"""步骤 5：标定发车间隔（frequency）。

公交（直接观测）：同一 (线路方向, 上车站) 的上车刷卡时刻按间隙 > ``--gap`` 秒切簇，一簇 ≈ 一班车；
相邻簇中心的间隔即该站观测到的班距。取各线路客流最大的若干站，按小时汇总中位数；同时用
"该小时簇数最多的站" 给出班距上界 3600/簇数，两者取较小者作为估计（漏掉无人上车的班次会使两者偏大）。

地铁（间接估计）：进出站刷卡无法观测到车次。利用 calibrate_speed 得到的截距
a = 进站 + 候车 + 上车站停站 + 出站，其中进出站时间取单站行程 (tt - d/v - c) 的低分位数
（候车接近 0 的乘客），候车 = a - AE - c，班距 ≈ 2 x 候车，按 线路 x 时段 估计并夹在
[``--subway-min``, ``--subway-max``] 内。该估计可在 line_headways_2019.csv 中人工覆盖。

输出 data/network/line_headways_2019.csv（scope, key, hour, headway_s, n_obs, method）
与 data/processed/calibrate_headway_report.json。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from calibration.calibrate_speed import SPEED_REPORT
from calibration.common import MATCHED_CSV, NETWORK_DIR, PROCESSED_DIR, ensure_dirs, load_matched, save_json
from transit_sim.utils import weighted_median

HEADWAYS_CSV = os.path.join(NETWORK_DIR, "line_headways_2019.csv")
HEADWAY_REPORT = os.path.join(PROCESSED_DIR, "calibrate_headway_report.json")
PERIOD_HOURS = {"am_peak": [7, 8], "pm_peak": [17, 18], "night": [22, 23, 0, 1, 2, 3, 4],
                "off_peak": [5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21]}


# ----------------------------------------------------------------------------- 公交
def _clusters(times: np.ndarray, gap: float) -> np.ndarray:
    """按时间间隙切簇，返回各簇的中位时刻。"""
    times = np.sort(times)
    if times.size == 0:
        return times
    breaks = np.where(np.diff(times) > gap)[0] + 1
    return np.array([np.median(c) for c in np.split(times, breaks)])


def bus_headways(m: pd.DataFrame, gap: float = 90.0, top_stops: int = 6, min_clusters: int = 3,
                 min_taps_stop: int = 40) -> Tuple[pd.DataFrame, dict]:
    bus = m[(m["mode"] == "bus") & m["route_name"].notna() & m["o_key"].notna()]
    rows = []
    n_routes = 0
    for rn, g in bus.groupby("route_name", sort=False):
        vol = g["o_key"].value_counts()
        stops = vol[vol >= min_taps_stop].index[:top_stops]
        if len(stops) == 0:
            continue
        n_routes += 1
        per_hour: Dict[int, List[Tuple[float, int]]] = {}
        max_clusters_hour: Dict[int, int] = {}
        for sk in stops:
            t = g.loc[g["o_key"] == sk, "dep_s"].values
            centers = _clusters(t, gap)
            if centers.size < 2:
                continue
            hours = (centers // 3600).astype(int) % 24
            gaps = np.diff(centers)
            gap_hours = hours[:-1]
            for h in np.unique(hours):
                n_c = int((hours == h).sum())
                max_clusters_hour[h] = max(max_clusters_hour.get(h, 0), n_c)
                sel = gap_hours == h
                if sel.sum() >= min_clusters - 1:
                    per_hour.setdefault(int(h), []).extend((float(x), 1) for x in gaps[sel] if 60 <= x <= 3600)
        for h, items in per_hour.items():
            if len(items) < min_clusters - 1:
                continue
            med = float(np.median([x for x, _ in items]))
            upper = 3600.0 / max(max_clusters_hour.get(h, 1), 1)
            est = min(med, max(upper, 60.0))
            rows.append({"scope": "line", "key": rn, "hour": h, "headway_s": est, "n_obs": len(items),
                         "median_gap_s": med, "upper_from_clusters_s": upper, "method": "bus_tapin_clusters"})
    table = pd.DataFrame(rows)
    rep = {"routes_with_estimates": int(table["key"].nunique()) if len(table) else 0,
           "routes_considered": n_routes, "rows": int(len(table))}
    return table, rep


# ----------------------------------------------------------------------------- 地铁
def estimate_access_egress(m: pd.DataFrame, speed_report: dict, quantile: float = 0.08) -> dict:
    """单站地铁行程：tt - d/v - c 的低分位数 ≈ 进站 + 出站（候车 ≈ 0 的乘客）。"""
    fit = speed_report["mode"]["subway"]["all"]
    s = m[(m["mode"] == "subway") & m["same_line"] & (m["n_intermediate"] == 0) & (m["route_dist_m"] > 300)]
    resid = s["tt_adj_s"] - s["route_dist_m"] / (fit["speed_kmh"] / 3.6) - fit["dwell_s"]
    resid = resid[(resid > 0) & (resid < 3600)]
    q = float(np.quantile(resid, quantile))
    return {"access_egress_s": q, "quantile": quantile, "n_obs": int(len(resid)),
            "resid_quantiles_s": {str(p): float(np.quantile(resid, p)) for p in (0.02, 0.05, 0.08, 0.1, 0.25, 0.5)}}


# 'prior' 模式下使用的典型 2019 北京地铁班距先验 (s)：高峰 ~3 min，平峰 ~6 min，夜间 ~8 min
SUBWAY_HEADWAY_PRIOR_S = {"all": 300.0, "am_peak": 180.0, "pm_peak": 180.0, "off_peak": 360.0, "night": 480.0}


def subway_headways(speed_report: dict, access_egress_s: float, hw_min: float, hw_max: float,
                    mode: str = "implied") -> Tuple[pd.DataFrame, dict]:
    """mode='implied'：班距 = 2 x (a - AE - c)；mode='prior'：班距取先验，AE_eff = a - c - 先验/2。"""
    rows = []
    rep = {"method": mode}
    mode_fits = speed_report["mode"]["subway"]
    method_tag = "subway_wait_intercept" if mode == "implied" else "subway_prior"

    def _hw(fit: dict, period: str, c: float) -> Tuple[float, float, float]:
        wait_implied = fit["intercept_s"] - access_egress_s - c
        hw_implied = float(np.clip(2.0 * wait_implied, hw_min, hw_max))
        if mode == "implied":
            return hw_implied, wait_implied, access_egress_s
        hw = SUBWAY_HEADWAY_PRIOR_S.get(period, SUBWAY_HEADWAY_PRIOR_S["all"])
        ae_eff = float(np.clip(fit["intercept_s"] - c - hw / 2.0, 60.0, 900.0))
        return hw, hw / 2.0, ae_eff

    for period, fit in mode_fits.items():
        hw, wait, ae = _hw(fit, period, fit["dwell_s"])
        rep[f"mode_{period}"] = {"intercept_s": fit["intercept_s"], "per_stop_s": fit["dwell_s"], "wait_s": wait,
                                 "headway_s": hw, "access_egress_effective_s": ae,
                                 "headway_implied_s": float(np.clip(2.0 * (fit["intercept_s"] - access_egress_s - fit["dwell_s"]), hw_min, hw_max)),
                                 "n_obs": fit["n_obs"]}
        hours = ["all"] if period == "all" else PERIOD_HOURS[period]
        for h in hours:
            rows.append({"scope": "mode", "key": "subway", "hour": h, "headway_s": hw, "n_obs": fit["n_obs"],
                         "method": method_tag})
    for rn, fits in speed_report["lines"].items():
        if not rn.startswith(("地铁", "首都机场线")):
            continue
        for period, fit in fits.items():
            c = mode_fits.get(period, mode_fits["all"])["dwell_s"]
            hw, _, _ = _hw(fit, period, c)
            hours = ["all"] if period == "all" else PERIOD_HOURS[period]
            for h in hours:
                rows.append({"scope": "line", "key": rn, "hour": h, "headway_s": hw, "n_obs": fit["n_obs"],
                             "method": method_tag})
    return pd.DataFrame(rows), rep


def calibrate(matched_path: str = MATCHED_CSV, gap: float = 90.0, subway_min: float = 120.0,
              subway_max: float = 900.0, subway_mode: str = "prior") -> dict:
    ensure_dirs()
    with open(SPEED_REPORT, "r", encoding="utf-8") as f:
        speed_report = json.load(f)
    m = load_matched(matched_path, usecols=["mode", "dep_s", "tt_adj_s", "o_key", "route_name", "route_dist_m",
                                             "n_intermediate", "same_line"])
    print("[headway] 公交：上车刷卡聚类 ...")
    bus_tab, bus_rep = bus_headways(m, gap=gap)
    # 公交模式级分小时：各线路估计的客流加权中位数
    mode_rows = []
    if len(bus_tab):
        for h, g in bus_tab.groupby("hour"):
            mode_rows.append({"scope": "mode", "key": "bus", "hour": int(h),
                              "headway_s": weighted_median(g["headway_s"], g["n_obs"]), "n_obs": int(g["n_obs"].sum()),
                              "method": "bus_tapin_clusters"})
        mode_rows.append({"scope": "mode", "key": "bus", "hour": "all",
                          "headway_s": weighted_median(bus_tab["headway_s"], bus_tab["n_obs"]),
                          "n_obs": int(bus_tab["n_obs"].sum()), "method": "bus_tapin_clusters"})
    print("[headway] 地铁：进出站时间与候车截距 ...")
    ae = estimate_access_egress(m, speed_report)
    sub_tab, sub_rep = subway_headways(speed_report, ae["access_egress_s"], subway_min, subway_max, subway_mode)

    table = pd.concat([pd.DataFrame(mode_rows), bus_tab, sub_tab], ignore_index=True)
    table = table[["scope", "key", "hour", "headway_s", "n_obs", "method"]
                  + [c for c in ("median_gap_s", "upper_from_clusters_s") if c in table.columns]]
    table.to_csv(HEADWAYS_CSV, index=False)

    bus_mode_by_hour = {int(r["hour"]): round(r["headway_s"]) for r in mode_rows if r["hour"] != "all"}
    report = {"gap_s": gap, "bus": bus_rep, "bus_mode_headway_by_hour_s": bus_mode_by_hour,
              "bus_mode_headway_all_s": next((r["headway_s"] for r in mode_rows if r["hour"] == "all"), None),
              "subway_access_egress": ae, "subway": sub_rep, "subway_mode": subway_mode,
              "subway_access_egress_effective_s": sub_rep["mode_all"]["access_egress_effective_s"],
              "subway_bounds_s": [subway_min, subway_max]}
    save_json(report, HEADWAY_REPORT)
    print(f"[headway] 公交: {bus_rep['routes_with_estimates']} 条线路(方向)有估计；模式级分小时(s): {bus_mode_by_hour}")
    print(f"[headway] 地铁({subway_mode}): 进出站(低分位) {ae['access_egress_s']:.0f}s, 有效进出站 "
          f"{report['subway_access_egress_effective_s']:.0f}s; "
          + ", ".join(f"{k.replace('mode_', '')}={v['headway_s']:.0f}s(wait {v['wait_s']:.0f}s)"
                      for k, v in sub_rep.items() if k.startswith("mode_")))
    print(f"[headway] 写出 {HEADWAYS_CSV} ({len(table)} 行)")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", default=MATCHED_CSV)
    ap.add_argument("--gap", type=float, default=90.0, help="切簇间隙 (s)")
    ap.add_argument("--subway-min", type=float, default=120.0)
    ap.add_argument("--subway-max", type=float, default=900.0)
    ap.add_argument("--subway-mode", default="prior", choices=["implied", "prior"],
                    help="implied: 由候车截距反推班距；prior: 用典型班距先验，把截距余量归入进出站时间")
    args = ap.parse_args()
    calibrate(args.matched, args.gap, args.subway_min, args.subway_max, args.subway_mode)


if __name__ == "__main__":
    main()
