"""步骤 6：标定地铁同站换乘时间（并汇总进出站时间）。

对需要换乘的地铁行程（same_line = False），用 2019 路网 + 已标定车速/班距求最短路径；仅取恰好一次
同站换乘的 OD：

    观测 tt = 进出站 AE + 上车站停站 c x 2 + 车内 ivt + 候车1 (h1/2) + 候车2 (h2/2) + 换乘步行 T

除 T 外均已知，逐行程残差的中位数即换乘步行时间；按换乘站分组给出站点级估计（样本 >= ``--min-obs``），
其余用模式级中位数。输出 data/network/transfer_times_2019.csv 与 data/processed/calibrate_transfer_report.json。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from calibration.calibrate_headway import HEADWAY_REPORT, HEADWAYS_CSV
from calibration.calibrate_speed import SPEED_REPORT, SPEEDS_CSV
from calibration.common import MATCHED_CSV, NETWORK_DIR, PROCESSED_DIR, ensure_dirs, load_matched, load_network_version, save_json
from transit_sim.network import HeadwayTable, SpeedTable
from transit_sim.routing import Router
from transit_sim.utils import classify_period

TRANSFER_CSV = os.path.join(NETWORK_DIR, "transfer_times_2019.csv")
TRANSFER_REPORT = os.path.join(PROCESSED_DIR, "calibrate_transfer_report.json")


def calibrate(matched_path: str = MATCHED_CSV, top_od: int = 4000, min_obs: int = 100,
              default_transfer_s: float = 180.0) -> dict:
    ensure_dirs()
    with open(SPEED_REPORT, "r", encoding="utf-8") as f:
        speed_report = json.load(f)
    with open(HEADWAY_REPORT, "r", encoding="utf-8") as f:
        hw_report = json.load(f)
    ae = float(hw_report["subway_access_egress_effective_s"])
    per_stop = {p: fit["dwell_s"] for p, fit in speed_report["mode"]["subway"].items()}

    net = load_network_version("2019", speed_table=SpeedTable.from_csv(SPEEDS_CSV),
                               same_station_transfer={"subway": default_transfer_s, "bus": 30.0, "intermodal": 180.0})
    hw_table = HeadwayTable.from_csv(HEADWAYS_CSV)

    def headway(line_id: str, mode: str, hour: int) -> float:
        h = hw_table.get(line_id, mode, hour)
        return float(h) if h else (300.0 if mode == "subway" else 600.0)

    router = Router(net, transfer_penalty=180.0, k=1,
                    headway_fn=lambda lid, mode: headway(lid, mode, 12))

    m = load_matched(matched_path, usecols=["mode", "dep_s", "tt_adj_s", "o_key", "d_key", "same_line"])
    s = m[(m["mode"] == "subway") & (~m["same_line"]) & m["o_key"].notna() & m["d_key"].notna()]
    s = s[(s["tt_adj_s"] > 120) & (s["tt_adj_s"] < 7200)]
    od_counts = s.groupby(["o_key", "d_key"]).size().sort_values(ascending=False)
    top = od_counts.head(top_od)
    print(f"[transfer] 换乘地铁行程 {len(s):,} 条，唯一 OD {len(od_counts):,}；取客流最大的 {len(top):,} 对 OD "
          f"(覆盖 {top.sum() / len(s):.1%})")

    t0 = time.time()
    od_info: Dict[tuple, dict] = {}
    n_single = 0
    for (o, d), n in top.items():
        path = router.shortest_path(o, d)
        if not path:
            continue
        route = router.build_route(path)
        transfers = [e for e in route["edge_path"] if e["mode"] == "transfer"]
        walks = [e for e in route["edge_path"] if e["mode"] == "walk"]
        if route["num_transfers"] != 1 or len(transfers) != 1 or walks:
            continue
        n_single += 1
        od_info[(o, d)] = {
            "ivt": route["in_vehicle_time"], "lines": route["lines"],
            "transfer_station": transfers[0]["from_node"], "default_transfer": transfers[0]["travel_time"],
        }
    print(f"[transfer] 最短路计算完成 {time.time() - t0:.0f}s；恰好一次同站换乘的 OD {n_single:,}")

    ss = s.merge(pd.DataFrame([{"o_key": o, "d_key": d, **v} for (o, d), v in od_info.items()]),
                 on=["o_key", "d_key"], how="inner")
    hours = (ss["dep_s"] // 3600).astype(int) % 24
    periods = [classify_period(t) for t in ss["dep_s"].values]
    c = np.array([per_stop.get(p, per_stop["all"]) for p in periods])
    h1 = np.array([headway(l[0], "subway", h) for l, h in zip(ss["lines"], hours)])
    h2 = np.array([headway(l[1], "subway", h) for l, h in zip(ss["lines"], hours)])
    pred_wo_transfer = ae + 2 * c + ss["ivt"].values + 0.5 * h1 + 0.5 * h2
    ss["resid"] = ss["tt_adj_s"].values - pred_wo_transfer
    ss["period"] = periods
    valid = ss[(ss["resid"] > -900) & (ss["resid"] < 2400)]

    overall = float(np.median(valid["resid"]))
    overall_clipped = float(np.clip(overall, 60.0, 600.0))
    by_station = (valid.groupby("transfer_station")["resid"]
                  .agg(n="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)))
    by_station = by_station[by_station["n"] >= min_obs]
    by_period = valid.groupby("period")["resid"].median().to_dict()

    rows = [{"scope": "mode", "key": "subway", "transfer_time_s": overall_clipped, "n_obs": int(len(valid)),
             "raw_median_s": overall}]
    for st, r in by_station.iterrows():
        rows.append({"scope": "station", "key": st, "transfer_time_s": float(np.clip(r["median"], 60.0, 600.0)),
                     "n_obs": int(r["n"]), "raw_median_s": float(r["median"])})
    table = pd.DataFrame(rows)
    table.to_csv(TRANSFER_CSV, index=False)

    report = {
        "access_egress_s": ae, "trips_used": int(len(valid)), "od_pairs_single_transfer": n_single,
        "transfer_time_median_s": overall, "transfer_time_clipped_s": overall_clipped,
        "transfer_time_by_period_s": by_period,
        "resid_quantiles_s": {str(q): float(valid["resid"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "stations_with_estimates": int(len(by_station)),
        "by_station": {st: {"n": int(r["n"]), "median_s": float(r["median"]), "name": net.stations[st].name}
                       for st, r in by_station.iterrows()},
    }
    save_json(report, TRANSFER_REPORT)
    print(f"[transfer] 换乘步行时间中位数 {overall:.0f}s（分时段 {{{', '.join(f'{k}: {v:.0f}' for k, v in by_period.items())}}}），"
          f"站点级估计 {len(by_station)} 站；写出 {TRANSFER_CSV}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", default=MATCHED_CSV)
    ap.add_argument("--top-od", type=int, default=4000)
    ap.add_argument("--min-obs", type=int, default=100)
    args = ap.parse_args()
    calibrate(args.matched, args.top_od, args.min_obs)


if __name__ == "__main__":
    main()
