"""步骤 4：标定车速与停站时间。

模型（对同一线路方向内直达的行程）::

    tt = a + d / v + n * dwell

* 公交：上车刷卡 -> 下车刷卡，tt 为纯车内时间（含中途停站），a 应接近 0；
* 地铁：进站刷卡 -> 出站刷卡，a = 进站步行 + 候车 + 出站步行（供 calibrate_transfer / headway 使用）。

分层估计：先按模式（含分时段）用稳健最小二乘联合估计 (a, 1/v, dwell)；再固定 dwell 为模式值，
按 线路 x 时段 估计 (a, 1/v)。输出 data/network/line_speeds_2019.csv 与
data/processed/calibrate_speed_report.json。

用法::

    python -m calibration.calibrate_speed
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from calibration.common import MATCHED_CSV, NETWORK_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_matched, save_json
from transit_sim.utils import classify_period

PERIODS = ["all", "am_peak", "pm_peak", "off_peak", "night"]
SPEEDS_CSV = os.path.join(NETWORK_DIR, "line_speeds_2019.csv")
SPEED_REPORT = os.path.join(PROCESSED_DIR, "calibrate_speed_report.json")

# 物理合理范围（公交的 "速度" 为扣除逐站固定延误后的边际速度，可高于商业速度）
SPEED_BOUNDS_KMH = {"bus": (4.0, 80.0), "subway": (15.0, 90.0)}
DWELL_BOUNDS_S = {"bus": (5.0, 180.0), "subway": (20.0, 60.0)}       # 逐站固定时间 c = dwell + stop_delay
HOLD_BOUNDS_S = {"bus": (0.0, 900.0), "subway": (0.0, 600.0)}
INTERCEPT_BOUNDS_S = {"bus": (0.0, 0.0), "subway": (0.0, 1800.0)}
F_SCALE_S = {"bus": 90.0, "subway": 120.0}
# 逐站固定时间 c 的拆分：开门停站 dwell 取先验（数据无法单独辨识），其余归为站间附加延误 stop_delay
DWELL_PRIOR_S = {"bus": 25.0, "subway": 35.0}


def _fit(d: np.ndarray, n: np.ndarray, tt: np.ndarray, mode: str, is_terminal: Optional[np.ndarray] = None,
         dwell_fixed: Optional[float] = None, hold_fixed: Optional[float] = None,
         f_scale: Optional[float] = None) -> dict:
    """稳健拟合仿真器的时间模型（b = 3.6 / v_kmh）::

        公交: tt = dwell * (n + 1) + hold * 1[首站上车] + b * d
              （上车刷卡 -> 下车刷卡；乘客经历上车站的停站，首站另有发车前等候 hold）
        地铁: tt = a + dwell * n + b * d
              （进站刷卡 -> 出站刷卡；a = 进站 + 候车 + 上车站停站 + 出站）

    线路级估计时固定 dwell（与 hold），只估 v（地铁另估 a）。返回参数与拟合质量。
    """
    vlo, vhi = SPEED_BOUNDS_KMH[mode]
    b_lo, b_hi = 3.6 / vhi, 3.6 / vlo
    f_scale = f_scale or F_SCALE_S[mode]
    if is_terminal is None:
        is_terminal = np.zeros_like(d, dtype=float)
    is_terminal = is_terminal.astype(float)
    b0 = 3.6 / (20.0 if mode == "bus" else 40.0)
    n_eff = n + 1.0 if mode == "bus" else n   # 公交乘客经历上车站停站

    names: list
    if mode == "bus":
        if dwell_fixed is None:
            def resid(p):
                return p[0] * n_eff + p[1] * is_terminal + p[2] * d - tt
            lo = [DWELL_BOUNDS_S[mode][0], HOLD_BOUNDS_S[mode][0], b_lo]
            hi = [DWELL_BOUNDS_S[mode][1], HOLD_BOUNDS_S[mode][1], b_hi]
            res = least_squares(resid, [25.0, 60.0, b0], loss="soft_l1", f_scale=f_scale, bounds=(lo, hi))
            c, hold, b = res.x
            names = ["dwell_s", "terminal_hold_s", "b"]
        else:
            c = float(dwell_fixed)
            hold = float(hold_fixed if hold_fixed is not None else 0.0)

            def resid(p):
                return c * n_eff + hold * is_terminal + p[0] * d - tt
            res = least_squares(resid, [b0], loss="soft_l1", f_scale=f_scale, bounds=([b_lo], [b_hi]))
            b = float(res.x[0])
            names = ["b"]
        a = 0.0
        pred = c * n_eff + hold * is_terminal + b * d
    else:
        a_lo, a_hi = INTERCEPT_BOUNDS_S[mode]
        if dwell_fixed is None:
            def resid(p):
                return p[0] + p[1] * n + p[2] * d - tt
            lo = [a_lo, DWELL_BOUNDS_S[mode][0], b_lo]
            hi = [a_hi, DWELL_BOUNDS_S[mode][1], b_hi]
            res = least_squares(resid, [300.0, 30.0, b0], loss="soft_l1", f_scale=f_scale, bounds=(lo, hi))
            a, c, b = res.x
            names = ["intercept_s", "dwell_s", "b"]
        else:
            c = float(dwell_fixed)

            def resid(p):
                return p[0] + c * n + p[1] * d - tt
            res = least_squares(resid, [300.0, b0], loss="soft_l1", f_scale=f_scale, bounds=([a_lo, b_lo], [a_hi, b_hi]))
            a, b = res.x
            names = ["intercept_s", "b"]
        hold = 0.0
        pred = a + c * n + b * d
    r = tt - pred
    # 近似标准误：最终残差的稳健尺度 × 雅可比
    J = res.jac
    scale = 1.4826 * np.median(np.abs(r - np.median(r))) if len(r) > 5 else np.nan
    try:
        cov = np.linalg.inv(J.T @ J) * scale ** 2
        se = dict(zip(names, np.sqrt(np.diag(cov))))
    except np.linalg.LinAlgError:
        se = {k: np.nan for k in names}
    v_kmh = 3.6 / b
    se_v = (3.6 / b ** 2) * se.get("b", np.nan)
    return {
        "n_obs": int(len(tt)), "intercept_s": float(a), "speed_kmh": float(v_kmh), "dwell_s": float(c),
        "terminal_hold_s": float(hold),
        "se_intercept_s": float(se.get("intercept_s", np.nan)), "se_speed_kmh": float(se_v),
        "se_dwell_s": float(se.get("dwell_s", np.nan)),
        "rmse_s": float(np.sqrt(np.mean(r ** 2))), "mae_s": float(np.mean(np.abs(r))),
        "median_resid_s": float(np.median(r)),
        "dwell_fixed": dwell_fixed is not None,
    }


def prepare(m: pd.DataFrame) -> pd.DataFrame:
    x = m[m["same_line"] & m["route_name"].notna() & (m["route_dist_m"] > 300)].copy()
    x["tt"] = np.where(x["mode"] == "subway", x["tt_adj_s"], x["tt_s"])
    x["period"] = [classify_period(t) for t in x["dep_s"].values]
    x["is_terminal"] = (x["o_idx"] == 0).astype(float)
    v_imp = x["route_dist_m"] / x["tt"] * 3.6
    ok = (x["tt"] >= 60) & (x["tt"] <= 7200)
    ok &= np.where(x["mode"] == "bus", (v_imp >= 2.0) & (v_imp <= 80.0), (v_imp >= 2.0) & (v_imp <= 120.0))
    return x[ok]


def _fit_df(xp: pd.DataFrame, mode: str, **kw) -> dict:
    return _fit(xp["route_dist_m"].values, xp["n_intermediate"].values.astype(float), xp["tt"].values, mode,
                is_terminal=xp["is_terminal"].values, **kw)


def calibrate(matched_path: str = MATCHED_CSV, min_line_obs: int = 150, make_plots: bool = True) -> dict:
    ensure_dirs()
    cols = ["mode", "dep_s", "tt_s", "tt_adj_s", "route_name", "route_dist_m", "n_intermediate", "same_line", "o_idx"]
    m = load_matched(matched_path, usecols=cols)
    x = prepare(m)
    rows = []
    report: Dict[str, dict] = {"mode": {}, "lines": {}, "filters": {"records_used": int(len(x))}}

    def _row(scope, key, period, fit, c_stop, hold, mode):
        dwell = min(DWELL_PRIOR_S[mode], c_stop)
        return {"scope": scope, "key": key, "period": period, "speed_kmh": fit["speed_kmh"],
                "dwell_s": dwell, "stop_delay_s": max(0.0, c_stop - dwell), "per_stop_time_s": c_stop,
                "terminal_hold_s": hold, "intercept_s": fit["intercept_s"], "n_obs": fit["n_obs"],
                "rmse_s": fit["rmse_s"], "se_speed_kmh": fit["se_speed_kmh"]}

    for mode in ("bus", "subway"):
        xm = x[x["mode"] == mode]
        report["mode"][mode] = {}
        mode_hold = None
        for period in PERIODS:
            xp = xm if period == "all" else xm[xm["period"] == period]
            if len(xp) < 500:
                continue
            # 模式级：公交各时段自由估计 (c, v)；地铁站距与站数近似共线、c 不可辨识，固定为先验 35 s 只估 (a, v)
            fit = _fit_df(xp, mode, dwell_fixed=DWELL_PRIOR_S["subway"] if mode == "subway" else None,
                          hold_fixed=None if period == "all" else mode_hold)
            if period == "all":
                mode_hold = fit["terminal_hold_s"]
            report["mode"][mode][period] = fit
            rows.append(_row("mode", mode, period, fit, fit["dwell_s"], mode_hold, mode))
            print(f"[speed] {mode:6s} {period:8s} n={fit['n_obs']:>9,} v={fit['speed_kmh']:5.1f} km/h "
                  f"(±{fit['se_speed_kmh']:.2f}) per-stop={fit['dwell_s']:5.1f}s hold={fit['terminal_hold_s']:5.1f}s "
                  f"a={fit['intercept_s']:6.1f}s rmse={fit['rmse_s']:.0f}s")
        # 线路级：固定该时段的模式级 c / hold，估 v（地铁另估 a）
        n_line = 0
        for rn, xl in xm.groupby("route_name"):
            for period in PERIODS:
                xp = xl if period == "all" else xl[xl["period"] == period]
                if len(xp) < min_line_obs or xp["route_dist_m"].std() < 300:
                    continue
                c_period = report["mode"][mode].get(period, report["mode"][mode]["all"])["dwell_s"]
                fit = _fit_df(xp, mode, dwell_fixed=c_period, hold_fixed=mode_hold)
                report["lines"].setdefault(rn, {})[period] = fit
                rows.append(_row("line", rn, period, fit, c_period, mode_hold, mode))
                if period == "all":
                    n_line += 1
        print(f"[speed] {mode}: 线路级估计 {n_line} 条(方向)")

    table = pd.DataFrame(rows)
    table.to_csv(SPEEDS_CSV, index=False)
    save_json(report, SPEED_REPORT)
    print(f"[speed] 写出 {SPEEDS_CSV} ({len(table)} 行) 与 {SPEED_REPORT}")
    if make_plots:
        _plot_fits(x, report)
    return report


def _plot_fits(x: pd.DataFrame, report: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, ("bus", "subway")):
        xm = x[x["mode"] == mode]
        fit = report["mode"].get(mode, {}).get("all")
        if fit is None or xm.empty:
            continue
        samp = xm.sample(min(len(xm), 200_000), random_state=0)
        n_eff = samp["n_intermediate"] + (1 if mode == "bus" else 0)
        pred = (fit["intercept_s"] + 3.6 / fit["speed_kmh"] * samp["route_dist_m"] + fit["dwell_s"] * n_eff
                + fit["terminal_hold_s"] * samp["is_terminal"])
        ax.hexbin(pred / 60, samp["tt"] / 60, gridsize=60, bins="log", cmap="viridis", extent=(0, 90, 0, 90))
        ax.plot([0, 90], [0, 90], "r--", lw=1)
        ax.set_xlabel("模型预测 (min)")
        ax.set_ylabel("观测 (min)")
        ax.set_title(f"{mode}: v={fit['speed_kmh']:.1f} km/h, dwell={fit['dwell_s']:.0f}s, "
                     f"hold={fit['terminal_hold_s']:.0f}s, a={fit['intercept_s']:.0f}s, n={fit['n_obs']:,}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "speed_fit_20190513.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", default=MATCHED_CSV)
    ap.add_argument("--min-line-obs", type=int, default=150)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    calibrate(args.matched, args.min_line_obs, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
