"""步骤 9：仿真 vs 观测 验证。

用标定配置（默认 demand.fraction=0.05, 05:00–24:00）跑一天，把每个仿真乘客的行程时间与其对应刷卡记录的
观测时间比较（demand 文件中的 observed_tt_s）。口径：

* 地铁出发的行程：观测 = 进站刷卡 -> 出站刷卡；仿真 = total_time（含进出站、候车、换乘）；
* 公交出发的行程：观测 = 上车刷卡 -> 下车刷卡；仿真 = total_time - 首次候车（刷卡发生在上车时）。

同时用旧版实验参数（地铁 75 / 公交 20 km/h，班距 800 / 60 s，dwell 30 / 20 s，同站换乘 60 s，无进出站时间）
在同一 2019 路网与需求上跑一组对照。输出 reports/validation_20190513.md 与图。

用法::

    python -m calibration.validate [--fraction 0.05] [--skip-legacy]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from calibration.build_config import CALIBRATED_YAML
from calibration.common import PROJECT_ROOT, REPORTS_DIR, ensure_dirs, save_json
from transit_sim import SimConfig, TransitSimulator

LEGACY_OVERRIDES = {
    "network.line_speeds_file": None, "network.line_headways_file": None, "network.transfer_times_file": None,
    "speeds.subway": 75.0, "speeds.bus": 20.0, "speeds.walk": 1.2,
    "dwell.subway": 30.0, "dwell.bus": 20.0, "stop_delay.subway": 0.0, "stop_delay.bus": 0.0,
    "terminal_hold.subway": 0.0, "terminal_hold.bus": 0.0,
    "headways.subway": 800, "headways.bus": 60,
    "transfer.same_station.subway": 60.0, "access_egress.subway": 0.0,
    "routing.cache_file": "data/processed/route_cache_2019_legacy.pkl",
}


def comparable_sim_time(trips: pd.DataFrame) -> pd.Series:
    t = trips["total_time"].copy()
    bus = trips["origin_mode"] == "bus"
    t[bus] = trips.loc[bus, "total_time"] - trips.loc[bus, "first_wait_time"]
    return t


def compare(trips: pd.DataFrame) -> Dict[str, dict]:
    df = trips[(trips["status"] == "arrived") & trips["observed_tt_s"].notna()].copy()
    df["sim_min"] = comparable_sim_time(df) / 60.0
    df["obs_min"] = df["observed_tt_s"] / 60.0
    out = {}
    for name, g in [("all", df), ("subway", df[df["origin_mode"] == "subway"]), ("bus", df[df["origin_mode"] == "bus"])]:
        if len(g) < 10:
            continue
        err = g["sim_min"] - g["obs_min"]
        out[name] = {
            "n": int(len(g)),
            "sim_mean": float(g["sim_min"].mean()), "obs_mean": float(g["obs_min"].mean()),
            "sim_median": float(g["sim_min"].median()), "obs_median": float(g["obs_min"].median()),
            "sim_p90": float(g["sim_min"].quantile(0.9)), "obs_p90": float(g["obs_min"].quantile(0.9)),
            "mean_error": float(err.mean()), "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "ks_stat": float(ks_2samp(g["sim_min"], g["obs_min"]).statistic),
            "wasserstein_min": float(wasserstein_distance(g["sim_min"], g["obs_min"])),
            "corr": float(np.corrcoef(g["sim_min"], g["obs_min"])[0, 1]),
        }
    hourly = (df.groupby(["origin_mode", "departure_hour"])
              .agg(n=("sim_min", "size"), sim_mean=("sim_min", "mean"), obs_mean=("obs_min", "mean"),
                   sim_median=("sim_min", "median"), obs_median=("obs_min", "median")).reset_index())
    return out, hourly, df


def run_sim(cfg: SimConfig, run_id: str) -> pd.DataFrame:
    sim = TransitSimulator(cfg, run_id=run_id, verbose=True)
    res = sim.run()
    return res.trips


def ensure_cache(config_path: str, overrides: Dict[str, object], fraction: float, workers: int) -> None:
    cfg = SimConfig.load(config_path, overrides=overrides)
    cache = cfg.resolve(cfg.routing.cache_file)
    if cache and os.path.exists(cache):
        return
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "precompute_paths.py"), "--config", config_path,
           "--workers", str(workers), "--fraction", str(fraction)]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={'null' if v is None else v}"]
    print("[validate] 预计算对照配置的路径缓存:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def plot_all(df_cal: pd.DataFrame, hourly_cal: pd.DataFrame, df_leg: Optional[pd.DataFrame],
             hourly_leg: Optional[pd.DataFrame], out_dir: str) -> Dict[str, str]:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = {}
    # 1. CDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ("subway", "bus")):
        g = df_cal[df_cal["origin_mode"] == mode]
        for col, label, ls in (("obs_min", "观测 (刷卡)", "-"), ("sim_min", "仿真 (标定参数)", "--")):
            x = np.sort(g[col].values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls, label=label)
        if df_leg is not None:
            gl = df_leg[df_leg["origin_mode"] == mode]
            x = np.sort(gl["sim_min"].values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), ":", label="仿真 (旧参数)")
        ax.set_xlim(0, 90)
        ax.set_xlabel("行程时间 (min)")
        ax.set_ylabel("累积概率")
        ax.set_title(f"{'地铁' if mode == 'subway' else '公交'}出发行程 (n={len(g):,})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "validation_cdf_20190513.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["cdf"] = p
    # 2. hourly means
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ("subway", "bus")):
        h = hourly_cal[hourly_cal["origin_mode"] == mode]
        ax.plot(h["departure_hour"], h["obs_mean"], "o-", label="观测均值")
        ax.plot(h["departure_hour"], h["sim_mean"], "s--", label="仿真均值 (标定)")
        if hourly_leg is not None:
            hl = hourly_leg[hourly_leg["origin_mode"] == mode]
            ax.plot(hl["departure_hour"], hl["sim_mean"], "^:", label="仿真均值 (旧参数)")
        ax.set_xlabel("出发小时")
        ax.set_ylabel("平均行程时间 (min)")
        ax.set_title("地铁" if mode == "subway" else "公交")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "validation_hourly_20190513.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["hourly"] = p
    # 3. scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, ("subway", "bus")):
        g = df_cal[df_cal["origin_mode"] == mode].sample(min(50_000, (df_cal["origin_mode"] == mode).sum()), random_state=0)
        ax.hexbin(g["obs_min"], g["sim_min"], gridsize=60, bins="log", cmap="viridis", extent=(0, 90, 0, 90))
        ax.plot([0, 90], [0, 90], "r--", lw=1)
        ax.set_xlabel("观测 (min)")
        ax.set_ylabel("仿真 (min)")
        ax.set_title("地铁" if mode == "subway" else "公交")
    fig.tight_layout()
    p = os.path.join(out_dir, "validation_scatter_20190513.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["scatter"] = p
    return paths


def write_report(stats_cal: dict, stats_leg: Optional[dict], hourly_cal: pd.DataFrame, cfg: SimConfig,
                 summary_cal: dict, summary_leg: Optional[dict], plots: Dict[str, str], path: str) -> None:
    L = ["# 2019-05-13 仿真 vs 观测 验证报告\n",
         f"配置：`{os.path.relpath(cfg.source_file or CALIBRATED_YAML, PROJECT_ROOT)}`，demand.fraction = {cfg.demand.fraction}，"
         f"{cfg.time.sim_start // 3600:02d}:00–{cfg.time.sim_end // 3600:02d}:00，路网 {cfg.network.version}，"
         f"路径选择 {cfg.choice.model}(k={cfg.choice.k_paths})。\n",
         "口径：地铁出发行程比较 进站刷卡→出站刷卡 总时间；公交出发行程比较 上车→下车（仿真扣除首次候车）。\n",
         "## 1. 总体指标（分钟）\n",
         "| 组 | n | 观测均值 | 仿真均值 (标定) | 仿真均值 (旧参数) | 观测中位 | 仿真中位 (标定) | 观测 P90 | 仿真 P90 (标定) | 均误差 | MAE | KS | Wasserstein |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("all", "subway", "bus"):
        c = stats_cal.get(k)
        if not c:
            continue
        lg = (stats_leg or {}).get(k)
        L.append(f"| {k} | {c['n']:,} | {c['obs_mean']:.1f} | {c['sim_mean']:.1f} | "
                 f"{lg['sim_mean']:.1f} | " if lg else f"| {k} | {c['n']:,} | {c['obs_mean']:.1f} | {c['sim_mean']:.1f} | - | ")
        L[-1] += (f"{c['obs_median']:.1f} | {c['sim_median']:.1f} | {c['obs_p90']:.1f} | {c['sim_p90']:.1f} | "
                  f"{c['mean_error']:+.1f} | {c['mae']:.1f} | {c['ks_stat']:.3f} | {c['wasserstein_min']:.2f} |")
    if stats_leg:
        L.append("\n旧参数对照（同一路网/需求/选择模型，仅参数不同）：")
        L.append("| 组 | 仿真均值 | 均误差 | MAE | KS | Wasserstein |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k in ("all", "subway", "bus"):
            lg = stats_leg.get(k)
            if lg:
                L.append(f"| {k} | {lg['sim_mean']:.1f} | {lg['mean_error']:+.1f} | {lg['mae']:.1f} | {lg['ks_stat']:.3f} | {lg['wasserstein_min']:.2f} |")
    L.append("\n## 2. 系统层面\n")
    L.append("| 指标 | 标定参数 | 旧参数 |")
    L.append("|---|---:|---:|")
    for key, label in (("n_trips", "乘客数"), ("completion_rate", "完成率"), ("n_timeout_exit", "超时退出"),
                       ("wait_time_min_mean", "平均候车 (min)"), ("walk_time_min_mean", "平均步行 (min)"),
                       ("in_vehicle_time_min_mean", "平均车内 (min)"), ("transfers_mean", "平均换乘次数"),
                       ("runtime_s", "运行时间 (s)")):
        a = summary_cal.get(key, "-")
        b = summary_leg.get(key, "-") if summary_leg else "-"
        fmt = (lambda v: f"{v:.3f}" if isinstance(v, float) else f"{v}")
        L.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
    L.append("\n## 3. 分小时均值（标定参数）\n")
    L.append("| 模式 | 小时 | n | 观测均值 | 仿真均值 | 观测中位 | 仿真中位 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in hourly_cal.itertuples(index=False):
        L.append(f"| {r.origin_mode} | {int(r.departure_hour)} | {int(r.n):,} | {r.obs_mean:.1f} | {r.sim_mean:.1f} | "
                 f"{r.obs_median:.1f} | {r.sim_median:.1f} |")
    L.append("\n## 4. 图\n")
    for k, p in plots.items():
        L.append(f"![{k}]({os.path.relpath(p, os.path.dirname(path))})\n")
    L.append("## 5. 说明\n")
    L.append("- 仿真中乘客按 logit 选择候选路径，观测行程未必对应仿真所选路径；公交记录只覆盖单次乘车，"
             "若仿真为该 OD 选择了含换乘的路径，其比较口径会偏长。")
    L.append("- 地铁的进出站 / 候车 / 换乘步行三者之和可识别、拆分不可识别，见 calibration_summary。")
    L.append("- 车辆容量 = 真实定员 × demand.fraction；小比例抽样下拥挤效应被同比例缩放。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CALIBRATED_YAML)
    ap.add_argument("--fraction", type=float, default=None, help="覆盖 demand.fraction")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    ensure_dirs()
    overrides = {"demand.fraction": args.fraction} if args.fraction is not None else {}
    cfg = SimConfig.load(args.config, overrides=overrides)

    print("[validate] 运行标定参数仿真 ...")
    sim = TransitSimulator(cfg, run_id="validate_calibrated", verbose=True)
    res_cal = sim.run()
    stats_cal, hourly_cal, df_cal = compare(res_cal.trips)

    stats_leg = hourly_leg = df_leg = None
    summary_leg = None
    if not args.skip_legacy:
        leg_over = dict(overrides)
        leg_over.update(LEGACY_OVERRIDES)
        ensure_cache(args.config, leg_over, cfg.demand.fraction, args.workers)
        cfg_leg = SimConfig.load(args.config, overrides=leg_over)
        print("[validate] 运行旧参数对照仿真 ...")
        res_leg = TransitSimulator(cfg_leg, run_id="validate_legacy", verbose=True).run()
        stats_leg, hourly_leg, df_leg = compare(res_leg.trips)
        summary_leg = res_leg.summary

    plots = plot_all(df_cal, hourly_cal, df_leg, hourly_leg, REPORTS_DIR)
    report_path = os.path.join(REPORTS_DIR, "validation_20190513.md")
    write_report(stats_cal, stats_leg, hourly_cal, cfg, res_cal.summary, summary_leg, plots, report_path)
    save_json({"calibrated": stats_cal, "legacy": stats_leg,
               "hourly_calibrated": hourly_cal.to_dict(orient="records")},
              os.path.join(REPORTS_DIR, "validation_20190513.json"))
    print(f"[validate] 报告 {report_path}")
    for k in ("all", "subway", "bus"):
        c = stats_cal.get(k)
        if c:
            lg = (stats_leg or {}).get(k)
            print(f"  {k:6s} n={c['n']:,} obs={c['obs_mean']:.1f} sim={c['sim_mean']:.1f} "
                  f"(err {c['mean_error']:+.1f}, MAE {c['mae']:.1f}, KS {c['ks_stat']:.3f})"
                  + (f" | legacy sim={lg['sim_mean']:.1f} (err {lg['mean_error']:+.1f}, KS {lg['ks_stat']:.3f})" if lg else ""))


if __name__ == "__main__":
    main()
