"""步骤 7：把各标定结果汇总成 config/calibrated_20190513.yaml 与 reports/calibration_summary_20190513.md。"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from calibration.calibrate_headway import HEADWAY_REPORT, HEADWAYS_CSV
from calibration.calibrate_speed import DWELL_PRIOR_S, SPEED_REPORT, SPEEDS_CSV
from calibration.calibrate_transfer import TRANSFER_CSV, TRANSFER_REPORT
from calibration.common import CONFIG_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_dirs
from transit_sim.config import SimConfig

CALIBRATED_YAML = os.path.join(CONFIG_DIR, "calibrated_20190513.yaml")
SUMMARY_MD = os.path.join(REPORTS_DIR, "calibration_summary_20190513.md")

# 旧仿真器实验脚本使用的参数（对照）
LEGACY_PARAMS = {"subway_speed_kmh": 75, "bus_speed_kmh": 20, "walking_speed_mps": 12, "subway_headway_s": 800,
                 "bus_headway_s": 60, "subway_capacity": 65, "bus_capacity": 10, "subway_dwell_s": 30, "bus_dwell_s": 20,
                 "subway_transfer_s": 60}


def _rel(p: str) -> str:
    return os.path.relpath(p, PROJECT_ROOT)


def build(demand_fraction: float = 0.05, out_yaml: str = CALIBRATED_YAML) -> SimConfig:
    ensure_dirs()
    with open(SPEED_REPORT, "r", encoding="utf-8") as f:
        sp = json.load(f)
    with open(HEADWAY_REPORT, "r", encoding="utf-8") as f:
        hw = json.load(f)
    with open(TRANSFER_REPORT, "r", encoding="utf-8") as f:
        tr = json.load(f)
    hw_table = pd.read_csv(HEADWAYS_CSV)

    cfg = SimConfig.load(os.path.join(CONFIG_DIR, "default.yaml"))
    sub, bus = sp["mode"]["subway"]["all"], sp["mode"]["bus"]["all"]
    cfg.network.version = "2019"
    cfg.network.line_speeds_file = _rel(SPEEDS_CSV)
    cfg.network.line_headways_file = _rel(HEADWAYS_CSV)
    cfg.network.transfer_times_file = _rel(TRANSFER_CSV)
    cfg.demand.file = "data/demand/demand_20190513.csv"
    cfg.demand.fraction = float(demand_fraction)
    cfg.speeds.subway = round(sub["speed_kmh"], 2)
    cfg.speeds.bus = round(bus["speed_kmh"], 2)
    cfg.speeds.walk = 1.2
    sub_dwell = min(DWELL_PRIOR_S["subway"], sub["dwell_s"])
    bus_dwell = min(DWELL_PRIOR_S["bus"], bus["dwell_s"])
    cfg.dwell.subway = round(sub_dwell, 1)
    cfg.dwell.bus = round(bus_dwell, 1)
    cfg.stop_delay.subway = round(max(0.0, sub["dwell_s"] - sub_dwell), 1)
    cfg.stop_delay.bus = round(max(0.0, bus["dwell_s"] - bus_dwell), 1)
    cfg.terminal_hold.subway = round(sub.get("terminal_hold_s", 0.0), 1)
    cfg.terminal_hold.bus = round(bus.get("terminal_hold_s", 0.0), 1)

    def mode_hourly(mode: str) -> dict:
        t = hw_table[(hw_table["scope"] == "mode") & (hw_table["key"] == mode) & (hw_table["hour"].astype(str) != "all")]
        return {int(float(r.hour)): round(float(r.headway_s)) for r in t.itertuples(index=False)}

    cfg.headways.subway = mode_hourly("subway") or 300
    cfg.headways.bus = mode_hourly("bus") or 600
    cfg.access_egress.subway = round(float(hw["subway_access_egress_effective_s"]), 1)
    cfg.access_egress.bus = 0.0
    cfg.transfer.same_station.subway = round(float(tr["transfer_time_clipped_s"]), 1)
    cfg.routing.cache_file = "data/processed/route_cache_2019.pkl"
    cfg.save(out_yaml)
    print(f"[config] 写出 {out_yaml}")
    _write_summary(cfg, sp, hw, tr)
    return cfg


def _write_summary(cfg: SimConfig, sp: dict, hw: dict, tr: dict) -> None:
    sub, bus = sp["mode"]["subway"], sp["mode"]["bus"]
    L = []
    L.append("# 2019-05-13 刷卡数据标定结果汇总\n")
    L.append("数据：北京一卡通 2019-05-13（周一）全日刷卡，路网范围内有效行程 3.27M 条（地铁 1.22M / 公交 2.05M）。\n")
    L.append("## 1. 车速 / 逐站时间（calibrate_speed）\n")
    L.append("模型：公交 `tt = c·(n+1) + hold·1[首站] + d/v`；地铁 `tt = a + c·n + d/v`（d 为沿线直线段距离之和，n 为中间站数）。"
             "c 为逐站固定时间（开门停站 + 加减速/信号），拆为 dwell（先验）与 stop_delay。\n")
    L.append("| 模式 | 时段 | n | v (km/h) | ±se | 逐站 c (s) | 首站停留 (s) | 截距 a (s) | RMSE (s) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode, fits in (("subway", sub), ("bus", bus)):
        for period, f in fits.items():
            L.append(f"| {mode} | {period} | {f['n_obs']:,} | {f['speed_kmh']:.1f} | {f['se_speed_kmh']:.2f} | "
                     f"{f['dwell_s']:.0f} | {f['terminal_hold_s']:.0f} | {f['intercept_s']:.0f} | {f['rmse_s']:.0f} |")
    n_line_sub = sum(1 for k in sp["lines"] if k.startswith(("地铁", "首都机场线")))
    L.append(f"\n线路级估计：地铁 {n_line_sub} 条(方向)，公交 {len(sp['lines']) - n_line_sub} 条(方向)，见 `data/network/line_speeds_2019.csv`。\n")
    L.append("## 2. 发车间隔（calibrate_headway）\n")
    L.append(f"公交：上车刷卡聚类（间隙 {hw['gap_s']:.0f} s），{hw['bus']['routes_with_estimates']} 条线路(方向)得到分小时估计。模式级分小时班距 (s)：\n")
    L.append("| 小时 | " + " | ".join(str(h) for h in sorted(hw["bus_mode_headway_by_hour_s"], key=int)) + " |")
    L.append("|---|" + "---|" * len(hw["bus_mode_headway_by_hour_s"]))
    L.append("| 公交 | " + " | ".join(str(hw["bus_mode_headway_by_hour_s"][h]) for h in sorted(hw["bus_mode_headway_by_hour_s"], key=int)) + " |\n")
    L.append(f"地铁（方法 `{hw['subway_mode']}`）：进出站刷卡不能观测车次。截距 a = 进站 + 候车 + 上车站停站 + 出站；"
             f"单站行程残差低分位给出进出站下界 {hw['subway_access_egress']['access_egress_s']:.0f} s。")
    L.append("| 时段 | 截距 a (s) | 采用班距 (s) | 平均候车 (s) | 有效进出站 (s) | 隐含班距 (s, 若进出站取下界) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, v in hw["subway"].items():
        if not k.startswith("mode_"):
            continue
        L.append(f"| {k[5:]} | {v['intercept_s']:.0f} | {v['headway_s']:.0f} | {v['wait_s']:.0f} | "
                 f"{v['access_egress_effective_s']:.0f} | {v['headway_implied_s']:.0f} |")
    L.append("\n说明：进出站时间、候车时间与换乘步行时间之和可由刷卡数据识别，但三者的拆分不可识别。"
             "默认采用典型班距先验（高峰 3 min / 平峰 6 min / 夜间 8 min），把余量归入进出站时间；"
             "`--subway-mode implied` 则反过来用进出站下界反推班距（约 7.5–9 min）。两者对仿真平均行程时间等价，"
             "但影响列车数与拥挤程度；可在 `line_headways_2019.csv` 中按线路人工覆盖。\n")
    L.append("## 3. 换乘 / 进出站（calibrate_transfer）\n")
    L.append(f"恰好一次同站换乘的地铁 OD {tr['od_pairs_single_transfer']:,} 对、{tr['trips_used']:,} 条行程：观测时间减去 "
             f"进出站 + 停站 + 车内 + 两次候车期望后的残差中位数 = **{tr['transfer_time_median_s']:.0f} s**（采用 "
             f"{tr['transfer_time_clipped_s']:.0f} s）；分时段：" +
             ", ".join(f"{k} {v:.0f} s" for k, v in tr["transfer_time_by_period_s"].items()) +
             f"；{tr['stations_with_estimates']} 个换乘站有站点级估计（`transfer_times_2019.csv`）。\n")
    L.append("## 4. 写入配置 `config/calibrated_20190513.yaml` 的参数与旧版对照\n")
    L.append("| 参数 | 标定值 | 旧版实验值 |")
    L.append("|---|---:|---:|")
    rows = [
        ("地铁运行速度 km/h", f"{cfg.speeds.subway:.1f}（线路级见表）", LEGACY_PARAMS["subway_speed_kmh"]),
        ("公交运行速度 km/h（边际）", f"{cfg.speeds.bus:.1f}（线路级见表）", LEGACY_PARAMS["bus_speed_kmh"]),
        ("步行速度 m/s", cfg.speeds.walk, LEGACY_PARAMS["walking_speed_mps"]),
        ("地铁 dwell / stop_delay s", f"{cfg.dwell.subway} / {cfg.stop_delay.subway}", LEGACY_PARAMS["subway_dwell_s"]),
        ("公交 dwell / stop_delay s", f"{cfg.dwell.bus} / {cfg.stop_delay.bus}", LEGACY_PARAMS["bus_dwell_s"]),
        ("公交首站停留 s", cfg.terminal_hold.bus, "-"),
        ("地铁班距 s", f"高峰 {cfg.headways.subway.get(8, '-')} / 平峰 {cfg.headways.subway.get(12, '-')}", LEGACY_PARAMS["subway_headway_s"]),
        ("公交班距 s", f"高峰 {cfg.headways.bus.get(8, '-')} / 平峰 {cfg.headways.bus.get(12, '-')}", LEGACY_PARAMS["bus_headway_s"]),
        ("地铁进出站 s", cfg.access_egress.subway, "-"),
        ("地铁同站换乘 s", cfg.transfer.same_station.subway, LEGACY_PARAMS["subway_transfer_s"]),
        ("单车定员（真实）", f"地铁 {cfg.capacity_real.subway} / 公交 {cfg.capacity_real.bus}，× demand.fraction",
         f"{LEGACY_PARAMS['subway_capacity']} / {LEGACY_PARAMS['bus_capacity']}"),
    ]
    for k, v, o in rows:
        L.append(f"| {k} | {v} | {o} |")
    L.append("")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[config] 写出 {SUMMARY_MD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demand-fraction", type=float, default=0.05)
    ap.add_argument("--out", default=CALIBRATED_YAML)
    args = ap.parse_args()
    build(args.demand_fraction, args.out)


if __name__ == "__main__":
    main()
