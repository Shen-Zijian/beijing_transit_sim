"""一键运行完整标定流水线。

    python -m calibration.run_all [--raw /path/20190513.csv] [--skip-preprocess] [--precompute] [--validate]

步骤：
  1. preprocess(full)  原始刷卡 -> 清洗 -> 按完整路网匹配（供裁剪）
  2. prune_network_2019 -> nodes_2019 / edges_2019
  3. preprocess(2019)  按 2019 路网重新匹配（标定用）
  4. convert_demand    -> demand_20190513.csv + hourly_profile
  5. calibrate_speed   -> line_speeds_2019.csv
  6. calibrate_headway -> line_headways_2019.csv
  7. calibrate_transfer-> transfer_times_2019.csv
  8. build_config      -> config/calibrated_20190513.yaml + reports/calibration_summary_20190513.md
  9. (可选) scripts/precompute_paths.py  预计算路径缓存
 10. (可选) calibration.validate           仿真 vs 观测验证报告
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from calibration import build_config, calibrate_headway, calibrate_speed, calibrate_transfer, convert_demand
from calibration import preprocess_smartcard, prune_network_2019
from calibration.common import (DEFAULT_RAW_CSV, FILTERED_CACHE_CSV, MATCHED_CSV, MATCHED_FULL_CSV, PROJECT_ROOT,
                                ensure_dirs)


def _stage(name):
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
    return time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW_CSV)
    ap.add_argument("--skip-preprocess", action="store_true", help="已有匹配文件时跳过 1–3 步")
    ap.add_argument("--subway-headway-mode", default="prior", choices=["prior", "implied"])
    ap.add_argument("--demand-fraction", type=float, default=0.05)
    ap.add_argument("--precompute", action="store_true", help="随后预计算路径缓存（多进程，数十分钟）")
    ap.add_argument("--validate", action="store_true", help="随后运行验证（需要路径缓存）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    ensure_dirs()
    t_all = time.time()

    if not args.skip_preprocess or not os.path.exists(MATCHED_CSV):
        t = _stage("1/8 preprocess (full network)")
        preprocess_smartcard.run(args.raw, MATCHED_FULL_CSV, "full", cached_filtered=FILTERED_CACHE_CSV)
        print(f"用时 {time.time() - t:.0f}s")
        t = _stage("2/8 prune_network_2019")
        prune_network_2019.prune(MATCHED_FULL_CSV)
        print(f"用时 {time.time() - t:.0f}s")
        t = _stage("3/8 preprocess (2019 network)")
        preprocess_smartcard.run(args.raw, MATCHED_CSV, "2019", cached_filtered=FILTERED_CACHE_CSV)
        print(f"用时 {time.time() - t:.0f}s")
    else:
        print("跳过 1–3 步（使用已有匹配文件）")

    t = _stage("4/8 convert_demand")
    convert_demand.convert(MATCHED_CSV, "2019")
    print(f"用时 {time.time() - t:.0f}s")
    t = _stage("5/8 calibrate_speed")
    calibrate_speed.calibrate(MATCHED_CSV)
    print(f"用时 {time.time() - t:.0f}s")
    t = _stage("6/8 calibrate_headway")
    calibrate_headway.calibrate(MATCHED_CSV, subway_mode=args.subway_headway_mode)
    print(f"用时 {time.time() - t:.0f}s")
    t = _stage("7/8 calibrate_transfer")
    calibrate_transfer.calibrate(MATCHED_CSV)
    print(f"用时 {time.time() - t:.0f}s")
    t = _stage("8/8 build_config")
    build_config.build(args.demand_fraction)
    print(f"用时 {time.time() - t:.0f}s")

    py = sys.executable
    if args.precompute:
        _stage("precompute_paths")
        subprocess.run([py, os.path.join(PROJECT_ROOT, "scripts", "precompute_paths.py"),
                        "--workers", str(args.workers)], check=True)
    if args.validate:
        _stage("validate")
        subprocess.run([py, "-m", "calibration.validate"], check=True, cwd=PROJECT_ROOT)
    print(f"\n全部完成，总用时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
