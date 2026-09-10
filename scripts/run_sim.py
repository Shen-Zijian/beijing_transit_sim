"""命令行运行仿真。

用法::

    python scripts/run_sim.py --config config/calibrated_20190513.yaml \
        [--run-id my_run] [--set demand.fraction=0.1 --set time.sim_end=43200] [--quiet]

结果写入 output/<run_id>/{trips.csv, hourly_stats.csv, summary.json, config.yaml}。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transit_sim import SimConfig, TransitSimulator, parse_set_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "calibrated_20190513.yaml"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--set", action="append", default=[], help="覆盖配置项，如 --set demand.fraction=0.1")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig.load(args.config, overrides=parse_set_args(args.set))
    sim = TransitSimulator(cfg, run_id=args.run_id, verbose=not args.quiet)
    result = sim.run()
    print(json.dumps({k: v for k, v in result.summary.items() if k != "by_main_mode"},
                     ensure_ascii=False, indent=2, default=str))
    if result.output_paths:
        print("输出:", result.output_paths)


if __name__ == "__main__":
    main()
