"""需求时间剖面：按小时（及 15 分钟）统计出行量，输出 CSV 与图。

用法::

    python -m calibration.demand_profile [--demand data/demand/demand_20190513.csv]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calibration.common import DEMAND_DIR, REPORTS_DIR, ensure_dirs

DEMAND_CSV = os.path.join(DEMAND_DIR, "demand_20190513.csv")
PROFILE_CSV = os.path.join(DEMAND_DIR, "hourly_profile_20190513.csv")


def hourly_profile(df: pd.DataFrame, time_col: str = "time", mode_col: str = "origin_type") -> pd.DataFrame:
    hour = (df[time_col] // 3600).astype(int).clip(0, 23)
    out = pd.DataFrame({"hour": range(24)}).set_index("hour")
    out["trips"] = hour.value_counts().reindex(range(24), fill_value=0)
    for mode in ("subway", "bus"):
        out[mode] = hour[df[mode_col] == mode].value_counts().reindex(range(24), fill_value=0)
    out["share"] = out["trips"] / max(out["trips"].sum(), 1)
    return out.reset_index()


def quarter_hour_profile(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    q = (df[time_col] // 900).astype(int)
    s = q.value_counts().sort_index()
    return pd.DataFrame({"quarter": s.index, "start_s": s.index * 900, "trips": s.values})


def plot_profile(profile: pd.DataFrame, path: str, title: str = "2019-05-13 需求时间剖面（路网范围内）") -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(profile["hour"] - 0.2, profile["subway"], width=0.4, label="地铁")
    ax.bar(profile["hour"] + 0.2, profile["bus"], width=0.4, label="公交")
    ax.set_xlabel("出发小时")
    ax.set_ylabel("出行数")
    ax.set_xticks(range(0, 24))
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand", default=DEMAND_CSV)
    ap.add_argument("--out", default=PROFILE_CSV)
    args = ap.parse_args()
    ensure_dirs()
    df = pd.read_csv(args.demand, usecols=["time", "origin_type"])
    prof = hourly_profile(df)
    prof.to_csv(args.out, index=False)
    plot_profile(prof, os.path.join(REPORTS_DIR, "demand_profile_20190513.png"))
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
