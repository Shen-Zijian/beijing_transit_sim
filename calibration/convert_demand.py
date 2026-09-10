"""步骤 3：匹配后的刷卡记录 -> 仿真需求文件 data/demand/demand_20190513.csv（+ 小时剖面）。

输出列与旧版 ``concentrated_demand.csv`` 兼容（sid, pid, o, d, o_lon, o_lat, d_lon, d_lat, date, time,
origin_id, origin_type, destination_id, destination_type, distance），并附加验证用列：
observed_tt_s（观测行程时间，地铁已做分钟精度补偿）、o_line、d_line、same_line、route_name。

仅保留两端节点均在所选路网版本（默认 2019）中的记录。

用法::

    python -m calibration.convert_demand [--network 2019|full]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from calibration.common import MATCHED_CSV, NETWORK_DIR, PROCESSED_DIR, ensure_dirs, load_matched, save_json
from calibration.demand_profile import DEMAND_CSV, PROFILE_CSV, hourly_profile, plot_profile
from calibration.common import REPORTS_DIR
from transit_sim.network import split_node_key
from transit_sim.utils import haversine_m


def convert(matched_path: str = MATCHED_CSV, network_version: str = "2019",
            out_path: str = DEMAND_CSV) -> dict:
    ensure_dirs()
    cols = ["card", "mode", "o_line", "d_line", "o_lon", "o_lat", "d_lon", "d_lat", "dep_s", "tt_adj_s",
            "o_key", "d_key", "route_name", "same_line"]
    m = load_matched(matched_path, usecols=cols)
    n0 = len(m)
    m = m.dropna(subset=["o_key", "d_key"])
    m = m[m["o_key"] != m["d_key"]]
    n1 = len(m)

    nodes = pd.read_csv(os.path.join(NETWORK_DIR, f"nodes_{network_version}.csv"))
    valid_keys = set(nodes["node_type"].astype(str) + ":" + nodes["node_id"].astype(str))
    m = m[m["o_key"].isin(valid_keys) & m["d_key"].isin(valid_keys)]
    n2 = len(m)

    o_mode_id = m["o_key"].map(lambda k: split_node_key(k))
    d_mode_id = m["d_key"].map(lambda k: split_node_key(k))
    out = pd.DataFrame({
        "sid": np.arange(len(m), dtype=int),
        "pid": m["card"].values,
        "o": (m["o_lon"].round(2).astype(str) + "," + m["o_lat"].round(2).astype(str)).values,
        "d": (m["d_lon"].round(2).astype(str) + "," + m["d_lat"].round(2).astype(str)).values,
        "o_lon": m["o_lon"].values, "o_lat": m["o_lat"].values,
        "d_lon": m["d_lon"].values, "d_lat": m["d_lat"].values,
        "date": "2019-05-13",
        "time": m["dep_s"].astype(int).values,
        "origin_id": [x[1] for x in o_mode_id],
        "origin_type": [x[0] for x in o_mode_id],
        "destination_id": [x[1] for x in d_mode_id],
        "destination_type": [x[0] for x in d_mode_id],
        "distance": haversine_m(m["o_lat"].values, m["o_lon"].values, m["d_lat"].values, m["d_lon"].values) / 1000.0,
        "observed_tt_s": m["tt_adj_s"].values,
        "o_line": m["o_line"].values, "d_line": m["d_line"].values,
        "same_line": m["same_line"].astype(bool).values,
        "route_name": m["route_name"].values,
    }).sort_values("time").reset_index(drop=True)
    out["sid"] = np.arange(len(out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    prof = hourly_profile(out)
    prof.to_csv(PROFILE_CSV, index=False)
    plot_profile(prof, os.path.join(REPORTS_DIR, "demand_profile_20190513.png"))

    rep = {
        "network_version": network_version, "matched_records": int(n0), "both_nodes": int(n1),
        "in_network_version": int(n2), "output_rows": int(len(out)),
        "by_origin_type": out["origin_type"].value_counts().to_dict(),
        "mean_observed_tt_min": float(out["observed_tt_s"].mean() / 60.0),
        "hourly_trips": prof.set_index("hour")["trips"].to_dict(),
        "unique_od_pairs": int(out[["origin_type", "origin_id", "destination_type", "destination_id"]]
                               .drop_duplicates().shape[0]),
    }
    save_json(rep, os.path.join(PROCESSED_DIR, "convert_demand_report.json"))
    print(f"[demand] 匹配记录 {n0:,} -> 两端有节点 {n1:,} -> 在 {network_version} 路网内 {n2:,}；"
          f"唯一 OD {rep['unique_od_pairs']:,}；写出 {out_path}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", default=MATCHED_CSV)
    ap.add_argument("--network", default="2019")
    ap.add_argument("--out", default=DEMAND_CSV)
    args = ap.parse_args()
    convert(args.matched, args.network, args.out)


if __name__ == "__main__":
    main()
