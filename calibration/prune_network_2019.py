"""步骤 2：生成与 2019-05 刷卡数据一致的路网版本 nodes_2019.csv / edges_2019.csv。

规则（数据驱动 + 少量已知事实）：

1. 地铁：剔除 2019-05 尚未开通的线路（3/11/12/13A/13B/17/19/22/28 号线、大兴机场线）；
   剔除全天刷卡次数 < ``--min-subway-taps`` 的车站（即 2019 年不存在的站，如 16 号线南段、14 号线中段、
   19/12/17 号线新站等）；剔除车站后把线路切成连续段，保留 >= 2 站的段（多段时以 _seg1/_seg2 命名）。
2. 公交：仅保留在刷卡数据中有匹配行程（任一方向 >= ``--min-bus-trips``）的线路，保留其全部站点。
3. 换乘边：仅保留两端节点都保留的边。

用法::

    python -m calibration.prune_network_2019
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

from calibration.common import (MATCHED_FULL_CSV, NETWORK_DIR, PROCESSED_DIR, ensure_dirs, line_variant_key,
                                load_matched, save_json)
from transit_sim.network import split_node_key

# 2019-05-13 尚未开通的地铁线路（按 route_name 正则匹配）；16 号线当时仅北安河--西苑段，全部在 bbox 之外
NOT_IN_2019_PATTERNS = [
    r"^地铁3号线", r"^地铁11号线", r"^地铁12号线", r"^地铁13A号线", r"^地铁13B号线", r"^地铁16号线",
    r"^地铁17号线", r"^地铁19号线", r"^地铁22号线", r"^地铁28号线", r"大兴国际机场线", r"^大兴机场",
]
# 2019 年后在既有线路上加开的中间站：当时列车通过不停，线路应跨站接续（合并两段距离）
PASS_THROUGH_2019 = {"二里沟", "红庙", "陶然桥"}
# 2019 年已存在但刷卡数据中无记录的车站（保留）
KEEP_DESPITE_NO_TAPS = {"动物园"}
# 某线路上 2019 年尚未通到的既有车站（车站本身因其他线路而存在）：从该线路的站序中剔除
LINE_STATION_EXCLUDE = {
    "地铁8号线": {"王府井", "前门", "金鱼胡同"},        # 8 号线 中国美术馆--珠市口 段 2021-12 开通
    "地铁14号线": {"景风门", "西铁营", "菜户营", "丽泽商务区", "东管头"},  # 14 号线中段(北京南站--西局) 2021-12 开通
    "首都机场线": {"北新桥"},                            # 机场线西延至北新桥 2021-12 开通
}


def _not_in_2019(route_name: str) -> bool:
    return any(re.search(p, route_name) for p in NOT_IN_2019_PATTERNS)


def _excluded_on_line(route_name: str, station_name: str) -> bool:
    for prefix, names in LINE_STATION_EXCLUDE.items():
        if route_name.startswith(prefix) and station_name in names:
            return True
    return False


def prune(matched_path: str = MATCHED_FULL_CSV, min_subway_taps: int = 20, min_bus_trips: int = 10,
          out_nodes: str | None = None, out_edges: str | None = None) -> dict:
    ensure_dirs()
    nodes = pd.read_csv(os.path.join(NETWORK_DIR, "nodes_full.csv"))
    edges = pd.read_csv(os.path.join(NETWORK_DIR, "edges_full.csv"))
    m = load_matched(matched_path, usecols=["mode", "o_key", "d_key", "route_name", "same_line"])

    # ---- 地铁车站证据
    sub = m[m["mode"] == "subway"]
    taps = pd.concat([sub["o_key"], sub["d_key"]]).dropna().value_counts()
    observed_subway = {split_node_key(k)[1] for k, n in taps.items() if n >= min_subway_taps}
    subway_nodes = nodes[nodes["node_type"] == "subway"]
    name_of = dict(zip(subway_nodes["node_id"].astype(str), subway_nodes["station_name"].astype(str)))
    observed_subway |= {nid for nid, nm in name_of.items() if nm in KEEP_DESPITE_NO_TAPS}
    all_subway = set(subway_nodes["node_id"].astype(str))
    dropped_subway_stations = sorted(all_subway - observed_subway)

    # ---- 公交线路证据（按变体键聚合两个方向）
    bus = m[(m["mode"] == "bus") & m["route_name"].notna()]
    trips_by_route = bus["route_name"].value_counts()
    trips_by_line = {}
    for rn, n in trips_by_route.items():
        trips_by_line[line_variant_key(rn)] = trips_by_line.get(line_variant_key(rn), 0) + int(n)

    kept_edge_rows = []
    line_report = {"subway_dropped_not_in_2019": [], "subway_split": {}, "subway_dropped_short": [],
                   "bus_dropped_no_trips": [], "bus_kept": 0, "subway_kept": 0}

    for mode in ("subway", "bus"):
        sub_e = edges[edges["edge_type"] == mode]
        for route_name, grp in sub_e.groupby("route_name", sort=False):
            grp = grp.sort_index()
            if mode == "subway":
                if _not_in_2019(route_name):
                    line_report["subway_dropped_not_in_2019"].append(route_name)
                    continue
                # 逐边扫描：车站分三类 —— 保留 / 通过不停(合并到下一段) / 不存在(切断)
                def _status(nid: str) -> str:
                    nm = name_of.get(nid, "")
                    if _excluded_on_line(route_name, nm):
                        return "cut"
                    if nid in observed_subway:
                        return "keep"
                    if nm in PASS_THROUGH_2019:
                        return "pass"
                    return "cut"

                runs = []
                cur = []
                pending = None   # (from_id, 累积距离, 累积时间) —— 跨通过站合并
                for r in grp.itertuples(index=False):
                    f, t = str(r.from_id), str(r.to_id)
                    sf, st = _status(f), _status(t)
                    if sf == "cut":
                        pending = None
                        if cur:
                            runs.append(cur)
                            cur = []
                        continue
                    if pending is None:
                        if sf == "pass":
                            continue   # 段起点不能是通过站
                        pending = (f, 0.0, 0.0)
                    f0, dist_acc, tt_acc = pending
                    dist_acc += float(r.distance)
                    tt_acc += float(r.travel_time)
                    if st == "pass":
                        pending = (f0, dist_acc, tt_acc)
                        continue
                    if st == "cut":
                        pending = None
                        if cur:
                            runs.append(cur)
                            cur = []
                        continue
                    d = r._asdict()
                    d["from_id"] = f0
                    d["distance"] = dist_acc
                    d["travel_time"] = tt_acc
                    if cur and str(cur[-1]["to_id"]) != f0:
                        runs.append(cur)
                        cur = []
                    cur.append(d)
                    pending = (t, 0.0, 0.0)
                if cur:
                    runs.append(cur)
                if not runs:
                    line_report["subway_dropped_short"].append(route_name)
                    continue
                if len(runs) > 1:
                    line_report["subway_split"][route_name] = [len(x) + 1 for x in runs]
                for k, run in enumerate(runs, 1):
                    name = route_name if len(runs) == 1 else f"{route_name}_seg{k}"
                    for r in run:
                        d = dict(r)
                        d["route_name"] = name
                        kept_edge_rows.append(d)
                line_report["subway_kept"] += len(runs)
            else:
                key = line_variant_key(route_name)
                if trips_by_line.get(key, 0) < min_bus_trips:
                    line_report["bus_dropped_no_trips"].append(route_name)
                    continue
                kept_edge_rows.extend(r._asdict() for r in grp.itertuples(index=False))
                line_report["bus_kept"] += 1

    kept = pd.DataFrame(kept_edge_rows)
    # 保留的节点：出现在保留车辆边中的节点（按模式）
    keep_subway = set(kept[kept["edge_type"] == "subway"][["from_id", "to_id"]].astype(str).values.ravel())
    keep_bus = set(kept[kept["edge_type"] == "bus"][["from_id", "to_id"]].astype(str).values.ravel())
    node_keep = nodes[((nodes["node_type"] == "subway") & nodes["node_id"].astype(str).isin(keep_subway))
                      | ((nodes["node_type"] == "bus") & nodes["node_id"].astype(str).isin(keep_bus))]

    # 换乘边：两端均保留
    def _keep_transfer(r):
        f, t = str(r.from_id), str(r.to_id)
        if r.edge_type == "subway_transfer":
            return f in keep_subway and t in keep_subway
        if r.edge_type == "bus_transfer":
            return f in keep_bus and t in keep_bus
        if r.edge_type == "intermodal_transfer":
            return (f in keep_subway and t in keep_bus) or (f in keep_bus and t in keep_subway)
        return False

    transfers = edges[edges["edge_type"].isin(["subway_transfer", "bus_transfer", "intermodal_transfer"])]
    transfers = transfers[[_keep_transfer(r) for r in transfers.itertuples(index=False)]]
    out_edges_df = pd.concat([kept, transfers], ignore_index=True)[list(edges.columns)]

    out_nodes = out_nodes or os.path.join(NETWORK_DIR, "nodes_2019.csv")
    out_edges = out_edges or os.path.join(NETWORK_DIR, "edges_2019.csv")
    node_keep.to_csv(out_nodes, index=False)
    out_edges_df.to_csv(out_edges, index=False)

    report = {
        "min_subway_taps": min_subway_taps, "min_bus_trips": min_bus_trips,
        "nodes_full": int(len(nodes)), "nodes_2019": int(len(node_keep)),
        "subway_stations_full": len(all_subway), "subway_stations_2019": len(keep_subway),
        "subway_stations_dropped": dropped_subway_stations,
        "subway_station_names_dropped": sorted(nodes[(nodes["node_type"] == "subway")
                                                     & nodes["node_id"].astype(str).isin(dropped_subway_stations)]
                                               ["station_name"].tolist()),
        "bus_stops_full": int((nodes["node_type"] == "bus").sum()), "bus_stops_2019": len(keep_bus),
        "edges_full": int(len(edges)), "edges_2019": int(len(out_edges_df)),
        "edge_type_counts_2019": out_edges_df["edge_type"].value_counts().to_dict(),
        **line_report,
    }
    save_json(report, os.path.join(PROCESSED_DIR, "prune_network_2019_report.json"))
    print(f"[prune] 节点 {report['nodes_full']} -> {report['nodes_2019']}（地铁 {report['subway_stations_full']} -> "
          f"{report['subway_stations_2019']}，公交 {report['bus_stops_full']} -> {report['bus_stops_2019']}）")
    print(f"[prune] 边 {report['edges_full']} -> {report['edges_2019']}: {report['edge_type_counts_2019']}")
    print(f"[prune] 地铁线路(方向) 保留 {report['subway_kept']}，剔除未开通 {len(report['subway_dropped_not_in_2019'])}，"
          f"无有效段 {len(report['subway_dropped_short'])}，切分 {len(report['subway_split'])}；"
          f"公交线路(方向) 保留 {report['bus_kept']}，剔除 {len(report['bus_dropped_no_trips'])}")
    print(f"[prune] 剔除的地铁站: {report['subway_station_names_dropped']}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", default=MATCHED_FULL_CSV)
    ap.add_argument("--min-subway-taps", type=int, default=20)
    ap.add_argument("--min-bus-trips", type=int, default=10)
    args = ap.parse_args()
    prune(args.matched, args.min_subway_taps, args.min_bus_trips)


if __name__ == "__main__":
    main()
