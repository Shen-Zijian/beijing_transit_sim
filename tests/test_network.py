import json
import os

import pandas as pd


def test_network_counts_match_prune_report(net_2019, project_root):
    rep_path = os.path.join(project_root, "data", "processed", "prune_network_2019_report.json")
    s = net_2019.summary()
    if os.path.exists(rep_path):
        rep = json.load(open(rep_path, encoding="utf-8"))
        assert s["subway_stations"] == rep["subway_stations_2019"]
        assert s["bus_stops"] == rep["bus_stops_2019"]
        assert s["subway_lines"] == rep["subway_kept"]
        assert s["bus_lines"] == rep["bus_kept"]
    # 与源 CSV 的一致性：每条车辆边都被建成线路段
    edges = pd.read_csv(net_2019.edges_file)
    n_vehicle_edges_csv = int(edges["edge_type"].isin(["subway", "bus"]).sum())
    assert s["vehicle_edges"] == n_vehicle_edges_csv


def test_line_sequences_are_contiguous(net_2019):
    for line in net_2019.lines.values():
        assert len(line.stations) == len(line.seg_distance) + 1
        assert all(d >= 0 for d in line.seg_distance)
        if line.is_loop:
            assert line.stations[0] == line.stations[-1]
        # 站序上相邻站之间必须有该线路的车辆边
        for i in range(len(line.stations) - 1):
            edges = [e for e in net_2019.adj[line.stations[i]] if e.line == line.id and e.to == line.stations[i + 1]]
            assert edges, f"{line.id} 缺少段 {i}"


def test_shared_id_stations_have_intermodal_link(cfg_2019):
    """完整路网中 28 个地铁/公交共用 node_id 的同名站必须有跨模式步行边。"""
    from transit_sim import TransitNetwork
    cfg = cfg_2019.copy()
    cfg.network.version = "full"
    if not os.path.exists(cfg.nodes_file):
        import pytest
        pytest.skip("缺少完整路网文件")
    net = TransitNetwork.from_config(cfg, verbose=False)
    shared = [raw for raw, keys in net._raw_index.items() if len(keys) == 2]
    assert len(shared) == 28
    for raw in shared:
        a, b = net._raw_index[raw]
        assert any(e.to == b and e.mode == "walk" for e in net.adj[a])
        assert any(e.to == a and e.mode == "walk" for e in net.adj[b])


def test_no_post_2019_subway_lines(net_2019):
    for bad in ("地铁3号线", "地铁12号线", "地铁13A号线", "地铁13B号线", "地铁16号线", "地铁17号线", "地铁19号线",
                "大兴国际机场线"):
        assert not any(name.startswith(bad) or bad in name for name in net_2019.lines), bad
    # 2019 年不存在的车站不应出现
    names = {s.name for s in net_2019.stations.values() if s.mode == "subway"}
    for gone in ("二里沟", "红庙", "金鱼胡同", "丽泽商务区", "北太平庄"):
        assert gone not in names, gone
    assert "动物园" in names


def test_speed_and_headway_tables(net_2019, cfg_2019):
    from transit_sim.vehicles import HeadwayModel
    hm = HeadwayModel(cfg_2019)
    for line in list(net_2019.lines.values())[:50]:
        v = net_2019.line_speed(line.id, line.mode, 8.5)
        assert 4.0 <= v <= 90.0
        seg = net_2019.segment_times(line, 8.5)
        assert all(t > 0 for t in seg)
        h = hm.headway(line, 8.5 * 3600)
        assert cfg_2019.time.time_step <= h <= 3600
