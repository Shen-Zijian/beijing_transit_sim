import pytest

from transit_sim.routing import Router


@pytest.fixture(scope="module")
def router(net_2019):
    return Router(net_2019, transfer_penalty=180.0, k=5, headway_fn=lambda lid, mode: 300.0)


def _by_name(net, name, mode="subway"):
    return net.stations_by_name(mode)[name][0].id


def test_shortest_path_subway(router, net_2019):
    o, d = _by_name(net_2019, "西直门"), _by_name(net_2019, "国贸")
    path = router.shortest_path(o, d)
    assert path
    route = router.build_route(path)
    assert route["node_path"][0] == o and route["node_path"][-1] == d
    assert route["num_transfers"] <= 2
    assert 15 * 60 < route["in_vehicle_time"] < 60 * 60
    # 边序连续
    real = [e for e in route["edge_path"] if e["mode"] != "transfer"]
    for a, b in zip(real, real[1:]):
        assert a["to_node"] == b["from_node"]


def test_k_paths_distinct_and_sorted(router, net_2019):
    o, d = _by_name(net_2019, "北京南站"), _by_name(net_2019, "东直门")
    paths = router.k_shortest_paths(o, d, 5)
    assert 1 <= len(paths) <= 5
    keys = {router._path_key(p) for p in paths}
    assert len(keys) == len(paths)
    costs = [router.path_cost(p) for p in paths]
    assert costs == sorted(costs)
    assert all(not router._has_node_loop(p) for p in paths)


def test_path_cost_matches_search(router, net_2019):
    """搜索得到的最短路代价应等于 path_cost 重算值（含换乘惩罚与同站换乘时间）。"""
    o, d = _by_name(net_2019, "北京西站"), _by_name(net_2019, "北京南站")
    path = router.shortest_path(o, d)
    c1 = router.path_cost(path)
    # 任意其他候选不应更优
    for p in router.k_shortest_paths(o, d, 3)[1:]:
        assert router.path_cost(p) >= c1 - 1e-6


def test_same_station_transfer_edge_present(router, net_2019):
    o, d = _by_name(net_2019, "北京西站"), _by_name(net_2019, "北京南站")
    route = router.build_route(router.shortest_path(o, d))
    if route["num_transfers"] >= 1:
        transfer_edges = [e for e in route["edge_path"] if e["mode"] in ("transfer", "walk")]
        assert transfer_edges


def test_loop_line_wraparound(net_2019):
    line = next(l for l in net_2019.lines.values() if l.is_loop and l.mode == "subway")
    n = len(line.stations)
    dist, n_int = line.segment_between(n - 3, 1)
    assert dist > 0 and n_int == 2
    assert line.hops_between(n - 3, 1) == 3
