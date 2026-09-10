import os

import pytest

from transit_sim import SimHooks, TransitSimulator


@pytest.fixture(scope="module")
def cfg_smoke(cfg_2019):
    cfg = cfg_2019.copy()
    if not os.path.exists(cfg.resolve(cfg.demand.file)):
        pytest.skip("缺少需求文件，请先运行 calibration.run_all")
    cfg.demand.fraction = 0.01
    cfg.time.sim_start = 7 * 3600
    cfg.time.sim_end = 8 * 3600
    cfg.output.save_trips = False
    cache = cfg.resolve(cfg.routing.cache_file)
    if not (cache and os.path.exists(cache)):
        cfg.routing.cache_file = None   # 无缓存时在线计算（慢但可运行）
        cfg.demand.fraction = 0.002
    return cfg


def _run(cfg, hooks=None):
    sim = TransitSimulator(cfg, hooks=hooks, run_id="pytest", verbose=False)
    return sim.run(save=False)


def test_smoke_run_completes(cfg_smoke):
    res = _run(cfg_smoke)
    assert res.counters["created"] > 0
    assert res.counters["arrived"] > 0
    trips = res.trips
    assert set(trips["status"].unique()) <= {"arrived", "timeout_exit", "unfinished"}
    done = trips[trips["status"] == "arrived"]
    assert (done["total_time"] > 0).all()
    assert (done["in_vehicle_time"] + done["walk_time"] + done["wait_time"] + done["access_egress_time"]
            <= done["total_time"] + cfg_smoke.time.time_step + 1e-6).all()
    assert (done["num_boardings"] >= 1).all() or (done["walk_time"] > 0).any()


def test_deterministic_with_seed(cfg_smoke):
    a = _run(cfg_smoke)
    b = _run(cfg_smoke)
    assert a.counters == b.counters
    assert a.trips["total_time"].sum() == b.trips["total_time"].sum()


def test_hooks_route_override(cfg_smoke):
    class Force(SimHooks):
        def __init__(self):
            self.n = 0

        def on_passenger_created(self, sim, passenger, routes):
            self.n += 1
            return routes[-1] if routes else None   # 强制选最后一条候选

    h = Force()
    res = _run(cfg_smoke, hooks=h)
    assert h.n == res.counters["created"] + res.counters["no_route"]


def test_capacity_scaling(cfg_smoke):
    assert cfg_smoke.effective_capacity("subway") == max(1, round(cfg_smoke.capacity_real.subway * cfg_smoke.demand.fraction))
