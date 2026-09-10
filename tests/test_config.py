import os

from transit_sim import SimConfig, parse_set_args


def test_load_default_and_override(project_root):
    cfg = SimConfig.load(os.path.join(project_root, "config", "default.yaml"),
                         overrides={"demand.fraction": 0.1, "speeds.subway": "36", "headways.subway": {7: 180, 12: 360}})
    assert cfg.demand.fraction == 0.1
    assert cfg.speeds.subway == 36.0
    assert cfg.headways.subway == {7: 180.0, 12: 360.0}
    assert cfg.nodes_file.endswith(os.path.join("data", "network", "nodes_2019.csv"))
    assert cfg.effective_capacity("subway") == max(1, round(cfg.capacity_real.subway * 0.1))


def test_parse_set_args():
    ov = parse_set_args(["demand.fraction=0.2", "network.line_speeds_file=null", "choice.model=shortest"])
    assert ov == {"demand.fraction": 0.2, "network.line_speeds_file": None, "choice.model": "shortest"}


def test_unknown_key_rejected(project_root):
    import pytest
    with pytest.raises(KeyError):
        SimConfig.from_dict({"nonexistent": 1})
