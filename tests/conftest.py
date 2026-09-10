import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transit_sim import SimConfig, TransitNetwork  # noqa: E402

CALIBRATED = os.path.join(PROJECT_ROOT, "config", "calibrated_20190513.yaml")
DEFAULT = os.path.join(PROJECT_ROOT, "config", "default.yaml")


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cfg_2019():
    path = CALIBRATED if os.path.exists(CALIBRATED) else DEFAULT
    cfg = SimConfig.load(path)
    if not os.path.exists(cfg.nodes_file):
        pytest.skip("缺少 2019 路网文件，请先运行 calibration.run_all")
    return cfg


@pytest.fixture(scope="session")
def net_2019(cfg_2019):
    return TransitNetwork.from_config(cfg_2019, verbose=False)
