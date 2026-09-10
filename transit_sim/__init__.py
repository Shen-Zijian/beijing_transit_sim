"""beijing_transit_sim —— 纯净的北京多模式公交/地铁仿真核心。

主要入口::

    from transit_sim import SimConfig, TransitNetwork, Router, TransitSimulator
    cfg = SimConfig.load("config/calibrated_20190513.yaml")
    sim = TransitSimulator(cfg)
    result = sim.run()
"""
from .config import SimConfig, parse_set_args
from .network import TransitNetwork, SpeedTable, HeadwayTable, Line, Station, Edge
from .routing import Router
from .choice import build_choice_model, RouteChoice, ShortestChoice, LogitChoice
from .demand import DemandTable, Demand
from .metrics import TripRecorder
from .simulator import TransitSimulator, SimulationResult, SimHooks

__all__ = [
    "SimConfig", "parse_set_args", "TransitNetwork", "SpeedTable", "HeadwayTable", "Line", "Station",
    "Edge", "Router", "build_choice_model", "RouteChoice", "ShortestChoice", "LogitChoice",
    "DemandTable", "Demand", "TripRecorder", "TransitSimulator", "SimulationResult", "SimHooks",
]
__version__ = "0.1.0"
