"""仿真配置：YAML -> 嵌套 dataclass。

用法::

    cfg = SimConfig.load("config/default.yaml")
    cfg = SimConfig.load("config/default.yaml", overrides={"demand.fraction": 0.1})
    cfg.time.time_step

所有路径字段若为相对路径，均相对于 ``cfg.root``（默认取配置文件所在目录的上一级，即项目根目录）。
"""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

HeadwaySpec = Union[float, int, Dict[int, float]]


@dataclass
class TimeConfig:
    time_step: int = 10
    sim_start: int = 5 * 3600
    sim_end: int = 24 * 3600
    seed: int = 42


@dataclass
class NetworkConfig:
    version: str = "2019"
    dir: str = "data/network"
    line_speeds_file: Optional[str] = None
    line_headways_file: Optional[str] = None
    transfer_times_file: Optional[str] = None
    nodes_file: Optional[str] = None   # 显式覆盖（否则由 version 推导）
    edges_file: Optional[str] = None


@dataclass
class DemandConfig:
    file: str = "data/demand/demand_20190513.csv"
    fraction: float = 0.05
    time_column: str = "time"


@dataclass
class SpeedConfig:
    subway: float = 35.0
    bus: float = 18.0
    walk: float = 1.2


@dataclass
class DwellConfig:
    subway: float = 30.0
    bus: float = 20.0


@dataclass
class StopDelayConfig:
    """每个站间段附加的固定运行延误 (s)：加减速、进出站、信号等（不含开门停站 dwell）。"""
    subway: float = 0.0
    bus: float = 0.0


@dataclass
class TerminalHoldConfig:
    """首站发车前停留 (s)：首站上车乘客经历该时间。"""
    subway: float = 0.0
    bus: float = 0.0


@dataclass
class HeadwayConfig:
    subway: HeadwaySpec = 300
    bus: HeadwaySpec = 600


@dataclass
class ServiceHoursConfig:
    subway: List[float] = field(default_factory=lambda: [5.0, 23.5])
    bus: List[float] = field(default_factory=lambda: [5.0, 23.0])


@dataclass
class CapacityConfig:
    subway: float = 1460
    bus: float = 90


@dataclass
class SameStationTransferConfig:
    subway: float = 180.0
    bus: float = 30.0
    intermodal: float = 180.0


@dataclass
class TransferConfig:
    same_station: SameStationTransferConfig = field(default_factory=SameStationTransferConfig)
    routing_penalty: float = 180.0


@dataclass
class AccessEgressConfig:
    subway: float = 240.0
    bus: float = 0.0


@dataclass
class PassengerConfig:
    max_wait_time: float = 1500.0
    timeout_action: str = "exit"
    max_reroutes: int = 2


@dataclass
class ChoiceConfig:
    model: str = "logit"
    k_paths: int = 5
    beta_ivt: float = 1.0
    beta_wait: float = 1.5
    beta_walk: float = 1.5
    beta_transfer: float = 300.0
    theta: float = 0.005


@dataclass
class RoutingConfig:
    cache_file: Optional[str] = None


@dataclass
class OutputConfig:
    dir: str = "output"
    log_every: int = 3600
    save_trips: bool = True


@dataclass
class SimConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    speeds: SpeedConfig = field(default_factory=SpeedConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    stop_delay: StopDelayConfig = field(default_factory=StopDelayConfig)
    terminal_hold: TerminalHoldConfig = field(default_factory=TerminalHoldConfig)
    headways: HeadwayConfig = field(default_factory=HeadwayConfig)
    service_hours: ServiceHoursConfig = field(default_factory=ServiceHoursConfig)
    capacity_real: CapacityConfig = field(default_factory=CapacityConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    access_egress: AccessEgressConfig = field(default_factory=AccessEgressConfig)
    passenger: PassengerConfig = field(default_factory=PassengerConfig)
    choice: ChoiceConfig = field(default_factory=ChoiceConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    root: str = "."          # 项目根目录，用于解析相对路径
    source_file: Optional[str] = None

    # ------------------------------------------------------------------ IO
    @classmethod
    def load(cls, path: str, overrides: Optional[Dict[str, Any]] = None,
             root: Optional[str] = None) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if root is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        cfg = cls.from_dict(raw, root=root)
        cfg.source_file = os.path.abspath(path)
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], root: str = ".") -> "SimConfig":
        cfg = cls(root=root)
        _update_dataclass(cfg, raw)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("root", None)
        d.pop("source_file", None)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    def copy(self) -> "SimConfig":
        return copy.deepcopy(self)

    # ------------------------------------------------------------ helpers
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """支持 ``{"demand.fraction": 0.1, "speeds.subway": 36}`` 形式的点路径覆盖。"""
        for key, value in overrides.items():
            parts = key.split(".")
            obj: Any = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            last = parts[-1]
            if dataclasses.is_dataclass(obj):
                cur = getattr(obj, last)
                setattr(obj, last, _coerce_like(cur, value))
            else:
                obj[last] = value

    def resolve(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.root, path))

    @property
    def nodes_file(self) -> str:
        if self.network.nodes_file:
            return self.resolve(self.network.nodes_file)
        return self.resolve(os.path.join(self.network.dir, f"nodes_{self.network.version}.csv"))

    @property
    def edges_file(self) -> str:
        if self.network.edges_file:
            return self.resolve(self.network.edges_file)
        return self.resolve(os.path.join(self.network.dir, f"edges_{self.network.version}.csv"))

    def effective_capacity(self, mode: str) -> int:
        real = getattr(self.capacity_real, mode)
        return max(1, int(round(real * self.demand.fraction)))


# ---------------------------------------------------------------- utils
def _update_dataclass(obj: Any, raw: Dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(obj, key):
            raise KeyError(f"未知配置项: {key}")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(value, dict):
            _update_dataclass(cur, value)
        else:
            setattr(obj, key, _coerce_like(cur, value))


def _coerce_like(current: Any, value: Any) -> Any:
    """按现有字段类型做温和转换（命令行传入的字符串 -> 数值/布尔）。"""
    if isinstance(value, str):
        if isinstance(current, bool):
            return value.lower() in ("1", "true", "yes", "y")
        if isinstance(current, int) and not isinstance(current, bool):
            try:
                return int(value)
            except ValueError:
                return float(value)
        if isinstance(current, float):
            return float(value)
        if value.lower() in ("none", "null", ""):
            return None
    if isinstance(value, dict) and current is not None and not isinstance(current, dict):
        # headway 等字段允许 常数 -> 分小时字典
        return {int(k): float(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {int(k) if str(k).lstrip("-").isdigit() else k: v for k, v in value.items()}
    return value


def parse_set_args(items: Optional[List[str]]) -> Dict[str, Any]:
    """把命令行 ``--set a.b=1 --set c.d=x`` 解析成覆盖字典。"""
    out: Dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set 需要 key=value 形式: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
