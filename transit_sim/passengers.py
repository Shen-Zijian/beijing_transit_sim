"""乘客对象。状态机与旧版一致：walking / waiting / onboard / arrived / timeout_exit。

``edge_index`` 指向 ``route['edge_path']`` 中乘客正在执行（步行中 / 候车等待 / 车上乘坐）的那条边。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Passenger:
    id: int
    sid: int
    pid: object
    origin: str
    destination: str
    origin_mode: str
    destination_mode: str
    departure_time: float
    route: Optional[dict] = None
    edge_index: int = 0
    current_node: str = ""
    status: str = "waiting"          # walking | waiting | onboard | arrived | timeout_exit
    state_since: float = 0.0
    state_version: int = 0           # 每次状态变化 +1，用于失效旧事件
    walking_end_time: Optional[float] = None
    walking_kind: str = ""           # access | egress | walk | transfer
    vehicle_id: Optional[int] = None
    boarded_at: Optional[float] = None
    queue_key: Optional[Tuple[str, str]] = None
    arrival_time: Optional[float] = None
    wait_time: float = 0.0
    first_wait_time: float = 0.0        # 首次上车前的候车（公交刷卡口径需扣除）
    walk_time: float = 0.0
    in_vehicle_time: float = 0.0
    access_egress_time: float = 0.0
    num_boardings: int = 0
    num_reroutes: int = 0
    num_timeouts: int = 0
    exit_reason: str = ""
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    @property
    def edge_path(self) -> list:
        return self.route["edge_path"] if self.route else []

    @property
    def current_edge(self) -> Optional[dict]:
        ep = self.edge_path
        return ep[self.edge_index] if 0 <= self.edge_index < len(ep) else None

    def set_status(self, status: str, now: float) -> None:
        self.status = status
        self.state_since = now
        self.state_version += 1

    def main_mode(self) -> str:
        if not self.route:
            return "Walk"
        n_sub = sum(1 for e in self.edge_path if e["mode"] == "subway")
        n_bus = sum(1 for e in self.edge_path if e["mode"] == "bus")
        if n_sub > 0 and n_sub >= n_bus:
            return "Subway"
        if n_bus > 0:
            return "Bus"
        return "Walk"
