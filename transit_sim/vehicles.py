"""车辆与车队：按 (线路, 小时) 发车间隔调度，事件驱动地沿线路运行。

车辆状态机（与旧版 ``_update_subway_batch`` / ``_update_bus_batch`` 语义一致）::

    running --到达--> dwell --停站结束--> running ... --到达终点--> 回收

差别：运行/停站时间来自线路级速度表与停站时间表；乘客可在到站时和发车前两次上车；
仿真开始时按发车间隔"热启动"，避免线路上空车的初始暂态。
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import SimConfig
from .network import HeadwayTable, Line, TransitNetwork


@dataclass
class Vehicle:
    id: int
    line_id: str
    mode: str
    capacity: int
    stations: List[str]
    seg_times: List[float]
    dwell: float
    dispatched_at: float
    stop_index: int = 0                 # 当前/最近到达的站序
    status: str = "running"             # running | dwell
    next_event_time: float = 0.0
    passengers: list = field(default_factory=list)

    @property
    def current_station(self) -> str:
        return self.stations[self.stop_index]

    @property
    def next_station(self) -> Optional[str]:
        return self.stations[self.stop_index + 1] if self.stop_index + 1 < len(self.stations) else None

    @property
    def at_terminal(self) -> bool:
        return self.stop_index >= len(self.stations) - 1

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - len(self.passengers)

    def fast_forward(self, now: float) -> bool:
        """无乘客推进到 now；返回 False 表示已到终点（应被丢弃）。"""
        while self.next_event_time <= now:
            if self.status == "running":
                self.stop_index += 1
                if self.at_terminal:
                    return False
                self.status = "dwell"
                self.next_event_time += self.dwell
            else:
                self.status = "running"
                self.next_event_time += self.seg_times[self.stop_index]
        return True


class HeadwayModel:
    """发车间隔查询：线路级/模式级分小时表 -> 配置中的模式级（常数或分小时字典）。"""

    def __init__(self, cfg: SimConfig, table: Optional[HeadwayTable] = None):
        self.cfg = cfg
        self.table = table or HeadwayTable.from_csv(cfg.resolve(cfg.network.line_headways_file))
        self._avg_cache: Dict[str, float] = {}

    def headway(self, line: Line, now: float) -> float:
        hour = int(now) % 86400 // 3600
        h = self.table.get(line.id, line.mode, hour)
        if h is None:
            spec = getattr(self.cfg.headways, line.mode)
            if isinstance(spec, dict):
                h = spec.get(hour)
                if h is None:
                    keys = sorted(spec)
                    h = spec[min(keys, key=lambda k: abs(k - hour))] if keys else 600.0
            else:
                h = float(spec)
        return max(float(h), float(self.cfg.time.time_step))

    def average(self, line: Line) -> float:
        """服务时段内各小时班距的均值（路由/选择模型中的期望等待用）。"""
        h = self._avg_cache.get(line.id)
        if h is None:
            lo, hi = getattr(self.cfg.service_hours, line.mode)
            hours = range(int(lo), int(-(-hi // 1)))
            vals = [self.headway(line, hh * 3600.0) for hh in hours] or [600.0]
            h = float(sum(vals) / len(vals))
            self._avg_cache[line.id] = h
        return h

    def in_service(self, mode: str, now: float) -> bool:
        lo, hi = getattr(self.cfg.service_hours, mode)
        h = (now % 86400) / 3600.0
        return lo <= h < hi


class Fleet:
    """管理所有线路的发车与车辆事件。"""

    def __init__(self, network: TransitNetwork, cfg: SimConfig,
                 headway_table: Optional[HeadwayTable] = None,
                 headway_model: Optional[HeadwayModel] = None):
        self.net = network
        self.cfg = cfg
        self.headway_model = headway_model or HeadwayModel(cfg, headway_table)
        self.headways = self.headway_model.table
        self.vehicles: Dict[int, Vehicle] = {}
        self._vid = itertools.count(1)
        self._events: List[Tuple[float, int]] = []       # (time, vehicle_id)
        self._dispatch_heap: List[Tuple[float, str]] = [] # (next_departure_time, line_id)
        self.last_departure: Dict[str, float] = {}
        self.n_dispatched = {"subway": 0, "bus": 0}
        self._capacity = {m: cfg.effective_capacity(m) for m in ("subway", "bus")}

    # ------------------------------------------------------------ headway
    def headway(self, line: Line, now: float) -> float:
        return self.headway_model.headway(line, now)

    def in_service(self, mode: str, now: float) -> bool:
        return self.headway_model.in_service(mode, now)

    # ------------------------------------------------------------ dispatch
    def initialize(self, now: float, warm_start: bool = True) -> None:
        for line in self.net.lines.values():
            if line.n_stops < 2:
                continue
            if warm_start and self.in_service(line.mode, now):
                self._warm_start_line(line, now)
            else:
                start = self._service_start(line.mode, now)
                heapq.heappush(self._dispatch_heap, (start, line.id))

    def _service_start(self, mode: str, now: float) -> float:
        """下一个服务开始时刻（>= now）；若已在服务时段内则返回 now。"""
        lo, hi = getattr(self.cfg.service_hours, mode)
        day = (now // 86400) * 86400
        h = (now - day) / 3600.0
        if lo <= h < hi:
            return now
        start = day + lo * 3600.0
        if start < now:
            start += 86400.0
        return start

    def _warm_start_line(self, line: Line, now: float) -> None:
        """把 now 之前按间隔发出的、尚未到终点的车辆放到线路上。"""
        h = self.headway(line, now)
        hour = (now % 86400) / 3600.0
        seg = self.net.segment_times(line, hour=hour)
        dwell = self.net.line_dwell(line.id, line.mode, hour)
        hold = self.net.line_terminal_hold(line.id, line.mode, hour)
        total = hold + sum(seg) + dwell * max(0, len(seg) - 1)
        t = now
        # 不受服务开始时刻限制：服务时段内的任一时刻，整条线路上都应已有按间隔分布的车辆
        # （相当于"首班车同时从沿线各点发出"，避免长线路末端在开始后数小时内无车）
        while t > now - total:
            v = self._create_vehicle(line, t, seg, dwell, hold)
            if v.fast_forward(now):
                self.vehicles[v.id] = v
                heapq.heappush(self._events, (v.next_event_time, v.id))
            t -= h
        self.last_departure[line.id] = now
        heapq.heappush(self._dispatch_heap, (now + h, line.id))

    def _create_vehicle(self, line: Line, depart: float, seg: Optional[List[float]] = None,
                        dwell: Optional[float] = None, hold: Optional[float] = None) -> Vehicle:
        """在首站生成车辆：先以 dwell 状态停留 hold 秒（首站上车），再按段运行。"""
        hour = (depart % 86400) / 3600.0
        seg = seg if seg is not None else self.net.segment_times(line, hour=hour)
        dwell = dwell if dwell is not None else self.net.line_dwell(line.id, line.mode, hour)
        hold = hold if hold is not None else self.net.line_terminal_hold(line.id, line.mode, hour)
        v = Vehicle(
            id=next(self._vid), line_id=line.id, mode=line.mode,
            capacity=self._capacity[line.mode], stations=line.stations,
            seg_times=seg, dwell=dwell, dispatched_at=depart,
            stop_index=0, status="dwell", next_event_time=depart + max(0.0, hold),
        )
        self.n_dispatched[line.mode] += 1
        return v

    def dispatch(self, now: float, on_departure: Optional[Callable[[Vehicle, str, float], None]] = None) -> int:
        """发出所有到期的班次；返回本步发车数。"""
        n = 0
        while self._dispatch_heap and self._dispatch_heap[0][0] <= now:
            t, line_id = heapq.heappop(self._dispatch_heap)
            line = self.net.lines[line_id]
            if not self.in_service(line.mode, now):
                nxt = self._service_start(line.mode, now + self.cfg.time.time_step)
                heapq.heappush(self._dispatch_heap, (nxt, line_id))
                continue
            if line_id not in self.last_departure:
                # 当日首次发车：整条线路热启动
                self._warm_start_line(line, now)
                n += 1
                continue
            v = self._create_vehicle(line, now)
            self.vehicles[v.id] = v
            self.last_departure[line_id] = now
            if on_departure is not None:
                on_departure(v, v.current_station, now)   # 首站上车
            heapq.heappush(self._events, (v.next_event_time, v.id))
            heapq.heappush(self._dispatch_heap, (now + self.headway(line, now), line_id))
            n += 1
        return n

    # ------------------------------------------------------------ events
    def process_events(self, now: float,
                       on_arrival: Callable[[Vehicle, str, float], None],
                       on_departure: Callable[[Vehicle, str, float], None],
                       on_terminal: Callable[[Vehicle, str, float], None]) -> None:
        ev = self._events
        while ev and ev[0][0] <= now:
            t, vid = heapq.heappop(ev)
            v = self.vehicles.get(vid)
            if v is None:
                continue
            if v.status == "running":
                v.stop_index += 1
                station = v.current_station
                if v.at_terminal:
                    on_terminal(v, station, now)
                    del self.vehicles[vid]
                    continue
                v.status = "dwell"
                on_arrival(v, station, now)
                v.next_event_time = t + v.dwell
            else:
                station = v.current_station
                on_departure(v, station, now)
                v.status = "running"
                v.next_event_time = t + v.seg_times[v.stop_index]
            heapq.heappush(ev, (v.next_event_time, vid))

    # ------------------------------------------------------------ stats
    def occupancy(self) -> Dict[str, float]:
        out = {}
        for mode in ("subway", "bus"):
            vs = [v for v in self.vehicles.values() if v.mode == mode]
            cap = sum(v.capacity for v in vs)
            out[f"{mode}_vehicles"] = len(vs)
            out[f"{mode}_onboard"] = sum(len(v.passengers) for v in vs)
            out[f"{mode}_occupancy"] = (out[f"{mode}_onboard"] / cap) if cap else 0.0
        return out
