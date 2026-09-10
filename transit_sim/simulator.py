"""纯净仿真主循环（单线程、固定步长、事件驱动的车辆与乘客计时）。

每步 ``now``::

    1. 注入本步出发的需求 -> 生成乘客、选路、进入 access 步行 / 候车队列
    2. 到期班次发车（首站上车）
    3. 处理车辆事件：到站(下车+上车) / 发车(再次上车) / 到终点(清空)
    4. 处理乘客事件：步行结束 / 候车超时
    5. 周期性快照；调用 hooks.on_step

扩展点（供后续研究注入策略，核心不含任何策略实现）::

    class MyHooks(SimHooks):
        def on_passenger_created(self, sim, passenger, routes):   # 返回一个 route 即覆盖选择模型
        def on_step(self, sim, now): ...
        def on_trip_end(self, sim, passenger): ...
"""
from __future__ import annotations

import heapq
import json
import os
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .choice import RouteChoice, build_choice_model
from .config import SimConfig
from .demand import Demand, DemandTable
from .metrics import TripRecorder
from .network import TransitNetwork
from .passengers import Passenger
from .routing import Router
from .utils import fmt_hms
from .vehicles import Fleet, HeadwayModel, Vehicle


def make_router(cfg: SimConfig, net: TransitNetwork, headway_model: Optional[HeadwayModel] = None,
                load_cache: bool = True) -> Router:
    """按配置构造与仿真器一致的 Router（预计算脚本与仿真器共用，保证 route dict 完全一致）。"""
    hm = headway_model or HeadwayModel(cfg)
    return Router(
        net, transfer_penalty=cfg.transfer.routing_penalty, k=cfg.choice.k_paths,
        headway_fn=lambda line_id, mode: hm.average(net.lines[line_id]),
        access_egress={"subway": cfg.access_egress.subway, "bus": cfg.access_egress.bus},
        cache_path=cfg.resolve(cfg.routing.cache_file) if load_cache else None,
    )


class SimHooks:
    """默认空钩子。"""

    def on_passenger_created(self, sim: "TransitSimulator", passenger: Passenger, routes: List[dict]) -> Optional[dict]:
        return None

    def on_step(self, sim: "TransitSimulator", now: float) -> None:
        return None

    def on_trip_end(self, sim: "TransitSimulator", passenger: Passenger) -> None:
        return None


@dataclass
class SimulationResult:
    trips: pd.DataFrame
    hourly: pd.DataFrame
    summary: dict
    counters: dict
    config: dict
    runtime_s: float
    output_paths: Dict[str, str] = field(default_factory=dict)


class TransitSimulator:
    def __init__(self, cfg: SimConfig, *, network: Optional[TransitNetwork] = None,
                 router: Optional[Router] = None, choice: Optional[RouteChoice] = None,
                 demand: Optional[DemandTable] = None, hooks: Optional[SimHooks] = None,
                 run_id: Optional[str] = None, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.rng = np.random.default_rng(cfg.time.seed)
        self.net = network or TransitNetwork.from_config(cfg, verbose=verbose)
        self.headway_model = HeadwayModel(cfg)
        self.headway_table = self.headway_model.table
        self.fleet = Fleet(self.net, cfg, headway_model=self.headway_model)
        self.router = router or make_router(cfg, self.net, self.headway_model)
        self.choice = choice or build_choice_model(cfg.choice)
        self.demand = demand or DemandTable.from_csv(
            cfg.resolve(cfg.demand.file), time_step=cfg.time.time_step, fraction=cfg.demand.fraction,
            seed=cfg.time.seed, time_column=cfg.demand.time_column,
            start=cfg.time.sim_start, end=cfg.time.sim_end,
        )
        self.hooks = hooks or SimHooks()
        self.recorder = TripRecorder()

        self.now: float = float(cfg.time.sim_start)
        self.passengers: Dict[int, Passenger] = {}
        self.queues: Dict[Tuple[str, str], deque] = {}
        self._walk_events: List[Tuple[float, int, int]] = []
        self._timeout_events: List[Tuple[float, int, int]] = []
        self._next_pid = 1
        self.counters = {
            "demand_total": len(self.demand), "created": 0, "no_node": 0, "no_route": 0,
            "arrived": 0, "timeout_exit": 0, "timeouts": 0, "reroutes": 0, "unfinished": 0,
        }
        self._last_log = None
        if verbose:
            print(f"[sim] run_id={self.run_id} 需求 {len(self.demand):,} 条 (fraction={cfg.demand.fraction}), "
                  f"容量 subway={cfg.effective_capacity('subway')} bus={cfg.effective_capacity('bus')}, "
                  f"步长 {cfg.time.time_step}s, {fmt_hms(cfg.time.sim_start)}–{fmt_hms(cfg.time.sim_end)}")

    # ------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def average_headway(self, line_id: str, mode: str) -> float:
        """路由用的线路平均发车间隔（服务时段内各小时均值）。"""
        return self.headway_model.average(self.net.lines[line_id])

    # ------------------------------------------------------------ run
    def run(self, save: Optional[bool] = None) -> SimulationResult:
        cfg = self.cfg
        t0 = _time.time()
        self.fleet.initialize(self.now, warm_start=True)
        self.log(f"[sim] 热启动车辆: subway={self.fleet.occupancy()['subway_vehicles']} "
                 f"bus={self.fleet.occupancy()['bus_vehicles']}")
        dt = cfg.time.time_step
        end = cfg.time.sim_end
        next_log = self.now
        while self.now < end:
            now = self.now
            self.step(now)
            if now >= next_log:
                self._snapshot(now)
                next_log += cfg.output.log_every
            self.now = now + dt
        self._finalize(self.now)
        runtime = _time.time() - t0
        summary = self.recorder.summary()
        summary["runtime_s"] = runtime
        summary["counters"] = dict(self.counters)
        summary["router"] = dict(self.router.stats)
        result = SimulationResult(
            trips=self.recorder.trips_df(), hourly=self.recorder.snapshots_df(), summary=summary,
            counters=dict(self.counters), config=cfg.to_dict(), runtime_s=runtime,
        )
        if save is None:
            save = cfg.output.save_trips
        if save:
            out_dir = os.path.join(cfg.resolve(cfg.output.dir), self.run_id)
            result.output_paths = self.recorder.save(out_dir)
            cfg.save(os.path.join(out_dir, "config.yaml"))
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            if self.router.cache_path:
                self.router.save_cache()
            self.log(f"[sim] 结果已保存到 {out_dir}")
        self.log(f"[sim] 完成: 创建 {self.counters['created']:,} / 到达 {self.counters['arrived']:,} / "
                 f"超时退出 {self.counters['timeout_exit']:,} / 未完成 {self.counters['unfinished']:,}; "
                 f"用时 {runtime:.1f}s")
        return result

    def step(self, now: float) -> None:
        for d in self.demand.pop(now):
            self._create_passenger(d, now)
        self.fleet.dispatch(now, on_departure=self._board)
        self.fleet.process_events(now, self._on_arrival, self._on_departure, self._on_terminal)
        self._process_walk_events(now)
        self._process_timeout_events(now)
        self.hooks.on_step(self, now)

    # ------------------------------------------------------------ passengers
    def _create_passenger(self, d: Demand, now: float) -> Optional[Passenger]:
        o = self.net.resolve_node(d.origin, d.origin_mode)
        dst = self.net.resolve_node(d.destination, d.destination_mode)
        if o is None or dst is None or o == dst:
            self.counters["no_node"] += 1
            return None
        routes = self.router.get_routes(o, dst)
        p = Passenger(
            id=self._next_pid, sid=d.sid, pid=d.pid, origin=o, destination=dst,
            origin_mode=self.net.stations[o].mode, destination_mode=self.net.stations[dst].mode,
            departure_time=d.time, current_node=o, state_since=now, extra=d.extra,
        )
        self._next_pid += 1
        route = self.hooks.on_passenger_created(self, p, routes)
        if route is None:
            route = self.choice.select(routes, self.rng, passenger=p, now=now)
        if route is None or not route.get("edge_path"):
            self.counters["no_route"] += 1
            return None
        p.route = route
        p.edge_index = 0
        self.passengers[p.id] = p
        self.counters["created"] += 1
        access = getattr(self.cfg.access_egress, p.origin_mode, 0.0)
        if access > 0:
            self._start_walking(p, now, access, "access")
        else:
            self._advance(p, now)
        return p

    def _advance(self, p: Passenger, now: float) -> None:
        """根据 edge_index 指向的边决定下一步：步行 / 候车 / 完成。"""
        edge = p.current_edge
        if edge is None:
            self._finish(p, now)
            return
        if edge["mode"] in ("walk", "transfer"):
            self._start_walking(p, now, edge["travel_time"], edge["mode"])
            return
        key = (edge["from_node"], edge["line"])
        p.set_status("waiting", now)
        p.queue_key = key
        self.queues.setdefault(key, deque()).append(p)
        heapq.heappush(self._timeout_events, (now + self.cfg.passenger.max_wait_time, p.state_version, p.id))

    def _start_walking(self, p: Passenger, now: float, duration: float, kind: str) -> None:
        p.set_status("walking", now)
        p.walking_kind = kind
        p.walking_end_time = now + max(0.0, float(duration))
        heapq.heappush(self._walk_events, (p.walking_end_time, p.state_version, p.id))

    def _finish(self, p: Passenger, now: float) -> None:
        egress = getattr(self.cfg.access_egress, p.destination_mode, 0.0)
        if egress > 0 and p.walking_kind != "egress":
            self._start_walking(p, now, egress, "egress")
            return
        p.set_status("arrived", now)
        p.arrival_time = now
        self.counters["arrived"] += 1
        self.recorder.record(p, now)
        self.passengers.pop(p.id, None)
        self.hooks.on_trip_end(self, p)

    def _exit(self, p: Passenger, now: float, reason: str) -> None:
        p.set_status("timeout_exit", now)
        p.exit_reason = reason
        p.arrival_time = None
        self.counters["timeout_exit"] += 1
        self.recorder.record(p, now)
        self.passengers.pop(p.id, None)
        self.hooks.on_trip_end(self, p)

    def _process_walk_events(self, now: float) -> None:
        ev = self._walk_events
        while ev and ev[0][0] <= now:
            _, ver, pid = heapq.heappop(ev)
            p = self.passengers.get(pid)
            if p is None or p.status != "walking" or p.state_version != ver:
                continue
            dur = (p.walking_end_time or now) - p.state_since
            kind = p.walking_kind
            if kind == "access":
                p.access_egress_time += dur
                p.walking_kind = ""
                self._advance(p, now)
            elif kind == "egress":
                p.access_egress_time += dur
                self._finish(p, now)
            else:
                p.walk_time += dur
                edge = p.current_edge
                if edge is not None:
                    p.current_node = edge["to_node"]
                    p.edge_index += 1
                p.walking_kind = ""
                self._advance(p, now)

    def _process_timeout_events(self, now: float) -> None:
        ev = self._timeout_events
        while ev and ev[0][0] <= now:
            _, ver, pid = heapq.heappop(ev)
            p = self.passengers.get(pid)
            if p is None or p.status != "waiting" or p.state_version != ver:
                continue
            self._handle_timeout(p, now)

    def _handle_timeout(self, p: Passenger, now: float) -> None:
        p.num_timeouts += 1
        self.counters["timeouts"] += 1
        p.wait_time += now - p.state_since
        action = self.cfg.passenger.timeout_action
        if action == "reroute" and p.num_reroutes < self.cfg.passenger.max_reroutes:
            if self._reroute(p, now):
                return
            self._exit(p, now, "reroute_failed")
            return
        self._exit(p, now, "max_wait_time_exceeded")

    def _reroute(self, p: Passenger, now: float) -> bool:
        if p.current_node == p.destination:
            self._finish(p, now)
            return True
        routes = self.router.get_routes(p.current_node, p.destination)
        if not routes:
            return False
        waiting_line = p.queue_key[1] if p.queue_key else None
        alts = [r for r in routes if not r["lines"] or r["lines"][0] != waiting_line] or routes
        route = self.choice.select(alts, self.rng, passenger=p, now=now)
        if route is None:
            return False
        p.route = route
        p.edge_index = 0
        p.queue_key = None
        p.num_reroutes += 1
        self.counters["reroutes"] += 1
        self._advance(p, now)
        return True

    # ------------------------------------------------------------ vehicles
    def _board(self, v: Vehicle, station: str, now: float) -> None:
        key = (station, v.line_id)
        q = self.queues.get(key)
        if not q:
            return
        while q and v.remaining_capacity > 0:
            p = q[0]
            if p.status != "waiting" or p.queue_key != key:
                q.popleft()
                continue
            q.popleft()
            spell = now - p.state_since
            p.wait_time += spell
            if p.num_boardings == 0:
                p.first_wait_time = spell
            p.set_status("onboard", now)
            p.queue_key = None
            p.vehicle_id = v.id
            p.boarded_at = now
            p.num_boardings += 1
            v.passengers.append(p)

    def _on_arrival(self, v: Vehicle, station: str, now: float) -> None:
        if v.passengers:
            staying = []
            for p in v.passengers:
                edge = p.current_edge
                if edge is not None and edge["to_node"] == station:
                    p.edge_index += 1
                    p.current_node = station
                    nxt = p.current_edge
                    if (nxt is not None and nxt["mode"] == v.mode and nxt["line"] == v.line_id
                            and nxt["from_node"] == station):
                        staying.append(p)      # 继续乘坐同一车辆
                        continue
                    self._alight(p, v, now)
                else:
                    staying.append(p)
            v.passengers = staying
        self._board(v, station, now)

    def _on_departure(self, v: Vehicle, station: str, now: float) -> None:
        self._board(v, station, now)

    def _on_terminal(self, v: Vehicle, station: str, now: float) -> None:
        for p in v.passengers:
            edge = p.current_edge
            if edge is not None and edge["to_node"] == station:
                p.edge_index += 1
            p.current_node = station
            p.in_vehicle_time += now - (p.boarded_at if p.boarded_at is not None else now)
            p.vehicle_id = None
            p.boarded_at = None
            nxt = p.current_edge
            if nxt is not None and nxt["from_node"] != station:
                # 路径与实际位置不一致（车辆在乘客预期下车前到达终点），重新规划
                if not self._reroute(p, now):
                    self._exit(p, now, "path_mismatch")
            else:
                self._advance(p, now)
        v.passengers = []

    def _alight(self, p: Passenger, v: Vehicle, now: float) -> None:
        p.in_vehicle_time += now - (p.boarded_at if p.boarded_at is not None else now)
        p.vehicle_id = None
        p.boarded_at = None
        self._advance(p, now)

    # ------------------------------------------------------------ stats
    def _snapshot(self, now: float) -> None:
        by_status = {"walking": 0, "waiting": 0, "onboard": 0}
        wait_sum = 0.0
        for p in self.passengers.values():
            by_status[p.status] = by_status.get(p.status, 0) + 1
            if p.status == "waiting":
                wait_sum += now - p.state_since
        occ = self.fleet.occupancy()
        stats = {
            "active": len(self.passengers),
            **by_status,
            "waiting_avg_min": (wait_sum / by_status["waiting"] / 60.0) if by_status["waiting"] else 0.0,
            "created": self.counters["created"],
            "arrived": self.counters["arrived"],
            "timeout_exit": self.counters["timeout_exit"],
            **occ,
        }
        self.recorder.snapshot(now, **stats)
        self.log(f"[sim] {fmt_hms(now)} 活跃 {stats['active']:>7,} (走 {stats['walking']:,} / 等 {stats['waiting']:,} / "
                 f"乘 {stats['onboard']:,}) 平均候车 {stats['waiting_avg_min']:.1f}min | 已到达 {stats['arrived']:,} "
                 f"超时 {stats['timeout_exit']:,} | 车辆 subway {occ['subway_vehicles']} ({occ['subway_occupancy']:.0%}) "
                 f"bus {occ['bus_vehicles']} ({occ['bus_occupancy']:.0%}) | 路由 {self.router.stats['computed']:,} 新算")

    def _finalize(self, now: float) -> None:
        for p in list(self.passengers.values()):
            if p.status == "waiting":
                p.wait_time += now - p.state_since
            elif p.status == "onboard" and p.boarded_at is not None:
                p.in_vehicle_time += now - p.boarded_at
            elif p.status == "walking":
                p.walk_time += now - p.state_since
            p.status = "unfinished"
            self.counters["unfinished"] += 1
            self.recorder.record(p, now)
        self.passengers.clear()
