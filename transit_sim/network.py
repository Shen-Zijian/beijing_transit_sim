"""多模式公交网络：站点、线路（含逐段距离/时间）、步行换乘边、线路级速度/发车间隔表。

与旧版 ``simulator_switch.load_multimodal_network`` 语义对应，但：

* 段运行时间 = 段距离 / 线路速度（线路级 > 模式级），可按时段查表；
* 环线保留闭合终点（车辆跑完整圈）；
* 支持平行线路（同一对站点被多条线路服务）——旧版基于 ``nx.DiGraph`` 会互相覆盖；
* 同站换乘（``*_transfer`` 自环边）记为站点属性 ``transfer_time``，不再是图上的自环。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import pandas as pd

from .config import SimConfig
from .utils import classify_period, haversine_scalar_m

VEHICLE_MODES = ("subway", "bus")


def node_key(mode: str, raw_id: object) -> str:
    """图节点键：'subway:BV10000102' / 'bus:BV10000102'。

    源数据中有 28 个 node_id 同时被地铁站和公交站使用（同名同 id），必须按模式区分。
    """
    return f"{mode}:{raw_id}"


def split_node_key(key: str) -> Tuple[str, str]:
    mode, _, raw = key.partition(":")
    return mode, raw


class Edge(NamedTuple):
    frm: str
    to: str
    mode: str                 # 'subway' | 'bus' | 'walk'
    line: Optional[str]       # 车辆边的线路 id；步行边为 None
    travel_time: float        # 秒（车辆边为路由用的静态平均值）
    distance: float           # 米
    seg_index: int = -1       # 车辆边在线路中的段序号（步行边为 -1）


@dataclass
class Station:
    id: str                          # 带模式前缀的节点键，如 'subway:BV10000102'
    raw_id: str                      # 源数据 node_id
    name: str
    mode: str
    lat: float
    lon: float
    lines: List[str] = field(default_factory=list)
    transfer_time: float = 0.0

    @property
    def location(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class Line:
    id: str
    mode: str
    direction: int
    stations: List[str]              # 有序站点 id；环线时 stations[-1] == stations[0]
    seg_distance: List[float]        # 米，len = len(stations) - 1
    is_loop: bool = False
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for i, s in enumerate(self.stations):
            self._index.setdefault(s, i)

    @property
    def n_stops(self) -> int:
        return len(self.stations)

    @property
    def length_m(self) -> float:
        return float(sum(self.seg_distance))

    def index_of(self, station_id: str) -> int:
        return self._index.get(station_id, -1)

    def segment_between(self, i: int, j: int) -> Tuple[float, int]:
        """站序 i -> j 的沿线距离（米）与中间站数；环线允许绕行（j <= i 时跨过闭合点）。"""
        if j > i:
            return float(sum(self.seg_distance[i:j])), j - i - 1
        if self.is_loop and j < i:
            n = len(self.seg_distance)          # = len(stations) - 1，闭合环的段数
            dist = float(sum(self.seg_distance[i:]) + sum(self.seg_distance[:j]))
            return dist, (n - i) + j - 1
        return 0.0, 0

    def hops_between(self, i: int, j: int) -> int:
        if j > i:
            return j - i
        if self.is_loop and j < i:
            return (len(self.seg_distance) - i) + j
        return 0


def _valid(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


class SpeedTable:
    """线路级 / 模式级 速度 / 停站 / 站间附加延误 / 首站停留 表。

    CSV 列: scope(line|mode), key, period(all|am_peak|pm_peak|off_peak|night), speed_kmh
            [, dwell_s, stop_delay_s, terminal_hold_s, n_obs]
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self._speed: Dict[Tuple[str, str, str], float] = {}
        self._dwell: Dict[Tuple[str, str, str], float] = {}
        self._stop_delay: Dict[Tuple[str, str, str], float] = {}
        self._hold: Dict[Tuple[str, str, str], float] = {}
        if df is not None and len(df):
            for r in df.itertuples(index=False):
                key = (str(r.scope), str(r.key), str(getattr(r, "period", "all")))
                for attr, table in (("speed_kmh", self._speed), ("dwell_s", self._dwell),
                                    ("stop_delay_s", self._stop_delay), ("terminal_hold_s", self._hold)):
                    v = getattr(r, attr, None)
                    if _valid(v):
                        table[key] = float(v)

    @classmethod
    def from_csv(cls, path: Optional[str]) -> "SpeedTable":
        if path and os.path.exists(path):
            return cls(pd.read_csv(path))
        return cls()

    def _lookup(self, table, line_id: str, mode: str, period: Optional[str]):
        keys = []
        if period:
            keys.append(("line", line_id, period))
        keys.append(("line", line_id, "all"))
        if period:
            keys.append(("mode", mode, period))
        keys.append(("mode", mode, "all"))
        for k in keys:
            if k in table:
                return table[k]
        return None

    def speed(self, line_id: str, mode: str, period: Optional[str] = None) -> Optional[float]:
        return self._lookup(self._speed, line_id, mode, period)

    def dwell(self, line_id: str, mode: str, period: Optional[str] = None) -> Optional[float]:
        return self._lookup(self._dwell, line_id, mode, period)

    def stop_delay(self, line_id: str, mode: str, period: Optional[str] = None) -> Optional[float]:
        return self._lookup(self._stop_delay, line_id, mode, period)

    def terminal_hold(self, line_id: str, mode: str, period: Optional[str] = None) -> Optional[float]:
        return self._lookup(self._hold, line_id, mode, period)

    def __len__(self):
        return len(self._speed)


class HeadwayTable:
    """线路级 / 模式级分小时发车间隔表。CSV 列: scope(line|mode), key, hour(0-23|all), headway_s[, n_obs]"""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self._hw: Dict[Tuple[str, str, str], float] = {}
        if df is not None and len(df):
            for r in df.itertuples(index=False):
                h = str(r.hour)
                if h not in ("all",):
                    h = str(int(float(h)))
                self._hw[(str(r.scope), str(r.key), h)] = float(r.headway_s)

    @classmethod
    def from_csv(cls, path: Optional[str]) -> "HeadwayTable":
        if path and os.path.exists(path):
            return cls(pd.read_csv(path))
        return cls()

    def get(self, line_id: str, mode: str, hour: Optional[int] = None) -> Optional[float]:
        keys = []
        if hour is not None:
            keys.append(("line", line_id, str(int(hour))))
        keys.append(("line", line_id, "all"))
        if hour is not None:
            keys.append(("mode", mode, str(int(hour))))
        keys.append(("mode", mode, "all"))
        for k in keys:
            if k in self._hw:
                return self._hw[k]
        return None

    def __len__(self):
        return len(self._hw)


class TransitNetwork:
    def __init__(self, nodes_file: str, edges_file: str, *,
                 speeds: Dict[str, float], walk_speed: float,
                 dwell: Dict[str, float],
                 same_station_transfer: Dict[str, float],
                 stop_delay: Optional[Dict[str, float]] = None,
                 terminal_hold: Optional[Dict[str, float]] = None,
                 speed_table: Optional[SpeedTable] = None,
                 transfer_times_file: Optional[str] = None,
                 verbose: bool = True):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.mode_speeds = dict(speeds)          # km/h
        self.walk_speed = float(walk_speed)      # m/s
        self.mode_dwell = dict(dwell)
        self.mode_stop_delay = dict(stop_delay or {m: 0.0 for m in VEHICLE_MODES})
        self.mode_terminal_hold = dict(terminal_hold or {m: 0.0 for m in VEHICLE_MODES})
        self.same_station_transfer = dict(same_station_transfer)
        self.speed_table = speed_table or SpeedTable()
        self.verbose = verbose

        self.stations: Dict[str, Station] = {}
        self.lines: Dict[str, Line] = {}
        self.adj: Dict[str, List[Edge]] = {}
        self.walk_edges: List[Edge] = []
        self.n_vehicle_edges = 0
        self._load(nodes_file, edges_file)
        self._apply_transfer_times(transfer_times_file)

    # ------------------------------------------------------------ 构建
    @classmethod
    def from_config(cls, cfg: SimConfig, verbose: bool = True) -> "TransitNetwork":
        return cls(
            cfg.nodes_file, cfg.edges_file,
            speeds={"subway": cfg.speeds.subway, "bus": cfg.speeds.bus},
            walk_speed=cfg.speeds.walk,
            dwell={"subway": cfg.dwell.subway, "bus": cfg.dwell.bus},
            stop_delay={"subway": cfg.stop_delay.subway, "bus": cfg.stop_delay.bus},
            terminal_hold={"subway": cfg.terminal_hold.subway, "bus": cfg.terminal_hold.bus},
            same_station_transfer={
                "subway": cfg.transfer.same_station.subway,
                "bus": cfg.transfer.same_station.bus,
                "intermodal": cfg.transfer.same_station.intermodal,
            },
            speed_table=SpeedTable.from_csv(cfg.resolve(cfg.network.line_speeds_file)),
            transfer_times_file=cfg.resolve(cfg.network.transfer_times_file),
            verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load(self, nodes_file: str, edges_file: str) -> None:
        nodes = pd.read_csv(nodes_file)
        edges = pd.read_csv(edges_file)
        for r in nodes.itertuples(index=False):
            mode = str(r.node_type)
            if mode not in VEHICLE_MODES:
                continue
            key = node_key(mode, r.node_id)
            self.stations[key] = Station(
                id=key, raw_id=str(r.node_id), name=str(r.station_name), mode=mode,
                lat=float(r.y), lon=float(r.x),
                transfer_time=float(self.same_station_transfer.get(mode, 0.0)),
            )
        for sid in self.stations:
            self.adj[sid] = []
        self._raw_index: Dict[str, List[str]] = {}
        for s in self.stations.values():
            self._raw_index.setdefault(s.raw_id, []).append(s.id)

        # 车辆线路
        broken = 0
        for mode in VEHICLE_MODES:
            sub = edges[edges["edge_type"] == mode]
            for route_name, grp in sub.groupby("route_name", sort=False):
                grp = grp.sort_index()
                stations: List[str] = []
                seg_dist: List[float] = []
                for e in grp.itertuples(index=False):
                    f, t = node_key(mode, e.from_id), node_key(mode, e.to_id)
                    if f not in self.stations or t not in self.stations:
                        continue
                    if not stations:
                        stations.append(f)
                    elif stations[-1] != f:
                        broken += 1
                        # 链断裂：以新的 from 为下一站接续（旧版行为近似）
                        stations.append(f)
                        seg_dist.append(0.0)
                    stations.append(t)
                    seg_dist.append(float(e.distance))
                if len(stations) < 2:
                    continue
                is_loop = stations[0] == stations[-1] and len(stations) > 2
                line = Line(id=str(route_name), mode=mode,
                            direction=int(grp["direction"].iloc[0]) if "direction" in grp else 0,
                            stations=stations, seg_distance=seg_dist, is_loop=is_loop)
                self.lines[line.id] = line
                for s in dict.fromkeys(stations):
                    self.stations[s].lines.append(line.id)
                self._add_line_edges(line)
        if broken:
            self._log(f"[network] 警告: {broken} 处线路边序不连续，已按顺序接续")

        # 步行换乘边（非自环）；自环记为同站换乘时间
        pair_modes = {
            "bus_transfer": [("bus", "bus")],
            "subway_transfer": [("subway", "subway")],
            "intermodal_transfer": [("subway", "bus"), ("bus", "subway")],
        }
        seen_walk = set()
        for et, mode_pairs in pair_modes.items():
            sub = edges[edges["edge_type"] == et]
            for e in sub.itertuples(index=False):
                if str(e.from_id) == str(e.to_id):
                    continue  # 同站换乘：使用 Station.transfer_time
                dist = float(e.distance) if not math.isnan(float(e.distance)) else 0.0
                for fm, tm in mode_pairs:
                    f, t = node_key(fm, e.from_id), node_key(tm, e.to_id)
                    if f not in self.stations or t not in self.stations or (f, t) in seen_walk:
                        continue
                    self._add_walk_edge(f, t, dist)
                    seen_walk.add((f, t))
        # 同一 node_id 同时作为地铁站与公交站（源数据 28 例）：补跨模式同站换乘边
        added_shared = 0
        for raw, keys in self._raw_index.items():
            if len(keys) < 2:
                continue
            for a in keys:
                for b in keys:
                    if a != b and (a, b) not in seen_walk:
                        sa, sb = self.stations[a], self.stations[b]
                        dist = haversine_scalar_m(sa.lat, sa.lon, sb.lat, sb.lon)
                        tt = max(dist / self.walk_speed if self.walk_speed > 0 else 0.0,
                                 float(self.same_station_transfer.get("intermodal", 0.0)))
                        edge = Edge(a, b, "walk", None, tt, dist, -1)
                        self.adj[a].append(edge)
                        self.walk_edges.append(edge)
                        seen_walk.add((a, b))
                        added_shared += 1
        if added_shared:
            self._log(f"[network] 为 {added_shared // 2} 个共用 id 的地铁/公交同名站补充跨模式换乘边")
        self._log(f"[network] {sum(1 for s in self.stations.values() if s.mode=='subway')} 地铁站 / "
                  f"{sum(1 for s in self.stations.values() if s.mode=='bus')} 公交站, "
                  f"{sum(1 for l in self.lines.values() if l.mode=='subway')} 条地铁线路(方向) / "
                  f"{sum(1 for l in self.lines.values() if l.mode=='bus')} 条公交线路(方向), "
                  f"{self.n_vehicle_edges} 条车辆边, {len(self.walk_edges)} 条步行边")

    def _add_walk_edge(self, f: str, t: str, dist: float) -> None:
        tt = dist / self.walk_speed if self.walk_speed > 0 else 0.0
        edge = Edge(f, t, "walk", None, tt, dist, -1)
        self.adj[f].append(edge)
        self.walk_edges.append(edge)

    def _add_line_edges(self, line: Line) -> None:
        seg = self.segment_times(line, None)
        dwell = self.line_dwell(line.id, line.mode, None)
        for i in range(len(line.stations) - 1):
            f, t = line.stations[i], line.stations[i + 1]
            # 路由用静态时间：段运行(含站间附加延误) + 中间站停站（首段不含起点站 dwell）
            tt = seg[i] + (dwell if i > 0 else 0.0)
            self.adj[f].append(Edge(f, t, line.mode, line.id, tt, line.seg_distance[i], i))
            self.n_vehicle_edges += 1

    def _apply_transfer_times(self, path: Optional[str]) -> None:
        if not path or not os.path.exists(path):
            return
        df = pd.read_csv(path)
        n = 0
        for r in df.itertuples(index=False):
            scope = str(getattr(r, "scope", "station"))
            key = str(r.key)
            val = float(r.transfer_time_s)
            if scope == "station" and key in self.stations:
                self.stations[key].transfer_time = val
                n += 1
            elif scope == "mode":
                for s in self.stations.values():
                    if s.mode == key:
                        s.transfer_time = val
                        n += 1
        self._log(f"[network] 应用同站换乘时间 {n} 条 ({path})")

    # ------------------------------------------------------------ 查询
    def line_speed(self, line_id: str, mode: str, hour: Optional[float] = None) -> float:
        period = classify_period(hour, is_hour=True) if hour is not None else None
        v = self.speed_table.speed(line_id, mode, period)
        return float(v) if v else float(self.mode_speeds[mode])

    def line_dwell(self, line_id: str, mode: str, hour: Optional[float] = None) -> float:
        period = classify_period(hour, is_hour=True) if hour is not None else None
        d = self.speed_table.dwell(line_id, mode, period)
        return float(d) if d is not None else float(self.mode_dwell[mode])

    def line_stop_delay(self, line_id: str, mode: str, hour: Optional[float] = None) -> float:
        period = classify_period(hour, is_hour=True) if hour is not None else None
        d = self.speed_table.stop_delay(line_id, mode, period)
        return float(d) if d is not None else float(self.mode_stop_delay.get(mode, 0.0))

    def line_terminal_hold(self, line_id: str, mode: str, hour: Optional[float] = None) -> float:
        period = classify_period(hour, is_hour=True) if hour is not None else None
        d = self.speed_table.terminal_hold(line_id, mode, period)
        return float(d) if d is not None else float(self.mode_terminal_hold.get(mode, 0.0))

    def segment_times(self, line: Line, hour: Optional[float] = None) -> List[float]:
        """各站间段运行时间 = 距离 / 线路速度 + 站间附加延误。"""
        v = self.line_speed(line.id, line.mode, hour) * 1000.0 / 3600.0
        delay = self.line_stop_delay(line.id, line.mode, hour)
        return [d / v + delay for d in line.seg_distance]

    def node_mode(self, node_id: str) -> str:
        return self.stations[node_id].mode

    def resolve_node(self, ident: object, mode: Optional[str] = None) -> Optional[str]:
        """把 'mode:id' 键、或 (raw id, mode)、或唯一的 raw id 解析成节点键；失败返回 None。"""
        if ident is None:
            return None
        s = str(ident)
        if s in self.stations:
            return s
        if mode in VEHICLE_MODES:
            k = node_key(mode, s)
            return k if k in self.stations else None
        cands = self._raw_index.get(s, [])
        if len(cands) == 1:
            return cands[0]
        return None

    def stations_by_name(self, mode: Optional[str] = None) -> Dict[str, List[Station]]:
        out: Dict[str, List[Station]] = {}
        for s in self.stations.values():
            if mode and s.mode != mode:
                continue
            out.setdefault(s.name, []).append(s)
        return out

    def vehicle_edges_from(self, node_id: str) -> Iterable[Edge]:
        return (e for e in self.adj.get(node_id, ()) if e.line is not None)

    def to_networkx(self):
        import networkx as nx
        G = nx.MultiDiGraph()
        for s in self.stations.values():
            G.add_node(s.id, name=s.name, mode=s.mode, location=s.location, lines=list(s.lines))
        for u, edges in self.adj.items():
            for e in edges:
                G.add_edge(u, e.to, mode=e.mode, line=e.line, travel_time=e.travel_time, distance=e.distance)
        return G

    def summary(self) -> Dict[str, int]:
        return {
            "subway_stations": sum(1 for s in self.stations.values() if s.mode == "subway"),
            "bus_stops": sum(1 for s in self.stations.values() if s.mode == "bus"),
            "subway_lines": sum(1 for l in self.lines.values() if l.mode == "subway"),
            "bus_lines": sum(1 for l in self.lines.values() if l.mode == "bus"),
            "vehicle_edges": self.n_vehicle_edges,
            "walk_edges": len(self.walk_edges),
        }
