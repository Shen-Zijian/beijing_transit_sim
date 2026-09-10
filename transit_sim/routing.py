"""线路感知的最短路 / Yen k 短路，以及带缓存的候选路径生成。

与旧版 ``_shortest_path_with_transfer_penalty`` / ``_k_shortest_paths_with_transfer_penalty`` 对应，改进点：

* 搜索状态为 (节点, 当前线路标签)，换乘惩罚与同站换乘步行时间计入代价，结果为真正的最优解；
* 支持平行线路（同一站对多条线路）；
* 同站换乘在 ``edge_path`` 中显式表示为 ``mode='transfer'`` 的自环边，供乘客状态机统一处理；
* A* 启发式 + 仅在"决策节点"（上车/下车/换乘点）做 Yen 分支，速度提升一个量级。

route dict 字段（与旧 pkl 兼容并扩展）::

    node_path, edge_path[{from_node,to_node,mode,line,travel_time,distance}],
    total_travel_time, in_vehicle_time, walk_time, expected_wait, num_boardings,
    num_transfers, total_distance(km), mode_type(1 bus/2 subway/7 mixed), lines, boarding_nodes
"""
from __future__ import annotations

import heapq
import os
import pickle
from typing import Callable, Dict, FrozenSet, Hashable, List, Optional, Set, Tuple

from .network import Edge, TransitNetwork
from .utils import haversine_scalar_m

FRESH = None        # 尚未乘车
AFTER_WALK = "*"    # 曾乘车、现处于步行后状态（下一次上车即为换乘）

EdgeKey = Tuple[str, str, Optional[str]]


class Router:
    def __init__(self, network: TransitNetwork, *, transfer_penalty: float = 180.0,
                 k: int = 5, headway_fn: Optional[Callable[[str, str], float]] = None,
                 access_egress: Optional[Dict[str, float]] = None,
                 cache_path: Optional[str] = None, use_astar: bool = True):
        self.net = network
        self.transfer_penalty = float(transfer_penalty)
        self.k = int(k)
        # headway_fn(line_id, mode) -> 平均发车间隔 (s)，用于候选路径的期望等待
        self.headway_fn = headway_fn or (lambda line_id, mode: 300.0 if mode == "subway" else 600.0)
        self.access_egress = access_egress or {"subway": 0.0, "bus": 0.0}
        self.cache_path = cache_path
        self.cache: Dict[Tuple[str, str], List[dict]] = {}
        self.use_astar = use_astar
        self.max_overlap = 0.8          # Yen 候选与已选路径的车辆边重叠比例上限（去近似重复）
        self.max_dijkstra_per_od = 80   # 单个 OD 的搜索次数上限
        vmax_kmh = max(list(network.mode_speeds.values()) +
                       [v for v in network.speed_table._speed.values()] + [1.0])
        self._vmax = vmax_kmh * 1000.0 / 3600.0
        self.stats = {"computed": 0, "cache_hits": 0, "no_path": 0, "dijkstra_calls": 0}
        self._build_fast_adj()
        if cache_path and os.path.exists(cache_path):
            self.load_cache(cache_path)

    def _build_fast_adj(self) -> None:
        """把 Edge 展开成纯元组，避免热循环中的属性访问开销。"""
        self._fadj: Dict[str, list] = {}
        for u, edges in self.net.adj.items():
            self._fadj[u] = [(e.to, e.line, e.travel_time, e) for e in edges]
        self._coords = {k: (s.lat, s.lon) for k, s in self.net.stations.items()}
        self._transfer_time = {k: s.transfer_time for k, s in self.net.stations.items()}

    # ------------------------------------------------------------ cache
    def load_cache(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.cache.update(data)
        print(f"[router] 已加载路径缓存 {len(self.cache)} 对 OD ({path})")

    def save_cache(self, path: Optional[str] = None) -> None:
        path = path or self.cache_path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------ 查询
    def get_routes(self, source: str, target: str, k: Optional[int] = None) -> List[dict]:
        key = (source, target)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        k = k or self.k
        paths = self.k_shortest_paths(source, target, k)
        routes = [self.build_route(p) for p in paths]
        self.cache[key] = routes
        self.stats["computed"] += 1
        if not routes:
            self.stats["no_path"] += 1
        return routes

    # ------------------------------------------------------------ Dijkstra / A*
    def shortest_path(self, source: str, target: str, *, banned_nodes: FrozenSet[str] = frozenset(),
                      banned_edges: FrozenSet[EdgeKey] = frozenset(),
                      start_tag: Hashable = FRESH) -> Optional[List[Edge]]:
        """返回边序列（含显式 transfer 自环边），找不到返回 None。"""
        if source == target:
            return []
        fadj = self._fadj
        coords = self._coords
        if target not in coords or source not in coords:
            return None
        self.stats["dijkstra_calls"] += 1
        tlat, tlon = coords[target]
        vmax = self._vmax
        penalty = self.transfer_penalty
        transfer_time = self._transfer_time
        use_astar = self.use_astar
        hcache: Dict[str, float] = {}

        def h(node: str) -> float:
            v = hcache.get(node)
            if v is None:
                if use_astar:
                    lat, lon = coords[node]
                    v = haversine_scalar_m(lat, lon, tlat, tlon) / vmax
                else:
                    v = 0.0
                hcache[node] = v
            return v

        start = (source, start_tag)
        best: Dict[Tuple[str, Hashable], float] = {start: 0.0}
        prev: Dict[Tuple[str, Hashable], Tuple[Tuple[str, Hashable], Optional[Edge], Optional[Edge]]] = {}
        heap = [(h(source), 0.0, source, start_tag)]
        closed: Set[Tuple[str, Hashable]] = set()
        inf = float("inf")
        has_banned_edges = bool(banned_edges)

        while heap:
            _, g, node, tag = heapq.heappop(heap)
            state = (node, tag)
            if state in closed:
                continue
            closed.add(state)
            if node == target:
                return self._reconstruct(prev, state)
            for to, line, tt_edge, e in fadj.get(node, ()):
                if to in banned_nodes:
                    continue
                if has_banned_edges and (node, to, line) in banned_edges:
                    continue
                extra_edge: Optional[Edge] = None
                if line is None:  # walk
                    new_tag = FRESH if tag is FRESH else AFTER_WALK
                    cost = tt_edge
                else:
                    if tag is FRESH or tag == line:
                        cost = tt_edge
                    elif tag == AFTER_WALK:
                        cost = tt_edge + penalty
                    else:  # 同站换乘（不同线路）
                        tt = transfer_time[node]
                        cost = tt_edge + penalty + tt
                        extra_edge = Edge(node, node, "transfer", None, tt, 0.0, -1)
                    new_tag = line
                ns = (to, new_tag)
                ng = g + cost
                if ng < best.get(ns, inf):
                    best[ns] = ng
                    prev[ns] = (state, e, extra_edge)
                    heapq.heappush(heap, (ng + h(to), ng, to, new_tag))
        return None

    @staticmethod
    def _reconstruct(prev, state) -> List[Edge]:
        edges: List[Edge] = []
        while state in prev:
            pstate, e, extra = prev[state]
            edges.append(e)
            if extra is not None:
                edges.append(extra)
            state = pstate
        edges.reverse()
        return edges

    # ------------------------------------------------------------ Yen
    def k_shortest_paths(self, source: str, target: str, k: Optional[int] = None) -> List[List[Edge]]:
        k = k or self.k
        first = self.shortest_path(source, target)
        if not first:
            return []
        A: List[List[Edge]] = [first]
        A_keys = {self._path_key(first)}
        B: List[Tuple[float, int, List[Edge]]] = []
        B_keys: Set[tuple] = set()
        counter = 0
        calls0 = self.stats["dijkstra_calls"]
        while len(A) < k:
            if self.stats["dijkstra_calls"] - calls0 > self.max_dijkstra_per_od:
                break
            prev_path = A[-1]
            real = [e for e in prev_path if e.mode != "transfer"]
            nodes = [real[0].frm] + [e.to for e in real]
            for i in self._spur_indices(real):
                spur = nodes[i]
                root_real = real[:i]
                root_key = tuple((e.frm, e.to, e.line) for e in root_real)
                banned_edges: Set[EdgeKey] = set()
                for p in A:
                    p_real = [e for e in p if e.mode != "transfer"]
                    if len(p_real) > i and tuple((e.frm, e.to, e.line) for e in p_real[:i]) == root_key:
                        banned_edges.add((p_real[i].frm, p_real[i].to, p_real[i].line))
                banned_nodes = frozenset(nodes[:i])
                start_tag = self._tag_after(root_real)
                spur_path = self.shortest_path(spur, target, banned_nodes=banned_nodes,
                                               banned_edges=frozenset(banned_edges), start_tag=start_tag)
                if not spur_path:
                    continue
                root_edges = self._root_with_transfers(prev_path, i)
                # spur 搜索以 start_tag 起步，若首边换线，Dijkstra 已自动插入同站 transfer 边
                cand = root_edges + spur_path
                if self._has_node_loop(cand):
                    continue
                key = self._path_key(cand)
                if key in A_keys or key in B_keys:
                    continue
                counter += 1
                heapq.heappush(B, (self.path_cost(cand), counter, cand))
                B_keys.add(key)
            # 从候选堆中取代价最低、且与已选路径不高度重叠的路径
            accepted = False
            while B:
                _, _, nxt = heapq.heappop(B)
                if self.max_overlap < 1.0 and any(self._overlap(nxt, a) > self.max_overlap for a in A):
                    continue
                A.append(nxt)
                A_keys.add(self._path_key(nxt))
                accepted = True
                break
            if not accepted:
                break
        return A

    @staticmethod
    def _overlap(p: List[Edge], q: List[Edge]) -> float:
        """两条路径共享车辆边的时间占比（相对较短者）。"""
        pe = {(e.frm, e.to, e.line): e.travel_time for e in p if e.line is not None}
        qe = {(e.frm, e.to, e.line): e.travel_time for e in q if e.line is not None}
        if not pe or not qe:
            return 0.0
        shared = sum(t for k, t in pe.items() if k in qe)
        return shared / min(sum(pe.values()), sum(qe.values()))

    @staticmethod
    def _spur_indices(real: List[Edge]) -> List[int]:
        """决策节点：起点、上车点、下车点（模式/线路变化处）。"""
        idx = [0]
        for i in range(1, len(real)):
            a, b = real[i - 1], real[i]
            if a.line != b.line:
                idx.append(i)
        return idx

    @staticmethod
    def _tag_after(root_real: List[Edge]) -> Hashable:
        if not root_real:
            return FRESH
        boarded = any(e.line is not None for e in root_real)
        last = root_real[-1]
        if last.line is not None:
            return last.line
        return AFTER_WALK if boarded else FRESH

    @staticmethod
    def _root_with_transfers(path: List[Edge], n_real: int) -> List[Edge]:
        """取前 n_real 条真实边（保留其间的 transfer 自环边）。"""
        out: List[Edge] = []
        cnt = 0
        for e in path:
            if e.mode == "transfer":
                if cnt < n_real:
                    out.append(e)
                continue
            if cnt >= n_real:
                break
            out.append(e)
            cnt += 1
        # 去掉末尾悬挂的 transfer 边（其后没有真实边）
        while out and out[-1].mode == "transfer":
            out.pop()
        return out

    @staticmethod
    def _has_node_loop(path: List[Edge]) -> bool:
        real = [e for e in path if e.mode != "transfer"]
        nodes = [real[0].frm] + [e.to for e in real] if real else []
        return len(nodes) != len(set(nodes))

    @staticmethod
    def _path_key(path: List[Edge]) -> tuple:
        return tuple((e.frm, e.to, e.line) for e in path if e.mode != "transfer")

    # ------------------------------------------------------------ 代价与属性
    def path_cost(self, path: List[Edge]) -> float:
        """与搜索一致的感知代价（含换乘惩罚与同站换乘时间）。"""
        cost = 0.0
        tag: Hashable = FRESH
        for e in path:
            if e.mode == "transfer":
                cost += e.travel_time
                continue
            if e.line is None:
                cost += e.travel_time
                tag = FRESH if tag is FRESH else AFTER_WALK
            else:
                cost += e.travel_time
                if tag is not FRESH and tag != e.line:
                    cost += self.transfer_penalty
                tag = e.line
        return cost

    def build_route(self, path: List[Edge]) -> dict:
        net = self.net
        ivt = walk = dist = 0.0
        boardings: List[Tuple[str, str]] = []  # (node, line)
        modes: Set[str] = set()
        tag: Hashable = FRESH
        edge_path = []
        for e in path:
            edge_path.append({
                "from_node": e.frm, "to_node": e.to, "mode": e.mode, "line": e.line,
                "travel_time": float(e.travel_time), "distance": float(e.distance),
            })
            if e.mode in ("walk", "transfer"):
                walk += e.travel_time
                dist += e.distance
                if e.mode == "walk":
                    tag = FRESH if tag is FRESH else AFTER_WALK
            else:
                ivt += e.travel_time
                dist += e.distance
                modes.add(e.mode)
                if tag != e.line:
                    boardings.append((e.frm, e.line))
                tag = e.line
        expected_wait = 0.0
        for node, line in boardings:
            mode = net.lines[line].mode
            expected_wait += 0.5 * float(self.headway_fn(line, mode))
        real = [e for e in path if e.mode != "transfer"]
        node_path = ([real[0].frm] + [e.to for e in real]) if real else []
        o_mode = net.stations[node_path[0]].mode if node_path else None
        d_mode = net.stations[node_path[-1]].mode if node_path else None
        access = float(self.access_egress.get(o_mode, 0.0)) if o_mode else 0.0
        egress = float(self.access_egress.get(d_mode, 0.0)) if d_mode else 0.0
        mode_type = 0
        if modes == {"bus"}:
            mode_type = 1
        elif modes == {"subway"}:
            mode_type = 2
        elif modes == {"bus", "subway"}:
            mode_type = 7
        return {
            "node_path": node_path,
            "edge_path": edge_path,
            "in_vehicle_time": ivt,
            "walk_time": walk,
            "expected_wait": expected_wait,
            "access_egress_time": access + egress,
            "total_travel_time": ivt + walk + expected_wait + access + egress,
            "num_boardings": len(boardings),
            "num_transfers": max(0, len(boardings) - 1),
            "total_distance": dist / 1000.0,
            "mode_type": mode_type,
            "lines": [l for _, l in boardings],
            "boarding_nodes": [n for n, _ in boardings],
            "perceived_cost": self.path_cost(path),
        }
