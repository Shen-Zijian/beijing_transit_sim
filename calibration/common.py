"""标定流水线公共部分：路径常量、bbox、线路名归一化、刷卡记录 -> 路网匹配器。"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transit_sim.network import TransitNetwork  # noqa: E402
from transit_sim.utils import haversine_m, normalize_station_name  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NETWORK_DIR = os.path.join(DATA_DIR, "network")
DEMAND_DIR = os.path.join(DATA_DIR, "demand")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

DEFAULT_RAW_CSV = "/Users/smlmac/Shen/simulator_beijing_simplified/20190513.csv"
FILTERED_CACHE_CSV = os.path.join(PROCESSED_DIR, "smartcard_20190513_filtered.csv")   # 清洗后未匹配
MATCHED_FULL_CSV = os.path.join(PROCESSED_DIR, "smartcard_20190513_matched_full.csv")  # 按完整路网匹配（裁剪用）
MATCHED_CSV = os.path.join(PROCESSED_DIR, "smartcard_20190513_matched.csv")            # 按 2019 路网匹配（标定用）
PREPROCESS_REPORT = os.path.join(PROCESSED_DIR, "preprocess_report.json")

# 路网覆盖范围 (min_lat, min_lon, max_lat, max_lon)，与旧仓库一致
BBOX = (39.833217, 116.271581, 39.989478, 116.487437)

RAW_COLUMNS = ["card", "o_mode", "o_line", "o_dir", "o_stopno", "o_name", "o_lon", "o_lat", "dep",
               "d_mode", "d_line", "d_dir", "d_stopno", "d_name", "d_lon", "d_lat", "arr"]
MODE_MAP = {"公交": "bus", "地铁": "subway"}
# 地铁进站刷卡时间为分钟精度（秒恒为 0，视为向下取整）：记录的出发时刻早于真实时刻 U(0,60) s，
# 记录的行程时间平均偏长 30 s，故 tt_adj = tt - 30。
SUBWAY_TAPIN_MINUTE_ADJ_S = -30.0

MATCHED_DTYPES = {"card": str, "mode": str, "line_raw": str, "o_line": str, "d_line": str,
                  "o_name": str, "d_name": str, "o_key": str, "d_key": str, "route_name": str}


def ensure_dirs() -> None:
    for d in (NETWORK_DIR, DEMAND_DIR, PROCESSED_DIR, REPORTS_DIR, CONFIG_DIR):
        os.makedirs(d, exist_ok=True)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_network_version(version: str = "full", verbose: bool = False, **kwargs) -> TransitNetwork:
    """以默认参数加载指定版本路网（匹配/裁剪用，速度参数无关紧要）。"""
    params = dict(speeds={"subway": 35.0, "bus": 18.0}, walk_speed=1.2, dwell={"subway": 30.0, "bus": 20.0},
                  same_station_transfer={"subway": 180.0, "bus": 30.0, "intermodal": 180.0}, verbose=verbose)
    params.update(kwargs)
    return TransitNetwork(os.path.join(NETWORK_DIR, f"nodes_{version}.csv"),
                          os.path.join(NETWORK_DIR, f"edges_{version}.csv"), **params)


def load_full_network(verbose: bool = False) -> TransitNetwork:
    return load_network_version("full", verbose=verbose)


def load_matched(path: str = MATCHED_CSV, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, dtype={k: v for k, v in MATCHED_DTYPES.items()
                                                     if usecols is None or k in usecols},
                       keep_default_na=True)


# --------------------------------------------------------------------------- 线路名归一化
_PAREN_RE = re.compile(r"[（(].*?[）)]")
_TRAIL_TOKEN_RE = re.compile(r"(内|外|快|区间|支|定班)$")


def line_variant_key(name: object) -> Optional[str]:
    """带方向/快慢变体的归一化键：'300路外环' -> '300外'，'345路快车' -> '345快'，'夜11路' -> '夜11'。"""
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return None
    s = _PAREN_RE.sub("", s)
    s = s.replace("内环", "内").replace("外环", "外").replace("快车", "快")
    s = re.sub(r"路$", "", s)
    s = s.replace("路", "")
    s = re.sub(r"线$", "", s) if re.search(r"\d线$", s) else s
    return s or None


def line_core_key(name: object) -> Optional[str]:
    """去掉内/外/快/区间等后缀的核心键：'300快外' -> '300'。"""
    v = line_variant_key(name)
    if v is None:
        return None
    prev = None
    while prev != v:
        prev = v
        v = _TRAIL_TOKEN_RE.sub("", v)
    return v or prev


# --------------------------------------------------------------------------- 匹配器
@dataclass
class PairMatch:
    o_key: Optional[str]
    d_key: Optional[str]
    route_name: Optional[str]
    o_idx: int
    d_idx: int
    route_dist_m: float
    n_intermediate: int
    same_line: bool
    match: str        # 匹配方式说明


class NetworkMatcher:
    """把刷卡记录（站名 / 线路名 / 坐标）匹配到路网节点与线路。"""

    def __init__(self, net: TransitNetwork, coord_tol_m: float = 200.0):
        self.net = net
        self.coord_tol_m = coord_tol_m
        # 地铁：站名 -> 节点键
        self.subway_by_name: Dict[str, str] = {}
        for s in net.stations.values():
            if s.mode == "subway":
                self.subway_by_name.setdefault(normalize_station_name(s.name), s.id)
        # 各线路：站键 -> 首次出现的站序；站名 -> 站序；坐标数组
        self.line_index: Dict[str, Dict[str, int]] = {}
        self.line_name_index: Dict[str, Dict[str, int]] = {}
        self.line_coords: Dict[str, np.ndarray] = {}
        for line in net.lines.values():
            idx: Dict[str, int] = {}
            nidx: Dict[str, int] = {}
            coords = []
            for i, sk in enumerate(line.stations):
                idx.setdefault(sk, i)
                st = net.stations[sk]
                nidx.setdefault(normalize_station_name(st.name), i)
                coords.append((st.lat, st.lon))
            self.line_index[line.id] = idx
            self.line_name_index[line.id] = nidx
            self.line_coords[line.id] = np.asarray(coords)
        # 站键 -> 所属线路
        self.lines_of_station: Dict[str, List[str]] = {k: list(s.lines) for k, s in net.stations.items()}
        # 公交：变体键 / 核心键 -> 线路 id 列表
        self.bus_by_variant: Dict[str, List[str]] = {}
        self.bus_by_core: Dict[str, List[str]] = {}
        for line in net.lines.values():
            if line.mode != "bus":
                continue
            v = line_variant_key(line.id)
            c = line_core_key(line.id)
            if v:
                self.bus_by_variant.setdefault(v, []).append(line.id)
            if c:
                self.bus_by_core.setdefault(c, []).append(line.id)

    # ------------------------------------------------------------------ 地铁
    def match_subway_pair(self, o_name: str, d_name: str) -> PairMatch:
        ok = self.subway_by_name.get(normalize_station_name(o_name))
        dk = self.subway_by_name.get(normalize_station_name(d_name))
        if ok is None or dk is None:
            return PairMatch(ok, dk, None, -1, -1, np.nan, -1, False, "no_station")
        best: Optional[Tuple[int, str, int, int]] = None
        for lid in self.lines_of_station.get(ok, []):
            idx = self.line_index[lid]
            if dk not in idx:
                continue
            io, id_ = idx[ok], idx[dk]
            line = self.net.lines[lid]
            hops = line.hops_between(io, id_)      # 环线允许绕过闭合点
            if hops <= 0:
                continue
            cand = (hops, lid, io, id_)
            if best is None or cand < best:
                best = cand
        if best is None:
            return PairMatch(ok, dk, None, -1, -1, np.nan, -1, False, "transfer")
        _, lid, io, id_ = best
        dist, n_int = self.net.lines[lid].segment_between(io, id_)
        return PairMatch(ok, dk, lid, io, id_, dist, n_int, True, "same_line")

    # ------------------------------------------------------------------ 公交
    def _bus_candidates(self, line_raw: str) -> Tuple[List[str], str]:
        v = line_variant_key(line_raw)
        if v and v in self.bus_by_variant:
            return self.bus_by_variant[v], "variant"
        c = line_core_key(line_raw)
        if c and c in self.bus_by_core:
            return self.bus_by_core[c], "core"
        return [], "no_line"

    def _locate_on_line(self, lid: str, name: str, lon: float, lat: float) -> Tuple[int, str]:
        nidx = self.line_name_index[lid]
        i = nidx.get(normalize_station_name(name), -1)
        if i >= 0:
            return i, "name"
        if lon is None or lat is None or np.isnan(lon) or np.isnan(lat):
            return -1, "none"
        coords = self.line_coords[lid]
        d = haversine_m(lat, lon, coords[:, 0], coords[:, 1])
        j = int(np.argmin(d))
        if d[j] <= self.coord_tol_m:
            return j, "coord"
        return -1, "none"

    def match_bus_pair(self, line_raw: str, o_name: str, d_name: str,
                       o_lon: float, o_lat: float, d_lon: float, d_lat: float) -> PairMatch:
        cands, how = self._bus_candidates(line_raw)
        if not cands:
            return PairMatch(None, None, None, -1, -1, np.nan, -1, False, "no_line")
        best = None
        for lid in cands:
            io, ho = self._locate_on_line(lid, o_name, o_lon, o_lat)
            if io < 0:
                continue
            id_, hd = self._locate_on_line(lid, d_name, d_lon, d_lat)
            if id_ < 0:
                continue
            line = self.net.lines[lid]
            hops = line.hops_between(io, id_)
            if hops <= 0:
                continue
            score = (0 if (ho == "name" and hd == "name") else 1, hops)
            if best is None or score < best[0]:
                best = (score, lid, io, id_, ho, hd)
        if best is None:
            return PairMatch(None, None, None, -1, -1, np.nan, -1, False, f"{how}_no_stop")
        _, lid, io, id_, ho, hd = best
        line = self.net.lines[lid]
        dist, n_int = line.segment_between(io, id_)
        return PairMatch(line.stations[io], line.stations[id_], lid, io, id_, dist, n_int, True,
                         f"{how}_{ho}_{hd}")


def bbox_mask(df: pd.DataFrame, bbox=BBOX) -> pd.Series:
    min_lat, min_lon, max_lat, max_lon = bbox
    return (df["o_lat"].between(min_lat, max_lat) & df["o_lon"].between(min_lon, max_lon)
            & df["d_lat"].between(min_lat, max_lat) & df["d_lon"].between(min_lon, max_lon))
