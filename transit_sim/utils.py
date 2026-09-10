"""通用工具：地理距离、时间格式化、时段划分、线路名归一化。"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

EARTH_RADIUS_M = 6371000.0

# 时段定义（小时，左闭右开）
PERIODS: Dict[str, Tuple[float, float]] = {
    "am_peak": (7, 9),
    "pm_peak": (17, 19),
    "off_peak": (9, 17),   # 9–17 之外的白天时段由 classify_period 归入 off_peak
    "night": (22, 5),
}


def haversine_m(lat1, lon1, lat2, lon2):
    """球面距离（米），支持标量或 numpy 数组。"""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def haversine_scalar_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hour_of(seconds: float) -> int:
    return int(seconds) % 86400 // 3600


def classify_period(seconds_or_hour: float, is_hour: bool = False) -> str:
    """把出发时刻归到 am_peak / pm_peak / off_peak / night。"""
    h = float(seconds_or_hour) if is_hour else (float(seconds_or_hour) % 86400) / 3600.0
    if 7 <= h < 9:
        return "am_peak"
    if 17 <= h < 19:
        return "pm_peak"
    if h >= 22 or h < 5:
        return "night"
    return "off_peak"


# ------------------------------------------------------------- 名称归一化
_BUS_SUFFIX_RE = re.compile(r"(路|线)$")
_BUS_PAREN_RE = re.compile(r"[（(].*?[）)]")
_LOOP_TOKEN_RE = re.compile(r"(内|外|内环|外环|快|快车|区间|支|支线|夜|定班)$")

SUBWAY_STATION_ALIASES: Dict[str, str] = {
    "灵镜胡同": "灵境胡同",
    "T2航站楼": "2号航站楼",
    "T3航站楼": "3号航站楼",
}


def normalize_bus_line(name: object) -> Optional[str]:
    """把刷卡数据/路网中的公交线路名统一为不带 '路' 后缀、不含括号的核心名，如 '300外' -> '300'。

    返回 None 表示无法解析。
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return None
    s = _BUS_PAREN_RE.sub("", s)
    s = _BUS_SUFFIX_RE.sub("", s)
    s = s.replace("路", "")
    s2 = _LOOP_TOKEN_RE.sub("", s)
    return s2 or s


def bus_line_variant(name: object) -> Optional[str]:
    """保留内/外/快等变体信息的归一化名（'300外路' -> '300外'）。"""
    if name is None:
        return None
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return None
    s = _BUS_PAREN_RE.sub("", s)
    s = _BUS_SUFFIX_RE.sub("", s)
    return s.replace("路", "") or None


def normalize_station_name(name: object) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    s = s.replace("站", "") if s.endswith("站") and len(s) > 2 else s
    return SUBWAY_STATION_ALIASES.get(s, s)


def route_terminals(route_name: str) -> Tuple[str, Optional[str], Optional[str]]:
    """'地铁10号线内环(巴沟--巴沟)' -> ('地铁10号线内环', '巴沟', '巴沟')。"""
    m = re.match(r"^(.*?)[（(](.*?)--(.*?)[）)]\s*$", str(route_name))
    if not m:
        return str(route_name), None, None
    return m.group(1), m.group(2), m.group(3)


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
    return float(v[min(idx, v.size - 1)])
