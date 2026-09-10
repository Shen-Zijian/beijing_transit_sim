"""需求加载：仿真格式 CSV -> 按仿真步长预分桶的出行请求。

兼容旧版 ``concentrated_demand.csv`` 列（sid, pid, time, origin_id, origin_type,
destination_id, destination_type, ...）以及标定流水线生成的 ``demand_20190513.csv``
（额外含 observed_tt_s 等列，原样带入 Demand.extra 供验证）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import pandas as pd


@dataclass
class Demand:
    sid: int
    pid: object
    time: float
    origin: str
    destination: str
    origin_mode: str
    destination_mode: str
    extra: dict = field(default_factory=dict)


class DemandTable:
    EXTRA_COLS = ("observed_tt_s", "o_line", "d_line", "card")

    def __init__(self, df: pd.DataFrame, time_step: int, time_column: str = "time",
                 start: Optional[float] = None, end: Optional[float] = None):
        self.time_step = int(time_step)
        df = df.copy()
        if time_column != "time":
            df = df.rename(columns={time_column: "time"})
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "origin_id", "destination_id"])
        if start is not None:
            df = df[df["time"] >= start]
        if end is not None:
            df = df[df["time"] < end]
        df = df[df["origin_id"].astype(str) != df["destination_id"].astype(str)]
        self.df = df.sort_values("time").reset_index(drop=True)
        self._buckets: Dict[int, List[Demand]] = {}
        self._build()

    @classmethod
    def from_csv(cls, path: str, time_step: int, fraction: float = 1.0, seed: int = 42,
                 time_column: str = "time", start: Optional[float] = None,
                 end: Optional[float] = None, usecols: Optional[List[str]] = None) -> "DemandTable":
        df = pd.read_csv(path, usecols=usecols)
        if fraction < 1.0:
            n = int(round(len(df) * fraction))
            df = df.sample(n=n, random_state=seed, replace=False)
        return cls(df, time_step=time_step, time_column=time_column, start=start, end=end)

    def _build(self) -> None:
        df = self.df
        has_sid = "sid" in df.columns
        has_pid = "pid" in df.columns
        has_ot = "origin_type" in df.columns
        has_dt = "destination_type" in df.columns
        extra_cols = [c for c in self.EXTRA_COLS if c in df.columns]
        step = self.time_step
        for i, row in enumerate(df.itertuples(index=False)):
            r = row._asdict()
            t = float(r["time"])
            bucket = int(t) // step * step
            om = str(r["origin_type"]) if has_ot else "unknown"
            dm = str(r["destination_type"]) if has_dt else "unknown"
            o_raw, d_raw = str(r["origin_id"]), str(r["destination_id"])
            d = Demand(
                sid=int(r["sid"]) if has_sid and not pd.isna(r["sid"]) else i,
                pid=r["pid"] if has_pid else None,
                time=t,
                origin=f"{om}:{o_raw}" if om in ("subway", "bus") and ":" not in o_raw else o_raw,
                destination=f"{dm}:{d_raw}" if dm in ("subway", "bus") and ":" not in d_raw else d_raw,
                origin_mode=om,
                destination_mode=dm,
                extra={c: r[c] for c in extra_cols},
            )
            self._buckets.setdefault(bucket, []).append(d)

    def pop(self, now: float) -> List[Demand]:
        return self._buckets.pop(int(now) // self.time_step * self.time_step, [])

    def peek(self, now: float) -> List[Demand]:
        return self._buckets.get(int(now) // self.time_step * self.time_step, [])

    def __len__(self) -> int:
        return len(self.df)

    def iter_all(self) -> Iterator[Demand]:
        for b in sorted(self._buckets):
            yield from self._buckets[b]

    def unique_od_pairs(self) -> List[tuple]:
        """返回去重后的 (origin_key, destination_key) 列表（键带模式前缀）。"""
        seen = set()
        out = []
        for d in self.iter_all():
            k = (d.origin, d.destination)
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def hourly_counts(self) -> pd.Series:
        return (self.df["time"] // 3600).astype(int).value_counts().sort_index()
