"""行程记录与系统指标。

* ``TripRecorder``：每个乘客完成（或超时退出）时记录一行，包含总时间及等待/步行/车内/进出站分量；
* 逐小时系统快照由 ``TransitSimulator`` 调用 ``snapshot`` 写入；
* ``summary`` 给出与观测数据可比的统计口径：地铁按刷卡进站到刷卡出站的总时间，公交按车内时间。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

TRIP_COLUMNS = [
    "passenger_id", "sid", "pid", "origin", "destination", "origin_mode", "destination_mode",
    "departure_time", "departure_hour", "arrival_time", "status",
    "total_time", "wait_time", "first_wait_time", "walk_time", "in_vehicle_time", "access_egress_time",
    "num_boardings", "num_transfers", "main_mode", "mode_type", "route_distance_km",
    "num_reroutes", "observed_tt_s",
]


class TripRecorder:
    def __init__(self):
        self.records: List[dict] = []
        self.snapshots: List[dict] = []

    # ------------------------------------------------------------ trips
    def record(self, p, now: float) -> None:
        """p: passengers.Passenger"""
        r = p.route or {}
        total = (p.arrival_time - p.departure_time) if p.arrival_time is not None else (now - p.departure_time)
        self.records.append({
            "passenger_id": p.id,
            "sid": p.sid,
            "pid": p.pid,
            "origin": p.origin,
            "destination": p.destination,
            "origin_mode": p.origin_mode,
            "destination_mode": p.destination_mode,
            "departure_time": p.departure_time,
            "departure_hour": int(p.departure_time) % 86400 // 3600,
            "arrival_time": p.arrival_time,
            "status": p.status,
            "total_time": float(total),
            "wait_time": float(p.wait_time),
            "first_wait_time": float(p.first_wait_time),
            "walk_time": float(p.walk_time),
            "in_vehicle_time": float(p.in_vehicle_time),
            "access_egress_time": float(p.access_egress_time),
            "num_boardings": int(p.num_boardings),
            "num_transfers": max(0, int(p.num_boardings) - 1),
            "main_mode": p.main_mode(),
            "mode_type": r.get("mode_type", 0),
            "route_distance_km": r.get("total_distance", np.nan),
            "num_reroutes": int(p.num_reroutes),
            "observed_tt_s": p.extra.get("observed_tt_s", np.nan) if p.extra else np.nan,
        })

    def snapshot(self, now: float, **stats) -> None:
        row = {"time": now, "hour": now / 3600.0}
        row.update(stats)
        self.snapshots.append(row)

    # ------------------------------------------------------------ export
    def trips_df(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=TRIP_COLUMNS)
        return pd.DataFrame(self.records, columns=TRIP_COLUMNS)

    def snapshots_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots)

    def save(self, out_dir: str, prefix: str = "") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        tp = os.path.join(out_dir, f"{prefix}trips.csv")
        self.trips_df().to_csv(tp, index=False)
        paths["trips"] = tp
        sp = os.path.join(out_dir, f"{prefix}hourly_stats.csv")
        self.snapshots_df().to_csv(sp, index=False)
        paths["hourly_stats"] = sp
        mp = os.path.join(out_dir, f"{prefix}summary.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2, default=_json_default)
        paths["summary"] = mp
        return paths

    def summary(self) -> dict:
        df = self.trips_df()
        if df.empty:
            return {"n_trips": 0}
        done = df[df["status"] == "arrived"]
        out = {
            "n_trips": int(len(df)),
            "n_arrived": int(len(done)),
            "n_timeout_exit": int((df["status"] == "timeout_exit").sum()),
            "n_unfinished": int((df["status"] == "unfinished").sum()),
            "completion_rate": float(len(done) / max(len(df), 1)),
        }
        if len(done):
            for col in ("total_time", "wait_time", "walk_time", "in_vehicle_time"):
                v = done[col] / 60.0
                out[f"{col}_min_mean"] = float(v.mean())
                out[f"{col}_min_median"] = float(v.median())
                out[f"{col}_min_p90"] = float(v.quantile(0.9))
            out["transfers_mean"] = float(done["num_transfers"].mean())
            out["by_main_mode"] = {}
            for mode, g in done.groupby("main_mode"):
                out["by_main_mode"][mode] = {
                    "n": int(len(g)),
                    "total_time_min_mean": float(g["total_time"].mean() / 60.0),
                    "total_time_min_median": float(g["total_time"].median() / 60.0),
                    "in_vehicle_min_mean": float(g["in_vehicle_time"].mean() / 60.0),
                    "wait_min_mean": float(g["wait_time"].mean() / 60.0),
                    "transfers_mean": float(g["num_transfers"].mean()),
                }
            if done["observed_tt_s"].notna().any():
                obs = done.dropna(subset=["observed_tt_s"])
                out["observed_comparison"] = {
                    "n": int(len(obs)),
                    "sim_total_min_mean": float(obs["total_time"].mean() / 60.0),
                    "obs_total_min_mean": float(obs["observed_tt_s"].mean() / 60.0),
                    "mean_abs_error_min": float((obs["total_time"] - obs["observed_tt_s"]).abs().mean() / 60.0),
                }
        return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
