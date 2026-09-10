"""步骤 1：读取原始刷卡文件（20190513.csv, ~10.7M 行），清洗并匹配到路网。

输出 data/processed/smartcard_20190513_matched.csv，每行一条有效行程：

    card, mode, line_raw, o_line, d_line, o_dir, o_stopno, d_stopno, o_name, d_name,
    o_lon, o_lat, d_lon, d_lat, dep_s, arr_s, tt_s, tt_adj_s,
    o_key, d_key, route_name, o_idx, d_idx, route_dist_m, n_intermediate, same_line, match

* bbox 过滤两端都在路网范围内的记录；
* tt_s = 到达 - 出发（秒），剔除 <= 0 或 > 3 h；公交剔除起终点同站；
* tt_adj_s：地铁进站刷卡为分钟精度，期望补偿 +30 s；公交不变；
* 地铁按站名匹配节点，并判断是否存在单一线路方向可直达（same_line）；
* 公交按线路名（变体键/核心键）+ 站名（失败则 200 m 内坐标）匹配到线路与站序。

用法::

    python -m calibration.preprocess_smartcard [--raw /path/20190513.csv] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from calibration.common import (BBOX, DEFAULT_RAW_CSV, FILTERED_CACHE_CSV, MATCHED_CSV, MATCHED_FULL_CSV, MODE_MAP,
                                PREPROCESS_REPORT, RAW_COLUMNS, SUBWAY_TAPIN_MINUTE_ADJ_S, NetworkMatcher,
                                bbox_mask, ensure_dirs, load_network_version, save_json)

DAY0 = pd.Timestamp("2019-05-13")


def read_and_filter(raw_path: str, chunksize: int = 1_000_000, limit: int | None = None,
                    max_tt_s: float = 3 * 3600) -> tuple[pd.DataFrame, dict]:
    """分块读取 -> bbox / 时间 / 同站过滤，返回合并后的 DataFrame 与统计。"""
    stats = {"rows_read": 0, "in_bbox": 0, "valid_time": 0, "kept": 0}
    parts = []
    t0 = time.time()
    reader = pd.read_csv(raw_path, header=0, names=RAW_COLUMNS, encoding="utf-8-sig", chunksize=chunksize,
                         dtype={"card": str, "o_line": str, "d_line": str, "o_name": str, "d_name": str},
                         nrows=limit)
    for i, chunk in enumerate(reader):
        stats["rows_read"] += len(chunk)
        chunk = chunk.dropna(subset=["o_lat", "o_lon", "d_lat", "d_lon", "dep", "arr"])
        chunk = chunk[bbox_mask(chunk, BBOX)]
        stats["in_bbox"] += len(chunk)
        if chunk.empty:
            continue
        dep = pd.to_datetime(chunk["dep"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        arr = pd.to_datetime(chunk["arr"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        ok = dep.notna() & arr.notna()
        chunk = chunk[ok]
        dep, arr = dep[ok], arr[ok]
        day0 = dep.dt.normalize()
        dep_s = (dep - day0).dt.total_seconds()
        arr_s = (arr - day0).dt.total_seconds()
        tt = arr_s - dep_s
        mode = chunk["o_mode"].map(MODE_MAP)
        valid = (tt > 0) & (tt <= max_tt_s) & mode.notna() & (chunk["o_mode"] == chunk["d_mode"])
        same_stop = (chunk["o_name"] == chunk["d_name"])
        valid &= ~((mode == "bus") & same_stop)
        stats["valid_time"] += int(valid.sum())
        out = pd.DataFrame({
            "card": chunk["card"].str.slice(0, 12),
            "mode": mode,
            "line_raw": chunk["o_line"].astype(str),
            "o_line": chunk["o_line"].astype(str),
            "d_line": chunk["d_line"].astype(str),
            "o_dir": chunk["o_dir"].astype("int16"),
            "o_stopno": chunk["o_stopno"].astype("int16"),
            "d_stopno": chunk["d_stopno"].astype("int16"),
            "o_name": chunk["o_name"].astype(str),
            "d_name": chunk["d_name"].astype(str),
            "o_lon": chunk["o_lon"].astype("float64"), "o_lat": chunk["o_lat"].astype("float64"),
            "d_lon": chunk["d_lon"].astype("float64"), "d_lat": chunk["d_lat"].astype("float64"),
            "dep_s": dep_s.astype("float64"), "arr_s": arr_s.astype("float64"), "tt_s": tt.astype("float64"),
        })[valid.values]
        parts.append(out)
        stats["kept"] += len(out)
        print(f"  chunk {i + 1}: 读取 {stats['rows_read']:,} | bbox 内 {stats['in_bbox']:,} | 保留 {stats['kept']:,} "
              f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, stats


def add_adjusted_tt(df: pd.DataFrame) -> pd.DataFrame:
    """地铁进站刷卡分钟精度补偿：tt_adj = tt - 30 s（公交不变）。"""
    df["tt_adj_s"] = df["tt_s"] + np.where(df["mode"] == "subway", SUBWAY_TAPIN_MINUTE_ADJ_S, 0.0)
    return df


def match_records(df: pd.DataFrame, matcher: NetworkMatcher) -> tuple[pd.DataFrame, dict]:
    """在唯一组合上做匹配再合并回全表。"""
    t0 = time.time()
    out_cols = ["o_key", "d_key", "route_name", "o_idx", "d_idx", "route_dist_m", "n_intermediate", "same_line", "match"]

    # ---- 地铁
    sub = df[df["mode"] == "subway"]
    sub_pairs = sub[["o_name", "d_name"]].drop_duplicates()
    rows = []
    for on, dn in sub_pairs.itertuples(index=False, name=None):
        m = matcher.match_subway_pair(on, dn)
        rows.append((on, dn, m.o_key, m.d_key, m.route_name, m.o_idx, m.d_idx, m.route_dist_m,
                     m.n_intermediate, m.same_line, m.match))
    sub_map = pd.DataFrame(rows, columns=["o_name", "d_name"] + out_cols)
    print(f"  地铁唯一站对 {len(sub_map):,} 组匹配完成 ({time.time() - t0:.0f}s)", flush=True)

    # ---- 公交（站名 + 坐标：坐标取该 (线路, 站名) 组的中位坐标）
    bus = df[df["mode"] == "bus"]
    grp = bus.groupby(["line_raw", "o_name", "d_name"], sort=False)
    bus_pairs = grp.agg(o_lon=("o_lon", "median"), o_lat=("o_lat", "median"),
                        d_lon=("d_lon", "median"), d_lat=("d_lat", "median")).reset_index()
    rows = []
    for r in bus_pairs.itertuples(index=False):
        m = matcher.match_bus_pair(r.line_raw, r.o_name, r.d_name, r.o_lon, r.o_lat, r.d_lon, r.d_lat)
        rows.append((r.line_raw, r.o_name, r.d_name, m.o_key, m.d_key, m.route_name, m.o_idx, m.d_idx,
                     m.route_dist_m, m.n_intermediate, m.same_line, m.match))
    bus_map = pd.DataFrame(rows, columns=["line_raw", "o_name", "d_name"] + out_cols)
    print(f"  公交唯一 (线路,起,终) {len(bus_map):,} 组匹配完成 ({time.time() - t0:.0f}s)", flush=True)

    sub_m = sub.merge(sub_map, on=["o_name", "d_name"], how="left")
    bus_m = bus.merge(bus_map, on=["line_raw", "o_name", "d_name"], how="left")
    out = pd.concat([sub_m, bus_m], ignore_index=True).sort_values("dep_s").reset_index(drop=True)
    out["same_line"] = out["same_line"].fillna(False).astype(bool)

    rep = {}
    for mode in ("subway", "bus"):
        x = out[out["mode"] == mode]
        rep[mode] = {
            "records": int(len(x)),
            "both_nodes_matched": int((x["o_key"].notna() & x["d_key"].notna()).sum()),
            "same_line_matched": int(x["same_line"].sum()),
            "match_breakdown": x["match"].value_counts().to_dict(),
        }
    return out, rep


def run(raw_path: str, out_path: str, network_version: str = "full", chunksize: int = 1_000_000,
        limit: int | None = None, cached_filtered: str | None = None) -> dict:
    """完整流程；network_version 决定匹配所用路网（'full' 用于首轮裁剪，'2019' 用于最终标定）。"""
    ensure_dirs()
    if not os.path.exists(raw_path):
        raise SystemExit(f"找不到原始文件: {raw_path}")
    if cached_filtered and os.path.exists(cached_filtered):
        print(f"[preprocess] 读取已过滤缓存 {cached_filtered}")
        df = pd.read_csv(cached_filtered, dtype={"card": str, "line_raw": str, "o_line": str, "d_line": str,
                                                  "o_name": str, "d_name": str})
        stats = {"rows_read": None, "in_bbox": None, "kept": int(len(df))}
    else:
        print(f"[preprocess] 读取 {raw_path}")
        df, stats = read_and_filter(raw_path, chunksize=chunksize, limit=limit)
        if cached_filtered:
            df.drop(columns=[c for c in ("tt_adj_s",) if c in df.columns]).to_csv(cached_filtered, index=False)
    df = add_adjusted_tt(df.drop(columns=[c for c in ("tt_adj_s",) if c in df.columns]))
    print(f"[preprocess] 有效记录 {len(df):,}（bbox 内 {stats['in_bbox']} / 总 {stats['rows_read']}）")

    net = load_network_version(network_version)
    matcher = NetworkMatcher(net)
    print(f"[preprocess] 匹配到路网 ({network_version}) ...")
    out, rep = match_records(df, matcher)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    report = {"raw_file": raw_path, "network_version": network_version, "bbox": BBOX, "filter_stats": stats,
              "match": rep, "n_output": int(len(out)),
              "hourly_departures": (out["dep_s"] // 3600).astype(int).value_counts().sort_index().to_dict()}
    save_json(report, PREPROCESS_REPORT if network_version != "full" else PREPROCESS_REPORT.replace(".json", "_full.json"))
    print(f"[preprocess] 写出 {out_path} ({len(out):,} 行)")
    for mode, r in rep.items():
        print(f"  {mode}: 记录 {r['records']:,}, 两端匹配 {r['both_nodes_matched']:,} "
              f"({r['both_nodes_matched'] / max(r['records'], 1):.1%}), 同线直达 {r['same_line_matched']:,}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW_CSV, help="原始刷卡 CSV 路径")
    ap.add_argument("--out", default=None, help="输出路径（默认按 --network 决定）")
    ap.add_argument("--network", default="2019", choices=["full", "2019"], help="匹配所用路网版本")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int, default=None, help="仅读取前 N 行（调试）")
    ap.add_argument("--cache-filtered", default=FILTERED_CACHE_CSV,
                    help="过滤后（未匹配）记录的缓存文件，避免二次读取 2 GB 原始文件")
    args = ap.parse_args()
    out = args.out or (MATCHED_CSV if args.network == "2019" else MATCHED_FULL_CSV)
    run(args.raw, out, args.network, args.chunksize, args.limit, args.cache_filtered)


if __name__ == "__main__":
    main()
