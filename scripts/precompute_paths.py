"""多进程预计算需求文件中所有唯一 OD 的 k 条候选路径，写入 routing.cache_file。

用法::

    python scripts/precompute_paths.py --config config/calibrated_20190513.yaml [--workers 20] [--fraction 1.0]

已存在于缓存中的 OD 会被跳过；可重复运行以增量补全。
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import get_context
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transit_sim import DemandTable, SimConfig, TransitNetwork  # noqa: E402
from transit_sim.simulator import make_router  # noqa: E402

_ROUTER = None


def _init(config_path: str, overrides: dict):
    global _ROUTER
    cfg = SimConfig.load(config_path, overrides=overrides)
    net = TransitNetwork.from_config(cfg, verbose=False)
    _ROUTER = make_router(cfg, net, load_cache=False)


def _work(pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], list]:
    out = {}
    for o, d in pairs:
        try:
            out[(o, d)] = _ROUTER.get_routes(o, d)
        except Exception as e:  # pragma: no cover
            out[(o, d)] = []
            print(f"[precompute] {o}->{d} 失败: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "calibrated_20190513.yaml"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--fraction", type=float, default=1.0, help="只为需求的这一比例（确定性抽样）预计算")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--set", action="append", default=[], help="覆盖配置项 key=value")
    args = ap.parse_args()

    from transit_sim.config import parse_set_args
    overrides = parse_set_args(args.set)
    cfg = SimConfig.load(args.config, overrides=overrides)
    cache_path = cfg.resolve(cfg.routing.cache_file)
    if not cache_path:
        raise SystemExit("配置中 routing.cache_file 为空")
    cache: Dict[Tuple[str, str], list] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[precompute] 已有缓存 {len(cache):,} 对 OD")

    print(f"[precompute] 读取需求 {cfg.resolve(cfg.demand.file)} (fraction={args.fraction}) ...")
    demand = DemandTable.from_csv(cfg.resolve(cfg.demand.file), time_step=cfg.time.time_step,
                                  fraction=args.fraction, seed=cfg.time.seed, time_column=cfg.demand.time_column)
    net = TransitNetwork.from_config(cfg, verbose=False)
    pairs = []
    for o, d in demand.unique_od_pairs():
        ok = net.resolve_node(o)
        dk = net.resolve_node(d)
        if ok is None or dk is None or ok == dk or (ok, dk) in cache:
            continue
        pairs.append((ok, dk))
    pairs = sorted(set(pairs))
    print(f"[precompute] 需计算 {len(pairs):,} 对 OD，{args.workers} 进程")
    if not pairs:
        return
    chunks = [pairs[i:i + args.chunk] for i in range(0, len(pairs), args.chunk)]
    t0 = time.time()
    done = 0
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.config, overrides)) as pool:
        for res in pool.imap_unordered(_work, chunks):
            cache.update(res)
            done += len(res)
            if done % (args.chunk * 10) < args.chunk or done == len(pairs):
                el = time.time() - t0
                print(f"[precompute] {done:,}/{len(pairs):,}  {el:.0f}s  ({done / max(el, 1e-9):.1f} OD/s)", flush=True)
            if done % (args.chunk * 50) < args.chunk:
                _save(cache, cache_path)
    _save(cache, cache_path)
    n_empty = sum(1 for v in cache.values() if not v)
    print(f"[precompute] 完成：缓存 {len(cache):,} 对 OD（无路径 {n_empty:,}），用时 {time.time() - t0:.0f}s -> {cache_path}")


def _save(cache, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
