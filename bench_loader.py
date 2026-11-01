"""
bench_loader.py — does a per-worker I/O thread pool fix the data-bound loader? (2026-09-29)
============================================================================================
profile_getitem.py measured ~1.0 s per sample, ~95 % waiting on cold GPFS reads. This times
SoilMoistureDataset.__getitems__ on batches of 128 (the train batch size), each batch drawn
from fresh random indices so no two runs share a GPFS read:

  A  one process, io_threads in --threads          -> samples/s per worker
  B  --procs processes x --proc-threads threads     -> aggregate samples/s, the realistic case
     (one GPU rank runs 12 train workers; the full node runs 4 ranks at once)

Target: one rank consumes a 128-batch in ~0.5-1 s of GPU compute, so it needs >~130-250
samples/s to stop being data-bound. Read-only.

Usage: python bench_loader.py [--threads 1 8 16 32] [--procs 12] [--proc-threads 16]
"""
import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import dataset as D  # noqa: E402
from splits_config import SM_CATEGORIES, TRAIN_YEARS  # noqa: E402

BS = 128
DS = None


def _batch(args):
    seed, threads = args
    DS.io_threads = threads
    rng = random.Random(seed)
    idx = [rng.randrange(len(DS)) for _ in range(BS)]
    t = time.perf_counter()
    DS.__getitems__(idx)
    return time.perf_counter() - t


def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--proc-threads", type=int, nargs="+", default=[16])
    ap.add_argument("--stations", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    DS = D.SoilMoistureDataset(splits_csv=str(REPO / "csvs" / "station_splits.csv"),
                               era5_stats_path=str(REPO / "csvs" / "era5_stats18.json"),
                               category_filter=list(SM_CATEGORIES), years=list(TRAIN_YEARS),
                               split_filter=["train"], training=True, max_stations=a.stations)
    print(f"dataset build: {time.time() - t0:.0f} s, {len(DS)} samples from {a.stations} stations",
          flush=True)
    _batch((0, 1 if 1 not in a.threads else 4))            # warm imports / pool code paths

    print("\nA  single process, one 128-batch per setting")
    seed = 100
    for n in a.threads:
        seed += 1
        dt = _batch((seed, n))
        print(f"   io_threads={n:3d}   {dt:7.1f} s/batch   {BS / dt:7.1f} samples/s", flush=True)

    ctx = mp.get_context("fork")
    for pt in a.proc_threads:
        print(f"\nB  {a.procs} processes x {pt} threads, one 128-batch each, concurrently")
        seeds = [(1000 * pt + i, pt) for i in range(a.procs)]
        t = time.perf_counter()
        with ctx.Pool(a.procs) as p:
            per = p.map(_batch, seeds)
        wall = time.perf_counter() - t
        print(f"   wall {wall:.1f} s   per-batch mean {sum(per) / len(per):.1f} s   "
              f"aggregate {a.procs * BS / wall:.1f} samples/s  (one rank's 12 workers)", flush=True)


if __name__ == "__main__":
    main()
