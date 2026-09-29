"""
profile_getitem.py — where does one training sample's load time go? (2026-09-29)
=================================================================================
Smoke 27340622 was data-bound: 60-210 s of loading vs 3-14 s of compute per epoch, GPU util
3-14 %. This builds the train dataset for the smoke's 20 stations (as train.py does), then
times N random __getitem__ calls in ONE process: wall time per sample, cProfile top entries,
and build_fine split into its S2 / S1 / DEM / LULC reads. Read-only.

Usage: python profile_getitem.py [--n 300] [--stations 20]
"""
import argparse
import cProfile
import pstats
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import dataset as D  # noqa: E402
from splits_config import SM_CATEGORIES, TRAIN_YEARS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stations", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    ds = D.SoilMoistureDataset(splits_csv=str(REPO / "csvs" / "station_splits.csv"),
                               era5_stats_path=str(REPO / "csvs" / "era5_stats18.json"),
                               category_filter=list(SM_CATEGORIES), years=list(TRAIN_YEARS),
                               split_filter=["train"], training=True, max_stations=a.stations)
    print(f"dataset build: {time.time() - t0:.1f} s, {len(ds)} samples", flush=True)

    # time build_fine's pieces by wrapping the zarr reads it makes
    import zarr.core
    orig = zarr.core.Array.__getitem__
    acc = {}

    def timed(self, sel):
        t = time.perf_counter()
        out = orig(self, sel)
        k = self.path.split("/")[0] + "/" + self.path.split("/")[-1]
        acc[k] = acc.get(k, 0.0) + time.perf_counter() - t
        return out
    zarr.core.Array.__getitem__ = timed

    random.seed(0)
    idx = [random.randrange(len(ds)) for _ in range(a.n)]
    for i in idx[:5]:
        ds[i]                                   # warm the per-worker caches
    acc.clear()
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    for i in idx:
        ds[i]
    pr.disable()
    wall = time.perf_counter() - t
    print(f"\n__getitem__: {wall / a.n * 1000:.1f} ms/sample over {a.n} samples "
          f"(=> {a.n / wall:.1f} samples/s per worker)")
    print("\nzarr reads (ms/sample by array):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v / a.n * 1000:8.1f}")
    print("\ncProfile, top 25 by cumulative time:")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
