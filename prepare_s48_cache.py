"""
prepare_s48_cache.py — the per-station read cache the §48 loader needs
=======================================================================
Pure re-layout of the token store; it computes nothing a model could learn from. Written
because the token store's chunking fights the §48 read pattern:

  s2/l12, s1_*/l12   chunked 32 acquisitions at a time -> one anchor read decompresses ~9 MB
  cm/masks           ONE chunk per station             -> one pixel mask decompresses ~5 MB

Per station, into CACHE_ROOT/{cat}/{station}/ (dataset.build_station_cache):
  pyr.npz            per-acquisition pooled pyramids (N,4,768) fp16, valid-token counts,
                     date ints, for s2 / s1_asc / s1_desc; DEM and LULC pyramids. Written
                     LAST and renamed into place — its presence means the station is complete.
  {orbit}_l12.npy    (N,196,768) fp16, flat, memmapped per sample for the anchor
  s2_cm.npy          (N_s2,224,224) u1 pixel cloud classes aligned to s2 dates, 255 = no mask

The token store is read-only (chmod'd, see §43.12) and is never written. Resume-safe: a
station with a pyr.npz is skipped unless --force.

Usage:  sbatch slurm/prepare_s48_cache.sh [--limit 8] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402
from dataset import CACHE_ROOT, ZARR_ROOT, _open_zarr, build_station_cache  # noqa: E402

SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_INDEX  = REPO / "csvs" / "s48_cache_index.csv"


def one_station(task: dict) -> dict:
    cat, folder = task["cat"], task["folder"]
    out = CACHE_ROOT / cat / folder
    rec = dict(folder=folder, cat=cat, status="OK", seconds=0.0)
    if (out / "pyr.npz").exists() and not task["force"]:
        rec["status"] = "SKIP_EXISTS"
        return rec
    t0 = time.time()
    zg = _open_zarr(ZARR_ROOT / cat / folder, cat)
    if zg is None:
        rec["status"] = "ZARR_INCOMPLETE"
        return rec
    try:
        rec.update(build_station_cache(zg, out))
    except Exception as e:                            # noqa: BLE001
        rec["status"] = f"FAIL:{type(e).__name__}:{str(e)[:80]}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["category"] = df.apply(category_of, axis=1)
    tasks, seen = [], set()
    for _, r in df[df["category"].isin(ALL_CATEGORIES)].iterrows():
        folder = station_dir_name(r)
        if folder not in seen:
            seen.add(folder)
            tasks.append(dict(cat=r["category"], folder=folder, force=args.force))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Stations: {len(tasks)}  ->  {CACHE_ROOT}", flush=True)

    recs = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one_station, tasks, chunksize=1), 1):
            recs.append(r)
            if not r["status"].startswith(("OK", "SKIP")):
                print(f"  !! {r['status']:<40s} {r['folder']}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    idx = pd.DataFrame(recs)
    print(f"\nstatus: {dict(Counter(idx['status'].str.split(':').str[0]))}")
    for col in sorted(c for c in idx.columns if c.endswith(("_n", "_no_cm", "_nonfinite",
                                                            "_no_token_mask"))):
        print(f"  {col:<24s} total {int(idx[col].fillna(0).sum()):>9,d}  "
              f"stations>0 {int((idx[col].fillna(0) > 0).sum()):>4d}")
    for col in ("dem_ok", "lulc_ok"):
        if col in idx:
            print(f"  {col:<24s} False at {int((idx[col] == False).sum())} stations")  # noqa: E712
    idx.to_csv(OUT_INDEX, index=False)
    print(f"wrote {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
