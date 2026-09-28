"""
consolidate_landsat_st.py — the per-station 22x22 @ 100 m Landsat ST target (§46.5-B, §48)
==========================================================================================
Warps every station's st30 bundle ONCE onto the thermal head's 22x22 @ 100 m grid, so the
training loader reads a (n_scenes, 22, 22) array instead of reprojecting 76x76 cubes per
sample. Same read as `compute_lst_stats.py`, so the target and `sigma_ST` share one grid:

  * grid:  `landsat_target.to_target` — tile west/north edge anchored, never re-derived here
  * mask:  `mask_1km` (cdist > 1 km, §42), applied BEFORE the warp so a rejected 30 m pixel
           cannot bleed into a 100 m cell through the area-weighted average (§49.1)
  * no QC recomputed (§46.5 item 3) — the 250-360 K guard and cdist rule live in the mask

ALL stations and ALL years are written. The loss only ever reads train stations in
TRAIN_YEARS (the dataset gates that); val / OOT targets exist so the thermal head can be
evaluated, never trained on.

Geometry check (§46.8 item 5). The grid centre is the station's lat/lon projected to UTM —
what compute_lst_stats.py fitted sigma_ST on. The model tile, though, is whatever the imagery
download produced, recorded in `satellite_zarr/{station}.zarr/.zattrs` as `bounds_utm`. If the
two centres disagree, every thermal cell is offset from the pixels predicting it, and nothing
raises. The offset is measured per station and reported; > 5 m is flagged.

OUTPUT  {DATA_ROOT}/{cat}/{folder}/LANDSAT_ST/{folder}_lst22.npz
            lst22  (n, 22, 22) float16 Kelvin, NaN = no valid retrieval in the cell
            dates  (n,)        int32 YYYYMMDD
            cells  (n,)        int16 valid cells per scene
        csvs/lst22_index.csv  one row per station: status, n_scenes, offset_m

Usage:  sbatch slurm/consolidate_landsat_st.sh [--limit 8] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402
from landsat_target import SRC_N, to_target, utm_epsg  # noqa: E402

DATA_ROOT  = Path("/gpfs/work3/0/prjs1968/data")
RAW_ROOT   = Path("/projects/prjs1968/satellite_zarr")
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_INDEX  = REPO / "csvs" / "lst22_index.csv"
MASK_KEY   = "mask_1km"
MAX_OFFSET_M = 5.0


def _tile_centre(folder: str):
    """Centre of the model tile from the raw imagery store, or None if it has no bounds."""
    p = RAW_ROOT / f"{folder}.zarr" / ".zattrs"
    if not p.exists():
        return None
    b = json.loads(p.read_text()).get("bounds_utm")
    if not b or len(b) != 4:
        return None
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def one_station(task: dict) -> dict:
    folder, cat, lat, lon = task["folder"], task["cat"], task["lat"], task["lon"]
    d   = DATA_ROOT / cat / folder / "LANDSAT_ST"
    rec = dict(folder=folder, cat=cat, status="OK", n_scenes=0, n_written=0, offset_m=np.nan)
    st  = sorted(d.glob(f"{folder}_st30_*.npz"))
    mk  = sorted(d.glob(f"{folder}_st30mask_*.npz"))
    if not st or not mk:
        rec["status"] = "NO_BUNDLE"
        return rec
    try:
        z = np.load(st[0], allow_pickle=False)
        m = np.load(mk[0], allow_pickle=False)
        if MASK_KEY not in m.files:
            rec["status"] = f"NO_MASK_KEY:{sorted(m.files)[:4]}"
            return rec
        lst   = z["lst30"].astype(np.float32)
        dates = np.asarray([int(str(x)[:8]) for x in z["dates"]], dtype=np.int32)
        mask  = np.unpackbits(m[MASK_KEY], axis=1, count=SRC_N * SRC_N
                              ).reshape(-1, SRC_N, SRC_N).astype(bool)
    except Exception as e:                            # noqa: BLE001
        rec["status"] = f"READ_FAIL:{type(e).__name__}"
        return rec
    rec["n_scenes"] = int(lst.shape[0])
    if mask.shape[0] != lst.shape[0]:
        rec["status"] = f"MASK_MISALIGNED:{mask.shape[0]}!={lst.shape[0]}"
        return rec

    keep = mask.reshape(mask.shape[0], -1).any(axis=1)
    epsg = utm_epsg(lat, lon)
    try:
        import pyproj
        tx = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        cx, cy = tx.transform(lon, lat)
    except Exception as e:                            # noqa: BLE001
        rec["status"] = f"PROJ_FAIL:{type(e).__name__}"
        return rec

    tc = _tile_centre(folder)
    if tc is not None:
        rec["offset_m"] = float(np.hypot(tc[0] - cx, tc[1] - cy))

    out = d / f"{folder}_lst22.npz"
    if not keep.any():
        rec["status"] = "NO_SUPERVISED_SCENE"
        tgt = np.zeros((0, 22, 22), dtype=np.float32)
        kd  = np.zeros(0, dtype=np.int32)
    else:
        src = np.where(mask[keep], lst[keep], np.nan)          # mask FIRST, then warp
        try:
            tgt = to_target(src, cx, cy, epsg)                  # (k, 22, 22) Kelvin
        except Exception as e:                                  # noqa: BLE001
            rec["status"] = f"WARP_FAIL:{type(e).__name__}"
            return rec
        kd = dates[keep]
        ok = np.isfinite(tgt).reshape(tgt.shape[0], -1).any(axis=1)
        tgt, kd = tgt[ok], kd[ok]

    rec["n_written"] = int(tgt.shape[0])
    if not task["dry_run"]:
        cells = np.isfinite(tgt).reshape(tgt.shape[0], -1).sum(1).astype(np.int16)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, lst22=tgt.astype(np.float16), dates=kd, cells=cells)
        tmp.rename(out)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N stations (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="read and warp, write nothing")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["category"] = df.apply(category_of, axis=1)
    df = df[df["category"].isin(ALL_CATEGORIES)]
    tasks, seen = [], set()
    for _, r in df.iterrows():
        folder = station_dir_name(r)
        if folder in seen:
            continue
        seen.add(folder)
        tasks.append(dict(folder=folder, cat=r["category"], lat=float(r["latitude"]),
                          lon=float(r["longitude"]), dry_run=args.dry_run))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Stations : {len(tasks)}   mask key: {MASK_KEY}   dry_run={args.dry_run}", flush=True)

    recs = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one_station, tasks, chunksize=1), 1):
            recs.append(r)
            if r["status"] not in ("OK", "NO_SUPERVISED_SCENE"):
                print(f"  !! {r['status']:<32s} {r['folder']}", flush=True)
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    idx = pd.DataFrame(recs)
    print(f"\nstatus: {dict(Counter(idx['status']))}")
    print(f"scenes on disk {int(idx['n_scenes'].sum()):,d}, written {int(idx['n_written'].sum()):,d}")
    off = idx["offset_m"].dropna()
    print(f"grid-vs-tile centre offset (m), {len(off)} stations with bounds_utm: "
          f"median {off.median():.3f}, max {off.max():.3f}")
    bad = idx[idx["offset_m"] > MAX_OFFSET_M]
    if len(bad):
        print(f"  !! {len(bad)} stations offset > {MAX_OFFSET_M} m — the thermal target does "
              f"NOT sit on the model tile for these:")
        for _, r in bad.sort_values("offset_m", ascending=False).head(20).iterrows():
            print(f"     {r['offset_m']:9.1f} m  {r['folder']}")
    if not args.dry_run:
        idx.to_csv(OUT_INDEX, index=False)
        print(f"\nwrote {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
