"""
compute_lst_stats.py — sigma_ST and sigma_level, before any training (§46)
===========================================================================

The thermal head's loss is

    L_lst = Huber(delta=1.0) on (That' - T') / sigma_ST

so `sigma_ST` is not a convenience: `delta = 1.0` only means "one SD of normal within-tile
thermal contrast" if sigma_ST is that SD. It is **global and fixed, never per-scene and never
per-batch** (§46 :14875) — and that is deliberate. The AMPLITUDE of within-tile contrast is
itself the signal: a dry heterogeneous tile under strong insolation spreads widely, a wet or
overcast one is nearly uniform, and per-scene normalisation would rescale both to unit
variance, keeping the shape of the pattern while discarding how strong it was.

Being global is also what makes it leakable, which is why this job obeys the same two rules
as `compute_era5_stats.py`:

    TRAIN STATIONS ONLY   (split == "train", categories from splits_config)
    PRE-OOT YEARS ONLY    (TRAIN_YEARS; 2023-2025 is the §47 holdout and is never read)

A constant fitted over held-out scenes would set the scale of every training gradient from
data the model is later scored on.

The 22x22 @ 100 m grid comes from `landsat_target.py`, which §46's loader must also import —
if the two derive it separately, sigma_ST is in different units from the residual it divides
and nothing raises.

OUTPUT  csvs/lst_stats.json          sigma_ST, sigma_level, provenance
        csvs/lst_tile_means.csv      (station, date, tile_mean, n_valid_cells) — so the
                                     "R2 of tile-mean ST on the 18 ERA5 drivers" gate can be
                                     settled without re-reading the 4.5 GB archive.

Usage:  sbatch slurm/compute_lst_stats.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from splits_config import SM_CATEGORIES, TRAIN_YEARS, category_of, station_dir_name  # noqa: E402
from landsat_target import OUT_N, OUT_RES_M, SRC_N, centre, to_target, utm_epsg  # noqa: E402

DATA_ROOT  = Path("/gpfs/work3/0/prjs1968/data")
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_JSON   = REPO / "csvs" / "lst_stats.json"
OUT_MEANS  = REPO / "csvs" / "lst_tile_means.csv"

MIN_VALID_CELLS = 2      # a "pattern" needs at least two cells to have any spread at all


def one_station(task: dict) -> dict:
    """Accumulate this station's contribution. Returns partials, never a global statistic."""
    folder, cat, lat, lon = task["folder"], task["cat"], task["lat"], task["lon"]
    d = DATA_ROOT / cat / folder / "LANDSAT_ST"
    st = sorted(d.glob(f"{folder}_st30_*.npz"))
    mk = sorted(d.glob(f"{folder}_st30mask_*.npz"))
    rec = dict(folder=folder, n_scenes=0, n_sup=0, n_used=0, sq=0.0, n_cells=0,
               mean_n=0, mean_sum=0.0, mean_sq=0.0, status="OK", mask_key="", rows=[])
    if not st or not mk:
        rec["status"] = "NO_BUNDLE"
        return rec

    try:
        z = np.load(st[0], allow_pickle=False)
        m = np.load(mk[0], allow_pickle=False)
        lst = z["lst30"].astype(np.float32)          # (n, 76, 76) Kelvin, NaN = no retrieval
        dates = np.asarray([str(x) for x in z["dates"]])
        # The bundles on disk carry TWO masks, at cdist > 1 km and > 0.5 km. 1 km is the
        # rule §42 settled on (the strict choice); `mask` alone is an older layout that
        # build_landsat_mask.py's docstring still describes but no bundle uses.
        mkey = next((k for k in ("mask_1km", "mask") if k in m.files), None)
        if mkey is None:
            rec["status"] = f"NO_MASK_KEY:{sorted(m.files)[:4]}"
            return rec
        rec["mask_key"] = mkey
        mask = np.unpackbits(m[mkey], axis=1, count=SRC_N * SRC_N
                             ).reshape(-1, SRC_N, SRC_N).astype(bool)
    except Exception as e:                            # noqa: BLE001
        rec["status"] = f"READ_FAIL:{type(e).__name__}"
        return rec

    rec["n_scenes"] = int(lst.shape[0])
    if mask.shape[0] != lst.shape[0]:
        rec["status"] = f"MASK_MISALIGNED:{mask.shape[0]}!={lst.shape[0]}"
        return rec

    years = np.asarray([int(str(x)[:4]) for x in dates])
    keep = np.isin(years, TRAIN_YEARS) & mask.reshape(mask.shape[0], -1).any(axis=1)
    rec["n_sup"] = int((np.isin(years, TRAIN_YEARS) & mask.reshape(mask.shape[0], -1).any(axis=1)).sum())
    if not keep.any():
        rec["status"] = "NO_PRE_CUT_SUPERVISED_SCENE"
        return rec

    # Mask first, warp second. Warping then masking would let a rejected 30 m pixel bleed
    # into a 100 m cell through the area-weighted average.
    src = np.where(mask[keep], lst[keep], np.nan)
    epsg = utm_epsg(lat, lon)
    try:
        import pyproj
        tx = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        cx, cy = tx.transform(lon, lat)
        tgt = to_target(src, cx, cy, epsg)            # (k, 22, 22) Kelvin
    except Exception as e:                            # noqa: BLE001
        rec["status"] = f"WARP_FAIL:{type(e).__name__}"
        return rec

    kept_dates = dates[keep]
    for i in range(tgt.shape[0]):
        field = tgt[i]
        n_valid = int(np.isfinite(field).sum())
        if n_valid < MIN_VALID_CELLS:
            continue
        cen, mu = centre(field)
        if not np.isfinite(mu):
            continue
        rec["n_used"]  += 1
        rec["sq"]      += float(np.nansum(cen ** 2))
        rec["n_cells"] += n_valid
        rec["mean_n"]  += 1
        rec["mean_sum"] += mu
        rec["mean_sq"]  += mu * mu
        rec["rows"].append((folder, kept_dates[i], round(mu, 4), n_valid))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N stations (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["category"] = df.apply(category_of, axis=1)
    sel = df[(df["split"] == "train") & df["category"].isin(SM_CATEGORIES)]
    print(f"Splits      : {SPLITS_CSV}")
    print(f"Split       : train")
    print(f"Categories  : {list(SM_CATEGORIES)}")
    print(f"Years       : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} "
          f"(the §47 OOT holdout is never read)")
    print(f"Target grid : {OUT_N}x{OUT_N} @ {OUT_RES_M} m, anchored to the tile's west/north "
          f"edge (landsat_target.py)")
    print(f"Stations    : {len(sel)} train stations\n", flush=True)

    tasks = [dict(folder=station_dir_name(r), cat=r["category"],
                  lat=float(r["latitude"]), lon=float(r["longitude"]))
             for _, r in sel.iterrows()]
    if args.limit:
        tasks = tasks[: args.limit]

    recs = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one_station, tasks, chunksize=1), 1):
            recs.append(r)
            if r["status"] != "OK":
                print(f"  !! {r['status']:<32s} {r['folder']}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    from collections import Counter
    print(f"\nstatus: {dict(Counter(r['status'] for r in recs))}")
    print(f"mask key used: {dict(Counter(r['mask_key'] for r in recs if r['mask_key']))}")
    # Cross-check against the index rather than trusting the bundle read.
    try:
        idx = pd.read_csv(REPO / "csvs" / "landsat_mask_index.csv")
        idx["yr"] = idx["date"].astype(str).str[:4].astype(int)
        idx = idx[idx["yr"].isin(TRAIN_YEARS) & (idx["supervised"] == 1)]
        want = {t["folder"] for t in tasks}
        short = {f.split("_", 1)[1] if f.startswith("ISMN_") else f.split("_", 1)[1]
                 for f in want}
        n_idx = int(idx["station_id"].isin([s.split("_")[-1] for s in want]).sum())
        print(f"supervised pre-cut scenes: bundles say {sum(r['n_sup'] for r in recs):,d}, "
              f"landsat_mask_index says {n_idx:,d} (station_id join is approximate)")
    except Exception as e:  # noqa: BLE001
        print(f"index cross-check skipped: {type(e).__name__}")

    sq      = sum(r["sq"] for r in recs)
    n_cells = sum(r["n_cells"] for r in recs)
    n_used  = sum(r["n_used"] for r in recs)
    n_scn   = sum(r["n_scenes"] for r in recs)
    if n_cells == 0:
        sys.exit("FATAL: no supervised pre-cut scene produced a usable 22x22 field — "
                 "refusing to emit a fabricated constant.")

    # Each scene is centred to zero mean over its own valid cells, so the pooled variance is
    # simply the summed squared residual over the cell count. One degree of freedom per scene
    # is consumed by that centring.
    dof = max(1, n_cells - n_used)
    sigma_st = float(np.sqrt(sq / dof))

    mn = sum(r["mean_n"] for r in recs)
    ms = sum(r["mean_sum"] for r in recs)
    mq = sum(r["mean_sq"] for r in recs)
    sigma_level = float(np.sqrt(max(0.0, mq / mn - (ms / mn) ** 2))) if mn > 1 else float("nan")

    print(f"\nscenes on disk (train stations) : {n_scn:,d}")
    print(f"scenes used (pre-cut, supervised, >= {MIN_VALID_CELLS} valid cells) : {n_used:,d}")
    print(f"target cells contributing       : {n_cells:,d}")
    print(f"\n  sigma_ST     = {sigma_st:.4f} K   (pooled SD of the centred 22x22 field)")
    print(f"  sigma_level  = {sigma_level:.4f} K   (SD of tile-mean ST)")
    print(f"  ratio        = {sigma_st / sigma_level:.4f}" if sigma_level else "")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0

    payload = {
        "sigma_ST": sigma_st,
        "sigma_level": sigma_level,
        "units": "kelvin",
        "grid": f"{OUT_N}x{OUT_N} @ {OUT_RES_M} m, tile west/north edge anchored",
        "centring": "per-scene mean over valid cells removed; alpha=0 means the level term "
                    "is reported, not trained",
        "split": "train",
        "categories": list(SM_CATEGORIES),
        "years": [TRAIN_YEARS[0], TRAIN_YEARS[-1]],
        "n_stations": len([r for r in recs if r["status"] == "OK"]),
        "n_scenes_used": n_used,
        "n_cells": n_cells,
        "min_valid_cells": MIN_VALID_CELLS,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    rows = [row for r in recs for row in r["rows"]]
    pd.DataFrame(rows, columns=["folder", "date", "tile_mean_K", "n_valid_cells"]).to_csv(
        OUT_MEANS, index=False)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MEANS}  ({len(rows):,d} rows) — input to the alpha=0 gate "
          f"(R2 of tile-mean ST on the 18 ERA5 drivers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
