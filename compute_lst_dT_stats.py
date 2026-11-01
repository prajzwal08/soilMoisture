"""
compute_lst_dT_stats.py — the dT_pixel Huber knee and head bias, fixed before training (§52)
===========================================================================================

`--lst-target dT_pixel` trains

    L_lst = mean over valid cells of Huber_c( D[i,j] - (LST_obs[i,j] - t2m_mean) )    (K)

Before this file, c and the head's starting bias were recomputed at every start from whatever
stations that run loaded (a 20-station smoke gave c = 5.3 K; the full run would differ). They
are now computed ONCE here and frozen:

    c         = dT_pixel_sd    SD of the per-cell target (LST_obs - t2m_mean) — the quantity
                               the loss actually compares, so "c = one SD of the target"
    bias init = dT_pixel_mean  its mean

Same leakage rules as compute_lst_stats.py / compute_era5_stats.py:

    TRAIN STATIONS ONLY   (split == "train", SM categories)
    PRE-OOT YEARS ONLY    (TRAIN_YEARS; 2023-2025 is never read)

Reads exactly what the training loader reads — the lst22 target bundle (dataset._load_lst22)
and raw ERA5-Land t2m_mean from the station store (dataset._load_zarr_era5, column
T2M_MEAN_IDX) matched by exact date — so the statistics are of the real training target, not
a re-derivation of it. Every Landsat day in TRAIN_YEARS counts (as for sigma_ST), not only days
that also carry an SM label. Sums are float64.

Tile-mean dT (>= LST_LEVEL_MIN_CELLS valid cells) is reported alongside for reference.

OUTPUT  csvs/lst_dT_stats.json   (a NEW file: lst_stats.json is SHA'd into every existing
                                  checkpoint and is left untouched)
Usage:  sbatch slurm/compute_lst_dT_stats.sh      (CPU; never on the login node)
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

from dataset import (LST_LEVEL_MIN_CELLS, T2M_MEAN_IDX, _load_lst22,  # noqa: E402
                     _load_zarr_era5, _open_zarr)
from splits_config import SM_CATEGORIES, TRAIN_YEARS, category_of, station_dir_name  # noqa: E402

SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_JSON   = REPO / "csvs" / "lst_dT_stats.json"


def one_station(task):
    d, cat = task
    r = dict(station=d, status="OK", n_scenes=0, px_n=0, px_s=0.0, px_q=0.0,
             tile_n=0, tile_s=0.0, tile_q=0.0)
    try:
        lst = _load_lst22(cat, d)
        if lst is None:
            r["status"] = "NO_LST22"
            return r
        idx, arr = lst
        zg = _open_zarr(Path(d), cat)
        era = _load_zarr_era5(zg) if zg is not None else None
        if era is None:
            r["status"] = "NO_ERA5"
            return r
        values, dints, _ = era
        erow = {int(x): i for i, x in enumerate(dints)}
        for date, i in idx.items():
            if date // 10000 not in TRAIN_YEARS:
                continue
            k = erow.get(int(date))
            if k is None:
                continue
            f = np.asarray(arr[i], dtype=np.float64)
            v = np.isfinite(f)
            if not v.any():
                continue
            dT = f[v] - float(values[k, T2M_MEAN_IDX])
            r["n_scenes"] += 1
            r["px_n"] += dT.size
            r["px_s"] += float(dT.sum())
            r["px_q"] += float((dT * dT).sum())
            if v.sum() >= LST_LEVEL_MIN_CELLS:
                m = float(dT.mean())
                r["tile_n"] += 1
                r["tile_s"] += m
                r["tile_q"] += m * m
    except Exception as e:                                  # noqa: BLE001
        r["status"] = f"ERROR:{type(e).__name__}:{e}"
    return r


def _mean_sd(n, s, q):
    if n < 2:
        return float("nan"), float("nan")
    m = s / n
    return m, float(np.sqrt(max(0.0, q / n - m * m) * n / (n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N stations (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args()

    df = pd.read_csv(SPLITS_CSV)                              # pandas: quoted commas
    df["category"] = df.apply(category_of, axis=1)
    sel = df[(df["split"] == "train") & df["category"].isin(SM_CATEGORIES)]
    tasks = [(station_dir_name(r), r["category"]) for _, r in sel.iterrows()]
    if a.limit:
        tasks = tasks[: a.limit]
    print(f"train stations: {len(tasks)}   years {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} "
          f"(OOT never read)", flush=True)

    with Pool(a.workers) as pool:
        recs = pool.map(one_station, tasks, chunksize=1)
    st = Counter(r["status"].split(":")[0] for r in recs)
    print(f"status: {dict(st)}")
    for r in recs:
        if r["status"].startswith("ERROR"):
            print(f"  {r['station']}: {r['status']}")

    px_n = sum(r["px_n"] for r in recs)
    tile_n = sum(r["tile_n"] for r in recs)
    if px_n == 0:
        sys.exit("FATAL: no training cell produced a dT value -- refusing to write a constant.")
    px_m, px_sd = _mean_sd(px_n, sum(r["px_s"] for r in recs), sum(r["px_q"] for r in recs))
    t_m, t_sd = _mean_sd(tile_n, sum(r["tile_s"] for r in recs), sum(r["tile_q"] for r in recs))
    n_ok = sum(r["status"] == "OK" for r in recs)
    print(f"\nscenes used: {sum(r['n_scenes'] for r in recs):,d}   cells: {px_n:,d}   "
          f"stations OK: {n_ok}")
    print(f"  per-cell dT = LST_obs - t2m_mean : mean {px_m:.4f} K   SD {px_sd:.4f} K"
          f"   <-- knee c and head bias")
    print(f"  tile-mean dT (>= {LST_LEVEL_MIN_CELLS} cells)     : mean {t_m:.4f} K   "
          f"SD {t_sd:.4f} K   (reference, n = {tile_n:,d})")
    if a.dry_run or a.limit:
        print("\nDRY RUN / --limit -- nothing written.")
        return 0
    payload = {
        "dT_pixel_mean": px_m, "dT_pixel_sd": px_sd,
        "dT_tile_mean": t_m, "dT_tile_sd": t_sd,
        "units": "kelvin",
        "definition": "per 100 m cell: lst22 LST_obs - raw ERA5-Land t2m_mean (era5/values18 "
                      "col 0), same calendar day; valid = finite LST cell and an ERA5 row",
        "knee_rule": "dT_pixel Huber knee c = dT_pixel_sd; head_lst bias init = dT_pixel_mean",
        "split": "train", "categories": list(SM_CATEGORIES),
        "years": [TRAIN_YEARS[0], TRAIN_YEARS[-1]],
        "n_stations": n_ok, "n_scenes": sum(r["n_scenes"] for r in recs),
        "n_cells": px_n, "n_tiles": tile_n,
        "tile_min_valid_cells": LST_LEVEL_MIN_CELLS,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
