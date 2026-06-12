"""
Scan SoilMoistureDataset samples for NaN/Inf, broken down by field, to find
the source of the train_loss=nan seen during training.

For every sample in the (small) dataset, checks every float tensor returned
by __getitem__ (tokens, era5, sif, twsa, soil_patch, label) for NaN and Inf.
"label" NaN is expected (depth not observed) and is reported separately from
the other fields, which should always be finite.

Usage:
    conda run -n terramind python scan_nan.py [--max-stations 3] [--split train,val]
"""

import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from dataset import SoilMoistureDataset, ERA5_VARS

SPLITS_CSV      = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv"
DATA_ROOT       = "/gpfs/work3/0/prjs1968/data"
ERA5_STATS_PATH = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json"
YEARS           = list(range(2016, 2024))
CATEGORY_FILTER = ["sm_only"]

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")

# Values with |x| above this (and finite) are likely sentinel/fill values
# (e.g. -9999, 1e20) rather than real physical quantities.
SENTINEL_ABS_THRESHOLD = 1e5

RAW_FIELDS = {
    "sif":  "sif/values",
    "twsa": "twsa/lwe",
    "soil": "soil",
    "era5": "era5/values",
    "dem":  "dem",
    "lulc": "lulc",
}


def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def _check_station_ranges(row: dict) -> list:
    """Per-station worker: return a list of (dir_name, field, msg) issues for
    Inf or |value| > SENTINEL_ABS_THRESHOLD in sif/twsa/soil/era5/dem/lulc."""
    cat, dir_name = row["cat"], row["dir_name"]
    path = ZARR_ROOT / cat / dir_name
    if not (path / ".complete").exists():
        return []
    try:
        zg = zarr.open_consolidated(str(path), mode="r")
    except Exception:
        try:
            zg = zarr.open_group(str(path), mode="r")
        except Exception as e:
            return [(dir_name, "OPEN_ERROR", str(e))]

    issues = []
    for name, key in RAW_FIELDS.items():
        if key not in zg:
            continue
        arr = np.asarray(zg[key][:])
        if arr.size == 0:
            continue
        has_inf = bool(np.isinf(arr).any())
        finite  = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        amin, amax = float(finite.min()), float(finite.max())
        if has_inf or max(abs(amin), abs(amax)) > SENTINEL_ABS_THRESHOLD:
            issues.append((dir_name, name, f"min={amin:.4g} max={amax:.4g} has_inf={has_inf}"))
    return issues


def check_raw_ranges(workers: int = 64):
    """Scan raw zarr arrays (all sm_only stations) for Inf or |value| > SENTINEL_ABS_THRESHOLD
    in sif/twsa/soil/era5/dem/lulc -- catches sentinel/fill values that pass
    audit_comprehensive.py's isnan-only checks."""
    df = pd.read_csv(SPLITS_CSV)
    df["cat"]      = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)
    df = df[df["cat"].isin(CATEGORY_FILTER)]

    rows = df[["cat", "dir_name"]].to_dict("records")
    with Pool(workers) as pool:
        results = pool.map(_check_station_ranges, rows)
    issues = [issue for station_issues in results for issue in station_issues]

    print(f"\n=== Raw zarr range check: {len(df)} sm_only stations ({workers} workers) ===")
    if not issues:
        print("  No Inf or |value| > 1e5 found in sif/twsa/soil/era5/dem/lulc.")
    else:
        for dir_name, field, msg in issues:
            print(f"  [{dir_name}] {field}: {msg}")


def scan(ds, name):
    bad_counts = defaultdict(int)
    first_bad  = {}
    label_nan_all = 0   # samples where every depth is NaN (-> zero-loss branch)

    for i in range(len(ds)):
        s = ds.samples[i]
        sample = ds[i]

        for k, v in sample.items():
            if not isinstance(v, torch.Tensor) or not v.is_floating_point():
                continue

            if k == "label":
                if torch.isnan(v).all():
                    label_nan_all += 1
                if torch.isinf(v).any():
                    bad_counts[f"{k}_inf"] += 1
                    first_bad.setdefault(f"{k}_inf", (s["station_key"], s["year"], s["doy"]))
                continue

            n_nan = torch.isnan(v).sum().item()
            n_inf = torch.isinf(v).sum().item()
            if n_nan:
                bad_counts[f"{k}_nan"] += 1
                first_bad.setdefault(f"{k}_nan", (s["station_key"], s["year"], s["doy"], n_nan))
            if n_inf:
                bad_counts[f"{k}_inf"] += 1
                first_bad.setdefault(f"{k}_inf", (s["station_key"], s["year"], s["doy"], n_inf))

    print(f"\n=== {name}: {len(ds)} samples, "
          f"{len(set(s['station_key'] for s in ds.samples))} stations ===")
    print(f"  samples with all-NaN label (zero-loss branch): {label_nan_all}")
    if not bad_counts:
        print("  No NaN/Inf found in any non-label field.")
    else:
        for k, n in sorted(bad_counts.items()):
            print(f"  {k:20s}: {n} samples affected, first={first_bad[k]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stations", type=int, default=3)
    parser.add_argument("--split", type=str, default="train,val")
    args = parser.parse_args()

    common_kwargs = dict(
        splits_csv      = SPLITS_CSV,
        data_root       = DATA_ROOT,
        era5_stats_path = ERA5_STATS_PATH,
        years           = YEARS,
        category_filter = CATEGORY_FILTER,
    )

    # Sanity-check ERA5 normalisation stats themselves
    import json
    stats = json.load(open(ERA5_STATS_PATH))
    stds = np.array(stats["stds"], dtype=np.float32)
    means = np.array(stats["means"], dtype=np.float32)
    bad_std = [ERA5_VARS[i] for i, sd in enumerate(stds) if abs(sd) < 1e-6]
    print(f"ERA5 stats: {len(stds)} vars, near-zero std vars: {bad_std or 'none'}")
    if np.isnan(means).any() or np.isnan(stds).any():
        print("  WARNING: era5_stats.json contains NaN!")

    check_raw_ranges()

    for split in args.split.split(","):
        train_flag = (split == "train")
        max_st = args.max_stations if split == "train" else max(1, args.max_stations // 5) or 1
        ds = SoilMoistureDataset(**common_kwargs, split_filter=[split], training=train_flag,
                                  max_stations=max_st)
        scan(ds, split)


if __name__ == "__main__":
    main()
