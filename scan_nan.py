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


def _iter_arrays(group, prefix=""):
    """Recursively yield (path, zarr.Array) for every array in a zarr group tree."""
    for key in sorted(group.array_keys()):
        yield f"{prefix}{key}", group[key]
    for key in sorted(group.group_keys()):
        yield from _iter_arrays(group[key], f"{prefix}{key}/")


def _check_station_all_arrays(row: dict) -> list:
    """Per-station worker: check isnan/isinf for every float array in the zarr
    group -- covers fields never checked before (s2/s1 L3/L6/L9/L12 fp16 token
    arrays, dem, lulc, labels/sm) in addition to era5/sif/twsa/soil."""
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
    for arr_path, arr in _iter_arrays(zg):
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        if arr.size == 0:
            continue
        data  = np.asarray(arr[:])
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        if n_nan or n_inf:
            issues.append((dir_name, arr_path,
                           f"nan={n_nan} inf={n_inf} shape={data.shape} dtype={data.dtype}"))
    return issues


def check_all_zarr_arrays(workers: int = 64):
    """Scan every float array in every sm_only station's zarr group for NaN/Inf."""
    df = pd.read_csv(SPLITS_CSV)
    df["cat"]      = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)
    df = df[df["cat"].isin(CATEGORY_FILTER)]

    rows = df[["cat", "dir_name"]].to_dict("records")
    with Pool(workers) as pool:
        results = pool.map(_check_station_all_arrays, rows)
    issues = [issue for station_issues in results for issue in station_issues]

    print(f"\n=== Full zarr array scan (isnan/isinf): {len(df)} sm_only stations ({workers} workers) ===")
    if not issues:
        print("  No NaN/Inf found in any float array.")
    else:
        for dir_name, arr_path, msg in issues:
            print(f"  [{dir_name}] {arr_path}: {msg}")


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

    check_all_zarr_arrays()

    for split in args.split.split(","):
        train_flag = (split == "train")
        max_st = args.max_stations if split == "train" else max(1, args.max_stations // 5) or 1
        ds = SoilMoistureDataset(**common_kwargs, split_filter=[split], training=train_flag,
                                  max_stations=max_st)
        scan(ds, split)


if __name__ == "__main__":
    main()
