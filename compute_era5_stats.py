"""
One-time script to compute per-variable ERA5 normalisation statistics
from all sm_only stations and write them to csvs/era5_stats.json.

Run once on a login node (no GPU needed):
    python compute_era5_stats.py

Output: csvs/era5_stats.json
  {
    "vars":        ["t2m_mean", ...],   # 19 variable names
    "means":       [float, ...],
    "stds":        [float, ...],
    "log1p_precip": true               # tp_sum is log1p-transformed before z-scoring
  }
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_ROOT  = Path("/gpfs/work3/0/prjs1968/data")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_PATH   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")

ERA5_VARS = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]
PRECIP_IDX = ERA5_VARS.index("tp_sum")


def main():
    splits = pd.read_csv(SPLITS_CSV)
    sm_only_train = splits[
        (splits["has_soil_moisture"] == True) &
        (splits["has_flux"] == False) &
        (splits["split"] == "train")
    ]
    print(f"Computing stats from {len(sm_only_train)} sm_only train stations...")

    # Two-pass Welford online algorithm for numerically stable mean/var
    n_total  = np.zeros(len(ERA5_VARS), dtype=np.float64)
    mean_acc = np.zeros(len(ERA5_VARS), dtype=np.float64)
    M2_acc   = np.zeros(len(ERA5_VARS), dtype=np.float64)

    skipped = 0
    for _, r in sm_only_train.iterrows():
        dir_name = f"ISMN_{r['network']}_{r['station_name']}"
        era5_dir = DATA_ROOT / "sm_only" / dir_name / "ERA5Land"
        nc_files = sorted(era5_dir.glob("meteo_*_*.nc")) if era5_dir.exists() else []
        if not nc_files:
            skipped += 1
            continue

        ds  = xr.open_dataset(nc_files[0])
        arr = np.stack([ds[v].values for v in ERA5_VARS], axis=-1).astype(np.float64)
        ds.close()

        # log1p precipitation
        arr[:, PRECIP_IDX] = np.log1p(arr[:, PRECIP_IDX].clip(0))

        # Online Welford update
        for row in arr:
            mask = np.isfinite(row)
            for j in range(len(ERA5_VARS)):
                if mask[j]:
                    n_total[j]  += 1
                    delta        = row[j] - mean_acc[j]
                    mean_acc[j] += delta / n_total[j]
                    M2_acc[j]   += delta * (row[j] - mean_acc[j])

    stds = np.sqrt(np.where(n_total > 1, M2_acc / (n_total - 1), 1.0))

    print(f"Skipped {skipped} stations (no ERA5Land files)")
    print(f"Total days processed: {int(n_total.min()):,}")

    result = {
        "vars":         ERA5_VARS,
        "means":        mean_acc.tolist(),
        "stds":         stds.tolist(),
        "log1p_precip": True,
        "precip_var":   "tp_sum",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved → {OUT_PATH}")
    print("\nPer-variable stats (mean ± std):")
    for v, m, s in zip(ERA5_VARS, mean_acc, stds):
        flag = " [log1p]" if v == "tp_sum" else ""
        print(f"  {v:<14s}  {m:12.4f} ± {s:.4f}{flag}")


if __name__ == "__main__":
    main()
