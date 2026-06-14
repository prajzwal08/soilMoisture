"""Check SM label values across all val stations for out-of-range values."""
import zarr
import numpy as np
import pandas as pd
from pathlib import Path

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
splits = pd.read_csv("csvs/station_splits.csv")

# Filter to val stations only (or change to "train" / all)
target_splits = ["val"]
splits = splits[splits["split"].isin(target_splits)]

print(f"Checking {len(splits)} val stations for out-of-range SM labels...\n")
print(f"{'Station':<55} {'depth':<10} {'min':>8} {'max':>8} {'n_neg':>8} {'n_gt06':>8} {'n_total':>8}")
print("-" * 110)

flagged = []

for _, r in splits.iterrows():
    has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
    has_fl = str(r.get("has_flux", "False")).lower() == "true"
    cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
    if str(r["source_network"]) == "ISMN":
        dir_name = f"ISMN_{r['network']}_{r['station_name']}"
    else:
        dir_name = f"{r['source_network']}_{r['station_id']}"

    path = ZARR_ROOT / cat / dir_name
    try:
        zg = zarr.open_group(str(path), mode="r")
    except Exception:
        continue

    if "labels" not in zg or "sm" not in zg["labels"]:
        continue

    sm = zg["labels/sm"][:]        # shape: (n_days, n_depths)
    depths = list(zg["labels/depths"][:]) if "depths" in zg["labels"] else [f"d{i}" for i in range(sm.shape[1])]

    for i, depth in enumerate(depths):
        vals = sm[:, i]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        vmin, vmax = float(vals.min()), float(vals.max())
        n_neg  = int((vals < 0).sum())
        n_gt06 = int((vals > 0.6).sum())
        if n_neg > 0 or n_gt06 > 0:
            print(f"{dir_name:<55} {str(depth):<10} {vmin:>8.4f} {vmax:>8.4f} {n_neg:>8d} {n_gt06:>8d} {len(vals):>8d}")
            flagged.append(dir_name)

print(f"\n{'='*110}")
print(f"Stations with out-of-range SM values: {len(set(flagged))}")
if not flagged:
    print("All val stations have SM values in [0, 0.6].")
