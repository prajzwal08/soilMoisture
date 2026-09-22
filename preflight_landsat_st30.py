#!/usr/bin/env python
"""Dry run for the 993-station Landsat ST pull: check every output path BEFORE writing any.

The folder name is {source_network}_{network}_{station_id} and the category is derived from
has_soil_moisture / has_flux.  If either disagrees with what the other modalities already built,
the pull silently creates 993 orphan directories next to the real ones.  Cheaper to check.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, SPLITS, category_of, station_folder, station_grid30

df = pd.read_csv(SPLITS)          # pandas, never awk
rows = list(df.itertuples(index=False))
print(f"stations in {SPLITS.name}: {len(rows)}")

cat_count, exists, missing, bad_grid = Counter(), 0, [], []
for r in rows:
    cat, folder = category_of(r), station_folder(r)
    cat_count[cat] += 1
    d = DATA_ROOT / cat / folder
    if d.is_dir():
        exists += 1
    else:
        missing.append(f"{cat}/{folder}")
    try:
        g = station_grid30(r)
        w, s, e, n = g["bounds"]
        if round(e - w) != 2280 or round(n - s) != 2280:
            bad_grid.append(f"{folder}: {e-w} x {n-s} m")
    except Exception as exc:
        bad_grid.append(f"{folder}: {str(exc)[:80]}")

print("\ncategory routing:")
for c in ("sm_only", "sm_and_flux", "flux_only"):
    on_disk = len(list((DATA_ROOT / c).iterdir())) if (DATA_ROOT / c).is_dir() else 0
    print(f"  {c:<14} csv={cat_count[c]:<5} dirs_on_disk={on_disk}")

print(f"\nstation dirs that already exist : {exists} / {len(rows)}")
print(f"would be CREATED (orphan risk)  : {len(missing)}")
for m in missing[:20]:
    print(f"    {m}")
if len(missing) > 20:
    print(f"    ... and {len(missing)-20} more")

print(f"\ngrid construction failures      : {len(bad_grid)}")
for b in bad_grid[:10]:
    print(f"    {b}")

print("\nEPSG zones spanned:", len({station_grid30(r)["epsg"] for r in rows}))
sys.exit(1 if (missing or bad_grid) else 0)
