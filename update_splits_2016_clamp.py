"""
update_splits_2016_clamp.py

One-off update to csvs/station_splits.csv:
  - Add `actual_start_date` column = original `start_date` (preserves the true
    record start used for n_years / oot_eligible / split derivation).
  - Clamp `start_date` = max(start_date, 20160101) — no satellite (S1/S2)
    imagery exists before 2016-01-01 anywhere in the dataset.

All other columns (n_years, split, oot_eligible, oost_eligible,
location_group_id, ablation_*, ...) are left unchanged.

Usage:
    python update_splits_2016_clamp.py
"""

import pandas as pd

CSV = "csvs/station_splits.csv"

df = pd.read_csv(CSV)

df["actual_start_date"] = df["start_date"]
df["start_date"] = df["start_date"].clip(lower=20160101)

n_clamped = (df["start_date"] != df["actual_start_date"]).sum()
print(f"Clamped start_date for {n_clamped} stations")

# Place actual_start_date right after start_date
cols = list(df.columns)
cols.remove("actual_start_date")
idx = cols.index("start_date") + 1
cols.insert(idx, "actual_start_date")
df = df[cols]

df.to_csv(CSV, index=False)
print(f"Wrote {CSV}")
