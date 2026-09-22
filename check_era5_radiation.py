#!/usr/bin/env python
"""
check_era5_radiation.py
=======================
Sanity-check the `rad_{year}.nc` files written by `download_era5_radiation.py`
BEFORE any of them reach the splice.  §43.12 run-order step 2.

The single most likely way this pipeline is silently wrong is fetching the
ACCUMULATED radiation band instead of the `_hourly` de-accumulated one.  The plain
band accumulates from 00 UTC, so summing 24 of its values gives roughly 12x the
day's true energy and destroys the diurnal information.  Two signatures catch it:

  1. MAGNITUDE.  Daily ssrd_sum belongs in ~[0, 3.5e7] J m-2 (up to ~35 MJ on a
     clear summer day at low latitude, 0 in polar night).  strd_sum belongs in
     ~[1.0e7, 4.0e7] (roughly 120-460 W m-2 held over 86400 s).
  2. SEASONALITY.  ssrd_sum is the most strongly seasonal quantity in the whole
     driver stack.  std/mean below ~0.15 means it is not tracking the sun, which
     is either the accumulated band or a constant fill.

It also checks the day count per file, which catches a silently truncated year,
and reports the ssrd/strd ratio -- shortwave should exceed longwave in summer at
most sites and fall well below it in winter, so a ratio pinned near a constant is
another way to see a broken fetch.

Usage
-----
    python check_era5_radiation.py                       # every rad_*.nc on disk
    python check_era5_radiation.py --stations A,B,C
    sbatch slurm/check_era5_radiation.sh

Env: `soilmoisture` or `terramind` -- needs only xarray + pandas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
REPO_ROOT = Path("/gpfs/work3/0/prjs1968/soilMoisture")
OUT_CSV   = REPO_ROOT / "csvs" / "era5_radiation_check.csv"

SSRD_LO, SSRD_HI = 0.0, 3.5e7     # J m-2 per day
STRD_LO, STRD_HI = 1.0e7, 4.0e7
SEASON_MIN       = 0.15           # std/mean of ssrd_sum


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", type=str, default=None)
    args = ap.parse_args()

    files = sorted(DATA_ROOT.glob("*/*/ERA5Land/rad_????.nc"))
    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        files = [f for f in files if f.parent.parent.name in wanted]
    if not files:
        print("no rad_*.nc found")
        return 1
    print(f"checking {len(files)} file(s)\n")

    rows = []
    for f in files:
        station, year = f.parent.parent.name, int(f.stem.split("_")[1])
        with xr.open_dataset(f) as ds:
            ssrd = ds["ssrd_sum"].values.astype(np.float64)
            strd = ds["strd_sum"].values.astype(np.float64)
            strategy = ds.attrs.get("strategy", "?")
        n = len(ssrd)
        exp_n = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        rows.append({
            "station": station, "year": year, "strategy": strategy,
            "n_days": n, "n_expected": exp_n, "n_ok": n == exp_n,
            "n_nan": int(np.isnan(ssrd).sum() + np.isnan(strd).sum()),
            "ssrd_min": ssrd.min(), "ssrd_max": ssrd.max(), "ssrd_mean": ssrd.mean(),
            "ssrd_season": ssrd.std() / ssrd.mean() if ssrd.mean() > 0 else 0.0,
            "strd_min": strd.min(), "strd_max": strd.max(), "strd_mean": strd.mean(),
            "ratio_mean": ssrd.mean() / strd.mean() if strd.mean() > 0 else np.nan,
        })

    d = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)

    fails = []
    if (~d.n_ok).any():
        fails.append(f"{int((~d.n_ok).sum())} file(s) with wrong day count")
    if (d.n_nan > 0).any():
        fails.append(f"{int((d.n_nan > 0).sum())} file(s) contain NaN")
    if (d.ssrd_min < SSRD_LO).any() or (d.ssrd_max > SSRD_HI).any():
        fails.append("ssrd_sum out of physical range -- ACCUMULATED BAND?")
    if (d.strd_min < STRD_LO).any() or (d.strd_max > STRD_HI).any():
        fails.append("strd_sum out of physical range -- ACCUMULATED BAND?")
    if (d.ssrd_season < SEASON_MIN).any():
        fails.append(f"{int((d.ssrd_season < SEASON_MIN).sum())} file(s) with "
                     f"near-flat ssrd_sum (std/mean < {SEASON_MIN})")

    pd.set_option("display.width", 200)
    print(d[["station", "year", "strategy", "n_days", "n_expected", "n_nan",
             "ssrd_mean", "ssrd_max", "ssrd_season", "strd_mean", "ratio_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n--- per station ---")
    g = d.groupby("station").agg(
        files=("year", "count"),
        ssrd_mean_MJ=("ssrd_mean", lambda s: s.mean() / 1e6),
        ssrd_max_MJ=("ssrd_max", lambda s: s.max() / 1e6),
        ssrd_season=("ssrd_season", "mean"),
        strd_mean_MJ=("strd_mean", lambda s: s.mean() / 1e6),
    )
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nexpected: ssrd_mean 5-25 MJ, ssrd_max 20-35 MJ, ssrd_season > 0.15,")
    print("          strd_mean 20-35 MJ")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        print(f"\nreport -> {OUT_CSV}")
        return 2

    print(f"\nPASS -- all {len(d)} file(s) physical.  report -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
