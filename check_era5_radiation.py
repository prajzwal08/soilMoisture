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

# ALIGNED TO verify_era5_18.py (§45).  These were 0/3.5e7 and 1.0e7/4.0e7, TIGHTER
# than the downstream verify, so this script failed on data verify would have passed
# -- the check must not be stricter than the gate it feeds.  Both false alarms it
# raised on the completed download were this, not the data:
#   * strd floor 1.0e7 cut a measured global minimum of 8.675 MJ (100 W m-2) at 32
#     stations, all Tibetan Plateau / interior Alaska / glacier / MT-MN winter, every
#     one of them on or above the Brutsaert clear-sky bound.
#   * SEASON_MIN 0.15 cut 3 Sahel stations at ~13 N whose measured std/mean is
#     0.106-0.139 at a mean of 22-23 MJ.  A weak solar cycle at 13 N is correct;
#     0.08 still catches a genuinely constant series, which reads ~0.
SSRD_LO, SSRD_HI = 0.0, 4.5e7     # J m-2 per day
STRD_LO, STRD_HI = 5.0e6, 5.0e7
SEASON_MIN       = 0.08           # std/mean of ssrd_sum


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
            # §45: the accumulated-band question is answerable from PROVENANCE, not
            # from magnitudes.  download_era5_radiation.py:380-390 stamps the bands it
            # requested into every file, and :107-112 requests the de-accumulated
            # `_hourly` variants.  This one assertion is stronger and cheaper than
            # every range argument below it.
            bands = str(ds.attrs.get("bands", ""))
        n = len(ssrd)
        exp_n = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        rows.append({
            "station": station, "year": year, "strategy": strategy,
            "bands": bands, "bands_hourly": "_hourly" in bands,
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
    if (~d.bands_hourly).any():
        fails.append(f"{int((~d.bands_hourly).sum())} file(s) NOT built from the "
                     f"de-accumulated `_hourly` bands -- ACCUMULATED BAND, confirmed "
                     f"from provenance")
    if (d.ssrd_min < SSRD_LO).any() or (d.ssrd_max > SSRD_HI).any():
        fails.append(f"ssrd_sum outside [{SSRD_LO:.3g}, {SSRD_HI:.3g}] J m-2 "
                     f"(min {d.ssrd_min.min():.4g}, max {d.ssrd_max.max():.4g})")
    if (d.strd_min < STRD_LO).any() or (d.strd_max > STRD_HI).any():
        fails.append(f"strd_sum outside [{STRD_LO:.3g}, {STRD_HI:.3g}] J m-2 "
                     f"(min {d.strd_min.min():.4g}, max {d.strd_max.max():.4g})")
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

    # The old text here advertised "strd_mean 20-35 MJ" while the test is on
    # strd_min/strd_max -- Ngari's strd_mean is 16.3 MJ, outside the advertised range,
    # untested, and fine. State what is ACTUALLY tested.
    print(f"\ntested, per file (the table above shows MEANS, which are NOT tested):")
    print(f"  bands contain '_hourly'                     (provenance, the real band check)")
    print(f"  ssrd_sum every day within [{SSRD_LO:.3g}, {SSRD_HI:.3g}] J m-2")
    print(f"  strd_sum every day within [{STRD_LO:.3g}, {STRD_HI:.3g}] J m-2")
    print(f"  ssrd_sum std/mean >= {SEASON_MIN}")
    print(f"  day count matches the calendar year, and no NaN")
    print(f"  ranges match verify_era5_18.py -- a check stricter than the gate it "
          f"feeds is a bug")

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
