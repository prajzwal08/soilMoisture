#!/usr/bin/env python
"""
verify_era5_18.py
=================
Read back every `era5/values18` written by `splice_era5_radiation.py` and prove
it is what it claims to be.  §43.12.

Six checks per station:

  1. `era5/values18` and `era5/vars18` exist.
  2. Shape is `(N, 18)` with N matching `era5/values` exactly.
  3. `vars18` equals the expected 18 names, in order.
  4. No NaN and no inf anywhere.  The datasets z-score without NaN handling, so
     one NaN trains the model on `nan`.
  5. COLUMN IDENTITY -- `values18[:, 0:16]` is BIT-IDENTICAL to
     `values[:, [0..5, 9..18]]`.  This is the check that proves the splice moved
     only what it was supposed to move and nothing drifted.
  6. Physical range.  Daily `ssrd_sum` should sit in [0, 4.5e7] J m-2 (up to
     ~35 MJ on a clear summer day, 0 in polar night) and `strd_sum` in
     [5e6, 5e7] (roughly 150-450 W m-2 over 86400 s).  Values far above these
     mean the ACCUMULATED band was fetched instead of the `_hourly` one -- the
     single most likely way for this pipeline to be silently wrong.

It also reports the seasonal amplitude of `ssrd_sum` per station.  A near-flat
`ssrd_sum` is the other signature of the accumulated-band mistake, and of a
station stuck on a constant fill value.

Usage
-----
    python verify_era5_18.py                    # all 993
    python verify_era5_18.py --stations A,B
    sbatch slurm/verify_era5_18.sh

Env: `terramind` (needs zarr 2.x).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

REPO_ROOT   = Path("/gpfs/work3/0/prjs1968/soilMoisture")
ZARR_ROOT   = Path("/projects/prjs1968/zarr_tokens")
STATION_CSV = REPO_ROOT / "csvs" / "station_splits.csv"
REPORT_CSV  = REPO_ROOT / "csvs" / "era5_18_verify.csv"

EXPECTED = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum", "ssrd_sum", "strd_sum",
]
KEEP_IDX = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
N_CARRIED = len(KEEP_IDX)          # 16

SSRD_LO, SSRD_HI = 0.0, 4.5e7      # J m-2 per day
STRD_LO, STRD_HI = 5.0e6, 5.0e7


def station_rows() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV)

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    def _cat(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        return "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

    df["folder"] = df.apply(_folder, axis=1)
    df["cat"]    = df.apply(_cat, axis=1)
    return df[["folder", "cat"]].reset_index(drop=True)


def check(args) -> dict:
    folder, cat, zarr_root = args
    r = {"folder": folder, "ok": False, "fail": "", "n": 0,
         "ssrd_min": np.nan, "ssrd_max": np.nan, "ssrd_mean": np.nan,
         "ssrd_season_ratio": np.nan,
         "strd_min": np.nan, "strd_max": np.nan}

    zpath = Path(zarr_root) / cat / folder
    try:
        if not (zpath / ".complete").exists():
            r["fail"] = "no-complete"
            return r

        zg = zarr.open_group(store=zarr.DirectoryStore(str(zpath)), mode="r")

        if "era5/values18" not in zg or "era5/vars18" not in zg:
            r["fail"] = "missing-values18"
            return r

        v18 = zg["era5/values18"][:]
        v19 = zg["era5/values"][:]
        names = [str(x) for x in zg["era5/vars18"][:]]
        r["n"] = int(v18.shape[0])

        if v18.shape != (v19.shape[0], 18):
            r["fail"] = f"shape {v18.shape} vs expected ({v19.shape[0]}, 18)"
            return r
        if names != EXPECTED:
            r["fail"] = "vars18 mismatch"
            return r
        if not np.isfinite(v18).all():
            n_bad = int((~np.isfinite(v18)).sum())
            r["fail"] = f"{n_bad} non-finite values"
            return r

        # The check that matters: carried columns must be untouched, bit for bit.
        if not np.array_equal(v18[:, :N_CARRIED], v19[:, KEEP_IDX]):
            n_diff = int((v18[:, :N_CARRIED] != v19[:, KEEP_IDX]).sum())
            r["fail"] = f"carried columns differ in {n_diff} cells"
            return r

        ssrd = v18[:, EXPECTED.index("ssrd_sum")]
        strd = v18[:, EXPECTED.index("strd_sum")]
        r["ssrd_min"], r["ssrd_max"] = float(ssrd.min()), float(ssrd.max())
        r["ssrd_mean"] = float(ssrd.mean())
        r["strd_min"], r["strd_max"] = float(strd.min()), float(strd.max())
        # Seasonal amplitude: flat means something is wrong.
        r["ssrd_season_ratio"] = float(ssrd.std() / ssrd.mean()) if ssrd.mean() > 0 else 0.0

        if not (SSRD_LO <= ssrd.min() and ssrd.max() <= SSRD_HI):
            r["fail"] = f"ssrd_sum out of range [{ssrd.min():.3g}, {ssrd.max():.3g}]"
            return r
        if not (STRD_LO <= strd.min() and strd.max() <= STRD_HI):
            r["fail"] = f"strd_sum out of range [{strd.min():.3g}, {strd.max():.3g}]"
            return r

        r["ok"] = True

    except Exception as exc:
        r["fail"] = str(exc)[:200]

    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--stations", type=str, default=None)
    ap.add_argument("--zarr-root", type=Path, default=ZARR_ROOT)
    args = ap.parse_args()

    rows = station_rows()
    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        rows = rows[rows["folder"].isin(wanted)]
    print(f"verifying {len(rows)} stations under {args.zarr_root}")

    tasks = [(r.folder, r.cat, args.zarr_root) for r in rows.itertuples()]
    with Pool(args.workers) as pool:
        results = pool.map(check, tasks)

    rep = pd.DataFrame(results)
    rep.to_csv(REPORT_CSV, index=False)

    n_ok = int(rep.ok.sum())
    print(f"\nPASS {n_ok} / {len(rep)}")

    bad = rep[~rep.ok]
    if len(bad):
        print(f"\nFAIL {len(bad)}:")
        for k, v in bad["fail"].value_counts().items():
            print(f"  {k[:70]:72s} {v}")
        print(bad[["folder", "fail"]].head(15).to_string(index=False))

    good = rep[rep.ok]
    if len(good):
        print("\nssrd_sum across stations (J m-2 per day):")
        print(f"  daily max   : {good.ssrd_max.min():.3g} .. {good.ssrd_max.max():.3g}")
        print(f"  station mean: {good.ssrd_mean.min():.3g} .. {good.ssrd_mean.max():.3g}")
        print(f"  seasonal std/mean: {good.ssrd_season_ratio.min():.3f} .. "
              f"{good.ssrd_season_ratio.max():.3f}  "
              f"(median {good.ssrd_season_ratio.median():.3f})")
        flat = good[good.ssrd_season_ratio < 0.15]
        if len(flat):
            print(f"  WARNING: {len(flat)} stations have near-flat ssrd_sum "
                  f"(std/mean < 0.15) -- check for the accumulated-band mistake")
            print(flat.nsmallest(10, "ssrd_season_ratio")
                  [["folder", "ssrd_mean", "ssrd_season_ratio"]].to_string(index=False))

    print(f"\nreport -> {REPORT_CSV}")
    return 0 if len(bad) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
