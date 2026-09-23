# ============================================================
# Station duration audit
# ============================================================
# The level-1 NetCDFs were produced with a 365-day minimum
# ("<1 year of valid daily data" in station_metadata.csv), but
# preprocessing_ISMN_soilMoisture.py:228 now requires 1095 days
# (3 years) and was never re-run.  This counts how many stations
# in the active inventory fall below the current threshold.
#
# Duration is taken from the NetCDF itself (n_days_total, and the
# per-depth observed/gap-filled counts) rather than from any CSV,
# so the answer reflects the data the model actually loads.
#
# Usage:
#   python audit_station_duration.py [--min-days 1095]
# ============================================================

import argparse
import re
from pathlib import Path

import pandas as pd
import xarray as xr

SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
META_CSV   = Path("/gpfs/work3/0/prjs1968/raw_soil_moisture/station_metadata.csv")
L1_DIRS    = [Path("/gpfs/work3/0/prjs1968/level1_organised"),
              Path("/gpfs/work3/0/prjs1968/raw_soil_moisture")]
OUT_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_duration_audit.csv")


def index_level1():
    """Map both naming conventions to a path.

    level1_organised: ISMN_{network}_{station}.nc
    raw_soil_moisture: {network}_{station}_{start}_{end}.nc
    """
    idx = {}
    for d in L1_DIRS:
        for p in d.rglob("*.nc"):
            stem = p.stem
            idx[stem] = p
            # strip the _{start}_{end} suffix if present
            m = re.match(r"^(.*)_\d{8}_\d{8}$", stem)
            if m:
                idx.setdefault(m.group(1), p)
            # strip a leading source prefix (ISMN_, ICOS_, ...)
            for pre in ("ISMN_", "ICOS_", "AmeriFlux_"):
                if stem.startswith(pre):
                    idx.setdefault(stem[len(pre):], p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=1095)
    args = ap.parse_args()

    splits = pd.read_csv(SPLITS_CSV)
    print(f"station_splits.csv : {len(splits)} rows")
    print(f"  split counts     : {splits['split'].value_counts().to_dict()}")

    idx = index_level1()
    print(f"level-1 files      : {len(set(idx.values()))} unique\n")

    rows = []
    for _, r in splits.iterrows():
        net, sta = str(r["network"]), str(r["station_id"])
        path = (idx.get(f"{net}_{sta}") or idx.get(sta)
                or idx.get(f"{r['source_network']}_{net}_{sta}"))
        rec = {
            "source_network": r["source_network"], "network": net, "station_id": sta,
            "split": r["split"], "n_years_csv": r.get("n_years"),
            "oot_eligible": r.get("oot_eligible"), "oost_eligible": r.get("oost_eligible"),
            "start_date": r.get("actual_start_date"), "end_date": r.get("end_date"),
        }
        if path is None:
            rec.update(n_days=pd.NA, n_observed_surface=pd.NA, found=False)
        else:
            with xr.open_dataset(path) as ds:
                rec["n_days"] = int(ds.sizes.get("date_time", 0))
                obs = ds.attrs.get("n_observed_0_10")
                rec["n_observed_surface"] = int(obs) if obs is not None else pd.NA
                rec["found"] = True
                rec["file"] = path.name
        rows.append(rec)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    missing = df[~df["found"].astype(bool)]
    have    = df[df["found"].astype(bool)].copy()
    have["n_days"] = have["n_days"].astype(int)

    print(f"matched to a level-1 file : {len(have)}")
    print(f"NOT matched               : {len(missing)}")
    if len(missing):
        print("  (first 10)", missing["station_id"].head(10).tolist())

    short = have[have["n_days"] < args.min_days]
    print(f"\n=== stations under {args.min_days} days ({args.min_days/365.25:.1f} yr) ===")
    print(f"  count : {len(short)} / {len(have)}  ({100*len(short)/max(len(have),1):.1f}%)")
    if len(short):
        print("\n  by split:")
        print(short["split"].value_counts().to_string())
        print("\n  by source_network:")
        print(short["source_network"].value_counts().to_string())
        print("\n  shortest 25:")
        cols = ["source_network", "network", "station_id", "split", "n_days", "n_years_csv"]
        print(short.nsmallest(25, "n_days")[cols].to_string(index=False))
        print("\n  of these, flagged for temporal holdout:")
        print(f"    oot_eligible  : {int(short['oot_eligible'].sum())}")
        print(f"    oost_eligible : {int(short['oost_eligible'].sum())}")

    print("\n=== duration distribution (days) ===")
    print(have["n_days"].describe().to_string())
    for cut in (365, 730, 1095, 1460, 1825):
        n = int((have["n_days"] < cut).sum())
        print(f"  < {cut:5d} d ({cut/365.25:.1f} yr): {n:4d}")

    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
