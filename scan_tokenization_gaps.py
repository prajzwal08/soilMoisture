"""
scan_tokenization_gaps.py

For every station, compare satellite_zarr/{station}.zarr (raw downloaded
imagery) against the production token zarr (ZARR_ROOT/{category}/{station}/)
to find dates that exist in satellite_zarr but were never tokenized into
s2/s1_asc/s1_desc (and therefore have no cloud mask either).

This is the same gap pattern found for AmeriFlux_US-xUN (80 missing S2
dates from 2022-2024 sitting in satellite_zarr, never run through Phase 1
TerraMind tokenization).

Usage:
    conda run --no-capture-output -n sensei python scan_tokenization_gaps.py [--workers N]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import zarr

ZARR_ROOT     = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ZARR_ROOT = Path("/projects/prjs1968/satellite_zarr")


def _dates(zg, key):
    if f"{key}/dates" not in zg:
        return set()
    return set(zg[f"{key}/dates"][:].astype(str).tolist())


def scan(args):
    station, cat = args
    sat_path = SAT_ZARR_ROOT / f"{station}.zarr"
    if not sat_path.exists():
        return {"station": station, "category": cat, "has_sat_zarr": False}

    try:
        sat = zarr.open_group(str(sat_path), mode="r")
        tok = zarr.open_group(str(ZARR_ROOT / cat / station), mode="r")

        result = {"station": station, "category": cat, "has_sat_zarr": True}
        for key in ("s2", "s1_asc", "s1_desc"):
            sat_dates = _dates(sat, key)
            tok_dates = _dates(tok, key)
            extra = sat_dates - tok_dates
            result[f"{key}_n_sat"]   = len(sat_dates)
            result[f"{key}_n_tok"]   = len(tok_dates)
            result[f"{key}_n_extra"] = len(extra)
        return result
    except Exception as exc:
        return {"station": station, "category": cat, "has_sat_zarr": True, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv("csvs/audit_zarr_complete.csv")
    jobs = list(zip(df["station"], df["category"]))

    with Pool(args.workers) as pool:
        results = pool.map(scan, jobs)

    out = pd.DataFrame(results)
    out.to_csv("csvs/tokenization_gap_scan.csv", index=False)

    print(f"Total stations: {len(out)}")
    print(f"Stations with satellite_zarr: {out['has_sat_zarr'].sum()}")
    print(f"Stations WITHOUT satellite_zarr: {(~out['has_sat_zarr']).sum()}")

    has_sat = out[out["has_sat_zarr"] & out.get("error").isna()] if "error" in out.columns else out[out["has_sat_zarr"]]
    for key in ("s2", "s1_asc", "s1_desc"):
        col = f"{key}_n_extra"
        if col in has_sat.columns:
            n_gap = (has_sat[col] > 0).sum()
            total_extra = has_sat[col].sum()
            print(f"{key}: {n_gap} stations with extra un-tokenized dates (total {total_extra} dates)")

    gap_stations = has_sat[
        (has_sat.get("s2_n_extra", 0) > 0)
        | (has_sat.get("s1_asc_n_extra", 0) > 0)
        | (has_sat.get("s1_desc_n_extra", 0) > 0)
    ]
    print(f"\nTotal stations needing re-tokenization: {len(gap_stations)}")
    print(gap_stations[["station","category","s2_n_sat","s2_n_tok","s2_n_extra",
                         "s1_asc_n_sat","s1_asc_n_tok","s1_asc_n_extra",
                         "s1_desc_n_sat","s1_desc_n_tok","s1_desc_n_extra"]].to_string(index=False))


if __name__ == "__main__":
    main()
