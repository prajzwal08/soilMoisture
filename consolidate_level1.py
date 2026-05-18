"""
Consolidate level1 label files from three sources into one organised directory.

Output structure:
    {out_root}/
        sm_only/         ISMN stations (soil moisture, no flux)
        sm_and_flux/     ICOS + AmeriFlux stations with both SM and flux
        flux_only/       ICOS + AmeriFlux flux-only stations

Naming convention:
    ISMN      → ISMN_{network}_{station_id}.nc
    ICOS      → ICOS_{station_id}.nc
    AmeriFlux → AmeriFlux_{station_id}.nc

Usage:
    python consolidate_level1.py             # dry run (print only)
    python consolidate_level1.py --execute   # actually copy files
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/home/khanalp/data/soilmoisture")
CSV_PATH   = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_splits.csv")
OUT_ROOT   = DATA_ROOT / "level1_organised"

SRC_DIRS = {
    "ISMN":      DATA_ROOT / "level1",
    "ICOS_swc":  DATA_ROOT / "icos_level1" / "swc_and_flux",
    "ICOS_flux": DATA_ROOT / "icos_level1" / "flux_only",
    "AF_swc":    DATA_ROOT / "ameriflux_level1" / "swc_and_flux",
    "AF_flux":   DATA_ROOT / "ameriflux_level1" / "flux_only",
}


def find_source_file(row: pd.Series) -> Path | None:
    """Locate the existing .nc file for a station."""
    src   = row["source_network"]
    net   = row["network"]
    sid   = row["station_id"]
    has_sm = row["has_soil_moisture"]
    has_fx = row["has_flux"]

    if src == "ISMN":
        # pattern: {network}_{station_id}_*.nc  in level1/
        matches = list(SRC_DIRS["ISMN"].glob(f"{net}_{sid}_*.nc"))

    elif src == "ICOS":
        subdir = "ICOS_swc" if has_sm else "ICOS_flux"
        matches = list(SRC_DIRS[subdir].glob(f"{sid}_*.nc"))

    elif src == "AmeriFlux":
        subdir = "AF_swc" if has_sm else "AF_flux"
        matches = list(SRC_DIRS[subdir].glob(f"{sid}_*.nc"))

    else:
        return None

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  WARNING: multiple matches for {src} {sid}: {matches}")
        return matches[0]
    return None


def target_subdir(row: pd.Series) -> str:
    if row["has_soil_moisture"] and row["has_flux"]:
        return "sm_and_flux"
    if row["has_soil_moisture"] and not row["has_flux"]:
        return "sm_only"
    return "flux_only"


def target_filename(row: pd.Series) -> str:
    src = row["source_network"]
    if src == "ISMN":
        return f"ISMN_{row['network']}_{row['station_id']}.nc"
    return f"{src}_{row['station_id']}.nc"


def main(execute: bool):
    df = pd.read_csv(CSV_PATH)
    print(f"Stations in CSV: {len(df)}")

    counts = {"copied": 0, "missing": 0}

    for _, row in df.iterrows():
        src_file = find_source_file(row)
        subdir   = target_subdir(row)
        fname    = target_filename(row)
        dst_file = OUT_ROOT / subdir / fname

        if src_file is None:
            print(f"  MISSING: {row['source_network']} {row['station_id']}")
            counts["missing"] += 1
            continue

        print(f"  {src_file.name}  →  {subdir}/{fname}")

        if execute:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            counts["copied"] += 1

    print(f"\n{'Copied' if execute else 'Would copy'}: {counts['copied'] + (0 if execute else len(df) - counts['missing'])}")
    print(f"Missing: {counts['missing']}")
    if not execute:
        print("\nRun with --execute to actually copy files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Copy files (default: dry run)")
    args = parser.parse_args()
    main(execute=args.execute)
