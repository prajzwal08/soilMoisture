"""
fill_tabular_recovery.py

CPU-only recovery for CRITICAL stations from audit_zarr_complete.csv that are
missing soil / era5 / labels (sm and/or le) but already have all token groups
(s2/s1/dem/lulc/cm) written. Re-runs `_fill_tabular` (idempotent: only writes a
modality if it's not already present in the zarr group AND the source file
exists on disk) and consolidates metadata.

Stations that also need cloud-mask inference (MISSING_cm) are handled
separately by `process_cm_fill --station` (GPU, sensei env), which calls
`_fill_tabular` itself — they are skipped here.

Usage:
    conda run --no-capture-output -n sensei python fill_tabular_recovery.py [--execute]
"""

import argparse
from pathlib import Path

import pandas as pd
import zarr

from retokenize_satellite_zarr import ZARR_ROOT, SPLITS_CSV, _fill_tabular


def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually write to zarr. Without this, dry-run only.")
    args = parser.parse_args()

    audit = pd.read_csv("csvs/audit_zarr_complete.csv")
    crit  = audit[audit["status"] == "CRITICAL"]

    # Skip stations needing cm inference — handled by process_cm_fill --station.
    crit = crit[~crit["flags"].str.contains("MISSING_cm")]

    # Only stations where the gap is soil/era5/labels (fill-only recovery).
    fill_flags = ("MISSING_soil", "MISSING_era5", "MISSING_labels_sm", "MISSING_labels_le")
    crit = crit[crit["flags"].apply(lambda f: any(x in f for x in fill_flags))]

    splits = pd.read_csv(SPLITS_CSV)
    splits = splits.dropna(subset=["source_network", "network", "station_id"])
    splits["cat"]      = splits.apply(_category, axis=1)
    splits["dir_name"] = splits.apply(_dir_name, axis=1)
    lookup = splits.set_index("dir_name")[["cat", "start_date", "end_date"]].to_dict("index")

    print(f"{len(crit)} stations to process (fill-only, MISSING_cm excluded)")

    for _, row in crit.iterrows():
        dir_name = row["station"]
        info = lookup.get(dir_name)
        if info is None:
            print(f"SKIP   {dir_name}: not found in station_splits.csv")
            continue
        cat, start_date, end_date = info["cat"], info["start_date"], info["end_date"]
        token_dir = ZARR_ROOT / cat / dir_name
        if not token_dir.exists():
            print(f"MISS   {dir_name}: {token_dir} does not exist")
            continue

        if not args.execute:
            print(f"DRY    {dir_name}  [{cat}]  flags={row['flags']}")
            continue

        store = zarr.DirectoryStore(str(token_dir))
        token_root = zarr.open_group(store=store, mode="a")
        check_paths = ["soil", "era5/values", "sif/values",
                        "labels/sm", "labels/le"]
        before = {p: (p in token_root) for p in check_paths}
        _fill_tabular(token_root, cat, dir_name, str(start_date), str(end_date))
        after = {p: (p in token_root) for p in check_paths}
        zarr.consolidate_metadata(store)
        added = [p for p in check_paths if after[p] and not before[p]]
        print(f"OK     {dir_name}  [{cat}]  added={added}")


if __name__ == "__main__":
    main()
