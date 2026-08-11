"""Audit all 993 stations: check zarr completeness on scratch."""
import zarr
import pandas as pd
from pathlib import Path

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
splits = pd.read_csv("csvs/station_splits.csv")

results = {"ok": [], "missing_zarr": [], "no_s2": [], "no_era5": [], "no_labels": [], "no_l12": []}

for _, r in splits.iterrows():
    has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
    has_fl = str(r.get("has_flux", "False")).lower() == "true"
    cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
    if str(r["source_network"]) == "ISMN":
        dir_name = f"ISMN_{r['network']}_{r['station_name']}"
    else:
        dir_name = f"{r['source_network']}_{r['station_id']}"

    path = ZARR_ROOT / cat / dir_name
    split = r.get("split", "?")

    if not path.exists():
        results["missing_zarr"].append((dir_name, split))
        continue
    try:
        zg = zarr.open_group(str(path), mode="r")
    except Exception as e:
        results["missing_zarr"].append((dir_name, f"{split} ERR:{e}"))
        continue

    issues = []
    if "s2" not in zg or "dates" not in zg["s2"]:      issues.append("no_s2")
    if "era5" not in zg or "values" not in zg["era5"]: issues.append("no_era5")
    if "labels" not in zg or "sm" not in zg["labels"]: issues.append("no_labels")
    if "s2" in zg and "l12" not in zg["s2"]:           issues.append("no_l12")

    if issues:
        for iss in issues:
            results[iss].append((dir_name, split))
    else:
        results["ok"].append(dir_name)

print(f"Total stations  : {len(splits)}")
print(f"Fully OK        : {len(results['ok'])}")
print(f"Missing zarr    : {len(results['missing_zarr'])}")
print(f"No S2 dates     : {len(results['no_s2'])}")
print(f"No ERA5         : {len(results['no_era5'])}")
print(f"No SM labels    : {len(results['no_labels'])}")
print(f"No S2/l12       : {len(results['no_l12'])}")

for key in ("missing_zarr", "no_s2", "no_era5", "no_labels", "no_l12"):
    if results[key]:
        print(f"\n--- {key} ---")
        for name, sp in results[key][:20]:
            print(f"  [{sp}] {name}")
        if len(results[key]) > 20:
            print(f"  ... and {len(results[key])-20} more")
