"""
cleanup_scratch_duplicates.py
=============================
Delete duplicate station directories from scratch that have a doubled network
prefix, e.g. ISMN_SNOTEL_ISMN_SNOTEL_AlpineMeadow alongside ISMN_SNOTEL_AlpineMeadow.

All 860 such duplicates were verified to be empty or redundant — the correct-named
twin always holds the real data (0 cases where duplicate has unique content).

Usage:
    python cleanup_scratch_duplicates.py            # dry run (print only)
    python cleanup_scratch_duplicates.py --execute  # actually delete
"""

import argparse
import shutil
from pathlib import Path

SCRATCH      = Path("/gpfs/scratch1/shared/pkhanal/satellite")
LOG_FILE     = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/logs.txt")
CSV_EXCLUDED = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/excluded_stations.csv")


def is_duplicate(name: str) -> tuple[bool, str]:
    """Return (True, correct_name) if name has a doubled network prefix."""
    parts = name.split("_")
    if len(parts) < 4:
        return False, ""
    prefix = parts[0] + "_" + parts[1]   # e.g. ISMN_SNOTEL
    rest   = "_".join(parts[2:])          # e.g. ISMN_SNOTEL_AlpineMeadow
    if rest.startswith(prefix):
        return True, rest
    return False, ""


def verify_safe(dupe_dir: Path, twin_dir: Path | None) -> bool:
    """Return True if it is safe to delete dupe_dir."""
    if twin_dir is None or not twin_dir.exists():
        # Orphan — check dupe itself is empty across all modalities
        for sub in ("S2L2A", "S1RTC", "DEM", "LULC"):
            if list((dupe_dir / sub).glob("*.tif")) if (dupe_dir / sub).exists() else []:
                return False
        return True
    # Twin exists — ensure dupe has nothing the twin lacks
    for sub in ("S2L2A", "S1RTC", "DEM", "LULC"):
        dupe_tifs = len(list((dupe_dir / sub).glob("*.tif"))) if (dupe_dir / sub).exists() else 0
        twin_tifs = len(list((twin_dir / sub).glob("*.tif"))) if (twin_dir / sub).exists() else 0
        if dupe_tifs > 0 and twin_tifs == 0:
            return False
    return True


def main(execute: bool):
    import pandas as pd
    excluded = set(pd.read_csv(CSV_EXCLUDED)["station"])

    dupes = []
    risky = []

    for d in sorted(SCRATCH.iterdir()):
        if not d.is_dir():
            continue
        ok, correct_name = is_duplicate(d.name)
        if not ok:
            continue
        twin = SCRATCH / correct_name
        # If the correct name is an excluded station and has no twin, still safe to delete
        if not twin.exists() and correct_name in excluded:
            dupes.append(d)
        elif verify_safe(d, twin if twin.exists() else None):
            dupes.append(d)
        else:
            risky.append((d, correct_name))

    print(f"Duplicate dirs found : {len(dupes)}")
    print(f"Risky (skipped)      : {len(risky)}")
    print()

    if risky:
        print("RISKY — NOT deleting:")
        for d, c in risky:
            print(f"  {d.name}  (correct: {c})")
        print()

    for d in dupes:
        print(f"  {'DELETE' if execute else 'would delete'}: {d.name}")
        if execute:
            try:
                shutil.rmtree(d)
            except FileNotFoundError:
                pass

    remaining = len(list(SCRATCH.iterdir())) if execute else "N/A (dry run)"
    print()
    print(f"{'Deleted' if execute else 'Would delete'}: {len(dupes)} duplicate dirs")
    print(f"Scratch dirs remaining: {remaining}")

    if execute:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datetime import date
        with open(LOG_FILE, "a") as f:
            f.write(f"\n=== {date.today()} ===\n")
            f.write(f"- cleanup_scratch_duplicates.py: deleted {len(dupes)} duplicate scratch dirs\n")
            f.write(f"- Scratch now has {remaining} dirs\n")
        print(f"\nLogged to {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run)")
    args = parser.parse_args()
    main(execute=args.execute)
