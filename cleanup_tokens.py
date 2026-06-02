"""
cleanup_tokens.py
=================
Two tasks:

1. Move 35 excluded stations from sm_only/sm_and_flux/flux_only to excluded_stations/
   (contents untouched).

2. For each of 993 active stations (station_splits.csv):
   - Audit: check consolidated *_L12_*.pt exists for S2L2A and S1RTC
   - geo.json: keep one renamed {station}_geo.json, delete all YYYYMMDD_*geo.json
   - Old .pt:  delete all YYYYMMDD_*_L*.pt  (per-date unconsolidated tokens)

Usage:
    python cleanup_tokens.py            # dry run (print only)
    python cleanup_tokens.py --execute  # actually move/delete
"""

import argparse
import shutil
from pathlib import Path
from datetime import date

DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
CSV_ACTIVE   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
CSV_EXCLUDED = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/excluded_stations.csv")
LOG_FILE     = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/logs.txt")


def folder_name(row) -> str:
    src = row["source_network"]
    if src == "ISMN":
        return f"ISMN_{row['network']}_{row['station_id']}"
    return f"{src}_{row['station_id']}"


def find_station_dir(station: str) -> Path | None:
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        p = DATA_ROOT / cat / station
        if p.exists():
            return p
    return None


def is_old_format(path: Path) -> bool:
    """True if filename starts with 8 digits (YYYYMMDD)."""
    return path.stem[:8].isdigit()


def has_consolidated(subdir: Path) -> bool:
    """True if subdir contains at least one consolidated *_L12_*.pt."""
    return bool(list(subdir.glob("*_L12_*.pt")))


def cleanup_subdir(subdir: Path, station: str, execute: bool) -> dict:
    """Clean geo.jsons and old .pt files in one modality dir. Returns counts."""
    counts = {"geo_kept": 0, "geo_deleted": 0, "pt_deleted": 0}
    if not subdir.exists():
        return counts

    old_geos = sorted(f for f in subdir.glob("*.json") if is_old_format(f))
    old_pts  = sorted(f for f in subdir.glob("*.pt")   if is_old_format(f))

    if old_geos:
        target = subdir / f"{station}_geo.json"
        if execute:
            if not target.exists():
                shutil.copy2(old_geos[0], target)
            for g in old_geos:
                g.unlink()
        counts["geo_kept"]    = 1
        counts["geo_deleted"] = len(old_geos)

    for pt in old_pts:
        if execute:
            pt.unlink()
        counts["pt_deleted"] += 1

    return counts


def main(execute: bool):
    import pandas as pd

    df_active   = pd.read_csv(CSV_ACTIVE)
    df_excluded = pd.read_csv(CSV_EXCLUDED)

    # ── Step 1: move excluded stations ────────────────────────────────────────
    excl_dest = DATA_ROOT / "excluded_stations"
    moved, already_there, not_found = 0, 0, 0

    print("=== Step 1: Move excluded stations ===")
    for station in df_excluded["station"]:
        src = find_station_dir(station)
        dst = excl_dest / station
        if dst.exists():
            print(f"  already moved: {station}")
            already_there += 1
        elif src is None:
            print(f"  NOT FOUND: {station}")
            not_found += 1
        else:
            print(f"  {'MOVE' if execute else 'would move'}: {src} → {dst}")
            if execute:
                excl_dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
            moved += 1

    print(f"\n  Moved: {moved}  Already there: {already_there}  Not found: {not_found}\n")

    # ── Step 2: audit + cleanup active stations ────────────────────────────────
    print("=== Step 2: Audit + cleanup active stations ===")
    total_geo_deleted = total_pt_deleted = 0
    missing_s2, missing_s1 = [], []

    for _, row in df_active.iterrows():
        station = folder_name(row)
        sdir    = find_station_dir(station)
        if sdir is None:
            missing_s2.append(station)
            missing_s1.append(station)
            continue

        s2dir = sdir / "S2L2A"
        s1dir = sdir / "S1RTC"

        # Audit
        if not s2dir.exists() or not has_consolidated(s2dir):
            missing_s2.append(station)
        if not s1dir.exists() or not has_consolidated(s1dir):
            missing_s1.append(station)

        # Cleanup geo + old .pt
        for subdir in (s2dir, s1dir):
            c = cleanup_subdir(subdir, station, execute)
            total_geo_deleted += c["geo_deleted"]
            total_pt_deleted  += c["pt_deleted"]

    print(f"\n  geo.json files deleted : {total_geo_deleted}")
    print(f"  old .pt files deleted  : {total_pt_deleted}")
    print(f"\n  Missing S2L2A consolidated ({len(missing_s2)}):")
    for s in missing_s2: print(f"    {s}")
    print(f"\n  Missing S1RTC consolidated ({len(missing_s1)}):")
    for s in missing_s1: print(f"    {s}")

    # ── Log ───────────────────────────────────────────────────────────────────
    if execute:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"\n=== {date.today()} ===\n")
            f.write(f"- cleanup_tokens.py --execute\n")
            f.write(f"  - Moved {moved} excluded stations to excluded_stations/\n")
            f.write(f"  - Deleted {total_geo_deleted} old geo.json files\n")
            f.write(f"  - Deleted {total_pt_deleted} old per-date .pt files\n")
            f.write(f"  - Missing S2 consolidated: {len(missing_s2)}\n")
            f.write(f"  - Missing S1 consolidated: {len(missing_s1)}\n")
        print(f"\nLogged to {LOG_FILE}")

    if not execute:
        print("\nRun with --execute to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Apply changes (default: dry run)")
    args = parser.parse_args()
    main(execute=args.execute)
