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
    python cleanup_tokens.py                    # dry run (print only)
    python cleanup_tokens.py --execute          # run with default 8 workers
    python cleanup_tokens.py --execute --workers 16
"""

import argparse
import shutil
from pathlib import Path
from datetime import date
from multiprocessing import Pool

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
    return path.stem[:8].isdigit()


def has_consolidated(subdir: Path) -> bool:
    return bool(list(subdir.glob("*_L12_*.pt")))


def cleanup_subdir(subdir: Path, station: str, execute: bool) -> dict:
    counts = {"geo_deleted": 0, "pt_deleted": 0}
    if not subdir.exists():
        return counts

    old_geos = sorted(f for f in subdir.glob("*.json") if is_old_format(f))
    old_pts  = sorted(f for f in subdir.glob("*.pt")   if is_old_format(f))

    if not has_consolidated(subdir):
        return counts

    for pt in old_pts:
        if execute:
            pt.unlink(missing_ok=True)
        counts["pt_deleted"] += 1

    if old_geos:
        target = subdir / f"{station}_geo.json"
        if execute:
            if not target.exists():
                try:
                    old_geos[0].rename(target)
                except FileNotFoundError:
                    pass
                for g in old_geos[1:]:
                    g.unlink(missing_ok=True)
            else:
                for g in old_geos:
                    g.unlink(missing_ok=True)
        counts["geo_deleted"] = len(old_geos)

    return counts


def _worker(args: tuple) -> tuple:
    """Process one station; returns (station, missing_s2, missing_s1, geo_del, pt_del)."""
    station, execute = args
    sdir = find_station_dir(station)
    if sdir is None:
        return station, True, True, 0, 0

    s2dir = sdir / "S2L2A"
    s1dir = sdir / "S1RTC"

    missing_s2 = not s2dir.exists() or not has_consolidated(s2dir)
    missing_s1 = not s1dir.exists() or not has_consolidated(s1dir)

    geo_del = pt_del = 0
    for subdir in (s2dir, s1dir):
        c = cleanup_subdir(subdir, station, execute)
        geo_del += c["geo_deleted"]
        pt_del  += c["pt_deleted"]

    return station, missing_s2, missing_s1, geo_del, pt_del


def main(execute: bool, workers: int):
    import pandas as pd

    df_active   = pd.read_csv(CSV_ACTIVE)
    df_excluded = pd.read_csv(CSV_EXCLUDED)

    # ── Step 1: move excluded stations (small, keep sequential) ───────────────
    excl_dest = DATA_ROOT / "excluded_stations"
    moved, already_there, not_found = 0, 0, 0

    print("=== Step 1: Move excluded stations ===", flush=True)
    for station in df_excluded["station"]:
        src = find_station_dir(station)
        dst = excl_dest / station
        if dst.exists():
            already_there += 1
        elif src is None:
            print(f"  NOT FOUND: {station}", flush=True)
            not_found += 1
        else:
            print(f"  {'MOVE' if execute else 'would move'}: {src} → {dst}", flush=True)
            if execute:
                excl_dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
            moved += 1

    print(f"  Moved: {moved}  Already there: {already_there}  Not found: {not_found}\n", flush=True)

    # ── Step 2: parallel audit + cleanup ──────────────────────────────────────
    stations = [folder_name(row) for _, row in df_active.iterrows()]
    print(f"=== Step 2: Audit + cleanup {len(stations)} stations with {workers} workers ===", flush=True)

    total_geo_del = total_pt_del = 0
    missing_s2, missing_s1 = [], []
    done = 0

    args_list = [(s, execute) for s in stations]
    with Pool(workers) as pool:
        for station, ms2, ms1, geo_del, pt_del in pool.imap_unordered(_worker, args_list, chunksize=4):
            done += 1
            total_geo_del += geo_del
            total_pt_del  += pt_del
            if ms2:
                missing_s2.append(station)
            if ms1:
                missing_s1.append(station)
            if done % 100 == 0 or done == len(stations):
                print(f"  [{done}/{len(stations)}] geo_deleted={total_geo_del} pt_deleted={total_pt_del}", flush=True)

    print(f"\n  geo.json files deleted : {total_geo_del}")
    print(f"  old .pt files deleted  : {total_pt_del}")
    print(f"\n  Missing S2L2A consolidated: {len(missing_s2)}")
    for s in missing_s2: print(f"    {s}")
    print(f"\n  Missing S1RTC consolidated: {len(missing_s1)}")
    for s in missing_s1: print(f"    {s}")

    if execute:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"\n=== {date.today()} ===\n")
            f.write(f"- cleanup_tokens.py --execute --workers {workers}\n")
            f.write(f"  - Moved {moved} excluded stations to excluded_stations/\n")
            f.write(f"  - Deleted {total_geo_del} old geo.json files\n")
            f.write(f"  - Deleted {total_pt_del} old per-date .pt files\n")
            f.write(f"  - Missing S2 consolidated: {len(missing_s2)}\n")
            f.write(f"  - Missing S1 consolidated: {len(missing_s1)}\n")
        print(f"\nLogged to {LOG_FILE}", flush=True)
    else:
        print("\nRun with --execute to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute",  action="store_true")
    parser.add_argument("--workers",  type=int, default=8)
    args = parser.parse_args()
    main(execute=args.execute, workers=args.workers)
