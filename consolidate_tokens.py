"""
consolidate_tokens.py
=====================
Bundle per-date .pt token files (output of precompute_terramind.py) into
consolidated dict files that dataset.py expects.

Input  (per acquisition, per layer):
    S2L2A/  YYYYMMDD_L{3,6,9,12}.pt          plain tensor [196, 768] fp16
    S1RTC/  YYYYMMDD_{ASC|DESC}_L{3,6,9,12}.pt

Output (per station, per layer, per orbit):
    S2L2A/  {station}_L{3,6,9,12}_{start}_{end}.pt
              dict { tokens: [N,196,768] fp16, dates: [N str], layer: str, geo: dict }
    S1RTC/  {station}_{ASC|DESC}_L{3,6,9,12}_{start}_{end}.pt

Resume-safe: skips groups where the consolidated file already exists.

Usage:
    python consolidate_tokens.py                    # dry run, all stations
    python consolidate_tokens.py --execute          # run all stations
    python consolidate_tokens.py --execute --workers 8
    python consolidate_tokens.py --execute --start-idx 0 --end-idx 50
"""

import argparse
import json
import re
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import torch

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
CSV_ACTIVE = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
LOG_FILE = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/logs.txt")
LAYERS = ("L3", "L6", "L9", "L12")


def folder_name(row) -> str:
    if row["source_network"] == "ISMN":
        return f"ISMN_{row['network']}_{row['station_id']}"
    return f"{row['source_network']}_{row['station_id']}"


def find_station_dir(station: str) -> Path | None:
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        p = DATA_ROOT / cat / station
        if p.exists():
            return p
    return None


def parse_s2_stem(stem: str) -> str | None:
    m = re.match(r'^(\d{8})_L\d+$', stem)
    return m.group(1) if m else None


def parse_s1_stem(stem: str) -> tuple[str, str] | None:
    m = re.match(r'^(\d{8})_(ASC|DESC)_L\d+$', stem)
    return (m.group(1), m.group(2)) if m else None


def load_geo(subdir: Path) -> dict | None:
    for f in subdir.glob("*_geo.json"):
        try:
            return json.loads(f.read_text())
        except Exception:
            continue
    return None


def consolidate_s2(subdir: Path, station: str, execute: bool) -> dict:
    counts = {"consolidated": 0, "skipped": 0, "no_files": 0}
    if not subdir.exists():
        return counts

    per_date: dict[str, list[Path]] = {}
    for pt in subdir.glob("*.pt"):
        date_str = parse_s2_stem(pt.stem)
        if date_str:
            per_date.setdefault(date_str, []).append(pt)

    if not per_date:
        counts["no_files"] = 1
        return counts

    dates_sorted = sorted(per_date.keys())
    start, end = dates_sorted[0], dates_sorted[-1]
    geo = load_geo(subdir)

    for layer in LAYERS:
        out_name = f"{station}_{layer}_{start}_{end}.pt"
        out_path = subdir / out_name

        if out_path.exists():
            counts["skipped"] += 1
            continue

        tensors, valid_dates = [], []
        for d in dates_sorted:
            pt = subdir / f"{d}_{layer}.pt"
            if not pt.exists():
                continue
            try:
                t = torch.load(pt, map_location="cpu", weights_only=True)
                tensors.append(t)
                valid_dates.append(d)
            except Exception as e:
                print(f"    WARN: failed to load {pt.name}: {e}", flush=True)

        if not tensors:
            counts["no_files"] += 1
            continue

        stacked = torch.stack(tensors)
        bundle = {"tokens": stacked, "dates": valid_dates, "layer": layer, "geo": geo}

        print(f"  {'SAVE' if execute else 'would save'} {subdir.parent.name}/S2L2A/{out_name}"
              f"  [{len(valid_dates)} dates, {stacked.shape}]", flush=True)

        if execute:
            tmp = out_path.with_suffix(".tmp")
            torch.save(bundle, tmp)
            tmp.rename(out_path)
        counts["consolidated"] += 1

    return counts


def consolidate_s1(subdir: Path, station: str, execute: bool) -> dict:
    counts = {"consolidated": 0, "skipped": 0, "no_files": 0}
    if not subdir.exists():
        return counts

    by_orbit: dict[str, dict[str, list]] = {}
    for pt in subdir.glob("*.pt"):
        parsed = parse_s1_stem(pt.stem)
        if parsed:
            date_str, orbit = parsed
            by_orbit.setdefault(orbit, {}).setdefault(date_str, []).append(pt)

    if not by_orbit:
        counts["no_files"] = 1
        return counts

    geo = load_geo(subdir)

    for orbit, date_map in by_orbit.items():
        dates_sorted = sorted(date_map.keys())
        start, end = dates_sorted[0], dates_sorted[-1]

        for layer in LAYERS:
            out_name = f"{station}_{orbit}_{layer}_{start}_{end}.pt"
            out_path = subdir / out_name

            if out_path.exists():
                counts["skipped"] += 1
                continue

            tensors, valid_dates = [], []
            for d in dates_sorted:
                pt = subdir / f"{d}_{orbit}_{layer}.pt"
                if not pt.exists():
                    continue
                try:
                    t = torch.load(pt, map_location="cpu", weights_only=True)
                    tensors.append(t)
                    valid_dates.append(d)
                except Exception as e:
                    print(f"    WARN: failed to load {pt.name}: {e}", flush=True)

            if not tensors:
                counts["no_files"] += 1
                continue

            stacked = torch.stack(tensors)
            bundle = {"tokens": stacked, "dates": valid_dates, "layer": layer, "geo": geo}

            print(f"  {'SAVE' if execute else 'would save'} {subdir.parent.name}/S1RTC/{out_name}"
                  f"  [{len(valid_dates)} dates, {stacked.shape}]", flush=True)

            if execute:
                tmp = out_path.with_suffix(".tmp")
                torch.save(bundle, tmp)
                tmp.rename(out_path)
            counts["consolidated"] += 1

    return counts


def _worker(args: tuple) -> tuple[str, dict, dict]:
    """Process one station; returns (station, s2_counts, s1_counts)."""
    station, execute = args
    sdir = find_station_dir(station)
    if sdir is None:
        return station, {"consolidated": 0, "skipped": 0, "no_files": 0}, {"consolidated": 0, "skipped": 0, "no_files": 0}

    s2_old = list((sdir / "S2L2A").glob("[0-9]*_L12.pt")) if (sdir / "S2L2A").exists() else []
    s1_old = list((sdir / "S1RTC").glob("[0-9]*_L12.pt")) if (sdir / "S1RTC").exists() else []

    if not s2_old and not s1_old:
        return station, {"consolidated": 0, "skipped": 0, "no_files": 0}, {"consolidated": 0, "skipped": 0, "no_files": 0}

    print(f"  {station}  (S2: {len(s2_old)} dates, S1: {len(s1_old)} dates)", flush=True)

    c_s2 = consolidate_s2(sdir / "S2L2A", station, execute)
    c_s1 = consolidate_s1(sdir / "S1RTC", station, execute)
    return station, c_s2, c_s1


def main(execute: bool, start_idx: int, end_idx: int | None, workers: int):
    import pandas as pd
    df = pd.read_csv(CSV_ACTIVE)
    stations = [folder_name(row) for _, row in df.iterrows()]
    stations = stations[start_idx:end_idx]

    total_consolidated = total_skipped = total_no_files = 0
    done = 0

    args_list = [(s, execute) for s in stations]

    print(f"Processing {len(stations)} stations with {workers} workers...", flush=True)

    with Pool(workers) as pool:
        for station, c_s2, c_s1 in pool.imap_unordered(_worker, args_list, chunksize=1):
            done += 1
            for c in (c_s2, c_s1):
                total_consolidated += c["consolidated"]
                total_skipped      += c["skipped"]
                total_no_files     += c["no_files"]
            if (done % 20) == 0 or done == len(stations):
                print(f"[{done}/{len(stations)}] consolidated={total_consolidated} skipped={total_skipped}", flush=True)

    print(f"\n{'Saved' if execute else 'Would save'}: {total_consolidated} bundle files")
    print(f"Already existed (skipped): {total_skipped}")
    print(f"No source files: {total_no_files}")

    if execute:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"\n=== {date.today()} ===\n")
            f.write(f"- consolidate_tokens.py --execute --workers {workers}: saved {total_consolidated} bundle files\n")
            f.write(f"  skipped (already existed): {total_skipped}\n")
        print(f"\nLogged to {LOG_FILE}")
    else:
        print("\nRun with --execute to apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute",    action="store_true")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx",   type=int, default=None)
    args = parser.parse_args()
    main(execute=args.execute, start_idx=args.start_idx, end_idx=args.end_idx, workers=args.workers)
