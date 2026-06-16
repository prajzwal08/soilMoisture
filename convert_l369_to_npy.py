#!/usr/bin/env python3
"""
Convert zarr L3/L6/L9 token arrays to uncompressed numpy memmap files.

Each zarr array (N, 196, 768) fp16 is written as a flat .npy file alongside
the zarr store. At training time, workers do:
    arr = np.memmap("s2_l3.npy", dtype="float16", mode="r", shape=(N,196,768))
    row = arr[best_idx]   # 0.3 MB direct read, zero decompression
instead of:
    row = zarr_group["s2/l3"][best_idx]  # 9.6 MB chunk read + zstd decompress

Usage:
    python convert_l369_to_npy.py [--execute] [--workers 64]

Without --execute: dry run, prints what would be written.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
CATEGORIES = ["sm_only", "sm_and_flux"]
ORBITS     = ["s2", "s1_asc", "s1_desc"]
LAYERS     = ["l3", "l6", "l9"]


def npy_path(station_dir: Path, orbit: str, layer: str) -> Path:
    return station_dir / f"{orbit}_{layer}.npy"


def meta_path(station_dir: Path, orbit: str, layer: str) -> Path:
    return station_dir / f"{orbit}_{layer}.json"


def convert_station(args) -> dict:
    station_dir, execute = args
    result = {"station": station_dir.name, "written": [], "skipped": [], "errors": []}
    try:
        zg = zarr.open(str(station_dir), "r")
    except Exception as e:
        result["errors"].append(f"open zarr: {e}")
        return result

    for orbit in ORBITS:
        for layer in LAYERS:
            key = f"{orbit}/{layer}"
            if key not in zg:
                continue

            np_path = npy_path(station_dir, orbit, layer)
            mt_path = meta_path(station_dir, orbit, layer)

            if np_path.exists() and mt_path.exists():
                result["skipped"].append(key)
                continue

            try:
                arr   = zg[key]                              # zarr array
                shape = arr.shape                            # (N, 196, 768)
                dtype = np.dtype("float16")

                if execute:
                    mm = np.memmap(str(np_path), dtype=dtype,
                                   mode="w+", shape=shape)
                    mm[:] = arr[:]                           # decompress once, write raw
                    mm.flush()
                    del mm
                    mt_path.write_text(json.dumps(
                        {"shape": list(shape), "dtype": "float16"}
                    ))

                result["written"].append(
                    f"{key} → {np_path.name}  shape={shape}"
                    f"  size={np.prod(shape)*2/1e6:.1f}MB"
                )
            except Exception as e:
                result["errors"].append(f"{key}: {e}")

    return result


def collect_stations() -> list[Path]:
    stations = []
    for cat in CATEGORIES:
        cat_path = ZARR_ROOT / cat
        if cat_path.exists():
            stations.extend(sorted(cat_path.iterdir()))
    return stations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually write files (default: dry run)")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    stations = collect_stations()
    print(f"Found {len(stations)} stations | execute={args.execute} | workers={args.workers}")

    # Estimate output size
    total_written = total_skipped = total_errors = 0
    est_gb = len(stations) * len(ORBITS) * len(LAYERS) * 65.6 / 1000  # rough 65 MB avg
    print(f"Estimated output size: ~{est_gb:.0f} GB (uncompressed fp16)")
    print()

    work = [(s, args.execute) for s in stations]
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(convert_station, work, chunksize=4):
            for w in r["written"]:
                print(f"  [WRITE]  {r['station'][:40]:40s}  {w}")
                total_written += 1
            for s in r["skipped"]:
                total_skipped += 1
            for e in r["errors"]:
                print(f"  [ERROR]  {r['station']}: {e}")
                total_errors += 1

    print(f"\nDone: {total_written} written, {total_skipped} skipped, {total_errors} errors")
    if not args.execute:
        print("\nDry run — re-run with --execute to write files.")


if __name__ == "__main__":
    main()
