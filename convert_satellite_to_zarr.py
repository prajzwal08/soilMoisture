"""
convert_satellite_to_zarr.py
=============================
Migrate raw satellite TIF imagery from scratch to a compressed Zarr v2 archive
at /gpfs/work3/0/prjs1968/satellite_zarr/.

Per-station structure:
  {station}.zarr/
    .zattrs         ← station metadata (lat, lon, epsg, bounds, etc.)
    s2/data         ← (N, 12, 224, 224)  int16    chunks=(1,12,224,224)
    s2/dates        ← (N,)               |S8
    s1_asc/data     ← (N, 2, 224, 224)   float16  chunks=(1,2,224,224)
    s1_asc/dates    ← (N,)               |S8
    s1_desc/...     ← same shape (only if DESC TIFs exist)
    dem/data        ← (1, 224, 224)       float32
    lulc/data       ← (N_years, 224, 224) uint8    chunks=(1,224,224)
    lulc/years      ← (N_years,)          int16
    .complete       ← sentinel; written only after all groups succeed

Resume-safe: stations with .complete sentinel are skipped.
Errors per station are caught and logged; job continues.

Usage:
    python convert_satellite_to_zarr.py --start 0 --end 10          # dry-run style: 10 stations
    python convert_satellite_to_zarr.py --start 0 --end 10 --execute
    python convert_satellite_to_zarr.py --start 0 --end 10 --execute --s1-dtype float32
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import zarr
import numcodecs

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_ROOT = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DST_ROOT = Path("/gpfs/work3/0/prjs1968/satellite_zarr")
LOG_FILE = Path("/gpfs/work3/0/prjs1968/data/logs/zarr_failures.log")

# ── constants ─────────────────────────────────────────────────────────────────
S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)


# ── per-station conversion ────────────────────────────────────────────────────

def convert_station(src_dir: Path, dst_root: Path, s1_dtype: str, execute: bool) -> str:
    """Convert one station directory to a Zarr store. Returns status string."""
    out_path = dst_root / f"{src_dir.name}.zarr"
    sentinel = out_path / ".complete"

    if sentinel.exists():
        return "skipped"

    if not execute:
        # dry-run: just count files and report
        n_s2  = len(list((src_dir / "S2L2A").glob("*.tif"))) if (src_dir / "S2L2A").exists() else 0
        n_s1  = len(list((src_dir / "S1RTC").glob("*.tif"))) if (src_dir / "S1RTC").exists() else 0
        n_lulc = len(list((src_dir / "LULC").glob("*.tif"))) if (src_dir / "LULC").exists() else 0
        return f"dry-run  S2={n_s2}  S1={n_s1}  LULC={n_lulc}"

    store = zarr.DirectoryStore(str(out_path))
    root  = zarr.open_group(store=store, mode="w")

    # ── station metadata ──────────────────────────────────────────────────────
    meta_path = src_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
        root.attrs.update(meta)
    root.attrs["source"]            = str(src_dir)
    root.attrs["zarr_creation_date"] = datetime.now(timezone.utc).date().isoformat()
    root.attrs["s1_dtype"]           = s1_dtype

    # ── S2L2A ─────────────────────────────────────────────────────────────────
    s2_dir = src_dir / "S2L2A"
    if s2_dir.exists():
        s2_files = sorted(s2_dir.glob("*.tif"))
        if s2_files:
            N = len(s2_files)
            dates = np.array([f.stem.encode() for f in s2_files], dtype="|S8")
            grp = root.require_group("s2")
            grp.attrs.update({"bands": S2_BANDS, "nodata": 0, "units": "DN [0,10000]"})
            arr = grp.zeros("data", shape=(N, 12, 224, 224), dtype="int16",
                            chunks=(1, 12, 224, 224), compressor=COMPRESSOR, overwrite=True)
            grp.array("dates", dates, dtype="|S8", overwrite=True)
            for i, f in enumerate(s2_files):
                with rasterio.open(f) as ds:
                    arr[i] = ds.read()

    # ── S1RTC ─────────────────────────────────────────────────────────────────
    s1_dir = src_dir / "S1RTC"
    np_dtype = np.float16 if s1_dtype == "float16" else np.float32
    z_dtype  = s1_dtype

    if s1_dir.exists():
        for orbit in ("ASC", "DESC"):
            orb_files = sorted(f for f in s1_dir.glob("*.tif")
                               if re.search(rf"_{orbit}\.tif$", f.name))
            if not orb_files:
                continue
            N = len(orb_files)
            dates = np.array([f.stem.split("_")[0].encode() for f in orb_files], dtype="|S8")
            grp = root.require_group(f"s1_{orbit.lower()}")
            grp.attrs.update({"bands": ["VV", "VH"], "orbit": orbit, "units": "dB"})
            arr = grp.zeros("data", shape=(N, 2, 224, 224), dtype=z_dtype,
                            chunks=(1, 2, 224, 224), compressor=COMPRESSOR, overwrite=True)
            grp.array("dates", dates, dtype="|S8", overwrite=True)
            for i, f in enumerate(orb_files):
                with rasterio.open(f) as ds:
                    arr[i] = ds.read().astype(np_dtype)

    # ── DEM ───────────────────────────────────────────────────────────────────
    dem_file = next((src_dir / "DEM").glob("*.tif"), None) if (src_dir / "DEM").exists() else None
    if dem_file:
        grp = root.require_group("dem")
        grp.attrs.update({"units": "m"})
        with rasterio.open(dem_file) as ds:
            data = ds.read().astype("float32")   # (1, 224, 224)
        grp.array("data", data, dtype="float32",
                  chunks=(1, 224, 224), compressor=COMPRESSOR, overwrite=True)

    # ── LULC ─────────────────────────────────────────────────────────────────
    lulc_dir = src_dir / "LULC"
    if lulc_dir.exists():
        lulc_files = sorted(
            (f for f in lulc_dir.glob("*.tif") if f.stem.isdigit()),
            key=lambda f: int(f.stem)
        )
        if lulc_files:
            N = len(lulc_files)
            years = np.array([int(f.stem) for f in lulc_files], dtype="int16")
            grp = root.require_group("lulc")
            grp.attrs.update({"source": "io-lulc-annual-v02"})
            arr = grp.zeros("data", shape=(N, 224, 224), dtype="uint8",
                            chunks=(1, 224, 224), compressor=COMPRESSOR, overwrite=True)
            grp.array("years", years, dtype="int16", overwrite=True)
            for i, f in enumerate(lulc_files):
                with rasterio.open(f) as ds:
                    arr[i] = ds.read(1)

    # ── completion sentinel ───────────────────────────────────────────────────
    sentinel.touch()
    return "done"


# ── main ──────────────────────────────────────────────────────────────────────

# ── worker wrapper (needed for Pool pickling) ─────────────────────────────────

def _worker(args_tuple: tuple) -> tuple[str, str]:
    """Return (station_name, status_or_error)."""
    station_dir, dst_root, s1_dtype, execute = args_tuple
    try:
        status = convert_station(station_dir, dst_root, s1_dtype, execute)
        return station_dir.name, status
    except Exception as exc:
        return station_dir.name, f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Convert satellite TIFs to Zarr archive")
    parser.add_argument("--src",      type=Path, default=SRC_ROOT)
    parser.add_argument("--dst",      type=Path, default=DST_ROOT)
    parser.add_argument("--start",    type=int,  default=0)
    parser.add_argument("--end",      type=int,  default=None)
    parser.add_argument("--s1-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--workers",  type=int,  default=4,
                        help="Parallel worker processes (default 4; match cpus-per-task - 1)")
    parser.add_argument("--execute",  action="store_true",
                        help="Write Zarr files. Without this flag runs in dry-run mode.")
    args = parser.parse_args()

    if args.execute:
        args.dst.mkdir(parents=True, exist_ok=True)

    all_stations = sorted(d for d in args.src.iterdir() if d.is_dir())
    subset = all_stations[args.start : args.end]
    n = len(subset)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode      : {mode}")
    print(f"S1 dtype  : {args.s1_dtype}")
    print(f"Workers   : {args.workers}")
    print(f"Src       : {args.src}")
    print(f"Dst       : {args.dst}")
    print(f"Stations  : {n}  (indices {args.start}:{args.end})")
    print()

    tasks = [(d, args.dst, args.s1_dtype, args.execute) for d in subset]
    done = skipped = errors = 0
    failures: list[str] = []

    with Pool(args.workers) as pool:
        for idx, (name, status) in enumerate(
            pool.imap_unordered(_worker, tasks, chunksize=1), 1
        ):
            if status == "skipped":
                skipped += 1
            elif status.startswith("ERROR"):
                errors += 1
                failures.append(f"{name}: {status}")
                print(f"  [{idx:4d}/{n}] {name}  {status}", file=sys.stderr)
            else:
                done += 1
                print(f"  [{idx:4d}/{n}] {name}  ✓  {status}", flush=True)

    print(f"\nDone: {done}  Skipped: {skipped}  Errors: {errors}")

    if failures and args.execute:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as fh:
            for msg in failures:
                fh.write(f"{ts}  {msg}\n")
        print(f"Failures logged to {LOG_FILE}")


if __name__ == "__main__":
    main()
