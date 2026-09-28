"""
restage_store.py — copy a zarr store back onto scratch after a purge
=====================================================================

The token store training reads from (`ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr`) was
purged on 2026-09-20: all 993 station directories survive as empty skeletons, so neither a
structural check nor `du` (which reports 51 GB of allocated directory blocks) reveals it.
The live copy at `/projects/prjs1968/zarr_tokens` is 1.4 TB and is the ONLY copy.

THREE RULES, and they are the whole design:

  1. THE SOURCE IS NEVER WRITTEN. Every rsync is `src/ -> dst/`, never `--delete`, never
     `--remove-source-files`. The source is the only copy in existence; a mistake here is
     not recoverable from anywhere.
  2. MERGE, DO NOT WIPE. The skeletons stay; rsync fills them. Nothing on the destination is
     removed, so a partially-staged station resumes rather than restarting.
  3. VERIFY BY COUNTING CHUNK FILES, NEVER `.complete`. The sentinel is copied along with
     everything else, so a `.complete` on the destination proves only that a rename happened.
     36 stations on scratch carry `.complete` today and 0 of 993 hold an `era5/values/0.0`
     chunk. `dataset.py:190` returns None per station when a store is incomplete and nothing
     raises, so a run against that store builds 0 samples and proceeds.

Sharded for a SLURM array: shard i of N takes stations [i::N], so any shard can be re-run
alone and re-running a completed shard is a no-op (rsync skips same-size, same-mtime files).

Usage:
  python restage_store.py --store tokens --shard 0 --n-shards 8 --workers 16
  python restage_store.py --store tokens --dry-run --limit 4       # smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

STORES = {
    # name        source (the only copy)                   destination (scratch, purged)
    "tokens":   (Path("/projects/prjs1968/zarr_tokens"),
                 Path("/gpfs/scratch1/shared/pkhanal/zarr")),
    "imagery":  (Path("/projects/prjs1968/satellite_zarr"),
                 Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr")),
    # §46's aux supervision target: the Landsat st30 bundles + masks, 4.5 GB in 1,991
    # files. ONLY LANDSAT_ST -- the sibling subdirs under this root are all dead for
    # training: ECOSTRESS was closed by the G0 verdict (§38, thermal arm shut), ERA5Land
    # is the pre-§43.12 per-station NetCDF superseded by era5/values18 in the token zarr,
    # and MERIT/terrain failed §32.10's sufficiency gate. Copying them would be 3.5 GB of
    # dead weight and an invitation to read the stale ERA5.
    # The category allowlist is NOT optional here. This root has nine top-level dirs --
    # _ecostress_staging, excluded_stations, landsat_st, landsat_st_stations and logs sit
    # alongside the three real categories -- and treating every top-level dir as a category
    # copies all of them, `logs/` included.
    "lst":      (Path("/gpfs/work3/0/prjs1968/data"),
                 Path("/gpfs/scratch1/shared/pkhanal/data"),
                 "LANDSAT_ST",
                 ("sm_only", "sm_and_flux", "flux_only")),
}

REPO = Path(__file__).resolve().parent


def unpack(store: str):
    """(source, destination, subdir-or-None, categories-or-None).

    `subdir` narrows the copy to one modality directory per station, so a shared root does
    not drag its dead siblings along. `categories` restricts which top-level directories
    count as categories at all -- without it, anything sitting beside the real three gets
    walked as if it were one."""
    spec = STORES[store]
    return (spec[0], spec[1],
            spec[2] if len(spec) > 2 else None,
            spec[3] if len(spec) > 3 else None)


def station_units(src_root: Path, subdir: str | None = None,
                  categories: tuple | None = None) -> list[tuple[Path, Path]]:
    """(relative path, absolute source) for every unit that copies independently.

    tokens:  {cat}/{station}              — three category subdirectories
    imagery: {station}.zarr               — flat
    lst:     {cat}/{station}/LANDSAT_ST   — one modality dir per station
    """
    units = []
    cats = [d for d in sorted(src_root.iterdir()) if d.is_dir()]
    if categories is not None:
        cats = [d for d in cats if d.name in categories]
        missing = set(categories) - {d.name for d in cats}
        if missing:
            raise SystemExit(f"FATAL: categories not found under {src_root}: {sorted(missing)}")
    if all(d.name.endswith(".zarr") for d in cats):
        return [(Path(d.name), d) for d in cats]
    for cat in cats:
        for st in sorted(cat.iterdir()):
            if not st.is_dir():
                continue
            if subdir is None:
                units.append((Path(cat.name) / st.name, st))
            elif (st / subdir).is_dir():
                units.append((Path(cat.name) / st.name / subdir, st / subdir))
    return units


def measure(path: Path) -> tuple[int, int]:
    """(file count, total bytes) under a directory. Follows no symlinks, counts real files."""
    n = total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
                n += 1
            except OSError:
                pass
    return n, total


def copy_one(task):
    rel, src, dst_root, dry = task
    dst = dst_root / rel
    src_n, src_b = measure(src)
    if src_n == 0:
        return dict(unit=str(rel), status="SOURCE_EMPTY", src_files=0, src_bytes=0,
                    dst_files=0, dst_bytes=0, seconds=0.0)
    if dry:
        dst_n, dst_b = measure(dst) if dst.exists() else (0, 0)
        return dict(unit=str(rel), status="DRY_RUN", src_files=src_n, src_bytes=src_b,
                    dst_files=dst_n, dst_bytes=dst_b, seconds=0.0)

    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # -a preserves times so a re-run is a no-op; no --delete, ever. The trailing slash on
    # the source copies its CONTENTS into dst, which is what merging into a skeleton means.
    r = subprocess.run(
        ["rsync", "-a", "--no-compress", f"{src}/", f"{dst}/"],
        capture_output=True, text=True,
    )
    dt = time.time() - t0
    if r.returncode != 0:
        return dict(unit=str(rel), status=f"RSYNC_FAIL_{r.returncode}", src_files=src_n,
                    src_bytes=src_b, dst_files=0, dst_bytes=0, seconds=dt,
                    err=r.stderr.strip()[:200])
    dst_n, dst_b = measure(dst)
    ok = (dst_n >= src_n) and (dst_b >= src_b)
    return dict(unit=str(rel), status="OK" if ok else "SHORT", src_files=src_n,
                src_bytes=src_b, dst_files=dst_n, dst_bytes=dst_b, seconds=dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", choices=sorted(STORES), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N units only (smoke)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure both sides, copy nothing")
    args = ap.parse_args()

    src_root, dst_root, subdir, cats = unpack(args.store)
    if not src_root.is_dir():
        print(f"FATAL: source {src_root} does not exist", file=sys.stderr)
        return 2

    units = station_units(src_root, subdir, cats)
    if args.limit:
        units = units[: args.limit]
    mine = units[args.shard :: args.n_shards]
    print(f"store={args.store}  src={src_root}  dst={dst_root}")
    print(f"units total {len(units)}, shard {args.shard}/{args.n_shards} -> {len(mine)} units, "
          f"{args.workers} workers, dry_run={args.dry_run}\n", flush=True)

    tasks = [(rel, src, dst_root, args.dry_run) for rel, src in mine]
    t0 = time.time()
    rows, done_b = [], 0
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(copy_one, tasks, chunksize=1), 1):
            rows.append(res)
            done_b += res["src_bytes"]
            if res["status"] not in ("OK", "DRY_RUN"):
                print(f"  !! {res['status']:<16s} {res['unit']}  {res.get('err', '')}",
                      flush=True)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i:4d}/{len(tasks)}  {done_b / 1e9:8.1f} GB  "
                      f"{done_b / 1e9 / max(el, 1):5.2f} GB/s  {el / 60:6.1f} min",
                      flush=True)

    log = REPO / "csvs" / f"restage_{args.store}.s{args.shard:02d}.csv"
    import csv as _csv
    keys = ["unit", "status", "src_files", "src_bytes", "dst_files", "dst_bytes", "seconds"]
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    tally = Counter(r["status"] for r in rows)
    src_b = sum(r["src_bytes"] for r in rows)
    dst_b = sum(r["dst_bytes"] for r in rows)
    print(f"\nstatus: {dict(tally)}")
    print(f"source {src_b / 1e12:.3f} TB in {sum(r['src_files'] for r in rows):,d} files")
    print(f"dest   {dst_b / 1e12:.3f} TB in {sum(r['dst_files'] for r in rows):,d} files")
    print(f"wrote {log}")
    bad = [r for r in rows if r["status"] not in ("OK", "DRY_RUN")]
    if bad:
        print(f"\n{len(bad)} UNIT(S) NOT OK — shard is NOT complete")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
