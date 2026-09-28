"""
verify_restage.py — did the re-stage actually land? Counted, not sentinelled.
=============================================================================

`.complete` is copied along with everything else, so its presence on the destination proves
a rename happened and nothing more. On 2026-09-20 scratch held 36 `.complete` markers and
**0 of 993** `era5/values/0.0` chunks, and `dataset.py:190` answers a missing store with
`None` per station rather than an exception — a training run against it builds 0 samples and
proceeds. So this script never looks at `.complete`.

What it checks, per station, both sides:

  1. file count and total bytes match (destination >= source; rsync adds, never removes)
  2. the REQUIRED KEYS exist as real chunk files on the destination, not empty directories:
     `era5/values18/0.0` (or `era5/values/0.0`), `labels/sm/0.0`, and at least one
     `s2_l*.npy` token bundle
  3. nothing on the destination is a zero-byte file

Exit code is non-zero if any station fails any of them.

Usage:  sbatch slurm/verify_restage.sh tokens
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from restage_store import STORES, station_units, measure, unpack

REPO = Path(__file__).resolve().parent

# Keys that must exist as real files. Alternatives within a tuple satisfy each other.
REQUIRED = {
    "tokens": [
        ("era5/values18/0.0", "era5/values/0.0"),
        ("labels/sm/0.0",),
    ],
    "imagery": [
        ("s2/data/0.0.0.0", "s2/data/0.0.0"),
    ],
    "lst": [],          # bundle filenames carry dates; the file-count match is the check
}


def check_one(task):
    rel, src, dst_root, store = task
    dst = dst_root / rel
    src_n, src_b = measure(src)
    if not dst.exists():
        return dict(unit=str(rel), ok=False, why="DST_MISSING",
                    src_files=src_n, dst_files=0, src_bytes=src_b, dst_bytes=0)
    dst_n, dst_b = measure(dst)

    problems = []
    if dst_n < src_n:
        problems.append(f"files {dst_n}<{src_n}")
    if dst_b < src_b:
        problems.append(f"bytes {dst_b}<{src_b}")

    for alts in REQUIRED.get(store, []):
        # only demand a key the SOURCE actually has -- stations legitimately differ
        if not any((src / a).is_file() for a in alts):
            continue
        if not any((dst / a).is_file() for a in alts):
            problems.append(f"missing {alts[0]}")

    if store == "tokens":
        has_bundle_src = any(p.name.startswith("s2_l") and p.suffix == ".npy"
                             for p in src.iterdir())
        if has_bundle_src and not any(p.name.startswith("s2_l") and p.suffix == ".npy"
                                      for p in dst.iterdir()):
            problems.append("no s2_l*.npy bundle")

    # A zero-byte file on the destination is only a fault if the SOURCE has content
    # there. Both stores legitimately carry empty `.complete` and `.tokens_complete`
    # sentinels, so a name-based exemption would have to be kept in step with them
    # forever; comparing against the source cannot go stale.
    truncated = []
    for root, _d, files in os.walk(dst):
        for f in files:
            p = os.path.join(root, f)
            try:
                if os.stat(p).st_size != 0:
                    continue
                s = src / os.path.relpath(p, dst)
                if s.is_file() and s.stat().st_size > 0:
                    truncated.append(os.path.relpath(p, dst))
            except OSError:
                pass
    if truncated:
        problems.append(f"{len(truncated)} truncated: {truncated[:3]}")

    return dict(unit=str(rel), ok=not problems, why=";".join(problems),
                src_files=src_n, dst_files=dst_n, src_bytes=src_b, dst_bytes=dst_b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", choices=sorted(STORES), required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    src_root, dst_root, subdir, cats = unpack(args.store)
    units = station_units(src_root, subdir, cats)
    print(f"verifying {args.store}: {len(units)} units, {src_root} -> {dst_root}\n", flush=True)

    tasks = [(rel, src, dst_root, args.store) for rel, src in units]
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check_one, tasks, chunksize=4), 1):
            rows.append(r)
            if not r["ok"]:
                print(f"  FAIL {r['unit']:<45s} {r['why']}", flush=True)
            if i % 200 == 0:
                print(f"  ...{i}/{len(tasks)}", flush=True)

    import csv as _csv
    log = REPO / "csvs" / f"verify_restage_{args.store}.csv"
    with open(log, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not r["ok"]]
    sb = sum(r["src_bytes"] for r in rows)
    db = sum(r["dst_bytes"] for r in rows)
    print(f"\nsource {sb / 1e12:.3f} TB in {sum(r['src_files'] for r in rows):,d} files")
    print(f"dest   {db / 1e12:.3f} TB in {sum(r['dst_files'] for r in rows):,d} files")
    print(f"wrote {log}")
    print("=" * 66)
    if bad:
        print(f"RESTAGE INCOMPLETE — {len(bad)}/{len(rows)} units failed")
        return 1
    print(f"RESTAGE VERIFIED — {len(rows)}/{len(rows)} units complete, counted not sentinelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
