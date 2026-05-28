"""
harmonize_s2_pre2022.py
=======================
Add +1000 DN offset to all pre-2022 S2L2A tiles to match the ESA
processing baseline 4.0 convention introduced on 2022-01-25.

After this script all tiles on scratch are in the same convention:
    0          = nodata (swath edge fill — unchanged)
    1000       = 0.0 reflectance
    11000      = 1.0 reflectance (max valid)

This means both downstream scripts can use a single normalization:
    cloud masking : (DN - 1000) / 10000  for ALL tiles
    TerraMind     : raw DNs as-is, z-score with pretraining stats

Run as a SLURM array job (one station per task):
    sbatch slurm/harmonize_s2.sh

Or single station for testing:
    python harmonize_s2_pre2022.py --station AmeriFlux_CA-Cbo
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRATCH_DIR   = Path("/gpfs/scratch1/shared/pkhanal/satellite")
CUTOFF_STEM   = "20220125"   # tiles with stem < this need the +1000 offset


def harmonize_station(station_dir: Path) -> tuple[int, int]:
    """
    Add +1000 to non-zero pixels of all pre-2022 S2L2A tiles for one station.
    Returns (n_rewritten, n_errors).
    """
    s2_dir = station_dir / "S2L2A"
    if not s2_dir.exists():
        return 0, 0

    pending = [f for f in sorted(s2_dir.glob("*.tif")) if f.stem < CUTOFF_STEM]
    if not pending:
        return 0, 0

    rewritten = skipped = errors = 0
    for path in pending:
        tmp = path.with_suffix(".tmp")
        try:
            with rasterio.open(path) as src:
                arr     = src.read()           # (C, H, W) int16
                profile = src.profile.copy()

            # ── idempotency check ─────────────────────────────────────────────
            # After harmonization every valid pixel has DN >= 1000 (the minimum
            # valid reflectance in baseline 4.0 convention). If the tile's
            # minimum non-zero DN is already >= 1000 it has been processed —
            # adding +1000 again would corrupt it. Safe to skip.
            arr32       = arr.astype(np.int32)
            valid       = arr32[arr32 != 0]
            if len(valid) > 0 and valid.min() >= 1000:
                skipped += 1
                continue

            # Add +1000 to valid pixels; leave nodata (DN=0) unchanged.
            arr32[arr32 != 0] += 1000

            # Write to .tmp first — if we crash mid-write the original is intact
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(arr32.astype(np.int16))

            tmp.replace(path)   # atomic rename
            rewritten += 1

        except Exception as exc:
            log.error("[error] %s: %s", path.name, exc)
            tmp.unlink(missing_ok=True)
            errors += 1

    return rewritten, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--station",     type=str,  default=None,
                        help="Process a single station (test mode)")
    parser.add_argument("--start-idx",   type=int,  default=0)
    parser.add_argument("--end-idx",     type=int,  default=None)
    args = parser.parse_args()

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())
        station_dirs = station_dirs[args.start_idx : args.end_idx]

    n = len(station_dirs)
    total_rewritten = total_skipped = total_errors = 0
    t0 = time.time()

    for idx, station_dir in enumerate(station_dirs, 1):
        rewritten, skipped, errors = harmonize_station(station_dir)
        total_rewritten += rewritten
        total_skipped   += skipped
        total_errors    += errors
        if rewritten or errors:
            log.info("[%4d/%d] %s  rewritten=%d  skipped=%d  errors=%d",
                     idx, n, station_dir.name, rewritten, skipped, errors)

    elapsed = time.time() - t0
    log.info("Done.  Rewritten: %d   Skipped (already done): %d   Errors: %d   Elapsed: %.1fs",
             total_rewritten, total_skipped, total_errors, elapsed)


if __name__ == "__main__":
    main()
