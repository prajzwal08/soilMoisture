"""
cloud_masking_inference.py
==========================
Run SEnSeIv2 (sensor-independent SegFormerB2, alldata-ambiguous) on every
S2L2A tile in scratch and write per-pixel cloud masks.

Uses the same model as TerraMesh (arXiv 2504.11172) to keep inputs
in-distribution for TerraMind tokenization.

Model: SEnSeIv2-SegFormerB2-alldata-ambiguous
  - Sensor-independent: accepts any band subset via wavelength descriptors
  - Handles our 12-band S2L2A tiles (no B10) natively — no zero-padding needed
  - 7-class output collapsed to 4-class + nodata

Input  (scratch):
    /gpfs/scratch1/shared/pkhanal/satellite/{station}/S2L2A/YYYYMMDD.tif

Output (scratch, alongside input):
    /gpfs/scratch1/shared/pkhanal/satellite/{station}/CloudMask/YYYYMMDD.tif

Output class encoding:
    0 = clear (land + water + snow)
    1 = thick cloud
    2 = thin cloud
    3 = cloud shadow
  255 = nodata (all bands == 0, i.e. swath edge)

Resume-safe: skips tiles whose CloudMask already exists.

Usage:
    python cloud_masking_inference.py                          # all stations
    python cloud_masking_inference.py --station ISMN_SCAN_Abo  # single station
    python cloud_masking_inference.py --start-idx 0 --end-idx 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import torch

from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_DESCRIPTORS

# ── constants ─────────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")

# Our S2L2A tiles have 12 bands: B01–B09, B11, B12 (B10 absent in L2A).
# Indices into SENTINEL2_DESCRIPTORS (0-based) for those 12 bands:
#   B01=0 B02=1 B03=2 B04=3 B05=4 B06=5 B07=6 B08=7 B8A=8 B09=9 B11=11 B12=12
S2_L2A_DESCRIPTOR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
S2_L2A_DESCRIPTORS    = [SENTINEL2_DESCRIPTORS[i] for i in S2_L2A_DESCRIPTOR_IDX]


# ── model loader ──────────────────────────────────────────────────────────────

def load_sensei(device: str = "cpu") -> CloudMask:
    cfg_path, wts_path = get_model_files("SEnSeIv2-SegFormerB2-alldata-ambiguous")
    return CloudMask(
        cfg_path, wts_path,
        device=device,
        output_style="4-class",
        categorise=True,
        batch_size=1,
    )


# ── per-station inference ─────────────────────────────────────────────────────

def process_station(model: CloudMask, station_dir: Path) -> tuple[int, int]:
    """Run SEnSeIv2 on all pending S2L2A tiles. Returns (done, errors)."""
    s2_dir  = station_dir / "S2L2A"
    out_dir = station_dir / "CloudMask"

    if not s2_dir.exists():
        return 0, 0

    pending = [f for f in sorted(s2_dir.glob("*.tif"))
               if not (out_dir / f.name).exists()]
    if not pending:
        return 0, 0

    out_dir.mkdir(exist_ok=True)
    done = errors = 0

    for tif_path in pending:
        out_path = out_dir / tif_path.name
        try:
            with rasterio.open(tif_path) as src:
                arr     = src.read().astype(np.float32) / 10_000  # (12, H, W)
                profile = src.profile

            # Pixels where ALL bands are 0 are swath-edge fill values (no nodata tag)
            nodata_mask = (arr == 0).all(axis=0)

            # Run SEnSeIv2 — pass wavelength descriptors so it knows which
            # bands are present (handles missing B10 natively)
            mask = model(arr, descriptors=S2_L2A_DESCRIPTORS)  # (H, W) uint8

            # 4-class output: 0=clear 1=thick 2=thin 3=shadow
            # Override nodata pixels regardless of what the model predicted
            mask = mask.astype(np.uint8)
            mask[nodata_mask] = 255

            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
            tmp = out_path.with_suffix(".tmp")
            with rasterio.open(tmp, "w", **out_profile) as dst:
                dst.write(mask[np.newaxis])
            tmp.rename(out_path)

            done += 1

        except (OSError, rasterio.errors.RasterioIOError) as exc:
            print(f"    [error] {tif_path.name}: {exc}")
            errors += 1

    return done, errors


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEnSeIv2 cloud masking for S2L2A tiles")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--station",     type=str,  default=None,
                        help="Process a single station (smoke test)")
    parser.add_argument("--start-idx",   type=int,  default=0,
                        help="First station index, inclusive (SLURM array slicing)")
    parser.add_argument("--end-idx",     type=int,  default=None,
                        help="Last station index, exclusive (SLURM array slicing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Scratch    : {args.scratch_dir}")
    print(f"Model      : SEnSeIv2-SegFormerB2-alldata-ambiguous\n")

    print("Loading SEnSeIv2...")
    model = load_sensei(device)
    print("Model loaded.\n")

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())
        station_dirs = station_dirs[args.start_idx : args.end_idx]

    n = len(station_dirs)
    print(f"Stations to process: {n}\n")

    total_done = total_errors = 0
    for idx, station_dir in enumerate(station_dirs, 1):
        t0 = time.time()
        done, errors = process_station(model, station_dir)
        total_done   += done
        total_errors += errors
        elapsed = time.time() - t0
        if done > 0:
            print(f"[{idx:4d}/{n}] {station_dir.name}  done={done}  errors={errors}  ({elapsed:.1f}s)")
        elif idx % 50 == 0:
            print(f"[{idx:4d}/{n}] ... (up to date)")

    print(f"\nFinished.  Masks written: {total_done}   errors: {total_errors}")


if __name__ == "__main__":
    main()
