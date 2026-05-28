"""
cloud_masking_inference.py
==========================
Run SEnSeIv2 (sensor-independent SegFormerB2, alldata-ambiguous) on every
S2L2A tile in scratch and write per-pixel cloud masks to permanent storage.

Uses the same model as TerraMesh (arXiv 2504.11172) to keep inputs
in-distribution for TerraMind tokenization.

Model: SEnSeIv2-SegFormerB2-alldata-ambiguous
  - Sensor-independent: accepts any band subset via wavelength descriptors
  - Handles our 12-band S2L2A tiles (no B10) natively — no zero-padding needed
  - 7-class output collapsed to 4-class + nodata

Input  (scratch):
    /gpfs/scratch1/shared/pkhanal/satellite/{station}/S2L2A/YYYYMMDD.tif

Output (permanent project storage):
    /gpfs/work3/0/prjs1968/data/{sm_only|sm_and_flux|flux_only}/{station}/CloudMask/YYYYMMDD.tif

Output class encoding (full 7-class, matching TerraMesh):
    0 = land
    1 = water
    2 = snow / ice
    3 = thin cloud
    4 = thick cloud
    5 = cloud shadow
  255 = nodata (all bands == 0, i.e. swath edge)

At training time, dataset.py reads these masks and uses them to build a
per-token validity mask for the temporal transformer attention.

Resume-safe: skips tiles whose CloudMask already exists in permanent storage.

Usage:
    python cloud_masking_inference.py                          # all stations
    python cloud_masking_inference.py --station ISMN_SCAN_Abo  # single station
    python cloud_masking_inference.py --start-idx 0 --end-idx 100
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import torch
from tqdm import tqdm

from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_DESCRIPTORS

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── GPFS / GDAL config (applied via rasterio.Env in main) ────────────────────
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
    "GDAL_SWATH_SIZE":               "200000000",
    "VSI_CACHE":                     "TRUE",
}

# ── constants ─────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_DIR    = Path("/gpfs/work3/0/prjs1968/data")

# Our S2L2A tiles have 12 bands: B01–B09, B11, B12 (B10 absent in L2A).
# Indices into SENTINEL2_DESCRIPTORS (0-based) for those 12 bands:
#   B01=0 B02=1 B03=2 B04=3 B05=4 B06=5 B07=6 B08=7 B8A=8 B09=9 B11=11 B12=12
S2_L2A_DESCRIPTOR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
S2_L2A_DESCRIPTORS    = [SENTINEL2_DESCRIPTORS[i] for i in S2_L2A_DESCRIPTOR_IDX]


def _station_data_dir(station_id: str, data_dir: Path) -> Path | None:
    """Find the permanent data directory for a station across sm_only/sm_and_flux/flux_only."""
    for subfolder in ("sm_only", "sm_and_flux", "flux_only"):
        d = data_dir / subfolder / station_id
        if d.exists():
            return d
    return None


# ── model loader ──────────────────────────────────────────────────────────────

def load_sensei(device: str = "cpu") -> CloudMask:
    cfg_path, wts_path = get_model_files("SEnSeIv2-SegFormerB2-alldata-ambiguous")
    return CloudMask(
        cfg_path, wts_path,
        device=device,
        output_style=None,   # preserve all 7 classes (matches TerraMesh storage)
        categorise=True,
    )


# ── tile loader (runs in thread pool) ────────────────────────────────────────

def _load_tile(
    path: Path,
) -> tuple[str, np.ndarray, np.ndarray, dict] | tuple[str, None, None, None]:
    """Read one S2L2A tile from disk. Designed to run in a worker thread."""
    try:
        with rasterio.open(path) as src:
            arr_int = src.read()
            profile = src.profile
        nodata_mask = (arr_int == 0).all(axis=0)
        arr = arr_int.astype(np.float32)
        del arr_int
        arr /= 10_000.0
        return path.name, arr, nodata_mask, profile
    except (OSError, rasterio.errors.RasterioIOError) as exc:
        log.error("[io error] %s: %s", path.name, exc)
        return path.name, None, None, None


# ── per-station inference ─────────────────────────────────────────────────────

def process_station(model: CloudMask, station_dir: Path,
                    out_dir: Path, batch_size: int,
                    io_workers: int = 3) -> tuple[int, int]:
    """Run SEnSeIv2 on all pending S2L2A tiles. Returns (done, errors).

    I/O and GPU are pipelined: all tile reads are submitted to a thread pool
    upfront so worker threads pre-fetch disk → RAM while the GPU computes the
    previous batch. On a GPFS filesystem where I/O (≈18 tiles/sec) is the
    bottleneck rather than the GPU (≈80 tiles/sec), this overlap is the main
    source of throughput gain.
    """
    s2_dir = station_dir / "S2L2A"
    if not s2_dir.exists():
        return 0, 0

    # Set-based resume check: two readdir ops instead of N individual stat calls
    all_inputs    = {f.name for f in s2_dir.glob("*.tif")}
    existing      = {f.name for f in out_dir.glob("*.tif")} if out_dir.exists() else set()
    pending_names = sorted(all_inputs - existing)
    if not pending_names:
        return 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.model.parameters()).device
    done = errors = 0

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        # Submit ALL reads immediately — threads start filling RAM while the
        # GPU works through earlier batches
        futures = [pool.submit(_load_tile, s2_dir / name) for name in pending_names]

        for i in tqdm(range(0, len(futures), batch_size),
                      desc=station_dir.name, unit="batch", leave=False):
            chunk_futures = futures[i : i + batch_size]

            # ── collect prefetched tiles ──────────────────────────────────────
            names, arrays, nodata_masks, profiles = [], [], [], []
            for fut in chunk_futures:
                name, arr, nodata_mask, profile = fut.result()
                if arr is None:
                    errors += 1
                    continue
                names.append(name)
                arrays.append(arr)
                nodata_masks.append(nodata_mask)
                profiles.append(profile)

            if not arrays:
                continue

            # ── batched GPU forward pass ──────────────────────────────────────
            try:
                batch = torch.from_numpy(np.stack(arrays)).to(device)  # (B, C, H, W)
                desc_batch = [S2_L2A_DESCRIPTORS for _ in range(len(arrays))]
                with torch.no_grad():
                    preds = model.model(batch, desc_batch)   # (B, num_classes, H, W)
                preds_np = preds.cpu().numpy()
                del batch, preds
            except Exception as exc:
                log.error("[model error] batch starting %s: %s", names[0], exc)
                errors += len(arrays)
                continue

            # ── write results ─────────────────────────────────────────────────
            for name, pred, nodata_mask, profile in zip(names, preds_np, nodata_masks, profiles):
                try:
                    mask = model.postprocess(pred)   # (H, W) argmax uint8
                    mask = mask.astype(np.uint8)
                    mask[nodata_mask] = 255

                    out_path = out_dir / name
                    out_profile = profile.copy()
                    out_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
                    tmp = out_path.with_suffix(".tmp")
                    with rasterio.open(tmp, "w", **out_profile) as dst:
                        dst.write(mask[np.newaxis])
                    tmp.rename(out_path)
                    done += 1
                except Exception as exc:
                    log.error("[write error] %s: %s", name, exc)
                    errors += 1

    return done, errors


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEnSeIv2 cloud masking for S2L2A tiles")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--data-dir",    type=Path, default=DATA_DIR,
                        help="Root of permanent station directories")
    parser.add_argument("--station",     type=str,  default=None,
                        help="Process a single station (smoke test)")
    parser.add_argument("--start-idx",   type=int,  default=0,
                        help="First station index, inclusive (SLURM array slicing)")
    parser.add_argument("--end-idx",     type=int,  default=None,
                        help="Last station index, exclusive (SLURM array slicing)")
    parser.add_argument("--batch-size",  type=int,  default=16,
                        help="GPU inference batch size (default 16)")
    parser.add_argument("--io-workers", type=int,  default=3,
                        help="Threads for async tile prefetch (default 3; "
                             "set to cpus-per-task minus 1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device     : %s", device)
    log.info("Scratch    : %s", args.scratch_dir)
    log.info("Data dir   : %s", args.data_dir)
    log.info("Batch size : %d  IO workers: %d", args.batch_size, args.io_workers)
    log.info("Model      : SEnSeIv2-SegFormerB2-alldata-ambiguous")

    log.info("Loading SEnSeIv2...")
    model = load_sensei(device)
    log.info("Model loaded.")

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())
        station_dirs = station_dirs[args.start_idx : args.end_idx]

    n = len(station_dirs)
    log.info("Stations to process: %d", n)

    total_done = total_errors = skipped = 0

    with rasterio.Env(**_GDAL_ENV):
        for idx, station_dir in enumerate(station_dirs, 1):
            perm_dir = _station_data_dir(station_dir.name, args.data_dir)
            if perm_dir is None:
                log.warning("[skip] %s — no permanent directory in %s (setup failure?)",
                            station_dir.name, args.data_dir)
                skipped += 1
                continue
            out_dir = perm_dir / "CloudMask"
            t0 = time.time()
            done, errors = process_station(model, station_dir, out_dir,
                                           args.batch_size, args.io_workers)
            total_done   += done
            total_errors += errors
            elapsed = time.time() - t0
            if done > 0:
                log.info("[%4d/%d] %s  done=%d  errors=%d  (%.1fs)",
                         idx, n, station_dir.name, done, errors, elapsed)
            elif idx % 50 == 0:
                log.info("[%4d/%d] ... (up to date)", idx, n)

    log.info("Finished.  Masks written: %d   errors: %d   skipped: %d",
             total_done, total_errors, skipped)


if __name__ == "__main__":
    main()
