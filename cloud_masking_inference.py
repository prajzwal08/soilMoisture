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
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, datetime, timezone
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

# ESA Sentinel-2 processing baseline 4.0 (introduced 2022-01-25) added a
# +1000 DN radio offset to all bands to allow negative reflectance storage.
# MPC delivers raw DNs (rescale=False), so we subtract this offset ourselves
# for post-2022 acquisitions — matching senseiv2/inference.py sentinel2_loader.
_S2_BASELINE4_DATE = _date(2022, 1, 25)


def _load_tile(
    path: Path,
) -> tuple[str, np.ndarray, np.ndarray, dict] | tuple[str, None, None, None]:
    """Read one S2L2A tile from disk. Designed to run in a worker thread."""
    try:
        with rasterio.open(path) as src:
            arr_int = src.read()
            profile = src.profile

        # Nodata check on raw integers (fill pixels are DN=0 in both baselines)
        nodata_mask = (arr_int == 0).all(axis=0)

        # Detect baseline >= 4 from filename date (YYYYMMDD.tif).
        # Fallback: any pixel > 10000 is physically impossible without the offset.
        stem = path.stem
        try:
            tile_date = _date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
            has_offset = tile_date >= _S2_BASELINE4_DATE
        except ValueError:
            has_offset = bool(arr_int.max() > 10_000)

        if has_offset:
            # Subtract offset; clip to 0 to avoid negative reflectance from dark pixels
            arr = np.clip(arr_int.astype(np.int32) - 1000, 0, None).astype(np.float32)
        else:
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

    Double-buffer pipeline: at any moment at most 2 * batch_size tiles occupy
    RAM — the batch the GPU is computing, and the next batch being prefetched
    by IO threads in parallel. Reads for batch N+1 are submitted immediately
    after batch N's futures are collected, so all io_workers threads stay busy
    during the GPU forward pass and deflate writes.
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

    chunks = [pending_names[i : i + batch_size]
              for i in range(0, len(pending_names), batch_size)]

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        # Pre-fetch the first batch before entering the loop
        next_futures = [pool.submit(_load_tile, s2_dir / n) for n in chunks[0]]

        for idx in tqdm(range(len(chunks)), desc=station_dir.name,
                        unit="batch", leave=False):
            # ── collect the batch that was prefetched last iteration ──────────
            names, arrays, nodata_masks, profiles = [], [], [], []
            for fut in next_futures:
                name, arr, nodata_mask, profile = fut.result()
                if arr is None:
                    errors += 1
                    continue
                names.append(name)
                arrays.append(arr)
                nodata_masks.append(nodata_mask)
                profiles.append(profile)

            # ── kick off next batch reads immediately ─────────────────────────
            # Threads now read batch idx+1 from disk while the GPU and write
            # loop below consume batch idx — true overlap, bounded to 2 batches
            if idx + 1 < len(chunks):
                next_futures = [pool.submit(_load_tile, s2_dir / n)
                                for n in chunks[idx + 1]]

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
                    mask = model.postprocess(pred).astype(np.uint8)
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
    job_start     = time.time()
    job_start_utc = datetime.now(timezone.utc).isoformat()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
            elapsed       = time.time() - t0
            tiles_per_sec = done / elapsed if elapsed > 0 and done > 0 else 0.0
            if done > 0:
                log.info("[%4d/%d] %s  done=%d  errors=%d  (%.1fs  %.1f tiles/s)",
                         idx, n, station_dir.name, done, errors, elapsed, tiles_per_sec)
            elif idx % 50 == 0:
                log.info("[%4d/%d] ... (up to date)", idx, n)

    job_elapsed  = time.time() - job_start
    overall_tps  = total_done / job_elapsed if job_elapsed > 0 else 0.0
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Masks written : %d", total_done)
    log.info("  Errors        : %d", total_errors)
    log.info("  Skipped       : %d", skipped)
    log.info("  Elapsed       : %.1f s  (%.1f min)", job_elapsed, job_elapsed / 60)
    log.info("  Throughput    : %.1f tiles/sec (end-to-end)", overall_tps)
    log.info("  Peak VRAM     : %.2f GB", peak_vram_gb)
    log.info("=" * 60)

    # ── write JSON summary (one file per array task) ──────────────────────────
    # After the full run, aggregate with:
    #   python3 -c "import json,pathlib; fs=list(pathlib.Path('data/logs').glob('cloud_mask_stats_*.json')); \
    #     print(sum(json.loads(f.read_text())['masks_written'] for f in fs), 'masks across', len(fs), 'tasks')"
    # SLURM_ARRAY_JOB_ID is the same for all tasks in the array (e.g. 23182278).
    # SLURM_JOB_ID is unique per task — using it would give every JSON a different
    # prefix, making glob-based aggregation impossible.
    job_id  = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    summary = {
        "job_id":        job_id,
        "task_id":       task_id,
        "node":          os.environ.get("SLURMD_NODENAME", "local"),
        "start_utc":     job_start_utc,
        "end_utc":       datetime.now(timezone.utc).isoformat(),
        "elapsed_s":     round(job_elapsed, 2),
        "batch_size":    args.batch_size,
        "io_workers":    args.io_workers,
        "masks_written": total_done,
        "errors":        total_errors,
        "skipped":       skipped,
        "tiles_per_sec": round(overall_tps, 2),
        "peak_vram_gb":  round(peak_vram_gb, 3),
    }
    log_dir  = args.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_json = log_dir / f"cloud_mask_stats_{job_id}_{task_id}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Stats saved → %s", out_json)


if __name__ == "__main__":
    main()
