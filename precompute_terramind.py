"""
precompute_terramind.py
=======================
Run TerraMind on every satellite .tif in scratch and save features to
permanent project storage.

Input  (scratch – purged ~14-30 days):
    /gpfs/scratch1/shared/pkhanal/satellite/{station}/

Output (project – permanent):
    /gpfs/work3/0/prjs1968/data/{sm_only|sm_and_flux|flux_only}/{station}/

Saved per temporal acquisition (S2L2A, S1RTC):
    {stem}_L12.pt   (196, 768) fp16   semantic tokens → history bottleneck
    {stem}_L9.pt    (196, 768) fp16   U-Net skip connection
    {stem}_L6.pt    (196, 768) fp16   U-Net skip connection
    {stem}_L3.pt    (196, 768) fp16   U-Net skip connection
    {stem}_geo.json              CRS, affine transform, bounds

Saved per static modality (DEM, LULC) — L12 only, no skips needed:
    dem_L12.pt    (196, 768) fp16
    dem_geo.json
    lulc_L12.pt   (196, 768) fp16
    lulc_geo.json

geo.json schema:
    {
      "crs"      : "EPSG:32631",
      "transform": [a, b, c, d, e, f],   // GDAL-order affine (pixel → world)
      "bounds"   : [west, south, east, north]
    }
    Each of the 196 tokens covers a 16×16 pixel (160 m) block.
    Token (row i, col j) has world-space centre:
        Affine(*geo["transform"]) * (j*16 + 8, i*16 + 8)

Resume-safe: skips any acquisition whose _L12.pt already exists in the output dir.

Usage:
    python precompute_terramind.py                          # all stations
    python precompute_terramind.py --station ISMN_SCAN_Abo  # single station (test)
    python precompute_terramind.py --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from affine import Affine
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import center_crop, S2_BAND_INDICES
from model import TerraMindEncoder

SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_DIR    = Path("/gpfs/work3/0/prjs1968/data")
BATCH_SIZE  = 8

TEMPORAL_LAYERS = ("L12", "L9", "L6", "L3")
IMAGE_SIZE  = 224   # pixel dimensions of each input patch
TOKEN_SIZE  = 16    # each token covers TOKEN_SIZE × TOKEN_SIZE pixels → 14×14 = 196 tokens


# ── geo helpers ───────────────────────────────────────────────────────────────

def _read_geo(src: rasterio.DatasetReader,
              crop_top: int = 0, crop_left: int = 0) -> dict:
    """Build geo dict from an open rasterio dataset, adjusted for any crop."""
    T = src.transform * Affine.translation(crop_left, crop_top)
    bounds = rasterio.transform.array_bounds(IMAGE_SIZE, IMAGE_SIZE, T)
    return {
        "crs"      : src.crs.to_string(),
        "transform": [T.a, T.b, T.c, T.d, T.e, T.f],
        "bounds"   : list(bounds),      # [west, south, east, north]
    }


def _save_geo(geo: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(geo, indent=2))
    tmp.rename(path)


# ── tif loader ────────────────────────────────────────────────────────────────

def _load_tif(path: Path,
              band_indices: list[int] | None = None,
              do_crop: bool = False) -> tuple[torch.Tensor | None, dict | None]:
    """Load .tif → (C, H, W) float32 tensor + geo dict. Returns (None, None) on I/O failure."""
    try:
        with rasterio.open(path) as src:
            arr    = src.read().astype(np.float32)
            nodata = src.nodata
            h, w   = src.shape
            if do_crop and (h < IMAGE_SIZE or w < IMAGE_SIZE):
                print(f"    [warn] {path.name}: image {h}×{w} smaller than IMAGE_SIZE={IMAGE_SIZE}; skipping")
                return None, None
            crop_top  = (h - IMAGE_SIZE) // 2 if do_crop else 0
            crop_left = (w - IMAGE_SIZE) // 2 if do_crop else 0
            geo = _read_geo(src, crop_top, crop_left)
    except (OSError, rasterio.errors.RasterioIOError) as exc:
        print(f"    [warn] cannot load {path.name}: {exc}")
        return None, None

    if nodata is not None:
        mask = np.isclose(arr, nodata) if np.issubdtype(arr.dtype, np.floating) else (arr == nodata)
        arr[mask] = 0.0
    if do_crop:
        arr = center_crop(arr)
    if band_indices is not None:
        arr = arr[band_indices]

    return torch.from_numpy(arr), geo


# ── batch runner ──────────────────────────────────────────────────────────────

def _save_feats(feats: dict, out_idx: int, src_idx: int,
                src_paths: list[Path], geos: list[dict | None],
                out_dir: Path, layers: tuple[str, ...]) -> None:
    """Save one item's features to disk. Raises ValueError on non-finite tensors."""
    stem = src_paths[src_idx].stem
    for layer in layers:
        t = feats[layer][out_idx]
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {layer} for {stem}")
        tmp = out_dir / f"{stem}_{layer}.tmp"
        torch.save(t.half().cpu(), tmp)
        tmp.rename(out_dir / f"{stem}_{layer}.pt")
    if geos[src_idx] is not None:
        _save_geo(geos[src_idx], out_dir / f"{stem}_geo.json")


def _run_batch(encoder:   TerraMindEncoder,
               patches:   list[torch.Tensor | None],
               geos:      list[dict | None],
               src_paths: list[Path],
               out_dir:   Path,
               modality:  str,
               layers:    tuple[str, ...],
               device:    torch.device,
               failures:  list[str]) -> int:
    """Forward a batch through TerraMind, save .pt and _geo.json. Returns count saved."""
    valid_i = [i for i, p in enumerate(patches) if p is not None]
    if not valid_i:
        return 0

    batch = torch.stack([patches[i] for i in valid_i]).float().to(device)
    oom = False
    try:
        with torch.no_grad():
            feats = encoder(batch, modality)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
            raise
        oom = True

    if not oom:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for out_idx, src_idx in enumerate(valid_i):
            try:
                _save_feats(feats, out_idx, src_idx, src_paths, geos, out_dir, layers)
                saved += 1
            except ValueError as exc:
                print(f"    [warn] {exc}")
                failures.append(str(src_paths[src_idx]))
        return saved

    print(f"    [warn] GPU OOM on batch of {len(valid_i)}; retrying one-at-a-time")
    torch.cuda.empty_cache()
    del batch
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src_idx in valid_i:
        single = patches[src_idx].float().unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                sf = encoder(single, modality)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc2:
            print(f"    [warn] OOM on single {src_paths[src_idx].name}: {exc2}")
            failures.append(str(src_paths[src_idx]))
            continue
        try:
            _save_feats(sf, 0, src_idx, src_paths, geos, out_dir, layers)
            saved += 1
        except ValueError as exc2:
            print(f"    [warn] {exc2}")
            failures.append(str(src_paths[src_idx]))
    return saved


# ── per-modality processors ───────────────────────────────────────────────────

def process_temporal(encoder:     TerraMindEncoder,
                     src_dir:     Path,
                     out_dir:     Path,
                     modality:    str,
                     device:      torch.device,
                     batch_size:  int,
                     band_indices: list[int] | None = None,
                     do_crop:     bool = False,
                     failures:    list[str] | None = None) -> int:
    """
    Process all .tif files for S2L2A or S1RTC.
    Saves L3/L6/L9/L12 + geo.json per acquisition to out_dir.
    Returns count of newly processed acquisitions.
    """
    if failures is None:
        failures = []
    if not src_dir.exists():
        return 0

    pending = [
        f for f in sorted(src_dir.glob("*.tif"))
        if not (out_dir / f"{f.stem}_L12.pt").exists()
    ]
    if not pending:
        return 0

    processed = 0
    for start in range(0, len(pending), batch_size):
        batch_files = pending[start : start + batch_size]
        results     = [_load_tif(f, band_indices, do_crop) for f in batch_files]
        patches, geos = zip(*results)
        processed += _run_batch(
            encoder, list(patches), list(geos),
            batch_files, out_dir, modality, TEMPORAL_LAYERS, device, failures,
        )

    return processed


def process_static(encoder:  TerraMindEncoder,
                   src_path: Path,
                   out_dir:  Path,
                   modality: str,
                   out_stem: str,
                   device:   torch.device,
                   do_crop:  bool = False,
                   failures: list[str] | None = None) -> None:
    """
    Process a single static .tif (DEM or LULC). Saves L12 + geo.json only.
    Skip connections (L3/L6/L9) are not needed for static modalities.
    """
    if failures is None:
        failures = []
    out_pt = out_dir / f"{out_stem}_L12.pt"
    if out_pt.exists() or not src_path.exists():
        return

    patch, geo = _load_tif(src_path, do_crop=do_crop)
    if patch is None:
        return

    with torch.no_grad():
        feats = encoder(patch.float().unsqueeze(0).to(device), modality)

    t = feats["L12"][0]
    if not torch.isfinite(t).all():
        print(f"    [warn] non-finite values in L12 for {out_stem} ({src_path.name})")
        failures.append(str(src_path))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{out_stem}_L12.tmp"
    torch.save(t.half().cpu(), tmp)
    tmp.rename(out_pt)

    if geo is not None:
        _save_geo(geo, out_dir / f"{out_stem}_geo.json")


def _latest_lulc_tif(lulc_dir: Path) -> Path | None:
    """Return the most recent year-named .tif in a LULC directory (e.g. 2024.tif)."""
    tifs = sorted(lulc_dir.glob("[0-9][0-9][0-9][0-9].tif"))
    return tifs[-1] if tifs else None


def _station_data_dir(station_id: str, data_dir: Path) -> Path | None:
    """Find the permanent data directory for a station across sm_only/sm_and_flux/flux_only."""
    for subfolder in ("sm_only", "sm_and_flux", "flux_only"):
        d = data_dir / subfolder / station_id
        if d.exists():
            return d
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute TerraMind features")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR,
                        help="Root of raw satellite tifs (scratch)")
    parser.add_argument("--data-dir",    type=Path, default=DATA_DIR,
                        help="Root of per-station data folders (sm_only/sm_and_flux/flux_only)")
    parser.add_argument("--batch-size",  type=int,  default=BATCH_SIZE)
    parser.add_argument("--station",     type=str,  default=None,
                        help="Process a single station by name (for testing)")
    parser.add_argument("--start-idx",   type=int,  default=0,
                        help="First station index, inclusive (for SLURM array slicing)")
    parser.add_argument("--end-idx",     type=int,  default=None,
                        help="Last station index, exclusive (for SLURM array slicing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Scratch    : {args.scratch_dir}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Batch size : {args.batch_size}\n")

    encoder = TerraMindEncoder(frozen=True).to(device).eval()

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())
        station_dirs = station_dirs[args.start_idx : args.end_idx]

    n = len(station_dirs)
    print(f"Stations to process: {n}\n")

    total_s2 = total_s1 = 0
    skipped: list[str] = []
    failures: list[str] = []

    for idx, src_station in enumerate(station_dirs, 1):
        t0          = time.time()
        sid         = src_station.name
        out_station = _station_data_dir(sid, args.data_dir)
        if out_station is None:
            print(f"[{idx:4d}/{n}] {sid}  [skip — no data directory found]")
            skipped.append(sid)
            continue

        s2_new = process_temporal(
            encoder,
            src_dir     = src_station / "S2L2A",
            out_dir     = out_station / "S2L2A",
            modality    = "S2L2A",
            device      = device,
            batch_size  = args.batch_size,
            band_indices= S2_BAND_INDICES,
            do_crop     = True,   # S2 tiles are 256×256; crop centre to IMAGE_SIZE
            failures    = failures,
        )
        s1_new = process_temporal(
            encoder,
            src_dir    = src_station / "S1RTC",
            out_dir    = out_station / "S1RTC",
            modality   = "S1RTC",
            device     = device,
            batch_size = args.batch_size,
            do_crop    = False,   # S1 tiles are natively IMAGE_SIZE×IMAGE_SIZE
            failures   = failures,
        )
        process_static(
            encoder,
            src_path = src_station / "DEM" / "dem.tif",
            out_dir  = out_station / "DEM",
            modality = "DEM",
            out_stem = "dem",
            device   = device,
            do_crop  = True,    # DEM tiles are 256×256; crop centre to IMAGE_SIZE
            failures = failures,
        )
        lulc_tif = _latest_lulc_tif(src_station / "LULC")
        if lulc_tif:
            process_static(
                encoder,
                src_path = lulc_tif,
                out_dir  = out_station / "LULC",
                modality = "LULC",
                out_stem = "lulc",
                device   = device,
                do_crop  = False,   # LULC tiles are natively IMAGE_SIZE×IMAGE_SIZE
                failures = failures,
            )

        total_s2 += s2_new
        total_s1 += s1_new

        elapsed = time.time() - t0
        if s2_new or s1_new:
            print(f"[{idx:4d}/{n}] {sid}  S2={s2_new}  S1={s1_new}  ({elapsed:.1f}s)")
        elif idx % 50 == 0:
            print(f"[{idx:4d}/{n}] ... (up to date)")

    print(f"\nDone.  New acquisitions → S2: {total_s2},  S1: {total_s1}")
    if skipped:
        tail = f" … and {len(skipped) - 10} more" if len(skipped) > 10 else ""
        print(f"Skipped {len(skipped)} stations (no data dir): {', '.join(skipped[:10])}{tail}")
    if failures:
        manifest = args.data_dir / "failures.log"
        with manifest.open("a") as fh:
            for path in failures:
                fh.write(f"{path}\n")
        print(f"{len(failures)} file(s) failed (NaN/OOM/IO) — appended to {manifest}")


if __name__ == "__main__":
    main()
