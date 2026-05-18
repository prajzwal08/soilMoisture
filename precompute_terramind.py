"""
precompute_terramind.py
=======================
One-time script: run TerraMind on every satellite .tif and save L3/L6/L9/L12
features (196×768, fp16) alongside the original file.

Also processes static modalities:
  DEM  → DEM/dem_L12.pt       (196×768 fp16)
  LULC → LULC/lulc_L12.pt    (196×768 fp16)
  NDVI → computed from S2L2A bands, saved per acquisition as {stem}_NDVI_L12.pt

Resume-safe: skips acquisitions where _L12.pt already exists.

Usage:
    python precompute_terramind.py
    python precompute_terramind.py --sat-dir /path/to/satellite --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

# ── shared imports from project ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import center_crop, S2_BAND_INDICES
from model import TerraMindEncoder

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SAT_DIR   = Path("/home/khanalp/data/satellite")
DEFAULT_BATCH     = 8


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_tif(path: Path, band_indices=None,
              do_crop: bool = False) -> torch.Tensor | None:
    """Load a .tif → (C, H, W) float32 tensor. Returns None on failure."""
    try:
        with rasterio.open(path) as src:
            arr    = src.read().astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = 0.0
        if do_crop:
            arr = center_crop(arr)
        if band_indices is not None:
            arr = arr[band_indices]
        return torch.from_numpy(arr)
    except Exception as exc:
        print(f"    [warn] cannot load {path.name}: {exc}")
        return None


def _run_batch(encoder: TerraMindEncoder,
               patches: list[torch.Tensor | None],
               paths:   list[Path],
               modality: str,
               device:   torch.device) -> int:
    """
    Forward a batch of patches through TerraMind, save L12 per acquisition.
    Returns the number of acquisitions successfully saved.
    """
    valid_i = [i for i, p in enumerate(patches) if p is not None]
    if not valid_i:
        return 0

    batch = torch.stack([patches[i] for i in valid_i]).float().to(device)

    with torch.no_grad():
        feats = encoder(batch, modality)          # hooks fill L3/L6/L9/L12

    for out_idx, src_idx in enumerate(valid_i):
        p = paths[src_idx]
        for layer in ("L12", "L9", "L6", "L3"):
            out_path = p.parent / f"{p.stem}_{layer}.pt"
            torch.save(feats[layer][out_idx].half().cpu(), out_path)

    return len(valid_i)


# ── per-modality processing ───────────────────────────────────────────────────

def process_modality(encoder:      TerraMindEncoder,
                     tif_dir:      Path,
                     modality:     str,
                     device:       torch.device,
                     batch_size:   int,
                     band_indices = None,
                     do_crop:      bool = False) -> int:
    """
    Iterate all .tif files in tif_dir, compute L12, save *_L12.pt.
    Skips files that already have a corresponding _L12.pt.

    Returns count of newly processed acquisitions.
    """
    if not tif_dir.exists():
        return 0

    pending = [
        f for f in sorted(tif_dir.glob("*.tif"))
        if not (tif_dir / f"{f.stem}_L12.pt").exists()
    ]
    if not pending:
        return 0

    processed = 0
    for start in range(0, len(pending), batch_size):
        batch_files = pending[start : start + batch_size]
        patches     = [_load_tif(f, band_indices, do_crop) for f in batch_files]
        processed  += _run_batch(encoder, patches, batch_files, modality, device)

    return processed


def process_dem(encoder:    TerraMindEncoder,
                dem_path:   Path,
                device:     torch.device) -> None:
    """
    Compute DEM pyramid tokens (4×768, fp32) via masked-mean pooling.
    Saved as  DEM/dem_pyramid.pt.  Skipped if already exists.
    """
    out_path = dem_path.parent / "dem_pyramid.pt"
    if out_path.exists():
        return

    patch = _load_tif(dem_path, do_crop=True)
    if patch is None:
        return

    with torch.no_grad():
        feats   = encoder(patch.float().unsqueeze(0).to(device), "DEM")
        l12     = feats["L12"]                                      # (1, 196, 768)
        pyramid = spatial_pyramid_pool(l12, token_valid=None, attn=None)  # (1, 4, 768)

    torch.save(pyramid[0].cpu(), out_path)   # (4, 768) fp32


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute TerraMind L12 features")
    parser.add_argument("--sat-dir",    type=Path, default=DEFAULT_SAT_DIR)
    parser.add_argument("--batch-size", type=int,  default=DEFAULT_BATCH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Sat dir    : {args.sat_dir}")
    print(f"Batch size : {args.batch_size}\n")

    encoder = TerraMindEncoder(frozen=True).to(device).eval()

    station_dirs = sorted(d for d in args.sat_dir.iterdir() if d.is_dir())
    print(f"Stations to process: {len(station_dirs)}\n")

    total_s2 = total_s1 = 0

    for idx, station_dir in enumerate(station_dirs, 1):
        s2_new = process_modality(
            encoder, station_dir / "S2L2A", "S2L2A", device, args.batch_size,
            band_indices=S2_BAND_INDICES, do_crop=True,
        )
        s1_new = process_modality(
            encoder, station_dir / "S1RTC", "S1RTC", device, args.batch_size,
            do_crop=False,
        )

        dem_path = station_dir / "DEM" / "dem.tif"
        if dem_path.exists():
            process_dem(encoder, dem_path, device)

        total_s2 += s2_new
        total_s1 += s1_new

        if s2_new or s1_new:
            print(f"[{idx:4d}/{len(station_dirs)}] {station_dir.name}"
                  f"  S2={s2_new}  S1={s1_new}")
        elif idx % 50 == 0:
            print(f"[{idx:4d}/{len(station_dirs)}] ... (up to date)")

    print(f"\nFinished.  New acquisitions → S2: {total_s2},  S1: {total_s1}")


if __name__ == "__main__":
    main()
