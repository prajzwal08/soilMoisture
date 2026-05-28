"""
analyze_cloud_masks.py
======================
Analyse cloud mask outputs to understand token-level validity.

For each CloudMask tile (224×224, 16×16 patches → 196 tokens):
  - A patch token is VALID if >= 50% of its valid (non-nodata) pixels are clear
    (class 0=land, 1=water, 2=snow/ice).
  - A patch token is MASKED if > 50% of valid pixels are cloud
    (class 3=thin cloud, 4=thick cloud, 5=cloud shadow)
    OR if all pixels are nodata (255).

Reports:
  - Overall: total tokens, % masked
  - Per-tile: distribution of masked-token fractions
  - Per-station: mean masked-token fraction
  - How many tiles have ALL tokens masked (effectively unusable)

Usage:
    python analyze_cloud_masks.py
    python analyze_cloud_masks.py --sample 5000   # random sample of tiles
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

DATA_DIR   = Path("/gpfs/work3/0/prjs1968/data")
PATCH_SIZE = 16      # pixels per patch side
TILE_SIZE  = 224     # pixels per tile side
N_PATCHES  = (TILE_SIZE // PATCH_SIZE) ** 2   # 196

NODATA_VAL        = 255
KEEP_THRESHOLD    = 0.5     # reject tile if > 50% of patches are invalid
NODATA_FRAC_THRESH = 0.01   # ≥1% nodata pixels per patch → patch invalid


def patch_validity(mask_arr: np.ndarray) -> np.ndarray:
    """
    Given a (1, 224, 224) uint8 cloud mask array, return a (196,) bool array
    where True = patch is valid.

    A patch is INVALID if:
      - any pixel is class 3 (thin cloud), 4 (thick cloud), or 5 (cloud shadow), OR
      - ≥1% of its pixels are nodata (class 255).
    Patches with <1% nodata and no cloud are valid (nodata pixels are NN-filled during encoding).
    """
    arr = mask_arr[0]  # (224, 224)
    n_side = TILE_SIZE // PATCH_SIZE   # 14
    valid_tokens = np.zeros(n_side * n_side, dtype=bool)
    n_pixels = PATCH_SIZE * PATCH_SIZE   # 256

    for i in range(n_side):
        for j in range(n_side):
            patch = arr[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            has_cloud   = np.any((patch == 3) | (patch == 4) | (patch == 5))
            nodata_frac = np.sum(patch == NODATA_VAL) / n_pixels
            valid_tokens[i * n_side + j] = (not has_cloud) and (nodata_frac < NODATA_FRAC_THRESH)

    return valid_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N tiles instead of processing all")
    args = parser.parse_args()

    # Collect all CloudMask tifs
    all_tifs = []
    for subfolder in ("sm_only", "sm_and_flux", "flux_only"):
        subdir = DATA_DIR / subfolder
        if not subdir.exists():
            continue
        for station_dir in subdir.iterdir():
            cm_dir = station_dir / "CloudMask"
            if cm_dir.exists():
                all_tifs.extend(cm_dir.glob("*.tif"))

    print(f"Total CloudMask tiles found: {len(all_tifs):,}")

    if args.sample and args.sample < len(all_tifs):
        all_tifs = random.sample(all_tifs, args.sample)
        print(f"Randomly sampled: {len(all_tifs):,} tiles")

    # Per-tile analysis
    tile_masked_fracs = []          # fraction of tokens masked per tile
    station_stats: dict[str, list[float]] = {}
    errors = 0

    for tif_path in tqdm(all_tifs, desc="Analysing", unit="tile"):
        station = tif_path.parents[1].name
        try:
            with rasterio.open(tif_path) as src:
                arr = src.read()
        except Exception:
            errors += 1
            continue

        valid_tokens = patch_validity(arr)
        masked_frac = 1.0 - valid_tokens.mean()
        tile_masked_fracs.append(masked_frac)

        if station not in station_stats:
            station_stats[station] = []
        station_stats[station].append(masked_frac)

    tile_masked_fracs = np.array(tile_masked_fracs)
    n_tiles = len(tile_masked_fracs)

    print(f"\n{'='*60}")
    print(f"TOKEN-LEVEL CLOUD ANALYSIS  (patch={PATCH_SIZE}px, nodata_thresh={NODATA_FRAC_THRESH*100:.0f}%)")
    print(f"{'='*60}")
    print(f"Tiles analysed   : {n_tiles:,}  (errors: {errors})")
    print(f"Total tokens     : {n_tiles * N_PATCHES:,}")
    print(f"Masked tokens    : {int(tile_masked_fracs.sum() * N_PATCHES):,}  "
          f"({tile_masked_fracs.mean()*100:.1f}% of all tokens)")
    print()

    # Tile-level summary
    thresholds = [0.0, 0.25, 0.50, 0.75, 1.0]
    print("Tiles by fraction of tokens masked:")
    print(f"  {'Threshold':<20}  {'N tiles':>8}  {'% of total':>10}")
    prev = 0.0
    for t in thresholds:
        n = int((tile_masked_fracs > t).sum())
        print(f"  > {t*100:.0f}% tokens masked  : {n:>8,}  ({n/n_tiles*100:>8.1f}%)")

    fully_masked = int((tile_masked_fracs == 1.0).sum())
    fully_clear  = int((tile_masked_fracs == 0.0).sum())
    print(f"\n  Fully masked (all 196 tokens bad) : {fully_masked:,}  ({fully_masked/n_tiles*100:.1f}%)")
    print(f"  Fully clear  (all 196 tokens good): {fully_clear:,}  ({fully_clear/n_tiles*100:.1f}%)")

    # Percentile distribution
    print(f"\nDistribution of per-tile masked-token fraction:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:02d}: {np.percentile(tile_masked_fracs, p)*100:.1f}%")

    # Worst stations (most cloud)
    station_mean = {s: np.mean(v) for s, v in station_stats.items()}
    sorted_stations = sorted(station_mean.items(), key=lambda x: -x[1])
    print(f"\nTop 10 cloudiest stations (mean masked-token fraction):")
    for name, frac in sorted_stations[:10]:
        n = len(station_stats[name])
        print(f"  {name:<45}  {frac*100:5.1f}%  ({n} tiles)")

    print(f"\nTop 10 clearest stations:")
    for name, frac in sorted_stations[-10:]:
        n = len(station_stats[name])
        print(f"  {name:<45}  {frac*100:5.1f}%  ({n} tiles)")

    # Tile-level survival after 50% invalid-patch threshold
    tiles_rejected = int((tile_masked_fracs > KEEP_THRESHOLD).sum())
    tiles_kept     = n_tiles - tiles_rejected
    print(f"\n{'='*60}")
    print(f"TILE-LEVEL FILTER  (reject if >{KEEP_THRESHOLD*100:.0f}% patches invalid)")
    print(f"{'='*60}")
    print(f"  Tiles kept : {tiles_kept:>8,}  ({tiles_kept/n_tiles*100:.1f}%)")
    print(f"  Tiles lost : {tiles_rejected:>8,}  ({tiles_rejected/n_tiles*100:.1f}%)")


if __name__ == "__main__":
    main()
