"""
Cloud mask inference using CloudSEN12 UNetMobV2_V2.

Runs on S2 L1C GeoTIFFs and writes 4-class cloud masks:
    0 = clear
    1 = thick cloud
    2 = thin cloud
    3 = cloud shadow
  255 = nodata (where all S2L1C bands are 0)

Output: ~/data/satellite/{station}/CloudMask/YYYYMMDD.tif

Usage:
    conda run -n geo python cloud_masking_inference.py
"""

from pathlib import Path

import numpy as np
import rasterio
import torch

from cloudsen12_models.cloudsen12 import load_model_by_name

# ── Config ────────────────────────────────────────────────────────────────────
STATION_NAME = "Berlin_PSA15bucherchaus"   # station to process
DATA_ROOT    = Path("~/data/satellite").expanduser()
WEIGHTS_DIR  = Path("~/.georeader").expanduser()
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES   = None   # set to None to process all images
# ─────────────────────────────────────────────────────────────────────────────


def run_inference(station_name: str) -> None:
    station_dir  = DATA_ROOT / station_name
    s2l1c_dir    = station_dir / "S2L1C"
    out_dir      = station_dir / "CloudMask"
    out_dir.mkdir(exist_ok=True)

    tif_files = sorted(s2l1c_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No S2L1C files found in {s2l1c_dir}")

    if MAX_IMAGES is not None:
        tif_files = tif_files[:MAX_IMAGES]

    print(f"Station : {station_name}")
    print(f"Device  : {DEVICE}")
    print(f"Images  : {len(tif_files)}")

    model = load_model_by_name(
        "UNetMobV2_V2",
        weights_folder=str(WEIGHTS_DIR),
        device=torch.device(DEVICE),
    )
    print("Model loaded.\n")

    for tif_path in tif_files:
        out_path = out_dir / tif_path.name

        with rasterio.open(tif_path) as src:
            arr      = src.read().astype(np.float32) / 10_000   # (13, H, W), 0–1
            profile  = src.profile
            nodata_mask = (arr == 0).all(axis=0)                 # all bands 0 → nodata

        mask = model.predict(arr)                                # (H, W) uint8, 0–3
        mask[nodata_mask] = 255

        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(mask[np.newaxis, :, :])

        clear_pct = 100 * (mask == 0).sum() / mask.size
        cloud_pct = 100 * ((mask == 1) | (mask == 2)).sum() / mask.size
        print(f"  {tif_path.stem}  clear={clear_pct:.1f}%  cloud={cloud_pct:.1f}%")

    print(f"\nDone. Masks saved to {out_dir}")


if __name__ == "__main__":
    run_inference(STATION_NAME)
