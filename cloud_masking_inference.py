"""
Batch cloud mask inference using CloudSEN12 cloudsen12l2a (S2L2A, 12 bands).

Processes all stations under DATA_ROOT. Skips images that already have a mask.

Output per image: ~/data/satellite/{station}/CloudMask/YYYYMMDD.tif
    0 = clear  1 = thick cloud  2 = thin cloud  3 = cloud shadow  255 = nodata

Usage:
    conda run -n geo python cloud_masking_inference.py
"""

from pathlib import Path

import numpy as np
import rasterio
import torch

from cloudsen12_models.cloudsen12 import load_model_by_name

DATA_ROOT = Path("~/data/satellite").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def run_inference(model, station_dir: Path) -> tuple[int, int]:
    s2l2a_dir = station_dir / "S2L2A"
    out_dir   = station_dir / "CloudMask"
    out_dir.mkdir(exist_ok=True)

    todo = [f for f in sorted(s2l2a_dir.glob("*.tif"))
            if not (out_dir / f.name).exists()]

    if not todo:
        return 0, 0

    done = skipped = 0
    for tif_path in todo:
        out_path = out_dir / tif_path.name
        try:
            with rasterio.open(tif_path) as src:
                arr     = src.read().astype(np.float32) / 10_000   # (12, H, W)
                profile = src.profile
                nodata_mask = (arr == 0).all(axis=0)

            mask = model.predict(arr)                               # (H, W) uint8
            mask[nodata_mask] = 255

            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(mask[np.newaxis])

            done += 1
        except Exception as e:
            print(f"    ERROR {tif_path.name}: {e}")
            skipped += 1

    return done, skipped


def main():
    station_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"Stations : {len(station_dirs)}")
    print(f"Device   : {DEVICE}\n")

    model = load_model_by_name("cloudsen12l2a", device=torch.device(DEVICE))
    print("Model loaded.\n")

    total_done = total_skipped = 0
    for station_dir in station_dirs:
        done, skipped = run_inference(model, station_dir)
        total_done    += done
        total_skipped += skipped
        if done > 0:
            print(f"  {station_dir.name}: {done} done, {skipped} errors")

    print(f"\nFinished. Total masks written: {total_done}  errors: {total_skipped}")


if __name__ == "__main__":
    main()
