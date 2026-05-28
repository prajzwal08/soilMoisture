"""
visualize_cloud_masks.py
========================
Side-by-side RGB / cloud-mask panels for quality-checking SEnSeIv2 output.

For each station that has CloudMask TIFs in permanent storage, picks
representative dates (early / mid / late in the time series) and renders:

  Left column  — true-colour RGB from S2L2A (B04/B03/B02, 2–98 % stretch)
  Right column — 7-class cloud mask (colour-coded)

Saves one SVG per station to the output directory.

Usage:
    # All stations that have cloud masks so far
    python visualize_cloud_masks.py

    # Single station
    python visualize_cloud_masks.py --station AmeriFlux_CA-Cbo

    # Limit to first N stations with cloud masks
    python visualize_cloud_masks.py --n-stations 5

    # More dates per station (default 3)
    python visualize_cloud_masks.py --n-dates 5

    # Custom output dir (default: same as data logs)
    python visualize_cloud_masks.py --out-dir /tmp/cloud_vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_DIR    = Path("/gpfs/work3/0/prjs1968/data")
LOGS_DIR    = DATA_DIR / "logs"
SPLITS      = ["sm_only", "sm_and_flux", "flux_only"]

# ── cloud mask colour scheme (7-class TerraMesh encoding) ─────────────────────
CM_CLASSES = [0,         1,        2,          3,            4,             5,        255      ]
CM_COLOURS = ["#4caf50", "#1e88e5", "#b3e5fc",  "#b0bec5",   "#607d8b",    "#37474f", "#111111"]
CM_LABELS  = ["land",    "water",   "snow/ice", "thin cloud", "thick cloud", "shadow", "nodata" ]
CM_CMAP    = ListedColormap(CM_COLOURS)
CM_NORM    = BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 256], ncolors=7)

# S2L2A band layout: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
# RGB = B04 B03 B02  →  indices 3, 2, 1
RGB_IDX = (3, 2, 1)


def _remap_mask(arr: np.ndarray) -> np.ndarray:
    """Map raw class values (0-5, 255) → 0-6 index for CM_CMAP."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for i, c in enumerate(CM_CLASSES):
        out[arr == c] = i
    return out


def _load_rgb(tif_path: Path) -> np.ndarray | None:
    """Load S2L2A TIF and return uint8 RGB (H, W, 3) with 2–98% stretch."""
    if not tif_path.exists():
        return None
    try:
        with rasterio.open(tif_path) as src:
            bands = src.read(list(b + 1 for b in RGB_IDX)).astype(np.float32)
    except Exception:
        return None

    rgb = np.empty((*bands.shape[1:], 3), dtype=np.uint8)
    for i, ch in enumerate(bands):
        lo, hi = np.percentile(ch[ch > 0], [2, 98]) if (ch > 0).any() else (0, 1)
        hi = max(hi, lo + 1e-6)
        rgb[..., i] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return rgb


def _load_mask(tif_path: Path) -> np.ndarray | None:
    """Load CloudMask TIF and return remapped uint8 array."""
    try:
        with rasterio.open(tif_path) as src:
            return _remap_mask(src.read(1))
    except Exception:
        return None


def find_stations_with_masks(data_dir: Path) -> list[tuple[str, Path, list[Path]]]:
    """Return [(station_name, station_perm_dir, [mask_tifs])] sorted by name."""
    result = []
    for split in SPLITS:
        d = data_dir / split
        if not d.exists():
            continue
        for station_dir in sorted(d.iterdir()):
            cm_dir = station_dir / "CloudMask"
            if not cm_dir.exists():
                continue
            masks = sorted(cm_dir.glob("*.tif"))
            if masks:
                result.append((station_dir.name, station_dir, masks))
    return result


def pick_dates(masks: list[Path], n: int) -> list[Path]:
    """Pick n evenly-spaced dates from the sorted mask list."""
    if len(masks) <= n:
        return masks
    indices = [int(round(i * (len(masks) - 1) / (n - 1))) for i in range(n)]
    return [masks[i] for i in sorted(set(indices))]


def render_station(station_name: str, station_perm_dir: Path,
                   masks: list[Path], n_dates: int,
                   out_dir: Path) -> Path:
    """Render RGB + mask panels and save as SVG. Returns the output path."""
    chosen = pick_dates(masks, n_dates)
    ncols  = len(chosen)

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    if ncols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(station_name, fontsize=11, fontweight="bold", y=1.01)

    # Row labels
    axes[0, 0].set_ylabel("RGB", fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("Cloud mask", fontsize=9, labelpad=4)

    for col, mask_tif in enumerate(chosen):
        date_str = mask_tif.stem   # e.g. "20230415"

        # ── find matching S2L2A in scratch ────────────────────────────────────
        scratch_s2 = SCRATCH_DIR / station_name / "S2L2A" / mask_tif.name
        rgb = _load_rgb(scratch_s2)
        mask = _load_mask(mask_tif)

        # top row — RGB
        ax_rgb = axes[0, col]
        if rgb is not None:
            ax_rgb.imshow(rgb, interpolation="nearest")
        else:
            ax_rgb.set_facecolor("#222222")
            ax_rgb.text(0.5, 0.5, "no S2 tile\nin scratch",
                        ha="center", va="center", color="white", fontsize=7,
                        transform=ax_rgb.transAxes)
        ax_rgb.set_title(date_str, fontsize=8)
        ax_rgb.axis("off")

        # bottom row — cloud mask
        ax_cm = axes[1, col]
        if mask is not None:
            ax_cm.imshow(mask, cmap=CM_CMAP, norm=CM_NORM, interpolation="nearest")
            # Class fraction annotation
            total = mask.size
            fracs = {CM_LABELS[i]: 100 * (mask == i).sum() / total
                     for i in range(len(CM_CLASSES))}
            ann = "\n".join(f"{l}: {v:.0f}%" for l, v in fracs.items() if v > 0.5)
            ax_cm.text(0.02, 0.02, ann, transform=ax_cm.transAxes,
                       fontsize=5.5, color="white", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
        else:
            ax_cm.set_facecolor("#222222")
        ax_cm.axis("off")

    # Legend on last column bottom
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(CM_COLOURS, CM_LABELS)]
    axes[1, -1].legend(handles=patches, loc="lower right", fontsize=6,
                       framealpha=0.8, ncol=1, borderpad=0.4)

    out_path = out_dir / f"cloud_mask_vis_{station_name}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualise S2L2A + cloud masks")
    parser.add_argument("--station",    type=str,  default=None,
                        help="Process a single station by name")
    parser.add_argument("--n-stations", type=int,  default=None,
                        help="Limit to first N stations with cloud masks")
    parser.add_argument("--n-dates",    type=int,  default=3,
                        help="Dates per station (default 3: early/mid/late)")
    parser.add_argument("--data-dir",   type=Path, default=DATA_DIR)
    parser.add_argument("--scratch-dir",type=Path, default=SCRATCH_DIR)
    parser.add_argument("--out-dir",    type=Path, default=LOGS_DIR)
    args = parser.parse_args()

    global SCRATCH_DIR
    SCRATCH_DIR = args.scratch_dir

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stations = find_stations_with_masks(args.data_dir)
    if not stations:
        print("No CloudMask TIFs found. Run cloud_masking_inference.py first.")
        return

    if args.station:
        stations = [(n, d, m) for n, d, m in stations if n == args.station]
        if not stations:
            print(f"Station '{args.station}' has no cloud masks yet.")
            return
    elif args.n_stations:
        stations = stations[:args.n_stations]

    print(f"Rendering {len(stations)} station(s), {args.n_dates} dates each...\n")

    for station_name, station_dir, masks in stations:
        out = render_station(station_name, station_dir, masks,
                             args.n_dates, args.out_dir)
        print(f"  {station_name:40s}  {len(masks):3d} masks  →  {out.name}")

    print(f"\nDone. SVGs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
