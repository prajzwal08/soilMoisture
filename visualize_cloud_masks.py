"""
visualize_cloud_masks.py
========================
Side-by-side RGB / cloud-mask panels for quality-checking SEnSeIv2 output.

Picks one representative tile per calendar month (Jan–Dec) so you can see
how cloud cover varies across seasons.  For months with multiple years of
data, the tile closest to the middle of that month is used.

Layout per station:
  Top row    — true-colour RGB (B04/B03/B02, 2–98 % contrast stretch)
  Bottom row — 7-class cloud mask with per-class pixel % annotated

Saves one SVG per station to the output directory.

Usage:
    python visualize_cloud_masks.py                        # all stations
    python visualize_cloud_masks.py --station AmeriFlux_CA-Cbo
    python visualize_cloud_masks.py --n-stations 5
    python visualize_cloud_masks.py --out-dir /tmp/cloud_vis
"""

from __future__ import annotations

import argparse
import calendar
from collections import defaultdict
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
# RGB = B04 B03 B02  →  band indices 3, 2, 1 (0-based)
RGB_IDX = (3, 2, 1)

MONTH_ABBR = [calendar.month_abbr[m] for m in range(1, 13)]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _remap_mask(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for i, c in enumerate(CM_CLASSES):
        out[arr == c] = i
    return out


def _load_rgb(tif_path: Path) -> np.ndarray | None:
    """Return uint8 (H, W, 3) RGB with 2–98 % contrast stretch, or None."""
    if not tif_path.exists():
        return None
    try:
        with rasterio.open(tif_path) as src:
            bands = src.read([b + 1 for b in RGB_IDX]).astype(np.float32)
    except Exception:
        return None
    rgb = np.empty((*bands.shape[1:], 3), dtype=np.uint8)
    for i, ch in enumerate(bands):
        valid = ch[ch > 0]
        lo, hi = (np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0))
        hi = max(hi, lo + 1e-6)
        rgb[..., i] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return rgb


def _load_mask(tif_path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(tif_path) as src:
            return _remap_mask(src.read(1))
    except Exception:
        return None


# ── date helpers ──────────────────────────────────────────────────────────────

def _parse_date(stem: str) -> tuple[int, int, int] | None:
    """Parse YYYYMMDD stem → (year, month, day), or None."""
    try:
        return int(stem[:4]), int(stem[4:6]), int(stem[6:8])
    except (ValueError, IndexError):
        return None


def pick_monthly_tiles(masks: list[Path]) -> list[tuple[str, Path]]:
    """
    For each calendar month (1–12) that has at least one tile, pick the tile
    whose day is closest to the 15th (middle of month).  Across years, prefer
    the most recent year.

    Returns a list of (label, path) sorted Jan → Dec, e.g. [("Jan 2022", ...), ...]
    """
    by_month: dict[int, list[tuple[int, int, Path]]] = defaultdict(list)
    for p in masks:
        parsed = _parse_date(p.stem)
        if parsed is None:
            continue
        year, month, day = parsed
        by_month[month].append((year, day, p))

    chosen = []
    for month in range(1, 13):
        tiles = by_month.get(month)
        if not tiles:
            continue
        # Most recent year first, then closest to day 15
        tiles.sort(key=lambda t: (-t[0], abs(t[1] - 15)))
        year, day, path = tiles[0]
        label = f"{MONTH_ABBR[month - 1]} {year}"
        chosen.append((label, path))
    return chosen


# ── station finder ────────────────────────────────────────────────────────────

def find_stations_with_masks(data_dir: Path) -> list[tuple[str, list[Path]]]:
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
                result.append((station_dir.name, masks))
    return result


# ── renderer ─────────────────────────────────────────────────────────────────

def render_station(station_name: str, masks: list[Path], out_dir: Path) -> Path:
    """
    One SVG per station: 2 rows (RGB / cloud mask) × N columns (one per month).
    """
    monthly = pick_monthly_tiles(masks)
    if not monthly:
        raise ValueError(f"No parseable dates for {station_name}")

    ncols = len(monthly)
    col_w = 3.0   # inches per column — keeps SVG manageable for 12 months
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(col_w * ncols, 6.5),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(station_name, fontsize=11, fontweight="bold", y=1.02)
    axes[0, 0].set_ylabel("RGB",        fontsize=8, labelpad=3)
    axes[1, 0].set_ylabel("Cloud mask", fontsize=8, labelpad=3)

    for col, (label, mask_tif) in enumerate(monthly):
        scratch_s2 = SCRATCH_DIR / station_name / "S2L2A" / mask_tif.name
        rgb  = _load_rgb(scratch_s2)
        mask = _load_mask(mask_tif)

        # ── RGB row ──────────────────────────────────────────────────────────
        ax_rgb = axes[0, col]
        ax_rgb.set_title(label, fontsize=7.5, pad=3)
        if rgb is not None:
            ax_rgb.imshow(rgb, interpolation="nearest")
        else:
            ax_rgb.set_facecolor("#1a1a1a")
            ax_rgb.text(0.5, 0.5, "no tile\nin scratch",
                        ha="center", va="center", color="#aaaaaa",
                        fontsize=6, transform=ax_rgb.transAxes)
        ax_rgb.axis("off")

        # ── cloud mask row ───────────────────────────────────────────────────
        ax_cm = axes[1, col]
        if mask is not None:
            ax_cm.imshow(mask, cmap=CM_CMAP, norm=CM_NORM, interpolation="nearest")
            total = mask.size
            lines = []
            for i, lbl in enumerate(CM_LABELS):
                pct = 100.0 * (mask == i).sum() / total
                if pct > 0.5:
                    lines.append(f"{lbl}: {pct:.0f}%")
            ax_cm.text(0.02, 0.02, "\n".join(lines),
                       transform=ax_cm.transAxes, fontsize=5,
                       color="white", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))
        else:
            ax_cm.set_facecolor("#1a1a1a")
        ax_cm.axis("off")

    # Legend on the last mask panel
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(CM_COLOURS, CM_LABELS)]
    axes[1, -1].legend(handles=patches, loc="lower right",
                       fontsize=5.5, framealpha=0.85, ncol=1, borderpad=0.3)

    out_path = out_dir / f"cloud_mask_vis_{station_name}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise S2L2A + cloud masks by month")
    parser.add_argument("--station",     type=str,  default=None)
    parser.add_argument("--n-stations",  type=int,  default=None)
    parser.add_argument("--data-dir",    type=Path, default=DATA_DIR)
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--out-dir",     type=Path, default=LOGS_DIR)
    args = parser.parse_args()

    global SCRATCH_DIR
    SCRATCH_DIR = args.scratch_dir

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stations = find_stations_with_masks(args.data_dir)
    if not stations:
        print("No CloudMask TIFs found yet.")
        return

    if args.station:
        stations = [(n, m) for n, m in stations if n == args.station]
        if not stations:
            print(f"Station '{args.station}' has no cloud masks yet.")
            return
    elif args.n_stations:
        stations = stations[:args.n_stations]

    print(f"Rendering {len(stations)} station(s) — one panel per calendar month\n")
    for station_name, masks in stations:
        try:
            out = render_station(station_name, masks, args.out_dir)
            monthly = pick_monthly_tiles(masks)
            print(f"  {station_name:45s}  {len(monthly):2d} months  →  {out.name}")
        except Exception as exc:
            print(f"  {station_name}: ERROR — {exc}")

    print(f"\nDone. SVGs in: {args.out_dir}")


if __name__ == "__main__":
    main()
