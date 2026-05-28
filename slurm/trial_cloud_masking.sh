#!/bin/bash
#SBATCH --job-name=cm_trial
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/trial_cloud_mask_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/trial_cloud_mask_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/sensei/bin/python
DATA_DIR=/gpfs/work3/0/prjs1968/data

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Started    : $(date)"
echo ""

# ── trial run ─────────────────────────────────────────────────────────────────
# 3 stations — enough to measure per-tile throughput and catch any API issues
$PYTHON cloud_masking_inference.py \
    --start-idx 0 \
    --end-idx   3

echo ""
echo "Masking finished : $(date)"
echo ""

# ── output validation ─────────────────────────────────────────────────────────
echo "=========================================="
echo "OUTPUT VALIDATION"
echo "=========================================="

$PYTHON - <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import rasterio

DATA_DIR = Path("/gpfs/work3/0/prjs1968/data")
SPLITS   = ["sm_only", "sm_and_flux", "flux_only"]
VALID_CLASSES = {0, 1, 2, 3, 4, 5, 255}

total_files = 0
total_bytes = 0
errors      = []
class_counts = {c: 0 for c in VALID_CLASSES}

for split in SPLITS:
    for station_dir in sorted((DATA_DIR / split).iterdir()):
        cm_dir = station_dir / "CloudMask"
        if not cm_dir.exists():
            continue
        for tif in sorted(cm_dir.glob("*.tif")):
            total_files += 1
            total_bytes += tif.stat().st_size
            try:
                with rasterio.open(tif) as src:
                    arr = src.read(1)
                    if arr.dtype != np.uint8:
                        errors.append(f"wrong dtype {tif.name}: {arr.dtype}")
                    unexpected = set(np.unique(arr)) - VALID_CLASSES
                    if unexpected:
                        errors.append(f"unexpected classes {tif.name}: {unexpected}")
                    for c in VALID_CLASSES:
                        class_counts[c] += int((arr == c).sum())
            except Exception as exc:
                errors.append(f"load error {tif.name}: {exc}")

print(f"  .tif files found : {total_files}")
print(f"  total size       : {total_bytes / 1e6:.1f} MB")
print(f"  avg size/file    : {total_bytes / max(total_files, 1) / 1024:.0f} KB")
print(f"  errors           : {len(errors)}")
for e in errors[:20]:
    print(f"    {e}")

total_px = sum(class_counts.values())
if total_px > 0:
    print(f"\n  Pixel class distribution (across all tiles):")
    labels = {0: "land", 1: "water", 2: "snow/ice", 3: "thin cloud",
              4: "thick cloud", 5: "cloud shadow", 255: "nodata"}
    for c, label in labels.items():
        pct = 100.0 * class_counts[c] / total_px
        print(f"    {c:3d}  {label:<14s}  {class_counts[c]:>10,d}  ({pct:5.1f}%)")

if not errors:
    print("\n  All masks valid ✓")
PYEOF

echo ""
echo "=========================================="
echo "CLOUD MASK VISUALISATION"
echo "=========================================="

$PYTHON - <<'PYEOF'
"""
For each station processed in this trial, pick 3 representative dates
(early / mid / late in the time series) and render a 7-class colour map.
Saved as SVG to the logs dir for offline inspection.
"""
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

DATA_DIR  = Path("/gpfs/work3/0/prjs1968/data")
LOGS_DIR  = Path("/gpfs/work3/0/prjs1968/data/logs")
SPLITS    = ["sm_only", "sm_and_flux", "flux_only"]

# 7-class colour scheme (matching TerraMesh class encoding)
CLASSES = [0, 1, 2, 3, 4, 5, 255]
COLOURS = ["#4caf50", "#1e88e5", "#b3e5fc", "#b0bec5", "#607d8b", "#37474f", "#000000"]
LABELS  = ["land", "water", "snow/ice", "thin cloud", "thick cloud", "shadow", "nodata"]
CMAP    = ListedColormap(COLOURS)
NORM    = BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 256], ncolors=7)

def remap(arr):
    """Map raw class values to 0-6 index for CMAP/NORM."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for i, c in enumerate(CLASSES):
        out[arr == c] = i
    return out

# Collect stations that have CloudMask output
stations = []
for split in SPLITS:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        continue
    for station_dir in sorted(split_dir.iterdir()):
        cm_dir = station_dir / "CloudMask"
        tifs = sorted(cm_dir.glob("*.tif")) if cm_dir.exists() else []
        if tifs:
            stations.append((station_dir.name, tifs))

if not stations:
    print("  No CloudMask outputs found — skipping visualisation.")
else:
    for station_name, tifs in stations:
        # Pick early / mid / late tiles
        n = len(tifs)
        indices = sorted({0, n // 2, n - 1})
        chosen  = [tifs[i] for i in indices]

        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5))
        if len(chosen) == 1:
            axes = [axes]

        fig.suptitle(station_name, fontsize=11, fontweight="bold")

        for ax, tif in zip(axes, chosen):
            with rasterio.open(tif) as src:
                arr = src.read(1)
            ax.imshow(remap(arr), cmap=CMAP, norm=NORM, interpolation="nearest")
            ax.set_title(tif.stem, fontsize=8)
            ax.axis("off")

        # Legend on last axis
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLOURS, LABELS)]
        axes[-1].legend(handles=patches, loc="lower right", fontsize=6,
                        framealpha=0.8, ncol=1)

        out_svg = LOGS_DIR / f"trial_cloud_mask_{station_name}.svg"
        fig.savefig(out_svg, format="svg", bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"  Saved: {out_svg}")

    print(f"\n  {len(stations)} SVG(s) written to {LOGS_DIR}")
PYEOF

echo ""
echo "Finished : $(date)"
