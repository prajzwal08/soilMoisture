"""
analyze_cloud_coverage.py
=========================
Analyse SEnSeIv2 CloudMask TIFs to understand what fraction of S2L2A
tiles would survive a pre-tokenization validity filter.

For each CloudMask TIF, computes the fraction of 196 (14×14) tokens
that are "valid" — i.e. the corresponding 16×16 patch has <1% invalid
pixels (cloud classes 3/4/5 or nodata 255).

Reports:
  1. Tile-level valid-token fraction distribution (histogram)
  2. Survival rate at thresholds: 10 / 20 / 30 / 50 / 70 % valid tokens
  3. Monthly breakdown — median valid-token fraction by calendar month
  4. Per-station summary table

Saves plots to --out-dir (default: data/logs/).
Can be re-run after full cloud masking to get definitive numbers.

Usage:
    python analyze_cloud_coverage.py                    # all available masks
    python analyze_cloud_coverage.py --station AmeriFlux_CA-Cbo
    python analyze_cloud_coverage.py --out-dir /tmp/analysis
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("/gpfs/work3/0/prjs1968/data")
LOGS_DIR = DATA_DIR / "logs"
SPLITS   = ["sm_only", "sm_and_flux", "flux_only"]

# SEnSeIv2 classes that make a pixel invalid for TerraMind tokenization
INVALID_CLASSES = {3, 4, 5, 255}   # thin cloud, thick cloud, shadow, nodata

THRESHOLDS = [0.10, 0.20, 0.30, 0.50, 0.70]
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── core computation ──────────────────────────────────────────────────────────

def valid_token_fraction(cm_path: Path) -> float | None:
    """
    Read a CloudMask TIF and return the fraction of 196 tokens that are valid.
    A token (16×16 patch) is valid if <1% of its pixels are in INVALID_CLASSES.
    Returns None if the file can't be read.
    """
    try:
        with rasterio.open(cm_path) as src:
            cm = src.read(1).astype(np.uint8)
    except Exception:
        return None

    h, w = cm.shape
    # Crop / pad to 224×224 (14 tokens × 16 px)
    cm = cm[:224, :224]
    if cm.shape[0] < 224 or cm.shape[1] < 224:
        pad = np.full((224, 224), 255, dtype=np.uint8)
        pad[:cm.shape[0], :cm.shape[1]] = cm
        cm = pad

    invalid = np.isin(cm, list(INVALID_CLASSES))           # (224, 224) bool
    patch   = invalid.reshape(14, 16, 14, 16)
    frac_invalid = patch.mean(axis=(1, 3))                 # (14, 14) float
    token_valid  = frac_invalid < 0.01                     # True = valid token
    return float(token_valid.mean())                       # fraction 0–1


def collect_records(data_dir: Path, station_filter: str | None = None):
    """
    Walk all CloudMask directories and return a list of dicts:
        {station, split, month (1-12), year, valid_frac}
    """
    records = []
    for split in SPLITS:
        d = data_dir / split
        if not d.exists():
            continue
        for station_dir in sorted(d.iterdir()):
            if station_filter and station_dir.name != station_filter:
                continue
            cm_dir = station_dir / "CloudMask"
            if not cm_dir.exists():
                continue
            for tif in sorted(cm_dir.glob("*.tif")):
                stem = tif.stem
                try:
                    year  = int(stem[:4])
                    month = int(stem[4:6])
                except (ValueError, IndexError):
                    continue
                vf = valid_token_fraction(tif)
                if vf is None:
                    continue
                records.append({
                    "station": station_dir.name,
                    "split":   split,
                    "month":   month,
                    "year":    year,
                    "vf":      vf,
                })
    return records


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_analysis(records: list[dict], out_dir: Path, tag: str = ""):
    if not records:
        print("No records to plot.")
        return

    vf_all = np.array([r["vf"] for r in records])
    n_total = len(vf_all)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Cloud coverage analysis  —  {n_total:,} tiles from "
                 f"{len({r['station'] for r in records})} station(s)", fontsize=11)

    # ── Panel 1: histogram of valid-token fraction ────────────────────────────
    ax = axes[0]
    ax.hist(vf_all, bins=40, range=(0, 1), color="#1e88e5", edgecolor="white",
            linewidth=0.4)
    for thr in THRESHOLDS:
        kept = (vf_all >= thr).mean() * 100
        ax.axvline(thr, color="red", lw=1, ls="--", alpha=0.6)
        ax.text(thr + 0.01, ax.get_ylim()[1] * 0.95,
                f"{thr:.0%}\n→{kept:.0f}%", fontsize=6.5,
                color="red", va="top")
    ax.set_xlabel("Valid-token fraction per tile")
    ax.set_ylabel("Number of tiles")
    ax.set_title("Distribution of valid-token fraction")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # ── Panel 2: survival rate at each threshold ──────────────────────────────
    ax = axes[1]
    survival = [(vf_all >= t).mean() * 100 for t in THRESHOLDS]
    bars = ax.bar([f"{t:.0%}" for t in THRESHOLDS], survival,
                  color="#43a047", edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, survival):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Min valid-token threshold")
    ax.set_ylabel("% tiles kept")
    ax.set_title("Tile survival rate by threshold")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Panel 3: monthly median valid-token fraction ──────────────────────────
    ax = axes[2]
    by_month: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_month[r["month"]].append(r["vf"])

    months_present = sorted(by_month)
    medians = [np.median(by_month[m]) for m in months_present]
    p25     = [np.percentile(by_month[m], 25) for m in months_present]
    p75     = [np.percentile(by_month[m], 75) for m in months_present]
    x       = np.arange(len(months_present))

    ax.bar(x, medians, color="#fb8c00", edgecolor="white", linewidth=0.4,
           label="median")
    ax.errorbar(x, medians,
                yerr=[np.subtract(medians, p25), np.subtract(p75, medians)],
                fmt="none", color="#333333", capsize=3, lw=1.2,
                label="IQR")
    ax.axhline(0.30, color="red", lw=1, ls="--", label="30% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_ABBR[m - 1] for m in months_present],
                       fontsize=8)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Valid-token fraction")
    ax.set_title("Seasonal cloud pattern  (median ± IQR)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)

    plt.tight_layout()
    stem = out_dir / f"cloud_coverage_analysis{'_' + tag if tag else ''}"
    fig.savefig(stem.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {stem}.png / .svg")


def print_summary(records: list[dict]):
    if not records:
        return

    vf_all   = np.array([r["vf"] for r in records])
    stations = sorted({r["station"] for r in records})

    print(f"\n{'='*60}")
    print(f"CLOUD COVERAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Tiles analysed : {len(records):,}")
    print(f"  Stations       : {len(stations)}")
    print(f"  Valid-frac mean: {vf_all.mean():.1%}")
    print(f"  Valid-frac med : {np.median(vf_all):.1%}")
    print()

    print(f"  Survival rate by threshold:")
    for thr in THRESHOLDS:
        kept    = (vf_all >= thr).sum()
        dropped = len(vf_all) - kept
        print(f"    ≥{thr:.0%} valid tokens : {kept:5,} kept  "
              f"({kept/len(vf_all):.1%})   {dropped:5,} dropped")

    print()
    print(f"  Per-station summary (sorted by avg valid-frac):")
    print(f"  {'Station':<45} {'Tiles':>6} {'Avg valid':>10} {'At ≥30%':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*10} {'-'*8}")
    by_station: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_station[r["station"]].append(r["vf"])
    for stn in sorted(by_station, key=lambda s: np.mean(by_station[s])):
        vfs = np.array(by_station[stn])
        print(f"  {stn:<45} {len(vfs):>6,} {np.mean(vfs):>9.1%}"
              f"  {(vfs>=0.30).mean():>7.1%}")

    print(f"\n{'='*60}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse cloud coverage to inform pre-tokenization filtering")
    parser.add_argument("--station",  type=str,  default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir",  type=Path, default=LOGS_DIR)
    parser.add_argument("--tag",      type=str,  default="",
                        help="Optional suffix for output filenames")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CloudMask TIFs...")
    records = collect_records(args.data_dir, station_filter=args.station)
    print(f"  Found {len(records):,} tiles across "
          f"{len({r['station'] for r in records})} station(s)\n")

    if not records:
        print("No CloudMask TIFs found yet. Run cloud_masking_inference.py first.")
        return

    print_summary(records)
    plot_analysis(records, args.out_dir, tag=args.tag)
    print("Done.")


if __name__ == "__main__":
    main()
