"""
cleanup_tier1_failures.py
=========================
Identify and remove satellite tiles that fail TerraMesh tier-1 filtering
(> 50 % invalid pixels).  Before any file is deleted the script:

  1. Writes a full CSV record of every file to be deleted.
  2. Saves example visualisations of the worst failures per modality.
  3. Saves a summary histogram of nan_frac distributions.

Then deletes the files (skipped in --dry-run mode).

Invalid pixel definition:
  S2L2A  — ALL 12 bands == 0  (fill pixel / outside MGRS boundary)
  S1RTC  — ANY band is NaN    (swath edge stored as NaN by download script)
  DEM    — ANY band is NaN    (reprojection gap)

Usage:
  python cleanup_tier1_failures.py --dry-run       # preview only, no deletion
  python cleanup_tier1_failures.py                  # scan → record → delete
  python cleanup_tier1_failures.py --workers 8
  python cleanup_tier1_failures.py --n-stations 10  # smoke-test on 10 stations
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── GDAL performance env vars ─────────────────────────────────────────────────
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE",                    "TRUE")
os.environ.setdefault("GDAL_MAX_RAW_BLOCK_CACHE_SIZE","200000000")

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
REPO_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture")
FIG_DIR     = REPO_DIR / "fig"
TEXT_DIR    = REPO_DIR / "text"

TIER1_THRESHOLD = 0.50   # > 50 % invalid pixels → tier-1 failure

RGB_IDX = (3, 2, 1)      # S2L2A: B04 / B03 / B02

DATA_DIR = Path("/gpfs/work3/0/prjs1968/data")

# Stations excluded entirely (>50% S2L2A tile loss — MGRS boundary issue).
# Both their scratch directory AND data directory will be deleted.
EXCLUDED_STATIONS: dict[str, str] = {
    # station_name: split
    "ISMN_SNOTEL_CopperMountain":             "sm_only",
    "ISMN_SNOTEL_DiamondPeak":                "sm_only",
    "ISMN_CW3E_WindyGap":                     "sm_only",
    "ISMN_SNOTEL_WhiteHorseLake":             "sm_only",
    "ISMN_SNOTEL_BurnsideLake":               "sm_only",
    "ISMN_SCAN_CochoraRanch":                 "sm_only",
    "ISMN_SNOTEL_Corral":                     "sm_only",
    "ISMN_REMEDHUS_LaAtalaya":                "sm_only",
    "ISMN_SCAN_Wedowee":                      "sm_only",
    "ISMN_SNOTEL_Lakefork#3":                 "sm_only",
    "ISMN_SNOTEL_MedanoPass":                 "sm_only",
    "ISMN_SCAN_JordanValleyCwma":             "sm_only",
    "ISMN_TAHMO_CRIG(SoilMoistureStation1)":  "sm_only",
    "ICOS_DE-Tha":                            "sm_and_flux",
    "ISMN_SCAN_Tuskegee":                     "sm_only",
    "ISMN_NGARI_SQ06":                        "sm_only",
    "ISMN_HOAL_Hoal-16":                      "sm_only",
    # Minimum coverage exclusions (<12 avg S2L2A tiles/year after tier-1)
    "ISMN_SCAN_WearyLake":                    "sm_only",
    "ISMN_SNOTEL_Brumley":                    "sm_only",
    "ISMN_USCRN_Darrington-21-NNE":           "sm_only",
}


# ── NaN / invalidity maps ─────────────────────────────────────────────────────

def _invalid_map(path: Path, modality: str) -> np.ndarray | None:
    """(H, W) bool: True = invalid pixel.  Returns None on read error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
                nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        if modality == "S2L2A":
            return (arr == 0.0).all(axis=0)          # fill = all bands zero
        else:
            return np.isnan(arr).any(axis=0)          # S1RTC / DEM: any NaN
    except Exception:
        return None


# ── per-station scanner ───────────────────────────────────────────────────────

def _scan_station(station_dir: Path) -> list[dict]:
    """Return list of tier-1 failure records for one station."""
    failures = []

    # S2L2A and S1RTC — one TIF per date
    for modality in ("S2L2A", "S1RTC"):
        mod_dir = station_dir / modality
        if not mod_dir.exists():
            continue
        for tif in sorted(mod_dir.glob("*.tif")):
            inv = _invalid_map(tif, modality)
            if inv is None:
                continue
            nan_frac = float(inv.mean())
            if nan_frac > TIER1_THRESHOLD:
                stem = tif.stem
                date = stem[:8] if len(stem) >= 8 else stem
                failures.append({
                    "station":  station_dir.name,
                    "modality": modality,
                    "filename": tif.name,
                    "date":     date,
                    "nan_frac": nan_frac,
                    "path":     str(tif),
                })

    # DEM — single static file
    dem_path = station_dir / "DEM" / "dem.tif"
    if dem_path.exists():
        inv = _invalid_map(dem_path, "DEM")
        if inv is not None:
            nan_frac = float(inv.mean())
            if nan_frac > TIER1_THRESHOLD:
                failures.append({
                    "station":  station_dir.name,
                    "modality": "DEM",
                    "filename": "dem.tif",
                    "date":     "static",
                    "nan_frac": nan_frac,
                    "path":     str(dem_path),
                })

    return failures


# ── CSV record ────────────────────────────────────────────────────────────────

CSV_FIELDS = ["station", "modality", "filename", "date", "nan_frac", "path"]


def save_csv(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  CSV saved: {out_path}  ({len(records):,} records)")


# ── visualisation helpers ─────────────────────────────────────────────────────

def _load_rgb(path: Path) -> np.ndarray | None:
    """Return (H, W, 3) uint8 RGB for S2L2A (2–98% stretch on non-zero pixels)."""
    try:
        with rasterio.open(path) as src:
            bands = src.read([b + 1 for b in RGB_IDX]).astype(np.float32)
    except Exception:
        return None
    rgb = np.zeros((*bands.shape[1:], 3), dtype=np.uint8)
    for i, ch in enumerate(bands):
        valid = ch[ch > 0]
        lo, hi = (np.percentile(valid, [2, 98]) if valid.size > 0 else (0.0, 1.0))
        hi = max(hi, lo + 1e-6)
        rgb[..., i] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return rgb


def _load_band0(path: Path) -> np.ndarray | None:
    """Return (H, W) float32 first band."""
    try:
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return None


def _render_examples(records: list[dict], modality: str, out_path: Path,
                     n: int = 6):
    """
    Grid of worst-N tier-1 failures for one modality.
    Shows background image + red overlay on invalid pixels.
    """
    mod_recs = sorted([r for r in records if r["modality"] == modality],
                      key=lambda r: r["nan_frac"], reverse=True)[:n]
    if not mod_recs:
        return

    ncols = min(len(mod_recs), 3)
    nrows = (len(mod_recs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"Tier-1 failures — {modality}  (worst {len(mod_recs)})",
                 fontsize=11, fontweight="bold")

    for idx, rec in enumerate(mod_recs):
        ax = axes[idx // ncols][idx % ncols]
        path = Path(rec["path"])
        inv  = _invalid_map(path, modality)

        # ── background image ─────────────────────────────────────────────────
        if modality == "S2L2A":
            bg = _load_rgb(path)
            if bg is not None:
                ax.imshow(bg, interpolation="nearest")
            else:
                ax.set_facecolor("#222")
        else:
            band = _load_band0(path)
            if band is not None:
                valid = band[np.isfinite(band)]
                vmin  = np.percentile(valid, 2)  if valid.size else 0
                vmax  = np.percentile(valid, 98) if valid.size else 1
                ax.imshow(band, cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
            else:
                ax.set_facecolor("#222")

        # ── red overlay on invalid pixels ────────────────────────────────────
        if inv is not None:
            overlay = np.zeros((*inv.shape, 4), dtype=np.float32)
            overlay[inv, 0] = 1.0    # R
            overlay[inv, 3] = 0.55   # alpha
            ax.imshow(overlay, interpolation="nearest")

        label = f"{rec['date']}\n{rec['nan_frac']:.1%} invalid"
        ax.set_title(f"{rec['station']}\n{label}", fontsize=6.5, pad=3)
        ax.axis("off")

    # hide unused axes
    for idx in range(len(mod_recs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # legend
    legend_patch = mpatches.Patch(facecolor=(1, 0, 0, 0.55), label="invalid pixels")
    axes[0][-1].legend(handles=[legend_patch], loc="lower right",
                       fontsize=7, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Examples saved: {out_path}")


def _render_summary(records: list[dict],
                    totals: dict[str, int],
                    out_path: Path):
    """
    One histogram per modality showing nan_frac distribution of tier-1 failures.
    """
    mods = [m for m in ("S2L2A", "S1RTC", "DEM") if any(r["modality"] == m for r in records)]
    if not mods:
        return

    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 4.5), squeeze=False)
    fig.suptitle("Tier-1 failures — nan_frac distribution per modality", fontsize=11)

    colors = {"S2L2A": "#1e88e5", "S1RTC": "#fb8c00", "DEM": "#43a047"}

    for i, mod in enumerate(mods):
        ax = axes[0][i]
        fracs = [r["nan_frac"] for r in records if r["modality"] == mod]
        n_del   = len(fracs)
        n_total = totals.get(mod, 0)
        pct     = n_del / n_total * 100 if n_total else 0

        ax.hist(fracs, bins=20, range=(TIER1_THRESHOLD, 1.0),
                color=colors.get(mod, "#888"), edgecolor="white", linewidth=0.4)
        ax.axvline(TIER1_THRESHOLD, color="red", lw=1.2, ls="--", label="50% threshold")
        ax.set_xlabel("Invalid-pixel fraction")
        ax.set_ylabel("Number of tiles deleted")
        ax.set_title(f"{mod}\n{n_del:,} deleted  ({pct:.1f}% of {n_total:,} total)")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot saved: {out_path}")


# ── deletion ──────────────────────────────────────────────────────────────────

def delete_files(records: list[dict]) -> tuple[int, int]:
    """Delete individual tier-1 failing tiles (skip excluded stations — whole
    dir is removed by delete_excluded_stations instead)."""
    deleted = errors = 0
    for rec in records:
        if rec["station"] in EXCLUDED_STATIONS:
            continue   # handled by delete_excluded_stations
        p = Path(rec["path"])
        try:
            p.unlink()
            deleted += 1
        except Exception as exc:
            print(f"  [error] could not delete {p}: {exc}")
            errors += 1
    return deleted, errors


def delete_excluded_stations(scratch_dir: Path, data_dir: Path) -> tuple[int, int]:
    """
    Remove entire scratch and data directories for the 17 excluded stations.
    Returns (stations_deleted, errors).
    """
    import shutil
    deleted = errors = 0
    for stn, split in EXCLUDED_STATIONS.items():
        for root in [scratch_dir / stn, data_dir / split / stn]:
            if root.exists():
                try:
                    shutil.rmtree(root)
                    print(f"  Removed: {root}")
                    deleted += 1
                except Exception as exc:
                    print(f"  [error] could not remove {root}: {exc}")
                    errors += 1
            else:
                print(f"  [skip]  {root}  (not found)")
    return deleted, errors


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remove tier-1 NaN failures from scratch; record everything first")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--fig-dir",     type=Path, default=FIG_DIR)
    parser.add_argument("--text-dir",    type=Path, default=TEXT_DIR)
    parser.add_argument("--workers",     type=int,  default=8)
    parser.add_argument("--n-stations",  type=int,  default=None,
                        help="Limit to first N stations (smoke-test)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Scan and record but do not delete any files")
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.text_dir.mkdir(parents=True, exist_ok=True)

    stations = sorted(s for s in args.scratch_dir.iterdir() if s.is_dir())
    if args.n_stations:
        stations = stations[:args.n_stations]

    print(f"Scanning {len(stations)} station(s) with {args.workers} worker(s)…")
    if args.dry_run:
        print("  [DRY RUN — no files will be deleted]\n")

    # ── phase 1: scan ─────────────────────────────────────────────────────────
    all_records: list[dict] = []
    totals: dict[str, int]  = {"S2L2A": 0, "S1RTC": 0, "DEM": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_scan_station, s): s for s in stations}
        done = 0
        for fut in as_completed(futures):
            done += 1
            station_dir = futures[fut]
            try:
                recs = fut.result()
                all_records.extend(recs)
                # count totals for summary denominator
                for mod, glob_pat, is_static in [
                    ("S2L2A", "S2L2A/*.tif", False),
                    ("S1RTC", "S1RTC/*.tif", False),
                    ("DEM",   "DEM/dem.tif", True),
                ]:
                    if is_static:
                        totals[mod] += 1 if (station_dir / "DEM" / "dem.tif").exists() else 0
                    else:
                        totals[mod] += sum(1 for _ in (station_dir / mod.split("/")[0]).glob("*.tif")
                                           if (station_dir / mod.split("/")[0]).exists())
            except Exception as exc:
                print(f"  [error] {futures[fut].name}: {exc}")
            if done % 100 == 0 or done == len(stations):
                print(f"  {done}/{len(stations)} stations scanned  "
                      f"({len(all_records):,} failures so far)…")

    # ── phase 2: report ───────────────────────────────────────────────────────
    print(f"\nTier-1 failures found:")
    for mod in ("S2L2A", "S1RTC", "DEM"):
        n  = sum(1 for r in all_records if r["modality"] == mod)
        tt = totals[mod]
        print(f"  {mod:<8}  {n:>6,} / {tt:>7,}  ({n/tt*100:.1f}%)" if tt else
              f"  {mod:<8}  {n:>6,}")

    if not all_records:
        print("\nNo tier-1 failures found. Nothing to do.")
        return

    # ── phase 3: save records & figures ──────────────────────────────────────
    print("\nSaving records and figures before any deletion…")

    csv_path = args.text_dir / "tier1_deleted_records.csv"
    save_csv(all_records, csv_path)

    for mod in ("S2L2A", "S1RTC", "DEM"):
        _render_examples(all_records, mod,
                         args.fig_dir / f"tier1_failures_{mod.lower()}.png",
                         n=6)

    _render_summary(all_records, totals,
                    args.fig_dir / "tier1_failures_summary.png")

    # ── phase 4: delete ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Skipping deletion. Re-run without --dry-run to delete.")
        return

    # Step A — delete 17 excluded station directories entirely
    print(f"\nDeleting {len(EXCLUDED_STATIONS)} excluded station directories…")
    st_del, st_err = delete_excluded_stations(args.scratch_dir, DATA_DIR)
    print(f"  Directories removed : {st_del}")
    if st_err:
        print(f"  Errors              : {st_err}")

    # Step B — delete individual tier-1 failing tiles from remaining stations
    remaining = [r for r in all_records if r["station"] not in EXCLUDED_STATIONS]
    print(f"\nDeleting {len(remaining):,} individual tier-1 failing tiles…")
    deleted, errors = delete_files(remaining)
    print(f"  Files deleted : {deleted:,}")
    if errors:
        print(f"  Errors        : {errors}")
    print("Done.")


if __name__ == "__main__":
    main()
