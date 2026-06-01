"""
filter_bad_tiles.py
===================
Final pre-tokenisation tile filter across all 4 modalities (S2L2A, S1RTC, DEM, LULC).

For every .tif, compute the fraction of nodata pixels at the TILE level.
Tiles where > BAD_TILE_THRESH (50%) of pixels are nodata are written to a
deletion manifest. Surviving tiles with < 1% nodata are NN-filled on-the-fly
in the tokeniser (precompute_terramind.py).

Nodata definitions (matching the tokeniser):
  S2L2A — any band == 0        (valid reflectance ≥ 1 post-harmonisation)
  S1RTC — any band is NaN      (orbit swath boundaries)
  DEM   — any band is NaN or 0 (ocean / sea-level ambiguity)
  LULC  — all bands == 0       (background / outside footprint; valid classes ≥ 10)

Outputs:
  csvs/bad_tile_manifest.txt  — one absolute path per line  →  delete these
  csvs/bad_tile_stats.csv     — per-station per-modality counts
  text/bad_tile_filter.txt    — human-readable summary

Usage:
    python filter_bad_tiles.py                          # all stations, 32 workers
    python filter_bad_tiles.py --station ISMN_SCAN_Abo  # single station (smoke test)
    python filter_bad_tiles.py --workers 16
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from tqdm import tqdm

SCRATCH_DIR     = Path("/gpfs/scratch1/shared/pkhanal/satellite")
BAD_TILE_THRESH = 0.50   # tile-level nodata fraction above which the tile is flagged
IMAGE_SIZE      = 224

# Mirrors the tokeniser's per-modality settings exactly
_MODALITY_CFG = {
    "S2L2A": {"subdir": "S2L2A", "glob": "*.tif",                      "do_crop": True},
    "S1RTC": {"subdir": "S1RTC", "glob": "*.tif",                      "do_crop": False},
    "DEM":   {"subdir": "DEM",   "glob": "*.tif",                      "do_crop": True},
    "LULC":  {"subdir": "LULC",  "glob": "[0-9][0-9][0-9][0-9].tif",  "do_crop": False},
}


def _nodata_mask(arr: np.ndarray, modality: str) -> np.ndarray:
    """(C, H, W) float32 → (H, W) bool; True = nodata pixel."""
    if modality == "S2L2A":
        return (arr == 0).any(axis=0)
    elif modality == "S1RTC":
        return np.isnan(arr).any(axis=0)
    elif modality == "DEM":
        return (np.isnan(arr) | (arr == 0)).any(axis=0)
    else:  # LULC: valid ESA WorldCover classes are multiples of 10 (10–95); 0 = nodata
        return (arr == 0).all(axis=0)


def _tile_nodata_frac(path: Path, modality: str, do_crop: bool) -> float | None:
    """Return fraction of nodata pixels for a tile, or None on load/size error."""
    try:
        with rasterio.open(path) as src:
            h, w = src.shape
            if do_crop:
                if h < IMAGE_SIZE or w < IMAGE_SIZE:
                    return None
                top    = (h - IMAGE_SIZE) // 2
                left   = (w - IMAGE_SIZE) // 2
                window = rasterio.windows.Window(left, top, IMAGE_SIZE, IMAGE_SIZE)
                arr    = src.read(window=window).astype(np.float32)
            else:
                if h != IMAGE_SIZE or w != IMAGE_SIZE:
                    return None
                arr = src.read().astype(np.float32)
    except Exception:
        return None
    bad = _nodata_mask(arr, modality)
    return float(bad.sum()) / bad.size


def _process_station(station_dir: Path) -> dict:
    result: dict = {"station": station_dir.name, "modalities": {}}

    for mod, cfg in _MODALITY_CFG.items():
        sub = station_dir / cfg["subdir"]
        if not sub.exists():
            continue

        tiles: list[Path] = sorted(sub.glob(cfg["glob"]))
        bad_paths: list[str] = []
        n_error = 0

        for tif in tiles:
            frac = _tile_nodata_frac(tif, mod, cfg["do_crop"])
            if frac is None:
                n_error += 1
            elif frac > BAD_TILE_THRESH:
                bad_paths.append(str(tif))

        result["modalities"][mod] = {
            "n_total"  : len(tiles),
            "n_bad"    : len(bad_paths),
            "n_error"  : n_error,
            "bad_paths": bad_paths,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag satellite tiles with >50% nodata for deletion")
    parser.add_argument("--station",     type=str,  default=None,
                        help="Single station name (smoke test)")
    parser.add_argument("--workers",     type=int,  default=32)
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    args = parser.parse_args()

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())

    n = len(station_dirs)
    print(f"Scratch dir : {args.scratch_dir}")
    print(f"Stations    : {n}")
    print(f"Workers     : {args.workers}")
    print()

    results: list[dict] = []
    if args.workers > 1 and n > 1:
        with Pool(processes=args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_process_station, station_dirs),
                          total=n, desc="Stations", unit="stn"):
                results.append(r)
    else:
        for sd in tqdm(station_dirs, desc="Stations", unit="stn"):
            results.append(_process_station(sd))
    results.sort(key=lambda r: r["station"])

    # ── Collect all bad paths ──────────────────────────────────────────────────
    all_bad: list[str] = []
    for r in results:
        for ms in r["modalities"].values():
            all_bad.extend(ms["bad_paths"])

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = Path("csvs/bad_tile_manifest.txt")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(all_bad) + ("\n" if all_bad else ""))
    print(f"Manifest  → {manifest}  ({len(all_bad):,} files to delete)")

    # ── Stats CSV ──────────────────────────────────────────────────────────────
    stats_path = Path("csvs/bad_tile_stats.csv")
    _ND_FIELDS = ["station", "modality", "n_total", "n_bad", "n_error", "pct_bad"]
    with stats_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_ND_FIELDS)
        w.writeheader()
        for r in results:
            for mod, s in r["modalities"].items():
                pct = round(s["n_bad"] / s["n_total"] * 100, 2) if s["n_total"] else 0.0
                w.writerow({"station": r["station"], "modality": mod,
                            "n_total": s["n_total"], "n_bad": s["n_bad"],
                            "n_error": s["n_error"], "pct_bad": pct})
    print(f"Stats CSV → {stats_path}")

    # ── Text summary ───────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "BAD TILE FILTER  (criterion: >50% nodata pixels at tile level)",
        "=" * 60,
        f"Stations scanned : {n}",
        "",
    ]
    for mod in _MODALITY_CFG:
        all_s = [r["modalities"][mod] for r in results if mod in r["modalities"]]
        if not all_s:
            lines += [f"  {mod}: no tiles found", ""]
            continue
        tot = sum(s["n_total"] for s in all_s)
        bad = sum(s["n_bad"]   for s in all_s)
        err = sum(s["n_error"] for s in all_s)
        pct = bad / tot * 100 if tot else 0.0
        hit = sum(1 for s in all_s if s["n_bad"] > 0)
        lines += [
            f"  {mod}:",
            f"    total tiles    : {tot:,}",
            f"    flagged (>50%) : {bad:,}  ({pct:.2f}%)",
            f"    load errors    : {err}",
            f"    stations hit   : {hit}",
            "",
        ]
    lines += [f"  TOTAL flagged  : {len(all_bad):,}", "=" * 60]
    summary = "\n".join(lines)

    txt = Path("text/bad_tile_filter.txt")
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(summary)
    print(f"Summary   → {txt}\n")
    print(summary)


if __name__ == "__main__":
    main()
