"""
audit_satellite_data.py
=======================
Pre-tokenisation QA audit for satellite tiles across all 1,028 stations.

Part A — Coverage: compare actual S2L2A / S1RTC tile dates vs expected
         start_date/end_date from station_splits.csv; per-year image counts.

Part B — Nodata: sample tiles per modality (S2L2A, S1RTC, DEM), compute
         per-patch nodata fractions, classify as clean / fillable / partial / bad.
         Nodata definition:
           S2L2A — pixel == 0 (standard S2 nodata; valid reflectance ≥ 1)
           S1RTC — np.isnan (NaN from orbit swath boundaries)
           DEM   — np.isnan OR pixel == 0 (ocean/nodata; 0 = sea level ambiguous)

Outputs (relative to working directory):
  csvs/dataset_coverage.csv
  csvs/dataset_nodata.csv
  text/dataset_audit.txt

Usage:
    python audit_satellite_data.py                        # all stations, 16 workers, 10% sample
    python audit_satellite_data.py --station ISMN_Berlin_PflABerlin11H
    python audit_satellite_data.py --coverage-only
    python audit_satellite_data.py --nodata-only --sample-frac 0.20
    python audit_satellite_data.py --workers 4 --sample-frac 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
SPLITS_CSV  = Path("csvs/station_splits.csv")

# ── patch geometry ────────────────────────────────────────────────────────────
PATCH_SIZE      = 16
N_SIDE          = 14
N_PATCHES       = N_SIDE * N_SIDE    # 196
N_PIX           = PATCH_SIZE * PATCH_SIZE  # 256
IMAGE_SIZE      = 224

FILLABLE_THRESH = 0.01   # < 1% nodata per patch → NN-fillable in tokeniser
BAD_THRESH      = 0.50   # ≥ 50% nodata per patch → bad token (skip)


# ── splits CSV helpers ────────────────────────────────────────────────────────

def _load_splits() -> dict[str, dict]:
    """Return {scratch_dir_name: row_dict} from station_splits.csv.

    Naming rule (derived from actual data):
      network == source_network → {source_network}_{station_id}   (AmeriFlux, ICOS)
      network != source_network → {source_network}_{network}_{station_id}  (ISMN)
    """
    lookup: dict[str, dict] = {}
    if not SPLITS_CSV.exists():
        print(f"[warn] splits CSV not found: {SPLITS_CSV}")
        return lookup
    with SPLITS_CSV.open() as fh:
        for row in csv.DictReader(fh):
            sn  = row["source_network"]
            nw  = row["network"]
            sid = row["station_id"]
            key = f"{sn}_{sid}" if sn == nw else f"{sn}_{nw}_{sid}"
            lookup[key] = row
    return lookup


# ── tile loading ──────────────────────────────────────────────────────────────

def _load_arr(path: Path, do_crop: bool = False) -> np.ndarray | None:
    """Load .tif → (C, 224, 224) float32, with optional centre-crop. None on error."""
    try:
        with rasterio.open(path) as src:
            h, w = src.shape
            if do_crop:
                if h < IMAGE_SIZE or w < IMAGE_SIZE:
                    return None
                top    = (h - IMAGE_SIZE) // 2
                left   = (w - IMAGE_SIZE) // 2
                window = rasterio.windows.Window(left, top, IMAGE_SIZE, IMAGE_SIZE)
                arr = src.read(window=window).astype(np.float32)
            else:
                if h != IMAGE_SIZE or w != IMAGE_SIZE:
                    return None
                arr = src.read().astype(np.float32)
    except Exception:
        return None
    return arr


# ── patch nodata analysis ─────────────────────────────────────────────────────

def _bad_mask(arr: np.ndarray, modality: str) -> np.ndarray:
    """(C, 224, 224) → (224, 224) bool mask of bad (nodata) pixels."""
    if modality == "S2L2A":
        return (arr == 0).any(axis=0)
    elif modality == "S1RTC":
        return np.isnan(arr).any(axis=0)
    else:  # DEM: NaN or 0 (ocean/missing)
        return (np.isnan(arr) | (arr == 0)).any(axis=0)


def _patch_nodata_fracs(arr: np.ndarray, modality: str) -> np.ndarray:
    """(C, 224, 224) float32 → (196,) nodata fraction per 16×16 patch."""
    bad   = _bad_mask(arr, modality)
    fracs = np.empty(N_PATCHES, dtype=np.float32)
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            patch = bad[i * PATCH_SIZE:(i + 1) * PATCH_SIZE,
                        j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
            fracs[i * N_SIDE + j] = patch.sum() / N_PIX
    return fracs


def _nodata_stats(fracs_list: list[np.ndarray], n_total: int,
                  n_sampled: int) -> dict:
    if not fracs_list:
        return dict(n_tiles_total=n_total, n_tiles_sampled=n_sampled,
                    pct_patches_clean=0.0, pct_patches_fillable=0.0,
                    pct_patches_partial=0.0, pct_patches_bad=0.0,
                    pct_tiles_with_bad_patch=0.0)
    all_fracs = np.concatenate(fracs_list)
    n         = len(all_fracs)
    clean     = float((all_fracs == 0).sum()) / n * 100
    fillable  = float(((all_fracs > 0) & (all_fracs < FILLABLE_THRESH)).sum()) / n * 100
    partial   = float(((all_fracs >= FILLABLE_THRESH) & (all_fracs < BAD_THRESH)).sum()) / n * 100
    bad       = float((all_fracs >= BAD_THRESH).sum()) / n * 100
    tiles_bad = sum(1 for f in fracs_list if (f >= BAD_THRESH).any())
    return dict(
        n_tiles_total            = n_total,
        n_tiles_sampled          = n_sampled,
        pct_patches_clean        = round(clean,    3),
        pct_patches_fillable     = round(fillable, 3),
        pct_patches_partial      = round(partial,  3),
        pct_patches_bad          = round(bad,      3),
        pct_tiles_with_bad_patch = round(tiles_bad / len(fracs_list) * 100, 3),
    )


# ── per-station worker (must be module-level for multiprocessing) ─────────────

def _analyze_station(args: tuple) -> dict:
    (station_dir, splits_row, sample_frac, rng_seed,
     do_coverage, do_nodata) = args

    sid = station_dir.name
    result: dict = {
        "station"        : sid,
        "in_splits"      : splits_row is not None,
        "split"          : splits_row.get("split",      "") if splits_row else "",
        "expected_start" : splits_row.get("start_date", "") if splits_row else "",
        "expected_end"   : splits_row.get("end_date",   "") if splits_row else "",
        "coverage"       : {},
        "nodata"         : {},
    }

    # ── Part A: coverage ──────────────────────────────────────────────────────
    if do_coverage:
        for mod in ("S2L2A", "S1RTC"):
            tifs  = sorted(station_dir.glob(f"{mod}/*.tif"))
            dates : list[str] = []
            per_year: dict[str, int] = defaultdict(int)
            for tif in tifs:
                # S2L2A: stem = YYYYMMDD
                # S1RTC: stem = YYYYMMDD_ASC or YYYYMMDD_DESC
                date_str = tif.stem.split("_")[0]
                dates.append(date_str)
                per_year[date_str[:4]] += 1

            cov: dict = dict(n=len(tifs), start="", end="",
                             per_year={}, outside_range=False, n_outside=0)
            if dates:
                cov["start"]    = min(dates)
                cov["end"]      = max(dates)
                cov["per_year"] = dict(sorted(per_year.items()))
                if splits_row:
                    exp_s = splits_row.get("start_date", "")
                    exp_e = splits_row.get("end_date",   "")
                    if exp_s and exp_e:
                        outside = [d for d in dates if d < exp_s or d > exp_e]
                        cov["outside_range"] = len(outside) > 0
                        cov["n_outside"]     = len(outside)
            result["coverage"][mod] = cov

    # ── Part B: nodata analysis ───────────────────────────────────────────────
    if do_nodata:
        rng = random.Random(rng_seed)

        # S2L2A — centre-crop (matches tokeniser do_crop=True)
        for mod, do_crop in (("S2L2A", True), ("S1RTC", False), ("DEM", False)):
            if mod == "DEM":
                all_tifs  = list({p for p in (station_dir / "DEM").glob("*.tif")})
                n_total   = len(all_tifs)
                to_sample = all_tifs
            else:
                all_tifs  = sorted(station_dir.glob(f"{mod}/*.tif"))
                n_total   = len(all_tifs)
                k         = max(1, round(n_total * sample_frac))
                to_sample = rng.sample(all_tifs, min(k, n_total))

            fracs_list: list[np.ndarray] = []
            for path in to_sample:
                arr = _load_arr(path, do_crop=do_crop)
                if arr is None:
                    continue
                fracs_list.append(_patch_nodata_fracs(arr, mod))

            result["nodata"][mod] = _nodata_stats(fracs_list, n_total, len(to_sample))

    return result


# ── output writers ────────────────────────────────────────────────────────────

_COV_FIELDS = [
    "station", "split", "expected_start", "expected_end",
    "s2_n", "s2_start", "s2_end", "s2_outside_range", "s2_n_outside",
    "s1_n", "s1_start", "s1_end", "s1_outside_range", "s1_n_outside",
    "s2_per_year", "s1_per_year",
]

_ND_FIELDS = [
    "station", "split", "modality",
    "n_tiles_total", "n_tiles_sampled",
    "pct_patches_clean", "pct_patches_fillable",
    "pct_patches_partial", "pct_patches_bad",
    "pct_tiles_with_bad_patch",
]


def _write_coverage_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COV_FIELDS)
        w.writeheader()
        for r in results:
            c = r["coverage"]
            s2 = c.get("S2L2A", {})
            s1 = c.get("S1RTC", {})
            w.writerow({
                "station"         : r["station"],
                "split"           : r["split"],
                "expected_start"  : r["expected_start"],
                "expected_end"    : r["expected_end"],
                "s2_n"            : s2.get("n", 0),
                "s2_start"        : s2.get("start", ""),
                "s2_end"          : s2.get("end", ""),
                "s2_outside_range": s2.get("outside_range", False),
                "s2_n_outside"    : s2.get("n_outside", 0),
                "s1_n"            : s1.get("n", 0),
                "s1_start"        : s1.get("start", ""),
                "s1_end"          : s1.get("end", ""),
                "s1_outside_range": s1.get("outside_range", False),
                "s1_n_outside"    : s1.get("n_outside", 0),
                "s2_per_year"     : json.dumps(s2.get("per_year", {})),
                "s1_per_year"     : json.dumps(s1.get("per_year", {})),
            })


def _write_nodata_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_ND_FIELDS)
        w.writeheader()
        for r in results:
            for mod, stats in r["nodata"].items():
                w.writerow({
                    "station" : r["station"],
                    "split"   : r["split"],
                    "modality": mod,
                    **stats,
                })


def _write_text_summary(results: list[dict], sample_frac: float,
                        do_coverage: bool, do_nodata: bool, path: Path) -> str:
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln("=" * 60)
    ln("SATELLITE DATA AUDIT")
    ln("=" * 60)
    ln(f"Stations analysed : {len(results)}")

    # ── coverage summary ──────────────────────────────────────────────────────
    if do_coverage:
        ln()
        ln("── COVERAGE ──────────────────────────────────────────────")
        not_in_splits     = sum(1 for r in results if not r["in_splits"])
        s2_outside        = sum(1 for r in results
                               if r["coverage"].get("S2L2A", {}).get("outside_range"))
        s1_outside        = sum(1 for r in results
                               if r["coverage"].get("S1RTC", {}).get("outside_range"))
        s2_counts = [r["coverage"].get("S2L2A", {}).get("n", 0) for r in results]
        s1_counts = [r["coverage"].get("S1RTC", {}).get("n", 0) for r in results]
        ln(f"Not in splits CSV     : {not_in_splits}")
        ln(f"S2 outside date range : {s2_outside} stations")
        ln(f"S1 outside date range : {s1_outside} stations")
        if s2_counts:
            ln(f"S2 images/station     : median={int(np.median(s2_counts))}  "
               f"min={min(s2_counts)}  max={max(s2_counts)}  "
               f"total={sum(s2_counts):,}")
        if s1_counts:
            ln(f"S1 images/station     : median={int(np.median(s1_counts))}  "
               f"min={min(s1_counts)}  max={max(s1_counts)}  "
               f"total={sum(s1_counts):,}")

        # per-year totals across all stations
        s2_yr: dict[str, int] = defaultdict(int)
        s1_yr: dict[str, int] = defaultdict(int)
        for r in results:
            for yr, cnt in r["coverage"].get("S2L2A", {}).get("per_year", {}).items():
                s2_yr[yr] += cnt
            for yr, cnt in r["coverage"].get("S1RTC", {}).get("per_year", {}).items():
                s1_yr[yr] += cnt
        if s2_yr:
            ln()
            ln("  S2L2A tiles per year (all stations):")
            for yr in sorted(s2_yr):
                ln(f"    {yr}: {s2_yr[yr]:,}")
        if s1_yr:
            ln()
            ln("  S1RTC tiles per year (all stations):")
            for yr in sorted(s1_yr):
                ln(f"    {yr}: {s1_yr[yr]:,}")

    # ── nodata summary ────────────────────────────────────────────────────────
    if do_nodata:
        ln()
        ln(f"── NODATA  ({int(sample_frac*100)}% tile sample for S2/S1; 100% for DEM) ─────")

        for mod in ("S2L2A", "S1RTC", "DEM"):
            all_stats = [r["nodata"][mod] for r in results if mod in r["nodata"]]
            if not all_stats:
                continue

            # Weighted average across stations (weight by n_tiles_sampled)
            keys = ("pct_patches_clean", "pct_patches_fillable",
                    "pct_patches_partial", "pct_patches_bad",
                    "pct_tiles_with_bad_patch")
            weights = np.array([s["n_tiles_sampled"] for s in all_stats], dtype=float)
            totw    = weights.sum()
            if totw == 0:
                continue
            avgs = {k: round(float(np.dot([s[k] for s in all_stats], weights) / totw), 2)
                    for k in keys}
            total_tiles   = sum(s["n_tiles_total"]   for s in all_stats)
            sampled_tiles = sum(s["n_tiles_sampled"] for s in all_stats)
            n_bad_stations = sum(1 for s in all_stats if s["pct_patches_bad"] > 0)

            label = "pct stations with bad DEM patch" if mod == "DEM" else "pct tiles with any bad patch"
            ln()
            ln(f"  {mod} ({sampled_tiles:,} tiles sampled of {total_tiles:,} total):")
            ln(f"    clean       : {avgs['pct_patches_clean']:.2f}%")
            ln(f"    fillable    : {avgs['pct_patches_fillable']:.2f}%  (0 < nodata < 1%)")
            ln(f"    partial     : {avgs['pct_patches_partial']:.2f}%  (1% ≤ nodata < 50%)")
            ln(f"    bad         : {avgs['pct_patches_bad']:.2f}%  (nodata ≥ 50%)")
            ln(f"    {label:<36}: {avgs['pct_tiles_with_bad_patch']:.2f}%")
            ln(f"    stations with any bad patch : {n_bad_stations}/{len(all_stats)}")

    ln()
    ln("=" * 60)
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit satellite data coverage and nodata")
    parser.add_argument("--station",       type=str,   default=None,
                        help="Process a single station by name")
    parser.add_argument("--sample-frac",   type=float, default=0.10,
                        help="Fraction of S2/S1 tiles to sample for nodata analysis (default 0.10)")
    parser.add_argument("--workers",       type=int,   default=16,
                        help="Multiprocessing workers (default 16)")
    parser.add_argument("--coverage-only", action="store_true",
                        help="Skip nodata analysis")
    parser.add_argument("--nodata-only",   action="store_true",
                        help="Skip coverage analysis")
    parser.add_argument("--scratch-dir",   type=Path,  default=SCRATCH_DIR,
                        help="Scratch root with per-station satellite dirs")
    args = parser.parse_args()

    do_coverage = not args.nodata_only
    do_nodata   = not args.coverage_only

    print(f"Scratch dir  : {args.scratch_dir}")
    print(f"Splits CSV   : {SPLITS_CSV}")
    print(f"Workers      : {args.workers}")
    print(f"Sample frac  : {args.sample_frac}")
    print(f"Coverage     : {do_coverage}")
    print(f"Nodata check : {do_nodata}")
    print()

    splits = _load_splits()
    print(f"Splits CSV entries loaded : {len(splits):,}")

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())

    n = len(station_dirs)
    print(f"Stations to process : {n}\n")

    rng_seed = 42
    tasks = [
        (sd, splits.get(sd.name), args.sample_frac, rng_seed, do_coverage, do_nodata)
        for sd in station_dirs
    ]

    results: list[dict] = []
    if args.workers > 1 and n > 1:
        with Pool(processes=args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_analyze_station, tasks),
                          total=n, desc="Stations", unit="stn"):
                results.append(r)
    else:
        for task in tqdm(tasks, desc="Stations", unit="stn"):
            results.append(_analyze_station(task))

    results.sort(key=lambda r: r["station"])

    # ── write outputs ─────────────────────────────────────────────────────────
    if do_coverage:
        cov_path = Path("csvs/dataset_coverage.csv")
        _write_coverage_csv(results, cov_path)
        print(f"Coverage CSV → {cov_path}")

    if do_nodata:
        nd_path = Path("csvs/dataset_nodata.csv")
        _write_nodata_csv(results, nd_path)
        print(f"Nodata CSV   → {nd_path}")

    txt_path = Path("text/dataset_audit.txt")
    summary  = _write_text_summary(results, args.sample_frac,
                                   do_coverage, do_nodata, txt_path)
    print(f"Summary      → {txt_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
