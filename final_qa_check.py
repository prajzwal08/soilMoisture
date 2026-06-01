"""
final_qa_check.py
=================
Comprehensive pre-tokenisation QA across all 1,010 active stations.

Five checks:
  1. Year coverage   — ≥3 years with ≥1 S1 OR S2 tile per station
  2. Bad tile scan   — any tile where >50% of 196 patches have >1% nodata → delete
  3. Satellite mods  — all 4 modalities present (S2L2A, S1RTC, DEM, LULC)
  4. Soil            — soil/soil_patch.tif exists + soil_patch_ok == True
  5. ERA5-Land       — at least 1 ERA5Land/meteo_YYYY.nc; full year range checked

Run AFTER downloads complete (LULC redownload + DEM missing fix).

Outputs:
  text/final_qa_report.txt      — full human-readable summary
  csvs/final_qa.csv             — per-station pass/fail per check
  csvs/final_qa_bad_tiles.txt   — absolute paths of tiles to delete

Usage:
    python final_qa_check.py
    python final_qa_check.py --workers 16
    python final_qa_check.py --station ISMN_ARM_Anthony   # smoke test
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR  = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
SPLITS_CSV   = Path("csvs/station_splits.csv")
IMG_CSV      = Path("csvs/images_per_station_year.csv")
OUT_CSV      = Path("csvs/final_qa.csv")
OUT_MANIFEST = Path("csvs/final_qa_bad_tiles.txt")
OUT_TXT      = Path("text/final_qa_report.txt")

IMAGE_SIZE       = 224
PATCH_SIZE       = 16
N_SIDE           = 14            # 14×14 = 196 patches
BAD_PATCH_THRESH = 0.01          # >1% nodata within a patch → patch invalid
BAD_TILE_THRESH  = 0.50          # >50% invalid patches → tile flagged

_MODALITY_CFG = {
    "S2L2A": {"subdir": "S2L2A", "glob": "*.tif",                     "do_crop": True},
    "S1RTC": {"subdir": "S1RTC", "glob": "*.tif",                     "do_crop": False},
    "DEM":   {"subdir": "DEM",   "glob": "*.tif",                     "do_crop": True},
    "LULC":  {"subdir": "LULC",  "glob": "[0-9][0-9][0-9][0-9].tif", "do_crop": False},
}

DATA_SUBDIRS = ("sm_only", "sm_and_flux", "flux_only")


# ── helpers ───────────────────────────────────────────────────────────────────

def _nodata_mask(arr: np.ndarray, modality: str) -> np.ndarray:
    if modality == "S2L2A":
        return (arr == 0).any(axis=0)
    elif modality == "S1RTC":
        return np.isnan(arr).any(axis=0)
    elif modality == "DEM":
        return (np.isnan(arr) | (arr == 0)).any(axis=0)
    else:  # LULC
        return (arr == 0).all(axis=0)


def _pct_bad_patches(path: Path, modality: str, do_crop: bool) -> float | None:
    """Fraction of patches with >1% nodata, or None on load/size error."""
    try:
        with rasterio.open(path) as src:
            h, w = src.shape
            if do_crop:
                if h < IMAGE_SIZE or w < IMAGE_SIZE:
                    return None
                top  = (h - IMAGE_SIZE) // 2
                left = (w - IMAGE_SIZE) // 2
                arr  = src.read(window=rasterio.windows.Window(
                    left, top, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
            else:
                if h != IMAGE_SIZE or w != IMAGE_SIZE:
                    return None
                arr = src.read().astype(np.float32)
    except Exception:
        return None

    nodata = _nodata_mask(arr, modality)
    bad = 0
    for pi in range(N_SIDE):
        for pj in range(N_SIDE):
            patch = nodata[pi*PATCH_SIZE:(pi+1)*PATCH_SIZE,
                           pj*PATCH_SIZE:(pj+1)*PATCH_SIZE]
            if patch.mean() > BAD_PATCH_THRESH:
                bad += 1
    return bad / (N_SIDE * N_SIDE)


def _station_data_dir(station_folder: str) -> Path | None:
    for sub in DATA_SUBDIRS:
        d = DATA_ROOT / sub / station_folder
        if d.exists():
            return d
    return None


# ── per-station scan ──────────────────────────────────────────────────────────

def _scan_station(args: tuple) -> dict:
    station_folder, n_years, soil_patch_ok, start_date, end_date = args

    result: dict = {
        "station":          station_folder,
        # check 1
        "n_years":          n_years,
        "fail_years":       n_years < 3,
        # check 2
        "n_bad_tiles":      0,
        "bad_tile_paths":   [],
        "fail_bad_tiles":   False,
        # check 3
        "missing_mods":     [],
        "fail_modalities":  False,
        # check 4
        "soil_file_ok":     False,
        "soil_flag_ok":     bool(soil_patch_ok),
        "fail_soil":        True,
        # check 5
        "era5_found":       0,
        "era5_expected":    0,
        "era5_missing_yrs": [],
        "fail_era5":        True,
    }

    scratch = SCRATCH_DIR / station_folder

    # ── check 2 + 3: satellite tiles ─────────────────────────────��───────────
    for mod, cfg in _MODALITY_CFG.items():
        sub = scratch / cfg["subdir"]
        tiles = sorted(sub.glob(cfg["glob"])) if sub.exists() else []
        if not tiles:
            result["missing_mods"].append(mod)
            continue
        for tif in tiles:
            frac = _pct_bad_patches(tif, mod, cfg["do_crop"])
            if frac is not None and frac > BAD_TILE_THRESH:
                result["n_bad_tiles"] += 1
                result["bad_tile_paths"].append(str(tif))

    result["fail_modalities"] = len(result["missing_mods"]) > 0
    result["fail_bad_tiles"]  = result["n_bad_tiles"] > 0

    # ── check 4: soil ─────────────────────────────────────────────────────────
    data_dir = _station_data_dir(station_folder)
    if data_dir is not None:
        result["soil_file_ok"] = (data_dir / "soil" / "soil_patch.tif").exists()
    result["fail_soil"] = not (result["soil_file_ok"] and result["soil_flag_ok"])

    # ── check 5: ERA5-Land ────────────────────────────────────────────────────
    if data_dir is not None:
        start_yr = int(str(start_date)[:4])
        end_yr   = int(str(end_date)[:4])
        expected = list(range(start_yr, end_yr + 1))
        result["era5_expected"] = len(expected)
        era5_dir = data_dir / "ERA5Land"
        missing  = [y for y in expected
                    if not (era5_dir / f"meteo_{y}.nc").exists()]
        result["era5_found"]       = len(expected) - len(missing)
        result["era5_missing_yrs"] = missing
        result["fail_era5"]        = result["era5_found"] == 0

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--station", type=str, default=None,
                        help="Single station name for smoke testing")
    args = parser.parse_args()

    splits = pd.read_csv(SPLITS_CSV)
    img    = pd.read_csv(IMG_CSV, dtype={"year": str})

    # Build station folder names
    def folder(r):
        return f"{r.source_network}_{r.station_id}" if r.source_network == r.network \
               else f"{r.source_network}_{r.network}_{r.station_id}"
    splits["folder"] = splits.apply(folder, axis=1)

    # Check 1 inputs: n_years per station (s1 OR s2 > 0)
    year_counts = (
        img.assign(combined=(img["s2"] > 0) | (img["s1"] > 0))
           .groupby("station")["combined"]
           .sum()
           .astype(int)
    )

    rows = splits if args.station is None else splits[splits["folder"] == args.station]

    scan_args = []
    for _, st in rows.iterrows():
        f = st["folder"]
        scan_args.append((
            f,
            int(year_counts.get(f, 0)),
            bool(st.get("soil_patch_ok", True)),
            int(st["start_date"]),
            int(st["end_date"]),
        ))

    print(f"\nScanning {len(scan_args)} stations with {args.workers} workers…")
    if args.workers > 1 and len(scan_args) > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(_scan_station, scan_args),
                                total=len(scan_args), unit="station"))
    else:
        results = [_scan_station(a) for a in tqdm(scan_args)]

    # ── collect bad tile paths ────────────────────────────────────────────────
    all_bad_tiles: list[str] = []
    for r in results:
        all_bad_tiles.extend(r["bad_tile_paths"])

    # ── write CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "station", "n_years", "fail_years",
            "n_bad_tiles", "fail_bad_tiles",
            "missing_modalities", "fail_modalities",
            "soil_file_ok", "soil_flag_ok", "fail_soil",
            "era5_found", "era5_expected", "era5_missing_years", "fail_era5",
            "overall_pass",
        ])
        for r in results:
            overall = not any([
                r["fail_years"], r["fail_bad_tiles"],
                r["fail_modalities"], r["fail_soil"], r["fail_era5"],
            ])
            w.writerow([
                r["station"], r["n_years"], r["fail_years"],
                r["n_bad_tiles"], r["fail_bad_tiles"],
                "|".join(r["missing_mods"]), r["fail_modalities"],
                r["soil_file_ok"], r["soil_flag_ok"], r["fail_soil"],
                r["era5_found"], r["era5_expected"],
                "|".join(str(y) for y in r["era5_missing_yrs"]), r["fail_era5"],
                overall,
            ])

    # ── write bad tile manifest ───────────────────────────────────────────────
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text("\n".join(all_bad_tiles) + ("\n" if all_bad_tiles else ""))

    # ── build report ─────────────────────────────────────────────────────────
    n = len(results)

    def _failing(key):
        return [r["station"] for r in results if r[key]]

    f1 = _failing("fail_years")
    f2 = _failing("fail_bad_tiles")
    f3 = _failing("fail_modalities")
    f4 = _failing("fail_soil")
    f5 = _failing("fail_era5")

    # ERA5 partial gaps (found > 0 but not complete)
    era5_partial = [r for r in results
                    if not r["fail_era5"] and r["era5_found"] < r["era5_expected"]]

    lines = [
        "=" * 64,
        "FINAL PRE-TOKENISATION QA REPORT",
        "=" * 64,
        f"  Stations checked : {n:,}",
        "",
        f"  CHECK 1 — Year coverage (≥3 years with ≥1 S1 or S2 tile)",
        f"    Failing : {len(f1)}",
        "",
        f"  CHECK 2 — Bad tiles (>50% patches with >1% nodata)",
        f"    Stations with bad tiles : {len(f2)}",
        f"    Total bad tiles         : {len(all_bad_tiles)}",
        "",
        f"  CHECK 3 — Satellite modality completeness",
        f"    Stations missing ≥1 modality : {len(f3)}",
        "",
        f"  CHECK 4 — Soil completeness",
        f"    Failing (missing file or soil_patch_ok=False) : {len(f4)}",
        "",
        f"  CHECK 5 — ERA5-Land completeness",
        f"    Stations with 0 ERA5 files  : {len(f5)}",
        f"    Stations with partial gaps  : {len(era5_partial)}",
        "",
        f"  OVERALL PASS : {sum(1 for r in results if not any([r['fail_years'], r['fail_bad_tiles'], r['fail_modalities'], r['fail_soil'], r['fail_era5']]))} / {n}",
        "=" * 64,
    ]

    def _section(title, items, detail_fn=None):
        lines.append(f"\n{title} ({len(items)}):")
        for s in sorted(items):
            suffix = f"  [{detail_fn(s)}]" if detail_fn else ""
            lines.append(f"  {s}{suffix}")

    if f1:
        lookup = {r["station"]: r["n_years"] for r in results}
        _section("CHECK 1 — Stations with <3 active years", f1,
                 lambda s: f"{lookup[s]} years")

    if f2:
        lookup = {r["station"]: r["n_bad_tiles"] for r in results}
        _section("CHECK 2 — Stations with bad tiles", f2,
                 lambda s: f"{lookup[s]} bad tiles")

    if f3:
        lookup = {r["station"]: "|".join(r["missing_mods"]) for r in results}
        _section("CHECK 3 — Stations missing modalities", f3,
                 lambda s: lookup[s])

    if f4:
        lookup = {r["station"]: ("no file" if not r["soil_file_ok"] else "flag=False")
                  for r in results}
        _section("CHECK 4 — Stations failing soil check", f4,
                 lambda s: lookup[s])

    if f5:
        _section("CHECK 5 — Stations with 0 ERA5 files", f5)

    if era5_partial:
        lines.append(f"\nCHECK 5 — Stations with partial ERA5 gaps ({len(era5_partial)}):")
        for r in sorted(era5_partial, key=lambda x: x["station"]):
            missing_str = ",".join(str(y) for y in r["era5_missing_yrs"][:5])
            if len(r["era5_missing_yrs"]) > 5:
                missing_str += f"… (+{len(r['era5_missing_yrs'])-5} more)"
            lines.append(f"  {r['station']}  found={r['era5_found']}/{r['era5_expected']}  missing={missing_str}")

    lines.append("")
    report = "\n".join(lines)

    print("\n" + report)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(report)

    print(f"\n  CSV      → {OUT_CSV}")
    print(f"  Manifest → {OUT_MANIFEST}  ({len(all_bad_tiles)} bad tiles)")
    print(f"  Report   → {OUT_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
