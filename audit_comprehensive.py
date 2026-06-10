"""
Comprehensive per-station dataset audit — single entry point covering:

  1. Modality existence — s2, s1_asc, s1_desc, cm, dem, lulc, soil, era5,
     sif, twsa, labels/sm, labels/le (per-category requirements)
  2. Cloud-mask <-> S2 date alignment — fraction of S2 acquisition dates
     that have a matching cloud-mask entry
  3. Per-year coverage within [start_date, end_date] for every dated
     modality (informational unless it strands a *required* modality)
  4. Token vs raw-image alignment — for s2/s1_asc/s1_desc, compare token
     date sets against the raw `satellite_zarr` date sets to catch dates
     that were tokenized but no longer have a source image (or vice versa);
     for dem/lulc, confirm the raw source image exists too
  5. Tabular/static sanity — NaN checks on dem, lulc, soil, era5, sif,
     twsa, labels/sm (these must always be fully clean); labels/le NaN
     fraction is reported informationally (flux gaps are expected)
  6. Label continuity — longest run of consecutive years of label
     coverage ending at the station's end_date; flagged if below
     MIN_TRAIN_YEARS for stations that need sm/le labels

Output:
  csvs/audit_comprehensive.csv          — per-station detail
  text/audit_comprehensive_summary.txt  — aggregate summary (also printed)

Usage:
    conda run -n sensei python audit_comprehensive.py [--workers 64]
"""

import argparse
import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ROOT   = Path("/projects/prjs1968/satellite_zarr")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_comprehensive.csv")
OUT_SUMM   = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/audit_comprehensive_summary.txt")

LAYERS = ["l12", "l9", "l6", "l3"]

CM_COVERAGE_THRESHOLD = 95.0  # pct
MIN_TRAIN_YEARS = 3

# dataset.py::fill_soil_nans() nearest-neighbour-fills any amount of NaN in the
# soil patch at load time, so raw NaN pixels are NOT a training blocker by
# themselves. Stations with soil_nan_pct >= this threshold were already
# excluded via soil_patch_ok=False (see csvs/excluded_stations.csv, range
# 0.58-1.00) -- anything still active above this would mean the exclusion
# step missed one.
SOIL_NAN_PCT_CRITICAL = 0.5

# A "missing" satellite year at the very start/end of a station's record is
# only a real gap if the record actually overlaps that year by more than a
# few days — otherwise it's just a 1-4 day sliver where a Sentinel pass was
# never going to land (S2 ~5-day, S1 ~6-12 day revisit).
MIN_YEAR_OVERLAP_DAYS = 7


# ── helpers ──────────────────────────────────────────────────────────────────

def _has_token_group(zg, key: str) -> bool:
    return all(f"{key}/{l}" in zg for l in LAYERS) and f"{key}/dates" in zg


def _date_set(zg, key: str):
    """Return a set of 'YYYYMMDD' strings for zg[f'{key}/dates'], or None."""
    if f"{key}/dates" not in zg:
        return None
    return {(d.decode() if isinstance(d, bytes) else str(d)) for d in zg[f"{key}/dates"][:]}


def _years_from_date_strs(dates) -> set:
    return {int(str(d)[:4]) for d in dates}


def _years_from_date_ints(date_ints: np.ndarray) -> set:
    return {int(d) // 10000 for d in date_ints}


def _missing_years(years_present: set, start_year: int, end_year: int) -> str:
    missing = [y for y in range(start_year, end_year + 1) if y not in years_present]
    return ",".join(str(y) for y in missing)


def _date_from_int(d: int) -> datetime.date:
    return datetime.date(d // 10000, (d // 100) % 100, d % 100)


def _year_overlap_days(year: int, start_date: int, end_date: int) -> int:
    lo = max(datetime.date(year, 1, 1), _date_from_int(start_date))
    hi = min(datetime.date(year, 12, 31), _date_from_int(end_date))
    return (hi - lo).days + 1 if hi >= lo else 0


def _missing_years_min_overlap(years_present: set, start_year: int, end_year: int,
                                start_date: int, end_date: int, min_days: int) -> str:
    missing = [
        y for y in range(start_year, end_year + 1)
        if y not in years_present and _year_overlap_days(y, start_date, end_date) >= min_days
    ]
    return ",".join(str(y) for y in missing)


def _max_continuous_years_ending(years_present: set, start_year: int, end_year: int) -> int:
    """Length of the run of consecutive years present, counting back from end_year."""
    count = 0
    y = end_year
    while y >= start_year and y in years_present:
        count += 1
        y -= 1
    return count


def _has_nan(arr) -> bool:
    a = np.asarray(arr)
    return bool(np.isnan(a).any())


# ── main per-station audit ──────────────────────────────────────────────────

def audit_station(row: dict) -> dict:
    cat      = row["cat"]
    dir_name = row["dir_name"]
    split    = row["split"]
    start_date = int(row["start_date"])
    end_date   = int(row["end_date"])
    start_year = start_date // 10000
    end_year   = end_date   // 10000

    needs_sm = cat in ("sm_only", "sm_and_flux")
    needs_le = cat in ("flux_only", "sm_and_flux")

    result = {
        "station": dir_name, "category": cat, "split": split,
        "start_date": row["start_date"], "end_date": row["end_date"],
        "status": "OK", "flags": "",

        # 1. modality existence
        "has_s2": False, "has_s1_asc": False, "has_s1_desc": False,
        "has_cm": False, "has_dem": False, "has_lulc": False, "has_soil": False,
        "has_era5": False, "has_sif": False, "has_twsa": False,
        "has_labels_sm": False, "has_labels_le": False,

        # 2. cm <-> s2 alignment
        "n_s2_dates": 0, "n_cm_dates": 0, "n_s2_missing_cm": 0, "cm_coverage_pct": 0.0,

        # 3. per-year coverage
        "s2_missing_years": "", "s1_asc_missing_years": "", "s1_desc_missing_years": "",
        "cm_missing_years": "", "satellite_missing_years": "", "era5_missing_years": "",
        "labels_sm_missing_years": "", "labels_le_missing_years": "",
        "sif_missing_years": "", "twsa_missing_years": "",

        # 4. token vs raw alignment
        "has_raw_satzarr": False,
        "n_s2_raw": None, "s2_token_minus_raw": None, "s2_raw_minus_token": None,
        "n_s1_asc_raw": None, "s1_asc_token_minus_raw": None, "s1_asc_raw_minus_token": None,
        "n_s1_desc_raw": None, "s1_desc_token_minus_raw": None, "s1_desc_raw_minus_token": None,
        "has_raw_dem": None, "has_raw_lulc": None,

        # 5. tabular/static sanity
        "dem_nan": False, "lulc_nan": False, "soil_nan_pct": 0.0,
        "era5_nan": False, "sif_nan": False, "twsa_nan": False,
        "labels_sm_nan": False, "labels_le_nan_frac": None,

        # 6. label continuity
        "sm_continuous_years": None, "le_continuous_years": None,
        "lt_3yr_labels": False,
    }

    flags = []
    path = ZARR_ROOT / cat / dir_name

    if not (path / ".complete").exists():
        result["flags"]  = "NOT_COMPLETE"
        result["status"] = "CRITICAL"
        return result

    try:
        zg = zarr.open_consolidated(str(path), mode="r")
    except KeyError:
        try:
            zg = zarr.open_group(str(path), mode="r")
        except Exception as e:
            result["flags"]  = f"OPEN_ERROR({e})"
            result["status"] = "CRITICAL"
            return result
    except Exception as e:
        result["flags"]  = f"OPEN_ERROR({e})"
        result["status"] = "CRITICAL"
        return result

    # ── 1. Existence ─────────────────────────────────────────────────────────
    result["has_s2"]      = _has_token_group(zg, "s2")
    result["has_s1_asc"]  = _has_token_group(zg, "s1_asc")
    result["has_s1_desc"] = _has_token_group(zg, "s1_desc")
    result["has_cm"]      = "cm/masks" in zg and "cm/dates" in zg
    result["has_dem"]     = "dem"  in zg
    result["has_lulc"]    = "lulc" in zg
    result["has_soil"]    = "soil" in zg
    result["has_era5"]    = all(f"era5/{k}" in zg for k in ("values", "date_ints", "doys"))
    result["has_sif"]     = all(f"sif/{k}"  in zg for k in ("values", "date_ints", "doys"))
    result["has_twsa"]    = all(f"twsa/{k}" in zg for k in ("lwe", "date_ints", "doys"))
    result["has_labels_sm"] = all(f"labels/{k}" in zg for k in ("sm", "depths", "dates"))
    result["has_labels_le"] = all(f"labels/{k}" in zg for k in ("le", "le_qc", "dates_flux"))

    if not result["has_s2"]:               flags.append("MISSING_s2")
    if not (result["has_s1_asc"] or result["has_s1_desc"]):
        flags.append("MISSING_s1 (both asc & desc)")
    if not result["has_cm"]:               flags.append("MISSING_cm")
    if not result["has_dem"]:              flags.append("MISSING_dem")
    if not result["has_lulc"]:             flags.append("MISSING_lulc")
    if not result["has_soil"]:             flags.append("MISSING_soil")
    if not result["has_era5"]:             flags.append("MISSING_era5")
    if needs_sm and not result["has_labels_sm"]: flags.append("MISSING_labels_sm")
    if needs_le and not result["has_labels_le"]: flags.append("MISSING_labels_le")
    if not result["has_sif"]:              flags.append("INFO_no_sif")
    if not result["has_twsa"]:             flags.append("INFO_no_twsa")

    critical_so_far = [f for f in flags if not f.startswith("INFO_")]

    # ── 2. Cloud mask <-> S2 date alignment ─────────────────────────────────
    s2_dates = _date_set(zg, "s2") if result["has_s2"] else None
    cm_dates = _date_set(zg, "cm") if result["has_cm"] else None
    if s2_dates is not None:
        result["n_s2_dates"] = len(s2_dates)
    if cm_dates is not None:
        result["n_cm_dates"] = len(cm_dates)

    if s2_dates is not None and cm_dates is not None:
        missing_cm = s2_dates - cm_dates
        result["n_s2_missing_cm"] = len(missing_cm)
        result["cm_coverage_pct"] = round(
            100 * (len(s2_dates) - len(missing_cm)) / len(s2_dates), 2
        ) if s2_dates else 0.0
        if result["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD:
            flags.append(f"CM_COVERAGE_LOW({result['cm_coverage_pct']}%)")

    # ── 3. Per-year coverage within [start_date, end_date] ──────────────────
    def _check_years(dates_arr, is_int=False):
        years = _years_from_date_ints(dates_arr) if is_int else _years_from_date_strs(dates_arr)
        return _missing_years(years, start_year, end_year)

    s1_asc_dates  = _date_set(zg, "s1_asc")  if result["has_s1_asc"]  else None
    s1_desc_dates = _date_set(zg, "s1_desc") if result["has_s1_desc"] else None

    s2_years      = _years_from_date_strs(s2_dates)      if s2_dates      else set()
    s1_asc_years  = _years_from_date_strs(s1_asc_dates)  if s1_asc_dates  else set()
    s1_desc_years = _years_from_date_strs(s1_desc_dates) if s1_desc_dates else set()

    if result["has_s2"]:
        result["s2_missing_years"] = _missing_years(s2_years, start_year, end_year)
        if result["s2_missing_years"]:
            flags.append(f"INFO_s2_missing_years({result['s2_missing_years']})")

    if result["has_s1_asc"]:
        result["s1_asc_missing_years"] = _missing_years(s1_asc_years, start_year, end_year)
        if result["s1_asc_missing_years"]:
            flags.append(f"INFO_s1_asc_missing_years({result['s1_asc_missing_years']})")

    if result["has_s1_desc"]:
        result["s1_desc_missing_years"] = _missing_years(s1_desc_years, start_year, end_year)
        if result["s1_desc_missing_years"]:
            flags.append(f"INFO_s1_desc_missing_years({result['s1_desc_missing_years']})")

    if result["has_cm"]:
        result["cm_missing_years"] = _check_years(zg["cm/dates"][:])
        if result["cm_missing_years"]:
            flags.append(f"INFO_cm_missing_years({result['cm_missing_years']})")

    # A year only counts as missing if NONE of s2/s1_asc/s1_desc cover it —
    # at least one satellite modality per year is sufficient for training —
    # and the station's record overlaps that year by more than a few days
    # (see MIN_YEAR_OVERLAP_DAYS).
    satellite_years = s2_years | s1_asc_years | s1_desc_years
    result["satellite_missing_years"] = _missing_years_min_overlap(
        satellite_years, start_year, end_year, start_date, end_date, MIN_YEAR_OVERLAP_DAYS,
    )
    if result["satellite_missing_years"]:
        flags.append(f"satellite_missing_years({result['satellite_missing_years']})")

    if result["has_era5"]:
        result["era5_missing_years"] = _check_years(zg["era5/date_ints"][:], is_int=True)
        if result["era5_missing_years"]:
            flags.append(f"era5_missing_years({result['era5_missing_years']})")

    if result["has_labels_sm"]:
        result["labels_sm_missing_years"] = _check_years(zg["labels/dates"][:])
        if result["labels_sm_missing_years"]:
            flags.append(f"labels_sm_missing_years({result['labels_sm_missing_years']})")

    if result["has_labels_le"]:
        result["labels_le_missing_years"] = _check_years(zg["labels/dates_flux"][:])
        if result["labels_le_missing_years"]:
            flags.append(f"labels_le_missing_years({result['labels_le_missing_years']})")

    if result["has_sif"]:
        result["sif_missing_years"] = _check_years(zg["sif/date_ints"][:], is_int=True)
        if result["sif_missing_years"]:
            flags.append(f"INFO_sif_missing_years({result['sif_missing_years']})")

    if result["has_twsa"]:
        result["twsa_missing_years"] = _check_years(zg["twsa/date_ints"][:], is_int=True)
        if result["twsa_missing_years"]:
            flags.append(f"INFO_twsa_missing_years({result['twsa_missing_years']})")

    # ── 4. Token vs raw-image alignment ─────────────────────────────────────
    raw_path = SAT_ROOT / f"{dir_name}.zarr"
    if raw_path.exists():
        result["has_raw_satzarr"] = True
        try:
            rzg = zarr.open_group(str(raw_path), mode="r")
        except Exception as e:
            flags.append(f"RAW_OPEN_ERROR({e})")
            rzg = None

        if rzg is not None:
            for mod, tdates in [("s2", s2_dates), ("s1_asc", s1_asc_dates), ("s1_desc", s1_desc_dates)]:
                rdates = _date_set(rzg, mod)
                result[f"n_{mod}_raw"] = len(rdates) if rdates is not None else None
                if tdates is not None and rdates is not None:
                    tmr = len(tdates - rdates)
                    rmt = len(rdates - tdates)
                    result[f"{mod}_token_minus_raw"] = tmr
                    result[f"{mod}_raw_minus_token"] = rmt
                    if tmr:
                        flags.append(f"{mod.upper()}_TOKEN_MINUS_RAW({tmr})")
                    if rmt:
                        flags.append(f"{mod.upper()}_RAW_MINUS_TOKEN({rmt})")

            result["has_raw_dem"]  = "dem/data"  in rzg
            result["has_raw_lulc"] = "lulc/data" in rzg
            if result["has_dem"] and not result["has_raw_dem"]:
                flags.append("DEM_TOKEN_NO_RAW")
            if result["has_lulc"] and not result["has_raw_lulc"]:
                flags.append("LULC_TOKEN_NO_RAW")
    else:
        flags.append("INFO_no_raw_satzarr")

    # ── 5. Tabular/static sanity (NaN checks) ───────────────────────────────
    if result["has_dem"]:
        result["dem_nan"] = _has_nan(zg["dem"][:])
        if result["dem_nan"]: flags.append("DEM_HAS_NAN")
    if result["has_lulc"]:
        result["lulc_nan"] = _has_nan(zg["lulc"][:])
        if result["lulc_nan"]: flags.append("LULC_HAS_NAN")
    if result["has_soil"]:
        soil = np.asarray(zg["soil"][:])
        # NaN mask is identical across channels (shared spatial nodata mask
        # from OpenLandMap); dataset.py::fill_soil_nans() nearest-neighbour
        # fills these at load time, so report the fraction informationally
        # unless it crosses the soil_patch_ok exclusion threshold.
        pct = float(np.isnan(soil[0]).mean()) if soil.shape[0] else 0.0
        result["soil_nan_pct"] = round(pct, 4)
        if pct > SOIL_NAN_PCT_CRITICAL:
            flags.append(f"SOIL_NAN_HIGH({result['soil_nan_pct']:.1%})")
        elif pct > 0:
            flags.append(f"INFO_soil_nan_pct({result['soil_nan_pct']:.1%})")
    if result["has_era5"]:
        result["era5_nan"] = _has_nan(zg["era5/values"][:])
        if result["era5_nan"]: flags.append("ERA5_HAS_NAN")
    if result["has_sif"]:
        result["sif_nan"] = _has_nan(zg["sif/values"][:])
        if result["sif_nan"]: flags.append("SIF_HAS_NAN")
    if result["has_twsa"]:
        result["twsa_nan"] = _has_nan(zg["twsa/lwe"][:])
        if result["twsa_nan"]: flags.append("TWSA_HAS_NAN")
    if result["has_labels_sm"]:
        result["labels_sm_nan"] = _has_nan(zg["labels/sm"][:])
        if result["labels_sm_nan"]: flags.append("LABELS_SM_HAS_NAN")
    if result["has_labels_le"]:
        le = np.asarray(zg["labels/le"][:])
        result["labels_le_nan_frac"] = round(float(np.isnan(le).mean()), 4) if len(le) else None

    # ── 6. Label continuity (>= MIN_TRAIN_YEARS ending at end_date) ─────────
    if result["has_labels_sm"]:
        sm_years = _years_from_date_strs(zg["labels/dates"][:])
        result["sm_continuous_years"] = _max_continuous_years_ending(sm_years, start_year, end_year)
        if needs_sm and result["sm_continuous_years"] < MIN_TRAIN_YEARS:
            flags.append(f"LT_{MIN_TRAIN_YEARS}YR_SM({result['sm_continuous_years']})")
            result["lt_3yr_labels"] = True

    if result["has_labels_le"]:
        le_years = _years_from_date_strs(zg["labels/dates_flux"][:])
        result["le_continuous_years"] = _max_continuous_years_ending(le_years, start_year, end_year)
        if needs_le and result["le_continuous_years"] < MIN_TRAIN_YEARS:
            flags.append(f"LT_{MIN_TRAIN_YEARS}YR_LE({result['le_continuous_years']})")
            result["lt_3yr_labels"] = True

    # ── Status rollup ────────────────────────────────────────────────────────
    required_year_gap_flags = [
        f for f in flags
        if "_missing_years(" in f and not f.startswith("INFO_")
    ]
    nan_flags = [f for f in flags if f.endswith("_HAS_NAN") or f.startswith("SOIL_NAN_HIGH")]
    raw_align_flags = [
        f for f in flags
        if "_TOKEN_MINUS_RAW(" in f or "_RAW_MINUS_TOKEN(" in f
        or f in ("DEM_TOKEN_NO_RAW", "LULC_TOKEN_NO_RAW")
    ]
    lt3yr_flags = [f for f in flags if f.startswith("LT_3YR")]

    if critical_so_far or required_year_gap_flags or nan_flags:
        result["status"] = "CRITICAL"
    elif (result["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD
          or raw_align_flags or lt3yr_flags):
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    result["flags"] = "|".join(flags)
    return result


# ── driver ───────────────────────────────────────────────────────────────────

def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["cat"]      = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)

    print(f"Auditing {len(df)} stations with {args.workers} workers...")

    rows = df[["cat", "dir_name", "split", "start_date", "end_date"]].to_dict("records")

    with Pool(args.workers) as pool:
        results = pool.map(audit_station, rows)

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"Comprehensive dataset audit — {len(out_df)} stations")
    lines.append("=" * 60)

    for status in ["OK", "WARN", "CRITICAL"]:
        n = (out_df["status"] == status).sum()
        lines.append(f"  {status:<10}: {n}")
    lines.append("")

    lines.append("1. Modality presence:")
    for col in ["has_s2", "has_s1_asc", "has_s1_desc", "has_cm", "has_dem", "has_lulc",
                "has_soil", "has_era5", "has_sif", "has_twsa",
                "has_labels_sm", "has_labels_le"]:
        n_ok = out_df[col].sum()
        lines.append(f"  {col:<16}: {n_ok}/{len(out_df)}")
    lines.append("")

    # Cloud mask coverage
    has_s2 = out_df[out_df["has_s2"]]
    lines.append(f"2. Cloud mask coverage (stations with S2, n={len(has_s2)}):")
    lines.append(f"  Mean cm coverage : {has_s2['cm_coverage_pct'].mean():.2f}%")
    lines.append(f"  Min  cm coverage : {has_s2['cm_coverage_pct'].min():.2f}%")
    n_low = (has_s2["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD).sum()
    lines.append(f"  Stations < {CM_COVERAGE_THRESHOLD}% cm coverage: {n_low}")
    lines.append("")

    # Per-year gaps (required modalities)
    lines.append("3. Stations with missing-year gaps (required modalities):")
    for col in ["satellite_missing_years", "era5_missing_years",
                "labels_sm_missing_years", "labels_le_missing_years"]:
        n = (out_df[col] != "").sum()
        if n:
            lines.append(f"  {col:<24}: {n} stations")
    lines.append("")

    lines.append("   Per-modality satellite gaps (informational only):")
    for col in ["s2_missing_years", "s1_asc_missing_years", "s1_desc_missing_years", "cm_missing_years"]:
        n = (out_df[col] != "").sum()
        if n:
            lines.append(f"     {col:<24}: {n} stations")
    lines.append("")

    lines.append("   SIF/TWSA (informational only):")
    lines.append(f"     has_sif  : {out_df['has_sif'].sum()}/{len(out_df)}")
    lines.append(f"     has_twsa : {out_df['has_twsa'].sum()}/{len(out_df)}")
    n_sif_gap  = (out_df["sif_missing_years"]  != "").sum()
    n_twsa_gap = (out_df["twsa_missing_years"] != "").sum()
    lines.append(f"     sif_missing_years gaps  : {n_sif_gap} stations")
    lines.append(f"     twsa_missing_years gaps : {n_twsa_gap} stations")
    lines.append("")

    # Token vs raw alignment
    lines.append(f"4. Token vs raw-image alignment (has_raw_satzarr: "
                  f"{out_df['has_raw_satzarr'].sum()}/{len(out_df)}):")
    for m in ["s2", "s1_asc", "s1_desc"]:
        tmr_col, rmt_col = f"{m}_token_minus_raw", f"{m}_raw_minus_token"
        if tmr_col in out_df:
            n_tmr = (out_df[tmr_col].fillna(0) > 0).sum()
            n_rmt = (out_df[rmt_col].fillna(0) > 0).sum()
            lines.append(f"  {m:<8}: token>raw on {n_tmr} stations, raw>token (untokenized images) on {n_rmt} stations")
    n_dem_norraw  = (out_df["has_raw_dem"]  == False).sum() if "has_raw_dem"  in out_df else 0
    n_lulc_norraw = (out_df["has_raw_lulc"] == False).sum() if "has_raw_lulc" in out_df else 0
    lines.append(f"  dem     : token without matching raw image: {(out_df['flags'].str.contains('DEM_TOKEN_NO_RAW')).sum()} stations")
    lines.append(f"  lulc    : token without matching raw image: {(out_df['flags'].str.contains('LULC_TOKEN_NO_RAW')).sum()} stations")
    lines.append("")

    # NaN sanity
    lines.append("5. Tabular/static NaN sanity (should all be 0, except soil):")
    for col in ["dem_nan", "lulc_nan", "era5_nan", "sif_nan", "twsa_nan", "labels_sm_nan"]:
        n = out_df[col].sum()
        lines.append(f"  {col:<16}: {n} stations with NaN")
    soil_pct = out_df["soil_nan_pct"]
    n_soil_any  = (soil_pct > 0).sum()
    n_soil_high = (soil_pct > SOIL_NAN_PCT_CRITICAL).sum()
    lines.append(f"  soil_nan_pct    : {n_soil_any} stations with some NaN "
                  f"(nearest-neighbour-filled at load time, see dataset.py::fill_soil_nans); "
                  f"max={soil_pct.max():.1%}")
    lines.append(f"                    {n_soil_high} stations > {SOIL_NAN_PCT_CRITICAL:.0%} "
                  f"(should already be excluded via soil_patch_ok=False)")
    le_frac = out_df["labels_le_nan_frac"].dropna()
    if len(le_frac):
        lines.append(f"  labels_le_nan_frac (informational, flux gaps expected): "
                      f"mean={le_frac.mean():.2%}, max={le_frac.max():.2%}")
    lines.append("")

    # Label continuity
    lines.append(f"6. Label continuity (>= {MIN_TRAIN_YEARS} yr continuous, ending at end_date):")
    for col, needs_col, name in [
        ("sm_continuous_years", "has_labels_sm", "SM"),
        ("le_continuous_years", "has_labels_le", "LE"),
    ]:
        sub = out_df[out_df[needs_col]]
        n_lt = (sub[col] < MIN_TRAIN_YEARS).sum()
        lines.append(f"  {name}: {n_lt}/{len(sub)} stations with < {MIN_TRAIN_YEARS} continuous years")
    lines.append("")

    # CRITICAL stations
    crit = out_df[out_df["status"] == "CRITICAL"][["station", "category", "split", "flags"]]
    lines.append(f"CRITICAL stations ({len(crit)}):")
    for _, r in crit.iterrows():
        lines.append(f"  [{r['category']}/{r['split']}] {r['station']}: {r['flags']}")
    lines.append("")

    # WARN stations
    warn = out_df[out_df["status"] == "WARN"][["station", "category", "split", "flags"]]
    lines.append(f"WARN stations ({len(warn)}):")
    for _, r in warn.iterrows():
        lines.append(f"  [{r['category']}/{r['split']}] {r['station']}: {r['flags']}")

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_SUMM.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMM.write_text(summary)
    print(f"\nDetailed results → {OUT_CSV}")
    print(f"Summary          → {OUT_SUMM}")


if __name__ == "__main__":
    main()
