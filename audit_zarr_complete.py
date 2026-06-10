"""
Comprehensive zarr modality + temporal coverage audit.

For every station's zarr store (ZARR_ROOT/{category}/{dir_name}/), check:
  1. Modality existence — s2, s1_asc, s1_desc, cm, dem, lulc, soil, era5,
     sif, twsa, labels/sm, labels/le (per-category requirements)
  2. Cloud-mask <-> S2 date alignment — fraction of S2 acquisition dates
     that have a matching cloud-mask entry
  3. Per-year coverage within [start_date, end_date] — for each modality
     with a date array, flag any year in the station's range with zero
     entries

Output:
  csvs/audit_zarr_complete.csv          — per-station detail
  text/audit_zarr_complete_summary.txt  — aggregate summary (also printed)

Usage:
    python audit_zarr_complete.py [--workers 64]
"""

import argparse
import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_zarr_complete.csv")
OUT_SUMM   = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/audit_zarr_complete_summary.txt")

LAYERS = ["l12", "l9", "l6", "l3"]

CM_COVERAGE_THRESHOLD = 95.0  # pct

# A "missing" satellite year at the very start/end of a station's record is
# only a real gap if the record actually overlaps that year by more than a
# few days — otherwise it's just a 1-4 day sliver where a Sentinel pass was
# never going to land (S2 ~5-day, S1 ~6-12 day revisit).
MIN_YEAR_OVERLAP_DAYS = 7


def _has_token_group(zg, key: str) -> bool:
    return all(f"{key}/{l}" in zg for l in LAYERS) and f"{key}/dates" in zg


def _years_from_date_strs(dates: np.ndarray) -> set:
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


def audit_station(row: dict) -> dict:
    cat      = row["cat"]
    dir_name = row["dir_name"]
    split    = row["split"]
    start_year = int(row["start_date"]) // 10000
    end_year   = int(row["end_date"])   // 10000

    needs_sm = cat in ("sm_only", "sm_and_flux")
    needs_le = cat in ("flux_only", "sm_and_flux")

    result = {
        "station": dir_name, "category": cat, "split": split,
        "start_date": row["start_date"], "end_date": row["end_date"],
        "complete": False,
        "has_s2": False, "has_s1_asc": False, "has_s1_desc": False,
        "has_cm": False, "has_dem": False, "has_lulc": False, "has_soil": False,
        "has_era5": False, "has_sif": False, "has_twsa": False,
        "has_labels_sm": False, "has_labels_le": False,
        "n_s2_dates": 0, "n_cm_dates": 0, "n_s2_missing_cm": 0, "cm_coverage_pct": 0.0,
        "s2_missing_years": "", "s1_asc_missing_years": "", "s1_desc_missing_years": "",
        "cm_missing_years": "", "satellite_missing_years": "", "era5_missing_years": "",
        "labels_sm_missing_years": "", "labels_le_missing_years": "",
        "sif_missing_years": "", "twsa_missing_years": "",
        "status": "OK", "flags": "",
    }

    flags = []
    path = ZARR_ROOT / cat / dir_name

    if not (path / ".complete").exists():
        result["flags"]  = "NOT_COMPLETE"
        result["status"] = "CRITICAL"
        return result
    result["complete"] = True

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
    s2_dates = cm_dates = None
    if result["has_s2"]:
        s2_dates = zg["s2/dates"][:]
        result["n_s2_dates"] = len(s2_dates)
    if result["has_cm"]:
        cm_dates = zg["cm/dates"][:]
        result["n_cm_dates"] = len(cm_dates)

    if s2_dates is not None and cm_dates is not None:
        s2_set = {str(d) for d in s2_dates}
        cm_set = {str(d) for d in cm_dates}
        missing_cm = s2_set - cm_set
        result["n_s2_missing_cm"] = len(missing_cm)
        result["cm_coverage_pct"] = round(
            100 * (len(s2_set) - len(missing_cm)) / len(s2_set), 2
        ) if s2_set else 0.0
        if result["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD:
            flags.append(f"CM_COVERAGE_LOW({result['cm_coverage_pct']}%)")

    # ── 3. Per-year coverage within [start_date, end_date] ──────────────────
    def _check_years(key, dates_arr, is_int=False):
        years = _years_from_date_ints(dates_arr) if is_int else _years_from_date_strs(dates_arr)
        return _missing_years(years, start_year, end_year)

    s2_years = _years_from_date_strs(s2_dates) if result["has_s2"] else set()
    s1_asc_years = _years_from_date_strs(zg["s1_asc/dates"][:]) if result["has_s1_asc"] else set()
    s1_desc_years = _years_from_date_strs(zg["s1_desc/dates"][:]) if result["has_s1_desc"] else set()

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
        result["cm_missing_years"] = _check_years("cm", cm_dates)
        if result["cm_missing_years"]:
            flags.append(f"INFO_cm_missing_years({result['cm_missing_years']})")

    # A year only counts as missing if NONE of s2/s1_asc/s1_desc cover it —
    # at least one satellite modality per year is sufficient for training —
    # and the station's record overlaps that year by more than a few days
    # (see MIN_YEAR_OVERLAP_DAYS).
    satellite_years = s2_years | s1_asc_years | s1_desc_years
    result["satellite_missing_years"] = _missing_years_min_overlap(
        satellite_years, start_year, end_year,
        int(row["start_date"]), int(row["end_date"]), MIN_YEAR_OVERLAP_DAYS,
    )
    if result["satellite_missing_years"]:
        flags.append(f"satellite_missing_years({result['satellite_missing_years']})")

    if result["has_era5"]:
        result["era5_missing_years"] = _check_years("era5", zg["era5/date_ints"][:], is_int=True)
        if result["era5_missing_years"]:
            flags.append(f"era5_missing_years({result['era5_missing_years']})")

    if result["has_labels_sm"]:
        result["labels_sm_missing_years"] = _check_years("labels_sm", zg["labels/dates"][:])
        if result["labels_sm_missing_years"]:
            flags.append(f"labels_sm_missing_years({result['labels_sm_missing_years']})")

    if result["has_labels_le"]:
        result["labels_le_missing_years"] = _check_years("labels_le", zg["labels/dates_flux"][:])
        if result["labels_le_missing_years"]:
            flags.append(f"labels_le_missing_years({result['labels_le_missing_years']})")

    if result["has_sif"]:
        result["sif_missing_years"] = _check_years("sif", zg["sif/date_ints"][:], is_int=True)
        if result["sif_missing_years"]:
            flags.append(f"INFO_sif_missing_years({result['sif_missing_years']})")

    if result["has_twsa"]:
        result["twsa_missing_years"] = _check_years("twsa", zg["twsa/date_ints"][:], is_int=True)
        if result["twsa_missing_years"]:
            flags.append(f"INFO_twsa_missing_years({result['twsa_missing_years']})")

    # ── Status rollup ────────────────────────────────────────────────────────
    required_year_gap_flags = [
        f for f in flags
        if "_missing_years(" in f and not f.startswith("INFO_")
    ]

    if critical_so_far:
        result["status"] = "CRITICAL"
    elif result["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD or required_year_gap_flags:
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    result["flags"] = "|".join(flags)
    return result


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
    lines.append(f"Comprehensive zarr audit — {len(out_df)} stations")
    lines.append("=" * 60)

    for status in ["OK", "WARN", "CRITICAL"]:
        n = (out_df["status"] == status).sum()
        lines.append(f"  {status:<10}: {n}")
    lines.append("")

    lines.append("Modality presence:")
    for col in ["has_s2", "has_s1_asc", "has_s1_desc", "has_cm", "has_dem", "has_lulc",
                "has_soil", "has_era5", "has_sif", "has_twsa",
                "has_labels_sm", "has_labels_le"]:
        n_ok = out_df[col].sum()
        lines.append(f"  {col:<16}: {n_ok}/{len(out_df)}")
    lines.append("")

    # Cloud mask coverage
    has_s2 = out_df[out_df["has_s2"]]
    lines.append(f"Cloud mask coverage (stations with S2, n={len(has_s2)}):")
    lines.append(f"  Mean cm coverage : {has_s2['cm_coverage_pct'].mean():.2f}%")
    lines.append(f"  Min  cm coverage : {has_s2['cm_coverage_pct'].min():.2f}%")
    n_low = (has_s2["cm_coverage_pct"] < CM_COVERAGE_THRESHOLD).sum()
    lines.append(f"  Stations < {CM_COVERAGE_THRESHOLD}% cm coverage: {n_low}")
    lines.append("")

    # Per-year gaps (required modalities)
    lines.append("Stations with missing-year gaps (required modalities):")
    for col in ["satellite_missing_years", "era5_missing_years",
                "labels_sm_missing_years", "labels_le_missing_years"]:
        n = (out_df[col] != "").sum()
        if n:
            lines.append(f"  {col:<24}: {n} stations")
    lines.append("")

    # Per-modality satellite gaps (informational only — a year is only a real
    # gap if NONE of s2/s1_asc/s1_desc cover it; see satellite_missing_years)
    lines.append("Per-modality satellite gaps (informational only):")
    for col in ["s2_missing_years", "s1_asc_missing_years", "s1_desc_missing_years", "cm_missing_years"]:
        n = (out_df[col] != "").sum()
        if n:
            lines.append(f"  {col:<24}: {n} stations")
    lines.append("")

    # SIF/TWSA informational
    lines.append("SIF/TWSA (informational only):")
    lines.append(f"  has_sif  : {out_df['has_sif'].sum()}/{len(out_df)}")
    lines.append(f"  has_twsa : {out_df['has_twsa'].sum()}/{len(out_df)}")
    n_sif_gap  = (out_df["sif_missing_years"]  != "").sum()
    n_twsa_gap = (out_df["twsa_missing_years"] != "").sum()
    lines.append(f"  sif_missing_years gaps  : {n_sif_gap} stations")
    lines.append(f"  twsa_missing_years gaps : {n_twsa_gap} stations")
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
