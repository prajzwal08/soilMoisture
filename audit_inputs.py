"""
audit_inputs.py
===============
Final pre-training data audit across all 993 active stations in
csvs/station_splits.csv.

For each station checks:
  1. Existence of every required input modality
  2. Date coverage alignment against the station's splits start_date / end_date

Outputs:
  csvs/audit_inputs.csv         — per-station status (one row per station)
  text/audit_inputs_summary.txt — aggregate stats (also printed to stdout)

Usage:
    python audit_inputs.py                # 8 parallel workers
    python audit_inputs.py --workers 16
"""

import argparse
import re
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
CSV_ACTIVE   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUTPUT_CSV   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_inputs.csv")
OUTPUT_SUMM  = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/audit_inputs_summary.txt")

# ── tolerances ────────────────────────────────────────────────────────────────
LABEL_TOL_DAYS     = 30   # label start/end vs splits start/end
CLOUDMASK_TOL_DAYS = 60   # cloudmask range vs S2 range


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s), fmt)
        except (ValueError, TypeError):
            pass
    return None


def _dates_from_stem(stem: str) -> tuple[datetime | None, datetime | None]:
    """Extract start/end from stems ending in _YYYYMMDD_YYYYMMDD."""
    m = re.search(r"_(\d{8})_(\d{8})$", stem)
    if m:
        return _parse_date(m.group(1)), _parse_date(m.group(2))
    return None, None


def _nc_time_range(path: Path) -> tuple[datetime | None, datetime | None]:
    """Read first/last time value from a NetCDF file (minimal I/O).
    Tries 'time' and 'date_time' dimension names (labels.nc uses 'date_time')."""
    try:
        import xarray as xr
        try:
            ds = xr.open_dataset(path, engine="netcdf4", use_cftime=False)
        except Exception:
            ds = xr.open_dataset(path, engine="netcdf4")
        for dim in ("time", "date_time"):
            if dim in ds:
                t = ds[dim].values
                ds.close()
                t0 = pd.Timestamp(t[0]).to_pydatetime().replace(tzinfo=None)
                t1 = pd.Timestamp(t[-1]).to_pydatetime().replace(tzinfo=None)
                return t0, t1
        ds.close()
        return None, None
    except Exception:
        return None, None


def _find_station_dir(station: str) -> tuple[Path | None, str | None]:
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        p = DATA_ROOT / cat / station
        if p.exists():
            return p, cat
    return None, None


def _fmt(d: datetime | None) -> str | None:
    return d.strftime("%Y-%m-%d") if d else None


# ── per-station worker ────────────────────────────────────────────────────────

def audit_station(args: tuple) -> dict:
    station, splits_start, splits_end = args
    base, category = _find_station_dir(station)

    # skeleton for missing dir case
    empty = {k: None for k in [
        "s2_start","s2_end","s1_start","s1_end",
        "era5_start","era5_end","label_start","label_end",
        "twsa_start","twsa_end","sif_start","sif_end",
        "cloudmask_start","cloudmask_end",
        "era5_covers_labels","labels_match_splits",
        "twsa_has_overlap","sif_has_overlap",
    ]}

    if base is None:
        return {
            "station": station, "category": "NOT_FOUND",
            "splits_start": _fmt(splits_start), "splits_end": _fmt(splits_end),
            **{f"has_{m}": False for m in ["S2","S1","DEM","LULC","ERA5","soil","TWSA","SIF","CloudMask","labels"]},
            **empty,
            "flags": "STATION_DIR_NOT_FOUND",
            "status": "MISSING",
        }

    # ── 1. Existence + date extraction ────────────────────────────────────────

    # S2L2A — consolidated: {station}_L12_{start}_{end}.pt
    s2_files = list((base / "S2L2A").glob("*_L12_*.pt")) if (base / "S2L2A").exists() else []
    has_S2 = bool(s2_files)
    s2_start = s2_end = None
    if has_S2:
        s2_start, s2_end = _dates_from_stem(s2_files[0].stem)

    # S1RTC — consolidated: {station}_{orbit}_L12_{start}_{end}.pt
    s1_files = list((base / "S1RTC").glob("*_L12_*.pt")) if (base / "S1RTC").exists() else []
    has_S1 = bool(s1_files)
    s1_start = s1_end = None
    if has_S1:
        s1_start, s1_end = _dates_from_stem(s1_files[0].stem)

    # DEM / LULC (static)
    has_DEM  = (base / "DEM"  / "dem_L12.pt").exists()
    has_LULC = (base / "LULC" / "lulc_L12.pt").exists()

    # ERA5Land — meteo_{start}_{end}.nc
    era5_files = list((base / "ERA5Land").glob("meteo_*.nc")) if (base / "ERA5Land").exists() else []
    has_ERA5 = bool(era5_files)
    era5_start = era5_end = None
    if has_ERA5:
        m = re.search(r"meteo_(\d{8})_(\d{8})\.nc", era5_files[0].name)
        if m:
            era5_start, era5_end = _parse_date(m.group(1)), _parse_date(m.group(2))

    # soil (static)
    has_soil = (base / "soil" / "soil_patch.tif").exists()

    # TWSA — twsa_{start}_{end}.nc
    twsa_files = list((base / "TWSA").glob("twsa_*.nc")) if (base / "TWSA").exists() else []
    has_TWSA = bool(twsa_files)
    twsa_start = twsa_end = None
    if has_TWSA:
        m = re.search(r"twsa_(\d{8})_(\d{8})\.nc", twsa_files[0].name)
        if m:
            twsa_start, twsa_end = _parse_date(m.group(1)), _parse_date(m.group(2))

    # SIF — sif_{start}_{end}.nc
    sif_files = list((base / "SIF").glob("sif_*.nc")) if (base / "SIF").exists() else []
    has_SIF = bool(sif_files)
    sif_start = sif_end = None
    if has_SIF:
        m = re.search(r"sif_(\d{8})_(\d{8})\.nc", sif_files[0].name)
        if m:
            sif_start, sif_end = _parse_date(m.group(1)), _parse_date(m.group(2))

    # CloudMask — {station}_{start}_{end}.pt
    cm_files = list((base / "CloudMask").glob("*.pt")) if (base / "CloudMask").exists() else []
    has_CloudMask = bool(cm_files)
    cm_start = cm_end = None
    if has_CloudMask:
        cm_start, cm_end = _dates_from_stem(cm_files[0].stem)

    # labels.nc — open to read time coordinate
    labels_path = base / "labels.nc"
    has_labels = labels_path.exists()
    label_start = label_end = None
    if has_labels:
        label_start, label_end = _nc_time_range(labels_path)

    # ── 2. Date alignment checks ──────────────────────────────────────────────
    flags: list[str] = []

    # a. labels vs splits (±30 days)
    if has_labels and label_start and label_end:
        if abs((label_start - splits_start).days) > LABEL_TOL_DAYS:
            flags.append("LABEL_START_MISMATCH")
        if abs((label_end - splits_end).days) > LABEL_TOL_DAYS:
            flags.append("LABEL_END_MISMATCH")

    # b. ERA5 must fully cover label period
    era5_covers_labels = None
    if has_ERA5 and era5_start and era5_end and label_start and label_end:
        era5_covers_labels = (era5_start <= label_start) and (era5_end >= label_end)
        if not era5_covers_labels:
            flags.append("ERA5_UNDERCOVERS")

    # c. TWSA overlaps label period
    twsa_has_overlap = None
    if has_TWSA and twsa_start and twsa_end:
        twsa_has_overlap = (twsa_end >= splits_start) and (twsa_start <= splits_end)
        if not twsa_has_overlap:
            flags.append("TWSA_NO_OVERLAP")

    # d. SIF overlaps label period
    sif_has_overlap = None
    if has_SIF and sif_start and sif_end:
        sif_has_overlap = (sif_end >= splits_start) and (sif_start <= splits_end)
        if not sif_has_overlap:
            flags.append("SIF_NO_OVERLAP")

    # e. S2 token range overlaps label period
    if has_S2 and s2_start and s2_end:
        if not (s2_end >= splits_start and s2_start <= splits_end):
            flags.append("S2_NO_OVERLAP")

    # f. S1 token range overlaps label period (only if S1 present)
    if has_S1 and s1_start and s1_end:
        if not (s1_end >= splits_start and s1_start <= splits_end):
            flags.append("S1_NO_OVERLAP")

    # g. CloudMask must cover S2 token range (cm_start <= s2_start, cm_end >= s2_end)
    # Allow ±60 days tolerance at start; cloud mask end must be >= s2_end - 60 days.
    # Cloud mask having MORE coverage than S2 tokens is fine (harmless extra data).
    if has_CloudMask and has_S2 and cm_start and cm_end and s2_start and s2_end:
        start_gap = (cm_start - s2_start).days   # positive = cm starts after s2
        end_gap   = (s2_end - cm_end).days        # positive = cm ends before s2
        if start_gap > CLOUDMASK_TOL_DAYS or end_gap > CLOUDMASK_TOL_DAYS:
            flags.append("CLOUDMASK_RANGE_MISMATCH")

    # ── 3. Missing modalities → prepend flags ─────────────────────────────────
    modality_checks = [
        ("S2", has_S2), ("S1", has_S1), ("DEM", has_DEM), ("LULC", has_LULC),
        ("ERA5", has_ERA5), ("soil", has_soil), ("TWSA", has_TWSA), ("SIF", has_SIF),
        ("CloudMask", has_CloudMask), ("labels", has_labels),
    ]
    missing = [name for name, present in modality_checks if not present]
    for name in missing:
        flags.insert(0, f"MISSING_{name}")

    has_missing    = bool(missing)
    has_date_issue = any(f for f in flags if not f.startswith("MISSING_"))
    labels_match_splits = ("LABEL_START_MISMATCH" not in flags and
                           "LABEL_END_MISMATCH"   not in flags)

    if has_missing and has_date_issue:
        status = "MISSING+DATE_MISMATCH"
    elif has_missing:
        status = "MISSING"
    elif has_date_issue:
        status = "DATE_MISMATCH"
    else:
        status = "OK"

    return {
        "station": station, "category": category,
        "splits_start": _fmt(splits_start), "splits_end": _fmt(splits_end),
        "has_S2": has_S2, "has_S1": has_S1, "has_DEM": has_DEM, "has_LULC": has_LULC,
        "has_ERA5": has_ERA5, "has_soil": has_soil, "has_TWSA": has_TWSA, "has_SIF": has_SIF,
        "has_CloudMask": has_CloudMask, "has_labels": has_labels,
        "s2_start": _fmt(s2_start), "s2_end": _fmt(s2_end),
        "s1_start": _fmt(s1_start), "s1_end": _fmt(s1_end),
        "era5_start": _fmt(era5_start), "era5_end": _fmt(era5_end),
        "label_start": _fmt(label_start), "label_end": _fmt(label_end),
        "twsa_start": _fmt(twsa_start), "twsa_end": _fmt(twsa_end),
        "sif_start": _fmt(sif_start), "sif_end": _fmt(sif_end),
        "cloudmask_start": _fmt(cm_start), "cloudmask_end": _fmt(cm_end),
        "era5_covers_labels": era5_covers_labels,
        "labels_match_splits": labels_match_splits,
        "twsa_has_overlap": twsa_has_overlap,
        "sif_has_overlap": sif_has_overlap,
        "flags": ",".join(flags),
        "status": status,
    }


# ── summary printer / writer ──────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    total = len(df)
    n_ok  = (df["status"] == "OK").sum()

    p("=" * 65)
    p("AUDIT INPUTS SUMMARY")
    p("=" * 65)
    p(f"Total stations audited : {total}")
    p(f"Fully OK               : {n_ok}  ({100*n_ok/total:.1f}%)")
    p(f"Needs attention        : {total - n_ok}")
    p()

    # ── existence breakdown ──────────────────────────────────────────────────
    modalities = ["S2","S1","DEM","LULC","ERA5","soil","TWSA","SIF","CloudMask","labels"]
    p("Missing by modality:")
    for mod in modalities:
        col = f"has_{mod}"
        missing_df = df[df[col] == False]
        n = len(missing_df)
        if n:
            p(f"  {mod:<12}: {n} stations")
            for row in missing_df.head(10).itertuples():
                p(f"    - {row.station}")
            if n > 10:
                p(f"    ... and {n - 10} more")
        else:
            p(f"  {mod:<12}: 0 missing  ✓")
    p()

    # ── date mismatch breakdown ───────────────────────────────────────────────
    flag_types = [
        "LABEL_START_MISMATCH", "LABEL_END_MISMATCH",
        "ERA5_UNDERCOVERS",
        "TWSA_NO_OVERLAP", "SIF_NO_OVERLAP",
        "S2_NO_OVERLAP", "S1_NO_OVERLAP",
        "CLOUDMASK_RANGE_MISMATCH",
    ]
    p("Date alignment issues:")
    any_date_issue = False
    for flag in flag_types:
        affected = df[df["flags"].str.contains(flag, na=False)]
        n = len(affected)
        if n:
            any_date_issue = True
            p(f"  {flag:<30}: {n} stations")
            for row in affected.head(5).itertuples():
                p(f"    - {row.station}")
            if n > 5:
                p(f"    ... and {n - 5} more")
    if not any_date_issue:
        p("  None  ✓")
    p()

    # ── stations needing attention ────────────────────────────────────────────
    problem = df[df["status"] != "OK"].sort_values(
        "status", key=lambda s: s.map({"MISSING+DATE_MISMATCH": 0, "MISSING": 1, "DATE_MISMATCH": 2})
    )
    if len(problem):
        p(f"Stations needing attention ({len(problem)}):")
        for row in problem.itertuples():
            p(f"  [{row.status}] {row.station}  →  {row.flags}")
    else:
        p("All stations OK — ready for training.")
    p()
    p(f"Full results: {OUTPUT_CSV}")
    p("=" * 65)

    path.write_text("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit all input modalities for training")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df_splits = pd.read_csv(CSV_ACTIVE)

    def folder_name(row) -> str:
        if row["source_network"] == "ISMN":
            return f"ISMN_{row['network']}_{row['station_id']}"
        return f"{row['source_network']}_{row['station_id']}"

    tasks = [
        (
            folder_name(row),
            _parse_date(str(row["start_date"])),
            _parse_date(str(row["end_date"])),
        )
        for _, row in df_splits.iterrows()
    ]

    print(f"Auditing {len(tasks)} stations with {args.workers} workers …\n")

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(audit_station, tasks, chunksize=4), 1):
            results.append(res)
            if i % 100 == 0 or i == len(tasks):
                n_ok = sum(1 for r in results if r["status"] == "OK")
                print(f"  [{i}/{len(tasks)}] OK so far: {n_ok}", flush=True)

    df = pd.DataFrame(results).sort_values("station").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}\n")

    write_summary(df, OUTPUT_SUMM)


if __name__ == "__main__":
    main()
