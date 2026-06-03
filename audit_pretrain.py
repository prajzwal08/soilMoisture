"""
Pre-training comprehensive audit.

Checks per station (train + val splits, sm_only category):
  1. Modality existence  — S2, S1, DEM, LULC, ERA5, soil, TWSA, SIF, CloudMask, labels
  2. ERA5 NaN           — any variable all-NaN across all days
  3. Cloud mask coverage — what fraction of S2 token dates have a cloud mask entry
                           (missing entry → acquisition treated as cloud-free)
  4. Token bundle sanity — S2/S1 bundles loadable, shapes sane, no NaN
  5. Label coverage     — labels.nc has enough non-NaN SM observations

Output:
  csvs/audit_pretrain.csv          — per-station detail
  text/audit_pretrain_summary.txt  — aggregate summary (also printed)

Usage:
    python audit_pretrain.py [--workers 8] [--split train,val] [--category sm_only]
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

DATA_ROOT  = Path("/gpfs/work3/0/prjs1968/data")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_pretrain.csv")
OUT_SUMM   = Path("/gpfs/work3/0/prjs1968/soilMoisture/text/audit_pretrain_summary.txt")

ERA5_VARS = [
    "t2m_mean","t2m_min","t2m_max","d2m_mean","d2m_min","d2m_max",
    "skt_mean","skt_min","skt_max","u10_mean","u10_min","u10_max",
    "v10_mean","v10_min","v10_max","sp_mean","sp_min","sp_max","tp_sum",
]


def _first(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern)) if directory.exists() else []
    return files[0] if files else None


def audit_station(row: dict) -> dict:
    station_id = row["station_id_dir"]   # actual directory name
    category   = row["category"]
    base       = DATA_ROOT / category / station_id

    result = {
        "station_id":         station_id,
        "split":              row["split"],
        "category":           category,
        # existence
        "has_S2":             False,
        "has_S1":             False,
        "has_DEM":            False,
        "has_LULC":           False,
        "has_ERA5":           False,
        "has_soil":           False,
        "has_TWSA":           False,
        "has_SIF":            False,
        "has_CloudMask":      False,
        "has_labels":         False,
        # ERA5 quality
        "era5_nan_vars":      "",       # comma-sep list of all-NaN variables
        # cloud mask coverage
        "s2_n_dates":         0,        # dates in S2 L12 bundle
        "cm_n_dates":         0,        # dates in cloud mask bundle
        "s2_dates_no_cm":     0,        # S2 dates missing from cloud mask
        "s2_dates_no_cm_pct": 0.0,
        # token sanity
        "s2_shape_ok":        True,
        "s1_shape_ok":        True,
        "s2_has_nan":         False,
        "s1_has_nan":         False,
        # label quality
        "label_n_days":       0,
        "label_obs_pct":      0.0,      # % days with real SM obs (non-NaN)
        # overall
        "flags":              "",
        "status":             "OK",
    }

    flags = []

    if not base.exists():
        result["flags"]  = "STATION_DIR_NOT_FOUND"
        result["status"] = "CRITICAL"
        return result

    # ── 1. Existence ──────────────────────────────────────────────────────────
    s2_dir  = base / "S2L2A"
    s1_dir  = base / "S1RTC"
    cm_dir  = base / "CloudMask"

    s2_pt   = _first(s2_dir,  "*_L12_*.pt")
    s1_asc  = _first(s1_dir,  "*_ASC_L12_*.pt")
    s1_desc = _first(s1_dir,  "*_DESC_L12_*.pt")
    s1_pt   = s1_asc or s1_desc
    cm_pt   = _first(cm_dir,  "*_*.pt")
    dem_pt  = base / "DEM"  / "dem_L12.pt"
    lulc_pt = base / "LULC" / "lulc_L12.pt"
    era5_nc = _first(base / "ERA5Land", "meteo_*_*.nc")
    soil_tf = base / "soil" / "soil_patch.tif"
    twsa_nc = _first(base / "TWSA", "twsa_*_*.nc")
    sif_nc  = _first(base / "SIF",  "sif_*_*.nc")
    label_nc = base / "labels.nc"

    result["has_S2"]       = s2_pt   is not None
    result["has_S1"]       = s1_pt   is not None
    result["has_DEM"]      = dem_pt.exists()
    result["has_LULC"]     = lulc_pt.exists()
    result["has_ERA5"]     = era5_nc is not None
    result["has_soil"]     = soil_tf.exists()
    result["has_TWSA"]     = twsa_nc is not None
    result["has_SIF"]      = sif_nc  is not None
    result["has_CloudMask"]= cm_pt   is not None
    result["has_labels"]   = label_nc.exists()

    for mod, present in [
        ("S2", result["has_S2"]), ("S1", result["has_S1"]),
        ("DEM", result["has_DEM"]), ("LULC", result["has_LULC"]),
        ("ERA5", result["has_ERA5"]), ("soil", result["has_soil"]),
        ("labels", result["has_labels"]),
    ]:
        if not present:
            flags.append(f"MISSING_{mod}")

    # TWSA and SIF optional — warn but not critical
    if not result["has_TWSA"]: flags.append("MISSING_TWSA")
    if not result["has_SIF"]:  flags.append("MISSING_SIF")
    if not result["has_CloudMask"]: flags.append("MISSING_CloudMask")

    # ── 2. ERA5 NaN check ────────────────────────────────────────────────────
    if era5_nc is not None:
        try:
            ds = xr.open_dataset(era5_nc)
            nan_vars = [v for v in ERA5_VARS if v in ds and
                        bool(np.isnan(ds[v].values).all())]
            ds.close()
            if nan_vars:
                result["era5_nan_vars"] = ",".join(nan_vars)
                flags.append(f"ERA5_ALL_NAN({len(nan_vars)}vars)")
        except Exception as e:
            flags.append(f"ERA5_LOAD_ERROR({e})")

    # ── 3. Cloud mask coverage vs S2 dates ───────────────────────────────────
    if s2_pt is not None:
        try:
            s2_data  = torch.load(s2_pt, map_location="cpu", weights_only=False)
            s2_dates = set(s2_data.get("dates", []))
            result["s2_n_dates"] = len(s2_dates)

            # Token shape check
            tokens = s2_data.get("tokens")
            if tokens is not None:
                if tokens.ndim != 3 or tokens.shape[1] != 196 or tokens.shape[2] != 768:
                    flags.append(f"S2_SHAPE_BAD({list(tokens.shape)})")
                    result["s2_shape_ok"] = False
                if torch.isnan(tokens.float()).any():
                    flags.append("S2_HAS_NAN")
                    result["s2_has_nan"] = True
        except Exception as e:
            flags.append(f"S2_LOAD_ERROR({e})")
            s2_dates = set()

        if cm_pt is not None and s2_dates:
            try:
                cm_data  = torch.load(cm_pt, map_location="cpu", weights_only=False)
                cm_dates = set(cm_data.get("dates", []))
                result["cm_n_dates"]     = len(cm_dates)
                missing_cm = s2_dates - cm_dates
                result["s2_dates_no_cm"] = len(missing_cm)
                result["s2_dates_no_cm_pct"] = round(
                    100 * len(missing_cm) / len(s2_dates), 1) if s2_dates else 0.0
                if result["s2_dates_no_cm_pct"] > 20:
                    flags.append(f"CM_COVERAGE_LOW({result['s2_dates_no_cm_pct']}%missing)")
            except Exception as e:
                flags.append(f"CM_LOAD_ERROR({e})")

    # ── 4. S1 token sanity ───────────────────────────────────────────────────
    if s1_pt is not None:
        try:
            s1_data = torch.load(s1_pt, map_location="cpu", weights_only=False)
            tokens  = s1_data.get("tokens")
            if tokens is not None:
                if tokens.ndim != 3 or tokens.shape[1] != 196 or tokens.shape[2] != 768:
                    flags.append(f"S1_SHAPE_BAD({list(tokens.shape)})")
                    result["s1_shape_ok"] = False
                if torch.isnan(tokens.float()).any():
                    flags.append("S1_HAS_NAN")
                    result["s1_has_nan"] = True
        except Exception as e:
            flags.append(f"S1_LOAD_ERROR({e})")

    # ── 5. Label quality ─────────────────────────────────────────────────────
    if label_nc.exists():
        try:
            ds = xr.open_dataset(label_nc)
            # surface SM variable (0-10 cm depth)
            sm_vars = [v for v in ds.data_vars if "0" in v and "10" in v]
            if not sm_vars:
                sm_vars = list(ds.data_vars)[:1]
            if sm_vars:
                vals = ds[sm_vars[0]].values.ravel()
                n_days = len(vals)
                obs_pct = round(100 * float(np.isfinite(vals).mean()), 1)
                result["label_n_days"]  = n_days
                result["label_obs_pct"] = obs_pct
                if obs_pct < 5:
                    flags.append(f"LABEL_SPARSE({obs_pct}%)")
            ds.close()
        except Exception as e:
            flags.append(f"LABEL_LOAD_ERROR({e})")

    # ── Final status ─────────────────────────────────────────────────────────
    critical = [f for f in flags if any(f.startswith(p) for p in
                ("MISSING_S2","MISSING_S1","MISSING_DEM","MISSING_LULC",
                 "MISSING_ERA5","MISSING_soil","MISSING_labels",
                 "ERA5_ALL_NAN","S2_LOAD_ERROR","S1_LOAD_ERROR","STATION_DIR"))]
    warnings = [f for f in flags if f not in critical]

    if critical:
        result["status"] = "CRITICAL"
    elif warnings:
        result["status"] = "WARN"

    result["flags"] = "|".join(flags)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int, default=8)
    parser.add_argument("--split",    default="train,val",
                        help="Comma-separated splits to audit")
    parser.add_argument("--category", default="sm_only",
                        help="Comma-separated categories to audit")
    args = parser.parse_args()

    splits     = [s.strip() for s in args.split.split(",")]
    categories = [c.strip() for c in args.category.split(",")]

    df = pd.read_csv(SPLITS_CSV)
    df = df[df["split"].isin(splits)]

    def _category(r):
        if r["has_soil_moisture"] and r["has_flux"]:
            return "sm_and_flux"
        if r["has_soil_moisture"]:
            return "sm_only"
        return "flux_only"

    df["category"] = df.apply(_category, axis=1)

    def _station_dir_name(r):
        # ISMN subnetworks: source_network=ISMN, network=SCAN/AMMA-CATCH/etc.
        # → ISMN_{network}_{station_id}
        # ICOS / AmeriFlux: source_network == network
        # → {source_network}_{station_id}
        if str(r["source_network"]) != str(r["network"]):
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['source_network']}_{r['station_id']}"

    df = df.dropna(subset=["source_network", "network", "station_id"])
    df["station_id_dir"] = df.apply(_station_dir_name, axis=1)
    df = df[df["category"].isin(categories)].reset_index(drop=True)

    print(f"Auditing {len(df)} stations (splits={splits}, categories={categories})...")

    rows_input = df[["station_id_dir","split","category"]].rename(
        columns={"station_id_dir": "station_id_dir"}).to_dict("records")

    with Pool(args.workers) as pool:
        results = pool.map(audit_station, rows_input)

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"Pre-training audit — {len(out_df)} stations")
    lines.append("=" * 60)

    for status in ["OK", "WARN", "CRITICAL"]:
        n = (out_df["status"] == status).sum()
        lines.append(f"  {status:<10}: {n}")
    lines.append("")

    # Modality presence
    lines.append("Modality presence:")
    for col in ["has_S2","has_S1","has_DEM","has_LULC","has_ERA5",
                "has_soil","has_TWSA","has_SIF","has_CloudMask","has_labels"]:
        n_ok  = out_df[col].sum()
        n_tot = len(out_df)
        lines.append(f"  {col:<18}: {n_ok}/{n_tot}")
    lines.append("")

    # ERA5 NaN
    era5_bad = out_df[out_df["era5_nan_vars"] != ""]
    lines.append(f"ERA5 all-NaN variables: {len(era5_bad)} stations affected")
    for _, r in era5_bad.iterrows():
        lines.append(f"  {r['station_id']}: {r['era5_nan_vars']}")
    lines.append("")

    # Cloud mask coverage
    has_both = out_df[(out_df["has_S2"]) & (out_df["has_CloudMask"])]
    no_cm    = out_df[(out_df["has_S2"]) & (~out_df["has_CloudMask"])]
    lines.append(f"Cloud mask coverage (stations with S2):")
    lines.append(f"  Have cloud mask   : {len(has_both)}")
    lines.append(f"  No cloud mask     : {len(no_cm)}")
    if len(has_both) > 0:
        mean_missing = has_both["s2_dates_no_cm_pct"].mean()
        max_missing  = has_both["s2_dates_no_cm_pct"].max()
        high_missing = (has_both["s2_dates_no_cm_pct"] > 20).sum()
        lines.append(f"  Mean S2 dates without CM entry : {mean_missing:.1f}%")
        lines.append(f"  Max  S2 dates without CM entry : {max_missing:.1f}%")
        lines.append(f"  Stations >20% S2 dates missing from CM: {high_missing}")
    lines.append("")

    # Token issues
    lines.append("Token issues:")
    for flag in ["S2_HAS_NAN","S1_HAS_NAN","S2_SHAPE_BAD","S1_SHAPE_BAD",
                 "S2_LOAD_ERROR","S1_LOAD_ERROR","CM_LOAD_ERROR"]:
        n = out_df["flags"].str.contains(flag, na=False).sum()
        if n:
            lines.append(f"  {flag}: {n} stations")
    lines.append("")

    # CRITICAL stations
    crit = out_df[out_df["status"] == "CRITICAL"][["station_id","split","flags"]]
    lines.append(f"CRITICAL stations ({len(crit)}):")
    for _, r in crit.iterrows():
        lines.append(f"  [{r['split']}] {r['station_id']}: {r['flags']}")
    lines.append("")

    # WARN stations with high CM missing
    warn_cm = has_both[has_both["s2_dates_no_cm_pct"] > 20][
        ["station_id","split","s2_n_dates","s2_dates_no_cm","s2_dates_no_cm_pct"]]
    if len(warn_cm):
        lines.append(f"Stations with >20% S2 dates missing cloud mask ({len(warn_cm)}):")
        for _, r in warn_cm.iterrows():
            lines.append(f"  [{r['split']}] {r['station_id']}: "
                         f"{r['s2_dates_no_cm']}/{r['s2_n_dates']} "
                         f"({r['s2_dates_no_cm_pct']}%) without CM entry")

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_SUMM.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMM.write_text(summary)
    print(f"\nDetailed results → {OUT_CSV}")
    print(f"Summary          → {OUT_SUMM}")


if __name__ == "__main__":
    main()
