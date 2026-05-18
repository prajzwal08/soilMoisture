"""
Preprocess AmeriFlux FLUXNET FULLSET data for all 205 downloaded sites.

SWC (sites with depth info from FLX15_site_info.csv):
  - Read HH/HR file; keep QC=0 (measured) timesteps only
  - Resample to daily mean; mask days with < 6 valid observations
  - Convert % -> fraction; bin into 0-10, 10-30, 30-100 cm
  - Gap analysis -> trim to surface valid period -> climatological gap-fill

Flux variables (all sites, from DD file, kept as-is with QC):
  NEE_VUT_REF, GPP_DT_VUT_REF, GPP_NT_VUT_REF,
  RECO_DT_VUT_REF, RECO_NT_VUT_REF, H_F_MDS, LE_F_MDS, NETRAD

Output: /home/khanalp/data/soilmoisture/ameriflux_level1/
"""

import zipfile
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import (
    longest_available_after_removing_long_gaps,
    trim_to_surface_valid_period_and_keep_well_covered_depths,
    gapfill_by_monthday_mean_with_feb29_fallback,
    _fill_short_nan_runs,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
AMF_DIR      = Path("/home/khanalp/data/flux/ameriflux_raw")
SITES_CSV    = Path("/home/khanalp/data/flux/ameriflux_raw/filtered_sites.csv")
FLX15_CSV    = Path("/home/khanalp/data/flux/FLX15_site_info.csv")
OUT_DIR_SWC  = Path("/home/khanalp/data/soilmoisture/ameriflux_level1/swc_and_flux")
OUT_DIR_FLUX = Path("/home/khanalp/data/soilmoisture/ameriflux_level1/flux_only")
OUT_DIR_SWC.mkdir(parents=True, exist_ok=True)
OUT_DIR_FLUX.mkdir(parents=True, exist_ok=True)

MIN_QC_MEASURED   = 0.60  # QC threshold: day valid if fraction measured > this
MIN_MEASURED_FRAC = 0.60  # HH → hourly → daily: valid day if >60% hourly slots measured
HOURS_PER_DAY     = 24

DEPTH_BINS      = [0.0, 10.0, 30.0, 100.0]
DEPTH_LABELS    = ["0-10", "10-30", "30-100"]
MIN_OBS_PER_DAY = 6
MIN_VALID_DAYS  = 1095  # require >= 3 years: 1yr context + 1yr train + 1yr test
TRAIN_START     = "2017-01-01"
TRAIN_END       = "2025-12-31"

FLUX_VARS = ["LE_F_MDS"]

FLUX_UNITS = {
    "NEE_VUT_REF":     "umol CO2 m-2 s-1",
    "GPP_DT_VUT_REF":  "umol CO2 m-2 s-1",
    "GPP_NT_VUT_REF":  "umol CO2 m-2 s-1",
    "RECO_DT_VUT_REF": "umol CO2 m-2 s-1",
    "RECO_NT_VUT_REF": "umol CO2 m-2 s-1",
    "H_F_MDS":         "W m-2",
    "LE_F_MDS":        "W m-2",
    "NETRAD":          "W m-2",
}

FLUX_LONGNAMES = {
    "NEE_VUT_REF":     "Net ecosystem exchange (VUT reference)",
    "GPP_DT_VUT_REF":  "Gross primary production (daytime partitioning)",
    "GPP_NT_VUT_REF":  "Gross primary production (nighttime partitioning)",
    "RECO_DT_VUT_REF": "Ecosystem respiration (daytime partitioning)",
    "RECO_NT_VUT_REF": "Ecosystem respiration (nighttime partitioning)",
    "H_F_MDS":         "Sensible heat flux (MDS gap-filled)",
    "LE_F_MDS":        "Latent heat flux (MDS gap-filled)",
    "NETRAD":          "Net radiation",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _open_subhourly(zip_path: Path) -> pd.DataFrame:
    """Open HH or HR CSV from a FULLSET zip."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(
            (n for n in zf.namelist()
             if any(f"_{r}_" in n for r in ("HH", "HR")) and "ERA5" not in n),
            None,
        )
        if name is None:
            raise FileNotFoundError("No HH/HR file found in zip")
        with zf.open(name) as f:
            df = pd.read_csv(f, na_values=[-9999, -9999.0])

    df["TIMESTAMP_START"] = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    )
    return df.set_index("TIMESTAMP_START").sort_index()


def _flux_hh_to_hourly(df_hh: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: HH → Hourly.
    Filter each variable to measured-only (QC==0), resample to hourly mean.
    NaN hourly slot = no measured sub-hourly data in that hour.

    - NEE, H, LE : filtered to QC==0
    - GPP, RECO  : inherit NEE's QC==0 mask (partitioned products)
    - NETRAD     : NaN already marks missing; no QC filter needed
    """
    if "NEE_VUT_REF_QC" in df_hh.columns:
        nee_mask = df_hh["NEE_VUT_REF_QC"] == 0
    else:
        nee_mask = pd.Series(True, index=df_hh.index)

    hourly: dict[str, pd.Series] = {}
    for var in FLUX_VARS:
        if var not in df_hh.columns:
            continue
        s = df_hh[var].copy().astype(float)
        if var in ("GPP_DT_VUT_REF", "GPP_NT_VUT_REF", "RECO_DT_VUT_REF", "RECO_NT_VUT_REF"):
            s[~nee_mask] = np.nan
        elif var != "NETRAD" and f"{var}_QC" in df_hh.columns:
            s[df_hh[f"{var}_QC"] != 0] = np.nan
        hourly[var] = s.resample("h").mean()

    return pd.DataFrame(hourly)


def _flux_hourly_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Hourly → Daily.
    Day is valid if >MIN_MEASURED_FRAC of 24 hourly slots have data.
    Invalid days → NaN.  daily_QC = n_valid_hours / 24  (0.0–1.0).
    """
    result: dict[str, pd.Series] = {}
    for var in FLUX_VARS:
        if var not in df_hourly.columns:
            continue
        s          = df_hourly[var]
        n_valid    = s.resample("D").count()
        daily_qc   = n_valid / HOURS_PER_DAY
        daily_mean = s.resample("D").mean()
        daily_mean[daily_qc <= MIN_MEASURED_FRAC] = np.nan
        result[var]         = daily_mean
        result[f"{var}_QC"] = daily_qc
    return pd.DataFrame(result)


def _flux_hh_to_daily(df_hh: pd.DataFrame) -> pd.DataFrame:
    """HH → Hourly → Daily pipeline."""
    return _flux_hourly_to_daily(_flux_hh_to_hourly(df_hh))


def _open_daily(zip_path: Path) -> pd.DataFrame:
    """Open DD CSV from a FULLSET zip."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if "_DD_" in n and "ERA5" not in n)
        with zf.open(name) as f:
            df = pd.read_csv(f, na_values=[-9999, -9999.0])

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"].astype(str), format="%Y%m%d")
    return df.set_index("TIMESTAMP").sort_index()


# ---------------------------------------------------------------------------
# SWC pipeline helpers
# ---------------------------------------------------------------------------
def _build_depth_map(flx15_row: pd.Series) -> dict[int, float]:
    """Return {N: depth_cm} for SWC_N columns that have a non-NaN depth value."""
    depth_map = {}
    for col in flx15_row.index:
        if not (col.startswith("SWC_") and col.endswith("_depth/cm")):
            continue
        val = flx15_row[col]
        if pd.isna(val):
            continue
        n = int(col.split("_")[1])
        depth_map[n] = float(val)
    return depth_map


def _swc_to_daily(df_hh: pd.DataFrame, depth_map: dict[int, float]) -> pd.DataFrame:
    """
    Per sensor:
      - keep QC=0 timesteps only
      - resample to daily mean; mask days with < MIN_OBS_PER_DAY valid obs
      - convert % -> fraction if values > 1.5
    Returns DataFrame with sensor indices as columns, daily date index.
    """
    daily_sensors: dict[int, pd.Series] = {}
    for n in depth_map:
        swc_col = f"SWC_F_MDS_{n}"
        qc_col  = f"SWC_F_MDS_{n}_QC"
        if swc_col not in df_hh.columns:
            continue

        s = df_hh[swc_col].copy()
        if qc_col in df_hh.columns:
            s[df_hh[qc_col] != 0] = np.nan

        daily_count = s.resample("D").count()
        daily_mean  = s.resample("D").mean()
        daily_mean[daily_count < MIN_OBS_PER_DAY] = np.nan

        valid = daily_mean.dropna()
        if len(valid) and valid.max() > 1.5:
            daily_mean = daily_mean / 100.0

        daily_sensors[n] = daily_mean

    if not daily_sensors:
        return pd.DataFrame()
    return pd.DataFrame(daily_sensors)


def _bin_sensors(daily_sensors: pd.DataFrame, depth_map: dict[int, float]) -> pd.DataFrame:
    """Average daily sensor columns into depth bins; return DataFrame."""
    bin_series: dict[str, list[pd.Series]] = {lbl: [] for lbl in DEPTH_LABELS}

    for n, depth_cm in depth_map.items():
        if n not in daily_sensors.columns:
            continue
        label = pd.cut(
            [depth_cm],
            bins=DEPTH_BINS,
            labels=DEPTH_LABELS,
            right=True,
            include_lowest=True,
        )[0]
        if pd.isna(label):
            continue
        bin_series[str(label)].append(daily_sensors[n])

    result = {}
    for lbl in DEPTH_LABELS:
        if bin_series[lbl]:
            result[lbl] = pd.concat(bin_series[lbl], axis=1).mean(axis=1)

    return pd.DataFrame(result) if result else pd.DataFrame()


def _make_swc_dataset(binned: pd.DataFrame) -> xr.Dataset:
    """Wrap binned daily DataFrame into xr.Dataset (utils.py conventions)."""
    depths = list(binned.columns)
    return xr.Dataset(
        {
            "soil_moisture": xr.DataArray(
                binned[depths].values,
                dims=["date_time", "depth"],
                attrs={
                    "units":         "m3 m-3",
                    "long_name":     "Volumetric soil moisture content",
                    "standard_name": "volume_fraction_of_condensed_water_in_soil",
                    "valid_range":   [0.0, 1.0],
                },
            )
        },
        coords={"date_time": binned.index.values, "depth": depths},
    )


def _run_swc_pipeline(ds: xr.Dataset) -> xr.Dataset | None:
    """Apply gap analysis, trim, and gap-fill to an SWC dataset."""
    surface_depth = str(ds["depth"].values[0])   # shallowest bin present

    longest_avail = longest_available_after_removing_long_gaps(
        ds, depth_dim="depth", time_dim="date_time",
        var="soil_moisture", max_gap_days=7,
    )

    L, start, end = longest_avail.get(surface_depth, (0, None, None))
    if L < MIN_VALID_DAYS:
        return None

    ds = trim_to_surface_valid_period_and_keep_well_covered_depths(
        ds, longest_avail,
        surface_depth=surface_depth,
        var="soil_moisture",
        depth_dim="depth",
        time_dim="date_time",
        min_frac=0.95,
    )

    ds = gapfill_by_monthday_mean_with_feb29_fallback(
        ds, var="soil_moisture", time_dim="date_time",
        flag_name="soil_moisture_qc",
    )

    if bool(ds["soil_moisture"].isnull().any()):
        return None

    ds["soil_moisture_qc"].attrs.update({
        "long_name":     "Soil moisture quality control flag",
        "flag_values":   "0 1 2",
        "flag_meanings": "observed gap_filled missing",
    })
    return ds


# ---------------------------------------------------------------------------
# Flux helper
# ---------------------------------------------------------------------------
def _add_flux_vars(ds: xr.Dataset, df_dd: pd.DataFrame) -> xr.Dataset:
    """Add flux variables from DD file aligned to ds time axis."""
    time_index = pd.DatetimeIndex(ds["date_time"].values)
    df_aligned = df_dd.reindex(time_index)

    for var in FLUX_VARS:
        if var not in df_dd.columns:
            continue
        da = xr.DataArray(df_aligned[var].values, dims=["date_time"])
        da.attrs["units"]     = FLUX_UNITS.get(var, "")
        da.attrs["long_name"] = FLUX_LONGNAMES.get(var, var)
        ds[var] = da

        qc_col = f"{var}_QC"
        if qc_col in df_dd.columns:
            qc = xr.DataArray(df_aligned[qc_col].values, dims=["date_time"])
            qc.attrs["long_name"] = (
                f"Quality flag for {var}: fraction of measured (QC=0) "
                "sub-daily values contributing to the daily aggregate"
            )
            ds[qc_col] = qc

    return ds


def _flux_only_dataset(df_dd: pd.DataFrame) -> xr.Dataset:
    """Create a Dataset containing only flux variables on the DD time axis."""
    time = df_dd.index.values
    ds = xr.Dataset(coords={"date_time": time})
    for var in FLUX_VARS:
        if var not in df_dd.columns:
            continue
        da = xr.DataArray(df_dd[var].values, dims=["date_time"])
        da.attrs["units"]     = FLUX_UNITS.get(var, "")
        da.attrs["long_name"] = FLUX_LONGNAMES.get(var, var)
        ds[var] = da

        qc_col = f"{var}_QC"
        if qc_col in df_dd.columns:
            qc = xr.DataArray(df_dd[qc_col].values, dims=["date_time"])
            qc.attrs["long_name"] = (
                f"Quality flag for {var}: fraction of measured (QC=0) "
                "sub-daily values contributing to the daily aggregate"
            )
            ds[qc_col] = qc
    return ds


# ---------------------------------------------------------------------------
# Flux quality helpers
# ---------------------------------------------------------------------------
def _unify_carbon_qc(ds: xr.Dataset) -> xr.Dataset:
    """GPP and RECO are partitioned from NEE — assign NEE_VUT_REF_QC as their QC."""
    if "NEE_VUT_REF_QC" not in ds:
        return ds
    for var in ("GPP_DT_VUT_REF", "GPP_NT_VUT_REF", "RECO_DT_VUT_REF", "RECO_NT_VUT_REF"):
        if var in ds:
            qc = ds["NEE_VUT_REF_QC"].copy()
            qc.attrs["long_name"] = (
                f"Quality flag for {var} (same as NEE_VUT_REF_QC; partitioned from NEE)"
            )
            ds[f"{var}_QC"] = qc
    return ds


QC_VARS_FOR_GAP_ANALYSIS = ("LE_F_MDS_QC",)
MAX_GAP_DAYS = 30


def _flux_longest_valid_period(
    ds: xr.Dataset,
    max_gap_days: int = MAX_GAP_DAYS,
) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Find the longest consecutive period of measured flux days, bridging gaps
    <= max_gap_days.  A day is 'measured' when ALL QC_VARS_FOR_GAP_ANALYSIS
    have QC >= MIN_QC_MEASURED.
    """
    available_qc = [v for v in QC_VARS_FOR_GAP_ANALYSIS if v in ds]
    if not available_qc:
        return 0, None, None

    # day is valid only if ALL required QC variables meet the threshold
    valid = np.ones(ds.sizes["date_time"], dtype=bool)
    for qc_var in available_qc:
        qc = ds[qc_var].values.astype(float)
        valid &= (~np.isnan(qc)) & (qc >= MIN_QC_MEASURED)

    # mask invalid days as NaN so _fill_short_nan_runs treats them as gaps
    proxy = np.where(valid, 1.0, np.nan)
    avail = _fill_short_nan_runs(proxy, max_gap_days=max_gap_days)
    dates = pd.DatetimeIndex(ds["date_time"].values)

    if not avail.any():
        return 0, None, None

    y = np.r_[False, avail, False]
    idx = np.flatnonzero(y[1:] != y[:-1])
    starts    = idx[::2]
    ends_excl = idx[1::2]
    lens      = ends_excl - starts

    k   = int(lens.argmax())
    s_i = int(starts[k])
    e_i = int(ends_excl[k] - 1)
    return int(lens[k]), pd.Timestamp(dates[s_i]), pd.Timestamp(dates[e_i])


# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------
def _site_metadata(site_row: pd.Series) -> dict:
    return {
        "site_id":            str(site_row["SITE_ID"]),
        "site_name":          str(site_row.get("SITE_NAME", "")),
        "latitude":           float(site_row["LOCATION_LAT"]),
        "longitude":          float(site_row["LOCATION_LONG"]),
        "elevation_m":        float(site_row["LOCATION_ELEV"]) if pd.notna(site_row.get("LOCATION_ELEV")) else float("nan"),
        "IGBP":               str(site_row.get("IGBP", "")),
        "climate_koeppen":    str(site_row.get("CLIMATE_KOEPPEN", "")),
        "MAT_degC":           float(site_row["MAT"]) if pd.notna(site_row.get("MAT")) else float("nan"),
        "MAP_mm":             float(site_row["MAP"]) if pd.notna(site_row.get("MAP")) else float("nan"),
        "tower_began":        str(site_row.get("TOWER_BEGAN", "")),
        "source":                  "AmeriFlux FLUXNET FULLSET",
        "depth_bins":              "0,10,30,100",
        "min_obs_per_day":         MIN_OBS_PER_DAY,
        "swc_max_gap_days":        7,
        "swc_min_coverage_frac":   0.95,
        "flux_aggregation":        "HH->hourly->daily; QC=fraction of measured hours per day",
        "flux_min_measured_frac":  MIN_MEASURED_FRAC,
        "flux_max_gap_days":       MAX_GAP_DAYS,
        "flux_qc_variables":       "NEE+LE joint validity; GPP/RECO inherit NEE QC",
        "processing_date":         datetime.now().strftime("%Y-%m-%d"),
    }


def process_site(site_row: pd.Series, flx15_row: pd.Series | None) -> str | None:
    site_id  = site_row["SITE_ID"]
    zip_path = AMF_DIR / f"{site_id}_FLUXNET_FULLSET.zip"
    if not zip_path.exists():
        return None

    # --- Open HH once — used for both SWC and flux ---
    try:
        df_hh = _open_subhourly(zip_path)
    except Exception as e:
        print(f"  [{site_id}] HH read error: {e}")
        return None

    # --- Clip to training window before any processing ---
    df_hh = df_hh.loc[TRAIN_START:TRAIN_END]
    if df_hh.empty:
        return None

    # --- Compute daily flux from HH (QC fractions derived, GPP/RECO inherit NEE QC) ---
    df_flux_daily = _flux_hh_to_daily(df_hh)

    has_swc = False
    ds = None

    # --- SWC pipeline (only if depth info available) ---
    if flx15_row is not None:
        depth_map = _build_depth_map(flx15_row)
        if depth_map:
            try:
                daily_sensors = _swc_to_daily(df_hh, depth_map)
                if not daily_sensors.empty:
                    binned = _bin_sensors(daily_sensors, depth_map)
                    if not binned.empty:
                        swc_ds = _make_swc_dataset(binned)
                        swc_ds = _run_swc_pipeline(swc_ds)
                        if swc_ds is not None:
                            ds = _add_flux_vars(swc_ds, df_flux_daily)
                            has_swc = True
            except Exception as e:
                print(f"  [{site_id}] SWC pipeline error: {e}")

    # --- Flux-only fallback ---
    if ds is None:
        ds = _flux_only_dataset(df_flux_daily)

    if ds is None or len(ds["date_time"]) == 0:
        return None

    # --- Gap analysis for flux-only stations ---
    if not has_swc:
        L, start, end = _flux_longest_valid_period(ds, max_gap_days=MAX_GAP_DAYS)
        if L < MIN_VALID_DAYS:
            return None
        ds = ds.sel(date_time=slice(start, end))

    # --- Metadata ---
    meta = _site_metadata(site_row)
    meta["has_soil_moisture"] = str(has_swc)
    ds.attrs.update(meta)

    time_vals = pd.DatetimeIndex(ds["date_time"].values)
    start_str = time_vals[0].strftime("%Y%m%d")
    end_str   = time_vals[-1].strftime("%Y%m%d")

    out_dir  = OUT_DIR_SWC if has_swc else OUT_DIR_FLUX
    out_path = out_dir / f"{site_id}_{start_str}_{end_str}.nc"
    ds.to_netcdf(out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> tuple[str, str]:
    """
    Returns (site_id, result) where result is one of:
      "swc+flux:<path>", "flux:<path>", "skipped"
    """
    site_row, flx15_row = args
    site_id = site_row["SITE_ID"]
    try:
        out = process_site(site_row, flx15_row)
    except Exception as e:
        return site_id, f"error:{e}"
    if out is None:
        return site_id, "skipped"
    if Path(out).parent == OUT_DIR_SWC:
        return site_id, f"swc+flux:{out}"
    return site_id, f"flux:{out}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
N_WORKERS = 16

def main():
    sites = pd.read_csv(SITES_CSV)
    flx15 = pd.read_csv(FLX15_CSV).set_index("site_id")
    swc_depth_cols = [c for c in flx15.columns if "SWC" in c]

    # Build argument list
    args = []
    for _, site_row in sites.iterrows():
        site_id = site_row["SITE_ID"]
        flx15_row = None
        if site_id in flx15.index:
            row = flx15.loc[site_id]
            if any(pd.notna(row[c]) for c in swc_depth_cols):
                flx15_row = row
        args.append((site_row, flx15_row))

    print(f"Processing {len(args)} sites with {N_WORKERS} workers ...")

    saved_swc, saved_flux_only, skipped, errors = [], [], [], []

    with Pool(N_WORKERS) as pool:
        for site_id, result in pool.imap_unordered(_worker, args):
            if result.startswith("swc+flux:"):
                print(f"  [{site_id}] SWC+flux -> {Path(result.split(':',1)[1]).name}")
                saved_swc.append(site_id)
            elif result.startswith("flux:"):
                print(f"  [{site_id}] flux only -> {Path(result.split(':',1)[1]).name}")
                saved_flux_only.append(site_id)
            elif result.startswith("error:"):
                print(f"  [{site_id}] ERROR: {result[6:]}")
                errors.append(site_id)
            else:
                print(f"  [{site_id}] skipped")
                skipped.append(site_id)

    print(f"\nDone.")
    print(f"  SWC + flux : {len(saved_swc)}")
    print(f"  Flux only  : {len(saved_flux_only)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Errors     : {len(errors)}")
    if errors:
        print(f"  Error IDs  : {errors}")


if __name__ == "__main__":
    main()
