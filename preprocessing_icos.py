"""
Preprocess ICOS Ecosystem flux and soil moisture data.

Merges ICOS2025 (Ecosystem Release, 79 stations, 2019-2025) and
ICOS2020 (Warm Winter / 2G60-ZHAK, 74 stations, 1989-2020).

Merge strategy per station:
  - ICOS2025-only or ICOS2020-only: use that source directly
  - In both: count years of ICOS2025 data from 2017 onwards (incl. 2017);
    if <= 3 years → merge with ICOS2020 (2025 wins on overlapping timestamps);
    if  > 3 years → use ICOS2025 alone (sufficient for training)

Sensor depth priority:
  1. ICOSETC_{site}_VARINFO_FLUXNET_HH_L2.csv  (VAR_INFO_HEIGHT, metres)
  2. ICOS_site_info.csv  (SWC_N_depth/cm columns)
  3. No depth info → flux-only output

SWC pipeline (identical to AmeriFlux):
  - QC=0 (measured) only; daily mean, mask days with < 6 obs
  - % → fraction if max > 1.5; bin into 0-10, 10-30, 30-100 cm
  - Gap analysis → trim to surface valid period → climatological gap-fill

Flux: LE_F_MDS only.

Output: /home/khanalp/data/soilmoisture/icos_level1/
"""

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
ICOS2025_DIR  = Path("/home/khanalp/data/flux/ICOS2025")
ICOS2020_DIR  = Path("/home/khanalp/data/flux/ICOS2020")
STATION_CSV   = Path("/home/khanalp/data/flux/icos_raw/ICOS_available_station.csv")
SITE_INFO_CSV = Path("/home/khanalp/data/flux/ICOS_site_info.csv")
FLX15_CSV     = Path("/home/khanalp/data/flux/FLX15_site_info.csv")
OUT_DIR_SWC   = Path("/home/khanalp/data/soilmoisture/icos_level1/swc_and_flux")
OUT_DIR_FLUX  = Path("/home/khanalp/data/soilmoisture/icos_level1/flux_only")
OUT_DIR_SWC.mkdir(parents=True, exist_ok=True)
OUT_DIR_FLUX.mkdir(parents=True, exist_ok=True)

MIN_QC_MEASURED   = 0.60
MIN_MEASURED_FRAC = 0.60
HOURS_PER_DAY     = 24

DEPTH_BINS   = [0.0, 10.0, 30.0, 100.0]
DEPTH_LABELS = ["0-10", "10-30", "30-100"]
MIN_OBS_PER_DAY = 6
MIN_VALID_DAYS  = 1095   # 3 years
TRAIN_START     = "2017-01-01"
TRAIN_END       = "2025-12-31"
MIN_YEARS_POST_2017 = 3  # skip station if fewer than this many years remain after 2017 clip

FLUX_VARS = ["LE_F_MDS"]

FLUX_UNITS     = {"LE_F_MDS": "W m-2"}
FLUX_LONGNAMES = {"LE_F_MDS": "Latent heat flux (MDS gap-filled)"}

QC_VARS_FOR_GAP_ANALYSIS = ("LE_F_MDS_QC",)
MAX_GAP_DAYS = 30

N_WORKERS = 16


# ---------------------------------------------------------------------------
# I/O helpers — read HH CSVs
# ---------------------------------------------------------------------------
def _read_hh_2025(site_id: str) -> pd.DataFrame | None:
    """Read ICOS2025 FLUXNET_HH_L2 CSV; returns DatetimeIndex DataFrame or None."""
    path = ICOS2025_DIR / site_id / f"ICOSETC_{site_id}_FLUXNET_HH_L2.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, na_values=[-9999, -9999.0])
    df["TIMESTAMP_START"] = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    )
    return df.set_index("TIMESTAMP_START").sort_index()


def _read_hh_2020(site_id: str) -> pd.DataFrame | None:
    """Read ICOS2020 FLUXNET2015 FULLSET_HH CSV; returns DatetimeIndex DataFrame or None."""
    dirs = list(ICOS2020_DIR.glob(f"FLX_{site_id}_FLUXNET2015_FULLSET_*"))
    if not dirs:
        return None
    files = list(dirs[0].glob(f"FLX_{site_id}_FLUXNET2015_FULLSET_HH_*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0], na_values=[-9999, -9999.0])
    df["TIMESTAMP_START"] = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    )
    return df.set_index("TIMESTAMP_START").sort_index()


def _merge_hh(site_id: str) -> pd.DataFrame | None:
    """
    Return merged HH DataFrame for site_id.

    Always merges ICOS2020 + ICOS2025 when both are available.
    ICOS2025 wins on overlapping timestamps.
    The post-2017 year count check is done in process_site after clipping.
    """
    in_2025 = (ICOS2025_DIR / site_id).exists() and (
        ICOS2025_DIR / site_id / f"ICOSETC_{site_id}_FLUXNET_HH_L2.csv"
    ).exists()
    in_2020 = bool(list(ICOS2020_DIR.glob(f"FLX_{site_id}_FLUXNET2015_FULLSET_*")))

    if not in_2025 and not in_2020:
        return None

    if not in_2025:
        return _read_hh_2020(site_id)

    df25 = _read_hh_2025(site_id)
    if df25 is None:
        return _read_hh_2020(site_id) if in_2020 else None

    if not in_2020:
        return df25

    # Both available — always merge; 2025 wins on overlap
    df20 = _read_hh_2020(site_id)
    if df20 is None:
        return df25
    merged = pd.concat([df20, df25]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------
def _swc_depths_from_varinfo(site_id: str) -> dict[int, float]:
    """
    Extract {sensor_index: depth_cm} from ICOS2025 VARINFO file.
    VAR_INFO_HEIGHT is in metres (negative = below ground).
    """
    varinfo = ICOS2025_DIR / site_id / f"ICOSETC_{site_id}_VARINFO_FLUXNET_HH_L2.csv"
    if not varinfo.exists():
        return {}
    df = pd.read_csv(varinfo)
    result: dict[int, float] = {}
    swc_mask = (df["VARIABLE"] == "VAR_INFO_VARNAME") & (
        df["DATAVALUE"].astype(str).str.match(r"^SWC_F_MDS_\d+$", na=False)
    )
    for _, row in df[swc_mask].iterrows():
        group_id = row["GROUP_ID"]
        varname  = str(row["DATAVALUE"])
        depth_rows = df[
            (df["GROUP_ID"] == group_id) & (df["VARIABLE"] == "VAR_INFO_HEIGHT")
        ]
        if depth_rows.empty:
            continue
        depth_m = float(depth_rows.iloc[0]["DATAVALUE"])
        n = int(varname.split("_")[-1])
        result[n] = abs(depth_m) * 100.0   # metres → cm, positive
    return result


def _swc_depths_from_site_info(site_id: str, site_info: pd.DataFrame) -> dict[int, float]:
    """Extract {sensor_index: depth_cm} from ICOS_site_info.csv (fallback)."""
    rows = site_info[site_info["site_id"] == site_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    result: dict[int, float] = {}
    for i in range(1, 20):
        col = f"SWC_{i}_depth/cm"
        if col in row.index and pd.notna(row[col]):
            result[i] = float(row[col])
    return result


def _get_swc_depths(
    site_id: str, site_info: pd.DataFrame
) -> dict[int, float]:
    """
    Return depth map with priority:
      1. VARINFO file (ICOS2025, exact depths)
      2. ICOS_site_info.csv (fallback)
      3. {} → flux-only output
    """
    depths = _swc_depths_from_varinfo(site_id)
    if depths:
        return depths
    return _swc_depths_from_site_info(site_id, site_info)


# ---------------------------------------------------------------------------
# Flux pipeline  (identical logic to AmeriFlux, LE only)
# ---------------------------------------------------------------------------
def _flux_hh_to_hourly(df_hh: pd.DataFrame) -> pd.DataFrame:
    """HH → hourly: keep QC=0 timesteps only, resample to hourly mean."""
    hourly: dict[str, pd.Series] = {}
    for var in FLUX_VARS:
        if var not in df_hh.columns:
            continue
        s = df_hh[var].copy().astype(float)
        qc_col = f"{var}_QC"
        if qc_col in df_hh.columns:
            s[df_hh[qc_col] != 0] = np.nan
        hourly[var] = s.resample("h").mean()
    return pd.DataFrame(hourly)


def _flux_hourly_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly → daily: valid day requires >MIN_MEASURED_FRAC of 24 hourly slots."""
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
    return _flux_hourly_to_daily(_flux_hh_to_hourly(df_hh))


# ---------------------------------------------------------------------------
# SWC pipeline  (identical logic to AmeriFlux)
# ---------------------------------------------------------------------------
def _swc_to_daily(df_hh: pd.DataFrame, depth_map: dict[int, float]) -> pd.DataFrame:
    """Per sensor: QC=0 only → daily mean (≥6 obs) → % to fraction."""
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
    """Average sensor columns into depth bins."""
    bin_series: dict[str, list[pd.Series]] = {lbl: [] for lbl in DEPTH_LABELS}
    for n, depth_cm in depth_map.items():
        if n not in daily_sensors.columns:
            continue
        label = pd.cut(
            [depth_cm], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=True, include_lowest=True
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
    """Wrap binned daily DataFrame into xr.Dataset."""
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
    """Gap analysis → trim → climatological gap-fill."""
    surface_depth = str(ds["depth"].values[0])

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
# Flux dataset helpers  (identical to AmeriFlux)
# ---------------------------------------------------------------------------
def _add_flux_vars(ds: xr.Dataset, df_flux_daily: pd.DataFrame) -> xr.Dataset:
    """Align flux daily DataFrame to ds time axis and attach variables."""
    time_index  = pd.DatetimeIndex(ds["date_time"].values)
    df_aligned  = df_flux_daily.reindex(time_index)
    for var in FLUX_VARS:
        if var not in df_flux_daily.columns:
            continue
        da = xr.DataArray(df_aligned[var].values, dims=["date_time"])
        da.attrs["units"]     = FLUX_UNITS.get(var, "")
        da.attrs["long_name"] = FLUX_LONGNAMES.get(var, var)
        ds[var] = da
        qc_col = f"{var}_QC"
        if qc_col in df_flux_daily.columns:
            qc = xr.DataArray(df_aligned[qc_col].values, dims=["date_time"])
            qc.attrs["long_name"] = (
                f"Quality flag for {var}: fraction of measured (QC=0) "
                "sub-daily values contributing to the daily aggregate"
            )
            ds[qc_col] = qc
    return ds


def _flux_only_dataset(df_flux_daily: pd.DataFrame) -> xr.Dataset:
    """Dataset containing only flux variables on the daily time axis."""
    ds = xr.Dataset(coords={"date_time": df_flux_daily.index.values})
    for var in FLUX_VARS:
        if var not in df_flux_daily.columns:
            continue
        da = xr.DataArray(df_flux_daily[var].values, dims=["date_time"])
        da.attrs["units"]     = FLUX_UNITS.get(var, "")
        da.attrs["long_name"] = FLUX_LONGNAMES.get(var, var)
        ds[var] = da
        qc_col = f"{var}_QC"
        if qc_col in df_flux_daily.columns:
            qc = xr.DataArray(df_flux_daily[qc_col].values, dims=["date_time"])
            qc.attrs["long_name"] = (
                f"Quality flag for {var}: fraction of measured (QC=0) "
                "sub-daily values contributing to the daily aggregate"
            )
            ds[qc_col] = qc
    return ds


def _flux_longest_valid_period(
    ds: xr.Dataset,
    max_gap_days: int = MAX_GAP_DAYS,
) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Longest consecutive period with LE QC >= MIN_QC_MEASURED, bridging gaps <= max_gap_days."""
    available_qc = [v for v in QC_VARS_FOR_GAP_ANALYSIS if v in ds]
    if not available_qc:
        return 0, None, None

    valid = np.ones(ds.sizes["date_time"], dtype=bool)
    for qc_var in available_qc:
        qc = ds[qc_var].values.astype(float)
        valid &= (~np.isnan(qc)) & (qc >= MIN_QC_MEASURED)

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
# Metadata
# ---------------------------------------------------------------------------
def _site_metadata(
    site_id: str,
    station_row: pd.Series | None,
    site_info_row: pd.Series | None,
) -> dict:
    """Build metadata dict from ICOS_available_station.csv + ICOS_site_info.csv."""
    def _f(row, col, default=float("nan")):
        if row is None or col not in row.index:
            return default
        v = row[col]
        return default if pd.isna(v) else v

    lat  = _f(station_row, "LOCATION_LAT")
    lon  = _f(station_row, "LOCATION_LONG")
    elev = _f(station_row, "LOCATION_ELEV")
    name = _f(station_row, "SITE_NAME", "")
    country      = _f(station_row, "COUNTRY", "")
    icos_class   = _f(station_row, "icosClass", "")
    station_theme = _f(station_row, "stationTheme", "")

    # lat/lon fallback from ICOS_site_info.csv
    if pd.isna(lat):
        lat = _f(site_info_row, "lat")
    if pd.isna(lon):
        lon = _f(site_info_row, "lon")

    igbp = _f(site_info_row, "IGBP", "")

    return {
        "site_id":                 site_id,
        "site_name":               str(name),
        "latitude":                float(lat),
        "longitude":               float(lon),
        "elevation_m":             float(elev),
        "country":                 str(country),
        "icos_class":              str(icos_class),
        "station_theme":           str(station_theme),
        "IGBP":                    str(igbp),
        "source":                  "ICOS FLUXNET L2 (ICOS2025 + ICOS2020)",
        "depth_bins":              "0,10,30,100",
        "min_obs_per_day":         MIN_OBS_PER_DAY,
        "swc_max_gap_days":        7,
        "swc_min_coverage_frac":   0.95,
        "flux_aggregation":        "HH->hourly->daily; QC=fraction of measured hours per day",
        "flux_min_measured_frac":  MIN_MEASURED_FRAC,
        "flux_max_gap_days":       MAX_GAP_DAYS,
        "processing_date":         datetime.now().strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------
def process_site(
    site_id: str,
    station_row: pd.Series | None,
    site_info: pd.DataFrame,
) -> str | None:
    # --- Load and merge HH data ---
    try:
        df_hh = _merge_hh(site_id)
    except Exception as e:
        print(f"  [{site_id}] HH read/merge error: {e}")
        return None
    if df_hh is None or df_hh.empty:
        return None

    # --- Clip to training window ---
    df_hh = df_hh.loc[TRAIN_START:TRAIN_END]
    if df_hh.empty:
        return None

    # --- Skip if not enough years after 2017 clip ---
    if df_hh.index.year.nunique() <= MIN_YEARS_POST_2017:
        return None

    # --- Compute daily flux ---
    df_flux_daily = _flux_hh_to_daily(df_hh)

    has_swc = False
    ds = None

    # --- SWC pipeline ---
    depth_map = _get_swc_depths(site_id, site_info)
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

    # --- Quality check for flux-only stations ---
    if not has_swc:
        L, start, end = _flux_longest_valid_period(ds)
        if L < MIN_VALID_DAYS:
            return None
        ds = ds.sel(date_time=slice(start, end))

    # --- Metadata ---
    site_info_rows = site_info[site_info["site_id"] == site_id]
    site_info_row  = site_info_rows.iloc[0] if not site_info_rows.empty else None
    meta = _site_metadata(site_id, station_row, site_info_row)
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
# Worker  (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> tuple[str, str]:
    site_id, station_row, site_info = args
    try:
        out = process_site(site_id, station_row, site_info)
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
def _discover_sites() -> list[str]:
    """Union of site IDs found in ICOS2025 and ICOS2020 directories."""
    sites_2025 = {d.name for d in ICOS2025_DIR.iterdir() if d.is_dir()}
    sites_2020 = set()
    for d in ICOS2020_DIR.iterdir():
        if d.is_dir() and d.name.startswith("FLX_"):
            parts = d.name.split("_")
            if len(parts) >= 2:
                sites_2020.add(parts[1])
    return sorted(sites_2025 | sites_2020)


def main():
    stations   = pd.read_csv(STATION_CSV).set_index("SITE_ID")
    site_info  = pd.read_csv(SITE_INFO_CSV)
    # FLX15 is a fallback for FLUXNET2015 sites absent from the ICOS station CSVs
    flx15      = pd.read_csv(FLX15_CSV).rename(columns={"site_id": "site_id"})
    site_info  = pd.concat([site_info, flx15[~flx15["site_id"].isin(site_info["site_id"])]], ignore_index=True)
    all_sites  = _discover_sites()

    args = []
    for site_id in all_sites:
        station_row = stations.loc[site_id] if site_id in stations.index else None
        args.append((site_id, station_row, site_info))

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
