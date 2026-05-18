# ============================================================
# ISMN Soil Moisture Preprocessing Pipeline — v1.1
# ============================================================
# Reads raw ISMN hourly data, applies quality control, resamples
# to daily, bins sensors into standard depth layers, removes long
# gaps, trims to the surface valid period, gap-fills with
# climatological means, and saves one NetCDF per station.
#
# Output: /home/khanalp/data/soilmoisture/level1/
# Depth bins: 0-10, 10-30, 30-100 cm
# ============================================================

# --- Libraries ---
import os
import sys
import ismn
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime
import warnings
from ismn.interface import ISMN_Interface
import random

# --- Plot style (publication template) ---
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

os.chdir('/home/khanalp/code/PhD/soilMoisture/')
from utils import longest_available_after_removing_long_gaps, \
                  trim_to_surface_valid_period_and_keep_well_covered_depths, \
                  gapfill_by_monthday_mean_with_feb29_fallback

plt.style.use(['science', 'no-latex'])
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# --- Paths ---
insitu_dir = '/home/khanalp/data/ISMNsoilMoisture/'
output_dir = Path('/home/khanalp/data/soilmoisture/level1/')
output_dir.mkdir(exist_ok=True)

# --- Load ISMN data ---
ds = ISMN_Interface(
    "/home/khanalp/data/ISMNsoilMoisture/Data_separate_files_header_20140101_20251231_13107_18mx_20260208",
    parallel=True
)

# Build list of all (network, station) pairs
networks_stations = []
for network in ds.collection.networks:
    for station in ds.collection[network].stations:
        networks_stations.append({'network': network, 'station': station})


# ============================================================
# process_station()
# ============================================================
# Applies QC, depth averaging, daily resampling, and depth
# binning for a single station. Steps:
#   1. Keep measurements with flag G (Good) or D* (Dubious).
#   2. Group sensors by depth_to and average across sensors.
#      (depth_to used instead of depth_from — some stations
#       have depth_from=0 for all sensors.)
#   3. Drop depths that are entirely NaN after QC.
#   4. Resample hourly → daily mean; count valid obs per day.
#   5. Mask days with fewer than 6 valid sub-daily observations.
#   6. Bin sensor depths into standard layers and average within
#      each bin: 0-10, 10-30, 30-100 cm.
#      (Bins chosen to match natural sensor placement clusters.)
#   7. Return xr.Dataset with soil_moisture + station attrs.
#      Returns None if no valid depths survive.
# ============================================================
def process_station(station_data, network, station):
    # Step 1: quality flag filter — keep G and D* flags
    mask = (station_data["soil_moisture_flag"] == "G") | \
           station_data["soil_moisture_flag"].astype(str).str.startswith("D")
    soil_moisture_masked = station_data['soil_moisture'].where(mask)

    # Step 2: group sensors by depth_to, average across sensors at same depth
    soil_moisture_masked = soil_moisture_masked.assign_coords(
        depth_group=('sensor', station_data['depth_to'].values)
    )
    depth_averaged = soil_moisture_masked.groupby('depth_group').mean(dim='sensor', skipna=True)
    depth_averaged = depth_averaged.rename({'depth_group': 'depth'})

    # Step 3: drop depths that are all NaN
    valid_count_per_depth = depth_averaged.count(dim='date_time')
    depth_averaged = depth_averaged.where(valid_count_per_depth > 0, drop=True)
    if len(depth_averaged.depth) == 0:
        return None

    # Step 4: resample to daily mean and count valid obs per day
    daily = depth_averaged.resample(date_time='1D').mean(dim='date_time', skipna=True)
    count = depth_averaged.resample(date_time='1D').count(dim='date_time')
    try:
        count = count.fillna(0).astype(int)
    except:
        count = count.astype(float)

    # Step 5: mask days with < 6 valid sub-daily observations
    daily_filtered = daily.where(count >= 6)

    # Step 6: bin sensor depths into standard layers
    depth_vals = daily_filtered["depth"].values.astype(float)
    depth_cm = depth_vals * 100.0 if np.nanmax(depth_vals) <= 3 else depth_vals

    depth_bin = pd.cut(
        depth_cm,
        bins=[0.0, 10.0, 30.0, 100.0],
        labels=["0-10", "10-30", "30-100"],
        right=True,
        include_lowest=True
    )
    # Drop sensors outside the defined bins (depth > 100 cm → NaN label)
    valid_sensor_mask = ~pd.isna(depth_bin)  # numpy bool array
    daily_filtered = daily_filtered.isel(depth=valid_sensor_mask)
    if len(daily_filtered.depth) == 0:
        return None
    depth_bin_valid = depth_bin[valid_sensor_mask]
    daily_filtered = daily_filtered.assign_coords(depth_bin=("depth", depth_bin_valid.astype(str)))
    daily_binned = (
        daily_filtered.groupby("depth_bin")
        .mean(dim="depth", skipna=True)
        .rename({"depth_bin": "depth"})
    )

    # Step 7: return dataset with station metadata attributes
    result_ds = xr.Dataset({"soil_moisture": daily_binned})
    result_ds.attrs['network']   = network
    result_ds.attrs['station']   = station
    result_ds.attrs['latitude']  = float(station_data.attrs.get('lat', np.nan))
    result_ds.attrs['longitude'] = float(station_data.attrs.get('lon', np.nan))
    result_ds.attrs["max_depth"] = float(np.nanmax(depth_vals))
    return result_ds


# ============================================================
# process_single_station()
# ============================================================
# Wrapper for parallel execution. Calls process_station() then
# applies the full quality pipeline:
#   1. Gap analysis: find longest continuous surface run
#      (gaps <=7 days are bridged).
#   2. Trim: anchor time window to surface (0-10 cm) valid
#      period; drop non-surface depths with <95% coverage.
#   3. Duration filter: skip if <1095 valid days remain (3 years,
#      consistent with ICOS/AmeriFlux MIN_VALID_DAYS; ensures ≥2 years
#      of valid training targets after the 365-day context window).
#   4. Gap-fill: climatological month-day means; Feb-29
#      fallback to Feb-28 then Mar-01.
#   5. Write NetCDF with metadata attrs and save station record.
#
# Returns a dict with status (saved / skipped / error) and
# metadata for the station_metadata.csv.
# ============================================================
def process_single_station(args):
    network, station, idx, total = args

    def out(status: str, remark: str, **extra):
        base = {"network": network, "station": station, "status": status, "remark": remark}
        base.update(extra)
        return base

    try:
        print(f"[{idx+1}/{total}] Processing: {network}/{station}")

        station_data = ds[network][station].to_xarray()
        result_ds = process_station(station_data, network, station)

        if result_ds is None:
            return out("skipped", "no valid data after process_station",
                       latitude=np.nan, longitude=np.nan)

        lat = float(result_ds.attrs.get("latitude", np.nan))
        lon = float(result_ds.attrs.get("longitude", np.nan))
        surface_depth = "0-10"

        # Step 1: gap analysis — find longest continuous run per depth
        longest_available = longest_available_after_removing_long_gaps(result_ds, max_gap_days=7)
        L, start, end = longest_available.get(surface_depth, (0, None, None))
        if (L is None) or (L == 0) or (start is None) or (end is None):
            return out("skipped", f"no surface run for {surface_depth} after removing long gaps",
                       latitude=lat, longitude=lon)

        # Step 2: trim to surface valid period; drop under-covered depths
        ds_clean = trim_to_surface_valid_period_and_keep_well_covered_depths(
            result_ds, longest_available, surface_depth=surface_depth, min_frac=0.95
        )
        if ds_clean is None:
            return out("skipped", "no clean data after trimming/coverage filter",
                       latitude=lat, longitude=lon)

        # Compute date range from observed (pre-gap-fill) data
        valid_dates = ds_clean["soil_moisture"].dropna(dim="date_time", how="all").date_time
        if len(valid_dates) > 0:
            start_date = pd.to_datetime(valid_dates.min().values).strftime("%Y%m%d")
            end_date   = pd.to_datetime(valid_dates.max().values).strftime("%Y%m%d")
        else:
            start_date, end_date = None, None

        # Step 3: duration filter — require at least 3 years of valid data
        if len(valid_dates) < 1095:
            return out("skipped", "<3 years of valid daily data",
                       latitude=lat, longitude=lon,
                       start_date=start_date, end_date=end_date,
                       n_days=int(len(valid_dates)))

        # Step 4: gap-fill using climatological month-day means
        ds_gap_filled = gapfill_by_monthday_mean_with_feb29_fallback(ds_clean)
        if ds_gap_filled["soil_moisture"].isnull().any().item():
            return out("skipped", "NaNs remain after gap-filling",
                       latitude=lat, longitude=lon,
                       start_date=start_date, end_date=end_date,
                       n_days=int(len(valid_dates)))

        # Step 5: write metadata attrs to NetCDF

        # Temporal provenance
        ds_gap_filled.attrs['start_date']      = start_date
        ds_gap_filled.attrs['end_date']        = end_date
        ds_gap_filled.attrs['n_days_total']    = int(len(ds_gap_filled.date_time))
        ds_gap_filled.attrs['processing_date'] = datetime.now().strftime("%Y-%m-%d")

        # Processing provenance
        ds_gap_filled.attrs['depth_bins']        = "0,10,30,100"
        ds_gap_filled.attrs['min_obs_per_day']   = 6
        ds_gap_filled.attrs['max_gap_days']      = 7
        ds_gap_filled.attrs['min_coverage_frac'] = 0.95
        ds_gap_filled.attrs['pipeline_version']  = "1.1"

        # Per-depth quality stats (n_observed, n_gapfilled, frac_observed)
        # qc flag: 0=observed, 1=gap-filled, 2=still missing
        qc = ds_gap_filled["soil_moisture_qc"]
        for depth in ds_gap_filled.depth.values:
            q        = qc.sel(depth=depth)
            n_obs    = int((q == 0).sum().item())
            n_filled = int((q == 1).sum().item())
            n_total  = int((q != 2).sum().item())
            frac_obs = round(n_obs / n_total, 4) if n_total > 0 else 0.0
            d = str(depth).replace("-", "_").replace(">", "gt")
            ds_gap_filled.attrs[f'n_observed_{d}']    = n_obs
            ds_gap_filled.attrs[f'n_gapfilled_{d}']   = n_filled
            ds_gap_filled.attrs[f'frac_observed_{d}'] = frac_obs

        # Variable-level attributes
        ds_gap_filled["soil_moisture"].attrs.update({
            "units":         "m3 m-3",
            "long_name":     "Volumetric soil moisture content",
            "standard_name": "volume_fraction_of_condensed_water_in_soil",
            "valid_range":   [0.0, 1.0],
        })
        ds_gap_filled["soil_moisture_qc"].attrs.update({
            "long_name":     "Soil moisture quality control flag",
            "flag_values":   "0 1 2",
            "flag_meanings": "observed gap_filled missing",
        })

        filename = f"{network}_{station}_{start_date}_{end_date}.nc"
        filepath = output_dir / filename
        ds_gap_filled.to_netcdf(filepath)

        return out(
            "saved", "",
            latitude=lat, longitude=lon,
            start_date=start_date, end_date=end_date,
            n_days=int(len(valid_dates)),
            max_depth_cm=float(result_ds.attrs.get("max_depth", np.nan)) * 100.0,
            depths=ds_gap_filled["depth"].values.tolist(),
            filename=filename, filepath=str(filepath),
        )

    except Exception as e:
        return out("error", str(e), latitude=np.nan, longitude=np.nan)


# ============================================================
# Collect all station tasks
# ============================================================
all_tasks = [
    (network, station)
    for network in ds.collection.networks
    for station in ds.collection[network].stations
]

# ============================================
# CONFIGURE TEST RUN HERE
# ============================================
TEST_MODE       = False   # True = random sample of N_TEST_STATIONS
N_TEST_STATIONS = 100

if TEST_MODE:
    rng   = random.Random(42)
    tasks = rng.sample(all_tasks, k=min(N_TEST_STATIONS, len(all_tasks)))
    print(f"=== TEST MODE: Processing random {len(tasks)} stations ===")
else:
    tasks = all_tasks
    print(f"=== FULL MODE: Processing all {len(tasks)} stations ===")

tasks_with_idx = [(net, sta, i, len(tasks)) for i, (net, sta) in enumerate(tasks)]
print(f"Output directory: {output_dir}/")
print(f"Using {cpu_count()} CPU cores available")

# ============================================
# CONFIGURE PARALLELIZATION HERE
# ============================================
USE_PARALLEL = True   # False = sequential (easier for debugging)
N_WORKERS    = 100

if USE_PARALLEL:
    print(f"Running in parallel with {N_WORKERS} workers")
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_single_station, tasks_with_idx)
else:
    print("Running sequentially (no parallelization)")
    results = [process_single_station(task) for task in tasks_with_idx]

# ============================================================
# Save metadata CSV
# ============================================================
metadata_list = [r for r in results if r is not None]

if len(metadata_list) > 0:
    df_metadata = pd.DataFrame(metadata_list)
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {len(df_metadata)}/{len(tasks)} stations")
    print(f"Files saved to: {output_dir}/")
    metadata_file = output_dir / 'station_metadata.csv'
    df_metadata.to_csv(metadata_file, index=False)
    print(f"Metadata saved to: {metadata_file}")
else:
    print("\nNo valid data found across all stations")
