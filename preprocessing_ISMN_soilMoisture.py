# Libraries
import os 
import sys
import ismn
import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool, cpu_count # 
from pathlib import Path
from datetime import datetime
import warnings
from ismn.interface import ISMN_Interface
import random


# ---------------- Plot style (my template) ----------------
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots  # registers 'science', 'no-latex', etc.

os.chdir('/home/khanalp/code/PhD/soilMoisture/')
from utils import longest_available_after_removing_long_gaps,trim_to_surface_valid_period_and_keep_well_covered_depths,gapfill_by_monthday_mean_with_feb29_fallback

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


# Insitu data directory (adjust as needed)
insitu_dir = '/home/khanalp/data/ISMNsoilMoisture/'

# Create output directory
output_dir = '/home/khanalp/code/PhD/soilMoisture/processed_soil_moisture/'
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)



# Read the data using ISMN_Interface
ds = ISMN_Interface("/home/khanalp/data/ISMNsoilMoisture/Data_separate_files_header_20140101_20251231_13107_18mx_20260208", parallel=True)

# List networks and stations using modern ISMN interface
networks_stations = []
for network in ds.collection.networks:
    for station in ds.collection[network].stations:
        networks_stations.append({
            'network': network,
            'station': station
        })


"""
To learn about ISMN data quality flag: check https://ismn.earth/en/data/flag-overview/
In short, G = good, D = Dubious, C= outside plausible range.

Inside process_station, \
1. Create a mask for good-quality measurements: soil_moisture_flag == "G".
2. Apply the mask to soil_moisture so non-G values become NaN.
3. Add a coordinate depth_group (per sensor) using depth_from so sensors can be grouped by depth.
4. Group by depth_group and take the mean across sensors (depth-average), ignoring NaNs.
5. Rename depth_group to depth.
6. Compute how many valid values each depth has over all times; drop depths that are all NaN.
7. If no depths remain, return None.
8. Resample to daily mean soil moisture (per depth).
9. Resample to a daily count of valid observations (per depth).
10.Try to convert the daily count to an integer (fallback to a float if conversion fails).
11. Mask daily soil moisture where the daily count is < 6 (those days become NaN).
12. Return an xarray.Dataset with:
13. soil_moisture (daily, filtered by count)
14. observation_count (daily count)
Add metadata attributes: network, station, latitude, and longitude.
"""

# Function to process each station
def process_station(station_data, network, station):
    """
    Process a single station dataset:
    - Depth average soil_moisture where flag = 'G'
    - Drop depths with no valid data
    - Convert to daily if >= 6 valid observations per day
    """
    # Create mask where soil_moisture_flag == "G" and also include "D" (Dubious) flags as valid data.
    # mask = station_data['soil_moisture_flag'] == "G"
    mask = (station_data["soil_moisture_flag"] == "G") | station_data["soil_moisture_flag"].astype(str).str.startswith("D")
    
    # Apply mask to soil_moisture
    soil_moisture_masked = station_data['soil_moisture'].where(mask)
    
    # Assign depth_from as a coordinate for grouping
    soil_moisture_masked = soil_moisture_masked.assign_coords(
        depth_group=('sensor', station_data['depth_to'].values) # Use depth_to instead of depth_from because some stations have depth_from = 0 for all sensors, but depth_to varies and can be used to group sensors by depth.
    )
    
    # Group by depth and average across sensors
    depth_averaged = soil_moisture_masked.groupby('depth_group').mean(dim='sensor', skipna=True)
    depth_averaged = depth_averaged.rename({'depth_group': 'depth'})
    
    # Drop depths that have ALL NaN values (no valid data)
    valid_count_per_depth = depth_averaged.count(dim='date_time')
    depths_with_data = valid_count_per_depth > 0
    depth_averaged = depth_averaged.where(depths_with_data, drop=True)
    
    # If no valid depths remain, return None
    if len(depth_averaged.depth) == 0:
        return None
    
    # Resample to daily
    daily = depth_averaged.resample(date_time='1D').mean(dim='date_time', skipna=True)
    
    # Count valid observations per day
    count = depth_averaged.resample(date_time='1D').count(dim='date_time')
    
    # Handle the casting more gracefully
    try:
        count = count.fillna(0).astype(int)
    except:
        count = count.astype(float)
    
    # Mask out days with < 6 valid observations
    daily_filtered = daily.where(count >= 6)
    
        # ---- depth-bin averaging AFTER daily filtering (no count saved) ----
    depth_vals = daily_filtered["depth"].values.astype(float)
    depth_cm = depth_vals * 100.0 if np.nanmax(depth_vals) <= 3 else depth_vals

    depth_bin = pd.cut(
        depth_cm,
        bins=[0.0, 10.0, 30.0, 50.0, np.inf],
        labels=["0-10", "10-30", "30-50", ">50"],
        right=True,
        include_lowest=True
    )

    daily_filtered = daily_filtered.assign_coords(depth_bin=("depth", depth_bin.astype(str)))

    daily_binned = (
        daily_filtered.groupby("depth_bin")
        .mean(dim="depth", skipna=True)
        .rename({"depth_bin": "depth"})
    )

    result_ds = xr.Dataset({"soil_moisture": daily_binned})
    
    # Add metadata as attributes
    result_ds.attrs['network'] = network
    result_ds.attrs['station'] = station
    result_ds.attrs['latitude'] = float(station_data.attrs.get('lat', np.nan)) #The attrs lat and variables latitude are latitude.
    result_ds.attrs['longitude'] = float(station_data.attrs.get('lon', np.nan)) # same for lon. 
    result_ds.attrs["max_depth"] = float(np.nanmax(depth_vals))
    return result_ds

'''
## 'process_single_station' is just a wrapper for parallel processing. 
What really happens is:
1. station data is read into xarray. 
2. then process_station is called, which does QC, filtering, converts hourly to daily, which returns result_ds. 
3. If result_ds is None there we skip to other stations, if not
4. We get metadata like network, stations, longitude, latitude, depths, start_date, end_date, etc.
5. Filename is saved as f"{network}_{station}_{start_date}_{end_date}.nc" in output_dir.
6. Function returns metadata.
'''

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

        # Always available for saved/skipped from here
        lat = float(result_ds.attrs.get("latitude", np.nan))
        lon = float(result_ds.attrs.get("longitude", np.nan))

        surface_depth = "0-10"

        longest_available = longest_available_after_removing_long_gaps(result_ds, max_gap_days=7)
        L, start, end = longest_available.get(surface_depth, (0, None, None))
        if (L is None) or (L == 0) or (start is None) or (end is None):
            return out("skipped", f"no surface run for {surface_depth} after removing long gaps",
                       latitude=lat, longitude=lon)

        ds_clean = trim_to_surface_valid_period_and_keep_well_covered_depths(
            result_ds, longest_available, surface_depth=surface_depth, min_frac=0.95
        )
        if ds_clean is None:
            return out("skipped", "no clean data after trimming/coverage filter",
                       latitude=lat, longitude=lon)

        valid_dates = ds_clean["soil_moisture"].dropna(dim="date_time", how="all").date_time

        # compute dates if any exist (even if < 1 year)
        if len(valid_dates) > 0:
            start_date = pd.to_datetime(valid_dates.min().values).strftime("%Y%m%d")
            end_date = pd.to_datetime(valid_dates.max().values).strftime("%Y%m%d")
        else:
            start_date, end_date = None, None

        if len(valid_dates) < 365:
            return out("skipped", "<1 year of valid daily data",
                       latitude=lat, longitude=lon,
                       start_date=start_date, end_date=end_date,
                       n_days=int(len(valid_dates)))

        ds_gap_filled = gapfill_by_monthday_mean_with_feb29_fallback(ds_clean)

        if ds_gap_filled["soil_moisture"].isnull().any().item():
            return out("skipped", "NaNs remain after gap-filling",
                       latitude=lat, longitude=lon,
                       start_date=start_date, end_date=end_date,
                       n_days=int(len(valid_dates)))

        filename = f"{network}_{station}_{start_date}_{end_date}.nc"
        filepath = output_dir / filename
        ds_gap_filled.to_netcdf(filepath)

        return out(
            "saved", "",
            latitude=lat, longitude=lon,
            start_date=start_date, end_date=end_date,
            n_days=int(len(valid_dates)),
            max_depth_cm=float(result_ds.attrs.get("max_depth", np.nan)) * 100.0,
            filename=filename, filepath=str(filepath),
        )

    except Exception as e:
        # If you want lat/lon even on errors, you can’t guarantee it exists.
        # But you can still return NaN placeholders.
        return out("error", str(e), latitude=np.nan, longitude=np.nan)

# Collect all station tasks
all_tasks = []
for network in ds.collection.networks:
    for station in ds.collection[network].stations:
        all_tasks.append((network, station))
        

# ============================================
# CONFIGURE TEST RUN HERE
# ============================================
TEST_MODE = True
N_TEST_STATIONS = 100

if TEST_MODE:
    # reproducible random sample
    rng = random.Random(42)
    tasks = rng.sample(all_tasks, k=min(N_TEST_STATIONS, len(all_tasks)))
    print(f"=== TEST MODE: Processing random {len(tasks)} stations ===")
else:
    tasks = all_tasks
    print(f"=== FULL MODE: Processing all {len(tasks)} stations ===")

# Add index and total count to tasks
tasks_with_idx = [(net, sta, i, len(tasks)) for i, (net, sta) in enumerate(tasks)]

print(f"Output directory: {output_dir}/")
print(f"Using {cpu_count()} CPU cores available")


# ============================================
# CONFIGURE PARALLELIZATION HERE
# ============================================
USE_PARALLEL = True  # Set to False for sequential (easier debugging)
N_WORKERS = 50  # Number of parallel workers (adjust as needed)

if USE_PARALLEL:
    print(f"Running in parallel with {N_WORKERS} workers")
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_single_station, tasks_with_idx)
else:
    print("Running sequentially (no parallelization)")
    results = [process_single_station(task) for task in tasks_with_idx]


# Filter out None results
metadata_list = [r for r in results if r is not None]

# Create metadata DataFrame
if len(metadata_list) > 0:
    df_metadata = pd.DataFrame(metadata_list)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {len(df_metadata)}/{len(tasks)} stations")
    print(f"Files saved to: {output_dir}/")
    print(f"\nMetadata summary:")
    print(df_metadata.head(20))
    
    # Save metadata
    metadata_file = output_dir / 'station_metadata.csv'
    df_metadata.to_csv(metadata_file, index=False)
    print(f"\nMetadata saved to: {metadata_file}")
    
    # # Print summary statistics
    # print(f"\nSummary:")
    # print(f"  Networks: {df_metadata['network'].nunique()}")
    # print(f"  Stations: {len(df_metadata)}")
    # print(f"  Date range: {df_metadata['start_date'].min()} to {df_metadata['end_date'].max()}")
    # print(f"  Depth range: {df_metadata['n_depths'].min()}-{df_metadata['n_depths'].max()} depths per station")
    
else:
    print("\nNo valid data found across all stations")

