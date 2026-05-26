# Explore raw sensor depth distribution across all ISMN stations
# Goal: find natural depth breakpoints to optimise depth binning
# Pipeline applied: QC flags + daily resampling + obs count filter (same as preprocessing)
# Stops BEFORE binning — raw depth_to values are collected here

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
from ismn.interface import ISMN_Interface
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

os.chdir('/home/khanalp/code/PhD/soilMoisture/')

plt.style.use(['science', 'no-latex'])
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load ISMN
ismn_path = "/home/khanalp/data/ISMNsoilMoisture/Data_separate_files_header_20140101_20251231_13107_18mx_20260208"
ds = ISMN_Interface(ismn_path, parallel=True)

output_dir = '/home/khanalp/data/soilmoisture/level1/'
plots_dir  = '/home/khanalp/code/PhD/soilMoisture/plots/'
os.makedirs(plots_dir, exist_ok=True)


def extract_depths_for_station(args):
    """
    Apply QC + daily resampling (steps 1-4 of pipeline), then return
    all unique depth_to values that survive (i.e. have at least one
    valid daily observation). Stops before any binning.
    """
    network, station = args
    try:
        station_data = ds[network][station].to_xarray()

        # Step 1: quality flag filter
        mask = (station_data["soil_moisture_flag"] == "G") | \
               station_data["soil_moisture_flag"].astype(str).str.startswith("D")
        sm = station_data['soil_moisture'].where(mask)

        # Step 2: attach depth_to as coordinate
        sm = sm.assign_coords(depth_group=('sensor', station_data['depth_to'].values))

        # Step 3: group by depth, average across sensors
        depth_avg = sm.groupby('depth_group').mean(dim='sensor', skipna=True)
        depth_avg = depth_avg.rename({'depth_group': 'depth'})

        # Drop depths that are all NaN
        valid = depth_avg.count(dim='date_time') > 0
        depth_avg = depth_avg.where(valid, drop=True)
        if len(depth_avg.depth) == 0:
            return []

        # Step 4: resample to daily, apply obs count filter (>=6)
        daily = depth_avg.resample(date_time='1D').mean(skipna=True)
        count = depth_avg.resample(date_time='1D').count()
        daily_filtered = daily.where(count >= 6)

        # Keep only depths that still have at least one valid daily value
        surviving = daily_filtered.count(dim='date_time') > 0
        surviving_depths = depth_avg.depth.values[surviving.values]

        # Convert to cm if stored as metres (values <= 3 assumed to be in metres)
        depth_vals = surviving_depths.astype(float)
        if len(depth_vals) > 0 and np.nanmax(depth_vals) <= 3:
            depth_vals = depth_vals * 100.0

        return depth_vals.tolist()

    except Exception:
        return []


# Collect all tasks
all_tasks = [
    (network, station)
    for network in ds.collection.networks
    for station in ds.collection[network].stations
]

print(f"Processing {len(all_tasks)} stations with {cpu_count()} CPUs...")

with Pool(80) as pool:
    results = pool.map(extract_depths_for_station, all_tasks)

# Flatten all depth values
all_depths = np.array([d for station_depths in results for d in station_depths])
print(f"Total sensor-depth observations collected: {len(all_depths)}")

# Save raw depths for later use
np.save(os.path.join(output_dir, 'raw_sensor_depths_cm.npy'), all_depths)
print(f"Saved raw depths to {output_dir}raw_sensor_depths_cm.npy")

# --- Unique depth counts ---
unique_depths, counts = np.unique(np.round(all_depths, 1), return_counts=True)
df_depths = pd.DataFrame({'depth_cm': unique_depths, 'n_stations': counts})
df_depths = df_depths.sort_values('depth_cm')
df_depths.to_csv(os.path.join(output_dir, 'raw_sensor_depth_counts.csv'), index=False)
print(f"\nTop 30 most common depths:")
print(df_depths.sort_values('n_stations', ascending=False).head(30).to_string(index=False))

# --- Plot 1: Histogram of all raw depths ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(all_depths, bins=200, color='steelblue', edgecolor='none', alpha=0.8)
ax.set_xlabel('Sensor depth (cm)')
ax.set_ylabel('Count')
ax.set_title('Raw sensor depth distribution (after QC + daily filter, before binning)')
ax.set_xlim(0, 200)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'raw_depth_histogram.svg'))
plt.close()
print("Saved: plots/raw_depth_histogram.svg")

# --- Plot 2: Bar chart of top 40 most common depths ---
top40 = df_depths.sort_values('n_stations', ascending=False).head(40).sort_values('depth_cm')
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(top40['depth_cm'].astype(str), top40['n_stations'], color='steelblue', edgecolor='none')
ax.set_xlabel('Sensor depth (cm)')
ax.set_ylabel('Number of station-sensors')
ax.set_title('Most common sensor depths across all ISMN stations')
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'raw_depth_bar.svg'))
plt.close()
print("Saved: plots/raw_depth_bar.svg")

# --- Plot 3: Cumulative distribution ---
sorted_depths = np.sort(all_depths)
cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sorted_depths, cdf, color='steelblue', lw=2)
ax.set_xlabel('Sensor depth (cm)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Cumulative distribution of sensor depths')
ax.set_xlim(0, 200)
ax.axvline(10,  color='grey', ls='--', lw=1, label='10 cm')
ax.axvline(30,  color='orange', ls='--', lw=1, label='30 cm')
ax.axvline(50,  color='red', ls='--', lw=1, label='50 cm')
ax.axvline(100, color='purple', ls='--', lw=1, label='100 cm')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'raw_depth_cdf.svg'))
plt.close()
print("Saved: plots/raw_depth_cdf.svg")

print("\nDone. Inspect the plots and raw_sensor_depth_counts.csv to choose new bin boundaries.")
