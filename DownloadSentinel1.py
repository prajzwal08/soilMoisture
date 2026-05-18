"""
Download first 10 Sentinel-1 images for a location
Cropped to 264x264 pixels at 10m resolution
Using Microsoft Planetary Computer (better S1 metadata)
"""
import pystac_client
import planetary_computer
import stackstac
import numpy as np
from pathlib import Path
import xarray as xr
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
STATION_NAME = "Twente_01"
LATITUDE = 52.2379
LONGITUDE = 6.8564

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

OUTPUT_DIR = "./sentinel_data"
PIXEL_SIZE = 264
OUT_EPSG = 32632

# ============================================================================
# FUNCTIONS
# ============================================================================

def bbox_for_pixels(lon, lat, pixels=264, res_m=10):
    """Calculate bbox for exact pixel dimensions"""
    half_m = (pixels * res_m) / 2
    lat_buf = half_m / 111320
    lon_buf = half_m / (111320 * np.cos(np.deg2rad(lat)))
    return [lon - lon_buf, lat - lat_buf, lon + lon_buf, lat + lon_buf]

def center_crop(data, target_size=264):
    """Crop center to target size"""
    _, _, h, w = data.shape
    
    if h < target_size or w < target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        data = np.pad(data, ((0,0), (0,0), (pad_h//2, pad_h-pad_h//2), 
                            (pad_w//2, pad_w-pad_w//2)), mode='edge')
        h, w = data.shape[2:]
    
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    return data[:, :, start_h:start_h+target_size, start_w:start_w+target_size]

# ============================================================================
# MAIN
# ============================================================================

output_path = Path(OUTPUT_DIR) / STATION_NAME
output_path.mkdir(parents=True, exist_ok=True)

bbox = bbox_for_pixels(LONGITUDE, LATITUDE, pixels=PIXEL_SIZE, res_m=10)

print("Connecting to Microsoft Planetary Computer...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace  # Handles auth automatically
)
print("✓ Connected\n")

print(f"Searching S1 data from {START_DATE} to {END_DATE}...")
s1_search = catalog.search(
    collections=["sentinel-1-rtc"],
    bbox=bbox,
    datetime=f"{START_DATE}/{END_DATE}"
)

s1_items_all = list(s1_search.items())
print(f"✓ Found {len(s1_items_all)} total scenes\n")

if len(s1_items_all) == 0:
    print("❌ No Sentinel-1 data found")
    exit()

# Group by date
items_by_date = defaultdict(list)
for item in s1_items_all:
    date = item.datetime.date()
    items_by_date[date].append(item)

sorted_dates = sorted(items_by_date.keys())[:10]
print(f"Selecting first 10 dates:")
for i, date in enumerate(sorted_dates, 1):
    print(f"  {i}. {date}")

selected_items = [items_by_date[date][0] for date in sorted_dates]
print(f"\n✓ Selected {len(selected_items)} images\n")

print("Loading S1 data...")
s1 = stackstac.stack(
    selected_items,
    assets=["vh", "vv"],  # Note: order matters for some catalogs
    bounds_latlon=bbox,
    epsg=OUT_EPSG,
    resolution=10,
    snap_bounds=True,
    dtype="float32",
    fill_value=np.float32(np.nan),
    chunksize=128,
    rescale=False,
)

print(f"Stack shape (before crop): {s1.shape}")
print("Computing (downloading data)...")

s1_data = s1.compute()
print(f"✓ Data loaded: {s1_data.shape}\n")

print(f"Cropping to {PIXEL_SIZE}x{PIXEL_SIZE}...")
s1_cropped = center_crop(s1_data.values, PIXEL_SIZE)
print(f"✓ Cropped shape: {s1_cropped.shape}\n")

s1_out = xr.DataArray(
    s1_cropped,
    dims=['time', 'band', 'y', 'x'],
    coords={
        'time': s1_data.time.values,
        'band': ['VH', 'VV']
    },
    attrs={
        'description': 'Sentinel-1 GRD backscatter',
        'station': STATION_NAME,
        'lat': LATITUDE,
        'lon': LONGITUDE,
        'resolution_m': 10,
        'pixel_size': PIXEL_SIZE
    }
)

output_file = output_path / f"s1_first10.nc"
s1_out.to_netcdf(output_file)

print("="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print(f"File: {output_file}")
print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Shape: {s1_cropped.shape}")
print(f"Dates: {[str(d)[:10] for d in s1_data.time.values]}")
print(f"\nTo load:")
print(f"  s1 = xr.open_dataarray('{output_file}')")

import matplotlib.pyplot as plt

# Get first timestep (both bands)
first_image = s1_out.isel(time=0)

# Plot both bands
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# VH band
im1 = axes[0].imshow(first_image.sel(band='VH'), cmap='gray')
axes[0].set_title(f'VH - {str(first_image.time.values)[:10]}')
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# VV band
im2 = axes[1].imshow(first_image.sel(band='VV'), cmap='gray')
axes[1].set_title(f'VV - {str(first_image.time.values)[:10]}')
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(output_path / 'first_image.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Plot saved to: {output_path / 'first_image.png'}")