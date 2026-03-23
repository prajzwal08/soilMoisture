import numpy as np
import pandas as pd
from pyproj import Transformer

PIXEL_SIZE = 10  # meters (Sentinel-1/2 resolution)
METADATA_PATH = "/home/khanalp/data/soilmoisture/level1/station_metadata.csv"

df = pd.read_csv(METADATA_PATH)
df = df[df["status"] == "saved"].dropna(subset=["latitude", "longitude"]).copy()

# Assign UTM zone and hemisphere vectorially
df["utm_zone"] = ((df["longitude"] + 180) / 6).astype(int) + 1
df["hem"] = np.where(df["latitude"] >= 0, "north", "south")

# Batch-transform per (zone, hemisphere) group — one Transformer per group
eastings  = np.empty(len(df))
northings = np.empty(len(df))

for (zone, hem), idx in df.groupby(["utm_zone", "hem"]).groups.items():
    proj = f"+proj=utm +zone={zone} +{hem} +datum=WGS84"
    t = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    lons = df.loc[idx, "longitude"].values
    lats = df.loc[idx, "latitude"].values
    eastings[df.index.get_indexer(idx)], northings[df.index.get_indexer(idx)] = t.transform(lons, lats)

df["pixel_x"] = (eastings  / PIXEL_SIZE).astype(int)
df["pixel_y"] = (northings / PIXEL_SIZE).astype(int)
df["patch_id"] = list(zip(df["utm_zone"], df["hem"], df["pixel_x"], df["pixel_y"]))

n_stations = len(df)
n_unique = df["patch_id"].nunique()
patch_counts = df.groupby("patch_id").size().rename("n_stations")

print(f"Total stations  : {n_stations}")
print(f"Unique patches  : {n_unique}")
print(f"Shared patches  : {n_stations - n_unique}  (stations that share a pixel with another)")

shared = patch_counts[patch_counts > 1]
if len(shared):
    print(f"\nPatches shared by ≥2 stations ({len(shared)} patches):")
    for pid, n in shared.items():
        stations_here = df[df["patch_id"] == pid][
            ["network", "station", "latitude", "longitude"]
        ]
        print(f"\n  Patch {pid}  ({n} stations):")
        print(stations_here.to_string(index=False))
else:
    print("\nNo shared patches found.")
