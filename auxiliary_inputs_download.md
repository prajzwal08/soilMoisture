# Auxiliary Inputs Download

Documentation for three auxiliary model inputs: TWSA, SIF, and OpenLandMap soil properties.

---

## 1. TWSA — Terrestrial Water Storage Anomaly (GRACE/GRACE-FO)

### Goal

Extract monthly TWSA at each station's lat/lon from the JPL mascon solution.
One NetCDF file per station per year, containing 12 monthly values.

### Data Source

- **Collection**: `NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI` on Google Earth Engine
- **Product**: JPL RL06.3 Mascon, CRI-filtered (Coastal Resolution Improvement)
- **Resolution**: 0.5° grid (~55 km spacing, effective ~300 km mascon size)
- **Coverage**: April 2002 – September 2024 (monthly, forward-streaming)
- **Auth**: Existing GEE credentials (same as ERA5-Land) — no new registration needed
- **CRI filter**: Removes land/ocean signal leakage near coastlines — recommended for land stations

### Variables

| Output variable | GEE band | Units | Notes |
|---|---|---|---|
| `lwe` | `lwe_thickness` | cm EWT | Liquid water equivalent thickness anomaly |
| `lwe_uncertainty` | `uncertainty` | cm EWT | 1-sigma measurement uncertainty |

**Total: 2 variables × 12 monthly values per year**

### Temporal notes

- Monthly values assigned to **mid-month DoY** for positional encoding:
  Jan=15, Feb=46, Mar=74, Apr=105, May=135, Jun=166, Jul=196, Aug=227, Sep=258, Oct=288, Nov=319, Dec=349
- **GRACE/GRACE-FO gap**: August 2017 – May 2018 (no satellite data, ~11 months)
  → no TWSA token for those months; model handles sparse modality naturally

### Flowchart

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  ee.Initialize()  (uses cached credentials)          │
│  Collection: NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI    │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  load_stations()                                     │
│  Read station_splits.csv — all 1,048 stations        │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  build_job_list()                                    │
│  For each station × year in [2002..2024]:            │
│    output = TWSA/twsa_{year}.nc                      │
│    if exists → SKIP (checkpoint)                     │
│    else → add job {station_id, lat, lon, year}       │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  ThreadPoolExecutor (N_WORKERS=10)                   │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼  [per job: one station × one year]
┌──────────────────────────────────────────────────────┐
│  process_station_year(job)                           │
│                                                      │
│  geometry = ee.Geometry.Point([lon, lat])            │
│                                                      │
│  ImageCollection                                     │
│    .filterDate(year_start, year_end)                 │
│    .select(["lwe_thickness", "uncertainty"])         │
│    .getRegion(geometry, scale=55000)                 │
│  → parse list-of-lists → DataFrame (time, lwe, unc) │
│                                                      │
│  Assign mid-month DoY per record                     │
│  Convert to xr.Dataset, shape (time=≤12,)            │
│  Save → TWSA/twsa_{year}.nc                         │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Append result to twsa_download_log.csv              │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼  (repeat for all jobs)
                          END
```

### Output

```
/home/khanalp/data/satellite/
  {network}_{station}/
    TWSA/
      twsa_{YYYY}.nc    ← lwe + lwe_uncertainty, shape (time=≤12,)
```

### Script

`download_twsa_gee.py`

### Usage

```bash
# Full download (background)
nohup conda run -n geo python download_twsa_gee.py \
  > /tmp/download_twsa.log 2>&1 &

# Monitor
tail -f /tmp/download_twsa.log
```

---

## 2. SIF — Solar-Induced Fluorescence (TROPOMI / TROPOSIF)

### Goal

Extract daily SIF observations at each station's lat/lon from the TROPOSIF L2B product.
One NetCDF file per station per year, containing all valid (cloud-free) daily observations.

### Data Source

- **Collection**: `L2B_SIF___` on S5P-PAL STAC API
- **Product**: TROPOSIF L2B daily — all valid orbits within a day merged, quality-filtered
- **STAC endpoint**: `https://data-portal.s5p-pal.com/api/s5p-l2`
- **Resolution**: ~3.5 km pixel footprint (ungridded point observations)
- **Coverage**: April 2018 – present (~1-4 week latency)
- **Auth**: None — publicly accessible, no registration required
- **L2 vs L2B**: L2B chosen — pre-filtered to valid cloud-free retrievals; one file/day vs ~14 orbits/day for L2

### Variables

| Output variable | Source variable | Units | Notes |
|---|---|---|---|
| `sif` | `SIF_743` | mW/m²/sr/nm | SIF radiance at 740 nm (743–758 nm fitting window) |
| `sif_uncertainty` | `SIF_743_uncertainty` | mW/m²/sr/nm | 1-sigma retrieval uncertainty |

**Total: 2 variables × ~40-50 valid days per year per station (cloud-free only)**

### Temporal notes

- SIF is a **sparse modality**: only cloud-free daytime overpasses produce valid values
- Average ~40-50 valid observations/year per station (highly variable by climate)
- Missing days have no token in the temporal transformer — sparse injection handled by model
- Modality dropout p=0.5 during training

### Flowchart

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  pystac_client.Client.open(S5P_PAL_STAC_URL)        │
│  No authentication required                          │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  load_stations()                                     │
│  Read station_splits.csv — all 1,048 stations        │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  build_job_list()                                    │
│  For each station × year in [2018..present]:         │
│    output = SIF/sif_{year}.nc                        │
│    if exists → SKIP (checkpoint)                     │
│    else → add job {station_id, lat, lon, year}       │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  ThreadPoolExecutor (N_WORKERS=5)                    │
│  (conservative — HTTP requests to S5P-PAL)           │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼  [per job: one station × one year]
┌──────────────────────────────────────────────────────┐
│  process_station_year(job)                           │
│                                                      │
│  For each day in year:                               │
│    STAC search: collection=L2B_SIF___, bbox=station  │
│    bbox = station ± 0.05° (~5.5 km buffer)           │
│    → get download URL for L2B daily file             │
│    → open NetCDF, filter pixels within 3.5 km radius │
│    → if valid pixels exist: take mean SIF + unc      │
│    → append to daily records                         │
│                                                      │
│  Convert valid days → xr.Dataset, shape (time=N,)    │
│  Save → SIF/sif_{year}.nc                           │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Append result to sif_download_log.csv               │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼  (repeat for all jobs)
                          END
```

### Output

```
/home/khanalp/data/satellite/
  {network}_{station}/
    SIF/
      sif_{YYYY}.nc    ← sif + sif_uncertainty, shape (time=N_valid_days,)
                          coordinate: time (datetime64)
```

### Script

`download_sif_tropomi.py`

### Usage

```bash
# Full download (background)
nohup python download_sif_tropomi.py \
  > /tmp/download_sif.log 2>&1 &

# Monitor
tail -f /tmp/download_sif.log
```

---

## 3. OpenLandMap-soildb — Soil Properties

### Goal

Extract a 74×74 pixel soil property patch centred on each station from the global
OpenLandMap-soildb GeoTIFFs. One multi-band GeoTIFF per station.

### Data Source

- **Dataset**: OpenLandMap-soildb (Hengl et al., 2026 ESSD)
- **DOI**: [10.5281/zenodo.15470431](https://doi.org/10.5281/zenodo.15470431)
- **Format**: Global GeoTIFFs (~2 GB compressed), downloaded once to Snellius
- **Resolution**: 30 m
- **Auth**: None — Zenodo public download
- **Approach**: Download global files once → clip 74×74 per station locally (no per-station download)

### Variables (7 channels)

| Channel | Variable | Units | Depth |
|---|---|---|---|
| 0 | Clay content | wt% | 0–30 cm |
| 1 | Sand content | wt% | 0–30 cm |
| 2 | Silt content | wt% | 0–30 cm |
| 3 | SOC content (organic carbon) | g/kg | 0–30 cm |
| 4 | SOC density | kg/m³ | 0–30 cm |
| 5 | Bulk density | t/m³ | 0–30 cm |
| 6 | pH (H₂O) | — | 0–30 cm |

### Temporal notes

- Static dataset — 5-year composites (2000, 2005, 2010, 2015, 2020, 2022)
- Use composite closest to station mid-record year
- Downloaded once per station, not per year

### Patch size

74×74 pixels at 30 m = **2.22 km × 2.22 km** centred on station
(matches S2/S1/DEM footprint of 2.24 km at 10 m)

### Flowchart

```
START
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Download global GeoTIFFs from Zenodo (once)         │
│  DOI: 10.5281/zenodo.15470431                        │
│  ~2 GB compressed → store on Snellius               │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  load_stations()                                     │
│  Read station_splits.csv — all 1,048 stations        │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  For each station:                                   │
│    output = soil/soil_patch.tif                      │
│    if exists → SKIP                                  │
│                                                      │
│    Select composite year closest to mid-record year  │
│    (mid_year = (start_year + end_year) // 2)         │
│                                                      │
│    Compute UTM bbox: station ± 37×30m = ±1110 m      │
│    Open global GeoTIFF → clip to bbox                │
│    Stack 7 channels → (7, 74, 74) float32            │
│    Save → soil/soil_patch.tif                       │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
                          END
```

### Output

```
/home/khanalp/data/satellite/
  {network}_{station}/
    soil/
      soil_patch.tif    ← float32, 7 bands, 74×74 px @ 30 m
```

### Script

`download_soil_openlandmap.py`

### Usage

```bash
# Step 1 — download global GeoTIFFs from Zenodo (once, on Snellius)
wget -O /home/khanalp/data/openlandmap/soildb.zip \
  https://zenodo.org/records/15470431/files/soildb_0_30cm.zip

# Step 2 — clip per station
python download_soil_openlandmap.py

# Monitor progress via log output
```

### Notes

- The global GeoTIFF (~2 GB) should be downloaded directly to Snellius — too large to transfer locally
- Clipping 1,048 stations is fast (~minutes) once the global file is on disk
- No parallel workers needed — rasterio clipping is CPU-bound and fast
