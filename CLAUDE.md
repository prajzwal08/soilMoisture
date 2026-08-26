# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD research codebase with two major pipelines:
1. **ISMN preprocessing** — clean and gap-fill in-situ soil moisture observations into daily NetCDF files per station
2. **Multimodal ML training** — extract TerraMind ViT tokens from satellite imagery (S2L2A, S1RTC, DEM, LULC) and train a soil moisture prediction model fusing satellite tokens with ERA5-Land, SIF, TWSA, and soil data

## Running the Scripts

```bash
# ISMN preprocessing (parallel, all stations)
python preprocessing_ISMN_soilMoisture.py

# Tokenization pipeline (run in order)
python precompute_terramind.py [--station NAME] [--csv-start-idx I] [--csv-end-idx J]
python consolidate_tokens.py [--execute] [--workers 8]
python cleanup_tokens.py [--execute] [--workers 8]

# Cloud masking
python cloud_masking_inference.py
python filter_cloudy_tiles.py

# Model training
python train.py [--run-name NAME] [--lr LR] [--batch-size N]
                [--max-stations N]  # smoke-test mode
                [--max-epochs N]    # smoke-test mode
# Resume is automatic: if checkpoints/{run_name}/last.pt exists it resumes
# Delete last.pt to start fresh

# Pre-compute ERA5 normalisation stats (run once before first training)
python compute_era5_stats.py  # → csvs/era5_stats.json

# Demo / meeting plots from a trained checkpoint
python demo_plot.py --run-name baseline_huber --year 2021 --n-stations 4
# → demo_output/*.png  (predicted vs observed SM + spatial map)

# Interactive analysis
jupyter notebook preprocessing_ISMN_soilMoisture.ipynb
```

SLURM submissions (all jobs require --mail-type=BEGIN,END,FAIL + --mail-user):
```bash
sbatch slurm/tokenize.sh          # GPU array job; indexes station_splits.csv via --csv-start-idx/end-idx
sbatch slurm/consolidate_tokens.sh
sbatch slurm/cleanup_tokens.sh
sbatch slurm/train.sh --run-name baseline_huber           # Phase 1 baseline (Huber loss)
```

## Key Configuration (in `preprocessing_ISMN_soilMoisture.py`)

| Variable | Default | Purpose |
|---|---|---|
| `TEST_MODE` | `False` | If `True`, processes a random sample of `N_TEST_STATIONS` (100) stations |
| `USE_PARALLEL` | `True` | Toggle parallel vs. sequential processing (use `False` for debugging) |
| `N_WORKERS` | `100` | Number of multiprocessing workers |

## Data Paths (hardcoded)

**ISMN preprocessing:**
- Input ISMN data: `/home/khanalp/data/ISMNsoilMoisture/Data_separate_files_header_20140101_20251231_13107_18mx_20260208`
- Output level-1 NetCDFs: `/home/khanalp/data/soilmoisture/level1/` — pattern: `{network}_{station}_{start_date}_{end_date}.nc`

**ML pipeline:**
- Permanent token + feature storage: `/gpfs/work3/0/prjs1968/data/{sm_only|sm_and_flux|flux_only}/{station}/`
- Raw satellite TIFs (scratch): `/gpfs/scratch1/shared/pkhanal/satellite/{station}/`
- Station inventory: `csvs/station_splits.csv` (993 active), `csvs/excluded_stations.csv` (35 excluded)

**Station categories** (subdirs under `/gpfs/work3/0/prjs1968/data/`):
- `sm_only` — ISMN-only stations (no flux tower)
- `sm_and_flux` — stations with both soil moisture and flux measurements
- `flux_only` — flux-only stations (ICOS/AmeriFlux)

## Architecture & Data Flow

### Processing pipeline (`preprocessing_ISMN_soilMoisture.py` + `utils.py`)

```
ISMN raw data (hourly, per-sensor)
  └─► process_station()
        • Keep flags G (Good) and D* (Dubious starting with D)
        • Group sensors by depth_to, average across sensors
        • Resample to daily mean; mask days with < 6 observations
        • Bin depths into: 0-10 cm, 10-30 cm, 30-100 cm  (cut points "0,10,30,100";
          sensors deeper than 100 cm are dropped). These three are SM_DEPTHS in dataset.py
          and the model's three output columns — the order is load-bearing.
  └─► longest_available_after_removing_long_gaps()  [utils.py]
        • Short gaps (≤ 7 days) are treated as available (bridged)
        • Returns longest continuous period per depth bin
  └─► trim_to_surface_valid_period_and_keep_well_covered_depths()  [utils.py]
        • Reference window = longest valid period of surface depth (0-10 cm)
        • Drop other depths with < 95% data coverage within that window
  └─► gapfill_by_monthday_mean_with_feb29_fallback()  [utils.py]
        • Fill remaining NaNs using climatological mean (same month-day across years)
        • Feb-29 fallback: uses Feb-28, then Mar-01 if climatology missing
        • Adds a QC flag variable: 0=observed, 1=gap-filled, 2=still missing
  └─► Save NetCDF + metadata CSV
```

### Station-level quality filters (applied in `process_single_station`)
1. Must have a valid continuous surface (0-10 cm) run after gap removal
2. Must have ≥ 1 year (365 days) of valid daily data
3. No NaNs may remain after gap-filling (otherwise station is skipped)

### `utils.py` — utility functions

- `longest_available_after_removing_long_gaps(ds, max_gap_days=7)` — core gap analysis
- `trim_to_surface_valid_period_and_keep_well_covered_depths(ds, longest_avail, surface_depth, min_frac=0.95)` — depth/time filtering
- `gapfill_by_monthday_mean_with_feb29_fallback(ds)` — climatological gap-filling
- `missing_days_per_year(ds)` — diagnostic: count missing days per year per depth
- `longest_missing_run_with_dates(is_nan, dates)` — diagnostic: find longest gap with start/end dates
- `trim_to_common_continuous_period(ds, longest_available)` — alternative trimming (align all depths to common window)

### `DownloadSentinel1.py` (legacy)
Original single-station Sentinel-1 downloader. Superseded by `download_s2_mpc.py`, `download_s1_lulc_mpc.py`, etc. which drive from `station_splits.csv`.

## Tokenization Pipeline

Converts raw satellite TIFs into consolidated TerraMind token bundles that `dataset.py` loads.

```
precompute_terramind.py
  • Runs TerraMind ViT-Base on each TIF in scratch storage
  • Extracts frozen layer outputs L3, L6, L9, L12 → per-acquisition: YYYYMMDD_L{3,6,9,12}.pt  ([196,768] fp16)
  • Static modalities: dem_L12.pt, lulc_L12.pt  (one per station)
  • Resume-safe: skips stations where _L12.pt already exists
  └─► consolidate_tokens.py
        • Bundles all per-date .pt files into one dict per station per layer
        • Output: {station}_L{3,6,9,12}_{start}_{end}.pt
                  dict { tokens:[N,196,768] fp16, dates:[N str], layer:str, geo:dict }
        • Parallel with Pool(N); resume-safe (skips existing bundles)
  └─► cleanup_tokens.py
        • Deletes old per-date .pt + geo.json files (only after consolidated bundles verified)
        • Moves excluded stations to excluded_stations/ folder
        • Always verify consolidate is complete before running cleanup
```

**Token storage layout per station:**
```
{data_root}/{category}/{station}/
  S2L2A/  {station}_L{3,6,9,12}_{start}_{end}.pt
  S1RTC/  {station}_{ASC|DESC}_L{3,6,9,12}_{start}_{end}.pt
  DEM/    dem_L12.pt
  LULC/   lulc_L12.pt
```

## Model Training Pipeline

- **`model.py`** — `SoilMoistureModel` with `TemporalTransformer` backbone; fuses satellite tokens (S2, S1, DEM, LULC) with ERA5-Land, SIF, TWSA, and soil tabular features
- **`dataset.py`** — `SoilMoistureDataset`; loads consolidated `.pt` token bundles + NetCDF label files; expects consolidated format (not per-date files)
- **`train.py`** — training entry point; reads from `station_splits.csv` for train/val/test splits

## Key Dependencies

**Preprocessing:** `ismn`, `xarray`, `pandas`, `numpy`, `matplotlib`, `scienceplots`

**Satellite & tokenization:** `pystac-client`, `planetary-computer`, `stackstac`, `torch`, `terramind` (ViT-Base), `rasterio`, `geopandas`

**Cloud masking:** `CloudSEN12` (inference), `opencv-python`

**Conda environments — there are two, and they are not interchangeable** (verified 2026-08-12):

| env | file | Python | use for |
|---|---|---|---|
| `soilmoisture` | `environment-download.yml` | 3.10 | downloading + preprocessing — cdsapi, earthengine-api, pystac-client, planetary-computer, stackstac, icoscp, ismn. **No torch/zarr.** |
| `terramind` | `environment-terramind.yml` | 3.11 | tokenisation, training, eval, analysis probes — torch (cu124), terratorch, zarr 2.x, scikit-learn, umap-learn. **No download APIs.** |

```bash
conda activate soilmoisture   # download_*.py, preprocessing_*.py
conda activate terramind      # precompute_terramind.py, train.py, eval_predict.py, probe_*.py
```

Two gotchas worth knowing: `torch` must come from the CUDA 12.4 index (the plain PyPI wheel is
CPU-only and training silently falls back to CPU), and `zarr` is pinned to **2.x** because the
code uses v2 `open_consolidated` semantics throughout.

`environment.yml` is the older combined spec, superseded by the two files above but retained
for now.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
