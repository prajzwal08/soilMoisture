# Zarr → Training: Comprehensive Plan
**Date:** 2026-06-09  
**Status:** Full audit — 991/993 stations recoverable

---

## 1. CURRENT STATE SNAPSHOT

### 1.1 Zarr stores (ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr/)

| Tier | Count | Fix needed | Source |
|------|-------|-----------|--------|
| **Fully OK** | 816 | None | zarr ready |
| **Group A** | 14 | Re-run `create_token_zarr.py` (no GPU) | `/projects/prjs1968/data/` — has `.pt` bundles + CloudMask |
| **Group B** | 81 | TerraMind + CloudSEN12 on satellite_zarr (GPU) | `/projects/prjs1968/satellite_zarr/` — raw pixels only |
| **Group C** | 62 | CloudSEN12 on satellite_zarr (GPU) + fill tabular (CPU) | satellite_zarr pixels + scratch/data tabular |
| **Group D** | 4 | Write labels from level1_organised | zarr complete except `labels/` |
| **Skip** | 2 | No source | Phillipsburg, CaveMountain |
| **Total** | 979+ | | |

### 1.2 Fully OK — split breakdown (816)

| Split | SM stations | Flux-only | Total |
|-------|------------|-----------|-------|
| train | 506 | 60 | 566 |
| val   | 65 | 14 | 79 |
| oos   | 155 | 16 | 171 |

Target: sm_only train=587, val=74. **Currently 506/587 (86%) train, 65/74 (88%) val.**

---

## 2. GROUP DETAILS

### Group A — 14 stations (`.pt` bundles in /projects/prjs1968/data/)

All 14 have full data in `/projects/prjs1968/data/{cat}/{station}/`:
- `S2L2A/` — TerraMind `.pt` token bundles ✅
- `CloudMask/` — precomputed `.pt` cloud masks ✅
- `S1RTC/`, `DEM/`, `LULC/`, `ERA5Land/`, `SIF/`, `TWSA/`, `soil/` ✅

**Fix:** Re-run `create_token_zarr.py` targeting these 14. No GPU needed.

Stations: ISMN_SNOTEL_Aniak [sm/train], ISMN_SCAN_Moccasin [sm/train],
ICOS_CH-Fru, ICOS_CZ-KrP, ICOS_CZ-wet, ICOS_CH-Cha, ICOS_CH-Lae, ICOS_FR-Aur [flux],
AmeriFlux_CA-Mer, AmeriFlux_US-BZB, AmeriFlux_US-ONA, AmeriFlux_US-MtB,
AmeriFlux_US-xDJ, AmeriFlux_US-xSE [flux]

### Group B — 81 stations (only in satellite_zarr, no tokens in zarr)

These 81 are NOT in `/projects/prjs1968/data/`. All 81 exist in
`/projects/prjs1968/satellite_zarr/{station}.zarr/` with:
- `s2/data`, `s2/dates` — raw Sentinel-2 pixels ✅
- `s1_asc/data`, `s1_asc/dates` (+ `s1_desc/` for most) ✅
- `dem/`, `lulc/` ✅
- **No CloudMask** anywhere ❌

Their token zarr already has `.complete` but the `s2/` dir is empty/absent.
Also need tabular data: ERA5/SIF/TWSA/soil/labels.

**Fix:**
1. Run **TerraMind inference** on raw pixels → write L3/L6/L9/L12 tokens to zarr `s2/`, `s1_asc/`, `s1_desc/`
2. Run **CloudSEN12 inference** on same raw S2 pixels → write to zarr `cm/`
3. Fill tabular: ERA5/SIF/TWSA from re-download OR scratch/data if available; labels from `level1_organised`
4. Rewrite `.complete` properly

Networks: SNOTEL (45), SCAN (29), ICOS (15), AmeriFlux (6). Splits: train=63, oos=25, val=7.

### Group C — 62 stations (S2 tokens in zarr, no .complete, no CloudMask)

These 62 already have `s2/`, `s1_asc/`, `dem/`, `lulc/` in their token zarr.
They are also all in `/projects/prjs1968/satellite_zarr/` (raw pixels, no CloudMask there either).
Tabular source: `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/` has ERA5Land, SIF, soil.
TWSA and CloudMask are missing.

**Fix:**
1. Run **CloudSEN12 inference** on satellite_zarr raw S2 pixels → write to zarr `cm/`
2. Fill ERA5/SIF/soil from `scratch/data/` → zarr
3. Download TWSA for these 62 stations → zarr
4. Write SM labels from `/projects/prjs1968/level1_organised/` → zarr
5. Write `.complete`

Splits: train=43, oos=17, val=4.

### Group D — 4 sm_and_flux ICOS stations (labels only missing)

ICOS_IT-Tor, ICOS_FI-Sii, ICOS_FR-Tou, ICOS_BE-Vie have all modalities except `labels/`.

**Fix:** Write SM labels from `/projects/prjs1968/level1_organised/sm_and_flux/` → zarr `labels/`.

---

## 3. CRITICAL BUG: dataset.py skips all zarr-only stations

`data_root = /projects/prjs1968/data` has only **66 sm_only + 103 flux_only = 169 stations**.

`dataset.py` line 796:
```python
if not sat_dir.exists() or not label_file.exists():
    continue
```
All 824 stations not in `data_root` are silently skipped. **Primary training blocker.**

The June 3 smoke test (584 stations, 1.16M samples) used old `.pt` files — those are gone.

---

## 4. ACTION PLAN

### Step 1 — Fix dataset.py [CRITICAL — 1h] ← DO FIRST

Change line 796:
```python
# Before:
if not sat_dir.exists() or not label_file.exists():
    continue

# After:
zarr_complete = (ZARR_ROOT / cat / dir_name / ".complete").exists()
if not zarr_complete and (not sat_dir.exists() or not label_file.exists()):
    continue
```

Add guard before line 855 to prevent KeyError on empty labels:
```python
if label_file not in self._label_cache:
    continue
sm_np, depths, times = self._label_cache[label_file]
```

**Effect:** Loads all 816 fully OK zarr stations → **506 sm train + 65 sm val**.

### Step 2 — Re-run create_token_zarr.py for Group A [CPU SLURM, ~1h]

Delete the 14 partial `.complete` files, then re-run `create_token_zarr.py`
with `--data-root /projects/prjs1968/data` for just these 14 stations.
No GPU needed — reads `.pt` bundles directly.

**Effect:** +2 sm_only (Aniak, Moccasin) → 508 sm train; +12 flux_only.

### Step 3 — START TRAINING [after Steps 1+2]

After Steps 1 and 2 (ETA: same day):
- **508 sm_only train** stations
- **65 sm_only val** stations
- Expected: ~1.01M train samples

Steps 4–6 run in parallel with training.

### Step 4 — GPU job: tokenize + cloud-mask Group B (81 stations) [~4–8h GPU]

Write `retokenize_satellite_zarr.py` that reads from
`/projects/prjs1968/satellite_zarr/{station}.zarr/`:
1. Read raw S2 pixels batch-wise → run **TerraMind** → write L3/L6/L9/L12 tokens
2. Read same raw S2 pixels → run **CloudSEN12** → write cloud masks
3. Fill tabular (ERA5/SIF/soil from scratch if available; else re-download)
4. Write SM labels from `/projects/prjs1968/level1_organised/`
5. Remove old `.complete`, write new `.complete`

**Effect:** +81 stations recoverable → ~909 total.

### Step 5 — GPU job: cloud-mask + fill Group C (62 stations) [~2–4h GPU]

Reuse `retokenize_satellite_zarr.py` for CloudSEN12 only (tokens already in zarr):
1. Read raw S2 from satellite_zarr → run **CloudSEN12** → write `cm/` to zarr
2. Fill ERA5/SIF/soil from `scratch/data/`
3. Download TWSA for 62 stations → zarr
4. Write SM labels from `level1_organised`
5. Write `.complete`

**Effect:** +62 stations → ~971 total.

### Step 6 — Write labels for Group D (4 ICOS sm_and_flux) [~30 min]

Read from `/projects/prjs1968/level1_organised/sm_and_flux/` → write to zarr `labels/`.

---

## 5. VERIFICATION

```bash
# After Step 1 — smoke test (no GPU needed)
python3 -c "
from dataset import SoilMoistureDataset
ds = SoilMoistureDataset(
    splits_csv='/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv',
    data_root='/projects/prjs1968/data',
    era5_stats_path='/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json',
    split_filter=['train'], category_filter=['sm_only']
)
print(f'Samples: {len(ds)}, Stations: {len(set(s[\"station_key\"] for s in ds.samples))}')
"
# Before fix: ~0 stations  |  After: ~506 stations

# Count .complete markers
find /gpfs/scratch1/shared/pkhanal/zarr -name '.complete' | wc -l
# Should stay at 929 until Group B/C jobs complete
```

---

## 6. SUMMARY TABLE

| Group | Count | Fix | GPU? | When |
|-------|-------|-----|------|------|
| Fully OK | 816 | dataset.py fix | No | Step 1 (today) |
| A: .pt bundles in projects/data | 14 | Re-run create_token_zarr.py | No | Step 2 (today) |
| B: satellite_zarr only, no tokens | 81 | TerraMind + CloudSEN12 | Yes | Step 4 (parallel) |
| C: tokens in zarr, no CM/tabular | 62 | CloudSEN12 + fill tabular | Yes (CM only) | Step 5 (parallel) |
| D: labels only missing | 4 | Write labels | No | Step 6 (parallel) |
| Unrecoverable | 2 | Skip | — | — |

**Total recoverable: 977/993**

---

## 7. KEY PATHS

| What | Path |
|------|------|
| Token zarr | `/gpfs/scratch1/shared/pkhanal/zarr/{cat}/{station}/` |
| Raw pixel zarr | `/projects/prjs1968/satellite_zarr/{station}.zarr/` |
| .pt token bundles | `/projects/prjs1968/data/{cat}/{station}/` |
| Scratch tabular | `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/` |
| SM labels | `/projects/prjs1968/level1_organised/{cat}/{station}.nc` |
| dataset.py | `/gpfs/work3/0/prjs1968/soilMoisture/dataset.py` |
| train.py | `/gpfs/work3/0/prjs1968/soilMoisture/train.py` |
| create_token_zarr.py | `/gpfs/work3/0/prjs1968/soilMoisture/create_token_zarr.py` |
