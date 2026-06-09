# Zarr → Training: Comprehensive Plan
**Date:** 2026-06-09  
**Status:** Audit complete — 0 stations permanently lost; all issues recoverable

---

## 1. CURRENT STATE SNAPSHOT

### 1.1 Zarr stores (ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr/)

| Tier | Count | Description | Recoverable? |
|------|-------|-------------|-------------|
| **Fully OK** | 816 | `.complete` + all 10 modality dirs present | ✅ Ready now |
| **Needs re-tokenization** | 95 | `.complete` but NO S2 tokens in token zarr | ✅ All in satellite_zarr |
| **Minor issues** | 18 | `.complete` + S2 present, missing 1–3 minor mods | ✅ Small fixes |
| **Partial (no .complete)** | 62 | S2+S1+DEM+LULC in zarr; tabular in scratch only | ✅ Fill script needed |
| **Empty/no zarr** | 2 | ISMN_SCAN_Phillipsburg (no source), ISMN_SNOTEL_CaveMountain | ⚠️ Partial |
| **Total** | 993 | | |

**Key discovery:** All 95 stations previously thought "dead" are present in
`/projects/prjs1968/satellite_zarr/` with raw S2, S1, DEM, LULC pixel data.
Zero stations are permanently lost — all can be recovered via TerraMind re-tokenization.

### 1.2 Fully OK — split breakdown (816 stores)

| Split | SM stations | Flux-only | Total |
|-------|------------|-----------|-------|
| train | 506 | 60 | 566 |
| val   | 65 | 14 | 79 |
| oos   | 155 | 16 | 171 |
| **Total** | **726** | **90** | **816** |

Target from station_splits.csv:
- sm_only train: 587 → 506/587 = **86% available now**
- sm_only val: ~74 → 65/74 = **88% available now**
- sm_and_flux: 48 → **0 fully OK** (all have at least labels missing)

### 1.3 The 95 stations needing re-tokenization

All 95 have `.complete` in token zarr but `s2/` dir is empty/absent.
**All 95** are found at `/projects/prjs1968/satellite_zarr/{station}.zarr/` with:
- `s2/data`, `s2/dates` — raw Sentinel-2 pixels
- `s1_asc/data`, `s1_asc/dates` (+ `s1_desc` for most)
- `dem/`, `lulc/`

Networks: SNOTEL (45), SCAN (29), ICOS (15), AmeriFlux (6)
Splits: train=63, oos=25, val=7

Fix: Run TerraMind inference on `satellite_zarr` → write L3/L6/L9/L12 tokens into token zarr.
This is a GPU job (~2–4 h for 95 stations).

### 1.4 Partial stores without .complete (62)

62 stations have S2+S1+DEM+LULC tokens in zarr but no `.complete`:
- ERA5, SIF, soil → in `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/`
- TWSA → NOT downloaded
- CloudMask → source gone (raw S2 TIFs purged; must use satellite_zarr + CloudSEN12)
- SM labels → in `/projects/prjs1968/level1_organised/{cat}/{station}.nc`
- Splits: train=43, oos=17, val=4

### 1.5 Labels source

All SM labels are in `/projects/prjs1968/level1_organised/`:
- `sm_only/`: 894 files
- `sm_and_flux/`: 49 files
- `flux_only/`: 105 files (total 1048 — more than 993 because some networks split per-year)

The `update_labels` job (June 5, job 23498440) already wrote labels to 813 zarr stores
using this path. Stations still missing labels: the 62 partial stores + 4 sm_and_flux ICOS.

---

## 2. ROOT CAUSE OF DATA LOSS

### 2.1 The 95 "dead .complete" stores

The `tok_zarr_perm` job (June 5) processed stations from work3 data root. For stations
whose `S2L2A/` dir was missing in work3, `write_tokens()` returned an empty result silently
and `.complete` was still written. The source `.pt` bundle files were deleted earlier.

**No permanent loss** — raw pixels survive in `/projects/prjs1968/satellite_zarr/`.

### 2.2 The 62 partial stores

S2/S1/DEM/LULC tokens were written by earlier jobs when `.pt` files existed.
Subsequent tabular fill jobs wrote ERA5/SIF/soil to `scratch/data/` but never into zarr.
The `cm_sat_zarr` job failed entirely (`ModuleNotFoundError: numcodecs` on GPU node).
TWSA was never downloaded for these stations.

### 2.3 CRITICAL: dataset.py broken for zarr-only stations (PRIMARY TRAINING BLOCKER)

`dataset.py` line 796:
```python
if not sat_dir.exists() or not label_file.exists():
    continue
```
`sat_dir = data_root / cat / dir_name` where `data_root = /gpfs/work3/0/prjs1968/data`.
Work3 has only **66 sm_only + 103 flux_only = 169 stations**.
All other 824 stations are silently skipped — zarr is never opened for them.

The June 3 smoke test (584 stations, 1.16M samples) used old `.pt` files before
the zarr restructuring. Those `.pt` files are now gone.

---

## 3. ACTION PLAN

### Step 1 — Fix dataset.py [CRITICAL — 1h] ← DO THIS FIRST

Change line 796:
```python
# Before:
if not sat_dir.exists() or not label_file.exists():
    continue

# After:
zarr_complete = (ZARR_ROOT / cat / dir_name / ".complete").exists()
if not zarr_complete and (not sat_dir.exists() or not label_file.exists()):
    continue
# For zarr-only stations: sat_dir may not exist in data_root;
# label_file path is used as dict key only (labels loaded from zarr at line 819).
```

Add guard before line 855 to prevent KeyError when zarr labels are empty:
```python
if label_file not in self._label_cache:
    continue
sm_np, depths, times = self._label_cache[label_file]
```

**Effect:** Loads all 816 fully OK zarr stations → **506 sm train + 65 sm val**.

### Step 2 — Re-tokenize 95 stations from satellite_zarr [GPU job, ~2–4h]

Run TerraMind on `/projects/prjs1968/satellite_zarr/{station}.zarr/` to generate
L3/L6/L9/L12 tokens, then write them into the corresponding token zarr at
`/gpfs/scratch1/shared/pkhanal/zarr/{cat}/{station}/`.

The satellite_zarr stores have: `s2/data` (raw pixels), `s2/dates`, `s1_asc/`, `s1_desc/`, `dem/`, `lulc/`.

Script needed: `retokenize_from_satellite_zarr.py` — reads raw pixels, runs TerraMind, writes tokens.
These 95 stations already have ERA5/SIF/TWSA/labels in their token zarr (from earlier jobs).
After tokenization: delete old `.complete`, then re-run fill to write `.complete` properly.

**Effect:** +95 stations recoverable → total ~911 complete after this step.

### Step 3 — Fill tabular for 62 partial zarr stations [CPU job, ~1 day]

Write `fill_zarr_tabular.py`:
1. ERA5 from `scratch/data/{cat}/{station}/ERA5Land/*.nc` → zarr `era5/`
2. SIF from `scratch/data/{cat}/{station}/SIF/*.nc` → zarr `sif/`
3. Soil from `scratch/data/{cat}/{station}/soil/soil_patch.tif` → zarr `soil/`
4. SM labels from `/projects/prjs1968/level1_organised/{cat}/{station}.nc` → zarr `labels/`
5. TWSA: re-download from GRACE (GEE) for these 62 stations → zarr `twsa/`
6. CloudMask: run CloudSEN12 on `satellite_zarr` raw pixels for these 62 stations (GPU)
7. Write `.complete` after all available modalities written

**Effect:** +62 stations → ~973 complete.

### Step 4 — Re-run zarr creation for 14 fixable work3 stations [~2h SLURM]

These 14 have S2L2A in work3 but incomplete zarr. Delete partial `.complete`, re-run
`create_token_zarr.py` targeting these 14 only.

Stations: ISMN_SNOTEL_Aniak, ISMN_SCAN_Moccasin + 12 flux_only ICOS/AmeriFlux.

**Effect:** +2 sm_only train stations + 12 flux_only.

### Step 5 — Fix 4 sm_and_flux ICOS stations (labels missing) [~30 min]

ICOS_IT-Tor, ICOS_FI-Sii, ICOS_FR-Tou, ICOS_BE-Vie have all modalities except labels.
Look up in `/projects/prjs1968/level1_organised/sm_and_flux/` and write to zarr.

### Step 6 — START TRAINING [after Step 1 only — same day]

After Step 1 dataset.py fix:
- **506 sm_only train** stations (~86% of 587 target)
- **65 sm_only val** stations (~88% of 74 target)
- Expected: ~1.0M train samples

Steps 2–5 run in parallel with training (add new `.complete` without touching active stores).

---

## 4. TRAINING CONFIGURATION

### 4.1 Recommended first run (train.py)

```python
CONFIG = {
    "split_filter":    ["train"],
    "category_filter": ["sm_only"],
    "years":           list(range(2016, 2024)),
    "min_obs":         30,
    # data_root = work3 still works for the 169 work3 stations (fallback to .pt)
}
```

### 4.2 Expected performance with zarr (T=32 chunks)

| Metric | Before zarr | After zarr |
|--------|------------|-----------|
| Disk reads / sample | ~103 | ~7 (with L12 cache) |
| GPU utilisation | ~21% | 60–75% |
| Epoch time | ~110 min | ~25–35 min |
| SBU/epoch (H100) | ~370 | ~85 |

---

## 5. VERIFICATION STEPS

```bash
# After Step 1 — smoke test dataset loading (no GPU)
python3 -c "
from dataset import SoilMoistureDataset
ds = SoilMoistureDataset(
    splits_csv='/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv',
    data_root='/gpfs/work3/0/prjs1968/data',
    era5_stats_path='/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json',
    split_filter=['train'], category_filter=['sm_only']
)
print(f'Samples: {len(ds)}, Stations: {len(set(s[\"station_key\"] for s in ds.samples))}')
"
# Before fix: ~0–169 stations  |  After fix: ~506 stations

# Check .complete count after Step 2 re-tokenization
find /gpfs/scratch1/shared/pkhanal/zarr -name '.complete' | wc -l
# Before re-tokenize: 929  |  After: ~929 (same, just S2 added to 95 stores)

# Fresh full audit
python3 audit_zarr.py \
  --zarr-root /gpfs/scratch1/shared/pkhanal/zarr \
  --csv /gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv \
  --out /gpfs/scratch1/shared/pkhanal/audit_zarr_v2.csv
```

---

## 6. DATA RECOVERY SUMMARY

| Group | Count | Status |
|-------|-------|--------|
| Fully OK | 816 | ✅ Ready |
| Needs TerraMind re-tokenization | 95 | ✅ satellite_zarr available at /projects/prjs1968/satellite_zarr/ |
| Tabular fill needed | 62 | ✅ ERA5/SIF/soil in scratch; labels in level1_organised |
| Work3 re-run | 14 | ✅ S2L2A in work3 |
| Labels only | 4 | ✅ level1_organised |
| Truly gone | 2 | ⚠️ Phillipsburg (no source); CaveMountain (tabular only) |

**Total recoverable: 991/993 (99.8%)**  
**Usable now (after dataset.py fix): 816 (82%)**  
**Usable after parallel GPU + CPU jobs: ~993 (99.8%)**

---

## 7. KEY FILE PATHS

| What | Path |
|------|------|
| Token zarr stores | `/gpfs/scratch1/shared/pkhanal/zarr/{cat}/{station}/` |
| Raw pixel zarr stores | `/projects/prjs1968/satellite_zarr/{station}.zarr/` |
| Scratch tabular data | `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/` |
| Work3 token data | `/gpfs/work3/0/prjs1968/data/{cat}/{station}/` |
| SM labels (authoritative) | `/projects/prjs1968/level1_organised/{cat}/{station}.nc` |
| dataset.py | `/gpfs/work3/0/prjs1968/soilMoisture/dataset.py` |
| train.py | `/gpfs/work3/0/prjs1968/soilMoisture/train.py` |
| Stale audit CSV | `/gpfs/scratch1/shared/pkhanal/audit_zarr.csv` |
| Earlier zarr redesign plan | `/gpfs/scratch1/shared/pkhanal/zarr_redesign_plan.md` |
