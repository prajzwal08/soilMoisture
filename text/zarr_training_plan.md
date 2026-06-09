# Zarr → Training: Comprehensive Plan
**Date:** 2026-06-09 (updated)
**Status:** Full audit — 991/993 stations recoverable. All fix scripts written.

---

## 1. CURRENT STATE SNAPSHOT (re-audited 2026-06-09)

| Group | Count | Status |
|-------|-------|--------|
| **Fully OK** | 818 | `.complete` + s2/l12 + era5 — ready for training |
| **Group A** | 24 | `.complete` + NO s2 tokens; have `.pt` bundles in `/projects/prjs1968/data/` |
| **Group B** | 83 | `.complete` + NO s2 tokens; satellite_zarr only (GPU needed) |
| **Group C** | 62 | s2 tokens present, NO `.complete`; need CloudSEN12 + tabular fill |
| **Group D** | 4 | `.complete` but missing `labels/sm` (sm_and_flux ICOS) |
| **Unresolved** | 2 | ISMN_SCAN_CaveValley, ISMN_SCAN_SplitMountain — ERA5 missing |
| **Total** | 993 | |

### Split breakdown — Fully OK (818)

| Split | sm_only | flux_only | Total |
|-------|---------|-----------|-------|
| train | 506 | 60 | 566 |
| val   | 65  | 14 | 79  |
| oos   | 155 | 16 | 171 |

---

## 2. CRITICAL BUG FIX — dataset.py (Step 1 — DONE ✅)

`data_root = /projects/prjs1968/data` has only 169 stations.
Old code at line 796:
```python
if not sat_dir.exists() or not label_file.exists():
    continue   # silently skips all 824 zarr-only stations!
```

**Fix applied (2026-06-09):**
```python
zarr_complete = (ZARR_ROOT / cat / dir_name / ".complete").exists()
if not zarr_complete and (not sat_dir.exists() or not label_file.exists()):
    continue
```
Also added guard before line 858 to prevent KeyError on missing labels:
```python
if label_file not in self._label_cache:
    continue
sm_np, depths, times = self._label_cache[label_file]
```

**Effect:** Dataset now loads all 818 fully OK stations → ~506 sm_only train + 65 sm_only val.

---

## 3. GROUP DETAILS AND FIX SCRIPTS

### Group A — 24 stations (re-run create_token_zarr.py)

Have `.pt` bundles in `/projects/prjs1968/data/{cat}/{station}/S2L2A/`.
`create_token_zarr.py --force` deletes the broken zarr and recreates from `.pt` bundles.

**Script changes (2026-06-09):**
- `create_token_zarr.py`: switched labels source to `level1_organised`, added `--stations`
  and `--force` flags.

**Fix:** `sbatch slurm/group_a_zarr.sh`

Stations (24):
`ISMN_SCAN_Moccasin, ISMN_SNOTEL_Aniak, ICOS_CH-Cha, ICOS_CH-Fru, ICOS_CH-Lae,
ICOS_CZ-KrP, ICOS_CZ-wet, ICOS_FR-Aur, ICOS_RU-Fy2, AmeriFlux_CA-Cbo,
AmeriFlux_CA-Mer, AmeriFlux_US-BZB, AmeriFlux_US-Bi2, AmeriFlux_US-DFC,
AmeriFlux_US-Mpj, AmeriFlux_US-MtB, AmeriFlux_US-ONA, AmeriFlux_US-Rls,
AmeriFlux_US-Rms, AmeriFlux_US-Rwf, AmeriFlux_US-Seg, AmeriFlux_US-Ses,
AmeriFlux_US-xDJ, AmeriFlux_US-xSE`

**+2 sm_only train stations (Aniak, Moccasin) → 508 sm train**

### Group B — 83 stations (TerraMind + CloudSEN12 from satellite_zarr, GPU)

These 83 stations have `.complete` in zarr but no s2/l12 tokens. All 83 exist in
`/projects/prjs1968/satellite_zarr/{station}.zarr/` with raw S2/S1/DEM/LULC pixels.
Some already have era5, labels, s1_asc partial tokens, and cm in zarr
(written by earlier partial jobs). The script fills only what's missing.

Networks: mostly SNOTEL (45) and SCAN (29) + ICOS/AmeriFlux.
Splits: train=63, oos=19, val=1.

**Fix:** `sbatch slurm/retokenize_b.sh`

Script `retokenize_satellite_zarr.py --mode all`:
1. Reads raw pixels from satellite_zarr
2. TerraMind inference → s2/l3,l6,l9,l12 + s2/dates
3. TerraMind on S1 ASC/DESC, DEM, LULC (if not already present)
4. CloudSEN12 inference → cm/masks, cm/dates (if not already present)
5. ERA5/SIF from scratch/data; labels from level1_organised
6. Consolidate metadata + touch .complete

### Group C — 62 stations (CloudSEN12 + fill tabular, GPU for CM)

These 62 already have s2/l12 tokens in zarr (+ s1, dem, lulc) but no `.complete`.
All 62 have ERA5/SIF in scratch/data. Need CloudSEN12 + labels + write `.complete`.

Splits: train=43, oos=17, val=2.

**Fix:** `sbatch slurm/retokenize_c.sh`

Script `retokenize_satellite_zarr.py --mode cm-only`:
1. CloudSEN12 on raw S2 from satellite_zarr → cm/masks, cm/dates
2. Fill ERA5/SIF from scratch/data → zarr
3. Labels from level1_organised → zarr
4. Write .complete

### Group D — 4 sm_and_flux ICOS stations (labels only missing)

ICOS_BE-Vie, ICOS_FI-Sii, ICOS_FR-Tou, ICOS_IT-Tor — all modalities present except
`labels/sm`. Labels available in `level1_organised/sm_and_flux/`.

**Fix:** `python fix_group_d_labels.py`
(Opens zarr in append mode, writes only labels/sm, labels/qc, labels/depths, labels/dates)

---

## 4. ACTION PLAN

| Step | Action | Script | GPU? | Status |
|------|--------|--------|------|--------|
| 1 | Fix dataset.py | inline edit | No | **DONE ✅** |
| 2 | Fix Group D (4 stations) | `python fix_group_d_labels.py` | No | Ready |
| 3 | Fix Group A (24 stations) | `sbatch slurm/group_a_zarr.sh` | No | Ready |
| 4 | Start training | `sbatch slurm/train.sh` | Yes | After Steps 1–3 |
| 5 | Fix Group B (83 stations) | `sbatch slurm/retokenize_b.sh` | Yes | Parallel with training |
| 6 | Fix Group C (62 stations) | `sbatch slurm/retokenize_c.sh` | Yes | Parallel with training |

**Sequence:**
1. Run Steps 2+3 (CPU, fast — ~1h each)
2. Start training with 818 + 4(D) + 24(A) = 846 stations ready
3. Run Steps 5+6 in parallel (GPU, adds 83+62=145 more stations as they complete)

---

## 5. LABELS SOURCE UPDATE (important)

All SM labels come from `/projects/prjs1968/level1_organised/{cat}/{station}.nc`.
**NOT** from `/projects/prjs1968/raw_soil_moisture/`.

`create_token_zarr.py` was updated to use `LEVEL1_DIR = /projects/prjs1968/level1_organised`.
The `write_labels()` function now opens `LEVEL1_DIR/{category}/{dir_name}.nc` directly.

---

## 6. KEY PATHS

| What | Path |
|------|------|
| Token zarr | `/gpfs/scratch1/shared/pkhanal/zarr/{cat}/{station}/` |
| Raw pixel zarr | `/projects/prjs1968/satellite_zarr/{station}.zarr/` |
| .pt token bundles | `/projects/prjs1968/data/{cat}/{station}/` |
| Scratch tabular | `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/` |
| SM labels | `/projects/prjs1968/level1_organised/{cat}/{station}.nc` |
| dataset.py | `/gpfs/work3/0/prjs1968/soilMoisture/dataset.py` |
| create_token_zarr.py | `/gpfs/work3/0/prjs1968/soilMoisture/create_token_zarr.py` |
| retokenize_satellite_zarr.py | `/gpfs/work3/0/prjs1968/soilMoisture/retokenize_satellite_zarr.py` |
| fix_group_d_labels.py | `/gpfs/work3/0/prjs1968/soilMoisture/fix_group_d_labels.py` |

---

## 7. VERIFICATION

```bash
# Count .complete after all steps
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
# Expect: 929 now → 991 after all groups fixed

# Quick dataset smoke test (after Step 1 — dataset.py fix)
conda run -n terramind python -c "
from dataset import SoilMoistureDataset
ds = SoilMoistureDataset(
    splits_csv='/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv',
    data_root='/projects/prjs1968/data',
    era5_stats_path='/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json',
    split_filter=['train'], category_filter=['sm_only'],
)
print(f'Samples: {len(ds)}, Stations: {len(set(s[\"station_key\"] for s in ds.samples))}')
"
# Before fix: ~0 stations  |  After: ~506 stations, ~1M samples
```
