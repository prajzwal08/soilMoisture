# Zarr → Training: Comprehensive Plan
**Date:** 2026-06-09  
**Status:** Audit complete — action plan below

---

## 1. CURRENT STATE SNAPSHOT

### 1.1 Zarr stores (ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr/)

| Tier | Count | Description |
|------|-------|-------------|
| **Fully OK** | 816 | `.complete` + all 10 modality dirs present |
| **Dead complete** | 81 | `.complete` but NO S2 tokens — source data gone permanently (VERIFY: check satellite_zarr) |
| **Minor issues** | 32 | `.complete` + S2 present, missing 1–3 minor modalities |
| **Partial (no complete)** | 62 | S2+S1+DEM+LULC in zarr; ERA5/SIF/soil in scratch, TWSA/CM gone |
| **Empty/no zarr** | 2 | ISMN_SCAN_Phillipsburg (nothing), ISMN_SNOTEL_CaveMountain (scratch tabular only) |
| **Total** | 993 | |

> **NOTE:** The 81 "dead .complete" entries need a cross-check against `satellite_zarr`
> (raw pixel zarr stores) at `/gpfs/scratch1/shared/pkhanal/zarr/` or
> `/gpfs/work3/0/prjs1968/data/` to confirm S2 truly cannot be recovered.

### 1.2 Fully OK — split breakdown (816 stores)

| Split | SM stations | Flux-only | Total |
|-------|------------|-----------|-------|
| train | 506 | 60 | 566 |
| val   | 65 | 14 | 79 |
| oos   | 155 | 16 | 171 |
| **Total** | **726** | **90** | **816** |

Target from station_splits.csv:
- sm_only train: 587 → currently 506/587 = **86% available**
- sm_only val: ~74 → currently 65/74 = **88% available**
- sm_and_flux: 48 → **0 fully OK** (all have at least labels missing)

### 1.3 Dead .complete stores (81) — status uncertain until satellite_zarr checked

These 81 have `.complete` but no `s2/` directory in the token zarr. Networks:
- SNOTEL: 45, SCAN: 29, ICOS: 15, AmeriFlux: 6
- Splits: train=63, oos=25, val=7

**Recovery question:** Does `/gpfs/scratch1/shared/pkhanal/satellite_zarr/` or an equivalent raw-pixel zarr exist for these stations? If yes, they can be re-tokenized via TerraMind. If no, their S2 tokens are permanently lost.

Current behaviour in dataset.py: skipped at line 847 (`"s2/dates" not in zg → continue`), but waste ~1 s each during dataset init.

### 1.4 Partial stores without .complete (62)

62 stations have S2+S1+DEM+LULC tokens in their zarr but no `.complete`:
- ERA5, SIF, soil → in `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/`
- TWSA → NOT downloaded
- CloudMask → NOT available (raw S2 TIFs were purged; CloudSEN12 inference needed)
- SM labels → available in `/projects/prjs1968/raw_soil_moisture/`
- Splits: train=43, oos=17, val=4

### 1.5 14 fixable stations (full source in work3)

These 14 have S2L2A in `/gpfs/work3/0/prjs1968/data/` but zarr is incomplete.
Fix: re-run `create_token_zarr.py` for just these 14.

Stations:
- ISMN_SNOTEL_Aniak [sm_only/train]
- ISMN_SCAN_Moccasin [sm_only/train]
- ICOS_CH-Fru, ICOS_CZ-KrP, ICOS_CZ-wet, ICOS_CH-Cha, ICOS_CH-Lae, ICOS_FR-Aur [flux_only]
- AmeriFlux_CA-Mer, AmeriFlux_US-BZB, AmeriFlux_US-ONA, AmeriFlux_US-MtB, AmeriFlux_US-xDJ, AmeriFlux_US-xSE [flux_only]

---

## 2. ROOT CAUSE OF DATA LOSS

### 2.1 The 81 dead .complete stores

These stations went through the `tok_zarr_perm` job which created `.complete` but the
`write_tokens()` function returned an empty result silently (no `S2L2A` in work3 data root at run time,
or the `.pt` bundle was already deleted). Source `.pt` files were deleted after earlier tokenization.

**To confirm permanently lost:** run the satellite_zarr cross-check (see §5 Verification).

### 2.2 The 62 partial stores

S2/S1/DEM/LULC tokens were written by earlier jobs (when source `.pt` files existed).
The subsequent tabular fill jobs (`era5_299`, `sif_299`) wrote to `scratch/data/` but
**never wrote into zarr**. The `cm_sat_zarr` job failed entirely
(`ModuleNotFoundError: numcodecs` on GPU node). TWSA was never downloaded for these.

### 2.3 CRITICAL: dataset.py broken for zarr-only stations

`dataset.py` line 796:
```python
if not sat_dir.exists() or not label_file.exists():
    continue
```
`sat_dir = data_root / cat / dir_name` where `data_root = /gpfs/work3/0/prjs1968/data`.
Work3 has only **66 sm_only + 103 flux_only = 169 stations**.
All other 824 stations are skipped silently before zarr is even opened.

The June 3 smoke test (`1160521 samples from 584 stations`) used old `.pt` files before
the zarr restructuring — those `.pt` files are now gone.

---

## 3. ACTION PLAN

### Step 1 — Fix dataset.py [CRITICAL — 1h] ← DO THIS FIRST

Change line 796 from:
```python
if not sat_dir.exists() or not label_file.exists():
    continue
```
To:
```python
zarr_complete = (ZARR_ROOT / cat / dir_name / ".complete").exists()
if not zarr_complete and (not sat_dir.exists() or not label_file.exists()):
    continue
# For zarr-only stations sat_dir may not exist in data_root;
# label_file path is used as dict key only (zarr labels loaded at line 819).
```

Add guard before line 855 to prevent KeyError when zarr labels array is empty:
```python
if label_file not in self._label_cache:
    continue
sm_np, depths, times = self._label_cache[label_file]
```

**Effect:** dataset.py loads all 816 fully OK zarr stations → **506 sm train + 65 sm val**.

### Step 2 — satellite_zarr cross-check for 81 dead stations [1h, CPU]

Check if raw-pixel satellite zarr stores (from the original `satellite_zarr` pipeline)
exist for any of the 81 dead stations. If they do, S2 tokens can be recovered by re-running
TerraMind tokenization on the raw pixels.

```bash
# Check for satellite_zarr or raw-pixel stores
ls /gpfs/scratch1/shared/pkhanal/ | grep -i sat
ls /gpfs/work3/0/prjs1968/data/ | grep -i sat

# For each of the 81 dead stations check satellite_zarr:
python3 << 'EOF'
import pathlib
dead = [...]  # list of (cat, station) tuples from audit
sat_zarr = pathlib.Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr")  # adjust path
for cat, station in dead:
    if (sat_zarr / cat / station).exists():
        print(f"RECOVERABLE: {station}")
EOF
```

### Step 3 — Remove .complete from 81 dead stations [15 min, CPU]

Whether or not satellite_zarr recovery is possible, remove the misleading `.complete`
markers from the 81 dead stores so dataset.py never opens them:

```bash
python3 << 'EOF'
import pathlib
zarr_root = pathlib.Path('/gpfs/scratch1/shared/pkhanal/zarr')
removed = 0
for cp in zarr_root.rglob('.complete'):
    station_dir = cp.parent
    groups = {d.name for d in station_dir.iterdir()
              if not d.name.startswith('.') and d.is_dir()}
    if 's2' not in groups:
        cp.unlink()
        removed += 1
        print(f'Removed: {station_dir.parent.name}/{station_dir.name}')
print(f'Total removed: {removed}')
EOF
```

**Effect:** 929 → 848 .complete markers (all 848 have S2 tokens).

### Step 4 — Re-run zarr creation for 14 fixable work3 stations [~2h SLURM]

Delete their partial `.complete` first, then re-run `create_token_zarr.py`
with `--station-list` pointing to the 14 stations.

**Effect:** 2 additional sm_only train stations + 12 flux_only.

### Step 5 — Fill tabular for 62 partial zarr stations [~1 day, CPU job]

Write `fill_zarr_tabular.py`:
1. ERA5 from `scratch/data/{cat}/{station}/ERA5Land/*.nc` → zarr `era5/`
2. SIF from `scratch/data/{cat}/{station}/SIF/*.nc` → zarr `sif/`
3. Soil from `scratch/data/{cat}/{station}/soil/soil_patch.tif` → zarr `soil/`
4. SM labels from `raw_soil_moisture/{network}_{station}_*.nc` → zarr `labels/`
5. TWSA: re-download from GRACE (GEE/ESA) → zarr `twsa/`
6. CloudMask: skip (source gone) — dataset.py handles missing CM gracefully
7. Write `.complete` after all available modalities written

**Effect:** ~62 additional complete stations → ~910 total.

### Step 6 — Fix 4 sm_and_flux ICOS stations (labels only) [~30 min]

ICOS_IT-Tor, ICOS_FI-Sii, ICOS_FR-Tou, ICOS_BE-Vie have all modalities except labels.
Check `/projects/prjs1968/raw_soil_moisture/` for matching NetCDF files.

### Step 7 — START TRAINING [after Steps 1+3]

After Steps 1 and 3 (ETA: same day), training can begin:
- **506 sm_only train** stations (~86% of 587 target)
- **65 sm_only val** stations (~88% of 74 target)
- Expected: ~1.0M train samples

Steps 4+5 run in parallel — new `.complete` files appear without touching active stores.

---

## 4. TRAINING CONFIGURATION

### 4.1 Expected dataset size (after Steps 1+3)

- Train: ~506 stations × ~2000 samples ≈ **~1.0M samples**
- Val: ~65 stations × ~2000 samples ≈ **~130k samples**

### 4.2 Recommended first run (train.py)

```python
CONFIG = {
    "split_filter":    ["train"],
    "category_filter": ["sm_only"],  # focus SM stations first
    "years":           list(range(2016, 2024)),
    "min_obs":         30,
    # data_root still works as fallback for the 169 work3 stations
}
```

### 4.3 GPU estimate

With zarr (T=32 chunks) vs old .pt:
- Disk reads per sample: ~7 (was 103)
- Expected GPU utilisation: 60–75% (was ~21%)
- Epoch time: ~25–35 min (was ~110 min)

---

## 5. VERIFICATION STEPS

```bash
# After Step 1 fix — smoke test dataset loading (no GPU needed)
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
# Expect: ~1.0M samples from ~506 stations (was 169 before fix)

# After Step 3 — confirm dead .complete are gone
find /gpfs/scratch1/shared/pkhanal/zarr -name '.complete' | wc -l
# Expect: ~848

# Fresh full audit (run before training)
python3 audit_zarr.py \
  --zarr-root /gpfs/scratch1/shared/pkhanal/zarr \
  --csv /gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv \
  --out /gpfs/scratch1/shared/pkhanal/audit_zarr_v2.csv
```

---

## 6. DATA LOSS SUMMARY

| Group | Count | Root cause | Recoverable? |
|-------|-------|-----------|-------------|
| Fully OK | 816 | — | ✅ Ready now |
| Dead .complete (no S2) | 81 | `.pt` deleted before zarr wrote S2 | ❓ Check satellite_zarr; else ❌ |
| Partial zarr (no tabular) | 62 | tabular fill job wrote to scratch, not zarr; CM job failed | ✅ ERA5/SIF/soil fixable; CM needs GPU |
| 14 fixable from work3 | 14 | Write error mid-job | ✅ Re-run zarr creation |
| sm_and_flux labels missing | 4 | labels never written | ✅ Easy fix |
| No zarr at all | 2 | Source data missing | ❌ Skip (1) / partial (1) |

**Usable NOW (after dataset.py fix only): ~816 stations (82%)**  
**Usable after quick fixes (Steps 1–4): ~848 stations (85%)**  
**Usable after full fixes (all steps): ~910 stations (92%)**

---

## 7. KEY FILE PATHS

| What | Path |
|------|------|
| Zarr stores | `/gpfs/scratch1/shared/pkhanal/zarr/{cat}/{station}/` |
| Scratch tabular data | `/gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/` |
| Work3 token data | `/gpfs/work3/0/prjs1968/data/{cat}/{station}/` |
| Raw SM labels | `/projects/prjs1968/raw_soil_moisture/` |
| Audit CSV (stale, June 5) | `/gpfs/scratch1/shared/pkhanal/audit_zarr.csv` |
| dataset.py | `/gpfs/work3/0/prjs1968/soilMoisture/dataset.py` |
| train.py | `/gpfs/work3/0/prjs1968/soilMoisture/train.py` |
| create_token_zarr.py | `/gpfs/work3/0/prjs1968/soilMoisture/create_token_zarr.py` |
| Earlier zarr redesign plan | `/gpfs/scratch1/shared/pkhanal/zarr_redesign_plan.md` |
