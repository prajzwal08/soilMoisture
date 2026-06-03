# Plan: Phase 1 Training — sm_only Soil Moisture Model

## Context

All 993 stations have complete `.pt` token bundles (S2L2A, S1RTC, DEM, LULC). The pipeline is ready to train. Phase 1 trains only on **sm_only stations (842 stations)** to predict soil moisture at 3 depths (`0-10`, `10-30`, `30-100` cm). Not all stations have all 3 depths; NaN masking in the loss handles this.

**Problems with current code:**
1. `dataset.py` uses stale paths from an old data layout (`metadata_csv`, `/home/khanalp/data/satellite`) — incompatible with actual `/gpfs/work3/0/prjs1968/data/{category}/{station}/` layout.
2. SIF and TWSA are not loaded anywhere (mentioned in architecture, missing from both `dataset.py` and `model.py`).
3. No ERA5 normalization — raw values passed directly to MLP.
4. `train.py` ignores pre-defined train/val splits in `station_splits.csv`; uses random 15% split instead.
5. No W&B logging, no argparse, no SLURM training script.

---

## Step 1 — Pre-compute ERA5 normalization stats

**New file: `compute_era5_stats.py`**

- Iterate over all `sm_only` stations' `ERA5Land/meteo_*.nc` files
- Compute per-variable mean and std across all stations and all days
- Apply `log1p` to `tp_sum` before computing its stats (heavy right skew)
- Write result to `csvs/era5_stats.json` — shape: `{"means": [19 floats], "stds": [19 floats], "log1p_precip": true}`
- Run once on CPU login node (~5–10 min): `python compute_era5_stats.py`

---

## Step 1b — Fix depth constants (dataset.py + model.py)

**Current (wrong):**
```python
SM_DEPTHS = ["0-10", "10-20", "20-40", "40-100"]  # n_depths = 4
```

**Correct** (from full label census across all 842 sm_only stations):
```
501 stations: ('0-10', '10-30', '30-100')   ← all 3 depths
155 stations: ('0-10', '10-30')
153 stations: ('0-10',)
 33 stations: ('0-10', '30-100')
```
```python
SM_DEPTHS = ["0-10", "10-30", "30-100"]  # n_depths = 3
```

Change `n_depths=4` → `n_depths=3` everywhere: `dataset.py` constant, `train.py` CONFIG, `model.py` default arg.
The existing NaN masking in `__getitem__` and `masked_huber_loss` already handles missing depths per station — no other logic changes needed for this fix.

---

## Step 2 — Refactor `dataset.py`

### 2a. Replace `__init__` signature

**Old (incompatible):**
```python
def __init__(self, metadata_csv, satellite_dir, ismn_dir, years, min_obs, station_filter, soil_data_root)
```

**New:**
```python
def __init__(self, splits_csv, data_root, era5_stats_path, years=None,
             min_obs=30, category_filter=None, split_filter=None, training=True)
```
- `splits_csv` → `/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv`
- `data_root` → `/gpfs/work3/0/prjs1968/data`
- `era5_stats_path` → `csvs/era5_stats.json`
- `category_filter` → e.g. `["sm_only"]` (Phase 1), `None` for all
- `split_filter` → e.g. `["train"]` or `["val"]` (uses pre-defined column in CSV)
- `training` → enables modality dropout for SIF/TWSA

### 2b. Station directory resolution (inside `__init__`)

Replace the old `metadata_csv` iteration with `station_splits.csv`:
```python
splits = pd.read_csv(splits_csv)
if category_filter:
    splits = splits[splits applies category logic]
if split_filter:
    splits = splits[splits["split"].isin(split_filter)]

for _, r in splits.iterrows():
    has_sm = r["has_soil_moisture"] == True
    has_fl = r["has_flux"] == True
    cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
    # Match actual dir naming: ISMN_{network}_{station_name} or {source}_{station_id}
    if r["source_network"] == "ISMN":
        dir_name = f"ISMN_{r['network']}_{r['station_name']}"
    else:
        dir_name = f"{r['source_network']}_{r['station_id']}"
    sat_dir    = Path(data_root) / cat / dir_name
    label_file = sat_dir / "labels.nc"
    soil_path  = sat_dir / "soil" / "soil_patch.tif"
```

### 2c. ERA5 normalization in `load_era5_rolling`

After extracting raw ERA5 values:
```python
# log1p precipitation before normalizing
era5[:, ERA5_VARS.index("tp_sum")] = np.log1p(era5[:, ERA5_VARS.index("tp_sum")])
era5 = (era5 - means) / (stds + 1e-8)   # z-score
```
Stats are loaded once in `__init__` from `era5_stats.json` and passed into the function.

### 2d. Add `load_sif_rolling(sat_dir, year, target_doy)`

```
SIF/sif_*.nc → variables: "sif"
Returns: sif_vals (N,1) float32, doys (N,) long, valid (N,) bool
If file missing → all-zero tensors, valid=False (model ignores via key_mask)
MAX_SIF = 50 (cap)
```

### 2e. Add `load_twsa_rolling(sat_dir, year, target_doy)`

```
TWSA/twsa_*.nc → variable: "lwe"
Monthly observations (~12 per year in window)
Returns: twsa_vals (N,1) float32, doys (N,) long, valid (N,) bool
If file missing → all-zero tensors
MAX_TWSA = 12
```

### 2f. Update `__getitem__` return dict

Add to the returned dict:
```python
"sif"       : sif_vals,          # (MAX_SIF, 1) fp32
"sif_doys"  : sif_doys,          # (MAX_SIF,) long
"sif_valid" : sif_valid,         # (MAX_SIF,) bool — False if dropped
"twsa"      : twsa_vals,         # (MAX_TWSA, 1) fp32
"twsa_doys" : twsa_doys,         # (MAX_TWSA,) long
"twsa_valid": twsa_valid,        # (MAX_TWSA,) bool
```

**Modality dropout** (only when `self.training=True`):
```python
if self.training and random.random() < 0.5:
    sif_valid[:] = False   # zero out entire SIF modality
if self.training and random.random() < 0.5:
    twsa_valid[:] = False
```

---

## Step 3 — Fix and extend `model.py`

### 3a. Current state of modality embeddings (what exists vs. what's missing)

**Exists:**
- `soil_modality_emb`: Embedding(1, 768) — soil tokens get a type signal
- `static_modality_emb`: Embedding(2, 768) — DEM (0), LULC (1)
- `spatial_modality_emb`: Embedding(2, 768) — target-day S2 (0), S1 (1)
- `scale_emb`: Embedding(4, 768) — pyramid levels for satellite + static tokens

**Missing (bug):** ERA5, S2-history, and S1-history tokens have **no modality type embedding**

In `_build_sequence` currently:
- S2 history gets: `pyr + doy_pe + rel_pos + scale_e` — no type identifier
- S1 history gets: same formula — **indistinguishable from S2 in the sequence**
- ERA5 gets: `era5_mlp_out + doy_pe + rel_pos` — no type identifier, no scale

The transformer cannot tell S2 from S1 from ERA5 tokens (only their learned representations differ). Fix before training.

### 3b. Add missing modality embeddings in `__init__`

```python
self.hist_modality_emb = nn.Embedding(2, d_model)  # 0=S2hist, 1=S1hist
self.era5_modality_emb = nn.Embedding(1, d_model)
self.sif_modality_emb  = nn.Embedding(1, d_model)
self.twsa_modality_emb = nn.Embedding(1, d_model)
```

### 3c. Fix `_build_sequence` — satellite history tokens

```python
for i, (pyr, doys, valid, rel_pos) in enumerate([
    (s2_pyr, s2_doys, s2_valid, batch["s2_rel_pos"]),
    (s1_pyr, s1_doys, s1_valid, batch["s1_rel_pos"]),
]):
    hist_mod = self.hist_modality_emb(torch.tensor(i, device=device))  # (768,)
    sat_tok  = pyr + pe + rp + scale_e + hist_mod   # broadcast over (B, MAX_ACQ, 4)
```

### 3d. Fix `_build_sequence` — ERA5 tokens

```python
era5_mod = self.era5_modality_emb(torch.zeros(1, dtype=torch.long, device=device))  # (1, 768)
era5_tok = era5_tok + era5_pe + era5_rp + era5_mod
```

### 3e. Add SIF and TWSA MLPs and their sequence blocks

In `__init__`:
```python
self.sif_mlp  = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, d_model))
self.twsa_mlp = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, d_model))
```

In `_build_sequence` after ERA5 block:
```python
for vals, doys, valid, mlp, mod_emb in [
    (batch["sif"],  batch["sif_doys"],  batch["sif_valid"],  self.sif_mlp,  self.sif_modality_emb),
    (batch["twsa"], batch["twsa_doys"], batch["twsa_valid"], self.twsa_mlp, self.twsa_modality_emb),
]:
    tok = mlp(vals.float().to(device))                                           # (B, N, 768)
    pe  = sinusoidal_pe(doys.reshape(-1), self.d_model).reshape(B, -1, self.d_model)
    rp  = self.rel_pos_emb(doys.reshape(-1).clamp(0, 364)).reshape(B, -1, self.d_model)
    m   = mod_emb(torch.zeros(1, dtype=torch.long, device=device))               # (1, 768)
    tok = tok + pe + rp + m
    tokens.append(tok)
    is_pad.append(~valid.to(device))
```

### 3f. How temporal encoding works

Two signals encode where each token sits in the 365-day rolling window — both are **additive** to every token that has a time dimension:

| Signal | Type | Encodes | Range |
|---|---|---|---|
| `sinusoidal_pe(doy)` | Fixed | Absolute calendar day-of-year | 1–365 |
| `rel_pos_emb(pos)` | Learned | Position in rolling window (0=oldest, 364=target day) | 0–364 |

ERA5 future masking: `key_padding_mask=True` for ERA5 tokens where `day_index >= target_doy` — prevents attending to future weather when predicting today's SM.

---

## Step 4 — Fix and extend `train.py`

### 4a. Update CONFIG paths

```python
CONFIG = {
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "data_root"     : "/gpfs/work3/0/prjs1968/data",
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",
    "category_filter": ["sm_only"],
    ...
}
```

### 4b. Use pre-defined splits

Replace `train_val_split()` with two separate dataset instances:
```python
train_dataset = SoilMoistureDataset(..., split_filter=["train"], training=True)
val_dataset   = SoilMoistureDataset(..., split_filter=["val"],   training=False)
```

### 4c. Add W&B logging

```python
import wandb
wandb.init(project="soil-moisture-phd", name=CONFIG["run_name"], config=CONFIG)

# Each epoch:
wandb.log({
    "epoch": epoch,
    "train/loss": train_loss,
    "val/loss": val_loss,
    **{f"val/{d}/ubRMSE": m["ubRMSE"] for d, m in metrics.items()},
    **{f"val/{d}/MAE":    m["MAE"]    for d, m in metrics.items()},
    "lr": optimizer.param_groups[0]["lr"],
})
```

### 4d. Add argparse

```python
parser = argparse.ArgumentParser()
parser.add_argument("--lr",         type=float, default=CONFIG["lr"])
parser.add_argument("--batch-size", type=int,   default=CONFIG["batch_size"])
parser.add_argument("--n-layers",   type=int,   default=CONFIG["n_layers"])
parser.add_argument("--run-name",   type=str,   default="baseline_sm_only")
args = parser.parse_args()
CONFIG.update({k: v for k, v in vars(args).items() if v is not None})
```

---

## Step 5 — Create `slurm/train.sh`

```bash
#!/bin/bash
#SBATCH --job-name=sm_train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

conda run -n terramind python train.py "$@"
```

Submit baseline: `sbatch slurm/train.sh --run-name baseline`
Submit sweep:    `sbatch slurm/train.sh --run-name sweep_lr3e5 --lr 3e-5`

---

## Step 6 — Hyperparameter tuning plan (manual SLURM sweeps)

**Recommendation: manual sweeps tracked via W&B** (simplest for a research context, no extra dependencies).

### Phase A — Baseline
Submit one job with default config. Expected wall-time: ~16–20h on A100 for 100 epochs. Monitor W&B for val ubRMSE convergence and loss curves.

**Accept baseline if:** val ubRMSE < 0.07 m³/m³ for 0-10 cm within 30 epochs.

### Phase B — 3 targeted sweep jobs (submit after baseline reaches epoch ~20)

| Run name | `--lr` | `--batch-size` | `--n-layers` | Why |
|---|---|---|---|---|
| `sweep_lr_low` | `3e-5` | 4 | 6 | Safer for convergence; tests underfitting |
| `sweep_lr_high` | `3e-4` | 4 | 6 | Faster convergence; tests instability |
| `sweep_deeper` | `1e-4` | 4 | 8 | More transformer capacity |

Compare all runs on W&B. Select best config for Phase 2 (sm_and_flux expansion).

### Phase C (only if needed)
- Gradient accumulation (effective batch 16–32) if Phase A/B show batch-size sensitivity
- Lower Huber delta (0.02) if model over-smooths dry/wet extremes

---

## Execution order

1. `python compute_era5_stats.py` (login node, ~10 min)
2. Edit `dataset.py` (Steps 2a–2f)
3. Edit `model.py` (Step 3)
4. Edit `train.py` (Step 4)
5. Create `slurm/train.sh` (Step 5)
6. `wandb login` on Snellius (one-time)
7. Quick smoke test: `python train.py --run-name smoke_test` on login node for 2 epochs with a 10-station subset (add `--max-stations 10` flag)
8. `sbatch slurm/train.sh --run-name baseline`
9. After epoch ~20: submit Phase B sweeps

---

## Verification

- Dataset builds without error and reports expected sample count (expect ~200k–400k samples for sm_only, 2016–2023)
- Model forward pass runs: `python -c "from model import *; ..."` with a dummy batch
- W&B run appears at wandb.ai/ktm.prajwalkhanal/soil-moisture-phd
- `sbatch` job starts, BEGIN email arrives, `logs/train_JOBID.out` shows epoch progress
- Checkpoint files written to `/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only/`

---

## Files modified

| File | Change |
|---|---|
| `compute_era5_stats.py` | **New** — one-time ERA5 stats computation |
| `dataset.py` | Refactor `__init__`, add `load_sif_rolling`, `load_twsa_rolling`, ERA5 z-score, modality dropout |
| `model.py` | Add `sif_mlp`, `twsa_mlp`, extend `_build_sequence` |
| `train.py` | Fix paths, use CSV splits, add W&B + argparse |
| `slurm/train.sh` | **New** — SLURM training job |
| `csvs/era5_stats.json` | **New** — generated by compute_era5_stats.py |

---

## Ablation Study: Temporal Window Size

**Question:** How much temporal context does the model actually need?

Surface SM (0–10 cm) responds to rain within hours; deep SM (30–100 cm) integrates over months.
Reducing the window size also reduces sequence length, cutting transformer compute roughly linearly.

### Experiment design

Submit all 5 runs as a SLURM array after the baseline (365-day) is stable (~20–30 epochs).

| Run name | `--window-days` | Expected behaviour |
|---|---|---|
| `window_365` | 365 (baseline) | Full annual cycle; captures seasonality for all depths |
| `window_270` | 270 | Loses ~3 months; may affect deep layer |
| `window_180` | 180 | Half-year; surface should still be fine |
| `window_90`  | 90  | One season; expect degradation in 30–100 cm |
| `window_30`  | 30  | One month; expect clear degradation in all but 0–10 cm |

Track **val ubRMSE per depth** in W&B. Primary signal: at what window size does 30–100 cm ubRMSE degrade by >10% relative to baseline.

### Code changes needed to run this sweep

`window_days` is currently hardcoded. Before submitting:

1. Add `"window_days": 365` to `CONFIG` in `train.py` and a `--window-days` CLI arg.
2. Pass `window_days` into `SoilMoistureDataset` and through to `load_era5_rolling` (change `n = 365` to `n = window_days`).
3. In `model.py` `_build_sequence`, change the ERA5 relative position encoding from `torch.arange(365)` to `torch.arange(window_days)` and the ERA5 future mask accordingly.
4. `MAX_S2` / `MAX_S1` can stay fixed — a shorter window naturally yields fewer acquisitions.
5. The `rel_pos_emb` is `nn.Embedding(365, d_model)` — clamp any rel_pos to `[0, window_days-1]` or expand the embedding to max 365 with indexing clamped to `window_days`.

### Expected wall-time per run

~16–20h on A100 (same as baseline). Submit as array:
```bash
for W in 270 180 90 30; do
    sbatch slurm/train.sh --run-name window_${W} --window-days ${W}
done
```
