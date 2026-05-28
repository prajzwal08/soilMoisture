#!/bin/bash
#SBATCH --job-name=tm_trial
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/terramind/bin/python
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj
DATA_DIR=/gpfs/work3/0/prjs1968/data

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Started    : $(date)"
echo ""

# ── background GPU utilisation monitor ───────────────────────────────────────
# Polls every 5 s; written to a separate log for post-analysis
GPU_LOG=$DATA_DIR/logs/trial_gpu_util_${SLURM_JOB_ID}.csv
nvidia-smi dmon -s u -d 5 -o T > "$GPU_LOG" &
DMON_PID=$!
echo "GPU monitor PID: $DMON_PID  (log: $GPU_LOG)"
echo ""

# ── trial run ─────────────────────────────────────────────────────────────────
# --trial 5     : 5 stations — enough to warm up torch.compile and measure throughput
# --batch-size 8: safe for 20 GB MIG slice
# --num-workers 3: leave 1 CPU for main thread
# --compile     : enable torch.compile kernel fusion
$PYTHON precompute_terramind.py \
    --trial      5    \
    --batch-size 8    \
    --num-workers 3   \
    --compile

echo ""
echo "Encoding finished : $(date)"
echo ""

# ── stop GPU monitor ──────────────────────────────────────────────────────────
kill $DMON_PID 2>/dev/null

# ── output validation ─────────────────────────────────────────────────────────
echo "=========================================="
echo "OUTPUT VALIDATION"
echo "=========================================="

$PYTHON - <<'PYEOF'
import sys
from pathlib import Path
import torch

DATA_DIR = Path("/gpfs/work3/0/prjs1968/data")
SPLITS   = ["sm_only", "sm_and_flux", "flux_only"]

total_files = 0
total_bytes = 0
errors      = []

for split in SPLITS:
    for station_dir in sorted((DATA_DIR / split).iterdir()):
        for modality, suffix, expected_shape in [
            ("S2L2A",  "_L12.pt", (196, 768)),
            ("S1RTC",  "_L12.pt", (196, 768)),
            ("DEM",    "_L12.pt", (196, 768)),
            ("LULC",   "_L12.pt", (196, 768)),
        ]:
            mod_dir = station_dir / modality
            if not mod_dir.exists():
                continue
            pts = sorted(mod_dir.glob(f"*{suffix}"))
            for pt in pts:
                total_files += 1
                total_bytes += pt.stat().st_size
                try:
                    t = torch.load(pt, weights_only=True, map_location="cpu")
                    if t.shape != torch.Size(expected_shape):
                        errors.append(f"shape mismatch {pt}: {t.shape}")
                    if not torch.isfinite(t).all():
                        errors.append(f"non-finite {pt}")
                except Exception as exc:
                    errors.append(f"load error {pt}: {exc}")

print(f"  .pt files found  : {total_files}")
print(f"  total size       : {total_bytes / 1e6:.1f} MB")
print(f"  errors           : {len(errors)}")
for e in errors[:20]:
    print(f"    {e}")
if not errors:
    print("  All tensors valid ✓")

# Expected sizes per file:
# L12.pt: (196, 768) fp16 = 196*768*2 bytes = 301,056 bytes ≈ 294 KB
# L9/L6/L3.pt: same
print(f"\n  Expected per L12.pt: {196*768*2/1024:.0f} KB  "
      f"(got avg {total_bytes/max(total_files,1)/1024:.0f} KB)")
PYEOF

echo ""
echo "=========================================="
echo "GPU UTILISATION SUMMARY"
echo "=========================================="
echo "Full log: $GPU_LOG"
# Print average SM utilisation (column 2 in dmon -s u output, skipping headers)
awk '/^[0-9]/ {sum+=$2; n++} END {if(n>0) printf "  avg SM util : %.0f%%\n  samples     : %d\n", sum/n, n}' "$GPU_LOG"

echo ""
echo "Finished : $(date)"
