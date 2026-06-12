#!/bin/bash
#SBATCH --job-name=sm_smoke_h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_h100_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

# Print GPU info
conda run -n terramind python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# Background GPU utilization sampler (restricted to this job's GPU)
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
  --format=csv -l 5 > "/gpfs/scratch1/shared/pkhanal/logs/gpu_usage_${SLURM_JOB_ID}.csv" &
GPU_MON_PID=$!

# Smoke test: 3 stations, 2 epochs, batch=64 — confirms H100 compatibility + VRAM usage
# num_workers/prefetch_factor reduced from defaults (16/4): at ~31MB/sample and
# batch_size=64 (~2GB/batch), 16x4=64 queued batches would need ~125GB of shared
# memory alone -- 4x2=8 queued batches (~16GB) leaves headroom under --mem=128G.
conda run -n terramind python train.py \
  --run-name smoke_h100 \
  --max-stations 3 \
  --max-epochs 2 \
  --batch-size 64 \
  --num-workers 4 \
  --prefetch-factor 2 \
  "$@"

kill "$GPU_MON_PID" 2>/dev/null || true
echo "--- GPU utilization summary ---"
awk -F', ' 'NR>1 {gsub(/ %/,"",$2); gsub(/ MiB/,"",$3); gsub(/ MiB/,"",$4); sum+=$2; n++; if($3>maxmem) maxmem=$3} END {if(n>0) printf "avg GPU util: %.1f%%  max VRAM used: %d MiB / %s\n", sum/n, maxmem, $4}' \
  "/gpfs/scratch1/shared/pkhanal/logs/gpu_usage_${SLURM_JOB_ID}.csv"
