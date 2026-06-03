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

# Smoke test: 3 stations, 2 epochs, batch=64 — confirms H100 compatibility + VRAM usage
conda run -n terramind python train.py \
  --run-name smoke_h100 \
  --max-stations 3 \
  --max-epochs 2 \
  --batch-size 64
