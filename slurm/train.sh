#!/bin/bash
#SBATCH --job-name=sm_train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=600G
#SBATCH --time=120:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
ulimit -n 65536   # memmap FDs: 612 stations × 3 keys × 8 workers ≈ 14k handles

cd /gpfs/work3/0/prjs1968/soilMoisture

# L12 cache re-enabled: 600G budget covers ~91GB shared pages at init.
# Eliminates zarr disk reads for L12 tokens during training, removing batch stalls.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=3600   # 1 hour; default 30 min risks timeout during zarr stalls

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py --num-workers 8 "$@"
