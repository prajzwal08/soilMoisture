#!/bin/bash
#SBATCH --job-name=sm_smoke_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# TEMPORARY smoke-only script: runs on the wide-open gpu_a100 partition instead of
# the saturated gpu_h100. A100 = 40 GB VRAM (vs 80 GB H100) and 480 GB RAM (vs 720 GB),
# so mem is trimmed and batch size MUST be reduced at the CLI (--batch-size 32) to fit.
# For functional plumbing checks only (e.g. confirm TV-off trains). Full runs use train.sh.

set -euo pipefail
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

echo "Cleaning stale SHM caches..."
rm -rf /dev/shm/sm_l12_* 2>/dev/null || true
echo "SHM clean."

conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py "$@"
