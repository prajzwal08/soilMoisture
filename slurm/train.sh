#!/bin/bash
#SBATCH --job-name=sm_train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=120:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

set -euo pipefail
ulimit -n 65536   # memmap FDs: 612 stations × 3 keys × 8 workers ≈ 14k handles

cd /gpfs/work3/0/prjs1968/soilMoisture

# /dev/shm L12 preload: ~145 GB measured (shared across 4 ranks as one physical copy).
# Budget at val→train boundary: 145+159+242+61 = 607 GB → 113 GB headroom vs 720G.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200   # 2 hours; covers cold-GPFS val on first epoch

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py "$@"
