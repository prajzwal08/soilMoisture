#!/bin/bash
#SBATCH --job-name=tier1_probe
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tier1_probe_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

conda run -n terramind --no-capture-output python tier1_probe.py "$@"
