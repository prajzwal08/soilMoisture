#!/bin/bash
#SBATCH --job-name=eval_stations
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_stations_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"

conda run -n terramind --no-capture-output python eval_stations.py "$@"
