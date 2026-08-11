#!/bin/bash
#SBATCH --job-name=capacity_check
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/capacity_check_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n terramind --no-capture-output python capacity_check.py "$@"
date
