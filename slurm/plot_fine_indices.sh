#!/bin/bash
#SBATCH --job-name=plot_fine_idx
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/plot_fine_indices_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n terramind --no-capture-output python plot_fine_indices.py "$@"
date
