#!/bin/bash
#SBATCH --job-name=lst_dT_stats
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/s48_lst_ablation/lst_dT_stats_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §52: freeze the dT_pixel Huber knee + head bias from the full training set, once.
# Smoke: sbatch slurm/compute_lst_dT_stats.sh --limit 10   (writes nothing)
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
conda run -n terramind --no-capture-output python compute_lst_dT_stats.py --workers 64 "$@"
