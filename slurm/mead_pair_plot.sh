#!/bin/bash
#SBATCH --job-name=mead_pair
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_pair_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_pair_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Runbook §69: Ne1 vs Ne2 per-year figure. Args pass through to plot_mead_pair.py.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID  commit $(git rev-parse --short HEAD)  $(date)"
$(which python) plot_mead_pair.py "$@"
