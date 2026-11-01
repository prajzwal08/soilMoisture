#!/bin/bash
#SBATCH --job-name=nl_grid
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=logs/nl_grid_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Netherlands land-only 2.24 km tile grid + map (plan §68). CPU only, a few minutes.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n soilmoisture --no-capture-output python make_nl_grid.py
conda run -n soilmoisture --no-capture-output python plot_nl_grid.py
date
