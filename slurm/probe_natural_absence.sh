#!/bin/bash
#SBATCH --job-name=probe_absence
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_natural_absence_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §73.7 natural whole-group absence q per Shapley player (train split, 5,000 random samples, no
# dropout), overall / by month / by |latitude| band. CPU only, no training.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python probe_natural_absence.py \
    --n 5000 --out csvs/natural_absence_${SLURM_JOB_ID}.csv
