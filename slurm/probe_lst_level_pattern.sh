#!/bin/bash
#SBATCH --job-name=lst_lvl_pat
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_lst_level_pattern_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §52: LST level (tile mean − T2m) vs within-tile pattern, against observed SM.
# Observed data only, no model. Smoke: sbatch slurm/probe_lst_level_pattern.sh --max-stations 20

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n terramind --no-capture-output python probe_lst_level_pattern.py --workers 64 "$@"
date
