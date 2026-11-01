#!/bin/bash
#SBATCH --job-name=mead_pattern
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_pattern_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_pattern_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Runbook §69.6: fixed-pattern tests 1 (overlap) + 2 (static) on the Mead daily maps.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID  commit $(git rev-parse --short HEAD)  $(date)"
python -m py_compile analyse_mead_map_pattern.py
$(which python) analyse_mead_map_pattern.py "$@"
