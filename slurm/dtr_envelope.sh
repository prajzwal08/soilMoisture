#!/bin/bash
#SBATCH --job-name=dtr_env
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== dtr_env job=$SLURM_JOB_ID $(date) ==="
conda run -n terramind --no-capture-output python analyse_dtr_envelope.py
echo "--- plausible band only ---"
conda run -n terramind --no-capture-output python analyse_dtr_envelope.py --band
echo "=== done $(date) ==="
