#!/bin/bash
#SBATCH --job-name=dtr_all
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §37.9 -- consolidate every station's DTR bundle.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== dtr_all job=$SLURM_JOB_ID host=$(hostname) $(date) ==="
conda run -n soilmoisture --no-capture-output \
    python consolidate_dtr.py --workers 64 --overwrite
echo "=== done $(date) ==="
