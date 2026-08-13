#!/bin/bash
#SBATCH --job-name=landsat_st
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §29 Phase A — Landsat 8/9 C2 L2 Surface Temperature over TxSON.
#
#   sbatch slurm/landsat_st.sh tile    # one 2.24 km window, 247 scenes, minutes
#   sbatch slurm/landsat_st.sh aoi     # whole 35x35 km domain, ~300 scenes, ~30 min
#
# Resume is automatic via csvs/landsat_st_download_log.csv, keyed on (item_id, extent).

set -eo pipefail
exec 2>&1

EXTENT="${1:-aoi}"
shift || true

export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== landsat_st  extent=$EXTENT  job=$SLURM_JOB_ID  $(date) ==="

conda run -n soilmoisture --no-capture-output python download_landsat_st_mpc.py \
    --extent "$EXTENT" \
    --workers 12 \
    "$@"

echo "=== done $(date) ==="
