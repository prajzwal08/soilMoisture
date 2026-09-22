#!/bin/bash
#SBATCH --job-name=ls_st30_smoke
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke + zone-fix regression in one job.
#
# These 8 stations are EXACTLY the ones assert_grid_invariants() killed with SystemExit in
# job 27015709 -- Landsat delivered their scenes in the neighbouring UTM zone (430 wrong-zone
# scenes at CentraliaLake, 282 at Cascade#2, 254 at CamidelsNerets, 128 at AnchorRiverDivide),
# and Aniak additionally got MPC's malformed 'EPSG:3264'.  All 8 must now complete.
# They also span 6 UTM zones, so this is the multi-zone smoke as well.
#
# soilmoisture env. Download only, never combined with analysis in one job.

set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

STATIONS='Alexandria,AnchorRiverDivide,Aniak,CamidelsNerets,CasaPeriles,Cascade#2,CentraliaLake,Condom'

echo "=== landsat st30 SMOKE / zone regression  job=$SLURM_JOB_ID  $(date) ==="
echo "stations: $STATIONS"

conda run -n soilmoisture --no-capture-output python -m py_compile download_landsat_st30.py
echo "--- syntax OK ---"

conda run -n soilmoisture --no-capture-output python download_landsat_st30.py \
    --stations "$STATIONS" --workers 6 --shard-tag smoke --overwrite "$@"

echo "=== done $(date) ==="
