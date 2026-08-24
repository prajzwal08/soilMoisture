#!/bin/bash
#SBATCH --job-name=twi_big
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --output=logs/twi_hand_big_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.4 — TWI/HAND for the 8 regions of 20e6 cells or more: 0.478e9 cells (half the
# total) over just 8 regions, 282 stations. The largest is region 0 at 262e6 cells
# (420 x 562 km, 146 stations).
#
# Only 4 workers, but a lot of memory per worker: a single region holds the DEM, the
# conditioned DEM, MFD and D8 accumulation, slope, TWI, HAND and the pyflwdir index
# arrays simultaneously, and WhiteboxTools holds its own copy in a separate process.
# For region 0 that is ~1 GB per float32 array, so tens of GB in flight.
#
# --tier3 is OFF here: it adds a second full-region accumulation pass on exactly the
# regions where a pass is most expensive. The cross-check runs on all 345 bulk
# regions instead, which is a far larger sample for the same question.
#
# Resume-safe: a region with terrain_30m.tif already present is skipped, so if the
# 24 h wall clock is hit, requeueing continues where it stopped.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n terramind --no-capture-output python -u build_twi_hand.py \
    --min-cells 20000000 --workers 4 \
    --region-log csvs/terrain_region_log_big.csv \
    --station-log csvs/terrain_station_log_big.csv

date
echo "regions with terrain: $(ls -d /gpfs/work3/0/prjs1968/data/terrain/region_*/terrain_30m.tif 2>/dev/null | wc -l) / 353"
