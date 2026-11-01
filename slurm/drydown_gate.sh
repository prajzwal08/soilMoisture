#!/bin/bash
#SBATCH --job-name=drygate
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/drygate_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.8 — the sufficiency gate. No GPU, no model: observed labels against derived
# terrain, on colocated pairs so station identity is differenced out, stratified by
# Koppen macro-climate and split wet/dry.
#
# Pool(64) over the 353 regions: each worker opens one region's terrain raster once
# and pulls every station inside it, rather than reopening a 1 GB raster per station.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date

conda run -n terramind --no-capture-output python -u gate_drydown_within_network.py --workers 64

date
