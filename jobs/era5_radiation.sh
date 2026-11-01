#!/bin/bash
#SBATCH --job-name=era5_rad
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 -- fetch ssrd/strd only, for all 993 stations.
#
# --cpus-per-task=16 with 6 in-process workers is deliberate, not a mismatch:
# rome bills a 16-CPU minimum regardless, and the workers are I/O-bound GEE
# threads, not compute.  `N_WORKERS = 6` is inherited from
# download_era5land_gee.py:70 ("reduced from 16 to avoid 429 rate limits").
#
# WALL TIME.  Measured from data/logs/: 4413 station-years took 13 h 47 m at 6
# workers (~11.3 s each).  ~9,000 station-years is therefore ~28 h against this
# 24 h wall.  DO NOT raise concurrency to fit -- resubmit instead.  Every
# station-year whose rad_{year}.nc exists is skipped, so a second submission
# picks up exactly where the first stopped and costs nothing.
#
# Smoke first:
#   sbatch --time=01:00:00 jobs/era5_radiation.sh --stations \
#     ISMN_SNOTEL_PortGraham,ISMN_USCRN_Cape-Charles-5-ENE,ISMN_SCAN_Combate
#   (the three STRATEGY_BUFFER stations -- they exercise the riskiest path)
# Full run only after the smoke output has been checked.

set -eo pipefail

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

cd /gpfs/work3/0/prjs1968/soilMoisture

python download_era5_radiation.py "$@"
