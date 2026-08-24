#!/bin/bash
#SBATCH --job-name=merit_hydro
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/merit_hydro_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.6 — MERIT Hydro reference windows, the necessary condition on our own terrain.
# 993 stations x 25 km, bands upa/upg/hnd/elv/dir at 90 m.
#
# 8 workers, not Pool(64): this is a remote API with rate limits, and the earlier
# ERA5-Land job had to drop from 16 to 6 workers to stop hitting 429s. Backoff is
# 2/4/8 s and hard errors (403, RESOURCE_EXHAUSTED, type errors) fail fast.
#
# Resume-safe: a station with merit_hydro_25km.tif already present is skipped, and
# the per-station log is merged rather than overwritten.
#
# Auth: earthengine authenticate --auth_mode=notebook (the default mode shells out
# to gcloud, which is not installed on the login node). Credentials are read from
# ~/.config/earthengine/credentials, which is on home and visible from the node.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n soilmoisture --no-capture-output python download_merit_hydro_gee.py \
    --workers 8 --n-verify 40

date
echo "stations written: $(ls /gpfs/work3/0/prjs1968/data/*/*/MERIT/merit_hydro_25km.tif 2>/dev/null | wc -l) / 993"
