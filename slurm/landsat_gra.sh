#!/bin/bash
#SBATCH --job-name=landsat_gra
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-9
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# DOWNLOAD ONLY -- soilmoisture env. Never combined with analysis in one job (36.18):
# the two conda envs are not interchangeable and terramind has no MPC stack.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
CID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" csvs/gra_thermal_cluster_ids.txt)
echo "cluster ${SLURM_ARRAY_TASK_ID}: ${CID}"
conda run -n soilmoisture --no-capture-output python download_landsat_st_mpc.py \
    --extent cluster --cluster "${CID}" --workers 12 "$@"
