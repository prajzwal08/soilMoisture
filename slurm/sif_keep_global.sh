#!/bin/bash
#SBATCH --job-name=sif_keep
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sif_keep_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68.12: keep the WHOLE global TROPOMI L2B SIF daily files (S5P-PAL, L2B_SIF___), download only.
#   smoke : sbatch --array=0 --time=00:30:00 slurm/sif_keep_global.sh smoke     (first 5 days of 2018)
#   full  : sbatch --array=0-7 slurm/sif_keep_global.sh                         (one task per year 2018-2025)
# A resubmit re-fetches only missing files (existing non-empty files are skipped).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
KEEP=/projects/prjs1968/sif_tropomi_l2b
YEARS=(2018 2019 2020 2021 2022 2023 2024 2025)
YEAR=${YEARS[${SLURM_ARRAY_TASK_ID:-0}]}
EXTRA=""; [[ "${1:-full}" == smoke ]] && EXTRA="--max-days 5"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} year ${YEAR} on $(hostname)  $(date)"
conda run -n soilmoisture --no-capture-output python download_sif_tropomi.py \
    --keep-dir "${KEEP}" --years "${YEAR}" --end-date 2025-12-31 --workers 6 ${EXTRA}
du -sh "${KEEP}/${YEAR}"; ls "${KEEP}/${YEAR}" | wc -l
date
