#!/bin/bash
#SBATCH --job-name=bf_preflight
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/bf_preflight_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate terramind
python -m py_compile backfill_merge.py backfill_repair.py backfill_verify.py backfill_cleanup.py \
    backfill_s2_download.py prepare_s48_cache.py cloud_masking_inference.py eval_predict.py
echo "compile OK"
