#!/bin/bash
#SBATCH --job-name=bf_verify
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bf_verify_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 phase 7. Usage: sbatch slurm/backfill_verify.sh BACKUP_DIR ST1 ST2 ...
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
BK=$1; shift
python -m py_compile backfill_catalogue.py backfill_s2_download.py backfill_merge.py backfill_verify.py prepare_s48_cache.py eval_predict.py && echo "compile OK"
python backfill_verify.py --backup "$BK" --stations "$@"
