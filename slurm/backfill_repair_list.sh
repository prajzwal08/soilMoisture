#!/bin/bash
#SBATCH --job-name=bf_rlist
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/bf_rlist_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50.8: build csvs/s2_repair_targets.csv and byte-compile the backfill/repair code.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python -m py_compile backfill_repair.py backfill_merge.py backfill_verify.py && echo "compile OK"
python backfill_repair.py --list
