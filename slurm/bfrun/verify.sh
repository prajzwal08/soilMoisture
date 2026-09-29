#!/bin/bash
#SBATCH --job-name=bf_verify
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_verify_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
R=$1   # run dir: stations.txt, backup.ok, *.ok
conda activate terramind
python backfill_verify.py --backup /gpfs/work3/0/prjs1968/backfill_backup/20260928 --stations-file $R/splice.ok --ok-out $R/verify.ok --workers 16
[ -s $R/verify.ok ]
