#!/bin/bash
#SBATCH --job-name=bf_rverify
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bf_rverify_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50.8 verify. Usage: sbatch slurm/backfill_repair_verify.sh BACKUP_DIR ST1 ST2 ...
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
BK=$1; shift
python backfill_repair.py --verify --backup "$BK" --stations "$@"
