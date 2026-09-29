#!/bin/bash
#SBATCH --job-name=bf_repair
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bf_repair_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50.8 phase 0d, up to the token splice: repaired raw rows (sibling+swap), their cloud masks
# (cloud_masking_inference.py redirected to s2_repair), their TerraMind tokens. The splice
# runs next under the unlock: BF_MODE=repair sbatch slurm/backfill_splice_guarded.sh ST...
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate terramind
python backfill_repair.py --raw --stations "$@"
conda activate sensei
python cloud_masking_inference.py --scratch-dir /gpfs/scratch1/shared/pkhanal/s2_repair \
    --data-dir /gpfs/scratch1/shared/pkhanal/s2_repair_cm --batch-size 16 --io-workers 3
conda activate terramind
python backfill_repair.py --encode --stations "$@"
