#!/bin/bash
#SBATCH --job-name=s60_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_s60_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_s60.sh: same flags, 20 stations, 3 epochs, warmup 20 (mechanics only).
# Check: CONFIG static_dropout=0.5 weight_decay=0.5; [static-dropout check] ~0.5 each, untied=0; epoch files.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
rm -rf $CK/s60_smoke_20261005
bash slurm/train.sh --run-name s60_smoke_20261005 \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 20 --save-every-epoch \
    --static-dropout 0.5 --weight-decay 0.5 \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "=== epoch files ==="; ls -la $CK/s60_smoke_20261005/epochs/
