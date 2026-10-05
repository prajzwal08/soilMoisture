#!/bin/bash
#SBATCH --job-name=s61_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_s61_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_s61.sh: same flags, 20 stations, 3 epochs, warmup 20 (mechanics only).
# Check: CONFIG fine_inputs=indices skip_zero_init=False coarse 0.5 wd 0.2 lr 1e-4; fine ratio > 0 at ep1.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
rm -rf $CK/s61_smoke_20261005
bash slurm/train.sh --run-name s61_smoke_20261005 \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.5 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 20 --save-every-epoch \
    --fine-inputs indices --no-skip-zero-init --weight-decay 0.2 --lr 1e-4 \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "=== epoch files ==="; ls -la $CK/s61_smoke_20261005/epochs/
