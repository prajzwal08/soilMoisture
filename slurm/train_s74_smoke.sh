#!/bin/bash
#SBATCH --job-name=nolst_L3_wu200_sd03_rng2_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_nolst_L3_wu200_sd03_rng2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_s74.sh (§74): same flags, 20 stations, 3 epochs, warmup 20 (mechanics only).
# Check: CONFIG static_dropout=0.3 coarse_dropout=0.3 lambda_range=2 sm_range_lo=0.0 sm_range_hi=0.9 n_layers=3;
#   a "range" line every epoch with a finite train_pen and out_of_range %; no NaN; 4-rank DDP reduce works;
#   epochs/epoch_00{1,2,3}.pt.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
rm -rf $CK/nolst_L3_wu200_sd03_rng2_smoke_20261013
bash slurm/train.sh --run-name nolst_L3_wu200_sd03_rng2_smoke_20261013 \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --fine-inputs bands --warmup-steps 20 --save-every-epoch \
    --static-dropout 0.3 --weight-decay 0.05 \
    --lambda-range 2 --sm-range-lo 0.0 --sm-range-hi 0.9 \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "(range lines: grep \"^ +range \" in this log)"
echo "=== epoch files ==="; ls -la $CK/nolst_L3_wu200_sd03_rng2_smoke_20261013/epochs/
