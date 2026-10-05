#!/bin/bash
#SBATCH --job-name=nolst_L3_wd03
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_L3_wd03_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §65: §59 (train_nolst_L3_wu200.sh) with --weight-decay 0.3 (was 0.05); nothing else changed. 10 epochs.
# AdamW max total shrink over ~2000 steps: lr*wd*steps ~12% (0.05 gave ~2%). Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_wd03_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 200 --save-every-epoch --weight-decay 0.3 --max-epochs 10
