#!/bin/bash
#SBATCH --job-name=nolst_L2_wu200_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_nolst_L2_wu200_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_s62.sh (§62): same flags, 20 stations, 3 epochs, warmup 20 (mechanics only).
# Check: CONFIG n_layers=2 fine_inputs=bands save_every_epoch=True; gap + [diag] lines each epoch; epochs/epoch_00{1,2,3}.pt.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
rm -rf $CK/nolst_L2_wu200_smoke_20261005
bash slurm/train.sh --run-name nolst_L2_wu200_smoke_20261005 \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 2 --fine-inputs bands --warmup-steps 20 --save-every-epoch \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "=== epoch files ==="; ls -la $CK/nolst_L2_wu200_smoke_20261005/epochs/
