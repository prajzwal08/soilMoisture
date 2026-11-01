#!/bin/bash
#SBATCH --job-name=s73_shapready_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_s73_shapready_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_s73.sh (§73 SHAP-ready retrain): same flags, 20 stations, 3 epochs, warmup 20.
# Check: [§73 shap-ready check] line, violations=0, every player withheld at about its p
#   (era5 .5, s2 .3, s1 .3, fine .1, dem/lulc/soil .5, sif/twsa .5, s2&s1 ~.09);
#   [static-dropout check] untied=0; stats shas == the §59 log (printed at the end).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
RUN=nolst_L3_wu200_shapready_smoke_20261012
rm -rf $CK/$RUN
bash slurm/train.sh --run-name $RUN \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --coarse-dropout-mode per_sensor \
    --fine-group-dropout 0.1 --sif-twsa-dropout 0.5 \
    --n-layers 3 --fine-inputs bands --warmup-steps 20 --save-every-epoch \
    --static-dropout 0.5 --weight-decay 0.05 \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "=== epoch files ==="; ls -la $CK/$RUN/epochs/
echo "=== stats shas: §59 log vs this smoke (must match) ==="
for k in era5_stats_sha driver_stats_sha fine_stats_sha; do
  a=$(grep -o "$k=[^ ]*" logs/lst_tmean_diff/train_nolst_L3_wu200_27575211.out | head -1)
  b=$(grep -o "$k=[^ ]*" logs/lst_tmean_diff/smoke_s73_shapready_${SLURM_JOB_ID}.out | head -1)
  [ "$a" = "$b" ] && echo "OK   $a" || echo "DIFF §59 $a vs smoke $b"
done
