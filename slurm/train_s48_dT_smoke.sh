#!/bin/bash
#SBATCH --job-name=sm_dT_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:30:00
#SBATCH --output=logs/s48_lst_ablation/dT_smoke_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §52 smoke for the dT_pixel thermal target (run afterok:slurm/verify_lst_level.sh): a
# 20-station, 4-epoch train with --lst-target dT_pixel and a 20-step warmup so lambda (and so
# the thermal term) switches on inside the smoke. Checkpoints in s48_lst_ablation/, never
# phase1_sm_only/. STOP after this; the full run needs the user's OK.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

rm -rf /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_lst_ablation/s48_dT_smoke_20260930
bash slurm/train.sh --run-name s48_dT_smoke_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_lst_ablation \
    --lst-target dT_pixel \
    --max-stations 20 --max-epochs 4 --warmup-steps 20 --max-val-batches 50
