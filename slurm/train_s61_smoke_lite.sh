#!/bin/bash
#SBATCH --job-name=s61_smoke_lite
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/lst_tmean_diff/smoke_s61_lite_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Light smoke for train_s61.sh (2026-10-05, H100 queue full): same §61 flags, 20 stations, 3 epochs,
# but 1 A100 (no torchrun / DDP), NO /dev/shm staging (the dataset reads the §48 cache on GPFS, which is
# the default S48_CACHE_ROOT) -> fits a partly-free node. Mechanics only; the full run stays on 4 H100.
set -euo pipefail
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU $CUDA_VISIBLE_DEVICES"

echo "=== store-integrity pre-flight ==="
conda run -n terramind --no-capture-output python verify_zarr_store.py \
    --root /gpfs/scratch1/shared/pkhanal/zarr --out "csvs/verify_preflight_${SLURM_JOB_ID}.csv" --workers 16 \
    || { echo "STORE VERIFICATION FAILED"; exit 1; }
conda run -n terramind --no-capture-output python preflight_s48.py \
    || { echo "§48 CACHE PRE-FLIGHT FAILED"; exit 1; }
echo "=== pre-flight passed (no shm staging: reading GPFS cache) ==="

CK=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff
rm -rf $CK/s61_smoke_20261005
conda run -n terramind --no-capture-output python train.py --run-name s61_smoke_20261005 \
    --checkpoint-dir $CK \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.5 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 20 --save-every-epoch \
    --fine-inputs indices --no-skip-zero-init --weight-decay 0.2 --lr 1e-4 \
    --max-stations 20 --max-epochs 3 --max-val-batches 50
echo "=== epoch files ==="; ls -la $CK/s61_smoke_20261005/epochs/
