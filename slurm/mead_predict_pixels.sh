#!/bin/bash
#SBATCH --job-name=mead_pred
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_pred_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_pred_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Runbook §69.3: frozen §66 model, Ne1 and Ne2 tiles, every day 2017-2024, both
# stations' pixels read from each tile's map + ERA5 identity check. Inference only.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ZARR_ROOT RAW_ZARR_ROOT S48_CACHE_ROOT          # training stores, never the NL ones
echo "Job $SLURM_JOB_ID  commit $(git rev-parse --short HEAD)  $(date)"
md5sum dataset.py eval_predict.py model.py            # dataset.py carries uncommitted §68 edits
$(which python) predict_mead_pixels.py --years "${1:-2017-2024}" --save-maps
echo "Finished $(date)"
