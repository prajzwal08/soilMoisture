#!/bin/bash
#SBATCH --job-name=spatial_het
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/spatial_het_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §23 spatial heterogeneity diagnostic.  Inference is only n_stations x 4 batch-1
# forward passes; wall time is dominated by dataset init and a few scene reads.
#   sbatch slurm/spatial_heterogeneity.sh --station ISMN_RSMN_Iasi --selftest \
#          --verify-against eval_output/predictions_oos.parquet
#   sbatch slurm/spatial_heterogeneity.sh

set -euo pipefail
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

conda run -n terramind --no-capture-output python plot_spatial_heterogeneity.py "$@"
