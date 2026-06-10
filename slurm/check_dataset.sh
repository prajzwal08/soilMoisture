#!/bin/bash
#SBATCH --job-name=sm_check_data
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch1/shared/pkhanal/logs/check_dataset_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

conda run -n terramind python check_dataset.py --n-samples 2000 --workers 8
