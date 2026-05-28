#!/bin/bash
#SBATCH --job-name=test_cm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/test_cloud_masking_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/test_cloud_masking_%j.err

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

python cloud_masking_inference.py \
    --station ISMN_SNOTEL_Truckee#2 \
    --batch-size 16 \
    --io-workers 3

echo ""
echo "Finished: $(date)"
