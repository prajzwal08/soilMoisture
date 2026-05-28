#!/bin/bash
#SBATCH --job-name=harmonize_s2
#SBATCH --array=0-1027
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/harmonize_s2_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/harmonize_s2_%A_%a.err

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job    : $SLURM_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

$PYTHON harmonize_s2_pre2022.py \
    --start-idx $SLURM_ARRAY_TASK_ID \
    --end-idx   $((SLURM_ARRAY_TASK_ID + 1))

echo ""
echo "Finished: $(date)"
