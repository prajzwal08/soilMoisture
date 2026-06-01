#!/bin/bash
#SBATCH --job-name=final_qa
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/final_qa_%j.out

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo ""

cd /gpfs/work3/0/prjs1968/soilMoisture
python final_qa_check.py --workers 32

echo ""
echo "Finished : $(date)"
