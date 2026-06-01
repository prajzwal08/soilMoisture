#!/bin/bash
#SBATCH --job-name=dem_missing
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/dem_missing_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/dem_missing_%j.err

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo ""

cd /gpfs/work3/0/prjs1968/soilMoisture
python redownload_dem_missing.py

echo ""
echo "Finished : $(date)"
