#!/bin/bash
#SBATCH --job-name=audit_duration
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/audit_duration_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/audit_duration_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Started : $(date)"
echo ""

$(which python) audit_station_duration.py --min-days 1095

echo ""
echo "Finished : $(date)"
