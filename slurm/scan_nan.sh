#!/bin/bash
#SBATCH --job-name=scan_nan
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scan_nan_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "Started: $(date)"

conda run -n terramind python scan_nan.py --max-stations 3

echo "Finished: $(date)"
