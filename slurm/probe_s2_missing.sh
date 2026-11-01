#!/bin/bash
#SBATCH --job-name=s2probe
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/s2probe_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Diagnostic: retry a sample of never-obtained S2 scenes once each, record why they fail.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture
python probe_s2_missing.py
