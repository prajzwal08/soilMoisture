#!/bin/bash
#SBATCH --job-name=dbl_check
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dbl_check_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50.8: per-scene evidence for the double-offset suspects (fresh download vs stored row).
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate soilmoisture
python check_double_offset.py --fetch
conda activate terramind
python check_double_offset.py --compare
