#!/bin/bash
#SBATCH --job-name=plot_lst_deseason
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=logs/lst_tmean_diff/plot_lst_deseason_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Deseasonalised LST level / pattern vs SM 0-10, all stations in the §52 probe scenes.csv (CPU, ~1 min).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python plot_lst_level_deseason_all.py "$@"
