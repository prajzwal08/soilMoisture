#!/bin/bash
#SBATCH --job-name=gra_landsat_plot
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# ANALYSIS ONLY -- terramind env, never combined with the download job (36.18).
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python -c "
import py_compile; py_compile.compile('plot_gra_landsat.py', doraise=True); print('syntax OK')"
conda run -n terramind --no-capture-output python plot_gra_landsat.py "$@"
