#!/bin/bash
#SBATCH --job-name=gra_thermal
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python -c "
import py_compile
for f in ('plot_gra_thermal.py','plot_gra_sm_timeseries.py','plot_gra_dtr_vs_sm.py'): py_compile.compile(f, doraise=True)
print('syntax OK')"
echo "--- step 2: thermal maps ---"
conda run -n terramind --no-capture-output python plot_gra_thermal.py "$@"
echo
echo "--- step 2b: soil-moisture time series ---"
conda run -n terramind --no-capture-output python plot_gra_sm_timeseries.py "$@"
echo
echo "--- step 2c: DTR vs soil moisture ---"
conda run -n terramind --no-capture-output python plot_gra_dtr_vs_sm.py "$@"
