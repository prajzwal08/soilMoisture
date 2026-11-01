#!/bin/bash
#SBATCH --job-name=dtr_g0
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# G0 / 36.21(i) -- the thermal-arm kill test, across every consolidated station.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== dtr_g0 job=$SLURM_JOB_ID host=$(hostname) $(date) ==="
conda run -n terramind --no-capture-output python -c "
import py_compile; py_compile.compile('gate_dtr_static.py', doraise=True); print('syntax OK')"
conda run -n terramind --no-capture-output python gate_dtr_static.py --workers 64
echo "=== done $(date) ==="
