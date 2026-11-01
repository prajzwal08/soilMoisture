#!/bin/bash
#SBATCH --job-name=dtr_txson
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §37.9 + §37.10 -- consolidate the TxSON DTR bundles, then look at them.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== dtr_txson job=$SLURM_JOB_ID host=$(hostname) $(date) ==="

conda run -n soilmoisture --no-capture-output python -c "
import py_compile
for f in ('consolidate_dtr.py','plot_dtr_txson.py'): py_compile.compile(f, doraise=True)
print('syntax OK')"

echo "--- consolidate (TxSON) ---"
conda run -n soilmoisture --no-capture-output \
    python consolidate_dtr.py --network TxSON --workers 40 --overwrite

echo "--- plot ---"
conda run -n terramind --no-capture-output \
    python plot_dtr_txson.py --network TxSON --n-dates 5

echo "=== done $(date) ==="
