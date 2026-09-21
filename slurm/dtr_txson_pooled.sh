#!/bin/bash
#SBATCH --job-name=dtr_tx
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
F=/gpfs/work3/0/prjs1968/soilMoisture/fig/dtr_txson
conda run -n terramind --no-capture-output python -c "
import py_compile; py_compile.compile('plot_dtr_sm_pooled.py', doraise=True); print('syntax OK')"
echo '--- TxSON, all stations, all dt ---'
conda run -n terramind --no-capture-output python plot_dtr_sm_pooled.py \
    --network TxSON --label "   —   TxSON only, all 40 stations" \
    --out "$F/dtr_sm_pooled_surface_TxSON.png"
echo '--- TxSON, dt 6-9 h ---'
conda run -n terramind --no-capture-output python plot_dtr_sm_pooled.py \
    --network TxSON --dt-lo 6 --dt-hi 9 --label "   —   TxSON, dt_hours in [6, 9)" \
    --out "$F/dtr_sm_pooled_surface_TxSON_6-9h.png"
