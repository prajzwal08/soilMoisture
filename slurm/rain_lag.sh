#!/bin/bash
#SBATCH --job-name=rain_lag
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/rain_lag_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §72 rain-event lag ladder (SM, S1 VV/VH, S2 bands + NDVI/NDMI around rain events).
#   sbatch --cpus-per-task=16 --mem=48G --time=00:30:00 slurm/rain_lag.sh smoke   20 stations
#   sbatch slurm/rain_lag.sh                                                      all SM stations
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
RUN_PY="conda run -n terramind --no-capture-output python"
MODE="${1:-full}"
echo "=== rain_lag ${MODE}  job ${SLURM_JOB_ID}  $(date)  git $(git rev-parse --short HEAD) ==="
if [[ "${MODE}" == smoke ]]; then
    OUT=eval_output/rain_lag_smoke; FIG=figures/rain_lag/lag_ladder_smoke
    $RUN_PY analyse_rain_lag.py --smoke 20 --workers "${SLURM_CPUS_PER_TASK}" --out "${OUT}"
else
    OUT=eval_output/rain_lag; FIG=figures/rain_lag/lag_ladder
    $RUN_PY analyse_rain_lag.py --workers "${SLURM_CPUS_PER_TASK}" --out "${OUT}"
fi
$RUN_PY plot_rain_lag.py --in-dir "${OUT}" --out "${FIG}"
echo "=== done $(date) ==="
