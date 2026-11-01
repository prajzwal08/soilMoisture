#!/bin/bash
#SBATCH --job-name=eval_twente
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_twente_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# Time series for the 8 Netherlands (ISMN TWENTE) stations. All are OOS (nl_holdout)
# and end 2020-12, so there is no OOT. Reads eval_output/<run>/predictions_oos.parquet
# written by slurm/eval_predict.sh; no GPU, no re-inference.
#
#   sbatch --dependency=afterok:<eval_predict id> slurm/eval_twente_ts.sh <run>

set -eo pipefail
exec 2>&1

cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

RUN="${1:?usage: sbatch slurm/eval_twente_ts.sh <run-name> [extra plot args]}"
shift
IN="eval_output/${RUN}"
OUT="${OUT:-figures/eval/${RUN}/timeseries_twente}"   # e.g. OUT=figures/eval/<run>_paper/timeseries_twente
RUN_PY="conda run -n terramind --no-capture-output python"

echo "=== eval_twente  job ${SLURM_JOB_ID}  started $(date) ==="
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" \
    --out-dir "${OUT}" --splits oos --per-page 4 \
    --select named --stations \
        ISMN_TWENTE_Hupsel ISMN_TWENTE_ITCSM-07b ISMN_TWENTE_ITCSM-08b \
        ISMN_TWENTE_ITCSM-09 ISMN_TWENTE_ITCSM-13 ISMN_TWENTE_ITCSM-15b \
        ISMN_TWENTE_ITCSM-16b ISMN_TWENTE_ITCSM-17c "$@"

echo ""
echo "=== All done $(date) ==="
echo "Figures: ${OUT}/"
