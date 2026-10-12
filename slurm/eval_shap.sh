#!/bin/bash
#SBATCH --job-name=eval_shap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_shap_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §73 exact group-Shapley (K = 5: era5, sat, dem, lulc, soil; 32 coalitions per sample).
#   sbatch slurm/eval_shap.sh verify [RUN]   verify_shap_groups.py: recipes == training dropout,
#                                            all-on == plain forward, efficiency, phi_sat dummy
#   sbatch slurm/eval_shap.sh smoke  [RUN]   1 OOS station  -> eval_output/<RUN>_shap_smoke (~5 SBU)
#   sbatch slurm/eval_shap.sh full   [RUN]   OOS (221), then OOT -> eval_output/<RUN>_shap (~460 SBU OOS)
# RUN defaults to the §64 SHAP-ready retrain. eval_predict.py refuses a checkpoint whose training
# never withheld a player. Every submit needs the user's OK; full only after the §73 gate (go).
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
unset S48_CACHE_ROOT          # eval reads the all-years GPFS cache, never a /dev/shm training copy
MODE="${1:-smoke}"
RUN="${2:-nolst_L3_wu200_sd05_20261005}"
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/${RUN}/best.pt
RUN_PY="conda run -n terramind --no-capture-output python"
echo "=== eval_shap ${MODE}  run ${RUN}  job ${SLURM_JOB_ID}  $(date)  on $(hostname) ==="
$RUN_PY shap_groups.py
case "${MODE}" in
  verify)
    $RUN_PY verify_shap_groups.py --ckpt "${CKPT}" --n 64 ;;
  smoke)
    OUT=eval_output/${RUN}_shap_smoke; rm -rf "${OUT}"
    $RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 64 --num-workers 8 \
        --splits oos --max-stations 1 --shap --shap-chunk 4 --out-dir "${OUT}" ;;
  full)
    OUT=eval_output/${RUN}_shap
    $RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 64 --num-workers 8 \
        --splits oos oot --shap --shap-chunk 4 --out-dir "${OUT}" ;;
  *) echo "unknown mode ${MODE} (verify | smoke | full)"; exit 2 ;;
esac
echo "=== done $(date) ==="
