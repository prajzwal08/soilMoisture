#!/bin/bash
#SBATCH --job-name=lst_residual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/lst_tmean_diff/probe_lst_residual_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Is observed LST − T2m information the model is MISSING?
# 1) val predictions of §59 (nolst_L3_wu200_20261005, best.pt = ep10, SELECT 0.0488)
# 2) on Landsat scene days, r(model error, LST − T2m anomaly) vs r(SM anomaly, LST − T2m anomaly)
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
unset S48_CACHE_ROOT
RUN=nolst_L3_wu200_20261005
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/${RUN}/best.pt
OUT=eval_output/${RUN}
RUN_PY="conda run -n terramind --no-capture-output python"
echo "=== probe_lst_residual job ${SLURM_JOB_ID} $(date) on $(hostname) ==="
$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --splits val \
    --batch-size 128 --num-workers 8 --out-dir "${OUT}"
$RUN_PY probe_lst_residual.py --pred "${OUT}/predictions_val.parquet" --label "§59 3L ep10"
echo "=== done $(date) ==="
