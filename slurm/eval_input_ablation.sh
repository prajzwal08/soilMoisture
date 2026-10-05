#!/bin/bash
#SBATCH --job-name=input_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/input_ablation_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §67 Phase B: input attribution (reliance) of the frozen baseline on OOS, eval-only.
# 13 passes from ONE dataset build: baseline + 10 cross-station shuffles + 2 within-station.
#   sbatch slurm/eval_input_ablation.sh          full  (222 OOS stations)
#   sbatch slurm/eval_input_ablation.sh smoke    5 stations -> eval_output/..._ablation_smoke
# --mem 180G = the 1-GPU share of an H100 node (300G billed two GPUs, §66).
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_CLIP=0,1
ulimit -n 65536
unset S48_CACHE_ROOT           # all-years GPFS cache, never a /dev/shm training copy
RUN=baseline_selected_20261005
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/nolst_L3_wu200_20261005/best.pt
RUN_PY="conda run -n terramind --no-capture-output python"
MODE="${1:-full}"
if [[ "${MODE}" == smoke ]]; then
    OUT=eval_output/${RUN}_ablation_smoke; LIMIT="--max-stations 5"; rm -rf "${OUT}"
else
    OUT=eval_output/${RUN}_ablation; LIMIT=""
fi
FIG=figures/eval/${RUN}_paper/ablation$([[ "${MODE}" == smoke ]] && echo _smoke)
PASSES="none era5 sat s2 s1 fine dem lulc soil sif twsa era5:within_station s1:within_station"
echo "=== input_ablation ${MODE}  job ${SLURM_JOB_ID}  $(date)  on $(hostname) ==="
echo "ckpt ${CKPT}   out ${OUT}   passes: ${PASSES}"

$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 128 --num-workers 8 \
    --splits oos ${LIMIT} --out-dir "${OUT}" --seed 0 --ablate ${PASSES}

echo ""; echo "───────── paired comparison vs the baseline pass ─────────"
$RUN_PY compare_ablation.py --base "${OUT}/predictions_oos.parquet" \
    ${OUT}/predictions_oos_*_s0.parquet --csv "${OUT}/ablation_summary.csv"

echo ""; echo "───────── positive control: ERA5 cross-station must clearly hurt ─────────"
$RUN_PY - "${OUT}/ablation_summary.csv" <<'EOF'
import sys, pandas as pd
s = pd.read_csv(sys.argv[1])
e = s[s["ablation"].str.contains("era5_cross_station")]
print(e[["depth", "n", "d_ubRMSE", "d_ubRMSE_lo", "d_ubRMSE_hi", "d_ubRMSE_pct"]].to_string(index=False))
ok = len(e) and (e["d_ubRMSE_lo"] > 0).all()
print("POSITIVE CONTROL", "PASS" if ok else "FAIL (ERA5 shuffle did not clearly hurt: check the harness)")
EOF

echo ""; echo "───────── figure ─────────"
$RUN_PY plot_input_ablation.py --summary "${OUT}/ablation_summary.csv" --out-dir "${FIG}"
echo "=== done $(date) ==="
