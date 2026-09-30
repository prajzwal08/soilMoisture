#!/bin/bash
#SBATCH --job-name=pw_ablation
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/pw_ablation_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# EVAL ONLY (user 2026-09-30): did the PATCHWISE model (pw_stage2a_L3 best.pt, ep2) use the
# TerraMind embeddings at all? Runs the §24 shuffle harness from the patchwise code
# (worktree at 2b04fe0, the last commit before that checkpoint's eval) on val, paired rows.
#   0 baseline | 1 sat cross_station | 2 sat within_station | 3 era5 cross_station (positive control)
# Each condition writes to its own dir (manifest.json is per dir; avoids a write race).
set -eo pipefail
exec 2>&1
WT=/gpfs/work3/0/prjs1968/wt_pw_ablation
OUT=/gpfs/work3/0/prjs1968/soilMoisture/eval_output/pw_stage2a_L3_ablation
cd $WT
echo "worktree HEAD: $(git rev-parse --short HEAD)   node $(hostname)   task ${SLURM_ARRAY_TASK_ID}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
case ${SLURM_ARRAY_TASK_ID} in
  0) A=(--ablate none);                                   D=base ;;
  1) A=(--ablate sat  --ablate-mode cross_station  --seed 0); D=sat_cross ;;
  2) A=(--ablate sat  --ablate-mode within_station --seed 0); D=sat_within ;;
  3) A=(--ablate era5 --ablate-mode cross_station  --seed 0); D=era5_cross ;;
esac
conda run -n terramind --no-capture-output python eval_predict.py \
    --run-name pw_stage2a_L3 --ckpt best.pt --batch-size 128 --num-workers 8 \
    --splits val --out-dir "$OUT/$D" "${A[@]}"
