#!/bin/bash
#SBATCH --job-name=smoke_eval
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/smoke_eval_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# CPU pre-flight for the eval path.  ALWAYS run this before slurm/eval_predict.sh.
#
#   sbatch slurm/smoke_eval.sh pw_stage2a_L3 best.pt      # must exit 0
#   sbatch slurm/eval_predict.sh pw_stage2a_L3 best.pt --out-dir eval_output/pw_stage2a_L3
#
# No GPU: it catches import errors, signature drift between eval/plot scripts and
# dataset.py / model.py, and argparse mismatches with slurm/eval_figures.sh -- the
# faults that killed three GPU jobs on 2026-08-27 without ever reaching CUDA.

set -eo pipefail
exec 2>&1

cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""     # prove the path does not secretly need a GPU
ulimit -n 65536

RUN="${1:-pw_stage2a_L3}"
CKPT="${2:-best.pt}"
shift 2 || true

echo "=== smoke_eval  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)   Run: ${RUN}   Checkpoint: ${CKPT}"

conda run -n terramind --no-capture-output python smoke_eval.py \
    --run-name "${RUN}" --ckpt "${CKPT}" "$@"

echo ""
echo "=== All done $(date) ==="
