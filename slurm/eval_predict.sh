#!/bin/bash
#SBATCH --job-name=eval_predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 16, deliberately NOT 64. The §35.33 shm preload fans out across a fork Pool, so more
# cores are faster -- but a gpu_h100 node is 64 cores / 4 GPUs shared by up to 4 jobs, and
# asking for all 64 takes the whole node and bills FOUR GPUs. 16 workers already cuts the
# preload from ~90 min to ~6, which is the win; the last 4x is not worth 2x the SBUs.
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=04:00:00
# --mem=300G, not the 120G used by evaluate_meeting.sh: the dataset preloads
# L12 tokens into RAM and the OOT split (train+val, 774 stations) needs ~156 GB
# on its own.  Measured from the zarr array shapes, not guessed.
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_predict_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -eo pipefail
exec 2>&1
echo "=== eval_predict  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536          # memmap FD pressure

RUN="${1:-cls_depth_star_reg}"
CKPT="${2:-best.pt}"
shift 2 || true
RUN_PY="conda run -n terramind --no-capture-output python"
echo "Run: ${RUN}  Checkpoint: ${CKPT}  Extra args: $*"

# §22.4 -- one GPU pass; every metric and figure downstream is CPU-only.
# "val" is included so the §22.6 hard gate (reproduce 0.0539/0.0500/0.0552)
# can be checked against the same code path as the held-out splits.
$RUN_PY eval_predict.py \
    --run-name    "${RUN}" \
    --ckpt        "${CKPT}" \
    --batch-size  128 \
    --num-workers 8 \
    --splits      val oos oot oost \
    "$@"

echo ""
echo "=== All done $(date) ==="
echo "Outputs in: /gpfs/work3/0/prjs1968/soilMoisture/eval_output/"
