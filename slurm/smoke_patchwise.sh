#!/bin/bash
#SBATCH --job-name=sm_smoke_pw
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_pw_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# Stage 2a smoke test (§35.20 step 5).
#
# TWO GPUs, not one, and not four. The failure this exists to catch is DDP with
# find_unused_parameters unset (train.py:902, default False): any parameter that is
# constructed but never gradiented raises RuntimeError on step 1. That is INVISIBLE on a
# single device, and four GPUs would just cost more to learn the same thing.
#
# Runs three things in sequence, cheapest and most decisive first:
#   1. --arch unet          the baseline must not have moved
#   2. --arch patchwise --driver-mode memory
#   3. --arch patchwise --driver-mode concat
#
# Each on 3 stations for 1-2 epochs. A pass here is not evidence the model is any good; it is
# evidence the plumbing does not crash and the baseline is unchanged.

set -euo pipefail
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200

echo "=================================================================="
echo "Store pre-flight (§35.6: a purged .zarray reads back as fill_value"
echo "with no exception, so soil silently becomes all zeros)"
echo "=================================================================="
if ! conda run -n terramind --no-capture-output python verify_zarr_store.py \
        --root /gpfs/scratch1/shared/pkhanal/zarr \
        --out "csvs/verify_smoke_${SLURM_JOB_ID}.csv" --workers 32; then
    echo "STORE VERIFICATION FAILED — run: sbatch slurm/restore_zarr.sh"
    exit 1
fi

run () {
    local name="$1"; shift
    echo
    echo "=================================================================="
    echo "SMOKE: $name"
    echo "  $*"
    echo "=================================================================="
    # Checkpoints live under CONFIG["checkpoint_dir"] (train.py:181/889), NOT a relative
    # ./checkpoints. A relative rm here silently matched nothing, and because train.py
    # auto-resumes from last.pt a second smoke run would have RESUMED the first one's weights
    # while looking like a clean pass. Fully-qualified, and scoped to smoke_* only.
    local CKROOT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only
    case "$name" in
        pw_*|unet) ;;                       # guard: never expand to anything but a smoke dir
        *) echo "refusing to clear checkpoints for unexpected run name '$name'"; exit 1 ;;
    esac
    rm -rf "${CKROOT:?}/smoke_${name:?}"
    conda run -n terramind --no-capture-output \
        torchrun --nproc_per_node=2 train.py \
            --run-name "smoke_${name}" \
            --max-stations 3 --max-epochs 2 \
            --batch-size 8 --num-workers 4 --prefetch-factor 2 \
            --max-train-batches 20 --max-val-batches 10 \
            "$@"
    echo "SMOKE $name: OK"
}

# 1. The baseline must not have moved. This runs FIRST: if the patchwise work broke the unet
#    path there is no point testing anything else.
run unet --arch unet --use-memmap

# 2. and 3. The new path, both driver wirings. --use-memmap is deliberately omitted: the .npy
#    memmaps exist only for the anchor L3/L6/L9 reads, and patchwise drops those entirely.
run pw_memory --arch patchwise --driver-mode memory --driver-layers 2
run pw_concat --arch patchwise --driver-mode concat --driver-layers 2

echo
echo "=================================================================="
echo "ALL SMOKE RUNS COMPLETED"
echo "=================================================================="
