#!/bin/bash
#SBATCH --job-name=sm_verify_pw
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/verify_pw_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# Full verification of the patchwise-only refactor (§35.22), in ONE job.
# Nothing here is run interactively — see the plan's verification list.
#
#   1  unit tests
#   2  the FROZEN snapshot still imports and still loads cls_depth_star_reg
#   3  no unet remnants in the live modules
#   4  smoke both driver modes on 2 GPUs (the DDP check is invisible on one)
#   5  measure the I/O win

set -uo pipefail
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200
FAIL=0
step () { echo; echo "=================== $* ==================="; }

step "1. unit tests"
for t in test_patchwise_model.py test_patchwise_dataset.py; do
    conda run -n terramind --no-capture-output python "$t" || { echo "FAILED: $t"; FAIL=1; }
done

step "2. FROZEN SNAPSHOT — the only route back to the baseline"
# §35.21: cls_depth_star_reg's 604 MB of weights exist exactly once, no backup. If this set
# stops importing, the baseline becomes unreachable.
conda run -n terramind --no-capture-output python - <<'PYEOF' || FAIL=1
import torch
import model_unet, dataset_unet, train_unet, ckpt_utils_unet          # noqa: F401
print("snapshot imports OK")
from pathlib import Path
p = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only/cls_depth_star_reg/best.pt")
m, cfg, ep = ckpt_utils_unet.load_checkpoint(p, "cpu")
n = sum(q.numel() for q in m.parameters())
assert hasattr(m, "decoder"), "snapshot did not build the U-Net decoder"
print(f"baseline loads: epoch {ep}, {n:,} params, decoder present")
PYEOF

step "3. no unet remnants in the live modules"
if grep -nE "UNetDecoder|_cpu_pyramid_pool|select_anchor_zarr|anchor_l[0-9]|STATION_ROW|total_variation_loss|use_mmap|\"arch\"" model.py dataset.py train.py; then
    echo "FAILED: unet remnants above"; FAIL=1
else
    echo "clean"
fi

step "4. smoke, 2 GPUs, both driver modes"
CKROOT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only
for mode in memory concat; do
    echo "--- driver_mode=$mode ---"
    rm -rf "${CKROOT:?}/smoke_pwonly_${mode:?}"
    conda run -n terramind --no-capture-output \
        torchrun --nproc_per_node=2 train.py \
            --run-name "smoke_pwonly_${mode}" \
            --driver-mode "$mode" --driver-layers 2 \
            --max-stations 3 --max-epochs 2 \
            --batch-size 8 --num-workers 4 --prefetch-factor 2 \
            --max-train-batches 20 --max-val-batches 10 \
        || { echo "FAILED: smoke $mode"; FAIL=1; }
done

step "5. I/O — per-sample payload after the narrow read"
conda run -n terramind --no-capture-output python - <<'PYEOF' || FAIL=1
from dataset import SoilMoistureDataset
ds = SoilMoistureDataset("csvs/station_splits.csv", "csvs/era5_stats.json",
                         category_filter=["sm_only"], split_filter=["val"],
                         training=False, max_stations=2, token_sel="station")
s = ds[0]
tot = 0
for k, v in sorted(s.items()):
    if hasattr(v, "nbytes"):
        tot += v.nbytes
        if v.nbytes > 20_000:
            print(f"  {k:16s} {tuple(v.shape)!s:22s} {v.nbytes/1024:9.1f} KB")
print(f"  TOTAL per sample: {tot/1024:.1f} KB")
print("  (was ~1.6 MB before the anchor drop; ~30 MB was materialised in the worker)")
assert "s2_pyr" not in s and "anchor_l12" not in s, "pooled/anchor keys still emitted"
print("  no pooled or anchor keys: OK")
PYEOF

step "RESULT"
[ "$FAIL" -eq 0 ] && echo "ALL VERIFICATION PASSED" || echo "VERIFICATION FAILED"
exit $FAIL
