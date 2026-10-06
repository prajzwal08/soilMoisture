#!/bin/bash
#SBATCH --job-name=effres_maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/effres_maps_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §70 effective resolution, GPU half (effres_maps.py) on the frozen baseline nolst_L3_wu200_20261005.
#   sbatch slurm/effres_maps.sh smoke   CR200-18 only, blocks 0,1,2,112 + checks + analysis on 16 CPUs
#   sbatch slurm/effres_maps.sh         all 40 TxSON tiles, blocks 0,2,4,8,14,28,56,112
# Then the CPU half: sbatch slurm/effres_analysis.sh
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
unset S48_CACHE_ROOT          # eval reads the all-years GPFS cache, as slurm/eval_final.sh
RUN_PY="conda run -n terramind --no-capture-output python"
MODE="${1:-full}"
echo "=== effres_maps ${MODE}  job ${SLURM_JOB_ID}  $(date)  on $(hostname)  git $(git rev-parse --short HEAD) ==="

if [[ "${MODE}" == smoke ]]; then
    OUT=eval_output/effres_txson_smoke; rm -rf "${OUT}"
    $RUN_PY effres_maps.py --blocks 0,1,2,112 --tiles ISMN_TxSON_CR200-18 --out-dir "${OUT}"

    echo ""; echo "───────── smoke checks ─────────"
    $RUN_PY - "${OUT}" <<'PY'
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); ok = True
r0 = pd.read_parquet(d / "readouts_L0.parquet")
k = ["tile", "station", "year", "doy", "depth"]
# (1) the plain pass reproduces eval_final's TxSON readout (bf16, different batch -> small diffs)
ref = pd.read_parquet("eval_output/baseline_selected_20261005/predictions_network_txson.parquet")
m = r0.merge(ref[k + ["pred"]], on=k, suffixes=("", "_ref"))
dmax = float((m["pred"] - m["pred_ref"]).abs().max())
print(f"(1) L0 vs eval_final readout: {len(m):,} of {len(r0):,} rows matched, max |diff| {dmax:.2e}")
ok &= len(m) > 0.95 * len(r0) and dmax < 2e-3
# (2) block 1 is the identity
r1 = pd.read_parquet(d / "readouts_L1.parquet")
d1 = float((r1["pred"].to_numpy() - r0["pred"].to_numpy()).__abs__().max())
print(f"(2) block 1 vs plain: max |diff| {d1:.2e}  (expect ~0; bf16 noise only)")
ok &= d1 < 2e-3
# (3) the whole-tile block moves the map at all (else the harness never reached the model)
for b in (2, 112):
    s = pd.read_parquet(d / f"summary_L{b}.parquet")
    print(f"(3) block {b}: median map RMS diff {s['rms_diff'].median():.2e}, "
          f"median map SD plain {s['sd_L0'].median():.2e} -> coarsened {s['sd_Lb'].median():.2e}")
s1 = pd.read_parquet(d / "summary_L1.parquet")
print(f"    block 1: median map RMS diff {s1['rms_diff'].median():.2e}")
print("SMOKE", "PASS" if ok else "FAIL")
PY
    echo ""; echo "───────── analysis on the smoke output ─────────"
    $RUN_PY effres_analysis.py --in-dir "${OUT}" --workers 16
else
    $RUN_PY effres_maps.py --out-dir eval_output/effres_txson
fi
echo "=== done $(date) ==="
