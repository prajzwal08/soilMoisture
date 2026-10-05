#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_final_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §66 Phase A: GPU half of the final-model test evaluation (§59 nolst_L3_wu200_20261005, best.pt = ep10).
#   1) predictions on OOS / OOT / OOST   (val parquet already written by job 27614390)
#   2) TxSON network readout (every station pixel in every TxSON tile) for the maps + tile time series
# CPU half afterwards: slurm/eval_figures.sh <run> --style bw ; slurm/eval_txson_figures.sh <run>
#
#   sbatch slurm/eval_final.sh            full
#   sbatch slurm/eval_final.sh smoke      5 stations per split + one TxSON tile -> eval_output/<run>_smoke
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
unset S48_CACHE_ROOT          # eval must read the all-years GPFS cache, never a /dev/shm training copy
RUN=nolst_L3_wu200_20261005
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/${RUN}/best.pt
RUN_PY="conda run -n terramind --no-capture-output python"
MODE="${1:-full}"
if [[ "${MODE}" == smoke ]]; then
    OUT=eval_output/${RUN}_smoke; LIMIT="--max-stations 5"; TILES="--pixel-tiles ISMN_TxSON_CR200-18"
    rm -rf "${OUT}"
else
    OUT=eval_output/${RUN}; LIMIT=""; TILES=""
fi
echo "=== eval_final ${MODE}  job ${SLURM_JOB_ID}  $(date)  on $(hostname) ==="
echo "Run ${RUN}  ckpt ${CKPT}  out ${OUT}"

echo ""; echo "───────── 1. splits OOS / OOT / OOST ─────────"
$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 128 --num-workers 8 \
    --splits oos oot oost ${LIMIT} --out-dir "${OUT}"

echo ""; echo "───────── 2. TxSON network readout ─────────"
$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 128 --num-workers 8 \
    --pixel-csv csvs/txson_readouts.csv --tag txson ${TILES} --out-dir "${OUT}"

echo ""; echo "───────── 3. sanity on the parquets ─────────"
$RUN_PY - "${OUT}" <<'EOF'
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); ok = True
for f in sorted(d.glob("predictions_*.parquet")):
    df = pd.read_parquet(f)
    p = df["pred"].to_numpy(np.float64)
    fin = bool(np.isfinite(p).all()); rng = bool(p.min() >= 0.0 and p.max() <= 0.6)
    ok &= fin and rng
    print(f"  {f.name:<44s} rows {len(df):>9,}  stations {(df['station_key'].nunique() if 'station_key' in df else -1):>4}  "
          f"pred {p.min():.3f}-{p.max():.3f}  finite {fin}  in[0,0.6] {rng}")
m = json.loads((d / "manifest.json").read_text())
print(f"  manifest checkpoint={m.get('checkpoint')}  epoch={m.get('epoch')}")
print("SANITY", "PASS" if ok else "FAIL")
EOF
echo "=== done $(date) ==="
