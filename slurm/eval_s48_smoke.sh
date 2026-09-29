#!/bin/bash
#SBATCH --job-name=eval_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_smoke_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# Smoke for the full s48 evaluation: every stage of the split eval and the TxSON eval,
# on a few stations / one tile / one year. Nothing is written to eval_output/<run>.
#
#   sbatch slurm/eval_s48_smoke.sh <run>

set -eo pipefail
exec 2>&1
echo "=== eval_s48_smoke  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
unset S48_CACHE_ROOT

RUN="${1:?usage: sbatch slurm/eval_s48_smoke.sh <run-name>}"
IN="eval_output/${RUN}_smoke"
OUT="figures/eval/${RUN}_smoke"
TILE="ISMN_TxSON_CR200-18"
RUN_PY="conda run -n terramind --no-capture-output python"
rm -rf "${IN}" "${OUT}"          # smoke dirs only; never the run's own output

echo ""; echo "───────── 1. splits: val + oot, 5 stations ─────────"
$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt best.pt --batch-size 128 \
    --num-workers 8 --splits val oot --max-stations 5 --out-dir "${IN}"

echo ""; echo "───────── 2. TxSON readout: tile ${TILE} ─────────"
$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt best.pt --batch-size 128 \
    --num-workers 8 --pixel-csv csvs/txson_readouts.csv --tag txson \
    --pixel-tiles "${TILE}" --out-dir "${IN}"

echo ""; echo "───────── 3. pass criteria on the parquets ─────────"
$RUN_PY - "${IN}" <<'EOF'
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); ok = True
for f in sorted(d.glob("predictions_*.parquet")):
    df = pd.read_parquet(f)
    p = df["pred"].to_numpy(np.float64)
    fin = np.isfinite(p).all(); rng = (p.min() >= 0.0) and (p.max() <= 0.6)
    line = f"  {f.name:<40s} rows {len(df):>7,}  pred {p.min():.3f}-{p.max():.3f}  finite {fin}  in[0,0.6] {rng}"
    if "oot" in f.name:
        y = sorted(df["year"].unique()); line += f"  years {y}"
        ok &= set(y) <= {2023, 2024, 2025}
    if "network" in f.name:
        n = df.groupby(["tile", "station"]).ngroups; line += f"  readouts {n}"
        ok &= n == 6
    ok &= bool(fin) and bool(rng)
    print(line)
m = json.loads((d / "manifest.json").read_text())
print("  manifest:", {k: m.get(k) for k in ("run_name", "epoch", "git_sha", "checkpoint")})
print("SMOKE PARQUET CHECKS:", "PASS" if ok else "FAIL")
EOF

echo ""; echo "───────── 4. metrics (val tie-back numbers are 5-station, not the gate) ─────────"
$RUN_PY eval_metrics.py --in-dir "${IN}" --out-dir "${IN}"

echo ""; echo "───────── 5. TxSON figures on the one tile ─────────"
TILES="${TILE}" IN="${IN}" OUT="${OUT}" bash slurm/eval_txson_figures.sh "${RUN}"

echo ""; echo "───────── 6. 20 m tile map, 2019, with the map/eval agreement check ─────────"
$RUN_PY plot_tile_sm_map.py --run-name "${RUN}" --tile "${TILE}" --years 2019 2019 \
    --out-dir "${OUT}/tile" --check-preds "${IN}/predictions_network_txson.parquet"

echo ""
echo "=== All done $(date) ==="
ls -R "${OUT}" | head -60
