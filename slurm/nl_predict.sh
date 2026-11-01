#!/bin/bash
#SBATCH --job-name=nl_predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/nl_predict_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68 E1/E2: every-day soil moisture for the Overijssel grid with the FROZEN §66 model
# (baseline_selected_20261005), centre pixel (56,56), 2016-01-01..2025-12-31, then the cube.
# --mem 180G = one GPU's share of a gpu_h100 node (eval_final.sh); s48 memmaps the cache.
#
#   sbatch --time=02:00:00 slurm/nl_predict.sh check
#          TRAINING stores, 2 TWENTE stations with labels (no NL data needed):
#          1) check_predict_all_days.py dataset  -- flag off == pre-§68 code; observed-day subset of
#             predict_all_days == normal mode (same samples, same tensors)
#          2) eval_predict.py --predict-all-days on those 2 stations, compared with the existing
#             allday predictions (eval_output/baseline_selected_20261005/allday) on shared rows
#   sbatch --time=03:00:00 slurm/nl_predict.sh smoke    3 tiles (nl/data_smoke), gate S4, assemble --n-tiles 1
#   sbatch slurm/nl_predict.sh full                     684 tiles, gate S4, assemble (682 grid tiles)
#   sbatch --partition=rome --gpus=0 --cpus-per-task=16 slurm/nl_predict.sh assemble   re-assemble only
#
# The eval args are the ones the baseline's allday predictions were made with (slurm/eval_allday.sh,
# job 27627836): --run-name baseline_selected_20261005 --ckpt <abs best.pt> --batch-size 128
# --num-workers 8. Frozen stats: eval_predict.py reads csvs/era5_stats18.json + driver_stats.json; no refit.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
MODE="${1:?mode: check | smoke | full | assemble}"
RUN=baseline_selected_20261005
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/nolst_L3_wu200_20261005/best.pt
NL=/gpfs/scratch1/shared/pkhanal/nl
ALLDAY=eval_output/${RUN}/allday/predictions_oos.parquet
YEARS=2016-2025
RUN_PY="conda run -n terramind --no-capture-output python"
EVAL_ARGS=(--run-name "${RUN}" --ckpt "${CKPT}" --batch-size 128 --num-workers 8)
echo "=== nl_predict ${MODE}  job ${SLURM_JOB_ID}  $(date)  on $(hostname)  git $(git rev-parse --short HEAD) ==="

nl_env() {   # point dataset.py / eval_predict.py at the NL stores (§68.3 isolation rule)
    export ZARR_ROOT=${NL}/zarr RAW_ZARR_ROOT=${NL}/satellite_zarr S48_CACHE_ROOT=${NL}/s48cache
    export SOIL_STATION_CSV="$1"
    echo "  ZARR_ROOT=${ZARR_ROOT}  RAW_ZARR_ROOT=${RAW_ZARR_ROOT}  S48_CACHE_ROOT=${S48_CACHE_ROOT}"
    echo "  SOIL_STATION_CSV=${SOIL_STATION_CSV}"
}

gate_s4() {   # $1 = pred dir, $2 = tile CSV. §68.5 gate S4 on the per-tile parquets.
    $RUN_PY - "$1" "$2" "${ALLDAY}" <<'EOF'
import sys
from pathlib import Path
import numpy as np, pandas as pd
pred_dir, csv, allday = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = pd.read_csv(csv)
want = 3653 * 3
ok = True
files = {f.stem: f for f in (pred_dir / "tiles").glob("*.parquet")}
miss = sorted(set(t["dir_name"]) - set(files))
print(f"S4 tiles: {len(files)} parquets for {len(t)} CSV rows; missing {len(miss)} {miss[:5]}")
ok &= not miss
bad = []
for k, f in files.items():
    df = pd.read_parquet(f)
    p = df["pred"].to_numpy(np.float64)
    n_days = df["date"].nunique()
    r = (len(df) == want and n_days == 3653 and np.isfinite(p).all()
         and not df.duplicated(["date", "depth"]).any())
    if not r:
        bad.append((k, len(df), n_days, int((~np.isfinite(p)).sum())))
print(f"S4 rows == 3653 x 3, unique, finite: {len(files) - len(bad)} / {len(files)} tiles; bad {bad[:5]}")
ok &= not bad
allp = pd.concat([pd.read_parquet(f, columns=["pred"]) for f in files.values()])["pred"]
print(f"S4 pred range {allp.min():.3f}..{allp.max():.3f}; outside [0,1]: "
      f"{int(((allp < 0) | (allp > 1)).sum())} of {len(allp):,} (clipped at assemble)")
ref = pd.read_parquet(allday)
ref["depth"] = ref["depth"].astype(str)
ctl = t[t["is_control"].astype(str).str.lower() == "true"]
for r in ctl.itertuples(index=False):
    if r.dir_name not in files:
        continue
    a = pd.read_parquet(files[r.dir_name]); a["depth"] = a["depth"].astype(str)
    b = ref[ref["station_key"].astype(str) == r.control_of]
    m = a.merge(b, on=["date", "depth"], suffixes=("", "_ref"))
    if m.empty:
        print(f"S4 control {r.dir_name} vs {r.control_of}: no shared rows in allday -- not gated")
        continue
    d = (m["pred"].clip(0, 1) - m["pred_ref"].clip(0, 1)).abs()
    o = m["obs_ref"].notna()
    med = float(d[o].median()) if o.any() else float("nan")
    print(f"S4 control {r.dir_name} vs {r.control_of}: {len(m):,} shared rows "
          f"({int(o.sum()):,} observed); median |diff| observed {med:.4f}, all {d.median():.4f}, "
          f"max {d.max():.4f}; per depth observed: "
          f"{m[o].assign(d=d[o]).groupby('depth')['d'].median().round(4).to_dict()}")
    ok &= bool(np.isfinite(med) and med < 0.02)
print("GATE S4", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
}

case "${MODE}" in
check)
    unset ZARR_ROOT RAW_ZARR_ROOT S48_CACHE_ROOT SOIL_STATION_CSV   # training stores
    OUT=eval_output/nl_check
    rm -rf "${OUT}"
    echo ""; echo "───────── 1. dataset identity (2 TWENTE stations) ─────────"
    $RUN_PY check_predict_all_days.py dataset \
        --stations ISMN_TWENTE_ITCSM-07b ISMN_TWENTE_ITCSM-09 --years "${YEARS}" --tmp-dir "${OUT}"
    echo ""; echo "───────── 2. eval_predict --predict-all-days on the same 2 stations ─────────"
    $RUN_PY eval_predict.py "${EVAL_ARGS[@]}" --splits oos --splits-csv "${OUT}/_check_pad_stations.csv" \
        --predict-all-days --years "${YEARS}" --out-dir "${OUT}/pred"
    echo ""; echo "───────── 3. vs the baseline's allday predictions ─────────"
    $RUN_PY check_predict_all_days.py compare --tiles-dir "${OUT}/pred/tiles" --reference "${ALLDAY}"
    ;;
smoke|full)
    if [[ "${MODE}" == smoke ]]; then
        CSV=${NL}/data_smoke/station_splits.csv; OUT=eval_output/nl_overijssel_smoke
        ZOUT=nl_output/smoke/overijssel_sm_daily.zarr; NT=1
    else
        CSV=${NL}/data/station_splits.csv; OUT=eval_output/nl_overijssel
        ZOUT=nl_output/overijssel_sm_daily.zarr; NT=682
    fi
    nl_env "${CSV}"
    if [[ -e "${OUT}/tiles" ]]; then echo "${OUT}/tiles exists -- move it away first"; exit 1; fi
    echo ""; echo "───────── E1 predict (${MODE}) ─────────"
    $RUN_PY eval_predict.py "${EVAL_ARGS[@]}" --splits nl_grid --splits-csv "${CSV}" \
        --predict-all-days --years "${YEARS}" --out-dir "${OUT}"
    echo ""; echo "───────── gate S4 ─────────"
    gate_s4 "${OUT}" "${CSV}"
    echo ""; echo "───────── E2 assemble ─────────"
    $RUN_PY assemble_nl_sm.py --pred-dir "${OUT}" --tiles-csv "${CSV}" --out "${ZOUT}" \
        --n-tiles "${NT}" --model-tag "${RUN}" --overwrite
    ;;
assemble)
    $RUN_PY assemble_nl_sm.py --pred-dir eval_output/nl_overijssel \
        --tiles-csv ${NL}/data/station_splits.csv --model-tag "${RUN}" --overwrite
    ;;
*)
    echo "unknown mode ${MODE}"; exit 2 ;;
esac
echo "=== done $(date) ==="
