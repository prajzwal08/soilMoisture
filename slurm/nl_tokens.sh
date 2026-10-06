#!/bin/bash
#SBATCH --job-name=nl_tok
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nl_tok_%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68 D1 -> B4 -> D2 -> D3 for the Overijssel grid: raw TIFs -> cloud mask + cloudy-scene filter
# -> raw zarr -> TerraMind tokens + cm + drivers -> token masks + values18 -> §48 cache.
# Same writers, same order as the training store (runbook §68.3, logs 2026-05-28..06-12):
#   cloud_masking_inference -> filter_cloudy_tiles --analyze/--delete -> convert_satellite_to_zarr
#   -> retokenize --mode terramind -> retokenize --mode cm-fill -> compute_s1_dem_lulc_token_masks
#   -> splice_era5_radiation -> prepare_s48_cache.
# Every root is a NEW scratch root under nl/ (exports below); each python step refuses to run if a
# §68 station CSV is set while any output equals or sits under a training root, and every mode ends with
# check_isolation (any training-store mtime newer than this job's start -> exit 1).
#
# The per-tile nc (ERA5Land meteo_*/rad_*, TWSA, SIF, soil) MUST be complete before `tok`
# (cm-fill runs with --require-drivers and refuses a tile without them).
#
#   smoke : sbatch -p gpu_h100 --gpus=1 --cpus-per-task=16 --mem=120G -t 06:00:00 \
#                  slurm/nl_tokens.sh smoke
#           every step, in order, on the 3 tiles of nl/data_smoke/station_splits.csv, one GPU node.
#   full  : (each needs its own OK; afterok chains stop on any failure)
#     J1=$(sbatch --parsable -p gpu_a100 --gpus=1 --cpus-per-task=18 --mem=64G -t 02:00:00 \
#              --array=0-7 slurm/nl_tokens.sh cm 8)
#     J2=$(sbatch --parsable --dependency=afterok:$J1 -t 06:00:00 slurm/nl_tokens.sh prep)
#     J3=$(sbatch --parsable -p gpu_h100 --gpus=1 --cpus-per-task=16 --mem=120G -t 08:00:00 \
#              --array=0-15 --dependency=afterok:$J2:<C1/C2 fanout>:<C3 SIF>:<C4 soil> \
#              slurm/nl_tokens.sh tok 16)
#     J4=$(sbatch --parsable --dependency=afterok:$J3 -t 06:00:00 slurm/nl_tokens.sh post)
#   modes:  cm   B4  SEnSeIv2 masks, array task i of n            (sensei,    gpu_a100)
#           prep B4  filter_cloudy_tiles analyze + delete; D1 raw zarr (sensei + terramind, rome)
#           tok  D2  TerraMind tokens, then cm-fill (cm + ERA5/SIF/TWSA/soil + .complete),
#                    array task i of n                                (terramind + sensei, gpu_h100)
#           post D2  token masks, values18 splice; D3 §48 cache        (terramind, rome)
# Resubmitting any mode resumes (every writer skips finished tiles).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj
NL=/gpfs/scratch1/shared/pkhanal/nl
MODE=${1:?smoke|cm|prep|tok|post}
NT=${2:-1}
TI=${SLURM_ARRAY_TASK_ID:-0}
CPUS=${SLURM_CPUS_PER_TASK:-16}

export SOIL_SCRATCH_DIR=${NL}/satellite            # raw TIFs (B1-B3)
export SOIL_DATA_ROOT=${NL_NC_ROOT:-${NL}/data}    # per-tile nc + CloudMask/, {root}/sm_only/{dir}
export RAW_ZARR_ROOT=${NL}/satellite_zarr          # D1
export ZARR_ROOT=${NL}/zarr                        # D2
export S48_CACHE_ROOT=${NL}/s48cache               # D3
if [[ ${MODE} == smoke ]]; then
    export SOIL_STATION_CSV=${NL}/data_smoke/station_splits.csv
else
    export SOIL_STATION_CSV=${NL}/data/station_splits.csv
fi
TM="conda run -n terramind --no-capture-output python"
SE="conda run -n sensei --no-capture-output python"
LOGD=${NL}/logs/tokens_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${MODE}_${TI}
mkdir -p ${LOGD}
MARK=${LOGD}/.start
touch ${MARK}
echo "Job ${SLURM_JOB_ID} task ${TI}/${NT} mode ${MODE} on $(hostname)  $(date)"
env | grep -E '^(SOIL_|ZARR_ROOT|RAW_ZARR_ROOT|S48_CACHE_ROOT)' | sort

# This task's tiles (dir_name column, rows i::n), comma-separated. All §68 tiles are sm_only
# (has_soil_moisture True, has_flux False; make_nl_splits.py), so paths below use sm_only.
TILES=$($TM - <<EOF
import pandas as pd
d = pd.read_csv("${SOIL_STATION_CSV}")
assert (d.has_soil_moisture.astype(str) == "True").all() and (d.has_flux.astype(str) == "False").all()
print(",".join(d["dir_name"].tolist()[${TI}::${NT}]))
EOF
)
TILE_ARR=(${TILES//,/ })
echo "tiles: ${#TILE_ARR[@]}"
[[ ${#TILE_ARR[@]} -gt 0 ]] || { echo "no tiles for task ${TI}/${NT}"; exit 1; }

# Fail unless "$1" (a path with {d}) exists for every tile of this task.
require_all() {
    local miss=0
    for d in "${TILE_ARR[@]}"; do
        [[ -e ${1//\{d\}/$d} ]] || { echo "  MISSING ${1//\{d\}/$d}"; miss=$((miss+1)); }
    done
    echo "check $1: $((${#TILE_ARR[@]}-miss))/${#TILE_ARR[@]}"
    [[ ${miss} -eq 0 ]]
}

# Training stores and training csvs must be untouched by this job.
check_isolation() {
    local bad=0 n r f
    for r in /gpfs/work3/0/prjs1968/zarr_tokens /gpfs/scratch1/shared/pkhanal/zarr \
             /gpfs/work3/0/prjs1968/satellite_zarr /gpfs/scratch1/shared/pkhanal/satellite_zarr \
             /gpfs/scratch1/shared/pkhanal/s48cache; do
        n=$(find "$r" -maxdepth 3 -newer ${MARK} 2>/dev/null | wc -l)
        [[ $n -eq 0 ]] || { echo "  ISOLATION: $n entries under $r changed during this job"; bad=1; }
    done
    for f in csvs/station_splits.csv csvs/s48_cache_index.csv csvs/s48_cache_index.partial.csv \
             csvs/era5_splice_report.csv text/cloudy_tile_manifest.csv \
             /gpfs/work3/0/prjs1968/data/logs/zarr_failures.log; do
        [[ ! $f -nt ${MARK} ]] || { echo "  ISOLATION: $f modified"; bad=1; }
    done
    [[ $bad -eq 0 ]] && echo "isolation OK (no training store / csv touched)"
    return $bad
}

step_cm() {        # B4a: SEnSeIv2 per S2 TIF -> {SOIL_DATA_ROOT}/sm_only/{dir}/CloudMask/*.tif
    for d in "${TILE_ARR[@]}"; do mkdir -p ${SOIL_DATA_ROOT}/sm_only/$d; done   # else silent skip
    printf "%s\n" "${TILE_ARR[@]}" > ${LOGD}/tiles.txt
    $SE cloud_masking_inference.py --stations-file ${LOGD}/tiles.txt \
        --batch-size 16 --io-workers $((CPUS - 1)) 2>&1 | tee ${LOGD}/cm.log
    grep -q "Errors        : 0" ${LOGD}/cm.log || { echo "cloud mask errors"; return 1; }
    grep -q "Skipped       : 0" ${LOGD}/cm.log || { echo "tiles skipped (no data dir)"; return 1; }
    # every S2 TIF has its mask
    local d n2 nc bad=0
    for d in "${TILE_ARR[@]}"; do
        n2=$(ls ${SOIL_SCRATCH_DIR}/$d/S2L2A/*.tif 2>/dev/null | wc -l)
        nc=$(ls ${SOIL_DATA_ROOT}/sm_only/$d/CloudMask/*.tif 2>/dev/null | wc -l)
        [[ $n2 -gt 0 && $nc -ge $n2 ]] || { echo "  $d S2=$n2 CM=$nc"; bad=1; }
    done
    return $bad
}

step_filter() {    # B4b: >50 % invalid 16x16 patches -> delete the S2 TIF (training rule, 2026-05-28)
    local man=${SOIL_DATA_ROOT}/logs/cloudy_tile_manifest_${SLURM_JOB_ID}.csv
    $SE filter_cloudy_tiles.py --analyze --manifest ${man} > ${LOGD}/filter_analyze.log 2>&1
    grep -A4 "GLOBAL TILE SURVIVAL" ${LOGD}/filter_analyze.log
    grep -A2 "Stations with 0 tiles" ${LOGD}/filter_analyze.log
    # NOTE: deletes the rejected S2 TIFs from nl/satellite (as training did); a later re-run of
    # the S2 downloader would re-fetch them, so do not re-run B1 after this step.
    $SE filter_cloudy_tiles.py --delete --manifest ${man} 2>&1 | tee ${LOGD}/filter_delete.log
    grep -q "Errors                  : 0" ${LOGD}/filter_delete.log \
        || { echo "filter delete errors"; return 1; }
}

step_zarr() {      # D1: TIFs -> {RAW_ZARR_ROOT}/{dir}.zarr (s2 int16, s1 f16 dB, dem, lulc)
    $TM convert_satellite_to_zarr.py --stations "${TILES}" --workers ${CPUS} \
        --s1-dtype float16 --execute 2>&1 | tee ${LOGD}/zarr.log
    grep -q "Errors: 0" ${LOGD}/zarr.log || { echo "zarr conversion errors"; return 1; }
    require_all "${RAW_ZARR_ROOT}/{d}.zarr/.complete"
}

step_tok() {       # D2a: TerraMind L3/L6/L9/L12 + DEM/LULC tokens, then cm + drivers + .complete
    $TM retokenize_satellite_zarr.py --mode terramind --stations "${TILES}" \
        --batch-size 8 --device cuda --execute 2>&1 | tee ${LOGD}/terramind.log
    ! grep -qE "^\[ *[0-9]+/ *[0-9]+\] (ERR|MISS)" ${LOGD}/terramind.log \
        || { echo "terramind phase failed"; return 1; }
    require_all "${ZARR_ROOT}/sm_only/{d}/s2/l12/.zarray"
    $SE retokenize_satellite_zarr.py --mode cm-fill --stations "${TILES}" \
        --with-twsa --require-drivers \
        --batch-size 16 --device cuda --execute 2>&1 | tee ${LOGD}/cmfill.log
    ! grep -qE "^\[ *[0-9]+/ *[0-9]+\] (ERR|MISS|NODRV)|NOTOK" ${LOGD}/cmfill.log \
        || { echo "cm-fill phase failed"; return 1; }
    require_all "${ZARR_ROOT}/sm_only/{d}/.complete"
    require_all "${ZARR_ROOT}/sm_only/{d}/cm/masks/.zarray"
    require_all "${ZARR_ROOT}/sm_only/{d}/era5/values/.zarray"
    require_all "${ZARR_ROOT}/sm_only/{d}/twsa/lwe/.zarray"
    require_all "${ZARR_ROOT}/sm_only/{d}/sif/values/.zarray"
    require_all "${ZARR_ROOT}/sm_only/{d}/soil/.zarray"
}

step_post() {      # D2b token masks + values18, D3 §48 cache
    $TM compute_s1_dem_lulc_token_masks.py --stations "${TILES}" --execute 2>&1 \
        | tee ${LOGD}/token_masks.log
    grep -q "Missing satellite_zarr: 0  |  Errors: 0" ${LOGD}/token_masks.log \
        || { echo "token mask errors"; return 1; }
    require_all "${ZARR_ROOT}/sm_only/{d}/dem_token_mask/.zarray"
    require_all "${ZARR_ROOT}/sm_only/{d}/lulc_token_mask/.zarray"

    local rep=${LOGD}/era5_splice_report.csv
    $TM splice_era5_radiation.py --stations "${TILES}" --workers ${CPUS} \
        --zarr-root ${ZARR_ROOT} --report ${rep} --execute 2>&1 | tee ${LOGD}/splice.log
    $TM - <<EOF
import sys, pandas as pd
r = pd.read_csv("${rep}")
bad = r[(r.status != "ok") | (r.n_gap > 0)]
print(f"splice: {len(r)} tiles, {len(r) - len(bad)} ok")
if len(bad): print(bad.to_string(index=False)); sys.exit(1)
EOF
    require_all "${ZARR_ROOT}/sm_only/{d}/era5/values18/.zarray"

    $TM prepare_s48_cache.py --workers ${CPUS} --stations "${TILE_ARR[@]}" 2>&1 \
        | tee ${LOGD}/s48cache.log
    require_all "${S48_CACHE_ROOT}/sm_only/{d}/pyr.npz"
}

case ${MODE} in
    smoke) step_cm; step_filter; step_zarr; step_tok; step_post ;;
    cm)    step_cm ;;
    prep)  step_filter; step_zarr ;;
    tok)   step_tok ;;
    post)  step_post ;;
    *)     echo "unknown mode ${MODE}"; exit 2 ;;
esac
check_isolation
echo "mode ${MODE} task ${TI} DONE  $(date)"
