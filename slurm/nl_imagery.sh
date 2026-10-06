#!/bin/bash
#SBATCH --job-name=nl_img
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=36:00:00
#SBATCH --output=logs/nl_img_%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68 imagery downloads for the Overijssel grid (S2 L2A + DEM, S1 RTC + LULC), Planetary Computer.
#   smoke : sbatch --time=06:00:00 slurm/nl_imagery.sh smoke
#           builds the tile CSV, downloads S2+DEM and S1+LULC for 2 TWENTE control tiles + 1 grid tile,
#           then verify_nl_download.py (exit 1 on any failed gate -> afterok dependants never start).
#   full  : sbatch --array=0-7  --dependency=afterok:<smoke> slurm/nl_imagery.sh s2  8
#           sbatch --array=0-11 --dependency=afterok:<smoke> slurm/nl_imagery.sh s1 12
# Resubmitting the same command resumes (per-task checkpoints; existing TIFs are skipped).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
NL=/gpfs/scratch1/shared/pkhanal/nl
export SOIL_SCRATCH_DIR=${NL}/satellite SOIL_S2_HARMONISE=1
PY="conda run -n soilmoisture --no-capture-output python"
MODE=${1:?smoke|s2|s1}
echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-none} mode ${MODE} on $(hostname)  $(date)"

if [[ ${MODE} == smoke ]]; then
    $PY make_nl_splits.py --out-root ${NL}/data
    mkdir -p ${NL}/data_smoke
    $PY - <<EOF
import pandas as pd
df = pd.read_csv("${NL}/data/station_splits.csv")
smoke = pd.concat([df[df.is_control].head(2), df[~df.is_control].head(1)])
smoke.to_csv("${NL}/data_smoke/station_splits.csv", index=False)
print(smoke[["dir_name", "latitude", "longitude", "control_of"]].to_string(index=False))
EOF
    export SOIL_DATA_ROOT=${NL}/data_smoke
    $PY download_s2_mpc.py > logs/nl_img_smoke_s2_${SLURM_JOB_ID}.log 2>&1 &
    P2=$!
    $PY download_s1_lulc_mpc.py > logs/nl_img_smoke_s1_${SLURM_JOB_ID}.log 2>&1 &
    P1=$!
    RC2=0; wait $P2 || RC2=$?; echo "S2 exit $RC2"; tail -5 logs/nl_img_smoke_s2_${SLURM_JOB_ID}.log
    RC1=0; wait $P1 || RC1=$?; echo "S1 exit $RC1"; tail -5 logs/nl_img_smoke_s1_${SLURM_JOB_ID}.log
    conda run -n terramind --no-capture-output python verify_nl_download.py \
        --csv ${NL}/data_smoke/station_splits.csv --scratch ${NL}/satellite --modalities s2 s1
    [[ $RC2 -eq 0 && $RC1 -eq 0 ]] || { echo "download exit codes S2=$RC2 S1=$RC1 -> FAIL"; exit 1; }
else
    export SOIL_DATA_ROOT=${NL}/data SOIL_TASK_INDEX=${SLURM_ARRAY_TASK_ID} SOIL_N_TASKS=${2:?n_tasks}
    if [[ ${MODE} == s2 ]]; then
        $PY download_s2_mpc.py
    else
        $PY download_s1_lulc_mpc.py
    fi
fi
date
