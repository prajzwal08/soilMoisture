#!/bin/bash
#SBATCH --job-name=eco_census
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36 TIER 1 -- ECOSTRESS day/night pair census.
#
#   sbatch slurm/ecostress_census.sh selftest      # ~3 granules, verify mechanics FIRST
#   sbatch slurm/ecostress_census.sh dryrun        # part A only, no pixels, no EDL
#   sbatch slurm/ecostress_census.sh controls      # >52 deg; MUST return zero granules
#   sbatch slurm/ecostress_census.sh sample 20     # stage 1a
#   sbatch slurm/ecostress_census.sh full          # stage 1b, all ~880 in-range stations
#
# Resume is automatic via csvs/ecostress_census_log.csv (keyed on station_id).
# Pass --fresh through to ignore it.
#
# --cpus-per-task=16, NOT 64: this job is network-bound and both CMR and LP DAAC throttle.
# HTTP concurrency is capped at 8 inside the script regardless of the allocation.

set -eo pipefail
exec 2>&1

MODE="${1:-selftest}"
shift || true

export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_census  mode=$MODE  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="

# -- pre-flight: the two things that make this job fail slowly instead of fast ----------
if [ "$MODE" != "dryrun" ]; then
    if ! grep -q "urs.earthdata.nasa.gov" "$HOME/.netrc" 2>/dev/null; then
        echo "FATAL: no urs.earthdata.nasa.gov entry in ~/.netrc -- see runbook §36.15"
        exit 1
    fi
    PERM=$(stat -c '%a' "$HOME/.netrc")
    if [ "$PERM" != "600" ]; then
        echo "FATAL: ~/.netrc is mode $PERM, must be 600 (curl/GDAL refuse otherwise)"
        exit 1
    fi
fi

echo "--- connectivity ---"
for HOST in cmr.earthdata.nasa.gov data.lpdaac.earthdatacloud.nasa.gov; do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "https://$HOST/" || echo "000")
    echo "  $HOST -> HTTP $CODE"
    if [ "$CODE" = "000" ]; then
        echo "FATAL: no route to $HOST from this compute node."
        exit 1
    fi
done
echo

case "$MODE" in
    selftest)
        ARGS="--selftest" ;;
    probe)
        UR="${1:-ECOv002_L2T_LSTE_00375_005_14RNU_20180730T222757_0712_01}"; shift || true
        ARGS="--probe $UR" ;;
    controls)
        ARGS="--controls" ;;
    full)
        ARGS="" ;;
    dryrun)
        N="${1:-20}"; shift || true
        ARGS="--dry-run --sample $N" ;;
    sample)
        N="${1:-20}"; shift || true
        ARGS="--sample $N" ;;
    *)
        echo "unknown mode: use selftest|probe|dryrun|controls|sample|full"
        exit 2 ;;
esac

echo "--- census_ecostress.py $ARGS $* ---"
conda run -n soilmoisture --no-capture-output python census_ecostress.py $ARGS "$@"

echo "=== done $(date) ==="
