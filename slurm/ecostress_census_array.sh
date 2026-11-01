#!/bin/bash
#SBATCH --job-name=eco_census_arr
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-43%12
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36 TIER 1 -- the full-network ECOSTRESS census, as a SLURM array.
#
#   sbatch slurm/ecostress_census_array.sh              # all in-range stations
#   sbatch slurm/ecostress_census_array.sh --all-phases # also read OUT-of-window granules
#   bash   slurm/ecostress_census_array.sh merge        # merge per-task CSVs afterwards
#
# WHY AN ARRAY.  Measured 2026-09-15 from the probe: ~3 s per layer read, ~210 in-window
# overpasses per station, 4 layers -> ~5 min/station at 8 concurrent HTTP.  Across ~880
# stations that is ~77 h, an order of magnitude past any single-job wall.  With 44 tasks
# of 20 stations each: ~100 min per task, comfortably inside 6 h with margin for retries.
#
# %12 throttles to 12 concurrent tasks x 8 HTTP workers = 96 concurrent requests to LP
# DAAC.  Raise only after confirming LP DAAC does not throttle -- the failure mode is
# intermittent 403/503 that with_retry absorbs into a much longer wall-clock, not a crash.
#
# Each task writes its OWN csvs/*.NNNNN_NNNNN.csv.  Appending from 44 processes to one
# file would interleave rows -- csv.DictWriter has no cross-process atomicity.

set -eo pipefail
exec 2>&1

cd /gpfs/work3/0/prjs1968/soilMoisture

# -------------------------------------------------------------------------- merge mode
if [ "${1:-}" = "merge" ]; then
    echo "=== merging per-task census CSVs ==="
    for BASE in ecostress_census_granules ecostress_census_pairs ecostress_census_log; do
        OUT="csvs/${BASE}.csv"
        PARTS=$(ls -1 csvs/${BASE}.[0-9]*_[0-9]*.csv 2>/dev/null | sort)
        N=$(echo "$PARTS" | grep -c . || true)
        if [ "$N" -eq 0 ]; then echo "  $BASE: no parts found, skipping"; continue; fi
        FIRST=$(echo "$PARTS" | head -1)
        head -1 "$FIRST" > "$OUT"
        for P in $PARTS; do tail -n +2 "$P" >> "$OUT"; done
        echo "  $BASE: $N parts -> $(( $(wc -l < "$OUT") - 1 )) rows"
    done
    echo "=== done. Per-task parts left in place; delete once the merge is verified. ==="
    exit 0
fi

# -------------------------------------------------------------------------- census mode
N_STATIONS=918          # in-range count at |lat| <= 54 (LAT_LIMIT); 993 - 75 excluded
N_TASKS=44
PER_TASK=$(( (N_STATIONS + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))

export PYTHONUNBUFFERED=1
ulimit -n 65536

echo "=== eco_census_arr task ${SLURM_ARRAY_TASK_ID}  stations [${START}, ${END})  \
job=${SLURM_ARRAY_JOB_ID}  node=$(hostname)  $(date) ==="

if ! grep -q "urs.earthdata.nasa.gov" "$HOME/.netrc" 2>/dev/null; then
    echo "FATAL: no urs.earthdata.nasa.gov entry in ~/.netrc -- see runbook §36.15"
    exit 1
fi
PERM=$(stat -c '%a' "$HOME/.netrc")
if [ "$PERM" != "600" ]; then
    echo "FATAL: ~/.netrc is mode $PERM, must be 600"
    exit 1
fi

for HOST in cmr.earthdata.nasa.gov data.lpdaac.earthdatacloud.nasa.gov; do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "https://$HOST/" || echo "000")
    echo "  $HOST -> HTTP $CODE"
    [ "$CODE" = "000" ] && { echo "FATAL: no route to $HOST"; exit 1; }
done

conda run -n soilmoisture --no-capture-output python census_ecostress.py \
    --start-idx "$START" --end-idx "$END" "$@"

echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
