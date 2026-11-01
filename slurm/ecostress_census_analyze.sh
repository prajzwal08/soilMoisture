#!/bin/bash
#SBATCH --job-name=eco_census_viz
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36.22 -- draw the ECOSTRESS census filter chain.
#
#   sbatch slurm/ecostress_census_analyze.sh
#   sbatch slurm/ecostress_census_analyze.sh BodieHills,Rothamsted
#
# Reads csvs/ecostress_census_{granules,pairs,log}.csv and writes PNGs plus
# viz_data.json into fig/ecostress_filter_viz/.
#
# NO downloads happen here.  §36.17: "never combine download and analysis in one job."
# That is also why this needs no EDL credentials and no connectivity pre-flight --
# it touches nothing but local CSVs.
#
# It is a batch job rather than a login-node one-liner because nothing in this project
# runs on the login node, not even a seconds-long plot.

set -eo pipefail
exec 2>&1

STATIONS="${1:-BodieHills,Rothamsted,PSA2Tiergarten,Banizoumbou}"
shift || true

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_census_viz  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="
echo "stations: $STATIONS"
echo

# -- pre-flight: fail fast and legibly if the census outputs are not where we expect ---
for F in csvs/ecostress_census_granules.csv \
         csvs/ecostress_census_pairs.csv \
         csvs/ecostress_census_log.csv \
         census_ecostress.py; do
    if [ ! -s "$F" ]; then
        echo "FATAL: missing or empty -- $F"
        exit 1
    fi
    printf '  %-46s %8s  %s\n' "$F" "$(du -h "$F" | cut -f1)" \
           "$(date -r "$F" '+%Y-%m-%d %H:%M')"
done
echo

echo "--- plot_ecostress_census.py ---"
conda run -n terramind --no-capture-output \
    python plot_ecostress_census.py \
        --stations "$STATIONS" \
        --outdir fig/ecostress_filter_viz \
        --emit-json "$@"

echo
echo "--- output ---"
ls -la fig/ecostress_filter_viz/
echo "=== done $(date) ==="
