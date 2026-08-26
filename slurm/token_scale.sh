#!/bin/bash
#SBATCH --job-name=token_scale
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/token_scale_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §35.25 — measure the scale of the frozen TerraMind L12 tokens.
#
# §35.24 added an input LayerNorm on those tokens (model.py s2_norm / s1_norm /
# dem_norm / lulc_norm) on an UNMEASURED argument: that a positional code at
# std 0.02 would be invisible against a raw token. Nobody has checked how big the
# raw tokens actually are. This job produces the one number that settles it —
# `per_elem_std`, the spread across the 768 features inside a token, which is
# exactly what LayerNorm divides by.
#
#   tag share ~= 2%   -> the input LayerNorms are a no-op; turn them off
#   tag share ~= 0%   -> staleness is invisible without them; keep them
#
# It also splits within-tile variance into the part carried by token MAGNITUDE
# (which LayerNorm deletes) and the part carried by DIRECTION (which it keeps).
# A large magnitude share means the input LayerNorm is discarding exactly the
# within-tile signal the patchwise design (§34) exists to find — in which case it
# should not be on by default whatever the tag share says.
#
# CPU only, reads the token zarr on scratch. ~10 min for 120 stations.

set -euo pipefail
mkdir -p logs csvs

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "started   : $(date -Is)"
echo

# terramind: needs zarr 2.x + numpy + pandas, and imports dataset.py / model.py for
# the real ZARR_ROOT, STATION_TOKEN and EMB_INIT_STD constants (so this measures what
# the model actually consumes). The soilmoisture env has neither torch nor zarr.
conda run -n terramind --no-capture-output python measure_token_scale.py

echo
echo "finished  : $(date -Is)"
echo "result    : csvs/token_scale.json"
