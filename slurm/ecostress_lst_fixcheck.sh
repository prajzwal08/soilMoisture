#!/bin/bash
#SBATCH --job-name=eco_lst_fix
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §37.8 VALIDATION -- does the per-thread cookie jar fix the 39.9% error rate?
#
# Array 26984229 lost 31,329 of 78,446 reads to
#   "not recognized as being in a supported file format"
# i.e. an EDL login page arriving where a TIFF should be.  Diagnosis in
# census_ecostress.thread_cookie_opts.  This job re-reads a sample of the FAILED keys
# only (--retry-failed makes read_ok=0 rows eligible again) at the same worker count, so
# the error rate is directly comparable to the 0.399 of the run it is fixing.
#
# PASS = err rate well under 0.399.  Then submit the full array.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture

WORKERS="${1:-64}"
LIMIT="${2:-600}"

echo "=== eco_lst_fix job=$SLURM_JOB_ID host=$(hostname) workers=$WORKERS limit=$LIMIT $(date) ==="

# Syntax first: a typo here would otherwise surface as a clean run with zero rows.
conda run -n soilmoisture --no-capture-output \
    python -c "import py_compile,sys; py_compile.compile('read_ecostress_lst.py', doraise=True); py_compile.compile('census_ecostress.py', doraise=True); print('syntax OK')"

[ -r "$HOME/.netrc" ] || { echo "FATAL: no ~/.netrc"; exit 1; }
grep -q urs.earthdata "$HOME/.netrc" || { echo "FATAL: ~/.netrc has no urs.earthdata"; exit 1; }

conda run -n soilmoisture --no-capture-output python read_ecostress_lst.py \
    --out-tag dtr --workers "$WORKERS" \
    --shard 0 --nshards 8 --retry-failed --limit "$LIMIT"
echo "=== done $(date) ==="
