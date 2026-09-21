#!/bin/bash
#SBATCH --job-name=eco_lst_smoke
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §37.6 -- measure the operating point before committing ~315k COG opens.
# 16 CPUs is the MINIMUM BILLABLE SLICE on rome (a node is shared by at most 8 jobs),
# so asking for 4 would save nothing and starve the TLS/decode work.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_lst_smoke job=$SLURM_JOB_ID host=$(hostname) $(date) ==="

# --- pre-flight: EDL auth fails SILENTLY (read_ok=0 everywhere, looks like total cloud)
[ -r "$HOME/.netrc" ] || { echo "FATAL: no ~/.netrc"; exit 1; }
grep -q urs.earthdata "$HOME/.netrc" || { echo "FATAL: ~/.netrc has no urs.earthdata"; exit 1; }
MODE=$(stat -c '%a' "$HOME/.netrc")
[ "$MODE" = "600" ] || { echo "FATAL: ~/.netrc mode $MODE, must be 600"; exit 1; }
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
       https://data.lpdaac.earthdatacloud.nasa.gov/ || echo 000)
[ "$CODE" = "000" ] && { echo "FATAL: LP DAAC unreachable from this node"; exit 1; }
echo "preflight ok (netrc 600, lpdaac http $CODE, TMPDIR=$TMPDIR)"

# --- syntax first: nothing runs on the login node, so this is where it gets checked
conda run -n soilmoisture --no-capture-output python -c \
  "import ast; ast.parse(open('read_ecostress_lst.py').read()); print('syntax ok')"

conda run -n soilmoisture --no-capture-output python read_ecostress_lst.py \
    --smoke --smoke-n "${1:-50}" --smoke-workers 16 32 64 --out-tag dtr
echo "=== done $(date) ==="
