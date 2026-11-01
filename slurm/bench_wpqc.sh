#!/bin/bash
#SBATCH --job-name=eco_bench
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture
# 16 cores / 28G is the SAME bill as 4 cores / 7G: a rome node is shared by at most 8
# jobs, so 128/8 = 16 CPUs is the minimum billable slice.  Asking for 4 bought nothing.
N="${1:-96}"; W="${2:-16}"
echo "=== eco_bench job=$SLURM_JOB_ID node=$(hostname) $(date)  n=$N workers=$W ==="
OFF=0
for C in A_percall B_shared C_seq E_sharednovza D_tuned; do
  # Separate PROCESS per config: env vars curl reads at connection setup cannot leak,
  # and no page cache carries over.  Disjoint --offset so no config warms another's.
  conda run -n soilmoisture --no-capture-output python bench_wpqc.py \
      --config "$C" --n "$N" --offset "$OFF" --workers "$W" 2>&1 | grep -v "GDAL signalled"
  OFF=$((OFF + N))
  echo
done
echo "=== done $(date) ==="
