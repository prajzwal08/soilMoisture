#!/bin/bash
#SBATCH --job-name=retok_recovery
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/retokenize_recovery_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Recovery for the 2 previously-excluded stations (Phillipsburg, CaveMountain).
# Both have complete satellite_zarr but no token zarr — run both phases.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

STATIONS="ISMN_SCAN_Phillipsburg ISMN_SNOTEL_CaveMountain"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

echo "=== Phase 1: TerraMind tokens (terramind env) ==="
for s in $STATIONS; do
    conda run --no-capture-output -n terramind python retokenize_satellite_zarr.py \
        --mode       terramind \
        --station    "$s" \
        --batch-size 8 \
        --device     cuda \
        --execute
done

echo "=== Phase 2: CloudSEN12 + tabular fill (sensei env) ==="
for s in $STATIONS; do
    conda run --no-capture-output -n sensei python retokenize_satellite_zarr.py \
        --mode       cm-fill \
        --station    "$s" \
        --batch-size 16 \
        --device     cuda \
        --execute
done

echo "Done. Total .complete:"
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
