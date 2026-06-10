#!/bin/bash
#SBATCH --job-name=cm_fill_4
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/cm_fill_4_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# CRITICAL recovery: 4 stations missing cloud masks (MISSING_cm).
# process_cm_fill --station also runs _fill_tabular, picking up the
# soil (Perthshire) / era5 (PerdidoRivFarms) data downloaded earlier.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

for STATION in ISMN_SCAN_MorrisFarms ISMN_SCAN_PerdidoRivFarms ISMN_SCAN_Perthshire ISMN_SNOTEL_AnchorRiverDivide; do
    echo "=== $STATION ==="
    conda run --no-capture-output -n sensei python retokenize_satellite_zarr.py \
        --mode    cm-fill \
        --station "$STATION" \
        --batch-size 16 \
        --device  cuda \
        --execute
done

echo "Done."
