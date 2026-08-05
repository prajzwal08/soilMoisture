#!/bin/bash
#SBATCH --job-name=station_mean_probe
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/station_mean_probe_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §20.14 station-mean probe. CPU only — never touches the L12 token path, so no GPU
# and modest memory. Runs in ~2 min interactively; this wrapper exists for
# reproducibility and for re-runs after the training job's best.pt advances.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n terramind --no-capture-output python station_mean_probe.py "$@"
