#!/bin/bash
#SBATCH --job-name=audit_inputs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/audit_inputs_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/audit_inputs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo ""

$PYTHON audit_inputs.py --workers 8

echo ""
echo "Finished : $(date)"
