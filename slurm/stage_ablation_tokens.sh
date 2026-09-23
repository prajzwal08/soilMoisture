#!/bin/bash
#SBATCH --job-name=stage_ablation_tokens
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/stage_ablation_tokens_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/stage_ablation_tokens_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §24.13 step 0 — stage the ablation_oos sm_only token stores from /projects to scratch, so the
# frozen U-Net arm runs UNMODIFIED at the path dataset_unet.py:40 already names.
#
# ~36 stations / ~50 GB, not the full 1.4 TB: category_filter=["sm_only"] (train_unet.py:191)
# admits only 36 of the 50 ablation_oos stations.
#
# Pass --execute to actually copy; with no argument this is a dry run that reports sizes.
#     sbatch slurm/stage_ablation_tokens.sh            # dry run
#     sbatch slurm/stage_ablation_tokens.sh --execute  # copy

set -eo pipefail
exec 2>&1

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "CPUs    : $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo "Args    : $*"
echo ""

# Syntax gate before the parallel section — a SyntaxError inside Pool workers is reported
# badly and would burn the whole allocation.
$PYTHON -c "import ast; ast.parse(open('stage_ablation_tokens.py').read()); print('syntax OK')"
echo ""

$PYTHON stage_ablation_tokens.py --workers "$SLURM_CPUS_PER_TASK" "$@"

echo ""
echo "=== done $(date) ==="
