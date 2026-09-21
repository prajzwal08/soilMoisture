#!/bin/bash
#SBATCH --job-name=s1_orb
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python /gpfs/scratch1/nodespecific/int5/77080/claude-77080/-gpfs-work3-0-prjs1968-soilMoisture/4cd3363e-bd3f-432d-bcc0-929bfd5b5ef9/scratchpad/orb.py
