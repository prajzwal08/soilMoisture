#!/bin/bash
#SBATCH --job-name=effres_explainer
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/effres_explainer_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §70 plain-language explainer: teaching figures + LaTeX -> text/effres_explainer.pdf
#   sbatch slurm/effres_explainer.sh
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
echo "=== effres_explainer  job ${SLURM_JOB_ID}  $(date)  git $(git rev-parse --short HEAD) ==="
conda run -n terramind --no-capture-output python plot_effres_explainer.py
cd text
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error effres_explainer.tex > /dev/null; done
rm -f effres_explainer.{aux,log,out}
ls -la effres_explainer.pdf
echo "=== done $(date) ==="
