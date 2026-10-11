#!/bin/bash
#SBATCH --job-name=compile_wrapup
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/compile_wrapup_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/compile_wrapup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Build text/paper_wrapup.pdf (pdflatex, bibtex, pdflatex x2; no latexmk on Snellius).
cd /gpfs/work3/0/prjs1968/soilMoisture/text
pdflatex -interaction=nonstopmode paper_wrapup.tex > /dev/null
bibtex paper_wrapup
pdflatex -interaction=nonstopmode paper_wrapup.tex > /dev/null
pdflatex -interaction=nonstopmode paper_wrapup.tex
echo "--- warnings ---"
grep -iE "undefined|warning" paper_wrapup.log | grep -v "Font shape" || echo "none"
