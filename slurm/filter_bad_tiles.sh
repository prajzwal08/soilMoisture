#!/bin/bash
#SBATCH --job-name=bad_tile_filter
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bad_tile_filter_%j.out

source ~/.bashrc
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture

python filter_bad_tiles.py --workers 32

echo ""
echo "Files flagged for deletion: $(wc -l < csvs/bad_tile_manifest.txt)"
echo "Review csvs/bad_tile_manifest.txt before deleting."
