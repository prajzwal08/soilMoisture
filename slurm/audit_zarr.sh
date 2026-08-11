#!/bin/bash
#SBATCH --job-name=audit_zarr
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/audit_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python audit_zarr.py
