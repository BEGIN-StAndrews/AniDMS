#!/bin/bash
#SBATCH --job-name=DMS08Q1
#SBATCH --partition=small-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/DMS_08Q1_%j.log
#SBATCH --error=logs/DMS_08Q1_%j.err

# Print job info, here we produce the data for 2008 Q1
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda environment, here ml340 is my user name, and geospatial is the name of the conda environment I created for this project. Please change these accordingly.
source /software/conda/ml340/conda/etc/profile.d/conda.sh
conda activate geospatial

# Change to working directory
cd /sharedscratch/ml340/DMSEstimation

# Run Python script
python running_cluster_08Q1.py

# Print completion
echo "Job finished at: $(date)"