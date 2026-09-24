#!/bin/bash
#SBATCH --job-name=fpl-ablat
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

set -e
cd /gpfs/workdir/dalbanal/fpl-optimizer

echo "============================================"
echo "Predictor Ablation: xP variants"
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "============================================"

/gpfs/workdir/dalbanal/envs/fpl-optimizer/bin/python scripts/predictor_ablation.py

echo "Finished: $(date)"
