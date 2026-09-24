#!/bin/bash
#SBATCH --job-name=fpl-retrain
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/retrain_%j.out
#SBATCH --error=logs/retrain_%j.err

set -e

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/fpl-optimizer
PYTHON=$WORKDIR/envs/fpl-optimizer/bin/python
cd $PROJECT

echo "============================================"
echo "FPL Predictor Retrain (xP shift fix)"
echo "  - fpl_xp now shifted by 1 GW"
echo "  - Retraining LightGBM from scratch"
echo "  - Comparing old vs new model"
echo "Job ID:    $SLURM_JOB_ID"
echo "Start:     $(date)"
echo "============================================"

$PYTHON scripts/retrain_and_test.py

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
