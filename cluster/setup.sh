#!/bin/bash
# Setup script for ruche cluster — run once after uploading the project.
#
# Usage:
#   ssh ruche
#   cd /gpfs/workdir/dalbanal/fpl-optimizer
#   bash cluster/setup.sh

set -e

echo "Setting up FPL-Optimizer environment on ruche..."

# Load Anaconda (has Python 3.11)
module load anaconda3/2023.09-0/none-none
echo "Python: $(python3 --version)"

# Create conda env with Python 3.11
if [ ! -d "$WORKDIR/envs/fpl-optimizer" ]; then
    conda create -y -p $WORKDIR/envs/fpl-optimizer python=3.11
    echo "Created conda environment"
fi

# Activate
source activate $WORKDIR/envs/fpl-optimizer

# Install the project and dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Verify key imports
python -c "
from fpl_optimizer.prediction.model import PointPredictor
from fpl_optimizer.optimizer.transfer_optimizer import optimize_transfers
print('All imports OK')
"

# Create logs directory
mkdir -p logs

echo ""
echo "Setup complete! Submit with:"
echo "  sbatch cluster/retrain_predictor.slurm"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/fpl_optimizer_*.out"
