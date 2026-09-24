#!/bin/bash
# Upload project to ruche cluster.
# Run from the project root on your local machine.
#
# Usage:
#   bash cluster/upload.sh

set -e

REMOTE="dalbanal@ruche.mesocentre.universite-paris-saclay.fr"
REMOTE_DIR="/gpfs/workdir/dalbanal/fpl-optimizer"

echo "Uploading FPL-Optimizer to ruche..."
echo "Remote: $REMOTE:$REMOTE_DIR"
echo ""

# Create remote directory
ssh $REMOTE "mkdir -p $REMOTE_DIR"

# Sync code, data, models, scripts, cluster configs
# Excludes: venv, __pycache__, .git, runs (will be created on cluster)
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude 'runs' \
    --exclude '.pytest_cache' \
    --exclude '*.egg-info' \
    --exclude 'notebooks' \
    src/ \
    $REMOTE:$REMOTE_DIR/src/

rsync -avz --progress \
    data/ \
    $REMOTE:$REMOTE_DIR/data/

rsync -avz --progress \
    models/ \
    $REMOTE:$REMOTE_DIR/models/

rsync -avz --progress \
    scripts/ \
    $REMOTE:$REMOTE_DIR/scripts/

rsync -avz --progress \
    cluster/ \
    $REMOTE:$REMOTE_DIR/cluster/

# Upload project config files
scp pyproject.toml $REMOTE:$REMOTE_DIR/
scp setup.cfg $REMOTE:$REMOTE_DIR/ 2>/dev/null || true

echo ""
echo "Upload complete!"
echo ""
echo "Next steps:"
echo "  ssh ruche"
echo "  cd $REMOTE_DIR"
echo "  bash cluster/setup.sh    # first time only"
echo "  sbatch cluster/retrain_predictor.slurm"
