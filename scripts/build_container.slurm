#!/bin/bash
#SBATCH --job-name=cns-pytorch-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/l091835/cns/logs/cns-pytorch-build-%j.log
#SBATCH --error=/home/l091835/cns/logs/cns-pytorch-build-%j.err

set -euo pipefail
set -x

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"

BASE="/home/l091835/cns"
mkdir -p "$BASE/logs" "$BASE/container"

# Set up temporary directories
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"
export APPTAINER_CACHEDIR="$SLURM_TMPDIR/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Build location
cd "$SLURM_TMPDIR"

# Copy definition file
DEF_SOURCE="$BASE/scripts/cns-pytorch.def"
cp "$DEF_SOURCE" ./cns-pytorch.def

# Compute nodes can't reach pypi.org directly — stage a transient copy of
# the user's Artifactory pip mirror config next to the .def file so the
# %files section can copy it to /etc/pip.conf for the build. The .def's
# %post deletes /etc/pip.conf again before the image is finalized, so the
# credential never ends up baked into the .sif; this scratch copy also
# lives only in $SLURM_TMPDIR and is removed at the end of this script.
PIP_CONF="$HOME/.pip/pip.conf"
if [ -f "$PIP_CONF" ]; then
    cp "$PIP_CONF" ./pip.conf
else
    echo "WARNING: $PIP_CONF not found — pip installs may fail without a reachable index."
    touch ./pip.conf
fi

# Build the container
echo "=== Starting CNS Segmentation Container Build (Python 3.12) ==="
apptainer build --fakeroot cns-pytorch.sif cns-pytorch.def
rm -f ./pip.conf

# Verify build
if [ -f cns-pytorch.sif ]; then
    echo "=== Build successful! Verifying container ==="
    apptainer exec cns-pytorch.sif python3 --version
    apptainer exec cns-pytorch.sif python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    apptainer exec cns-pytorch.sif python3 -c "import monai, nibabel, trimesh, mlflow; print('All imports successful')"
fi

# Copy to final location
FINAL_DIR="$BASE/container"
mkdir -p "$FINAL_DIR"

# Backup existing container if present
if [ -f "$FINAL_DIR/cns-pytorch.sif" ]; then
    mv "$FINAL_DIR/cns-pytorch.sif" "$FINAL_DIR/cns-pytorch.sif.bak"
    echo "Backed up existing container to cns-pytorch.sif.bak"
fi

mv cns-pytorch.sif "$FINAL_DIR/"

echo "=== Build completed successfully ==="
echo "Container location: $FINAL_DIR/cns-pytorch.sif"
echo "End Time: $(date)"
