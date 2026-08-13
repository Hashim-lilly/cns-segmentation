#!/bin/bash
# =============================================================================
# CNS Spinal Cord Segmentation — Batch Inference
#
# Runs scripts/predict.py on a GPU node inside the container, producing
# CPU-viewable artifacts (prediction NIfTIs, overlay PNGs, Dice CSV/summary)
# under experiments/predictions/. A CPU-only Jupyter kernel (e.g. on
# magtrain) can load these afterwards without needing GPU access itself.
#
# Env vars (all optional):
#   CHECKPOINT  - path to model checkpoint, relative to project root
#                 (default: model.checkpoint from configs/inference.yaml)
#   SPLIT       - 'val' or 'train' (default: val)
#   OUTPUT_DIR  - override output.output_dir from the inference config
#   LIMIT       - only process the first N subjects (quick smoke test)
#
# Examples:
#   sbatch scripts/predict.slurm
#   CHECKPOINT=experiments/spine_segresnet_phase1_20260811_060000/checkpoints/best_model.pth sbatch scripts/predict.slurm
#   LIMIT=3 sbatch --partition=interactive --time=00:30:00 scripts/predict.slurm
# =============================================================================
#SBATCH --job-name=cns-predict
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/l091835/cns/logs/predict-%j.log
#SBATCH --error=/home/l091835/cns/logs/predict-%j.err

set -euo pipefail

BASE="/home/l091835/cns"
CONTAINER="$BASE/container/cns-pytorch.sif"

SPLIT="${SPLIT:-val}"
CLI_ARGS="--split $SPLIT"
[ -n "${CHECKPOINT:-}" ] && CLI_ARGS="$CLI_ARGS --checkpoint $CHECKPOINT"
[ -n "${OUTPUT_DIR:-}" ] && CLI_ARGS="$CLI_ARGS --output-dir $OUTPUT_DIR"
[ -n "${LIMIT:-}" ] && CLI_ARGS="$CLI_ARGS --limit $LIMIT"

mkdir -p "$BASE/logs"

echo "========================================"
echo "CNS Spinal Cord Segmentation — Inference"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Split      : $SPLIT"
echo "Command    : python scripts/predict.py $CLI_ARGS"
echo "Start time : $(date)"
echo "========================================"

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$BASE":"$BASE" \
  "$CONTAINER" \
  bash -c "
    set -euo pipefail
    cd '$BASE'
    export PYTHONPATH='$BASE:\${PYTHONPATH:-}'

    echo '--- GPU Info ---'
    nvidia-smi
    echo '--- Python / Device ---'
    python -c \"from src.models.segresnet import get_device; print(f'Resolved device: {get_device()}')\"
    echo '----------------'

    python scripts/predict.py $CLI_ARGS
  "

echo "========================================"
echo "Inference Complete"
echo "End time: $(date)"
echo "========================================"
