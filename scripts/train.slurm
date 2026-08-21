#!/bin/bash
# =============================================================================
# CNS Spinal Cord Segmentation — Training Submission
#
# Env vars (all optional):
#   CONFIG      - path to YAML config, relative to project root (default: configs/train_spine.yaml)
#   EPOCHS      - override training.epochs
#   LR          - override training.lr
#   BATCH_SIZE  - override training.batch_size
#   SEED        - override random seed
#   EXPERIMENT_NAME - override output.experiment_name
#
# SLURM overrides via sbatch flags: --partition, --mem, --gpus-per-node, --time
#
# Examples:
#   # Default Phase 1 baseline run
#   sbatch scripts/train.slurm
#
#   # Override epochs / lr
#   EPOCHS=150 LR=3e-4 sbatch scripts/train.slurm
#
#   # Different partition (e.g. shorter queue for a quick run)
#   sbatch --partition=interactive --time=12:00:00 scripts/train.slurm
# =============================================================================
#SBATCH --job-name=cns-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/train-%j.log
#SBATCH --error=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/train-%j.err

set -euo pipefail

BASE="/home/l091835/CNS-Digital-Twin/cns-segmentation"
CONTAINER="$BASE/container/cns-pytorch.sif"

CONFIG="${CONFIG:-configs/train_spine.yaml}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SEED="${SEED:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

mkdir -p "$BASE/logs"

# ── Build CLI args ──────────────────────────────────────────────────────────
CLI_ARGS="--config $CONFIG"
[ -n "$EPOCHS" ] && CLI_ARGS="$CLI_ARGS --epochs $EPOCHS"
[ -n "$LR" ] && CLI_ARGS="$CLI_ARGS --lr $LR"
[ -n "$BATCH_SIZE" ] && CLI_ARGS="$CLI_ARGS --batch-size $BATCH_SIZE"
[ -n "$SEED" ] && CLI_ARGS="$CLI_ARGS --seed $SEED"
[ -n "$EXPERIMENT_NAME" ] && CLI_ARGS="$CLI_ARGS --name $EXPERIMENT_NAME"

# ── Job info ─────────────────────────────────────────────────────────────────
echo "========================================"
echo "CNS Spinal Cord Segmentation — Training"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Config     : $CONFIG"
echo "Command    : python scripts/train.py $CLI_ARGS"
echo "Start time : $(date)"
echo "========================================"

# ── Run inside container ────────────────────────────────────────────────────
apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind "$BASE":"$BASE" \
  "$CONTAINER" \
  bash -c "
    set -euo pipefail
    cd '$BASE'
    export PYTHONPATH='$BASE/src:\${PYTHONPATH:-}'

    echo '--- GPU Info ---'
    nvidia-smi
    echo '--- Python / Device ---'
    python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')\"
    python -c \"from cns_segmentation.models.segresnet import get_device; print(f'Resolved device: {get_device()}')\"
    echo '----------------'

    python scripts/train.py $CLI_ARGS
  "

echo "========================================"
echo "Training Complete"
echo "End time: $(date)"
echo "========================================"
