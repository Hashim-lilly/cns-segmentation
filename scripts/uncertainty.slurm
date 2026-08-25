#!/bin/bash
# =============================================================================
# CNS Spinal Cord Segmentation — MC-Dropout Uncertainty & Calibration
#
# Runs scripts/uncertainty.py on a GPU node inside the container, producing
# entropy/mutual-information/variance NIfTI maps, a per-subject Dice CSV
# with uncertainty summary columns, and a calibration_summary.yaml (ECE +
# reliability diagrams) under experiments/uncertainty_<structure>/.
#
# N=8 stochastic forward passes per subject means roughly 8x the sliding-
# window compute of a single-pass scripts/predict.py run — the default
# --time below is bumped accordingly from predict.slurm's 2h.
#
# Env vars (all optional except CONFIG/TRAIN_CONFIG/CHECKPOINT, which follow
# scripts/uncertainty.py's own required-vs-optional rules):
#   CONFIG        - path to inference YAML config (default: configs/inference.yaml)
#   TRAIN_CONFIG  - path to training YAML config (default: configs/train_spine.yaml)
#   CHECKPOINT    - path to model checkpoint, relative to project root
#                   (default: model.checkpoint from CONFIG)
#   SPLIT         - 'val' or 'train' (default: val)
#   OUTPUT_DIR    - override output.output_dir from CONFIG
#   LIMIT         - only process the first N subjects (quick smoke test)
#   ENABLED       - '1'/'true' to pass --enabled, '0'/'false' to pass
#                   --no-enabled; unset leaves the config's own
#                   inference.uncertainty.enabled value in effect
#
# Examples:
#   CONFIG=configs/inference_canal.yaml TRAIN_CONFIG=configs/train_spine_canal.yaml \
#     CHECKPOINT=experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth \
#     ENABLED=true sbatch scripts/uncertainty.slurm
#
#   LIMIT=1 ENABLED=true sbatch --partition=interactive --time=00:30:00 scripts/uncertainty.slurm
# =============================================================================
#SBATCH --job-name=cns-uncertainty
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/uncertainty-%j.log
#SBATCH --error=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/uncertainty-%j.err

set -euo pipefail

BASE="/home/l091835/CNS-Digital-Twin/cns-segmentation"
CONTAINER="$BASE/container/cns-pytorch.sif"

CONFIG="${CONFIG:-configs/inference.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_spine.yaml}"
SPLIT="${SPLIT:-val}"
CLI_ARGS="--config $CONFIG --train-config $TRAIN_CONFIG --split $SPLIT"
[ -n "${CHECKPOINT:-}" ] && CLI_ARGS="$CLI_ARGS --checkpoint $CHECKPOINT"
[ -n "${OUTPUT_DIR:-}" ] && CLI_ARGS="$CLI_ARGS --output-dir $OUTPUT_DIR"
[ -n "${LIMIT:-}" ] && CLI_ARGS="$CLI_ARGS --limit $LIMIT"
if [ -n "${ENABLED:-}" ]; then
  case "${ENABLED,,}" in
    1|true|yes) CLI_ARGS="$CLI_ARGS --enabled" ;;
    0|false|no) CLI_ARGS="$CLI_ARGS --no-enabled" ;;
    *) echo "Unrecognized ENABLED value: $ENABLED" >&2; exit 1 ;;
  esac
fi

mkdir -p "$BASE/logs"

echo "========================================"
echo "CNS Spinal Cord Segmentation — Uncertainty"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Config     : $CONFIG"
echo "Command    : python scripts/uncertainty.py $CLI_ARGS"
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
    export PYTHONPATH='$BASE/src:\${PYTHONPATH:-}'

    echo '--- GPU Info ---'
    nvidia-smi
    echo '--- Python / Device ---'
    python -c \"from cns_segmentation.models.segresnet import get_device; print(f'Resolved device: {get_device()}')\"
    echo '----------------'

    python scripts/uncertainty.py $CLI_ARGS
  "

echo "========================================"
echo "Uncertainty Analysis Complete"
echo "End time: $(date)"
echo "========================================"
