#!/bin/bash
# =============================================================================
# CNS Spinal Cord Segmentation — External Cross-Dataset Validation
#
# Runs scripts/evaluate_external.py on a GPU node inside the container,
# scoring an already-trained checkpoint's single-structure channel against
# an independent role="validation" registry dataset (never seen during
# training) — producing real Dice/HD95/surface-Dice numbers under
# experiments/external_eval_<dataset>/.
#
# Env vars (required):
#   DATASET          - registry key, e.g. spider_canal
#   STRUCTURE        - structure name to score, e.g. canal
#   TRAIN_CONFIG     - training config the checkpoint came from
#   INFERENCE_CONFIG - inference config matching the checkpoint's architecture
#   CHECKPOINT       - path to the trained checkpoint, relative to project root
#
# Env vars (optional):
#   OUTPUT_DIR  - override default experiments/external_eval_<dataset>/
#   LIMIT       - only process the first N subjects (quick smoke test)
#
# Examples:
#   DATASET=spider_canal STRUCTURE=canal \
#     TRAIN_CONFIG=configs/train_spine_canal.yaml \
#     INFERENCE_CONFIG=configs/inference_canal.yaml \
#     CHECKPOINT=experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth \
#     sbatch --partition=MiG20 scripts/evaluate_external.slurm
#
#   DATASET=openneuro_ds004507 STRUCTURE=rootlets \
#     TRAIN_CONFIG=configs/train_spine_rootlets.yaml \
#     INFERENCE_CONFIG=configs/inference_rootlets.yaml \
#     CHECKPOINT=experiments/spine_segresnet_rootlets_20260819_053526/checkpoints/best_model.pth \
#     sbatch --partition=MiG20 scripts/evaluate_external.slurm
# =============================================================================
#SBATCH --job-name=cns-eval-external
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/eval-external-%j.log
#SBATCH --error=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/eval-external-%j.err

set -euo pipefail

BASE="/home/l091835/CNS-Digital-Twin/cns-segmentation"
CONTAINER="$BASE/container/cns-pytorch.sif"

: "${DATASET:?Must set DATASET, e.g. spider_canal}"
: "${STRUCTURE:?Must set STRUCTURE, e.g. canal}"
: "${TRAIN_CONFIG:?Must set TRAIN_CONFIG, e.g. configs/train_spine_canal.yaml}"
: "${INFERENCE_CONFIG:?Must set INFERENCE_CONFIG, e.g. configs/inference_canal.yaml}"
: "${CHECKPOINT:?Must set CHECKPOINT, e.g. experiments/.../checkpoints/best_model.pth}"

CLI_ARGS="--dataset $DATASET --structure $STRUCTURE --train-config $TRAIN_CONFIG --inference-config $INFERENCE_CONFIG --checkpoint $CHECKPOINT"
[ -n "${OUTPUT_DIR:-}" ] && CLI_ARGS="$CLI_ARGS --output-dir $OUTPUT_DIR"
[ -n "${LIMIT:-}" ] && CLI_ARGS="$CLI_ARGS --limit $LIMIT"

mkdir -p "$BASE/logs"

echo "========================================"
echo "CNS External Cross-Dataset Validation"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Dataset    : $DATASET"
echo "Structure  : $STRUCTURE"
echo "Command    : python scripts/evaluate_external.py $CLI_ARGS"
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

    python scripts/evaluate_external.py $CLI_ARGS
  "

echo "========================================"
echo "External Validation Complete"
echo "End time: $(date)"
echo "========================================"
