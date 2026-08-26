#!/bin/bash
# =============================================================================
# CNS Merged Evaluation Report
#
# Runs scripts/evaluate.py on a GPU node inside the container, producing
# experiments/evaluate_report/{merged_metrics.csv,provenance.yaml} across all
# four trained structures (cord/canal/csf/rootlets) x held-out sites x
# external validation datasets. Reuse-first by default (loads existing
# metrics_summary.yaml artifacts); set FORCE_RERUN=1 to re-run inference.
#
# Env vars (all optional):
#   STRUCTURES    - comma-separated subset, e.g. 'cord,canal' (default: all)
#   FORCE_RERUN   - set to 1 to re-run inference instead of reusing artifacts
#   OUTPUT_DIR    - override output dir (default: experiments/evaluate_report)
#   LIMIT         - only process the first N subjects per run (force-rerun only)
#
# Examples:
#   sbatch scripts/evaluate.slurm
#   FORCE_RERUN=1 sbatch scripts/evaluate.slurm
#   STRUCTURES=cord,canal FORCE_RERUN=1 LIMIT=3 sbatch --partition=interactive --time=00:30:00 scripts/evaluate.slurm
# =============================================================================
#SBATCH --job-name=cns-evaluate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/evaluate-%j.log
#SBATCH --error=/home/l091835/CNS-Digital-Twin/cns-segmentation/logs/evaluate-%j.err

set -euo pipefail

BASE="/home/l091835/CNS-Digital-Twin/cns-segmentation"
CONTAINER="$BASE/container/cns-pytorch.sif"

CLI_ARGS=""
[ -n "${STRUCTURES:-}" ] && CLI_ARGS="$CLI_ARGS --structures $STRUCTURES"
[ "${FORCE_RERUN:-0}" = "1" ] && CLI_ARGS="$CLI_ARGS --force-rerun"
[ -n "${OUTPUT_DIR:-}" ] && CLI_ARGS="$CLI_ARGS --output-dir $OUTPUT_DIR"
[ -n "${LIMIT:-}" ] && CLI_ARGS="$CLI_ARGS --limit $LIMIT"

mkdir -p "$BASE/logs"

echo "========================================"
echo "CNS Merged Evaluation Report"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Command    : python scripts/evaluate.py $CLI_ARGS"
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

    python scripts/evaluate.py $CLI_ARGS
  "

echo "========================================"
echo "Evaluation Complete"
echo "End time: $(date)"
echo "========================================"
