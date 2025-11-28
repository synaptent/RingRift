#!/bin/bash
# Launch distributed training with torchrun
#
# Usage:
#   ./run_distributed_training.sh [num_gpus] [additional_args]
#
# Examples:
#   ./run_distributed_training.sh 2 --data path/to/data.npz --epochs 100
#   ./run_distributed_training.sh 4 --data data.npz --scale-lr --lr-scale-mode sqrt
#   ./run_distributed_training.sh 1 --data data.npz  # Single GPU (still uses DDP)

set -e

# Default number of GPUs (auto-detect if not specified)
NUM_GPUS="${1:-auto}"

if [ "$NUM_GPUS" = "auto" ]; then
    # Auto-detect number of GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
    echo "Auto-detected $NUM_GPUS GPU(s)"
else
    # Shift to remove first argument (num_gpus) from additional args
    shift
fi

# Validate NUM_GPUS is a number
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer, got: $NUM_GPUS"
    exit 1
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: Need at least 1 GPU, got: $NUM_GPUS"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to ai-service directory
cd "$AI_SERVICE_DIR"

echo "=============================================="
echo "RingRift Distributed Training Launcher"
echo "=============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Working directory: $AI_SERVICE_DIR"
echo "Additional arguments: $@"
echo "=============================================="

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun not found. Please install PyTorch >= 1.9"
    echo "Install with: pip install torch"
    exit 1
fi

# Run distributed training
# - nproc_per_node: number of processes per node (typically = number of GPUs)
# - master_addr: address of the master node (localhost for single machine)
# - master_port: port for distributed communication (use any free port)
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr=localhost \
    --master_port=29500 \
    app/training/train.py \
    --distributed \
    "$@"

echo ""
echo "=============================================="
echo "Distributed training completed!"
echo "=============================================="