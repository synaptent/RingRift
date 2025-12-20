#!/bin/bash
# Deploy EBMO training to cluster nodes

set -e

# Node definitions
declare -A NODES=(
    ["rtx5090"]="ssh1.vast.ai:15166"
    ["rtx5070"]="ssh2.vast.ai:10042"
    ["rtx4080s"]="ssh3.vast.ai:19940"
    ["a40"]="ssh8.vast.ai:38742"
)

REPO_PATH="/workspace/ringrift-ai"
LOCAL_PATH="$(dirname $(dirname $(realpath $0)))"

# Function to deploy to a node
deploy_to_node() {
    local name=$1
    local addr=$2
    local port=$3

    echo "=== Deploying to $name ($addr:$port) ==="

    # Create workspace
    ssh -o StrictHostKeyChecking=no -p $port root@$addr "mkdir -p $REPO_PATH" 2>/dev/null

    # Sync code (exclude large files)
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='*.pt' \
        --exclude='*.npz' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='venv' \
        --exclude='node_modules' \
        -e "ssh -o StrictHostKeyChecking=no -p $port" \
        "$LOCAL_PATH/" "root@$addr:$REPO_PATH/"

    # Install dependencies
    ssh -o StrictHostKeyChecking=no -p $port root@$addr "cd $REPO_PATH && pip install -q -e . 2>/dev/null || pip install -q torch numpy" 2>/dev/null

    echo "=== $name deployed ==="
}

# Function to start training on a node
start_training() {
    local name=$1
    local addr=$2
    local port=$3
    local script=$4

    echo "=== Starting training on $name ==="

    # Run training in background
    ssh -o StrictHostKeyChecking=no -p $port root@$addr "cd $REPO_PATH && nohup python $script > /tmp/training_$name.log 2>&1 &" 2>/dev/null

    echo "=== Training started on $name (check /tmp/training_$name.log) ==="
}

# Parse arguments
ACTION=${1:-"deploy"}

case $ACTION in
    "deploy")
        for name in "${!NODES[@]}"; do
            IFS=':' read -r addr port <<< "${NODES[$name]}"
            deploy_to_node "$name" "$addr" "$port"
        done
        ;;
    "train")
        SCRIPT=${2:-"scripts/train_ebmo_contrastive.py"}
        for name in "${!NODES[@]}"; do
            IFS=':' read -r addr port <<< "${NODES[$name]}"
            start_training "$name" "$addr" "$port" "$SCRIPT"
        done
        ;;
    "status")
        for name in "${!NODES[@]}"; do
            IFS=':' read -r addr port <<< "${NODES[$name]}"
            echo "=== $name ==="
            ssh -o StrictHostKeyChecking=no -p $port root@$addr "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader && ps aux | grep python | grep -v grep | head -3" 2>/dev/null || echo "Connection failed"
        done
        ;;
    "logs")
        NODE=${2:-"rtx5090"}
        IFS=':' read -r addr port <<< "${NODES[$NODE]}"
        ssh -o StrictHostKeyChecking=no -p $port root@$addr "tail -50 /tmp/training_$NODE.log" 2>/dev/null
        ;;
    *)
        echo "Usage: $0 {deploy|train|status|logs [node]}"
        exit 1
        ;;
esac
