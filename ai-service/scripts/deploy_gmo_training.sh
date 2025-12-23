#!/bin/bash
# Deploy GMO variant training to cluster nodes
# Usage: ./scripts/deploy_gmo_training.sh [sync|train|status|logs]

set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LOCAL_PATH="$(dirname "$SCRIPT_DIR")"

# Node definitions: name -> "user@host:port:ssh_key:repo_path"
declare -A NODES=(
    # GH200 nodes (CUDA) - largest memory
    ["gh200_e"]="ubuntu@100.88.176.74:22:~/.ssh/id_ed25519:~/ringrift"
    ["gh200_f"]="ubuntu@100.104.165.116:22:~/.ssh/id_ed25519:~/ringrift"
    # Vast.ai nodes
    ["vast_a40"]="root@ssh8.vast.ai:38742:~/.ssh/id_cluster:/workspace/ringrift"
    ["vast_5090"]="root@ssh1.vast.ai:15166:~/.ssh/id_cluster:/workspace/ringrift"
)

# Training assignment: model -> node
declare -A TRAINING_ASSIGNMENT=(
    ["gmo"]="gh200_e"
    ["gmo_v2"]="vast_5090"
    ["ig_gmo"]="vast_a40"
)

# Parse node connection string
parse_node() {
    local name=$1
    IFS=':' read -r userhost port ssh_key repo_path <<< "${NODES[$name]}"
    echo "$userhost $port $ssh_key $repo_path"
}

# Sync code and data to a node
sync_to_node() {
    local name=$1
    read -r userhost port ssh_key repo_path <<< "$(parse_node "$name")"

    echo "=== Syncing to $name ($userhost:$port) ==="

    # Expand ssh key path
    ssh_key="${ssh_key/#\~/$HOME}"

    # Create repo directory
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$ssh_key" -p "$port" "$userhost" \
        "mkdir -p $repo_path/data/training $repo_path/models/gmo $repo_path/models/gmo_v2 $repo_path/models/ig_gmo" 2>/dev/null

    # Sync code (exclude large files, include training data)
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='*.pt' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='checkpoints' \
        --exclude='data/training/*.npz' \
        --exclude='data/training/*.db' \
        -e "ssh -o StrictHostKeyChecking=no -i $ssh_key -p $port" \
        "$LOCAL_PATH/app/" "$userhost:$repo_path/app/"

    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -i $ssh_key -p $port" \
        "$LOCAL_PATH/scripts/" "$userhost:$repo_path/scripts/"

    # Sync training data
    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -i $ssh_key -p $port" \
        "$LOCAL_PATH/data/training/gmo_full_sq8_2p.jsonl" \
        "$userhost:$repo_path/data/training/"

    # Sync pyproject.toml and setup
    rsync -avz \
        -e "ssh -o StrictHostKeyChecking=no -i $ssh_key -p $port" \
        "$LOCAL_PATH/pyproject.toml" "$userhost:$repo_path/"

    # Install dependencies
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" -p "$port" "$userhost" \
        "cd $repo_path && pip install -q -e . 2>/dev/null || pip install -q torch numpy tqdm pydantic" 2>/dev/null || true

    echo "=== $name synced ==="
}

# Start training on a node
start_training() {
    local model=$1
    local name="${TRAINING_ASSIGNMENT[$model]}"

    if [ -z "$name" ]; then
        echo "No node assigned for model: $model"
        return 1
    fi

    read -r userhost port ssh_key repo_path <<< "$(parse_node "$name")"
    ssh_key="${ssh_key/#\~/$HOME}"

    echo "=== Starting $model training on $name ==="

    case $model in
        "gmo")
            TRAIN_CMD="python -m app.training.train_gmo \
                --data-path data/training/gmo_full_sq8_2p.jsonl \
                --output-dir models/gmo/sq8_2p_full \
                --epochs 100 --batch-size 128 --lr 0.0005 \
                --early-stopping-patience 20 --device cuda"
            ;;
        "gmo_v2")
            TRAIN_CMD="python -m app.training.train_gmo_v2 \
                --data-path data/training/gmo_full_sq8_2p.jsonl \
                --output-dir models/gmo_v2/sq8_2p_full \
                --epochs 100 --batch-size 64 --lr 0.0003 --device cuda"
            ;;
        "ig_gmo")
            TRAIN_CMD="python -m app.training.train_ig_gmo \
                --data-path data/training/gmo_full_sq8_2p.jsonl \
                --output-dir models/ig_gmo/sq8_2p_full \
                --epochs 80 --batch-size 64 --lr 0.0003 --device cuda"
            ;;
        *)
            echo "Unknown model: $model"
            return 1
            ;;
    esac

    # Run training in background with nohup
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" -p "$port" "$userhost" \
        "cd $repo_path && nohup $TRAIN_CMD > /tmp/training_${model}.log 2>&1 &" 2>/dev/null

    echo "=== $model training started on $name (log: /tmp/training_${model}.log) ==="
}

# Check training status on all nodes
check_status() {
    for name in "${!NODES[@]}"; do
        read -r userhost port ssh_key repo_path <<< "$(parse_node "$name")"
        ssh_key="${ssh_key/#\~/$HOME}"

        echo "=== $name ==="
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$ssh_key" -p "$port" "$userhost" \
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; \
             ps aux | grep 'train_' | grep python | grep -v grep | head -3 || echo 'No training running'" \
            2>/dev/null || echo "Connection failed"
        echo
    done
}

# Show training logs
show_logs() {
    local model=${1:-"gmo"}
    local name="${TRAINING_ASSIGNMENT[$model]}"

    if [ -z "$name" ]; then
        echo "No node assigned for model: $model"
        return 1
    fi

    read -r userhost port ssh_key repo_path <<< "$(parse_node "$name")"
    ssh_key="${ssh_key/#\~/$HOME}"

    echo "=== Logs for $model on $name ==="
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" -p "$port" "$userhost" \
        "tail -100 /tmp/training_${model}.log 2>/dev/null || echo 'No log file found'" 2>/dev/null
}

# Fetch trained models back to local
fetch_models() {
    for model in "${!TRAINING_ASSIGNMENT[@]}"; do
        local name="${TRAINING_ASSIGNMENT[$model]}"
        read -r userhost port ssh_key repo_path <<< "$(parse_node "$name")"
        ssh_key="${ssh_key/#\~/$HOME}"

        echo "=== Fetching $model from $name ==="

        # Create local directory
        mkdir -p "$LOCAL_PATH/models/$model/sq8_2p_full"

        # Sync model files
        rsync -avz --progress \
            -e "ssh -o StrictHostKeyChecking=no -i $ssh_key -p $port" \
            "$userhost:$repo_path/models/$model/sq8_2p_full/*.pt" \
            "$LOCAL_PATH/models/$model/sq8_2p_full/" 2>/dev/null || echo "No models found yet"
    done
}

# Main
ACTION=${1:-"help"}

case $ACTION in
    "sync")
        for name in "${!NODES[@]}"; do
            sync_to_node "$name"
        done
        ;;
    "train")
        MODEL=${2:-"all"}
        if [ "$MODEL" == "all" ]; then
            for model in "${!TRAINING_ASSIGNMENT[@]}"; do
                start_training "$model"
            done
        else
            start_training "$MODEL"
        fi
        ;;
    "status")
        check_status
        ;;
    "logs")
        show_logs "${2:-gmo}"
        ;;
    "fetch")
        fetch_models
        ;;
    *)
        echo "Usage: $0 {sync|train [model]|status|logs [model]|fetch}"
        echo
        echo "Commands:"
        echo "  sync       - Sync code and data to all cluster nodes"
        echo "  train      - Start training (all|gmo|gmo_v2|ig_gmo)"
        echo "  status     - Check training status on all nodes"
        echo "  logs       - Show training logs (gmo|gmo_v2|ig_gmo)"
        echo "  fetch      - Fetch trained models back to local"
        echo
        echo "Node assignments:"
        for model in "${!TRAINING_ASSIGNMENT[@]}"; do
            echo "  $model -> ${TRAINING_ASSIGNMENT[$model]}"
        done
        ;;
esac
