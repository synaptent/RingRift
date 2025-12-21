#!/bin/bash
# Deploy Multi-Config NNUE Training Across Cluster
#
# Trains all 12 board/player combinations (4 boards × 3 player counts)
# on Lambda GH200 cluster nodes for comprehensive 2000+ Elo coverage.
#
# Usage:
#   ./scripts/deploy_multi_config_training.sh deploy   # Deploy training jobs
#   ./scripts/deploy_multi_config_training.sh status   # Check status
#   ./scripts/deploy_multi_config_training.sh collect  # Collect trained models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/cluster_common.sh" 2>/dev/null || true

# SSH options for cluster access
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes}"
REMOTE_PATH="~/ringrift/ai-service"

# All 12 board/player combinations
CONFIGS=(
    "square8:2"
    "square8:3"
    "square8:4"
    "square19:2"
    "square19:3"
    "square19:4"
    "hex8:2"
    "hex8:3"
    "hex8:4"
    "hexagonal:2"
    "hexagonal:3"
    "hexagonal:4"
)

# Lambda GH200 nodes (via Tailscale)
GH200_HOSTS=(
    "ubuntu@lambda-gh200-a"
    "ubuntu@lambda-gh200-c"
    "ubuntu@lambda-gh200-d"
    "ubuntu@lambda-gh200-e"
    "ubuntu@lambda-gh200-f"
    "ubuntu@lambda-gh200-g"
    "ubuntu@lambda-gh200-h"
    "ubuntu@lambda-gh200-i"
    "ubuntu@lambda-gh200-k"
    "ubuntu@lambda-gh200-l"
)

# H100 nodes for larger board training (square19, hexagonal)
H100_HOSTS=(
    "ubuntu@lambda-h100"
    "ubuntu@lambda-2xh100"
)

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

check_node_available() {
    local host="$1"
    ssh $SSH_OPTS "$host" "echo ok" 2>/dev/null && return 0 || return 1
}

get_available_nodes() {
    local available=()
    print_header "Checking Node Availability"

    for host in "${GH200_HOSTS[@]}" "${H100_HOSTS[@]}"; do
        if check_node_available "$host"; then
            available+=("$host")
            echo "  ✓ $host available"
        else
            echo "  ✗ $host offline"
        fi
    done

    echo "${available[@]}"
}

deploy_training() {
    print_header "Deploying Multi-Config Training (12 Combinations)"

    # Get available nodes
    readarray -t AVAILABLE < <(get_available_nodes)

    if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
        echo "ERROR: No cluster nodes available"
        exit 1
    fi

    echo ""
    echo "Available nodes: ${#AVAILABLE[@]}"
    echo "Configurations: ${#CONFIGS[@]}"

    # Distribute configs across nodes (round-robin)
    local node_idx=0

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r board players <<< "$config"
        local host="${AVAILABLE[$node_idx]}"
        local config_key="${board}_${players}p"

        echo ""
        echo "--- Deploying $config_key to $host ---"

        # Training command using train_nnue.py with canonical data
        local train_cmd="cd $REMOTE_PATH && source venv/bin/activate && \\
            export PYTHONPATH=\$PYTHONPATH:$REMOTE_PATH && \\
            mkdir -p logs models/nnue data/training && \\
            nohup python scripts/train_nnue.py \\
                --board-type $board \\
                --num-players $players \\
                --epochs 100 \\
                --batch-size 256 \\
                --learning-rate 0.0003 \\
                --model-name nnue_policy_${config_key} \\
                --data-dir data/training \\
                --output-dir models/nnue \\
                > logs/train_${config_key}.log 2>&1 &
            echo 'Started training for $config_key (PID: '\$!')'
            sleep 2
            tail -5 logs/train_${config_key}.log 2>/dev/null || echo 'Log not ready'"

        ssh $SSH_OPTS "$host" "$train_cmd" 2>&1 || echo "  Failed to start on $host"

        # Round-robin to next node
        node_idx=$(( (node_idx + 1) % ${#AVAILABLE[@]} ))
    done

    echo ""
    print_header "Training Deployed"
    echo "Monitor with: $0 status"
}

check_status() {
    print_header "Multi-Config Training Status"

    for host in "${GH200_HOSTS[@]}" "${H100_HOSTS[@]}"; do
        echo ""
        echo "--- $host ---"

        ssh $SSH_OPTS "$host" "
            cd $REMOTE_PATH 2>/dev/null || exit 1

            # Show running training processes
            echo 'Running:'
            ps aux | grep 'train_nnue\|train_policy' | grep -v grep | awk '{print \"  \" \$11 \" \" \$12 \" \" \$13}' || echo '  None'

            # Show recent logs
            echo ''
            echo 'Recent logs:'
            for log in logs/train_*.log; do
                if [[ -f \"\$log\" ]]; then
                    name=\$(basename \"\$log\" .log)
                    last_line=\$(tail -1 \"\$log\" 2>/dev/null)
                    echo \"  \$name: \$last_line\"
                fi
            done 2>/dev/null || echo '  No logs found'

            # Show GPU utilization
            echo ''
            echo 'GPU:'
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo '  N/A'
        " 2>&1 || echo "  Unreachable"
    done
}

collect_models() {
    print_header "Collecting Trained Models"

    local collect_dir="models/nnue/cluster_collected_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$collect_dir"

    for host in "${GH200_HOSTS[@]}" "${H100_HOSTS[@]}"; do
        echo "Collecting from $host..."

        scp $SSH_OPTS "$host:$REMOTE_PATH/models/nnue/nnue_policy_*_$(date +%Y%m%d)*.pth" \
            "$collect_dir/" 2>/dev/null || echo "  No models found on $host"
    done

    echo ""
    echo "Models collected to: $collect_dir"
    ls -la "$collect_dir" 2>/dev/null || echo "No models collected"
}

generate_selfplay_all() {
    print_header "Starting Canonical Selfplay for All 12 Combinations"

    # Get available nodes
    readarray -t AVAILABLE < <(get_available_nodes)

    if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
        echo "ERROR: No cluster nodes available"
        exit 1
    fi

    local node_idx=0

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r board players <<< "$config"
        local host="${AVAILABLE[$node_idx]}"
        local config_key="${board}_${players}p"

        echo ""
        echo "--- Starting canonical selfplay for $config_key on $host ---"

        # Use generate_canonical_selfplay.py which validates parity + canonical phase history
        # Uses --min-recorded-games + --max-soak-attempts for iterative scale-up
        local selfplay_cmd="cd $REMOTE_PATH && source venv/bin/activate && \\
            export PYTHONPATH=\$PYTHONPATH:$REMOTE_PATH && \\
            mkdir -p data/games logs data/training && \\
            nohup python scripts/generate_canonical_selfplay.py \\
                --board $board \\
                --num-players $players \\
                --num-games 50 \\
                --min-recorded-games 200 \\
                --max-soak-attempts 10 \\
                --difficulty-band light \\
                --db data/games/canonical_${board}_${players}p.db \\
                --summary data/games/db_health.canonical_${board}_${players}p.json \\
                > logs/canonical_selfplay_${config_key}.log 2>&1 &
            echo 'Canonical selfplay started for $config_key (PID: '\$!')'
            sleep 2
            tail -3 logs/canonical_selfplay_${config_key}.log 2>/dev/null || echo 'Log not ready'"

        ssh $SSH_OPTS "$host" "$selfplay_cmd" 2>&1 || echo "  Failed on $host"

        node_idx=$(( (node_idx + 1) % ${#AVAILABLE[@]} ))
    done

    echo ""
    print_header "Canonical Selfplay Deployed"
    echo "Monitor with: $0 status"
    echo ""
    echo "Once selfplay completes (check logs/canonical_selfplay_*.log for 'canonical_ok: true'):"
    echo ""
    echo "  1. Export to NPZ format per-config:"
    echo "     ssh <host> 'cd $REMOTE_PATH && python scripts/db_to_training_npz.py \\"
    echo "       --db data/games/canonical_<board>_<N>p.db \\"
    echo "       --output data/training/canonical_<board>_<N>p.npz \\"
    echo "       --board-type <board> --num-players <N>'"
    echo ""
    echo "  2. Train models: $0 deploy"
    echo ""
    echo "  3. Collect trained models: $0 collect"
}

case "${1:-help}" in
    deploy)
        deploy_training
        ;;
    status)
        check_status
        ;;
    collect)
        collect_models
        ;;
    selfplay)
        generate_selfplay_all
        ;;
    help|*)
        echo "Multi-Config NNUE Training Deployment"
        echo ""
        echo "Usage: $0 {deploy|status|collect|selfplay}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy training for all 12 board/player combinations"
        echo "  status   - Check training status on all nodes"
        echo "  collect  - Collect trained models from cluster"
        echo "  selfplay - Start selfplay data generation for all configs"
        echo ""
        echo "Configurations (12 total):"
        for config in "${CONFIGS[@]}"; do
            echo "  - $config"
        done
        ;;
esac
