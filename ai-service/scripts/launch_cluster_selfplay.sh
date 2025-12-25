#!/bin/bash
# Launch Gumbel MCTS selfplay across cluster for underrepresented configs
# Created: December 24, 2025
# Target: 5 hours of continuous selfplay generation

SSH_KEY=~/.ssh/id_cluster
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "=========================================="
echo "Cluster Selfplay Launcher"
echo "Target: 5 hours of continuous generation"
echo "=========================================="
echo ""

# GH200 nodes (96GB each)
GH200_NODES=(
    "192.222.51.29"    # lambda-gh200-a
    "192.222.51.162"   # lambda-gh200-c
    "192.222.58.122"   # lambda-gh200-d
    "192.222.57.162"   # lambda-gh200-e
    "192.222.57.178"   # lambda-gh200-f
    "192.222.57.79"    # lambda-gh200-g
    "192.222.56.123"   # lambda-gh200-h
    "192.222.50.112"   # lambda-gh200-i
    "192.222.51.150"   # lambda-gh200-k
    "192.222.51.233"   # lambda-gh200-l
    "192.222.50.219"   # lambda-gh200-m
    "192.222.51.204"   # lambda-gh200-n
    "192.222.51.161"   # lambda-gh200-b
    "192.222.51.92"    # lambda-gh200-o
    "192.222.51.215"   # lambda-gh200-p
    "192.222.51.18"    # lambda-gh200-q
    "192.222.50.172"   # lambda-gh200-r
    "192.222.51.167"   # lambda-gh200-s
    "192.222.50.211"   # lambda-gh200-t
)

# A10 nodes (23GB each)
A10_NODES=(
    "150.136.65.197"   # lambda-a10
    "129.153.159.191"  # lambda-a10-b
    "150.136.56.240"   # lambda-a10-c
)

# Underrepresented configs (priority order)
CONFIGS=(
    "hexagonal 2 10000"
    "hexagonal 4 10000"
    "square19 4 10000"
    "square8 3 10000"
    "hexagonal 3 10000"
    "square19 2 10000"
    "square19 3 10000"
    "hex8 3 5000"
    "hex8 2 5000"
    "hex8 4 5000"
)

echo "Checking node availability..."

# Collect available nodes
AVAILABLE_NODES=()

for host in "${GH200_NODES[@]}"; do
    gpu_util=$(timeout 5 ssh $SSH_OPTS -i $SSH_KEY ubuntu@$host "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null" 2>/dev/null)
    if [ -n "$gpu_util" ]; then
        util_val=$(echo $gpu_util | tr -d ' %')
        if [ "$util_val" -lt 50 ] 2>/dev/null; then
            AVAILABLE_NODES+=("$host:gh200")
            echo "  $host (GH200): ${gpu_util} - AVAILABLE"
        else
            echo "  $host (GH200): ${gpu_util} - BUSY"
        fi
    fi
done

for host in "${A10_NODES[@]}"; do
    gpu_util=$(timeout 5 ssh $SSH_OPTS -i $SSH_KEY ubuntu@$host "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null" 2>/dev/null)
    if [ -n "$gpu_util" ]; then
        util_val=$(echo $gpu_util | tr -d ' %')
        if [ "$util_val" -lt 50 ] 2>/dev/null; then
            AVAILABLE_NODES+=("$host:a10")
            echo "  $host (A10): ${gpu_util} - AVAILABLE"
        fi
    fi
done

echo ""
echo "Found ${#AVAILABLE_NODES[@]} available nodes"

if [ ${#AVAILABLE_NODES[@]} -eq 0 ]; then
    echo "No available nodes found."
    exit 1
fi

echo ""
echo "Launching selfplay jobs..."

node_idx=0
for config in "${CONFIGS[@]}"; do
    read -r board players games <<< "$config"

    if [ $node_idx -ge ${#AVAILABLE_NODES[@]} ]; then
        node_idx=0
    fi

    node_info="${AVAILABLE_NODES[$node_idx]}"
    host=$(echo $node_info | cut -d: -f1)
    gpu_type=$(echo $node_info | cut -d: -f2)

    if [ "$gpu_type" = "gh200" ]; then
        if [ "$board" = "hexagonal" ] || [ "$board" = "square19" ]; then
            batch_size=64
        else
            batch_size=128
        fi
    else
        if [ "$board" = "hexagonal" ] || [ "$board" = "square19" ]; then
            batch_size=32
        else
            batch_size=64
        fi
    fi

    session_name="gumbel_${board}_${players}p"

    echo "[$host] $session_name ($games games, batch=$batch_size)..."

    ssh $SSH_OPTS -i $SSH_KEY ubuntu@$host "
        screen -X -S $session_name quit 2>/dev/null || true
        cd ~/ringrift/ai-service && \
        screen -dmS $session_name bash -c '
            source venv/bin/activate && \
            export PYTHONPATH=. && \
            export CUDA_VISIBLE_DEVICES=0 && \
            python3 scripts/run_gpu_mcts_selfplay.py \
                --board $board --num-players $players \
                --batch-size $batch_size --budget 64 \
                --eval-mode heuristic --games $games \
                --output data/selfplay/${session_name} --continuous \
                2>&1 | tee /tmp/${session_name}.log
            exec bash
        '
    " 2>/dev/null && echo "  OK" || echo "  FAILED"

    ((node_idx++))
done

echo ""
echo "Done! Monitor with: ssh ubuntu@<host> 'screen -ls'"
