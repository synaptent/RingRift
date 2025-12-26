#!/bin/bash
# Quick setup script for new Lambda Labs GPU instances
# Usage: ./scripts/setup_new_lambda_instances.sh [host]
# Example: ./scripts/setup_new_lambda_instances.sh gh200-m

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="${SCRIPT_DIR}/.."

# New instances to set up (already running)
RUNNING_HOSTS=(
    "ubuntu@192.222.50.219"   # gh200_m
    "ubuntu@192.222.51.204"   # gh200_n
    "ubuntu@192.222.51.92"    # gh200_o
    "ubuntu@192.222.51.215"   # gh200_p
    "ubuntu@129.153.159.191"  # lambda_a10_b
    "ubuntu@150.136.56.240"   # lambda_a10_c
)

# Booting instances (wait for these)
BOOTING_HOSTS=(
    "ubuntu@192.222.51.18"    # gh200_q
    "ubuntu@192.222.50.172"   # gh200_r
    "ubuntu@192.222.51.89"    # gh200_s
)

setup_host() {
    local HOST="$1"
    echo "=============================================="
    echo "Setting up: $HOST"
    echo "=============================================="

    # Test SSH connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "echo 'SSH OK'" 2>/dev/null; then
        echo "ERROR: Cannot SSH to $HOST - skipping"
        return 1
    fi

    # Create directories and run setup
    ssh "$HOST" << 'REMOTE_SCRIPT'
set -e
echo "=== Creating project directories ==="
mkdir -p ~/ringrift/ai-service/{app,scripts,data/games,models,config}

echo "=== Checking GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

echo "=== Setting up Python venv ==="
cd ~/ringrift/ai-service
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip wheel -q

echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Verifying CUDA ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== Installing dependencies ==="
pip install numpy scipy cma tqdm pydantic aiohttp prometheus_client pyyaml -q

echo "=== Setup complete for $(hostname) ==="
REMOTE_SCRIPT

    echo ""
    echo "=== Syncing code to $HOST ==="
    rsync -avz --exclude='venv' --exclude='*.db' --exclude='__pycache__' \
        --exclude='.git' --exclude='data/games' --exclude='models/*.pth' \
        "${AI_SERVICE_DIR}/" "${HOST}:~/ringrift/ai-service/"

    echo "=== Syncing config ==="
    rsync -avz "${AI_SERVICE_DIR}/config/" "${HOST}:~/ringrift/ai-service/config/"

    echo ""
    echo "SUCCESS: $HOST is ready!"
    echo ""
}

# Main
if [ -n "$1" ]; then
    # Setup specific host
    setup_host "$1"
else
    echo "Setting up all RUNNING Lambda instances..."
    echo ""

    for host in "${RUNNING_HOSTS[@]}"; do
        setup_host "$host" || true
        echo ""
    done

    echo "=============================================="
    echo "BOOTING instances (check later):"
    for host in "${BOOTING_HOSTS[@]}"; do
        echo "  $host"
    done
    echo ""
    echo "Run this script again for booting instances once they're ready:"
    echo "  ./scripts/setup_new_lambda_instances.sh ubuntu@192.222.51.18"
    echo "=============================================="
fi
