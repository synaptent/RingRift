#!/bin/bash
#
# Vast.ai Cluster Setup Script - Fix model sync and PYTHONPATH issues
#
# This script:
# 1. Syncs model files to all Vast.ai cluster instances
# 2. Sets up proper PYTHONPATH in startup scripts
# 3. Verifies environment before running training jobs
#
# Usage:
#   ./scripts/cluster_vast_setup.sh --sync-models      # Sync models to all instances
#   ./scripts/cluster_vast_setup.sh --fix-env          # Fix PYTHONPATH on all instances
#   ./scripts/cluster_vast_setup.sh --verify           # Verify environment on all instances
#   ./scripts/cluster_vast_setup.sh --full             # Do all of the above
#   ./scripts/cluster_vast_setup.sh --instance 28844401  # Fix single instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$AI_SERVICE_DIR/models"
LOG_DIR="$AI_SERVICE_DIR/logs"

mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Model files to sync (most commonly needed)
ESSENTIAL_MODELS=(
    "ringrift_best_sq8_2p.pth"
    "ringrift_best_sq8_3p.pth"
    "ringrift_best_sq8_4p.pth"
    "ringrift_best_hex_2p.pth"
    "ringrift_best_sq19_2p.pth"
)

# Get all running Vast instances
get_vast_instances() {
    vastai show instances --raw 2>/dev/null | \
        python3 -c "
import sys, json
instances = json.load(sys.stdin)
for inst in instances:
    if inst.get('actual_status') == 'running':
        print(f\"{inst['id']}|{inst['ssh_host']}|{inst['ssh_port']}|{inst.get('gpu_name', 'Unknown')}\")
"
}

# SSH to a Vast instance
ssh_vast() {
    local host=$1
    local port=$2
    shift 2
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$port" "root@$host" "$@"
}

# SCP to a Vast instance
scp_vast() {
    local host=$1
    local port=$2
    local src=$3
    local dst=$4
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no -P "$port" "$src" "root@$host:$dst"
}

# Sync models to a single instance
sync_models_to_instance() {
    local instance_id=$1
    local host=$2
    local port=$3
    local gpu=$4

    log_info "Syncing models to instance $instance_id ($gpu)..."

    # Find RingRift path on instance
    local ringrift_path=$(ssh_vast "$host" "$port" 'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null' 2>/dev/null || echo "/root/ringrift")

    # Create models directory if it doesn't exist
    ssh_vast "$host" "$port" "mkdir -p $ringrift_path/ai-service/models" 2>/dev/null || true

    # Sync each essential model
    local synced=0
    for model in "${ESSENTIAL_MODELS[@]}"; do
        local local_path="$MODELS_DIR/$model"
        if [ -f "$local_path" ]; then
            # Check if model exists on remote and is same size
            local remote_size=$(ssh_vast "$host" "$port" "stat -c %s $ringrift_path/ai-service/models/$model 2>/dev/null || echo 0" 2>/dev/null || echo "0")
            local local_size=$(stat -f %z "$local_path" 2>/dev/null || stat -c %s "$local_path" 2>/dev/null)

            if [ "$remote_size" != "$local_size" ]; then
                log_info "  Syncing $model (${local_size} bytes)..."
                scp_vast "$host" "$port" "$local_path" "$ringrift_path/ai-service/models/$model"
                ((synced++))
            else
                log_info "  $model already synced"
            fi
        else
            log_warn "  Local model not found: $model"
        fi
    done

    log_info "Synced $synced models to instance $instance_id"
    return 0
}

# Fix PYTHONPATH on a single instance
fix_env_on_instance() {
    local instance_id=$1
    local host=$2
    local port=$3
    local gpu=$4

    log_info "Fixing environment on instance $instance_id ($gpu)..."

    # Find RingRift path
    local ringrift_path=$(ssh_vast "$host" "$port" 'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null' 2>/dev/null || echo "/root/ringrift")

    # Create a startup script that sets PYTHONPATH
    ssh_vast "$host" "$port" "cat > $ringrift_path/ai-service/scripts/setup_env.sh << 'ENVSCRIPT'
#!/bin/bash
# RingRift environment setup
cd $ringrift_path/ai-service
export PYTHONPATH=\"\$(pwd):\$PYTHONPATH\"

# Activate virtual environment if it exists
if [ -f \"venv/bin/activate\" ]; then
    source \"venv/bin/activate\"
fi

echo \"Environment ready: PYTHONPATH=\$PYTHONPATH\"
ENVSCRIPT
chmod +x $ringrift_path/ai-service/scripts/setup_env.sh" 2>/dev/null

    # Add to .bashrc if not already there
    ssh_vast "$host" "$port" "
grep -q 'setup_env.sh' ~/.bashrc 2>/dev/null || \
echo 'cd $ringrift_path/ai-service && source scripts/setup_env.sh 2>/dev/null' >> ~/.bashrc
" 2>/dev/null

    log_info "Environment fixed on instance $instance_id"
    return 0
}

# Verify environment on a single instance
verify_instance() {
    local instance_id=$1
    local host=$2
    local port=$3
    local gpu=$4

    log_info "Verifying instance $instance_id ($gpu)..."

    # Find RingRift path
    local ringrift_path=$(ssh_vast "$host" "$port" 'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null' 2>/dev/null || echo "/root/ringrift")

    # Check critical items
    local errors=0

    # 1. Check models directory
    local models_exist=$(ssh_vast "$host" "$port" "ls $ringrift_path/ai-service/models/*.pth 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    if [ "$models_exist" -gt 0 ]; then
        log_info "  Models: $models_exist .pth files found"
    else
        log_error "  Models: No .pth files found!"
        ((errors++))
    fi

    # 2. Check PYTHONPATH setup
    local pythonpath_ok=$(ssh_vast "$host" "$port" "cd $ringrift_path/ai-service && export PYTHONPATH=\$(pwd) && python3 -c 'import app; print(\"OK\")' 2>/dev/null" 2>/dev/null || echo "FAIL")
    if [ "$pythonpath_ok" = "OK" ]; then
        log_info "  PYTHONPATH: OK"
    else
        log_error "  PYTHONPATH: Cannot import app module!"
        ((errors++))
    fi

    # 3. Check GPU availability
    local gpu_available=$(ssh_vast "$host" "$port" "python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null" 2>/dev/null || echo "Error")
    if [ "$gpu_available" = "True" ]; then
        log_info "  GPU: Available"
    else
        log_warn "  GPU: $gpu_available"
    fi

    # 4. Check P2P orchestrator
    local p2p_health=$(ssh_vast "$host" "$port" "curl -s http://localhost:8770/health 2>/dev/null | head -c 50" 2>/dev/null || echo "Not running")
    if echo "$p2p_health" | grep -q "healthy"; then
        log_info "  P2P: Running"
    else
        log_warn "  P2P: Not running (may be OK if not needed)"
    fi

    if [ "$errors" -eq 0 ]; then
        log_info "Instance $instance_id: ALL CHECKS PASSED"
        return 0
    else
        log_error "Instance $instance_id: $errors CHECKS FAILED"
        return 1
    fi
}

# Run Gumbel selfplay on a single instance
run_gumbel_on_instance() {
    local instance_id=$1
    local host=$2
    local port=$3
    local gpu=$4
    local num_games=${5:-100}

    log_info "Starting Gumbel selfplay on instance $instance_id ($gpu)..."

    # Find RingRift path
    local ringrift_path=$(ssh_vast "$host" "$port" 'ls -d /workspace/ringrift 2>/dev/null || ls -d /root/ringrift 2>/dev/null' 2>/dev/null || echo "/root/ringrift")

    # Run Gumbel selfplay with proper environment
    ssh_vast "$host" "$port" "
cd $ringrift_path/ai-service
export PYTHONPATH=\$(pwd):\$PYTHONPATH
source venv/bin/activate 2>/dev/null || true

# Kill any existing Gumbel process
pkill -f generate_gumbel_selfplay || true

# Start Gumbel selfplay in background
nohup python3 scripts/generate_gumbel_selfplay.py \
    --num-games $num_games \
    --board-type square8 \
    --num-players 2 \
    --output-dir data/gumbel_selfplay \
    > logs/gumbel_selfplay.log 2>&1 &

echo \"Started Gumbel selfplay with PID \$!\"
sleep 2
ps aux | grep -v grep | grep generate_gumbel || echo \"Process not running\"
" 2>/dev/null

    return 0
}

# Process command line arguments
SYNC_MODELS=false
FIX_ENV=false
VERIFY=false
RUN_GUMBEL=false
SPECIFIC_INSTANCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sync-models)
            SYNC_MODELS=true
            shift
            ;;
        --fix-env)
            FIX_ENV=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --run-gumbel)
            RUN_GUMBEL=true
            shift
            ;;
        --full)
            SYNC_MODELS=true
            FIX_ENV=true
            VERIFY=true
            shift
            ;;
        --instance)
            SPECIFIC_INSTANCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sync-models    Sync model files to all instances"
            echo "  --fix-env        Fix PYTHONPATH on all instances"
            echo "  --verify         Verify environment on all instances"
            echo "  --run-gumbel     Start Gumbel selfplay on all instances"
            echo "  --full           Do sync-models, fix-env, and verify"
            echo "  --instance ID    Only process specific instance ID"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to verify if no action specified
if ! $SYNC_MODELS && ! $FIX_ENV && ! $VERIFY && ! $RUN_GUMBEL; then
    VERIFY=true
fi

# Get instances
log_info "Getting Vast.ai instances..."
instances=$(get_vast_instances)

if [ -z "$instances" ]; then
    log_error "No running Vast instances found"
    exit 1
fi

echo ""
echo "========================================"
echo "VAST.AI CLUSTER STATUS"
echo "========================================"

# Process each instance
while IFS='|' read -r id host port gpu; do
    # Skip if specific instance requested and this isn't it
    if [ -n "$SPECIFIC_INSTANCE" ] && [ "$id" != "$SPECIFIC_INSTANCE" ]; then
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "Instance: $id ($gpu)"
    echo "Host: $host:$port"
    echo "----------------------------------------"

    if $SYNC_MODELS; then
        sync_models_to_instance "$id" "$host" "$port" "$gpu" || log_warn "Failed to sync models to $id"
    fi

    if $FIX_ENV; then
        fix_env_on_instance "$id" "$host" "$port" "$gpu" || log_warn "Failed to fix env on $id"
    fi

    if $VERIFY; then
        verify_instance "$id" "$host" "$port" "$gpu" || log_warn "Verification issues on $id"
    fi

    if $RUN_GUMBEL; then
        run_gumbel_on_instance "$id" "$host" "$port" "$gpu" 100 || log_warn "Failed to start Gumbel on $id"
    fi

done <<< "$instances"

echo ""
echo "========================================"
echo "CLUSTER SETUP COMPLETE"
echo "========================================"
