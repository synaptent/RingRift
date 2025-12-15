#!/bin/bash
# Deploy Unified AI Loop to Cluster
#
# This script deploys all components of the unified AI loop to the cluster,
# installs systemd services, and enables hot model reload on selfplay workers.
#
# Usage:
#   ./scripts/deploy_unified_loop.sh              # Deploy to all hosts
#   ./scripts/deploy_unified_loop.sh --primary    # Deploy only to primary (lambda_h100)
#   ./scripts/deploy_unified_loop.sh --selfplay   # Deploy only to selfplay hosts
#   ./scripts/deploy_unified_loop.sh --dry-run    # Show what would be done

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Host configurations
PRIMARY_HOST="ubuntu@209.20.157.81"  # lambda_h100
PRIMARY_NAME="lambda_h100"

TRAINING_HOSTS=(
    "ubuntu@209.20.157.81:lambda_h100"
    "ubuntu@192.222.53.22:lambda_2xh100"
    "ubuntu@150.136.65.197:lambda_a10"
)

SELFPLAY_HOSTS=(
    "ubuntu@192.222.51.29:gh200_a"
    "ubuntu@192.222.51.167:gh200_b"
    "ubuntu@192.222.51.162:gh200_c"
    "ubuntu@192.222.58.122:gh200_d"
    "ubuntu@192.222.57.162:gh200_e"
    "ubuntu@192.222.57.178:gh200_f"
    "ubuntu@100.88.176.74:lambda_gh200_e"
    "ubuntu@100.104.165.116:lambda_gh200_f"
    "ubuntu@100.104.126.58:lambda_gh200_g"
    "ubuntu@100.65.88.62:lambda_gh200_h"
)

AWS_HOSTS=(
    "ubuntu@100.121.198.28:aws_staging"
    "ubuntu@100.115.97.24:aws_worker"
)

# Files to deploy
CORE_FILES=(
    "scripts/unified_ai_loop.py"
    "scripts/unified_data_sync.py"
    "scripts/streaming_data_collector.py"
    "scripts/shadow_tournament_service.py"
    "scripts/regression_gate.py"
    "scripts/run_model_elo_tournament.py"
    "scripts/model_promotion_manager.py"
    "scripts/run_self_play_soak.py"
    "scripts/curriculum_training.py"
    "scripts/cluster_orchestrator.py"
    "config/unified_loop.yaml"
    "config/remote_hosts.yaml"
    "app/distributed/data_events.py"
)

SYSTEMD_FILES=(
    "config/systemd/ringrift-ai-loop.service"
    "config/systemd/ringrift-selfplay@.service"
)

MONITORING_FILES=(
    "config/monitoring/grafana-dashboard.json"
    "config/monitoring/alerting-rules.yaml"
)

# Parse arguments
DRY_RUN=false
PRIMARY_ONLY=false
SELFPLAY_ONLY=false
INSTALL_SYSTEMD=false
ENABLE_HOT_RELOAD=true
RESTART_SERVICES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --primary)
            PRIMARY_ONLY=true
            shift
            ;;
        --selfplay)
            SELFPLAY_ONLY=true
            shift
            ;;
        --install-systemd)
            INSTALL_SYSTEMD=true
            shift
            ;;
        --restart)
            RESTART_SERVICES=true
            shift
            ;;
        --no-hot-reload)
            ENABLE_HOT_RELOAD=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

deploy_to_host() {
    local addr="${1%:*}"
    local name="${1#*:}"
    local remote_path="~/ringrift/ai-service"

    echo -n "  [$name] "

    if $DRY_RUN; then
        echo "DRY-RUN: Would deploy to $addr"
        return 0
    fi

    # Check connectivity
    if ! timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes "$addr" "echo OK" &>/dev/null; then
        echo -e "${RED}UNREACHABLE${NC}"
        return 1
    fi

    # Create directories
    ssh -o ConnectTimeout=10 "$addr" "mkdir -p $remote_path/scripts $remote_path/config/systemd $remote_path/config/monitoring $remote_path/app/distributed $remote_path/logs/unified_loop $remote_path/logs/selfplay" 2>/dev/null

    # Deploy core files
    for file in "${CORE_FILES[@]}"; do
        rsync -az --timeout=30 "$AI_SERVICE_ROOT/$file" "$addr:$remote_path/$file" 2>/dev/null || true
    done

    # Deploy systemd files
    for file in "${SYSTEMD_FILES[@]}"; do
        if [[ -f "$AI_SERVICE_ROOT/$file" ]]; then
            rsync -az --timeout=30 "$AI_SERVICE_ROOT/$file" "$addr:$remote_path/$file" 2>/dev/null || true
        fi
    done

    # Deploy monitoring files
    for file in "${MONITORING_FILES[@]}"; do
        if [[ -f "$AI_SERVICE_ROOT/$file" ]]; then
            rsync -az --timeout=30 "$AI_SERVICE_ROOT/$file" "$addr:$remote_path/$file" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}OK${NC}"
    return 0
}

install_systemd_service() {
    local addr="${1%:*}"
    local name="${1#*:}"

    echo -n "  [$name] Installing systemd services... "

    if $DRY_RUN; then
        echo "DRY-RUN"
        return 0
    fi

    ssh -o ConnectTimeout=10 "$addr" "
        sudo cp ~/ringrift/ai-service/config/systemd/ringrift-ai-loop.service /etc/systemd/system/ 2>/dev/null || true
        sudo cp ~/ringrift/ai-service/config/systemd/ringrift-selfplay@.service /etc/systemd/system/ 2>/dev/null || true
        sudo systemctl daemon-reload
    " 2>/dev/null

    echo -e "${GREEN}OK${NC}"
}

enable_hot_reload_on_host() {
    local addr="${1%:*}"
    local name="${1#*:}"

    echo -n "  [$name] Configuring hot reload... "

    if $DRY_RUN; then
        echo "DRY-RUN"
        return 0
    fi

    # Check if selfplay is running and what config it's using
    local selfplay_pids=$(ssh -o ConnectTimeout=5 "$addr" "pgrep -f 'run_self_play_soak' || echo ''" 2>/dev/null)

    if [[ -z "$selfplay_pids" ]]; then
        echo -e "${YELLOW}NO SELFPLAY RUNNING${NC}"
        return 0
    fi

    # Create hot reload signal file that workers can watch
    ssh -o ConnectTimeout=5 "$addr" "
        mkdir -p ~/ringrift/ai-service/data/model_updates
        touch ~/ringrift/ai-service/data/model_updates/.hot_reload_enabled
    " 2>/dev/null

    echo -e "${GREEN}ENABLED${NC} (${selfplay_pids// /,})"
}

restart_unified_loop() {
    local addr="${1%:*}"
    local name="${1#*:}"

    echo -n "  [$name] Restarting unified loop... "

    if $DRY_RUN; then
        echo "DRY-RUN"
        return 0
    fi

    ssh -o ConnectTimeout=10 "$addr" "
        cd ~/ringrift/ai-service
        # Kill existing process
        pkill -f 'unified_ai_loop.py' 2>/dev/null || true
        sleep 2
        # Start new process
        nohup python3 scripts/unified_ai_loop.py --foreground -v > logs/unified_loop/daemon.log 2>&1 &
        sleep 2
        # Verify
        pgrep -f 'unified_ai_loop.py' > /dev/null && echo 'STARTED' || echo 'FAILED'
    " 2>/dev/null
}

# Main execution
echo "=============================================="
echo "  RingRift Unified AI Loop Deployment"
echo "=============================================="
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Step 1: Deploy files
log_info "Step 1: Deploying files to hosts"

if ! $SELFPLAY_ONLY; then
    echo "Training hosts:"
    for host in "${TRAINING_HOSTS[@]}"; do
        deploy_to_host "$host"
    done
fi

if ! $PRIMARY_ONLY; then
    echo "Selfplay hosts:"
    for host in "${SELFPLAY_HOSTS[@]}"; do
        deploy_to_host "$host"
    done

    echo "AWS hosts:"
    for host in "${AWS_HOSTS[@]}"; do
        deploy_to_host "$host"
    done
fi

echo ""

# Step 2: Install systemd services (if requested)
if $INSTALL_SYSTEMD; then
    log_info "Step 2: Installing systemd services"

    if ! $SELFPLAY_ONLY; then
        echo "Training hosts:"
        for host in "${TRAINING_HOSTS[@]}"; do
            install_systemd_service "$host"
        done
    fi

    if ! $PRIMARY_ONLY; then
        echo "Selfplay hosts:"
        for host in "${SELFPLAY_HOSTS[@]}"; do
            install_systemd_service "$host"
        done
    fi

    echo ""
fi

# Step 3: Enable hot reload on selfplay hosts
if $ENABLE_HOT_RELOAD && ! $PRIMARY_ONLY; then
    log_info "Step 3: Enabling hot model reload on selfplay hosts"

    for host in "${SELFPLAY_HOSTS[@]}"; do
        enable_hot_reload_on_host "$host"
    done

    echo ""
fi

# Step 4: Restart services (if requested)
if $RESTART_SERVICES; then
    log_info "Step 4: Restarting unified loop on primary"
    restart_unified_loop "$PRIMARY_HOST:$PRIMARY_NAME"
    echo ""
fi

# Summary
echo "=============================================="
log_info "Deployment complete!"
echo ""
echo "To start the unified loop on primary:"
echo "  ssh $PRIMARY_HOST 'cd ~/ringrift/ai-service && python3 scripts/unified_ai_loop.py --foreground -v'"
echo ""
echo "To check status:"
echo "  ssh $PRIMARY_HOST 'cd ~/ringrift/ai-service && python3 scripts/unified_ai_loop.py --status'"
echo ""
echo "To view metrics (once running):"
echo "  curl http://209.20.157.81:9090/metrics"
echo "=============================================="
