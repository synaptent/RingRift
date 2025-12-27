#!/bin/bash
# RingRift AI Cluster Orchestration Startup Script
# This script starts all components of the AI self-improvement loop across the cluster.
#
# Usage:
#   ./scripts/start_cluster.sh [--dry-run]
#
# Components started:
# 1. Unified AI Loop (coordinator) - vast-rtx4060ti
# 2. Job State Sync Daemon (coordinator) - vast-rtx4060ti
# 3. Data Sync Service - vast-512cpu
# 4. Balanced Selfplay across all GPU nodes
#
# Prerequisites:
# - Tailscale connected to all vast nodes
# - SSH keys configured for root@vast-* hosts
# - Config files in config/distributed_hosts.yaml and config/unified_loop.yaml

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] Would execute the following commands:"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Node configurations
COORDINATOR="vast-rtx4060ti"
DATA_AGGREGATOR="vast-512cpu"
GPU_NODES=("vast-4080s-2x" "vast-rtx4060ti" "vast-2080ti" "vast-3070-24cpu" "vast-2060s-22cpu" "vast-3060ti-64cpu" "vast-5070-4x")

# Selfplay configurations to distribute across GPU nodes
# Format: "board_type:num_players:max_moves"
SELFPLAY_CONFIGS=(
    "square19:4:2000"
    "square19:3:1600"
    "square19:2:1200"
    "hexagonal:4:2000"
    "hexagonal:3:1600"
    "hexagonal:2:1200"
    "square8:4:1200"
    "square8:3:800"
)

# Check Tailscale connectivity
check_tailscale() {
    log_info "Checking Tailscale connectivity..."
    if ! tailscale status | grep -q "vast-"; then
        log_error "No vast nodes found on Tailscale. Please run 'tailscale status' to check."
        exit 1
    fi
    local connected=$(tailscale status | grep "vast-" | grep -c "active\|idle" || echo 0)
    log_info "Found $connected vast nodes on Tailscale"
}

# Sync configs to all nodes
sync_configs() {
    log_info "Syncing config files to all nodes..."
    local config_dir="$(dirname "$0")/../config"

    local config_files=("$config_dir/unified_loop.yaml" "$config_dir/distributed_hosts.yaml")
    if [[ ! -f "$config_dir/distributed_hosts.yaml" && -f "$config_dir/remote_hosts.yaml" ]]; then
        config_files+=("$config_dir/remote_hosts.yaml")
    fi

    for node in "${GPU_NODES[@]}" "$DATA_AGGREGATOR"; do
        if $DRY_RUN; then
            echo "  scp ${config_files[*]} root@$node:..."
        else
            # Find the ringrift directory on this node
            local remote_path=$(ssh -o ConnectTimeout=5 root@$node 'ls -d /workspace/ringrift /root/ringrift /workspace/RingRift /root/RingRift 2>/dev/null | head -1' 2>/dev/null || echo "")
            if [[ -n "$remote_path" ]]; then
                scp -q "${config_files[@]}" "root@$node:$remote_path/ai-service/config/" 2>/dev/null || true
                log_info "  Synced configs to $node"
            else
                log_warn "  Could not find ringrift directory on $node"
            fi
        fi
    done
}

# Start the unified AI loop on coordinator
start_unified_loop() {
    log_info "Starting Unified AI Loop on $COORDINATOR..."

    if $DRY_RUN; then
        echo "  ssh root@$COORDINATOR 'cd .../ai-service && python3 scripts/unified_ai_loop.py --start --verbose'"
        return
    fi

    # Find ringrift directory
    local remote_path=$(ssh -o ConnectTimeout=10 root@$COORDINATOR 'ls -d /workspace/ringrift /root/ringrift 2>/dev/null | head -1')

    # Check if already running
    if ssh root@$COORDINATOR "pgrep -f 'unified_ai_loop.py'" >/dev/null 2>&1; then
        log_info "  Unified AI Loop already running on $COORDINATOR"
        return
    fi

    ssh root@$COORDINATOR "cd $remote_path/ai-service && mkdir -p logs/unified_loop data/coordination && nohup python3 scripts/unified_ai_loop.py --start --verbose > logs/unified_loop/startup.log 2>&1 &"
    sleep 2

    if ssh root@$COORDINATOR "pgrep -f 'unified_ai_loop.py'" >/dev/null 2>&1; then
        log_info "  Unified AI Loop started successfully"
    else
        log_error "  Failed to start Unified AI Loop"
    fi
}

# Start job-state sync daemon on coordinator
start_job_state_sync() {
    log_info "Starting Job State Sync Daemon on $COORDINATOR..."

    if $DRY_RUN; then
        echo "  ssh root@$COORDINATOR 'cd .../ai-service && python3 scripts/job_state_sync_daemon.py --interval 60'"
        return
    fi

    local remote_path=$(ssh -o ConnectTimeout=10 root@$COORDINATOR 'ls -d /workspace/ringrift /root/ringrift 2>/dev/null | head -1')

    if ssh root@$COORDINATOR "pgrep -f 'job_state_sync_daemon.py'" >/dev/null 2>&1; then
        log_info "  Job State Sync Daemon already running on $COORDINATOR"
        return
    fi

    ssh root@$COORDINATOR "cd $remote_path/ai-service && mkdir -p logs && nohup python3 scripts/job_state_sync_daemon.py --interval 60 > logs/job_state_sync_daemon.log 2>&1 &"
    sleep 2

    if ssh root@$COORDINATOR "pgrep -f 'job_state_sync_daemon.py'" >/dev/null 2>&1; then
        log_info "  Job State Sync Daemon started successfully"
    else
        log_warn "  Job State Sync Daemon may not have started - check logs"
    fi
}

# Start data sync service
start_data_sync() {
    log_info "Starting Data Sync Service on $DATA_AGGREGATOR..."

    if $DRY_RUN; then
        echo "  ssh root@$DATA_AGGREGATOR 'cd .../ai-service && python3 scripts/unified_data_sync.py --watchdog'"
        return
    fi

    local remote_path=$(ssh -o ConnectTimeout=10 root@$DATA_AGGREGATOR 'ls -d /workspace/ringrift /root/ringrift 2>/dev/null | head -1')

    # Check if already running
    if ssh root@$DATA_AGGREGATOR "pgrep -f 'unified_data_sync.py'" >/dev/null 2>&1; then
        log_info "  Data Sync Service already running on $DATA_AGGREGATOR"
        return
    fi

    ssh root@$DATA_AGGREGATOR "cd $remote_path/ai-service && mkdir -p logs data/games && nohup python3 scripts/unified_data_sync.py --watchdog > logs/unified_data_sync.log 2>&1 &"
    sleep 2

    if ssh root@$DATA_AGGREGATOR "pgrep -f 'unified_data_sync.py'" >/dev/null 2>&1; then
        log_info "  Data Sync Service started successfully"
    else
        log_warn "  Data Sync Service may not have started - check logs"
    fi
}

# Start balanced selfplay across GPU nodes
start_selfplay() {
    log_info "Starting balanced selfplay across GPU nodes..."

    local config_idx=0
    for node in "${GPU_NODES[@]}"; do
        # Skip coordinator for selfplay (it runs the loop)
        # Assign configs round-robin
        local config="${SELFPLAY_CONFIGS[$config_idx]}"
        IFS=':' read -r board players max_moves <<< "$config"

        if $DRY_RUN; then
            echo "  ssh root@$node '... run_gpu_selfplay.py --board $board --num-players $players --max-moves $max_moves'"
            config_idx=$(( (config_idx + 1) % ${#SELFPLAY_CONFIGS[@]} ))
            continue
        fi

        local remote_path=$(ssh -o ConnectTimeout=10 root@$node 'ls -d /workspace/ringrift /root/ringrift /workspace/RingRift /root/RingRift 2>/dev/null | head -1' 2>/dev/null || echo "")

        if [[ -z "$remote_path" ]]; then
            log_warn "  Skipping $node - ringrift directory not found"
            continue
        fi

        # Check if selfplay already running for this config
        if ssh root@$node "pgrep -f 'run_gpu_selfplay.*--board $board.*--num-players $players'" >/dev/null 2>&1; then
            log_info "  Selfplay $board ${players}p already running on $node"
        else
            local logfile="logs/selfplay_${board}_${players}p.log"
            ssh root@$node "cd $remote_path/ai-service && mkdir -p logs data/games && nohup python3 scripts/run_gpu_selfplay.py --board $board --num-players $players --engine-mode nnue-guided --num-games 50000 --batch-size 64 --output-dir data/games > $logfile 2>&1 &" 2>/dev/null || true
            log_info "  Started $board ${players}p selfplay on $node"
        fi

        config_idx=$(( (config_idx + 1) % ${#SELFPLAY_CONFIGS[@]} ))
    done
}

# Show cluster status
show_status() {
    log_info "Cluster Status:"
    echo ""
    echo "=== Unified AI Loop ==="
    ssh -o ConnectTimeout=5 root@$COORDINATOR 'ps aux | grep unified_ai_loop | grep -v grep | head -1' 2>/dev/null || echo "  Not running"
    echo ""
    echo "=== Job State Sync ==="
    ssh -o ConnectTimeout=5 root@$COORDINATOR 'ps aux | grep job_state_sync_daemon | grep -v grep | head -1' 2>/dev/null || echo "  Not running"
    echo ""
    echo "=== Data Sync Service ==="
    ssh -o ConnectTimeout=5 root@$DATA_AGGREGATOR 'ps aux | grep unified_data_sync | grep -v grep | head -1' 2>/dev/null || echo "  Not running"
    echo ""
    echo "=== Selfplay Processes ==="
    for node in "${GPU_NODES[@]}"; do
        local count=$(ssh -o ConnectTimeout=5 root@$node 'pgrep -f "run_gpu_selfplay|run_hybrid" | wc -l' 2>/dev/null || echo 0)
        echo "  $node: $count processes"
    done
}

# Main
main() {
    log_info "RingRift AI Cluster Startup"
    echo ""

    check_tailscale
    sync_configs
    start_unified_loop
    start_job_state_sync
    start_data_sync
    start_selfplay

    echo ""
    show_status

    echo ""
    log_info "Cluster startup complete!"
    log_info "Monitor with: tail -f logs/unified_loop/startup.log"
}

main "$@"
