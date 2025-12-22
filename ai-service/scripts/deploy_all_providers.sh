#!/bin/bash
# =============================================================================
# Unified Multi-Provider Cluster Deployment
# =============================================================================
# Updates code and optionally restarts services across all cluster nodes:
#   - Lambda/GH200 nodes (via Tailscale)
#   - vast.ai instances (via vastai CLI)
#   - AWS instances (via SSH)
#
# Usage:
#   ./scripts/deploy_all_providers.sh [command] [options]
#
# Commands:
#   update          - Update code on all nodes (git fetch && git reset --hard)
#   status          - Show status of all nodes
#   restart         - Restart selfplay/p2p services on all nodes
#   all             - Update + restart (default)
#   count           - Count nodes by provider
#
# Options:
#   --providers P   - Comma-separated list: lambda,vast,aws (default: all)
#   --parallel N    - Max parallel operations (default: 10)
#   --dry-run       - Show what would be done without executing
#   --verbose       - Show detailed output
#
# December 2025 - RingRift Unified Cluster Management
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Configuration
# =============================================================================

# Lambda/GH200 nodes via Tailscale
LAMBDA_TAILSCALE_IPS=(
    "100.123.183.70"   # lambda-gh200-a
    "100.104.34.73"    # lambda-gh200-b
    "100.88.35.19"     # lambda-gh200-c
    "100.75.84.47"     # lambda-gh200-d
    "100.88.176.74"    # lambda-gh200-e
    "100.104.165.116"  # lambda-gh200-f
    "100.104.126.58"   # lambda-gh200-g
    "100.65.88.62"     # lambda-gh200-h
    "100.99.27.56"     # lambda-gh200-i
    "100.96.142.42"    # lambda-gh200-k
    "100.76.145.60"    # lambda-gh200-l
    "100.78.101.123"   # lambda-h100
)

# AWS instances (hostname or IP)
AWS_HOSTS=(
    "52.15.114.79"     # aws-staging (relay proxy)
)

# SSH configuration
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes -o LogLevel=ERROR"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_cluster}"
SSH_USER="${SSH_USER:-ubuntu}"

# Defaults
PROVIDERS="lambda,vast,aws"
MAX_PARALLEL=10
DRY_RUN=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_debug() { [[ "$VERBOSE" == "true" ]] && echo -e "${CYAN}[DEBUG]${NC} $*" || true; }

# SSH to a node
ssh_node() {
    local host="$1"
    shift
    ssh $SSH_OPTS -i "$SSH_KEY" "${SSH_USER}@${host}" "$@" 2>/dev/null
}

# Get vast.ai instances using vastai CLI
get_vast_instances() {
    local vastai_cmd=""

    # Find vastai CLI
    for path in "vastai" "$HOME/.local/bin/vastai" "/usr/local/bin/vastai"; do
        if command -v "$path" &>/dev/null; then
            vastai_cmd="$path"
            break
        fi
    done

    if [[ -z "$vastai_cmd" ]]; then
        log_debug "vastai CLI not found"
        return
    fi

    # Get running instances
    $vastai_cmd show instances --raw 2>/dev/null | \
        python3 -c "
import json, sys
try:
    instances = json.load(sys.stdin)
    for inst in instances:
        if inst.get('actual_status') == 'running':
            ssh_host = inst.get('ssh_host', '')
            ssh_port = inst.get('ssh_port', 22)
            instance_id = inst.get('id', '')
            if ssh_host:
                print(f'{instance_id}|{ssh_host}|{ssh_port}')
except:
    pass
" 2>/dev/null || true
}

# =============================================================================
# Update Functions
# =============================================================================

update_lambda_node() {
    local host="$1"
    local result=""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would update $host"
        return 0
    fi

    result=$(ssh_node "$host" '
        cd ~/ringrift/ai-service 2>/dev/null || cd ~/RingRift/ai-service 2>/dev/null || exit 1
        git fetch origin 2>&1
        git reset --hard origin/main 2>&1
        git log -1 --oneline
    ' 2>&1) || result="FAILED: connection error"

    echo "$result"
}

update_vast_instance() {
    local instance_id="$1"
    local ssh_host="$2"
    local ssh_port="$3"
    local result=""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would update vast:$instance_id ($ssh_host:$ssh_port)"
        return 0
    fi

    result=$(ssh $SSH_OPTS -p "$ssh_port" "root@$ssh_host" '
        cd /root/RingRift/ai-service 2>/dev/null || cd /root/ringrift/ai-service 2>/dev/null || exit 1
        git fetch origin 2>&1
        git reset --hard origin/main 2>&1
        git log -1 --oneline
    ' 2>&1) || result="FAILED: connection error"

    echo "$result"
}

update_aws_node() {
    local host="$1"
    local result=""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would update $host"
        return 0
    fi

    result=$(ssh_node "$host" '
        cd ~/ringrift/ai-service 2>/dev/null || cd ~/RingRift/ai-service 2>/dev/null || exit 1
        git fetch origin 2>&1
        git reset --hard origin/main 2>&1
        git log -1 --oneline
    ' 2>&1) || result="FAILED: connection error"

    echo "$result"
}

# =============================================================================
# Commands
# =============================================================================

cmd_count() {
    echo ""
    echo "=== Cluster Node Count ==="
    echo ""

    local lambda_count=${#LAMBDA_TAILSCALE_IPS[@]}
    local aws_count=${#AWS_HOSTS[@]}
    local vast_count=0

    # Count vast instances
    while IFS='|' read -r id host port; do
        [[ -n "$id" ]] && ((vast_count++))
    done < <(get_vast_instances)

    local total=$((lambda_count + aws_count + vast_count))

    printf "%-15s %5d nodes\n" "Lambda/GH200:" "$lambda_count"
    printf "%-15s %5d nodes\n" "vast.ai:" "$vast_count"
    printf "%-15s %5d nodes\n" "AWS:" "$aws_count"
    echo "-------------------------"
    printf "%-15s %5d nodes\n" "TOTAL:" "$total"
    echo ""
}

cmd_update() {
    log_info "Updating code on all cluster nodes..."
    echo ""

    local updated=0
    local failed=0
    local pids=()
    local results_dir=$(mktemp -d)

    # Lambda/GH200 nodes
    if [[ "$PROVIDERS" == *"lambda"* ]]; then
        log_info "Updating ${#LAMBDA_TAILSCALE_IPS[@]} Lambda/GH200 nodes..."
        for host in "${LAMBDA_TAILSCALE_IPS[@]}"; do
            (
                result=$(update_lambda_node "$host")
                echo "$result" > "$results_dir/lambda_$host"
            ) &
            pids+=($!)

            # Throttle parallel jobs
            if (( ${#pids[@]} >= MAX_PARALLEL )); then
                wait "${pids[0]}"
                pids=("${pids[@]:1}")
            fi
        done
    fi

    # vast.ai instances
    if [[ "$PROVIDERS" == *"vast"* ]]; then
        local vast_instances=()
        while IFS='|' read -r id host port; do
            [[ -n "$id" ]] && vast_instances+=("$id|$host|$port")
        done < <(get_vast_instances)

        if (( ${#vast_instances[@]} > 0 )); then
            log_info "Updating ${#vast_instances[@]} vast.ai instances..."
            for instance in "${vast_instances[@]}"; do
                IFS='|' read -r id host port <<< "$instance"
                (
                    result=$(update_vast_instance "$id" "$host" "$port")
                    echo "$result" > "$results_dir/vast_$id"
                ) &
                pids+=($!)

                if (( ${#pids[@]} >= MAX_PARALLEL )); then
                    wait "${pids[0]}"
                    pids=("${pids[@]:1}")
                fi
            done
        fi
    fi

    # AWS nodes
    if [[ "$PROVIDERS" == *"aws"* ]]; then
        log_info "Updating ${#AWS_HOSTS[@]} AWS nodes..."
        for host in "${AWS_HOSTS[@]}"; do
            (
                result=$(update_aws_node "$host")
                echo "$result" > "$results_dir/aws_$host"
            ) &
            pids+=($!)
        done
    fi

    # Wait for all jobs
    wait

    # Display results
    echo ""
    printf "%-12s %-20s %s\n" "PROVIDER" "NODE" "STATUS"
    printf "%-12s %-20s %s\n" "--------" "----" "------"

    for result_file in "$results_dir"/*; do
        [[ -f "$result_file" ]] || continue
        local filename=$(basename "$result_file")
        local provider="${filename%%_*}"
        local node="${filename#*_}"
        local status=$(cat "$result_file" | tail -1)

        if [[ "$status" == *"FAILED"* ]]; then
            printf "${RED}%-12s %-20s %s${NC}\n" "$provider" "$node" "$status"
            ((failed++))
        else
            printf "${GREEN}%-12s %-20s %s${NC}\n" "$provider" "$node" "${status:0:50}"
            ((updated++))
        fi
    done

    rm -rf "$results_dir"

    echo ""
    log_info "Update complete: $updated succeeded, $failed failed"
}

cmd_status() {
    log_info "Checking status of all cluster nodes..."
    echo ""

    printf "%-12s %-20s %-10s %s\n" "PROVIDER" "NODE" "STATUS" "DETAILS"
    printf "%-12s %-20s %-10s %s\n" "--------" "----" "------" "-------"

    # Lambda/GH200 nodes
    if [[ "$PROVIDERS" == *"lambda"* ]]; then
        for host in "${LAMBDA_TAILSCALE_IPS[@]}"; do
            if ping -c1 -W2 "$host" &>/dev/null; then
                details=$(ssh_node "$host" 'pgrep -f "python.*selfplay" >/dev/null && echo "selfplay running" || echo "idle"' 2>/dev/null || echo "ssh failed")
                printf "${GREEN}%-12s %-20s %-10s %s${NC}\n" "lambda" "$host" "online" "$details"
            else
                printf "${RED}%-12s %-20s %-10s %s${NC}\n" "lambda" "$host" "offline" "-"
            fi
        done
    fi

    # vast.ai instances
    if [[ "$PROVIDERS" == *"vast"* ]]; then
        while IFS='|' read -r id host port; do
            [[ -n "$id" ]] || continue
            details=$(ssh $SSH_OPTS -p "$port" "root@$host" 'pgrep -f "python.*selfplay" >/dev/null && echo "selfplay running" || echo "idle"' 2>/dev/null || echo "ssh failed")
            if [[ "$details" != *"failed"* ]]; then
                printf "${GREEN}%-12s %-20s %-10s %s${NC}\n" "vast" "id:$id" "running" "$details"
            else
                printf "${YELLOW}%-12s %-20s %-10s %s${NC}\n" "vast" "id:$id" "unreachable" "-"
            fi
        done < <(get_vast_instances)
    fi

    # AWS nodes
    if [[ "$PROVIDERS" == *"aws"* ]]; then
        for host in "${AWS_HOSTS[@]}"; do
            if ping -c1 -W2 "$host" &>/dev/null; then
                details=$(ssh_node "$host" 'pgrep -f "python.*worker" >/dev/null && echo "worker running" || echo "idle"' 2>/dev/null || echo "ssh failed")
                printf "${GREEN}%-12s %-20s %-10s %s${NC}\n" "aws" "$host" "online" "$details"
            else
                printf "${RED}%-12s %-20s %-10s %s${NC}\n" "aws" "$host" "offline" "-"
            fi
        done
    fi

    echo ""
}

cmd_restart() {
    log_info "Restarting services on all cluster nodes..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would restart services on all nodes"
        return 0
    fi

    # Lambda/GH200 - restart p2p orchestrator
    if [[ "$PROVIDERS" == *"lambda"* ]]; then
        for host in "${LAMBDA_TAILSCALE_IPS[@]}"; do
            (
                ssh_node "$host" '
                    pkill -f "p2p_orchestrator" 2>/dev/null || true
                    cd ~/ringrift/ai-service 2>/dev/null || cd ~/RingRift/ai-service
                    source venv/bin/activate 2>/dev/null || true
                    nohup PYTHONPATH=. python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &
                ' &>/dev/null
                echo "Restarted: $host"
            ) &
        done
    fi

    wait
    log_info "Services restarted"
}

cmd_all() {
    cmd_update
    echo ""
    cmd_restart
}

# =============================================================================
# Main
# =============================================================================

main() {
    local cmd="${1:-all}"
    shift || true

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --providers)   PROVIDERS="$2"; shift 2 ;;
            --parallel)    MAX_PARALLEL="$2"; shift 2 ;;
            --dry-run)     DRY_RUN=true; shift ;;
            --verbose|-v)  VERBOSE=true; shift ;;
            *)             shift ;;
        esac
    done

    echo ""
    echo "============================================================"
    echo "RingRift Multi-Provider Cluster Deployment"
    echo "============================================================"
    echo "Providers: $PROVIDERS"
    echo "Parallel jobs: $MAX_PARALLEL"
    [[ "$DRY_RUN" == "true" ]] && echo "Mode: DRY RUN"
    echo ""

    case "$cmd" in
        update)   cmd_update ;;
        status)   cmd_status ;;
        restart)  cmd_restart ;;
        count)    cmd_count ;;
        all)      cmd_all ;;
        *)
            echo "Usage: $0 [update|status|restart|count|all] [options]"
            echo ""
            echo "Commands:"
            echo "  update    - Update code on all nodes"
            echo "  status    - Show status of all nodes"
            echo "  restart   - Restart services on all nodes"
            echo "  count     - Count nodes by provider"
            echo "  all       - Update + restart (default)"
            echo ""
            echo "Options:"
            echo "  --providers P   - Providers: lambda,vast,aws (default: all)"
            echo "  --parallel N    - Max parallel operations (default: 10)"
            echo "  --dry-run       - Show without executing"
            echo "  --verbose       - Show detailed output"
            exit 1
            ;;
    esac
}

main "$@"
