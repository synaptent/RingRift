#!/bin/bash
# ============================================================================
# RingRift Cluster Monitoring - Comprehensive with Auto-Recovery
# ============================================================================
#
# Features:
#   - Tailscale mesh network status
#   - Vast.ai CLI instance monitoring
#   - Lambda node health (H100, GH200 cluster)
#   - P2P orchestrator status
#   - Selfplay and gauntlet monitoring
#   - Automatic recovery of failed processes
#
# Usage:
#   ./cluster_monitoring.sh              # Single status check
#   ./cluster_monitoring.sh --loop       # Continuous monitoring (30s interval)
#   ./cluster_monitoring.sh --loop 60    # Custom interval in seconds
#
# ============================================================================

set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$AI_SERVICE_DIR/logs"
LOG_FILE="$LOG_DIR/cluster_monitor_$(date +%Y%m%d).log"

# Node IPs (Tailscale)
H100_IP="100.78.101.123"
GH200_NODES=(
    "100.88.176.74:GH200-e"
    "100.88.35.19:GH200-c"
    "100.75.84.47:GH200-d"
    "100.104.165.116:GH200-f"
    "100.104.126.58:GH200-g"
    "100.65.88.62:GH200-h"
    "100.99.27.56:GH200-i"
    "100.96.142.42:GH200-k"
    "100.76.145.60:GH200-l"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

header() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}    ${BLUE}🔄 RINGRIFT CLUSTER MONITOR${NC} - $(date '+%Y-%m-%d %H:%M:%S')                              ${CYAN}║${NC}"
    if [ -n "$1" ]; then
        echo -e "${CYAN}║${NC}    Iteration: ${GREEN}$1${NC} | Interval: ${2:-30}s                                                   ${CYAN}║${NC}"
    fi
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝${NC}"
}

# ============================================================================
# Tailscale Status
# ============================================================================
check_tailscale() {
    echo -e "\n${MAGENTA}═══ 🌐 TAILSCALE NETWORK ═══${NC}"

    if ! command -v tailscale &>/dev/null; then
        echo -e "  ${RED}✗${NC} Tailscale CLI not found"
        return 1
    fi

    local ts_status
    ts_status=$(tailscale status 2>/dev/null)

    # Count nodes
    local total=$(echo "$ts_status" | grep -cE "^100\." || echo "0")
    local online=$(echo "$ts_status" | grep -c "active" || echo "0")
    local lambda_online=$(echo "$ts_status" | grep -i "lambda" | grep -c "active" || echo "0")
    local vast_online=$(echo "$ts_status" | grep -i "vast" | grep -c "active" || echo "0")

    echo -e "  ${CYAN}Total nodes:${NC} ${GREEN}$online${NC}/$total online"
    echo -e "  ${CYAN}Lambda nodes:${NC} ${GREEN}$lambda_online${NC} active"
    echo -e "  ${CYAN}Vast nodes:${NC} ${GREEN}$vast_online${NC} active"

    # Show offline nodes
    local offline
    offline=$(echo "$ts_status" | grep "offline" | head -5)
    if [ -n "$offline" ]; then
        echo -e "  ${YELLOW}Recently offline:${NC}"
        echo "$offline" | while read line; do
            local name=$(echo "$line" | awk '{print $2}')
            echo -e "    ${RED}○${NC} $name"
        done
    fi
}

# ============================================================================
# Vast.ai CLI Status
# ============================================================================
check_vast_cli() {
    echo -e "\n${MAGENTA}═══ 🖥️  VAST.AI INSTANCES ═══${NC}"

    if ! command -v vastai &>/dev/null; then
        echo -e "  ${RED}✗${NC} Vast CLI not found"
        return 1
    fi

    local vast_json
    vast_json=$(vastai show instances --raw 2>/dev/null)

    if [ -z "$vast_json" ] || [ "$vast_json" = "[]" ]; then
        echo -e "  ${YELLOW}No instances or API error${NC}"
        return 1
    fi

    # Parse with Python
    echo "$vast_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
running = [i for i in data if i.get('actual_status') == 'running']
total_gpus = sum(i.get('num_gpus', 0) or 0 for i in running)
total_cost = sum(i.get('dph_total', 0) or 0 for i in running)

print(f'  Running: \033[0;32m{len(running)}\033[0m/{len(data)} | GPUs: \033[0;32m{total_gpus}\033[0m | Cost: \${total_cost:.2f}/hr')
print()
print(f'  \033[0;36m{\"ID\":<10} {\"GPU\":<14} {\"Qty\":>3} {\"\$/hr\":>7} {\"GPU%\":>5} {\"Disk%\":>6}\033[0m')
print(f'  {\"─\"*50}')

for inst in sorted(running, key=lambda x: -(x.get('num_gpus') or 0))[:10]:
    gpu = inst.get('gpu_name', '?')[:12]
    num = inst.get('num_gpus', 0) or 0
    cost = inst.get('dph_total', 0) or 0
    util = inst.get('gpu_util', 0) or 0
    disk = inst.get('disk_usage', 0) or 0
    iid = inst.get('id', 0)

    util_color = '\033[0;32m' if util > 50 else '\033[1;33m' if util > 10 else '\033[0;31m'
    print(f'  {iid:<10} {gpu:<14} {num:>3} \${cost:>6.3f} {util_color}{util:>4.0f}%\033[0m {disk:>5.0f}%')
" 2>/dev/null || echo -e "  ${RED}Failed to parse Vast data${NC}"
}

# ============================================================================
# Lambda Node Health
# ============================================================================
check_lambda_node() {
    local ip=$1
    local name=$2

    local result
    result=$(timeout 15 ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no ubuntu@$ip "
        procs=\$(ps aux | grep -c '[p]ython')
        selfplay=\$(ps aux | grep -E 'selfplay|hybrid' | grep -v grep | wc -l)
        hex_sp=\$(ps aux | grep hexagonal | grep -v grep | wc -l)
        gauntlet=\$(ps aux | grep gauntlet | grep -v grep | wc -l)
        training=\$(ps aux | grep -E 'train.*\.py' | grep -v grep | wc -l)
        disk=\$(df -h / | tail -1 | awk '{print \$5}' | tr -d '%')
        gpu_util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '0')
        load=\$(uptime | awk -F'load average:' '{print \$2}' | awk '{print \$1}' | tr -d ',')
        echo \"\$procs|\$selfplay|\$hex_sp|\$gauntlet|\$training|\$disk|\$gpu_util|\$load\"
    " 2>/dev/null)

    if [ -z "$result" ]; then
        echo -e "    ${RED}✗${NC} $name: ${RED}UNREACHABLE${NC}"
        return 1
    fi

    IFS='|' read -r procs selfplay hex_sp gauntlet training disk gpu_util load <<< "$result"

    # Color coding
    local disk_color="${GREEN}" && [ "${disk:-0}" -gt 80 ] && disk_color="${YELLOW}" && [ "${disk:-0}" -gt 90 ] && disk_color="${RED}"
    local selfplay_color="${GREEN}" && [ "${selfplay:-0}" -lt 5 ] && selfplay_color="${YELLOW}" && [ "${selfplay:-0}" -eq 0 ] && selfplay_color="${RED}"
    local gpu_color="${GREEN}" && [ "${gpu_util:-0}" -lt 20 ] && gpu_color="${YELLOW}"

    printf "    ${GREEN}●${NC} %-10s procs=%-4s selfplay=${selfplay_color}%-3s${NC} hex=%-3s gntlt=%-2s GPU=${gpu_color}%3s%%${NC} disk=${disk_color}%3s%%${NC}\n" \
        "$name" "$procs" "$selfplay" "$hex_sp" "$gauntlet" "$gpu_util" "$disk"

    # Return data for recovery check
    echo "$selfplay"
}

check_lambda_nodes() {
    echo -e "\n${MAGENTA}═══ 🖥️  LAMBDA NODES ═══${NC}"

    # H100
    local h100_selfplay
    h100_selfplay=$(check_lambda_node "$H100_IP" "H100")

    # GH200 nodes in parallel
    for node in "${GH200_NODES[@]}"; do
        local ip="${node%%:*}"
        local name="${node##*:}"
        check_lambda_node "$ip" "$name" &
    done
    wait

    echo "$h100_selfplay"
}

# ============================================================================
# P2P Orchestrator Status
# ============================================================================
check_p2p() {
    echo -e "\n${MAGENTA}═══ 🔗 P2P ORCHESTRATOR ═══${NC}"

    local p2p_output
    p2p_output=$(cd "$AI_SERVICE_DIR" && timeout 90 python scripts/vast_p2p_manager.py status 2>&1)

    local summary
    summary=$(echo "$p2p_output" | grep "P2P Running")

    if [ -n "$summary" ]; then
        local running=$(echo "$summary" | grep -oE '[0-9]+/[0-9]+' | head -1)
        local selfplay=$(echo "$summary" | grep -oE 'Selfplay: [0-9]+' | grep -oE '[0-9]+')
        local gpus=$(echo "$summary" | grep -oE 'GPUs: [0-9]+' | grep -oE '[0-9]+')

        local run_num=$(echo "$running" | cut -d'/' -f1)
        local total_num=$(echo "$running" | cut -d'/' -f2)
        local run_color="${GREEN}"
        [ "$run_num" -lt "$((total_num * 3 / 4))" ] && run_color="${YELLOW}"
        [ "$run_num" -lt "$((total_num / 2))" ] && run_color="${RED}"

        echo -e "  P2P Active: ${run_color}${running}${NC} | GPUs: ${GREEN}$gpus${NC} | Selfplay Jobs: ${GREEN}$selfplay${NC}"
        echo "$run_num|$total_num"
    else
        echo -e "  ${RED}Failed to get P2P status${NC}"
        echo "0|0"
    fi
}

# ============================================================================
# Gauntlet Status
# ============================================================================
check_gauntlet() {
    echo -e "\n${MAGENTA}═══ 🏆 GAUNTLET STATUS ═══${NC}"

    local info
    info=$(ssh -o ConnectTimeout=8 -o BatchMode=yes ubuntu@$H100_IP "
        cd ringrift/ai-service
        procs=\$(ps aux | grep baseline_gauntlet | grep -v grep | wc -l)
        hex_last=\$(tail -1 logs/gauntlet_hex_2p.log 2>/dev/null | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' || echo 'N/A')
        sq_last=\$(tail -1 logs/gauntlet_sq19_2p.log 2>/dev/null | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' || echo 'N/A')
        echo \"\$procs|\$hex_last|\$sq_last\"
    " 2>/dev/null)

    IFS='|' read -r procs hex_time sq_time <<< "$info"

    local procs_color="${GREEN}"
    [ "${procs:-0}" -lt 10 ] && procs_color="${YELLOW}"
    [ "${procs:-0}" -eq 0 ] && procs_color="${RED}"

    echo -e "  Processes: ${procs_color}${procs:-0}${NC} | Hex-2p: ${hex_time:-N/A} | Sq19-2p: ${sq_time:-N/A}"
}

# ============================================================================
# Auto-Recovery
# ============================================================================
recover_if_needed() {
    local h100_selfplay=$1
    local vast_running=$2
    local vast_total=$3

    echo -e "\n${MAGENTA}═══ 🔧 AUTO-RECOVERY ═══${NC}"

    local actions=0

    # H100 selfplay recovery
    if [ "${h100_selfplay:-0}" -lt 3 ]; then
        echo -e "  ${YELLOW}⚡${NC} H100 selfplay low ($h100_selfplay), restarting hex selfplay..."
        ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@$H100_IP "
            cd ringrift/ai-service && source venv/bin/activate
            pkill -f 'hexagonal.*selfplay' 2>/dev/null || true
            nohup bash -c 'PYTHONPATH=. python scripts/hybrid_selfplay.py --board hexagonal --players 2 --games 500 --parallel 8 2>&1 >> logs/hex_selfplay.log' &
        " &>/dev/null
        log "ACTION: Restarted hex selfplay on H100"
        actions=$((actions + 1))
    fi

    # Vast P2P recovery
    if [ "${vast_running:-0}" -lt "$((${vast_total:-14} / 2))" ]; then
        echo -e "  ${YELLOW}⚡${NC} Vast P2P low ($vast_running/$vast_total), triggering restart..."
        cd "$AI_SERVICE_DIR" && python scripts/vast_p2p_manager.py start --parallel 5 &>/dev/null &
        log "ACTION: Triggered Vast P2P restart"
        actions=$((actions + 1))
    fi

    if [ $actions -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} All systems nominal"
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    mkdir -p "$LOG_DIR"

    local loop_mode=false
    local interval=30

    if [ "$1" = "--loop" ]; then
        loop_mode=true
        [ -n "$2" ] && interval=$2
    fi

    local iteration=1
    local h100_selfplay=0
    local vast_running=0
    local vast_total=14

    while true; do
        header "$iteration" "$interval"

        # Tailscale (every 5 iterations in loop mode)
        if [ "$loop_mode" = false ] || [ $((iteration % 5)) -eq 1 ]; then
            check_tailscale
        fi

        # Lambda nodes
        h100_selfplay=$(check_lambda_nodes | tail -1)

        # Vast CLI (every 3 iterations in loop mode)
        if [ "$loop_mode" = false ] || [ $((iteration % 3)) -eq 0 ]; then
            check_vast_cli
        fi

        # P2P status
        local p2p_data
        p2p_data=$(check_p2p)
        vast_running=$(echo "$p2p_data" | tail -1 | cut -d'|' -f1)
        vast_total=$(echo "$p2p_data" | tail -1 | cut -d'|' -f2)

        # Gauntlet
        check_gauntlet

        # Auto-recovery (every 5 iterations in loop mode)
        if [ "$loop_mode" = false ] || [ $((iteration % 5)) -eq 0 ]; then
            recover_if_needed "$h100_selfplay" "$vast_running" "$vast_total"
        fi

        echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        if [ "$loop_mode" = false ]; then
            break
        fi

        log "Iteration $iteration complete"
        echo -e "Next check in ${interval}s... (Ctrl+C to stop)"
        iteration=$((iteration + 1))
        sleep "$interval"
    done
}

main "$@"
