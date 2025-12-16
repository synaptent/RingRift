#!/bin/bash
# RingRift AI Cluster Dashboard
# Shows real-time status of all cluster components
#
# Usage:
#   ./scripts/cluster_dashboard.sh           # One-shot status
#   ./scripts/cluster_dashboard.sh --watch   # Continuous monitoring (refresh every 30s)
#   ./scripts/cluster_dashboard.sh --json    # Output as JSON for programmatic use

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Node configurations
COORDINATOR="vast-rtx4060ti"
DATA_AGGREGATOR="vast-512cpu"
GPU_NODES=("vast-4080s-2x" "vast-rtx4060ti" "vast-2080ti" "vast-3070-24cpu" "vast-2060s-22cpu" "vast-3060ti-64cpu" "vast-5070-4x")

# Parse args
WATCH_MODE=false
JSON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --watch|-w) WATCH_MODE=true; shift ;;
        --json|-j) JSON_MODE=true; shift ;;
        *) shift ;;
    esac
done

status_icon() {
    if [[ "$1" == "ok" ]]; then echo -e "${GREEN}●${NC}"
    elif [[ "$1" == "warn" ]]; then echo -e "${YELLOW}●${NC}"
    else echo -e "${RED}●${NC}"; fi
}

get_cluster_status() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if $JSON_MODE; then
        echo "{"
        echo "  \"timestamp\": \"$timestamp\","
    else
        echo ""
        echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${CYAN}║          RINGRIFT AI CLUSTER DASHBOARD                       ║${NC}"
        echo -e "${BOLD}${CYAN}║          $timestamp                              ║${NC}"
        echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi

    # 1. Unified AI Loop Status
    local loop_status="error"
    local loop_pid=""
    loop_pid=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$COORDINATOR 'pgrep -f unified_ai_loop' 2>/dev/null || echo "")
    if [[ -n "$loop_pid" ]]; then loop_status="ok"; fi

    if $JSON_MODE; then
        echo "  \"unified_loop\": {\"status\": \"$loop_status\", \"pid\": \"$loop_pid\", \"host\": \"$COORDINATOR\"},"
    else
        echo -e "${BOLD}1. UNIFIED AI LOOP${NC}"
        echo -e "   Host: $COORDINATOR"
        echo -e "   Status: $(status_icon $loop_status) ${loop_status^^} (PID: ${loop_pid:-none})"
        echo ""
    fi

    # 2. Data Sync Status
    local sync_status="error"
    local sync_procs=""
    sync_procs=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$DATA_AGGREGATOR 'pgrep -f unified_data_sync | wc -l' 2>/dev/null || echo "0")
    if [[ "$sync_procs" -gt 0 ]]; then sync_status="ok"; fi

    if $JSON_MODE; then
        echo "  \"data_sync\": {\"status\": \"$sync_status\", \"processes\": $sync_procs, \"host\": \"$DATA_AGGREGATOR\"},"
    else
        echo -e "${BOLD}2. DATA SYNC SERVICE${NC}"
        echo -e "   Host: $DATA_AGGREGATOR"
        echo -e "   Status: $(status_icon $sync_status) ${sync_status^^} ($sync_procs processes)"
        echo ""
    fi

    # 3. Selfplay Status
    if ! $JSON_MODE; then
        echo -e "${BOLD}3. SELFPLAY PROCESSES${NC}"
        printf "   %-20s %10s %10s %10s\n" "NODE" "PROCESSES" "LOAD" "STATUS"
        echo "   ────────────────────────────────────────────────────"
    else
        echo "  \"selfplay\": ["
    fi

    local total_procs=0
    local first=true
    for node in "${GPU_NODES[@]}"; do
        local procs=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$node 'pgrep -f "run_gpu_selfplay\|run_hybrid" | wc -l' 2>/dev/null || echo "0")
        local load=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$node 'cat /proc/loadavg 2>/dev/null | cut -d" " -f1' 2>/dev/null || echo "?")

        local node_status="ok"
        if [[ "$procs" == "0" ]]; then node_status="error"
        elif [[ "${load%.*}" -gt 100 ]]; then node_status="warn"; fi

        total_procs=$((total_procs + procs))

        if $JSON_MODE; then
            if ! $first; then echo ","; fi
            echo -n "    {\"node\": \"$node\", \"processes\": $procs, \"load\": \"$load\", \"status\": \"$node_status\"}"
            first=false
        else
            printf "   %-20s %10s %10s %10s\n" "$node" "$procs" "$load" "$(status_icon $node_status)"
        fi
    done

    if $JSON_MODE; then
        echo ""
        echo "  ],"
        echo "  \"total_selfplay_processes\": $total_procs,"
    else
        echo "   ────────────────────────────────────────────────────"
        echo -e "   ${BOLD}TOTAL: $total_procs processes${NC}"
        echo ""
    fi

    # 4. Game Data Status
    local game_counts=""
    game_counts=$(ssh -o ConnectTimeout=10 -o BatchMode=yes root@$COORDINATOR 'sqlite3 /workspace/ringrift/ai-service/data/games/selfplay.db "SELECT board_type || '\''_'\'' || num_players || '\''p'\'' as c, COUNT(*) FROM games GROUP BY 1" 2>/dev/null' || echo "")

    if $JSON_MODE; then
        echo "  \"game_counts\": {"
        if [[ -n "$game_counts" ]]; then
            echo "$game_counts" | while IFS='|' read -r config count; do
                echo "    \"$config\": $count,"
            done
        else
            echo "    \"status\": \"no_data\""
        fi
        echo "  },"
    else
        echo -e "${BOLD}4. GAME DATA (on coordinator)${NC}"
        if [[ -n "$game_counts" ]]; then
            printf "   %-15s %10s\n" "CONFIG" "GAMES"
            echo "   ─────────────────────────────"
            echo "$game_counts" | while IFS='|' read -r config count; do
                printf "   %-15s %10s\n" "$config" "$count"
            done
        else
            echo "   No game data found yet"
        fi
        echo ""
    fi

    # 5. Tailscale Status
    local ts_connected=$(tailscale status 2>/dev/null | grep -c "vast-.*active\|vast-.*idle" || echo "0")
    local ts_total=$(tailscale status 2>/dev/null | grep -c "vast-" || echo "0")

    if $JSON_MODE; then
        echo "  \"tailscale\": {\"connected\": $ts_connected, \"total\": $ts_total}"
        echo "}"
    else
        echo -e "${BOLD}5. TAILSCALE CONNECTIVITY${NC}"
        if [[ "$ts_connected" -eq "$ts_total" && "$ts_total" -gt 0 ]]; then
            echo -e "   $(status_icon ok) $ts_connected/$ts_total nodes connected"
        else
            echo -e "   $(status_icon warn) $ts_connected/$ts_total nodes connected"
        fi
        echo ""
    fi

    # Summary
    if ! $JSON_MODE; then
        echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        local overall="ok"
        if [[ "$loop_status" != "ok" || "$sync_status" != "ok" ]]; then overall="error"; fi
        if [[ "$total_procs" -lt 5 ]]; then overall="warn"; fi

        if [[ "$overall" == "ok" ]]; then
            echo -e "   ${GREEN}${BOLD}CLUSTER HEALTHY${NC} - All systems operational"
        elif [[ "$overall" == "warn" ]]; then
            echo -e "   ${YELLOW}${BOLD}CLUSTER DEGRADED${NC} - Some components need attention"
        else
            echo -e "   ${RED}${BOLD}CLUSTER UNHEALTHY${NC} - Critical components down"
        fi
        echo ""
    fi
}

# Main
if $WATCH_MODE; then
    while true; do
        clear
        get_cluster_status
        echo -e "${CYAN}Refreshing in 30 seconds... (Ctrl+C to exit)${NC}"
        sleep 30
    done
else
    get_cluster_status
fi
