#!/bin/bash
# Tunnel Metrics - collect and display tunnel statistics

METRICS_FILE="/tmp/tunnel_metrics.json"

# Collect metrics
collect_metrics() {
    local timestamp=$(date -Iseconds)
    local primary_total=0
    local primary_up=0
    local backup_total=0
    local backup_up=0
    
    # Check primary hub tunnels
    for port in 8771 8772 8773 8774 8775 8776 8777 8778 8779 8780 8781 8782; do
        ((primary_total++))
        nc -z -w 1 127.0.0.1 $port 2>/dev/null && ((primary_up++))
    done
    
    # Check backup hub tunnels (via SSH)
    for port in 8872 8873 8874 8875 8876 8877 8878; do
        ((backup_total++))
        ssh -o ConnectTimeout=3 ubuntu@100.88.176.74 "nc -z -w 1 127.0.0.1 $port" 2>/dev/null && ((backup_up++))
    done
    
    # Get cluster health
    local cluster_health=$(curl -s --connect-timeout 3 http://127.0.0.1:8770/health 2>/dev/null)
    local active_peers=$(echo "$cluster_health" | grep -o '"active_peers":[0-9]*' | cut -d: -f2)
    local total_peers=$(echo "$cluster_health" | grep -o '"total_peers":[0-9]*' | cut -d: -f2)
    local selfplay_jobs=$(echo "$cluster_health" | grep -o '"selfplay_jobs":[0-9]*' | cut -d: -f2)
    
    # Output JSON
    cat << EOJSON
{
    "timestamp": "$timestamp",
    "primary_hub": {
        "total_tunnels": $primary_total,
        "active_tunnels": $primary_up,
        "health_pct": $(echo "scale=1; $primary_up * 100 / $primary_total" | bc)
    },
    "backup_hub": {
        "total_tunnels": $backup_total,
        "active_tunnels": $backup_up,
        "health_pct": $(echo "scale=1; $backup_up * 100 / $backup_total" | bc)
    },
    "cluster": {
        "active_peers": ${active_peers:-0},
        "total_peers": ${total_peers:-0},
        "selfplay_jobs": ${selfplay_jobs:-0}
    }
}
EOJSON
}

# Display metrics
display_metrics() {
    echo "=== Tunnel Metrics ==="
    collect_metrics | tee $METRICS_FILE
}

case ${1:-display} in
    collect)
        collect_metrics > $METRICS_FILE
        ;;
    display)
        display_metrics
        ;;
    json)
        collect_metrics
        ;;
    *)
        echo "Usage: $0 {display|collect|json}"
        ;;
esac
