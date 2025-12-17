#!/bin/bash
# Tunnel Health Monitor - checks all SSH tunnel ports and reports status
# Run with: ./tunnel_health_monitor.sh [--json]

PRIMARY_HUB="127.0.0.1"
BACKUP_HUB="192.222.57.162"

# Primary hub tunnel mappings (port -> node_id)
declare -A PRIMARY_TUNNELS=(
    [8771]="vast-5070-4x"
    [8772]="lambda-gh200-h"
    [8773]="lambda-gh200-a"
    [8774]="lambda-gh200-c"
    [8775]="lambda-gh200-g"
    [8776]="lambda-gh200-i"
    [8777]="lambda-a10"
    [8778]="vast-3070"
    [8779]="vast-2060s"
    [8780]="vast-262969f8"
    [8781]="vast-112df53d"
    [8782]="vast-3060ti"
)

# Backup hub tunnel mappings
declare -A BACKUP_TUNNELS=(
    [8872]="lambda-gh200-h"
    [8873]="lambda-gh200-a"
    [8874]="lambda-gh200-c"
    [8875]="lambda-gh200-g"
    [8876]="lambda-gh200-i"
    [8877]="vast-3070"
    [8878]="vast-2060s"
)

check_tunnel() {
    local hub=$1
    local port=$2
    local node=$3
    
    if nc -z -w 2 $hub $port 2>/dev/null; then
        echo "OK"
    else
        echo "DOWN"
    fi
}

echo "=== Tunnel Health Report ==="
echo "Timestamp: $(date -Iseconds)"
echo ""
echo "=== Primary Hub (lambda-h100) ==="

total=0
healthy=0
for port in "${!PRIMARY_TUNNELS[@]}"; do
    node="${PRIMARY_TUNNELS[$port]}"
    status=$(check_tunnel $PRIMARY_HUB $port $node)
    ((total++))
    [[ $status == "OK" ]] && ((healthy++))
    printf "  Port %s (%s): %s\n" "$port" "$node" "$status"
done | sort

echo ""
echo "Primary: $healthy/$total tunnels healthy"

echo ""
echo "=== Backup Hub (lambda-gh200-e) ==="

backup_total=0
backup_healthy=0
for port in "${!BACKUP_TUNNELS[@]}"; do
    node="${BACKUP_TUNNELS[$port]}"
    # Check via SSH since we can't reach backup hub directly
    status=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@100.88.176.74 "nc -z -w 2 127.0.0.1 $port" 2>/dev/null && echo "OK" || echo "DOWN")
    ((backup_total++))
    [[ $status == "OK" ]] && ((backup_healthy++))
    printf "  Port %s (%s): %s\n" "$port" "$node" "$status"
done | sort

echo ""
echo "Backup: $backup_healthy/$backup_total tunnels healthy"
echo ""
echo "Total: $((healthy + backup_healthy))/$((total + backup_total)) tunnels operational"
