#!/bin/bash
# Cluster Health Check Script
# Monitors P2P network health and sends alerts for critical issues

set -e

# Configuration
COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8770}"  # Set to your coordinator IP
ALERT_EMAIL="${ALERT_EMAIL:-}"  # Set to receive email alerts
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"  # Set for Slack alerts
LOG_FILE="${LOG_FILE:-/tmp/cluster_health.log}"
DISK_THRESHOLD=85  # Alert when disk > 85%
VOTER_THRESHOLD=6  # Alert when voters < 6

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    local level="$1"
    local message="$2"

    log "[$level] $message"

    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"[$level] RingRift Cluster: $message\"}" \
            "$SLACK_WEBHOOK" >/dev/null 2>&1 || true
    fi
}

check_leader() {
    local status=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/status" 2>/dev/null)
    if [ -z "$status" ]; then
        alert "CRITICAL" "Cannot reach coordinator at $COORDINATOR_URL"
        return 1
    fi

    local leader=$(echo "$status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('leader_id', 'None'))" 2>/dev/null)
    if [ "$leader" = "None" ] || [ -z "$leader" ]; then
        alert "WARNING" "No leader elected in P2P cluster"
        return 1
    fi

    echo -e "${GREEN}Leader: $leader${NC}"
    return 0
}

check_voters() {
    local status=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/status" 2>/dev/null)
    if [ -z "$status" ]; then
        return 1
    fi

    local voters_alive=$(echo "$status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('voters_alive', 0))" 2>/dev/null)
    local voters_total=$(echo "$status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('voter_node_ids', [])))" 2>/dev/null)
    local quorum_ok=$(echo "$status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('voter_quorum_ok', False))" 2>/dev/null)

    if [ "$quorum_ok" != "True" ]; then
        alert "CRITICAL" "Voter quorum FAILED: $voters_alive/$voters_total voters alive"
        return 1
    fi

    if [ "$voters_alive" -lt "$VOTER_THRESHOLD" ]; then
        alert "WARNING" "Low voter count: $voters_alive/$voters_total (threshold: $VOTER_THRESHOLD)"
    fi

    echo -e "${GREEN}Voters: $voters_alive/$voters_total alive, quorum OK${NC}"
    return 0
}

check_disk_usage() {
    local status=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/status" 2>/dev/null)
    if [ -z "$status" ]; then
        return 1
    fi

    local high_disk_nodes=$(echo "$status" | python3 -c "
import json,sys
d = json.load(sys.stdin)
high_disk = []
for pid, p in d.get('peers', {}).items():
    disk = p.get('disk_percent', 0)
    if disk > $DISK_THRESHOLD:
        high_disk.append(f'{pid}: {disk:.1f}%')
if high_disk:
    print('; '.join(high_disk))
" 2>/dev/null)

    if [ -n "$high_disk_nodes" ]; then
        alert "WARNING" "High disk usage: $high_disk_nodes"
        echo -e "${YELLOW}High disk nodes: $high_disk_nodes${NC}"
    else
        echo -e "${GREEN}Disk usage: All nodes below ${DISK_THRESHOLD}%${NC}"
    fi
}

check_peer_count() {
    local health=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/health" 2>/dev/null)
    if [ -z "$health" ]; then
        return 1
    fi

    local active_peers=$(echo "$health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('active_peers', 0))" 2>/dev/null)

    if [ "$active_peers" -lt 5 ]; then
        alert "WARNING" "Low peer count: $active_peers active peers"
    fi

    echo -e "${GREEN}Active peers: $active_peers${NC}"
}

# Main
echo "========================================"
echo "RingRift Cluster Health Check"
echo "$(date)"
echo "========================================"
echo ""

# Run checks
echo "Checking leader election..."
check_leader

echo ""
echo "Checking voter quorum..."
check_voters

echo ""
echo "Checking disk usage..."
check_disk_usage

echo ""
echo "Checking peer connectivity..."
check_peer_count

echo ""
echo "========================================"
log "Health check completed"
