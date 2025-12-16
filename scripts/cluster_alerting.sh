#!/bin/bash
# Cluster Alerting Script - Sends alerts to Discord/Slack webhooks
# Usage: ./cluster_alerting.sh [--test]
# Configure webhook URL in /home/ubuntu/ringrift/.env as ALERT_WEBHOOK_URL

set -e

# Load configuration
ENV_FILE="/home/ubuntu/ringrift/.env"
STATE_FILE="/tmp/cluster_alert_state.json"
ALERT_COOLDOWN=300  # 5 minutes between repeated alerts

# Load webhook URL from .env
if [ -f "$ENV_FILE" ]; then
    export $(grep -E "^ALERT_WEBHOOK_URL=" "$ENV_FILE" | xargs)
fi

WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

if [ -z "$WEBHOOK_URL" ]; then
    echo "WARNING: No ALERT_WEBHOOK_URL configured in $ENV_FILE"
    echo "Add: ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/... or Slack webhook"
    exit 0
fi

# Detect webhook type (Discord vs Slack)
if [[ "$WEBHOOK_URL" == *"discord.com"* ]]; then
    WEBHOOK_TYPE="discord"
elif [[ "$WEBHOOK_URL" == *"slack.com"* ]]; then
    WEBHOOK_TYPE="slack"
else
    WEBHOOK_TYPE="generic"
fi

# Initialize state file
if [ ! -f "$STATE_FILE" ]; then
    echo "{}" > "$STATE_FILE"
fi

send_alert() {
    local title="$1"
    local message="$2"
    local severity="$3"  # info, warning, critical
    local alert_key="$4"
    
    # Check cooldown
    local last_alert=$(cat "$STATE_FILE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$alert_key', 0))" 2>/dev/null || echo 0)
    local now=$(date +%s)
    local diff=$((now - last_alert))
    
    if [ "$diff" -lt "$ALERT_COOLDOWN" ] && [ "$1" != "--test" ]; then
        echo "Skipping alert  - in cooldown ($diff < $ALERT_COOLDOWN seconds)"
        return
    fi
    
    # Update state
    python3 -c "import json; d=json.load(open('$STATE_FILE')); d['$alert_key']=$now; json.dump(d, open('$STATE_FILE','w'))"
    
    # Set color based on severity
    local color
    case "$severity" in
        critical) color=15158332 ;;  # Red
        warning)  color=16776960 ;;  # Yellow
        info)     color=3447003 ;;   # Blue
        success)  color=3066993 ;;   # Green
        *)        color=10070709 ;;  # Gray
    esac
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    if [ "$WEBHOOK_TYPE" == "discord" ]; then
        # Discord webhook format
        curl -s -H "Content-Type: application/json" -X POST "$WEBHOOK_URL" -d "{
            \"embeds\": [{
                \"title\": \"🔔 $title\",
                \"description\": \"$message\",
                \"color\": $color,
                \"timestamp\": \"$timestamp\",
                \"footer\": {\"text\": \"RingRift Cluster Alert\"}
            }]
        }"
    elif [ "$WEBHOOK_TYPE" == "slack" ]; then
        # Slack webhook format
        local emoji
        case "$severity" in
            critical) emoji="🚨" ;;
            warning)  emoji="⚠️" ;;
            info)     emoji="ℹ️" ;;
            success)  emoji="✅" ;;
            *)        emoji="📢" ;;
        esac
        curl -s -H "Content-Type: application/json" -X POST "$WEBHOOK_URL" -d "{
            \"text\": \"$emoji *$title*\n$message\"
        }"
    else
        # Generic webhook
        curl -s -H "Content-Type: application/json" -X POST "$WEBHOOK_URL" -d "{
            \"title\": \"$title\",
            \"message\": \"$message\",
            \"severity\": \"$severity\",
            \"timestamp\": \"$timestamp\"
        }"
    fi
    
    echo "Alert sent: $title ($severity)"
}

# Test mode
if [ "$1" == "--test" ]; then
    send_alert "Test Alert" "This is a test alert from RingRift cluster monitoring." "info" "test_alert"
    exit 0
fi

# Collect cluster health data
echo "Checking cluster health..."

# Get health endpoint
HEALTH=$(curl -s --connect-timeout 5 http://localhost:8770/health 2>/dev/null || echo "{}")

if [ "$HEALTH" == "{}" ]; then
    send_alert "P2P Service Down" "The P2P orchestrator on lambda-h100 is not responding!" "critical" "p2p_down"
    exit 1
fi

# Parse health data
HEALTHY=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('healthy', False))")
ACTIVE_PEERS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_peers', 0))")
TOTAL_PEERS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_peers', 0))")
DISK_PCT=$(echo "$HEALTH" | python3 -c "import sys,json; print(int(json.load(sys.stdin).get('disk_percent', 0)))")
MEM_PCT=$(echo "$HEALTH" | python3 -c "import sys,json; print(int(json.load(sys.stdin).get('memory_percent', 0)))")
SELFPLAY=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('selfplay_jobs', 0))")
LEADER=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('leader_id', 'none'))")

# Check tunnel health
DOWN_TUNNELS=""
for port in 8771 8772 8773 8774 8775 8776 8777 8778 8779 8780 8781 8782; do
    if ! nc -z -w1 localhost $port 2>/dev/null; then
        DOWN_TUNNELS="$DOWN_TUNNELS $port"
    fi
done

ALERTS_SENT=0

# Alert: P2P unhealthy
if [ "$HEALTHY" != "True" ]; then
    send_alert "Cluster Unhealthy" "P2P orchestrator reports unhealthy status.\\nActive: $ACTIVE_PEERS/$TOTAL_PEERS peers\\nDisk: $DISK_PCT%\\nMemory: $MEM_PCT%" "warning" "cluster_unhealthy"
    ALERTS_SENT=$((ALERTS_SENT + 1))
fi

# Alert: Low peer count (less than 50%)
if [ "$TOTAL_PEERS" -gt 0 ]; then
    PEER_PCT=$((ACTIVE_PEERS * 100 / TOTAL_PEERS))
    if [ "$PEER_PCT" -lt 50 ]; then
        send_alert "Low Peer Count" "Only $ACTIVE_PEERS/$TOTAL_PEERS peers active ($PEER_PCT%).\\nCheck node health and network connectivity." "critical" "low_peers"
        ALERTS_SENT=$((ALERTS_SENT + 1))
    fi
fi

# Alert: High disk usage
if [ "$DISK_PCT" -gt 85 ]; then
    send_alert "High Disk Usage" "Disk usage at $DISK_PCT% on lambda-h100.\\nConsider cleaning old game databases." "warning" "high_disk"
    ALERTS_SENT=$((ALERTS_SENT + 1))
fi

# Alert: High memory usage
if [ "$MEM_PCT" -gt 90 ]; then
    send_alert "High Memory Usage" "Memory usage at $MEM_PCT% on lambda-h100.\\nCheck for memory leaks or excessive processes." "warning" "high_memory"
    ALERTS_SENT=$((ALERTS_SENT + 1))
fi

# Alert: No leader elected
if [ "$LEADER" == "none" ] || [ "$LEADER" == "None" ] || [ "$LEADER" == "null" ]; then
    send_alert "No Leader Elected" "Cluster has no leader. Leader election may be stuck.\\nActive peers: $ACTIVE_PEERS" "warning" "no_leader"
    ALERTS_SENT=$((ALERTS_SENT + 1))
fi

# Alert: Tunnels down
if [ -n "$DOWN_TUNNELS" ]; then
    send_alert "Tunnels Down" "The following tunnel ports are down:$DOWN_TUNNELS\\nCheck Vast.ai instances or autossh processes." "warning" "tunnels_down"
    ALERTS_SENT=$((ALERTS_SENT + 1))
fi

if [ "$ALERTS_SENT" -eq 0 ]; then
    echo "✓ All checks passed - no alerts needed"
else
    echo "⚠ Sent $ALERTS_SENT alert(s)"
fi
