#!/bin/bash
# RingRift Cluster Alert Script
# Sends alerts to Slack when cluster issues are detected
#
# Usage:
#   ./scripts/cluster_alert.sh                    # Run checks and alert if issues
#   ./scripts/cluster_alert.sh --test             # Send test message
#   ./scripts/cluster_alert.sh --cron             # Run as cron job (silent unless issues)
#
# Environment:
#   SLACK_WEBHOOK_URL - Your Slack webhook URL (required)
#   Or set it in ~/.ringrift_slack_webhook
#
# Cron example (check every 15 minutes):
#   */15 * * * * /path/to/ringrift/ai-service/scripts/cluster_alert.sh --cron

set -e

# Load webhook URL from env or file
if [[ -z "$SLACK_WEBHOOK_URL" ]]; then
    if [[ -f ~/.ringrift_slack_webhook ]]; then
        SLACK_WEBHOOK_URL=$(cat ~/.ringrift_slack_webhook)
    else
        echo "Error: SLACK_WEBHOOK_URL not set and ~/.ringrift_slack_webhook not found"
        echo ""
        echo "To set up:"
        echo "  1. Create a Slack webhook at https://api.slack.com/apps"
        echo "  2. Either export SLACK_WEBHOOK_URL=<your-webhook-url>"
        echo "  3. Or save it to ~/.ringrift_slack_webhook"
        exit 1
    fi
fi

# Node configurations
COORDINATOR="vast-rtx4060ti"
DATA_AGGREGATOR="vast-512cpu"
GPU_NODES=("vast-4080s-2x" "vast-rtx4060ti" "vast-2080ti" "vast-3070-24cpu" "vast-2060s-22cpu" "vast-3060ti-64cpu" "vast-5070-4x")

# Thresholds
MIN_SELFPLAY_PROCESSES=3
MIN_TOTAL_PROCESSES=10

CRON_MODE=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cron|-c) CRON_MODE=true; shift ;;
        --test|-t) TEST_MODE=true; shift ;;
        *) shift ;;
    esac
done

send_slack_alert() {
    local severity="$1"  # warning, critical, info
    local title="$2"
    local message="$3"

    local color="#36a64f"  # green for info
    if [[ "$severity" == "warning" ]]; then color="#ff9900"; fi
    if [[ "$severity" == "critical" ]]; then color="#ff0000"; fi

    local emoji=":white_check_mark:"
    if [[ "$severity" == "warning" ]]; then emoji=":warning:"; fi
    if [[ "$severity" == "critical" ]]; then emoji=":rotating_light:"; fi

    local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "$emoji RingRift Cluster Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": "*Severity:*\n${severity^}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Time:*\n$(date '+%Y-%m-%d %H:%M:%S')"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Issue:* $title\n\n$message"
                    }
                }
            ]
        }
    ]
}
EOF
)

    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" > /dev/null
}

# Test mode - send a test message
if $TEST_MODE; then
    echo "Sending test alert to Slack..."
    send_slack_alert "info" "Test Alert" "This is a test message from the RingRift cluster alert system. If you see this, alerts are working correctly!"
    echo "Test alert sent!"
    exit 0
fi

# Run cluster health checks
issues=()
severity="info"

# Check 1: Unified AI Loop
loop_running=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$COORDINATOR 'pgrep -f unified_ai_loop' 2>/dev/null || echo "")
if [[ -z "$loop_running" ]]; then
    issues+=("• Unified AI Loop is DOWN on $COORDINATOR")
    severity="critical"
fi

# Check 2: Data Sync Service
sync_procs=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$DATA_AGGREGATOR 'pgrep -f unified_data_sync | wc -l' 2>/dev/null || echo "0")
if [[ "$sync_procs" -eq 0 ]]; then
    issues+=("• Data Sync Service is DOWN on $DATA_AGGREGATOR")
    if [[ "$severity" != "critical" ]]; then severity="warning"; fi
fi

# Check 3: Selfplay processes
total_procs=0
low_nodes=()
for node in "${GPU_NODES[@]}"; do
    procs=$(ssh -o ConnectTimeout=5 -o BatchMode=yes root@$node 'pgrep -f "run_gpu_selfplay\|run_hybrid" | wc -l' 2>/dev/null || echo "0")
    total_procs=$((total_procs + procs))
    if [[ "$procs" -lt "$MIN_SELFPLAY_PROCESSES" ]]; then
        low_nodes+=("$node ($procs)")
    fi
done

if [[ "$total_procs" -lt "$MIN_TOTAL_PROCESSES" ]]; then
    issues+=("• Total selfplay processes ($total_procs) below threshold ($MIN_TOTAL_PROCESSES)")
    if [[ "$severity" != "critical" ]]; then severity="warning"; fi
fi

if [[ ${#low_nodes[@]} -gt 0 ]]; then
    issues+=("• Low/no selfplay on: ${low_nodes[*]}")
fi

# Check 4: Tailscale connectivity
ts_connected=$(tailscale status 2>/dev/null | grep -c "vast-.*active\|vast-.*idle" || echo "0")
ts_total=$(tailscale status 2>/dev/null | grep -c "vast-" || echo "0")
if [[ "$ts_connected" -lt "$ts_total" && "$ts_total" -gt 0 ]]; then
    issues+=("• Tailscale: Only $ts_connected/$ts_total nodes connected")
    if [[ "$severity" != "critical" ]]; then severity="warning"; fi
fi

# Send alert if there are issues
if [[ ${#issues[@]} -gt 0 ]]; then
    issue_text=$(printf '%s\n' "${issues[@]}")
    send_slack_alert "$severity" "Cluster Health Issues Detected" "$issue_text"

    if ! $CRON_MODE; then
        echo "Alert sent to Slack ($severity):"
        echo "$issue_text"
    fi
    exit 1
else
    if ! $CRON_MODE; then
        echo "✓ Cluster healthy - no alerts sent"
        echo "  - Unified Loop: running"
        echo "  - Data Sync: $sync_procs processes"
        echo "  - Selfplay: $total_procs total processes"
        echo "  - Tailscale: $ts_connected/$ts_total nodes"
    fi
    exit 0
fi
