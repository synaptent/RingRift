#!/bin/bash
# RingRift Slack Alert Setup Script
# Configures Slack webhook alerts across all cluster nodes
#
# Usage:
#   ./scripts/setup_slack_alerts.sh <webhook_url>
#
# The webhook URL should be from a Slack Incoming Webhook integration:
# https://api.slack.com/messaging/webhooks

set -e

WEBHOOK_URL="${1:-}"

if [[ -z "$WEBHOOK_URL" ]]; then
    echo "RingRift Slack Alert Setup"
    echo "=========================="
    echo ""
    echo "Usage: $0 <slack_webhook_url>"
    echo ""
    echo "To get a Slack webhook URL:"
    echo "  1. Go to https://api.slack.com/apps"
    echo "  2. Create a new app or select existing one"
    echo "  3. Go to 'Incoming Webhooks' and enable it"
    echo "  4. Click 'Add New Webhook to Workspace'"
    echo "  5. Select the channel for alerts"
    echo "  6. Copy the webhook URL and run this script with it"
    echo ""
    echo "Example:"
    echo "  $0 'https://hooks.slack.com/services/T00/B00/xxxx'"
    exit 1
fi

# Validate webhook URL format
if [[ ! "$WEBHOOK_URL" =~ ^https://hooks\.slack\.com/ ]]; then
    echo "Warning: URL doesn't look like a Slack webhook. Continuing anyway..."
fi

echo "Setting up Slack alerts with webhook..."
echo ""

# Save to local file
echo "$WEBHOOK_URL" > ~/.ringrift_slack_webhook
echo "✓ Saved webhook to ~/.ringrift_slack_webhook"

# Export for current session
export SLACK_WEBHOOK_URL="$WEBHOOK_URL"
export RINGRIFT_SLACK_WEBHOOK="$WEBHOOK_URL"

# Add to bashrc for persistence
if ! grep -q "RINGRIFT_SLACK_WEBHOOK" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# RingRift Slack webhook" >> ~/.bashrc
    echo "export RINGRIFT_SLACK_WEBHOOK=\"$WEBHOOK_URL\"" >> ~/.bashrc
    echo "export SLACK_WEBHOOK_URL=\"$WEBHOOK_URL\"" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
fi

# Update notification_hooks.yaml
CONFIG_FILE="$(dirname "$0")/../config/notification_hooks.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
    # Enable slack and set webhook URL
    sed -i.bak "s|^  enabled: false|  enabled: true|" "$CONFIG_FILE"
    sed -i "s|^  webhook_url: ''|  webhook_url: '$WEBHOOK_URL'|" "$CONFIG_FILE"
    echo "✓ Updated config/notification_hooks.yaml"
fi

# Deploy to Lambda nodes
echo ""
echo "Deploying to Lambda nodes..."

# Load node configuration from config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../config/cluster_nodes.env"

declare -a LAMBDA_NODES=()
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    # Add H100 node if configured
    [[ -n "$H100_IP" ]] && LAMBDA_NODES+=("ubuntu@$H100_IP")
    # Add A10 node if configured
    [[ -n "$A10_IP" ]] && LAMBDA_NODES+=("ubuntu@$A10_IP")
    # Add GH200 nodes from array
    if [[ -n "${GH200_NODES+x}" ]]; then
        for node in "${GH200_NODES[@]}"; do
            ip="${node%%:*}"
            LAMBDA_NODES+=("ubuntu@$ip")
        done
    fi
else
    echo "Warning: No config/cluster_nodes.env found. No nodes to configure."
fi

for node in "${LAMBDA_NODES[@]}"; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $node "
        echo '$WEBHOOK_URL' > ~/.ringrift_slack_webhook
        if ! grep -q RINGRIFT_SLACK_WEBHOOK ~/.bashrc 2>/dev/null; then
            echo 'export RINGRIFT_SLACK_WEBHOOK=\"$WEBHOOK_URL\"' >> ~/.bashrc
            echo 'export SLACK_WEBHOOK_URL=\"$WEBHOOK_URL\"' >> ~/.bashrc
        fi
    " 2>/dev/null; then
        echo "  ✓ Configured ${node##*@}"
    else
        echo "  ✗ Failed to reach ${node##*@}"
    fi
done

# Test the webhook
echo ""
echo "Testing webhook..."
TEST_PAYLOAD=$(cat <<EOF
{
    "text": ":white_check_mark: RingRift Alert System Configured",
    "attachments": [
        {
            "color": "#36a64f",
            "fields": [
                {"title": "Status", "value": "Alerts are now active", "short": true},
                {"title": "Time", "value": "$(date '+%Y-%m-%d %H:%M:%S')", "short": true}
            ],
            "footer": "RingRift Cluster Monitoring"
        }
    ]
}
EOF
)

if curl -s -X POST -H 'Content-type: application/json' --data "$TEST_PAYLOAD" "$WEBHOOK_URL" > /dev/null; then
    echo "✓ Test message sent to Slack!"
else
    echo "✗ Failed to send test message"
fi

echo ""
echo "Setup complete! Alerts will be sent for:"
echo "  - Node failures (keepalive)"
echo "  - P2P orchestrator issues"
echo "  - Tailscale connectivity problems"
echo "  - Training pipeline events"
echo "  - Model rollbacks"
echo ""
echo "To manually test alerts:"
echo "  ./scripts/cluster_alert.sh --test"
