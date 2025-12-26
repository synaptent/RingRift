# Slack Webhook Setup for RingRift Cluster Alerting

This guide explains how to set up Slack webhook alerting for the RingRift AI training cluster.

## Overview

The RingRift cluster uses Slack webhooks to send alerts for:

- Node offline/online status changes
- Training job failures
- Resource exhaustion warnings
- Leader election changes
- Selfplay job completion

## Setup Steps

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name it "RingRift Alerts" and select your workspace
4. Click "Create App"

### 2. Enable Incoming Webhooks

1. In the app settings, go to "Incoming Webhooks"
2. Toggle "Activate Incoming Webhooks" to ON
3. Click "Add New Webhook to Workspace"
4. Select the channel for alerts (e.g., `#ringrift-alerts`)
5. Click "Allow"
6. Copy the Webhook URL (starts with `https://hooks.slack.com/services/...`)

### 3. Configure Environment Variable

Set the `RINGRIFT_SLACK_WEBHOOK` environment variable on each node:

```bash
# Add to ~/.bashrc or node startup script
export RINGRIFT_SLACK_WEBHOOK="https://hooks.slack.com/services/T.../B.../xxx..."
```

For systemd services, add to the service file:

```ini
[Service]
Environment="RINGRIFT_SLACK_WEBHOOK=https://hooks.slack.com/services/T.../B.../xxx..."
```

### 4. Verify Setup

Test the webhook connection:

```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"RingRift cluster alerting configured!"}' \
  "$RINGRIFT_SLACK_WEBHOOK"
```

### 5. Deploy to Cluster

To deploy to all cluster nodes:

```bash
# From coordinator (edit host list as needed)
for host in host1 host2 host3; do
  ssh ubuntu@$host 'echo "export RINGRIFT_SLACK_WEBHOOK=YOUR_WEBHOOK_URL" >> ~/.bashrc'
done
```

Or manually on each node:

```bash
ssh ubuntu@<node-ip> 'echo "export RINGRIFT_SLACK_WEBHOOK=your_url" >> ~/.bashrc'
```

## Alert Configuration

### Severity Levels

| Level    | Description      | Slack Color |
| -------- | ---------------- | ----------- |
| DEBUG    | Debug info       | Gray        |
| INFO     | Informational    | Green       |
| WARNING  | Needs attention  | Orange      |
| ERROR    | Action required  | Red         |
| CRITICAL | Immediate action | Dark Red    |

### Rate Limiting

- Same alert suppressed for 30 minutes after first occurrence
- Maximum 20 alerts per hour per alert type
- Partition-aware: Suppresses if >50% of nodes have the same issue

### Alert Types

- `node_offline` - Node unreachable
- `node_online` - Node recovered
- `training_failed` - Training job failed
- `selfplay_complete` - Selfplay batch finished
- `leader_election` - New leader elected
- `resource_exhaustion` - GPU/CPU/memory low
- `model_promoted` - New model promoted to production

## Optional: Discord Integration

You can also configure Discord webhooks:

```bash
export RINGRIFT_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

## Optional: PagerDuty Integration

For critical alerts that require on-call response:

```bash
export PAGERDUTY_ROUTING_KEY="your-pagerduty-routing-key"
```

## Troubleshooting

### No alerts received

1. Verify environment variable is set: `echo $RINGRIFT_SLACK_WEBHOOK`
2. Check webhook URL is valid (should start with `https://hooks.slack.com/services/`)
3. Test with curl command above
4. Check process logs for "Slack alert sent" or error messages

### Too many alerts

Adjust rate limiting in `app/monitoring/alert_router.py`:

- `MIN_ALERT_INTERVAL`: Time between duplicate alerts (default: 1800s)
- `MAX_ALERTS_PER_HOUR`: Maximum alerts per hour (default: 20)

## Files

- `app/monitoring/alert_router.py` - Alert routing and Slack integration
- `app/coordination/unified_node_health_daemon.py` - Node health monitoring
- `scripts/universal_keepalive.py` - Node keepalive with alerting
