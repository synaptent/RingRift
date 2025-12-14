#!/bin/bash
# Install RingRift Grafana dashboard
# Usage: ./install-grafana-dashboard.sh [grafana_url] [api_key]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_FILE="${SCRIPT_DIR}/grafana/unified-ai-loop-dashboard.json"

GRAFANA_URL="${1:-http://localhost:3000}"
API_KEY="${2:-}"

if [[ ! -f "$DASHBOARD_FILE" ]]; then
    echo "Error: Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi

echo "Installing RingRift AI Self-Improvement Loop dashboard..."
echo "Grafana URL: $GRAFANA_URL"

# Prepare the dashboard payload
PAYLOAD=$(jq '{
    dashboard: .,
    overwrite: true,
    folderId: 0
}' "$DASHBOARD_FILE")

# Build curl command
CURL_CMD="curl -s -X POST"
CURL_CMD+=" -H 'Content-Type: application/json'"

if [[ -n "$API_KEY" ]]; then
    CURL_CMD+=" -H 'Authorization: Bearer ${API_KEY}'"
else
    echo "Warning: No API key provided. Using anonymous access."
    echo "For production, create an API key in Grafana and pass it as the second argument."
fi

CURL_CMD+=" -d '${PAYLOAD}'"
CURL_CMD+=" '${GRAFANA_URL}/api/dashboards/db'"

# Execute
echo ""
echo "Uploading dashboard..."
RESPONSE=$(eval "$CURL_CMD")

# Check response
if echo "$RESPONSE" | jq -e '.status == "success"' > /dev/null 2>&1; then
    echo "Dashboard installed successfully!"
    echo "URL: ${GRAFANA_URL}$(echo "$RESPONSE" | jq -r '.url')"
else
    echo "Error installing dashboard:"
    echo "$RESPONSE" | jq .
    exit 1
fi

echo ""
echo "Prometheus Configuration"
echo "========================"
echo "Add the following to your prometheus.yml scrape_configs:"
echo ""
echo "  - job_name: 'ringrift-ai-loop'"
echo "    static_configs:"
echo "      - targets: ['localhost:9090']"
echo "    scrape_interval: 15s"
