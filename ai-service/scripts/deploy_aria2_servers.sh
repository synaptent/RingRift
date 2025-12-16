#!/bin/bash
# Deploy aria2 data server across cluster nodes
# Usage: ./scripts/deploy_aria2_servers.sh [--dry-run] [host1 host2 ...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$AI_SERVICE_DIR/deploy/systemd/aria2-data-server.service"

# Default hosts (Tailscale IPs for reliable connectivity)
DEFAULT_HOSTS=(
    "100.78.101.123"   # lambda-h100 (209-20-157-81)
    "100.123.183.70"   # gh200_a (192-222-51-29)
    "100.104.34.73"    # gh200_b (192-222-51-167)
    "100.88.35.19"     # gh200_c (192-222-51-162)
    "100.75.84.47"     # gh200_d (192-222-58-122)
    "100.121.198.28"   # aws_staging
    "100.115.97.24"    # aws_worker
)

DRY_RUN=false
HOSTS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            HOSTS+=("$1")
            shift
            ;;
    esac
done

# Use default hosts if none specified
if [ ${#HOSTS[@]} -eq 0 ]; then
    HOSTS=("${DEFAULT_HOSTS[@]}")
fi

echo "=== Aria2 Data Server Deployment ==="
echo "Hosts: ${HOSTS[*]}"
echo "Dry run: $DRY_RUN"
echo ""

deploy_to_host() {
    local host=$1
    local user="ubuntu"

    echo "--- Deploying to $host ---"

    # Check connectivity
    if ! timeout 5 ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$user@$host" "echo ok" &>/dev/null; then
        echo "  SKIP: Cannot connect to $host"
        return 1
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would deploy to $host"
        return 0
    fi

    # Pull latest code
    echo "  Pulling latest code..."
    ssh "$user@$host" "cd ~/ringrift && git pull --ff-only" || true

    # Copy service file
    echo "  Copying service file..."
    scp "$SERVICE_FILE" "$user@$host:/tmp/aria2-data-server.service"

    # Install and start service
    echo "  Installing service..."
    ssh "$user@$host" "sudo cp /tmp/aria2-data-server.service /etc/systemd/system/ && \
        sudo systemctl daemon-reload && \
        sudo systemctl enable aria2-data-server && \
        sudo systemctl restart aria2-data-server"

    # Check status
    echo "  Checking status..."
    ssh "$user@$host" "sudo systemctl status aria2-data-server --no-pager | head -15" || true

    # Test endpoint
    echo "  Testing endpoint..."
    if ssh "$user@$host" "curl -s http://localhost:8766/health" | grep -q "ok"; then
        echo "  SUCCESS: Service running on $host"
    else
        echo "  WARNING: Service may not be fully ready yet"
    fi

    echo ""
}

# Deploy to each host
SUCCESS=0
FAILED=0

for host in "${HOSTS[@]}"; do
    if deploy_to_host "$host"; then
        ((SUCCESS++))
    else
        ((FAILED++))
    fi
done

echo "=== Deployment Summary ==="
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "Active data servers:"
    for host in "${HOSTS[@]}"; do
        echo "  http://$host:8766"
    done
fi
