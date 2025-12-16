#!/bin/bash
# Deploy systemd services to cluster nodes
# Run from ai-service directory

set -e

# GH200 nodes (via Tailscale IPs)
GH200_HOSTS=(
    "100.123.183.70"  # lambda-gh200-a
    "100.104.34.73"   # lambda-gh200-b
    "100.88.35.19"    # lambda-gh200-c
    "100.75.84.47"    # lambda-gh200-d
    "100.88.176.74"   # lambda-gh200-e
    "100.104.165.116" # lambda-gh200-f
    "100.104.126.58"  # lambda-gh200-g
    "100.65.88.62"    # lambda-gh200-h
    "100.99.27.56"    # lambda-gh200-i
    "100.96.142.42"   # lambda-gh200-k
    "100.76.145.60"   # lambda-gh200-l
)

# Other Lambda instances
LAMBDA_HOSTS=(
    "100.91.25.13"    # lambda-a10
    "100.78.101.123"  # lambda-h100
    "100.97.104.89"   # lambda-2xh100
)

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
SERVICE_FILE="systemd/ringrift-elo-sync.service"

echo "=============================================="
echo "Deploying RingRift Elo Sync Service"
echo "=============================================="

deploy_to_host() {
    local host=$1
    local user=$2
    echo "  Deploying to $host..."

    # Copy service file
    scp $SSH_OPTS -i ~/.ssh/id_cluster $SERVICE_FILE $user@$host:/tmp/ringrift-elo-sync.service || return 1

    # Install and enable service
    ssh $SSH_OPTS -i ~/.ssh/id_cluster $user@$host << 'EOF'
        sudo cp /tmp/ringrift-elo-sync.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable ringrift-elo-sync
        sudo systemctl restart ringrift-elo-sync
        sudo systemctl status ringrift-elo-sync --no-pager || true
EOF
    return $?
}

echo ""
echo "Deploying to GH200 nodes..."
for host in "${GH200_HOSTS[@]}"; do
    deploy_to_host "$host" "ubuntu" &
done
wait

echo ""
echo "Deploying to Lambda nodes..."
for host in "${LAMBDA_HOSTS[@]}"; do
    deploy_to_host "$host" "ubuntu" &
done
wait

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "Check status with:"
echo "  ssh ubuntu@<node> 'sudo systemctl status ringrift-elo-sync'"
echo ""
