#!/bin/bash
# Deploy GPU turn fix to all cluster nodes and clean corrupted data
# Run this script from the ai-service directory

set -e

# GH200 nodes (via Tailscale IPs for reliability)
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
    # "100.105.66.41" # lambda-gh200-j - STOPPED
    "100.96.142.42"   # lambda-gh200-k
    "100.76.145.60"   # lambda-gh200-l
)

# Other Lambda instances
LAMBDA_HOSTS=(
    "100.91.25.13"    # lambda-a10
    "100.78.101.123"  # lambda-h100
    "100.97.104.89"   # lambda-2xh100
)

# Vast instances
VAST_HOSTS=(
    "100.118.201.85"  # vast-5090-quad-new
    "100.100.242.64"  # vast-rtx4060ti
    "100.74.154.36"   # vast-rtx3070-a-new
    "100.75.98.13"    # vast-rtx2060s
)

# SSH options
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "=============================================="
echo "GPU Turn Fix Deployment Script"
echo "=============================================="
echo ""

# Step 1: Stop selfplay on all nodes in parallel
echo "STEP 1: Stopping selfplay on all nodes..."
for host in "${GH200_HOSTS[@]}" "${LAMBDA_HOSTS[@]}"; do
    (
        echo "  Stopping selfplay on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "pkill -f 'python.*selfplay\|unified_ai_loop' 2>/dev/null || true" &
    ) &
done
for host in "${VAST_HOSTS[@]}"; do
    (
        echo "  Stopping selfplay on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster root@$host "pkill -f 'python.*selfplay\|unified_ai_loop' 2>/dev/null || true" &
    ) &
done
wait
echo "  Done stopping selfplay."
echo ""

# Step 2: Pull latest code on all nodes
echo "STEP 2: Pulling latest code on all nodes..."
for host in "${GH200_HOSTS[@]}" "${LAMBDA_HOSTS[@]}"; do
    (
        echo "  Pulling on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "cd ~/ringrift && git fetch origin && git reset --hard origin/main" 2>&1 | head -3
    ) &
done
for host in "${VAST_HOSTS[@]}"; do
    (
        echo "  Pulling on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster root@$host "cd ~/ringrift && git fetch origin && git reset --hard origin/main" 2>&1 | head -3
    ) &
done
wait
echo "  Done pulling code."
echo ""

# Step 3: Delete corrupted hex/square19 JSONL files
echo "STEP 3: Deleting corrupted hex/square19 JSONL files..."
DELETE_CMD='rm -rf ~/ringrift/ai-service/data/games/nfs_sync/hex_*/*.jsonl ~/ringrift/ai-service/data/games/nfs_sync/sq19_*/*.jsonl ~/ringrift/ai-service/data/games/gpu_selfplay/hex_*/*.jsonl ~/ringrift/ai-service/data/games/gpu_selfplay/sq19_*/*.jsonl 2>/dev/null; echo "Cleaned corrupted hex/sq19 games"'

for host in "${GH200_HOSTS[@]}" "${LAMBDA_HOSTS[@]}"; do
    (
        echo "  Cleaning on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "$DELETE_CMD"
    ) &
done
for host in "${VAST_HOSTS[@]}"; do
    (
        echo "  Cleaning on $host..."
        ssh $SSH_OPTS -i ~/.ssh/id_cluster root@$host "$DELETE_CMD"
    ) &
done
wait
echo "  Done cleaning corrupted data."
echo ""

# Step 4: Verify the fix is deployed
echo "STEP 4: Verifying fix deployment..."
VERIFY_CMD='grep -c "BUG FIX 2025-12-15" ~/ringrift/ai-service/app/ai/gpu_parallel_games.py 2>/dev/null || echo "0"'
for host in "${GH200_HOSTS[@]:0:3}"; do
    result=$(ssh $SSH_OPTS -i ~/.ssh/id_cluster ubuntu@$host "$VERIFY_CMD")
    echo "  $host: $result fix markers found"
done
echo ""

echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Restart selfplay with: ./scripts/start_cluster_selfplay.sh"
echo "2. Monitor with: ./scripts/cluster_status.sh"
