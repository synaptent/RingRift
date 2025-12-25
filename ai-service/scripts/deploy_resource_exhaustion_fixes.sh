#!/bin/bash
# Deploy resource exhaustion prevention fixes to cluster nodes
# Created: Dec 25, 2025
#
# Fixes included:
# 1. scripts/p2p/constants.py - Gauntlet resource limits (MAX_CONCURRENT_GAUNTLETS, etc.)
# 2. scripts/auto_promote.py - Pre-execution resource checks
# 3. scripts/run_self_play_soak.py - Fixed unbound num_players bug
# 4. app/training/game_gauntlet.py - Added adaptive worker scaling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"

# Files to deploy
FILES=(
    "scripts/p2p/constants.py"
    "scripts/auto_promote.py"
    "scripts/run_self_play_soak.py"
    "app/training/game_gauntlet.py"
)

# Target nodes (can be overridden via environment)
NODES="${RINGRIFT_DEPLOY_NODES:-100.88.35.19 100.123.183.70 192.222.51.161 192.222.53.22}"

echo "========================================"
echo "Resource Exhaustion Fixes Deployment"
echo "========================================"
echo ""
echo "Files to deploy:"
for f in "${FILES[@]}"; do
    echo "  - $f"
done
echo ""
echo "Target nodes: $NODES"
echo ""

# Check connectivity first
echo "Checking node connectivity..."
ACCESSIBLE_NODES=()
for node in $NODES; do
    if timeout 10 ssh -o ConnectTimeout=8 -o BatchMode=yes ubuntu@$node "echo 'OK'" 2>/dev/null; then
        echo "  ✓ $node - accessible"
        ACCESSIBLE_NODES+=("$node")
    else
        echo "  ✗ $node - unreachable"
    fi
done

if [ ${#ACCESSIBLE_NODES[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: No nodes are accessible. Aborting deployment."
    exit 1
fi

echo ""
echo "Deploying to ${#ACCESSIBLE_NODES[@]} accessible nodes..."

for node in "${ACCESSIBLE_NODES[@]}"; do
    echo ""
    echo "--- Deploying to $node ---"
    for file in "${FILES[@]}"; do
        src="${AI_SERVICE_ROOT}/${file}"
        dst="ubuntu@${node}:~/ringrift/ai-service/${file}"

        if [ -f "$src" ]; then
            echo "  Copying $file..."
            scp -o ConnectTimeout=10 "$src" "$dst" && echo "    ✓ Done" || echo "    ✗ Failed"
        else
            echo "  ✗ Source file not found: $src"
        fi
    done
done

echo ""
echo "========================================"
echo "Deployment complete!"
echo "========================================"
echo ""
echo "To verify, run on each node:"
echo "  grep -n 'MAX_CONCURRENT_GAUNTLETS' ~/ringrift/ai-service/scripts/p2p/constants.py"
echo "  grep -n 'check_gauntlet_resources' ~/ringrift/ai-service/scripts/auto_promote.py"
