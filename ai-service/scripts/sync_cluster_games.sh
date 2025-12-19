#!/bin/bash
# Sync GPU selfplay games from cluster nodes to local machine
# Usage: ./scripts/sync_cluster_games.sh

set -e
cd "$(dirname "$0")/.."

LOCAL_DIR="data/games/gpu_selfplay"
mkdir -p "$LOCAL_DIR"

if [ -z "${RINGRIFT_LEGACY_SYNC:-}" ]; then
    echo "[sync_cluster_games] Deprecated; use cluster_sync_coordinator.py --mode games"
    python3 scripts/cluster_sync_coordinator.py --mode games
    exit $?
fi

# All hosts
GH200_HOSTS=(
    "100.123.183.70" "100.104.34.73" "100.88.35.19" "100.75.84.47"
    "100.88.176.74" "100.104.165.116" "100.104.126.58" "100.65.88.62"
    "100.99.27.56" "100.96.142.42" "100.76.145.60"
)
LAMBDA_HOSTS=("100.91.25.13" "100.78.101.123" "100.97.104.89")
VAST_HOSTS=("100.118.201.85" "100.100.242.64" "100.74.154.36" "100.75.98.13")

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "=============================================="
echo "Syncing GPU Selfplay Games from Cluster"
echo "=============================================="

sync_from_host() {
    local host=$1
    local user=$2
    local host_name=$3

    echo "Syncing from $host_name ($host)..."

    for config in hexagonal_2p hexagonal_3p hexagonal_4p square19_2p square19_3p square19_4p; do
        remote_file="~/ringrift/ai-service/data/games/gpu_selfplay/${config}/games.jsonl"
        local_file="$LOCAL_DIR/${config}/games_${host_name}.jsonl"
        mkdir -p "$LOCAL_DIR/${config}"

        # Copy file if it exists and has content
        ssh $SSH_OPTS -i ~/.ssh/id_cluster ${user}@${host} "test -s $remote_file" 2>/dev/null && \
            scp $SSH_OPTS -i ~/.ssh/id_cluster ${user}@${host}:${remote_file} "$local_file" 2>/dev/null && \
            echo "  $config: $(wc -l < "$local_file") games" || \
            echo "  $config: skipped (no data)"
    done
}

# Sync in parallel
echo ""
echo "Syncing from GH200 nodes..."
for i in "${!GH200_HOSTS[@]}"; do
    host="${GH200_HOSTS[$i]}"
    letter=$(printf "\\x$(printf '%02x' $((97 + i)))")  # a, b, c, ...
    sync_from_host "$host" "ubuntu" "gh200-$letter" &
done
wait

echo ""
echo "Syncing from Lambda nodes..."
sync_from_host "${LAMBDA_HOSTS[0]}" "ubuntu" "lambda-a10" &
sync_from_host "${LAMBDA_HOSTS[1]}" "ubuntu" "lambda-h100" &
sync_from_host "${LAMBDA_HOSTS[2]}" "ubuntu" "lambda-2xh100" &
wait

echo ""
echo "Syncing from Vast nodes..."
sync_from_host "${VAST_HOSTS[0]}" "root" "vast-5090" &
sync_from_host "${VAST_HOSTS[1]}" "root" "vast-4060ti" &
sync_from_host "${VAST_HOSTS[2]}" "root" "vast-3070" &
sync_from_host "${VAST_HOSTS[3]}" "root" "vast-2060s" &
wait

echo ""
echo "=============================================="
echo "Merging games into combined files..."
echo "=============================================="

for config in hexagonal_2p hexagonal_3p hexagonal_4p square19_2p square19_3p square19_4p; do
    combined="$LOCAL_DIR/${config}/games_combined.jsonl"
    rm -f "$combined"

    # Merge all per-host files
    cat "$LOCAL_DIR/${config}"/games_*.jsonl > "$combined" 2>/dev/null || true

    if [ -f "$combined" ]; then
        count=$(wc -l < "$combined")
        echo "$config: $count total games"
    else
        echo "$config: 0 games"
    fi
done

echo ""
echo "Sync complete!"
