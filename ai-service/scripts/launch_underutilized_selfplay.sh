#!/bin/bash
# Launch Gumbel MCTS selfplay on UNDERUTILIZED nodes for 5 hours
# Targets underrepresented configs: square19_3p, square19_4p, hex8_3p, hex8_4p
# Date: Dec 24, 2025

set -e

SSH_KEY=~/.ssh/id_cluster
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

# Underutilized GH200 nodes (no training per cluster monitor)
IDLE_GH200=(
  "100.83.234.82"    # lambda-gh200-b
  "100.88.176.74"    # lambda-gh200-e
  "100.99.27.56"     # lambda-gh200-i
  "100.66.65.33"     # lambda-gh200-q
  "100.79.109.120"   # lambda-gh200-s
  "100.126.108.12"   # lambda-gh200-t
)

# Underutilized A10 node
IDLE_A10=(
  "100.78.55.103"    # lambda-a10-c
)

# Hetzner CPU nodes
HETZNER_CPU=(
  "100.94.174.19"    # hetzner-cpu1
  "100.67.131.72"    # hetzner-cpu2
  "100.126.21.102"   # hetzner-cpu3
)

# Vast.ai nodes with no training
IDLE_VAST=(
  # Note: Many vast nodes are offline, using only reachable ones
)

# Config assignments for 5-hour run
# Target: square19_3p (10k games), square19_4p (7.8k games) - most underrepresented
# Also: hex8_3p, hex8_4p for variety

# GH200 assignments (6 nodes): 2 each for critical configs
CONFIGS_GH200=(
  "square19:4:800"     # lambda-gh200-b
  "square19:4:800"     # lambda-gh200-e
  "square19:3:800"     # lambda-gh200-i
  "square19:3:800"     # lambda-gh200-q
  "hex8:4:1000"        # lambda-gh200-s (faster for smaller board)
  "hex8:3:1000"        # lambda-gh200-t
)

# A10 assignment (1 node)
CONFIGS_A10=(
  "square8:3:1500"     # lambda-a10-c (fast on smaller board)
)

# Hetzner CPU assignments (3 nodes) - CPU only, use heuristic
CONFIGS_HETZNER=(
  "square8:3:200"      # hetzner-cpu1 (CPU-only, slower)
  "square8:4:200"      # hetzner-cpu2
  "hex8:3:200"         # hetzner-cpu3
)

launch_gumbel() {
  local host=$1
  local user=$2
  local board=$3
  local players=$4
  local games=$5
  local path=$6

  local screen_name="gumbel_${board}_${players}p"
  local db_name="${board}_${players}p_gumbel_$(date +%Y%m%d_%H%M%S).db"

  echo "Launching on $host: $board ${players}p ($games games)"

  ssh -i $SSH_KEY $SSH_OPTS ${user}@${host} "
    cd ${path} && \
    screen -X -S ${screen_name} quit 2>/dev/null || true && \
    screen -dmS ${screen_name} bash -c '
      source venv/bin/activate 2>/dev/null || source /workspace/ringrift/ai-service/venv/bin/activate 2>/dev/null
      mkdir -p logs data/games
      PYTHONPATH=. python3 -u scripts/generate_gumbel_selfplay.py \
        --board ${board} \
        --num-players ${players} \
        --num-games ${games} \
        --output data/games/${db_name} \
        2>&1 | tee logs/gumbel_${board}_${players}p_\$(date +%Y%m%d_%H%M%S).log
      exec bash
    '
  " 2>/dev/null && echo "  -> Started" || echo "  -> FAILED (check SSH)"
}

echo "============================================================"
echo "CLUSTER SELFPLAY LAUNCH (UNDERUTILIZED NODES)"
echo "Target: 5 hours runtime"
echo "Date: $(date)"
echo "============================================================"
echo ""

# Launch on idle GH200 nodes
echo "=== Launching on idle GH200 nodes (6 nodes) ==="
for i in "${!IDLE_GH200[@]}"; do
  host="${IDLE_GH200[$i]}"
  config="${CONFIGS_GH200[$i]}"
  IFS=':' read -r board players games <<< "$config"
  launch_gumbel "$host" "ubuntu" "$board" "$players" "$games" "~/ringrift/ai-service" &
  sleep 0.3
done
wait

echo ""
echo "=== Launching on idle A10 nodes (1 node) ==="
for i in "${!IDLE_A10[@]}"; do
  host="${IDLE_A10[$i]}"
  config="${CONFIGS_A10[$i]}"
  IFS=':' read -r board players games <<< "$config"
  launch_gumbel "$host" "ubuntu" "$board" "$players" "$games" "~/ringrift/ai-service" &
done
wait

echo ""
echo "=== Launching on Hetzner CPU nodes (3 nodes) ==="
for i in "${!HETZNER_CPU[@]}"; do
  host="${HETZNER_CPU[$i]}"
  config="${CONFIGS_HETZNER[$i]}"
  IFS=':' read -r board players games <<< "$config"
  launch_gumbel "$host" "root" "$board" "$players" "$games" "~/ringrift/ai-service" &
done
wait

echo ""
echo "============================================================"
echo "LAUNCH COMPLETE - $(date)"
echo "============================================================"
echo ""
echo "Distribution (10 nodes total):"
echo "  square19_4p: 2 nodes (GH200-b, GH200-e) - 1600 games"
echo "  square19_3p: 2 nodes (GH200-i, GH200-q) - 1600 games"
echo "  hex8_4p: 1 node (GH200-s) - 1000 games"
echo "  hex8_3p: 1 node (GH200-t) - 1000 games"
echo "  square8_3p: 2 nodes (A10-c, hetzner-cpu1) - 1700 games"
echo "  square8_4p: 1 node (hetzner-cpu2) - 200 games"
echo "  hex8_3p: 1 node (hetzner-cpu3) - 200 games"
echo ""
echo "Expected total: ~7,300 new games across underrepresented configs"
echo "Runtime: 5+ hours (Gumbel MCTS with 800-1500 sims/move)"
