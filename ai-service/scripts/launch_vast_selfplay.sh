#!/bin/bash
# Launch Gumbel MCTS selfplay on idle Vast.ai nodes
# Date: Dec 24, 2025

set -e

SSH_KEY=~/.ssh/id_cluster
SSH_OPTS="-o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes"

# Vast.ai nodes that are online with no training (from cluster monitor)
# Format: "ssh_host:ssh_port:path:config"
VAST_NODES=(
  "ssh5.vast.ai:31158:/workspace/ringrift/ai-service:hexagonal:2:300"   # vast-29031159 (RTX 5080)
  "ssh8.vast.ai:38470:~/ringrift/ai-service:hexagonal:3:500"            # vast-29118471 (8x RTX 3090)
  "ssh9.vast.ai:38472:~/ringrift/ai-service:hexagonal:4:500"            # vast-29118472 (4x RTX 5090)
  "ssh5.vast.ai:16088:/workspace/ringrift/ai-service:square8:3:400"     # vast-29126088 (RTX 4060 Ti)
  "ssh9.vast.ai:18352:/workspace/ringrift/ai-service:square19:2:600"    # vast-29128352 (2x RTX 5090)
  "ssh7.vast.ai:18356:/workspace/ringrift/ai-service:square19:4:600"    # vast-29128356 (RTX 5090)
  "ssh5.vast.ai:18356:/workspace/ringrift/ai-service:hexagonal:2:500"   # vast-29128357 (2x RTX 5090)
  "ssh4.vast.ai:19150:/workspace/ringrift/ai-service:hexagonal:4:600"   # vast-29129151 (4x RTX 5090)
  "ssh6.vast.ai:19528:~/ringrift/ai-service:hexagonal:3:600"            # vast-29129529 (8x RTX 4090)
)

launch_vast_gumbel() {
  local ssh_host=$1
  local ssh_port=$2
  local path=$3
  local board=$4
  local players=$5
  local games=$6

  local screen_name="gumbel_${board}_${players}p"
  local db_name="${board}_${players}p_gumbel_$(date +%Y%m%d_%H%M%S).db"

  echo "Launching on ${ssh_host}:${ssh_port}: $board ${players}p ($games games)"

  ssh -i $SSH_KEY $SSH_OPTS -p ${ssh_port} root@${ssh_host} "
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
  " 2>/dev/null && echo "  -> Started" || echo "  -> FAILED (node may be offline)"
}

echo "============================================================"
echo "VAST.AI SELFPLAY LAUNCH"
echo "Date: $(date)"
echo "============================================================"
echo ""

echo "=== Launching on Vast.ai nodes (9 nodes) ==="
for node in "${VAST_NODES[@]}"; do
  IFS=':' read -r ssh_host ssh_port path board players games <<< "$node"
  launch_vast_gumbel "$ssh_host" "$ssh_port" "$path" "$board" "$players" "$games" &
  sleep 0.5
done
wait

echo ""
echo "============================================================"
echo "VAST.AI LAUNCH COMPLETE - $(date)"
echo "============================================================"
echo ""
echo "Distribution (9 nodes):"
echo "  hexagonal_2p: 2 nodes - 800 games"
echo "  hexagonal_3p: 2 nodes - 1100 games"
echo "  hexagonal_4p: 2 nodes - 1100 games"
echo "  square19_2p: 1 node - 600 games"
echo "  square19_4p: 1 node - 600 games"
echo "  square8_3p: 1 node - 400 games"
echo ""
echo "Expected total: ~4,600 new games"
