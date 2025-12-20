#!/bin/bash
# Collect selfplay data from Lambda cluster nodes

DEST_DIR="data/games/cluster_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST_DIR"

echo "Collecting selfplay data to $DEST_DIR..."

# Lambda nodes
NODES="100.65.88.62 100.79.109.120 100.117.177.83 100.99.27.56 100.97.98.26 100.66.65.33 100.104.126.58 100.83.234.82"

for ip in $NODES; do
  echo "Collecting from $ip..."
  mkdir -p "$DEST_DIR/$ip"
  rsync -az --timeout=60 \
    ubuntu@$ip:~/ringrift/ai-service/data/games/diverse/ \
    "$DEST_DIR/$ip/" 2>/dev/null &
done

wait
echo "Collection complete. Data in $DEST_DIR"

# Count games
total=0
for f in $(find "$DEST_DIR" -name "*.jsonl" 2>/dev/null); do
  count=$(wc -l < "$f" 2>/dev/null || echo 0)
  total=$((total + count))
done
echo "Total games collected: $total"
