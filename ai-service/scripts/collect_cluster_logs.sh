#!/bin/bash
# Collect logs from all cluster nodes
set -e

LOG_DIR=logs/cluster_$(date +%Y%m%d)
mkdir -p $LOG_DIR

echo "[$(date)] Collecting cluster logs..."

# All cluster nodes
declare -A NODES=(
  ["gh200_a"]="192.222.51.29"
  ["gh200_b"]="192.222.51.167"
  ["gh200_c"]="192.222.51.162"
  ["gh200_d"]="192.222.58.122"
  ["gh200_e"]="192.222.57.162"
  ["gh200_f"]="192.222.57.178"
  ["gh200_g"]="192.222.57.79"
  ["gh200_h"]="192.222.56.123"
  ["gh200_i"]="192.222.50.112"
  ["gh200_j"]="192.222.50.210"
  ["gh200_k"]="192.222.51.150"
  ["gh200_l"]="192.222.51.233"
  ["h100"]="209.20.157.81"
  ["a10"]="150.136.65.197"
)

for name in "${!NODES[@]}"; do
  ip=${NODES[$name]}
  echo "Collecting from $name ($ip)..."
  mkdir -p $LOG_DIR/$name
  rsync -az --timeout=30 -e "ssh -o ConnectTimeout=10 -o BatchMode=yes" ubuntu@$ip:~/ringrift/ai-service/logs/*.log $LOG_DIR/$name/ 2>/dev/null || echo "  Failed: $name"
done

# Summary
echo ""
echo "=== Log Collection Summary ==="
for name in "${!NODES[@]}"; do
  count=$(ls $LOG_DIR/$name/*.log 2>/dev/null | wc -l || echo 0)
  echo "$name: $count log files"
done

echo ""
echo "[$(date)] Logs collected to $LOG_DIR"
