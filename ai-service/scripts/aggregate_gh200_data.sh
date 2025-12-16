#!/bin/bash
# Aggregate selfplay JSONL data from all GH200 nodes
set -e

AGGREGATED_DIR=data/selfplay/aggregated
OUTPUT_DB=data/games/selfplay_aggregated.db
TMP_DIR=/tmp/gh200_jsonl_$(date +%Y%m%d_%H%M%S)

mkdir -p $TMP_DIR
mkdir -p $AGGREGATED_DIR

echo "[$(date)] Starting JSONL aggregation from GH200 nodes..."

# GH200 node IPs and names
declare -A GH200_NODES=(
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
)

TOTAL_GAMES=0

# Collect JSONL files from each node
for name in "${!GH200_NODES[@]}"; do
  ip=${GH200_NODES[$name]}
  echo "Collecting from $name ($ip)..."
  
  # Get remote file size and line count if exists
  REMOTE_INFO=$(ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@$ip \
    "if [ -f ~/ringrift/ai-service/data/selfplay/gpu/games.jsonl ]; then \
       wc -l ~/ringrift/ai-service/data/selfplay/gpu/games.jsonl | cut -d\" \" -f1; \
     else echo 0; fi" 2>/dev/null) || REMOTE_INFO=0
  
  if [ "$REMOTE_INFO" -gt 0 ]; then
    echo "  Found $REMOTE_INFO games on $name"
    
    # Copy to local tmp with node name prefix
    scp -o ConnectTimeout=30 -o BatchMode=yes \
      ubuntu@$ip:~/ringrift/ai-service/data/selfplay/gpu/games.jsonl \
      $TMP_DIR/${name}_games.jsonl 2>/dev/null || echo "  Failed to copy from $name"
    
    if [ -f "$TMP_DIR/${name}_games.jsonl" ]; then
      TOTAL_GAMES=$((TOTAL_GAMES + REMOTE_INFO))
    fi
  else
    echo "  No data or unreachable: $name"
  fi
done

echo ""
echo "Total games collected: $TOTAL_GAMES"

if [ $TOTAL_GAMES -gt 0 ]; then
  # Merge all JSONL files
  echo "Merging JSONL files..."
  cat $TMP_DIR/*.jsonl > $AGGREGATED_DIR/all_gh200_games.jsonl
  
  # Count merged
  MERGED_COUNT=$(wc -l < $AGGREGATED_DIR/all_gh200_games.jsonl)
  echo "Merged file has $MERGED_COUNT games"
  
  # Convert to SQLite
  echo "Converting to SQLite database..."
  cd ~/ringrift/ai-service
  source venv/bin/activate
  
  python3 scripts/aggregate_jsonl_to_db.py \
    --input-dir $AGGREGATED_DIR \
    --output-db $OUTPUT_DB \
    --verbose
  
  echo ""
  echo "[$(date)] Aggregation complete: $OUTPUT_DB"
  
  # Show stats
  if [ -f "$OUTPUT_DB" ]; then
    python3 -c "
import sqlite3
conn = sqlite3.connect('$OUTPUT_DB')
cursor = conn.cursor()
try:
    cursor.execute('SELECT board_type, num_players, COUNT(*) FROM games GROUP BY board_type, num_players')
    print('Database stats:')
    for row in cursor.fetchall():
        print(f'  {row[0]}_{row[1]}p: {row[2]} games')
    cursor.execute('SELECT COUNT(*) FROM games')
    total = cursor.fetchone()[0]
    print(f'  Total: {total} games')
except Exception as e:
    print(f'Error querying DB: {e}')
conn.close()
"
  fi
fi

# Cleanup
rm -rf $TMP_DIR

echo "[$(date)] Done"
