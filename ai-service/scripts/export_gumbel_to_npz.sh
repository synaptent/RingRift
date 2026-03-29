#!/bin/bash
# Automated gumbel JSONL → NPZ conversion for Lambda nodes.
# Merges all complete gumbel games from JSONL files, converts to NPZ,
# and pushes to S3 for training nodes to pull.
#
# Install: crontab -e → 0 */2 * * * ~/ringrift/ai-service/scripts/export_gumbel_to_npz.sh >> /tmp/gumbel_export.log 2>&1
#
# Mar 29, 2026: Created after discovering gumbel selfplay writes JSONL
# but the training pipeline reads NPZ. The auto_export_daemon reads
# SQLite DBs (which are empty for gumbel) and misses all JSONL data.

set -e
cd ~/ringrift/ai-service
export PYTHONPATH=.
LOG_PREFIX="[gumbel-export $(date +%H:%M)]"

for config in hex8_2p hex8_3p hex8_4p square8_2p square8_3p square8_4p; do
    board=$(echo $config | sed 's/_[0-9]p//')
    players=$(echo $config | grep -o '[0-9]p' | tr -d 'p')

    MERGED="/tmp/${config}_gumbel_complete.jsonl"
    rm -f "$MERGED"

    # Merge only complete games (with winners and >10 moves)
    python3 -c "
import json, glob
output = open('$MERGED', 'w')
count = 0
for f in glob.glob('data/selfplay/gumbel/${config}/*/gumbel_${config}.jsonl'):
    for line in open(f):
        try:
            d = json.loads(line)
            if d.get('winner') is not None and d.get('num_moves', 0) > 10:
                output.write(line)
                count += 1
        except: pass
output.close()
print(f'${config}: {count} complete games')
" 2>/dev/null

    GAME_COUNT=$(wc -l < "$MERGED" 2>/dev/null || echo 0)

    if [ "$GAME_COUNT" -gt 50 ]; then
        echo "$LOG_PREFIX Exporting $config: $GAME_COUNT games"
        venv/bin/python scripts/jsonl_to_npz.py \
            --input "$MERGED" \
            --output "data/training/${config}.npz" \
            --board-type "$board" --num-players "$players" \
            --gpu-selfplay 2>/dev/null | tail -3

        # Also pull cross-strategy data from S3 and combine for optimal training mix
        # Cross-strategy data (heuristic, minimax, BRS games) provides position diversity
        # Gumbel data provides MCTS policy quality. Combined = strongest training signal.
        CROSS_NPZ="/tmp/${config}_crossstrat.npz"
        aws s3 cp "s3://ringrift-models-20251214/consolidated/training/${config}_crossstrat.npz" \
            "$CROSS_NPZ" --quiet 2>/dev/null || true

        if [ -f "$CROSS_NPZ" ] && [ -f "data/training/${config}.npz" ]; then
            echo "$LOG_PREFIX Combining gumbel + cross-strategy for $config"
            # numpy.load with allow_pickle=True is required for NPZ files containing
            # object arrays (policy indices). These are trusted local training data files.
            venv/bin/python3 -c "
import numpy as np
g = np.load('data/training/${config}.npz', allow_pickle=True)
c = np.load('$CROSS_NPZ', allow_pickle=True)
combined = {}
for k in g.files:
    if k in c.files:
        combined[k] = np.concatenate([g[k], c[k]])
    else:
        combined[k] = g[k]
np.savez_compressed('data/training/${config}.npz', **combined)
print(f'Combined: {len(g[\"features\"])} gumbel + {len(c[\"features\"])} cross = {len(combined[\"features\"])} total')
" 2>/dev/null
            rm -f "$CROSS_NPZ"
        fi

        # Push to S3
        if [ -f "data/training/${config}.npz" ]; then
            aws s3 cp "data/training/${config}.npz" \
                "s3://ringrift-models-20251214/consolidated/training/${config}.npz" --quiet 2>/dev/null
            echo "$LOG_PREFIX Pushed ${config}.npz to S3"
        fi
    fi

    rm -f "$MERGED"
done
