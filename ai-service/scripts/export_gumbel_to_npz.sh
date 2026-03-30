#!/bin/bash
# Automated gumbel JSONL → NPZ conversion for Lambda nodes.
# Converts local gumbel games to NPZ, COMBINES with the existing canonical
# NPZ from S3 (which has database-sourced games from the coordinator), and
# pushes the combined result back to S3.
#
# This ensures the canonical NPZ always contains BOTH:
# - Gumbel MCTS data (policy quality from search)
# - Database-sourced data (position diversity from heuristic/NN/tournament games)
#
# The coordinator's auto_export_daemon creates the base NPZ from large game
# databases. This script ADDS gumbel data on top, never replaces.
#
# Install: crontab -e → 0 */2 * * * ~/ringrift/ai-service/scripts/export_gumbel_to_npz.sh >> /tmp/gumbel_export.log 2>&1
#
# Mar 29, 2026: Created for gumbel JSONL → NPZ conversion.
# Mar 30, 2026: Rewritten to combine with existing S3 canonical NPZ instead
#   of overwriting. Previous version created tiny NPZs (2-3 MB) that replaced
#   the coordinator's large NPZs (200-800 MB), causing hex8/square8 regression.

set -e
cd ~/ringrift/ai-service
export PYTHONPATH=.
LOG_PREFIX="[gumbel-export $(date +%H:%M)]"

# Max samples per config to prevent OOM during training (500K positions)
MAX_SAMPLES=500000

for config in hex8_2p hex8_3p hex8_4p square8_2p square8_3p square8_4p square19_2p square19_3p square19_4p hexagonal_2p hexagonal_3p hexagonal_4p; do
    board=$(echo $config | sed 's/_[0-9]p//')
    players=$(echo $config | grep -o '[0-9]p' | tr -d 'p')

    MERGED="/tmp/${config}_gumbel_complete.jsonl"
    GUMBEL_NPZ="/tmp/${config}_gumbel_only.npz"
    EXISTING_NPZ="/tmp/${config}_existing.npz"
    COMBINED_NPZ="data/training/${config}.npz"
    rm -f "$MERGED" "$GUMBEL_NPZ" "$EXISTING_NPZ"

    # Step 1: Merge only complete gumbel games (with winners and >10 moves)
    python3 -c "
import json, glob, os
output = open('$MERGED', 'w')
count = 0
# Check both gumbel selfplay directories
for pattern in [
    'data/selfplay/gumbel/${config}/*/gumbel_${config}.jsonl',
    'data/selfplay/p2p_gpu/${board}_${players}p/*.jsonl',
]:
    for f in sorted(glob.glob(pattern)):
        try:
            for line in open(f):
                try:
                    d = json.loads(line)
                    if d.get('winner') is not None and d.get('num_moves', 0) > 10:
                        output.write(line)
                        count += 1
                except: pass
        except: pass
output.close()
print(f'${config}: {count} complete gumbel games')
" 2>/dev/null

    GAME_COUNT=$(wc -l < "$MERGED" 2>/dev/null || echo 0)

    # Step 2: Convert gumbel JSONL to NPZ (if we have games)
    HAVE_GUMBEL=false
    if [ "$GAME_COUNT" -gt 5 ]; then
        echo "$LOG_PREFIX Converting $config: $GAME_COUNT gumbel games to NPZ"
        venv/bin/python scripts/jsonl_to_npz.py \
            --input "$MERGED" \
            --output "$GUMBEL_NPZ" \
            --board-type "$board" --num-players "$players" \
            --gpu-selfplay 2>/dev/null | tail -3
        if [ -f "$GUMBEL_NPZ" ] && [ $(stat -c%s "$GUMBEL_NPZ" 2>/dev/null || stat -f%z "$GUMBEL_NPZ" 2>/dev/null) -gt 100 ]; then
            HAVE_GUMBEL=true
        fi
    fi

    # Step 3: Pull existing canonical NPZ from S3 (the coordinator's export)
    HAVE_EXISTING=false
    aws s3 cp "s3://ringrift-models-20251214/consolidated/training/${config}.npz" \
        "$EXISTING_NPZ" --quiet 2>/dev/null || true
    if [ -f "$EXISTING_NPZ" ] && [ $(stat -c%s "$EXISTING_NPZ" 2>/dev/null || stat -f%z "$EXISTING_NPZ" 2>/dev/null) -gt 100 ]; then
        HAVE_EXISTING=true
    fi

    # Step 4: Combine gumbel + existing, or use whichever is available
    if [ "$HAVE_GUMBEL" = true ] && [ "$HAVE_EXISTING" = true ]; then
        echo "$LOG_PREFIX Combining gumbel + existing canonical for $config"
        venv/bin/python3 -c "
import numpy as np
import sys

MAX_SAMPLES = $MAX_SAMPLES

# Load both NPZ files
g = np.load('$GUMBEL_NPZ', allow_pickle=True)
e = np.load('$EXISTING_NPZ', allow_pickle=True)

gumbel_n = len(g['features'])
existing_n = len(e['features'])
total = gumbel_n + existing_n

print(f'Gumbel: {gumbel_n} samples, Existing: {existing_n} samples')

# If combined exceeds max, downsample the existing (keep ALL gumbel)
if total > MAX_SAMPLES and gumbel_n < MAX_SAMPLES:
    # Keep all gumbel data, sample from existing to fill remaining budget
    keep_existing = MAX_SAMPLES - gumbel_n
    indices = np.random.default_rng(42).choice(existing_n, size=keep_existing, replace=False)
    indices.sort()
    print(f'Capping: keeping all {gumbel_n} gumbel + {keep_existing}/{existing_n} existing = {MAX_SAMPLES}')
elif gumbel_n >= MAX_SAMPLES:
    # Gumbel alone exceeds cap — sample from gumbel only
    keep_existing = 0
    indices = np.array([], dtype=int)
    gumbel_indices = np.random.default_rng(42).choice(gumbel_n, size=MAX_SAMPLES, replace=False)
    gumbel_indices.sort()
    print(f'Capping: {MAX_SAMPLES}/{gumbel_n} gumbel only')
else:
    indices = np.arange(existing_n)
    keep_existing = existing_n

# Build combined arrays
combined = {}
for k in g.files:
    gumbel_arr = g[k]
    if gumbel_n >= MAX_SAMPLES:
        gumbel_arr = gumbel_arr[gumbel_indices]

    if k in e.files and keep_existing > 0:
        existing_arr = e[k][indices]
        combined[k] = np.concatenate([gumbel_arr, existing_arr])
    else:
        combined[k] = gumbel_arr

# Also include any keys only in existing
for k in e.files:
    if k not in combined and keep_existing > 0:
        combined[k] = e[k][indices]

np.savez_compressed('$COMBINED_NPZ', **combined)
final_n = len(combined['features'])
print(f'Combined NPZ: {final_n} samples')
" 2>/dev/null
    elif [ "$HAVE_GUMBEL" = true ]; then
        echo "$LOG_PREFIX No existing canonical on S3, using gumbel-only for $config"
        cp "$GUMBEL_NPZ" "$COMBINED_NPZ"
    elif [ "$HAVE_EXISTING" = true ]; then
        echo "$LOG_PREFIX No new gumbel games, keeping existing canonical for $config"
        # Don't overwrite — the existing S3 version is already good
        rm -f "$MERGED" "$GUMBEL_NPZ" "$EXISTING_NPZ"
        continue
    else
        echo "$LOG_PREFIX No data available for $config, skipping"
        rm -f "$MERGED" "$GUMBEL_NPZ" "$EXISTING_NPZ"
        continue
    fi

    # Step 5: Push combined NPZ to S3
    if [ -f "$COMBINED_NPZ" ]; then
        SIZE=$(stat -c%s "$COMBINED_NPZ" 2>/dev/null || stat -f%z "$COMBINED_NPZ" 2>/dev/null)
        SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc)
        aws s3 cp "$COMBINED_NPZ" \
            "s3://ringrift-models-20251214/consolidated/training/${config}.npz" --quiet 2>/dev/null
        echo "$LOG_PREFIX Pushed ${config}.npz to S3 (${SIZE_MB}MB)"
    fi

    rm -f "$MERGED" "$GUMBEL_NPZ" "$EXISTING_NPZ"
done
