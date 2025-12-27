#!/bin/bash
# Unified auto-training trigger for all configs
# Monitors canonical databases and triggers training when thresholds reached

cd ~/ringrift/ai-service
source venv/bin/activate

LOG="logs/unified_auto_train.log"
mkdir -p logs

# Training thresholds per config (games needed with move data)
# Lowered thresholds for configs with actual move data available
declare -A THRESHOLDS=(
    ["hex8_2p"]=500      # Has 796 games with moves - ready!
    ["hex8_3p"]=300      # Has 397 games with moves
    ["hex8_4p"]=100      # Has 115 games with moves
    ["square8_2p"]=5000  # Has 15491 games with moves - ready!
    ["square8_3p"]=500   # Low data
    ["square8_4p"]=500   # canonical has games but no moves - use non-canonical
    ["square19_2p"]=50   # Has 77 games with moves
    ["square19_3p"]=100  # Has 150 games with moves - ready!
    ["square19_4p"]=100  # Low data
    ["hexagonal_2p"]=50  # Has 69 games with moves
    ["hexagonal_3p"]=50  # Low data
    ["hexagonal_4p"]=50  # Low data
)

# Database mapping - prefer databases with actual move data
declare -A DB_OVERRIDES=(
    ["hex8_3p"]="data/games/hex8_3p.db"
    ["hex8_4p"]="data/games/hex8_4p.db"
    ["square8_4p"]="data/games/square8_4p.db"
    ["square19_2p"]="data/games/canonical_square19_2p.db"
    ["hexagonal_2p"]="data/games/hexagonal_2p.db"
)

# Track what we have already trained this session
TRAINED_FILE="/tmp/trained_configs.txt"
touch $TRAINED_FILE

check_and_train() {
    local config=$1
    local threshold=$2
    local board_type=$(echo $config | cut -d_ -f1)
    local num_players=$(echo $config | cut -d_ -f2 | tr -d "p")

    # Use override DB if available, otherwise canonical
    local db="${DB_OVERRIDES[$config]:-data/games/canonical_${config}.db}"

    # Skip if already trained this session
    if grep -q "^${config}$" $TRAINED_FILE 2>/dev/null; then
        return 0
    fi

    if [ ! -f "$db" ]; then
        return 0
    fi

    # Count games that have move data (not just games table)
    local count=$(python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('$db')
    # Check both games and moves exist
    games = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
    moves = 0
    for tbl in ['game_moves', 'moves']:
        try:
            moves = conn.execute('SELECT COUNT(*) FROM ' + tbl).fetchone()[0]
            if moves > 0:
                break
        except:
            pass
    # Only count if we have move data
    if moves > 0:
        print(games)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)

    echo "$(date): $config: $count / $threshold games" >> $LOG

    if [ "$count" -ge "$threshold" ]; then
        echo "$(date): *** THRESHOLD REACHED for $config! Starting training... ***" | tee -a $LOG

        # Export training data
        echo "$(date): Exporting NPZ for $config..." | tee -a $LOG
        PYTHONPATH=. python scripts/export_replay_dataset.py \
            --db "$db" \
            --board-type $board_type --num-players $num_players \
            --output "data/training/${config}.npz" \
            >> $LOG 2>&1

        if [ ! -f "data/training/${config}.npz" ]; then
            echo "$(date): ERROR: NPZ export failed for $config" | tee -a $LOG
            return 1
        fi

        # Start training
        echo "$(date): Training $config..." | tee -a $LOG
        PYTHONPATH=. python -m app.training.train \
            --board-type $board_type --num-players $num_players \
            --data-path "data/training/${config}.npz" \
            --save-path "models/${config}_trained.pth" \
            --batch-size 512 --epochs 50 \
            >> $LOG 2>&1

        if [ -f "models/${config}_trained.pth" ]; then
            echo "$(date): *** Training complete for $config! ***" | tee -a $LOG

            # Run gauntlet evaluation
            echo "$(date): Running gauntlet for $config..." | tee -a $LOG
            PYTHONPATH=. python scripts/auto_promote.py --gauntlet \
                --model "models/${config}_trained.pth" \
                --board-type $board_type --num-players $num_players \
                --games 30 \
                >> $LOG 2>&1

            echo "$config" >> $TRAINED_FILE
        else
            echo "$(date): ERROR: Training failed for $config" | tee -a $LOG
        fi
    fi
}

echo "$(date): Starting unified auto-training monitor" | tee -a $LOG
echo "$(date): Configs monitored: ${!THRESHOLDS[*]}" | tee -a $LOG

while true; do
    echo "" >> $LOG
    echo "$(date): === Checking all configs ===" >> $LOG

    for config in "${!THRESHOLDS[@]}"; do
        check_and_train "$config" "${THRESHOLDS[$config]}"
    done

    sleep 120  # Check every 2 minutes
done
