#!/bin/bash
# Complete HP Tuning Pipeline - integrates with existing orchestration
#
# This script:
# 1. Monitors HP tuning jobs on Lambda hosts
# 2. Monitors selfplay generation on GH200 hosts
# 3. Collects completed selfplay data using existing sync scripts
# 4. Triggers curriculum training when HP tuning completes
#
# Usage: ./scripts/run_hp_tuning_pipeline.sh [--watch] [--trigger-training]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$AI_SERVICE_DIR"

# Source venv
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Configuration
HP_HOSTS=(
  "lambda-h100:ubuntu@209.20.157.81:square8_2p"
  "lambda-2xh100:ubuntu@192.222.53.22:square8_3p"
  "lambda-a10:ubuntu@150.136.65.197:square8_4p"
)

SELFPLAY_HOSTS=(
  "GH200-a:ubuntu@192.222.51.29:hexagonal_4p"
  "GH200-b:ubuntu@192.222.51.167:square19_4p"
  "GH200-c:ubuntu@192.222.51.162:hexagonal_2p"
  "GH200-d:ubuntu@192.222.58.122:square19_2p"
  "GH200-e:ubuntu@192.222.57.162:hexagonal_3p"
  "GH200-f:ubuntu@192.222.57.178:square19_3p"
)

WATCH_MODE=false
TRIGGER_TRAINING=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --watch|-w) WATCH_MODE=true; shift ;;
    --trigger-training) TRIGGER_TRAINING=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--watch] [--trigger-training]"
      echo "  --watch: Continuously monitor jobs"
      echo "  --trigger-training: Auto-trigger training when HP tuning completes"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

check_hp_tuning() {
  local all_done=true
  local any_running=false

  echo "=== HP TUNING STATUS ==="
  for entry in "${HP_HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"

    running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f tune_hyperparameters' 2>/dev/null || true)

    if [ -n "$running" ]; then
      any_running=true
      all_done=false

      # Get progress
      result=$(ssh -o ConnectTimeout=5 "$host" \
        "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | \
         python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get(\"trials\",[])), d.get(\"best_score\",-1))' 2>/dev/null" 2>/dev/null || echo "0 -1")
      trials=$(echo "$result" | awk '{print $1}')
      best=$(echo "$result" | awk '{printf "%.4f", $2}')

      echo "  ⏳ $name ($config): Trial $trials/50, best=$best"
    else
      # Check if it completed successfully
      if ssh -o ConnectTimeout=5 "$host" "test -f ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json" 2>/dev/null; then
        result=$(ssh -o ConnectTimeout=5 "$host" \
          "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | \
           python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get(\"trials\",[])), d.get(\"best_score\",-1))' 2>/dev/null" 2>/dev/null || echo "0 -1")
        trials=$(echo "$result" | awk '{print $1}')
        best=$(echo "$result" | awk '{printf "%.4f", $2}')
        echo "  ✓ $name ($config): COMPLETE - $trials trials, best=$best"
      else
        all_done=false
        echo "  ✗ $name ($config): Not started or failed"
      fi
    fi
  done

  if [ "$all_done" = true ]; then
    return 0  # All done
  else
    return 1  # Still running or not started
  fi
}

check_selfplay() {
  local all_done=true
  local total_games=0

  echo ""
  echo "=== SELFPLAY GENERATION STATUS ==="
  for entry in "${SELFPLAY_HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"

    running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f "run_self_play_soak|generate_canonical_selfplay"' 2>/dev/null || true)
    games=$(ssh -o ConnectTimeout=5 "$host" "sqlite3 ~/ringrift/ai-service/data/games/new_${config}.db 'SELECT COUNT(*) FROM games' 2>/dev/null" 2>/dev/null || echo "0")

    if [ -n "$running" ]; then
      all_done=false
      echo "  ⏳ $name ($config): $games games generated"
    else
      if [ "$games" != "0" ] && [ -n "$games" ]; then
        echo "  ✓ $name ($config): COMPLETE - $games games"
        total_games=$((total_games + games))
      else
        echo "  ✗ $name ($config): No data"
      fi
    fi
  done

  echo ""
  echo "Total new games across all sparse configs: $total_games"

  if [ "$all_done" = true ]; then
    return 0
  else
    return 1
  fi
}

collect_selfplay_data() {
  echo ""
  echo "=== COLLECTING SELFPLAY DATA ==="

  # Use existing sync script if available
  if [ -f "scripts/sync_selfplay_data.sh" ]; then
    echo "Using existing sync_selfplay_data.sh..."
    ./scripts/sync_selfplay_data.sh --merge 2>/dev/null || true
  else
    echo "Collecting manually..."
    mkdir -p data/games/collected

    for entry in "${SELFPLAY_HOSTS[@]}"; do
      IFS=':' read -r name host config <<< "$entry"

      games=$(ssh -o ConnectTimeout=5 "$host" "sqlite3 ~/ringrift/ai-service/data/games/new_${config}.db 'SELECT COUNT(*) FROM games' 2>/dev/null" 2>/dev/null || echo "0")

      if [ "$games" != "0" ] && [ -n "$games" ]; then
        echo "  Downloading ${config}.db ($games games)..."
        scp -o ConnectTimeout=10 "${host}:~/ringrift/ai-service/data/games/new_${config}.db" \
            "data/games/collected/${config}.db" 2>/dev/null || echo "  Failed to download ${config}"
      fi
    done

    # Merge if we have collected DBs
    if ls data/games/collected/*.db 1>/dev/null 2>&1; then
      echo ""
      echo "Merging collected DBs into selfplay.db..."
      python scripts/merge_game_dbs.py \
        --output data/games/selfplay.db \
        --dedupe-by-game-id \
        --db data/games/collected/*.db 2>/dev/null || echo "Merge failed"
    fi
  fi
}

sync_hyperparameters() {
  echo ""
  echo "=== SYNCING OPTIMIZED HYPERPARAMETERS ==="

  # Collect HP results from Lambda hosts and update local config
  for entry in "${HP_HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"

    # Download the tuning session results
    session_file="logs/hp_tuning/${config}/tuning_session.json"
    mkdir -p "logs/hp_tuning/${config}"

    if scp -o ConnectTimeout=10 "${host}:~/ringrift/ai-service/${session_file}" "$session_file" 2>/dev/null; then
      echo "  Downloaded $config tuning results"
    fi
  done

  # Update hyperparameters.json with best results
  if [ -f "scripts/update_hyperparameters_from_tuning.py" ]; then
    python scripts/update_hyperparameters_from_tuning.py
  else
    echo "  Note: Run manual HP update or create update script"
  fi
}

trigger_curriculum_training() {
  echo ""
  echo "=== TRIGGERING CURRICULUM TRAINING ==="

  if [ -f "scripts/curriculum_training.py" ]; then
    echo "Starting curriculum training with optimized hyperparameters..."
    python scripts/curriculum_training.py --auto-progress --db "data/games/selfplay.db" &
    echo "Curriculum training started in background (PID: $!)"
  else
    echo "curriculum_training.py not found"
  fi
}

run_once() {
  echo "=============================================="
  echo "  HP TUNING PIPELINE STATUS"
  echo "  $(date)"
  echo "=============================================="

  hp_done=false
  selfplay_done=false

  if check_hp_tuning; then
    hp_done=true
  fi

  if check_selfplay; then
    selfplay_done=true
  fi

  # Actions based on completion status
  if [ "$selfplay_done" = true ]; then
    collect_selfplay_data
  fi

  if [ "$hp_done" = true ]; then
    sync_hyperparameters

    if [ "$TRIGGER_TRAINING" = true ]; then
      trigger_curriculum_training
    else
      echo ""
      echo "HP tuning complete. To trigger training, run:"
      echo "  python scripts/curriculum_training.py --auto-progress"
    fi
  fi

  echo ""
  echo "=============================================="
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
  while true; do
    clear
    run_once
    echo ""
    echo "Refreshing in 5 minutes... (Ctrl+C to stop)"
    sleep 300
  done
else
  run_once
fi
