#!/bin/bash
# Scale selfplay workers across multiple parallel processes
# This script launches parallel selfplay workers for efficient data collection
#
# Usage:
#   ./scripts/scale_selfplay_workers.sh [config_file]
#   ./scripts/scale_selfplay_workers.sh config/selfplay_workers.yaml
#
# Or with command line arguments:
#   ./scripts/scale_selfplay_workers.sh --2p-workers 50 --3p-workers 20 --4p-workers 10
#
# For Vast/cloud instances with limited disk but ample RAM, use --ram-storage
# to store databases and logs in /dev/shm (tmpfs):
#   ./scripts/scale_selfplay_workers.sh --ram-storage --2p-workers 80

set -e

# Default configuration
NUM_2P_WORKERS=${SELFPLAY_2P_WORKERS:-10}
NUM_3P_WORKERS=${SELFPLAY_3P_WORKERS:-5}
NUM_4P_WORKERS=${SELFPLAY_4P_WORKERS:-3}
SEED_BASE=${SELFPLAY_SEED_BASE:-1000}
BOARD_TYPE=${SELFPLAY_BOARD_TYPE:-square8}
ENGINE_MODE=${SELFPLAY_ENGINE_MODE:-descent-only}
GAMES_PER_WORKER_2P=${SELFPLAY_GAMES_2P:-100}
GAMES_PER_WORKER_3P=${SELFPLAY_GAMES_3P:-80}
GAMES_PER_WORKER_4P=${SELFPLAY_GAMES_4P:-50}
MAX_MOVES_2P=${SELFPLAY_MAX_MOVES_2P:-500}
MAX_MOVES_3P=${SELFPLAY_MAX_MOVES_3P:-600}
MAX_MOVES_4P=${SELFPLAY_MAX_MOVES_4P:-700}
USE_RAM_STORAGE=false
DISABLE_NEURAL_NET=${SELFPLAY_DISABLE_NNUE:-false}

# Output directories (can be overridden for RAM storage)
DATA_DIR="data/games"
LOG_DIR="logs/selfplay"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --2p-workers N      Number of 2-player workers (default: $NUM_2P_WORKERS)"
    echo "  --3p-workers N      Number of 3-player workers (default: $NUM_3P_WORKERS)"
    echo "  --4p-workers N      Number of 4-player workers (default: $NUM_4P_WORKERS)"
    echo "  --seed-base N       Base seed for random number generation (default: $SEED_BASE)"
    echo "  --board TYPE        Board type: square8, square19, hexagonal (default: $BOARD_TYPE)"
    echo "  --engine MODE       Engine mode: descent-only, mixed (default: $ENGINE_MODE)"
    echo "  --ram-storage       Use /dev/shm for databases (for disk-constrained instances)"
    echo "  --disable-nnue      Disable NNUE neural net evaluation"
    echo "  --kill-existing     Kill existing selfplay workers before starting"
    echo "  --config FILE       Load configuration from YAML/JSON file"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment variables (override defaults):"
    echo "  SELFPLAY_2P_WORKERS, SELFPLAY_3P_WORKERS, SELFPLAY_4P_WORKERS"
    echo "  SELFPLAY_SEED_BASE, SELFPLAY_BOARD_TYPE, SELFPLAY_ENGINE_MODE"
    echo "  SELFPLAY_GAMES_2P, SELFPLAY_GAMES_3P, SELFPLAY_GAMES_4P"
    echo "  SELFPLAY_DISABLE_NNUE"
}

# Parse command line arguments
KILL_EXISTING=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --2p-workers)
            NUM_2P_WORKERS="$2"
            shift 2
            ;;
        --3p-workers)
            NUM_3P_WORKERS="$2"
            shift 2
            ;;
        --4p-workers)
            NUM_4P_WORKERS="$2"
            shift 2
            ;;
        --seed-base)
            SEED_BASE="$2"
            shift 2
            ;;
        --board)
            BOARD_TYPE="$2"
            shift 2
            ;;
        --engine)
            ENGINE_MODE="$2"
            shift 2
            ;;
        --ram-storage)
            USE_RAM_STORAGE=true
            shift
            ;;
        --disable-nnue)
            DISABLE_NEURAL_NET=true
            shift
            ;;
        --kill-existing)
            KILL_EXISTING=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Load config file if specified
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "Loading configuration from $CONFIG_FILE"
    # Simple YAML/env parser (for basic key: value or KEY=VALUE formats)
    while IFS=': =' read -r key value; do
        key=$(echo "$key" | tr -d ' ' | tr '[:lower:]' '[:upper:]')
        value=$(echo "$value" | tr -d ' "'"'"'')
        case $key in
            2P_WORKERS|SELFPLAY_2P_WORKERS) NUM_2P_WORKERS="$value" ;;
            3P_WORKERS|SELFPLAY_3P_WORKERS) NUM_3P_WORKERS="$value" ;;
            4P_WORKERS|SELFPLAY_4P_WORKERS) NUM_4P_WORKERS="$value" ;;
            SEED_BASE|SELFPLAY_SEED_BASE) SEED_BASE="$value" ;;
            BOARD_TYPE|SELFPLAY_BOARD_TYPE) BOARD_TYPE="$value" ;;
        esac
    done < "$CONFIG_FILE"
fi

# Configure RAM storage if requested
if [[ "$USE_RAM_STORAGE" == true ]]; then
    if [[ -d /dev/shm ]]; then
        DATA_DIR="/dev/shm/games"
        LOG_DIR="/dev/shm/logs"
        echo "Using RAM-backed storage at /dev/shm"
    else
        echo "Warning: /dev/shm not available, falling back to disk storage"
    fi
fi

# Kill existing workers if requested
if [[ "$KILL_EXISTING" == true ]]; then
    echo "Killing existing selfplay workers..."
    pkill -f run_self_play_soak || true
    sleep 2
fi

# Create directories
mkdir -p "$DATA_DIR" "$LOG_DIR"

# Set environment variables
export PYTHONPATH="${PYTHONPATH:-.}"
if [[ "$DISABLE_NEURAL_NET" == true ]]; then
    export RINGRIFT_DISABLE_NEURAL_NET=1
fi

echo "=== Selfplay Worker Configuration ==="
echo "Board type:      $BOARD_TYPE"
echo "Engine mode:     $ENGINE_MODE"
echo "2P workers:      $NUM_2P_WORKERS (${GAMES_PER_WORKER_2P} games each)"
echo "3P workers:      $NUM_3P_WORKERS (${GAMES_PER_WORKER_3P} games each)"
echo "4P workers:      $NUM_4P_WORKERS (${GAMES_PER_WORKER_4P} games each)"
echo "Data directory:  $DATA_DIR"
echo "Log directory:   $LOG_DIR"
echo "NNUE disabled:   $DISABLE_NEURAL_NET"
echo ""

# Launch 2-player workers
if [[ $NUM_2P_WORKERS -gt 0 ]]; then
    echo "Starting $NUM_2P_WORKERS 2-player workers..."
    for i in $(seq 1 $NUM_2P_WORKERS); do
        nohup python3 scripts/run_self_play_soak.py \
            --num-games "$GAMES_PER_WORKER_2P" \
            --board-type "$BOARD_TYPE" \
            --engine-mode "$ENGINE_MODE" \
            --num-players 2 \
            --max-moves "$MAX_MOVES_2P" \
            --seed $((SEED_BASE + i)) \
            --record-db "$DATA_DIR/${BOARD_TYPE}_2p_$i.db" \
            --log-jsonl "$LOG_DIR/${BOARD_TYPE}_2p_$i.jsonl" \
            > "$LOG_DIR/${BOARD_TYPE}_2p_$i.log" 2>&1 &
    done
fi

# Launch 3-player workers
if [[ $NUM_3P_WORKERS -gt 0 ]]; then
    echo "Starting $NUM_3P_WORKERS 3-player workers..."
    for i in $(seq 1 $NUM_3P_WORKERS); do
        nohup python3 scripts/run_self_play_soak.py \
            --num-games "$GAMES_PER_WORKER_3P" \
            --board-type "$BOARD_TYPE" \
            --engine-mode "$ENGINE_MODE" \
            --num-players 3 \
            --max-moves "$MAX_MOVES_3P" \
            --seed $((SEED_BASE + 1000 + i)) \
            --record-db "$DATA_DIR/${BOARD_TYPE}_3p_$i.db" \
            --log-jsonl "$LOG_DIR/${BOARD_TYPE}_3p_$i.jsonl" \
            > "$LOG_DIR/${BOARD_TYPE}_3p_$i.log" 2>&1 &
    done
fi

# Launch 4-player workers
if [[ $NUM_4P_WORKERS -gt 0 ]]; then
    echo "Starting $NUM_4P_WORKERS 4-player workers..."
    for i in $(seq 1 $NUM_4P_WORKERS); do
        nohup python3 scripts/run_self_play_soak.py \
            --num-games "$GAMES_PER_WORKER_4P" \
            --board-type "$BOARD_TYPE" \
            --engine-mode "$ENGINE_MODE" \
            --num-players 4 \
            --max-moves "$MAX_MOVES_4P" \
            --seed $((SEED_BASE + 2000 + i)) \
            --record-db "$DATA_DIR/${BOARD_TYPE}_4p_$i.db" \
            --log-jsonl "$LOG_DIR/${BOARD_TYPE}_4p_$i.jsonl" \
            > "$LOG_DIR/${BOARD_TYPE}_4p_$i.log" 2>&1 &
    done
fi

# Wait a moment for processes to start
sleep 3

# Report status
TOTAL_WORKERS=$(ps aux | grep -c run_self_play_soak | grep -v grep || echo 0)
echo ""
echo "=== Workers Launched ==="
echo "Total selfplay workers running: $TOTAL_WORKERS"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_DIR/${BOARD_TYPE}_2p_1.log"
echo ""
echo "Check worker count:"
echo "  ps aux | grep run_self_play | grep -v grep | wc -l"
