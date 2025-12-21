#!/usr/bin/env bash
#
# Sequential self-play soak matrix runner.
#
# Runs mixed-engine self-play soaks for all 12 combinations of:
#   - board_type ∈ {square8, square19, hex8, hexagonal}
#   - num_players ∈ {2, 3, 4}
# using the "light" difficulty band (Random/Heuristic/low-depth Minimax),
# and records:
#   - per-config JSONL logs + summaries under logs/selfplay_matrix/
#   - per-config replay DBs under data/games/selfplay_<board>_<players>p.db
#
# All runs are sequential (no background jobs) to keep memory usage bounded.
#
# Tunable via environment variables:
#   GAMES_2P / GAMES_3P / GAMES_4P                – games per config (defaults 5/3/2)
#   <BOARD>_MAX_MOVES_<N>P                        – board-specific max moves
#   BASE_SEED                                     – base RNG seed (default 1764142864)
#   RINGRIFT_SKIP_SHADOW_CONTRACTS                – forwarded as-is (default true)
#   SKIP_HEX_BOARDS                               – set true to skip hex boards on macOS
#
# Usage (from ai-service/):
#   chmod +x scripts/run_selfplay_matrix.sh
#   PYTHONPATH=. scripts/run_selfplay_matrix.sh
#
# For cluster (GH200/H100):
#   PYTHONPATH=. scripts/run_selfplay_matrix.sh
#
# For local macOS (skip hex due to MPS issues):
#   SKIP_HEX_BOARDS=true PYTHONPATH=. scripts/run_selfplay_matrix.sh

set -euo pipefail

# Limit threads to avoid memory exhaustion and OMP crashes on macOS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Always run relative to ai-service root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# NOTE: hexagonal had MPS adaptive pooling crash on Apple Silicon.
# This is NOT an issue on cluster (GH200/H100 Linux nodes).
# To exclude hex boards locally on macOS, set SKIP_HEX_BOARDS=true
if [[ "${SKIP_HEX_BOARDS:-false}" == "true" ]]; then
  BOARD_TYPES=(square8 square19)
else
  BOARD_TYPES=(square8 square19 hex8 hexagonal)
fi
PLAYER_COUNTS=(2 3 4)

# Conservative defaults to avoid memory exhaustion
GAMES_2P="${GAMES_2P:-5}"
GAMES_3P="${GAMES_3P:-3}"
GAMES_4P="${GAMES_4P:-2}"

# Board-specific max_moves (square19 needs much higher limits)
# These are overridable via SQUARE8_MAX_MOVES_2P, SQUARE19_MAX_MOVES_2P, etc.
SQUARE8_MAX_MOVES_2P="${SQUARE8_MAX_MOVES_2P:-150}"
SQUARE8_MAX_MOVES_3P="${SQUARE8_MAX_MOVES_3P:-200}"
SQUARE8_MAX_MOVES_4P="${SQUARE8_MAX_MOVES_4P:-250}"

SQUARE19_MAX_MOVES_2P="${SQUARE19_MAX_MOVES_2P:-350}"
SQUARE19_MAX_MOVES_3P="${SQUARE19_MAX_MOVES_3P:-450}"
SQUARE19_MAX_MOVES_4P="${SQUARE19_MAX_MOVES_4P:-550}"

# Hex8 (radius 8 hexagonal) - similar complexity to square8
HEX8_MAX_MOVES_2P="${HEX8_MAX_MOVES_2P:-200}"
HEX8_MAX_MOVES_3P="${HEX8_MAX_MOVES_3P:-250}"
HEX8_MAX_MOVES_4P="${HEX8_MAX_MOVES_4P:-300}"

# Hexagonal (standard 469-space board) - long games, needs high limits
HEXAGONAL_MAX_MOVES_2P="${HEXAGONAL_MAX_MOVES_2P:-800}"
HEXAGONAL_MAX_MOVES_3P="${HEXAGONAL_MAX_MOVES_3P:-1000}"
HEXAGONAL_MAX_MOVES_4P="${HEXAGONAL_MAX_MOVES_4P:-1200}"

BASE_SEED="${BASE_SEED:-1764142864}"

LOG_DIR="logs/selfplay_matrix"
mkdir -p "${LOG_DIR}"
mkdir -p "data/games"

echo "Starting self-play soak matrix..."
echo "  Boards:    ${BOARD_TYPES[*]}"
echo "  Players:   ${PLAYER_COUNTS[*]}"
echo "  Games:     2p=${GAMES_2P}, 3p=${GAMES_3P}, 4p=${GAMES_4P}"
echo "  Max moves:"
echo "    square8:    2p=${SQUARE8_MAX_MOVES_2P} 3p=${SQUARE8_MAX_MOVES_3P} 4p=${SQUARE8_MAX_MOVES_4P}"
echo "    square19:   2p=${SQUARE19_MAX_MOVES_2P} 3p=${SQUARE19_MAX_MOVES_3P} 4p=${SQUARE19_MAX_MOVES_4P}"
echo "    hex8:       2p=${HEX8_MAX_MOVES_2P} 3p=${HEX8_MAX_MOVES_3P} 4p=${HEX8_MAX_MOVES_4P}"
echo "    hexagonal:  2p=${HEXAGONAL_MAX_MOVES_2P} 3p=${HEXAGONAL_MAX_MOVES_3P} 4p=${HEXAGONAL_MAX_MOVES_4P}"
echo
echo "NOTE: hexagonal games are long (~800+ moves). Expect slow runs on large boards."
echo

idx=0
for board in "${BOARD_TYPES[@]}"; do
  for players in "${PLAYER_COUNTS[@]}"; do
    # Get board-specific max_moves
    board_upper=$(echo "${board}" | tr '[:lower:]' '[:upper:]')
    max_moves_var="${board_upper}_MAX_MOVES_${players}P"
    max_moves="${!max_moves_var}"

    case "${players}" in
      2) num_games="${GAMES_2P}" ;;
      3) num_games="${GAMES_3P}" ;;
      4) num_games="${GAMES_4P}" ;;
      *)
        echo "Unsupported player count: ${players}" >&2
        exit 1
        ;;
    esac

    seed=$((BASE_SEED + idx * 100000))
    log_jsonl="${LOG_DIR}/${board}_${players}p.mixed.light.jsonl"
    summary_json="${LOG_DIR}/${board}_${players}p.mixed.light.summary.json"
    record_db="data/games/selfplay_${board}_${players}p.db"

    echo "=== Running ${board} ${players}p (${num_games} games, max_moves=${max_moves}, seed=${seed}) ==="

    RINGRIFT_SKIP_SHADOW_CONTRACTS="${RINGRIFT_SKIP_SHADOW_CONTRACTS:-true}" \
    PYTHONPATH=. \
      python scripts/run_self_play_soak.py \
        --num-games "${num_games}" \
        --board-type "${board}" \
        --engine-mode mixed \
        --difficulty-band light \
        --num-players "${players}" \
        --max-moves "${max_moves}" \
        --seed "${seed}" \
        --gc-interval 10 \
        --log-jsonl "${log_jsonl}" \
        --summary-json "${summary_json}" \
        --record-db "${record_db}"

    echo "Completed ${board} ${players}p."
    echo

    idx=$((idx + 1))
  done
done

echo "Self-play soak matrix complete."

