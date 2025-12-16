#!/usr/bin/env bash
#
# Sequential minimal self-play matrix (RandomAI only).
#
# Uses scripts/run_minimal_selfplay.py (RandomAI, no neural net) to generate
# replay DBs for all combinations of:
#   - board_type ∈ {square8, square19, hexagonal}
#   - num_players ∈ {2, 3, 4}
#
# All runs are sequential and lightweight, with threads capped, so this should
# be safe on your Mac without triggering PyTorch/MPS or OMP crashes.
#
# Tunable via environment variables:
#   GAMES_2P / GAMES_3P / GAMES_4P           – games per config (defaults 5/4/3)
#   MAX_SQUARE8_2P / _3P / _4P              – max moves for square8 (150/200/250)
#   MAX_SQUARE19_2P / _3P / _4P             – max moves for square19 (400/500/600)
#   MAX_HEXAGONAL_2P / _3P / _4P            – max moves for hexagonal (400/500/600)
#   BASE_SEED                               – base RNG seed (default 1764142864)
#
# Usage (from ai-service/):
#   chmod +x scripts/run_minimal_selfplay_matrix.sh
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. scripts/run_minimal_selfplay_matrix.sh

set -euo pipefail

# Keep threads bounded by default; caller can override.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BOARD_TYPES=(square8 square19 hexagonal)
PLAYER_COUNTS=(2 3 4)

GAMES_2P="${GAMES_2P:-5}"
GAMES_3P="${GAMES_3P:-4}"
GAMES_4P="${GAMES_4P:-3}"

MAX_SQUARE8_2P="${MAX_SQUARE8_2P:-150}"
MAX_SQUARE8_3P="${MAX_SQUARE8_3P:-200}"
MAX_SQUARE8_4P="${MAX_SQUARE8_4P:-250}"

MAX_SQUARE19_2P="${MAX_SQUARE19_2P:-400}"
MAX_SQUARE19_3P="${MAX_SQUARE19_3P:-500}"
MAX_SQUARE19_4P="${MAX_SQUARE19_4P:-600}"

# Hexagonal boards need more moves to complete naturally
MAX_HEXAGONAL_2P="${MAX_HEXAGONAL_2P:-2000}"
MAX_HEXAGONAL_3P="${MAX_HEXAGONAL_3P:-3000}"
MAX_HEXAGONAL_4P="${MAX_HEXAGONAL_4P:-4000}"

BASE_SEED="${BASE_SEED:-1764142864}"

mkdir -p "data/games"

echo "Starting minimal self-play matrix (RandomAI only)..."
echo "  Boards:    ${BOARD_TYPES[*]}"
echo "  Players:   ${PLAYER_COUNTS[*]}"
echo "  Games 2p:  ${GAMES_2P}"
echo "  Games 3p:  ${GAMES_3P}"
echo "  Games 4p:  ${GAMES_4P}"
echo

idx=0
for board in "${BOARD_TYPES[@]}"; do
  for players in "${PLAYER_COUNTS[@]}"; do
    case "${players}" in
      2)
        num_games="${GAMES_2P}"
        case "${board}" in
          square8)   max_moves="${MAX_SQUARE8_2P}" ;;
          square19)  max_moves="${MAX_SQUARE19_2P}" ;;
          hexagonal) max_moves="${MAX_HEXAGONAL_2P}" ;;
        esac
        ;;
      3)
        num_games="${GAMES_3P}"
        case "${board}" in
          square8)   max_moves="${MAX_SQUARE8_3P}" ;;
          square19)  max_moves="${MAX_SQUARE19_3P}" ;;
          hexagonal) max_moves="${MAX_HEXAGONAL_3P}" ;;
        esac
        ;;
      4)
        num_games="${GAMES_4P}"
        case "${board}" in
          square8)   max_moves="${MAX_SQUARE8_4P}" ;;
          square19)  max_moves="${MAX_SQUARE19_4P}" ;;
          hexagonal) max_moves="${MAX_HEXAGONAL_4P}" ;;
        esac
        ;;
      *)
        echo "Unsupported player count: ${players}" >&2
        exit 1
        ;;
    esac

    seed=$((BASE_SEED + idx * 100000))
    db_path="data/games/selfplay_${board}_${players}p.db"

    echo "=== ${board} ${players}p (${num_games} games, max_moves=${max_moves}, seed=${seed}) ==="

    PYTHONPATH=. \
      python scripts/run_minimal_selfplay.py \
        --board-type "${board}" \
        --num-players "${players}" \
        --num-games "${num_games}" \
        --max-moves "${max_moves}" \
        --seed "${seed}" \
        --record-db "${db_path}"

    echo
    idx=$((idx + 1))
  done
done

echo "Minimal self-play matrix complete."

