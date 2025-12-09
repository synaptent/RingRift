#!/usr/bin/env bash
#
# Distributed self-play soak matrix runner.
#
# Runs self-play soaks across multiple machines via SSH.
# Each machine gets a subset of board/player combinations.
#
# Usage (from ai-service/):
#   chmod +x scripts/run_distributed_selfplay_matrix.sh
#   PYTHONPATH=. scripts/run_distributed_selfplay_matrix.sh [MODE]
#
# Modes:
#   lan      - Local Mac cluster only (default)
#   aws      - AWS staging only (suitable for square8 games only - 16GB RAM limit)
#   hybrid   - Both LAN and AWS workers (maximum parallelism)
#   local    - Localhost only (no remote workers)
#
# Environment variables:
#   CLUSTER_WORKERS_FILE - path to workers file (default: scripts/cluster_workers.txt)
#   LOCAL_ONLY           - set to 1 to skip remote workers (overrides mode)
#   REMOTE_PROJECT_DIR   - project dir on remote machines (default: ~/Development/RingRift)
#   AWS_PROJECT_DIR      - project dir on AWS (default: /home/ubuntu/ringrift)
#   AWS_SSH_KEY          - SSH key for AWS (default: ~/.ssh/ringrift-staging-key.pem)
#
# All other variables from run_selfplay_matrix.sh are supported.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Parse mode argument
MODE="${1:-lan}"

CLUSTER_WORKERS_FILE="${CLUSTER_WORKERS_FILE:-${SCRIPT_DIR}/cluster_workers.txt}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Development/RingRift}"

# AWS configuration
AWS_HOST="3.236.54.231"
AWS_USER="ubuntu"
AWS_PROJECT_DIR="${AWS_PROJECT_DIR:-/home/ubuntu/ringrift}"
AWS_SSH_KEY="${AWS_SSH_KEY:-~/.ssh/ringrift-staging-key.pem}"

# Limit threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Load worker list based on mode
declare -a WORKERS=()
declare -A WORKER_PROJECT_DIRS=()
declare -A WORKER_SSH_KEYS=()

case "$MODE" in
    local)
        WORKERS+=("localhost")
        ;;
    lan)
        WORKERS+=("localhost")
        if [[ "${LOCAL_ONLY}" != "1" ]] && [[ -f "${CLUSTER_WORKERS_FILE}" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
                line="${line%%#*}"
                line="${line//[[:space:]]/}"
                # Skip AWS entries (those with :8766 port)
                [[ "$line" == *":8766" ]] && continue
                [[ -n "$line" ]] && WORKERS+=("$line")
            done < "${CLUSTER_WORKERS_FILE}"
        fi
        ;;
    aws)
        WORKERS+=("localhost")
        WORKERS+=("${AWS_USER}@${AWS_HOST}")
        WORKER_PROJECT_DIRS["${AWS_USER}@${AWS_HOST}"]="${AWS_PROJECT_DIR}"
        WORKER_SSH_KEYS["${AWS_USER}@${AWS_HOST}"]="${AWS_SSH_KEY}"
        ;;
    hybrid)
        WORKERS+=("localhost")
        # Add LAN workers
        if [[ "${LOCAL_ONLY}" != "1" ]] && [[ -f "${CLUSTER_WORKERS_FILE}" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
                line="${line%%#*}"
                line="${line//[[:space:]]/}"
                [[ "$line" == *":8766" ]] && continue
                [[ -n "$line" ]] && WORKERS+=("$line")
            done < "${CLUSTER_WORKERS_FILE}"
        fi
        # Add AWS worker
        WORKERS+=("${AWS_USER}@${AWS_HOST}")
        WORKER_PROJECT_DIRS["${AWS_USER}@${AWS_HOST}"]="${AWS_PROJECT_DIR}"
        WORKER_SSH_KEYS["${AWS_USER}@${AWS_HOST}"]="${AWS_SSH_KEY}"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [local|lan|aws|hybrid]"
        exit 1
        ;;
esac

NUM_WORKERS="${#WORKERS[@]}"
echo "Distributed selfplay matrix"
echo "  Mode: ${MODE}"
echo "  Workers (${NUM_WORKERS}):"
for w in "${WORKERS[@]}"; do
    echo "    - ${w}"
done
echo

# Build job list (board x players combinations)
# NOTE: hexagonal excluded due to MPS crash
BOARD_TYPES=(square8 square19)
PLAYER_COUNTS=(2 3 4)

declare -a JOBS=()
for board in "${BOARD_TYPES[@]}"; do
    for players in "${PLAYER_COUNTS[@]}"; do
        JOBS+=("${board}:${players}")
    done
done

NUM_JOBS="${#JOBS[@]}"
echo "Total jobs: ${NUM_JOBS}"
echo

# Game counts and max moves (forwarded to each worker)
GAMES_2P="${GAMES_2P:-5}"
GAMES_3P="${GAMES_3P:-3}"
GAMES_4P="${GAMES_4P:-2}"

SQUARE8_MAX_MOVES_2P="${SQUARE8_MAX_MOVES_2P:-150}"
SQUARE8_MAX_MOVES_3P="${SQUARE8_MAX_MOVES_3P:-200}"
SQUARE8_MAX_MOVES_4P="${SQUARE8_MAX_MOVES_4P:-250}"

SQUARE19_MAX_MOVES_2P="${SQUARE19_MAX_MOVES_2P:-350}"
SQUARE19_MAX_MOVES_3P="${SQUARE19_MAX_MOVES_3P:-450}"
SQUARE19_MAX_MOVES_4P="${SQUARE19_MAX_MOVES_4P:-550}"

BASE_SEED="${BASE_SEED:-1764142864}"

LOG_DIR="logs/selfplay_matrix"
mkdir -p "${LOG_DIR}"
mkdir -p "data/games"

# Function to run a job on a worker
run_job() {
    local worker="$1"
    local board="$2"
    local players="$3"
    local job_idx="$4"

    # Get board-specific max_moves
    local board_upper
    board_upper=$(echo "${board}" | tr '[:lower:]' '[:upper:]')
    local max_moves_var="${board_upper}_MAX_MOVES_${players}P"
    local max_moves="${!max_moves_var}"

    local num_games
    case "${players}" in
        2) num_games="${GAMES_2P}" ;;
        3) num_games="${GAMES_3P}" ;;
        4) num_games="${GAMES_4P}" ;;
    esac

    local seed=$((BASE_SEED + job_idx * 100000))
    local log_jsonl="${LOG_DIR}/${board}_${players}p.mixed.light.jsonl"
    local summary_json="${LOG_DIR}/${board}_${players}p.mixed.light.summary.json"
    local record_db="data/games/selfplay_${board}_${players}p.db"

    echo "[${worker}] Starting ${board} ${players}p (${num_games} games, max_moves=${max_moves})"

    if [[ "${worker}" == "localhost" ]]; then
        # Run locally
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
    else
        # Determine project directory and SSH key for this worker
        local project_dir="${REMOTE_PROJECT_DIR}"
        local ssh_opts=""

        if [[ -n "${WORKER_PROJECT_DIRS[$worker]:-}" ]]; then
            project_dir="${WORKER_PROJECT_DIRS[$worker]}"
        fi

        if [[ -n "${WORKER_SSH_KEYS[$worker]:-}" ]]; then
            ssh_opts="-i ${WORKER_SSH_KEYS[$worker]}"
        fi

        # Run via SSH on remote worker
        ssh ${ssh_opts} "${worker}" "cd ${project_dir}/ai-service && \
            source venv/bin/activate && \
            RINGRIFT_SKIP_SHADOW_CONTRACTS=true \
            PYTHONPATH=. \
            python scripts/run_self_play_soak.py \
                --num-games ${num_games} \
                --board-type ${board} \
                --engine-mode mixed \
                --difficulty-band light \
                --num-players ${players} \
                --max-moves ${max_moves} \
                --seed ${seed} \
                --gc-interval 10 \
                --log-jsonl ${log_jsonl} \
                --summary-json ${summary_json} \
                --record-db ${record_db}"

        # Copy results back from remote (use same SSH options)
        echo "[${worker}] Copying results back..."
        local scp_opts=""
        if [[ -n "${WORKER_SSH_KEYS[$worker]:-}" ]]; then
            scp_opts="-i ${WORKER_SSH_KEYS[$worker]}"
        fi

        # Clean worker name for file suffix (replace @ and . with _)
        local worker_suffix="${worker//@/_}"
        worker_suffix="${worker_suffix//./_}"

        scp ${scp_opts} "${worker}:${project_dir}/ai-service/${log_jsonl}" "${log_jsonl}.remote_${worker_suffix}" 2>/dev/null || true
        scp ${scp_opts} "${worker}:${project_dir}/ai-service/${summary_json}" "${summary_json}.remote_${worker_suffix}" 2>/dev/null || true
        scp ${scp_opts} "${worker}:${project_dir}/ai-service/${record_db}" "${record_db}.remote_${worker_suffix}" 2>/dev/null || true
    fi

    echo "[${worker}] Completed ${board} ${players}p"
}

# Distribute jobs across workers in round-robin fashion
# Run in parallel with one background process per worker
WORKER_PIDS=""

echo "Job distribution:"
for ((w=0; w<NUM_WORKERS; w++)); do
    worker="${WORKERS[$w]}"
    # Calculate which jobs this worker gets (round-robin)
    job_indices=""
    for ((j=w; j<NUM_JOBS; j+=NUM_WORKERS)); do
        job_indices="${job_indices} ${j}"
    done
    # Display job list
    job_list=""
    for j in ${job_indices}; do
        job_list="${job_list} ${JOBS[$j]}"
    done
    echo "  ${worker}:${job_list}"
done
echo

# Run each worker's jobs in a background subshell
for ((w=0; w<NUM_WORKERS; w++)); do
    worker="${WORKERS[$w]}"

    # Calculate which jobs this worker gets (round-robin)
    (
        for ((j=w; j<NUM_JOBS; j+=NUM_WORKERS)); do
            job="${JOBS[$j]}"
            board="${job%%:*}"
            players="${job##*:}"
            run_job "${worker}" "${board}" "${players}" "${j}"
        done
    ) &
    WORKER_PIDS="${WORKER_PIDS} $!"
done

echo "Waiting for ${NUM_WORKERS} worker(s) to complete..."
echo

# Wait for all workers
all_success=true
for pid in ${WORKER_PIDS}; do
    if ! wait "${pid}"; then
        all_success=false
    fi
done

echo
if $all_success; then
    echo "Distributed selfplay matrix complete."
else
    echo "Some workers failed. Check logs for details."
    exit 1
fi
