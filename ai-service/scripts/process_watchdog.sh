#!/bin/bash
# process_watchdog.sh - Monitor and limit selfplay process counts
#
# Prevents runaway process accumulation that can overload nodes.
# Run via cron: */5 * * * * /path/to/process_watchdog.sh >> /var/log/process_watchdog.log 2>&1
#
# Usage:
#   ./scripts/process_watchdog.sh [OPTIONS]
#
# Options:
#   --max-python MAX   Maximum Python processes before alert (default: 50)
#   --max-load MAX     Maximum load average before action (default: 100)
#   --kill             Kill excess processes (default: alert only)
#   --slack URL        Send alerts to Slack webhook
#   -h, --help         Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Defaults
MAX_PYTHON_PROCS=50
MAX_LOAD=100
KILL_MODE=false
SLACK_WEBHOOK=""

# Detect hardware capabilities
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "8")
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
MEMORY_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "16")

# Scale limits based on hardware
# High-core machines (192+ cores like GH200) get much higher limits
if [[ "$CPU_CORES" -ge 192 ]]; then
    MACHINE_TIER="high-end"
    MAX_CPU_SELFPLAY=$((CPU_CORES * 2))  # Up to 2x cores for high-end
    MAX_GPU_SELFPLAY=16
    MAX_PYTHON_PROCS=$((CPU_CORES * 3))
elif [[ "$CPU_CORES" -ge 64 ]]; then
    MACHINE_TIER="medium"
    MAX_CPU_SELFPLAY=$((CPU_CORES))  # 1x cores for medium
    MAX_GPU_SELFPLAY=8
    MAX_PYTHON_PROCS=$((CPU_CORES * 2))
else
    MACHINE_TIER="standard"
    MAX_CPU_SELFPLAY=$((CPU_CORES * 2))  # 2x cores for small machines
    MAX_GPU_SELFPLAY=4
    MAX_PYTHON_PROCS=$((CPU_CORES * 4))
fi

# Orchestrators: should be exactly 1 per node
MAX_ORCHESTRATORS=2
# Training: typically 1-2 per node, scale with memory
MAX_TRAINING=$((MEMORY_GB / 32 + 1))
[[ "$MAX_TRAINING" -lt 2 ]] && MAX_TRAINING=2
[[ "$MAX_TRAINING" -gt 8 ]] && MAX_TRAINING=8

# State file for tracking kills and implementing backoff
STATE_DIR="${HOME}/.ringrift"
STATE_FILE="${STATE_DIR}/watchdog_state.json"
mkdir -p "$STATE_DIR"

# Backoff settings - progressively reduce kills over time
# Tier 1: First few kills use short backoff
# Tier 2: After multiple kills, extend to 1 per hour max
# Tier 3: After repeated issues, stop killing entirely (alert only)
MAX_KILLS_PER_HOUR=2
BASE_BACKOFF_MINUTES=15
# Extended tracking windows
FOUR_HOURS_EPOCH=$(($(date +%s) - 14400))
DAY_EPOCH=$(($(date +%s) - 86400))

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-python) MAX_PYTHON_PROCS="$2"; shift 2 ;;
        --max-load) MAX_LOAD="$2"; shift 2 ;;
        --kill) KILL_MODE=true; shift ;;
        --slack) SLACK_WEBHOOK="$2"; shift 2 ;;
        --help|-h)
            head -20 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get current metrics
PYTHON_COUNT=$(pgrep -c python 2>/dev/null || echo "0")
LOAD_1M=$(awk '{print $1}' /proc/loadavg 2>/dev/null || uptime | awk -F'[, ]+' '{print $(NF-2)}')

# Convert load to integer for comparison
LOAD_INT=${LOAD_1M%.*}

send_alert() {
    local message="$1"
    local severity="$2"

    echo "[$TIMESTAMP] [$severity] $message"

    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local color="warning"
        [[ "$severity" == "CRITICAL" ]] && color="danger"
        [[ "$severity" == "OK" ]] && color="good"

        curl -s -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Process Watchdog: $severity\",
                    \"text\": \"$message\",
                    \"footer\": \"$HOSTNAME | $TIMESTAMP\"
                }]
            }" >/dev/null 2>&1 || true
    fi
}

# Check for problems
PROBLEMS=()

if [[ "$PYTHON_COUNT" -gt "$MAX_PYTHON_PROCS" ]]; then
    PROBLEMS+=("Python processes: $PYTHON_COUNT (max: $MAX_PYTHON_PROCS)")
fi

if [[ "$LOAD_INT" -gt "$MAX_LOAD" ]]; then
    PROBLEMS+=("Load average: $LOAD_1M (max: $MAX_LOAD)")
fi

# Report status
if [[ ${#PROBLEMS[@]} -eq 0 ]]; then
    echo "[$TIMESTAMP] [OK] Host: $HOSTNAME | Tier: $MACHINE_TIER (${CPU_CORES}c/${MEMORY_GB}G) | Python: $PYTHON_COUNT (max:$MAX_PYTHON_PROCS) | Load: $LOAD_1M"
    exit 0
fi

# Problems detected
PROBLEM_MSG="${PROBLEMS[*]}"
send_alert "$HOSTNAME: $PROBLEM_MSG" "WARNING"

# Kill mode - try to recover
if [[ "$KILL_MODE" == "true" ]]; then
    # Check for backoff before taking action
    NOW_EPOCH=$(date +%s)
    HOUR_AGO_EPOCH=$((NOW_EPOCH - 3600))

    # Read state file
    KILL_TIMESTAMPS=""
    LAST_KILL_EPOCH=0
    if [[ -f "$STATE_FILE" ]]; then
        KILL_TIMESTAMPS=$(grep -oE '"[0-9]+"' "$STATE_FILE" 2>/dev/null | tr -d '"' || true)
        LAST_KILL_EPOCH=$(echo "$KILL_TIMESTAMPS" | sort -rn | head -1)
        LAST_KILL_EPOCH=${LAST_KILL_EPOCH:-0}
    fi

    # Count kills in different time windows
    KILLS_IN_HOUR=0
    KILLS_IN_4H=0
    KILLS_IN_DAY=0
    for ts in $KILL_TIMESTAMPS; do
        if [[ "$ts" -gt "$HOUR_AGO_EPOCH" ]]; then
            KILLS_IN_HOUR=$((KILLS_IN_HOUR + 1))
        fi
        if [[ "$ts" -gt "$FOUR_HOURS_EPOCH" ]]; then
            KILLS_IN_4H=$((KILLS_IN_4H + 1))
        fi
        if [[ "$ts" -gt "$DAY_EPOCH" ]]; then
            KILLS_IN_DAY=$((KILLS_IN_DAY + 1))
        fi
    done

    SECONDS_SINCE_LAST=$((NOW_EPOCH - LAST_KILL_EPOCH))

    # Tier 3: After 6+ kills in 24 hours, stop killing entirely (alert only mode)
    if [[ "$KILLS_IN_DAY" -ge 6 ]]; then
        echo "[$TIMESTAMP] [TIER3-DISABLED] Too many kills in 24h ($KILLS_IN_DAY). Kill mode disabled, alert only."
        send_alert "$HOSTNAME: Watchdog disabled after $KILLS_IN_DAY kills in 24h. Manual intervention required." "CRITICAL"
        exit 0
    fi

    # Tier 2: After 4+ kills in 4 hours, only allow 1 kill per hour with 1h minimum gap
    if [[ "$KILLS_IN_4H" -ge 4 ]]; then
        if [[ "$SECONDS_SINCE_LAST" -lt 3600 ]]; then
            WAIT_MORE=$((3600 - SECONDS_SINCE_LAST))
            echo "[$TIMESTAMP] [TIER2-BACKOFF] $KILLS_IN_4H kills in 4h. Max 1/hour. Waiting ${WAIT_MORE}s."
            exit 0
        fi
    fi

    # Tier 1: Normal backoff - exponential with 15 minute base
    # After 2 kills/hour, require increasingly long waits
    BACKOFF_SECONDS=$((BASE_BACKOFF_MINUTES * 60 * (2 ** KILLS_IN_HOUR)))
    BACKOFF_SECONDS=$((BACKOFF_SECONDS > 7200 ? 7200 : BACKOFF_SECONDS))  # Cap at 2 hours

    if [[ "$KILLS_IN_HOUR" -ge "$MAX_KILLS_PER_HOUR" ]] && [[ "$SECONDS_SINCE_LAST" -lt "$BACKOFF_SECONDS" ]]; then
        WAIT_MORE=$((BACKOFF_SECONDS - SECONDS_SINCE_LAST))
        echo "[$TIMESTAMP] [TIER1-BACKOFF] $KILLS_IN_HOUR kills in hour. Waiting ${WAIT_MORE}s (backoff=${BACKOFF_SECONDS}s)."
        exit 0
    fi

    echo "[$TIMESTAMP] [ACTION] Classifying processes... (kills_in_hour=$KILLS_IN_HOUR)"

    # Classify all Python processes by type
    ORCHESTRATOR_PIDS=""
    TRAINING_PIDS=""
    GPU_SELFPLAY_PIDS=""
    CPU_SELFPLAY_PIDS=""
    OTHER_PIDS=""

    for pid in $(pgrep python 2>/dev/null); do
        CMDLINE=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || true)
        if [[ -z "$CMDLINE" ]]; then
            continue
        fi

        if echo "$CMDLINE" | grep -qE "p2p_orchestrator|unified_ai_loop"; then
            ORCHESTRATOR_PIDS="$ORCHESTRATOR_PIDS $pid"
        elif echo "$CMDLINE" | grep -qE "train_nnue|run_nn_training|improvement_cycle"; then
            TRAINING_PIDS="$TRAINING_PIDS $pid"
        elif echo "$CMDLINE" | grep -qE "run_gpu_selfplay"; then
            GPU_SELFPLAY_PIDS="$GPU_SELFPLAY_PIDS $pid"
        elif echo "$CMDLINE" | grep -qE "run_self_play|run_hybrid_selfplay|selfplay"; then
            CPU_SELFPLAY_PIDS="$CPU_SELFPLAY_PIDS $pid"
        else
            OTHER_PIDS="$OTHER_PIDS $pid"
        fi
    done

    # Count each type
    ORCH_COUNT=$(echo $ORCHESTRATOR_PIDS | wc -w)
    TRAIN_COUNT=$(echo $TRAINING_PIDS | wc -w)
    GPU_SP_COUNT=$(echo $GPU_SELFPLAY_PIDS | wc -w)
    CPU_SP_COUNT=$(echo $CPU_SELFPLAY_PIDS | wc -w)
    OTHER_COUNT=$(echo $OTHER_PIDS | wc -w)

    echo "[$TIMESTAMP] [CLASSIFY] orchestrators=$ORCH_COUNT training=$TRAIN_COUNT gpu_selfplay=$GPU_SP_COUNT cpu_selfplay=$CPU_SP_COUNT other=$OTHER_COUNT"

    # NEVER kill orchestrators or training - these are critical
    KILLED=0

    # Kill excess GPU selfplay (beyond MAX_GPU_SELFPLAY)
    if [[ "$GPU_SP_COUNT" -gt "$MAX_GPU_SELFPLAY" ]]; then
        EXCESS=$((GPU_SP_COUNT - MAX_GPU_SELFPLAY))
        echo "[$TIMESTAMP] [ACTION] Killing $EXCESS excess GPU selfplay processes (limit: $MAX_GPU_SELFPLAY)..."
        echo $GPU_SELFPLAY_PIDS | tr ' ' '\n' | tail -n $EXCESS | xargs -r kill -9 2>/dev/null || true
        KILLED=$((KILLED + EXCESS))
    fi

    # Kill excess CPU selfplay (beyond MAX_CPU_SELFPLAY)
    if [[ "$CPU_SP_COUNT" -gt "$MAX_CPU_SELFPLAY" ]]; then
        EXCESS=$((CPU_SP_COUNT - MAX_CPU_SELFPLAY))
        echo "[$TIMESTAMP] [ACTION] Killing $EXCESS excess CPU selfplay processes (limit: $MAX_CPU_SELFPLAY)..."
        echo $CPU_SELFPLAY_PIDS | tr ' ' '\n' | tail -n $EXCESS | xargs -r kill -9 2>/dev/null || true
        KILLED=$((KILLED + EXCESS))
    fi

    if [[ "$KILLED" -gt 0 ]]; then
        send_alert "$HOSTNAME: Killed $KILLED excess selfplay processes (GPU:$GPU_SP_COUNT->$MAX_GPU_SELFPLAY, CPU:$CPU_SP_COUNT->$MAX_CPU_SELFPLAY). Protected: $ORCH_COUNT orchestrators, $TRAIN_COUNT training." "ACTION"
    else
        echo "[$TIMESTAMP] [OK] All process counts within limits, no action taken"
    fi

    # Alert if load is still high but don't kill critical processes
    sleep 3
    NEW_LOAD=$(awk '{print $1}' /proc/loadavg 2>/dev/null || uptime | awk -F'[, ]+' '{print $(NF-2)}')
    NEW_LOAD_INT=${NEW_LOAD%.*}

    if [[ "$NEW_LOAD_INT" -gt "$MAX_LOAD" ]]; then
        echo "[$TIMESTAMP] [HIGH-LOAD] Load still high ($NEW_LOAD). Protected: $ORCH_COUNT orchestrators, $TRAIN_COUNT training jobs."
        send_alert "$HOSTNAME: HIGH LOAD ($NEW_LOAD) after cleanup. Orchestrators ($ORCH_COUNT) and training ($TRAIN_COUNT) protected." "WARNING"
    fi

    # Record this kill action for backoff tracking
    # Keep timestamps from the past 24 hours for tiered backoff
    NEW_TIMESTAMPS="$NOW_EPOCH"
    for ts in $KILL_TIMESTAMPS; do
        if [[ "$ts" -gt "$DAY_EPOCH" ]]; then
            NEW_TIMESTAMPS="$NEW_TIMESTAMPS $ts"
        fi
    done
    echo "{\"kills\": [$(echo $NEW_TIMESTAMPS | tr ' ' ',')]}" > "$STATE_FILE"
    echo "[$TIMESTAMP] [STATE] Recorded kill at $NOW_EPOCH (hour=$((KILLS_IN_HOUR + 1)), 4h=$((KILLS_IN_4H + 1)), day=$((KILLS_IN_DAY + 1)))"
fi

# Exit with error code to indicate problems were found
exit 1
