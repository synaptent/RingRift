#!/usr/bin/env bash
# rotate_and_start.sh — Log rotation wrapper for RingRift LaunchAgent services.
#
# Rotates log files that exceed a size threshold before starting the actual
# process. Keeps at most MAX_BACKUPS rotated copies (.1, .2, .3, ...).
#
# Usage in a LaunchAgent plist (ProgramArguments):
#   /bin/bash
#   /path/to/rotate_and_start.sh
#   --log-dir /path/to/logs
#   --log-names master_loop.log,master_loop.err
#   --max-size-mb 50
#   --max-backups 3
#   --
#   /usr/bin/python3
#   /path/to/master_loop.py
#
# Everything after "--" is the actual command to exec.
#
# Can also be used standalone (no command after --) to just rotate logs:
#   /bin/bash rotate_and_start.sh --log-dir ~/logs --log-names app.log --max-size-mb 50
#
# Designed for macOS launchd where newsyslog is awkward for user agents and
# the process itself doesn't manage its own log rotation (stdout/stderr
# redirected by launchd).

set -euo pipefail

# --- Defaults ---
LOG_DIR=""
LOG_NAMES=""
MAX_SIZE_MB=50
MAX_BACKUPS=3

# --- Parse arguments ---
COMMAND_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --log-names)
            LOG_NAMES="$2"
            shift 2
            ;;
        --max-size-mb)
            MAX_SIZE_MB="$2"
            shift 2
            ;;
        --max-backups)
            MAX_BACKUPS="$2"
            shift 2
            ;;
        --)
            shift
            COMMAND_ARGS=("$@")
            break
            ;;
        *)
            echo "rotate_and_start.sh: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$LOG_DIR" ]]; then
    echo "rotate_and_start.sh: --log-dir is required" >&2
    exit 1
fi

if [[ -z "$LOG_NAMES" ]]; then
    echo "rotate_and_start.sh: --log-names is required" >&2
    exit 1
fi

# --- Rotate logs ---
MAX_SIZE_BYTES=$(( MAX_SIZE_MB * 1024 * 1024 ))

# Split comma-separated log names
IFS=',' read -ra LOGS <<< "$LOG_NAMES"

for log_name in "${LOGS[@]}"; do
    log_path="${LOG_DIR}/${log_name}"

    # Skip if the log file doesn't exist
    if [[ ! -f "$log_path" ]]; then
        continue
    fi

    # Check file size
    file_size=$(stat -f%z "$log_path" 2>/dev/null || echo 0)

    if (( file_size > MAX_SIZE_BYTES )); then
        echo "rotate_and_start.sh: rotating ${log_name} ($(( file_size / 1024 / 1024 ))MB > ${MAX_SIZE_MB}MB)"

        # Rotate existing backups: .2 -> .3, .1 -> .2, etc.
        # Start from the highest to avoid overwriting.
        for (( i = MAX_BACKUPS - 1; i >= 1; i-- )); do
            src="${log_path}.${i}"
            dst="${log_path}.$(( i + 1 ))"
            if [[ -f "$src" ]]; then
                mv "$src" "$dst"
            fi
        done

        # Current log -> .1
        mv "$log_path" "${log_path}.1"

        # Delete the oldest backup if it exceeds MAX_BACKUPS
        oldest="${log_path}.$(( MAX_BACKUPS + 1 ))"
        if [[ -f "$oldest" ]]; then
            rm "$oldest"
        fi

        echo "rotate_and_start.sh: rotated ${log_name} -> ${log_name}.1"
    fi
done

# --- Start the actual process ---
if [[ ${#COMMAND_ARGS[@]} -gt 0 ]]; then
    echo "rotate_and_start.sh: starting: ${COMMAND_ARGS[*]}"
    exec "${COMMAND_ARGS[@]}"
fi
