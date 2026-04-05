#!/bin/bash
# Launch wrapper for master_loop.py — used by com.ringrift.master-loop LaunchAgent.
# Uses .venv python (3.11) instead of system python (3.9) which can't import the codebase.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MASTER_LOOP_PROFILE="${RINGRIFT_MASTER_LOOP_PROFILE:-lean}"

resolve_python_bin() {
    local candidate
    for candidate in \
        "$AI_SERVICE_DIR/.venv/bin/python3" \
        "$AI_SERVICE_DIR/.venv/bin/python" \
        "$AI_SERVICE_DIR/venv/bin/python3" \
        "$AI_SERVICE_DIR/venv/bin/python"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    command -v python3
}

has_profile_arg() {
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--profile" || "$arg" == --profile=* ]]; then
            return 0
        fi
    done

    return 1
}

export PATH="/opt/homebrew/bin:$PATH"
cd "$AI_SERVICE_DIR"
export PYTHONPATH="$AI_SERVICE_DIR"

if ! has_profile_arg "$@"; then
    set -- --profile "$MASTER_LOOP_PROFILE" "$@"
fi

exec "$(resolve_python_bin)" scripts/master_loop.py "$@"
