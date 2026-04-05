#!/usr/bin/env bash
# Install all coordinator cron jobs on mac-studio.
# Run once after deployment. Idempotent — safe to re-run.
#
# Usage: bash scripts/install_coordinator_crons.sh

set -euo pipefail

VENV_PYTHON="$HOME/Development/RingRift/ai-service/.venv/bin/python"
AI_DIR="$HOME/Development/RingRift/ai-service"

# Verify python exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: $VENV_PYTHON not found"
    exit 1
fi

# Build the cron entries
CRONS=(
    # Memory guard: kill master loop above 8GB RSS (every 5 min)
    "*/5 * * * * cd $AI_DIR && $VENV_PYTHON scripts/memory_guard.py >> /tmp/memory_guard.log 2>&1"
    # Pipeline watchdog: check infrastructure health (every 30 min)
    "*/30 * * * * cd $AI_DIR && PYTHONPATH=. $VENV_PYTHON scripts/pipeline_watchdog.py --mode p2p >> /tmp/watchdog.log 2>&1"
    # Regression test: validate full pipeline every 6 hours
    "0 */6 * * * cd $AI_DIR && PYTHONPATH=. $VENV_PYTHON scripts/regression_test_pipeline.py --quick >> /tmp/regression.log 2>&1"
)

# Get current crontab
existing=$(crontab -l 2>/dev/null || true)

# Add each cron entry if not already present
for cron in "${CRONS[@]}"; do
    # Extract the script name as a unique identifier
    script=$(echo "$cron" | grep -o 'scripts/[a-z_]*\.py')
    if echo "$existing" | grep -q "$script"; then
        echo "Already installed: $script"
    else
        existing="$existing
$cron"
        echo "Added: $script"
    fi
done

# Install updated crontab
echo "$existing" | crontab -
echo ""
echo "Installed crons:"
crontab -l | grep -E 'memory_guard|watchdog|regression'
