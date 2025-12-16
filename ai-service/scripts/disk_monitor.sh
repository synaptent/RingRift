#!/bin/bash
# Disk monitoring and cleanup script for selfplay instances
# Run via cron every 15 minutes: */15 * * * * /workspace/ringrift/ai-service/scripts/disk_monitor.sh >> /var/log/disk_monitor.log 2>&1

set -e

# Configuration
DISK_THRESHOLD=85  # Percentage threshold to trigger cleanup
SELFPLAY_DATA_DIR="/workspace/ringrift/ai-service/data/selfplay"
LOG_DIR="/workspace/ringrift/ai-service/logs"
ALERT_FILE="/tmp/disk_alert_sent"

# Get current disk usage percentage
get_disk_usage() {
    df / | tail -1 | awk '{print $5}' | sed 's/%//'
}

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Clean pip cache
clean_pip_cache() {
    log "Cleaning pip cache..."
    pip cache purge 2>/dev/null || true
    rm -rf ~/.cache/pip /tmp/pip* 2>/dev/null || true
}

# Clean Python bytecode
clean_pycache() {
    log "Cleaning Python bytecode..."
    find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find /workspace -name "*.pyc" -delete 2>/dev/null || true
}

# Clean old log files (>1 day old)
clean_old_logs() {
    log "Cleaning old log files..."
    find "$LOG_DIR" -name "*.log" -mtime +1 -delete 2>/dev/null || true
    find /workspace -name "*.log" -size +100M -delete 2>/dev/null || true
}

# Clean temporary files
clean_temp() {
    log "Cleaning temporary files..."
    rm -rf /tmp/* /var/tmp/* 2>/dev/null || true
}

# Archive and compress old selfplay data
archive_selfplay_data() {
    log "Archiving old selfplay data..."
    if [ -d "$SELFPLAY_DATA_DIR" ]; then
        for dir in "$SELFPLAY_DATA_DIR"/*/; do
            if [ -d "$dir" ]; then
                games_file="$dir/games.jsonl"
                if [ -f "$games_file" ]; then
                    # Get line count
                    lines=$(wc -l < "$games_file" 2>/dev/null || echo 0)
                    # If more than 1000 games, compress older ones
                    if [ "$lines" -gt 1000 ]; then
                        log "Compressing $games_file ($lines games)"
                        # Keep last 500 games, archive rest
                        tail -500 "$games_file" > "${games_file}.tmp"
                        head -n -500 "$games_file" | gzip >> "${games_file%.jsonl}_archived.jsonl.gz"
                        mv "${games_file}.tmp" "$games_file"
                    fi
                fi
            fi
        done
    fi
}

# Emergency cleanup - more aggressive
emergency_cleanup() {
    log "EMERGENCY: Disk at critical level, aggressive cleanup..."

    # Remove all pycache
    find / -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    # Remove old selfplay data entirely
    if [ -d "$SELFPLAY_DATA_DIR" ]; then
        # Keep only the most recent directory
        cd "$SELFPLAY_DATA_DIR"
        ls -t | tail -n +2 | xargs rm -rf 2>/dev/null || true
    fi

    # Clear apt cache
    apt-get clean 2>/dev/null || true

    # Remove old kernels/headers if any
    apt-get autoremove -y 2>/dev/null || true
}

# Main monitoring logic
main() {
    log "=== Disk Monitor Check ==="

    USAGE=$(get_disk_usage)
    log "Current disk usage: ${USAGE}%"

    if [ "$USAGE" -ge 95 ]; then
        log "CRITICAL: Disk usage at ${USAGE}%"
        emergency_cleanup

    elif [ "$USAGE" -ge "$DISK_THRESHOLD" ]; then
        log "WARNING: Disk usage at ${USAGE}% (threshold: ${DISK_THRESHOLD}%)"

        # Standard cleanup
        clean_pip_cache
        clean_pycache
        clean_old_logs
        clean_temp
        archive_selfplay_data

    else
        log "OK: Disk usage at ${USAGE}% (below threshold: ${DISK_THRESHOLD}%)"
    fi

    # Report final usage
    FINAL_USAGE=$(get_disk_usage)
    log "Final disk usage: ${FINAL_USAGE}%"

    if [ "$USAGE" -ne "$FINAL_USAGE" ]; then
        log "Freed $((USAGE - FINAL_USAGE))% disk space"
    fi
}

# Run main
main
