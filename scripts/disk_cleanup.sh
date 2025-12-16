#\!/bin/bash
# Automated Disk Cleanup Script
# Runs daily to maintain disk below 80%

LOG_FILE="/var/log/disk_cleanup.log"
THRESHOLD=80
GAMES_DIR="/home/ubuntu/ringrift/ai-service/data/games"
CORRUPTED_DIR="/home/ubuntu/ringrift/ai-service/data/corrupted"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Disk Cleanup Started"

# Get current disk usage
DISK_USAGE=$(df -h /home | awk "NR==2 {gsub(/%/,\"\"); print \$5}")
log "Current disk usage: ${DISK_USAGE}%"

if [ "$DISK_USAGE" -lt "$THRESHOLD" ]; then
    log "Disk usage below threshold ($THRESHOLD%), no cleanup needed"
    exit 0
fi

log "Disk usage above threshold, starting cleanup..."

# 1. Remove corrupted files
if [ -d "$CORRUPTED_DIR" ]; then
    SIZE=$(du -sh "$CORRUPTED_DIR" 2>/dev/null | cut -f1)
    log "Removing corrupted directory ($SIZE)..."
    rm -rf "$CORRUPTED_DIR"
fi

# 2. Remove old log files (>7 days)
log "Cleaning old logs..."
find /home/ubuntu/ringrift/ai-service/logs -name "*.log" -mtime +7 -delete 2>/dev/null
find /var/log -name "*.log.*" -mtime +7 -delete 2>/dev/null

# 3. Clean Python cache
log "Cleaning Python cache..."
find /home/ubuntu/ringrift -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/ubuntu/ringrift -name "*.pyc" -delete 2>/dev/null

# 4. Remove empty game databases
log "Removing empty game databases..."
find "$GAMES_DIR" -name "*.db" -size 0 -delete 2>/dev/null

# 5. Compress old game databases (>30 days, not already compressed)
log "Compressing old databases..."
find "$GAMES_DIR" -name "*.db" -mtime +30 \! -name "*.gz" -exec gzip {} \; 2>/dev/null

# Final disk usage
DISK_USAGE_AFTER=$(df -h /home | awk "NR==2 {gsub(/%/,\"\"); print \$5}")
log "Disk usage after cleanup: ${DISK_USAGE_AFTER}%"
log "Freed: $((DISK_USAGE - DISK_USAGE_AFTER))%"
log "Cleanup complete"
