#!/bin/bash
# RingRift Disk Cleanup Script
# Prevents disk space exhaustion from Claude Code caches and temp files

LOG_FILE="/Users/armand/Development/RingRift/ai-service/logs/disk_cleanup.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

log() {
    echo "[$TIMESTAMP] $1" >> "$LOG_FILE"
}

# Check disk usage first
USAGE=$(df -h /System/Volumes/Data | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
log "Disk usage before cleanup: ${USAGE}%"

# 1. Clean Claude debug logs (safe - just debug output)
if [ -d "$HOME/.claude/debug" ]; then
    COUNT=$(find "$HOME/.claude/debug" -type f 2>/dev/null | wc -l)
    rm -rf "$HOME/.claude/debug"/* 2>/dev/null
    log "Cleaned Claude debug logs: ~$COUNT files"
fi

# 2. Clean old Claude file-history (older than 7 days)
if [ -d "$HOME/.claude/file-history" ]; then
    COUNT=$(find "$HOME/.claude/file-history" -mtime +7 -type f 2>/dev/null | wc -l)
    find "$HOME/.claude/file-history" -mtime +7 -type f -delete 2>/dev/null
    log "Cleaned Claude file-history older than 7 days: ~$COUNT files"
fi

# 3. Clean temp model files (older than 7 days)
COUNT=$(find /tmp -name "*.pth" -mtime +7 2>/dev/null | wc -l)
find /tmp -name "*.pth" -mtime +7 -delete 2>/dev/null
log "Cleaned temp .pth files older than 7 days: ~$COUNT files"

# 4. Clean old cluster backups in /tmp (older than 3 days)
if [ -d "/tmp/cluster_backup" ]; then
    SIZE=$(du -sh /tmp/cluster_backup 2>/dev/null | cut -f1)
    find /tmp/cluster_backup -mtime +3 -delete 2>/dev/null
    log "Cleaned old cluster backups: $SIZE"
fi

# 5. Clean ringrift coordination temp files (older than 7 days)
COUNT=$(find /tmp -name "ringrift_*" -mtime +7 2>/dev/null | wc -l)
find /tmp -name "ringrift_*" -mtime +7 -type f -delete 2>/dev/null
find /tmp -name "ringrift_*" -mtime +7 -type d -exec rm -rf {} + 2>/dev/null
log "Cleaned ringrift temp files older than 7 days: ~$COUNT"

# Check disk usage after
USAGE_AFTER=$(df -h /System/Volumes/Data | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
log "Disk usage after cleanup: ${USAGE_AFTER}%"

# Alert if still high
if [ "$USAGE_AFTER" -gt 85 ]; then
    log "WARNING: Disk usage still high at ${USAGE_AFTER}%"
    echo "WARNING: Disk usage at ${USAGE_AFTER}% - manual cleanup may be needed"
    echo "Large directories:"
    du -sh ~/.claude 2>/dev/null
    du -sh ~/Development/RingRift/ai-service/data 2>/dev/null
    du -sh ~/Development/RingRift/ai-service/models 2>/dev/null
fi

log "Cleanup complete"
