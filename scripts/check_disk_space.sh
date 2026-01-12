#!/bin/bash
# Alert if disk usage exceeds 85%
USAGE=$(df -h /System/Volumes/Data | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
if [ "$USAGE" -gt 85 ]; then
    echo "WARNING: Disk usage at ${USAGE}%"
    echo "Large directories:"
    du -sh ~/.claude 2>/dev/null
    du -sh ~/Development/RingRift/ai-service/data 2>/dev/null
    du -sh ~/Development/RingRift/ai-service/models 2>/dev/null
fi
