#!/bin/bash
# Vast P2P Maintenance Cron Job
# Add to crontab: */10 * * * * /path/to/cron_vast_maintenance.sh >> /path/to/logs/vast_cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$AI_SERVICE_DIR/logs/vast_cron_$(date +%Y%m%d).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$AI_SERVICE_DIR"
source venv/bin/activate 2>/dev/null || true

log "Starting Vast P2P maintenance..."

# Run full sync with SOCKS support
python3 scripts/vast_p2p_sync.py --full 2>&1 | while read line; do
    log "  $line"
done

# Check cluster health
log "Checking cluster health..."
LEADER_IPS=("100.91.25.13" "100.78.101.123" "100.104.165.116")

for ip in "${LEADER_IPS[@]}"; do
    status=$(curl -s --connect-timeout 5 "http://$ip:8770/status" 2>/dev/null) || continue
    
    if [ -n "$status" ]; then
        echo "$status" | python3 -c "
import sys, json
from collections import Counter
d = json.load(sys.stdin)
peers = d.get('peers', {})
vast = sum(1 for p in peers if 'vast' in p.lower())
jobs = sum(p.get('selfplay_jobs', 0) for p in peers.values())
print(f'Cluster: {len(peers)} peers, {vast} vast, {jobs} jobs, leader={d.get(\"leader_id\", \"unknown\")}')
" 2>/dev/null && break
    fi
done

log "Vast P2P maintenance complete"
