#\!/bin/bash
# P2P Stability Tuning Script
# Monitors and adjusts P2P parameters for optimal stability

LOG_FILE="/home/ubuntu/ringrift/logs/p2p_tuning.log"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Get current P2P status
get_status() {
    curl -s http://localhost:8770/health 2>/dev/null
}

# Restart P2P with optimized settings
restart_p2p() {
    log "Restarting P2P service with optimized settings..."
    sudo systemctl restart ringrift-p2p
    sleep 10
}

# Main tuning loop
tune_network() {
    HEALTH=$(get_status)
    
    if [ -z "$HEALTH" ]; then
        log "ERROR: P2P service not responding"
        restart_p2p
        return 1
    fi
    
    ACTIVE=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"active_peers\", 0))")
    TOTAL=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"total_peers\", 0))")
    
    RATIO=0
    if [ "$TOTAL" -gt 0 ]; then
        RATIO=$((ACTIVE * 100 / TOTAL))
    fi
    
    log "Active peers: $ACTIVE/$TOTAL ($RATIO%)"
    
    # Trigger gossip anti-entropy if peer ratio is low
    if [ "$RATIO" -lt 50 ]; then
        log "Low peer ratio - triggering anti-entropy sync"
        curl -s -X POST http://localhost:8770/gossip/trigger-anti-entropy 2>/dev/null
    fi
    
    # Check for stale peers and attempt recovery
    log "Checking for NAT-blocked peers..."
    curl -s http://localhost:8770/status | python3 -c "
import json, sys
d = json.load(sys.stdin)
nat_blocked = 0
for name, peer in d.get(peers, {}).items():
    if peer.get(nat_blocked, False):
        nat_blocked += 1
        relay = peer.get(relay_via, none)
        print(f {name}: NAT-blocked, relay={relay})
print(fTotal NAT-blocked: {nat_blocked})
" 2>/dev/null
    
    log "Tuning complete"
}

log "========================================="
log "P2P Tuning Started"
tune_network
