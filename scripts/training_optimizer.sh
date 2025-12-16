#!/bin/bash
# Training Pipeline Optimizer
# Monitors training backlog and triggers training on capable nodes

LOG_FILE="/home/ubuntu/ringrift/logs/training_optimizer.log"
mkdir -p "$(dirname $LOG_FILE)"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Training Optimizer Started"

# Get training status
STATUS=$(curl -s http://localhost:8770/status 2>/dev/null)

if [ -z "$STATUS" ]; then
    log "ERROR: Could not get cluster status"
    exit 1
fi

# Count current training jobs
TRAINING_JOBS=$(echo "$STATUS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
total = d.get(self, {}).get(training_jobs, 0)
for p in d.get(peers, {}).values():
    total += p.get(training_jobs, 0)
print(total)
" 2>/dev/null)

log "Current training jobs: $TRAINING_JOBS"

# Get boards ready for training
READY_BOARDS=$(echo "$STATUS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
icm = d.get(improvement_cycle_manager, {})
cycles = icm.get(cycles, {})
ready = []
for board, data in cycles.items():
    if data.get(pending_training):
        since = data.get(games_since_training, 0)
        ready.append(f{board}:{since})
print(,.join(ready) if ready else none)
" 2>/dev/null)

log "Boards ready for training: $READY_BOARDS"

# Get available GH200 nodes for training
AVAILABLE_NODES=$(echo "$STATUS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
available = []
for name, peer in d.get(peers, {}).items():
    if gh200 in name.lower() or h100 in name.lower():
        jobs = peer.get(training_jobs, 0)
        gpu = peer.get(gpu_name, )
        if jobs < 2:  # Can take more training
            available.append(f{name}:{jobs})
print(,.join(available) if available else none)
" 2>/dev/null)

log "Available training nodes: $AVAILABLE_NODES"

# Summary
log "Training pipeline status:"
log "  - Active jobs: $TRAINING_JOBS"
log "  - Boards waiting: $READY_BOARDS"
log "  - Available nodes: $AVAILABLE_NODES"

# Recommendations
if [ "$READY_BOARDS" != "none" ] && [ "$AVAILABLE_NODES" != "none" ]; then
    log "RECOMMENDATION: Training can be parallelized on available GH200 nodes"
fi

log "Optimizer complete"
