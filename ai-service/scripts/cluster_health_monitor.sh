#!/bin/bash
# DEPRECATED: Use cluster_monitor_unified.sh instead (December 2025)
# Migration: ./scripts/cluster_monitor_unified.sh --local-only
echo "WARNING: This script is deprecated. Use cluster_monitor_unified.sh instead."
echo "For equivalent functionality: ./scripts/cluster_monitor_unified.sh --local-only"
echo ""

# Cluster Health Monitor - logs status every 2 minutes
# Run with: nohup ./scripts/cluster_health_monitor.sh &

LOG_FILE="/Users/armand/Development/RingRift/ai-service/logs/cluster_health.log"
ALERT_FILE="/Users/armand/Development/RingRift/ai-service/logs/cluster_alerts.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $1" | tee -a "$ALERT_FILE"
}

check_cluster() {
    log "=== Cluster Health Check ==="
    
    # Check if AI service is running
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        log "AI Service: OK"
    else
        alert "AI Service: DOWN"
    fi
    
    # Check P2P orchestrator if available
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        log "P2P Orchestrator: OK"
    else
        log "P2P Orchestrator: Not running (optional)"
    fi
    
    # Check GPU utilization if nvidia-smi available
    if command -v nvidia-smi &> /dev/null; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "GPU Utilization: ${GPU_UTIL}%"
        log "GPU Memory: ${GPU_MEM} MB"
        
        if [ -n "$GPU_UTIL" ] && [ "$GPU_UTIL" -lt 10 ]; then
            alert "GPU underutilized: ${GPU_UTIL}%"
        fi
    fi
    
    # Check system resources
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    MEM_FREE=$(vm_stat 2>/dev/null | grep "Pages free" | awk '{print $3}' | tr -d '.')
    log "Load Average: $LOAD"
    
    # Check for any training processes
    TRAINING_PROCS=$(pgrep -f "train.py|generate_data.py|selfplay" | wc -l | tr -d ' ')
    log "Training Processes: $TRAINING_PROCS"
    
    # Check disk space
    DISK_USE=$(df -h /Users/armand/Development/RingRift 2>/dev/null | tail -1 | awk '{print $5}')
    log "Disk Usage: $DISK_USE"
    
    log "---"
}

log "Starting cluster health monitor (checking every 2 minutes)"
log "Alerts will be written to: $ALERT_FILE"

while true; do
    check_cluster
    sleep 120  # 2 minutes
done

# Add summary generation every hour
HOUR_COUNT=0
generate_hourly_summary() {
    SUMMARY_FILE="/Users/armand/Development/RingRift/ai-service/logs/hourly_summary.log"
    echo "=== HOURLY SUMMARY $(date '+%Y-%m-%d %H:%M') ===" >> "$SUMMARY_FILE"
    echo "Checks completed: $HOUR_COUNT" >> "$SUMMARY_FILE"
    echo "Alerts: $(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)" >> "$SUMMARY_FILE"
    tail -20 "$LOG_FILE" >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
}
