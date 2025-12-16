#!/bin/bash
# Prometheus-compatible metrics exporter
# Run with: ./prometheus_metrics.sh | nc -l -p 9100

METRICS_FILE="/tmp/ringrift_metrics.prom"

generate_metrics() {
    # Get cluster health
    HEALTH=$(curl -s http://localhost:8770/health 2>/dev/null)
    STATUS=$(curl -s http://localhost:8770/status 2>/dev/null)
    
    if [ -z "$HEALTH" ]; then
        echo "# P2P service not responding"
        return
    fi
    
    # Parse metrics
    ACTIVE_PEERS=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"active_peers\", 0))" 2>/dev/null)
    TOTAL_PEERS=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"total_peers\", 0))" 2>/dev/null)
    DISK_PCT=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"disk_percent\", 0))" 2>/dev/null)
    MEM_PCT=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"memory_percent\", 0))" 2>/dev/null)
    CPU_PCT=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"cpu_percent\", 0))" 2>/dev/null)
    
    # Calculate total selfplay jobs
    SELFPLAY_JOBS=$(echo "$STATUS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
total=d.get(\"self\",{}).get(\"selfplay_jobs\",0)
for p in d.get(\"peers\",{}).values(): total+=p.get(\"selfplay_jobs\",0)
print(total)
" 2>/dev/null)
    
    TRAINING_JOBS=$(echo "$STATUS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
total=d.get(\"self\",{}).get(\"training_jobs\",0)
for p in d.get(\"peers\",{}).values(): total+=p.get(\"training_jobs\",0)
print(total)
" 2>/dev/null)
    
    # Count tunnel status
    TUNNELS_UP=0
    TUNNELS_DOWN=0
    for port in 8771 8772 8773 8774 8775 8776 8777 8778 8779 8780 8781 8782; do
        if nc -z -w1 localhost $port 2>/dev/null; then
            TUNNELS_UP=$((TUNNELS_UP + 1))
        else
            TUNNELS_DOWN=$((TUNNELS_DOWN + 1))
        fi
    done
    
    # Output Prometheus format
    cat << PROM
# HELP ringrift_active_peers Number of active peers in the cluster
# TYPE ringrift_active_peers gauge
ringrift_active_peers $ACTIVE_PEERS

# HELP ringrift_total_peers Total peers known to the cluster
# TYPE ringrift_total_peers gauge
ringrift_total_peers $TOTAL_PEERS

# HELP ringrift_selfplay_jobs Total selfplay jobs running
# TYPE ringrift_selfplay_jobs gauge
ringrift_selfplay_jobs $SELFPLAY_JOBS

# HELP ringrift_training_jobs Total training jobs running
# TYPE ringrift_training_jobs gauge
ringrift_training_jobs $TRAINING_JOBS

# HELP ringrift_disk_percent Disk usage percentage on hub
# TYPE ringrift_disk_percent gauge
ringrift_disk_percent $DISK_PCT

# HELP ringrift_memory_percent Memory usage percentage on hub
# TYPE ringrift_memory_percent gauge
ringrift_memory_percent $MEM_PCT

# HELP ringrift_cpu_percent CPU usage percentage on hub
# TYPE ringrift_cpu_percent gauge
ringrift_cpu_percent $CPU_PCT

# HELP ringrift_tunnels_up Number of tunnels up
# TYPE ringrift_tunnels_up gauge
ringrift_tunnels_up $TUNNELS_UP

# HELP ringrift_tunnels_down Number of tunnels down
# TYPE ringrift_tunnels_down gauge
ringrift_tunnels_down $TUNNELS_DOWN
PROM
}

generate_metrics > "$METRICS_FILE"
cat "$METRICS_FILE"
