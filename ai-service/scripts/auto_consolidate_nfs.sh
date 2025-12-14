#!/bin/bash
# Auto-consolidate selfplay data from all GH200 nodes to shared NFS
# Run via cron every 30 minutes

NFS_BASE="/lambda/nfs/RingRift/selfplay_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/consolidate_${TIMESTAMP}.log"

echo "=== Auto-consolidation started at $(date) ===" | tee -a "$LOG_FILE"

# GH200 hosts (all have access to same NFS)
GH200_HOSTS=(
    "ubuntu@192.222.51.29"
    "ubuntu@192.222.51.167"
    "ubuntu@192.222.51.162"
    "ubuntu@192.222.58.122"
    "ubuntu@192.222.57.162"
    "ubuntu@192.222.57.178"
    "ubuntu@192.222.57.79"
    "ubuntu@192.222.56.123"
)

for host in "${GH200_HOSTS[@]}"; do
    node_name=$(echo "$host" | cut -d@ -f2 | tr '.' '_')
    echo "Checking $host..." | tee -a "$LOG_FILE"
    
    # Get selfplay data from node's local storage
    for board_dir in sq8_2p sq8_3p sq8_4p sq19_2p sq19_3p sq19_4p hex_2p hex_3p hex_4p; do
        # Sync JSONL files
        timeout 60 ssh "$host" "
            for f in ~/ringrift/ai-service/data/selfplay/gpu/*.jsonl; do
                if [ -f \"\$f\" ] && [ -s \"\$f\" ]; then
                    dest=\"$NFS_BASE/${board_dir}/\${HOSTNAME}_\$(basename \$f)\"
                    cp -n \"\$f\" \"\$dest\" 2>/dev/null && echo \"Copied \$f to \$dest\"
                fi
            done
        " 2>/dev/null || echo "  Timeout/error for $host" | tee -a "$LOG_FILE"
    done
done

echo "=== Consolidation complete at $(date) ===" | tee -a "$LOG_FILE"
