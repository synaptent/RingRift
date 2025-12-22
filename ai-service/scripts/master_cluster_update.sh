#!/bin/bash
# Master cluster update script - updates all nodes from config/cluster_nodes.yaml
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$AI_SERVICE_ROOT/config/cluster_nodes.yaml"
LOCAL_CONFIG="$AI_SERVICE_ROOT/config/cluster_nodes.local.yaml"

# Use local config if it exists
if [[ -f "$LOCAL_CONFIG" ]]; then
    CONFIG_FILE="$LOCAL_CONFIG"
fi

echo "============================================================"
echo "    MASTER CLUSTER UPDATE - ALL NODES"
echo "============================================================"
echo "Config: $CONFIG_FILE"
echo ""

# SSH update function
update_ssh() {
    local name="$1" host="$2" user="$3" port="${4:-22}" ssh_key="$5"
    local ssh_opts="-o ConnectTimeout=12 -o BatchMode=yes -o StrictHostKeyChecking=no"
    if [[ -n "$ssh_key" ]]; then
        ssh_key="${ssh_key/#\~/$HOME}"
        ssh_opts="$ssh_opts -i $ssh_key"
    fi
    result=$(timeout 25 ssh $ssh_opts -p "$port" "${user}@${host}" '
        for d in ~/ringrift ~/RingRift ~/ai-service /root/RingRift /root/ringrift ~/Development/RingRift /workspace/RingRift /workspace/ringrift; do
            if [ -d "$d/.git" ]; then
                cd "$d"
                git fetch origin 2>/dev/null
                git reset --hard origin/main 2>/dev/null
                git log -1 --oneline
                exit 0
            fi
        done
        echo NOREPO
    ' 2>&1)

    ec=$?
    if [[ $ec -eq 0 ]] && [[ "$result" != *"NOREPO"* ]] && [[ "$result" != *"fatal"* ]] && [[ "$result" != *"error"* ]]; then
        commit=$(echo "$result" | tail -1 | head -c 45)
        printf "  \033[32m✓\033[0m %-22s %-20s %s\n" "$name" "$host" "$commit"
        return 0
    else
        printf "  \033[31m✗\033[0m %-22s %-20s FAILED\n" "$name" "$host"
        return 1
    fi
}

export -f update_ssh

# Parse YAML and update nodes using Python - output to temp file
TMPFILE=$(mktemp)
python3 - "$CONFIG_FILE" > "$TMPFILE" << 'PYTHON_SCRIPT'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)

for group_name, group_data in config.items():
    if group_name == "config":
        continue
    if isinstance(group_data, dict) and "nodes" in group_data:
        desc = group_data.get("description", group_name)
        ssh_key = group_data.get("ssh_key", "")
        print(f"GROUP:{group_name}:{desc}:{len(group_data['nodes'])}:{ssh_key}")
        for node in group_data["nodes"]:
            name = node.get("name", "unknown")
            host = node.get("host", "")
            user = node.get("user", "ubuntu")
            port = node.get("port", 22)
            print(f"NODE:{name}:{host}:{user}:{port}")
PYTHON_SCRIPT

# Process nodes from temp file
total_count=0
current_ssh_key=""

while IFS=: read -r type arg1 arg2 arg3 arg4 arg5; do
    if [[ "$type" == "GROUP" ]]; then
        # Wait for previous group to complete
        wait
        desc="$arg2"
        count="$arg3"
        current_ssh_key="$arg4"
        echo ""
        echo "=== $desc ($count nodes) ==="
    elif [[ "$type" == "NODE" ]]; then
        name="$arg1"
        host="$arg2"
        user="$arg3"
        port="$arg4"
        ((total_count++))
        update_ssh "$name" "$host" "$user" "$port" "$current_ssh_key" </dev/null &
    fi
done < "$TMPFILE"

# Wait for last group and cleanup
wait
rm -f "$TMPFILE"

echo ""
echo "============================================================"
echo "    UPDATE COMPLETE - $total_count NODES PROCESSED"
echo "============================================================"
