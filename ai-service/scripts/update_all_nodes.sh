#!/bin/bash
# Comprehensive cluster update script

echo "=============================================="
echo "    Comprehensive Cluster Update"
echo "=============================================="

update_node() {
    local name="$1"
    local host="$2"
    local user="$3"
    local port="${4:-22}"

    result=$(timeout 30 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "${user}@${host}" '
        for dir in ~/ringrift/ai-service ~/RingRift/ai-service /root/RingRift/ai-service /root/ringrift/ai-service /workspace/ringrift/ai-service; do
            if [ -d "$dir/.git" ]; then
                cd "$dir"
                git fetch origin 2>/dev/null
                git reset --hard origin/main 2>/dev/null
                git log -1 --oneline 2>&1
                exit 0
            fi
        done
        echo NO_GIT_REPO
    ' 2>&1)

    exit_code=$?
    if [[ $exit_code -eq 0 ]] && [[ "$result" != "NO_GIT_REPO" ]] && [[ "$result" != *fatal* ]]; then
        commit=$(echo "$result" | tail -1 | cut -c1-45)
        printf "%-22s %-20s %s\n" "$name" "$host" "OK: $commit"
    else
        printf "%-22s %-20s %s\n" "$name" "$host" "FAILED"
    fi
}

export -f update_node

echo ""
echo "--- Lambda Cloud (23 instances) ---"
printf "%-22s %-20s %s\n" "NAME" "HOST" "STATUS"
printf "%-22s %-20s %s\n" "----" "----" "------"

# Run updates in parallel
update_node "lambda-a10-a" "150.136.65.197" "ubuntu" &
update_node "lambda-h100" "209.20.157.81" "ubuntu" &
update_node "lambda-2xh100" "192.222.53.22" "ubuntu" &
update_node "lambda-gh200-a" "192.222.51.29" "ubuntu" &
update_node "lambda-gh200-b" "192.222.51.161" "ubuntu" &
update_node "lambda-gh200-c" "192.222.51.162" "ubuntu" &
update_node "lambda-gh200-d" "192.222.58.122" "ubuntu" &
update_node "lambda-gh200-e" "192.222.57.162" "ubuntu" &
update_node "lambda-gh200-f" "192.222.57.178" "ubuntu" &
update_node "lambda-gh200-g" "192.222.57.79" "ubuntu" &
update_node "lambda-gh200-h" "192.222.56.123" "ubuntu" &
update_node "lambda-gh200-i" "192.222.50.112" "ubuntu" &
update_node "lambda-gh200-k" "192.222.51.150" "ubuntu" &
update_node "lambda-gh200-l" "192.222.51.233" "ubuntu" &
update_node "lambda-gh200-m" "192.222.50.219" "ubuntu" &
update_node "lambda-gh200-n" "192.222.51.204" "ubuntu" &
update_node "lambda-gh200-o" "192.222.51.92" "ubuntu" &
update_node "lambda-gh200-p" "192.222.51.215" "ubuntu" &
update_node "lambda-gh200-q" "192.222.51.18" "ubuntu" &
update_node "lambda-gh200-r" "192.222.50.172" "ubuntu" &
update_node "lambda-gh200-s" "192.222.51.89" "ubuntu" &
update_node "lambda-a10-b" "129.153.159.191" "ubuntu" &
update_node "lambda-a10-c" "150.136.56.240" "ubuntu" &
wait

echo ""
echo "--- Vast.ai Instances ---"
printf "%-22s %-20s %s\n" "NAME" "HOST" "STATUS"
printf "%-22s %-20s %s\n" "----" "----" "------"
update_node "vast-28844365" "ssh5.vast.ai" "root" "14364" &
update_node "vast-28844370" "ssh2.vast.ai" "root" "14370" &
update_node "vast-28844401" "ssh1.vast.ai" "root" "14400" &
update_node "vast-28889768" "ssh2.vast.ai" "root" "19768" &
update_node "vast-28920043" "ssh2.vast.ai" "root" "10042" &
update_node "vast-28925166" "ssh1.vast.ai" "root" "15166" &
update_node "vast-29046315" "ssh2.vast.ai" "root" "16314" &
update_node "vast-29046316" "ssh6.vast.ai" "root" "16316" &
update_node "vast-29046317" "ssh7.vast.ai" "root" "16316" &
update_node "vast-28889766" "ssh3.vast.ai" "root" "19766" &
update_node "vast-28889941" "ssh3.vast.ai" "root" "19940" &
update_node "vast-28890015" "ssh9.vast.ai" "root" "10014" &
update_node "vast-28918742" "ssh8.vast.ai" "root" "38742" &
update_node "vast-29031159" "ssh5.vast.ai" "root" "31158" &
update_node "vast-29031161" "ssh2.vast.ai" "root" "31160" &
wait

echo ""
echo "--- Mac Cluster ---"
printf "%-22s %-20s %s\n" "NAME" "HOST" "STATUS"
printf "%-22s %-20s %s\n" "----" "----" "------"
update_node "mac-studio" "100.107.168.125" "armand" &
update_node "mbp-64gb" "100.92.222.49" "armand" &
update_node "mbp-16gb" "100.66.142.46" "armand" &
wait

echo ""
echo "=============================================="
echo "    Complete: 41 nodes processed"
echo "=============================================="
