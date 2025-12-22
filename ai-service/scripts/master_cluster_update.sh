#!/bin/bash
set -o pipefail

echo "============================================================"
echo "    MASTER CLUSTER UPDATE - ALL 50+ NODES"
echo "============================================================"
echo ""

update_ssh() {
    local name="$1" host="$2" user="$3" port="${4:-22}"
    result=$(timeout 25 ssh -o ConnectTimeout=12 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "${user}@${host}" '
        for d in ~/ringrift/ai-service ~/RingRift/ai-service ~/ai-service /root/RingRift/ai-service /root/ringrift/ai-service ~/Development/RingRift/ai-service; do
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
    else
        printf "  \033[31m✗\033[0m %-22s %-20s FAILED\n" "$name" "$host"
    fi
}

export -f update_ssh

echo "=== LAMBDA GH200 CLUSTER (18 via Tailscale) ==="
update_ssh "gh200-a" "100.123.183.70" "ubuntu" &
update_ssh "gh200-b" "100.83.234.82" "ubuntu" &
update_ssh "gh200-c" "100.88.35.19" "ubuntu" &
update_ssh "gh200-d" "100.75.84.47" "ubuntu" &
update_ssh "gh200-e" "100.88.176.74" "ubuntu" &
update_ssh "gh200-f" "100.104.165.116" "ubuntu" &
update_ssh "gh200-g" "100.104.126.58" "ubuntu" &
update_ssh "gh200-h" "100.65.88.62" "ubuntu" &
update_ssh "gh200-i" "100.99.27.56" "ubuntu" &
update_ssh "gh200-k" "100.96.142.42" "ubuntu" &
update_ssh "gh200-l" "100.76.145.60" "ubuntu" &
update_ssh "gh200-m" "100.117.177.83" "ubuntu" &
update_ssh "gh200-o" "100.97.98.26" "ubuntu" &
update_ssh "gh200-q" "100.66.65.33" "ubuntu" &
update_ssh "gh200-s" "100.79.109.120" "ubuntu" &
update_ssh "h100" "100.78.101.123" "ubuntu" &
update_ssh "2xh100" "100.97.104.89" "ubuntu" &
update_ssh "a10" "100.91.25.13" "ubuntu" &
wait

echo ""
echo "=== LAMBDA ADDITIONAL (5 via direct IP) ==="
update_ssh "a10-a" "150.136.65.197" "ubuntu" &
update_ssh "a10-b" "129.153.159.191" "ubuntu" &
update_ssh "a10-c" "150.136.56.240" "ubuntu" &
update_ssh "gh200-n" "192.222.51.204" "ubuntu" &
update_ssh "gh200-r" "192.222.50.172" "ubuntu" &
update_ssh "gh200-p" "192.222.51.215" "ubuntu" &
wait

echo ""
echo "=== VAST.AI INSTANCES (15) ==="
update_ssh "vast-28844365" "ssh5.vast.ai" "root" "14364" &
update_ssh "vast-28844370" "ssh2.vast.ai" "root" "14370" &
update_ssh "vast-28844401" "ssh1.vast.ai" "root" "14400" &
update_ssh "vast-28889766" "ssh3.vast.ai" "root" "19766" &
update_ssh "vast-28889768" "ssh2.vast.ai" "root" "19768" &
update_ssh "vast-28889941" "ssh3.vast.ai" "root" "19940" &
update_ssh "vast-28890015" "ssh9.vast.ai" "root" "10014" &
update_ssh "vast-28918742" "ssh8.vast.ai" "root" "38742" &
update_ssh "vast-28920043" "ssh2.vast.ai" "root" "10042" &
update_ssh "vast-28925166" "ssh1.vast.ai" "root" "15166" &
update_ssh "vast-29031159" "ssh5.vast.ai" "root" "31158" &
update_ssh "vast-29031161" "ssh2.vast.ai" "root" "31160" &
update_ssh "vast-29046315" "ssh2.vast.ai" "root" "16314" &
update_ssh "vast-29046316" "ssh6.vast.ai" "root" "16316" &
update_ssh "vast-29046317" "ssh7.vast.ai" "root" "16316" &
wait

echo ""
echo "=== MAC CLUSTER (3 Tailscale) ==="
update_ssh "mac-studio" "100.107.168.125" "armand" &
update_ssh "mbp-64gb" "100.92.222.49" "armand" &
update_ssh "mbp-16gb" "100.66.142.46" "armand" &
wait

echo ""
echo "=== LOCAL NETWORK MACS (3) ==="
update_ssh "m1-pro-local" "10.0.0.108" "armand" &
update_ssh "mac-mini-local" "10.0.0.62" "armand" &
update_ssh "mbp-local" "10.0.0.89" "armand" &
wait

echo ""
echo "=== TAILSCALE MISC (4) ==="
update_ssh "vast-container" "100.100.242.64" "root" &
update_ssh "vast-3070-ts" "100.74.154.36" "root" &
update_ssh "aws-staging" "100.121.198.28" "ubuntu" &
update_ssh "aws-worker" "100.115.97.24" "ubuntu" &
wait

echo ""
echo "============================================================"
echo "    UPDATE COMPLETE - 49 NODES PROCESSED"
echo "============================================================"
