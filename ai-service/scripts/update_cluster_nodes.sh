#!/bin/bash
# Update all cluster nodes with latest code from GitHub
# Generated: 2025-12-25

set -u

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timeout per node (seconds)
TIMEOUT=30

# Track results
declare -a SUCCESS_NODES
declare -a FAILED_NODES

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RingRift Cluster Update Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to update a node
update_node() {
    local name=$1
    local ssh_host=$2
    local ssh_port=$3
    local ssh_key=$4
    local ssh_user=$5
    local ringrift_path=$6

    echo -e "${YELLOW}Updating ${name}...${NC}"

    # Build SSH command
    local ssh_cmd="ssh -o ConnectTimeout=${TIMEOUT} -o StrictHostKeyChecking=no"
    ssh_cmd="${ssh_cmd} -p ${ssh_port} -i ${ssh_key} ${ssh_user}@${ssh_host}"

    # Run git pull
    local git_cmd="cd ${ringrift_path} && git pull origin main 2>&1"
    local output

    if output=$(timeout ${TIMEOUT} ${ssh_cmd} "${git_cmd}" 2>&1); then
        echo -e "${GREEN}✓ ${name}: SUCCESS${NC}"
        echo "  Output: ${output}"
        SUCCESS_NODES+=("${name}")
        return 0
    else
        echo -e "${RED}✗ ${name}: FAILED${NC}"
        echo "  Error: ${output}"
        FAILED_NODES+=("${name}")
        return 1
    fi
}

echo -e "${BLUE}=== Vast.ai Nodes ===${NC}"
echo ""

# Vast.ai nodes (12 total)
update_node "vast-29129529" "ssh6.vast.ai" "19528" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29118471" "ssh8.vast.ai" "38470" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29128352" "ssh9.vast.ai" "18352" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
update_node "vast-28925166" "ssh1.vast.ai" "15166" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29128356" "ssh7.vast.ai" "18356" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
update_node "vast-28918742" "ssh8.vast.ai" "38742" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29031159" "ssh5.vast.ai" "31158" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29126088" "ssh5.vast.ai" "16088" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
update_node "vast-29031161" "ssh2.vast.ai" "31160" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-28890015" "ssh9.vast.ai" "10014" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-28889766" "ssh3.vast.ai" "19766" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
update_node "vast-29046315" "ssh2.vast.ai" "16314" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== RunPod Nodes ===${NC}"
echo ""

# RunPod nodes (5 total)
update_node "runpod-h100" "102.210.171.65" "30690" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
update_node "runpod-a100-1" "38.128.233.145" "33085" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
update_node "runpod-a100-2" "104.255.9.187" "11681" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
update_node "runpod-l40s-2" "193.183.22.62" "1630" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
update_node "runpod-3090ti-1" "174.94.157.109" "29473" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Vultr Nodes ===${NC}"
echo ""

# Vultr nodes (2 total)
update_node "vultr-a100-20gb" "208.167.249.164" "22" "$HOME/.ssh/id_ed25519" "root" "/root/ringrift/ai-service"
update_node "vultr-a100-20gb-2" "140.82.15.69" "22" "$HOME/.ssh/id_ed25519" "root" "/root/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Nebius Nodes ===${NC}"
echo ""

# Nebius nodes (2 total)
update_node "nebius-backbone-1" "89.169.112.47" "22" "$HOME/.ssh/id_cluster" "ubuntu" "~/ringrift/ai-service"
update_node "nebius-l40s-2" "89.169.108.182" "22" "$HOME/.ssh/id_cluster" "ubuntu" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Hetzner Nodes ===${NC}"
echo ""

# Hetzner nodes (3 total - CPU only)
update_node "hetzner-cpu1" "46.62.147.150" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"
update_node "hetzner-cpu2" "135.181.39.239" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"
update_node "hetzner-cpu3" "46.62.217.168" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Update Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}Successful updates (${#SUCCESS_NODES[@]}):${NC}"
for node in "${SUCCESS_NODES[@]}"; do
    echo -e "  ${GREEN}✓${NC} ${node}"
done

echo ""
echo -e "${RED}Failed updates (${#FAILED_NODES[@]}):${NC}"
if [ ${#FAILED_NODES[@]} -eq 0 ]; then
    echo -e "  ${GREEN}None${NC}"
else
    for node in "${FAILED_NODES[@]}"; do
        echo -e "  ${RED}✗${NC} ${node}"
    done
fi

echo ""
echo -e "${BLUE}Total nodes attempted: $((${#SUCCESS_NODES[@]} + ${#FAILED_NODES[@]}))${NC}"

# Exit with error if any nodes failed
if [ ${#FAILED_NODES[@]} -gt 0 ]; then
    exit 1
fi

exit 0
