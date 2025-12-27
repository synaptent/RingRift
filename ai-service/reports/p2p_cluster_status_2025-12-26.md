# P2P Cluster Status Report

**Date:** December 26, 2025
**Report Generated At:** 2025-12-26T05:15:00 UTC

## Executive Summary

- **Total Nodes with P2P Enabled:** 30 (in config)
- **Nodes in P2P Network:** 31 (includes 2 not in config)
- **Nodes Alive (< 2min heartbeat):** 24
- **Nodes Needing Deployment:** 1
- **Nodes Needing Restart:** 6
- **Retired Nodes:** 3
- **Active Selfplay Jobs:** 41
- **Active Training Jobs:** 2

## P2P Network Health

- **Leader:** None (using voter-based consensus)
- **Voter Quorum:** 4/3 alive (quorum: ✓ OK)
- **Voter Nodes:** hetzner-cpu1, nebius-backbone-1, nebius-h100-3, runpod-h100, vultr-a100-20gb

## Detailed Node Status

| Node                                            | P2P Status | Port 8770 | Last HB | Disk Usage | NAT Status  | Container | GPU              | Jobs        | Recommendation      |
| ----------------------------------------------- | ---------- | --------- | ------- | ---------- | ----------- | --------- | ---------------- | ----------- | ------------------- |
| **COORDINATOR NODES**                           |
| mac-studio                                      | ✓ running  | ✓ yes     | 32s     | 71%        | NAT-blocked | none      | M3 Max           | 0           | OK                  |
| local-mac                                       | ⚠ stale    | -         | 14m     | 77%        | NAT-blocked | none      | Apple MPS        | 0           | RESTART (retired)   |
| **NEBIUS NODES (GPU Training Primary)**         |
| nebius-backbone-1                               | ✓ running  | ✓ yes     | 0s      | 39%        | direct      | none      | L40S 48GB        | 1 training  | OK                  |
| nebius-h100-1                                   | ✓ running  | ✓ yes     | 1m      | 50%        | NAT-blocked | none      | H100 80GB        | 0           | OK                  |
| nebius-h100-3                                   | ⚠ stale    | -         | 4m      | 6%         | direct      | none      | H100 80GB        | 1 selfplay  | RESTART             |
| nebius-l40s-2                                   | ⚠ stale    | -         | 3h      | 33%        | NAT-blocked | none      | L40S 48GB        | 1 selfplay  | RETIRED             |
| **RUNPOD NODES (High-End GPUs)**                |
| runpod-h100                                     | ✓ running  | ✓ yes     | 1m      | 42%        | NAT-blocked | docker    | H100 PCIe 80GB   | 7 selfplay  | OK                  |
| runpod-a100-1                                   | ✓ running  | ✓ yes     | 1m      | 29%        | NAT-blocked | docker    | A100 80GB        | 1 training  | OK                  |
| runpod-a100-2                                   | ✓ running  | ✓ yes     | 1m      | 57%        | NAT-blocked | docker    | A100 80GB        | 14 selfplay | OK                  |
| runpod-a100-storage-1                           | ✓ running  | ✓ yes     | 2s      | 10%        | NAT-blocked | docker    | A100 80GB        | 1 selfplay  | OK                  |
| runpod-l40s-2                                   | ✗ stopped  | ✗ no      | N/A     | 23%        | direct      | docker    | L40S 48GB        | 0           | **DEPLOY P2P**      |
| runpod-3090ti-1                                 | ✓ running  | ✓ yes     | 11s     | 82%        | NAT-blocked | docker    | RTX 3090 Ti      | 0           | Disk high (monitor) |
| **VULTR NODES (vGPU)**                          |
| vultr-a100-20gb                                 | ⚠ stale    | -         | 2m      | 4%         | NAT-blocked | none      | A100D-20C vGPU   | 3 selfplay  | RESTART             |
| vultr-a100-20gb-2                               | ✓ running  | ✓ yes     | 1m      | 5%         | direct      | none      | A100D-20C vGPU   | 0           | OK                  |
| **HETZNER NODES (CPU-only)**                    |
| hetzner-cpu1                                    | ✓ running  | ✓ yes     | 17s     | 7%         | NAT-blocked | none      | none             | 0           | OK                  |
| hetzner-cpu2                                    | ⚠ stale    | -         | 3m      | 5%         | NAT-blocked | none      | none             | 0           | RESTART             |
| hetzner-cpu3                                    | ⚠ stale    | -         | 3m      | 6%         | NAT-blocked | none      | none             | 0           | RESTART             |
| **VAST.AI NODES (Tier 1: Multi-GPU)**           |
| vast-29129529                                   | ✓ running  | ✓ yes     | 18s     | 4%         | NAT-blocked | docker    | 8x RTX 4090      | 1 selfplay  | OK                  |
| vast-29118471                                   | ✓ running  | ✓ yes     | 20s     | 71%        | NAT-blocked | docker    | 8x RTX 3090      | 0           | OK                  |
| vast-29128352                                   | ⚠ stale    | -         | 9h      | 30%        | NAT-blocked | docker    | 2x RTX 5090      | 1 selfplay  | RETIRED             |
| **VAST.AI NODES (Tier 2: High-end Single GPU)** |
| vast-28925166                                   | ✓ running  | ✓ yes     | 1m      | 53%        | NAT-blocked | docker    | RTX 5090 32GB    | 0           | OK                  |
| vast-29128356                                   | ✓ running  | ✓ yes     | 3s      | 30%        | NAT-blocked | docker    | RTX 5090 32GB    | 12 selfplay | OK                  |
| vast-29118472                                   | ✓ running  | ✓ yes     | 1m      | 45%        | NAT-blocked | docker    | RTX 5090 32GB    | 1 selfplay  | OK                  |
| vast-29129151                                   | ✓ running  | ✓ yes     | 7s      | 41%        | NAT-blocked | docker    | RTX 5090 32GB    | 0           | OK                  |
| vast-28918742                                   | ✓ running  | ✓ yes     | 23s     | 70%        | NAT-blocked | docker    | A40 46GB         | 0           | OK                  |
| vast-29031159                                   | ✓ running  | ✓ yes     | 46s     | 14%        | NAT-blocked | docker    | RTX 5080 16GB    | 1 selfplay  | OK                  |
| **VAST.AI NODES (Tier 3: Mid-range)**           |
| vast-29126088                                   | ✓ running  | ✓ yes     | 23s     | 5%         | NAT-blocked | docker    | RTX 4060 Ti 16GB | 1 selfplay  | OK                  |
| vast-29031161                                   | ✓ running  | ✓ yes     | 30s     | 26%        | NAT-blocked | docker    | RTX 3060 12GB    | 0           | OK                  |
| vast-28890015                                   | ✓ running  | ✓ yes     | 29s     | 65%        | NAT-blocked | docker    | RTX 2080 Ti 11GB | 1 selfplay  | OK                  |
| **VAST.AI NODES (Tier 4: Entry-level)**         |
| vast-28889766                                   | ✓ running  | ✓ yes     | 4s      | 24%        | NAT-blocked | docker    | RTX 3060 Ti 8GB  | 2 selfplay  | OK                  |
| vast-29046315                                   | ✓ running  | ✓ yes     | 28s     | 32%        | NAT-blocked | docker    | RTX 3060 Ti 8GB  | 0           | OK                  |
| **OTHER NODES**                                 |
| aws-staging                                     | ✓ running  | ✓ yes     | 54s     | 29%        | direct      | none      | none             | 0           | OK (not in config)  |

## Action Items

### Priority 1: Deploy P2P (1 node)

These nodes have P2P enabled in config but are not running the daemon:

1. **runpod-l40s-2** (L40S 48GB, direct IP)
   - SSH: `root@193.183.22.62:1182`
   - Status: P2P not running, port not responding
   - Disk: 23% used (adequate space)
   - NAT: Direct IP (excellent for P2P)
   - **Action:** Deploy P2P daemon

### Priority 2: Restart P2P (6 nodes)

These nodes have stale heartbeats (> 2min) and should restart their P2P daemon:

1. **nebius-h100-3** (H100 80GB, direct IP)
   - Last heartbeat: 4 minutes ago
   - Jobs: 1 selfplay running
   - **Action:** Restart P2P daemon (consider graceful restart to preserve running job)

2. **vultr-a100-20gb** (A100D-20C vGPU)
   - Last heartbeat: 2 minutes ago
   - Jobs: 3 selfplay running
   - **Action:** Restart P2P daemon

3. **hetzner-cpu2** (CPU-only)
   - Last heartbeat: 3 minutes ago
   - **Action:** Restart P2P daemon

4. **hetzner-cpu3** (CPU-only)
   - Last heartbeat: 3 minutes ago
   - **Action:** Restart P2P daemon

### Priority 3: Monitor (3 retired nodes)

These nodes are marked as retired in P2P network:

1. **local-mac** (Apple MPS) - Retired, last HB 14m ago
2. **nebius-l40s-2** (L40S) - Retired, last HB 3h ago
3. **vast-29128352** (2x RTX 5090) - Retired, last HB 9h ago

**Action:** Consider removing from config or investigating why they were retired.

### Priority 4: Monitor Disk Usage

1. **runpod-3090ti-1** - 82% disk usage (approaching 85% threshold)

## P2P Network Notes

### Nodes Not in Config (2)

- **aws-staging** - Running and healthy, but not in distributed_hosts.yaml
- **nebius-l40s-2** - Retired node, likely old deployment

**Recommendation:** Add aws-staging to config if it should be managed, or investigate its purpose.

### NAT Status Summary

- **Direct IP (good for P2P):** 6 nodes
  - nebius-backbone-1, nebius-h100-3, vultr-a100-20gb-2, runpod-l40s-2, aws-staging, local network nodes
- **NAT-blocked (limited P2P utility):** 24 nodes
  - All Vast.ai nodes (expected, containerized)
  - Most RunPod nodes (containerized)
  - Hetzner CPU nodes

### Container Type

- **Docker:** 14 nodes (all Vast.ai, most RunPod)
- **None (bare metal/VM):** 16 nodes

## Cluster Workload Distribution

### Active Jobs by Node Type

| Provider  | Selfplay Jobs | Training Jobs | Total Nodes |
| --------- | ------------- | ------------- | ----------- |
| Vast.ai   | 12            | 0             | 14          |
| RunPod    | 23            | 1             | 6           |
| Nebius    | 2             | 1             | 4           |
| Vultr     | 3             | 0             | 2           |
| Hetzner   | 0             | 0             | 3           |
| Other     | 0             | 0             | 2           |
| **TOTAL** | **41**        | **2**         | **31**      |

### GPU Utilization by Tier

| GPU Tier        | Node Count | Active Nodes | Utilization |
| --------------- | ---------- | ------------ | ----------- |
| H100 (80GB)     | 3          | 3            | 100%        |
| A100 (80GB)     | 4          | 4            | 100%        |
| L40S (48GB)     | 3          | 2            | 67%         |
| RTX 5090 (32GB) | 5          | 4            | 80%         |
| RTX 4090 (24GB) | 1          | 1            | 100%        |
| RTX 3090 (24GB) | 2          | 1            | 50%         |
| Other GPUs      | 6          | 5            | 83%         |
| CPU-only        | 3          | 1            | 33%         |

## Health Indicators

✅ **Healthy:**

- Voter quorum is satisfied (4/3 alive)
- Leader election is stable
- 24/31 nodes (77%) have fresh heartbeats
- 43 total jobs running across cluster
- No nodes with critical disk space

⚠️ **Warnings:**

- 6 nodes have stale P2P heartbeats (> 2min)
- 1 node missing P2P deployment
- 3 retired nodes still in network
- 1 node approaching disk limit (82%)

## Recommendations

1. **Immediate Actions:**
   - Deploy P2P on runpod-l40s-2 (adds L40S GPU capacity)
   - Restart P2P on nebius-h100-3 (H100 is high-value node)
   - Restart P2P on vultr-a100-20gb (voter node)

2. **Short-term:**
   - Restart P2P on hetzner-cpu2, hetzner-cpu3 (CPU nodes, low priority)
   - Add aws-staging to distributed_hosts.yaml or document its purpose
   - Monitor runpod-3090ti-1 disk usage

3. **Long-term:**
   - Remove retired nodes (local-mac, nebius-l40s-2, vast-29128352) from P2P network
   - Consider removing p2p_enabled for retired nodes in config
   - Investigate NAT configuration for nodes that could benefit from direct IPs

## Deployment Commands

### Deploy P2P on runpod-l40s-2

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -p 1182 root@193.183.22.62
cd ~/workspace/ringrift/ai-service
nohup python3 scripts/p2p_orchestrator.py --node-id runpod-l40s-2 --port 8770 --advertise-host 193.183.22.62 > logs/p2p.log 2>&1 &
```

### Restart P2P on nodes (template)

```bash
# Example: nebius-h100-3
ssh -i ~/.ssh/id_cluster ubuntu@89.169.110.128
pkill -f p2p_orchestrator
cd ~/ringrift/ai-service
nohup python3 scripts/p2p_orchestrator.py --node-id nebius-h100-3 --port 8770 --advertise-host 89.169.110.128 > logs/p2p.log 2>&1 &
```
