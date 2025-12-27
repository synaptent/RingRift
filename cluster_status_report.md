# RingRift Cluster P2P Status Report

**Generated:** 2025-12-26
**Total Nodes in Config:** 31

---

## Executive Summary

- **P2P Running:** 13 nodes (42%)
- **P2P Not Running (Reachable):** 9 nodes (29%)
- **Unreachable:** 8 nodes (26%)
- **P2P Not Enabled:** 1 node (3%)

**P2P Voter Configuration:**

- Total voters: 5
- Quorum requires: 3 nodes
- Voters with P2P running: 3/5 (nebius-backbone-1, nebius-h100-3, vultr-a100-20gb-2)
- Voters offline: 2/5 (hetzner-cpu1, runpod-h100)

⚠️ **Critical Issue:** Only 3/5 voters are online, exactly meeting quorum requirements. If one more voter goes offline, the cluster will lose quorum.

---

## Nodes with P2P Running (13)

| Name                  | Host            | NAT | Storage      | Container | GPU Util | GPU Mem     | Jobs | Notes                       |
| --------------------- | --------------- | --- | ------------ | --------- | -------- | ----------- | ---- | --------------------------- |
| mac-studio            | 100.107.168.125 | NO  | N/A          | No        | N/A      | N/A         | 15   | Coordinator                 |
| nebius-backbone-1     | 89.169.112.47   | NO  | 59GB free    | No        | 44.0%    | 2.5/45GB    | 2    | **VOTER**, Good utilization |
| nebius-h100-1         | 89.169.111.139  | NO  | ⚠️ 48GB free | No        | 98.0%    | 33.8/79.6GB | 163  | High load, low storage      |
| nebius-h100-3         | 89.169.110.128  | NO  | 451GB free   | No        | 12.0%    | 1.8/79.6GB  | 3    | **VOTER**, Idle GPU         |
| runpod-a100-1         | 38.128.233.145  | NO  | N/A          | No        | N/A      | N/A         | 0    | Idle GPU                    |
| runpod-a100-2         | 104.255.9.187   | NO  | N/A          | No        | N/A      | N/A         | 0    | Idle GPU                    |
| runpod-a100-storage-1 | 213.173.105.7   | NO  | 90GB free    | Yes       | 13.0%    | 1.1/80GB    | 1    | Containerized               |
| vast-29031159         | ssh5.vast.ai    | NO  | N/A          | No        | N/A      | N/A         | 0    | RTX 5080, Idle              |
| vast-29046315         | ssh2.vast.ai    | NO  | N/A          | No        | N/A      | N/A         | 0    | RTX 3060 Ti, Idle           |
| vast-29118471         | ssh8.vast.ai    | NO  | N/A          | No        | N/A      | N/A         | 0    | 8x RTX 3090, Idle           |
| vast-29118472         | ssh9.vast.ai    | NO  | N/A          | No        | N/A      | N/A         | 0    | RTX 5090, Idle              |
| vast-29126088         | ssh5.vast.ai    | NO  | 190GB free   | No        | N/A      | N/A         | 0    | RTX 4060 Ti, Idle           |
| vultr-a100-20gb-2     | 140.82.15.69    | NO  | N/A          | No        | N/A      | N/A         | 0    | **VOTER**, Idle GPU         |

### Key Observations:

- **No NAT blocking:** All P2P nodes have direct connectivity
- **Idle GPUs:** 9 nodes with P2P running have 0 jobs - ready for selfplay allocation
- **Storage issues:**
  - nebius-h100-1: Only 48GB free (under 50GB threshold)
- **High utilization:** nebius-h100-1 at 98% GPU utilization with 163 jobs

---

## Nodes Without P2P (9 reachable, P2P enabled)

| Name            | Host            | Storage      | Container | GPU              | Priority                 |
| --------------- | --------------- | ------------ | --------- | ---------------- | ------------------------ |
| local-mac       | localhost       | N/A          | No        | none             | Low (coordinator-only)   |
| runpod-3090ti-1 | 174.94.157.109  | N/A          | No        | RTX 3090 Ti      | **HIGH**                 |
| runpod-l40s-2   | 193.183.22.62   | ⚠️ 39GB free | Yes       | L40S             | Medium (low storage)     |
| vast-28889766   | ssh3.vast.ai    | N/A          | No        | RTX 3060 Ti      | **HIGH**                 |
| vast-28925166   | ssh1.vast.ai    | N/A          | No        | RTX 5090         | **HIGH**                 |
| vast-29128352   | ssh9.vast.ai    | N/A          | No        | 2x RTX 5090      | **CRITICAL** (multi-GPU) |
| vast-29128356   | ssh7.vast.ai    | N/A          | No        | RTX 5090         | **HIGH**                 |
| vast-29129529   | ssh6.vast.ai    | N/A          | No        | 8x RTX 4090      | **CRITICAL** (multi-GPU) |
| vultr-a100-20gb | 208.167.249.164 | N/A          | No        | A100D-20C (vGPU) | **HIGH**                 |

### Deployment Priority:

1. **CRITICAL** (multi-GPU nodes):
   - vast-29128352 (2x RTX 5090)
   - vast-29129529 (8x RTX 4090)

2. **HIGH** (powerful single GPUs):
   - runpod-3090ti-1 (RTX 3090 Ti)
   - vast-28925166 (RTX 5090)
   - vast-29128356 (RTX 5090)
   - vast-28889766 (RTX 3060 Ti)
   - vultr-a100-20gb (A100 vGPU)

3. **MEDIUM**:
   - runpod-l40s-2 (L40S, but only 39GB storage - needs cleanup first)

---

## Unreachable Nodes (8)

| Name          | Host           | Error       | Notes                           |
| ------------- | -------------- | ----------- | ------------------------------- |
| hetzner-cpu1  | 46.62.147.150  | SSH timeout | **VOTER** - Critical for quorum |
| hetzner-cpu2  | 135.181.39.239 | SSH timeout | CPU node                        |
| hetzner-cpu3  | 46.62.217.168  | SSH timeout | CPU node                        |
| runpod-h100   | 102.210.171.65 | SSH timeout | **VOTER** - Critical for quorum |
| vast-28890015 | ssh9.vast.ai   | SSH timeout | RTX 2080 Ti                     |
| vast-28918742 | ssh8.vast.ai   | SSH timeout | A40                             |
| vast-29031161 | ssh2.vast.ai   | SSH timeout | RTX 3060                        |
| vast-29129151 | ssh4.vast.ai   | SSH timeout | RTX 5090                        |

### Investigation Needed:

- **Voter nodes offline:** hetzner-cpu1, runpod-h100 need immediate attention
- **Hetzner cluster:** All 3 Hetzner nodes unreachable - possible network issue
- **Vast.ai nodes:** 4 instances may have been terminated or are in maintenance

---

## Storage Analysis

### Low Storage (< 50GB free):

1. nebius-h100-1: 48GB free - needs immediate cleanup
2. runpod-l40s-2: 39GB free - needs cleanup before P2P deployment

### Adequate Storage (> 50GB free):

1. nebius-backbone-1: 59GB free
2. nebius-h100-3: 451GB free (excellent)
3. runpod-a100-storage-1: 90GB free
4. vast-29126088: 190GB free

### Unknown Storage:

- Most Vast.ai and RunPod nodes returned N/A for storage (likely containerized with shared storage)
- mac-studio: N/A (coordinator, external storage on /Volumes/RingRift-Data)

---

## Containerization Status

- **Containerized:** 2 nodes
  - runpod-a100-storage-1
  - runpod-l40s-2

- **Bare-metal:** All other reachable nodes

---

## Immediate Action Items

### 1. Restore Voter Quorum (CRITICAL)

- [ ] Investigate hetzner-cpu1 connectivity (SSH timeout)
- [ ] Investigate runpod-h100 connectivity (SSH timeout)
- [ ] Consider adding more voters from currently running nodes:
  - Candidates: nebius-h100-1, runpod-a100-1, runpod-a100-2

### 2. Deploy P2P to High-Value Nodes (HIGH PRIORITY)

```bash
cd /Users/armand/Development/RingRift/ai-service

# Deploy to multi-GPU nodes first (highest impact)
python scripts/deploy_p2p_cluster.py --nodes vast-29128352,vast-29129529

# Then deploy to high-end single GPUs
python scripts/deploy_p2p_cluster.py --nodes runpod-3090ti-1,vast-28925166,vast-29128356,vast-28889766,vultr-a100-20gb
```

### 3. Storage Cleanup (HIGH PRIORITY)

```bash
# Clean up nebius-h100-1 (currently at 48GB, high load)
ssh -i ~/.ssh/id_cluster ubuntu@89.169.111.139
# Investigate large files, old checkpoints, etc.

# Clean up runpod-l40s-2 (39GB free)
ssh -i ~/.ssh/id_ed25519 -p 1182 root@193.183.22.62
# Free up space, then deploy P2P
```

### 4. Investigate Unreachable Nodes (MEDIUM PRIORITY)

- Check Hetzner billing/network status (all 3 nodes down)
- Check Vast.ai instance status for 4 unreachable nodes
- Verify runpod-h100 instance status

### 5. Utilize Idle GPUs (LOW PRIORITY)

9 nodes with P2P running but 0 jobs:

- runpod-a100-1, runpod-a100-2 (A100 80GB)
- vast-29031159 (RTX 5080)
- vast-29046315, vast-28889766 (RTX 3060 Ti)
- vast-29118471 (8x RTX 3090)
- vast-29118472 (RTX 5090)
- vast-29126088 (RTX 4060 Ti)
- vultr-a100-20gb-2 (A100 vGPU)
- nebius-h100-3 (H100 - voter node, only 12% util)

---

## Deployment Commands

### Check current status again:

```bash
cd /Users/armand/Development/RingRift/ai-service
python scripts/p2p_cluster_status.py
```

### Deploy P2P to all reachable nodes without it:

```bash
# Deploy all at once (parallel)
python scripts/deploy_p2p_cluster.py --nodes runpod-3090ti-1,vast-28889766,vast-28925166,vast-29128352,vast-29128356,vast-29129529,vultr-a100-20gb

# Or deploy with dry-run first to preview
python scripts/deploy_p2p_cluster.py --nodes vast-29129529 --dry-run
```

### Deploy P2P to specific node:

```bash
python scripts/deploy_p2p_cluster.py --nodes <node-name>
```

### Check individual node P2P status:

```bash
ssh -i ~/.ssh/id_cluster root@<host> "curl -s http://localhost:8770/status | python3 -m json.tool"
```

---

## Voter Configuration Recommendations

Current voters:

1. nebius-backbone-1 (L40S) - ✅ Online, stable
2. nebius-h100-3 (H100) - ✅ Online, stable
3. hetzner-cpu1 (CPU) - ❌ Offline
4. vultr-a100-20gb (A100 vGPU) - Note: Wrong voter in config, should be vultr-a100-20gb-2
5. runpod-h100 (H100) - ❌ Offline

### Recommended Changes:

1. Fix voter config: Change `vultr-a100-20gb` to `vultr-a100-20gb-2` (the one actually online)
2. Replace offline voters once connectivity is restored:
   - Option A: Keep hetzner-cpu1 and runpod-h100 if they come back online
   - Option B: Replace with stable online nodes:
     - runpod-a100-1 (stable, A100)
     - runpod-a100-2 (stable, A100)

### Update voter config:

```bash
# Edit config/distributed_hosts.yaml
# Change p2p_voters section to include only stable, online nodes
```

---

## Summary Statistics

| Metric                                           | Count | Percentage    |
| ------------------------------------------------ | ----- | ------------- |
| Total nodes in config                            | 31    | 100%          |
| P2P running                                      | 13    | 42%           |
| P2P deployable (reachable, enabled, not running) | 9     | 29%           |
| Unreachable                                      | 8     | 26%           |
| Not P2P enabled                                  | 1     | 3%            |
| Idle GPUs (P2P running, 0 jobs)                  | 9     | 29%           |
| Active voters                                    | 3     | 60% of voters |
| Nodes with adequate storage                      | 4+    | -             |

---

## Next Steps

1. **Immediate (< 1 hour):**
   - Restore voter connectivity (hetzner-cpu1, runpod-h100)
   - Deploy P2P to vast-29128352 and vast-29129529 (multi-GPU)

2. **Short-term (< 24 hours):**
   - Deploy P2P to all 9 reachable nodes
   - Clean up storage on nebius-h100-1 and runpod-l40s-2
   - Investigate unreachable Vast.ai nodes

3. **Medium-term (< 1 week):**
   - Update voter configuration with stable nodes
   - Set up monitoring for voter quorum health
   - Investigate and restore Hetzner cluster

4. **Long-term (ongoing):**
   - Implement automated storage cleanup
   - Add automated P2P health monitoring and recovery
   - Optimize selfplay job distribution to utilize idle GPUs
