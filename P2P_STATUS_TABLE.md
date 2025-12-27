# P2P Daemon Status - Quick Reference Table

> **SUPERSEDED**: This report has been consolidated into
> `ai-service/docs/infrastructure/reports/2025-12-26_p2p_cluster_status.md`
>
> This file is kept for historical reference only.

> Snapshot note: This table reflects Dec 26, 2025 data. Current voter list and node health can differ; see `ai-service/P2P_DEPLOYMENT_REPORT.md` and `ai-service/config/distributed_hosts.yaml`.
> Current voters (Dec 27, 2025): nebius-backbone-1, nebius-h100-3, hetzner-cpu1, hetzner-cpu2, vultr-a100-20gb (runpod-h100 removed).

**Investigation Date**: December 26, 2025 20:40 CST
**Cluster Size**: 26 GPU nodes + 3 CPU nodes + 2 coordinators = 31 total nodes
**P2P Health**: 10/31 nodes running P2P (32% cluster coverage)

## Summary by Status

| Status                               | Count | Percentage | Notes                           |
| ------------------------------------ | ----- | ---------- | ------------------------------- |
| ✅ **HEALTHY** (Port 8770 listening) | 10    | 32%        | P2P running correctly           |
| ❌ **NOT STARTED**                   | 19    | 61%        | No evidence of P2P ever running |
| 💥 **CRASHED** (Missing deps)        | 2     | 7%         | Import errors (psutil, yaml)    |

## P2P Voter Quorum Status

**Required for leader election**: 3 out of 5 voters
**Current status**: ✅ **HEALTHY QUORUM** (5/5 voters online) _(historical snapshot)_

| Voter Node        | Status         | Port 8770    | Role                        |
| ----------------- | -------------- | ------------ | --------------------------- |
| nebius-backbone-1 | ✅ **HEALTHY** | ✅ LISTENING | Backbone node (L40S)        |
| nebius-h100-3     | ✅ **HEALTHY** | ✅ LISTENING | Training node (H100)        |
| hetzner-cpu1      | ✅ **HEALTHY** | ✅ LISTENING | CPU selfplay node           |
| vultr-a100-20gb   | ✅ **HEALTHY** | ✅ LISTENING | GPU selfplay (A100 vGPU)    |
| runpod-h100       | ✅ **HEALTHY** | ✅ LISTENING | GPU selfplay primary (H100) |

**Quorum Analysis**: All 5 voter nodes are online and healthy. Leader election can proceed normally.
**Note:** The voter list above is outdated; use `ai-service/config/distributed_hosts.yaml` as the source of truth.

## Complete Node Status Table

### Vast.ai Nodes (14 GPU nodes) - 2/14 healthy

| Node          | GPU                 | P2P Running | Port 8770    | Last Error     | Suggested Fix                     |
| ------------- | ------------------- | ----------- | ------------ | -------------- | --------------------------------- |
| vast-29118471 | 8x RTX 3090 (192GB) | ✅ YES      | ✅ LISTENING | None           | None - healthy                    |
| vast-28925166 | RTX 5090 (32GB)     | ✅ YES      | ✅ LISTENING | None           | None - healthy                    |
| vast-29129529 | 8x RTX 4090 (192GB) | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29128352 | 2x RTX 5090 (64GB)  | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29128356 | RTX 5090 (32GB)     | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-28918742 | A40 (46GB)          | 💥 CRASHED  | ❌ NO        | Missing psutil | `pip install psutil` then restart |
| vast-29031159 | RTX 5080 (16GB)     | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29126088 | RTX 4060 Ti (16GB)  | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29031161 | RTX 3060 (12GB)     | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-28890015 | RTX 2080 Ti (11GB)  | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-28889766 | RTX 3060 Ti (8GB)   | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29046315 | RTX 3060 Ti (8GB)   | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29118472 | RTX 5090 (32GB)     | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |
| vast-29129151 | RTX 5090 (32GB)     | ❌ NO       | ❌ NO        | No logs        | Start P2P daemon                  |

### RunPod Nodes (6 GPU nodes) - 3/6 healthy

| Node                  | GPU                | P2P Running | Port 8770    | Last Error | Suggested Fix          |
| --------------------- | ------------------ | ----------- | ------------ | ---------- | ---------------------- |
| runpod-h100           | H100 PCIe (80GB)   | ✅ YES      | ✅ LISTENING | None       | None - healthy (VOTER) |
| runpod-a100-2         | A100 PCIe (80GB)   | ✅ YES      | ✅ LISTENING | None       | None - healthy         |
| runpod-a100-storage-1 | A100 PCIe (80GB)   | ✅ YES      | ✅ LISTENING | None       | None - healthy         |
| runpod-a100-1         | A100 PCIe (80GB)   | ❌ NO       | ❌ NO        | No logs    | Start P2P daemon       |
| runpod-l40s-2         | L40S (48GB)        | ❌ NO       | ❌ NO        | No logs    | Start P2P daemon       |
| runpod-3090ti-1       | RTX 3090 Ti (24GB) | ❌ NO       | ❌ NO        | No logs    | Start P2P daemon       |

### Vultr Nodes (2 GPU nodes) - 1/2 healthy

| Node              | GPU               | P2P Running | Port 8770    | Last Error   | Suggested Fix                     |
| ----------------- | ----------------- | ----------- | ------------ | ------------ | --------------------------------- |
| vultr-a100-20gb   | A100D vGPU (20GB) | ✅ YES      | ✅ LISTENING | None         | None - healthy (VOTER)            |
| vultr-a100-20gb-2 | A100D vGPU (20GB) | 💥 CRASHED  | ❌ NO        | Missing yaml | `pip install PyYAML` then restart |

### Nebius Nodes (3 GPU nodes) - 2/3 healthy

| Node              | GPU             | P2P Running | Port 8770    | Last Error | Suggested Fix          |
| ----------------- | --------------- | ----------- | ------------ | ---------- | ---------------------- |
| nebius-backbone-1 | L40S (48GB)     | ✅ YES      | ✅ LISTENING | None       | None - healthy (VOTER) |
| nebius-h100-3     | H100 SXM (80GB) | ✅ YES      | ✅ LISTENING | None       | None - healthy (VOTER) |
| nebius-h100-1     | H100 SXM (80GB) | ❌ NO       | ❌ NO        | No logs    | Start P2P daemon       |

### Hetzner Nodes (3 CPU nodes) - 1/3 checked

| Node         | Type           | P2P Running    | Port 8770      | Last Error | Suggested Fix          |
| ------------ | -------------- | -------------- | -------------- | ---------- | ---------------------- |
| hetzner-cpu1 | CPU (8 cores)  | ✅ YES         | ✅ LISTENING   | None       | None - healthy (VOTER) |
| hetzner-cpu2 | CPU (16 cores) | ⚠️ NOT CHECKED | ⚠️ NOT CHECKED | -          | Check status           |
| hetzner-cpu3 | CPU (16 cores) | ⚠️ NOT CHECKED | ⚠️ NOT CHECKED | -          | Check status           |

### Coordinator Nodes (2 nodes) - Not checked

| Node       | Type          | Notes                                     |
| ---------- | ------------- | ----------------------------------------- |
| mac-studio | M3 Max (64GB) | Coordinator - P2P enabled but not checked |
| local-mac  | MBP (36GB)    | Coordinator - P2P enabled but not checked |

## Root Cause Breakdown

### Missing Dependencies (2 nodes - 7%)

- **vast-28918742**: `ModuleNotFoundError: No module named 'psutil'`
- **vultr-a100-20gb-2**: `ModuleNotFoundError: No module named 'yaml'`

**Fix**: Install missing packages in venv

```bash
source venv/bin/activate
pip install psutil PyYAML
```

### Never Started (19 nodes - 61%)

Most likely causes:

1. Nodes provisioned after initial P2P deployment
2. Nodes rebooted and P2P not configured to auto-start
3. Manual deployment script never executed on these nodes
4. Different ringrift paths (~/ringrift vs /workspace/ringrift)

**Fix**: Start P2P manually or via deployment script

```bash
cd ~/ringrift/ai-service  # or /workspace/ringrift/ai-service
nohup python scripts/p2p_orchestrator.py \
  --node-id $(hostname) \
  --host 0.0.0.0 \
  --port 8770 \
  --peers http://89.169.110.128:8770 \
  >/tmp/p2p.log 2>&1 &
```

## Impact Analysis

### Data Synchronization

- **Game data**: Only 32% of cluster can receive/send game databases
- **Model distribution**: Severely limited - most nodes won't get new models automatically
- **NPZ training data**: Fragmented across nodes with no sync

### Cluster Coordination

- **Leader election**: ✅ Working (all voters healthy)
- **Job distribution**: Limited to nodes with P2P
- **Resource discovery**: Incomplete cluster view

### Training Pipeline

- **Selfplay generation**: Happens on all nodes (doesn't require P2P)
- **Data collection**: Only 10 nodes can contribute to central pool
- **Model updates**: Only 10 nodes receive new models automatically

## Recommendations

### Priority 1 (Critical - Cluster Coordination)

1. ✅ **COMPLETE**: Verify all voter nodes (5/5 healthy)
2. ⚠️ **TODO**: Install missing dependencies on 2 crashed nodes
3. ⚠️ **TODO**: Start P2P on high-value nodes first:
   - Multi-GPU nodes: vast-29129529 (8x 4090), vast-29128352 (2x 5090)
   - Training nodes: nebius-h100-1 (H100 80GB), runpod-a100-1 (A100 80GB)

### Priority 2 (Important - Data Flow)

4. Start P2P on remaining RunPod nodes (3 nodes)
5. Start P2P on remaining Vast.ai nodes (12 nodes)
6. Create systemd/supervisor service for auto-restart
7. Add P2P startup to deployment/provisioning scripts

### Priority 3 (Maintenance - Monitoring)

8. Fix pgrep detection in monitoring (false positives)
9. Add P2P health checks to cluster_monitor.py
10. Create alerts for P2P outages on voter nodes
11. Document P2P startup procedure in deployment guide

## Files Generated

1. **Investigation script**: `/Users/armand/Development/RingRift/ai-service/scripts/investigate_p2p_status.sh`
2. **Raw report**: `/tmp/p2p_status_report_20251226_203240.txt`
3. **Analysis**: `/Users/armand/Development/RingRift/P2P_INVESTIGATION_SUMMARY.md`
4. **This table**: `/Users/armand/Development/RingRift/P2P_STATUS_TABLE.md`

---

**Next Steps**: DO NOT auto-fix. Await user approval before:

- Installing dependencies
- Starting P2P daemons
- Modifying configurations
- Creating systemd services

_Report completed: 2025-12-26 20:45 CST_
