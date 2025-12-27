# P2P Daemon Status Investigation Summary

> **SUPERSEDED**: This report has been consolidated into
> `ai-service/docs/infrastructure/reports/2025-12-26_p2p_cluster_status.md`
>
> This file is kept for historical reference only.

> Snapshot note: This report reflects Dec 26, 2025 data. Current voter list and deployment status may differ; see `ai-service/P2P_DEPLOYMENT_REPORT.md` and `ai-service/config/distributed_hosts.yaml`.
> Current voters (Dec 27, 2025): nebius-backbone-1, nebius-h100-3, hetzner-cpu1, hetzner-cpu2, vultr-a100-20gb (runpod-h100 removed).

**Investigation Date**: December 26, 2025
**Total Nodes Checked**: 26 GPU nodes across Vast.ai, RunPod, Vultr, and Nebius

## Executive Summary

Out of 26 GPU nodes:

- **6 nodes** have P2P running correctly (port 8770 listening)
- **20 nodes** have P2P NOT running (despite some false-positive PIDs)
- **2 nodes** have crashed P2P with missing dependencies (psutil, yaml)

### Critical Issues

1. **Missing Python Dependencies**: Several nodes crash on startup due to missing `psutil` and `yaml` modules
2. **False Positive PIDs**: `pgrep -f "p2p_daemon|p2p_orchestrator"` matches strings in other processes, showing PIDs that don't actually exist
3. **No Logs**: Most nodes have no P2P logs in expected locations, suggesting daemon was never successfully started

## Detailed Status by Provider

### VAST.AI NODES (14 total)

| Node                        | P2P Running | Port 8770    | Last Error                                      | Status          |
| --------------------------- | ----------- | ------------ | ----------------------------------------------- | --------------- |
| vast-29118471 (8x RTX 3090) | ✅ YES      | ✅ LISTENING | None                                            | **HEALTHY**     |
| vast-28925166 (RTX 5090)    | ✅ YES      | ✅ LISTENING | None                                            | **HEALTHY**     |
| vast-29129529 (8x RTX 4090) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29128352 (2x RTX 5090) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29128356 (RTX 5090)    | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-28918742 (A40)         | ❌ NO       | ❌ NO        | `ModuleNotFoundError: No module named 'psutil'` | **CRASHED**     |
| vast-29031159 (RTX 5080)    | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29126088 (RTX 4060 Ti) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29031161 (RTX 3060)    | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-28890015 (RTX 2080 Ti) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-28889766 (RTX 3060 Ti) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29046315 (RTX 3060 Ti) | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29118472 (RTX 5090)    | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |
| vast-29129151 (RTX 5090)    | ❌ NO       | ❌ NO        | No logs                                         | **NOT STARTED** |

**Vast.ai Summary**: 2/14 healthy (14% success rate)

### RUNPOD NODES (6 total)

| Node                              | P2P Running | Port 8770    | Last Error | Status          |
| --------------------------------- | ----------- | ------------ | ---------- | --------------- |
| runpod-h100 (H100 PCIe)           | ✅ YES      | ✅ LISTENING | None       | **HEALTHY**     |
| runpod-a100-2 (A100 PCIe)         | ✅ YES      | ✅ LISTENING | None       | **HEALTHY**     |
| runpod-a100-storage-1 (A100 PCIe) | ✅ YES      | ✅ LISTENING | None       | **HEALTHY**     |
| runpod-a100-1 (A100 PCIe)         | ❌ NO       | ❌ NO        | No logs    | **NOT STARTED** |
| runpod-l40s-2 (L40S)              | ❌ NO       | ❌ NO        | No logs    | **NOT STARTED** |
| runpod-3090ti-1 (RTX 3090 Ti)     | ❌ NO       | ❌ NO        | No logs    | **NOT STARTED** |

**RunPod Summary**: 3/6 healthy (50% success rate)

### VULTR NODES (2 total)

| Node                           | P2P Running | Port 8770    | Last Error                                    | Status      |
| ------------------------------ | ----------- | ------------ | --------------------------------------------- | ----------- |
| vultr-a100-20gb (A100D vGPU)   | ✅ YES      | ✅ LISTENING | None                                          | **HEALTHY** |
| vultr-a100-20gb-2 (A100D vGPU) | ❌ NO       | ❌ NO        | `ModuleNotFoundError: No module named 'yaml'` | **CRASHED** |

**Vultr Summary**: 1/2 healthy (50% success rate)

### NEBIUS NODES (3 total)

| Node                     | P2P Running | Port 8770    | Last Error | Status                  |
| ------------------------ | ----------- | ------------ | ---------- | ----------------------- |
| nebius-backbone-1 (L40S) | ✅ YES      | ✅ LISTENING | None       | **HEALTHY** (P2P Voter) |
| nebius-h100-3 (H100 SXM) | ✅ YES      | ✅ LISTENING | None       | **HEALTHY** (P2P Voter) |
| nebius-h100-1 (H100 SXM) | ❌ NO       | ❌ NO        | No logs    | **NOT STARTED**         |

**Nebius Summary**: 2/3 healthy (67% success rate)

### HETZNER NODES (Not checked in this investigation)

These are CPU-only nodes and were not part of this GPU-focused investigation.

## P2P Voter Status

**Note:** This section reflects the config at the time of the investigation. The current voter list is now defined solely by `ai-service/config/distributed_hosts.yaml`.

According to `config/distributed_hosts.yaml`, the P2P voters should be:

- nebius-backbone-1 ✅ **HEALTHY**
- nebius-h100-3 ✅ **HEALTHY**
- aws-staging ⚠️ **NOT CHECKED**
- hetzner-cpu1 ⚠️ **NOT CHECKED**

**Voter Quorum**: 2/4 confirmed healthy (quorum requires 3). **QUORUM AT RISK** (historical snapshot).

## Root Causes Analysis

### 1. Missing Python Dependencies (2 nodes)

- **vast-28918742**: Missing `psutil`
- **vultr-a100-20gb-2**: Missing `yaml` (PyYAML)

These nodes have supervisor attempting to restart P2P every 10 seconds but failing immediately due to import errors.

**Error trace example**:

```
File "/workspace/ringrift/ai-service/app/coordination/safeguards.py", line 40
    import psutil
ModuleNotFoundError: No module named 'psutil'
```

### 2. P2P Never Started (18 nodes)

Most nodes show no evidence of P2P ever running:

- No log files in expected locations
- No listening port 8770
- `pgrep` false positives (matching strings in other process command lines)

Possible reasons:

- Initial deployment script didn't include P2P daemon startup
- Nodes were provisioned after P2P infrastructure was set up
- Manual startup required but never performed
- Nodes rebooted and P2P not configured to auto-start

### 3. False Positive PIDs

The investigation script used `pgrep -f "p2p_daemon|p2p_orchestrator"` which matches:

- Actual running processes ✅
- Strings in other process arguments/environment ❌
- Old/stale references ❌

This led to incorrect "P2P Running: YES" reports when verifying with `ps aux | grep <PID>` showed the process doesn't exist.

## Network Architecture Impact

With only 6/26 GPU nodes running P2P:

- **Data sync coverage**: Severely limited
- **Model distribution**: Manual intervention likely needed
- **Cluster coordination**: Fragmented
- **Leader election**: At risk (only 2 voters confirmed)

## Recommended Actions (DO NOT EXECUTE - REPORT ONLY)

### Immediate Actions

1. **Install missing dependencies** on crashed nodes:

   ```bash
   # On vast-28918742, vultr-a100-20gb-2
   pip install psutil PyYAML
   ```

2. **Start P2P on healthy nodes** (18 nodes):

   ```bash
   cd ~/ringrift/ai-service  # or /workspace/ringrift/ai-service
   nohup python scripts/p2p_orchestrator.py \
     --node-id $(hostname) \
     --host 0.0.0.0 \
     --port 8770 \
     --peers http://89.169.110.128:8770 \
     >/tmp/p2p.log 2>&1 &
   ```

   Use the current `p2p_voters` list in `ai-service/config/distributed_hosts.yaml` to populate peers.

3. **Verify voter quorum**: Ensure all 4 voter nodes are running:
   - nebius-backbone-1 ✅
   - nebius-h100-3 ✅
   - aws-staging ⚠️ (need to check)
   - hetzner-cpu1 ⚠️ (need to check)

### Medium-term Actions

4. **Create systemd/supervisor services** for auto-restart
5. **Add P2P health monitoring** to cluster_monitor.py
6. **Update deployment scripts** to include P2P startup
7. **Fix pgrep detection** in monitoring scripts (use `pgrep -x` or stricter patterns)

### Long-term Actions

8. **Automated P2P deployment** via Ansible/Salt
9. **P2P health alerts** to Slack/Discord
10. **Dependency verification** in setup scripts

## Files Generated

- Investigation script: `/Users/armand/Development/RingRift/ai-service/scripts/investigate_p2p_status.sh`
- Raw report: `/tmp/p2p_status_report_20251226_203240.txt`
- This summary: `/Users/armand/Development/RingRift/P2P_INVESTIGATION_SUMMARY.md`

## Next Steps

**DO NOT auto-fix. Wait for user confirmation before:**

1. Installing missing dependencies
2. Starting P2P daemons
3. Modifying any node configurations
4. Updating deployment scripts

---

_Investigation completed: 2025-12-26 20:40 CST_
