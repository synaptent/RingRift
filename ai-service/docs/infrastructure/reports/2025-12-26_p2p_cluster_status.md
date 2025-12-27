# P2P Cluster Status Report

> **Report Date**: December 26-27, 2025
> **Status**: CONSOLIDATED from multiple investigation sources
> **Config Source of Truth**: `ai-service/config/distributed_hosts.yaml`

## Executive Summary

P2P cluster underwent voter reconfiguration on Dec 27, 2025. Current state:

- **Voter Quorum**: 5 voters configured, quorum requires 3 (majority)
- **Cluster Coverage**: ~10-14 nodes with P2P running (32-47% of cluster)
- **Leader Election**: Functional with current voter set

## Current Voter Configuration

From `config/distributed_hosts.yaml` (authoritative):

| Voter Node        | Provider | Hardware       | IP              | Status |
| ----------------- | -------- | -------------- | --------------- | ------ |
| nebius-backbone-1 | Nebius   | L40S 48GB      | 89.169.112.47   | Active |
| nebius-h100-3     | Nebius   | H100 SXM 80GB  | 89.169.110.128  | Active |
| hetzner-cpu1      | Hetzner  | CPU (8 cores)  | 46.62.147.150   | Active |
| hetzner-cpu2      | Hetzner  | CPU (16 cores) | 135.181.39.239  | Active |
| vultr-a100-20gb   | Vultr    | A100 vGPU 20GB | 208.167.249.164 | Active |

**Voter Selection Criteria** (Dec 27 update):

- Non-containerized (bare metal preferred for stability)
- Direct IP (not NAT-blocked)
- Stable providers (Nebius, Hetzner, Vultr)
- Removed: runpod-h100 (containerized)

## Cluster Node Summary

### By Provider

| Provider | Nodes | P2P Enabled | Notes                  |
| -------- | ----- | ----------- | ---------------------- |
| Vast.ai  | 14    | Yes         | NAT-blocked, ephemeral |
| RunPod   | 7     | Yes         | Containerized          |
| Nebius   | 3     | Yes         | Bare metal, 2 voters   |
| Vultr    | 2     | Yes         | VM, 1 voter            |
| Hetzner  | 3     | Yes         | Bare metal, 2 voters   |
| Local    | 2     | Yes         | Coordinators           |

### Known Issues (Dec 26 Investigation)

1. **Missing Dependencies** (2 nodes):
   - `vast-28918742`: Missing `psutil`
   - `vultr-a100-20gb-2`: Missing `PyYAML`

2. **P2P Never Started** (~19 nodes):
   - No logs or listening ports found
   - Nodes provisioned without P2P deployment

3. **SSH Connectivity Issues** (~9 nodes):
   - Timeout during investigation
   - May be temporary network issues or node offline

## Deployment Commands

### Check Current Status

```bash
# Check all nodes
python scripts/check_p2p_status_all_nodes.py

# Check specific nodes
python scripts/deploy_p2p_systemd.py --check --nodes "runpod-*"
```

### Deploy Systemd Service

```bash
# Deploy to all P2P-enabled nodes
python scripts/deploy_p2p_systemd.py

# Deploy to specific pattern
python scripts/deploy_p2p_systemd.py --nodes "vast-*"

# Dry run (preview)
python scripts/deploy_p2p_systemd.py --dry-run

# Include local nodes (default: excluded)
python scripts/deploy_p2p_systemd.py --include-local
```

### Manual P2P Start

```bash
# On a remote node
ssh <node> 'cd ~/ringrift/ai-service && \
  nohup python scripts/p2p_orchestrator.py \
    --node-id $(hostname) \
    --host 0.0.0.0 \
    --port 8770 \
    --peers http://89.169.112.47:8770,http://89.169.110.128:8770 \
    >/tmp/p2p.log 2>&1 &'
```

### Fix Missing Dependencies

```bash
# Install psutil and PyYAML
ssh <node> 'source ~/ringrift/ai-service/venv/bin/activate && pip install psutil PyYAML'
```

## Health Check Endpoints

```bash
# Check P2P health
curl -s http://<node-ip>:8770/health

# Check P2P status (includes leader, peers)
curl -s http://<node-ip>:8770/status | python3 -m json.tool

# Quick cluster summary (from any node)
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive peers: {d.get(\"alive_peers\")}")
print(f"Role: {d.get(\"role\")}")
'
```

## Configuration Reference

### Seed Peers (from voters)

The deployment script automatically derives seed peers from `p2p_voters` in `distributed_hosts.yaml`:

```
http://89.169.112.47:8770   # nebius-backbone-1
http://89.169.110.128:8770  # nebius-h100-3
http://46.62.147.150:8770   # hetzner-cpu1
http://135.181.39.239:8770  # hetzner-cpu2
http://208.167.249.164:8770 # vultr-a100-20gb
```

### Environment Variables

| Variable        | Default       | Description                    |
| --------------- | ------------- | ------------------------------ |
| `NODE_ID`       | hostname      | Node identifier for P2P mesh   |
| `P2P_PORT`      | 8770          | P2P orchestrator port          |
| `P2P_SEEDS`     | (from config) | Comma-separated seed peer URLs |
| `RINGRIFT_PATH` | ~/ringrift    | Root ringrift directory        |

## Report Provenance

This report consolidates data from:

1. `P2P_INVESTIGATION_SUMMARY.md` (Dec 26, 20:40) - Initial investigation
2. `P2P_STATUS_TABLE.md` (Dec 26, 20:45) - Detailed node status
3. `P2P_STATUS_SUMMARY.txt` (Dec 26) - Quick reference
4. `P2P_DEPLOYMENT_REPORT.md` (Dec 26) - Deployment results

The original investigation used a different voter list. The Dec 27 config update
(replacing runpod-h100 with hetzner-cpu2) is now reflected in this consolidated report.

---

_Consolidated: December 27, 2025_
_Source of truth for voters: `ai-service/config/distributed_hosts.yaml`_
