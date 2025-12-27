# SWIM/Raft Protocol Deployment Guide

**Last Updated:** 2025-12-27
**Status:** Production Ready (Optional Enhancement)

---

## Overview

The P2P orchestrator supports two optional protocols for improved cluster reliability:

| Protocol | Purpose                 | Benefit                                       |
| -------- | ----------------------- | --------------------------------------------- |
| **SWIM** | Gossip-based membership | 5s failure detection (vs 60-90s HTTP polling) |
| **Raft** | Replicated consensus    | Sub-second leader failover (vs 10-30s Bully)  |

**Important:** These protocols are _optional enhancements_. The default HTTP polling + Bully election is battle-tested and production-ready. Only enable SWIM/Raft if you need faster failure detection or leader failover.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    P2P Orchestrator                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐│
│  │ Membership Layer    │    │ Consensus Layer                 ││
│  │                     │    │                                 ││
│  │ HTTP Polling (default)   │ Bully Election (default)        ││
│  │ ────────────────────│    │ ─────────────────────────────   ││
│  │ SWIM Gossip (optional)   │ Raft Replication (optional)     ││
│  │                     │    │                                 ││
│  │ Feature Flag:       │    │ Feature Flag:                   ││
│  │ MEMBERSHIP_MODE     │    │ CONSENSUS_MODE                  ││
│  └─────────────────────┘    └─────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Hybrid Coordinator                                          ││
│  │ Routes operations based on feature flags                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Dependencies

```bash
# Install SWIM protocol library
pip install swim-p2p>=1.2.0

# Install Raft library
pip install pysyncobj>=0.3.14
```

### Port Requirements

| Port | Protocol | Purpose                   |
| ---- | -------- | ------------------------- |
| 8770 | TCP/HTTP | P2P orchestrator REST API |
| 7947 | UDP      | SWIM gossip protocol      |
| 4321 | TCP      | Raft consensus RPC        |

Ensure these ports are accessible between all cluster nodes.

---

## Configuration

### Environment Variables

#### Membership (SWIM)

```bash
# Enable SWIM membership protocol
export RINGRIFT_SWIM_ENABLED=true

# Membership mode: http, swim, or hybrid
# - http: HTTP polling only (default, most stable)
# - swim: SWIM gossip only
# - hybrid: SWIM primary, HTTP fallback
export RINGRIFT_MEMBERSHIP_MODE=hybrid

# SWIM port (default: 7947)
export RINGRIFT_SWIM_PORT=7947
```

#### Consensus (Raft)

```bash
# Enable Raft consensus protocol
export RINGRIFT_RAFT_ENABLED=true

# Consensus mode: bully, raft, or hybrid
# - bully: Bully election only (default, most stable)
# - raft: Raft replication only
# - hybrid: Raft primary, Bully fallback
export RINGRIFT_CONSENSUS_MODE=hybrid

# Raft port (default: 4321)
export RINGRIFT_RAFT_PORT=4321
```

---

## Deployment Stages

### Stage 1: SWIM Membership Only (Recommended Start)

The safest way to start is with SWIM in hybrid mode:

```bash
# On each node
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=hybrid
export RINGRIFT_RAFT_ENABLED=false
export RINGRIFT_CONSENSUS_MODE=bully
```

**What this does:**

- SWIM handles member discovery and failure detection
- Falls back to HTTP polling if SWIM fails
- Leader election continues using Bully algorithm

**Verify:**

```bash
curl -s http://localhost:8770/status | jq '.swim_raft'
# Expected:
# {
#   "membership_mode": "hybrid",
#   "consensus_mode": "bully",
#   "swim": { "enabled": true, "available": true },
#   "raft": { "enabled": false, "available": false }
# }
```

### Stage 2: SWIM as Primary

After verifying Stage 1 works for 24+ hours:

```bash
export RINGRIFT_MEMBERSHIP_MODE=swim
```

**What this does:**

- SWIM is now the only membership layer
- HTTP polling is disabled
- Faster failure detection (5s vs 60-90s)

### Stage 3: Add Raft Consensus (Optional)

If you need faster leader failover:

```bash
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=hybrid
```

**What this does:**

- Raft handles leader election and state replication
- Falls back to Bully if Raft fails
- Sub-second leader failover

### Stage 4: Full SWIM + Raft (Advanced)

Only for stable clusters after extensive testing:

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=raft
```

---

## Rollback Plan

Instant rollback via environment variables:

```bash
# Disable SWIM (revert to HTTP polling)
export RINGRIFT_SWIM_ENABLED=false
export RINGRIFT_MEMBERSHIP_MODE=http

# Disable Raft (revert to Bully election)
export RINGRIFT_RAFT_ENABLED=false
export RINGRIFT_CONSENSUS_MODE=bully
```

Restart P2P orchestrator on affected nodes:

```bash
pkill -f p2p_orchestrator
cd ~/ringrift/ai-service && nohup python scripts/p2p_orchestrator.py &
```

---

## Monitoring

### Status Endpoint

```bash
curl -s http://localhost:8770/status | jq '.swim_raft'
```

**Healthy Response:**

```json
{
  "membership_mode": "hybrid",
  "consensus_mode": "bully",
  "swim": {
    "enabled": true,
    "available": true,
    "members": 15,
    "alive": 14,
    "suspected": 1
  },
  "raft": {
    "enabled": false,
    "available": false
  },
  "hybrid_status": {
    "swim_fallback_active": false,
    "raft_fallback_active": true
  }
}
```

### Protocol-Specific Endpoints

```bash
# SWIM status
curl -s http://localhost:8770/swim/status

# Raft status
curl -s http://localhost:8770/raft/status
```

### Logs

SWIM/Raft protocol messages are logged with prefixes:

```
[Startup Validation] SWIM protocol available (membership_mode=hybrid)
[Startup Validation] Raft protocol available (consensus_mode=hybrid)
[SWIM] node-123 is now ALIVE
[SWIM] node-456 is now FAILED
[Raft] Became leader for term 42
```

---

## Troubleshooting

### SWIM Not Available

**Symptom:** Status shows `"swim_available": false`

**Cause:** `swim-p2p` library not installed

**Fix:**

```bash
pip install swim-p2p>=1.2.0
# Restart P2P orchestrator
```

### SWIM Port Binding Failed

**Symptom:** Log shows "Failed to bind to SWIM port"

**Cause:** Port 7947 already in use

**Fix:**

```bash
# Check what's using the port
lsof -i :7947

# Kill the process or use a different port
export RINGRIFT_SWIM_PORT=7948
```

### Raft Not Available

**Symptom:** Status shows `"raft_available": false`

**Cause:** `pysyncobj` library not installed

**Fix:**

```bash
pip install pysyncobj>=0.3.14
# Restart P2P orchestrator
```

### SWIM Partition Detected

**Symptom:** Multiple nodes show different membership views

**Cause:** Network partition or firewall blocking UDP port 7947

**Fix:**

1. Check firewall rules: `sudo iptables -L -n | grep 7947`
2. Check UDP connectivity between nodes
3. Temporarily revert to HTTP mode: `export RINGRIFT_MEMBERSHIP_MODE=http`

### Raft Split-Brain

**Symptom:** Multiple nodes claim to be leader

**Cause:** Network partition causing quorum loss

**Fix:**

1. Identify and resolve network partition
2. Temporarily revert to Bully: `export RINGRIFT_CONSENSUS_MODE=bully`
3. Once network stable, re-enable Raft

---

## Performance Tuning

### SWIM Configuration

```python
# In app/p2p/swim_adapter.py

@dataclass
class SwimConfig:
    failure_timeout: float = 5.0      # Time before marking node failed
    suspicion_timeout: float = 3.0    # Time in suspicion before failed
    ping_interval: float = 1.0        # Seconds between protocol rounds
    ping_request_group_size: int = 3  # Indirect ping group size
    max_transmissions: int = 10       # Max gossip transmissions
```

**Faster Detection (more bandwidth):**

```python
SwimConfig(
    failure_timeout=3.0,
    suspicion_timeout=2.0,
    ping_interval=0.5,
)
```

**Lower Bandwidth (slower detection):**

```python
SwimConfig(
    failure_timeout=10.0,
    suspicion_timeout=5.0,
    ping_interval=2.0,
)
```

### Raft Configuration

```python
# In app/p2p/raft_state.py

# Voter nodes (recommend 3 or 5 for quorum)
RAFT_VOTERS = [
    "nebius-backbone-1",
    "nebius-h100-3",
    "hetzner-cpu1",
    "hetzner-cpu2",
    "vultr-a100-20gb",
]
```

**Important:** Use an odd number of voters (3 or 5) for clean quorum decisions.

---

## File Reference

| File                              | Purpose                     |
| --------------------------------- | --------------------------- |
| `app/p2p/swim_adapter.py`         | SWIM protocol integration   |
| `app/p2p/raft_state.py`           | Raft state machines         |
| `app/p2p/hybrid_coordinator.py`   | Routes operations           |
| `app/p2p/constants.py`            | Feature flags and defaults  |
| `scripts/p2p/membership_mixin.py` | SWIM mixin for orchestrator |
| `scripts/p2p/consensus_mixin.py`  | Raft mixin for orchestrator |
| `scripts/p2p/handlers/swim.py`    | SWIM HTTP endpoints         |
| `scripts/p2p/handlers/raft.py`    | Raft HTTP endpoints         |

---

## Migration Checklist

- [ ] Install dependencies: `pip install swim-p2p>=1.2.0 pysyncobj>=0.3.14`
- [ ] Open UDP port 7947 between all nodes
- [ ] Open TCP port 4321 between voter nodes
- [ ] Set environment variables on all nodes
- [ ] Start with Stage 1 (SWIM hybrid mode)
- [ ] Monitor for 24+ hours before Stage 2
- [ ] Document node IPs and ports in cluster config
- [ ] Test rollback procedure

---

## See Also

- [CLUSTER_INTEGRATION_GUIDE.md](../CLUSTER_INTEGRATION_GUIDE.md) - Overall cluster architecture
- [P2P_MIGRATION_GUIDE.md](../P2P_MIGRATION_GUIDE.md) - P2P orchestrator setup
- [CLUSTER_OPERATIONS_RUNBOOK.md](./CLUSTER_OPERATIONS_RUNBOOK.md) - Operational procedures
