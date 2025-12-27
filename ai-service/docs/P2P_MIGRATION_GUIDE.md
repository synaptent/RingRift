# P2P SWIM/Raft Migration Guide

This guide documents the migration of the P2P orchestrator from HTTP-based membership and Bully election to SWIM gossip and Raft consensus protocols.

## Overview

The P2P orchestrator is transitioning to battle-tested protocols:

| Component       | Current                           | Target                        | Benefit                         |
| --------------- | --------------------------------- | ----------------------------- | ------------------------------- |
| Membership      | HTTP polling (60-90s detection)   | SWIM gossip (5s detection)    | 12-18x faster failure detection |
| Leader Election | Bully algorithm (10-30s failover) | Raft consensus (<1s failover) | 10-30x faster failover          |
| Work Queue      | SQLite + HTTP                     | Raft ReplDict                 | Consistent replicated state     |

## Prerequisites

Install the required dependencies:

```bash
pip install pysyncobj>=0.3.14 swim-p2p>=1.2.0
```

Verify installation:

```bash
python -c "import pysyncobj; print(f'pysyncobj {pysyncobj.__version__}')"
python -c "import swimprotocol; print('swim-p2p available')"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    P2POrchestrator                              │
│  (27K+ lines, mixin-based, HTTP REST API)                       │
├─────────────────────────────────────────────────────────────────┤
│  MembershipMixin          │  ConsensusMixin                     │
│  (SWIM gossip)            │  (Raft replication)                 │
│  ├─ 5s failure detection  │  ├─ Replicated work queue           │
│  ├─ O(1) bandwidth        │  ├─ Distributed locks               │
│  └─ Leaderless            │  └─ Sub-second failover             │
├─────────────────────────────────────────────────────────────────┤
│  HybridCoordinator (app/p2p/hybrid_coordinator.py)              │
│  Routes operations to SWIM/Raft/HTTP based on feature flags     │
└─────────────────────────────────────────────────────────────────┘
```

## Gradual Rollout Phases

### Phase 1: HTTP Baseline (Current)

All nodes continue using HTTP polling and Bully election:

```bash
# Default configuration - no changes needed
export RINGRIFT_SWIM_ENABLED=false
export RINGRIFT_RAFT_ENABLED=false
export RINGRIFT_MEMBERSHIP_MODE=http
export RINGRIFT_CONSENSUS_MODE=bully
```

### Phase 2: SWIM Hybrid Mode

Enable SWIM alongside HTTP for membership. SWIM is preferred when available, HTTP is fallback:

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=hybrid
export RINGRIFT_CONSENSUS_MODE=bully
```

Monitor with:

```bash
curl -s http://localhost:8770/status | jq '.swim_raft'
curl -s http://localhost:8770/swim/status
```

### Phase 3: SWIM Primary

Use SWIM as the primary membership protocol (HTTP disabled):

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim
export RINGRIFT_CONSENSUS_MODE=bully
```

### Phase 4: Raft Hybrid Mode

Enable Raft alongside Bully for consensus. Raft is preferred when quorum available:

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=hybrid
```

Monitor with:

```bash
curl -s http://localhost:8770/raft/status
```

### Phase 5: Full SWIM + Raft

Complete migration to SWIM and Raft:

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=raft
```

## Feature Flags

| Flag                       | Default | Options           | Description                 |
| -------------------------- | ------- | ----------------- | --------------------------- |
| `RINGRIFT_SWIM_ENABLED`    | false   | true/false        | Enable SWIM gossip protocol |
| `RINGRIFT_MEMBERSHIP_MODE` | http    | http/swim/hybrid  | Membership protocol to use  |
| `RINGRIFT_RAFT_ENABLED`    | false   | true/false        | Enable Raft consensus       |
| `RINGRIFT_CONSENSUS_MODE`  | bully   | bully/raft/hybrid | Consensus protocol to use   |
| `RINGRIFT_SWIM_PORT`       | 7947    | integer           | Port for SWIM gossip        |
| `RINGRIFT_RAFT_PORT`       | 4321    | integer           | Port for Raft RPC           |

## New Files

### Core Components

| File                              | Purpose                                     |
| --------------------------------- | ------------------------------------------- |
| `scripts/p2p/membership_mixin.py` | SWIM membership mixin for P2POrchestrator   |
| `scripts/p2p/consensus_mixin.py`  | Raft consensus mixin for P2POrchestrator    |
| `app/p2p/hybrid_coordinator.py`   | Routes operations based on feature flags    |
| `app/p2p/swim_adapter.py`         | SWIM protocol adapter                       |
| `app/p2p/raft_state.py`           | Raft state machines (replicated work queue) |

### HTTP Endpoints

| File                           | Endpoints                       |
| ------------------------------ | ------------------------------- |
| `scripts/p2p/handlers/swim.py` | `/swim/status`, `/swim/members` |
| `scripts/p2p/handlers/raft.py` | `/raft/status`, `/raft/work`    |

## Monitoring

### Status Endpoint

The `/status` endpoint includes SWIM/Raft status in the `swim_raft` field:

```json
{
  "swim_raft": {
    "membership_mode": "hybrid",
    "consensus_mode": "raft",
    "swim": {
      "enabled": true,
      "available": true,
      "started": true,
      "alive_count": 15,
      "suspected_count": 0,
      "failed_count": 1
    },
    "raft": {
      "enabled": true,
      "available": true,
      "initialized": true,
      "is_leader": true,
      "leader_address": "100.88.35.19:4321",
      "work_queue_size": 42,
      "pending_jobs": 5
    },
    "hybrid_status": {
      "swim_fallback_active": false,
      "raft_fallback_active": false
    }
  }
}
```

### SWIM-Specific Status

```bash
curl -s http://localhost:8770/swim/status
```

Returns membership details including alive, suspected, and failed nodes.

### Raft-Specific Status

```bash
curl -s http://localhost:8770/raft/status
```

Returns Raft state including leader status, term, and replicated state.

## Rollback Procedure

Instant rollback via environment variables:

```bash
# Emergency rollback to HTTP/Bully
export RINGRIFT_SWIM_ENABLED=false
export RINGRIFT_RAFT_ENABLED=false
export RINGRIFT_MEMBERSHIP_MODE=http
export RINGRIFT_CONSENSUS_MODE=bully

# Restart P2P orchestrator
pkill -f p2p_orchestrator.py
python scripts/p2p_orchestrator.py
```

Nodes without the new dependencies continue working normally with HTTP/Bully.

## Troubleshooting

### SWIM Not Starting

1. Check dependency: `python -c "import swimprotocol"`
2. Check port availability: `lsof -i :7947`
3. Check logs for SWIM initialization errors

### Raft Not Forming Quorum

1. Check dependency: `python -c "import pysyncobj"`
2. Verify voter nodes are reachable on port 4321
3. Check that at least 3/5 voters are online
4. Check logs for Raft RPC errors

### Fallback Active

If `swim_fallback_active` or `raft_fallback_active` is true:

1. Dependencies may be missing - check installation
2. Protocol may have failed to initialize - check logs
3. This is expected during gradual rollout

### High Memory Usage

SWIM and Raft maintain in-memory state. If memory is constrained:

```bash
# Reduce SWIM gossip frequency
export RINGRIFT_SWIM_GOSSIP_INTERVAL=5  # Default: 1s

# Reduce Raft snapshot frequency
export RINGRIFT_RAFT_SNAPSHOT_INTERVAL=300  # Default: 60s
```

## Testing

Run integration tests for the new endpoints:

```bash
# Unit tests
pytest tests/unit/p2p/test_membership_mixin.py -v
pytest tests/unit/p2p/test_consensus_mixin.py -v

# Integration tests (requires running cluster)
pytest tests/integration/p2p/test_swim_endpoints.py -v
pytest tests/integration/p2p/test_raft_endpoints.py -v
```

## Migration Timeline

| Phase                   | Status  | Target               |
| ----------------------- | ------- | -------------------- |
| Phase 1: HTTP Baseline  | Current | -                    |
| Phase 2: SWIM Hybrid    | Ready   | After deps installed |
| Phase 3: SWIM Primary   | Ready   | After hybrid stable  |
| Phase 4: Raft Hybrid    | Ready   | After SWIM stable    |
| Phase 5: Full SWIM+Raft | Ready   | After hybrid stable  |

## See Also

- `ai-service/CLAUDE.md` - P2P SWIM/Raft Transition section
- `scripts/p2p/README.md` - P2P module architecture
- `scripts/p2p/managers/README.md` - Manager module documentation
