# P2P Orchestrator Module

This package provides the distributed P2P orchestrator for RingRift AI training.
The orchestrator coordinates selfplay, training, and data sync across a cluster of nodes.

## Architecture

```
scripts/p2p/
├── __init__.py           # Package exports, backward compatibility
├── constants.py          # Configuration constants (421 lines)
├── types.py              # Enums: NodeRole, JobType (34 lines)
├── models.py             # Dataclasses: NodeInfo, ClusterJob (771 lines)
├── network.py            # HTTP client, circuit breaker (181 lines)
├── resource.py           # Resource checking utilities (115 lines)
├── cluster_config.py     # Cluster configuration loading (265 lines)
├── client.py             # P2P client for external use (441 lines)
├── utils.py              # General utilities (46 lines)
└── README.md             # This file
```

## Decomposition Plan

The main `p2p_orchestrator.py` (29,767 lines, 510 methods) needs decomposition into:

### Phase 1: HTTP Handler Extraction (TARGET: ~15,000 lines)

Extract handlers into `scripts/p2p/handlers/`:

| Module          | Lines | Methods | Status                |
| --------------- | ----- | ------- | --------------------- |
| `work_queue.py` | 471   | 11      | ✅ Extracted          |
| `election.py`   | 349   | 5       | ✅ Extracted          |
| `relay.py`      | 368   | 4       | ✅ Extracted          |
| `gauntlet.py`   | 382   | 3       | ✅ Extracted          |
| `gossip.py`     | 226   | 2       | ✅ Extracted          |
| `admin.py`      | 131   | 3       | ✅ Extracted          |
| `elo_sync.py`   | 175   | 4       | ✅ Extracted          |
| `data_sync.py`  | ~1000 | 12      | Pending (interleaved) |
| `dashboard.py`  | ~2000 | 20      | Pending               |
| `api.py`        | ~2500 | 25      | Pending (scattered)   |

**Progress: 29,767 → 28,194 lines (-1,573 lines, ~5.3% reduction)**

### Phase 2: Core Logic Extraction (TARGET: ~10,000 lines)

| Module               | Lines | Methods | Description                     |
| -------------------- | ----- | ------- | ------------------------------- |
| `peer_manager.py`    | ~3000 | 40      | Peer discovery, gossip, cache   |
| `leader_election.py` | ~2000 | 25      | Bully algorithm, leases, voters |
| `job_scheduler.py`   | ~3000 | 35      | Work queue, auto-scaling        |
| `data_sync_core.py`  | ~2000 | 20      | P2P rsync, manifests            |

### Phase 3: Final Orchestrator (~5,000 lines)

The remaining P2POrchestrator class should coordinate:

- Lifecycle management (start, stop, run loops)
- Route registration
- State persistence
- Top-level coordination

## Handler Groups (Current p2p_orchestrator.py)

### Work Queue Handlers (lines 6776-7112)

- `handle_work_add`
- `handle_work_add_batch`
- `handle_work_claim`
- `handle_work_start`
- `handle_work_complete`
- `handle_work_fail`
- `handle_work_status`
- `handle_populator_status`
- `handle_work_for_node`
- `handle_work_cancel`
- `handle_work_history`

### Election Handlers (lines 7112-7349)

- `handle_election`
- `handle_lease_request`
- `handle_voter_grant_status`
- `handle_election_reset`
- `handle_election_force_leader`

### Gauntlet Handlers (lines 9055-9529)

- `handle_gauntlet_execute`
- `handle_gauntlet_status`
- `handle_gauntlet_quick_eval`

### Dashboard/API Handlers (lines 19073-21370)

- Analytics: `handle_games_analytics`, `handle_training_metrics`
- A/B Testing: `handle_abtest_*`
- Jobs API: `handle_api_jobs_*`
- Dashboard: `handle_dashboard`, `handle_work_queue_dashboard`

## Migration Strategy

1. **Create mixin classes** for handler groups
2. **P2POrchestrator inherits** from mixins
3. **Test after each extraction**
4. **Update imports** for backward compatibility

## Usage

```python
# External client usage
from scripts.p2p import P2PClient

client = P2PClient("http://localhost:8770")
status = await client.get_status()

# Constants
from scripts.p2p.constants import PEER_TIMEOUT, HEARTBEAT_INTERVAL

# Models
from scripts.p2p.models import NodeInfo, ClusterJob
```

## File Locations

- Main orchestrator: `scripts/p2p_orchestrator.py`
- Package: `scripts/p2p/`
- Tests: `tests/unit/scripts/test_p2p_*.py`
