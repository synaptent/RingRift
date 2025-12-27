# P2P Orchestrator Handlers Reference

This document describes the handler mixins used by the P2P orchestrator to manage distributed cluster operations.

## Overview

The P2P orchestrator uses a mixin-based architecture with 12 handler classes, each responsible for a specific domain:

| Handler                      | Purpose                    | Key Endpoints                              |
| ---------------------------- | -------------------------- | ------------------------------------------ |
| `AdminHandlersMixin`         | Cluster admin operations   | `/admin/shutdown`, `/admin/restart`        |
| `CMAESHandlersMixin`         | Hyperparameter evolution   | `/cmaes/start`, `/cmaes/status`            |
| `ElectionHandlersMixin`      | Leader election            | `/election/vote`, `/election/heartbeat`    |
| `EloSyncHandlersMixin`       | Elo rating synchronization | `/elo/sync`, `/elo/rankings`               |
| `GauntletHandlersMixin`      | Model evaluation gauntlet  | `/gauntlet/run`, `/gauntlet/results`       |
| `GossipHandlersMixin`        | Peer discovery & gossip    | `/gossip`, `/peers`                        |
| `RaftHandlersMixin`          | Raft consensus protocol    | `/raft/append`, `/raft/vote`               |
| `RelayHandlersMixin`         | NAT traversal relay        | `/relay/forward`, `/relay/register`        |
| `SSHTournamentHandlersMixin` | SSH-based tournaments      | `/ssh_tournament/run`                      |
| `SwimHandlersMixin`          | SWIM membership protocol   | `/swim/ping`, `/swim/indirect-ping`        |
| `TournamentHandlersMixin`    | Model tournaments          | `/tournament/start`, `/tournament/bracket` |
| `WorkQueueHandlersMixin`     | Distributed work queue     | `/work/claim`, `/work/complete`            |

## Handler Details

### AdminHandlersMixin (`admin.py`)

Provides administrative control over cluster nodes.

**Endpoints:**

- `POST /admin/shutdown` - Gracefully shutdown node
- `POST /admin/restart` - Restart P2P orchestrator
- `GET /admin/health` - Detailed health check

**Usage:** Used by cluster operators for maintenance and troubleshooting.

### CMAESHandlersMixin (`cmaes.py`)

Manages distributed CMA-ES hyperparameter optimization.

**Endpoints:**

- `POST /cmaes/start` - Start CMA-ES optimization job
- `GET /cmaes/status` - Get optimization status
- `POST /cmaes/report` - Report candidate evaluation result

**Features:**

- Distributed candidate evaluation across GPU nodes
- Automatic best hyperparameter tracking
- Early stopping support

### ElectionHandlersMixin (`election.py`)

Implements Bully algorithm leader election with voter quorum.

**Endpoints:**

- `POST /election/vote` - Cast election vote
- `POST /election/heartbeat` - Leader heartbeat
- `GET /election/leader` - Get current leader

**Key Concepts:**

- **Voters**: Stable nodes (Nebius, Hetzner) with quorum requirement
- **Non-voters**: Ephemeral nodes (Vast.ai) that follow but don't vote
- **Heartbeat interval**: 15 seconds

### EloSyncHandlersMixin (`elo_sync.py`)

Synchronizes Elo ratings across the cluster.

**Endpoints:**

- `POST /elo/sync` - Push local Elo updates
- `GET /elo/rankings` - Get global rankings
- `POST /elo/merge` - Merge rating updates

**Features:**

- Eventual consistency model
- Conflict resolution via timestamp ordering
- Per-board-type rating tracking

### GauntletHandlersMixin (`gauntlet.py`)

Manages model evaluation gauntlet runs.

**Endpoints:**

- `POST /gauntlet/run` - Start gauntlet evaluation
- `GET /gauntlet/results` - Get evaluation results
- `POST /gauntlet/claim` - Claim gauntlet work item

**Gauntlet Types:**

- `quick`: 20 games, fast feedback
- `standard`: 50 games, promotion decisions
- `thorough`: 100 games, canonical validation

### GossipHandlersMixin (`gossip.py`)

Implements gossip protocol for peer discovery and state dissemination.

**Endpoints:**

- `POST /gossip` - Receive gossip message
- `GET /peers` - List known peers
- `POST /gossip/push` - Push state to peers

**Protocol:**

- Anti-entropy with pull-push sync
- Exponential backoff for failed nodes
- Automatic stale peer pruning

### RaftHandlersMixin (`raft.py`)

Implements Raft consensus for replicated state machines.

**Endpoints:**

- `POST /raft/append` - Append entries (leader only)
- `POST /raft/vote` - Request vote
- `GET /raft/state` - Get Raft state

**Features:**

- Log replication with persistence
- Snapshot-based log compaction
- Automatic leader election on timeout

**Integration:** Used with `app/p2p/raft_state.py` for replicated work queues.

### RelayHandlersMixin (`relay.py`)

Provides NAT traversal for nodes behind firewalls.

**Endpoints:**

- `POST /relay/register` - Register for relay service
- `POST /relay/forward` - Forward message to target
- `GET /relay/status` - Relay connection status

**Use Case:** Vast.ai and other containerized nodes often can't accept inbound connections. The relay hub (usually leader) forwards messages on their behalf.

### SSHTournamentHandlersMixin (`ssh_tournament.py`)

Runs tournaments via SSH to worker nodes.

**Endpoints:**

- `POST /ssh_tournament/run` - Start SSH tournament
- `GET /ssh_tournament/status` - Get tournament status

**Features:**

- Parallel execution across multiple workers
- Automatic result aggregation
- Failure recovery with retries

### SwimHandlersMixin (`swim.py`)

Implements SWIM membership protocol for failure detection.

**Endpoints:**

- `POST /swim/ping` - Direct ping
- `POST /swim/indirect-ping` - Indirect ping via proxy
- `GET /swim/members` - Get membership list

**Protocol Details:**

- Probabilistic failure detection
- Suspicion mechanism before marking dead
- Dissemination via piggyback gossip

### TournamentHandlersMixin (`tournament.py`)

Manages model tournaments with bracket/round-robin formats.

**Endpoints:**

- `POST /tournament/start` - Start tournament
- `GET /tournament/bracket` - Get bracket/standings
- `POST /tournament/result` - Report match result

**Formats:**

- Single elimination
- Double elimination
- Round robin

### WorkQueueHandlersMixin (`work_queue.py`)

Distributed work queue for selfplay/training jobs.

**Endpoints:**

- `POST /work/add` - Add work to queue
- `POST /work/claim` - Claim work item
- `POST /work/complete` - Mark work complete
- `GET /work/pending` - List pending work

**Features:**

- Priority-based scheduling
- Board-type affinity for GPUs
- Automatic requeue on worker failure
- Work stealing for load balancing

## Adding New Handlers

To add a new handler:

1. Create `handlers/your_handler.py`:

```python
class YourHandlersMixin:
    """Mixin for your feature."""

    async def handle_your_endpoint(self, request):
        """Handle POST /your/endpoint"""
        # Implementation
        pass
```

2. Add to `handlers/__init__.py`:

```python
from .your_handler import YourHandlersMixin
__all__.append("YourHandlersMixin")
```

3. Add to P2POrchestrator class inheritance:

```python
class P2POrchestrator(
    ...,
    YourHandlersMixin,
):
    pass
```

4. Register routes in orchestrator setup.

## Event Integration

P2P handlers are wired to the coordination EventRouter via `scripts/p2p/p2p_event_bridge.py`.

### Event Bridge Functions

| Handler                  | Event Function                  | Event Type Emitted                                                |
| ------------------------ | ------------------------------- | ----------------------------------------------------------------- |
| `WorkQueueHandlersMixin` | `emit_p2p_work_completed()`     | `SELFPLAY_COMPLETE`, `TRAINING_COMPLETED`, `EVALUATION_COMPLETED` |
| `WorkQueueHandlersMixin` | `emit_p2p_work_failed()`        | `TRAINING_FAILED`                                                 |
| `GauntletHandlersMixin`  | `emit_p2p_gauntlet_completed()` | `EVALUATION_COMPLETED`                                            |
| `GossipHandlersMixin`    | `emit_p2p_node_online()`        | `HOST_ONLINE`                                                     |
| `ElectionHandlersMixin`  | `emit_p2p_leader_changed()`     | `LEADER_CHANGED`                                                  |
| `EloSyncHandlersMixin`   | `emit_p2p_elo_updated()`        | `ELO_UPDATED`                                                     |

Emitter availability in `scripts/p2p_orchestrator.py` is cached: positive checks are reused,
and negative checks are retried every 30 seconds to allow late EventRouter initialization.

### Background Loops (`scripts/p2p/loops`)

The P2P orchestrator delegates periodic work to LoopManager-managed loops:

- `EloSyncLoop` - Periodic Elo synchronization
- `IdleDetectionLoop` - GPU idle resource detection
- `AutoScalingLoop` - Dynamic node scaling
- `JobReaperLoop` - Stale job cleanup
- `QueuePopulatorLoop` - Work queue maintenance
- `ValidationLoop` - Queues model validation work for newly trained models

Enable/disable via `RINGRIFT_EXTRACTED_LOOPS` (default `true`).

### Usage Example

```python
from scripts.p2p.p2p_event_bridge import (
    emit_p2p_work_completed,
    emit_p2p_node_online,
    emit_p2p_leader_changed,
)

# After work completes in work_queue handler
await emit_p2p_work_completed(
    work_id="abc123",
    work_type="selfplay",
    config_key="hex8_2p",
    result={"games_generated": 500},
    node_id="runpod-h100",
)

# When new peer discovered in gossip handler
await emit_p2p_node_online(
    node_id="new-node-1",
    host_type="gpu",
    capabilities={"has_gpu": True, "gpu_name": "H100"},
)

# When leadership changes in election handler
await emit_p2p_leader_changed(
    new_leader_id="nebius-h100-1",
    old_leader_id="runpod-h100",
    term=42,
)
```

### Direct Event Router Usage

For custom events not covered by the bridge:

```python
from app.coordination.event_router import publish

await publish(
    event_type="CUSTOM_P2P_EVENT",
    payload={"work_id": work_id, "node_id": self.node_id},
    source="p2p_handler",
)
```

See `docs/EVENT_CATALOG.md` for all available event types.

## Configuration

Handler behavior is configured via environment variables and `config/distributed_hosts.yaml`:

| Variable                      | Default | Description               |
| ----------------------------- | ------- | ------------------------- |
| `RINGRIFT_ELECTION_TIMEOUT`   | 30      | Election timeout seconds  |
| `RINGRIFT_GOSSIP_INTERVAL`    | 15      | Gossip interval seconds   |
| `RINGRIFT_WORK_CLAIM_TIMEOUT` | 300     | Work claim expiry seconds |

## See Also

- `scripts/p2p/managers/` - Domain logic managers
- `docs/EVENT_CATALOG.md` - Event types reference
- `docs/COORDINATION_ARCHITECTURE.md` - Overall architecture
