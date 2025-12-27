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

Handlers can emit events to the coordination layer:

```python
from app.coordination.event_router import get_event_router

router = get_event_router()
await router.emit("WORK_COMPLETED", {
    "work_id": work_id,
    "node_id": self.node_id,
})
```

See `docs/EVENT_CATALOG.md` for available event types.

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
