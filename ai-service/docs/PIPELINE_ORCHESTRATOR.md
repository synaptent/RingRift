# Pipeline Orchestrator

The pipeline orchestrator (`scripts/pipeline_orchestrator.py`) provides unified coordination for the canonical AI training pipeline across distributed compute resources.

## Overview

The pipeline orchestrator manages three core phases:

1. **Canonical Self-Play** - Generate training games using the canonical 7-phase FSM
2. **Parity Validation** - Verify TS/Python replay parity
3. **NPZ Export** - Export validated games to training format

## Backends

The orchestrator supports two backend modes for job dispatch:

### SSH Backend (Default)

Traditional SSH-based job execution. Requires SSH key authentication to worker hosts.

```bash
python scripts/pipeline_orchestrator.py \
  --backend ssh \
  --config config/pipeline_config.yaml \
  run canonical_selfplay
```

### P2P Backend

Uses the P2P orchestrator REST API for job dispatch. Nodes communicate via HTTP without requiring SSH access between them.

```bash
python scripts/pipeline_orchestrator.py \
  --backend p2p \
  --p2p-leader http://lambda-gpu:8770 \
  --p2p-auth-token "your-cluster-token" \
  run canonical_selfplay
```

## P2P Integration

When using the P2P backend, the pipeline orchestrator communicates with the P2P orchestrator cluster:

```
┌─────────────────────┐
│ Pipeline Orchestrator│
│  (pipeline_orchestrator.py)
│                     │
│  --backend p2p      │
│  --p2p-leader URL   │
└─────────┬───────────┘
          │ REST API
          v
┌─────────────────────┐     UDP Discovery    ┌─────────────────┐
│  P2P Leader Node    │<-------------------->│  P2P Worker Node │
│  (p2p_orchestrator) │                      │  (p2p_orchestrator)
│                     │                      │                  │
│  Port 8770          │                      │  Port 8770       │
│  Coordinates jobs   │                      │  Executes jobs   │
└─────────────────────┘                      └─────────────────┘
```

### P2P Orchestrator Endpoints

The P2P orchestrator exposes these endpoints for pipeline integration:

| Endpoint                    | Method | Description                       |
| --------------------------- | ------ | --------------------------------- |
| `/pipeline/start`           | POST   | Start a pipeline phase            |
| `/pipeline/status`          | GET    | Get pipeline execution status     |
| `/pipeline/selfplay_worker` | POST   | Start selfplay on a specific node |

### Starting P2P Orchestrator

On each node that should participate:

```bash
# Basic start (auto-discovers peers via UDP broadcast)
python scripts/p2p_orchestrator.py --host 0.0.0.0 --port 8770

# With authentication (recommended for non-LAN deployments)
python scripts/p2p_orchestrator.py \
  --host 0.0.0.0 \
  --port 8770 \
  --require-auth

# With Tailscale for cross-network clusters
python scripts/p2p_orchestrator.py \
  --host 0.0.0.0 \
  --port 8770 \
  --tailscale-only \
  --require-auth
```

See [P2P_ORCHESTRATOR_AUTH.md](P2P_ORCHESTRATOR_AUTH.md) for authentication setup.

## CLI Reference

### Global Options

| Option                   | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| `--backend {ssh,p2p}`    | Job dispatch backend (default: ssh)                       |
| `--p2p-leader URL`       | P2P leader node URL (required for p2p backend)            |
| `--p2p-auth-token TOKEN` | Cluster auth token (or set `RINGRIFT_CLUSTER_AUTH_TOKEN`) |
| `--config PATH`          | Pipeline configuration file                               |
| `--state PATH`           | State file for resumption                                 |
| `--dry-run`              | Show what would be executed                               |

### Commands

#### `run <phase>`

Execute a pipeline phase:

```bash
# Run canonical self-play
python scripts/pipeline_orchestrator.py run canonical_selfplay \
  --board square8 \
  --num-games 1000 \
  --output-db data/games/canonical_square8.db

# Run parity validation
python scripts/pipeline_orchestrator.py run parity_validation \
  --db data/games/canonical_square8.db

# Export to NPZ
python scripts/pipeline_orchestrator.py run npz_export \
  --db data/games/canonical_square8.db \
  --output-dir data/training/
```

#### `status`

Check pipeline status:

```bash
python scripts/pipeline_orchestrator.py status
```

#### `cluster`

Manage the compute cluster:

```bash
# List nodes
python scripts/pipeline_orchestrator.py cluster list

# Check health
python scripts/pipeline_orchestrator.py cluster health

# Sync data across nodes
python scripts/pipeline_orchestrator.py cluster sync

# Update code from git
python scripts/pipeline_orchestrator.py cluster update
```

## Configuration

### Pipeline Config (`config/pipeline_config.yaml`)

```yaml
# Default board configuration
default_board: square8

# Self-play settings
selfplay:
  games_per_worker: 100
  max_moves: 300
  timeout_seconds: 300

# Parity validation
parity:
  sample_size: 100 # Games to validate (0 = all)
  fail_threshold: 0.01 # Max allowed divergence rate

# NPZ export
export:
  compression: true
  split_ratio: 0.9 # train/val split
```

### Worker Config (`config/distributed_hosts.yaml`)

```yaml
hosts:
  lambda-gpu:
    ssh_host: lambda-gpu # SSH config alias
    role: mixed # selfplay + training
    capabilities:
      - gpu
      - large_memory
    max_parallel_jobs: 4

  ringrift-staging:
    ssh_host: ringrift-staging
    role: selfplay
    max_parallel_jobs: 2
```

## Pipeline Phases

### Canonical Self-Play

Generates training games using the canonical 7-phase FSM:

1. **ring_placement** - Place ring from hand
2. **ring_movement** - Move ring on board
3. **line_processing** - Handle formed lines
4. **territory_processing** - Process territory claims
5. **stack_collapse** - Collapse oversized stacks
6. **recovery_action** - Recovery from elimination
7. **game_over** - Terminal state

Games are stored in SQLite databases under `data/games/canonical_*.db`.

### Parity Validation

Verifies TypeScript and Python engines produce identical results:

```bash
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8.db \
  --sample 100
```

### NPZ Export

Exports validated games to NumPy format for training:

```bash
python scripts/export_replay_dataset.py \
  --db data/games/canonical_square8.db \
  --output data/training/canonical_square8.npz
```

## Workflow Examples

### Full Training Pipeline (SSH)

```bash
# 1. Generate games across cluster
python scripts/pipeline_orchestrator.py \
  --backend ssh \
  run canonical_selfplay \
  --board square8 \
  --num-games 10000

# 2. Validate parity
python scripts/pipeline_orchestrator.py run parity_validation

# 3. Export for training
python scripts/pipeline_orchestrator.py run npz_export

# 4. Train model
python scripts/train_from_canonical.py \
  --data data/training/canonical_square8.npz
```

### Full Training Pipeline (P2P)

```bash
# Start P2P orchestrators on all nodes first
# (See P2P_ORCHESTRATOR_AUTH.md)

# Then run pipeline via P2P
python scripts/pipeline_orchestrator.py \
  --backend p2p \
  --p2p-leader http://leader-host:8770 \
  run canonical_selfplay \
  --board square8 \
  --num-games 10000

# Continue with validation and export...
```

### Distributed Self-Play with Auto-Sync

```bash
# Sync code to all nodes, run selfplay, sync results back
python scripts/pipeline_orchestrator.py \
  --backend p2p \
  --p2p-leader http://leader:8770 \
  cluster update && \
python scripts/pipeline_orchestrator.py \
  run canonical_selfplay \
  --board square8 \
  --num-games 5000 && \
python scripts/pipeline_orchestrator.py \
  cluster sync
```

## Monitoring

### Pipeline State

State is persisted to `~/.ringrift/pipeline_state.json`:

```json
{
  "current_phase": "canonical_selfplay",
  "phase_status": "running",
  "games_completed": 4523,
  "games_target": 10000,
  "started_at": "2025-01-15T10:30:00Z",
  "workers": {
    "lambda-gpu": { "status": "active", "games": 2100 },
    "staging": { "status": "active", "games": 1200 }
  }
}
```

### Logs

- Pipeline logs: `logs/pipeline_orchestrator.log`
- P2P orchestrator logs: `logs/p2p_orchestrator.log`
- Per-worker logs: `logs/selfplay_<host>_<timestamp>.log`

## Troubleshooting

### P2P Backend Connection Failed

```
Error: Cannot connect to P2P leader at http://host:8770
```

1. Verify P2P orchestrator is running on leader: `curl http://host:8770/health`
2. Check firewall allows port 8770
3. For Tailscale, verify both nodes are on the tailnet

### Parity Validation Failures

```
Error: 5 games failed parity check
```

1. Check state bundles: `python scripts/diff_state_bundle.py <bundle.json>`
2. Review phase transitions in failing games
3. Ensure both TS and Python engines are at same version

### Worker Not Receiving Jobs

1. Check worker registration: `curl http://leader:8770/cluster/status`
2. Verify worker has required capabilities for the job
3. Check worker health: `curl http://worker:8770/health`

## Related Documentation

- [DISTRIBUTED_SELFPLAY.md](DISTRIBUTED_SELFPLAY.md) - Basic distributed setup
- [P2P_ORCHESTRATOR_AUTH.md](P2P_ORCHESTRATOR_AUTH.md) - P2P authentication
- [GPU_PIPELINE_ROADMAP.md](GPU_PIPELINE_ROADMAP.md) - GPU acceleration plans
