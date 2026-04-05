# CLAUDE.md - AI Assistant Context for RingRift

This file provides context for AI assistants working on this codebase.

## What is RingRift?

A multiplayer territory control board game where players place pieces to claim territory. Features:

- Multiple board geometries (square, hexagonal) and sizes
- 2-4 player support
- Neural network AI opponents trained via self-play
- Real-time multiplayer with matchmaking

## Repository Structure

```
RingRift/
├── src/                    # TypeScript source
│   ├── client/            # React frontend
│   ├── server/            # Node.js game server
│   └── shared/            # Shared game engine (SOURCE OF TRUTH for rules)
│       ├── engine/        # Core game logic
│       └── types/         # Type definitions
├── ai-service/            # Python ML pipeline (see ai-service/CLAUDE.md)
│   ├── app/              # Core modules
│   ├── scripts/          # CLI tools
│   └── data/             # Databases and training data
├── tests/                 # Integration tests
└── config/               # Configuration files
```

## Key Principle: TypeScript is Source of Truth

The game rules are defined in `src/shared/engine/`. The Python `ai-service` **mirrors** these rules for training. When rules change:

1. Update TypeScript first
2. Update Python to match
3. Run parity tests to verify they agree

## Quick Start Commands

```bash
# Frontend development
cd src/client && npm run dev

# Backend server
cd src/server && npm run dev

# AI service (Python)
cd ai-service
python -m app.training.train --help

# Run tests
npm test                           # TypeScript tests
cd ai-service && pytest           # Python tests
```

## Cluster Automation (Recommended)

For long-term cluster utilization, use `master_loop.py`:

```bash
cd ai-service

# Full automation (24/7 cluster operation)
python scripts/master_loop.py

# Watch mode (show status)
python scripts/master_loop.py --watch

# Dry run (preview actions)
python scripts/master_loop.py --dry-run
```

This orchestrates:

- **SelfplayScheduler**: Priority-based selfplay allocation (staleness, Elo velocity, curriculum weights)
- **DaemonManager**: 127 daemon types for sync, training, evaluation (107 active, 20 deprecated)
- **FeedbackLoopController**: Training feedback signals and curriculum adjustments
- **DataPipelineOrchestrator**: Export -> training -> evaluation -> promotion

## Board Configurations

| Type        | Sizes                 | Description             |
| ----------- | --------------------- | ----------------------- |
| `square8`   | 8x8 (64 cells)        | Standard square board   |
| `square19`  | 19x19 (361 cells)     | Large square (Go-sized) |
| `hex8`      | radius 4 (61 cells)   | Small hexagonal         |
| `hexagonal` | radius 12 (469 cells) | Large hexagonal         |

All board types support 2, 3, or 4 players.

## Common Workflows

### Train a New Model

```bash
cd ai-service

# 1. Export training data from game databases
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# 2. Train the model
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

### Transfer Learning (2p to 4p)

```bash
cd ai-service

# Resize value head for 4-player model
python scripts/transfer_2p_to_4p.py \
  --source models/my_2p_model.pth \
  --output models/my_4p_init.pth \
  --board-type square8

# Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/my_4p_init.pth \
  --data-path data/training/sq8_4p.npz
```

### Check Game Data Quality

```bash
cd ai-service
python -m app.training.data_quality --db data/games/selfplay.db
```

### Verify TS/Python Parity

```bash
cd ai-service
python scripts/check_ts_python_replay_parity.py --db data/games/my_games.db
```

## Cluster Infrastructure

RingRift uses a P2P mesh network for distributed training across ~12 active nodes.

| Provider     | Nodes | GPUs                        | Status  |
| ------------ | ----- | --------------------------- | ------- |
| Lambda GH200 | 7     | GH200 96GB x 7              | Active  |
| Nebius       | 3     | H100 80GB x 2, L40S         | Stopped |
| Hetzner      | 3     | CPU only (P2P voters)       | Active  |
| Local        | 2     | Mac Studio M3 (coordinator) | Active  |

```bash
# Check cluster status via P2P
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print("Leader:", d.get("leader_id")); print("Alive:", d.get("alive_peers"))'

# Or use the monitor
cd ai-service && python -m app.distributed.cluster_monitor

# Update all nodes to latest code
cd ai-service && python scripts/update_all_nodes.py --restart-p2p
```

See `ai-service/config/distributed_hosts.yaml` for full cluster configuration.

## Neural Network Architectures

| Version          | Parameters | Description                               |
| ---------------- | ---------- | ----------------------------------------- |
| `v2`             | ~2-4M      | Standard architecture (default for most)  |
| `v4`             | ~3-5M      | Improved residual blocks                  |
| `v5-heavy`       | ~8-12M     | Wider with heuristic features (49 inputs) |
| `v5-heavy-large` | ~25-35M    | Scaled v5-heavy for complex boards        |

## Key Features

- **GPU Selfplay**: Vectorized game simulation on CUDA (`app/ai/gpu_parallel_games.py`)
- **Gumbel MCTS**: Quality-focused tree search for training data
- **Transfer Learning**: Train 4-player models from 2-player checkpoints
- **Parity Testing**: Verify Python engine matches TypeScript rules
- **48-Hour Autonomous Operation**: Cluster runs unattended with automatic recovery
- **5 Feedback Loops**: Quality->Training, Elo->Selfplay, Regression->Curriculum, Loss->Exploration, Promotion->Curriculum

## 48-Hour Autonomous Operation

The cluster runs 48+ hours unattended with comprehensive resilience:

| Daemon              | Purpose                                      |
| ------------------- | -------------------------------------------- |
| `PROGRESS_WATCHDOG` | Detects Elo stalls, triggers recovery        |
| `P2P_RECOVERY`      | Restarts unhealthy P2P orchestrator          |
| `STALE_FALLBACK`    | Uses older models when sync fails            |
| `MEMORY_MONITOR`    | Prevents OOM via proactive GPU VRAM tracking |
| `LeaderProbeLoop`   | Fast leader failure detection (70s recovery) |

**Resilience Features:**

- Adaptive circuit breaker cascade prevention (9 CB types with 4-tier escalation)
- Graceful degradation with stale training data after sync failures
- Multi-transport failover (Tailscale -> SSH -> Base64 -> HTTP)
- Automatic parity gate bypass on cluster nodes without Node.js

## Known Issues

### Parity Gates on Cluster Nodes

Cluster nodes (Vast.ai, RunPod, Nebius) lack Node.js runtime, so TypeScript parity gates fail with "pending_gate" status in databases.

**Workaround**:

```bash
# Skip parity gates on cluster nodes (selfplay only, no TS validation)
export RINGRIFT_ALLOW_PENDING_GATE=1

# Run parity validation locally (has npx) before syncing to cluster
python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db
```

**Root cause**: Container images and cloud nodes don't include Node.js. The parity gate script (`scripts/selfplay-db-ts-replay.ts`) requires `npx ts-node`.

## See Also

- `ai-service/CLAUDE.md` - Detailed AI service context
- `ai-service/AGENTS.md` - Coding guidelines for AI service
- `ai-service/docs/architecture/` - Architecture documentation
- `ai-service/docs/QUICK_START_TRAINING.md` - Training quick-start guide
- `AGENTS.md` - Root-level coding guidelines
