# CLAUDE.md - AI Assistant Memory for RingRift

This file provides persistent context for AI assistants working on this codebase.

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

## Cluster Infrastructure

Training runs on a P2P mesh network of GPU nodes. See `ai-service/config/distributed_hosts.yaml` for configuration.

### Active Nodes (Dec 2025)

| Type         | Nodes                               | GPU Memory   |
| ------------ | ----------------------------------- | ------------ |
| Lambda GH200 | b-new, d, g, h, i, o, p, q, r, s, t | 96GB each    |
| Lambda H100  | lambda-h100, lambda-2xh100          | 80GB / 160GB |
| Vast.ai      | vast-5090, vast-4x5090              | Various      |
| Hetzner      | cpu1, cpu2, cpu3                    | CPU only     |

Note: GH200 nodes a, e, f are retired. Node names in config may differ from actual hostnames.

```bash
# Check cluster status via P2P
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Leader: {d.get(\"leader_id\")}"); print(f"Alive: {d.get(\"alive_peers\")}")'

# Or use the monitor
cd ai-service && python -m app.distributed.cluster_monitor
```

## Common Workflows

### Train a New Model

```bash
cd ai-service

# 1. Export training data
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# 2. Train
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

### Transfer Learning (2p → 4p)

```bash
cd ai-service

# Resize value head for 4-player model
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/transfer_sq8_4p_init.pth \
  --board-type square8

# Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/transfer_sq8_4p_init.pth \
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
python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db
```

## Board Configurations

| Type        | Sizes                 | Description             |
| ----------- | --------------------- | ----------------------- |
| `square8`   | 8×8 (64 cells)        | Standard square board   |
| `square19`  | 19×19 (361 cells)     | Large square (Go-sized) |
| `hex8`      | radius 4 (61 cells)   | Small hexagonal         |
| `hexagonal` | radius 12 (469 cells) | Large hexagonal         |

All support 2, 3, or 4 players.

## Current State (Dec 2025)

### Recent Additions

- **Transfer Learning**: `--init-weights` flag for 2p→4p model initialization
- **Value Head Resizing**: `scripts/transfer_2p_to_4p.py` for 2→4 player transfer

### Trained Models

| Model                   | Size  | Status                             |
| ----------------------- | ----- | ---------------------------------- |
| `canonical_hex8_2p.pth` | 125MB | Production (76.2% policy accuracy) |
| `canonical_sq8_2p.pth`  | 96MB  | Production                         |
| `canonical_sq8_3p.pth`  | 96MB  | Production                         |
| `canonical_sq19_2p.pth` | 107MB | Production                         |

### GPU Selfplay

The GPU selfplay pipeline (`ai-service/app/ai/gpu_parallel_games.py`) is production-ready:

- **Parity**: 100% verified against TypeScript (10K seeds tested)
- **Performance**: ~6.5x speedup on CUDA (partial vectorization)
- **Status**: ~80 `.item()` calls remain; full vectorization would yield 10-20x speedup

### Cluster Status

~21 active nodes running 400+ selfplay jobs:

- Lambda GH200 (b-new, d, g, h, i, o, p, q, r, s): Primary training
- Lambda H100/2xH100: Secondary training
- Vast.ai: vast-5090, vast-4x5090
- Hetzner CPU: cpu1, cpu2, cpu3
- P2P orchestrator on port 8770 with leader election

### Active Training

Check with: `python -m app.distributed.cluster_monitor --watch`

## See Also

- `ai-service/CLAUDE.md` - Detailed AI service context
- `ai-service/AGENTS.md` - Coding guidelines for AI service
- `AGENTS.md` - Root-level coding guidelines
