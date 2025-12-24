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

Training runs on GPU nodes managed via SSH. See `ai-service/config/distributed_hosts.yaml` for the full list. Key nodes:

- `lambda-gh200-a` through `lambda-gh200-p`: GH200 GPUs (96GB each)
- `lambda-h100`, `lambda-2xh100`: H100 GPUs
- `mac-studio`: Coordinator node

```bash
# Check cluster status
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

## Current State (Dec 2024)

### Trained Models

- `hex8_2p`: v2 model, 76.2% policy accuracy
- `square8_4p`: Training in progress on cluster

### Recent Additions

- `app/utils/game_discovery.py`: Unified database discovery
- `app/distributed/cluster_monitor.py`: Cluster status dashboard
- `app/training/data_quality.py`: Data validation tools

### Active Training

Check with: `python -m app.distributed.cluster_monitor --watch`

## See Also

- `ai-service/CLAUDE.md` - Detailed AI service context
- `ai-service/AGENTS.md` - Coding guidelines for AI service
- `AGENTS.md` - Root-level coding guidelines
