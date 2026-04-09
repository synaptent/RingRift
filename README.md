# RingRift

RingRift is a deterministic abstract strategy game and a research codebase for training neural agents on a novel multiplayer ruleset. The repository contains the playable web app, the canonical TypeScript rules engine, a Python AI service that mirrors those rules, parity tooling, and a minimal AlphaZero-style training loop used to produce the current results.

## Research Status

Status below is current as of April 9, 2026.

| Config       | Best Reported Elo | Promotions | Evidence                                                                         |
| ------------ | ----------------: | ---------: | -------------------------------------------------------------------------------- |
| `hex8_2p`    |          `1967.6` |        `6` | Strongest result so far; clear iterative improvement from the 1500 baseline      |
| `square8_2p` |          `1601.8` |        `2` | Clean fixed-LR run now promotes under the corrected experiment harness           |
| `square8_3p` |          `1534.9` |        `1` | Promising multiplayer result; useful evidence, but weaker than the 2-player runs |

These results came from long-running GH200 cluster experiments. Reproducing the same behavior locally is possible, but it requires GPU time and multiple training iterations.

For the concrete evidence and caveats, see [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md).

If you want the shortest external-facing orientation before diving into code, read [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md).
If you want the shortest shareable research summary, read [docs/RESEARCH_SNAPSHOT.md](/Users/armand/Development/RingRift/docs/RESEARCH_SNAPSHOT.md).

## Quick Start

### 1. Run the web app

```bash
git clone https://github.com/synaptent/RingRift.git
cd RingRift
npm install
cp .env.example .env
docker compose up -d postgres redis
npm run db:migrate
npm run db:generate
npm run dev
```

- Client: `http://localhost:5173`
- Server: `http://localhost:3000`

More setup detail is in [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md).

### 2. Run the AI service

```bash
cd ai-service
./setup.sh
./run.sh
```

- Health: `http://localhost:8001/health`
- Docs: `http://localhost:8001/docs`

### 3. Launch a proven training configuration

From the repo root:

```bash
./scripts/run_proven_experiment.sh square8_2p --print-only
./scripts/run_proven_experiment.sh square8_2p --iterations 10
```

Supported experiment presets:

- `hex8_2p`
- `square8_2p`

The script launches the same minimal loop configurations used for the published results and writes artifacts under `ai-service/data/proven_experiments/<config>/`.

### 4. Inspect results

```bash
cat ai-service/data/proven_experiments/square8_2p/summary.json
tail -n 5 ai-service/data/proven_experiments/square8_2p/metrics.jsonl
```

## Architecture Overview

The supported architecture is intentionally narrow:

1. The canonical game rules live in TypeScript under [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine).
2. The web app and backend both use that TypeScript engine.
3. The Python AI service mirrors those rules for inference, replay validation, and training.
4. The supported training engine is the minimal loop at [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py).
5. TS↔Python replay parity is checked with [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py).

The broader coordinator, daemon, and P2P infrastructure remains in the repository for cluster operations and historical experiments, but it is not required to understand or reproduce the core results.

See [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md) for the external-facing architecture guide.

## Repository Guide

Start here if you are new to the repo:

- [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md) for the shortest external-facing summary
- [docs/RESEARCH_SNAPSHOT.md](/Users/armand/Development/RingRift/docs/RESEARCH_SNAPSHOT.md) for the shortest shareable research summary
- [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md) for local setup
- [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md) for the current evidence
- [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md) for the system model
- [docs/REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md) for what is active versus legacy

Core code directories:

- [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine): canonical game rules
- [`src/server`](/Users/armand/Development/RingRift/src/server): Node/Express backend
- [`src/client`](/Users/armand/Development/RingRift/src/client): React frontend
- [`ai-service/app`](/Users/armand/Development/RingRift/ai-service/app): Python AI service
- [`ai-service/scripts`](/Users/armand/Development/RingRift/ai-service/scripts): training, parity, and ops scripts

## Validation

Useful trust-building checks for the supported path:

```bash
bash scripts/check_supported_path.sh
```

To refresh the checked-in results snapshot and SVG artifacts from local metrics:

```bash
npm run results:refresh
```

## Supported vs Legacy

RingRift is a large, historically layered repository. Not every subsystem is equally current.

- Supported for external readers:
  - web app
  - canonical TS rules engine
  - Python parity tooling
  - minimal training loop
  - proven experiment script and results docs
- Useful but secondary:
  - production deployment docs
  - cluster monitoring and operational scripts
  - AI ladder and calibration tooling
- Historical or operationally specialized:
  - `archive/`
  - `docs/archive/`
  - deprecated training and orchestration paths
  - many cluster automation scripts under `ai-service/scripts`

If your goal is to understand the research result, follow the supported path first and treat the rest as secondary context.
