# RingRift Comprehensive AI Training Pipeline Plan

## Overview

This plan defines a unified AI training pipeline that orchestrates selfplay, CMA-ES heuristic optimization, NNUE training, and neural network training across all board types (square8, square19, hexagonal) and player counts (2p, 3p, 4p). The pipeline achieves AlphaZero-style self-improvement through continuous feedback loops between all training components.

## Compute Resources

### Available Infrastructure

| Resource                         | Role                           | Capabilities         |
| -------------------------------- | ------------------------------ | -------------------- |
| **AWS Staging** (54.198.219.106) | Primary selfplay, CMA-ES       | 8 vCPU, 16GB RAM     |
| **AWS Extra** (selfplay-extra)   | Parallel selfplay, tournaments | 4 vCPU, 8GB RAM      |
| **Mac Studio** (mac-studio)      | NN training (MPS), NNUE        | M2 Ultra, 192GB, MPS |
| **M1 Pro** (m1-pro)              | Local selfplay, light training | M1 Pro, 64GB         |
| **Laptop**                       | Coordination, data aggregation | M3 Max, 64GB         |

### Cloud GPU (Future)

- AWS p3.2xlarge or Lambda Labs for large-scale NN training
- Used when Mac Studio MPS is insufficient for batch sizes > 512

## Pipeline Architecture

```
                    +-------------------+
                    |   ORCHESTRATOR    |
                    | (pipeline_master) |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
    +----v----+        +-----v-----+       +-----v-----+
    | SELFPLAY |        |  CMA-ES   |       |  TRAINING |
    | CLUSTER  |        | OPTIMIZER |       |   LOOP    |
    +----+----+        +-----+-----+       +-----+-----+
         |                   |                   |
         |    +--------------+                   |
         |    |                                  |
    +----v----v----+                       +-----v-----+
    | GameReplayDB |<----------------------|   SYNC    |
    | (per-machine)|                       |  SERVICE  |
    +------+-------+                       +-----------+
           |
    +------v-------+
    | MERGED DATA  |
    | (laptop/mac) |
    +------+-------+
           |
    +------v-------+
    |  NPZ EXPORT  |
    | (training)   |
    +--------------+
```

## Phase 1: Bootstrap Selfplay (No NN Required)

### 1.1 Parallel Selfplay Deployment

Deploy descent-only and heuristic-only selfplay across all machines:

**AWS Staging** (primary volume):

```bash
# Square8 - fastest, highest volume
python scripts/run_self_play_soak.py --num-games 500 --board-type square8 \
  --engine-mode descent-only --num-players 2 --max-moves 500 --seed 1

# Square19 - medium volume
python scripts/run_self_play_soak.py --num-games 200 --board-type square19 \
  --engine-mode descent-only --num-players 2 --max-moves 800 --seed 2

# Hexagonal - medium volume
python scripts/run_self_play_soak.py --num-games 200 --board-type hexagonal \
  --engine-mode descent-only --num-players 2 --max-moves 800 --seed 3
```

**AWS Extra** (3p/4p focus):

```bash
# 3-player configurations
python scripts/run_self_play_soak.py --num-games 150 --board-type square8 \
  --engine-mode descent-only --num-players 3 --max-moves 600 --seed 10

# 4-player configurations
python scripts/run_self_play_soak.py --num-games 100 --board-type square8 \
  --engine-mode descent-only --num-players 4 --max-moves 700 --seed 11
```

**M1 Pro** (supplemental):

```bash
# Mixed heuristic engines for diversity
python scripts/run_self_play_soak.py --num-games 100 --board-type square8 \
  --engine-mode heuristic-only --num-players 2 --max-moves 500 --seed 20
```

### 1.2 Target Game Counts for Phase 1

| Board     | 2p   | 3p  | 4p  | Total    |
| --------- | ---- | --- | --- | -------- |
| square8   | 1000 | 300 | 200 | 1500     |
| square19  | 400  | 150 | 100 | 650      |
| hexagonal | 400  | 150 | 100 | 650      |
| **Total** | 1800 | 600 | 400 | **2800** |

## Phase 2: Data Aggregation and First Training

### 2.1 Automated Data Sync

Create `scripts/sync_selfplay_data.sh`:

```bash
#!/bin/bash
# Sync all selfplay DBs to laptop and Mac Studio

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_DIR="data/games/synced_${TIMESTAMP}"
mkdir -p "$LOCAL_DIR"

# AWS Staging
rsync -avz ringrift-staging:~/ringrift/ai-service/data/games/*.db "$LOCAL_DIR/"

# AWS Extra
rsync -avz ringrift-selfplay-extra:~/ringrift/ai-service/data/games/*.db "$LOCAL_DIR/"

# M1 Pro
rsync -avz m1-pro:~/Development/RingRift/ai-service/data/games/*.db "$LOCAL_DIR/"

# Merge all DBs
python scripts/merge_game_dbs.py \
  --output data/games/merged_${TIMESTAMP}.db \
  $(find "$LOCAL_DIR" -name "*.db" -printf "--db %p ")

# Sync merged DB to Mac Studio for training
rsync -avz "data/games/merged_${TIMESTAMP}.db" mac-studio:~/Development/RingRift/ai-service/data/games/
```

### 2.2 NPZ Export for Training

```bash
# Export per-configuration training data
for board in square8 square19 hexagonal; do
  for players in 2 3 4; do
    python scripts/export_replay_dataset.py \
      --db data/games/merged.db \
      --board-type $board \
      --num-players $players \
      --output data/training/${board}_${players}p.npz \
      --require-completed \
      --min-moves 20
  done
done
```

### 2.3 Initial CMA-ES Optimization

Run CMA-ES for all configurations using distributed workers:

```bash
# On AWS Staging - orchestrate across all workers
python scripts/run_all_iterative_cmaes.sh hybrid
```

This runs iterative CMA-ES for all 9 configurations:

- square8: 2p, 3p, 4p
- square19: 2p, 3p, 4p
- hex: 2p, 3p, 4p

CMA-ES optimizes heuristic weights using selfplay evaluation.

### 2.4 Initial Neural Network Training (Mac Studio)

```bash
# On Mac Studio with MPS acceleration
cd ~/Development/RingRift/ai-service
source venv/bin/activate

# Train square8 2p model first (most data)
PYTHONPATH=. python app/training/train.py \
  --data-path data/training/square8_2p.npz \
  --board-type square8 \
  --num-players 2 \
  --epochs 100 \
  --batch-size 256 \
  --device mps \
  --save-path models/square8_2p_v1.pth

# Then other configurations as data becomes available
```

### 2.5 Initial NNUE Training

```bash
# Train NNUE from game databases
python scripts/train_nnue.py \
  --db data/games/merged.db \
  --board-type square8 \
  --num-players 2 \
  --epochs 50 \
  --batch-size 1024 \
  --output models/nnue/square8_2p_v1.nnue
```

## Phase 3: AlphaZero-Style Self-Improvement Loop

### 3.1 Feedback Architecture

```
+-------------+     +-------------+     +-------------+
|   CMA-ES    |---->|    NNUE     |---->|     NN      |
| (heuristic) |     | (minimax)   |     | (MCTS/eval) |
+------+------+     +------+------+     +------+------+
       ^                   ^                   ^
       |                   |                   |
       +-------------------+-------------------+
                           |
                    +------v------+
                    |  SELFPLAY   |
                    | (improved   |
                    |   engines)  |
                    +-------------+
```

### 3.2 Continuous Improvement Script

Create `scripts/run_improvement_loop.py`:

```python
#!/usr/bin/env python
"""AlphaZero-style continuous improvement loop."""

import argparse
import subprocess
import time
from pathlib import Path

def run_improvement_iteration(iteration: int, config: dict):
    """Single iteration of the improvement loop."""

    # 1. Run selfplay with current best models
    selfplay_cmd = [
        "python", "scripts/run_self_play_soak.py",
        "--num-games", str(config["games_per_iter"]),
        "--board-type", config["board"],
        "--engine-mode", "mixed",  # Use all available engines
        "--num-players", str(config["players"]),
        "--max-moves", str(config["max_moves"]),
        "--seed", str(iteration * 1000),
        "--log-jsonl", f"logs/improvement/iter_{iteration}.jsonl",
    ]
    subprocess.run(selfplay_cmd, check=True)

    # 2. Export new training data
    export_cmd = [
        "python", "scripts/export_replay_dataset.py",
        "--db", "data/games/selfplay.db",
        "--output", f"data/training/iter_{iteration}.npz",
        "--require-completed",
    ]
    subprocess.run(export_cmd, check=True)

    # 3. Fine-tune NN on new data (incremental)
    train_cmd = [
        "python", "app/training/train.py",
        "--data-path", f"data/training/iter_{iteration}.npz",
        "--board-type", config["board"],
        "--num-players", str(config["players"]),
        "--epochs", "10",  # Short fine-tuning
        "--resume-from", f"models/{config['board']}_{config['players']}p_best.pth",
        "--save-path", f"models/{config['board']}_{config['players']}p_iter{iteration}.pth",
    ]
    subprocess.run(train_cmd, check=True)

    # 4. Run evaluation tournament
    eval_cmd = [
        "python", "scripts/run_ai_tournament.py",
        "--p1", "Neural",
        "--p1-model", f"models/{config['board']}_{config['players']}p_iter{iteration}.pth",
        "--p2", "Neural",
        "--p2-model", f"models/{config['board']}_{config['players']}p_best.pth",
        "--board", config["board"],
        "--games", "20",
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)

    # 5. Promote if improved
    if "P1 wins" in result.stdout and parse_winrate(result.stdout) > 0.55:
        shutil.copy(
            f"models/{config['board']}_{config['players']}p_iter{iteration}.pth",
            f"models/{config['board']}_{config['players']}p_best.pth"
        )
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="square8")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=100)
    args = parser.parse_args()

    config = {
        "board": args.board,
        "players": args.players,
        "games_per_iter": args.games_per_iter,
        "max_moves": 500 if args.board == "square8" else 1000,
    }

    for i in range(args.iterations):
        print(f"=== Improvement Iteration {i+1}/{args.iterations} ===")
        improved = run_improvement_iteration(i, config)
        if improved:
            print(f"Model improved at iteration {i+1}!")
        time.sleep(10)  # Brief pause between iterations

if __name__ == "__main__":
    main()
```

### 3.3 CMA-ES Integration with NN

When NN models are available, CMA-ES can use them for evaluation:

```bash
# CMA-ES with neural evaluator
python scripts/run_iterative_cmaes.py \
  --board square8 \
  --num-players 2 \
  --evaluator neural \
  --model-path models/square8_2p_best.pth \
  --generations-per-iter 10 \
  --max-iterations 5
```

### 3.4 NNUE Updates from NN Outputs

```bash
# Distill NN knowledge into NNUE
python scripts/distill_to_nnue.py \
  --nn-model models/square8_2p_best.pth \
  --db data/games/selfplay.db \
  --output models/nnue/square8_2p_distilled.nnue
```

## Phase 4: Pipeline Orchestration

### 4.1 Master Orchestrator Script

Create `scripts/pipeline_orchestrator.py`:

```python
#!/usr/bin/env python
"""Master pipeline orchestrator for distributed AI training."""

import argparse
import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class WorkerConfig:
    name: str
    host: str
    role: str  # "selfplay", "training", "cmaes"
    capabilities: List[str]

WORKERS = [
    WorkerConfig("staging", "ringrift-staging", "selfplay", ["square8", "square19", "hex"]),
    WorkerConfig("extra", "ringrift-selfplay-extra", "selfplay", ["square8"]),
    WorkerConfig("m1pro", "m1-pro", "selfplay", ["square8", "hex"]),
    WorkerConfig("mac-studio", "mac-studio", "training", ["nn", "nnue"]),
]

class PipelineOrchestrator:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.state = {"phase": "bootstrap", "iteration": 0}

    async def run_selfplay_phase(self):
        """Distribute selfplay across workers."""
        tasks = []
        for worker in WORKERS:
            if worker.role == "selfplay":
                task = self.dispatch_selfplay(worker)
                tasks.append(task)
        await asyncio.gather(*tasks)

    async def run_sync_phase(self):
        """Sync all data to central location."""
        subprocess.run(["scripts/sync_selfplay_data.sh"], check=True)

    async def run_training_phase(self):
        """Run NN, NNUE, and CMA-ES training."""
        # Parallel training tasks
        await asyncio.gather(
            self.run_nn_training(),
            self.run_nnue_training(),
            self.run_cmaes_optimization(),
        )

    async def run_evaluation_phase(self):
        """Evaluate improved models via tournaments."""
        subprocess.run([
            "python", "scripts/run_ai_tournament.py",
            "--round-robin", "--all-engines"
        ], check=True)

    async def run_full_loop(self, iterations: int):
        """Execute complete pipeline loop."""
        for i in range(iterations):
            print(f"=== Pipeline Iteration {i+1}/{iterations} ===")

            await self.run_selfplay_phase()
            await self.run_sync_phase()
            await self.run_training_phase()
            await self.run_evaluation_phase()

            self.state["iteration"] = i + 1
            self.save_state()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(args.config)
    asyncio.run(orchestrator.run_full_loop(args.iterations))

if __name__ == "__main__":
    main()
```

### 4.2 Pipeline Configuration

Create `config/pipeline.json`:

```json
{
  "selfplay": {
    "games_per_iteration": {
      "square8_2p": 200,
      "square8_3p": 100,
      "square8_4p": 80,
      "square19_2p": 100,
      "square19_3p": 60,
      "square19_4p": 40,
      "hex_2p": 100,
      "hex_3p": 60,
      "hex_4p": 40
    },
    "engine_modes": ["descent-only", "heuristic-only", "mixed"],
    "max_moves": {
      "square8": 500,
      "square19": 1000,
      "hex": 800
    }
  },
  "training": {
    "nn": {
      "epochs_per_iteration": 10,
      "batch_size": 256,
      "device": "mps"
    },
    "nnue": {
      "epochs_per_iteration": 20,
      "batch_size": 1024
    },
    "cmaes": {
      "generations_per_iteration": 10,
      "population_size": 20,
      "games_per_eval": 12
    }
  },
  "evaluation": {
    "tournament_games": 20,
    "promotion_threshold": 0.55
  },
  "sync": {
    "interval_minutes": 60,
    "destinations": ["laptop", "mac-studio"]
  }
}
```

## Phase 5: Cloud GPU Training (Scale-Up)

### 5.1 AWS GPU Instance Setup

For large-scale NN training when Mac Studio is insufficient:

```bash
# Launch p3.2xlarge (V100 GPU)
aws ec2 run-instances \
  --instance-type p3.2xlarge \
  --image-id ami-0xxx \
  --key-name ringrift-gpu \
  --security-groups ringrift-gpu-sg

# Install dependencies
ssh ringrift-gpu 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118'
```

### 5.2 Distributed Training with DDP

```bash
# Multi-GPU training (future scaling)
torchrun --nproc_per_node=4 app/training/train.py \
  --data-path data/training/combined.npz \
  --board-type square8 \
  --distributed \
  --epochs 200 \
  --batch-size 1024
```

## Monitoring and Observability

### Dashboard Metrics

1. **Selfplay Progress**: Games completed per config, completion rate
2. **Training Metrics**: Loss curves, learning rate, validation accuracy
3. **Model Quality**: Tournament win rates, Elo ratings
4. **Resource Utilization**: CPU/GPU usage per worker

### Alerting

- Selfplay stalls (no new games in 30 minutes)
- Training loss divergence
- Model regression (tournament winrate < 45%)
- Worker offline

## Implementation Tasks

1. [ ] Create `scripts/pipeline_orchestrator.py`
2. [ ] Create `scripts/sync_selfplay_data.sh`
3. [ ] Create `scripts/run_improvement_loop.py`
4. [ ] Create `config/pipeline.json`
5. [ ] Add NNUE distillation script
6. [ ] Add tournament round-robin mode
7. [ ] Set up monitoring dashboard
8. [ ] Configure cloud GPU provisioning

## Success Criteria

- 10,000+ completed selfplay games across all configurations
- CMA-ES converged weights for all 9 configs
- NN models for square8 2p/3p achieving >60% vs heuristic baseline
- NNUE providing 10%+ speedup at Minimax difficulty 4+
- Continuous improvement loop running autonomously
