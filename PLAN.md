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

### Resource Scheduling

To prevent memory pressure when running concurrent workloads:

| Worker      | Primary Task      | Secondary Task  | Memory Limit |
| ----------- | ----------------- | --------------- | ------------ |
| AWS Staging | Selfplay (6 vCPU) | CMA-ES (2 vCPU) | 12GB / 4GB   |
| AWS Extra   | Selfplay only     | -               | 6GB          |
| Mac Studio  | NN Training       | NNUE Training   | 160GB / 32GB |
| M1 Pro      | Selfplay          | Light eval      | 48GB / 16GB  |

CMA-ES on staging should be scheduled during selfplay cooldown periods or run with `--max-workers 2` to limit concurrency.

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
# Supports incremental sync to avoid re-processing already-merged games

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_DIR="data/games/synced_${TIMESTAMP}"
CURSOR_FILE="data/games/.sync_cursor.json"
mkdir -p "$LOCAL_DIR"

# AWS Staging
rsync -avz ringrift-staging:~/ringrift/ai-service/data/games/*.db "$LOCAL_DIR/"

# AWS Extra
rsync -avz ringrift-selfplay-extra:~/ringrift/ai-service/data/games/*.db "$LOCAL_DIR/"

# M1 Pro
rsync -avz m1-pro:~/Development/RingRift/ai-service/data/games/*.db "$LOCAL_DIR/"

# Merge all DBs with deduplication
python scripts/merge_game_dbs.py \
  --output data/games/merged_${TIMESTAMP}.db \
  --cursor-file "$CURSOR_FILE" \
  --dedupe-by-game-id \
  $(find "$LOCAL_DIR" -name "*.db" -type f | sed 's/^/--db /' | tr '\n' ' ')

# Update cursor for next incremental sync
python -c "
import json
from pathlib import Path
cursor = json.loads(Path('$CURSOR_FILE').read_text()) if Path('$CURSOR_FILE').exists() else {}
cursor['last_merge'] = '$TIMESTAMP'
cursor['merged_db'] = 'data/games/merged_${TIMESTAMP}.db'
Path('$CURSOR_FILE').write_text(json.dumps(cursor, indent=2))
"

# Sync merged DB to Mac Studio for training
rsync -avz "data/games/merged_${TIMESTAMP}.db" mac-studio:~/Development/RingRift/ai-service/data/games/

echo "Sync complete. Merged DB: data/games/merged_${TIMESTAMP}.db"
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
"""AlphaZero-style continuous improvement loop with checkpointing and rollback."""

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LoopState:
    """Checkpoint state for resumable improvement loop."""
    iteration: int = 0
    best_model_path: Optional[str] = None
    best_winrate: float = 0.0
    consecutive_failures: int = 0
    total_improvements: int = 0


def load_state(state_path: Path) -> LoopState:
    """Load checkpoint state or return fresh state."""
    if state_path.exists():
        data = json.loads(state_path.read_text())
        return LoopState(**data)
    return LoopState()


def save_state(state: LoopState, state_path: Path) -> None:
    """Persist checkpoint state."""
    state_path.write_text(json.dumps(asdict(state), indent=2))


def parse_winrate(output: str) -> float:
    """Extract win rate from tournament output."""
    # Match patterns like "P1 wins: 12/20 (60.0%)" or "Win rate: 0.60"
    match = re.search(r"(\d+\.?\d*)%", output)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r"Win rate:\s*(\d+\.?\d*)", output)
    if match:
        return float(match.group(1))
    # Fallback: count wins
    p1_wins = output.count("P1 wins") + output.count("Winner: P1")
    total = p1_wins + output.count("P2 wins") + output.count("Winner: P2")
    return p1_wins / max(total, 1)


def validate_model(model_path: Path) -> bool:
    """Quick sanity check that model file is valid."""
    if not model_path.exists():
        return False
    # Check file size is reasonable (> 1KB, < 1GB)
    size = model_path.stat().st_size
    return 1024 < size < 1_000_000_000


def run_improvement_iteration(
    iteration: int, config: dict, state: LoopState
) -> tuple[bool, float]:
    """Single iteration of the improvement loop. Returns (improved, winrate)."""
    board = config["board"]
    players = config["players"]
    model_prefix = f"models/{board}_{players}p"

    # 1. Run selfplay with current best models
    selfplay_cmd = [
        "python", "scripts/run_self_play_soak.py",
        "--num-games", str(config["games_per_iter"]),
        "--board-type", board,
        "--engine-mode", "mixed",  # Use all available engines
        "--num-players", str(players),
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
    iter_model = Path(f"{model_prefix}_iter{iteration}.pth")
    best_model = Path(f"{model_prefix}_best.pth")

    train_cmd = [
        "python", "app/training/train.py",
        "--data-path", f"data/training/iter_{iteration}.npz",
        "--board-type", board,
        "--num-players", str(players),
        "--epochs", "10",  # Short fine-tuning
        "--save-path", str(iter_model),
    ]
    if best_model.exists():
        train_cmd.extend(["--resume-from", str(best_model)])

    subprocess.run(train_cmd, check=True)

    # Validate model was created successfully
    if not validate_model(iter_model):
        print(f"WARNING: Model validation failed for {iter_model}")
        return False, 0.0

    # 4. Run evaluation tournament
    eval_cmd = [
        "python", "scripts/run_ai_tournament.py",
        "--p1", "Neural",
        "--p1-model", str(iter_model),
        "--p2", "Neural" if best_model.exists() else "Heuristic",
        "--board", board,
        "--games", "20",
    ]
    if best_model.exists():
        eval_cmd.extend(["--p2-model", str(best_model)])

    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    winrate = parse_winrate(result.stdout)

    # 5. Promote if improved
    threshold = config.get("promotion_threshold", 0.55)
    if winrate > threshold:
        # Backup previous best before overwriting
        if best_model.exists():
            backup_path = Path(f"{model_prefix}_prev_best.pth")
            shutil.copy(best_model, backup_path)

        shutil.copy(iter_model, best_model)
        print(f"Model promoted! Win rate: {winrate:.1%} > {threshold:.1%}")
        return True, winrate

    return False, winrate


def rollback_model(config: dict) -> bool:
    """Rollback to previous best model if available."""
    board = config["board"]
    players = config["players"]
    model_prefix = f"models/{board}_{players}p"

    best_model = Path(f"{model_prefix}_best.pth")
    backup_model = Path(f"{model_prefix}_prev_best.pth")

    if backup_model.exists():
        shutil.copy(backup_model, best_model)
        print(f"Rolled back to previous best model: {backup_model}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style continuous improvement loop"
    )
    parser.add_argument("--board", default="square8")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Rollback after this many failures")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--state-file", type=Path,
                        default=Path("logs/improvement/state.json"))
    args = parser.parse_args()

    config = {
        "board": args.board,
        "players": args.players,
        "games_per_iter": args.games_per_iter,
        "max_moves": 500 if args.board == "square8" else 1000,
        "promotion_threshold": args.promotion_threshold,
    }

    # Load or initialize state
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(args.state_file) if args.resume else LoopState()
    start_iter = state.iteration if args.resume else 0

    if args.dry_run:
        print(f"DRY RUN - would execute {args.iterations} iterations")
        print(f"Config: {json.dumps(config, indent=2)}")
        return

    for i in range(start_iter, args.iterations):
        print(f"\n{'='*60}")
        print(f"=== Improvement Iteration {i+1}/{args.iterations} ===")
        print(f"{'='*60}")

        try:
            improved, winrate = run_improvement_iteration(i, config, state)

            state.iteration = i + 1
            if improved:
                state.total_improvements += 1
                state.best_winrate = winrate
                state.consecutive_failures = 0
                print(f"Model improved at iteration {i+1}! "
                      f"(Total improvements: {state.total_improvements})")
            else:
                state.consecutive_failures += 1
                print(f"No improvement (winrate: {winrate:.1%}, "
                      f"consecutive failures: {state.consecutive_failures})")

                # Rollback if too many consecutive failures
                if state.consecutive_failures >= args.max_consecutive_failures:
                    print(f"WARNING: {state.consecutive_failures} consecutive "
                          f"failures, attempting rollback...")
                    if rollback_model(config):
                        state.consecutive_failures = 0

            save_state(state, args.state_file)

        except subprocess.CalledProcessError as e:
            print(f"ERROR in iteration {i+1}: {e}")
            state.consecutive_failures += 1
            save_state(state, args.state_file)

        time.sleep(10)  # Brief pause between iterations

    print(f"\n{'='*60}")
    print(f"Improvement loop complete!")
    print(f"Total iterations: {state.iteration}")
    print(f"Total improvements: {state.total_improvements}")
    print(f"Best win rate achieved: {state.best_winrate:.1%}")


if __name__ == "__main__":
    main()
```

### 3.3 CMA-ES Integration with NN

When NN models are available, CMA-ES can use them for evaluation. However, this should only be enabled once the NN model reaches a quality threshold to avoid optimizing toward a weak target.

**Quality Gate for NN-guided CMA-ES:**

```bash
# First, verify NN quality against heuristic baseline
python scripts/run_ai_tournament.py \
  --p1 Neural --p1-model models/square8_2p_best.pth \
  --p2 Heuristic \
  --board square8 \
  --games 50

# Only proceed if NN achieves > 55% win rate against heuristic
# Then run CMA-ES with neural evaluator
python scripts/run_iterative_cmaes.py \
  --board square8 \
  --num-players 2 \
  --evaluator neural \
  --model-path models/square8_2p_best.pth \
  --nn-quality-gate 0.55 \
  --generations-per-iter 10 \
  --max-iterations 5
```

### 3.4 NNUE Distillation from NN

Create `scripts/distill_to_nnue.py`:

```python
#!/usr/bin/env python
"""Distill neural network knowledge into NNUE for faster minimax evaluation.

This script generates position evaluations from the NN and trains NNUE to
approximate them, providing 10-100x faster evaluation at minimax depths 4+.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from app.ai.nnue import RingRiftNNUE
from app.ai.neural_net import RingRiftNet
from app.db.game_replay import GameReplayDB
from app.models import BoardType


def distill_to_nnue(
    nn_model_path: Path,
    db_path: Path,
    output_path: Path,
    board_type: BoardType,
    num_positions: int = 100_000,
    batch_size: int = 1024,
    epochs: int = 50,
):
    """Generate NNUE training data from NN evaluations."""

    # Load NN model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nn_model = RingRiftNet.load(nn_model_path, device=device)
    nn_model.eval()

    # Load positions from game database
    db = GameReplayDB(db_path)
    positions = []
    nn_evals = []

    print(f"Generating {num_positions} position evaluations from NN...")

    with torch.no_grad():
        for game_id in tqdm(db.list_games()):
            game_data = db.load_game(game_id)
            if game_data.board_type != board_type:
                continue

            for move_idx in range(game_data.total_moves):
                state = db.get_state_at_move(game_id, move_idx)
                if state is None:
                    continue

                # Convert state to NN input format
                features = nn_model.state_to_features(state)
                features_tensor = torch.tensor(features, device=device).unsqueeze(0)

                # Get NN evaluation
                policy, value = nn_model(features_tensor)
                nn_eval = value.item()

                positions.append(features)
                nn_evals.append(nn_eval)

                if len(positions) >= num_positions:
                    break

            if len(positions) >= num_positions:
                break

    print(f"Collected {len(positions)} positions")

    # Train NNUE to approximate NN evaluations
    nnue = RingRiftNNUE(board_type=board_type)

    positions_np = np.array(positions, dtype=np.float32)
    evals_np = np.array(nn_evals, dtype=np.float32)

    print(f"Training NNUE for {epochs} epochs...")
    nnue.train_from_evaluations(
        positions_np,
        evals_np,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
    )

    # Save NNUE model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nnue.save(output_path)
    print(f"NNUE model saved to {output_path}")

    # Validate distillation quality
    test_positions = positions_np[:1000]
    test_evals = evals_np[:1000]
    nnue_evals = np.array([nnue.evaluate(p) for p in test_positions])
    mse = np.mean((nnue_evals - test_evals) ** 2)
    correlation = np.corrcoef(nnue_evals, test_evals)[0, 1]

    print(f"Distillation quality: MSE={mse:.4f}, correlation={correlation:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Distill NN to NNUE")
    parser.add_argument("--nn-model", type=Path, required=True,
                        help="Path to trained NN model")
    parser.add_argument("--db", type=Path, required=True,
                        help="Game database for position sampling")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for NNUE model")
    parser.add_argument("--board-type", type=str, default="square8",
                        choices=["square8", "square19", "hexagonal"])
    parser.add_argument("--num-positions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    distill_to_nnue(
        nn_model_path=args.nn_model,
        db_path=args.db,
        output_path=args.output,
        board_type=BoardType(args.board_type),
        num_positions=args.num_positions,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
```

## Phase 4: Pipeline Orchestration

### 4.1 Master Orchestrator Script

Create `scripts/pipeline_orchestrator.py`:

```python
#!/usr/bin/env python
"""Master pipeline orchestrator for distributed AI training.

Features:
- Distributed selfplay across multiple workers
- Automatic data synchronization and deduplication
- Parallel NN, NNUE, and CMA-ES training
- Model evaluation and promotion
- Checkpointing and resume capability
- Dry-run mode for testing
"""

import argparse
import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class WorkerConfig:
    name: str
    host: str
    role: str  # "selfplay", "training", "cmaes"
    capabilities: List[str]
    max_memory_gb: int = 8


@dataclass
class PipelineState:
    """Persistent state for pipeline resumption."""
    phase: str = "bootstrap"
    iteration: int = 0
    last_sync: Optional[str] = None
    models_promoted: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


WORKERS = [
    WorkerConfig("staging", "ringrift-staging", "selfplay",
                 ["square8", "square19", "hex"], max_memory_gb=12),
    WorkerConfig("extra", "ringrift-selfplay-extra", "selfplay",
                 ["square8"], max_memory_gb=6),
    WorkerConfig("m1pro", "m1-pro", "selfplay",
                 ["square8", "hex"], max_memory_gb=48),
    WorkerConfig("mac-studio", "mac-studio", "training",
                 ["nn", "nnue"], max_memory_gb=160),
]


class PipelineOrchestrator:
    def __init__(self, config_path: str, state_path: str, dry_run: bool = False):
        self.config = self.load_config(config_path)
        self.state_path = Path(state_path)
        self.state = self.load_state()
        self.dry_run = dry_run

    def load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def load_state(self) -> PipelineState:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            return PipelineState(**data)
        return PipelineState()

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def run_cmd(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run command with dry-run support."""
        if self.dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, **kwargs)

    async def dispatch_selfplay(self, worker: WorkerConfig) -> None:
        """Dispatch selfplay jobs to a worker."""
        for board in worker.capabilities:
            if board not in ["square8", "square19", "hex"]:
                continue

            games = self.config["selfplay"]["games_per_iteration"].get(
                f"{board}_2p", 50
            )
            max_moves = self.config["selfplay"]["max_moves"].get(board, 500)

            cmd = [
                "ssh", worker.host,
                f"cd ~/ringrift/ai-service && "
                f"python scripts/run_self_play_soak.py "
                f"--num-games {games} --board-type {board} "
                f"--engine-mode mixed --num-players 2 "
                f"--max-moves {max_moves}"
            ]
            self.run_cmd(cmd)

    async def run_selfplay_phase(self) -> None:
        """Distribute selfplay across workers."""
        print("\n=== SELFPLAY PHASE ===")
        tasks = []
        for worker in WORKERS:
            if worker.role == "selfplay":
                task = self.dispatch_selfplay(worker)
                tasks.append(task)
        await asyncio.gather(*tasks)

    async def run_sync_phase(self) -> None:
        """Sync all data to central location."""
        print("\n=== SYNC PHASE ===")
        self.run_cmd(["bash", "scripts/sync_selfplay_data.sh"], check=True)
        self.state.last_sync = datetime.now().isoformat()

    async def run_nn_training(self) -> None:
        """Run neural network training on Mac Studio."""
        print("Starting NN training...")
        cfg = self.config["training"]["nn"]
        cmd = [
            "ssh", "mac-studio",
            f"cd ~/Development/RingRift/ai-service && "
            f"PYTHONPATH=. python app/training/train.py "
            f"--data-path data/training/square8_2p.npz "
            f"--epochs {cfg['epochs_per_iteration']} "
            f"--batch-size {cfg['batch_size']} "
            f"--device {cfg['device']}"
        ]
        self.run_cmd(cmd)

    async def run_nnue_training(self) -> None:
        """Run NNUE training."""
        print("Starting NNUE training...")
        cfg = self.config["training"]["nnue"]
        cmd = [
            "python", "scripts/train_nnue.py",
            "--db", "data/games/merged.db",
            "--epochs", str(cfg["epochs_per_iteration"]),
            "--batch-size", str(cfg["batch_size"]),
        ]
        self.run_cmd(cmd)

    async def run_cmaes_optimization(self) -> None:
        """Run CMA-ES heuristic optimization."""
        print("Starting CMA-ES optimization...")
        cfg = self.config["training"]["cmaes"]
        cmd = [
            "python", "scripts/run_iterative_cmaes.py",
            "--generations", str(cfg["generations_per_iteration"]),
            "--population-size", str(cfg["population_size"]),
            "--games-per-eval", str(cfg["games_per_eval"]),
            "--max-workers", "2",  # Limit to avoid memory pressure
        ]
        self.run_cmd(cmd)

    async def run_training_phase(self) -> None:
        """Run NN, NNUE, and CMA-ES training."""
        print("\n=== TRAINING PHASE ===")
        # Run training tasks (can be parallelized on different machines)
        await asyncio.gather(
            self.run_nn_training(),
            self.run_nnue_training(),
            self.run_cmaes_optimization(),
        )

    async def run_evaluation_phase(self) -> None:
        """Evaluate improved models via tournaments."""
        print("\n=== EVALUATION PHASE ===")
        cfg = self.config["evaluation"]
        cmd = [
            "python", "scripts/run_ai_tournament.py",
            "--round-robin", "--all-engines",
            "--games", str(cfg["tournament_games"]),
        ]
        self.run_cmd(cmd, check=True)

    async def run_full_loop(self, iterations: int) -> None:
        """Execute complete pipeline loop."""
        start_iter = self.state.iteration

        for i in range(start_iter, iterations):
            print(f"\n{'='*60}")
            print(f"=== Pipeline Iteration {i+1}/{iterations} ===")
            print(f"{'='*60}")

            try:
                await self.run_selfplay_phase()
                await self.run_sync_phase()
                await self.run_training_phase()
                await self.run_evaluation_phase()

                self.state.iteration = i + 1
                self.state.phase = "completed"
                self.save_state()

            except subprocess.CalledProcessError as e:
                error_msg = f"Iteration {i+1} failed: {e}"
                print(f"ERROR: {error_msg}")
                self.state.errors.append(error_msg)
                self.save_state()
                raise

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Total iterations: {self.state.iteration}")


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for distributed AI training"
    )
    parser.add_argument("--config", default="config/pipeline.json",
                        help="Pipeline configuration file")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of pipeline iterations")
    parser.add_argument("--state-file", default="logs/pipeline/state.json",
                        help="State file for checkpointing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        args.config,
        args.state_file,
        dry_run=args.dry_run,
    )

    if not args.resume:
        orchestrator.state = PipelineState()

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
      "device": "mps",
      "quality_gate_winrate": 0.55
    },
    "nnue": {
      "epochs_per_iteration": 20,
      "batch_size": 1024,
      "distill_from_nn": true,
      "distill_positions": 100000
    },
    "cmaes": {
      "generations_per_iteration": 10,
      "population_size": 20,
      "games_per_eval": 12,
      "use_nn_evaluator": false,
      "nn_quality_gate": 0.55
    }
  },
  "evaluation": {
    "tournament_games": 20,
    "promotion_threshold": 0.55,
    "regression_threshold": 0.45
  },
  "sync": {
    "interval_minutes": 60,
    "destinations": ["laptop", "mac-studio"],
    "dedupe_games": true
  },
  "rollback": {
    "max_consecutive_failures": 5,
    "keep_backup_models": 3
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

### Experiment Tracking (Optional)

For production deployments, consider integrating:

```python
# Weights & Biases integration example
import wandb

wandb.init(project="ringrift-ai", config={
    "board": "square8",
    "players": 2,
    "iteration": iteration,
})

wandb.log({
    "train_loss": loss,
    "val_accuracy": accuracy,
    "tournament_winrate": winrate,
})
```

### Alerting

- Selfplay stalls (no new games in 30 minutes)
- Training loss divergence
- Model regression (tournament winrate < 45%)
- Worker offline
- Consecutive failures exceed threshold

## Implementation Tasks

1. [x] Create `scripts/pipeline_orchestrator.py` (documented above)
2. [x] Create `scripts/sync_selfplay_data.sh` (documented above)
3. [x] Create `scripts/run_improvement_loop.py` (documented above)
4. [x] Create `config/pipeline.json` (documented above)
5. [x] Add NNUE distillation script (documented above)
6. [x] Add tournament round-robin mode - EXISTS in `run_tournament.py` with `--scheduler round-robin`
7. [x] Set up monitoring - Created `app/training/metrics_logger.py` with TensorBoard + W&B support
8. [x] Configure cloud GPU provisioning scripts - Created `lambda_gpu_setup.sh` and `sync_to_lambda.sh`
9. [x] Add `--dedupe-by-game-id` to `merge_game_dbs.py`
10. [x] Add `--nn-quality-gate` to `run_iterative_cmaes.py` (gates CMA-ES on NN quality)
11. [ ] Add `--evaluator neural` to CMA-ES (future: use NN for position scoring instead of game outcomes)

## Success Criteria

- 10,000+ completed selfplay games across all configurations
- CMA-ES converged weights for all 9 configs
- NN models for square8 2p/3p achieving >60% vs heuristic baseline
- NNUE providing 10%+ speedup at Minimax difficulty 4+
- NNUE distillation achieving >0.9 correlation with NN evaluations
- Continuous improvement loop running autonomously with <5% failure rate
- Automatic rollback working on model regressions
