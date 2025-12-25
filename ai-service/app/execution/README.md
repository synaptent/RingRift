# Execution Module

Unified execution framework for running commands locally, via SSH, or on cluster workers.

## Overview

This module provides abstracted execution backends that all orchestrators should use instead of implementing their own SSH/subprocess logic:

- Low-level executors (SSH, Local)
- High-level backends (Slurm, SSH pool, Local)
- Game execution utilities

## Key Components

### Low-Level Executors

```python
from app.execution import SSHExecutor, LocalExecutor, ExecutionResult

# SSH execution
executor = SSHExecutor(host="worker-1", user="ringrift")
result = await executor.run("python scripts/run_selfplay.py")
print(f"Exit code: {result.exit_code}")
print(f"Output: {result.stdout}")

# Local execution
executor = LocalExecutor()
result = await executor.run("python scripts/train.py")
```

### High-Level Backends

```python
from app.execution import get_backend, BackendType

# Get configured backend (auto-detects from config)
backend = get_backend()

# Run selfplay across all available workers
results = await backend.run_selfplay(
    games=100,
    board_type="square8",
    num_players=2,
)

# Run tournament
result = await backend.run_tournament(
    agent_ids=["random", "heuristic"],
    games_per_pairing=20,
)
```

### Executor Pool

```python
from app.execution import ExecutorPool

# Pool of SSH executors for parallel work
pool = ExecutorPool(
    hosts=["worker-1", "worker-2", "worker-3"],
    max_concurrent=10,
)

# Run command on all hosts
results = await pool.run_all("nvidia-smi")
```

### Game Execution

```python
from app.execution import GameExecutor, run_quick_game, run_selfplay_batch

# Run a single game
executor = GameExecutor(board_type="square8", num_players=2)
result = executor.run_game(
    player_configs=[
        {"ai_type": "mcts", "difficulty": 5},
        {"ai_type": "heuristic", "difficulty": 3},
    ],
)
print(f"Winner: {result.winner}")
print(f"Moves: {result.move_count}")

# Quick game helper
result = run_quick_game(p1_type="mcts", p2_type="heuristic")

# Batch selfplay
results = run_selfplay_batch(
    board_type="hex8",
    num_players=2,
    num_games=100,
    engine="gumbel",
)
```

### Parallel Game Execution

```python
from app.execution import ParallelGameExecutor

# Run many games in parallel
executor = ParallelGameExecutor(
    board_type="hex8",
    num_players=2,
    max_workers=8,
)

results = executor.run_games(
    num_games=1000,
    player_configs=configs,
)
```

## Backend Types

| Type                  | Description                 |
| --------------------- | --------------------------- |
| `LocalBackend`        | Run on local machine        |
| `SSHBackend`          | Run via SSH on remote hosts |
| `SlurmBackend`        | Submit to Slurm cluster     |
| `OrchestratorBackend` | Use P2P orchestrator        |

## ExecutionResult

```python
@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    host: str | None
```

## GameResult

```python
@dataclass
class GameResult:
    winner: int | None  # Player number or None for draw
    outcome: GameOutcome  # WIN, DRAW, TIMEOUT
    move_count: int
    duration: float
    final_state: dict
```

## Configuration

Backends can be configured via environment or config file:

```yaml
execution:
  backend: ssh # local, ssh, slurm, orchestrator
  ssh:
    hosts:
      - worker-1
      - worker-2
    user: ringrift
    key_path: ~/.ssh/id_rsa
  slurm:
    partition: gpu
    time_limit: '4:00:00'
```

## See Also

- `app.p2p` - P2P orchestration for distributed work
- `app.distributed` - Cluster monitoring
