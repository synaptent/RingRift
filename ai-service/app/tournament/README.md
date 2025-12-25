# Tournament Module

Tournament system for AI agent evaluation with Elo ratings.

## Overview

This module provides a unified tournament framework for running AI agent competitions:

- Round-robin and Swiss scheduling
- Elo rating tracking and persistence
- Composite gauntlet (algorithm + NN tournaments)
- Hierarchical culling for model selection

## Quick Start

```python
from app.tournament import run_quick_tournament

# Run a quick tournament
results = run_quick_tournament(
    agent_ids=["random", "heuristic", "mcts_100"],
    board_type="square8",
    num_players=2,
    games_per_pairing=10,
)

for agent_id, rating in results.final_ratings.items():
    stats = results.agent_stats.get(agent_id, {})
    print(f"{agent_id}: {rating:.0f} Elo ({stats.get('win_rate', 0)*100:.0f}% win)")
```

## Key Components

### Tournament Runner

```python
from app.tournament import TournamentRunner, AIAgentRegistry, RoundRobinScheduler

registry = AIAgentRegistry()
scheduler = RoundRobinScheduler()
runner = TournamentRunner(registry, scheduler)

results = runner.run_tournament(
    agent_ids=["random", "heuristic"],
    board_type="hex8",
    num_players=2,
    games_per_pairing=20,
)
```

### Tournament Orchestrator

```python
from app.tournament import TournamentOrchestrator, run_quick_evaluation

# High-level orchestrator
orchestrator = TournamentOrchestrator()

# Quick evaluation against baselines
result = run_quick_evaluation(
    model_path="models/hex8_2p_v3.pth",
    board_type="hex8",
    num_players=2,
)
print(f"Win rate vs random: {result.vs_random_win_rate}")
print(f"Win rate vs heuristic: {result.vs_heuristic_win_rate}")

# Elo calibration
run_elo_calibration(num_games=100)
```

### Schedulers

```python
from app.tournament import RoundRobinScheduler, SwissScheduler

# Round-robin: everyone plays everyone
rr = RoundRobinScheduler()
matches = rr.generate_matches(agent_ids, games_per_pairing=10)

# Swiss: adaptive pairing based on standings
swiss = SwissScheduler()
matches = swiss.generate_round(agent_ids, current_standings)
```

### Elo Service

```python
from app.tournament import get_elo_service, EloService

# Get singleton service
elo = get_elo_service()

# Record match result
elo.record_match(
    winner_id="model_v3",
    loser_id="heuristic",
    board_type="hex8",
    num_players=2,
)

# Get current rating
rating = elo.get_rating("model_v3")
print(f"Rating: {rating.elo} (uncertainty: {rating.uncertainty})")

# Get match history
history = elo.get_match_history("model_v3", limit=100)
```

### Composite Gauntlet

Two-phase evaluation: algorithm tournament then NN tournament.

```python
from app.tournament import run_two_phase_gauntlet, CompositeGauntlet

# Run two-phase gauntlet
results = run_two_phase_gauntlet(
    model_path="models/hex8_2p_v3.pth",
    board_type="hex8",
    num_players=2,
    algorithm_games=50,
    nn_games=50,
)

# Separate tournaments
from app.tournament import run_algorithm_tournament, run_nn_tournament

# Algorithm tournament (vs heuristics)
alg_results = run_algorithm_tournament(
    agent_ids=["random", "heuristic", "defensive"],
    board_type="square8",
    games_per_pairing=20,
)

# NN tournament (vs other models)
nn_results = run_nn_tournament(
    model_paths=["v1.pth", "v2.pth", "v3.pth"],
    board_type="square8",
    games_per_pairing=10,
)
```

### Hierarchical Culling

```python
from app.tournament import run_hierarchical_culling, check_culling_needed

# Check if culling is needed
if check_culling_needed(board_type="hex8", num_players=2):
    # Run culling to remove weak models
    report = run_hierarchical_culling(
        board_type="hex8",
        num_players=2,
        keep_top_n=5,
    )
    print(f"Culled: {report.culled_models}")
    print(f"Kept: {report.kept_models}")
```

### Consistency Monitoring

```python
from app.tournament import ConsistencyMonitor, run_consistency_checks

# Check for rating consistency issues
report = run_consistency_checks(board_type="hex8")
if not report.is_consistent:
    for issue in report.issues:
        print(f"Issue: {issue}")
```

## Agent Types

| Type        | Description               |
| ----------- | ------------------------- |
| `random`    | Random valid moves        |
| `heuristic` | Rule-based player         |
| `mcts_N`    | MCTS with N simulations   |
| `gumbel_N`  | Gumbel MCTS with budget N |
| `nn:path`   | Neural network from path  |

## Scheduler Types

| Type          | Description                         |
| ------------- | ----------------------------------- |
| `round_robin` | Every agent plays every other agent |
| `swiss`       | Adaptive pairing based on standings |

## Configuration

```python
from app.tournament import TournamentConfig

config = TournamentConfig(
    games_per_pairing=20,
    max_workers=8,
    timeout_seconds=300,
    persist_results=True,
)
```

## See Also

- `app.gauntlet` - Game gauntlet runner
- `app.evaluation` - Benchmark framework
- `app.training.elo_service` - Canonical Elo service
