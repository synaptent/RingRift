# SelfplayScheduler Usage Guide

## Overview

The `SelfplayScheduler` class was extracted from `p2p_orchestrator.py` to provide modular, testable selfplay configuration selection and job targeting.

## Location

- **Module**: `scripts/p2p/managers/selfplay_scheduler.py`
- **Export**: `scripts/p2p/managers/__init__.py`

## Extracted Methods

The following methods were extracted from the orchestrator:

1. `_get_elo_based_priority_boost` → `get_elo_based_priority_boost()`
2. `_pick_weighted_selfplay_config` → `pick_weighted_config()`
3. `_target_selfplay_jobs_for_node` → `get_target_jobs_for_node()`
4. `_get_hybrid_job_targets` → `get_hybrid_job_targets()`
5. `_should_spawn_cpu_only_jobs` → `should_spawn_cpu_only_jobs()`
6. `_track_selfplay_diversity` → `track_diversity()`
7. `_get_diversity_metrics` → `get_diversity_metrics()`

## Classes

### `DiversityMetrics`

Dataclass for tracking selfplay diversity:

```python
@dataclass
class DiversityMetrics:
    games_by_engine_mode: dict[str, int]
    games_by_board_config: dict[str, int]
    games_by_difficulty: dict[str, int]
    asymmetric_games: int
    symmetric_games: int
    last_reset: float
```

### `SelfplayScheduler`

Main scheduler class with dependency injection:

```python
class SelfplayScheduler:
    def __init__(
        self,
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        should_stop_production_fn: Callable[[Any], bool] | None = None,
        should_throttle_production_fn: Callable[[Any], bool] | None = None,
        get_throttle_factor_fn: Callable[[Any], float] | None = None,
        record_utilization_fn: Callable[[str, float, float, float, int], None] | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[[str, int, float, float], int] | None = None,
        should_scale_up_fn: Callable[[str, float, float, int], tuple[bool, str]] | None = None,
        should_scale_down_fn: Callable[[str, float, float, float], tuple[bool, int, str]] | None = None,
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        verbose: bool = False,
    )
```

## Usage Example in Orchestrator

### Before (Original Code)

```python
class P2POrchestrator:
    def __init__(self):
        # ... many fields ...
        self.diversity_metrics = {
            "games_by_engine_mode": {},
            "games_by_board_config": {},
            # ... more fields ...
        }

    def _pick_weighted_selfplay_config(self, node):
        # 100+ lines of implementation
        pass

    def _target_selfplay_jobs_for_node(self, node):
        # 150+ lines of implementation
        pass
```

### After (With SelfplayScheduler)

```python
from scripts.p2p.managers import SelfplayScheduler

class P2POrchestrator:
    def __init__(self):
        # ... other initialization ...

        # Initialize scheduler with dependency injection
        self.selfplay_scheduler = SelfplayScheduler(
            get_cluster_elo_fn=self._get_cluster_elo_summary,
            load_curriculum_weights_fn=self._load_curriculum_weights,
            get_board_priority_overrides_fn=self._get_board_priority_overrides,
            should_stop_production_fn=lambda qt: should_stop_production(qt),
            should_throttle_production_fn=lambda qt: should_throttle_production(qt),
            get_throttle_factor_fn=lambda qt: get_throttle_factor(qt),
            record_utilization_fn=lambda nid, cpu, gpu, mem, jobs: record_utilization(nid, cpu, gpu, mem, jobs),
            get_host_targets_fn=lambda nid: get_host_targets(nid),
            get_target_job_count_fn=lambda nid, cpu_cnt, cpu_pct, gpu_pct: get_target_job_count(nid, cpu_cnt, cpu_pct, gpu_pct),
            should_scale_up_fn=lambda nid, cpu, gpu, jobs: should_scale_up(nid, cpu, gpu, jobs),
            should_scale_down_fn=lambda nid, cpu, gpu, mem: should_scale_down(nid, cpu, gpu, mem),
            get_max_selfplay_for_node_fn=get_max_selfplay_for_node,
            get_hybrid_selfplay_limits_fn=get_hybrid_selfplay_limits,
            verbose=self.verbose,
        )

    def _pick_weighted_selfplay_config(self, node):
        """Delegate to scheduler."""
        return self.selfplay_scheduler.pick_weighted_config(node)

    def _target_selfplay_jobs_for_node(self, node):
        """Delegate to scheduler."""
        return self.selfplay_scheduler.get_target_jobs_for_node(node)

    def _get_diversity_metrics(self):
        """Delegate to scheduler."""
        return self.selfplay_scheduler.get_diversity_metrics()
```

## Standalone Usage (Testing)

```python
from scripts.p2p.managers import SelfplayScheduler
from scripts.p2p.models import NodeInfo

# Create scheduler with minimal dependencies
scheduler = SelfplayScheduler(verbose=True)

# Create a test node
node = NodeInfo(
    node_id="test-gpu",
    host="127.0.0.1",
    port=8770,
    cpu_count=16,
    memory_gb=64,
    has_gpu=True,
    gpu_name="RTX 4090",
    cpu_percent=30.0,
    gpu_percent=25.0,
    selfplay_jobs=4,
)

# Pick a weighted config
config = scheduler.pick_weighted_config(node)
print(f"Config: {config['board_type']}_{config['num_players']}p")

# Get job target
target = scheduler.get_target_jobs_for_node(node)
print(f"Target jobs: {target}")

# Track diversity
scheduler.track_diversity(config)
metrics = scheduler.get_diversity_metrics()
print(f"Metrics: {metrics}")
```

## Features

### 1. Priority-Based Config Selection

Combines multiple priority sources:

- **Static priority**: Base priority for each config (3-8)
- **Elo-based boost**: +0 to +3 based on model performance
- **Curriculum weights**: -2 to +3 based on learning curriculum
- **Board priority overrides**: +0 to +6 from config file

### 2. Resource-Aware Job Targeting

Considers:

- **Backpressure**: Reduces/stops production when training queue is full
- **Hardware capabilities**: GPU type, CPU count, memory
- **Current utilization**: CPU %, GPU %, memory %
- **Adaptive scaling**: Scales up when underutilized, down when overloaded

### 3. Diversity Tracking

Monitors:

- Games by engine mode (mixed, mcts-only, etc.)
- Games by board configuration (hex8_2p, square19_4p, etc.)
- Asymmetric vs symmetric games
- Difficulty band distribution

## Benefits

1. **Modularity**: Selfplay scheduling logic is isolated from orchestrator
2. **Testability**: Easy to unit test without full orchestrator context
3. **Dependency Injection**: All external dependencies are injected, no globals
4. **Reusability**: Can be used in other contexts (e.g., standalone scheduler)
5. **Maintainability**: 737 lines in dedicated module vs inline in 30K+ line orchestrator

## Migration Path

To integrate into p2p_orchestrator.py:

1. Import the scheduler: `from scripts.p2p.managers import SelfplayScheduler`
2. Initialize in `__init__`: Create scheduler instance with dependency injection
3. Replace method calls: Change `self._pick_weighted_selfplay_config(node)` to `self.selfplay_scheduler.pick_weighted_config(node)`
4. Remove old methods: Delete the extracted methods from orchestrator

## Testing

Run syntax check:

```bash
python3 -m py_compile scripts/p2p/managers/selfplay_scheduler.py
```

Run basic functionality test:

```bash
python3 -c "from scripts.p2p.managers import SelfplayScheduler; print('✓ Import successful')"
```

See inline test in this file for comprehensive testing example.
