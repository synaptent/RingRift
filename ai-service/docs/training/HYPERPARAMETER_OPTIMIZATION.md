# Hyperparameter Optimization

RingRift uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) and NAS (Neural Architecture Search) for automated hyperparameter and architecture optimization.

## Overview

```
                    OPTIMIZATION SYSTEM

    +-------------------+     +-------------------+
    | CMA-ES Heuristic  |     | NAS Architecture  |
    | Weight Optimizer  |     | Search            |
    +--------+----------+     +--------+----------+
             |                         |
             v                         v
    +-------------------+     +-------------------+
    | GPU-Accelerated   |     | Distributed       |
    | Fitness Eval      |     | Worker Cluster    |
    +--------+----------+     +--------+----------+
             |                         |
             +------------+------------+
                          |
                          v
    +-----------------------------------------------------+
    |                 MODEL REGISTRY                       |
    +-----------------------------------------------------+
    |  Auto-promotion based on fitness improvement         |
    |  Version tracking with lineage                       |
    |  Prometheus metrics integration                      |
    +-----------------------------------------------------+
```

## CMA-ES Optimization

CMA-ES optimizes heuristic evaluation weights for the AI engine using evolutionary search with covariance matrix adaptation.

### Heuristic Weights

The optimizer tunes these evaluation weights:

| Weight                  | Default | Description                   |
| ----------------------- | ------- | ----------------------------- |
| `material_weight`       | 1.0     | Value of pieces on board      |
| `ring_count_weight`     | 0.5     | Importance of ring structures |
| `stack_height_weight`   | 0.3     | Value of tall stacks          |
| `center_control_weight` | 0.4     | Board center control value    |
| `territory_weight`      | 0.8     | Territorial influence         |
| `mobility_weight`       | 0.2     | Movement options value        |
| `line_potential_weight` | 0.6     | Potential winning line value  |
| `defensive_weight`      | 0.3     | Defensive positioning value   |

### GPU-Accelerated CMA-ES

The GPU accelerator provides 10-100x faster fitness evaluation.

**Usage:**

```bash
# Single GPU optimization
python scripts/run_gpu_cmaes.py \
    --board square8 \
    --num-players 2 \
    --generations 50 \
    --population-size 20 \
    --games-per-eval 50 \
    --output-dir logs/cmaes/gpu_square8_2p

# Multi-GPU on same machine
python scripts/run_gpu_cmaes.py \
    --board square8 \
    --num-players 2 \
    --generations 50 \
    --multi-gpu \
    --output-dir logs/cmaes/gpu_multi
```

**Parameters:**
| Parameter | Default | Description |
|----------------------|---------|--------------------------------|
| `--generations` | 50 | Number of CMA-ES generations |
| `--population-size` | 20 | Population size per generation |
| `--games-per-eval` | 50 | Games for fitness evaluation |
| `--sigma` | 0.5 | Initial step size |
| `--opponent-mode` | baseline| Opponent selection strategy |

### Distributed CMA-ES

For cluster-wide optimization:

```bash
# Coordinator
python scripts/run_distributed_gpu_cmaes.py \
    --mode coordinator \
    --board square8 \
    --generations 100

# Workers (on each GPU node)
python scripts/cmaes_cloud_worker.py \
    --coordinator-url http://coordinator:8080
```

### Registry Integration

Optimized weights are automatically registered in the model registry.

```python
from app.training.cmaes_registry_integration import (
    register_cmaes_result,
    get_best_heuristic_model,
    load_heuristic_weights_from_registry,
)

# After optimization completes
model_id, version = register_cmaes_result(
    weights_path=Path("logs/cmaes/optimized_weights.json"),
    board_type="square8",
    num_players=2,
    fitness=0.85,
    generation=50,
    cmaes_config={
        "population_size": 20,
        "sigma": 0.5,
        "games_per_eval": 50,
    },
)

# Load production weights
weights = load_heuristic_weights_from_registry(
    board_type="square8",
    num_players=2,
    stage="production",
)
```

**Auto-Promotion:**

- Models are auto-promoted to STAGING if fitness improves by >2%
- Manual promotion to PRODUCTION after validation

### CMA-ES Scripts

| Script                         | Purpose                       |
| ------------------------------ | ----------------------------- |
| `run_gpu_cmaes.py`             | Single/multi-GPU optimization |
| `run_distributed_gpu_cmaes.py` | Cluster-wide optimization     |
| `cmaes_cloud_worker.py`        | Remote worker for distributed |
| `run_iterative_cmaes.py`       | Iterative refinement          |
| `run_cmaes_optimization.py`    | Basic CMA-ES (CPU)            |

## Neural Architecture Search (NAS)

NAS explores network architectures for the neural network policy/value heads.

### Distributed NAS

```bash
# Launch coordinator
python scripts/launch_distributed_nas.py \
    --mode coordinator \
    --search-space default \
    --budget 100

# Launch workers
python scripts/distributed_nas.py \
    --coordinator-url http://coordinator:8080 \
    --gpu-id 0
```

### Search Space

The NAS explores:

| Component         | Options                     |
| ----------------- | --------------------------- |
| `hidden_layers`   | [2, 3, 4, 5]                |
| `hidden_size`     | [128, 256, 512, 1024]       |
| `activation`      | [relu, gelu, silu, mish]    |
| `normalization`   | [none, batch, layer, group] |
| `dropout`         | [0.0, 0.1, 0.2, 0.3]        |
| `residual`        | [true, false]               |
| `attention_heads` | [0, 2, 4, 8]                |

## Configuration

### Environment Variables

| Variable                        | Default | Description                |
| ------------------------------- | ------- | -------------------------- |
| `RINGRIFT_CMAES_GENERATIONS`    | 50      | Default generations        |
| `RINGRIFT_CMAES_POPULATION`     | 20      | Default population size    |
| `RINGRIFT_CMAES_GAMES_PER_EVAL` | 50      | Games per fitness eval     |
| `RINGRIFT_NAS_BUDGET`           | 100     | Total architectures to try |

### Task Coordinator Integration

CMA-ES and NAS tasks are coordinated to prevent resource conflicts:

```python
from app.coordination.task_coordinator import TaskType, get_coordinator

coordinator = get_coordinator()

# Check before starting CMA-ES
allowed, reason = coordinator.can_spawn_task(TaskType.CMAES, node_id)
if allowed:
    coordinator.register_task(task_id, TaskType.CMAES, node_id)
```

**Limits:**

- Max 1 CMA-ES task cluster-wide
- CMA-ES uses GPU resource type
- Estimated duration: 8 hours (in duration_scheduler)

## Monitoring

### Prometheus Metrics

```
# CMA-ES metrics
ringrift_cmaes_runs_total{config="square8_2p", board_type="square8"} 5
ringrift_cmaes_best_fitness{config="square8_2p", board_type="square8"} 0.82
ringrift_cmaes_generations_total{config="square8_2p"} 250

# NAS metrics
ringrift_nas_architectures_evaluated 45
ringrift_nas_best_accuracy 0.89
```

### Log Files

| Path                                   | Content                      |
| -------------------------------------- | ---------------------------- |
| `logs/cmaes/gpu_{board}_{players}/`    | Generation logs, checkpoints |
| `logs/cmaes/optimized_weights.json`    | Final optimized weights      |
| `logs/nas/search_log.json`             | NAS search history           |
| `data/model_registry/heuristic_*.json` | Registered models            |

## Fitness Evaluation

### GPU-Accelerated Games

The `ParallelGameRunner` runs batch games on GPU:

```python
from app.ai.gpu_parallel_games import (
    ParallelGameRunner,
    evaluate_candidate_fitness_gpu,
)

runner = ParallelGameRunner(
    batch_size=50,
    board_size=8,
    num_players=2,
    device=torch.device("cuda"),
)

# Evaluate candidate vs baseline
win_rate = evaluate_candidate_fitness_gpu(
    candidate_weights=optimized_weights,
    opponent_weights=baseline_weights,
    num_games=100,
    board_size=8,
    num_players=2,
)
```

### Fitness Calculation

Fitness is calculated as win rate against baseline:

```
fitness = wins / total_games
```

Fitness to Elo conversion:

```
elo_diff = -400 * log10((1/fitness) - 1)
```

| Fitness | Elo Diff | Interpretation      |
| ------- | -------- | ------------------- |
| 0.50    | 0        | Equal strength      |
| 0.60    | +72      | Slightly stronger   |
| 0.70    | +147     | Moderately stronger |
| 0.80    | +240     | Much stronger       |
| 0.90    | +382     | Dominant            |

## Iterative Refinement

The `run_iterative_cmaes.py` script supports:

1. **Warm start** - Initialize from previous best weights
2. **Population adaptation** - Increase population on plateau
3. **Sigma annealing** - Reduce step size over generations
4. **Opponent diversity** - Mix baseline with self-play

```bash
python scripts/run_iterative_cmaes.py \
    --board square8 \
    --num-players 2 \
    --iterations 5 \
    --generations-per-iteration 20 \
    --warm-start \
    --output-dir logs/cmaes/iterative
```

## Best Practices

1. **Start with baseline** - Run against baseline-only first
2. **Sufficient games** - Use 50+ games per evaluation for stable fitness
3. **Population size** - 2x number of weights is a good starting point
4. **Sigma tuning** - Start with 0.5, reduce if progress stalls
5. **Multi-GPU** - Use all available GPUs for faster evaluation
6. **Checkpointing** - Save every generation for recovery
7. **Registry integration** - Register results for traceability

## Troubleshooting

### CMA-ES Not Converging

1. Increase population size
2. Reduce sigma
3. Increase games per evaluation
4. Check baseline is reasonable

### GPU Memory Issues

```python
from app.ai.gpu_batch import clear_gpu_memory
clear_gpu_memory()
```

Or reduce batch size:

```bash
python scripts/run_gpu_cmaes.py --games-per-eval 20
```

### Fitness Plateau

1. Try different opponent modes:
   - `baseline-only` - Pure baseline comparison
   - `self-play` - Mix with self-play
   - `diverse` - Mix multiple opponents

2. Increase generation count
3. Reset sigma to explore more

## Related Documentation

- [TRAINING_OPTIMIZATIONS.md](TRAINING_OPTIMIZATIONS.md) - Training pipeline
- [COORDINATION_SYSTEM.md](COORDINATION_SYSTEM.md) - Task coordination
- [scripts/README.md](../scripts/README.md) - Script reference
