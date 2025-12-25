# Metrics Module

Unified metrics collection system for RingRift AI service.

## Overview

This module consolidates all metrics collection:

- Prometheus metrics for monitoring and alerting
- Training metrics for experiment tracking
- Orchestrator metrics for pipeline observability

## Key Components

### Recording Metrics

```python
from app.metrics import (
    record_selfplay_batch,
    record_training_run,
    record_model_promotion,
    record_evaluation_result,
)

# Record selfplay progress
record_selfplay_batch(
    board_type="square8",
    num_players=2,
    games=100,
    duration_seconds=60.5,
)

# Record training completion
record_training_run(
    board_type="hex8",
    num_players=2,
    epochs=50,
    final_loss=0.025,
    final_accuracy=0.76,
    duration_seconds=3600,
)

# Record model promotion
record_model_promotion(
    board_type="hex8",
    num_players=2,
    model_id="hex8_2p_v3",
    elo_improvement=45,
)
```

### Metrics Server

Start a Prometheus-compatible metrics endpoint:

```python
from app.metrics import start_metrics_server

# Start on port 9090
start_metrics_server(port=9090)

# Prometheus can scrape: http://localhost:9090/metrics
```

### Prometheus Metrics

| Metric                            | Type      | Description                    |
| --------------------------------- | --------- | ------------------------------ |
| `selfplay_games_total`            | Counter   | Total selfplay games generated |
| `selfplay_batch_duration_seconds` | Histogram | Selfplay batch duration        |
| `training_runs_total`             | Counter   | Training runs completed        |
| `training_duration_seconds`       | Histogram | Training run duration          |
| `model_promotions_total`          | Counter   | Models promoted to production  |
| `evaluation_games_total`          | Counter   | Evaluation games played        |

### Labels

All metrics include labels for filtering:

- `board_type`: hex8, square8, square19, hexagonal
- `num_players`: 2, 3, 4
- `engine`: heuristic, gumbel, mcts
- `status`: success, failed

### API Request Metrics

For application-level metrics (imported from `app.metrics_base`):

```python
from app.metrics import (
    AI_MOVE_REQUESTS,
    AI_MOVE_LATENCY,
    ACTIVE_GAMES,
)

# Increment request counter
AI_MOVE_REQUESTS.labels(board_type="hex8", player_count=2).inc()

# Record latency
AI_MOVE_LATENCY.labels(board_type="hex8").observe(0.15)

# Set gauge
ACTIVE_GAMES.set(42)
```

## Grafana Integration

Metrics can be visualized in Grafana dashboards:

```promql
# Games per hour by board type
rate(selfplay_games_total[1h])

# Average training duration
histogram_quantile(0.5, training_duration_seconds_bucket)

# Model promotion rate
increase(model_promotions_total[24h])
```

## Configuration

| Environment Variable | Default    | Description               |
| -------------------- | ---------- | ------------------------- |
| `METRICS_PORT`       | 9090       | Prometheus endpoint port  |
| `METRICS_ENABLED`    | true       | Enable/disable collection |
| `METRICS_PREFIX`     | ringrift\_ | Metric name prefix        |

## Usage with Orchestrator

The P2P orchestrator automatically records metrics:

```python
# Orchestrator records on job completion
from app.metrics import record_job_completion

record_job_completion(
    job_type="selfplay",
    node_id="lambda-gh200-b",
    success=True,
    duration_seconds=120.5,
)
```
