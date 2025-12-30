# Event Payload Schemas

This document defines the expected payload structure for critical events in the RingRift AI training pipeline.

**Created**: December 30, 2025

## Overview

Events are published via the `EventBus` with a `payload` dictionary containing event-specific data. This document specifies the required and optional fields for each critical event type.

For emitters and subscribers, see `EVENT_CATALOG.md`. For cross-process bridging, see `EVENT_SYSTEM_REFERENCE.md`.

## Payload Extraction Utilities

Use the canonical extraction utilities from `app/coordination/event_utils.py`:

```python
from app.coordination.event_utils import (
    normalize_event_payload,  # Extract payload from event object
    parse_config_key,         # Parse "hex8_2p" -> (board_type, num_players)
    extract_config_key,       # Get config_key with fallbacks
    extract_model_path,       # Get model path with fallbacks
    extract_evaluation_data,  # Parse EVALUATION_COMPLETED payload
    extract_training_data,    # Parse TRAINING_COMPLETED payload
    make_config_key,          # Create "hex8_2p" from components
)
```

---

## Training Pipeline Events

### TRAINING_COMPLETED

Emitted when a training job finishes successfully.

| Field              | Type    | Required | Description                               |
| ------------------ | ------- | -------- | ----------------------------------------- |
| `config_key`       | `str`   | Yes      | Config identifier (e.g., `"hex8_2p"`)     |
| `board_type`       | `str`   | Yes      | Board type (e.g., `"hex8"`, `"square19"`) |
| `num_players`      | `int`   | Yes      | Player count (2, 3, or 4)                 |
| `model_path`       | `str`   | Yes      | Absolute path to trained model            |
| `epochs`           | `int`   | Yes      | Number of epochs trained                  |
| `final_loss`       | `float` | Yes      | Final training loss                       |
| `samples_trained`  | `int`   | Yes      | Total samples used                        |
| `job_id`           | `str`   | No       | Job identifier (for tracking)             |
| `duration_seconds` | `float` | No       | Training duration                         |
| `early_stopped`    | `bool`  | No       | Whether early stopping triggered          |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "board_type": "hex8",
    "num_players": 2,
    "model_path": "/models/canonical_hex8_2p.pth",
    "epochs": 50,
    "final_loss": 0.0423,
    "samples_trained": 85000,
    "job_id": "train_hex8_2p_1704100200",
    "duration_seconds": 1832.5,
    "early_stopped": true,
}
```

**Dataclass:** `TrainingEventData` in `event_utils.py`

---

### EVALUATION_COMPLETED

Emitted when model evaluation (gauntlet) finishes.

| Field                       | Type        | Required | Description                          |
| --------------------------- | ----------- | -------- | ------------------------------------ |
| `config_key`                | `str`       | Yes      | Config identifier                    |
| `board_type`                | `str`       | Yes      | Board type                           |
| `num_players`               | `int`       | Yes      | Player count                         |
| `model_path`                | `str`       | Yes      | Path to evaluated model              |
| `elo`                       | `float`     | Yes      | Estimated Elo rating                 |
| `games_played`              | `int`       | Yes      | Games used in evaluation             |
| `win_rate`                  | `float`     | Yes      | Win rate (0.0-1.0)                   |
| `is_multi_harness`          | `bool`      | No       | If evaluated with multiple harnesses |
| `harness_results`           | `dict`      | No       | Per-harness Elo results              |
| `best_harness`              | `str`       | No       | Best-performing harness type         |
| `composite_participant_ids` | `list[str]` | No       | Composite IDs for Elo tracking       |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "board_type": "hex8",
    "num_players": 2,
    "model_path": "/models/canonical_hex8_2p.pth",
    "elo": 1650.5,
    "games_played": 100,
    "win_rate": 0.72,
    "is_multi_harness": true,
    "harness_results": {
        "gumbel_mcts": {"elo": 1650, "games": 50},
        "minimax": {"elo": 1580, "games": 50},
    },
    "best_harness": "gumbel_mcts",
}
```

**Dataclass:** `EvaluationEventData` in `event_utils.py`

---

### MODEL_PROMOTED

Emitted when a model is promoted to canonical status.

| Field          | Type    | Required | Description            |
| -------------- | ------- | -------- | ---------------------- |
| `config_key`   | `str`   | Yes      | Config identifier      |
| `model_path`   | `str`   | Yes      | Path to promoted model |
| `previous_elo` | `float` | No       | Previous canonical Elo |
| `new_elo`      | `float` | Yes      | New model's Elo        |
| `elo_delta`    | `float` | No       | Elo improvement        |
| `promoted_at`  | `str`   | No       | ISO timestamp          |
| `supersedes`   | `str`   | No       | Path to previous model |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "model_path": "/models/canonical_hex8_2p.pth",
    "previous_elo": 1580.0,
    "new_elo": 1650.5,
    "elo_delta": 70.5,
    "promoted_at": "2025-12-30T14:32:00Z",
    "supersedes": "/models/canonical_hex8_2p_v41.pth",
}
```

---

### TRAINING_THRESHOLD_REACHED

Emitted when sufficient data is available to trigger training.

| Field           | Type    | Required | Description                  |
| --------------- | ------- | -------- | ---------------------------- |
| `config_key`    | `str`   | Yes      | Config identifier            |
| `sample_count`  | `int`   | Yes      | Available samples            |
| `threshold`     | `int`   | Yes      | Required threshold           |
| `quality_score` | `float` | No       | Data quality score (0.0-1.0) |
| `triggered_at`  | `str`   | No       | ISO timestamp                |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "sample_count": 50000,
    "threshold": 5000,
    "quality_score": 0.85,
    "triggered_at": "2025-12-30T10:15:00Z",
}
```

---

## Data Sync Events

### DATA_SYNC_COMPLETED

Emitted when data synchronization finishes.

| Field               | Type        | Required | Description                                      |
| ------------------- | ----------- | -------- | ------------------------------------------------ |
| `source`            | `str`       | Yes      | Source node identifier                           |
| `targets`           | `list[str]` | No       | Target nodes                                     |
| `files_synced`      | `int`       | No       | Number of files transferred                      |
| `bytes_transferred` | `int`       | No       | Total bytes synced                               |
| `duration_seconds`  | `float`     | No       | Sync duration                                    |
| `sync_type`         | `str`       | No       | Type: `"broadcast"`, `"ephemeral"`, `"priority"` |

**Example:**

```python
{
    "source": "coordinator",
    "targets": ["runpod-h100", "vast-4090-1"],
    "files_synced": 5,
    "bytes_transferred": 524288000,
    "duration_seconds": 12.5,
    "sync_type": "broadcast",
}
```

---

### SYNC_REQUEST

Explicit sync request (router-driven).

| Field       | Type        | Required | Description                                   |
| ----------- | ----------- | -------- | --------------------------------------------- |
| `source`    | `str`       | Yes      | Requesting component                          |
| `targets`   | `list[str]` | No       | Target nodes (empty = all)                    |
| `data_type` | `str`       | Yes      | Data type: `"games"`, `"models"`, `"npz"`     |
| `reason`    | `str`       | No       | Trigger reason                                |
| `priority`  | `str`       | No       | Priority: `"normal"`, `"high"`, `"emergency"` |

**Example:**

```python
{
    "source": "TrainingCoordinator",
    "targets": ["nebius-h100-1"],
    "data_type": "models",
    "reason": "post_training_distribution",
    "priority": "high",
}
```

---

## Selfplay Events

### SELFPLAY_COMPLETE

Emitted when a batch of selfplay games finishes.

| Field              | Type    | Required | Description                 |
| ------------------ | ------- | -------- | --------------------------- |
| `config_key`       | `str`   | Yes      | Config identifier           |
| `games_completed`  | `int`   | Yes      | Games in this batch         |
| `total_games`      | `int`   | No       | Cumulative total for config |
| `node_id`          | `str`   | No       | Node that ran selfplay      |
| `model_id`         | `str`   | No       | Model used                  |
| `duration_seconds` | `float` | No       | Batch duration              |
| `games_per_second` | `float` | No       | Throughput metric           |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "games_completed": 100,
    "total_games": 5000,
    "node_id": "vast-4090-1",
    "model_id": "canonical_hex8_2p",
    "duration_seconds": 180.5,
    "games_per_second": 0.55,
}
```

---

### NEW_GAMES_AVAILABLE

Signal that new games are ready for processing.

| Field           | Type  | Required | Description                |
| --------------- | ----- | -------- | -------------------------- |
| `host`          | `str` | Yes      | Host with new data         |
| `new_games`     | `int` | Yes      | Count of new games         |
| `config_key`    | `str` | No       | Specific config (if known) |
| `database_path` | `str` | No       | Path to database           |

**Example:**

```python
{
    "host": "runpod-h100",
    "new_games": 500,
    "config_key": "hex8_2p",
    "database_path": "/data/games/selfplay_hex8_2p.db",
}
```

---

## Curriculum Events

### CURRICULUM_REBALANCED

Emitted when curriculum weights are updated.

| Field              | Type               | Required | Description              |
| ------------------ | ------------------ | -------- | ------------------------ |
| `weights`          | `dict[str, float]` | Yes      | New config weights       |
| `trigger`          | `str`              | No       | What triggered rebalance |
| `previous_weights` | `dict[str, float]` | No       | Previous weights         |

**Example:**

```python
{
    "weights": {
        "hex8_2p": 0.25,
        "hex8_4p": 0.15,
        "square8_2p": 0.30,
        "square8_4p": 0.30,
    },
    "trigger": "elo_velocity_update",
    "previous_weights": {
        "hex8_2p": 0.20,
        "hex8_4p": 0.20,
        "square8_2p": 0.30,
        "square8_4p": 0.30,
    },
}
```

---

### ELO_UPDATED

Emitted when Elo ratings are recalculated.

| Field          | Type    | Required | Description            |
| -------------- | ------- | -------- | ---------------------- |
| `config_key`   | `str`   | Yes      | Config identifier      |
| `model_id`     | `str`   | Yes      | Model identifier       |
| `new_elo`      | `float` | Yes      | New Elo rating         |
| `previous_elo` | `float` | No       | Previous rating        |
| `elo_delta`    | `float` | No       | Rating change          |
| `games_played` | `int`   | No       | Games in rating period |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "model_id": "canonical_hex8_2p",
    "new_elo": 1675.0,
    "previous_elo": 1650.0,
    "elo_delta": 25.0,
    "games_played": 50,
}
```

---

## Health & Monitoring Events

### HEALTH_CHECK_FAILED

Emitted when a node health check fails.

| Field                  | Type  | Required | Description                   |
| ---------------------- | ----- | -------- | ----------------------------- |
| `node_id`              | `str` | Yes      | Node identifier               |
| `reason`               | `str` | Yes      | Failure reason                |
| `consecutive_failures` | `int` | No       | Failure count                 |
| `last_success`         | `str` | No       | ISO timestamp of last success |

**Example:**

```python
{
    "node_id": "vast-4090-1",
    "reason": "connection_timeout",
    "consecutive_failures": 3,
    "last_success": "2025-12-30T10:00:00Z",
}
```

---

### NODE_RECOVERED

Emitted when a previously unhealthy node recovers.

| Field              | Type    | Required | Description             |
| ------------------ | ------- | -------- | ----------------------- |
| `node_id`          | `str`   | Yes      | Node identifier         |
| `downtime_seconds` | `float` | No       | Duration of outage      |
| `recovery_action`  | `str`   | No       | What triggered recovery |

**Example:**

```python
{
    "node_id": "vast-4090-1",
    "downtime_seconds": 300.0,
    "recovery_action": "automatic_restart",
}
```

---

### MEMORY_PRESSURE

Emitted when GPU/CPU memory reaches critical levels.

| Field                | Type    | Required | Description              |
| -------------------- | ------- | -------- | ------------------------ |
| `node_id`            | `str`   | Yes      | Node identifier          |
| `memory_type`        | `str`   | Yes      | `"gpu"` or `"cpu"`       |
| `usage_percent`      | `float` | Yes      | Current usage (0-100)    |
| `threshold`          | `float` | Yes      | Threshold that triggered |
| `recommended_action` | `str`   | No       | Suggested action         |

**Example:**

```python
{
    "node_id": "runpod-h100",
    "memory_type": "gpu",
    "usage_percent": 87.5,
    "threshold": 85.0,
    "recommended_action": "pause_spawning",
}
```

---

## Regression Events

### REGRESSION_DETECTED

Emitted when model performance regression is detected.

| Field          | Type    | Required | Description                                       |
| -------------- | ------- | -------- | ------------------------------------------------- |
| `config_key`   | `str`   | Yes      | Config identifier                                 |
| `model_path`   | `str`   | Yes      | Path to regressing model                          |
| `severity`     | `str`   | Yes      | `"minor"`, `"moderate"`, `"severe"`, `"critical"` |
| `elo_drop`     | `float` | Yes      | Elo decrease                                      |
| `baseline_elo` | `float` | No       | Expected Elo                                      |
| `current_elo`  | `float` | No       | Actual Elo                                        |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "model_path": "/models/candidate_hex8_2p.pth",
    "severity": "moderate",
    "elo_drop": -45.0,
    "baseline_elo": 1650.0,
    "current_elo": 1605.0,
}
```

---

## Work Queue Events

### WORK_QUEUED

Emitted when work is added to the queue.

| Field                | Type    | Required | Description                                |
| -------------------- | ------- | -------- | ------------------------------------------ |
| `work_id`            | `str`   | Yes      | Work item identifier                       |
| `work_type`          | `str`   | Yes      | `"selfplay"`, `"training"`, `"evaluation"` |
| `config_key`         | `str`   | No       | Associated config                          |
| `priority`           | `int`   | No       | Queue priority (higher = more urgent)      |
| `estimated_duration` | `float` | No       | Estimated seconds                          |

**Example:**

```python
{
    "work_id": "selfplay-hex8-2p-1704100200",
    "work_type": "selfplay",
    "config_key": "hex8_2p",
    "priority": 100,
    "estimated_duration": 3600.0,
}
```

---

### WORK_COMPLETED

Emitted when work finishes successfully.

| Field              | Type    | Required | Description           |
| ------------------ | ------- | -------- | --------------------- |
| `work_id`          | `str`   | Yes      | Work item identifier  |
| `work_type`        | `str`   | Yes      | Work type             |
| `node_id`          | `str`   | No       | Node that executed    |
| `duration_seconds` | `float` | No       | Actual duration       |
| `result`           | `dict`  | No       | Work-specific results |

**Example:**

```python
{
    "work_id": "selfplay-hex8-2p-1704100200",
    "work_type": "selfplay",
    "node_id": "vast-4090-1",
    "duration_seconds": 3200.0,
    "result": {"games_completed": 500},
}
```

---

## Consolidation Events

### NPZ_COMBINATION_COMPLETE

Emitted when NPZ files are combined.

| Field              | Type   | Required | Description                  |
| ------------------ | ------ | -------- | ---------------------------- |
| `config_key`       | `str`  | Yes      | Config identifier            |
| `output_path`      | `str`  | Yes      | Path to combined NPZ         |
| `source_files`     | `int`  | No       | Number of input files        |
| `total_samples`    | `int`  | No       | Combined sample count        |
| `quality_weighted` | `bool` | No       | If quality weighting applied |

**Example:**

```python
{
    "config_key": "hex8_2p",
    "output_path": "/data/training/hex8_2p_combined.npz",
    "source_files": 5,
    "total_samples": 250000,
    "quality_weighted": true,
}
```

---

## Common Fields (Present in Most Events)

| Field       | Type  | Description                                 |
| ----------- | ----- | ------------------------------------------- |
| `timestamp` | `str` | ISO 8601 timestamp (auto-added by EventBus) |
| `source`    | `str` | Component that emitted the event            |
| `trace_id`  | `str` | Distributed tracing ID (if enabled)         |

---

## Field Name Aliases

Some fields have aliases for backward compatibility:

| Canonical         | Aliases                   |
| ----------------- | ------------------------- |
| `config_key`      | `config`                  |
| `model_path`      | `checkpoint_path`, `path` |
| `final_loss`      | `loss`                    |
| `samples_trained` | `samples`                 |
| `games_played`    | `games`                   |

The extraction utilities in `event_utils.py` and `event_handler_utils.py` handle these aliases automatically.

---

## See Also

- `docs/EVENT_CATALOG.md` - Event emitters and subscribers
- `docs/EVENT_SYSTEM_REFERENCE.md` - Event system architecture
- `app/coordination/event_utils.py` - Payload extraction utilities
- `app/coordination/event_handler_utils.py` - Handler utilities
- `app/distributed/data_events.py` - DataEventType enum
