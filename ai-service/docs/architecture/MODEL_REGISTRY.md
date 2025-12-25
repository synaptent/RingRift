# Model Registry Architecture

Documentation for the model registry system that tracks trained models and their lifecycle.

## Overview

The model registry provides:

- Versioned model storage
- Training metadata tracking
- Model promotion workflow
- Cluster-wide model sync

## Database Schema

Located in `data/models/registry.db`:

```sql
-- Model versions table
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    board_type TEXT NOT NULL,
    num_players INTEGER NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT DEFAULT 'candidate',  -- candidate, promoted, archived
    created_at TEXT NOT NULL,
    promoted_at TEXT,

    -- Training metadata
    training_epochs INTEGER,
    training_samples INTEGER,
    final_loss REAL,

    -- Evaluation metrics
    policy_accuracy REAL,
    value_accuracy REAL,
    vs_random_winrate REAL,
    vs_heuristic_winrate REAL,

    UNIQUE(board_type, num_players, version)
);

-- Training runs table
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',

    -- Configuration
    config_json TEXT,

    -- Results
    epochs_completed INTEGER,
    best_loss REAL,
    checkpoint_path TEXT
);

-- Promotion history
CREATE TABLE promotions (
    id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    promoted_at TEXT NOT NULL,
    promoted_by TEXT,
    reason TEXT,
    gauntlet_results_json TEXT
);
```

## Model Lifecycle

```
[Training] → [Candidate] → [Gauntlet] → [Promoted] → [Deployed]
                              ↓
                         [Archived]
```

### States

| State       | Description                         |
| ----------- | ----------------------------------- |
| `candidate` | Newly trained, awaiting evaluation  |
| `promoted`  | Passed gauntlet, active for serving |
| `archived`  | Superseded by newer model           |

## Usage

### Register a New Model

```python
from app.training.model_registry import ModelRegistry

registry = ModelRegistry()

model_id = registry.register_model(
    board_type="hex8",
    num_players=2,
    version="v3.1",
    path="models/hex8_2p/v3.1.pth",
    metadata={
        "training_epochs": 50,
        "training_samples": 100000,
        "final_loss": 0.45,
    }
)
```

### Record Evaluation Results

```python
registry.update_metrics(
    model_id,
    policy_accuracy=0.762,
    value_accuracy=0.85,
    vs_random_winrate=0.95,
    vs_heuristic_winrate=0.72,
)
```

### Promote Model

```python
registry.promote_model(
    board_type="hex8",
    num_players=2,
    version="v3.1",
    reason="Passed gauntlet with 72% vs heuristic",
)
```

### Get Active Model

```python
model = registry.get_promoted_model("hex8", 2)
print(f"Active model: {model.path}")
```

## Promotion Criteria

Default thresholds (configurable):

| Opponent  | Required Win Rate |
| --------- | ----------------- |
| RANDOM    | 85%               |
| HEURISTIC | 60%               |

## Cluster Sync

Models are synced across the cluster using rsync:

```bash
# Sync promoted model to all nodes
python scripts/auto_promote.py --sync-to-cluster \
    --model models/hex8_2p/best.pth \
    --board-type hex8 --num-players 2
```

## File Structure

```
models/
├── hex8_2p/
│   ├── best.pth          # Symlink to promoted
│   ├── v3.1.pth
│   ├── v3.0.pth
│   └── checkpoints/
│       ├── epoch_10.pth
│       └── epoch_20.pth
├── square8_2p/
│   └── ...
└── registry.db           # SQLite database
```

## API Reference

### ModelRegistry

```python
class ModelRegistry:
    def register_model(board_type, num_players, version, path, metadata) -> int
    def update_metrics(model_id, **metrics) -> None
    def promote_model(board_type, num_players, version, reason) -> None
    def get_promoted_model(board_type, num_players) -> Model | None
    def list_models(board_type, num_players) -> list[Model]
    def archive_model(model_id) -> None
```

### Model Dataclass

```python
@dataclass
class Model:
    id: int
    board_type: str
    num_players: int
    version: str
    path: str
    status: str
    created_at: str
    policy_accuracy: float | None
    value_accuracy: float | None
```

## See Also

- `app/training/model_registry.py` - Implementation
- `scripts/auto_promote.py` - Promotion script
- `app/training/checkpoint_unified.py` - Checkpoint management
