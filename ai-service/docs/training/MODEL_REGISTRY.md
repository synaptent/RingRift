# Model Registry

The Model Registry provides comprehensive model lifecycle management for RingRift AI models, including neural network models and CMA-ES optimized heuristic weights.

## Overview

The registry tracks models through their complete lifecycle:

- **Development**: Initial training/optimization
- **Staging**: Under evaluation
- **Production**: Deployed for inference
- **Archived**: Retired models (available for rollback)
- **Rejected**: Failed evaluation

## Components

### Core Registry (`app/training/model_registry.py`)

The main registry handles:

- Model registration with versioning
- Stage transitions with validation
- Metrics tracking (Elo, win rate, games played)
- Training configuration storage
- Model comparison and tagging

```python
from app.training.model_registry import ModelRegistry, ModelStage, ModelType

registry = ModelRegistry(Path("data/model_registry"))

# Register a new model
model_id, version = registry.register_model(
    name="square8_2p",
    model_path=Path("models/latest.pt"),
    model_type=ModelType.POLICY_VALUE,
    metrics=ModelMetrics(elo=1500, games_played=100),
)

# Promote through stages
registry.promote(model_id, version, ModelStage.STAGING, reason="Passed dev tests")
registry.promote(model_id, version, ModelStage.PRODUCTION, reason="Passed evaluation")
```

### CLI Tool (`scripts/model_registry_cli.py`)

Command-line interface for registry management:

```bash
# List all models
python scripts/model_registry_cli.py list

# Show model details
python scripts/model_registry_cli.py show square8_2p:1

# Promote a model
python scripts/model_registry_cli.py promote square8_2p -v 3 --to staging

# Rollback to previous version
python scripts/model_registry_cli.py rollback square8_2p

# Show statistics
python scripts/model_registry_cli.py stats
```

### CMA-ES Integration (`app/training/cmaes_registry_integration.py`)

Integrates CMA-ES hyperparameter optimization results with the registry:

```python
from app.training.cmaes_registry_integration import register_cmaes_result

# After CMA-ES optimization completes
model_id, version = register_cmaes_result(
    weights_path=Path("logs/cmaes/optimized_weights.json"),
    board_type="square8",
    num_players=2,
    fitness=0.85,
    generation=50,
    cmaes_config={
        "population_size": 20,
        "sigma": 0.5,
        "games_per_eval": 10,
    },
    auto_promote=True,  # Auto-promote if fitness improved
)
```

### Rollback Manager (`app/training/rollback_manager.py`)

Handles automatic and manual rollbacks:

```python
from app.training.rollback_manager import RollbackManager

manager = RollbackManager(registry)

# Set performance baseline after promotion
manager.set_baseline("square8_2p", {"elo": 1500, "games_played": 100})

# Check if rollback is needed
should_rollback, reason = manager.should_rollback("square8_2p")

# Execute rollback
result = manager.rollback_model(
    "square8_2p",
    reason="Performance regression detected",
)
```

### Backup/Recovery (`app/training/registry_backup.py`)

Automated backup and recovery:

```python
from app.training.registry_backup import RegistryBackupManager

backup_mgr = RegistryBackupManager(registry_path)

# Create backup
backup_path = backup_mgr.create_backup(reason="Before sync")

# List backups
backups = backup_mgr.list_backups()

# Restore from backup
result = backup_mgr.restore_backup(backup_id="20240101_120000")
```

### Distributed Sync (`app/training/registry_sync_manager.py`)

Cluster-wide registry synchronization:

```python
from app.training.registry_sync_manager import RegistrySyncManager

sync_manager = RegistrySyncManager(registry_path=Path("data/model_registry.db"))
await sync_manager.initialize()

# Sync with cluster
result = await sync_manager.sync_with_cluster()
print(f"Synced {result['nodes_synced']} nodes, {result['models_merged']} models merged")
```

## Prometheus Metrics

The registry exposes the following metrics:

| Metric                                        | Type    | Description            |
| --------------------------------------------- | ------- | ---------------------- |
| `ringrift_model_registry_count`               | Gauge   | Models by stage        |
| `ringrift_model_registry_registrations_total` | Counter | Total registrations    |
| `ringrift_model_registry_promotions_total`    | Counter | Stage promotions       |
| `ringrift_cmaes_runs_total`                   | Counter | CMA-ES runs registered |
| `ringrift_cmaes_best_fitness`                 | Gauge   | Best fitness by config |
| `ringrift_model_rollbacks_total`              | Counter | Rollback events        |

## Grafana Dashboard

Import `deploy/grafana/model-registry-dashboard.json` for visualizations:

- Models by stage (pie chart)
- Promotion activity timeline
- CMA-ES fitness trends
- Parity validation status
- Rollback history

## Model Types

| Type           | Description               |
| -------------- | ------------------------- |
| `policy_value` | Main neural network model |
| `heuristic`    | CMA-ES optimized weights  |
| `ensemble`     | Ensemble of models        |
| `compressed`   | Quantized/pruned model    |
| `experimental` | Research models           |

## Stage Transitions

Valid transitions:

```
DEVELOPMENT -> STAGING, ARCHIVED, REJECTED
STAGING -> PRODUCTION, DEVELOPMENT, ARCHIVED, REJECTED
PRODUCTION -> ARCHIVED, STAGING
ARCHIVED -> DEVELOPMENT
REJECTED -> DEVELOPMENT
```

## Health Monitoring

The following components report health status:

- `registry_sync`: Cluster synchronization
- `model_registry`: Registry operations
- `parity_validation`: Data quality validation

Check health via the unified loop's health endpoint or Prometheus metrics.

## Backup Strategy

Recommended backup configuration:

- Auto-backup interval: 24 hours
- Maximum backups retained: 10
- Pre-sync backup: Always enabled

Backups are stored in `data/registry_backups/` with:

- SQLite database copy
- JSON metadata (hash, counts, timestamp)

## Troubleshooting

### Model not found

```bash
python scripts/model_registry_cli.py list --group
```

### Promotion failed

Check allowed transitions above. Use CLI to inspect current stage:

```bash
python scripts/model_registry_cli.py show model_id:version
```

### Registry sync failed

Check circuit breaker status:

```python
sync_manager.get_sync_status()['circuit_breakers']
```

### Rollback not available

Verify archived versions exist:

```bash
python scripts/model_registry_cli.py list --stage archived
```
