# Coordination Training Package

Training orchestration and scheduling for the RingRift AI training pipeline.

## Overview

This package manages the training phase of the self-improvement loop:

- Job scheduling with priority queues
- ELO-based curriculum weighting
- Training slot coordination across cluster

## Public API

Use the package-level exports instead of the removed compatibility shims:

```python
from app.coordination.training import (
    TrainingCoordinator,
    PriorityJobScheduler,
    UnifiedScheduler,
)
```

### `TrainingCoordinator`

Coordinates cluster training jobs and training-slot lifecycle.

### `PriorityJobScheduler`

Ranks pending training work using curriculum weights and freshness signals.

### `UnifiedScheduler`

Provides the higher-level scheduling facade used by coordination services.

**Priority factors**:

1. Curriculum weights (from `CurriculumFeedback`)
2. Time since last training
3. Data freshness (newer data = higher priority)
4. ELO velocity (configs with faster ELO growth get priority)

## Integration

### Event Subscriptions

The training coordination layer subscribes to:

- `NPZ_EXPORT_COMPLETE` - Trigger training after export
- `TRAINING_BLOCKED_BY_QUALITY` - Handle quality gate blocks
- `CLUSTER_HEALTH_CHANGED` - Adjust for node availability

### Event Emissions

The canonical training modules emit:

- `TRAINING_STARTED` - Training job began
- `TRAINING_COMPLETE` - Training finished successfully
- `TRAINING_FAILED` - Training job failed

## Configuration

From `config/distributed_hosts.yaml`:

```yaml
training:
  default_epochs: 50
  batch_size: 512
  learning_rate: 0.001
  early_stopping_patience: 5
```

## See Also

- `../data_pipeline_orchestrator.py` - Pipeline stage coordination
- `../training_coordinator.py` - Cluster-wide training coordination
- `../job_scheduler.py` - Priority scheduling implementation
- `../unified_scheduler.py` - Scheduler facade
- `../../training/train.py` - Actual training implementation
