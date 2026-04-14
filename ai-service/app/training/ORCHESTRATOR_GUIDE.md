# Training Orchestrator Guide

**Created**: December 2025
**Purpose**: Clarify orchestrator roles and migration path

This guide covers in-process training orchestration. For supported trainer
canaries, operators should deploy `ai-service/scripts/minimal_alphazero_loop.py`
via `ai-service/scripts/deploy_minimal_loops.sh` and monitor `progress.json`
plus `metrics.jsonl`; that rollout path is separate from the orchestration
classes described below.

## Current Architecture (5 Orchestrators)

| Class                         | File                                                         | Lines | Role                              |
| ----------------------------- | ------------------------------------------------------------ | ----- | --------------------------------- |
| `UnifiedTrainingOrchestrator` | unified_orchestrator.py                                      | 2,216 | **PRIMARY** - Step-level training |
| `TrainingCoordinator`         | `../coordination/training_coordinator.py`                    | 1,862 | Cluster-wide job coordination     |
| `TrainingLifecycleManager`    | lifecycle_integration.py                                     | 548   | Service lifecycle management      |
| `TrainingOrchestrator`        | `../../archive/deprecated_training/orchestrated_training.py` | 386   | _ARCHIVED_ compatibility surface  |
| `IntegratedTrainingManager`   | integrated_enhancements.py                                   | 1,350 | _DEPRECATED_ - Enhancements       |

## Target Architecture (3 Orchestrators)

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedTrainingOrchestrator                  │
│  (Primary training orchestrator - step-level + enhancements)    │
│                                                                 │
│  Features:                                                      │
│  - Forward/backward pass execution                              │
│  - Hot buffer (PER)                                            │
│  - Integrated enhancements (auxiliary, gradient surgery)        │
│  - Checkpoint management                                        │
│  - Background evaluation                                        │
│  - Adaptive controllers (LR, batch size)                        │
│  - Online learning (EBMO)                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ coordinates with
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TrainingCoordinator                        │
│  (Cluster-wide job coordination via NFS-backed SQLite)          │
│                                                                 │
│  Features:                                                      │
│  - Job registration and status tracking                         │
│  - Cross-node coordination                                      │
│  - Heartbeat monitoring                                         │
│  - Training slot allocation                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ manages lifecycle of
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TrainingLifecycleManager                     │
│  (Service startup/shutdown coordination)                        │
│                                                                 │
│  Features:                                                      │
│  - Service registration                                         │
│  - Dependency-ordered startup                                   │
│  - Health checking                                              │
│  - Graceful shutdown                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Migration Guide

### For `TrainingOrchestrator` users (archived compatibility layer)

`TrainingOrchestrator` now comes from the archived compatibility module
`archive.deprecated_training.orchestrated_training` and is re-exported from
`app.training` for migrations.

Replace with:

```python
# Old
from app.training import TrainingOrchestrator
orchestrator = TrainingOrchestrator(config)
await orchestrator.initialize()

# New
from app.training.unified_orchestrator import UnifiedTrainingOrchestrator, OrchestratorConfig
config = OrchestratorConfig(
    board_type="square8",
    num_players=2,
    # Manager coordination is now built-in
)
orchestrator = UnifiedTrainingOrchestrator(model, config)
# No separate initialize() needed - uses context manager
with orchestrator:
    for epoch in range(epochs):
        for batch in orchestrator.get_dataloader():
            loss = orchestrator.train_step(batch)
```

### For `IntegratedTrainingManager` users (integrated_enhancements.py)

**DEPRECATED**: Features are now integrated into `UnifiedTrainingOrchestrator`.

Replace with:

```python
# Old
from app.training.integrated_enhancements import IntegratedTrainingManager
manager = IntegratedTrainingManager(config, model)
manager.initialize_all()
loss = manager.apply_step_enhancements(batch_loss, step)

# New
from app.training.unified_orchestrator import UnifiedTrainingOrchestrator, OrchestratorConfig
config = OrchestratorConfig(
    # Enhancements are now config flags
    enable_auxiliary_tasks=True,
    enable_gradient_surgery=True,
    enable_curriculum=True,
)
orchestrator = UnifiedTrainingOrchestrator(model, config)
# Enhancements applied automatically in train_step()
loss = orchestrator.train_step(batch)
```

## When to Use Each Class

| Use Case                       | Class                                    |
| ------------------------------ | ---------------------------------------- |
| Training a model               | `UnifiedTrainingOrchestrator`            |
| Coordinating cluster-wide jobs | `TrainingCoordinator`                    |
| Managing service lifecycle     | `TrainingLifecycleManager`               |
| P2P cluster operations         | `P2PTrainingBridge` (p2p_integration.py) |

## Feature Integration Status

| Feature               | From                      | Into                        | Status  |
| --------------------- | ------------------------- | --------------------------- | ------- |
| Checkpoint management | TrainingOrchestrator      | UnifiedTrainingOrchestrator | ✅ Done |
| Rollback management   | TrainingOrchestrator      | UnifiedTrainingOrchestrator | ✅ Done |
| Auxiliary tasks       | IntegratedTrainingManager | UnifiedTrainingOrchestrator | ✅ Done |
| Gradient surgery      | IntegratedTrainingManager | UnifiedTrainingOrchestrator | ✅ Done |
| Curriculum controller | IntegratedTrainingManager | UnifiedTrainingOrchestrator | ✅ Done |
| Background evaluation | IntegratedTrainingManager | UnifiedTrainingOrchestrator | ✅ Done |
| ELO sampling          | IntegratedTrainingManager | UnifiedTrainingOrchestrator | ✅ Done |

## Archived Compatibility Surface

- `archive/deprecated_training/orchestrated_training.py` contains the archived
  `TrainingOrchestrator` compatibility layer.
- `app/training/integrated_enhancements.py` remains present for legacy imports,
  but new code should migrate to `UnifiedTrainingOrchestrator`.
