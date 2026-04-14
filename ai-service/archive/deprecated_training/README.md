# Deprecated Training Modules

This directory contains training modules that have been superseded by consolidated implementations.

## orchestrated_training.py

**Archived**: December 26, 2025
**Last Updated**: April 2026

**Reason**: Functionality consolidated into `app/training/unified_orchestrator.py`,
with the archived compatibility layer now living at
`archive/deprecated_training/orchestrated_training.py` and re-exported from
`app.training`.

**Superseded By**:

The manager lifecycle coordination functionality is now handled by:

- `app/training/unified_orchestrator.py` - UnifiedTrainingOrchestrator (step-level + manager coordination)
- `app/coordination/training_coordinator.py` - Cluster-wide training coordination
- `app/training/checkpoint_unified.py` - UnifiedCheckpointManager
- `app/training/rollback_manager.py` - RollbackManager with regression detection
- `app/training/promotion_controller.py` - PromotionController for model promotion

**Original Purpose**:

Manager LIFECYCLE orchestrator for training infrastructure that coordinated:

- Unified initialization and shutdown of training managers
- Coordinated checkpointing via UnifiedCheckpointManager
- Automatic rollback on regression via RollbackManager
- Promotion evaluation triggers via PromotionController
- Data coordination via DataCoordinator
- Elo rating updates via EloService
- Curriculum weight adjustments via CurriculumFeedback

**Migration**:

Compatibility import during migration:

```python
from app.training import TrainingOrchestrator, TrainingOrchestratorConfig

config = TrainingOrchestratorConfig()
orchestrator = TrainingOrchestrator(config)
await orchestrator.initialize()
```

The direct `app.training.orchestrated_training` module path no longer exists in
the active package tree. Use the `app.training` re-export only while migrating.

New code using `UnifiedTrainingOrchestrator`:

```python
from app.training.unified_orchestrator import (
    UnifiedTrainingOrchestrator,
    OrchestratorConfig,
)
orchestrator = UnifiedTrainingOrchestrator(model, config)
with orchestrator:
    for batch in orchestrator.get_dataloader():
        loss = orchestrator.train_step(batch)
```

See `app/training/ORCHESTRATOR_GUIDE.md` for complete migration instructions.

**Verification**:

The archived implementation lives in
`archive/deprecated_training/orchestrated_training.py`, and the supported
compatibility import is `from app.training import TrainingOrchestrator`.
New training code should use `UnifiedTrainingOrchestrator` directly.

**Deprecation Timeline**:

- December 2025: Deprecation warning added to module
- December 26, 2025: Module archived to `archive/deprecated_training/`
- Q2 2026: Planned removal (if no usage detected)
