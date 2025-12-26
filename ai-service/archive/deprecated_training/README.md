# Deprecated Training Modules

This directory contains training modules that have been superseded by consolidated implementations.

## orchestrated_training.py

**Archived**: December 26, 2025

**Reason**: Functionality consolidated into `app/training/unified_orchestrator.py`

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

Old code using `TrainingOrchestrator`:

```python
from app.training.orchestrated_training import TrainingOrchestrator
orchestrator = TrainingOrchestrator(config)
await orchestrator.initialize()
```

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

Grep analysis confirmed only 4 references (all documentation and self-references):

```bash
grep -r "from app.training.orchestrated_training import" --include="*.py" .
# Results:
# - app/training/__init__.py: Documentation reference
# - archive/deprecated_training/orchestrated_training.py: Self-reference
# - docs/MIGRATION_GUIDE.md: Migration documentation
# - app/training/ORCHESTRATOR_GUIDE.md: Migration documentation
```

No active code imports this module - safe to archive.

**Deprecation Timeline**:

- December 2025: Deprecation warning added to module
- December 26, 2025: Module archived to `archive/deprecated_training/`
- Q2 2026: Planned removal (if no usage detected)
