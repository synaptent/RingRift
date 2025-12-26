# Deprecation Migration Guide

**Last Updated**: December 2025
**Document Purpose**: Comprehensive migration guide for deprecated modules in the ai-service codebase.

---

## Table of Contents

1. [Training Orchestrators](#1-training-orchestrators)
   - [orchestrated_training.py to unified_orchestrator.py](#11-orchestrated_trainingpy--unified_orchestratorpy)
   - [integrated_enhancements.py to unified_orchestrator.py](#12-integrated_enhancementspy--unified_orchestratorpy)
2. [Legacy Model Formats](#2-legacy-model-formats)
   - [v1/v1_mps to v2 Migration](#21-v1v1_mps-to-v2-migration)
   - [Converting Old Checkpoints](#22-converting-old-checkpoints)
3. [Database Naming Changes](#3-database-naming-changes)
   - [Old Patterns to Canonical Naming](#31-old-patterns-to-canonical-naming)
   - [Migration Steps](#32-migration-steps)
4. [Event System Consolidation](#4-event-system-consolidation)
   - [stage_events.py Migration](#41-stage_eventspy-migration)
   - [cross_process_events.py Migration](#42-cross_process_eventspy-migration)
   - [Unified Subscription Model](#43-unified-subscription-model)

---

## 1. Training Orchestrators

### 1.1 orchestrated_training.py -> unified_orchestrator.py

**Status**: Deprecated December 2025, scheduled for removal Q2 2026

**What was deprecated and why**:

- `TrainingOrchestrator` from `orchestrated_training.py` was a manager lifecycle coordinator
- Its functionality (checkpoint management, rollback, promotion, curriculum) has been integrated into `UnifiedTrainingOrchestrator`
- Consolidation reduces maintenance burden and provides a single entry point for training

**Deprecation Warning**:

```python
# This warning is emitted on import:
DeprecationWarning: orchestrated_training.py is deprecated.
Use UnifiedTrainingOrchestrator from unified_orchestrator.py instead.
See app/training/ORCHESTRATOR_GUIDE.md for migration instructions.
```

#### Code Migration Examples

**Before (Deprecated)**:

```python
from app.training.orchestrated_training import (
    TrainingOrchestrator,
    TrainingOrchestratorConfig,
)

config = TrainingOrchestratorConfig(
    checkpoint_dir="checkpoints",
    checkpoint_interval_steps=1000,
    enable_rollback=True,
    enable_promotion=True,
)

orchestrator = TrainingOrchestrator(config)
await orchestrator.initialize()

async with orchestrator.training_context():
    # Training loop
    for batch in dataloader:
        loss = train_step(batch)
        orchestrator.record_step(step)

        if orchestrator.should_checkpoint():
            orchestrator.save_checkpoint(model_state=model.state_dict())

await orchestrator.shutdown()
```

**After (New API)**:

```python
from app.training.unified_orchestrator import (
    UnifiedTrainingOrchestrator,
    OrchestratorConfig,
)

config = OrchestratorConfig(
    board_type="square8",
    num_players=2,
    checkpoint_dir="checkpoints",
    checkpoint_interval=1000,
    enable_rollback=True,  # Rollback now built-in
    auto_rollback=True,    # Auto-rollback on CRITICAL regressions
)

orchestrator = UnifiedTrainingOrchestrator(model, config)

# Context manager handles initialization and cleanup
with orchestrator:
    for epoch in range(epochs):
        for batch in orchestrator.get_dataloader():
            # train_step handles checkpointing automatically
            loss = orchestrator.train_step(batch)

        # Report epoch completion for improvement optimizer
        orchestrator.complete_epoch(val_loss=val_loss)
```

#### Feature Mapping

| Old (TrainingOrchestrator)         | New (UnifiedTrainingOrchestrator)      |
| ---------------------------------- | -------------------------------------- |
| `orchestrator.initialize()`        | `with orchestrator:` (context manager) |
| `orchestrator.record_step(step)`   | Automatic in `train_step()`            |
| `orchestrator.should_checkpoint()` | Automatic in `train_step()`            |
| `orchestrator.save_checkpoint()`   | Automatic via `checkpoint_interval`    |
| `orchestrator.training_context()`  | `with orchestrator:`                   |
| `orchestrator.shutdown()`          | Automatic on context exit              |

---

### 1.2 integrated_enhancements.py -> unified_orchestrator.py

**Status**: Deprecated December 2025, scheduled for removal Q2 2026

**What was deprecated and why**:

- `IntegratedTrainingManager` was a separate wrapper for training enhancements
- All enhancement features (auxiliary tasks, gradient surgery, curriculum, etc.) are now integrated directly into `UnifiedTrainingOrchestrator`
- Individual enhancement modules remain available for direct use if needed

**Deprecation Warning**:

```python
# This warning is emitted on import:
DeprecationWarning: IntegratedTrainingManager from integrated_enhancements.py is deprecated.
Use UnifiedTrainingOrchestrator from unified_orchestrator.py instead.
Individual enhancement modules are still available for direct use.
See app/training/ORCHESTRATOR_GUIDE.md for migration instructions.
```

#### Code Migration Examples

**Before (Deprecated)**:

```python
from app.training.integrated_enhancements import (
    IntegratedTrainingManager,
    IntegratedEnhancementsConfig,
)

config = IntegratedEnhancementsConfig(
    auxiliary_tasks_enabled=True,
    gradient_surgery_enabled=True,
    batch_scheduling_enabled=True,
    curriculum_enabled=True,
    augmentation_enabled=True,
    reanalysis_enabled=True,
)

manager = IntegratedTrainingManager(config, model, board_type="square8")
manager.initialize_all()

# Manual enhancement application
for batch in dataloader:
    # Apply augmentation
    features, policy = manager.augment_batch_dense(batch[0], batch[1])

    # Compute auxiliary loss
    aux_loss, aux_metrics = manager.compute_auxiliary_loss(features, targets)

    # Apply gradient surgery
    combined_loss = manager.apply_gradient_surgery(model, losses)

    manager.update_step()

    # Check for reanalysis trigger
    if manager.should_reanalyze():
        manager.process_reanalysis(npz_path)
```

**After (New API)**:

```python
from app.training.unified_orchestrator import (
    UnifiedTrainingOrchestrator,
    OrchestratorConfig,
)

config = OrchestratorConfig(
    board_type="square8",
    num_players=2,
    # All enhancements are now config flags
    enable_enhancements=True,
    enable_auxiliary_tasks=True,
    enable_gradient_surgery=True,
    enable_batch_scheduling=True,
    enable_curriculum=True,
    enable_augmentation=True,
    enable_reanalysis=True,
    reanalysis_blend_ratio=0.7,
)

orchestrator = UnifiedTrainingOrchestrator(model, config)

with orchestrator:
    for batch in dataloader:
        # All enhancements applied automatically in train_step()
        metrics = orchestrator.train_step(batch)

        # Access current Elo from background evaluator
        current_elo = orchestrator._background_eval.get_current_elo()

        # Access curriculum metrics
        curriculum_weights = orchestrator.get_curriculum_weights()
```

#### Enhancement Feature Mapping

| Enhancement Feature    | Old Method                         | New Behavior                                      |
| ---------------------- | ---------------------------------- | ------------------------------------------------- |
| Data augmentation      | `manager.augment_batch_dense()`    | Automatic in `train_step()`                       |
| Auxiliary tasks        | `manager.compute_auxiliary_loss()` | Automatic in `train_step()`                       |
| Gradient surgery       | `manager.apply_gradient_surgery()` | Automatic when `enable_gradient_surgery=True`     |
| Batch scheduling       | `manager.get_batch_size()`         | Automatic via config                              |
| Background evaluation  | `manager.should_early_stop()`      | `orchestrator.should_stop()`                      |
| Current Elo            | `manager.get_current_elo()`        | Via `_background_eval.get_current_elo()`          |
| Reanalysis             | `manager.should_reanalyze()`       | Automatic trigger based on config                 |
| Knowledge distillation | `manager.should_distill()`         | Automatic based on `distillation_interval_epochs` |

---

## 2. Legacy Model Formats

### 2.1 v1/v1_mps to v2 Migration

**Status**: v1/v1_mps models deprecated, scheduled for removal Q1 2026

**Background**:

- v1 models used the original architecture with 40 input channels
- v1_mps was an MPS-optimized variant for Apple Silicon
- v2 models use the canonical architecture with encoder/decoder separation
- v2 provides better separation of concerns and cleaner code

**Key Differences**:

| Aspect            | v1/v1_mps               | v2 (Current)                    |
| ----------------- | ----------------------- | ------------------------------- |
| Input channels    | 40 (square) or varying  | Board-specific (56 for square8) |
| Architecture file | `_neural_net_legacy.py` | `neural_net/network.py`         |
| Encoding          | Monolithic              | Modular (`square_encoding.py`)  |
| Policy decoding   | Inline in network       | Separate decoder classes        |

### 2.2 Converting Old Checkpoints

**Migration Script**: `scripts/migrate_legacy_models.py`

#### Usage

```bash
# Dry run - preview what would be migrated
python scripts/migrate_legacy_models.py --dry-run

# Execute migration (adds models to registry)
python scripts/migrate_legacy_models.py --execute

# Migrate a specific model
python scripts/migrate_legacy_models.py --model models/my_old_model.pt --execute
```

#### Manual Checkpoint Migration

For checkpoints that need manual conversion:

```python
from app.utils.torch_utils import safe_load_checkpoint

# Load legacy checkpoint
checkpoint = safe_load_checkpoint("models/legacy_model.pt")

# Get state dict (handle different storage formats)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Infer model configuration from state dict
# Check input channels from first conv layer
for key in state_dict.keys():
    if "conv1.weight" in key or "stem.0.weight" in key:
        in_channels = state_dict[key].shape[1]
        if in_channels == 40:
            model_version = "v1"
        elif in_channels == 56:
            model_version = "v2"
        break

# Create new model with correct architecture
from app.ai.neural_net import RingRiftNet

model = RingRiftNet(
    board_type="square8",
    num_players=2,
    in_channels=56,  # v2 channels
)

# Load weights (with partial loading if architecture changed)
model.load_state_dict(state_dict, strict=False)

# Save in new format
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "board_type": "square8",
        "num_players": 2,
        "model_version": "v2",
    }
}, "models/migrated_model.pth")
```

#### Transfer Learning (2p to 4p)

When migrating a 2-player model to 4-player:

```bash
# Use the transfer script to resize value head
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/transfer_sq8_4p_init.pth \
  --board-type square8

# Then train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/transfer_sq8_4p_init.pth \
  --data-path data/training/sq8_4p.npz
```

---

## 3. Database Naming Changes

### 3.1 Old Patterns to Canonical Naming

**Status**: Non-canonical databases deprecated for training use

**Old Naming Patterns** (Deprecated for training):

- `selfplay_*.db` - Legacy selfplay databases
- `square8_2p.db` - Implicit naming without prefix
- `hex*.db` - Inconsistent hex naming

**Canonical Naming Pattern** (Required):

- `canonical_{board_type}_{num_players}p.db`
- Examples:
  - `canonical_square8_2p.db`
  - `canonical_hex8_4p.db`
  - `canonical_hexagonal_2p.db`

### 3.2 Migration Steps

#### Step 1: Identify Legacy Databases

```python
from app.utils.game_discovery import GameDiscovery

discovery = GameDiscovery()
for db in discovery.find_all_databases():
    if not db.path.name.startswith("canonical_"):
        print(f"Legacy: {db.path} ({db.game_count} games)")
```

#### Step 2: Validate Parity

Before migrating, ensure games pass parity validation:

```bash
# Check parity for a database
python scripts/check_ts_python_replay_parity.py \
  --db data/games/selfplay_square8_2p.db \
  --sample 100
```

#### Step 3: Create Canonical Copy

```python
import shutil
from pathlib import Path

# Copy with canonical naming
old_path = Path("data/games/selfplay_square8_2p.db")
new_path = Path("data/games/canonical_square8_2p.db")

# Create snapshot to avoid WAL issues
import sqlite3
src = sqlite3.connect(str(old_path))
dst = sqlite3.connect(str(new_path))
src.backup(dst)
src.close()
dst.close()
```

#### Step 4: Verify and Tag

```bash
# Tag games with parity status
python scripts/tag_games_parity_status.py data/games/canonical_square8_2p.db

# Validate the canonical database
python -m app.training.data_quality --db data/games/canonical_square8_2p.db
```

#### Database Naming Reference

| Board Type | 2-Player                    | 3-Player                    | 4-Player                    |
| ---------- | --------------------------- | --------------------------- | --------------------------- |
| square8    | `canonical_square8_2p.db`   | `canonical_square8_3p.db`   | `canonical_square8_4p.db`   |
| square19   | `canonical_square19_2p.db`  | `canonical_square19_3p.db`  | `canonical_square19_4p.db`  |
| hex8       | `canonical_hex8_2p.db`      | `canonical_hex8_3p.db`      | `canonical_hex8_4p.db`      |
| hexagonal  | `canonical_hexagonal_2p.db` | `canonical_hexagonal_3p.db` | `canonical_hexagonal_4p.db` |

---

## 4. Event System Consolidation

### 4.1 stage_events.py Migration

**Status**: Deprecated December 2025, scheduled for removal Q2 2026

**What changed**:

- `StageEventBus` from `stage_events.py` has been superseded by `UnifiedEventRouter` in `event_router.py`
- The unified router automatically routes events to all event buses
- Old module remains functional for backwards compatibility

**Deprecation Warning**:

```python
# This warning is emitted when calling get_event_bus():
DeprecationWarning: get_event_bus() from stage_events is deprecated.
Use get_router() from app.coordination.event_router instead.
```

#### Migration Example

**Before (Deprecated)**:

```python
from app.coordination.stage_events import (
    StageEventBus,
    StageEvent,
    StageCompletionResult,
    get_event_bus,
)

bus = get_event_bus()

# Subscribe to events
async def on_selfplay_done(result):
    if result.success:
        await start_data_sync()

bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_done)

# Emit events
await bus.emit(StageCompletionResult(
    event=StageEvent.SELFPLAY_COMPLETE,
    success=True,
    iteration=1,
    timestamp=datetime.now().isoformat(),
    games_generated=500
))
```

**After (New API)**:

```python
from app.coordination.event_router import (
    get_router,
    publish,
    subscribe,
    StageEvent,  # Re-exported for compatibility
    StageCompletionResult,  # Re-exported for compatibility
)

router = get_router()

# Subscribe to events (receives from all buses)
def on_selfplay_done(event):
    if event.payload.get("success"):
        start_data_sync_sync()  # Use sync version or fire_and_forget

subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_done)

# Publish events (routes to all buses automatically)
await publish(
    event_type=StageEvent.SELFPLAY_COMPLETE,
    payload={
        "success": True,
        "iteration": 1,
        "games_generated": 500,
    },
    source="selfplay",
)
```

### 4.2 cross_process_events.py Migration

**Status**: Deprecated December 2025, scheduled for removal Q2 2026

**What changed**:

- Direct use of `cross_process_events` module is deprecated
- The unified event router handles cross-process event routing automatically
- Use `route_to_cross_process=True` (default) when publishing

**Deprecation Warning**:

```python
# This warning is emitted on direct import:
DeprecationWarning: cross_process_events is deprecated. Import from event_router instead:
  from app.coordination.event_router import (
      CrossProcessEvent, CrossProcessEventQueue, bridge_to_cross_process, ...
  )
This module will be removed in Q2 2026.
```

#### Migration Example

**Before (Deprecated)**:

```python
from app.coordination.cross_process_events import (
    CrossProcessEventQueue,
    publish_event,
    poll_events,
    subscribe_process,
)

# Publish cross-process event
publish_event(
    event_type="MODEL_PROMOTED",
    payload={"model_id": "abc123", "elo": 1850},
    source="improvement_daemon"
)

# Subscribe and poll
subscriber_id = subscribe_process("pipeline_orchestrator")
events = poll_events(subscriber_id, event_types=["MODEL_PROMOTED"])
```

**After (New API)**:

```python
from app.coordination.event_router import (
    get_router,
    publish,
    subscribe,
    # Re-exported for direct access if needed
    CrossProcessEvent,
    cp_poll_events,
    subscribe_process,
)

# Publish (automatically routes to cross-process queue)
await publish(
    event_type="MODEL_PROMOTED",
    payload={"model_id": "abc123", "elo": 1850},
    source="improvement_daemon",
)

# Subscribe via unified router
def on_model_promoted(event):
    handle_promotion(event.payload)

subscribe("MODEL_PROMOTED", on_model_promoted)
```

### 4.3 Unified Subscription Model

The new `UnifiedEventRouter` provides a single subscription API that receives events from all sources:

```python
from app.coordination.event_router import (
    get_router,
    subscribe,
    publish,
    EventSource,
    RouterEvent,
)

router = get_router()

# Subscribe to specific event type
def on_training_complete(event: RouterEvent):
    print(f"Training complete from {event.origin}: {event.payload}")

    # Check event origin if needed
    if event.origin == EventSource.CROSS_PROCESS:
        print("Event came from another process")
    elif event.origin == EventSource.STAGE_BUS:
        print("Event came from stage bus")

subscribe("TRAINING_COMPLETED", on_training_complete)

# Subscribe to ALL events (global subscriber)
def log_all_events(event: RouterEvent):
    print(f"[{event.origin.value}] {event.event_type}: {event.payload}")

subscribe(None, log_all_events)  # None = all events
```

#### Event Type Mapping

The router automatically maps between event systems:

| DataEventType (data_events.py) | StageEvent (stage_events.py) |
| ------------------------------ | ---------------------------- |
| `TRAINING_COMPLETED`           | `training_complete`          |
| `MODEL_PROMOTED`               | `promotion_complete`         |
| `SELFPLAY_BATCH_COMPLETE`      | `selfplay_complete`          |
| `EVALUATION_COMPLETED`         | `evaluation_complete`        |
| `DATA_SYNC_COMPLETED`          | `sync_complete`              |

---

## Timeline Summary

| Module                       | Deprecated    | Removal Target |
| ---------------------------- | ------------- | -------------- |
| `orchestrated_training.py`   | December 2025 | Q2 2026        |
| `integrated_enhancements.py` | December 2025 | Q2 2026        |
| `_neural_net_legacy.py`      | December 2025 | Q1 2026        |
| `stage_events.py`            | December 2025 | Q2 2026        |
| `cross_process_events.py`    | December 2025 | Q2 2026        |
| Non-canonical databases      | December 2025 | Ongoing        |
| v1/v1_mps model formats      | Pre-2025      | Q1 2026        |

---

## Related Documentation

- [ORCHESTRATOR_GUIDE.md](../app/training/ORCHESTRATOR_GUIDE.md) - Detailed orchestrator migration
- [DEPRECATION_TIMELINE.md](./DEPRECATION_TIMELINE.md) - Full deprecation schedule
- [AGENTS.md](../AGENTS.md) - Canonical database requirements
- [TRAINING_DATA_REGISTRY.md](../TRAINING_DATA_REGISTRY.md) - Database provenance

---

## Getting Help

If you encounter issues during migration:

1. Check the deprecation warning message for specific guidance
2. Review the related documentation linked above
3. Run with `PYTHONWARNINGS=default` to see all deprecation warnings
4. Use `--allow-noncanonical` flag temporarily while migrating databases
