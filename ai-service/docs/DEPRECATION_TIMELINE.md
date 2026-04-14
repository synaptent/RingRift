# RingRift AI Service Deprecation Timeline

**Last Updated:** 2025-12-27
**Document Purpose:** Track deprecated modules and their planned removal dates.

---

## Active Deprecations

### Coordination Module Deprecations (December 2025)

| Module                                      | Status             | Deprecation Date | Target Removal   | Replacement                                                                      |
| ------------------------------------------- | ------------------ | ---------------- | ---------------- | -------------------------------------------------------------------------------- |
| `app/coordination/node_health_monitor.py`   | Archived (removed) | Dec 2025         | Removed Dec 2025 | `health_check_orchestrator.py`                                                   |
| `app/coordination/sync_coordinator.py`      | Deprecated shim    | Dec 2025         | Q3 2026          | `auto_sync_daemon.py` + `sync_facade.py` + `app/distributed/sync_coordinator.py` |
| `app/coordination/system_health_monitor.py` | Archived (removed) | Dec 2025         | Removed Dec 2025 | `unified_health_manager.py`                                                      |
| `app/coordination/cluster_data_sync.py`     | Archived (removed) | Dec 2025         | Removed Dec 2025 | `AutoSyncDaemon(strategy="broadcast")`                                           |
| `app/coordination/ephemeral_sync.py`        | Archived (removed) | Dec 2025         | Removed Dec 2025 | `AutoSyncDaemon(strategy="ephemeral")`                                           |

### Training Module Deprecations (December 2025)

| Module/Class                                           | Status     | Deprecation Date | Target Removal | Replacement                                          |
| ------------------------------------------------------ | ---------- | ---------------- | -------------- | ---------------------------------------------------- |
| `SmartCheckpointManager` (advanced_training.py)        | Deprecated | Dec 2025         | Q2 2026        | `UnifiedCheckpointManager` (checkpoint_unified.py)   |
| `SelfPlayConfig` (config.py)                           | Deprecated | Dec 2025         | Q2 2026        | `SelfplayConfig` (selfplay_config.py)                |
| `app/training/checkpointing.py`                        | Deprecated | Dec 2025         | Q2 2026        | `checkpoint_unified.py`                              |
| `app/training/data_pipeline_controller.py`             | Deprecated | Dec 2025         | Q2 2026        | `data_pipeline_orchestrator.py`                      |
| `app/training/distributed.py` (DistributedTrainer)     | Deprecated | Dec 2025         | Q2 2026        | `distributed_unified.py` (UnifiedDistributedTrainer) |
| `app/training/fault_tolerance.py` (retry_with_backoff) | Deprecated | Dec 2025         | Q2 2026        | `app.core.error_handler.retry`                       |

### Archived Modules (December 2025)

The following modules have been archived to `archive/deprecated_coordination/`:

- `queue_populator_daemon.py` → Use `unified_queue_populator.py`
- `queue_populator.py` (original) → Now re-export module
- `replication_monitor.py` → Use `unified_replication_daemon.py`
- `replication_repair_daemon.py` → Use `unified_replication_daemon.py`
- `model_distribution_daemon.py` → Use `unified_distribution_daemon.py`
- `npz_distribution_daemon.py` → Use `unified_distribution_daemon.py`

### Legacy Rules Engine Components

| Module                       | Status                           | Deprecation Date | Target Removal | Replacement                     |
| ---------------------------- | -------------------------------- | ---------------- | -------------- | ------------------------------- |
| `app/_game_engine_legacy.py` | Deprecated compatibility symlink | Dec 2025         | Q3 2026        | `app.game_engine` stable facade |
| `app/rules/phase_machine.py` | Transitional                     | Dec 2025         | Q2 2026        | `app/rules/fsm.py`              |
| `app/rules/legacy/*`         | Deprecated                       | Dec 2025         | Q4 2026        | Canonical rules v9+             |

### Legacy Rules Submodules

| Module                                     | Purpose                  | Target Removal | Notes                  |
| ------------------------------------------ | ------------------------ | -------------- | ---------------------- |
| `app/rules/legacy/move_type_aliases.py`    | Convert v1-v7 move types | Q4 2026        | For replay only        |
| `app/rules/legacy/replay_compatibility.py` | Replay old games         | Q4 2026        | Supports v1-v7         |
| `app/rules/legacy/phase_auto_advance.py`   | GPU selfplay compat      | Q2 2026        | Violates RR-CANON-R075 |
| `app/rules/legacy/state_normalization.py`  | Normalize old states     | Q4 2026        | For v1-v7 games        |

### Neural Network Legacy

| Module                         | Status                 | Target Removal | Replacement           |
| ------------------------------ | ---------------------- | -------------- | --------------------- |
| `app/ai/_neural_net_legacy.py` | Constants consolidated | Q2 2026        | `app/ai/neural_net/*` |

---

## Deprecation Phases

### Phase 1: Q1 2026 - Preparation

- [ ] Complete extraction of line formation logic to `app/rules/line.py`
- [ ] Complete extraction of territory logic to `app/rules/territory.py`
- [ ] Ensure all validators/mutators have shadow contracts
- [ ] Run full parity test suite

### Phase 2: Q2 2026 - Phase Machine Migration

- [ ] Migrate all phase logic from `phase_machine.py` to `fsm.py`
- [ ] Update GameEngine to use `fsm.py` for phase updates
- [ ] Mark `phase_machine.py` as deprecated (emit warnings)
- [ ] Remove `phase_auto_advance.py` (all selfplay now canonical)

### Phase 3: Q3 2026 - GameEngine Compatibility Retirement

- [ ] Enable mutator-first orchestration as default
- [ ] Migrate remaining direct `app._game_engine_legacy` callers to `app.game_engine`
- [ ] Keep `archive/deprecated_ai/_game_engine_legacy.py` as reference behind the `app.game_engine` facade while the compatibility path exists
- [ ] Remove direct legacy imports once all callers use the stable package surface

### Phase 4: Q4 2026 - Legacy Game Removal

- [ ] Remove `app/rules/legacy/*` modules
- [ ] Drop support for game schema versions < 8
- [ ] Retire the `app._game_engine_legacy.py` compatibility symlink after migration
- [ ] Complete migration to canonical-only rules

---

## Migration Guides

### Migrating from direct legacy GameEngine imports to app.game_engine

```python
# OLD (deprecated direct legacy import)
from app._game_engine_legacy import GameEngine
state = GameEngine.apply_move(state, move)

# NEW (stable public API)
from app.game_engine import GameEngine
state = GameEngine.apply_move(state, move)
```

### Migrating from phase_machine to fsm

```python
# OLD (deprecated)
from app.rules.phase_machine import advance_phases
state = advance_phases(state, player)

# NEW (canonical)
from app.rules.fsm import FSMValidator
validator = FSMValidator()
# Phase advancement handled by mutators
```

### Handling Legacy Game Replays

```python
# For v1-v7 games, use legacy replay path
from app.rules.legacy import requires_legacy_replay, replay_with_legacy_fallback

if requires_legacy_replay(game_record):
    states = replay_with_legacy_fallback(game_record)
else:
    # Canonical replay
    states = canonical_replay(game_record)
```

---

## Deprecation Warnings

Deprecated functions will emit warnings using:

```python
import warnings
warnings.warn(
    "Direct import from app._game_engine_legacy is deprecated. Use from app.game_engine import GameEngine instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Suppressing Warnings (During Migration)

```python
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Legacy code here
```

---

## Data Migration Requirements

### Game Records

| Schema Version | Status       | Action Required           |
| -------------- | ------------ | ------------------------- |
| v1-v7          | Legacy       | Migrate to v9+ or archive |
| v8             | Transitional | Verify hex geometry       |
| v9+            | Canonical    | No action needed          |

### Training Data

| Data Source                      | Status | Action                               |
| -------------------------------- | ------ | ------------------------------------ |
| GPU selfplay without bookkeeping | Legacy | Regenerate with `--canonical-mode`   |
| Gumbel MCTS before Dec 2025      | Verify | Check for explicit bookkeeping moves |
| Canonical selfplay v9+           | OK     | Use for training                     |

See `TRAINING_DATA_REGISTRY.md` for canonical database requirements.

---

## Contact & Ownership

For deprecation questions:

- Rules engine: See RULES_ENGINE_SURFACE_AUDIT.md
- Neural network: See AI_IMPROVEMENT_PLAN.md
- Training data: See TRAINING_DATA_REGISTRY.md
