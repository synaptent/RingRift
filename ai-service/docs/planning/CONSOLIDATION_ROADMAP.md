# AI-Service Consolidation Roadmap

**Created:** December 24, 2025
**Status:** Active Implementation

## Executive Summary

Comprehensive architectural assessment identified significant consolidation opportunities:

- 34+ selfplay scripts → 3 canonical entry points
- 25+ database connection patterns → 1 pooled manager
- 3 event systems → unified router (partially done)
- 10 coordination gaps in training pipeline

## Phase 1: Quick Wins (COMPLETED)

### 1.1 Archive Deprecated Scripts ✅

- **38 scripts** moved to `scripts/archive/`
- Categories: selfplay (13), training (19), export (6)
- Preserved in archive with README documentation

### 1.2 Pending: Deprecation Warnings

Add warnings to legacy imports that redirect to canonical modules:

- `model_registry.py` → `unified_model_store.py`
- `checkpointing.py` → `checkpoint_unified.py`

## Phase 2: Core Consolidation

### 2.1 Database Connection Pooling (CRITICAL)

**Problem:** 25+ files use direct `sqlite3.connect()` with inconsistent timeouts.

**Solution:** Centralize in `app/distributed/db_utils.py`:

```python
class ConnectionPool:
    def get_connection(db_path, profile="STANDARD") -> Connection
    def get_quick_connection(db_path) -> Connection  # 5s timeout
    def get_extended_connection(db_path) -> Connection  # 30s timeout
```

**Timeout Tiers:**

- QUICK (5s): Fast registry lookups
- STANDARD (15s): Normal operations
- EXTENDED (30s): Long-running transactions

**Files to Update:** 25+ modules in coordination/, training/, distributed/

### 2.2 Selfplay Script Consolidation (HIGH)

**Current State:** 34+ scripts with 70-80% overlap

**Target Architecture:**

```
scripts/selfplay.py              # Unified CLI entry point
├── app/training/selfplay_runner.py    # Base class (EXISTS)
├── app/training/selfplay_config.py    # Configuration (EXISTS)
├── app/training/gpu_mcts_selfplay.py  # GPU implementation (EXISTS)
└── app/coordination/selfplay_orchestrator.py  # Event coordination (EXISTS)
```

**Migration:**

1. Create `scripts/selfplay.py` wrapper (~100 lines)
2. Reduce each variant script to config + runner invocation
3. Archive 20+ redundant scripts

### 2.3 Model Factory Unification (MEDIUM)

**Current:** Two factories with 60-70% overlap

- `app/training/model_factory.py` - Training-centric
- `app/ai/neural_net/model_factory.py` - Inference-centric

**Target:** Single factory in `app/ai/neural_net/model_factory.py`

- Keep memory tier support (v2_lite, v3_lite, etc.)
- Add training config wrapper for backwards compatibility

### 2.4 Export Caching Consolidation (MEDIUM)

**Current:** 3 caching strategies in 6 files

- Mtime-based (`export_cache.py`)
- Hash-based (`dynamic_export.py`)
- Game-ID-based (`incremental_export.py`)

**Target:** Single `ExportManager` with configurable strategy

## Phase 3: Pipeline Integration

### 3.1 Auto-Triggering Pipeline Stages

**Gap:** DataPipelineOrchestrator tracks stages but doesn't auto-trigger.

**Solution:** Enable `auto_trigger=True` by default:

```python
# In app/coordination/__init__.py
initialize_all_coordinators(auto_trigger_pipeline=True)
```

**Event Flow:**

```
SELFPLAY_COMPLETE → auto-trigger SYNC
SYNC_COMPLETE → auto-trigger EXPORT
EXPORT_COMPLETE → auto-trigger TRAINING
TRAINING_COMPLETE → auto-trigger EVALUATION
```

### 3.2 Implement ExportOrchestrator

**New Component:** `app/coordination/export_orchestrator.py`

- Schedule export tasks based on data freshness
- Coordinate with training queue (backpressure)
- Track export metrics and failures

### 3.3 Training Queue Backpressure

**Problem:** Selfplay generates data faster than training consumes.

**Solution:**

- Monitor training queue depth
- Emit BACKPRESSURE_ACTIVATED when queue > threshold
- Selfplay reduces throughput in response

### 3.4 Curriculum Feedback Loop

**Current:** One-way (Elo → curriculum weights)
**Gap:** Curriculum changes don't affect selfplay config

**Solution:**

```
ELO_UPDATED → CurriculumFeedback.adjust_weights()
             → emit CURRICULUM_UPDATED
             → SelfplayOrchestrator.update_config()
             → Adjust difficulty/temperature
```

## Phase 4: Event System Hardening

### 4.1 Event Ordering Guarantees

**Problem:** No happens-before guarantees for related events.

**Solution:**

- Add `sequence_number` to all events (global monotonic)
- Add `parent_event_id` for causal relationships
- Process events in sequence order within subscriptions

### 4.2 Dead Letter Queue

**Problem:** Failed handler exceptions are logged but events lost.

**Solution:**

```sql
CREATE TABLE dead_letter (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,
    payload JSON,
    handler_name TEXT,
    error TEXT,
    retry_count INTEGER,
    created_at TIMESTAMP
);
```

- Background task retries with exponential backoff
- Manual inspection/replay tool

### 4.3 Event Topology Registry

**Problem:** No central knowledge of subscriptions.

**Solution:**

```python
@coordinator.register_subscription(
    event_type=DataEventType.NEW_GAMES_AVAILABLE,
    priority=10,
    required=True  # Startup fails if not subscribed
)
async def on_new_games(event): ...
```

### 4.4 Unified Backpressure System

**Combine signals into global gauge:**

- Event queue depth (all 3 buses)
- Work queue size
- Node resource utilization
- Selfplay games pending training

**Levels:** LOW (<30%), MEDIUM (30-60%), HIGH (60-80%), CRITICAL (>80%)

## Canonical Entry Points (Post-Consolidation)

| Function        | Canonical Module                                                   |
| --------------- | ------------------------------------------------------------------ |
| Selfplay        | `scripts/selfplay.py` → `app/training/selfplay_runner.py`          |
| Export          | `scripts/export_replay_dataset.py` → `app/training/export_core.py` |
| Training        | `app/training/train.py` → `app/training/unified_orchestrator.py`   |
| Model Storage   | `app/training/unified_model_store.py`                              |
| Event Routing   | `app/coordination/event_router.py`                                 |
| Cluster Monitor | `app/distributed/cluster_monitor.py`                               |
| Database Pool   | `app/distributed/db_utils.py`                                      |

## Files to Archive (Complete List)

### Already Archived (38 files)

- `scripts/archive/selfplay/` - 13 scripts
- `scripts/archive/training/` - 19 scripts
- `scripts/archive/export/` - 6 scripts

### Pending Archive

- `app/training/checkpointing.py` → checkpoint_unified.py
- `app/training/model_registry.py` (deprecated, keep for backwards compat)
- Additional 20+ selfplay variants after consolidation

## Success Metrics

| Metric                 | Before         | Target      |
| ---------------------- | -------------- | ----------- |
| Selfplay entry points  | 34+            | 3           |
| Export scripts         | 13             | 2           |
| DB connection patterns | 25+            | 1 (pooled)  |
| Event systems          | 3 (fragmented) | 1 (unified) |
| Pipeline auto-trigger  | Manual         | Automatic   |

## Timeline

- **Phase 1:** Complete ✅
- **Phase 2:** 3-5 days
- **Phase 3:** 5-7 days
- **Phase 4:** 5-10 days

**Total:** 2-3 weeks for full consolidation
