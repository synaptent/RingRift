# Integration Migration Plan

> **Status Update (2025-12-14)**: Migration largely complete. `continuous_improvement_daemon.py`
> is now DEPRECATED in favor of `unified_ai_loop.py`. See [ORCHESTRATOR_SELECTION.md](ORCHESTRATOR_SELECTION.md)
> for current guidance on which script to use.

This document outlines the migration plan for consolidating duplicate components and improving integration across the RingRift AI training infrastructure.

## Current State (December 2024)

### Orchestrator Integration

All three main orchestrators now share:

- **TaskCoordinator**: Global task limits preventing runaway spawning
- **EloService**: Centralized ELO operations
- **PipelineFeedbackController**: Closed-loop adaptation signals

### Components Consolidated

| Component           | Canonical Location                     | Status |
| ------------------- | -------------------------------------- | ------ |
| ELO Service         | `app/training/elo_service.py`          | Active |
| ELO Database        | `app/tournament/unified_elo_db.py`     | Active |
| ELO Calculator      | `app/tournament/elo.py`                | Active |
| Task Coordinator    | `app/coordination/task_coordinator.py` | Active |
| Feedback Controller | `app/integration/pipeline_feedback.py` | Active |

## Deprecated Components

### 1. ELO Functions in `scripts/run_model_elo_tournament.py`

**Status**: Deprecated (use `app/training/elo_service.py` instead)

**Migration Path**:

```python
# OLD (deprecated)
from scripts.run_model_elo_tournament import (
    init_elo_database,
    register_models,
    update_elo_after_match,
    get_leaderboard,
)

# NEW (canonical)
from app.training.elo_service import (
    get_elo_service,
    init_elo_database,  # Compatibility layer
    register_models,     # Compatibility layer
    update_elo_after_match,  # Compatibility layer
    get_leaderboard,    # Compatibility layer
)
```

**Timeline**:

- Compatibility layer added to `elo_service.py`
- Scripts updated: `pipeline_orchestrator.py`, `continuous_improvement_daemon.py`
- Full deprecation: Next major version

### 2. Direct SQLite ELO Access in `unified_ai_loop.py`

**Status**: Technical debt (should use EloService)

**Current Code** (lines 1531, 1673, 2235):

```python
# Direct SQLite access - needs refactoring
conn = sqlite3.connect(elo_db_path)
cursor.execute("SELECT ... FROM elo_ratings ...")
```

**Migration Path**:

```python
# Use EloService instead
if HAS_ELO_SERVICE:
    elo = get_elo_service()
    leaderboard = elo.get_leaderboard(board_type, num_players)
```

**Affected Locations**:

- `ModelPromoter` class (line ~1515)
- `AdaptiveCurriculum` class (line ~1653)
- `UnifiedAILoop._export_prometheus_metrics()` (line ~2234)

### 3. Multiple Tournament Scripts

**Status**: Candidates for consolidation

| Script                        | Purpose                    | Recommendation                                        |
| ----------------------------- | -------------------------- | ----------------------------------------------------- |
| `run_model_elo_tournament.py` | Model vs model tournaments | Keep (uses EloService)                                |
| `run_diverse_tournaments.py`  | Multi-config tournaments   | Keep (orchestrator integration)                       |
| `run_ai_tournament.py`        | AI type tournaments        | Keep (evaluation phase)                               |
| `distributed_tournament.py`   | Cluster tournaments        | Evaluate - may duplicate `run_diverse_tournaments.py` |

## Pending Refactoring Tasks

### Priority 1: Complete EloService Migration

1. Update `unified_ai_loop.py` to use `get_elo_service()` instead of direct SQLite
2. Update `ModelPromoter._check_promotion_candidates()` to use `EloService.get_leaderboard()`
3. Update `AdaptiveCurriculum._recompute_weights()` to use EloService

### Priority 2: Consolidate Tournament Scripts

1. Analyze overlap between `distributed_tournament.py` and `run_diverse_tournaments.py`
2. Create unified tournament interface if significant overlap exists
3. Update orchestrators to use consolidated interface

### Priority 3: Data Event Integration

The `app/distributed/data_events.py` module provides event emission for:

- `emit_game_generated()` - Game data available
- `emit_training_started()` / `emit_training_completed()` - Training lifecycle
- `emit_elo_updated()` - ELO changes
- `emit_model_promoted()` - Model deployment

Currently only partially integrated. Full integration would enable:

- Real-time dashboard updates
- Cross-node event propagation
- Automated alerting

## Architecture After Migration

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATORS                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│ unified_ai_loop │ pipeline_orch   │ cluster_orchestrator    │
│ (CANONICAL)     │ (CI/CD)         │ (distributed)           │
└────────┬────────┴────────┬────────┴────────┬────────────────┘
         │                 │                 │
         ├─────────────────┼─────────────────┤
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│               SHARED INFRASTRUCTURE                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│ TaskCoordinator │ EloService      │ FeedbackController      │
│ (rate limiting) │ (unified ELO)   │ (closed-loop adapt)     │
└────────┬────────┴────────┬────────┴────────┬────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
├─────────────────┬─────────────────┬─────────────────────────┤
│ unified_elo.db  │ game databases  │ model registry          │
│ (ELO ratings)   │ (training data) │ (trained models)        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Verification Checklist

After migration, verify:

- [ ] All orchestrators initialize TaskCoordinator on startup
- [ ] All orchestrators initialize EloService on startup
- [ ] All orchestrators initialize FeedbackController on startup
- [ ] ELO operations go through `elo_service.py`
- [ ] Feedback signals emitted after each pipeline stage
- [ ] No direct SQLite access for ELO data
- [ ] Task limits enforced (max 1 improvement loop cluster-wide)

## Notes

- Compatibility layer in `elo_service.py` provides same interface as deprecated scripts
- Feedback hooks added to all major pipeline stages
- TaskCoordinator integration prevents runaway spawning across cluster
