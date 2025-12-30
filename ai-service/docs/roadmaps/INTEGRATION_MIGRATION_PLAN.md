# Integration Migration Plan

> **Status Update (2025-12-14)**: Migration largely complete. `continuous_improvement_daemon.py`
> is now DEPRECATED in favor of `master_loop.py`. See [ORCHESTRATOR_SELECTION.md](../infrastructure/ORCHESTRATOR_SELECTION.md)
> for current guidance on which script to use.

> **Note:** References to `unified_ai_loop.py` in this document refer to the legacy monolithic loop.
> The canonical orchestrator is now `scripts/master_loop.py`.

This document outlines the migration plan for consolidating duplicate components and improving integration across the RingRift AI training infrastructure.

## Current State (December 2025)

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
- Scripts updated: ~~`pipeline_orchestrator.py`~~ (now deprecated), `continuous_improvement_daemon.py`
- Full deprecation: Next major version

### 2. Direct SQLite ELO Access in legacy `unified_ai_loop.py`

**Status**: ✅ COMPLETE (Dec 2025)

All Elo database access in the legacy `unified_ai_loop.py` now uses the centralized `EloService`:

```python
# Current implementation uses EloService throughout
if get_elo_service is not None:
    elo_svc = get_elo_service()
    rows = elo_svc.execute_query("SELECT ... FROM elo_ratings ...")
```

**Migrated Locations**:

- Priority scheduler model count query (line ~1555)
- Prometheus metrics export (line ~2996)
- All other Elo operations delegate to external components that use EloService

### 3. Multiple Tournament Scripts

**Status**: Candidates for consolidation

| Script                        | Purpose                    | Recommendation                                        |
| ----------------------------- | -------------------------- | ----------------------------------------------------- |
| `run_model_elo_tournament.py` | Model vs model tournaments | Keep (uses EloService)                                |
| `run_diverse_tournaments.py`  | Multi-config tournaments   | Keep (orchestrator integration)                       |
| `run_tournament.py` (basic)   | AI type tournaments        | Keep (evaluation phase)                               |
| `distributed_tournament.py`   | Cluster tournaments        | Evaluate - may duplicate `run_diverse_tournaments.py` |

## Pending Refactoring Tasks

### Priority 1: Complete EloService Migration ✅ COMPLETE

All EloService migration tasks have been completed:

1. ✅ Legacy `unified_ai_loop.py` uses `get_elo_service()` for all Elo database access
2. ✅ Model promotion and curriculum components use EloService or delegate to services that do
3. ✅ `elo_service.py` updated to use new `app.coordination` module (Dec 2025)

### Priority 2: Consolidate Tournament Scripts ✅ ANALYZED - NO CONSOLIDATION NEEDED

**Analysis Result (Dec 2025):** These scripts should remain separate as they serve orthogonal purposes:

| Script                          | Purpose                                                 | Execution Model                      |
| ------------------------------- | ------------------------------------------------------- | ------------------------------------ |
| `run_distributed_tournament.py` | Tier-based AI strength evaluation (D1-D10), Elo ratings | ThreadPoolExecutor, in-process games |
| `run_diverse_tournaments.py`    | Board/player config sampling for training data          | AsyncIO + subprocess orchestration   |

**Key Differences:**

- Different data models (MatchResult/TierStats vs ClusterHost/TournamentConfig)
- Different outputs (Elo ratings vs training samples)
- Different scheduling needs (discrete tiers vs exhaustive configs)

**Recommendations:**

1. Keep scripts separate
2. Consider extracting shared utilities (cluster host management, SSH) to common library
3. `run_diverse_tournaments.py` is already integrated into legacy `unified_ai_loop.py`

### Priority 3: Data Event Integration ✅ CORE COMPLETE

**Current Status (Dec 2025):** Core event infrastructure is fully integrated.

**Implemented:**

- `app/distributed/data_events.py` - Full event type definitions (DataEventType enum with 202 event types)
- `app/distributed/event_helpers.py` - Safe wrappers (`emit_*_safe()` functions)
- Legacy `unified_ai_loop.py` - Has its own EventBus + StageEventBus integration
- 12+ scripts import and use the event system

**Event Types Available:**

- Data: `NEW_GAMES_AVAILABLE`, `DATA_SYNC_*`
- Training: `TRAINING_STARTED`, `TRAINING_COMPLETED`, `TRAINING_FAILED`
- Evaluation: `EVALUATION_*`, `ELO_UPDATED`
- Promotion: `MODEL_PROMOTED`, `PROMOTION_*`
- Curriculum: `CURRICULUM_REBALANCED`, `WEIGHT_UPDATED`
- System: `DAEMON_*`, `HOST_ONLINE/OFFLINE`, `ERROR`

**Future Enhancements (not blocking):**

- Real-time dashboard updates (requires WebSocket/SSE infrastructure)
- Cross-node event propagation (requires distributed pub/sub)
- Automated alerting (requires integration with Prometheus/PagerDuty)

## Architecture After Migration

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATORS                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│ unified_ai_loop │ pipeline_orch   │ p2p_orchestrator        │
│ (CANONICAL)     │ (CI/CD)         │ (distributed P2P)       │
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
