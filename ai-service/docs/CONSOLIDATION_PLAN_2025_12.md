# RingRift AI-Service Consolidation & Quality Improvement Plan

**Created**: December 25, 2025
**Status**: Active
**Focus**: Code quality, module consolidation, exception handling, test coverage

> Status: Historical plan (Dec 2025). Kept for reference; current consolidation status lives in `COORDINATION_MODULE_INVENTORY.md` and `README.md`.

---

## Executive Summary

Based on comprehensive codebase exploration, the ai-service has grown to **651 Python files** with **430K+ lines of code**. While recent consolidation efforts (Dec 2025) addressed many issues, significant technical debt remains in exception handling, module proliferation, and test coverage.

### Key Metrics

| Metric                           | Current    | Target      | Priority |
| -------------------------------- | ---------- | ----------- | -------- |
| Broad `except Exception:`        | 1,885      | <300        | HIGH     |
| Orchestrator/Coordinator classes | 34         | 5-7 facades | MEDIUM   |
| Exception hierarchy adoption     | 5%         | 80%         | HIGH     |
| Critical paths without tests     | 6+ modules | 0           | HIGH     |
| Deprecated imports still active  | 3+         | 0           | HIGH     |

---

## Phase 1: Exception Handling (Priority: CRITICAL)

### 1.1 Enforce Exception Type Adoption

**File**: `app/core/exceptions.py` (exists, 25+ types defined)

**Action Items**:

1. **Add linter configuration** (`.flake8` or `pyproject.toml`):

   ```toml
   [tool.ruff.lint]
   select = ["E", "F", "W", "BLE"]  # BLE = blind-except
   ```

2. **Priority modules for migration** (by catch count):
   | Module | Catches | Status |
   |--------|---------|--------|
   | training/ | 79 | PENDING |
   | coordination/ | 65 | PENDING |
   | distributed/ | 38 | PENDING |
   | scripts/ | 623 | LOW PRIORITY |

3. **Migration pattern**:

   ```python
   # BEFORE
   try:
       sync_to_host(host)
   except Exception as e:
       logger.error(f"Sync failed: {e}")

   # AFTER
   from app.core.exceptions import NetworkError, SyncError
   try:
       sync_to_host(host)
   except NetworkError as e:
       logger.error(f"Network error during sync: {e}")
       # Can retry
   except SyncError as e:
       logger.error(f"Sync logic error: {e}")
       # Cannot retry, need to fix
   ```

### 1.2 Exception Type Mapping

| Current Catch           | Recommended Type                       | Retryable          |
| ----------------------- | -------------------------------------- | ------------------ |
| SSH/connection failures | `NetworkError`, `SSHError`             | Yes                |
| Model loading           | `ModelLoadError`                       | No                 |
| Training crashes        | `TrainingError`                        | Depends            |
| GPU OOM                 | `GPUError`, `MemoryError`              | Yes (reduce batch) |
| Sync failures           | `SyncError`, `SyncTimeoutError`        | Yes                |
| Daemon lifecycle        | `DaemonError`, `DaemonStartupError`    | Depends            |
| Cluster issues          | `ClusterError`, `NodeUnreachableError` | Yes                |

---

## Phase 2: Deprecated Import Cleanup (Priority: HIGH)

### 2.1 Modules Still Importing Deprecated Code

| Deprecated Module                       | Active Imports          | Replacement                     |
| --------------------------------------- | ----------------------- | ------------------------------- |
| `app.training.distributed`              | 3+ files                | `distributed_unified.py`        |
| `app.training.data_pipeline_controller` | Used in docstrings      | `data_pipeline_orchestrator.py` |
| `app.tournament.unified_elo_db`         | Planned removal Q2 2026 | `elo_sync_manager.py`           |

### 2.2 Action Items

1. **Find all deprecated imports**:

   ```bash
   grep -r "from app.training.distributed import" app/ --include="*.py"
   ```

2. **Add deprecation warnings**:

   ```python
   # In deprecated module __init__.py
   import warnings
   warnings.warn(
       "app.training.distributed is deprecated. Use distributed_unified.py",
       DeprecationWarning,
       stacklevel=2
   )
   ```

3. **Set removal deadline**: Add to module docstring with date

---

## Phase 3: Module Consolidation (Priority: MEDIUM)

### 3.1 Orchestrator Proliferation

**Current**: 17 orchestrator files
**Target**: 3-5 unified facades

| Category | Current Modules                                                                                                             | Proposed Consolidation          |
| -------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Training | `training_orchestrator.py`, `unified_orchestrator.py`, `optimization_orchestrator.py`, `per_orchestrator.py`                | `TrainingOrchestrator` facade   |
| Data     | `data_pipeline_orchestrator.py`, `cache_coordination_orchestrator.py`, `data_quality_orchestrator.py`                       | `DataOrchestrator` facade       |
| Cluster  | `health_check_orchestrator.py`, `recovery_orchestrator.py`, `multi_provider_orchestrator.py`, `node_health_orchestrator.py` | `ClusterOrchestrator` facade    |
| Sync     | `sync_orchestrator.py` (legacy; prefer `sync_facade.py`)                                                                    | Keep separate (execution layer) |

### 3.2 Coordinator Proliferation

**Current**: 17 coordinator files
**Target**: 5-7 focused coordinators

| Keep (Core)                          | Merge Into Facades               |
| ------------------------------------ | -------------------------------- |
| `training_coordinator.py`            | -                                |
| `sync_coordinator.py` (scheduling)   | -                                |
| `task_coordinator.py`                | -                                |
| `leadership_coordinator.py`          | -                                |
| `model_lifecycle_coordinator.py`     | -                                |
| `resource_monitoring_coordinator.py` | Merge into `ClusterCoordinator`  |
| `optimization_coordinator.py`        | Merge into `TrainingCoordinator` |

### 3.3 Sync Module Consolidation

**Current**: 12 sync-related modules
**Target**: 6 focused modules

| Keep                                | Archive                                  |
| ----------------------------------- | ---------------------------------------- |
| `sync_coordinator.py` (scheduling)  | `sync_coordination_core.py` (superseded) |
| `sync_coordinator.py` (distributed) | `gossip_sync.py` (merged into auto_sync) |
| `auto_sync_daemon.py`               |                                          |
| `ephemeral_sync.py`                 |                                          |
| `sync_router.py`                    |                                          |
| `sync_bandwidth.py`                 |                                          |

---

## Phase 4: Test Coverage Expansion (Priority: HIGH)

### 4.1 Critical Paths Without Tests

| Module         | Files | Risk     | Action                |
| -------------- | ----- | -------- | --------------------- |
| `rules/`       | 26    | CRITICAL | Game rule correctness |
| `game_engine/` | 6     | CRITICAL | Move generation       |
| `db/`          | 10    | HIGH     | Data integrity        |
| `routes/`      | 8     | HIGH     | HTTP routing          |
| `validation/`  | 6     | MEDIUM   | Input sanitization    |
| `notation/`    | 6     | LOW      | Game notation         |

### 4.2 Test Templates

**Unit Test Template** (`tests/unit/rules/test_line_generator.py`):

```python
import pytest
from app.rules.generators.line import LineGenerator
from app.rules import GameState, Position

class TestLineGenerator:
    def test_horizontal_line_detection(self):
        state = GameState.from_fen("...")
        generator = LineGenerator(state)
        lines = generator.find_lines_from(Position(3, 3))
        assert len(lines) == expected_count

    def test_no_line_when_blocked(self):
        # Test edge case
        pass
```

### 4.3 Coverage Targets

| Phase   | Target                         | Timeline |
| ------- | ------------------------------ | -------- |
| Phase 1 | 60% (rules/, game_engine/)     | Q1 2026  |
| Phase 2 | 75% (add db/, validation/)     | Q2 2026  |
| Phase 3 | 85% (add routes/, integration) | Q3 2026  |

---

## Phase 5: Legacy File Cleanup (Priority: MEDIUM)

### 5.1 Large Legacy Files

| File                     | Lines | Status     | Action                                     |
| ------------------------ | ----- | ---------- | ------------------------------------------ |
| `_game_engine_legacy.py` | 4,479 | Superseded | Delete after generator extraction verified |
| `main.py`                | 2,537 | Active     | Split into route handlers + middleware     |
| `helpers.py`             | 800+  | Active     | Split into domain-specific modules         |

### 5.2 Archive Organization

**Current**: `app/coordination/deprecated/`, `app/ai/archive/`, `scripts/archive/deprecated/`

**Action**: Consolidate all deprecated code under `archive/` root with READMEs documenting migration paths.

---

## Implementation Timeline

### Week 1 (Dec 26 - Jan 1)

- [ ] Add ruff BLE linter rule
- [ ] Fix deprecated imports in 3 priority files
- [ ] Add deprecation warnings to legacy modules

### Week 2 (Jan 2 - Jan 8)

- [ ] Migrate 20 highest-impact `except Exception:` to typed exceptions
- [ ] Create test templates for rules/ module
- [ ] Write 5 unit tests for LineGenerator

### Week 3 (Jan 9 - Jan 15)

- [ ] Migrate remaining training/ exceptions
- [ ] Create TrainingOrchestrator facade
- [ ] Archive superseded orchestrators

### Week 4 (Jan 16 - Jan 22)

- [ ] Migrate coordination/ exceptions
- [ ] Create ClusterOrchestrator facade
- [ ] Write unit tests for game_engine/

### Month 2+

- [ ] Continue exception migration (target <300)
- [ ] Complete test coverage for critical paths
- [ ] Consolidate sync modules

---

## Metrics Tracking

### Weekly Review Checklist

- [ ] Count of `except Exception:` (target: decreasing)
- [ ] Test coverage % (target: increasing)
- [ ] Deprecated import count (target: 0)
- [ ] Active orchestrator/coordinator count (target: stable then decreasing)

### Success Criteria

1. **Exception handling**: <300 broad catches (down from 1,885)
2. **Test coverage**: >60% for critical paths
3. **Module consolidation**: 5-7 orchestrator facades (down from 17)
4. **Deprecated imports**: 0 active deprecated imports
5. **Legacy files**: `_game_engine_legacy.py` deleted

---

## Quick Wins (Can Do Today)

1. **Add ruff BLE rule** - 5 minutes, catches new broad exceptions
2. **Fix 5 highest-impact exceptions** in `training_coordinator.py` - 30 minutes
3. **Add deprecation warning to `app.training.distributed`** - 10 minutes
4. **Create test template for rules/** - 20 minutes

---

## Dependencies & Risks

### Dependencies

- Linter configuration approved by team
- Test framework (pytest) already configured
- Exception hierarchy already exists in `app/core/exceptions.py`

### Risks

- Breaking changes during exception migration
- Test coverage may reveal bugs
- Facade pattern may obscure debugging

### Mitigations

- Phased rollout with canary testing
- Run test suite after each change
- Add logging at facade boundaries
