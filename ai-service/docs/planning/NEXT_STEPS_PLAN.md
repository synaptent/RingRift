# AI-Service Next Steps Plan

**Created:** December 24, 2025
**Last Updated:** December 29, 2025
**Based on:** Exploration analysis of testing, training pipeline, distributed reliability, and documentation

## Executive Summary

> **December 29, 2025 Update:** Many items from the original assessment have been completed.
> Critical modules now have comprehensive test coverage. See "Completed Items" section below.

Four exploration agents analyzed the codebase and identified significant opportunities:

| Area                    | Finding                                                     | Priority | Status (Dec 29)               |
| ----------------------- | ----------------------------------------------------------- | -------- | ----------------------------- |
| Testing                 | 40-50% coverage, critical gaps in coordination/distributed  | CRITICAL | ✅ Key modules covered        |
| Training Pipeline       | Board type mapping duplicated 17x, hardcoded values         | HIGH     | ⏳ Partially addressed        |
| Distributed Reliability | Missing timeout budgets, no circuit breakers on data server | CRITICAL | ✅ Fixed                      |
| Documentation           | 3 READMEs for 31 major modules                              | HIGH     | ✅ Module docstrings complete |

## Completed Items (December 29, 2025)

### Test Coverage - Critical Modules Now Covered

| Module                  | LOC    | Tests                 | Coverage         |
| ----------------------- | ------ | --------------------- | ---------------- |
| `selfplay_scheduler.py` | 3,823  | 138 tests (2,005 LOC) | ✅ Comprehensive |
| `resource_optimizer.py` | 2,563  | 113 tests (2,188 LOC) | ✅ Comprehensive |
| `daemon_manager.py`     | ~2,000 | 45+ tests             | ✅ Good          |
| `handler_base.py`       | 550    | 45 tests              | ✅ Good          |
| `dead_letter_queue.py`  | ~400   | 45 tests              | ✅ Complete      |

### Documentation - Module Docstrings

All coordination modules now have module-level docstrings. Only `tournament/config.py` was missing (now added).

## Phase 1: Critical Reliability Fixes

**Priority: CRITICAL - Production stability**
**Estimated effort: 3-5 days**

### 1.1 Add Timeout Budgets to Sync Fallback Chain

**Problem:** Individual operations have timeouts but full sync has none. If one transport fails slowly, retry chain could exceed cluster expectations.

**Location:** `app/distributed/sync_coordinator.py:321-376`

**Fix:**

```python
class SyncOperationBudget:
    def __init__(self, total_seconds: float = 300, per_attempt: float = 30):
        self.total = total_seconds
        self.per_attempt = per_attempt
        self.start_time = time.time()

    @property
    def remaining(self) -> float:
        return max(0, self.total - (time.time() - self.start_time))

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0
```

### 1.2 Add Background Sync Watchdog

**Problem:** Sync loop swallows all exceptions and continues. If sync hangs, daemon appears alive but makes no progress.

**Location:** `app/distributed/sync_coordinator.py:1342-1376`

**Fix:**

- Add progress tracking: `self._last_successful_sync = time.time()`
- Implement deadline: `await asyncio.wait_for(full_cluster_sync(), timeout=deadline)`
- Emit health metric: `time_since_last_successful_sync`

### 1.3 Add Data Server Health Monitoring

**Problem:** Data server started as subprocess but no liveness monitoring. If process crashes, aria2 clients get 404s but server marked "running".

**Location:** `app/distributed/sync_coordinator.py:586-650`

**Fix:**

- Periodic health check: `POST /health` to data server
- Auto-restart if `returncode is not None`
- Circuit breaker for unavailable data server

### 1.4 Fix SSH Retry Total Timeout

**Problem:** Retries grow delay but no cumulative timeout. With retries=2 and 30s timeout: 30 + 2 + 30 + 4 + 30 = 96s total per operation.

**Location:** `app/distributed/ssh_transport.py:412-427`

**Fix:**

```python
def _ssh_with_budget(self, cmd, budget: SyncOperationBudget, retries=2):
    for attempt in range(retries + 1):
        if budget.exhausted:
            raise TimeoutError("Total sync budget exhausted")
        timeout = min(self.timeout, budget.remaining)
        # ... execute with timeout
```

## Phase 2: Critical Testing Coverage

**Priority: CRITICAL - Prevent regressions**
**Status: ✅ LARGELY COMPLETE (Dec 29, 2025)**

> **Update:** Critical coordination modules now have comprehensive test suites.
> Total coordination tests: 300+ passing tests across 50+ test files.

### 2.1 Coordination Module Tests - COMPLETED

**Previously untested modules now covered:**

| Module                         | Tests     | Status      |
| ------------------------------ | --------- | ----------- |
| `selfplay_scheduler.py`        | 138 tests | ✅ Complete |
| `resource_optimizer.py`        | 113 tests | ✅ Complete |
| `daemon_manager.py`            | 45+ tests | ✅ Complete |
| `handler_base.py`              | 45 tests  | ✅ Complete |
| `dead_letter_queue.py`         | 45 tests  | ✅ Complete |
| `quality_monitor_daemon.py`    | 42 tests  | ✅ Complete |
| `disk_space_manager_daemon.py` | 38 tests  | ✅ Complete |
| `priority_calculator.py`       | 25+ tests | ✅ Complete |
| `budget_calculator.py`         | 20+ tests | ✅ Complete |

**Previously listed as gaps (now covered - Dec 30, 2025):**

1. `event_router.py` (926 lines) - ✅ 670-line test file
2. `pipeline_actions.py` (853 lines) - ✅ 1,051-line test file

### 2.2 Distributed Module Tests - MOSTLY COMPLETE

**Previously listed as critical (now covered - Dec 30, 2025):**

1. `circuit_breaker.py` - ✅ 826-line test file
2. `cluster_coordinator.py` - ❌ Still needs tests
3. `sync_coordinator.py` - ✅ 1,170-line test file

**Remaining gap:** Only `cluster_coordinator.py` lacks dedicated tests.

### 2.3 Core Module Tests

**Critical gaps:**

1. `event_bus.py` - Core event system
2. `state_machine.py` - State transitions
3. `lifecycle.py` - Component lifecycle

## Phase 3: Code Quality Improvements

**Priority: HIGH - Maintainability**
**Estimated effort: 2-3 days**

### 3.1 Extract Board Type Mapping Utility

**Problem:** Board type string-to-enum conversion duplicated in 17+ locations.

**Files affected:**

- `app/training/event_driven_selfplay.py` (4 duplicates)
- `app/training/config.py` (2 instances)
- `app/training/train_cli.py` (2 instances)
- Plus 10+ other files

**Solution:** Create `app/utils/board_type_utils.py`:

```python
def string_to_board_type(board_str: str) -> BoardType:
    """Single source of truth for string→enum conversion."""
    BOARD_TYPE_MAP = {
        "square8": BoardType.SQUARE_8X8,
        "square19": BoardType.SQUARE_19X19,
        "hex8": BoardType.HEXAGONAL_9X9,
        "hexagonal": BoardType.HEXAGONAL_25X25,
    }
    return BOARD_TYPE_MAP.get(board_str, BoardType.SQUARE_8X8)
```

### 3.2 Make Hardcoded Values Configurable

| File                       | Line    | Value                            | Fix                     |
| -------------------------- | ------- | -------------------------------- | ----------------------- |
| `event_driven_selfplay.py` | 461     | `max_moves=500`                  | Add to `SelfplayConfig` |
| `train_gmo.py`             | 499     | `board_type="square8"`           | Make parameter          |
| `train_loop.py`            | 291-295 | `think_time=500, randomness=0.1` | Extract to config       |

### 3.3 Log Environment Variable Parse Failures

**Location:** `app/training/config.py:104-105`

**Current:** Silent `pass` on invalid env var
**Fix:** `logger.warning(f"Invalid value for {var}: {e}, using default")`

## Phase 4: Documentation

**Priority: HIGH - Developer onboarding**
**Estimated effort: 3-4 days**

### 4.1 Module READMEs (Top 5 by impact)

| Module             | LOC  | Priority | Key Topics                                          |
| ------------------ | ---- | -------- | --------------------------------------------------- |
| `app/training/`    | 104K | CRITICAL | Pipeline overview, data flow, hyperparameter tuning |
| `app/ai/`          | 70K  | CRITICAL | Neural net architecture, model I/O, NNUE status     |
| `app/rules/`       | 9.7K | HIGH     | Generator pattern, validator system                 |
| `app/distributed/` | 35K  | HIGH     | Cluster ops, sync strategies, fault tolerance       |
| `app/tournament/`  | 7K   | MEDIUM   | Elo system, gauntlet, distributed tournament        |

### 4.2 Architecture Guides

1. **Training Pipeline Architecture** - Unify scattered guides (ORCHESTRATOR_GUIDE.md, COORDINATOR_GUIDE.md)
2. **Cluster Configuration Guide** - P2P config, GPU rankings, troubleshooting
3. **Config Documentation** - Explain each YAML/JSON config file

### 4.3 Quick Wins

1. Add section to CLAUDE.md linking all scattered guides
2. Create table of contents for `/docs` directory (42 files)
3. Add "See Also" cross-links in existing docs

## Phase 5: Medium Priority Improvements

**Priority: MEDIUM - Polish**
**Estimated effort: 2-3 days**

### 5.1 Structured Logging for Sync Operations

**Problem:** Errors logged at WARNING even when sync succeeds via fallback. No visibility into which transport worked.

**Fix:** Emit structured log entries:

```json
{
  "sync_operation": "training",
  "transport_attempt": 1,
  "transport": "aria2",
  "success": false,
  "error": "...",
  "duration_ms": 1234,
  "fallback_to": "ssh"
}
```

### 5.2 Background Task Backpressure

**Problem:** Fire-and-forget `asyncio.create_task()` can accumulate unbounded.

**Location:** `sync_coordinator.py:1793, 1862`

**Fix:**

```python
self._task_semaphore = asyncio.Semaphore(10)
async with self._task_semaphore:
    await self._execute_priority_sync(hosts)
```

### 5.3 Graceful Shutdown for Background Loops

**Problem:** `stop_background_sync()` sets flag but doesn't wait for current sync to finish.

**Fix:** Make async, store task reference, use `asyncio.CancelledError` handling.

## Implementation Order

```
Week 1:
├── Phase 1.1: Timeout budgets (1 day)
├── Phase 1.2: Sync watchdog (1 day)
├── Phase 1.3: Data server health (1 day)
├── Phase 1.4: SSH retry fix (0.5 day)
└── Phase 2.1: event_router tests (1.5 days)

Week 2:
├── Phase 2.2: circuit_breaker tests (1 day)
├── Phase 2.2: sync_coordinator tests (2 days)
├── Phase 3.1: Board type utility (0.5 day)
└── Phase 3.2: Configurable values (1 day)

Week 3:
├── Phase 4.1: app/training README (1.5 days)
├── Phase 4.1: app/ai README (1.5 days)
├── Phase 4.2: Training pipeline architecture doc (1 day)
└── Phase 4.3: Quick wins (0.5 day)

Week 4:
├── Phase 5.1: Structured logging (1 day)
├── Phase 5.2: Task backpressure (0.5 day)
├── Phase 5.3: Graceful shutdown (0.5 day)
└── Buffer/overflow (2 days)
```

## Success Metrics

| Metric                     | Current   | Target               |
| -------------------------- | --------- | -------------------- |
| Coordination test coverage | 36%       | 70%                  |
| Distributed test coverage  | 9%        | 50%                  |
| Module READMEs             | 3/31      | 8/31                 |
| Sync operation visibility  | Logs only | Structured + metrics |
| Background loop health     | None      | Watchdog + deadlines |

## Dependencies

- Phase 2 (testing) depends on Phase 1 (reliability fixes) being stable
- Phase 4 (docs) can proceed in parallel with any phase
- Phase 5 depends on Phase 1 patterns being established

## Risks

1. **Circuit breaker changes** could affect production sync if not tested thoroughly
2. **Timeout budget** changes require careful tuning to avoid premature failures
3. **Testing gaps** are large - prioritize by production impact

## Out of Scope

- GPU MPS performance fix (documented as future work)
- Full training pipeline refactoring (train.py is 4,150 lines but stable)
- P2P cluster consensus protocol (would require significant architecture changes)
