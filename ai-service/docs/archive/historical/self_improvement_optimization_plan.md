# Self-Improvement Loop Optimization Plan

> **STATUS: COMPLETED/MIGRATED**
>
> This optimization plan has been implemented. The features described here have been
> migrated to `scripts/unified_ai_loop.py`, which is now the canonical self-improvement
> loop implementation. See [ORCHESTRATOR_SELECTION.md](ORCHESTRATOR_SELECTION.md) for
> current script guidance.
>
> The `continuous_improvement_daemon.py` referenced below is DEPRECATED.

## Current State Analysis

### Feedback Latency: 95-190 minutes

From game generated → model in production:

1. Play game: 0-30 min
2. Wait for threshold: 0-30 min
3. Train: 60 min
4. Tournament: 30 min
5. Promotion: 0-60 min

### Key Bottlenecks Identified

| Bottleneck                           | Impact                       | Current                  | Target           |
| ------------------------------------ | ---------------------------- | ------------------------ | ---------------- |
| Sequential blocking subprocess calls | 3+ hours blocked per cycle   | Blocking `run_command()` | Async subprocess |
| Synchronous JSONL game counting      | O(total_data_size) per cycle | Read all files           | Cached counts    |
| Rate-limited promotion (1/hour)      | Delays improvements          | 60 min interval          | 15 min + events  |
| Sequential selfplay per config       | 1x throughput                | One at a time            | 3-5x parallel    |
| No training trigger events           | Wait for next cycle          | Polling every 60s        | Event-driven     |

---

## Implementation Plan

### Phase 1: Quick Wins (Immediate Impact)

#### 1.1 Cached Game Counting

**File**: `scripts/continuous_improvement_daemon.py`
**Change**: Cache JSONL line counts in metadata; only recount modified files
**Impact**: ~80% reduction in threshold-check time
**Risk**: Low (cache invalidation on file modification)

#### 1.2 Reduce Promotion Interval

**File**: `scripts/continuous_improvement_daemon.py`
**Change**: Reduce `AUTO_PROMOTE_INTERVAL` from 3600s to 900s (15 min)
**Impact**: 4x faster promotion cycles
**Risk**: Low (more frequent checks, same logic)

#### 1.3 Event-Driven Training Trigger

**File**: `scripts/continuous_improvement_daemon.py`
**Change**: Check training threshold immediately after selfplay completes
**Impact**: Eliminates 0-30 min wait for next cycle
**Risk**: Low

#### 1.4 Parallel Selfplay Configs

**File**: `scripts/continuous_improvement_daemon.py`
**Change**: Run selfplay for multiple board configs concurrently with semaphore
**Impact**: 3-5x faster data generation
**Risk**: Medium (resource contention)

### Phase 2: Pipeline Integration

#### 2.1 Training → Tournament Auto-Chain

**Change**: Automatically queue tournament when training completes
**Impact**: Eliminates manual phase coordination

#### 2.2 Tournament → Promotion Event Hook

**Change**: Trigger promotion check immediately after tournament
**Impact**: No waiting for scheduled promotion

#### 2.3 Unified Elo State

**Change**: Use single `unified_elo.db` as source of truth
**Impact**: Consistent rankings, no sync lag

### Phase 3: Async Execution

#### 3.1 Async Subprocess Calls

**Change**: Use `asyncio.create_subprocess_exec()`
**Impact**: Non-blocking training/tournament execution

#### 3.2 Concurrent Phase Execution

**Change**: Start next phase while previous completes
**Impact**: 2-3x cycle throughput

---

## Expected Results

### Before Optimization

- Cycle time: 90-120 min
- Feedback latency: 95-190 min
- Throughput: ~40 games/config/cycle

### After Phase 1

- Cycle time: 60-90 min (-30%)
- Feedback latency: 70-120 min (-35%)
- Throughput: ~120 games/config/cycle (3x)

### After All Phases

- Cycle time: 30-45 min (-65%)
- Feedback latency: 40-75 min (-60%)
- Throughput: ~200 games/config/cycle (5x)

---

## Implementation Status

- [x] 1.1 Cached game counting (count_games_in_jsonl with mtime cache)
- [x] 1.2 Reduce promotion interval (60min → 15min)
- [x] 1.3 Event-driven training trigger (after selfplay completes)
- [x] 1.4 Parallel selfplay configs (run_parallel_selfplay with semaphore)
- [x] 2.1 Training → Tournament chain (early training triggers tournament)
- [x] 2.2 Tournament → Promotion hook (promotion after early tournament)
- [ ] 2.3 Unified Elo state (already using unified_elo.db)
- [x] 3.1 Async subprocess calls (run_command_async)
- [ ] 3.2 Concurrent phases (future enhancement)

## Changes Made (2025-12-14)

### continuous_improvement_daemon.py

1. `AUTO_PROMOTE_INTERVAL`: 3600s → 900s (15 min)
2. `TRAINING_COOLDOWN_SECONDS`: 1800s → 900s (15 min)
3. Added `count_games_in_jsonl()` with mtime-based caching
4. Added `load_game_count_cache()` / `save_game_count_cache()`
5. Added `run_command_async()` for non-blocking subprocess
6. Added `run_single_selfplay_job()` with semaphore
7. Added `run_parallel_selfplay()` for concurrent data generation
8. Updated fallback selfplay to use parallel execution
9. Added event-driven training trigger after Phase 1a
10. Added immediate tournament + promotion after early training
