# Consolidated Next Steps - December 2025

**Assessment Date**: December 26, 2025
**Overall System Health**: 72% (Functional but fragile)

## Executive Summary

The RingRift AI training system has a solid foundation but suffers from **integration gaps**. The feedback loop that should drive continuous improvement has 8 points of failure (ghost events, type mismatches). The data flow pipeline works but has redundant quality scorers causing inconsistent gating.

**Key Insight**: Individual components are well-built but poorly integrated. Fixing the feedback loop alone will improve reliability from 72% to ~85%.

---

## Current State

| Aspect        | Status | Severity | Key Issues                                    |
| ------------- | ------ | -------- | --------------------------------------------- |
| Feedback Loop | 75%    | HIGH     | 8 ghost events, curriculum feedback broken    |
| Data Flow     | 85%    | CRITICAL | 8 quality scorers, weak export fallbacks      |
| Cluster Util. | 70%    | HIGH     | Allocation unbalanced, ephemeral data at risk |
| Code Quality  | 65%    | MEDIUM   | Consolidation needed, test gaps exist         |
| Documentation | 60%    | MEDIUM   | Architecture diagrams missing                 |

---

## P0 - Critical (This Week)

### P0.5: Wire Missing Event Emitters

**Status**: Pending
**Effort**: 2 hours
**Files**: `data_events.py`, `curriculum_integration.py`, `daemon_manager.py`

Missing emitters break the feedback loop:

1. **`emit_data_stale()`** - Signal for stale training data
   - Add to `app/distributed/data_events.py`
   - Wire to `training_freshness.py` checks

2. **`emit_curriculum_advanced()`** - Signal when curriculum tier progresses
   - Add to `app/coordination/curriculum_integration.py`
   - Wire to `SelfplayScheduler` for priority adjustment

3. **`emit_daemon_status_changed()`** - Signal for daemon health transitions
   - Add to `app/coordination/daemon_manager.py`
   - Wire to `NODE_HEALTH_MONITOR` daemon

### P0.6: Fix Event Type Mismatches

**Status**: Pending
**Effort**: 1 hour
**Files**: `feedback_loop_controller.py`, `curriculum_integration.py`

String literals instead of DataEventType enums cause silent event drops:

```python
# BAD - string literal, events silently dropped
bus.subscribe("QUALITY_SCORE_UPDATED", handler)

# GOOD - type-safe, events properly routed
bus.subscribe(DataEventType.QUALITY_SCORE_UPDATED, handler)
```

Locations to fix:

- `feedback_loop_controller.py:361` - "SCHEDULER_REGISTERED"
- `curriculum_integration.py:543` - "QUALITY_FEEDBACK_ADJUSTED"
- `curriculum_integration.py:544` - "QUALITY_SCORE_UPDATED"

### P0.7: Wire TRAINING_THRESHOLD_REACHED

**Status**: Pending
**Effort**: 1 hour
**File**: `data_pipeline_orchestrator.py`

The TRAINING_THRESHOLD_REACHED event is emitted but has no subscriber. It should trigger automatic NPZ export:

```python
# In data_pipeline_orchestrator.py
bus.subscribe(DataEventType.TRAINING_THRESHOLD_REACHED, self._trigger_auto_export)
```

---

## P1 - High Priority (This Month)

### P1.1: Add Test Coverage for Feedback Loop

**Effort**: 6 hours
**Files**: New test files

Critical modules have no tests:

- `feedback_loop_controller.py` (1443 LOC) - core feedback orchestration
- `selfplay_scheduler.py` - priority allocation logic
- `idle_resource_daemon.py` - spawn safety checks

Create:

- `test_feedback_loop_controller.py` (~200 LOC)
- `test_selfplay_scheduler.py` (~150 LOC)
- Test event type mismatches (string vs enum)

### P1.2: Deprecate Legacy Quality Scorers

**Effort**: 4 hours
**Files**: 7 modules with quality scoring

8 different quality scorers exist - consolidate to `UnifiedQualityScorer`:

| Module                                       | Status       | Action                   |
| -------------------------------------------- | ------------ | ------------------------ |
| `unified_quality.py:UnifiedQualityScorer`    | CANONICAL    | Keep                     |
| `training_enhancements.py:DataQualityScorer` | Deprecated   | Add warning              |
| `quality_bridge.py:QualityBridge`            | Legacy       | Add warning              |
| `quality_extractor.py:QualityExtractor`      | Sync-only    | Add warning              |
| `data_quality.py:DatabaseQualityChecker`     | Validation   | Keep (different purpose) |
| `generate_data.py:DataQualityTracker`        | Generation   | Redirect to Unified      |
| `data_quality_orchestrator.py`               | Coordination | Redirect to Unified      |
| `unified_manifest.py:GameQualityMetadata`    | Manifest     | Keep (storage only)      |

### P1.3: Fix IdleResourceDaemon GPU Memory Safety

**Effort**: 6 hours
**File**: `idle_resource_daemon.py`

Current: Can spawn 40 jobs on any GPU → OOM on smaller GPUs
Fix: Calculate max_spawns = GPU_MEMORY / BOARD_SIZE_REQUIREMENT

```python
# board_memory_requirements (GB)
MEMORY_REQUIREMENTS = {
    "hex8": 2,
    "square8": 2,
    "square19": 8,
    "hexagonal": 16,
}
```

### P1.4: Add NPZ Distribution Polling Fallback

**Effort**: 4 hours
**File**: `npz_distribution_daemon.py`

Currently event-driven only. If event missed, NPZ stays local.

Add:

```python
async def _poll_for_new_npz(self):
    """Fallback polling every 2 minutes for new NPZ files."""
    while self._running:
        new_files = self._check_for_undistributed_npz()
        for npz_path in new_files:
            await self._distribute_npz(npz_path)
        await asyncio.sleep(120)
```

---

## P2 - Medium Priority (Q1 2026)

### P2.1: Consolidate Feedback Controllers

**Effort**: 8 hours

Merge `gauntlet_feedback_controller.py` into `feedback_loop_controller.py` using strategy pattern.

- Lines saved: ~800 LOC
- Benefit: Single event subscription point

### P2.2: Consolidate Sync Coordinators

**Effort**: 12 hours

5 sync coordinators with overlapping responsibilities:

- `sync_coordinator.py` (1344 LOC)
- `auto_sync_daemon.py`
- `sync_bandwidth.py`
- `ephemeral_sync.py`
- `cluster_data_sync.py`

Create single `SyncOrchestrator` with pluggable strategies.

### P2.3: Consolidate Promotion Coordinators

**Effort**: 12 hours

4 promotion-related modules:

- `auto_promotion_daemon.py`
- `promotion_controller.py`
- `model_distribution_daemon.py`
- `unified_health_manager.py` (rollback)

Create single `PromotionOrchestrator` with lifecycle phases.

### P2.4: Create Architecture Documentation

**Effort**: 8 hours

Missing diagrams:

1. Feedback loop flow (Selfplay → Training → Evaluation → Promotion → Curriculum)
2. Data flow (Game → Export → NPZ → Training → Model)
3. Daemon dependency graph
4. Event type mapping table
5. Sync strategy selection flowchart

---

## P3 - Nice to Have

### P3.1: Incremental NPZ Export

**Effort**: 10 hours

Cache export offsets, only process new games since last export.
Impact: Large database exports from hours → minutes.

### P3.2: Global Bandwidth Limiter

**Effort**: 6 hours

Cap total cluster sync bandwidth to prevent saturation.

### P3.3: PreTrainingQualityValidator

**Effort**: 8 hours

Blocking quality check before training starts.

---

## Implementation Order

Week 1 (P0):

1. P0.5: Wire missing emitters
2. P0.6: Fix type mismatches
3. P0.7: Wire TRAINING_THRESHOLD_REACHED

Week 2-4 (P1): 4. P1.1: Add feedback loop tests 5. P1.2: Deprecate legacy scorers 6. P1.3: Fix IdleResourceDaemon memory 7. P1.4: Add NPZ polling fallback

Q1 2026 (P2): 8. P2.1-P2.4: Consolidation work

---

## Quick Wins Already Completed (Dec 2025)

- ✅ P0.1: Connected feedback signals to event system
- ✅ P0.2: Added SELFPLAY_COMPLETE event emission
- ✅ P0.3: Fixed daemon dependency ordering
- ✅ P0.4: Wired TRAINING_ROLLBACK_NEEDED subscription
- ✅ Created CLUSTER_UTILIZATION_STRATEGY.md
- ✅ Updated cluster nodes with latest code

---

## Metrics to Track

| Metric                       | Current | Target | How to Measure                      |
| ---------------------------- | ------- | ------ | ----------------------------------- |
| Ghost events                 | 8       | 0      | Grep for emitted-but-not-subscribed |
| Quality scorers              | 8       | 2      | Count unique scorer classes         |
| Test coverage (coordination) | ~19%    | 50%    | pytest --cov                        |
| Feedback loop completeness   | 75%     | 95%    | Manual audit                        |
| System health                | 72%     | 90%    | Composite score                     |

---

## Commands for Verification

```bash
# Find ghost events (emitted but not subscribed)
grep -r "emit_" app/coordination --include="*.py" | grep "def emit_" | \
  while read line; do
    func=$(echo $line | grep -oE "emit_\w+")
    if ! grep -r "subscribe.*$func" app --include="*.py" -q; then
      echo "Ghost: $func"
    fi
  done

# Find string-literal subscriptions (should use DataEventType)
grep -r 'subscribe("[^"]*")' app --include="*.py"

# Count quality scorer implementations
grep -r "class.*Quality\|class.*Scorer" app --include="*.py" | grep -v test
```
