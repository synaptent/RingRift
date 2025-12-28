# RingRift AI Training System - Priority Action Plan

> **⚠️ HISTORICAL DOCUMENT**: This action plan was created on December 26, 2025.
> Many P0/P1 items have been addressed in subsequent work (Dec 27-28, 2025).
> See `DEPRECATION_TIMELINE.md` and `CLAUDE.md` for current status.

**Date**: December 26, 2025
**Assessment Summary**: Comprehensive review of feedback loops, cluster utilization, data flow, and code quality.

## Current State Scores

| Dimension                 | Score  | Status                                         |
| ------------------------- | ------ | ---------------------------------------------- |
| Feedback Loop Integration | 6/10   | Ghost events, weak coupling                    |
| Cluster Utilization       | 50-60% | Primary GH200 fleet offline, 38 daemons active |
| Data Flow                 | 8.5/10 | Strong write-through, P2P sync                 |
| Code Quality              | 7/10   | 27.5% test coverage                            |

## Priority 0: Critical Fixes (This Session)

### P0.1: Wire SELFPLAY_RATE_CHANGED to SelfplayScheduler

**Problem**: FeedbackAccelerator publishes SELFPLAY_RATE_CHANGED but no subscriber
**Fix**: Add subscription in selfplay_scheduler.py
**Impact**: Enables Elo momentum → Selfplay rate feedback loop

### P0.2: Make train.py Query Pending Hyperparameter Updates

**Problem**: GauntletFeedbackController stores updates, train.py never reads them
**Fix**: Add call to `get_pending_hyperparameter_updates()` in training loop
**Impact**: Evaluation → Training parameter coupling

### P0.3: Add DLQ Monitor Daemon

**Problem**: Dead letter queue grows without remediation
**Fix**: Create DLQMonitorDaemon that auto-retries and escalates
**Impact**: Prevents unbounded failure accumulation

---

## Priority 1: Strengthen Feedback Loops (1-2 Days)

### P1.1: Connect QUALITY_DEGRADED to Training Threshold Reduction

**Current**: Quality events emitted but not consumed for training decisions
**Fix**: Subscribe FeedbackLoopController to QUALITY_DEGRADED, reduce training thresholds

### P1.2: Subscribe CurriculumFeedback to MODEL_PROMOTED

**Current**: Promoted models continue in same curriculum stage
**Fix**: Reset curriculum weights on MODEL_PROMOTED event

### P1.3: Implement MODEL_ROLLBACK_REQUESTED Handler

**Current**: REGRESSION_CRITICAL emits rollback request but no handler
**Fix**: Add handler in TrainingCoordinator to actually rollback model

---

## Priority 2: Cluster Utilization (3-5 Days)

### P2.1: Add Fair Allocation Quotas per Config

**Current**: Configs can monopolize GPU time
**Fix**: Implement resource quotas (e.g., 30% square8, 20% hex8)
**Improvement**: +5% utilization

### P2.2: Implement Starvation Prevention for LOW Priority Jobs

**Current**: LOW priority jobs wait indefinitely
**Fix**: Boost job priority after 4 hours in queue
**Improvement**: +3% utilization

### P2.3: Add Cost-Aware Scheduling for Multi-Provider

**Current**: Same treatment for Vast.ai and RunPod
**Fix**: Prefer cheaper providers for equivalent workloads
**Improvement**: 10-20% cost reduction

### P2.4: Optimize Ephemeral Node Job Allocation

**Current**: Same timeout for ephemeral and persistent nodes
**Fix**: Shorter jobs on Vast.ai, longer on Runpod
**Improvement**: Reduced data loss risk

---

## Priority 3: Code Quality (1-2 Weeks)

### P3.1: Test Coverage for Critical Daemons

**Target Modules** (10 tests each):

1. `auto_promotion_daemon.py`
2. `idle_resource_daemon.py`
3. `ephemeral_sync.py`
4. `model_distribution_daemon.py`
5. `training_freshness.py`

### P3.2: Refactor daemon_manager.py

**Current**: 2,888-line god class
**Fix**: Split into daemon-type-specific factory modules
**Risk Reduction**: Easier maintenance and testing

### P3.3: Consolidate Event Publishing

**Current**: 5 different publish() variants
**Fix**: Single canonical `event_router.publish()` API
**Benefit**: Consistent event handling

### P3.4: Add Daemon Lifecycle Documentation

**Missing**: Guide for daemon patterns
**Deliverable**: `docs/architecture/DAEMON_LIFECYCLE.md`

---

## Priority 4: Long-Term Sustainability

### P4.1: Predictive Resource Allocation

**Current**: Historical averages for job duration
**Upgrade**: ML-based prediction with variance tracking

### P4.2: Human-in-the-Loop Pause Mechanism

**Current**: No way to pause pipeline for intervention
**Add**: `pause_pipeline`, `resume_pipeline` CLI commands

### P4.3: Distributed Deadlock Detection

**Current**: No mechanism to detect waiting cycles
**Add**: Watchdog for cyclic dependencies

---

## Success Metrics

| Metric                       | Current   | Target |
| ---------------------------- | --------- | ------ |
| Feedback Coupling            | 6/10      | 9/10   |
| Events with Subscribers      | 70%       | 100%   |
| Ghost Events                 | 5+        | 0      |
| Cluster Utilization          | 50-60%    | 80-85% |
| Test Coverage (coordination) | 27.5%     | 60%    |
| DLQ Unresolved               | Unbounded | <50    |

---

## Immediate Next Steps

1. **Fix SELFPLAY_RATE_CHANGED subscription** (15 min)
2. **Add train.py hyperparameter query** (30 min)
3. **Create DLQMonitorDaemon** (1-2 hours)
4. **Commit and push** (5 min)
5. **Deploy when the suspended GH200 fleet comes back online**

---

## Files Modified This Session

- `app/coordination/job_scheduler.py` - Added preemption support
- `app/coordination/selfplay_scheduler.py` - Added weight normalization
- `app/distributed/data_events.py` - Added JOB_PREEMPTED event
- `app/training/selfplay_runner.py` - Added SELFPLAY_RATE_CHANGED subscription, ephemeral sync

## Commits Pushed

1. `feat(scheduler): add job preemption for critical priority work`
2. `feat(feedback): strengthen feedback loop integration and ephemeral sync`
