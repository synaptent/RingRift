# RingRift AI-Service Strategic Improvement Plan

**Created**: December 25, 2025
**Status**: Active
**Last Assessment**: Comprehensive 4-agent exploration

---

## Executive Summary

After thorough exploration of 651 Python files across coordination, training, distributed, and quality dimensions, this plan prioritizes improvements to maximize AI training loop effectiveness while reducing technical debt.

### Overall Health Scores

| Dimension                | Score | Key Issues                                    |
| ------------------------ | ----- | --------------------------------------------- |
| **AI Training Loop**     | 92%   | 3 critical gaps in feedback wiring            |
| **Daemon Orchestration** | 75%   | 21 orphaned daemons, 4-way training decisions |
| **Data Sync**            | 85%   | Ephemeral timeout, NPZ not tracked            |
| **Code Quality**         | 60%   | 358 broad exceptions, test gaps               |

### Cluster Capacity

- **Total GPU VRAM**: 1,221GB across 23 nodes
- **Training Capable (≥40GB)**: 10 nodes (H100, A100, L40S, 8x4090/3090)
- **Active Nodes**: 22/26 updated and synced

---

## Priority 1: AI Training Loop Strengthening (CRITICAL)

### Gap 1.1: TRAINING_BLOCKED_BY_QUALITY → Selfplay Acceleration

**Status**: NOT WIRED
**Impact**: Quality gates don't trigger data generation

**Files to Modify**:

1. `app/training/train_cli.py` (~line 200-300)
   - Emit `TRAINING_BLOCKED_BY_QUALITY` when `--check-data-freshness` fails

2. `app/coordination/selfplay_scheduler.py` (~line 800+)
   - Add subscription to `TRAINING_BLOCKED_BY_QUALITY`
   - Boost selfplay allocation by 1.5-2x for blocked configs

```python
# selfplay_scheduler.py - Add handler
async def _on_training_blocked_by_quality(self, event: dict) -> None:
    config_key = event.get("config_key")
    current_weight = self._config_weights.get(config_key, 1.0)
    self._config_weights[config_key] = current_weight * 1.5
    logger.info(f"[SelfplayScheduler] Boosted {config_key} weight by 1.5x due to quality block")
```

### Gap 1.2: Verify REGRESSION_DETECTED Emission

**Status**: HANDLERS EXIST, EMISSION UNCLEAR
**Files**: `app/coordination/gauntlet_feedback_controller.py:440-460`

**Action**: Verify emit call exists after regression detection in gauntlet.

### Gap 1.3: Orphaned Event Types

5 events published but no subscribers:

| Event                      | Should Trigger              |
| -------------------------- | --------------------------- |
| `QUALITY_PENALTY_APPLIED`  | Curriculum weight reduction |
| `LOW_QUALITY_DATA_WARNING` | Selfplay throttle           |
| `OPPONENT_MASTERED`        | Curriculum advancement      |
| `EXPLORATION_BOOST`        | Temperature adjustment      |
| `TRAINING_LOSS_ANOMALY`    | Quality check               |

**Action**: Wire each to appropriate handler in feedback_loop_controller.py.

---

## Priority 2: Daemon Consolidation (HIGH)

### 2.1 Remove 21 Orphaned Daemons

Daemons defined in `DaemonType` enum but not in any production profile:

```
adaptive_resources, auto_promotion, cache_coordination, cluster_data_sync,
cross_process_poller, data_server, distillation, elo_sync, external_drive_sync,
gauntlet_feedback, gossip_sync, health_check, high_quality_sync, maintenance,
model_sync, multi_provider, queue_monitor, recovery_orchestrator,
selfplay_coordinator, sync_coordinator, vast_cpu_pipeline
```

**Action**: Either add to profiles or remove from `daemon_manager.py:58-195`.

### 2.2 Consolidate Training Decision Authority

Currently 4 systems decide when to train:

1. `TRAINING_TRIGGER` daemon
2. `DataPipelineOrchestrator`
3. `FeedbackLoopController`
4. `MasterLoopController`

**Solution**: Make `TRAINING_TRIGGER` the sole authority:

- Other systems feed signals INTO it
- Only TRAINING_TRIGGER emits TRAINING_STARTED

### 2.3 Merge AUTO_PROMOTION into UNIFIED_PROMOTION

Two overlapping promotion daemons exist:

- `AUTO_PROMOTION` (orphaned, complex logic)
- `UNIFIED_PROMOTION` (active in profiles)

**Action**: Merge and archive `auto_promotion_daemon.py`.

---

## Priority 3: Data Sync Resilience (HIGH)

### 3.1 Increase Ephemeral Write-Through Timeout

**Current**: 30 seconds
**Issue**: Vast.ai nodes can have 50-500ms latency
**Solution**: Increase to 60s with adaptive retry

**File**: `app/coordination/ephemeral_sync.py:68`

### 3.2 Track NPZ Files in ClusterManifest

**Current**: Only game databases tracked
**Gap**: NPZ training files not replicated systematically

**Solution**: Extend `cluster_manifest.py` with:

```python
def register_npz(self, npz_path: str, board_type: str, num_players: int, node_id: str) -> None:
    """Register NPZ file location for replication."""
```

### 3.3 Add Capacity Refresh to SyncRouter

**Issue**: Router loads capacity at init, never refreshes
**Risk**: Stale disk capacity data

**File**: `app/coordination/sync_router.py:138-184`
**Solution**: Refresh every 5 minutes or on DISK_USAGE_UPDATED event.

---

## Priority 4: Code Quality (MEDIUM)

### 4.1 Replace Top 20 Broad Exception Catches

**Target Files**:
| File | Catches | Priority |
|------|---------|----------|
| `helpers.py` | 15 | HIGH |
| `p2p_backend.py` | 6 | HIGH |
| `metrics/orchestrator.py` | 4 | MEDIUM |
| `registry_base.py` | 3 | MEDIUM |
| `marshalling.py` | 2 | MEDIUM |

**Replacement Strategy**:

```python
# Before
except Exception:
    pass

# After
from app.core.exceptions import NetworkError, SyncError
except NetworkError as e:
    logger.warning(f"Network error during sync: {e}")
except SyncError as e:
    logger.error(f"Sync failed: {e}")
    raise
```

### 4.2 Fix Bare `except:` in Scripts

**5 critical instances** that catch KeyboardInterrupt:

- `scripts/audit_cluster_data.py:312`
- `scripts/monitor_gpu_mcts_jobs.py:67, 92, 108`
- `scripts/validate_db_games.py:187`

### 4.3 Complete Deprecated Module Migration

Remove active imports of:

- `app/training/orchestrated_training.py` (386 lines)
- `app/training/integrated_enhancements.py` (1,350 lines)

Both are superseded by `UnifiedTrainingOrchestrator`.

---

## Priority 5: Test Coverage (MEDIUM)

### Critical Untested Modules

| Module Path              | Files | Impact                |
| ------------------------ | ----- | --------------------- |
| `app/rules/validators/`  | 5     | Game rule correctness |
| `app/rules/mutators/`    | 4     | State mutations       |
| `app/game_engine/`       | 2     | Core engine           |
| `app/rules/placement.py` | 1     | Ring placement        |

**Template**: See `tests/unit/rules/test_line_generator.py` for pattern.

---

## Cluster Utilization Optimization

### Current State

- **IdleResourceDaemon**: Monitors GPU every 60s, spawns selfplay at <10% utilization
- **SelfplayScheduler**: Calculates priorities based on staleness, ELO velocity, curriculum
- **Coverage**: All 12 configs supported (hex8, square8, square19, hexagonal × 2p/3p/4p)

### Gaps Identified

1. **No model availability check** before spawning selfplay
2. **No launch failure tracking** with backoff
3. **IDLE_RESOURCE on coordinator** - should be training-node only

### Recommended Enhancements

```python
# idle_resource_daemon.py - Add model validation
def _select_config_for_gpu(self, gpu_memory_gb: int) -> Optional[str]:
    config = self._scheduler.get_priority_config()
    if not ModelDiscovery().has_model(config):
        logger.warning(f"No model for {config}, falling back")
        return self._fallback_selection(gpu_memory_gb)
    return config
```

---

## Data Flow Optimization

### Current Flow

```
Selfplay → GameDB → SELFPLAY_COMPLETE → auto_export_daemon
    → NPZ → TRAINING_STARTED → Training
    → TRAINING_COMPLETED → evaluation_daemon
    → EVALUATION_COMPLETED → auto_promotion_daemon
    → MODEL_PROMOTED → model_distribution_daemon + curriculum
```

### Gaps

1. **No NPZ validation** before training
2. **No stale data detection** in pipeline
3. **Ephemeral hosts** may lose games before sync

### Solutions

1. Add NPZ validation step after export
2. Check data age in TRAINING_TRIGGER
3. Increase ephemeral sync priority for active selfplay nodes

---

## Implementation Phases

### Phase 1: Critical Fixes (Completed Dec 25, 2025)

- [x] Wire TRAINING_BLOCKED_BY_QUALITY → selfplay acceleration
- [x] Verify REGRESSION_DETECTED emission (already implemented)
- [x] Increase ephemeral write-through timeout to 60s
- [x] Replace 10 highest-risk broad exception catches
- [x] Wire orphaned events (EXPLORATION_BOOST, OPPONENT_MASTERED)

### Phase 2: Consolidation (Completed Dec 25, 2025)

- [x] Document 19 orphaned daemons (see docs/ORPHANED_DAEMONS.md)
- [x] AUTO_PROMOTION superseded by UNIFIED_PROMOTION (documented as archived)
- [x] NPZ tracking already implemented in ClusterManifest (first-class citizen)
- [x] Added SyncRouter capacity refresh mechanism (5-min interval)

### Phase 3: Quality & Testing (In Progress)

- [ ] Complete deprecated module migration
- [x] Add SyncRouter capacity refresh (moved to Phase 2)
- [ ] Create test templates for validators/mutators
- [ ] Wire 5 orphaned events to handlers

### Phase 4: Optimization (Ongoing)

- [ ] Add model validation to IdleResourceDaemon
- [ ] Move IDLE_RESOURCE from coordinator profile
- [ ] Add provider-specific bandwidth config
- [ ] Implement Merkle tree reconciliation for gossip

---

## Success Metrics

| Metric                     | Current    | Target |
| -------------------------- | ---------- | ------ |
| Training loop gaps         | 3 critical | 0      |
| Orphaned daemons           | 21         | 0      |
| Broad exception catches    | 358        | <50    |
| Rules test coverage        | 5%         | >60%   |
| Ephemeral data loss events | Unknown    | 0      |
| Active deprecated imports  | 2          | 0      |

---

## Quick Wins Already Completed (Dec 25, 2025)

- [x] Added ruff BLE linter rule for broad exception detection
- [x] Added deprecation warning to `app/training/distributed.py`
- [x] Fixed 5 `sqlite3.Error` catches in `resource_optimizer.py`
- [x] Created test template for `rules/test_line_generator.py`
- [x] Updated 22/26 cluster nodes with latest code

### Phase 1 Fixes Completed (Dec 25, 2025)

- [x] Wired TRAINING_BLOCKED_BY_QUALITY → selfplay acceleration (train.py, selfplay_scheduler.py)
- [x] Verified REGRESSION_DETECTED emission in gauntlet_feedback_controller.py:444
- [x] Increased ephemeral write-through timeout from 30s to 60s (ephemeral_sync.py)
- [x] Fixed 10+ bare `except:` catches in p2p_backend.py, monitor_gpu_mcts_jobs.py, audit_cluster_data.py, validate_db_games.py
- [x] Added EXPLORATION_BOOST and OPPONENT_MASTERED event types to data_events.py
- [x] Wired EXPLORATION_BOOST → temperature adjustment (temperature_scheduling.py)
- [x] Wired OPPONENT_MASTERED → curriculum advancement (selfplay_scheduler.py)

---

## Files Reference

| Category           | Key Files                                                    |
| ------------------ | ------------------------------------------------------------ |
| Event Wiring       | `feedback_loop_controller.py`, `training_coordinator.py`     |
| Daemon Registry    | `daemon_manager.py:58-195` (types), `2906-2985` (profiles)   |
| Sync Architecture  | `sync_router.py`, `cluster_manifest.py`, `ephemeral_sync.py` |
| Exceptions         | `app/core/exceptions.py`                                     |
| Consolidation Plan | `docs/CONSOLIDATION_PLAN_2025_12.md`                         |
