# RingRift Q1 2025 Roadmap

**Generated:** December 26, 2025
**Status:** Archived (historical snapshot)

> This roadmap is retained for historical context only and is no longer updated.

---

## Executive Summary

The RingRift AI training infrastructure is **85% complete** with all 12 model configurations trained and a mature P2P cluster. The primary focus areas for Q1 2025 are:

1. **Large File Decomposition** - Critical for maintainability
2. **GPU Utilization Fix** - Stop wasting $8-10/hr on idle GPUs
3. **Observability Activation** - Enable real-time monitoring
4. **Testing Coverage** - Fill gaps in coordination layer

---

## Phase 1: Immediate Fixes (This Week)

### 1.1 Fix GPU Underutilization [CRITICAL]

**Problem:** 11/14 Vast.ai nodes and 2/3 RunPod nodes show 0% GPU utilization despite having powerful GPUs.

**Root Causes Identified:**

1. Parameter name mismatch: `engine` vs `engine_mode` in idle_resource_daemon.py
2. `gumbel-mcts` not in P2P orchestrator's supported modes list
3. UtilizationOptimizer never called in P2P job spawning

**Files to Modify:**

- `app/coordination/idle_resource_daemon.py` - Fix `engine` → `engine_mode`
- `scripts/p2p_orchestrator.py` - Add `gumbel-mcts` to supported modes
- `scripts/p2p_orchestrator.py` - Integrate UtilizationOptimizer

**Effort:** 2-4 hours
**Impact:** Save ~$8-10/hr in wasted GPU compute

### 1.2 Activate Alerting

**Problem:** Slack/Discord alerting infrastructure exists but webhooks not deployed.

**Action:**

```bash
# Deploy to all nodes
for host in $(yq '.hosts | keys | .[]' ai-service/config/distributed_hosts.yaml); do
  ssh $host "echo 'SLACK_WEBHOOK_URL=https://...' >> ~/.bashrc"
done
```

**Effort:** 1 hour
**Impact:** Real-time ops visibility

### 1.3 Verify Curriculum Integration

**Problem:** `DaemonType.CURRICULUM_INTEGRATION` may not be in active daemon profile.

**Action:**

```bash
python -c "
from app.coordination.daemon_manager import DaemonManager, DaemonProfile
dm = DaemonManager()
print('standard' in [d.name for d in dm.get_daemons_for_profile(DaemonProfile.STANDARD)])
"
```

**Effort:** 30 minutes
**Impact:** Enable automatic curriculum progression

---

## Phase 2: Architecture Improvements (Weeks 1-4)

### 2.1 Decompose p2p_orchestrator.py [CRITICAL]

**Current:** 28,406 lines, 309 commits, 60+ imports

**Target Architecture:**

```
scripts/p2p/
├── __init__.py
├── orchestrator.py          # Main entry point (~4,000 lines)
├── discovery.py             # Node discovery, broadcasting (~2,000 lines)
├── leader_election.py       # Bully algorithm (~1,500 lines)
├── webhook_server.py        # WebSocket/HTTP handlers (~1,500 lines)
├── state_persistence.py     # SQLite state management (~1,000 lines)
├── job_scheduler.py         # Job spawning, work queue (~2,000 lines)
├── health_checks.py         # Health monitoring, failure detection (~1,500 lines)
├── agent_management.py      # Agent delegation, config loading (~1,500 lines)
└── constants.py             # Shared constants (exists)
```

**Effort:** 2-3 weeks (split across 3-4 PRs)
**Risk:** HIGH - requires integration testing

### 2.2 Decompose turnOrchestrator.ts [CRITICAL]

**Current:** 3,963 lines, 117 commits - SOURCE OF TRUTH for rules

**Target Architecture:**

```
src/shared/engine/orchestration/
├── turnOrchestrator.ts      # Main entry (~400 lines)
├── phaseHandlers.ts         # Placement/Movement/Capture/Territory (~1,200 lines)
├── phases.ts                # Phase transitions and validation (~800 lines)
├── victory.ts               # Victory detection, tiebreakers (~600 lines)
├── specialRules.ts          # Pie rule, LPS, forced eliminations (~400 lines)
└── fsmIntegration.ts        # FSM validation layer (~500 lines)
```

**Effort:** 3-4 weeks (requires extensive parity testing)
**Risk:** EXTREME - any regression breaks game rules

### 2.3 Refactor train.py Parameters

**Problem:** `train_model()` has 80+ parameters

**Solution:** Use single `FullTrainingConfig` dataclass throughout

**Effort:** 1 week
**Risk:** LOW - config already exists, just enforce usage

---

## Phase 3: Observability & Testing (Weeks 2-6)

### 3.1 Integrate GPU Monitoring

**Problem:** GPU metrics exist but not displayed in ClusterMonitor

**Action:**

- Add `nvidia-smi` polling to ClusterMonitor watch output
- Expose per-node GPU utilization percentage
- Add GPU memory usage tracking

**Effort:** 4 hours
**Impact:** Detect underutilized expensive hardware

### 3.2 Implement Prometheus Exporter

**Problem:** Metrics exist in registry but not exposed to Prometheus

**Action:**

- Add `/metrics` endpoint to health server
- Expose Prometheus format (OpenMetrics compatible)
- Create basic Grafana dashboard template

**Effort:** 1-2 days
**Impact:** Enable standard monitoring stack

### 3.3 Add Coordination Layer Tests

**Problem:** 32% test coverage in `app/coordination/`

**Target:** 70% coverage for critical paths

**Priority Files:**

1. `daemon_manager.py` - Daemon lifecycle tests
2. `event_router.py` - Event routing tests
3. `selfplay_scheduler.py` - Scheduling algorithm tests
4. `feedback_loop_controller.py` - Feedback integration tests

**Effort:** 2-3 weeks
**Impact:** Catch regressions before production

---

## Phase 4: Code Quality (Weeks 4-8)

### 4.1 Unify Error Handling (Python)

**Problem:** Multiple error hierarchies, fragmented handling

**Action:**

- Create `app/errors/` module with unified hierarchy
- Migrate from deprecated `core/exceptions.py`
- Standardize exception handling in coordination layer

**Effort:** 1 week
**Impact:** Better debugging, consistent P2P behavior

### 4.2 Add Module Docstrings

**Problem:** `app/coordination/` (111 files) lacks documentation

**Target:** Module-level docstrings for all 35 daemon implementations

**Effort:** 3-4 days
**Impact:** Easier onboarding, better IDE support

### 4.3 Consolidate Configuration

**Problem:** 5 different config patterns across codebase

**Action:**

- Complete migration to `unified_config.py`
- Remove direct `os.environ.get()` calls
- Centralize all thresholds in `app/config/`

**Effort:** 1 week
**Impact:** Single source of truth for configuration

---

## Phase 5: Cleanup (Ongoing)

### 5.1 Archive Legacy Code

**Candidates for removal:**

- `app/ai/_neural_net_legacy.py` (7,080 lines, never called)
- `app/_game_engine_legacy.py` (4,479 lines, deprecated)

**Effort:** 2 hours per file
**Impact:** Reduce codebase size by ~11K lines

### 5.2 Remove Unused Event Types

**Dead APIs in event_router.py:**

- `GAME_QUALITY_UPDATED`
- `MODEL_CHECKPOINT_CREATED`
- `CLUSTER_SYNC_COMPLETED`
- `TRAINING_MILESTONE_REACHED`

**Effort:** 1 hour
**Impact:** Cleaner API surface

### 5.3 Update Checkpoint Format

**Security:** Migrate to `weights_only=True` by default

**Effort:** 2-3 hours
**Impact:** Reduce pickle deserialization attack surface

---

## Success Metrics

| Metric                     | Current          | Q1 Target    |
| -------------------------- | ---------------- | ------------ |
| GPU Utilization (Vast.ai)  | 21% (3/14 nodes) | >80%         |
| Largest Python File        | 28,406 lines     | <5,000 lines |
| Largest TS File            | 4,474 lines      | <2,000 lines |
| Coordination Test Coverage | 32%              | 70%          |
| Alerting Active            | No               | Yes          |
| Prometheus Metrics         | No               | Yes          |

---

## Implementation Order

```
Week 1:  [1.1] GPU Fix + [1.2] Alerting + [1.3] Curriculum Verify
Week 2:  [2.3] train.py params + [3.1] GPU Monitoring
Week 3:  [2.1] p2p_orchestrator.py (PR 1/3)
Week 4:  [2.1] p2p_orchestrator.py (PR 2/3) + [3.2] Prometheus
Week 5:  [2.1] p2p_orchestrator.py (PR 3/3) + [3.3] Tests start
Week 6:  [2.2] turnOrchestrator.ts (PR 1/2)
Week 7:  [2.2] turnOrchestrator.ts (PR 2/2) + [4.1] Error handling
Week 8:  [4.2] Docstrings + [4.3] Config consolidation
Ongoing: [5.1-5.3] Cleanup tasks
```

---

## Risk Mitigation

### For turnOrchestrator.ts Decomposition

1. Create feature branch: `feature/turn-orchestrator-decomposition`
2. Run full parity test suite (10K+ seeds) before AND after
3. Use Python parity tests: `check_ts_python_replay_parity.py`
4. Keep all aggregate imports at top level
5. No changes to public API
6. Merge only after 100% parity verification

### For p2p_orchestrator.py Decomposition

1. Create feature branch: `feature/p2p-modular`
2. Maintain lazy imports pattern for circular dependencies
3. Integration test each PR before merge
4. Deploy to staging cluster before production
5. Monitor P2P mesh health for 24h after each PR

---

## Resources Required

- **Engineering Time:** ~6-8 weeks total
- **Testing Infrastructure:** Existing (parity tests, jest, pytest)
- **Staging Environment:** Use Hetzner CPU nodes for testing
- **Monitoring:** Need to configure Slack webhook (1 hour)

---

## Appendix: File Decomposition Details

See detailed analysis in exploration agent outputs:

- Large file analysis: 14 files identified, ranked by priority
- Architecture assessment: Module structure, duplication, type safety
- Infrastructure status: 38+ daemons, P2P mesh health, data flow
- Training pipeline: 90% complete, curriculum integration TBD
- Tech debt: 4 critical TODOs, 5 deprecated modules properly archived
