# Comprehensive Integration Assessment - December 2025

## Executive Summary

Comprehensive architectural assessment of RingRift AI-Service covering:

- **367K lines** of Python across 42 app subdirectories
- **35 distributed modules** managing 70+ cluster nodes
- **3 event buses** with 50+ event types
- **30+ scripts** with duplicated patterns

### Key Findings

| Area                   | Status                              | Critical Gaps                             |
| ---------------------- | ----------------------------------- | ----------------------------------------- |
| Training Pipeline      | Components exist, poorly integrated | No event-driven triggers between stages   |
| Event Systems          | Sophisticated but fragmented        | Missing subscribers, no dead letter queue |
| Code Duplication       | Some consolidation done             | Model factories, CLI parsing, logging     |
| Cluster Infrastructure | Production-ready                    | No feedback to training decisions         |
| Valuable Patterns      | 14+ patterns identified             | Underutilized across codebase             |

---

## Part 1: Training Pipeline Integration Gaps

### Current Pipeline (Manual Steps)

```
Selfplay (SQL) → [MANUAL] → Export (NPZ) → [MANUAL] → Training
    → [MANUAL] → Evaluation → [MANUAL] → Promotion
```

### Target Pipeline (Event-Driven)

```
Selfplay → [EVENT: games_ready] → Export → [EVENT: export_complete]
    → Training → [EVENT: training_complete] → Evaluation
    → [EVENT: eval_complete] → Promotion → [FEEDBACK: curriculum]
```

### Gap #1: Selfplay → Export (30-60 min lag)

**Severity:** HIGH

**Current State:**

- `selfplay_runner.py` generates games → SQLite
- `auto_export_training_data.py` polls on fixed interval
- No event notification when games ready

**Fix Required:**

```python
# In selfplay_runner.py line 232 (after run loop):
from app.coordination.event_emitters import emit_selfplay_complete
emit_selfplay_complete(config_key, games_completed, samples_total)
```

**Files:** `app/training/selfplay_runner.py:232`, `scripts/auto_export_training_data.py`

### Gap #2: Export → Training (30 min lag)

**Severity:** HIGH

**Current State:**

- Export completes, writes NPZ file
- No notification sent
- `auto_retrain.py` polls every 30 minutes

**Fix Required:**

```python
# In export_core.py after successful export:
from app.coordination.event_emitters import emit_sync_complete
emit_sync_complete(config_key, npz_path, game_count)
```

**Files:** `app/training/export_core.py`, `scripts/auto_retrain.py`

### Gap #3: Training → Evaluation Disconnect

**Severity:** MEDIUM

**Current State:**

- `train_loop.py` does local tournament
- `background_eval.py` runs in parallel
- No information exchange between them

**Impact:** Training may waste compute on models that background eval already shows are poor.

### Gap #4: Evaluation → Curriculum Feedback Missing

**Severity:** MEDIUM

**Current State:**

- `promotion_controller.py` promotes models
- `curriculum_feedback.py` tracks games but not promotion feedback
- No adaptive weight adjustment

**Fix Required:** Wire `on_promotion_success()` to adjust curriculum weights.

### Gap #5: Cluster Status Not Used for Training Decisions

**Severity:** MEDIUM

**Current State:**

- `cluster_monitor.py` tracks disk/memory/node health
- Training jobs don't check resources before starting
- No backpressure when cluster constrained

---

## Part 2: Event System Analysis

### Three-Layer Architecture

| Layer             | Module                                     | Purpose                   | Lines |
| ----------------- | ------------------------------------------ | ------------------------- | ----- |
| DataEventBus      | `app/distributed/data_events.py`           | Async in-memory events    | 1,841 |
| StageEventBus     | `app/coordination/stage_events.py`         | Pipeline stage events     | ~300  |
| CrossProcessQueue | `app/coordination/cross_process_events.py` | SQLite-backed persistence | ~500  |

### Unified Router

**File:** `app/coordination/event_router.py`

- Single facade over all 3 buses
- Bidirectional routing
- History tracking (1000 events)

### Event Emitters

**File:** `app/coordination/event_emitters.py` (1,980+ lines)

- 40+ typed emit functions
- Automatic fallback handling

### Coordination Gaps

1. **Missing Subscribers:**
   - `CMAES_TRIGGERED`, `NAS_TRIGGERED` - no visible handlers
   - `CURRICULUM_REBALANCED` - no clear subscriber applies weights
   - `CACHE_INVALIDATED` - unclear who listens

2. **No Dead Letter Queue:**
   - Failed events logged but lost
   - No retry mechanism
   - `events_dropped` counter but no recovery

3. **Race Conditions:**
   - Coordinator initialization order not enforced
   - Cross-process event lag (up to 1 second)
   - No event ordering guarantees

4. **Circular Dependencies:**
   - Promotion → Curriculum → Training → Promotion
   - If any handler fails, loop breaks

---

## Part 3: Code Duplication Opportunities

### Priority 1: CLI Argument Parsing (30+ scripts)

**Impact:** 500+ lines of boilerplate

**Current State:**

```python
# Repeated in 30+ scripts:
parser.add_argument("--board-type", ...)
parser.add_argument("--num-players", ...)
parser.add_argument("--db", ...)
```

**Solution:** Extend `scripts/lib/cli.py`:

```python
def add_board_args(parser):
    parser.add_argument("--board-type", choices=[...], default="square8")
    parser.add_argument("--num-players", type=int, choices=[2,3,4], default=2)

def add_db_args(parser):
    parser.add_argument("--db", type=Path, help="Database path")
    parser.add_argument("--db-pattern", help="Glob pattern for DBs")
```

### Priority 2: Model Factory Unification (3 implementations)

**Impact:** ~500 lines of duplicate logic

**Files:**

- `app/ai/neural_net/model_factory.py` (402 lines) - Inference
- `app/training/model_factory.py` (347 lines) - Training
- `app/ai/unified_loader.py` (~150 lines) - Detection

**Solution:** Single `app/ai/model_loading.py` with:

- Board-specific model instantiation
- Memory tier handling (high/low/v3/v4)
- Architecture detection
- Policy size mappings

### Priority 3: Logging Configuration (2 implementations)

**Impact:** Inconsistent logging across scripts

**Files:**

- `app/core/logging_config.py`
- `scripts/lib/logging_config.py`

**Solution:** Consolidate to single module, import everywhere.

### Priority 4: Export Pipeline Variants (4 scripts)

**Files:**

- `scripts/export_replay_dataset.py` (350+ lines)
- `scripts/jsonl_to_npz.py` (300+ lines)
- `scripts/db_to_training_npz.py` (200+ lines)
- `scripts/export_replay_dataset_parallel.py` (deprecated)

**Common Logic:**

- Database loading and filtering
- Game record iteration
- Feature extraction
- NPZ output

---

## Part 4: Valuable Patterns to Harvest

### Core Patterns (Immediately Reusable)

| Pattern             | File                          | Lines | Value                       |
| ------------------- | ----------------------------- | ----- | --------------------------- |
| UnifiedRegistryBase | `app/core/registry_base.py`   | 500   | SQLite registry with events |
| StateMachine        | `app/core/state_machine.py`   | 633   | Type-safe state transitions |
| SingletonMixin      | `app/core/singleton_mixin.py` | 303   | Three singleton variants    |
| Initializable       | `app/core/initializable.py`   | 570   | Dependency injection        |
| RetryDecorators     | `app/core/error_handler.py`   | 800+  | Retry with circuit breaker  |

### Infrastructure Patterns

| Pattern           | File                                | Lines | Value                      |
| ----------------- | ----------------------------------- | ----- | -------------------------- |
| GPUBatchEvaluator | `app/ai/gpu_batch.py`               | 400+  | Hardware-agnostic batching |
| ParallelEncoder   | `app/training/parallel_encoding.py` | 400+  | CPU parallel processing    |
| ResourceGuard     | `app/utils/resource_guard.py`       | 1400+ | OOM prevention             |
| GameDiscovery     | `app/utils/game_discovery.py`       | 600+  | Database discovery         |
| HotDataBuffer     | `app/training/hot_data_buffer.py`   | 400+  | Streaming game data        |

### Already Consolidated (Good Examples)

| Module               | Status  | Impact                                      |
| -------------------- | ------- | ------------------------------------------- |
| `gumbel_common.py`   | ✅ Done | Unified 3 copies of GumbelAction/GumbelNode |
| `selfplay_runner.py` | ✅ Done | Base class for all selfplay                 |
| `selfplay_config.py` | ✅ Done | Unified configuration                       |
| `selfplay.py`        | ✅ Done | Single CLI entry point                      |

---

## Part 5: Distributed Infrastructure Gaps

### Cluster Overview

- **70+ nodes** across Lambda, Vast.ai, Hetzner
- **35 distributed modules** (27K+ lines)
- **Multi-protocol networking** (HTTP → Tailscale → SSH)

### Integration Opportunities

1. **Cluster Status → Training Decisions**

   ```python
   # Before starting training:
   status = cluster_monitor.get_cluster_status()
   if status.avg_disk_usage > 85:
       logger.warning("Cluster disk high, deferring training")
       wait_for_resources(check_disk=True)
   ```

2. **Node Health → Job Routing**

   ```python
   # Route to healthiest nodes:
   priority_nodes = sorted(nodes, key=lambda n: n.response_time_ms)
   ```

3. **Resource-Aware Batching**
   ```python
   # Adjust batch size based on GPU memory:
   free_gb = get_node_free_gpu_memory(node)
   batch_size = 512 if free_gb > 32 else 256 if free_gb > 16 else 128
   ```

---

## Part 6: Implementation Roadmap

### Phase 1: Pipeline Event Triggers (HIGH PRIORITY)

**Effort:** 1-2 days | **Impact:** Reduce training latency from 30+ min to <2 min

1. Add `emit_selfplay_complete()` in `selfplay_runner.py`
2. Add `emit_export_complete()` in `export_core.py`
3. Subscribe `auto_retrain.py` to export events instead of polling
4. Subscribe `auto_export_training_data.py` to selfplay events

### Phase 2: Code Consolidation Quick Wins (MEDIUM PRIORITY)

**Effort:** 1 day | **Impact:** 500+ lines removed, better maintainability

1. Extend `scripts/lib/cli.py` with `add_board_args()`, `add_db_args()`, `add_model_args()`
2. Unify logging to single module
3. Document consolidated patterns in CLAUDE.md

### Phase 3: Event System Hardening (MEDIUM PRIORITY)

**Effort:** 2-3 days | **Impact:** Reliability, debugging

1. Add dead letter queue for failed handlers
2. Enforce coordinator initialization order
3. Add missing event subscribers
4. Add global event log for debugging

### Phase 4: Feedback Loop Completion (LOWER PRIORITY)

**Effort:** 3-5 days | **Impact:** Adaptive training

1. Wire evaluation results to curriculum feedback
2. Connect cluster status to training decisions
3. Add resource-aware job routing

---

## Summary Table: All Identified Gaps

| Gap                              | Severity | Effort | Impact | Phase  |
| -------------------------------- | -------- | ------ | ------ | ------ |
| Selfplay → Export event trigger  | HIGH     | Low    | High   | 1      |
| Export → Training event trigger  | HIGH     | Low    | High   | 1      |
| CLI argument consolidation       | MEDIUM   | Low    | Medium | 2      |
| Logging unification              | MEDIUM   | Low    | Medium | 2      |
| Dead letter queue                | MEDIUM   | Medium | High   | 3      |
| Coordinator init ordering        | MEDIUM   | Medium | Medium | 3      |
| Evaluation → Curriculum feedback | MEDIUM   | Medium | Medium | 4      |
| Cluster → Training feedback      | MEDIUM   | High   | High   | 4      |
| Model factory unification        | LOW      | High   | Medium | Future |
| Export pipeline consolidation    | LOW      | High   | Medium | Future |

---

## Key Files Referenced

### Training Pipeline

- `app/training/selfplay_runner.py` (421 lines)
- `app/training/train_loop.py` (340 lines)
- `app/training/unified_orchestrator.py` (1,936 lines)
- `app/training/curriculum_feedback.py`
- `app/training/promotion_controller.py`

### Event System

- `app/coordination/event_router.py`
- `app/coordination/event_emitters.py` (1,980 lines)
- `app/coordination/event_mappings.py`
- `app/distributed/data_events.py` (1,841 lines)

### Duplication Sources

- `app/ai/neural_net/model_factory.py` (402 lines)
- `app/training/model_factory.py` (347 lines)
- `scripts/lib/cli.py` (partial helpers)
- `scripts/lib/logging_config.py`

### Distributed Infrastructure

- `app/distributed/sync_coordinator.py` (1,975 lines)
- `app/distributed/cluster_monitor.py` (801 lines)
- `config/distributed_hosts.yaml` (70+ nodes)

---

_Assessment generated: December 24, 2025_
_Based on exploration of 50+ files across training, coordination, distributed modules_
