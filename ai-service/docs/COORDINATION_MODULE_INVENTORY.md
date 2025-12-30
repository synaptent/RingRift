# Coordination Module Inventory

**Last Updated:** December 30, 2025
**Total Modules:** 219 Python modules in `app/coordination/`
**Status:** Active

Counts are snapshots unless noted; use:
`rg --files -g "*.py" app/coordination | wc -l` for a precise count.

This document is a curated inventory of primary modules in `app/coordination/` (not exhaustive). For detailed daemon documentation, see [DAEMON_REGISTRY.md](DAEMON_REGISTRY.md). For event system details, see [COORDINATION_ARCHITECTURE.md](COORDINATION_ARCHITECTURE.md).

---

## Quick Stats

| Category            | Count | Description                                  |
| ------------------- | ----- | -------------------------------------------- |
| Core Infrastructure | ~16   | Event system, types, enums, protocols        |
| Daemons             | 85    | Background services (see DAEMON_REGISTRY.md) |
| Sync                | ~18   | Data synchronization modules                 |
| Health & Monitoring | ~14   | Health checks, status, metrics               |
| Coordination        | ~15   | Orchestrators, coordinators, bridges         |
| Queue & Work        | ~10   | Work queue, scheduling, backpressure         |
| Deprecated          | 11    | Q2 2026 removal scheduled                    |

---

## Module Categories

### Core Infrastructure

Essential modules for event system, types, and base classes.

| Module                           | LOC  | Status | Purpose                                                           |
| -------------------------------- | ---- | ------ | ----------------------------------------------------------------- |
| `enums.py`                       | ~94  | Active | Central enum re-exports (LeadershipRole, DaemonType, etc.)        |
| `types.py`                       | ~350 | Active | Coordination types (BackpressureLevel, TaskType, BoardType, etc.) |
| `daemon_types.py`                | 700  | Active | DaemonType enum, DaemonInfo, DaemonManagerConfig, startup order   |
| `protocols.py`                   | ~300 | Active | Protocol definitions (CoordinatorProtocol, DaemonProtocol)        |
| `base_daemon.py`                 | ~200 | Active | BaseDaemon base class with lifecycle hooks                        |
| `coordinator_base.py`            | ~400 | Active | CoordinatorBase with SQLite persistence, singleton pattern        |
| `event_router.py`                | ~800 | Active | Unified event bus with deduplication, DLQ                         |
| `event_mappings.py`              | ~200 | Active | Event type mappings and conversions                               |
| `event_normalization.py`         | ~150 | Active | Canonical event name normalization                                |
| `event_emitters.py`              | ~600 | Active | 70+ typed event emitter functions                                 |
| `event_subscription_registry.py` | ~416 | Active | Delegated event wiring (DELEGATION_REGISTRY)                      |
| `core_utils.py`                  | ~100 | Active | Re-exports: tracing, locking, YAML utils                          |
| `core_events.py`                 | ~100 | Active | Re-exports: event router, mappings, emitters                      |
| `alert_types.py`                 | ~50  | Active | Alert dataclasses                                                 |
| `node_status.py`                 | ~300 | Active | NodeHealthState enum, NodeMonitoringStatus                        |

### Daemon Management

Daemon lifecycle and factory modules.

| Module                              | LOC    | Status | Purpose                                                |
| ----------------------------------- | ------ | ------ | ------------------------------------------------------ |
| `daemon_manager.py`                 | ~2,000 | Active | Main DaemonManager class, lifecycle, health monitoring |
| `daemon_runners.py`                 | ~1,100 | Active | 85 async runner functions for all daemon types         |
| `daemon_registry.py`                | ~150   | Active | Declarative DaemonSpec registry                        |
| `daemon_factory.py`                 | ~300   | Active | Factory pattern for daemon creation                    |
| `daemon_factory_implementations.py` | ~200   | Active | Factory method implementations                         |
| `daemon_lifecycle.py`               | ~150   | Active | Lifecycle state machine                                |
| `daemon_metrics.py`                 | ~100   | Active | Daemon performance metrics                             |
| `daemon_stats.py`                   | ~50    | Active | Statistics collection                                  |
| `daemon_adapters.py`                | ~200   | Active | Wrappers for legacy daemons                            |

### Sync Modules

Data synchronization infrastructure.

| Module                     | LOC  | Status     | Purpose                                         |
| -------------------------- | ---- | ---------- | ----------------------------------------------- |
| `auto_sync_daemon.py`      | ~500 | Active     | P2P sync daemon (HYBRID/EPHEMERAL/BROADCAST)    |
| `sync_facade.py`           | ~600 | Active     | Unified programmatic sync entry point           |
| `sync_router.py`           | ~400 | Active     | Intelligent routing based on node capabilities  |
| `sync_bandwidth.py`        | ~700 | Active     | Bandwidth-coordinated rsync, BatchRsync         |
| `sync_coordinator.py`      | ~300 | DEPRECATED | Use AUTO_SYNC, removal Q2 2026                  |
| `sync_base.py`             | ~100 | Active     | Base classes for sync modules                   |
| `sync_constants.py`        | ~50  | Active     | Sync configuration constants                    |
| `sync_durability.py`       | ~150 | Active     | Durability guarantees for sync                  |
| `sync_mutex.py`            | ~100 | Active     | Distributed mutex for sync                      |
| `sync_stall_handler.py`    | ~150 | Active     | Handle stalled sync operations                  |
| `sync_bloom_filter.py`     | ~100 | Active     | Bloom filter for sync deduplication             |
| `database_sync_manager.py` | ~500 | Active     | Base class for database sync (Elo, Registry)    |
| `cluster_transport.py`     | ~500 | Active     | Multi-transport failover (Tailscale→SSH→Base64) |
| `training_freshness.py`    | ~200 | Active     | Pre-training data freshness checks              |

### Health & Monitoring

Health checks, status monitoring, metrics.

| Module                             | LOC  | Status | Purpose                                          |
| ---------------------------------- | ---- | ------ | ------------------------------------------------ |
| `health_check_orchestrator.py`     | ~700 | Active | Node-level health tracking, HealthCheckResult    |
| `unified_health_manager.py`        | ~600 | Active | System-level health scoring                      |
| `health_facade.py`                 | ~150 | Active | Unified health interface re-exports              |
| `cluster_status_monitor.py`        | ~600 | Active | ClusterMonitor with async status methods         |
| `model_performance_watchdog.py`    | ~400 | Active | Track rolling win rates, detect regression       |
| `node_status.py`                   | ~300 | Active | NodeHealthState, NodeMonitoringStatus            |
| `node_availability_cache.py`       | ~390 | Active | Unified node availability cache (P2P/SSH/health) |
| `node_circuit_breaker.py`          | ~465 | Active | Per-node health circuit breaker for node probes  |
| `stall_detection.py`               | ~200 | Active | Detect stalled operations                        |
| `metrics_analysis_orchestrator.py` | ~500 | Active | Continuous metrics monitoring, plateau detection |
| `handler_resilience.py`            | ~150 | Active | Handler retry/circuit breaker logic              |

### Coordination & Orchestration

High-level coordinators and orchestrators.

| Module                               | LOC    | Status | Purpose                                              |
| ------------------------------------ | ------ | ------ | ---------------------------------------------------- |
| `data_pipeline_orchestrator.py`      | ~1,500 | Active | Pipeline stage tracking (SYNC→EXPORT→TRAIN→EVAL)     |
| `coordination_bootstrap.py`          | ~2,000 | Active | Bootstrap all coordinators from COORDINATOR_REGISTRY |
| `training_coordinator.py`            | ~800   | Active | Training dispatch, completion, model promotion       |
| `selfplay_orchestrator.py`           | ~600   | Active | Selfplay job coordination                            |
| `selfplay_scheduler.py`              | ~800   | Active | Priority-based selfplay allocation                   |
| `curriculum_integration.py`          | ~700   | Active | Bridges feedback loops for self-improvement          |
| `recovery_orchestrator.py`           | ~500   | Active | Model/training state recovery                        |
| `cache_coordination_orchestrator.py` | ~400   | Active | Model caching across cluster                         |
| `optimization_coordinator.py`        | ~300   | Active | Hyperparameter optimization coordination             |
| `task_lifecycle_coordinator.py`      | ~250   | Active | Task state machine management                        |
| `leadership_coordinator.py`          | ~400   | Active | Raft-like leader election                            |
| `multi_provider_orchestrator.py`     | ~600   | Active | Coordinates across Vast/RunPod/Nebius                |
| `base_orchestrator.py`               | ~200   | Active | Base class for orchestrators                         |
| `facade.py`                          | ~300   | Active | High-level coordination facade                       |
| `integration_bridge.py`              | ~200   | Active | Bridge between coordination systems                  |

### Queue & Work Management

Work queue, scheduling, backpressure.

| Module                       | LOC  | Status     | Purpose                                       |
| ---------------------------- | ---- | ---------- | --------------------------------------------- |
| `work_queue.py`              | ~800 | Active     | SQLite-backed work queue with claims          |
| `unified_queue_populator.py` | ~700 | Active     | Maintains work queue until Elo targets met    |
| `queue_populator.py`         | ~75  | DEPRECATED | Re-export of unified_queue_populator, Q2 2026 |
| `queue_monitor.py`           | ~300 | Active     | Queue depth, latency metrics                  |
| `job_scheduler.py`           | ~500 | Active     | PID-based resource allocation                 |
| `backpressure.py`            | ~200 | Active     | Backpressure level calculation                |
| `bandwidth_manager.py`       | ~300 | Active     | Bandwidth allocation across nodes             |
| `dead_letter_queue.py`       | ~400 | Active     | DLQ for failed events                         |

### Feedback & Curriculum

Training feedback loops and curriculum learning.

| Module                            | LOC  | Status | Purpose                                 |
| --------------------------------- | ---- | ------ | --------------------------------------- |
| `feedback_loop_controller.py`     | ~600 | Active | Central feedback loop orchestration     |
| `gauntlet_feedback_controller.py` | ~400 | Active | Bridges gauntlet evaluation to training |
| `unified_feedback.py`             | ~500 | Active | Unified feedback signal processing      |
| `feedback_signals.py`             | ~200 | Active | Feedback signal dataclasses             |
| `curriculum_weights.py`           | ~200 | Active | Curriculum weight management            |

### Pipeline Actions & Triggers

Pipeline stage actions and triggers.

| Module                       | LOC  | Status | Purpose                                |
| ---------------------------- | ---- | ------ | -------------------------------------- |
| `pipeline_actions.py`        | ~600 | Active | Stage invokers with circuit breaker    |
| `pipeline_triggers.py`       | ~300 | Active | Trigger conditions for pipeline stages |
| `training_trigger_daemon.py` | ~400 | Active | Decides when to trigger training       |
| `stage_events.py`            | ~150 | Active | Pipeline stage event definitions       |

### Daemons (Active)

Active daemon implementations. See [DAEMON_REGISTRY.md](DAEMON_REGISTRY.md) for full details.

| Module                            | Status | Purpose                                  |
| --------------------------------- | ------ | ---------------------------------------- |
| `auto_export_daemon.py`           | Active | Triggers NPZ export when thresholds met  |
| `auto_promotion_daemon.py`        | Active | Auto-promote models based on evaluation  |
| `auto_scaler.py`                  | Active | Cluster auto-scaling                     |
| `cluster_watchdog_daemon.py`      | Active | Self-healing cluster utilization         |
| `data_cleanup_daemon.py`          | Active | Auto-quarantine poor quality databases   |
| `data_consolidation_daemon.py`    | Active | Consolidate scattered selfplay games     |
| `disk_space_manager_daemon.py`    | Active | Proactive disk space management          |
| `evaluation_daemon.py`            | Active | Auto-evaluate models after training      |
| `idle_resource_daemon.py`         | Active | Monitors idle GPUs, spawns selfplay      |
| `maintenance_daemon.py`           | Active | Log rotation, DB vacuum, cleanup         |
| `node_recovery_daemon.py`         | Active | Auto-recover terminated nodes            |
| `orphan_detection_daemon.py`      | Active | Detect orphaned games not in manifest    |
| `p2p_auto_deployer.py`            | Active | Ensure P2P runs on all nodes             |
| `quality_monitor_daemon.py`       | Active | Continuous selfplay quality monitoring   |
| `s3_backup_daemon.py`             | Active | Backup models to S3                      |
| `tournament_daemon.py`            | Active | Automatic tournament scheduling          |
| `training_activity_daemon.py`     | Active | Detects training, triggers priority sync |
| `unified_distribution_daemon.py`  | Active | Model + NPZ distribution                 |
| `unified_idle_shutdown_daemon.py` | Active | Provider-agnostic idle detection         |
| `unified_replication_daemon.py`   | Active | Monitoring + repair combined             |
| `work_queue_monitor_daemon.py`    | Active | Work queue lifecycle + backpressure      |

### Utilities & Helpers

Small utility modules.

| Module                     | LOC  | Status     | Purpose                                            |
| -------------------------- | ---- | ---------- | -------------------------------------------------- |
| `utils.py`                 | ~100 | Active     | Miscellaneous utilities                            |
| `helpers.py`               | ~150 | Active     | Helper functions                                   |
| `handler_base.py`          | ~550 | Active     | Canonical handler base (HandlerBase, HandlerStats) |
| `base_handler.py`          | ~450 | Deprecated | Legacy helpers (superseded by handler_base.py)     |
| `singleton_mixin.py`       | ~500 | Active     | Singleton patterns (5 variants)                    |
| `distributed_lock.py`      | ~200 | Active     | Distributed locking                                |
| `tracing.py`               | ~150 | Active     | Distributed tracing context                        |
| `npz_validation.py`        | ~100 | Active     | NPZ file validation                                |
| `stability_heuristic.py`   | ~360 | Active     | Rating volatility + stability assessment           |
| `async_training_bridge.py` | ~150 | Active     | Async/sync training bridge                         |
| `async_bridge_manager.py`  | ~200 | Active     | Async bridge lifecycle                             |
| `task_decorators.py`       | ~100 | Active     | Task-related decorators                            |
| `duration_scheduler.py`    | ~100 | Active     | Duration-based scheduling                          |
| `master_loop_guard.py`     | ~50  | Active     | Guard against multiple master loops                |

### P2P Backend

P2P networking modules.

| Module               | LOC  | Status | Purpose                           |
| -------------------- | ---- | ------ | --------------------------------- |
| `p2p_backend.py`     | ~400 | Active | P2P communication backend         |
| `p2p_integration.py` | ~300 | Active | P2P integration with coordination |

### Resource Management

Resource allocation and optimization.

| Module                               | LOC  | Status | Purpose                                      |
| ------------------------------------ | ---- | ------ | -------------------------------------------- |
| `resource_monitoring_coordinator.py` | ~400 | Active | Resource monitoring                          |
| `unified_resource_coordinator.py`    | ~500 | Active | Unified resource management                  |
| `resource_optimizer.py`              | ~300 | Active | Resource optimization                        |
| `resource_targets.py`                | ~400 | Active | Resource target management                   |
| `adaptive_resource_manager.py`       | ~500 | Active | Dynamic resource scaling                     |
| `load_forecaster.py`                 | ~784 | Active | Cluster load prediction + scheduling windows |
| `utilization_optimizer.py`           | ~400 | Active | Optimize cluster workloads                   |
| `node_policies.py`                   | ~200 | Active | Node allocation policies                     |
| `host_health_policy.py`              | ~150 | Active | Host health policies                         |

### Coordinator Infrastructure

Coordinator patterns and dependencies.

| Module                        | LOC  | Status | Purpose                       |
| ----------------------------- | ---- | ------ | ----------------------------- |
| `coordinator_config.py`       | ~100 | Active | Coordinator configuration     |
| `coordinator_dependencies.py` | ~150 | Active | Coordinator dependency graph  |
| `coordinator_persistence.py`  | ~200 | Active | Coordinator state persistence |
| `unified_registry.py`         | ~300 | Active | Unified coordinator registry  |
| `unified_inventory.py`        | ~200 | Active | Inventory management          |
| `event_subscription_mixin.py` | ~100 | Active | Event subscription helpers    |

### Delivery & Distribution

Delivery tracking and distribution.

| Module                    | LOC  | Status | Purpose                 |
| ------------------------- | ---- | ------ | ----------------------- |
| `delivery_ledger.py`      | ~200 | Active | Track delivery status   |
| `delivery_retry_queue.py` | ~150 | Active | Retry failed deliveries |

### SLURM Backend

HPC cluster support (optional).

| Module             | LOC  | Status | Purpose                         |
| ------------------ | ---- | ------ | ------------------------------- |
| `slurm_backend.py` | ~200 | Active | SLURM job scheduler integration |

---

## Deprecated Modules (Q2 2026 Removal)

| Module                | Replacement                  |
| --------------------- | ---------------------------- |
| `sync_coordinator.py` | `auto_sync_daemon.py`        |
| `queue_populator.py`  | `unified_queue_populator.py` |

## Archived/Removed Modules (Dec 2025 Consolidation)

| Legacy Module Name         | Replacement / Notes                                            |
| -------------------------- | -------------------------------------------------------------- |
| `ephemeral_sync.py`        | Consolidated into `auto_sync_daemon.py` (strategy="ephemeral") |
| `cluster_data_sync.py`     | Consolidated into `auto_sync_daemon.py` (strategy="broadcast") |
| `node_health_monitor.py`   | Consolidated into `health_check_orchestrator.py`               |
| `system_health_monitor.py` | Consolidated into `unified_health_manager.py`                  |

---

## Related Documentation

- [DAEMON_REGISTRY.md](DAEMON_REGISTRY.md) - Detailed daemon catalog with factory methods
- [COORDINATION_ARCHITECTURE.md](COORDINATION_ARCHITECTURE.md) - Event system architecture
- [CLUSTER_INTEGRATION_GUIDE.md](CLUSTER_INTEGRATION_GUIDE.md) - P2P manager integration
- [DAEMON_INTEGRATION_GUIDE.md](DAEMON_INTEGRATION_GUIDE.md) - Adding new daemons

---

## Key Import Paths

```python
# Enums and types
from app.coordination.enums import DaemonType, DaemonState, LeadershipRole
from app.coordination.types import BackpressureLevel, TaskType, BoardType

# Event handler base (canonical)
from app.coordination.handler_base import HandlerBase

# Event system
from app.coordination.event_router import get_router, publish, subscribe

# Daemon management
from app.coordination.daemon_manager import get_daemon_manager, DaemonManager
from app.coordination.daemon_runners import get_runner

# Health
from app.coordination.health_facade import get_node_health, get_healthy_nodes

# Sync
from app.coordination.sync_facade import get_sync_facade

# Work queue
from app.coordination.work_queue import WorkQueue
from app.coordination.unified_queue_populator import UnifiedQueuePopulator
```
