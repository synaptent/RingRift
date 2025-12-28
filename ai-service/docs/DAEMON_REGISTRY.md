# Daemon Registry

This document provides a comprehensive reference for all daemons managed by the RingRift AI service `DaemonManager`.

**Last updated:** December 28, 2025 (Session - Event Subscriptions & Documentation)
**Total Daemon Types:** 73 (72 in `daemon_runners.py` + 1 inline in `daemon_manager.py`, 7 deprecated)
**Startup Order:** 18 daemons in `DAEMON_STARTUP_ORDER` (see `daemon_types.py`)
**Dependencies:** All 73 daemons have entries in `DAEMON_DEPENDENCIES`

> **Architecture Note (December 2025):** Factory methods have been extracted from `daemon_manager.py` to `daemon_runners.py`. Factory methods named `create_*()` are in `daemon_runners.py`; methods named `_create_*()` remain in `daemon_manager.py` for legacy or special cases.

## Table of Contents

- [Overview](#overview)
- [Daemon Types](#daemon-types)
  - [Core Infrastructure](#core-infrastructure)
  - [Sync Daemons](#sync-daemons)
  - [Training & Pipeline](#training--pipeline)
  - [Evaluation & Tournament](#evaluation--tournament)
  - [Health & Monitoring](#health--monitoring)
  - [Cluster Management](#cluster-management)
  - [Job Scheduling](#job-scheduling)
  - [Backup & Storage](#backup--storage)
  - [Resource & Utilization](#resource--utilization)
  - [Pipeline Automation](#pipeline-automation)
  - [System Health](#system-health)
  - [Cost Optimization](#cost-optimization)
  - [Feedback & Curriculum](#feedback--curriculum)
  - [Disk Space Management](#disk-space-management-december-2025)
- [Daemon Profiles](#daemon-profiles)
- [Dependency Graph](#dependency-graph)
- [Priority Levels](#priority-levels)
- [Factory Methods Reference](#factory-methods-reference)

## Overview

The `DaemonManager` coordinates the lifecycle of 60+ background services across the RingRift cluster. Daemons are organized into profiles based on node roles (coordinator, training_node, ephemeral, selfplay).
It also listens for backpressure events (`BACKPRESSURE_ACTIVATED`/`BACKPRESSURE_RELEASED`) and
pauses or resumes non-essential daemons when supported to reduce cluster load.

**Key Concepts:**

- **CRITICAL daemons**: Core infrastructure that requires faster failure detection (15s health checks)
- **Dependencies**: Daemons can depend on other daemons being started first
- **Profiles**: Predefined sets of daemons for different node types
- **Auto-restart**: Most daemons automatically restart on failure (with exponential backoff)

## Daemon Types

### Core Infrastructure

These daemons form the foundation of the cluster coordination system.

| Daemon Type            | Priority     | Description                                                                                                                      | Dependencies |
| ---------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `EVENT_ROUTER`         | **CRITICAL** | Unified event bus for all coordination. All pipeline events, feedback signals, and cross-daemon communication flow through this. | None         |
| `CROSS_PROCESS_POLLER` | HIGH         | Polls for events from other processes and forwards them to the event router.                                                     | None         |
| `DAEMON_WATCHDOG`      | HIGH         | Monitors daemon health and triggers restarts when failures are detected.                                                         | None         |
| `DLQ_RETRY`            | MEDIUM       | Dead letter queue remediation. Retries failed events from the DLQ.                                                               | EVENT_ROUTER |

**Factory Methods:**

- `_create_event_router()` → Uses `event_router.get_router()`
- `_create_cross_process_poller()` → Creates `CrossProcessEventPoller`
- `_create_daemon_watchdog()` → Uses `daemon_watchdog.start_watchdog()`
- `_create_dlq_retry()` → Creates `DLQRetryDaemon`

---

### Sync Daemons

Data synchronization across the cluster.

| Daemon Type             | Priority     | Description                                                                                                                | Dependencies               |
| ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `AUTO_SYNC`             | **CRITICAL** | Primary data sync mechanism. Pulls game data to coordinator; includes min-move completeness checks to avoid mid-write DBs. | None                       |
| `SYNC_COORDINATOR`      | DEPRECATED   | Legacy sync coordinator. Use `AUTO_SYNC` instead. Scheduled for removal Q2 2026.                                           | None                       |
| `GOSSIP_SYNC`           | MEDIUM       | P2P gossip-based data synchronization for eventual consistency.                                                            | None                       |
| `EPHEMERAL_SYNC`        | HIGH         | Aggressive 5-second sync for Vast.ai ephemeral nodes to prevent data loss on termination.                                  | None                       |
| `MODEL_SYNC`            | MEDIUM       | Syncs trained models across the cluster.                                                                                   | None                       |
| `MODEL_DISTRIBUTION`    | MEDIUM       | Auto-distributes models after promotion. Subscribes to MODEL_PROMOTED events.                                              | EVENT_ROUTER               |
| `NPZ_DISTRIBUTION`      | MEDIUM       | Syncs training data (NPZ files) after export.                                                                              | EVENT_ROUTER               |
| `EXTERNAL_DRIVE_SYNC`   | LOW          | Backup to external drives for disaster recovery.                                                                           | None                       |
| `CLUSTER_DATA_SYNC`     | MEDIUM       | Full cluster-wide data distribution and replication.                                                                       | EVENT_ROUTER               |
| `TRAINING_NODE_WATCHER` | MEDIUM       | Detects active training processes and triggers priority data sync.                                                         | None                       |
| `HIGH_QUALITY_SYNC`     | MEDIUM       | Priority sync for high-quality game data (quality score > 0.7).                                                            | None                       |
| `ELO_SYNC`              | MEDIUM       | Synchronize ELO ratings across cluster.                                                                                    | None                       |
| `TRAINING_DATA_SYNC`    | MEDIUM       | Pre-training data sync from OWC/S3. Ensures training nodes have fresh data before training.                                | EVENT_ROUTER               |
| `SYNC_PUSH`             | MEDIUM       | Push-based sync for GPU nodes. Pushes data to coordinator before cleanup to prevent data loss.                             | EVENT_ROUTER               |
| `S3_NODE_SYNC`          | MEDIUM       | Bi-directional S3 sync for cluster nodes. Syncs game data, models, and training files to/from S3.                          | EVENT_ROUTER               |
| `S3_CONSOLIDATION`      | LOW          | Consolidates data from all nodes to S3 (coordinator only). Runs after S3_NODE_SYNC.                                        | EVENT_ROUTER, S3_NODE_SYNC |

**Factory Methods:**

- `_create_auto_sync()` → Creates `AutoSyncDaemon`
- `_create_sync_coordinator()` → Creates `SyncCoordinator` (deprecated)
- `_create_gossip_sync()` → Creates `GossipSyncDaemon`
- `_create_ephemeral_sync()` → Uses `ephemeral_sync.get_ephemeral_sync_daemon()`
- `_create_model_sync()` → Creates `ModelSyncDaemon`
- `_create_model_distribution()` → Creates `ModelDistributionDaemon`
- `_create_npz_distribution()` → Creates `NPZDistributionDaemon`
- `_create_external_drive_sync()` → Creates `ExternalDriveSyncDaemon`
- `_create_cluster_data_sync()` → Creates `ClusterDataSyncDaemon`
- `_create_training_node_watcher()` → Uses `cluster_data_sync.get_training_node_watcher()`
- `_create_high_quality_sync()` → Creates `HighQualitySyncDaemon`
- `_create_elo_sync()` → Creates `EloSyncDaemon`
- `create_training_data_sync()` → Creates `TrainingDataSyncDaemon` (in daemon_runners.py)
- `create_sync_push()` → Creates `SyncPushDaemon` (in daemon_runners.py)
- `create_s3_node_sync()` → Creates `S3NodeSyncDaemon` (in daemon_runners.py)
- `create_s3_consolidation()` → Creates `S3ConsolidationDaemon` (in daemon_runners.py)

---

### Training & Pipeline

Training pipeline orchestration and coordination.

| Daemon Type                | Priority | Description                                                                                                        | Dependencies                |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| `DATA_PIPELINE`            | HIGH     | Orchestrates pipeline stages: selfplay → sync → export → train → evaluate → promote.                               | EVENT_ROUTER                |
| `DATA_CONSOLIDATION`       | HIGH     | Merges scattered selfplay games into canonical databases. Runs after sync, before training.                        | EVENT_ROUTER, DATA_PIPELINE |
| `CONTINUOUS_TRAINING_LOOP` | MEDIUM   | Continuous training loop that runs indefinitely.                                                                   | EVENT_ROUTER                |
| `UNIFIED_PROMOTION`        | HIGH     | Auto-promotes models after evaluation. Subscribes to EVALUATION_COMPLETED events.                                  | EVENT_ROUTER                |
| `AUTO_PROMOTION`           | HIGH     | Auto-promotes models based on evaluation thresholds. Emits MODEL_PROMOTED.                                         | EVENT_ROUTER, EVALUATION    |
| `DISTILLATION`             | LOW      | Creates smaller student models from larger teacher models for deployment.                                          | EVENT_ROUTER                |
| `SELFPLAY_COORDINATOR`     | MEDIUM   | Distributes selfplay workloads across the cluster.                                                                 | EVENT_ROUTER                |
| `NPZ_COMBINATION`          | MEDIUM   | Quality-weighted NPZ combination daemon. Combines multiple NPZ training files with quality-based sampling weights. | EVENT_ROUTER, DATA_PIPELINE |

**DATA_CONSOLIDATION Details (December 2025):**

Fixes critical training pipeline gap where selfplay games remain scattered across 30+ cluster nodes.

- **Event Subscriptions:** `NEW_GAMES_AVAILABLE`, `SELFPLAY_COMPLETE`
- **Event Emissions:** `CONSOLIDATION_STARTED`, `CONSOLIDATION_COMPLETE`
- **Data Flow:** Scattered DBs → canonical*{board}*{n}p.db → NPZ export → training
- **Configuration:**
  - `RINGRIFT_CONSOLIDATION_INTERVAL` - Seconds between consolidation cycles (default: 300)
  - `RINGRIFT_CONSOLIDATION_MIN_GAMES` - Minimum games to trigger consolidation (default: 50)
  - `RINGRIFT_CONSOLIDATION_BATCH_SIZE` - Games to process per batch (default: 100)

**Factory Methods:**

- `_create_data_pipeline()` → Creates `DataPipelineOrchestrator`
- `create_data_consolidation()` → Creates `DataConsolidationDaemon` (in daemon_runners.py)
- `_create_continuous_training_loop()` → Creates `ContinuousTrainingLoop`
- `_create_unified_promotion()` → Creates `PromotionController`
- `_create_auto_promotion()` → Uses `auto_promotion_daemon.get_auto_promotion_daemon()`
- `_create_distillation()` → Creates `DistillationDaemon`
- `_create_selfplay_coordinator()` → Creates `SelfplayScheduler`
- `create_npz_combination()` → Creates `NPZCombinationDaemon` (in daemon_runners.py)

---

### Evaluation & Tournament

Model evaluation and tournament scheduling.

| Daemon Type         | Priority | Description                                                                                      | Dependencies |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------ | ------------ |
| `EVALUATION`        | HIGH     | Auto-triggers evaluation after TRAINING_COMPLETE events.                                         | EVENT_ROUTER |
| `TOURNAMENT_DAEMON` | MEDIUM   | Automatic tournament scheduling for model comparison.                                            | EVENT_ROUTER |
| `GAUNTLET_FEEDBACK` | MEDIUM   | Bridges gauntlet evaluation results to training feedback. Emits REGRESSION_CRITICAL on failures. | EVENT_ROUTER |

**Factory Methods:**

- `_create_evaluation_daemon()` → Uses `evaluation_daemon.get_evaluation_daemon()`
- `_create_tournament_daemon()` → Uses `tournament_daemon.get_tournament_daemon()`
- `_create_gauntlet_feedback()` → Creates `GauntletFeedbackController`

---

### Health & Monitoring

Cluster health monitoring and alerting.

| Daemon Type                  | Priority   | Description                                                                                                                                                    | Dependencies                      |
| ---------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `NODE_HEALTH_MONITOR`        | HIGH       | Unified cluster health maintenance. Canonical health daemon.                                                                                                   | None                              |
| `COORDINATOR_HEALTH_MONITOR` | HIGH       | Tracks coordinator lifecycle events (healthy/unhealthy/degraded, heartbeat, init).                                                                             | EVENT_ROUTER                      |
| `WORK_QUEUE_MONITOR`         | HIGH       | Tracks work queue lifecycle (depth, latency, stuck jobs, backpressure).                                                                                        | EVENT_ROUTER, QUEUE_POPULATOR     |
| `HEALTH_CHECK`               | DEPRECATED | Legacy health checker. Use `NODE_HEALTH_MONITOR` instead. Scheduled for removal Q2 2026.                                                                       | None                              |
| `QUALITY_MONITOR`            | MEDIUM     | Continuous selfplay data quality monitoring. Triggers throttling feedback.                                                                                     | EVENT_ROUTER                      |
| `MODEL_PERFORMANCE_WATCHDOG` | MEDIUM     | Monitors model win rates and performance metrics.                                                                                                              | EVENT_ROUTER                      |
| `ORPHAN_DETECTION`           | LOW        | Detects orphaned game databases not in the cluster manifest.                                                                                                   | None                              |
| `SYSTEM_HEALTH_MONITOR`      | HIGH       | Global system health with pipeline pause capability.                                                                                                           | EVENT_ROUTER, NODE_HEALTH_MONITOR |
| `HEALTH_SERVER`              | HIGH       | HTTP endpoints: /health, /ready, /metrics for external monitoring.                                                                                             | None                              |
| `INTEGRITY_CHECK`            | MEDIUM     | Data integrity checking daemon. Scans for games without moves, orphaned databases, and schema violations. Quarantines invalid games to `orphaned_games` table. | EVENT_ROUTER                      |

**Factory Methods:**

- `_create_node_health_monitor()` → Creates `UnifiedNodeHealthDaemon`
- `create_coordinator_health_monitor()` → Creates `CoordinatorHealthMonitorDaemon` (in daemon_runners.py)
- `create_work_queue_monitor()` → Creates `WorkQueueMonitorDaemon` (in daemon_runners.py)
- `_create_health_check()` → Creates `HealthChecker` (deprecated)
- `_create_quality_monitor()` → Uses `quality_monitor_daemon.create_quality_monitor()`
- `_create_model_performance_watchdog()` → Creates `ModelPerformanceWatchdog`
- `_create_orphan_detection()` → Creates `OrphanDetectionDaemon`
- `_create_system_health_monitor()` → Creates `SystemHealthMonitorDaemon`
- `_create_health_server()` → Creates `HealthServerDaemon`
- `create_integrity_check()` → Creates `IntegrityCheckDaemon` (in daemon_runners.py)

---

### Cluster Management

P2P cluster coordination and monitoring.

| Daemon Type           | Priority | Description                                                                 | Dependencies |
| --------------------- | -------- | --------------------------------------------------------------------------- | ------------ |
| `CLUSTER_MONITOR`     | MEDIUM   | Real-time cluster monitoring (game counts, disk usage, training processes). | None         |
| `P2P_BACKEND`         | HIGH     | P2P node coordination and leader election.                                  | None         |
| `QUEUE_MONITOR`       | MEDIUM   | Monitors queue depths and applies backpressure.                             | None         |
| `REPLICATION_MONITOR` | MEDIUM   | Monitors data replication health across cluster.                            | None         |
| `REPLICATION_REPAIR`  | MEDIUM   | Actively repairs under-replicated data.                                     | None         |
| `P2P_AUTO_DEPLOY`     | MEDIUM   | Ensures P2P orchestrator runs on all cluster nodes.                         | EVENT_ROUTER |
| `CLUSTER_WATCHDOG`    | HIGH     | Self-healing cluster utilization monitor. Standalone daemon on coordinator. | None         |

**Factory Methods:**

- `_create_cluster_monitor()` → Creates `ClusterMonitor`
- `_create_p2p_backend()` → Creates `P2PNode`
- `_create_queue_monitor()` → Creates `QueueMonitor`
- `_create_replication_monitor()` → Uses `replication_monitor.get_replication_monitor()`
- `_create_replication_repair()` → Uses `replication_repair_daemon.get_replication_repair_daemon()`
- `_create_p2p_auto_deploy()` → Creates `P2PAutoDeployDaemon`
- `_create_cluster_watchdog()` → Creates `ClusterWatchdogDaemon`

---

### Job Scheduling

Resource allocation and job scheduling.

| Daemon Type          | Priority | Description                                                    | Dependencies |
| -------------------- | -------- | -------------------------------------------------------------- | ------------ |
| `JOB_SCHEDULER`      | HIGH     | Centralized job scheduling with PID-based resource allocation. | EVENT_ROUTER |
| `RESOURCE_OPTIMIZER` | MEDIUM   | Optimizes resource allocation across the cluster.              | None         |
| `VAST_CPU_PIPELINE`  | LOW      | CPU-only preprocessing pipeline for Vast.ai CPU nodes.         | None         |

**Factory Methods:**

- `_create_job_scheduler()` → Creates `JobScheduler`
- `_create_resource_optimizer()` → Creates `ResourceOptimizer` (not implemented yet)
- `_create_vast_cpu_pipeline()` → Creates `VastCpuPipelineDaemon`

---

### Backup & Storage

Data backup and external storage.

| Daemon Type   | Priority | Description                                                                     | Dependencies                     |
| ------------- | -------- | ------------------------------------------------------------------------------- | -------------------------------- |
| `S3_BACKUP`   | MEDIUM   | Backs up models to S3 after promotion. Runs after MODEL_DISTRIBUTION completes. | EVENT_ROUTER, MODEL_DISTRIBUTION |
| `DATA_SERVER` | MEDIUM   | HTTP server for serving game data and models over P2P network (port 8771).      | None                             |

**Factory Methods:**

- `_create_s3_backup()` → Creates `S3BackupDaemon`
- `_create_data_server()` → Creates `DataServer`

---

### Resource & Utilization

GPU/CPU resource optimization.

| Daemon Type             | Priority     | Description                                                                                                                     | Dependencies                  |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `IDLE_RESOURCE`         | **CRITICAL** | Monitors idle GPUs and auto-spawns selfplay jobs. Uses SelfplayScheduler priorities.                                            | EVENT_ROUTER                  |
| `QUEUE_POPULATOR`       | **CRITICAL** | Auto-populates work queue until Elo targets met (60% selfplay, 30% training, 10% tournament). Uses UnifiedQueuePopulatorDaemon. | EVENT_ROUTER                  |
| `NODE_RECOVERY`         | MEDIUM       | Auto-recovers terminated cluster nodes.                                                                                         | EVENT_ROUTER                  |
| `UTILIZATION_OPTIMIZER` | HIGH         | Matches GPU capabilities to board sizes. Stops CPU selfplay on GPU nodes.                                                       | EVENT_ROUTER, IDLE_RESOURCE   |
| `ADAPTIVE_RESOURCES`    | MEDIUM       | Dynamic resource scaling based on workload.                                                                                     | EVENT_ROUTER, CLUSTER_MONITOR |
| `MULTI_PROVIDER`        | MEDIUM       | Coordinates workloads across Vast/Nebius/RunPod/Vultr providers (Lambda legacy).                                                | EVENT_ROUTER, CLUSTER_MONITOR |

**Factory Methods:**

- `_create_idle_resource()` → Creates `IdleResourceDaemon`
- `_create_queue_populator()` → Creates `UnifiedQueuePopulatorDaemon`
- `_create_node_recovery()` → Creates `NodeRecoveryDaemon`
- `_create_utilization_optimizer()` → Creates `UtilizationOptimizer`
- `_create_adaptive_resources()` → Creates `AdaptiveResourceManager` (not implemented yet)
- `_create_multi_provider()` → Creates `MultiProviderOrchestrator` (not implemented yet)

---

### Pipeline Automation

Automated pipeline stage triggering.

| Daemon Type        | Priority | Description                                                                     | Dependencies              |
| ------------------ | -------- | ------------------------------------------------------------------------------- | ------------------------- |
| `AUTO_EXPORT`      | MEDIUM   | Auto-triggers NPZ export when game count thresholds are met.                    | EVENT_ROUTER              |
| `TRAINING_TRIGGER` | MEDIUM   | Decides when to auto-trigger training based on data freshness and availability. | EVENT_ROUTER, AUTO_EXPORT |
| `DATA_CLEANUP`     | LOW      | Auto-quarantines/deletes poor quality game databases.                           | None                      |

**Factory Methods:**

- `_create_auto_export()` → Creates `AutoExportDaemon`
- `_create_training_trigger()` → Creates `TrainingTriggerDaemon`
- `_create_data_cleanup()` → Creates `DataCleanupDaemon` (not implemented yet)

---

### System Health

System-level maintenance and monitoring.

| Daemon Type             | Priority | Description                                           | Dependencies                      |
| ----------------------- | -------- | ----------------------------------------------------- | --------------------------------- |
| `MAINTENANCE`           | LOW      | Log rotation, database vacuum, cleanup tasks.         | None                              |
| `METRICS_ANALYSIS`      | MEDIUM   | Continuous metrics monitoring and plateau detection.  | EVENT_ROUTER                      |
| `CACHE_COORDINATION`    | MEDIUM   | Coordinates model caching across the cluster.         | EVENT_ROUTER, CLUSTER_MONITOR     |
| `RECOVERY_ORCHESTRATOR` | HIGH     | Handles model/training state recovery after failures. | EVENT_ROUTER, NODE_HEALTH_MONITOR |

**Factory Methods:**

- `_create_maintenance()` → Creates `MaintenanceDaemon`
- `_create_metrics_analysis()` → Creates `MetricsAnalysisDaemon` (not implemented yet)
- `_create_cache_coordination()` → Creates `CacheCoordinationDaemon` (not implemented yet)
- `_create_recovery_orchestrator()` → Creates `RecoveryOrchestrator` (not implemented yet)

---

### Cost Optimization

Cloud provider cost management.

| Daemon Type   | Priority | Description                                                                                                              | Dependencies                  |
| ------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `LAMBDA_IDLE` | LOW      | Auto-terminates idle Lambda nodes to save costs. (NOTE: Lambda account terminated; code kept for historical restoration) | EVENT_ROUTER, CLUSTER_MONITOR |
| `VAST_IDLE`   | MEDIUM   | Auto-terminates idle Vast.ai nodes to save costs. Important for ephemeral marketplace instances.                         | EVENT_ROUTER, CLUSTER_MONITOR |

**Factory Methods:**

- `_create_lambda_idle()` → Creates `LambdaIdleDaemon`
- `_create_vast_idle()` → Creates `VastIdleDaemon`

---

### Feedback & Curriculum

Training feedback and curriculum learning.

| Daemon Type              | Priority     | Description                                                                                              | Dependencies |
| ------------------------ | ------------ | -------------------------------------------------------------------------------------------------------- | ------------ |
| `FEEDBACK_LOOP`          | **CRITICAL** | Orchestrates all training feedback signals (quality, performance, Elo velocity).                         | EVENT_ROUTER |
| `CURRICULUM_INTEGRATION` | MEDIUM       | Bridges all feedback loops for self-improvement. Adjusts training curriculum based on model performance. | EVENT_ROUTER |

**Factory Methods:**

- `_create_feedback_loop()` → Creates `FeedbackLoopController`
- `_create_curriculum_integration()` → Creates `CurriculumIntegrationDaemon`

---

### Disk Space Management (December 2025)

Disk space monitoring and cleanup for coordinator nodes.

| Daemon Type                | Priority | Description                                                                                    | Dependencies       |
| -------------------------- | -------- | ---------------------------------------------------------------------------------------------- | ------------------ |
| `DISK_SPACE_MANAGER`       | MEDIUM   | Proactive disk space monitoring. Cleanup at 60% usage (before 70% threshold).                  | EVENT_ROUTER       |
| `COORDINATOR_DISK_MANAGER` | HIGH     | Specialized disk manager for coordinator nodes. Auto-syncs to external storage before cleanup. | DISK_SPACE_MANAGER |

**COORDINATOR_DISK_MANAGER Details:**

Prevents coordinator node disk fill-up by syncing data to external storage (e.g., OWC drive on mac-studio):

- **Sync Target:** `/Volumes/RingRift-Data` on mac-studio
- **Data Routed:**
  - `data/games/*.db` → `RingRift-Data/selfplay_repository/`
  - `data/training/*.npz` → `RingRift-Data/canonical_data/`
  - `models/*.pth` → `RingRift-Data/canonical_models/`
- **Cleanup Threshold:** 50% disk usage (more aggressive than GPU nodes)
- **Retention:** Canonical databases kept locally; other data synced then removed after 24hr

**Configuration:**

- `RINGRIFT_COORDINATOR_REMOTE_HOST` - Remote host for sync (default: mac-studio)
- `RINGRIFT_COORDINATOR_REMOTE_PATH` - Remote path (default: /Volumes/RingRift-Data)
- `RINGRIFT_DISK_SPACE_CHECK_INTERVAL` - Check interval in seconds (default: 1800)

**Factory Methods:**

- `create_disk_space_manager()` → Creates `DiskSpaceManagerDaemon` (in daemon_runners.py)
- `create_coordinator_disk_manager()` → Creates `CoordinatorDiskManager` (in daemon_runners.py)

---

## Daemon Profiles

Profiles group daemons by node role for easier management.

### Coordinator Profile

Runs on the central coordinator node (typically MacBook M3).

**Daemon Count:** 35

**Daemons:**

- `EVENT_ROUTER` (CRITICAL)
- `HEALTH_SERVER`
- `P2P_BACKEND`
- `TOURNAMENT_DAEMON`
- `MODEL_DISTRIBUTION`
- `S3_BACKUP`
- `REPLICATION_MONITOR`
- `REPLICATION_REPAIR`
- `CLUSTER_MONITOR`
- `QUEUE_MONITOR`
- `FEEDBACK_LOOP` (CRITICAL)
- `QUALITY_MONITOR`
- `MODEL_PERFORMANCE_WATCHDOG`
- `NPZ_DISTRIBUTION`
- `ORPHAN_DETECTION`
- `NODE_HEALTH_MONITOR`
- `SYSTEM_HEALTH_MONITOR`
- `UNIFIED_PROMOTION`
- `JOB_SCHEDULER`
- `IDLE_RESOURCE` (CRITICAL)
- `NODE_RECOVERY`
- `LAMBDA_IDLE`
- `QUEUE_POPULATOR` (CRITICAL)
- `CURRICULUM_INTEGRATION`
- `AUTO_EXPORT`
- `TRAINING_TRIGGER`
- `DLQ_RETRY`
- `GAUNTLET_FEEDBACK`
- `AUTO_SYNC` (CRITICAL)
- `CLUSTER_DATA_SYNC`
- `CLUSTER_WATCHDOG`
- `METRICS_ANALYSIS`
- `DATA_CONSOLIDATION` (December 2025)
- `DISK_SPACE_MANAGER` (December 2025)
- `COORDINATOR_DISK_MANAGER` (December 2025)

**Use Case:** Centralized coordination, monitoring, and job scheduling for the entire cluster.

---

### Training Node Profile

Runs on GPU training nodes (H100, Nebius, RunPod, Vultr; Lambda legacy).

**Daemon Count:** 19

**Daemons:**

- `EVENT_ROUTER` (CRITICAL)
- `HEALTH_SERVER`
- `DATA_PIPELINE`
- `CONTINUOUS_TRAINING_LOOP`
- `AUTO_SYNC` (CRITICAL)
- `TRAINING_NODE_WATCHER`
- `EVALUATION`
- `QUALITY_MONITOR`
- `ORPHAN_DETECTION`
- `UNIFIED_PROMOTION`
- `P2P_AUTO_DEPLOY`
- `IDLE_RESOURCE` (CRITICAL)
- `UTILIZATION_OPTIMIZER`
- `CURRICULUM_INTEGRATION`
- `AUTO_EXPORT`
- `TRAINING_TRIGGER`
- `FEEDBACK_LOOP` (CRITICAL)
- `METRICS_ANALYSIS`
- `DLQ_RETRY`
- `DISK_SPACE_MANAGER` (December 2025)

**Use Case:** Dedicated training and evaluation on stable GPU nodes.

---

### Ephemeral Profile

Runs on ephemeral/spot instances (Vast.ai marketplace).

**Daemon Count:** 9

**Daemons:**

- `EVENT_ROUTER` (CRITICAL)
- `HEALTH_SERVER`
- `EPHEMERAL_SYNC` (5-second aggressive sync)
- `DATA_PIPELINE`
- `IDLE_RESOURCE` (CRITICAL)
- `QUALITY_MONITOR`
- `ORPHAN_DETECTION`
- `AUTO_SYNC` (CRITICAL)
- `FEEDBACK_LOOP` (CRITICAL)

**Use Case:** Ephemeral nodes with aggressive data sync to prevent loss on termination.

---

### Selfplay Profile

Runs on selfplay-only nodes (generates training data).

**Daemon Count:** 6

**Daemons:**

- `EVENT_ROUTER` (CRITICAL)
- `HEALTH_SERVER`
- `AUTO_SYNC` (CRITICAL)
- `QUALITY_MONITOR`
- `IDLE_RESOURCE` (CRITICAL)
- `FEEDBACK_LOOP` (CRITICAL)

**Use Case:** Dedicated selfplay generation with quality monitoring and resource utilization.

---

### Full Profile

All daemons (for testing/development).

**Daemon Count:** 60+

**Use Case:** Complete daemon suite for integration testing.

---

### Minimal Profile

Just event routing (for debugging).

**Daemon Count:** 1

**Daemons:**

- `EVENT_ROUTER`

**Use Case:** Minimal setup for debugging event-driven workflows.

---

## Dependency Graph

This graph shows which daemons depend on other daemons being started first.

```
EVENT_ROUTER (CRITICAL - no dependencies)
├── DLQ_RETRY
├── CONTINUOUS_TRAINING_LOOP
├── TOURNAMENT_DAEMON
├── DATA_PIPELINE
├── SELFPLAY_COORDINATOR
├── MODEL_DISTRIBUTION
├── UNIFIED_PROMOTION
├── FEEDBACK_LOOP (CRITICAL)
├── EVALUATION
│   └── AUTO_PROMOTION
├── QUALITY_MONITOR
├── MODEL_PERFORMANCE_WATCHDOG
├── NPZ_DISTRIBUTION
├── INTEGRITY_CHECK
├── DISTILLATION
├── CLUSTER_DATA_SYNC
├── JOB_SCHEDULER
├── IDLE_RESOURCE (CRITICAL)
│   └── UTILIZATION_OPTIMIZER
├── NODE_RECOVERY
├── P2P_AUTO_DEPLOY
├── QUEUE_POPULATOR (CRITICAL)
├── CURRICULUM_INTEGRATION
├── AUTO_EXPORT
│   └── TRAINING_TRIGGER
├── GAUNTLET_FEEDBACK
├── METRICS_ANALYSIS
├── TRAINING_DATA_SYNC
├── SYNC_PUSH
└── S3_NODE_SYNC
    └── S3_CONSOLIDATION

NODE_HEALTH_MONITOR
├── SYSTEM_HEALTH_MONITOR
└── RECOVERY_ORCHESTRATOR

CLUSTER_MONITOR
├── CACHE_COORDINATION
├── ADAPTIVE_RESOURCES
├── MULTI_PROVIDER
├── LAMBDA_IDLE
└── VAST_IDLE

MODEL_DISTRIBUTION
└── S3_BACKUP

EVALUATION
└── AUTO_PROMOTION

DATA_PIPELINE (CRITICAL - event subscribers before emitters!)
├── AUTO_SYNC (CRITICAL) - depends on DATA_PIPELINE
├── EPHEMERAL_SYNC - depends on DATA_PIPELINE
├── TRAINING_NODE_WATCHER - depends on DATA_PIPELINE
├── NPZ_DISTRIBUTION
├── DATA_CONSOLIDATION
└── NPZ_COMBINATION

FEEDBACK_LOOP (CRITICAL - subscribes to training events!)
└── AUTO_SYNC (CRITICAL) - depends on FEEDBACK_LOOP

December 2025 Critical Order:
- DATA_PIPELINE (position 3) must start BEFORE AUTO_SYNC (position 5)
- FEEDBACK_LOOP (position 4) must start BEFORE AUTO_SYNC (position 5)
- AUTO_SYNC emits DATA_SYNC_COMPLETED which DATA_PIPELINE must catch

DISK_SPACE_MANAGER
└── COORDINATOR_DISK_MANAGER - depends on DISK_SPACE_MANAGER

Standalone (no dependencies):
- SYNC_COORDINATOR (deprecated)
- HIGH_QUALITY_SYNC
- ELO_SYNC
- GOSSIP_SYNC
- MODEL_SYNC
- EXTERNAL_DRIVE_SYNC
- REPLICATION_MONITOR
- REPLICATION_REPAIR
- HEALTH_CHECK (deprecated)
- ORPHAN_DETECTION
- P2P_BACKEND
- QUEUE_MONITOR
- RESOURCE_OPTIMIZER
- VAST_CPU_PIPELINE
- DATA_SERVER
- MAINTENANCE
- CLUSTER_WATCHDOG
- CROSS_PROCESS_POLLER
- DAEMON_WATCHDOG
- HEALTH_SERVER
- DATA_CLEANUP
- DISK_SPACE_MANAGER
```

---

## Startup Ordering

**Critical:** Event subscribers must start before event emitters to prevent lost events.

The canonical startup order is defined in `DAEMON_STARTUP_ORDER` (daemon_types.py:367-403).
This ensures daemons start in dependency-safe order.

```
============================================================================
DAEMON_STARTUP_ORDER (18 daemons) - December 27, 2025 fix
============================================================================

Core Infrastructure (positions 1-4)
 1. EVENT_ROUTER        # Event system must be first
 2. DAEMON_WATCHDOG     # Self-healing for daemon crashes
 3. DATA_PIPELINE       # Pipeline processor (before sync!)
 4. FEEDBACK_LOOP       # Training feedback (before sync!)

Sync and Queue Management (positions 5-10)
 5. AUTO_SYNC           # Data sync (emits events)
 6. QUEUE_POPULATOR     # Work queue maintenance
 7. WORK_QUEUE_MONITOR  # Queue visibility (after populator)
 8. COORDINATOR_HEALTH  # Coordinator visibility
 9. IDLE_RESOURCE       # GPU utilization
10. TRAINING_TRIGGER    # Training trigger (after pipeline)

Monitoring Daemons (positions 11-15)
11. CLUSTER_MONITOR     # Cluster monitoring (depends on EVENT_ROUTER)
12. NODE_HEALTH_MONITOR # Node health (depends on EVENT_ROUTER)
13. HEALTH_SERVER       # Health endpoints (depends on EVENT_ROUTER)
14. CLUSTER_WATCHDOG    # Cluster watchdog (depends on CLUSTER_MONITOR)
15. NODE_RECOVERY       # Node recovery (depends on NODE_HEALTH_MONITOR)

Evaluation and Promotion Chain (positions 16-18)
16. EVALUATION          # Model evaluation (depends on TRAINING_TRIGGER)
17. AUTO_PROMOTION      # Auto-promotion (depends on EVALUATION)
18. MODEL_DISTRIBUTION  # Model distribution (depends on AUTO_PROMOTION)
```

**Validation:** The startup order is validated against `DAEMON_DEPENDENCIES` at startup
using `validate_startup_order_consistency()`. Any violations cause startup failure.

**Why order matters:** If `AUTO_SYNC` starts before `DATA_PIPELINE`, the `DATA_SYNC_COMPLETED`
events are lost because the subscriber isn't running yet. This causes the pipeline to miss
sync completions and never trigger NPZ exports.

**Auto-ready timeout:** Daemons are marked "ready" after 2s initialization delay (increased
from 0.5s in Dec 27, 2025 fix). Daemons can call `mark_daemon_ready()` earlier for explicit signaling.

**Implementation:** See `daemon_types.py:DAEMON_STARTUP_ORDER` and `scripts/master_loop.py:_get_daemon_list()`.

---

## Priority Levels

Daemons are categorized by priority for health monitoring and restart behavior.

### CRITICAL Priority

These daemons get 15-second health check intervals (vs 60s for others):

1. **EVENT_ROUTER** - Core event bus. All coordination depends on this.
2. **AUTO_SYNC** - Primary data sync mechanism. Ensures fresh data.
3. **QUEUE_POPULATOR** - Keeps work queue populated. Prevents idle cluster.
4. **IDLE_RESOURCE** - Ensures GPUs stay utilized. Prevents waste.
5. **FEEDBACK_LOOP** - Coordinates training feedback signals.

**Health Check Interval:** 15 seconds
**Auto-Restart:** Yes (with exponential backoff, max 5 attempts)

### HIGH Priority

Important daemons for cluster operation:

- P2P_BACKEND
- NODE_HEALTH_MONITOR
- HEALTH_SERVER
- DATA_PIPELINE
- UNIFIED_PROMOTION
- AUTO_PROMOTION
- EVALUATION
- SYSTEM_HEALTH_MONITOR
- UTILIZATION_OPTIMIZER
- RECOVERY_ORCHESTRATOR
- CLUSTER_WATCHDOG
- JOB_SCHEDULER
- EPHEMERAL_SYNC

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes

### MEDIUM Priority

Standard operational daemons:

- All sync daemons (GOSSIP_SYNC, MODEL_SYNC, etc.)
- Training support (CONTINUOUS_TRAINING_LOOP, DISTILLATION)
- Monitoring (QUALITY_MONITOR, CLUSTER_MONITOR, etc.)
- Pipeline automation (AUTO_EXPORT, TRAINING_TRIGGER)
- Tournaments and evaluation (TOURNAMENT_DAEMON, GAUNTLET_FEEDBACK)
- Resource management (NODE_RECOVERY, VAST_IDLE)

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes

### LOW Priority

Optional/non-critical daemons:

- ORPHAN_DETECTION
- EXTERNAL_DRIVE_SYNC
- VAST_CPU_PIPELINE
- MAINTENANCE
- DISTILLATION
- LAMBDA_IDLE
- DATA_CLEANUP

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes (but lower urgency)

---

## Factory Methods Reference

All daemon factory methods follow the pattern `_create_<daemon_name>()`. They either:

1. **Direct instantiation:** Create the daemon class directly
2. **Factory function:** Call a module-level factory function (e.g., `get_router()`, `get_evaluation_daemon()`)
3. **Not implemented:** Return a placeholder/not implemented error

### Import Paths

See `daemon_factory.py` for the complete registry mapping daemon types to:

- Import path (e.g., `app.coordination.auto_sync_daemon`)
- Class name (e.g., `AutoSyncDaemon`)
- Optional factory function (e.g., `get_auto_sync_daemon`)

Example:

```python
DaemonType.AUTO_SYNC: DaemonSpec(
    import_path="app.coordination.auto_sync_daemon",
    class_name="AutoSyncDaemon",
    singleton=True,
)
```

---

## Usage Examples

### Starting a Daemon Profile

```python
from app.coordination.daemon_manager import get_daemon_manager, start_profile

# Start coordinator profile
await start_profile("coordinator")

# Or manually
manager = get_daemon_manager()
await manager.start_all(DAEMON_PROFILES["training_node"])
```

### Starting Individual Daemons

```python
from app.coordination.daemon_manager import get_daemon_manager, DaemonType

manager = get_daemon_manager()

# Start a single daemon
await manager.start(DaemonType.AUTO_SYNC)

# Start multiple daemons
await manager.start_all([
    DaemonType.EVENT_ROUTER,
    DaemonType.FEEDBACK_LOOP,
    DaemonType.IDLE_RESOURCE,
])
```

### Checking Daemon Status

```python
manager = get_daemon_manager()

# Get all daemon statuses
status = manager.get_status()
for daemon_type, info in status.items():
    print(f"{daemon_type.value}: {info.state.value} (uptime: {info.uptime_seconds}s)")

# Check specific daemon
info = manager.get_daemon_info(DaemonType.AUTO_SYNC)
if info.state == DaemonState.RUNNING:
    print(f"AUTO_SYNC running for {info.uptime_seconds}s")
```

### Graceful Shutdown

```python
manager = get_daemon_manager()

# Shutdown all daemons
await manager.shutdown()

# Shutdown specific daemon
await manager.stop(DaemonType.CLUSTER_MONITOR)
```

---

## Configuration

Daemon behavior is controlled by `DaemonManagerConfig`:

```python
from app.coordination.daemon_types import DaemonManagerConfig

config = DaemonManagerConfig(
    auto_start=False,  # Don't auto-start on init
    health_check_interval=30.0,  # Global health check interval
    critical_daemon_health_interval=15.0,  # Faster for critical daemons
    shutdown_timeout=10.0,  # Max time for graceful shutdown
    auto_restart_failed=True,  # Auto-restart failed daemons
    max_restart_attempts=5,  # Max restart attempts per daemon
    recovery_cooldown=10.0,  # Time before attempting recovery (reduced from 300s)
)

manager = get_daemon_manager(config)
```

---

## Environment Variables

Daemon-related environment variables (via `app.config.env`):

- `RINGRIFT_NODE_ROLE` - Node role: `coordinator`, `training_node`, `ephemeral`, `selfplay`
- `RINGRIFT_IS_COORDINATOR` - Boolean flag for coordinator node
- `RINGRIFT_DAEMON_PROFILE` - Override auto-detected profile
- `RINGRIFT_SKIP_DAEMONS` - Comma-separated list of daemons to skip
- `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` - Grace period for P2P startup (default: 120s)

### Coordination Defaults (December 2025)

Configuration defaults are centralized in `app/config/coordination_defaults.py`. Key classes:

| Class                  | Purpose            | Key Settings                                                      |
| ---------------------- | ------------------ | ----------------------------------------------------------------- |
| `P2PDefaults`          | P2P network config | `DEFAULT_PORT=8770`, `HEARTBEAT_INTERVAL=15s`, `PEER_TIMEOUT=60s` |
| `JobTimeoutDefaults`   | Per-job timeouts   | `GPU_SELFPLAY=1hr`, `TRAINING=4hr`, `TOURNAMENT=1hr`              |
| `BackpressureDefaults` | Spawn rate control | Component weights, queue thresholds, multipliers                  |
| `DaemonHealthDefaults` | Health monitoring  | `CHECK_INTERVAL=60s`, `MAX_FAILURES=3`                            |
| `SQLiteDefaults`       | Database timeouts  | `READ=5s`, `WRITE=30s`, `HEAVY=60s`                               |

```python
from app.config.coordination_defaults import (
    P2PDefaults,
    JobTimeoutDefaults,
    BackpressureDefaults,
    get_p2p_port,
    get_job_timeout,
)

# Get P2P port (env override: RINGRIFT_P2P_PORT)
port = get_p2p_port()  # 8770

# Get job timeout (env override: RINGRIFT_JOB_TIMEOUT_TRAINING)
timeout = get_job_timeout("training")  # 14400 seconds
```

---

### Base Handler Classes (December 2025)

Event handlers should inherit from the canonical base in `app/coordination/handler_base.py`.
Legacy wrappers in `base_event_handler.py` and `base_handler.py` remain for compatibility but
are deprecated (Q2 2026).

| Class                  | Purpose                                                                    | Use Case                |
| ---------------------- | -------------------------------------------------------------------------- | ----------------------- |
| `HandlerBase`          | Canonical base for handlers (subscription, stats, lifecycle)               | New handlers            |
| `HandlerStats`         | Unified statistics dataclass (success_rate, errors_count, last_event_time) | Handler metrics         |
| `EventHandlerConfig`   | Handler configuration (async/sync, timeouts)                               | Custom handler behavior |
| `BaseEventHandler`     | Legacy alias for HandlerBase (backward-compatible)                         | Existing handlers       |
| `BaseSingletonHandler` | Legacy alias for HandlerBase (singleton-style usage)                       | Existing handlers       |
| `MultiEventHandler`    | Legacy alias for HandlerBase (multi-event patterns)                        | Existing handlers       |

**When to use:**

- New handlers that subscribe to `DataEventType` events via the event bus
- Handlers that need consistent stats tracking and health reporting
- Singleton handlers that need thread-safe instance management

**When NOT to use:**

- Utility classes that don't subscribe to events (e.g., `SyncStallHandler`)
- Decorators/wrappers (e.g., `handler_resilience.py`)
- Handlers with sync-to-async patterns (existing code with `fire_and_forget`)

```python
from app.coordination.handler_base import HandlerBase, HandlerStats

class MyHandler(HandlerBase):
    def __init__(self):
        super().__init__("MyHandler")

    def _do_subscribe(self) -> bool:
        from app.coordination.event_router import DataEventType, get_event_bus
        bus = get_event_bus()
        bus.subscribe(DataEventType.MY_EVENT, self._handle_event)
        self._subscribed = True
        return True

    async def _handle_event(self, event) -> None:
        # Business logic here
        self._record_success()  # Tracks stats automatically
```

**Legacy helper functions (from base_handler.py):**

- `create_handler_stats(**custom)` - Create HandlerStats with custom stats
- `safe_subscribe(handler)` - Subscribe with exception handling

---

## Deprecation Notices

### SYNC_COORDINATOR (Q2 2026)

**Replacement:** `AUTO_SYNC`
**Status:** Deprecated December 2025, removal Q2 2026
**Migration:** Replace all references to `DaemonType.SYNC_COORDINATOR` with `DaemonType.AUTO_SYNC`

### HEALTH_CHECK (Q2 2026)

**Replacement:** `NODE_HEALTH_MONITOR`
**Status:** Deprecated December 2025, removal Q2 2026
**Migration:** Replace all references to `DaemonType.HEALTH_CHECK` with `DaemonType.NODE_HEALTH_MONITOR`

---

## Infrastructure Verification (December 2025)

Automated verification confirmed the following architecture is properly configured:

### Daemon Coverage

| Metric                         | Value | Status                |
| ------------------------------ | ----- | --------------------- |
| Total DaemonType values        | 66    | ✓ All accounted       |
| Runners in `daemon_runners.py` | 65    | ✓ Complete            |
| Inline runners (HEALTH_SERVER) | 1     | ✓ Needs `self` access |
| Missing runners                | 0     | ✓ None                |

### Startup Order

| Metric                             | Value  | Notes                                  |
| ---------------------------------- | ------ | -------------------------------------- |
| Daemons in `DAEMON_STARTUP_ORDER`  | 18     | Critical path daemons                  |
| Daemons with `DAEMON_DEPENDENCIES` | 66     | All have deps defined                  |
| Order/Dependency consistency       | Passes | `validate_startup_order_consistency()` |

### P2P Event Subscriptions

The P2P orchestrator subscribes to 18+ events across three subscription methods:

- `_subscribe_to_daemon_events`: DAEMON_STATUS_CHANGED
- `_subscribe_to_feedback_signals`: QUALITY_DEGRADED, ELO_VELOCITY_CHANGED, EVALUATION_COMPLETED, PLATEAU_DETECTED, EXPLORATION_BOOST, PROMOTION_FAILED, HANDLER_FAILED
- `_subscribe_to_manager_events`: TRAINING_STARTED, TRAINING_COMPLETED, TASK_SPAWNED, TASK_COMPLETED, TASK_FAILED, DATA_SYNC_STARTED, DATA_SYNC_COMPLETED, NODE_UNHEALTHY, NODE_RECOVERED, P2P_CLUSTER_HEALTHY, P2P_CLUSTER_UNHEALTHY

### Circular Dependency Status

Previously identified circular dependencies have been resolved:

| Modules                                          | Resolution                        | Status  |
| ------------------------------------------------ | --------------------------------- | ------- |
| `selfplay_scheduler` ↔ `unified_queue_populator` | TYPE_CHECKING guard + lazy import | ✓ Fixed |

### Verification Commands

```bash
# Verify all daemons have runners
cd ai-service && PYTHONPATH=. python3 -c "
from app.coordination.daemon_types import DaemonType
from app.coordination.daemon_runners import get_all_runners
print(f'Total: {len(DaemonType)}, Runners: {len(get_all_runners())}')
"

# Verify startup order consistency
cd ai-service && PYTHONPATH=. python3 -c "
from app.coordination.daemon_types import validate_startup_order_consistency
valid, errors = validate_startup_order_consistency()
print('✓ Valid' if valid else f'✗ Errors: {errors}')
"
```

---

## See Also

- `daemon_manager.py` - Main daemon lifecycle management
- `daemon_runners.py` - 65 async runner functions for daemon types
- `daemon_types.py` - Type definitions and enums
- `daemon_factory.py` - Centralized daemon creation factory
- `daemon_adapters.py` - Daemon wrappers for legacy code
- `handler_base.py` - Canonical base classes for event handlers (Dec 2025)
- `base_handler.py` - Legacy helpers (deprecated, Q2 2026)
- `app/config/coordination_defaults.py` - Centralized configuration defaults (Dec 2025)
- `CONFIG_REFERENCE.md` - Environment variable configuration
- `CLAUDE.md` - Cluster infrastructure overview
