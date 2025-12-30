# Daemon Registry

This document provides a comprehensive reference for all daemons managed by the RingRift AI service `DaemonManager`.

**Last updated:** December 30, 2025 (availability + connectivity daemons)
**Total Daemon Types:** 85 (85 in `daemon_runners.py`, 11 deprecated)
**Startup Order:** 21 daemons in `DAEMON_STARTUP_ORDER` (see `daemon_types.py`)
**Dependencies:** Canonical dependencies live in `DAEMON_REGISTRY` (85 entries); `DAEMON_DEPENDENCIES` covers 79 legacy entries

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
| `CROSS_PROCESS_POLLER` | HIGH         | Polls for events from other processes and forwards them to the event router.                                                     | EVENT_ROUTER |
| `DAEMON_WATCHDOG`      | HIGH         | Monitors daemon health and triggers restarts when failures are detected.                                                         | EVENT_ROUTER |
| `DLQ_RETRY`            | MEDIUM       | Dead letter queue remediation. Retries failed events from the DLQ.                                                               | EVENT_ROUTER |

**Factory Methods:**

- `create_event_router()` → Uses `event_router.get_router()`
- `create_cross_process_poller()` → Creates `CrossProcessEventPoller`
- `create_daemon_watchdog()` → Uses `daemon_watchdog.start_watchdog()`
- `create_dlq_retry()` → Creates `DLQRetryDaemon`

---

### Sync Daemons

Data synchronization across the cluster.

| Daemon Type             | Priority     | Description                                                                                                                | Dependencies                               |
| ----------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `AUTO_SYNC`             | **CRITICAL** | Primary data sync mechanism. Pulls game data to coordinator; includes min-move completeness checks to avoid mid-write DBs. | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP |
| `UNIFIED_DATA_PLANE`    | HIGH         | Unified data plane consolidating sync planners, transport, and event bridging.                                             | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP |
| `SYNC_COORDINATOR`      | DEPRECATED   | Legacy sync coordinator. Use `AUTO_SYNC` instead. Scheduled for removal Q2 2026.                                           | EVENT_ROUTER                               |
| `GOSSIP_SYNC`           | MEDIUM       | P2P gossip-based data synchronization for eventual consistency.                                                            | EVENT_ROUTER                               |
| `EPHEMERAL_SYNC`        | DEPRECATED   | Legacy 5-second sync for ephemeral nodes. Use AutoSyncDaemon(strategy="ephemeral").                                        | EVENT_ROUTER, DATA_PIPELINE                |
| `MODEL_SYNC`            | MEDIUM       | Syncs trained models across the cluster.                                                                                   | EVENT_ROUTER                               |
| `MODEL_DISTRIBUTION`    | MEDIUM       | Auto-distributes models after promotion. Subscribes to MODEL_PROMOTED events.                                              | EVENT_ROUTER, EVALUATION, AUTO_PROMOTION   |
| `NPZ_DISTRIBUTION`      | DEPRECATED   | Legacy NPZ sync. Use unified_distribution_daemon.py with DataType.NPZ.                                                     | EVENT_ROUTER                               |
| `EXTERNAL_DRIVE_SYNC`   | LOW          | Backup to external drives for disaster recovery.                                                                           | EVENT_ROUTER                               |
| `CLUSTER_DATA_SYNC`     | DEPRECATED   | Legacy cluster-wide sync; use AutoSyncDaemon(strategy="broadcast").                                                        | EVENT_ROUTER                               |
| `TRAINING_NODE_WATCHER` | MEDIUM       | Detects active training processes, triggers priority sync, and tags training-active nodes for SyncRouter prioritization.   | EVENT_ROUTER, DATA_PIPELINE                |
| `HIGH_QUALITY_SYNC`     | MEDIUM       | Priority sync for high-quality game data (quality score > 0.7).                                                            | EVENT_ROUTER                               |
| `ELO_SYNC`              | MEDIUM       | Synchronize ELO ratings across cluster.                                                                                    | EVENT_ROUTER                               |
| `TRAINING_DATA_SYNC`    | MEDIUM       | Pre-training data sync from OWC/S3. Ensures training nodes have fresh data before training.                                | EVENT_ROUTER                               |
| `OWC_IMPORT`            | LOW          | Periodic import from the OWC external archive for underserved configs.                                                     | EVENT_ROUTER, DATA_PIPELINE                |
| `SYNC_PUSH`             | MEDIUM       | Push-based sync for GPU nodes. Pushes data to coordinator before cleanup to prevent data loss.                             | EVENT_ROUTER                               |
| `S3_NODE_SYNC`          | MEDIUM       | Bi-directional S3 sync for cluster nodes. Syncs game data, models, and training files to/from S3.                          | EVENT_ROUTER                               |
| `S3_CONSOLIDATION`      | LOW          | Consolidates data from all nodes to S3 (coordinator only). Runs after S3_NODE_SYNC.                                        | EVENT_ROUTER, S3_NODE_SYNC                 |

**Factory Methods:**

- `create_auto_sync()` → Creates `AutoSyncDaemon`
- `create_unified_data_plane()` → Creates `UnifiedDataPlaneDaemon`
- `create_sync_coordinator()` → Creates `SyncCoordinator` (deprecated)
- `create_gossip_sync()` → Creates `GossipSyncDaemon`
- `create_ephemeral_sync()` → Uses `ephemeral_sync.get_ephemeral_sync_daemon()` (deprecated)
- `create_model_sync()` → Creates `ModelSyncDaemon`
- `create_model_distribution()` → Creates `ModelDistributionDaemon`
- `create_npz_distribution()` → Creates `NPZDistributionDaemon` (deprecated)
- `create_external_drive_sync()` → Creates `ExternalDriveSyncDaemon`
- `create_cluster_data_sync()` → AutoSyncDaemon(strategy="broadcast") (deprecated wrapper)
- `create_training_node_watcher()` → Creates `TrainingActivityDaemon` (in `training_activity_daemon.py`)
- `create_high_quality_sync()` → Creates `HighQualitySyncDaemon`
- `create_elo_sync()` → Creates `EloSyncDaemon`
- `create_training_data_sync()` → Creates `TrainingDataSyncDaemon`
- `create_owc_import()` → Creates `OWCImportDaemon`
- `create_sync_push()` → Creates `SyncPushDaemon`
- `create_s3_node_sync()` → Creates `S3NodeSyncDaemon`
- `create_s3_consolidation()` → Creates `S3ConsolidationDaemon`

---

### Training & Pipeline

Training pipeline orchestration and coordination.

| Daemon Type                | Priority | Description                                                                                                        | Dependencies                |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| `DATA_PIPELINE`            | HIGH     | Orchestrates pipeline stages: selfplay → sync → export → train → evaluate → promote.                               | EVENT_ROUTER                |
| `DATA_CONSOLIDATION`       | HIGH     | Merges scattered selfplay games into canonical databases. Runs after sync, before training.                        | EVENT_ROUTER, DATA_PIPELINE |
| `CONTINUOUS_TRAINING_LOOP` | MEDIUM   | Continuous training loop that runs indefinitely.                                                                   | EVENT_ROUTER                |
| `NNUE_TRAINING`            | MEDIUM   | Auto-trains NNUE models when per-config game thresholds are met.                                                   | EVENT_ROUTER, DATA_PIPELINE |
| `PER_ORCHESTRATOR`         | MEDIUM   | Monitors prioritized experience replay (PER) buffers and orchestrates sampling refresh.                            | EVENT_ROUTER                |
| `UNIFIED_PROMOTION`        | HIGH     | Auto-promotes models after evaluation. Subscribes to EVALUATION_COMPLETED events.                                  | EVENT_ROUTER                |
| `AUTO_PROMOTION`           | HIGH     | Auto-promotes models based on evaluation thresholds. Emits MODEL_PROMOTED.                                         | EVENT_ROUTER, EVALUATION    |
| `DISTILLATION`             | LOW      | Creates smaller student models from larger teacher models for deployment.                                          | EVENT_ROUTER                |
| `SELFPLAY_COORDINATOR`     | MEDIUM   | Distributes selfplay workloads across the cluster.                                                                 | EVENT_ROUTER                |
| `CASCADE_TRAINING`         | MEDIUM   | Orchestrates 2p→3p→4p training cascade with transfer learning between tiers.                                       | EVENT_ROUTER, DATA_PIPELINE |
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

- `create_data_pipeline()` → Creates `DataPipelineOrchestrator`
- `create_data_consolidation()` → Creates `DataConsolidationDaemon`
- `create_continuous_training_loop()` → Creates `ContinuousTrainingLoop`
- `create_nnue_training()` → Creates `NNUETrainingDaemon`
- `create_per_orchestrator()` → Wires `per_orchestrator` events via `wire_per_events()`
- `create_unified_promotion()` → Creates `PromotionController`
- `create_auto_promotion()` → Uses `auto_promotion_daemon.get_auto_promotion_daemon()`
- `create_distillation()` → Creates `DistillationDaemon`
- `create_selfplay_coordinator()` → Creates `SelfplayScheduler`
- `create_cascade_training()` → Creates `CascadeTrainingOrchestrator`
- `create_npz_combination()` → Creates `NPZCombinationDaemon`

---

### Evaluation & Tournament

Model evaluation and tournament scheduling.

| Daemon Type         | Priority | Description                                                                                      | Dependencies |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------ | ------------ |
| `EVALUATION`        | HIGH     | Auto-triggers evaluation after TRAINING_COMPLETE events.                                         | EVENT_ROUTER |
| `TOURNAMENT_DAEMON` | MEDIUM   | Automatic tournament scheduling for model comparison.                                            | EVENT_ROUTER |
| `GAUNTLET_FEEDBACK` | MEDIUM   | Bridges gauntlet evaluation results to training feedback. Emits REGRESSION_CRITICAL on failures. | EVENT_ROUTER |

**Factory Methods:**

- `create_evaluation_daemon()` → Uses `evaluation_daemon.get_evaluation_daemon()`
- `create_tournament_daemon()` → Uses `tournament_daemon.get_tournament_daemon()`
- `create_gauntlet_feedback()` → Creates `GauntletFeedbackController`

---

### Health & Monitoring

Cluster health monitoring and alerting.

| Daemon Type                  | Priority   | Description                                                                                                                                                    | Dependencies                       |
| ---------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `NODE_HEALTH_MONITOR`        | DEPRECATED | Legacy node health monitor. Use `health_check_orchestrator.py` instead.                                                                                        | EVENT_ROUTER                       |
| `COORDINATOR_HEALTH_MONITOR` | HIGH       | Tracks coordinator lifecycle events (healthy/unhealthy/degraded, heartbeat, init).                                                                             | EVENT_ROUTER                       |
| `AVAILABILITY_NODE_MONITOR`  | HIGH       | Multi-layer node monitor for availability (P2P, SSH, GPU, provider API).                                                                                       | EVENT_ROUTER                       |
| `WORK_QUEUE_MONITOR`         | HIGH       | Tracks work queue lifecycle (depth, latency, stuck jobs, backpressure).                                                                                        | EVENT_ROUTER, QUEUE_POPULATOR      |
| `HEALTH_CHECK`               | DEPRECATED | Legacy health checker. Use `NODE_HEALTH_MONITOR` instead. Scheduled for removal Q2 2026.                                                                       | None                               |
| `QUALITY_MONITOR`            | MEDIUM     | Continuous selfplay data quality monitoring. Triggers throttling feedback.                                                                                     | EVENT_ROUTER                       |
| `MODEL_PERFORMANCE_WATCHDOG` | MEDIUM     | Monitors model win rates and performance metrics.                                                                                                              | EVENT_ROUTER                       |
| `PROGRESS_WATCHDOG`          | MEDIUM     | Detects Elo velocity stalls and emits recovery signals for training plateaus.                                                                                  | EVENT_ROUTER, SELFPLAY_COORDINATOR |
| `P2P_RECOVERY`               | MEDIUM     | Auto-restarts P2P orchestration on partition/failure to maintain cluster mesh stability.                                                                       | EVENT_ROUTER                       |
| `TAILSCALE_HEALTH`           | MEDIUM     | Monitors Tailscale connectivity and auto-recovers on disconnects.                                                                                              | None                               |
| `ORPHAN_DETECTION`           | LOW        | Detects orphaned game databases not in the cluster manifest.                                                                                                   | EVENT_ROUTER                       |
| `SYSTEM_HEALTH_MONITOR`      | DEPRECATED | Legacy system health scoring and pipeline pause. Use `unified_health_manager.py` instead.                                                                      | EVENT_ROUTER, NODE_HEALTH_MONITOR  |
| `HEALTH_SERVER`              | HIGH       | HTTP endpoints: /health, /ready, /metrics for external monitoring.                                                                                             | EVENT_ROUTER                       |
| `INTEGRITY_CHECK`            | MEDIUM     | Data integrity checking daemon. Scans for games without moves, orphaned databases, and schema violations. Quarantines invalid games to `orphaned_games` table. | EVENT_ROUTER                       |

**Factory Methods:**

- `create_node_health_monitor()` → Creates `UnifiedNodeHealthDaemon` (deprecated)
- `create_coordinator_health_monitor()` → Creates `CoordinatorHealthMonitorDaemon`
- `create_availability_node_monitor()` → Creates `AvailabilityNodeMonitor`
- `create_work_queue_monitor()` → Creates `WorkQueueMonitorDaemon`
- `create_health_check()` → Creates `HealthChecker` (deprecated)
- `create_quality_monitor()` → Uses `quality_monitor_daemon.create_quality_monitor()`
- `create_model_performance_watchdog()` → Creates `ModelPerformanceWatchdog`
- `create_progress_watchdog()` → Creates `ProgressWatchdogDaemon`
- `create_p2p_recovery()` → Creates `P2PRecoveryDaemon`
- `create_tailscale_health()` → Creates `TailscaleHealthDaemon`
- `create_orphan_detection()` → Creates `OrphanDetectionDaemon`
- `create_system_health_monitor()` → Creates `SystemHealthMonitorDaemon` (deprecated)
- `create_health_server()` → Creates `HealthServerDaemon`
- `create_integrity_check()` → Creates `IntegrityCheckDaemon`

---

### Cluster Management

P2P cluster coordination and monitoring.

| Daemon Type           | Priority   | Description                                                                 | Dependencies                  |
| --------------------- | ---------- | --------------------------------------------------------------------------- | ----------------------------- |
| `CLUSTER_MONITOR`     | MEDIUM     | Real-time cluster monitoring (game counts, disk usage, training processes). | EVENT_ROUTER                  |
| `P2P_BACKEND`         | HIGH       | P2P node coordination and leader election.                                  | EVENT_ROUTER                  |
| `QUEUE_MONITOR`       | MEDIUM     | Monitors queue depths and applies backpressure.                             | EVENT_ROUTER                  |
| `REPLICATION_MONITOR` | DEPRECATED | Legacy replication monitor. Use unified_replication_daemon.py instead.      | EVENT_ROUTER                  |
| `REPLICATION_REPAIR`  | DEPRECATED | Legacy replication repair. Use unified_replication_daemon.py instead.       | EVENT_ROUTER                  |
| `P2P_AUTO_DEPLOY`     | MEDIUM     | Ensures P2P orchestrator runs on all cluster nodes.                         | EVENT_ROUTER                  |
| `CLUSTER_WATCHDOG`    | HIGH       | Self-healing cluster utilization monitor. Standalone daemon on coordinator. | EVENT_ROUTER, CLUSTER_MONITOR |

**Factory Methods:**

- `create_cluster_monitor()` → Creates `ClusterMonitor`
- `create_p2p_backend()` → Creates `P2PNode`
- `create_queue_monitor()` → Creates `QueueMonitor`
- `create_replication_monitor()` → Uses `replication_monitor.get_replication_monitor()`
- `create_replication_repair()` → Uses `replication_repair_daemon.get_replication_repair_daemon()`
- `create_p2p_auto_deploy()` → Creates `P2PAutoDeployDaemon`
- `create_cluster_watchdog()` → Creates `ClusterWatchdogDaemon`

---

### Job Scheduling

Resource allocation and job scheduling.

| Daemon Type          | Priority | Description                                                    | Dependencies                |
| -------------------- | -------- | -------------------------------------------------------------- | --------------------------- |
| `JOB_SCHEDULER`      | HIGH     | Centralized job scheduling with PID-based resource allocation. | EVENT_ROUTER                |
| `RESOURCE_OPTIMIZER` | MEDIUM   | Optimizes resource allocation across the cluster.              | EVENT_ROUTER, JOB_SCHEDULER |
| `VAST_CPU_PIPELINE`  | LOW      | CPU-only preprocessing pipeline for Vast.ai CPU nodes.         | EVENT_ROUTER                |

**Factory Methods:**

- `create_job_scheduler()` → Creates `JobScheduler`
- `create_resource_optimizer()` → Creates `ResourceOptimizer` (not implemented yet)
- `create_vast_cpu_pipeline()` → Creates `VastCpuPipelineDaemon`

---

### Backup & Storage

Data backup and external storage.

| Daemon Type   | Priority | Description                                                                     | Dependencies                     |
| ------------- | -------- | ------------------------------------------------------------------------------- | -------------------------------- |
| `S3_BACKUP`   | MEDIUM   | Backs up models to S3 after promotion. Runs after MODEL_DISTRIBUTION completes. | EVENT_ROUTER, MODEL_DISTRIBUTION |
| `DATA_SERVER` | MEDIUM   | HTTP server for serving game data and models over P2P network (port 8771).      | EVENT_ROUTER                     |

**Factory Methods:**

- `create_s3_backup()` → Creates `S3BackupDaemon`
- `create_data_server()` → Creates `DataServer`

---

### Resource & Utilization

GPU/CPU resource optimization.

| Daemon Type                     | Priority     | Description                                                                                                                     | Dependencies                                |
| ------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `IDLE_RESOURCE`                 | **CRITICAL** | Monitors idle GPUs and auto-spawns selfplay jobs. Uses SelfplayScheduler priorities.                                            | EVENT_ROUTER                                |
| `QUEUE_POPULATOR`               | **CRITICAL** | Auto-populates work queue until Elo targets met (60% selfplay, 30% training, 10% tournament). Uses UnifiedQueuePopulatorDaemon. | EVENT_ROUTER, SELFPLAY_COORDINATOR          |
| `NODE_RECOVERY`                 | MEDIUM       | Auto-recovers terminated cluster nodes.                                                                                         | EVENT_ROUTER                                |
| `UTILIZATION_OPTIMIZER`         | HIGH         | Matches GPU capabilities to board sizes. Stops CPU selfplay on GPU nodes.                                                       | EVENT_ROUTER, IDLE_RESOURCE                 |
| `ADAPTIVE_RESOURCES`            | MEDIUM       | Dynamic resource scaling based on workload.                                                                                     | EVENT_ROUTER, CLUSTER_MONITOR               |
| `MULTI_PROVIDER`                | MEDIUM       | Coordinates workloads across Vast/Nebius/RunPod/Vultr providers (Lambda legacy).                                                | EVENT_ROUTER, CLUSTER_MONITOR               |
| `NODE_AVAILABILITY`             | MEDIUM       | Syncs provider instance state with distributed_hosts.yaml to prevent stale node config.                                         | EVENT_ROUTER                                |
| `AVAILABILITY_CAPACITY_PLANNER` | MEDIUM       | Budget-aware capacity planning and scaling recommendations.                                                                     | EVENT_ROUTER                                |
| `AVAILABILITY_PROVISIONER`      | MEDIUM       | Auto-provisions new instances when capacity drops below thresholds.                                                             | EVENT_ROUTER, AVAILABILITY_CAPACITY_PLANNER |

**Factory Methods:**

- `create_idle_resource()` → Creates `IdleResourceDaemon`
- `create_queue_populator()` → Creates `UnifiedQueuePopulatorDaemon`
- `create_node_recovery()` → Creates `NodeRecoveryDaemon`
- `create_utilization_optimizer()` → Creates `UtilizationOptimizer`
- `create_adaptive_resources()` → Creates `AdaptiveResourceManager` (not implemented yet)
- `create_multi_provider()` → Creates `MultiProviderOrchestrator` (not implemented yet)
- `create_node_availability()` → Creates `NodeAvailabilityDaemon`
- `create_availability_capacity_planner()` → Creates `CapacityPlanner`
- `create_availability_provisioner()` → Creates `Provisioner`

---

### Pipeline Automation

Automated pipeline stage triggering.

| Daemon Type        | Priority | Description                                                                     | Dependencies              |
| ------------------ | -------- | ------------------------------------------------------------------------------- | ------------------------- |
| `AUTO_EXPORT`      | MEDIUM   | Auto-triggers NPZ export when game count thresholds are met.                    | EVENT_ROUTER              |
| `TRAINING_TRIGGER` | MEDIUM   | Decides when to auto-trigger training based on data freshness and availability. | EVENT_ROUTER, AUTO_EXPORT |
| `DATA_CLEANUP`     | LOW      | Auto-quarantines/deletes poor quality game databases.                           | EVENT_ROUTER              |

**Factory Methods:**

- `create_auto_export()` → Creates `AutoExportDaemon`
- `create_training_trigger()` → Creates `TrainingTriggerDaemon`
- `create_data_cleanup()` → Creates `DataCleanupDaemon` (not implemented yet)

---

### System Health

System-level maintenance and monitoring.

| Daemon Type                    | Priority | Description                                             | Dependencies                            |
| ------------------------------ | -------- | ------------------------------------------------------- | --------------------------------------- |
| `MAINTENANCE`                  | LOW      | Log rotation, database vacuum, cleanup tasks.           | EVENT_ROUTER                            |
| `METRICS_ANALYSIS`             | MEDIUM   | Continuous metrics monitoring and plateau detection.    | EVENT_ROUTER                            |
| `CACHE_COORDINATION`           | MEDIUM   | Coordinates model caching across the cluster.           | EVENT_ROUTER, CLUSTER_MONITOR           |
| `RECOVERY_ORCHESTRATOR`        | HIGH     | Handles model/training state recovery after failures.   | EVENT_ROUTER, NODE_HEALTH_MONITOR       |
| `AVAILABILITY_RECOVERY_ENGINE` | MEDIUM   | Escalating recovery strategies for availability issues. | EVENT_ROUTER, AVAILABILITY_NODE_MONITOR |
| `CONNECTIVITY_RECOVERY`        | MEDIUM   | Unified event-driven connectivity recovery coordinator. | EVENT_ROUTER, TAILSCALE_HEALTH          |

**Factory Methods:**

- `create_maintenance()` → Creates `MaintenanceDaemon`
- `create_metrics_analysis()` → Creates `MetricsAnalysisDaemon` (not implemented yet)
- `create_cache_coordination()` → Creates `CacheCoordinationDaemon` (not implemented yet)
- `create_recovery_orchestrator()` → Creates `RecoveryOrchestrator` (not implemented yet)
- `create_availability_recovery_engine()` → Creates `RecoveryEngine`
- `create_connectivity_recovery()` → Creates `ConnectivityRecoveryCoordinator`

---

### Cost Optimization

Cloud provider cost management.

| Daemon Type   | Priority   | Description                                                                        | Dependencies                  |
| ------------- | ---------- | ---------------------------------------------------------------------------------- | ----------------------------- |
| `LAMBDA_IDLE` | DEPRECATED | Lambda idle shutdown (legacy). Dedicated GH200 nodes no longer need idle shutdown. | EVENT_ROUTER, CLUSTER_MONITOR |
| `VAST_IDLE`   | DEPRECATED | Vast idle shutdown (legacy). Use unified_idle_shutdown_daemon.                     | EVENT_ROUTER, CLUSTER_MONITOR |

**Factory Methods:**

- `create_lambda_idle()` → Creates `LambdaIdleDaemon` (deprecated)
- `create_vast_idle()` → Creates `VastIdleDaemon` (deprecated)

---

### Feedback & Curriculum

Training feedback and curriculum learning.

| Daemon Type              | Priority     | Description                                                                                              | Dependencies |
| ------------------------ | ------------ | -------------------------------------------------------------------------------------------------------- | ------------ |
| `FEEDBACK_LOOP`          | **CRITICAL** | Orchestrates all training feedback signals (quality, performance, Elo velocity).                         | EVENT_ROUTER |
| `CURRICULUM_INTEGRATION` | MEDIUM       | Bridges all feedback loops for self-improvement. Adjusts training curriculum based on model performance. | EVENT_ROUTER |

**Factory Methods:**

- `create_feedback_loop()` → Creates `FeedbackLoopController`
- `create_curriculum_integration()` → Creates `CurriculumIntegrationDaemon`

---

### Disk Space Management (December 2025)

Disk space monitoring and cleanup for coordinator nodes.

| Daemon Type                | Priority | Description                                                                                    | Dependencies                     |
| -------------------------- | -------- | ---------------------------------------------------------------------------------------------- | -------------------------------- |
| `DISK_SPACE_MANAGER`       | MEDIUM   | Proactive disk space monitoring. Cleanup at 60% usage (before 70% threshold).                  | EVENT_ROUTER                     |
| `COORDINATOR_DISK_MANAGER` | HIGH     | Specialized disk manager for coordinator nodes. Auto-syncs to external storage before cleanup. | EVENT_ROUTER, DISK_SPACE_MANAGER |

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

**Daemon Count:** 33

**Daemons:**

- `EVENT_ROUTER` (CRITICAL)
- `HEALTH_SERVER`
- `DAEMON_WATCHDOG`
- `P2P_BACKEND`
- `TOURNAMENT_DAEMON`
- `MODEL_DISTRIBUTION`
- `S3_BACKUP`
- `REPLICATION_MONITOR` (deprecated)
- `REPLICATION_REPAIR` (deprecated)
- `CLUSTER_MONITOR`
- `QUEUE_MONITOR`
- `FEEDBACK_LOOP` (CRITICAL)
- `QUALITY_MONITOR`
- `MODEL_PERFORMANCE_WATCHDOG`
- `NPZ_DISTRIBUTION` (deprecated)
- `ORPHAN_DETECTION`
- `UNIFIED_PROMOTION`
- `JOB_SCHEDULER`
- `IDLE_RESOURCE` (CRITICAL)
- `NODE_RECOVERY`
- `QUEUE_POPULATOR` (CRITICAL)
- `CURRICULUM_INTEGRATION`
- `AUTO_EXPORT`
- `NPZ_COMBINATION`
- `TRAINING_TRIGGER`
- `DLQ_RETRY`
- `GAUNTLET_FEEDBACK`
- `AUTO_SYNC` (CRITICAL)
- `CLUSTER_WATCHDOG`
- `METRICS_ANALYSIS`
- `ELO_SYNC`
- `DATA_CONSOLIDATION` (December 2025)
- `COORDINATOR_DISK_MANAGER` (December 2025)

**Use Case:** Centralized coordination, monitoring, and job scheduling for the entire cluster.

---

### Training Node Profile

Runs on GPU training nodes (H100, Nebius, RunPod, Vultr; Lambda legacy).

**Daemon Count:** 21

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
- `NPZ_COMBINATION`
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
- `DATA_PIPELINE`
- `IDLE_RESOURCE` (CRITICAL)
- `QUALITY_MONITOR`
- `ORPHAN_DETECTION`
- `AUTO_SYNC` (CRITICAL)
- `FEEDBACK_LOOP` (CRITICAL)
- `DISK_SPACE_MANAGER` (December 2025)

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

**Daemon Count:** 85

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

This graph is a simplified snapshot. For canonical dependencies, see `app/coordination/daemon_registry.py`.

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

Quality + Training Enhancement (positions 16-17)
16. QUALITY_MONITOR     # Quality monitoring (depends on DATA_PIPELINE)
17. DISTILLATION        # Distillation (depends on TRAINING_TRIGGER)

Evaluation and Promotion Chain (positions 18-21)
18. EVALUATION          # Model evaluation (depends on TRAINING_TRIGGER)
19. UNIFIED_PROMOTION   # Unified promotion (depends on EVALUATION)
20. AUTO_PROMOTION      # Auto-promotion (depends on EVALUATION)
21. MODEL_DISTRIBUTION  # Model distribution (depends on AUTO_PROMOTION)
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

These daemons get 15-second health check intervals (vs 60s for others), per `CRITICAL_DAEMONS`:

1. **EVENT_ROUTER** - Core event bus. All coordination depends on this.
2. **DAEMON_WATCHDOG** - Self-healing for daemon crashes.
3. **DATA_PIPELINE** - Pipeline processor (before sync).
4. **AUTO_SYNC** - Primary data sync mechanism. Ensures fresh data.
5. **QUEUE_POPULATOR** - Keeps work queue populated. Prevents idle cluster.
6. **IDLE_RESOURCE** - Ensures GPUs stay utilized. Prevents waste.
7. **FEEDBACK_LOOP** - Coordinates training feedback signals.

**Health Check Interval:** 15 seconds
**Auto-Restart:** Yes (with exponential backoff, max 5 attempts)

### HIGH Priority

Important daemons for cluster operation (examples, non-exhaustive):

- HEALTH_SERVER
- WORK_QUEUE_MONITOR
- COORDINATOR_HEALTH_MONITOR
- CLUSTER_WATCHDOG
- EVALUATION
- UTILIZATION_OPTIMIZER
- JOB_SCHEDULER

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes

### MEDIUM Priority

Standard operational daemons (examples, non-exhaustive):

- Sync daemons (GOSSIP_SYNC, MODEL_SYNC, HIGH_QUALITY_SYNC)
- Training support (CONTINUOUS_TRAINING_LOOP, DISTILLATION, CASCADE_TRAINING)
- Monitoring (QUALITY_MONITOR, CLUSTER_MONITOR, METRICS_ANALYSIS)
- Pipeline automation (AUTO_EXPORT, TRAINING_TRIGGER)
- Tournaments and evaluation (TOURNAMENT_DAEMON, GAUNTLET_FEEDBACK)
- Resource management (NODE_RECOVERY, ADAPTIVE_RESOURCES)

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes

### LOW Priority

Optional/non-critical daemons (examples, non-exhaustive):

- ORPHAN_DETECTION
- EXTERNAL_DRIVE_SYNC
- VAST_CPU_PIPELINE
- MAINTENANCE
- DATA_CLEANUP

**Health Check Interval:** 60 seconds
**Auto-Restart:** Yes (but lower urgency)

---

## Factory Methods Reference

All daemon factory methods follow the pattern `create_<daemon_name>()` (defined in `daemon_runners.py`). Legacy `_create_*()` helpers remain only for special cases (e.g., health server wiring). They either:

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
Legacy wrappers are consolidated here; `base_handler.py` remains as a deprecated shim, and the
old `base_event_handler.py` module is archived at `ai-service/archive/deprecated_coordination/_deprecated_base_event_handler.py`.

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

### Deprecated Daemons (Q2 2026 Removal)

| Daemon Type             | Replacement / Guidance                                             | Notes               |
| ----------------------- | ------------------------------------------------------------------ | ------------------- |
| `SYNC_COORDINATOR`      | Use `AUTO_SYNC`.                                                   | Deprecated Dec 2025 |
| `EPHEMERAL_SYNC`        | Use AutoSyncDaemon(strategy="ephemeral").                          | Deprecated Dec 2025 |
| `CLUSTER_DATA_SYNC`     | Use AutoSyncDaemon(strategy="broadcast").                          | Deprecated Dec 2025 |
| `HEALTH_CHECK`          | Use `NODE_HEALTH_MONITOR` (or unified health orchestrator).        | Deprecated Dec 2025 |
| `NODE_HEALTH_MONITOR`   | Use `health_check_orchestrator.py`.                                | Deprecated Dec 2025 |
| `SYSTEM_HEALTH_MONITOR` | Use `unified_health_manager.py`.                                   | Deprecated Dec 2025 |
| `NPZ_DISTRIBUTION`      | Use `unified_distribution_daemon.py` with `DataType.NPZ`.          | Deprecated Dec 2025 |
| `REPLICATION_MONITOR`   | Use `unified_replication_daemon.py`.                               | Deprecated Dec 2025 |
| `REPLICATION_REPAIR`    | Use `unified_replication_daemon.py`.                               | Deprecated Dec 2025 |
| `LAMBDA_IDLE`           | Dedicated GH200 nodes no longer need idle shutdown.                | Deprecated Dec 2025 |
| `VAST_IDLE`             | Use unified idle shutdown daemon (`unified_idle_shutdown_daemon`). | Deprecated Dec 2025 |

---

## Infrastructure Verification (December 2025)

Automated verification confirmed the following architecture is properly configured:

### Daemon Coverage

| Metric                         | Value | Status          |
| ------------------------------ | ----- | --------------- |
| Total DaemonType values        | 85    | ✓ All accounted |
| Runners in `daemon_runners.py` | 85    | ✓ Complete      |
| Inline runners                 | 0     | ✓ None          |
| Missing runners                | 0     | ✓ None          |

### Startup Order

| Metric                             | Value  | Notes                                  |
| ---------------------------------- | ------ | -------------------------------------- |
| Daemons in `DAEMON_STARTUP_ORDER`  | 21     | Critical path daemons                  |
| Daemons with `DAEMON_DEPENDENCIES` | 79     | Legacy dependency map                  |
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
