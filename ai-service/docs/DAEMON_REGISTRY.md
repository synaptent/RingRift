# Daemon Registry

Complete reference of all background daemons in the RingRift AI training infrastructure.

**Last Updated:** December 2025
**Total Daemons:** 53

## Quick Reference

| Category    | Daemons | Purpose                             |
| ----------- | ------- | ----------------------------------- |
| Sync        | 6       | Data synchronization across cluster |
| Training    | 9       | Training pipeline automation        |
| Monitoring  | 7       | Health and status monitoring        |
| Events      | 3       | Event routing and processing        |
| P2P         | 3       | Peer-to-peer mesh network           |
| Replication | 2       | Data replication enforcement        |
| Feedback    | 4       | Training feedback loops             |
| Resources   | 4       | Resource management                 |
| Recovery    | 3       | Error recovery and resilience       |
| External    | 2       | External storage/providers          |
| Utility     | 10      | Various utilities                   |

## Daemon Profiles

Daemons are grouped into profiles for different node roles:

Canonical definitions live in `app/coordination/daemon_manager.py` as
`DAEMON_PROFILES` and are used by `scripts/launch_daemons.py --profile`.

### Coordinator Profile (mac-studio)

Primary orchestration node - runs most daemons:

- EVENT_ROUTER, HEALTH_SERVER, P2P_BACKEND
- TOURNAMENT_DAEMON, MODEL_DISTRIBUTION
- REPLICATION_MONITOR, REPLICATION_REPAIR
- CLUSTER_MONITOR, FEEDBACK_LOOP
- QUALITY_MONITOR, MODEL_PERFORMANCE_WATCHDOG
- NPZ_DISTRIBUTION, ORPHAN_DETECTION
- NODE_HEALTH_MONITOR, SYSTEM_HEALTH_MONITOR
- UNIFIED_PROMOTION, JOB_SCHEDULER
- IDLE_RESOURCE, NODE_RECOVERY
- QUEUE_POPULATOR, CURRICULUM_INTEGRATION
- AUTO_EXPORT, TRAINING_TRIGGER

### Training Node Profile (GPU nodes)

For nodes actively running training:

- EVENT_ROUTER, HEALTH_SERVER
- DATA_PIPELINE, CONTINUOUS_TRAINING_LOOP
- AUTO_SYNC, TRAINING_NODE_WATCHER
- EVALUATION, QUALITY_MONITOR
- ORPHAN_DETECTION, UNIFIED_PROMOTION
- P2P_AUTO_DEPLOY, IDLE_RESOURCE
- CURRICULUM_INTEGRATION
- AUTO_EXPORT, TRAINING_TRIGGER

### Ephemeral Profile (Vast.ai spot instances)

Minimal daemons for short-lived nodes:

- EVENT_ROUTER, EPHEMERAL_SYNC
- DATA_PIPELINE, IDLE_RESOURCE

### Selfplay Profile (GPU selfplay nodes)

For dedicated selfplay generation:

- EVENT_ROUTER, AUTO_SYNC
- QUALITY_MONITOR, IDLE_RESOURCE

---

## Master Loop Profiles (scripts/master_loop.py)

The unified controller supports `--profile` to choose a daemon set. These
profiles are separate from the node role profiles above and control which
daemons the master loop starts.

### minimal

Baseline sync + health:

- EVENT_ROUTER, NODE_HEALTH_MONITOR, CLUSTER_MONITOR
- SYSTEM_HEALTH_MONITOR, HEALTH_SERVER
- AUTO_SYNC, CLUSTER_DATA_SYNC, ELO_SYNC

### standard (default)

Core automation on top of `minimal`:

- FEEDBACK_LOOP, DATA_PIPELINE, MODEL_DISTRIBUTION
- IDLE_RESOURCE, UTILIZATION_OPTIMIZER, QUEUE_POPULATOR
- AUTO_EXPORT, EVALUATION, AUTO_PROMOTION, TOURNAMENT_DAEMON
- CURRICULUM_INTEGRATION, NODE_RECOVERY, TRAINING_NODE_WATCHER
- QUALITY_MONITOR

### full

All `DaemonType` entries except deprecated `SYNC_COORDINATOR` and `HEALTH_CHECK`.

---

## Daemon Details

### Sync Daemons

#### SYNC_COORDINATOR

**Status:** Deprecated (use AUTO_SYNC)

- **Purpose:** Legacy sync coordination
- **File:** `app/coordination/sync_coordinator.py`

#### HIGH_QUALITY_SYNC

- **Purpose:** Priority sync for high-quality game data
- **Triggers:** Quality score > 0.7

#### ELO_SYNC

- **Purpose:** Synchronize ELO ratings across cluster
- **Interval:** 60 seconds
- **File:** Integrated into P2P

#### MODEL_SYNC

- **Purpose:** Sync model files between nodes
- **Triggers:** MODEL_PROMOTED event

#### CLUSTER_DATA_SYNC

- **Purpose:** Cluster-wide game database sync
- **Interval:** 300 seconds
- **File:** `app/coordination/cluster_data_sync.py`

#### AUTO_SYNC

- **Purpose:** Automated P2P data sync with gossip replication
- **Features:** Push-from-generator, excludes coordinator, respects bandwidth limits
- **File:** `app/coordination/auto_sync_daemon.py`

---

### Training Daemons

#### DATA_PIPELINE

- **Purpose:** Orchestrate training data pipeline
- **Stages:** Selfplay → Sync → Export → Train → Evaluate → Promote
- **File:** `app/coordination/data_pipeline_orchestrator.py`

#### SELFPLAY_COORDINATOR

- **Purpose:** Coordinate selfplay across cluster
- **Features:** Priority-based allocation, curriculum weights

#### CONTINUOUS_TRAINING_LOOP

- **Purpose:** Run continuous training loop
- **Depends on:** EVENT_ROUTER

#### TOURNAMENT_DAEMON

- **Purpose:** Schedule evaluation tournaments
- **Triggers:** TRAINING_COMPLETED event
- **Interval:** Periodic ladder (configurable)
- **File:** `app/coordination/tournament_daemon.py`

#### EVALUATION

- **Purpose:** Auto-evaluate models after training
- **Triggers:** TRAINING_COMPLETED event
- **File:** `app/coordination/evaluation_daemon.py`

#### AUTO_EXPORT

- **Purpose:** Auto-export NPZ when game threshold met
- **Threshold:** Configurable (default: 500 games)
- **File:** `app/coordination/auto_export_daemon.py`

#### TRAINING_TRIGGER

- **Purpose:** Decide when to trigger training
- **Logic:** Games threshold, quality gates, staleness
- **File:** `app/coordination/training_trigger_daemon.py`

#### DISTILLATION

- **Purpose:** Knowledge distillation for smaller models
- **File:** `app/coordination/daemon_adapters.py`

#### UNIFIED_PROMOTION

- **Purpose:** Auto-promote models based on evaluation
- **Thresholds:** 60% vs heuristic, 85% vs random
- **File:** `app/coordination/auto_promotion_daemon.py`

---

### Monitoring Daemons

#### HEALTH_CHECK

**Status:** Deprecated (use NODE_HEALTH_MONITOR)

- **Purpose:** Legacy health checks

#### CLUSTER_MONITOR

- **Purpose:** Real-time cluster status
- **File:** `app/distributed/cluster_monitor.py`

#### QUEUE_MONITOR

- **Purpose:** Work queue depth monitoring
- **Alerts:** Queue overflow, starvation

#### NODE_HEALTH_MONITOR

- **Purpose:** Unified cluster health maintenance
- **Features:** SSH health, GPU util, disk space
- **File:** `app/coordination/unified_node_health_daemon.py`

#### QUALITY_MONITOR

- **Purpose:** Continuous selfplay quality monitoring
- **Metrics:** Policy accuracy, value accuracy, game length
- **File:** `app/coordination/quality_monitor_daemon.py`

#### MODEL_PERFORMANCE_WATCHDOG

- **Purpose:** Monitor model win rates for regression
- **Alerts:** Win rate drops > 10%

#### SYSTEM_HEALTH_MONITOR

- **Purpose:** Global system health with pipeline pause
- **Features:** Resource constraints, coordinator health
- **File:** `app/coordination/system_health_monitor.py`

---

### Replication Daemons

#### REPLICATION_MONITOR

- **Purpose:** Monitor data replication health
- **Alerts:** Under-replicated, single-copy, zero-copy games
- **Emergency:** Triggers sync when threshold exceeded
- **File:** `app/coordination/replication_monitor.py`

#### REPLICATION_REPAIR

- **Purpose:** Actively repair under-replicated data
- **Priority:** Zero-copy > single-copy > under-replicated
- **Limits:** 500 repairs/hour, 3 concurrent
- **File:** `app/coordination/replication_repair_daemon.py`

---

### Event Daemons

#### EVENT_ROUTER

- **Purpose:** Unified event bus routing
- **Features:** SHA256 deduplication, cross-process bridging
- **File:** `app/coordination/event_router.py`

#### CROSS_PROCESS_POLLER

- **Purpose:** Poll cross-process event queue
- **Backend:** SQLite-based queue
- **File:** `app/coordination/cross_process_events.py`

#### DLQ_RETRY

- **Purpose:** Retry failed event handlers
- **Depends on:** EVENT_ROUTER
- **File:** `app/coordination/dead_letter_queue.py`

---

### P2P Daemons

#### P2P_BACKEND

- **Purpose:** P2P mesh network server
- **Port:** 8770
- **Features:** Leader election, gossip protocol
- **File:** `app/distributed/p2p.py`

#### GOSSIP_SYNC

- **Purpose:** Gossip-based data discovery
- **Interval:** 60 seconds

#### DATA_SERVER

- **Purpose:** HTTP data server for file transfers

---

### Feedback Daemons

#### FEEDBACK_LOOP

- **Purpose:** Central feedback loop controller
- **Signals:** Intensity, exploration, curriculum
- **File:** `app/coordination/feedback_loop_controller.py`

#### CURRICULUM_INTEGRATION

- **Purpose:** Bridge feedback loops for self-improvement
- **Features:** Weight adjustment, ELO velocity tracking

#### GAUNTLET_FEEDBACK

- **Purpose:** Bridge gauntlet evaluation to training
- **File:** `app/coordination/gauntlet_feedback_controller.py`

#### METRICS_ANALYSIS

- **Purpose:** Continuous metrics and plateau detection
- **Triggers:** PLATEAU_DETECTED event

---

### Resource Daemons

#### IDLE_RESOURCE

- **Purpose:** Monitor idle GPUs, spawn selfplay
- **Threshold:** GPU util < 10%
- **File:** `app/coordination/idle_resource_daemon.py`

#### JOB_SCHEDULER

- **Purpose:** Centralized job scheduling
- **Features:** PID-based resource allocation

#### QUEUE_POPULATOR

- **Purpose:** Auto-populate work queue
- **Mix:** 60% selfplay, 30% training, 10% tournament
- **File:** `app/coordination/queue_populator_daemon.py`

#### ADAPTIVE_RESOURCES

- **Purpose:** Dynamic resource scaling
- **Thresholds:** Disk 85%/92%, Memory 85%/95%

---

### Recovery Daemons

#### NODE_RECOVERY

- **Purpose:** Auto-recover terminated nodes
- **Providers:** Vast.ai, RunPod restart
- **File:** `app/coordination/node_recovery_daemon.py`

#### RECOVERY_ORCHESTRATOR

- **Purpose:** Model/training state recovery
- **Depends on:** NODE_HEALTH_MONITOR

#### P2P_AUTO_DEPLOY

- **Purpose:** Ensure P2P runs on recovered nodes

---

### External Daemons

#### EXTERNAL_DRIVE_SYNC

- **Purpose:** Sync to external storage (/Volumes/OWC)
- **File:** `app/coordination/daemon_adapters.py`

#### VAST_CPU_PIPELINE

- **Purpose:** Coordinate Vast.ai CPU instances

---

### Utility Daemons

#### TRAINING_NODE_WATCHER

- **Purpose:** Detect active training, trigger priority sync
- **File:** `app/coordination/training_freshness.py`

#### EPHEMERAL_SYNC

- **Purpose:** Aggressive sync for spot instances
- **Interval:** 5 seconds
- **File:** `app/coordination/ephemeral_sync.py`

#### NPZ_DISTRIBUTION

- **Purpose:** Distribute training data after export
- **File:** `app/coordination/npz_distribution_daemon.py`

#### ORPHAN_DETECTION

- **Purpose:** Detect unregistered game databases
- **File:** `app/coordination/orphan_detection_daemon.py`

#### MODEL_DISTRIBUTION

- **Purpose:** Distribute models after promotion
- **Triggers:** MODEL_PROMOTED event
- **File:** `app/coordination/model_distribution_daemon.py`

#### CACHE_COORDINATION

- **Purpose:** Coordinate model caching across cluster
- **Depends on:** CLUSTER_MONITOR

#### MULTI_PROVIDER

- **Purpose:** Coordinate across Runpod/Vast
- **Depends on:** CLUSTER_MONITOR

#### HEALTH_SERVER

- **Purpose:** HTTP endpoints (/health, /ready, /metrics)
- **Port:** 8765

#### MAINTENANCE

- **Purpose:** Log rotation, DB vacuum, cleanup
- **File:** `app/coordination/maintenance_daemon.py`

#### AUTO_PROMOTION

- **Purpose:** Legacy auto-promotion (see UNIFIED_PROMOTION)

---

## Starting Daemons

### Via master_loop.py (Recommended)

```bash
# Full automation
python scripts/master_loop.py

# Specific configs
python scripts/master_loop.py --configs hex8_2p,square8_2p
```

### Via launch_daemons.py

```bash
# Start all
python scripts/launch_daemons.py --all

# By category
python scripts/launch_daemons.py --monitoring
python scripts/launch_daemons.py --training

# Specific daemon
python scripts/launch_daemons.py --daemon replication_repair
```

### Programmatically

```python
from app.coordination.daemon_manager import get_daemon_manager, DaemonType

manager = get_daemon_manager()
await manager.start(DaemonType.REPLICATION_REPAIR)
```

---

## Daemon Dependencies

```
EVENT_ROUTER (core - most daemons depend on this)
    ├── DLQ_RETRY
    ├── CONTINUOUS_TRAINING_LOOP
    ├── TOURNAMENT_DAEMON
    ├── DATA_PIPELINE
    ├── SELFPLAY_COORDINATOR
    ├── MODEL_DISTRIBUTION
    ├── UNIFIED_PROMOTION
    ├── FEEDBACK_LOOP
    ├── EVALUATION
    │   └── AUTO_PROMOTION
    ├── QUALITY_MONITOR
    ├── MODEL_PERFORMANCE_WATCHDOG
    ├── NPZ_DISTRIBUTION
    ├── JOB_SCHEDULER
    ├── IDLE_RESOURCE
    ├── NODE_RECOVERY
    ├── QUEUE_POPULATOR
    ├── CURRICULUM_INTEGRATION
    ├── AUTO_EXPORT
    │   └── TRAINING_TRIGGER
    ├── GAUNTLET_FEEDBACK
    ├── CACHE_COORDINATION
    │   └── (CLUSTER_MONITOR)
    ├── METRICS_ANALYSIS
    ├── ADAPTIVE_RESOURCES
    │   └── (CLUSTER_MONITOR)
    ├── MULTI_PROVIDER
    │   └── (CLUSTER_MONITOR)
    └── SYSTEM_HEALTH_MONITOR
        └── (NODE_HEALTH_MONITOR)

RECOVERY_ORCHESTRATOR
    └── (NODE_HEALTH_MONITOR)
```
