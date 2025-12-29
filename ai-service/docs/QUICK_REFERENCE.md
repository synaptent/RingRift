# Quick Reference Card

Single-page reference for RingRift AI Service coordination infrastructure.

---

## Daemon Types Quick Lookup

| DaemonType                   | Description                                  | Dependencies                               |
| ---------------------------- | -------------------------------------------- | ------------------------------------------ |
| **Core Infrastructure**      |
| `EVENT_ROUTER`               | Event bus - all coordination depends on this | None                                       |
| `DAEMON_WATCHDOG`            | Self-healing daemon crash monitor            | EVENT_ROUTER                               |
| `DATA_PIPELINE`              | Pipeline stage orchestration                 | EVENT_ROUTER                               |
| `FEEDBACK_LOOP`              | Training feedback signal coordination        | EVENT_ROUTER                               |
| **Sync**                     |
| `AUTO_SYNC`                  | Primary P2P data sync mechanism              | EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP |
| `HIGH_QUALITY_SYNC`          | Priority sync for high-quality data          | EVENT_ROUTER, DATA_PIPELINE                |
| `ELO_SYNC`                   | Elo rating synchronization                   | EVENT_ROUTER                               |
| `MODEL_SYNC`                 | Model file synchronization                   | EVENT_ROUTER                               |
| `MODEL_DISTRIBUTION`         | Auto-distribute models after promotion       | EVENT_ROUTER, AUTO_PROMOTION               |
| `NPZ_DISTRIBUTION`           | Training data distribution                   | EVENT_ROUTER, DATA_PIPELINE                |
| `SYNC_PUSH`                  | GPU nodes push data before cleanup           | EVENT_ROUTER                               |
| **Queue & Resources**        |
| `QUEUE_POPULATOR`            | Maintains work queue until Elo targets       | EVENT_ROUTER                               |
| `WORK_QUEUE_MONITOR`         | Queue depth, latency, backpressure           | EVENT_ROUTER, QUEUE_POPULATOR              |
| `IDLE_RESOURCE`              | Spawns selfplay on idle GPUs                 | EVENT_ROUTER, QUEUE_POPULATOR              |
| `JOB_SCHEDULER`              | Centralized job scheduling                   | EVENT_ROUTER                               |
| `RESOURCE_OPTIMIZER`         | Resource allocation optimization             | EVENT_ROUTER                               |
| **Training**                 |
| `TRAINING_TRIGGER`           | Decides when to trigger training             | EVENT_ROUTER, DATA_PIPELINE, AUTO_SYNC     |
| `CONTINUOUS_TRAINING_LOOP`   | Continuous training orchestration            | EVENT_ROUTER, DATA_PIPELINE                |
| `SELFPLAY_COORDINATOR`       | Selfplay job coordination                    | EVENT_ROUTER                               |
| `DISTILLATION`               | Model distillation                           | EVENT_ROUTER, TRAINING_TRIGGER             |
| **Evaluation & Promotion**   |
| `EVALUATION`                 | Model evaluation after training              | EVENT_ROUTER, TRAINING_TRIGGER             |
| `AUTO_PROMOTION`             | Auto-promote based on eval results           | EVENT_ROUTER, EVALUATION                   |
| `TOURNAMENT_DAEMON`          | Automatic tournament scheduling              | EVENT_ROUTER                               |
| `GAUNTLET_FEEDBACK`          | Bridges gauntlet to training feedback        | EVENT_ROUTER, EVALUATION                   |
| **Health & Monitoring**      |
| `HEALTH_SERVER`              | HTTP endpoints /health, /ready, /metrics     | EVENT_ROUTER                               |
| `NODE_HEALTH_MONITOR`        | Node health tracking                         | EVENT_ROUTER                               |
| `CLUSTER_MONITOR`            | Cluster-wide monitoring                      | EVENT_ROUTER                               |
| `CLUSTER_WATCHDOG`           | Self-healing cluster monitor                 | EVENT_ROUTER, CLUSTER_MONITOR              |
| `QUALITY_MONITOR`            | Selfplay data quality monitoring             | EVENT_ROUTER, DATA_PIPELINE                |
| `COORDINATOR_HEALTH_MONITOR` | Coordinator health tracking                  | EVENT_ROUTER                               |
| **Recovery & Maintenance**   |
| `NODE_RECOVERY`              | Auto-recovers terminated nodes               | EVENT_ROUTER, NODE_HEALTH_MONITOR          |
| `MAINTENANCE`                | Log rotation, DB vacuum, cleanup             | EVENT_ROUTER                               |
| `DISK_SPACE_MANAGER`         | Proactive disk space management              | EVENT_ROUTER                               |
| `COORDINATOR_DISK_MANAGER`   | Coordinator-only disk management             | EVENT_ROUTER                               |
| `VAST_IDLE`                  | Terminates idle Vast.ai nodes                | EVENT_ROUTER                               |
| **P2P**                      |
| `P2P_BACKEND`                | P2P mesh network backend                     | None                                       |
| `GOSSIP_SYNC`                | Gossip protocol sync                         | EVENT_ROUTER                               |
| `P2P_AUTO_DEPLOY`            | Ensure P2P runs on all nodes                 | EVENT_ROUTER                               |

> **Deprecated (Q2 2026):** `SYNC_COORDINATOR`, `HEALTH_CHECK`, `CLUSTER_DATA_SYNC`, `EPHEMERAL_SYNC`, `SYSTEM_HEALTH_MONITOR`, `LAMBDA_IDLE`

---

## Event Types by Category

### Training Events

| Event                        | Description                     |
| ---------------------------- | ------------------------------- |
| `TRAINING_STARTED`           | Training has started            |
| `TRAINING_COMPLETED`         | Training completed successfully |
| `TRAINING_FAILED`            | Training failed with error      |
| `TRAINING_PROGRESS`          | Epoch progress update           |
| `TRAINING_EARLY_STOPPED`     | Early stopping triggered        |
| `TRAINING_THRESHOLD_REACHED` | Sufficient data for training    |
| `TRAINING_ROLLBACK_NEEDED`   | Rollback recommended            |

### Sync & Data Events

| Event                 | Description                 |
| --------------------- | --------------------------- |
| `DATA_SYNC_STARTED`   | Sync operation started      |
| `DATA_SYNC_COMPLETED` | Sync completed successfully |
| `DATA_SYNC_FAILED`    | Sync failed                 |
| `NEW_GAMES_AVAILABLE` | New training data ready     |
| `DATA_STALE`          | Training data is stale      |
| `DATA_FRESH`          | Training data is fresh      |
| `SYNC_REQUEST`        | Explicit sync request       |

### Health & Cluster Events

| Event                          | Description               |
| ------------------------------ | ------------------------- |
| `HEALTH_CHECK_PASSED`          | Node health check passed  |
| `HEALTH_CHECK_FAILED`          | Node health check failed  |
| `NODE_UNHEALTHY`               | Node marked unhealthy     |
| `NODE_RECOVERED`               | Node recovered to healthy |
| `CLUSTER_STATUS_CHANGED`       | Cluster status changed    |
| `P2P_CLUSTER_HEALTHY`          | Cluster is healthy        |
| `P2P_CLUSTER_UNHEALTHY`        | Cluster is unhealthy      |
| `HOST_ONLINE` / `HOST_OFFLINE` | Node connectivity         |

### Evaluation & Promotion Events

| Event                    | Description                   |
| ------------------------ | ----------------------------- |
| `EVALUATION_STARTED`     | Evaluation started            |
| `EVALUATION_COMPLETED`   | Evaluation completed          |
| `MODEL_PROMOTED`         | Model promoted to production  |
| `PROMOTION_FAILED`       | Promotion failed              |
| `PROMOTION_REJECTED`     | Below threshold               |
| `ELO_UPDATED`            | Elo rating updated            |
| `ELO_SIGNIFICANT_CHANGE` | Triggers curriculum rebalance |

### Quality & Regression Events

| Event                   | Description             |
| ----------------------- | ----------------------- |
| `QUALITY_SCORE_UPDATED` | Quality recalculated    |
| `QUALITY_DEGRADED`      | Below threshold         |
| `REGRESSION_DETECTED`   | Any regression detected |
| `REGRESSION_CRITICAL`   | Rollback recommended    |
| `REGRESSION_CLEARED`    | Model recovered         |

### Work Queue Events

| Event            | Description             |
| ---------------- | ----------------------- |
| `WORK_QUEUED`    | Work added to queue     |
| `WORK_CLAIMED`   | Work claimed by node    |
| `WORK_COMPLETED` | Work completed          |
| `WORK_FAILED`    | Work failed permanently |
| `WORK_TIMEOUT`   | Work timed out          |

### System Events

| Event                               | Description           |
| ----------------------------------- | --------------------- |
| `DAEMON_STARTED` / `DAEMON_STOPPED` | Daemon lifecycle      |
| `DAEMON_STATUS_CHANGED`             | Health status change  |
| `LEADER_ELECTED`                    | New cluster leader    |
| `BACKPRESSURE_ACTIVATED`            | Queue full            |
| `RECOVERY_INITIATED`                | Auto-recovery started |

---

## Health Check Endpoints

| Endpoint   | Port | Purpose                                             |
| ---------- | ---- | --------------------------------------------------- |
| `/health`  | 8790 | Liveness probe - is the service running?            |
| `/ready`   | 8790 | Readiness probe - is the service ready for traffic? |
| `/metrics` | 8790 | Prometheus-style metrics                            |
| `/status`  | 8770 | P2P cluster status (leader, peers, jobs)            |
| `/health`  | 8770 | P2P node health                                     |

**Environment variables:**

- `RINGRIFT_HEALTH_PORT` - Health server port (default: 8790)
- `RINGRIFT_P2P_PORT` / `P2P_DEFAULT_PORT` - P2P port (default: 8770)

---

## Common Troubleshooting Flows

### Check Daemon Status

```bash
# CLI status
python scripts/launch_daemons.py --status

# Programmatic
from app.coordination.daemon_manager import get_daemon_manager
dm = get_daemon_manager()
all_health = dm.get_all_daemon_health()
for dtype, health in all_health.items():
    print(f"{dtype.value}: {health}")

# Liveness probe (for K8s/Docker)
curl http://localhost:8790/health
```

### Verify Event Flow

```bash
# Check event subscriptions are wired
from app.coordination.data_pipeline_orchestrator import get_data_pipeline_orchestrator
orch = get_data_pipeline_orchestrator()
# Subscribed events are logged at startup

# Check cross-process queue
from app.coordination.cross_process_events import get_cross_process_queue
queue = get_cross_process_queue()
pending = queue.get_pending_count()
```

### Check Cluster Health

```bash
# Quick P2P status
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} nodes")
print(f"Role: {d.get(\"role\")}")
'

# Python cluster monitor
python -m app.distributed.cluster_monitor --watch

# Programmatic
from app.coordination.cluster_status_monitor import ClusterMonitor
monitor = ClusterMonitor()
status = await monitor.get_cluster_status_async()
```

### Common Issues

| Symptom                 | Check                     | Fix                                       |
| ----------------------- | ------------------------- | ----------------------------------------- |
| Daemon not starting     | `--status`, check logs    | Check dependencies, restart DaemonManager |
| Events not flowing      | Cross-process queue depth | Restart EVENT_ROUTER                      |
| Handlers failing        | DLQ backlog               | Run dlq_dashboard.py, retry or purge DLQ  |
| Sync stalled            | `/status` endpoint        | Check network, BACKPRESSURE events        |
| Node not responding     | `/health` on 8770         | SSH to node, check P2P process            |
| Training not triggering | TRAINING_THRESHOLD events | Check DATA_PIPELINE, game counts          |

---

## Quick Commands

```bash
# Launch all daemons
python scripts/launch_daemons.py --all

# Full automation (recommended)
python scripts/master_loop.py

# Check specific daemon health
python scripts/launch_daemons.py --status | grep AUTO_SYNC

# Force sync
python scripts/unified_data_sync.py --force

# Check P2P voter quorum
curl -s http://localhost:8770/status | jq '.voter_quorum'

# DLQ dashboard (failed event handlers)
python scripts/dlq_dashboard.py --pending --limit 20
```

---

_Generated: December 2025 | See `CLAUDE.md` and `docs/EVENT_SYSTEM_REFERENCE.md` for full details_
