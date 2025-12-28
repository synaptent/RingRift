# Master Runbook Index - RingRift AI Service

Centralized navigation for all 32 AI service operational runbooks.

**Created**: December 28, 2025
**Last Updated**: December 28, 2025

---

## Decision Tree: What's the Problem?

Use this tree to quickly find the right runbook:

```
Is the cluster responding?
├── NO → DISASTER_RECOVERY.md
└── YES
    │
    ├── Is P2P/Leader failing?
    │   ├── Leader election failed → P2P_LEADER_FAILOVER.md
    │   └── P2P orchestrator issues → P2P_ORCHESTRATOR_OPERATIONS.md
    │
    ├── Are daemons unhealthy?
    │   ├── Daemon won't start → DAEMON_FAILURE_RECOVERY.md
    │   └── Daemon manager issues → DAEMON_MANAGER_OPERATIONS.md
    │
    ├── Is data sync failing?
    │   ├── Cluster-wide sync → CLUSTER_SYNCHRONIZATION.md
    │   ├── Host-level sync → SYNC_HOST_CRITICAL.md
    │   ├── Transport failures → TRANSPORT_ESCALATION_FAILURES.md
    │   └── Orphan games → ORPHAN_GAMES_DETECTION.md
    │
    ├── Are events not flowing?
    │   ├── Event wiring verification → EVENT_WIRING_VERIFICATION.md
    │   └── Event system issues → COORDINATION_EVENT_SYSTEM.md
    │
    ├── Is training/parity failing?
    │   ├── Training loop stalled → TRAINING_LOOP_STALLED.md
    │   ├── Pipeline stage failed → PIPELINE_STAGE_FAILURES.md
    │   ├── Parity gate stuck → PARITY_GATE_RESOLUTION.md
    │   ├── Parity mismatch → PARITY_MISMATCH_DEBUG.md
    │   ├── Hexagonal-specific → HEXAGONAL_PARITY_BUG.md
    │   ├── Model promotion issues → MODEL_PROMOTION_WORKFLOW.md
    │   ├── Model distribution → MODEL_DISTRIBUTION_TROUBLESHOOTING.md
    │   ├── Feedback loops broken → FEEDBACK_LOOP_TROUBLESHOOTING.md
    │   ├── Feedback degraded → FEEDBACK_LOOP_DEGRADATION.md
    │   ├── Data corruption → GAME_DATA_CORRUPTION_RECOVERY.md
    │   └── Work queue stalled → WORK_QUEUE_STALLS.md
    │
    ├── Are GPUs stuck?
    │   ├── GPU OOM errors → GPU_OOM_DEBUG.md
    │   ├── GPU processes stuck → CLUSTER_GPU_STUCK.md
    │   └── Disk space issues → DISK_SPACE_MANAGEMENT.md
    │
    ├── Is cluster health low?
    │   ├── Health < 50% → CLUSTER_HEALTH_CRITICAL.md
    │   └── Connectivity issues → CLUSTER_CONNECTIVITY.md
    │
    └── General issues?
        ├── Coordinator errors → COORDINATOR_ERROR.md
        ├── Cluster deployment → cluster_deployment.md
        └── General troubleshooting → TROUBLESHOOTING.md
```

---

## Runbooks by Category

### Cluster & P2P Operations

| Runbook                                                          | Severity | When to Use                             |
| ---------------------------------------------------------------- | -------- | --------------------------------------- |
| [P2P_LEADER_FAILOVER.md](P2P_LEADER_FAILOVER.md)                 | Critical | Leader election failed, cluster stalled |
| [P2P_ORCHESTRATOR_OPERATIONS.md](P2P_ORCHESTRATOR_OPERATIONS.md) | High     | P2P orchestrator not responding         |
| [CLUSTER_HEALTH_CRITICAL.md](CLUSTER_HEALTH_CRITICAL.md)         | Critical | Cluster health dropped below 50%        |
| [CLUSTER_CONNECTIVITY.md](CLUSTER_CONNECTIVITY.md)               | High     | Nodes unreachable, P2P issues           |
| [cluster_deployment.md](cluster_deployment.md)                   | Info     | Adding/removing cluster nodes           |

### Daemon Management

| Runbook                                                      | Severity | When to Use                        |
| ------------------------------------------------------------ | -------- | ---------------------------------- |
| [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md)     | High     | Daemon won't start, keeps crashing |
| [DAEMON_MANAGER_OPERATIONS.md](DAEMON_MANAGER_OPERATIONS.md) | Medium   | DaemonManager health issues        |

### Data Synchronization

| Runbook                                                              | Severity | When to Use                       |
| -------------------------------------------------------------------- | -------- | --------------------------------- |
| [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md)             | High     | Cluster-wide sync failures        |
| [SYNC_HOST_CRITICAL.md](SYNC_HOST_CRITICAL.md)                       | Critical | Host stuck in critical sync state |
| [TRANSPORT_ESCALATION_FAILURES.md](TRANSPORT_ESCALATION_FAILURES.md) | High     | All transports failed for a node  |
| [ORPHAN_GAMES_DETECTION.md](ORPHAN_GAMES_DETECTION.md)               | High     | Games not synced from nodes       |

### Event System

| Runbook                                                      | Severity | When to Use                              |
| ------------------------------------------------------------ | -------- | ---------------------------------------- |
| [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) | High     | Events not flowing, subscribers failing  |
| [EVENT_WIRING_VERIFICATION.md](EVENT_WIRING_VERIFICATION.md) | Medium   | Verify event subscriptions are connected |

### Training & Parity

| Runbook                                                              | Severity | When to Use                          |
| -------------------------------------------------------------------- | -------- | ------------------------------------ |
| [TRAINING_LOOP_STALLED.md](TRAINING_LOOP_STALLED.md)                 | High     | Training jobs not progressing        |
| [PIPELINE_STAGE_FAILURES.md](PIPELINE_STAGE_FAILURES.md)             | High     | Export/train/evaluate stage failures |
| [PARITY_GATE_RESOLUTION.md](PARITY_GATE_RESOLUTION.md)               | High     | Parity gate blocking training        |
| [PARITY_MISMATCH_DEBUG.md](PARITY_MISMATCH_DEBUG.md)                 | Critical | TS/Python game replay mismatch       |
| [HEXAGONAL_PARITY_BUG.md](HEXAGONAL_PARITY_BUG.md)                   | Critical | Hexagonal board parity issues        |
| [GAME_DATA_CORRUPTION_RECOVERY.md](GAME_DATA_CORRUPTION_RECOVERY.md) | Critical | Corrupted game data (moves, phases)  |
| [WORK_QUEUE_STALLS.md](WORK_QUEUE_STALLS.md)                         | High     | Work queue not processing jobs       |

### Model Promotion & Feedback

| Runbook                                                                        | Severity | When to Use                              |
| ------------------------------------------------------------------------------ | -------- | ---------------------------------------- |
| [MODEL_PROMOTION_WORKFLOW.md](MODEL_PROMOTION_WORKFLOW.md)                     | High     | Model promotion issues, gauntlet failing |
| [MODEL_DISTRIBUTION_TROUBLESHOOTING.md](MODEL_DISTRIBUTION_TROUBLESHOOTING.md) | High     | Model sync, symlink, distribution issues |
| [FEEDBACK_LOOP_TROUBLESHOOTING.md](FEEDBACK_LOOP_TROUBLESHOOTING.md)           | High     | Feedback loops not updating              |
| [FEEDBACK_LOOP_DEGRADATION.md](FEEDBACK_LOOP_DEGRADATION.md)                   | Medium   | Feedback loops losing effectiveness      |

### GPU & Resources

| Runbook                                              | Severity | When to Use               |
| ---------------------------------------------------- | -------- | ------------------------- |
| [GPU_OOM_DEBUG.md](GPU_OOM_DEBUG.md)                 | High     | GPU out of memory errors  |
| [CLUSTER_GPU_STUCK.md](CLUSTER_GPU_STUCK.md)         | High     | GPU processes hanging     |
| [DISK_SPACE_MANAGEMENT.md](DISK_SPACE_MANAGEMENT.md) | High     | Disk full, cleanup needed |

### General Operations

| Runbook                                      | Severity | When to Use                |
| -------------------------------------------- | -------- | -------------------------- |
| [COORDINATOR_ERROR.md](COORDINATOR_ERROR.md) | Critical | Coordinator in error state |
| [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) | Critical | Major cluster failure      |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md)     | Info     | General debugging guidance |

---

## Quick Diagnosis Commands

### Check Overall Health

```bash
# P2P cluster status
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} nodes")
print(f"Role: {d.get(\"role\")}")
'

# Health endpoints
curl -s http://localhost:8790/health | jq
curl -s http://localhost:8790/ready | jq

# Daemon status
python scripts/launch_daemons.py --status
```

### Check Specific Subsystems

```bash
# Sync status
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
print(SyncCoordinator.get_instance().get_stats())
"

# Event router health
python -c "
from app.coordination.event_router import EventRouter
print(EventRouter.get_instance().get_health_summary())
"

# Daemon manager health
python -c "
from app.coordination.daemon_manager import DaemonManager
print(DaemonManager.get_instance().get_health_report())
"
```

---

## Severity Levels

| Level        | Response Time | Description                         |
| ------------ | ------------- | ----------------------------------- |
| **Critical** | < 5 minutes   | Cluster/training completely blocked |
| **High**     | < 30 minutes  | Major functionality impaired        |
| **Medium**   | < 2 hours     | Partial degradation                 |
| **Low**      | < 24 hours    | Minor issues, workarounds available |
| **Info**     | N/A           | Reference documentation             |

---

## Escalation Path

1. **On-call engineer** reviews alert, references this index
2. **Follow runbook** for specific issue type
3. **Check related runbooks** listed at bottom of each runbook
4. **Escalate** if issue persists after runbook steps

---

## Health Check Ports

| Service          | Port | Endpoint                        |
| ---------------- | ---- | ------------------------------- |
| P2P Orchestrator | 8770 | `/status`, `/health`            |
| Health Server    | 8790 | `/health`, `/ready`, `/metrics` |
| AI Service       | 8001 | `/health`                       |

---

## Related Documentation

- [../architecture/COORDINATION_SYSTEM.md](../architecture/COORDINATION_SYSTEM.md) - System architecture
- [../ENV_REFERENCE.md](../ENV_REFERENCE.md) - Environment variables
- [../../CLAUDE.md](../../CLAUDE.md) - AI assistant context
