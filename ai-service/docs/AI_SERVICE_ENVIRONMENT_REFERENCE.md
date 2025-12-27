# AI Service Environment Variables Reference

This document covers environment variables specific to the Python AI service cluster operations, coordination, and training infrastructure. For TypeScript/server variables, see the main `docs/operations/ENVIRONMENT_VARIABLES.md`.

## Table of Contents

- [Node Identity & Roles](#node-identity--roles)
- [Cluster Coordination](#cluster-coordination)
- [P2P Mesh Network](#p2p-mesh-network)
- [SSH & Connectivity](#ssh--connectivity)
- [Resource Management](#resource-management)
- [Process Management](#process-management)
- [Training Pipeline](#training-pipeline)
- [Sync & Transfer](#sync--transfer)
- [Circuit Breakers & Timeouts](#circuit-breakers--timeouts)
- [Health & Recovery](#health--recovery)
- [Logging & Debug](#logging--debug)
- [Storage & Paths](#storage--paths)

---

## Node Identity & Roles

### `RINGRIFT_NODE_ID`

| Property | Default  | Description            |
| -------- | -------- | ---------------------- |
| Type     | `string` | `socket.gethostname()` |

Unique identifier for this node in the cluster. Used for:

- Registering databases in ClusterManifest
- Tracking job ownership
- Event source identification

### `RINGRIFT_IS_COORDINATOR`

| Property | Default            | Description            |
| -------- | ------------------ | ---------------------- |
| Type     | `boolean`          | Auto-detected          |
| Values   | `1`, `true`, `yes` | Force coordinator role |

When set, node acts as cluster coordinator (orchestrates training, runs daemons).

### `RINGRIFT_ORCHESTRATOR`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `unknown`   |

Name of the orchestrator managing this node (for logging/identification).

### `RINGRIFT_BUILD_VERSION`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `dev`       |

Build/deployment version for tracking in events and logs.

---

## Cluster Coordination

### `RINGRIFT_COORDINATOR_URL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | Empty       |

URL of the cluster coordinator for centralized coordination.

### `RINGRIFT_CLUSTER_AUTH_TOKEN`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Authentication token for P2P cluster communication.

### `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Path to file containing auth token (alternative to inline token).

### `RINGRIFT_NFS_COORDINATION_PATH`

| Property | Default  | Description                         |
| -------- | -------- | ----------------------------------- |
| Type     | `string` | `/lambda/nfs/RingRift/coordination` |

Path to NFS-based coordination directory (Lambda GH200 nodes).

### `RINGRIFT_WORK_QUEUE_DB`

| Property | Default  | Description                       |
| -------- | -------- | --------------------------------- |
| Type     | `string` | `data/coordination/work_queue.db` |

Path to the work queue SQLite database.

### `RINGRIFT_COORDINATOR_DB`

| Property | Default  | Description                        |
| -------- | -------- | ---------------------------------- |
| Type     | `string` | `data/coordination/coordinator.db` |

Path to coordinator state database.

### `RINGRIFT_DLQ_PATH`

| Property | Default  | Description                              |
| -------- | -------- | ---------------------------------------- |
| Type     | `string` | `data/coordination/dead_letter_queue.db` |

Path to dead letter queue for failed events.

---

## P2P Mesh Network

### `RINGRIFT_P2P_SEEDS`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Comma-separated list of P2P seed node URLs for mesh discovery.

Example: `http://node1:8770,http://node2:8770`

### `RINGRIFT_P2P_PORT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `8770`      |

Port for P2P orchestrator HTTP endpoints.

### `RINGRIFT_P2P_LEADER_URL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Override URL for P2P leader node.

### `RINGRIFT_P2P_AGENT_MODE`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |

Enable P2P agent mode (worker node behavior).

### `RINGRIFT_P2P_AUTO_UPDATE`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |

Enable automatic P2P code updates from leader.

### `RINGRIFT_P2P_STARTUP_GRACE_PERIOD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |

Seconds to wait during P2P startup before killing processes.

---

## SSH & Connectivity

### `RINGRIFT_SSH_USER`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `ubuntu`    |

Default SSH username for cluster nodes.

### `RINGRIFT_SSH_KEY`

| Property | Default  | Description         |
| -------- | -------- | ------------------- |
| Type     | `string` | `~/.ssh/id_cluster` |

Path to SSH private key for cluster connections.

### `RINGRIFT_SSH_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |

Default SSH command timeout in seconds.

### `RINGRIFT_SSH_CONNECT_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |

SSH connection establishment timeout in seconds.

### `RINGRIFT_SSH_MAX_RETRIES`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `2`         |

Maximum SSH connection retry attempts.

### `RINGRIFT_SSH_RETRY_DELAY`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `1.0`       |

Delay between SSH retry attempts in seconds.

### `RINGRIFT_TCP_PROBE_TIMEOUT`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `3.0`       |

Timeout for TCP probe when selecting transport.

### `RINGRIFT_VAST_SSH_USER`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `root`      |

SSH username for Vast.ai instances.

---

## Resource Management

### `RINGRIFT_TARGET_UTIL_MIN`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `60`        |

Minimum target GPU utilization percentage.

### `RINGRIFT_TARGET_UTIL_MAX`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `80`        |

Maximum target GPU utilization percentage.

### `RINGRIFT_SCALE_UP_THRESHOLD`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `55`        |

Utilization threshold to scale up resources.

### `RINGRIFT_SCALE_DOWN_THRESHOLD`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `85`        |

Utilization threshold to scale down resources.

### `RINGRIFT_PID_KP` / `RINGRIFT_PID_KI` / `RINGRIFT_PID_KD`

| Property | Default | Description            |
| -------- | ------- | ---------------------- |
| Type     | `float` | `0.3` / `0.05` / `0.1` |

PID controller coefficients for resource allocation.

### `RINGRIFT_IDLE_CHECK_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |

Interval in seconds between idle resource checks.

### `RINGRIFT_IDLE_THRESHOLD`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `10.0`      |

GPU utilization percentage below which resource is "idle".
Legacy alias: `RINGRIFT_IDLE_GPU_THRESHOLD` (percent). Not the same as
`RINGRIFT_GPU_IDLE_THRESHOLD` (seconds before killing stuck processes).

### `RINGRIFT_IDLE_DURATION`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |

Seconds resource must be idle before action.

### `RINGRIFT_IDLE_RESOURCE_ENABLED`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |

Enable idle resource detection daemon.

### `RINGRIFT_MIN_MEMORY_GB`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `64`        |

Minimum system memory in GB for training.

### `RINGRIFT_MAX_QUEUE_SIZE`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1000`      |

Maximum work queue size.

### `RINGRIFT_MAX_SELFPLAY_CLUSTER`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `500`       |

Maximum concurrent selfplay jobs across cluster.

---

## Process Management

### `RINGRIFT_JOB_GRACE_PERIOD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |

Seconds to wait after SIGTERM before SIGKILL.

### `RINGRIFT_GPU_IDLE_THRESHOLD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `600`       |

Seconds of GPU idle before killing stuck processes.

### `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `128`       |

Maximum selfplay processes per node before cleanup.

---

## Training Pipeline

### `RINGRIFT_TRAINING_THRESHOLD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `500`       |

Minimum games before triggering training.

### `RINGRIFT_TRAINING_ENABLED`

| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |

Enable training on this node.

### `RINGRIFT_SELFPLAY_ENABLED`

| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |

Enable selfplay on this node.

### `RINGRIFT_RECORD_SELFPLAY_GAMES`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `true`      |

When enabled, selfplay games are recorded to the replay database. When disabled,
selfplay runs skip DB writes (JSONL/log output can still be emitted).

### `RINGRIFT_GAUNTLET_ENABLED`

| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |

Enable gauntlet evaluation on this node.

### `RINGRIFT_EXPORT_ENABLED`

| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |

Enable NPZ export on this node.

### `RINGRIFT_MAX_TRAINING_SAME_CONFIG`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |

Maximum concurrent training jobs for same board/player config.

### `RINGRIFT_MAX_TRAINING_TOTAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3`         |

Maximum total concurrent training jobs.

### `RINGRIFT_TRAINING_TIMEOUT_HOURS`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `24.0`      |

Training job timeout in hours.

### `RINGRIFT_TRAINING_MIN_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1200`      |

Minimum seconds between training runs for same config.

### `RINGRIFT_TRAINING_LOCK_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `7200`      |

Training lock timeout in seconds (2 hours).

---

## Sync & Transfer

### `RINGRIFT_DATA_SYNC_INTERVAL`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `120.0`     |

Interval between data sync cycles in seconds.

### `RINGRIFT_MODEL_SYNC_INTERVAL`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `600.0`     |

Interval between model sync cycles in seconds.

### `RINGRIFT_ELO_SYNC_INTERVAL`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `60.0`      |

Interval between Elo rating sync cycles in seconds.

### `RINGRIFT_REGISTRY_SYNC_INTERVAL`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `120.0`     |

Interval between registry sync cycles in seconds.

### `RINGRIFT_SYNC_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

Sync operation timeout in seconds.

### `RINGRIFT_MAX_SYNCS_PER_HOST`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `2`         |

Maximum concurrent syncs to single host.

### `RINGRIFT_MAX_SYNCS_CLUSTER`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `10`        |

Maximum concurrent syncs across cluster.

### `RINGRIFT_SYNC_LOCK_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |

Sync lock timeout in seconds.

### `RINGRIFT_S3_BUCKET`

| Property | Default  | Description                |
| -------- | -------- | -------------------------- |
| Type     | `string` | `ringrift-models-20251214` |

S3 bucket for model backup.

### `RINGRIFT_STORAGE_PROVIDER`

| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |

Storage provider type (for distributed storage).

---

## Circuit Breakers & Timeouts

### `RINGRIFT_CB_FAILURE_THRESHOLD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `5`         |

Failures before circuit breaker opens.

### `RINGRIFT_CB_RECOVERY_TIMEOUT`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `60.0`      |

Circuit breaker recovery timeout in seconds.

### `RINGRIFT_CB_MAX_BACKOFF`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `600.0`     |

Maximum circuit breaker backoff in seconds.

### `RINGRIFT_CB_HALF_OPEN_MAX_CALLS`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |

Calls allowed in half-open state.

### `RINGRIFT_CIRCUIT_BREAKER_RECOVERY`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

General circuit breaker recovery time in seconds.

### `RINGRIFT_LOCK_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3600`      |

General lock timeout in seconds.

### `RINGRIFT_LOCK_ACQUIRE_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |

Lock acquisition timeout in seconds.

### `RINGRIFT_LOCK_RETRY_INTERVAL`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `1.0`       |

Interval between lock acquisition retries.

### `RINGRIFT_CONNECT_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `45`        |

General connection timeout in seconds.

### `RINGRIFT_OPERATION_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `180`       |

General operation timeout in seconds.

### `RINGRIFT_HTTP_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |

HTTP request timeout in seconds.

### `RINGRIFT_EVENT_HANDLER_TIMEOUT`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `30.0`      |

Event handler timeout in seconds.

---

## Health & Recovery

### `RINGRIFT_HEALTH_PORT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `8790`      |

Port for health check HTTP endpoints.

### `RINGRIFT_HEALTH_CACHE_TTL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |

Health check cache TTL in seconds.

### `RINGRIFT_HEALTH_SSH_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `5`         |

SSH timeout for health checks in seconds.

### `RINGRIFT_HEARTBEAT_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |

Heartbeat interval in seconds.

### `RINGRIFT_HEARTBEAT_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `90`        |

Heartbeat timeout (node considered dead).

### `RINGRIFT_STALE_CLEANUP_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |

Interval for stale resource cleanup in seconds.

### `RINGRIFT_NODE_RECOVERY_ENABLED`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |

Enable automatic node recovery.

### `RINGRIFT_NODE_RECOVERY_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

Node recovery check interval in seconds.

### `RINGRIFT_PREEMPTIVE_RECOVERY`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |

Enable preemptive node recovery.

### `RINGRIFT_EPHEMERAL_HEARTBEAT_TIMEOUT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |

Heartbeat timeout for ephemeral nodes.

### `RINGRIFT_CHECKPOINT_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

Checkpoint save interval in seconds.

---

## Lambda Idle Management

### `RINGRIFT_LAMBDA_IDLE_ENABLED`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |

Enable Lambda idle detection daemon.

### `RINGRIFT_LAMBDA_IDLE_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

Lambda idle check interval in seconds.

### `RINGRIFT_LAMBDA_IDLE_THRESHOLD`

| Property | Default | Description |
| -------- | ------- | ----------- |
| Type     | `float` | `5.0`       |

Lambda GPU utilization threshold for idle.

### `RINGRIFT_LAMBDA_IDLE_DURATION`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1800`      |

Lambda idle duration before shutdown (30 min).

### `RINGRIFT_LAMBDA_MIN_NODES`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |

Minimum Lambda nodes to retain.

---

## Logging & Debug

### `RINGRIFT_LOG_LEVEL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `INFO`      |

Log level (DEBUG, INFO, WARNING, ERROR).

### `RINGRIFT_LOG_FORMAT`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `default`   |

Log format (default, compact, detailed, json).

### `RINGRIFT_LOG_JSON`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |

Enable JSON structured logging.

### `RINGRIFT_LOG_FILE`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Path to log file.

### `RINGRIFT_TRACE_DEBUG`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |

Enable trace-level debug logging.

### `RINGRIFT_REQUIRE_CRITICAL_IMPORTS`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `0`         |

Require critical imports at startup.

### `RINGRIFT_SKIP_MASTER_LOOP_CHECK`

| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `0`         |

Skip master loop guard check.

---

## Storage & Paths

### `RINGRIFT_AI_SERVICE_PATH`

| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |

Path to ai-service directory.

### `RINGRIFT_AI_SERVICE_DIR`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `.`         |

Alternative path specification.

### `RINGRIFT_DATA_DIR`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `data`      |

Data directory path.

### `RINGRIFT_CONFIG_PATH`

| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |

Configuration file path.

### `RINGRIFT_ELO_DB`

| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |

Path to Elo database.

### `RINGRIFT_SELFPLAY_DB_PATH`

| Property | Default  | Description              |
| -------- | -------- | ------------------------ |
| Type     | `string` | `data/games/selfplay.db` |

Override path for the selfplay replay database when no explicit DB path is supplied.

### `RINGRIFT_SNAPSHOT_INTERVAL`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `20`        |

Snapshot interval (in moves) for replay DB snapshots.

### `RINGRIFT_NFS_PATH`

| Property | Default  | Description            |
| -------- | -------- | ---------------------- |
| Type     | `string` | `/lambda/nfs/RingRift` |

NFS mount path for shared storage.

### `RINGRIFT_WAL_DIR`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `data/wal/` |

Write-ahead log directory.

---

## Transport Health

### `RINGRIFT_TRANSPORT_FAILURE_THRESHOLD`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3`         |

Failures before transport is disabled.

### `RINGRIFT_TRANSPORT_DISABLE_DURATION`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |

Duration to disable failed transport.

---

## Replay & Debug

### `RINGRIFT_TS_REPLAY_DUMP_DIR`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Directory for TypeScript replay dumps.

### `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K`

| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |

Comma-separated move numbers to dump state.

---

## Watchdog Configuration

Watchdog daemons use environment variable prefixes for configuration:

```bash
# Cluster watchdog
RINGRIFT_WATCHDOG_CHECK_INTERVAL=60
RINGRIFT_WATCHDOG_TIMEOUT=300

# Node recovery
RINGRIFT_NODE_RECOVERY_ENABLED=1
RINGRIFT_NODE_RECOVERY_INTERVAL=300
```

The `from_env()` factory method on config classes reads these automatically.

---

## See Also

- `docs/operations/ENVIRONMENT_VARIABLES.md` - Main TypeScript/server variables
- `docs/coordination/DAEMON_INDEX.md` - Daemon documentation
- `app/config/env.py` - Centralized environment configuration
- `app/config/coordination_defaults.py` - Coordination default values
