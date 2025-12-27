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

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `socket.gethostname()` |

Unique identifier for this node in the cluster. Used for:
=======
| Property | Default  | Description            |
| -------- | -------- | ---------------------- |
| Type     | `string` | `socket.gethostname()` |

Unique identifier for this node in the cluster. Used for:

>>>>>>> Stashed changes
- Registering databases in ClusterManifest
- Tracking job ownership
- Event source identification

### `RINGRIFT_IS_COORDINATOR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | Auto-detected |
| Values | `1`, `true`, `yes` | Force coordinator role |
=======
| Property | Default            | Description            |
| -------- | ------------------ | ---------------------- |
| Type     | `boolean`          | Auto-detected          |
| Values   | `1`, `true`, `yes` | Force coordinator role |
>>>>>>> Stashed changes

When set, node acts as cluster coordinator (orchestrates training, runs daemons).

### `RINGRIFT_ORCHESTRATOR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `unknown` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `unknown`   |
>>>>>>> Stashed changes

Name of the orchestrator managing this node (for logging/identification).

### `RINGRIFT_BUILD_VERSION`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `dev` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `dev`       |
>>>>>>> Stashed changes

Build/deployment version for tracking in events and logs.

---

## Cluster Coordination

### `RINGRIFT_COORDINATOR_URL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | Empty |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | Empty       |
>>>>>>> Stashed changes

URL of the cluster coordinator for centralized coordination.

### `RINGRIFT_CLUSTER_AUTH_TOKEN`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Authentication token for P2P cluster communication.

### `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Path to file containing auth token (alternative to inline token).

### `RINGRIFT_NFS_COORDINATION_PATH`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `/lambda/nfs/RingRift/coordination` |
=======
| Property | Default  | Description                         |
| -------- | -------- | ----------------------------------- |
| Type     | `string` | `/lambda/nfs/RingRift/coordination` |
>>>>>>> Stashed changes

Path to NFS-based coordination directory (Lambda GH200 nodes).

### `RINGRIFT_WORK_QUEUE_DB`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `data/coordination/work_queue.db` |
=======
| Property | Default  | Description                       |
| -------- | -------- | --------------------------------- |
| Type     | `string` | `data/coordination/work_queue.db` |
>>>>>>> Stashed changes

Path to the work queue SQLite database.

### `RINGRIFT_COORDINATOR_DB`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `data/coordination/coordinator.db` |
=======
| Property | Default  | Description                        |
| -------- | -------- | ---------------------------------- |
| Type     | `string` | `data/coordination/coordinator.db` |
>>>>>>> Stashed changes

Path to coordinator state database.

### `RINGRIFT_DLQ_PATH`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `data/coordination/dead_letter_queue.db` |
=======
| Property | Default  | Description                              |
| -------- | -------- | ---------------------------------------- |
| Type     | `string` | `data/coordination/dead_letter_queue.db` |
>>>>>>> Stashed changes

Path to dead letter queue for failed events.

---

## P2P Mesh Network

### `RINGRIFT_P2P_SEEDS`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Comma-separated list of P2P seed node URLs for mesh discovery.

Example: `http://node1:8770,http://node2:8770`

### `RINGRIFT_P2P_PORT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `8770` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `8770`      |
>>>>>>> Stashed changes

Port for P2P orchestrator HTTP endpoints.

### `RINGRIFT_P2P_LEADER_URL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Override URL for P2P leader node.

### `RINGRIFT_P2P_AGENT_MODE`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `false` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |
>>>>>>> Stashed changes

Enable P2P agent mode (worker node behavior).

### `RINGRIFT_P2P_AUTO_UPDATE`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `false` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |
>>>>>>> Stashed changes

Enable automatic P2P code updates from leader.

### `RINGRIFT_P2P_STARTUP_GRACE_PERIOD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `120` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |
>>>>>>> Stashed changes

Seconds to wait during P2P startup before killing processes.

---

## SSH & Connectivity

### `RINGRIFT_SSH_USER`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `ubuntu` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `ubuntu`    |
>>>>>>> Stashed changes

Default SSH username for cluster nodes.

### `RINGRIFT_SSH_KEY`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `~/.ssh/id_cluster` |
=======
| Property | Default  | Description         |
| -------- | -------- | ------------------- |
| Type     | `string` | `~/.ssh/id_cluster` |
>>>>>>> Stashed changes

Path to SSH private key for cluster connections.

### `RINGRIFT_SSH_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `60` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |
>>>>>>> Stashed changes

Default SSH command timeout in seconds.

### `RINGRIFT_SSH_CONNECT_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `30` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |
>>>>>>> Stashed changes

SSH connection establishment timeout in seconds.

### `RINGRIFT_SSH_MAX_RETRIES`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `2` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `2`         |
>>>>>>> Stashed changes

Maximum SSH connection retry attempts.

### `RINGRIFT_SSH_RETRY_DELAY`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `1.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `1.0`       |
>>>>>>> Stashed changes

Delay between SSH retry attempts in seconds.

### `RINGRIFT_TCP_PROBE_TIMEOUT`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `3.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `3.0`       |
>>>>>>> Stashed changes

Timeout for TCP probe when selecting transport.

### `RINGRIFT_VAST_SSH_USER`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `root` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `root`      |
>>>>>>> Stashed changes

SSH username for Vast.ai instances.

---

## Resource Management

### `RINGRIFT_TARGET_UTIL_MIN`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `60` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `60`        |
>>>>>>> Stashed changes

Minimum target GPU utilization percentage.

### `RINGRIFT_TARGET_UTIL_MAX`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `80` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `80`        |
>>>>>>> Stashed changes

Maximum target GPU utilization percentage.

### `RINGRIFT_SCALE_UP_THRESHOLD`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `55` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `55`        |
>>>>>>> Stashed changes

Utilization threshold to scale up resources.

### `RINGRIFT_SCALE_DOWN_THRESHOLD`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `85` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `85`        |
>>>>>>> Stashed changes

Utilization threshold to scale down resources.

### `RINGRIFT_PID_KP` / `RINGRIFT_PID_KI` / `RINGRIFT_PID_KD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `float` | `0.3` / `0.05` / `0.1` |
=======
| Property | Default | Description            |
| -------- | ------- | ---------------------- |
| Type     | `float` | `0.3` / `0.05` / `0.1` |
>>>>>>> Stashed changes

PID controller coefficients for resource allocation.

### `RINGRIFT_IDLE_CHECK_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `60` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |
>>>>>>> Stashed changes

Interval in seconds between idle resource checks.

### `RINGRIFT_IDLE_THRESHOLD`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `10.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `10.0`      |
>>>>>>> Stashed changes

GPU utilization percentage below which resource is "idle".

### `RINGRIFT_IDLE_DURATION`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `120` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |
>>>>>>> Stashed changes

Seconds resource must be idle before action.

### `RINGRIFT_IDLE_RESOURCE_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `1` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |
>>>>>>> Stashed changes

Enable idle resource detection daemon.

### `RINGRIFT_MIN_MEMORY_GB`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `64` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `64`        |
>>>>>>> Stashed changes

Minimum system memory in GB for training.

### `RINGRIFT_MAX_QUEUE_SIZE`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1000` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1000`      |
>>>>>>> Stashed changes

Maximum work queue size.

### `RINGRIFT_MAX_SELFPLAY_CLUSTER`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `500` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `500`       |
>>>>>>> Stashed changes

Maximum concurrent selfplay jobs across cluster.

---

## Process Management

### `RINGRIFT_JOB_GRACE_PERIOD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `60` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |
>>>>>>> Stashed changes

Seconds to wait after SIGTERM before SIGKILL.

### `RINGRIFT_GPU_IDLE_THRESHOLD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `600` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `600`       |
>>>>>>> Stashed changes

Seconds of GPU idle before killing stuck processes.

### `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `128` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `128`       |
>>>>>>> Stashed changes

Maximum selfplay processes per node before cleanup.

---

## Training Pipeline

### `RINGRIFT_TRAINING_THRESHOLD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `500` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `500`       |
>>>>>>> Stashed changes

Minimum games before triggering training.

### `RINGRIFT_TRAINING_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | Auto-detected |
=======
| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |
>>>>>>> Stashed changes

Enable training on this node.

### `RINGRIFT_SELFPLAY_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | Auto-detected |
=======
| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |
>>>>>>> Stashed changes

Enable selfplay on this node.

### `RINGRIFT_GAUNTLET_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | Auto-detected |
=======
| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |
>>>>>>> Stashed changes

Enable gauntlet evaluation on this node.

### `RINGRIFT_EXPORT_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | Auto-detected |
=======
| Property | Default   | Description   |
| -------- | --------- | ------------- |
| Type     | `boolean` | Auto-detected |
>>>>>>> Stashed changes

Enable NPZ export on this node.

### `RINGRIFT_MAX_TRAINING_SAME_CONFIG`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |
>>>>>>> Stashed changes

Maximum concurrent training jobs for same board/player config.

### `RINGRIFT_MAX_TRAINING_TOTAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `3` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3`         |
>>>>>>> Stashed changes

Maximum total concurrent training jobs.

### `RINGRIFT_TRAINING_TIMEOUT_HOURS`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `24.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `24.0`      |
>>>>>>> Stashed changes

Training job timeout in hours.

### `RINGRIFT_TRAINING_MIN_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1200` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1200`      |
>>>>>>> Stashed changes

Minimum seconds between training runs for same config.

### `RINGRIFT_TRAINING_LOCK_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `7200` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `7200`      |
>>>>>>> Stashed changes

Training lock timeout in seconds (2 hours).

---

## Sync & Transfer

### `RINGRIFT_DATA_SYNC_INTERVAL`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `120.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `120.0`     |
>>>>>>> Stashed changes

Interval between data sync cycles in seconds.

### `RINGRIFT_MODEL_SYNC_INTERVAL`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `600.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `600.0`     |
>>>>>>> Stashed changes

Interval between model sync cycles in seconds.

### `RINGRIFT_ELO_SYNC_INTERVAL`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `60.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `60.0`      |
>>>>>>> Stashed changes

Interval between Elo rating sync cycles in seconds.

### `RINGRIFT_REGISTRY_SYNC_INTERVAL`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `120.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `120.0`     |
>>>>>>> Stashed changes

Interval between registry sync cycles in seconds.

### `RINGRIFT_SYNC_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

Sync operation timeout in seconds.

### `RINGRIFT_MAX_SYNCS_PER_HOST`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `2` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `2`         |
>>>>>>> Stashed changes

Maximum concurrent syncs to single host.

### `RINGRIFT_MAX_SYNCS_CLUSTER`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `10` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `10`        |
>>>>>>> Stashed changes

Maximum concurrent syncs across cluster.

### `RINGRIFT_SYNC_LOCK_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `120` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |
>>>>>>> Stashed changes

Sync lock timeout in seconds.

### `RINGRIFT_S3_BUCKET`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `ringrift-models-20251214` |
=======
| Property | Default  | Description                |
| -------- | -------- | -------------------------- |
| Type     | `string` | `ringrift-models-20251214` |
>>>>>>> Stashed changes

S3 bucket for model backup.

### `RINGRIFT_STORAGE_PROVIDER`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | Auto-detected |
=======
| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |
>>>>>>> Stashed changes

Storage provider type (for distributed storage).

---

## Circuit Breakers & Timeouts

### `RINGRIFT_CB_FAILURE_THRESHOLD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `5` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `5`         |
>>>>>>> Stashed changes

Failures before circuit breaker opens.

### `RINGRIFT_CB_RECOVERY_TIMEOUT`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `60.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `60.0`      |
>>>>>>> Stashed changes

Circuit breaker recovery timeout in seconds.

### `RINGRIFT_CB_MAX_BACKOFF`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `600.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `600.0`     |
>>>>>>> Stashed changes

Maximum circuit breaker backoff in seconds.

### `RINGRIFT_CB_HALF_OPEN_MAX_CALLS`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |
>>>>>>> Stashed changes

Calls allowed in half-open state.

### `RINGRIFT_CIRCUIT_BREAKER_RECOVERY`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

General circuit breaker recovery time in seconds.

### `RINGRIFT_LOCK_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `3600` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3600`      |
>>>>>>> Stashed changes

General lock timeout in seconds.

### `RINGRIFT_LOCK_ACQUIRE_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `60` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |
>>>>>>> Stashed changes

Lock acquisition timeout in seconds.

### `RINGRIFT_LOCK_RETRY_INTERVAL`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `1.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `1.0`       |
>>>>>>> Stashed changes

Interval between lock acquisition retries.

### `RINGRIFT_CONNECT_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `45` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `45`        |
>>>>>>> Stashed changes

General connection timeout in seconds.

### `RINGRIFT_OPERATION_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `180` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `180`       |
>>>>>>> Stashed changes

General operation timeout in seconds.

### `RINGRIFT_HTTP_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `30` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |
>>>>>>> Stashed changes

HTTP request timeout in seconds.

### `RINGRIFT_EVENT_HANDLER_TIMEOUT`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `30.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `30.0`      |
>>>>>>> Stashed changes

Event handler timeout in seconds.

---

## Health & Recovery

### `RINGRIFT_HEALTH_PORT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `8790` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `8790`      |
>>>>>>> Stashed changes

Port for health check HTTP endpoints.

### `RINGRIFT_HEALTH_CACHE_TTL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `30` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |
>>>>>>> Stashed changes

Health check cache TTL in seconds.

### `RINGRIFT_HEALTH_SSH_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `5` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `5`         |
>>>>>>> Stashed changes

SSH timeout for health checks in seconds.

### `RINGRIFT_HEARTBEAT_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `30` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `30`        |
>>>>>>> Stashed changes

Heartbeat interval in seconds.

### `RINGRIFT_HEARTBEAT_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `90` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `90`        |
>>>>>>> Stashed changes

Heartbeat timeout (node considered dead).

### `RINGRIFT_STALE_CLEANUP_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `60` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `60`        |
>>>>>>> Stashed changes

Interval for stale resource cleanup in seconds.

### `RINGRIFT_NODE_RECOVERY_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `1` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |
>>>>>>> Stashed changes

Enable automatic node recovery.

### `RINGRIFT_NODE_RECOVERY_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

Node recovery check interval in seconds.

### `RINGRIFT_PREEMPTIVE_RECOVERY`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `1` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |
>>>>>>> Stashed changes

Enable preemptive node recovery.

### `RINGRIFT_EPHEMERAL_HEARTBEAT_TIMEOUT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `120` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `120`       |
>>>>>>> Stashed changes

Heartbeat timeout for ephemeral nodes.

### `RINGRIFT_CHECKPOINT_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

Checkpoint save interval in seconds.

---

## Lambda Idle Management

### `RINGRIFT_LAMBDA_IDLE_ENABLED`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `1` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `1`         |
>>>>>>> Stashed changes

Enable Lambda idle detection daemon.

### `RINGRIFT_LAMBDA_IDLE_INTERVAL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

Lambda idle check interval in seconds.

### `RINGRIFT_LAMBDA_IDLE_THRESHOLD`

| Property | Default | Description |
<<<<<<< Updated upstream
|----------|---------|-------------|
| Type | `float` | `5.0` |
=======
| -------- | ------- | ----------- |
| Type     | `float` | `5.0`       |
>>>>>>> Stashed changes

Lambda GPU utilization threshold for idle.

### `RINGRIFT_LAMBDA_IDLE_DURATION`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1800` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1800`      |
>>>>>>> Stashed changes

Lambda idle duration before shutdown (30 min).

### `RINGRIFT_LAMBDA_MIN_NODES`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `1` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `1`         |
>>>>>>> Stashed changes

Minimum Lambda nodes to retain.

---

## Logging & Debug

### `RINGRIFT_LOG_LEVEL`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `INFO` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `INFO`      |
>>>>>>> Stashed changes

Log level (DEBUG, INFO, WARNING, ERROR).

### `RINGRIFT_LOG_FORMAT`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `default` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `default`   |
>>>>>>> Stashed changes

Log format (default, compact, detailed, json).

### `RINGRIFT_LOG_JSON`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `false` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |
>>>>>>> Stashed changes

Enable JSON structured logging.

### `RINGRIFT_LOG_FILE`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Path to log file.

### `RINGRIFT_TRACE_DEBUG`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `false` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `false`     |
>>>>>>> Stashed changes

Enable trace-level debug logging.

### `RINGRIFT_REQUIRE_CRITICAL_IMPORTS`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `0` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `0`         |
>>>>>>> Stashed changes

Require critical imports at startup.

### `RINGRIFT_SKIP_MASTER_LOOP_CHECK`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `boolean` | `0` |
=======
| Property | Default   | Description |
| -------- | --------- | ----------- |
| Type     | `boolean` | `0`         |
>>>>>>> Stashed changes

Skip master loop guard check.

---

## Storage & Paths

### `RINGRIFT_AI_SERVICE_PATH`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | Auto-detected |
=======
| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |
>>>>>>> Stashed changes

Path to ai-service directory.

### `RINGRIFT_AI_SERVICE_DIR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `.` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `.`         |
>>>>>>> Stashed changes

Alternative path specification.

### `RINGRIFT_DATA_DIR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `data` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `data`      |
>>>>>>> Stashed changes

Data directory path.

### `RINGRIFT_CONFIG_PATH`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | Auto-detected |
=======
| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |
>>>>>>> Stashed changes

Configuration file path.

### `RINGRIFT_ELO_DB`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | Auto-detected |
=======
| Property | Default  | Description   |
| -------- | -------- | ------------- |
| Type     | `string` | Auto-detected |
>>>>>>> Stashed changes

Path to Elo database.

### `RINGRIFT_NFS_PATH`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `/lambda/nfs/RingRift` |
=======
| Property | Default  | Description            |
| -------- | -------- | ---------------------- |
| Type     | `string` | `/lambda/nfs/RingRift` |
>>>>>>> Stashed changes

NFS mount path for shared storage.

### `RINGRIFT_WAL_DIR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | `data/wal/` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | `data/wal/` |
>>>>>>> Stashed changes

Write-ahead log directory.

---

## Transport Health

### `RINGRIFT_TRANSPORT_FAILURE_THRESHOLD`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `3` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `3`         |
>>>>>>> Stashed changes

Failures before transport is disabled.

### `RINGRIFT_TRANSPORT_DISABLE_DURATION`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `number` | `300` |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `number` | `300`       |
>>>>>>> Stashed changes

Duration to disable failed transport.

---

## Replay & Debug

### `RINGRIFT_TS_REPLAY_DUMP_DIR`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

Directory for TypeScript replay dumps.

### `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K`

<<<<<<< Updated upstream
| Property | Default | Description |
|----------|---------|-------------|
| Type | `string` | None |
=======
| Property | Default  | Description |
| -------- | -------- | ----------- |
| Type     | `string` | None        |
>>>>>>> Stashed changes

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
