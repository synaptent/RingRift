# P2P Orchestrator Admin API Reference

This document describes the HTTP API endpoints exposed by the P2P orchestrator (`scripts/p2p_orchestrator.py`). The orchestrator runs on port **8770** by default.

## Base URL

```
http://<node-ip>:8770
```

## Authentication

All endpoints accept an optional `Authorization` header with the P2P auth token:

```
Authorization: Bearer <P2P_AUTH_TOKEN>
```

## Core Endpoints

### Health & Status

#### GET /health

Check if the orchestrator is running.

**Response:**

```json
{
  "status": "ok",
  "node_id": "my-node",
  "uptime_seconds": 3600,
  "is_leader": true
}
```

#### GET /status

Get detailed cluster status including all peers.

**Response:**

```json
{
  "node_id": "my-node",
  "is_leader": true,
  "leader_id": "my-node",
  "peers": [...],
  "jobs": [...],
  "cluster_health": "healthy"
}
```

#### GET /

Root endpoint returning basic orchestrator info.

---

## Admin Endpoints

### POST /admin/unretire

Unretire a specific peer node that was marked as retired.

**Request Body:**

```json
{
  "node_id": "vast-123456"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Node 'vast-123456' has been unretired",
  "node_id": "vast-123456"
}
```

**Use Case:** Restore a node that was incorrectly marked as retired due to temporary connectivity issues.

### GET /admin/purge_retired

Remove all retired peers from the cluster state.

**Response:**

```json
{
  "status": "success",
  "purged_count": 5,
  "purged_nodes": ["node-1", "node-2", ...]
}
```

### GET /admin/purge_stale

Remove peers that haven't sent heartbeats recently.

**Query Parameters:**

- `max_age_hours`: Maximum heartbeat age (default: 24)

**Response:**

```json
{
  "status": "success",
  "purged_count": 3
}
```

---

## Election & Leadership

### POST /heartbeat

Send heartbeat to maintain peer connectivity.

**Request Body:**

```json
{
  "node_id": "sender-node",
  "timestamp": 1702900000,
  "jobs": [...],
  "resources": {...}
}
```

### POST /election

Participate in leader election (Bully algorithm).

### POST /election/lease

Request or renew a leader lease from a voter.

### GET /election/grant

Check current lease grant status.

### POST /coordinator

Announce new leader to cluster.

---

## Job Management

### POST /start_job

Start a job on this node (sent by leader).

**Request Body:**

```json
{
  "job_id": "selfplay-001",
  "job_type": "selfplay",
  "config": {...}
}
```

### POST /stop_job

Stop a running job.

**Request Body:**

```json
{
  "job_id": "selfplay-001"
}
```

### POST /job/kill

Forcefully kill a stuck job (SIGKILL).

**Request Body:**

```json
{
  "job_id": "selfplay-001"
}
```

### POST /cleanup

Trigger disk cleanup on this node.

### POST /restart_stuck_jobs

Restart jobs that are stuck.

---

## Git Operations

### GET /git/status

Get current git status (branch, commit, dirty state).

**Response:**

```json
{
  "branch": "main",
  "commit": "abc123",
  "dirty": false,
  "behind": 0,
  "ahead": 0
}
```

### POST /git/update

Pull latest code from remote.

**Response:**

```json
{
  "status": "success",
  "previous_commit": "abc123",
  "current_commit": "def456"
}
```

---

## Selfplay & Training

### POST /selfplay/start

Dispatch GPU selfplay to a node.

### POST /reduce_selfplay

Reduce selfplay parallelism across cluster.

### POST /training/start

Start a training run.

### GET /training/status

Get current training status.

### POST /training/update

Update training progress.

### POST /training/nnue/start

Start NNUE training specifically.

### POST /training/cmaes/start

Start CMA-ES optimization.

---

## Data Sync

### POST /sync/start

Initiate data sync between nodes.

### GET /sync/status

Check sync status.

### POST /sync/pull

Pull data from this node.

**Request Body:**

```json
{
  "patterns": ["data/selfplay/*.jsonl"],
  "since_timestamp": 1702900000
}
```

### GET /sync/file

Download a specific file.

**Query Parameters:**

- `path`: File path relative to ai-service

### POST /sync/job_update

Update job sync status.

### POST /sync/training

Priority sync for training data.

---

## Tournament & Evaluation

### POST /tournament/start

Start an Elo tournament.

### POST /tournament/match

Report match result.

### POST /tournament/play_elo_match

Execute a single Elo match.

### GET /tournament/status

Get tournament progress.

### POST /tournament/result

Submit tournament result.

### POST /tournament/ssh_start

Start SSH-based distributed tournament.

### GET /tournament/ssh_status

Check SSH tournament status.

### POST /tournament/ssh_cancel

Cancel SSH tournament.

---

## CMA-ES Optimization

### POST /cmaes/start

Start CMA-ES optimization run.

### POST /cmaes/evaluate

Submit evaluation result.

### GET /cmaes/status

Check CMA-ES progress.

### POST /cmaes/result

Submit final CMA-ES result.

---

## Gauntlet Evaluation

### POST /gauntlet/execute

Run gauntlet evaluation.

### GET /gauntlet/status

Check gauntlet progress.

### POST /gauntlet/quick-eval

Run quick evaluation.

---

## Improvement Pipeline

### POST /improvement/start

Start self-improvement loop.

### GET /improvement/status

Check improvement status.

### POST /improvement/phase_complete

Report phase completion.

### GET /improvement_cycles/status

Get improvement cycle metrics.

### GET /improvement_cycles/leaderboard

Get model improvement leaderboard.

---

## A/B Testing

### POST /abtest/create

Create new A/B test.

### POST /abtest/result

Submit A/B test result.

### GET /abtest/status

Get A/B test status.

### GET /abtest/list

List all A/B tests.

### POST /abtest/cancel

Cancel running A/B test.

### POST /abtest/run

Execute A/B test manually.

---

## Rollback Management

### GET /rollback/status

Get rollback system status.

### GET /rollback/candidates

List rollback candidates.

### POST /rollback/execute

Execute manual rollback.

### POST /rollback/auto

Trigger automatic rollback evaluation.

---

## Metrics & Analytics

### GET /metrics

Get orchestrator metrics.

### GET /metrics/prometheus

Get Prometheus-formatted metrics.

### GET /elo/table

Get Elo leaderboard table.

### GET /elo/history

Get Elo history over time.

### GET /nodes/table

Get cluster nodes table.

### GET /games/analytics

Get game analytics.

### GET /training/metrics

Get training metrics.

### GET /mcts/stats

Get MCTS statistics.

### GET /trends/summary

Get trend summary.

### GET /trends/history

Get historical trends.

---

## Connectivity Diagnostics

### GET /connectivity/diagnose/{node_id}

Diagnose connectivity to a specific node.

### GET /connectivity/transport_stats

Get transport layer statistics.

### POST /connectivity/probe_vast

Probe Vast.ai nodes for connectivity.

---

## Registry Management

### POST /register

Register a new peer.

### GET /registry/status

Get registry status.

### POST /registry/update_vast

Update Vast.ai node info.

### POST /registry/update_aws

Update AWS node info.

### POST /registry/update_tailscale

Update Tailscale node info.

### POST /registry/save_yaml

Save registry to YAML.

---

## Data Manifest

### GET /data_manifest

Get local data manifest.

### GET /cluster_data_manifest

Get cluster-wide data manifest.

### POST /refresh_manifest

Refresh data manifest.

---

## Pipeline Control

### POST /pipeline/start

Start training pipeline.

### GET /pipeline/status

Get pipeline status.

### POST /pipeline/selfplay_worker

Register as selfplay worker.

---

## Dashboard

### GET /dashboard

HTML dashboard for monitoring.

---

## Example Usage

### Check Cluster Health

```bash
curl http://localhost:8770/health
```

### Unretire a Node

```bash
curl -X POST http://localhost:8770/admin/unretire \
  -H "Content-Type: application/json" \
  -d '{"node_id": "vast-123456"}'
```

### Trigger Git Update Across Cluster

```bash
curl -X POST http://localhost:8770/api/cluster/git/update
```

### Get Elo Leaderboard

```bash
curl http://localhost:8770/api/elo/leaderboard
```

### Start Tournament

```bash
curl -X POST http://localhost:8770/tournament/start \
  -H "Content-Type: application/json" \
  -d '{
    "agents": ["gumbel_150", "mcts_1000", "descent"],
    "games_per_match": 100,
    "board_type": "square8"
  }'
```

---

## Error Handling

All endpoints return JSON responses. Errors include:

```json
{
  "status": "error",
  "error": "Description of the error",
  "code": 400
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (missing/invalid auth token)
- `404`: Resource not found
- `500`: Internal server error

---

## Port Configuration

The orchestrator port can be configured via environment variable:

```bash
export P2P_PORT=8770
```

Or command line:

```bash
python scripts/p2p_orchestrator.py --port 8770
```
