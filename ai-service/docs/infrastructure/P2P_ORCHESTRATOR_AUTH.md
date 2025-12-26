# P2P Orchestrator Auth

This note documents the security model for the peer-to-peer orchestrator
(`scripts/p2p_orchestrator.py`) so cluster deployments stay consistent and we
don’t accidentally run unauthenticated control planes on public hosts.

## When to require auth

If **any** orchestrator node is reachable outside a trusted LAN/VPC (public IP,
shared reverse proxy, wide VPN), run with `--require-auth`.

## Auth model

- A single shared bearer token is used across the cluster.
- Nodes attach it to peer requests via:
  - `Authorization: Bearer <token>`
- Nodes validate that header on non-read endpoints.

## Token sources (priority order)

1. CLI: `--auth-token`
2. Env: `RINGRIFT_CLUSTER_AUTH_TOKEN`
3. Env file: `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE` (preferred)

When `--require-auth` is set, startup fails unless a token is resolved from one
of the sources above.

## Recommended: token file

Use a file so the secret isn’t exposed in process listings or environment dumps.

Example:

```bash
sudo install -d -m 700 /etc/ringrift
sudo sh -c 'openssl rand -hex 32 > /etc/ringrift/p2p_orchestrator.token'
sudo chmod 600 /etc/ringrift/p2p_orchestrator.token
```

Then set:

```bash
export RINGRIFT_CLUSTER_AUTH_TOKEN_FILE=/etc/ringrift/p2p_orchestrator.token
```

## Reverse proxy requirements

If you run the orchestrator behind nginx, the proxy must forward the auth
header:

```
proxy_set_header Authorization $http_authorization;
```

## Dashboard

- The orchestrator serves a dashboard at `GET /dashboard`.
- The dashboard is read-only by default; when a token is configured, POST actions require it.
- The dashboard UI includes an “Auth token (Bearer)” field; paste the shared token to enable POST actions.
- If the public dashboard endpoint is backed by a follower, it proxies leader-only APIs to the current leader (so you don’t need DNS to track leader changes).
- Node build/version strings are surfaced in `/api/cluster/status` (default: `git branch@sha`, override via `RINGRIFT_BUILD_VERSION`).

## Service templates

- macOS (launchd): `ai-service/config/launchd/com.ringrift.p2p-orchestrator.plist`
- Linux (systemd): `ai-service/config/systemd/ringrift-p2p-orchestrator.service`
- Env example: `ai-service/config/p2p_orchestrator.env.example`

## Job Submission via REST

The P2P orchestrator accepts job requests over HTTP (see [P2P_ADMIN_API.md](P2P_ADMIN_API.md)).

```bash
# Submit a selfplay job to the current leader
export RINGRIFT_CLUSTER_AUTH_TOKEN="$(cat /etc/ringrift/p2p_orchestrator.token)"

curl -X POST http://leader-host:8770/start_job \
  -H "Authorization: Bearer ${RINGRIFT_CLUSTER_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "selfplay-sq8-2p-001",
    "job_type": "selfplay",
    "config": {
      "board_type": "square8",
      "num_players": 2
    }
  }'
```

Supported `job_type` values include `selfplay`, `gpu_selfplay`, `hybrid_selfplay`, `nnue`, and `cmaes`.
For a full endpoint list, see [P2P_ADMIN_API.md](P2P_ADMIN_API.md).

## CLI Arguments Reference

### Core Arguments

| Argument           | Description                      | Default      |
| ------------------ | -------------------------------- | ------------ |
| `--node-id`        | Unique node identifier           | hostname     |
| `--host`           | Host address to bind to          | 0.0.0.0      |
| `--port`           | HTTP API port                    | 8770         |
| `--advertise-host` | Host address advertised to peers | auto-detect  |
| `--advertise-port` | Port advertised to peers         | same as port |
| `--peers`          | Comma-separated peer URLs        | -            |
| `--ringrift-path`  | Path to RingRift installation    | auto-detect  |

### Authentication Arguments

| Argument         | Description              | Default |
| ---------------- | ------------------------ | ------- |
| `--require-auth` | Require token validation | False   |
| `--auth-token`   | Bearer token value       | -       |

### Network Resilience Arguments

| Argument        | Description                       | Default |
| --------------- | --------------------------------- | ------- |
| `--relay-peers` | Relay peers for NAT-blocked nodes | -       |

### Storage Arguments

| Argument                  | Description                          | Default |
| ------------------------- | ------------------------------------ | ------- |
| `--storage-type`          | Storage backend (disk/ramdrive/auto) | auto    |
| `--sync-to-disk-interval` | Ramdrive sync interval (seconds)     | 300     |

### Environment Variables

| Variable                           | Description           |
| ---------------------------------- | --------------------- |
| `RINGRIFT_CLUSTER_AUTH_TOKEN`      | Bearer token value    |
| `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE` | Path to token file    |
| `RINGRIFT_BUILD_VERSION`           | Build version string  |
| `NAT_SYMMETRIC_DETECTION_ENABLED`  | Enable NAT detection  |
| `DYNAMIC_VOTER_ENABLED`            | Enable dynamic quorum |

## Network Resilience Features

### NAT Traversal with Relay Support

For nodes behind NAT that cannot receive direct connections:

```bash
python scripts/p2p_orchestrator.py \
  --node-id nat-node \
  --relay-peers http://relay1:8770,http://relay2:8770
```

Relay peers forward heartbeats and gossip to NAT-blocked nodes.

### Ramdrive Auto-Detection

The orchestrator automatically detects RAM disk availability:

```bash
# Auto-detect (default)
--storage-type auto

# Force disk storage
--storage-type disk

# Force ramdrive
--storage-type ramdrive
```

When using ramdrive, data is synced to disk periodically:

```bash
--sync-to-disk-interval 300  # Sync every 5 minutes
```

### Gossip Protocol

The P2P orchestrator uses a gossip protocol for:

- Peer discovery
- Data replication
- Health propagation

Gossip runs over the main HTTP API port.

## Related Documentation

- [VAST_P2P_ORCHESTRATION.md](VAST_P2P_ORCHESTRATION.md) - Vast.ai P2P setup
- [DISTRIBUTED_SELFPLAY.md](../training/DISTRIBUTED_SELFPLAY.md) - Cluster selfplay
- [TRAINING_PIPELINE.md](../training/TRAINING_PIPELINE.md) - Training workflow
