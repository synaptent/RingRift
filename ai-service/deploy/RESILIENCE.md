# RingRift Cluster Resilience (P2P + Fallback)

## What Runs On Every Node

- `scripts/p2p_orchestrator.py` (HTTP P2P control plane, default port `8770`)
- `scripts/node_resilience.py` (keeps P2P up; runs fallback selfplay when disconnected; triggers disk cleanup)

## Linux (systemd + cron backup)

1. On the node (as root, or via `sudo`), run:
   - `ai-service/deploy/setup_node_resilience.sh <node-id> <coordinator-url>`
2. Start services:
   - `systemctl start ringrift-p2p`
   - `systemctl start ringrift-resilience`
3. Verify:
   - `curl http://localhost:8770/health`
   - `tail -f /var/log/ringrift/p2p.log`
   - `tail -f /var/log/ringrift/resilience.log`

Notes:

- `/etc/ringrift/node.conf` is the per-node config (includes `NODE_ID`, `COORDINATOR_URL`, `RINGRIFT_DIR`, `P2P_PORT`, `SSH_PORT`).
- If `SSH_PORT` isn’t provided, the setup script tries to infer it from `ai-service/config/distributed_hosts.yaml`.
- For hardened clusters, export `RINGRIFT_CLUSTER_AUTH_TOKEN` before running setup; it will be persisted into `node.conf` and used for authenticated requests.

## macOS (launchd)

1. On the Mac:
   - `ai-service/deploy/setup_node_resilience_macos.sh <node-id> <coordinator-url>`
2. Verify:
   - `curl http://localhost:8770/health`
   - `tail -f ~/Library/Logs/RingRift/p2p.log`
   - `tail -f ~/Library/Logs/RingRift/resilience.log`

## Data Sync + Cleanup

- Training sync (leader-only): `POST /sync/training`
- Cluster-wide sync (leader-only): `POST /sync/start`
- Cleanup (leader-triggered): `POST /cleanup` and `POST /cleanup/files`

`p2p_orchestrator.py` sync now pulls files over HTTP via `GET /sync/file` (served only from `ai-service/data/**` and requires auth when `RINGRIFT_CLUSTER_AUTH_TOKEN` is set).

## Training Loop Automation

- `scripts/p2p_orchestrator.py` exposes endpoints for training triggers (e.g. `POST /training/nnue/start`, `POST /training/cmaes/start`) and includes an improvement-loop scaffold.
- `scripts/pipeline_orchestrator.py` is the SSH-driven end-to-end automation script (selfplay → sync → training → eval → promote). Run it from a coordinator/training node once the cluster is stable.
