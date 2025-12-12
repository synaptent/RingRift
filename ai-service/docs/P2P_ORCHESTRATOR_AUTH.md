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

## Service templates

- macOS (launchd): `ai-service/config/launchd/com.ringrift.p2p-orchestrator.plist`
- Linux (systemd): `ai-service/config/systemd/ringrift-p2p-orchestrator.service`
- Env example: `ai-service/config/p2p_orchestrator.env.example`
