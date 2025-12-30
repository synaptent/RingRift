# SSH Transport Fallback Chain

## Overview

`app/core/ssh.py` now supports a 3-layer transport fallback chain with automatic health tracking.

## Transport Priority

```
1. Tailscale (if configured)
   ↓ (on failure)
2. Direct SSH
   ↓ (on failure)
3. Cloudflare Zero Trust (if configured)
```

## Configuration Examples

### Tailscale + Direct Fallback

```yaml
# distributed_hosts.yaml
runpod-h100:
  ssh_host: 102.210.171.65
  ssh_port: 30178
  tailscale_ip: 100.100.100.10
  ssh_user: root
  ssh_key: ~/.ssh/id_ed25519
```

**Behavior**: Tries Tailscale first (100.100.100.10), falls back to direct (102.210.171.65:30178)

### Cloudflare Zero Trust

```yaml
# distributed_hosts.yaml
internal-server:
  ssh_host: internal.corp.example.com
  ssh_user: ubuntu
  ssh_key: ~/.ssh/id_ed25519
  cloudflare_tunnel: ssh.example.com
  cloudflare_service_token_id: cf_token_abc123
  cloudflare_service_token_secret: cf_secret_xyz789
```

**Behavior**: Tries direct first, falls back to Cloudflare tunnel via ProxyCommand

### Full Stack (All Transports)

```yaml
# distributed_hosts.yaml
hybrid-node:
  ssh_host: 203.0.113.10
  ssh_port: 22
  tailscale_ip: 100.100.100.20
  ssh_user: ubuntu
  ssh_key: ~/.ssh/id_ed25519
  cloudflare_tunnel: ssh.example.com
  cloudflare_service_token_id: cf_token
  cloudflare_service_token_secret: cf_secret
```

**Behavior**: Tries Tailscale → Direct → Cloudflare (3 attempts)

## Cloudflare ProxyCommand

When Cloudflare is configured, the SSH command includes:

```bash
ssh -o "ProxyCommand=cloudflared access ssh --hostname ssh.example.com \
    --service-token-id cf_token --service-token-secret cf_secret" \
    ubuntu@internal.corp.example.com
```

Requires `cloudflared` CLI installed on the client machine.

## Health Tracking

Each transport attempt updates connection health:

```python
client = get_ssh_client("runpod-h100")
result = await client.run_async("hostname")

print(f"Transport used: {result.transport_used}")  # "tailscale", "direct", or "cloudflare"
print(f"Health: {client.health.is_healthy}")       # False if ≥3 consecutive failures
print(f"Success rate: {client.health.success_rate:.1%}")
```

### Health Properties

- `is_healthy` - True if < 3 consecutive failures
- `success_rate` - Ratio of successes to total attempts
- `consecutive_failures` - Current failure streak
- `last_success` - Unix timestamp of last successful command

## Usage in Code

### Automatic Fallback

```python
from app.core.ssh import get_ssh_client

# Config loaded from distributed_hosts.yaml
client = get_ssh_client("runpod-h100")

# Automatically tries all configured transports
result = await client.run_async("nvidia-smi")

if result.success:
    print(f"Connected via {result.transport_used}")
else:
    print(f"All transports failed: {result.stderr}")
```

### Manual Configuration

```python
from app.core.ssh import SSHClient, SSHConfig

config = SSHConfig(
    host="203.0.113.10",
    tailscale_ip="100.100.100.20",
    cloudflare_tunnel="ssh.example.com",
    cloudflare_service_token_id="token",
    cloudflare_service_token_secret="secret",
)

client = SSHClient(config)
result = await client.run_async("uptime")
```

## Performance Characteristics

| Transport  | Latency      | Reliability | Security        | Notes                    |
| ---------- | ------------ | ----------- | --------------- | ------------------------ |
| Tailscale  | Low (2-10ms) | High        | End-to-end enc. | Best for cluster nodes   |
| Direct SSH | Varies       | Medium      | SSH only        | Standard fallback        |
| Cloudflare | Medium       | High        | Zero Trust      | Good for restricted nets |

## Timeout Behavior

Default timeouts:

- **Connect timeout**: 10 seconds per transport
- **Command timeout**: 60 seconds (configurable)

With 3 transports, total timeout can be up to 30 seconds for connection establishment.

```python
config = SSHConfig(
    host="example.com",
    connect_timeout=5,   # Faster connection attempts
    command_timeout=30,  # Lower command timeout
)
```

## Troubleshooting

### Tailscale Not Working

Check if Tailscale is running and IP is correct:

```bash
tailscale status | grep <node-name>
```

### Cloudflare Tunnel Fails

Verify `cloudflared` is installed:

```bash
cloudflared --version
```

Test tunnel manually:

```bash
cloudflared access ssh --hostname ssh.example.com \
  --service-token-id token --service-token-secret secret
```

### All Transports Fail

Check health tracking:

```python
if not client.health.is_healthy:
    print(f"Consecutive failures: {client.health.consecutive_failures}")
    print(f"Last failure: {client.health.last_failure}")
```

Consider increasing retry count:

```python
result = await client.run_async_with_retry(
    "hostname",
    max_retries=5,
    retry_delay=3.0,
)
```

## See Also

- `SSH_CONSOLIDATION_PHASE1.md` - Phase 1 features
- `ai-service/app/core/ssh.py` - Implementation
- `config/distributed_hosts.template.yaml` - Configuration reference
