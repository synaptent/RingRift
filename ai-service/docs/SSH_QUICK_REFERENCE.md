# SSH Quick Reference - app/core/ssh.py

**Updated**: December 26, 2025 (Phase 1 Complete)

## Basic Usage

```python
from app.core.ssh import get_ssh_client

# Get client (auto-loads from distributed_hosts.yaml)
client = get_ssh_client("runpod-h100")

# Run command
result = await client.run_async("nvidia-smi")
print(result.stdout)
```

## Transport Fallback Chain

Automatic: **Tailscale → Direct → Cloudflare**

```python
result = await client.run_async("hostname")
print(f"Connected via: {result.transport_used}")  # "tailscale", "direct", or "cloudflare"
```

## Health Tracking

```python
# Check connection health
if client.health.is_healthy:
    print(f"Success rate: {client.health.success_rate:.1%}")
else:
    print(f"Unhealthy: {client.health.consecutive_failures} failures")
```

## Background Jobs

```python
# Start background process
result = await client.run_background(
    "python train.py --epochs 100",
    log_file="~/logs/training.log"
)
pid = int(result.stdout.strip())

# Monitor memory
memory_mb, cmd = await client.get_process_memory(pid)
print(f"{cmd} using {memory_mb}MB")
```

## Configuration

### From cluster_hosts.yaml

```python
client = get_ssh_client("node-id")  # Auto-loads config
```

### Manual Configuration

```python
from app.core.ssh import SSHClient, SSHConfig

config = SSHConfig(
    host="203.0.113.10",
    port=22,
    user="ubuntu",
    key_path="~/.ssh/id_ed25519",
    tailscale_ip="100.100.100.20",           # Optional: Tailscale fallback
    cloudflare_tunnel="ssh.example.com",     # Optional: Cloudflare fallback
    cloudflare_service_token_id="token",     # Optional: CF auth
    cloudflare_service_token_secret="secret" # Optional: CF auth
)

client = SSHClient(config)
```

### From HostConfig Object

```python
from app.sync.cluster_hosts import get_cluster_nodes

nodes = get_cluster_nodes()
host_config = nodes["runpod-h100"]

config = SSHConfig.from_host_config(host_config)
client = SSHClient(config)
```

## Advanced Features

### Retry on Failure

```python
result = await client.run_async_with_retry(
    "hostname",
    max_retries=5,
    retry_delay=3.0,
)
```

### Custom Timeout

```python
result = await client.run_async("long_command", timeout=300)  # 5 minutes
```

### Working Directory

```python
result = await client.run_async(
    "git status",
    cwd="/opt/ringrift/ai-service"
)
```

### Environment Variables

```python
result = await client.run_async(
    "python train.py",
    env={"CUDA_VISIBLE_DEVICES": "0,1"}
)
```

### File Transfers

```python
# Upload
result = client.scp_to("local.txt", "/remote/path/file.txt")

# Download
result = client.scp_from("/remote/path/file.txt", "local.txt")
```

### Connectivity Check

```python
is_alive, message = await client.check_connectivity()
if is_alive:
    print(f"Connected: {message}")
```

## Common Patterns

### Job Submission & Monitoring

```python
# Submit job
result = await client.run_background(
    "python selfplay.py --games 1000",
    log_file="~/logs/selfplay.log"
)
pid = int(result.stdout.strip())

# Monitor progress
while True:
    mem_info = await client.get_process_memory(pid)
    if mem_info is None:
        print("Job finished")
        break
    memory_mb, _ = mem_info
    print(f"Job using {memory_mb}MB RAM")
    await asyncio.sleep(60)
```

### Health-Aware Scheduling

```python
clients = [get_ssh_client(node) for node in nodes]

# Test connectivity
for client in clients:
    await client.run_async("echo test")

# Filter healthy nodes
healthy = [c for c in clients if c.health.is_healthy]

# Schedule on best node
best = max(healthy, key=lambda c: c.health.success_rate)
result = await best.run_async("start_training.sh")
```

### Cloudflare Zero Trust

```yaml
# distributed_hosts.yaml
internal-server:
  ssh_host: internal.corp.example.com
  ssh_user: ubuntu
  cloudflare_tunnel: ssh.example.com
  cloudflare_service_token_id: ${CF_TOKEN_ID}
  cloudflare_service_token_secret: ${CF_TOKEN_SECRET}
```

```python
client = get_ssh_client("internal-server")
# Automatically uses Cloudflare as fallback
result = await client.run_async("hostname")
```

## Data Classes

### SSHConfig

16 fields including:

- `host`, `port`, `user`, `key_path`
- `tailscale_ip` (optional Tailscale fallback)
- `cloudflare_tunnel`, `cloudflare_service_token_id`, `cloudflare_service_token_secret`
- `connect_timeout`, `command_timeout`
- `use_control_master`, `control_persist`
- `work_dir`, `venv_activate`

### SSHResult

```python
result.success: bool           # True if returncode == 0
result.returncode: int         # Exit code
result.stdout: str            # Standard output
result.stderr: str            # Standard error
result.elapsed_ms: float      # Execution time in milliseconds
result.transport_used: str    # "tailscale", "direct", or "cloudflare"
result.timed_out: bool        # True if command timed out
result.error: str | None      # Error message if failed
```

### SSHHealth

```python
health.last_success: float | None          # Unix timestamp
health.last_failure: float | None          # Unix timestamp
health.consecutive_failures: int           # Current failure streak
health.consecutive_successes: int          # Current success streak
health.total_successes: int                # Total successful commands
health.total_failures: int                 # Total failed commands
health.is_healthy: bool                    # False if ≥3 consecutive failures
health.success_rate: float                 # 0.0 to 1.0
```

## Troubleshooting

### Connection Fails

```python
# Check health
print(f"Consecutive failures: {client.health.consecutive_failures}")
print(f"Last failure: {client.health.last_failure}")

# Try connectivity test
is_alive, msg = await client.check_connectivity()
print(msg)
```

### Timeouts

```python
# Increase timeout
config.connect_timeout = 30
config.command_timeout = 300

# Or per-command
result = await client.run_async("slow_command", timeout=600)
```

### Cloudflare Not Working

```bash
# Verify cloudflared installed
cloudflared --version

# Test tunnel manually
cloudflared access ssh --hostname ssh.example.com
```

## See Also

- **Full Documentation**: `docs/SSH_CONSOLIDATION_PHASE1.md`
- **Transport Chain**: `docs/SSH_TRANSPORT_CHAIN.md`
- **Implementation**: `app/core/ssh.py`
- **Examples**: `PHASE1_SUMMARY.md`
