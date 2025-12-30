# SSH Consolidation - Phase 1 Complete

## Overview

Extended `ai-service/app/core/ssh.py` with missing features to prepare for consolidation of duplicated SSH utilities.

## Features Added

### 1. Cloudflare Zero Trust Support

**New fields in `SSHConfig`:**

- `cloudflare_tunnel: str | None` - Cloudflare Access tunnel hostname
- `cloudflare_service_token_id: str | None` - Service token ID for authentication
- `cloudflare_service_token_secret: str | None` - Service token secret

**New properties:**

- `SSHConfig.use_cloudflare` - Returns True if Cloudflare tunnel is configured

**New methods:**

- `SSHClient._build_cloudflare_proxy_command()` - Builds `cloudflared access ssh` ProxyCommand

**Transport fallback chain updated:**

```
Tailscale (if configured) → Direct → Cloudflare (if configured)
```

### 2. SSHHealth Tracking

**New dataclass `SSHHealth`:**

```python
@dataclass
class SSHHealth:
    last_success: float | None
    last_failure: float | None
    consecutive_failures: int
    consecutive_successes: int
    total_successes: int
    total_failures: int

    @property
    def is_healthy(self) -> bool:
        """Healthy if < 3 consecutive failures."""
        return self.consecutive_failures < 3

    @property
    def success_rate(self) -> float:
        """Overall success rate (0.0 to 1.0)."""
```

**Integrated into `SSHClient`:**

- `SSHClient.health` property to access health status
- `SSHClient._record_success()` - Records successful execution
- `SSHClient._record_failure()` - Records failed execution
- Automatically tracked on every `run()` and `run_async()` call

### 3. Background Execution

**New method `SSHClient.run_background()`:**

```python
async def run_background(
    self,
    command: str,
    log_file: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SSHResult:
    """Execute command in background using nohup.

    Returns SSHResult with PID in stdout if successful.
    """
```

Uses `nohup` to run commands that survive SSH disconnection. Returns the PID for tracking.

### 4. Process Memory Monitoring

**New method `SSHClient.get_process_memory()`:**

```python
async def get_process_memory(self, pid: int) -> tuple[int, str] | None:
    """Get memory usage of a process in MB.

    Args:
        pid: Process ID to check

    Returns:
        Tuple of (memory_mb, command_name) or None if process not found
    """
```

Uses `ps` to retrieve RSS (resident set size) for monitoring remote processes.

### 5. SSHConfig from HostConfig

**New classmethod `SSHConfig.from_host_config()`:**

```python
@classmethod
def from_host_config(cls, host_config: Any) -> SSHConfig:
    """Create SSHConfig from a HostConfig object.

    Extracts all relevant SSH fields including Cloudflare settings.
    """
```

Enables easy conversion from `distributed_hosts.yaml` HostConfig objects.

## Usage Examples

### Cloudflare Zero Trust

```python
from app.core.ssh import SSHClient, SSHConfig

config = SSHConfig(
    host="internal-server.example.com",
    cloudflare_tunnel="ssh.example.com",
    cloudflare_service_token_id="cf_token_id",
    cloudflare_service_token_secret="cf_secret",
)

client = SSHClient(config)
result = await client.run_async("hostname")
# Automatically tries: Direct → Cloudflare
```

### Health Tracking

```python
client = get_ssh_client("runpod-h100")

# Execute commands
result = await client.run_async("nvidia-smi")

# Check health
if not client.health.is_healthy:
    print(f"Warning: {client.health.consecutive_failures} consecutive failures")
    print(f"Success rate: {client.health.success_rate:.1%}")
```

### Background Execution

```python
# Start a long-running process
result = await client.run_background(
    "python train.py --epochs 100",
    log_file="~/logs/training.log"
)

if result.success:
    pid = int(result.stdout.strip())
    print(f"Training started with PID {pid}")

    # Monitor memory usage
    memory_info = await client.get_process_memory(pid)
    if memory_info:
        memory_mb, cmd_name = memory_info
        print(f"Process using {memory_mb}MB")
```

### From HostConfig

```python
from app.sync.cluster_hosts import get_cluster_nodes

nodes = get_cluster_nodes()
host_config = nodes["runpod-h100"]

config = SSHConfig.from_host_config(host_config)
client = SSHClient(config)
```

## Testing

All features tested in `ai-service/test_ssh_phase1.py`:

```bash
$ python test_ssh_phase1.py

Testing Phase 1 SSH consolidation features...

✓ SSHHealth tests passed
✓ Cloudflare config tests passed
✓ from_host_config() tests passed
✓ Cloudflare proxy command tests passed
✓ Health tracking tests passed
✓ run_background() method exists and executes
✓ get_process_memory() works (PID 28096: 22MB, python)

✅ All Phase 1 SSH consolidation tests passed!
```

## Backward Compatibility

All existing functionality preserved:

- ✅ Connection pooling via ControlMaster
- ✅ Tailscale → Direct fallback
- ✅ Automatic retry with `run_async_with_retry()`
- ✅ SCP file transfers
- ✅ Sync and async execution
- ✅ Convenience function exports

New features are additive only.

## Next Steps

**Phase 2**: Migrate consumers to use new features

- `app/distributed/hosts.py` → use `SSHHealth` tracking
- `app/execution/executor.py` → use `run_background()` for job submission
- Update cluster monitoring to check `client.health.is_healthy`

**Phase 3**: Remove deprecated modules

- Archive old SSH utilities once all consumers migrated
- Update documentation to reference unified `app/core/ssh.py`

## Files Modified

- `ai-service/app/core/ssh.py` - Extended with Phase 1 features
- `ai-service/test_ssh_phase1.py` - Test suite
- `SSH_CONSOLIDATION_PHASE1.md` - This document
