# Cloud Provider Managers

This module contains cloud provider-specific management code for the distributed training cluster.

## Modules

| Module                 | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `vast_manager.py`      | Vast.ai instance management                                 |
| `base.py`              | Base class `CloudProviderManager` defining common interface |
| `aws_manager.py`       | AWS EC2 instance management                                 |
| `lambda_manager.py`    | Lambda Labs GPU instance management                         |
| `hetzner_manager.py`   | Hetzner Cloud server management                             |
| `tailscale_manager.py` | Tailscale VPN mesh management                               |

## Usage

```python
from app.providers import LambdaManager, VastManager

vast_manager = VastManager()
instances = await vast_manager.list_instances()

lambda_manager = LambdaManager()
lambda_instances = await lambda_manager.list_instances()
```

## Common Interface

All managers implement:

- `list_instances()` - List active instances
- `get_instance(instance_id)` - Get current instance details
- `check_health(instance)` - Run provider-specific health checks
- `terminate_instance(instance_id)` - Terminate instance
- `run_ssh_command(instance, command)` - Run remote commands over SSH

## Integration

Used by:

- `daemon_manager.py` for multi-provider orchestration
- `node_recovery.py` for auto-recovery
- `unified_idle_shutdown_daemon.py` for cost optimization across all providers

## Status Updates (December 2025)

**Lambda Account Status**: `LambdaManager` remains available via the package root and is loaded lazily.
Use it when Lambda Labs infrastructure is configured for the current environment.

**Idle Daemon Consolidation**: `lambda_idle_daemon.py` and `vast_idle_daemon.py` have been
consolidated into `unified_idle_shutdown_daemon.py` which provides provider-agnostic idle detection
and shutdown functionality. See `app/coordination/unified_idle_shutdown_daemon.py`.
