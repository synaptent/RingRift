# Cloud Provider Managers

This module contains cloud provider-specific management code for the distributed training cluster.

## Modules

| Module                 | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `base.py`              | Base class `CloudProviderManager` defining common interface |
| `aws_manager.py`       | AWS EC2 instance management                                 |
| `lambda_manager.py`    | Lambda Labs GPU instance management                         |
| `hetzner_manager.py`   | Hetzner Cloud server management                             |
| `tailscale_manager.py` | Tailscale VPN mesh management                               |

## Usage

```python
from app.providers.lambda_manager import LambdaManager

manager = LambdaManager()
instances = await manager.list_instances()
```

## Common Interface

All managers implement:

- `list_instances()` - List active instances
- `get_instance_status(instance_id)` - Get instance health
- `terminate_instance(instance_id)` - Terminate instance
- `get_ssh_config(instance_id)` - Get SSH connection config

## Integration

Used by:

- `daemon_manager.py` for multi-provider orchestration
- `node_recovery.py` for auto-recovery
- ~~`lambda_idle.py` for cost optimization~~ (DEPRECATED: Lambda account terminated Dec 2025)
