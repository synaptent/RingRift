# Coordination Providers Package

Cloud provider integrations for the RingRift training cluster.

## Overview

This package provides abstractions for different cloud GPU providers:

- Instance lifecycle management (start, stop, terminate)
- SSH connection handling
- Provider-specific path conventions
- Cost tracking and optimization

## Modules

### `base.py` - CloudProvider

Abstract base class for all providers:

```python
from app.coordination.providers import CloudProvider

class MyProvider(CloudProvider):
    async def get_instances(self) -> list[Instance]:
        ...

    async def start_instance(self, instance_id: str) -> bool:
        ...
```

### Provider Implementations

| Provider        | File                  | GPU Types                 | Notes                   |
| --------------- | --------------------- | ------------------------- | ----------------------- |
| **Lambda Labs** | `lambda_provider.py`  | GH200, H100, A10          | ⚠️ TERMINATED Dec 2025  |
| **Vast.ai**     | `vast_provider.py`    | RTX 5090, 4090, 3090, A40 | Ephemeral, cheap        |
| **Vultr**       | `vultr_provider.py`   | A100 (vGPU)               | Persistent              |
| **Hetzner**     | `hetzner_provider.py` | CPU only                  | P2P voters, data sync   |
| **RunPod**      | (via cluster_config)  | H100, A100, L40S          | Persistent `/workspace` |
| **Nebius**      | (via cluster_config)  | H100 80GB, L40S           | Training backbone       |

## Usage

### Get Provider by Name

```python
from app.coordination.providers import get_provider

provider = get_provider("lambda")
instances = await provider.get_instances()
```

### SSH Connection

```python
from app.coordination.providers import get_provider

provider = get_provider("vast")
ssh_config = provider.get_ssh_config("vast-29129529")
# Returns: {"host": "ssh6.vast.ai", "port": 19528, "user": "root", ...}
```

### Path Conventions

Each provider has different ringrift path conventions:

```python
provider = get_provider("runpod")
path = provider.get_ringrift_path()  # "/workspace/ringrift/ai-service"

provider = get_provider("lambda")
path = provider.get_ringrift_path()  # "~/ringrift/ai-service" (NFS mount)
```

## Provider-Specific Notes

### Lambda Labs (TERMINATED Dec 2025)

> **Note**: Lambda Labs account terminated December 2025. Provider code kept for reference only.

- Shared NFS storage at `/home/ubuntu/ringrift`
- Skip sync between Lambda nodes (same filesystem)
- All nodes permanently removed from cluster

### Vast.ai

- Ephemeral instances - aggressive sync required
- 15-30 second termination notice
- `EphemeralSyncDaemon` handles data protection
- Path varies: `~/ringrift` or `/workspace/ringrift`

### RunPod

- Persistent storage at `/workspace`
- Various GPU types (H100, A100, L40S, RTX 3090 Ti)
- Custom SSH key: `~/.runpod/ssh/RunPod-Key-Go`

### Vultr

- vGPU instances (A100 20GB slice)
- Persistent at `/root/ringrift`
- Standard SSH key

### Hetzner

- CPU-only nodes for data sync and coordination
- 8-16 CPUs, 16-32GB RAM
- Used for NPZ export, P2P gossip

## Configuration

From `config/distributed_hosts.yaml`:

```yaml
hosts:
  lambda-gh200-a:
    provider: lambda
    gpu_type: GH200
    ssh_host: 100.123.183.70
    ssh_key: ~/.ssh/id_cluster

  vast-29129529:
    provider: vast
    gpu_type: RTX 4090
    is_ephemeral: true
    ssh_host: ssh6.vast.ai
    ssh_port: 19528
```

## See Also

- `../daemon_manager.py` - Starts provider-specific daemons
- `../ephemeral_sync.py` - Aggressive sync for Vast.ai
- `../../distributed/cluster_manifest.py` - Node discovery
