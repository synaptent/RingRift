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
from app.coordination.providers import CloudProvider, GPUType, Instance

class MyProvider(CloudProvider):
    @property
    def provider_type(self):
        ...

    @property
    def name(self) -> str:
        ...

    async def list_instances(self) -> list[Instance]:
        ...

    async def scale_up(self, gpu_type: GPUType, count: int = 1) -> list[Instance]:
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
from app.coordination.providers import ProviderType, get_provider

provider = get_provider(ProviderType.VAST)
instances = await provider.list_instances()
```

### Provider Metadata For A Node

```python
from app.coordination.providers import ProviderRegistry

config = ProviderRegistry.get_for_node("vast-29129529")
print(config.ringrift_path)
print(config.ssh_user)
print(config.ssh_key)
```

### Capacity Snapshot

```python
from app.coordination.providers import get_all_providers

for provider in get_all_providers():
    if provider.is_configured():
        gpus = await provider.get_available_gpus()
        print(provider.name, gpus)
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
  my-training-node:
    provider: lambda
    gpu_type: GH200
    ssh_host: 100.x.x.x # Replace with your Tailscale IP
    ssh_key: ~/.ssh/id_cluster

  vast-example:
    provider: vast
    gpu_type: RTX 4090
    is_ephemeral: true
    ssh_host: ssh6.vast.ai
    ssh_port: 12345 # Replace with your port
```

## See Also

- `../daemon_manager.py` - Starts provider-specific daemons
- `../auto_sync_daemon.py` - Aggressive sync for Vast.ai (strategy=ephemeral)
- `../../distributed/cluster_manifest.py` - Node discovery
