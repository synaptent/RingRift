"""Registry modules for ClusterManifest decomposition.

Provides specialized registries for different data types:
- GameLocationRegistry: Game ID to node location mappings
- ModelRegistry: Model path to node location mappings
- NPZRegistry: NPZ file to node location mappings
- CheckpointRegistry: Training checkpoint tracking
- NodeInventoryManager: Node capacity and inventory tracking
- ReplicationManager: Sync target selection and replication

December 2025 - ClusterManifest god object decomposition.
"""

from app.distributed.registries.base import BaseRegistry
from app.distributed.registries.game_registry import GameLocationRegistry
from app.distributed.registries.model_registry import ModelRegistry
from app.distributed.registries.npz_registry import NPZRegistry
from app.distributed.registries.checkpoint_registry import CheckpointRegistry
from app.distributed.registries.node_inventory import NodeInventoryManager
from app.distributed.registries.replication import ReplicationManager

__all__ = [
    "BaseRegistry",
    "CheckpointRegistry",
    "GameLocationRegistry",
    "ModelRegistry",
    "NodeInventoryManager",
    "NPZRegistry",
    "ReplicationManager",
]
