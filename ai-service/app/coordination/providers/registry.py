"""Provider registry for cloud provider abstraction.

Centralizes provider-specific logic so adding new providers requires
only a single file change.

Usage:
    from app.coordination.providers.registry import ProviderRegistry

    provider = ProviderRegistry.get_for_node("vast-12345")
    idle_threshold = provider.idle_threshold_seconds
    shutdown_method = provider.get_shutdown_command
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class CloudProviderProtocol(Protocol):
    """Protocol for cloud provider implementations.

    This is the expected interface for provider classes.
    Implementations should provide these attributes and methods.
    """

    name: str
    idle_threshold_seconds: int
    min_nodes_to_keep: int

    def get_shutdown_command(self, node_id: str) -> str:
        """Get command to shut down a node."""
        ...

    def is_ephemeral(self) -> bool:
        """Whether nodes are ephemeral (can be terminated anytime)."""
        ...


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a cloud provider.

    Immutable configuration that describes provider-specific behavior
    for idle detection, shutdown, and node management.

    Attributes:
        name: Provider identifier (e.g., "vast", "lambda", "runpod")
        idle_threshold_seconds: Seconds of idle time before eligible for shutdown
        min_nodes_to_keep: Minimum nodes to retain even when idle
        is_ephemeral: Whether provider instances can terminate unexpectedly
        node_pattern: Regex pattern to match node names to this provider
        ringrift_path: Default path to ringrift on this provider's nodes
        ssh_key: Default SSH key path for this provider
        ssh_user: Default SSH username for this provider
    """
    name: str
    idle_threshold_seconds: int = 1800  # 30 min default
    min_nodes_to_keep: int = 0
    is_ephemeral: bool = False
    node_pattern: str = ""  # Regex pattern to match node names
    ringrift_path: str = "~/ringrift/ai-service"
    ssh_key: str = "~/.ssh/id_ed25519"
    ssh_user: str = "root"
    # Additional provider-specific settings
    extra: Dict[str, str] = field(default_factory=dict)

    def matches(self, node_name: str) -> bool:
        """Check if this provider matches a node name."""
        if not self.node_pattern:
            return node_name.startswith(self.name)
        return bool(re.match(self.node_pattern, node_name, re.IGNORECASE))

    def get_shutdown_command(self, node_id: str) -> str:
        """Get provider-specific shutdown command.

        Override in subclass or use extra["shutdown_command_template"] for custom commands.
        """
        template = self.extra.get("shutdown_command_template", "")
        if template:
            return template.format(node_id=node_id)
        # Default: graceful shutdown
        return f"sudo shutdown -h now  # {node_id}"


# Default provider configurations
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "vast": ProviderConfig(
        name="vast",
        idle_threshold_seconds=900,  # 15 min (hourly billing)
        min_nodes_to_keep=0,
        is_ephemeral=True,
        node_pattern=r"vast-\d+",
        ringrift_path="~/ringrift/ai-service",
        ssh_user="root",
        extra={
            "billing": "hourly",
            "shutdown_command_template": "vastai destroy instance {node_id}",
        },
    ),
    "runpod": ProviderConfig(
        name="runpod",
        idle_threshold_seconds=1200,  # 20 min
        min_nodes_to_keep=1,
        is_ephemeral=True,
        node_pattern=r"runpod-.*",
        ringrift_path="/workspace/ringrift/ai-service",
        ssh_key="~/.runpod/ssh/RunPod-Key-Go",
        ssh_user="root",
        extra={
            "billing": "per-second",
        },
    ),
    "nebius": ProviderConfig(
        name="nebius",
        idle_threshold_seconds=3600,  # 60 min (reserved instances)
        min_nodes_to_keep=1,
        is_ephemeral=False,
        node_pattern=r"nebius-.*",
        ringrift_path="~/ringrift/ai-service",
        ssh_user="ubuntu",
        ssh_key="~/.ssh/id_cluster",
    ),
    "lambda": ProviderConfig(
        name="lambda",
        idle_threshold_seconds=1800,  # 30 min
        min_nodes_to_keep=0,
        is_ephemeral=False,
        node_pattern=r"lambda-.*",
        ringrift_path="~/ringrift/ai-service",
        ssh_user="ubuntu",
        ssh_key="~/.ssh/id_cluster",
        extra={
            "has_nfs": "true",  # Lambda uses shared NFS
        },
    ),
    "vultr": ProviderConfig(
        name="vultr",
        idle_threshold_seconds=1800,  # 30 min
        min_nodes_to_keep=0,
        is_ephemeral=False,
        node_pattern=r"vultr-.*",
        ringrift_path="/root/ringrift/ai-service",
        ssh_user="root",
        ssh_key="~/.ssh/id_ed25519",
    ),
    "hetzner": ProviderConfig(
        name="hetzner",
        idle_threshold_seconds=7200,  # 2 hr (CPU nodes, less urgent)
        min_nodes_to_keep=3,  # Keep for P2P voting quorum
        is_ephemeral=False,
        node_pattern=r"hetzner-.*",
        ringrift_path="/root/ringrift/ai-service",
        ssh_user="root",
        ssh_key="~/.ssh/id_ed25519",
        extra={
            "gpu": "none",
            "role": "p2p_voter",
        },
    ),
    "local": ProviderConfig(
        name="local",
        idle_threshold_seconds=86400,  # 24 hr (never auto-shutdown)
        min_nodes_to_keep=1,
        is_ephemeral=False,
        node_pattern=r"(local-.*|mac-.*|macbook-.*)",
        ringrift_path="~/Development/RingRift/ai-service",
        ssh_user="armand",
        extra={
            "is_coordinator": "true",
        },
    ),
}


class ProviderRegistry:
    """Registry for cloud provider configurations.

    Thread-safe registry that maps node names to provider configurations.
    Supports both static configuration and dynamic registration.

    Example:
        >>> config = ProviderRegistry.get_for_node("vast-12345")
        >>> config.idle_threshold_seconds
        900
        >>> config.is_ephemeral
        True

        >>> # Register custom provider
        >>> custom = ProviderConfig(name="mycloud", idle_threshold_seconds=600)
        >>> ProviderRegistry.register(custom)
    """

    _configs: Dict[str, ProviderConfig] = dict(PROVIDER_CONFIGS)
    _node_cache: Dict[str, ProviderConfig] = {}  # Cache node->provider lookups

    @classmethod
    def register(cls, config: ProviderConfig) -> None:
        """Register a new provider configuration.

        Args:
            config: Provider configuration to register
        """
        cls._configs[config.name] = config
        # Invalidate cache when config changes
        cls._node_cache.clear()

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister a provider configuration.

        Args:
            provider_name: Name of provider to unregister

        Returns:
            True if provider was unregistered, False if not found
        """
        if provider_name in cls._configs:
            del cls._configs[provider_name]
            cls._node_cache.clear()
            return True
        return False

    @classmethod
    def get_for_node(cls, node_name: str) -> ProviderConfig:
        """Get provider config for a node name.

        Uses pattern matching to determine which provider owns a node.
        Results are cached for performance.

        Args:
            node_name: Name of the node (e.g., "vast-12345", "runpod-h100")

        Returns:
            ProviderConfig for the matching provider, or generic config if no match

        Example:
            >>> ProviderRegistry.get_for_node("vast-29129529")
            ProviderConfig(name='vast', idle_threshold_seconds=900, ...)
        """
        # Check cache first
        if node_name in cls._node_cache:
            return cls._node_cache[node_name]

        # Find matching provider
        for config in cls._configs.values():
            if config.matches(node_name):
                cls._node_cache[node_name] = config
                return config

        # Return generic config for unknown nodes
        generic = ProviderConfig(name="generic")
        cls._node_cache[node_name] = generic
        return generic

    @classmethod
    def get(cls, provider_name: str) -> Optional[ProviderConfig]:
        """Get provider config by name.

        Args:
            provider_name: Provider name (e.g., "vast", "runpod")

        Returns:
            ProviderConfig if found, None otherwise
        """
        return cls._configs.get(provider_name)

    @classmethod
    def all_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._configs.keys())

    @classmethod
    def all_configs(cls) -> Dict[str, ProviderConfig]:
        """Get all provider configurations.

        Returns:
            Dict mapping provider name to config
        """
        return dict(cls._configs)

    @classmethod
    def get_ephemeral_providers(cls) -> list[str]:
        """Get names of all ephemeral providers.

        Ephemeral providers have instances that can terminate unexpectedly
        and require aggressive data sync.

        Returns:
            List of ephemeral provider names
        """
        return [
            name for name, config in cls._configs.items()
            if config.is_ephemeral
        ]

    @classmethod
    def reset(cls) -> None:
        """Reset registry to default configurations.

        Useful for testing.
        """
        cls._configs = dict(PROVIDER_CONFIGS)
        cls._node_cache.clear()

    @classmethod
    def get_by_idle_threshold(cls, max_threshold: int) -> list[ProviderConfig]:
        """Get providers with idle threshold at or below max.

        Args:
            max_threshold: Maximum idle threshold in seconds

        Returns:
            List of matching ProviderConfigs sorted by threshold ascending
        """
        matching = [
            config for config in cls._configs.values()
            if config.idle_threshold_seconds <= max_threshold
        ]
        return sorted(matching, key=lambda c: c.idle_threshold_seconds)
