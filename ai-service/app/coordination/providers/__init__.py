"""Cloud provider abstraction layer for multi-provider cluster management.

This module provides a unified interface for managing compute resources across
multiple cloud providers (Vast.ai, RunPod, Nebius, Vultr, Hetzner).

DEPRECATED (Dec 2025): Lambda Labs account terminated.
ProviderType.LAMBDA is kept for backward compatibility but will emit warnings.

Usage:
    from app.coordination.providers import get_provider, ProviderType

    # Get a specific provider
    vultr = get_provider(ProviderType.VULTR)
    instances = await vultr.list_instances()

    # Get all available providers
    for provider in get_all_providers():
        print(f"{provider.name}: {await provider.get_available_gpus()}")

    # Get provider config for a node (centralized registry)
    from app.coordination.providers import ProviderRegistry

    config = ProviderRegistry.get_for_node("vast-12345")
    print(f"Idle threshold: {config.idle_threshold_seconds}s")
    print(f"Ephemeral: {config.is_ephemeral}")
"""

import logging
import warnings

from app.coordination.providers.base import (
    CloudProvider,
    Instance,
    InstanceStatus,
    ProviderType,
    GPUType,
)
from app.coordination.providers.registry import (
    ProviderRegistry,
    ProviderConfig,
    CloudProviderProtocol,
    PROVIDER_CONFIGS,
)

logger = logging.getLogger(__name__)

# Provider implementations (lazy imports to avoid dependency issues)
_provider_cache: dict[ProviderType, "CloudProvider"] = {}


def get_provider(provider_type: ProviderType) -> "CloudProvider":
    """Get a cloud provider instance by type.

    Args:
        provider_type: The type of provider to get

    Returns:
        CloudProvider instance for the specified type

    Raises:
        ValueError: If provider type is unknown
        DeprecationWarning: If Lambda provider is requested (account terminated Dec 2025)
    """
    if provider_type in _provider_cache:
        return _provider_cache[provider_type]

    if provider_type == ProviderType.VULTR:
        from app.coordination.providers.vultr_provider import VultrProvider
        _provider_cache[provider_type] = VultrProvider()
    elif provider_type == ProviderType.HETZNER:
        from app.coordination.providers.hetzner_provider import HetznerProvider
        _provider_cache[provider_type] = HetznerProvider()
    elif provider_type == ProviderType.LAMBDA:
        warnings.warn(
            "Lambda Labs account terminated Dec 2025. Lambda provider is deprecated. "
            "Returning None - use VAST or RUNPOD providers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning("Lambda provider requested but account is terminated - returning None")
        return None  # Lambda Labs account terminated Dec 2025
    elif provider_type == ProviderType.VAST:
        from app.coordination.providers.vast_provider import VastProvider
        _provider_cache[provider_type] = VastProvider()
    elif provider_type == ProviderType.RUNPOD:
        from app.coordination.providers.runpod_provider import RunPodProvider
        _provider_cache[provider_type] = RunPodProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return _provider_cache[provider_type]


def get_all_providers() -> list["CloudProvider"]:
    """Get all configured cloud providers.

    Returns:
        List of available CloudProvider instances
    """
    providers = []
    for provider_type in ProviderType:
        try:
            provider = get_provider(provider_type)
            if provider.is_configured():
                providers.append(provider)
        except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
            pass  # Provider not available
    return providers


def reset_providers() -> None:
    """Reset provider cache (useful for testing)."""
    global _provider_cache
    _provider_cache = {}


__all__ = [
    # Base types
    "CloudProvider",
    "Instance",
    "InstanceStatus",
    "ProviderType",
    "GPUType",
    # Provider registry (centralized configuration)
    "ProviderRegistry",
    "ProviderConfig",
    "CloudProviderProtocol",
    "PROVIDER_CONFIGS",
    # Factory functions
    "get_provider",
    "get_all_providers",
    "reset_providers",
]
