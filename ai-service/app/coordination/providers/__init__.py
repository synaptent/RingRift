"""Cloud provider abstraction layer for multi-provider cluster management.

This module provides a unified interface for managing compute resources across
multiple cloud providers (Lambda, Vultr, Vast.ai, Hetzner).

Usage:
    from app.coordination.providers import get_provider, ProviderType

    # Get a specific provider
    vultr = get_provider(ProviderType.VULTR)
    instances = await vultr.list_instances()

    # Get all available providers
    for provider in get_all_providers():
        print(f"{provider.name}: {await provider.get_available_gpus()}")
"""

from app.coordination.providers.base import (
    CloudProvider,
    Instance,
    InstanceStatus,
    ProviderType,
    GPUType,
)

# Provider implementations (lazy imports to avoid dependency issues)
_provider_cache: dict[ProviderType, "CloudProvider"] = {}


def get_provider(provider_type: ProviderType) -> "CloudProvider":
    """Get a cloud provider instance by type.

    Args:
        provider_type: The type of provider to get

    Returns:
        CloudProvider instance for the specified type
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
        from app.coordination.providers.lambda_provider import LambdaProvider
        _provider_cache[provider_type] = LambdaProvider()
    elif provider_type == ProviderType.VAST:
        from app.coordination.providers.vast_provider import VastProvider
        _provider_cache[provider_type] = VastProvider()
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
    "CloudProvider",
    "Instance",
    "InstanceStatus",
    "ProviderType",
    "GPUType",
    "get_provider",
    "get_all_providers",
    "reset_providers",
]
