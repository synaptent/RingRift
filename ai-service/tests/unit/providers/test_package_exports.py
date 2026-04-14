"""Focused tests for app.providers package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_provider_surface() -> None:
    module = importlib.import_module("app.providers")

    expected = [
        "Provider",
        "ProviderInstance",
        "ProviderManager",
        "InstanceState",
        "HealthCheckResult",
        "RecoveryResult",
        "LambdaManager",
        "VastManager",
        "HetznerManager",
        "AWSManager",
        "TailscaleManager",
        "TailscalePeer",
        "TailscaleStatus",
        "VastOffer",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
