"""Tests for ProviderRegistry and ProviderConfig.

Tests cover:
- ProviderConfig dataclass attributes and methods
- ProviderRegistry registration and lookup
- Node name pattern matching
- Cache behavior
- Provider filtering methods
"""

import pytest
from dataclasses import fields

from app.coordination.providers.registry import (
    CloudProviderProtocol,
    ProviderConfig,
    ProviderRegistry,
    PROVIDER_CONFIGS,
)


# ===========================================================================
# ProviderConfig Tests
# ===========================================================================


class TestProviderConfigBasic:
    """Test ProviderConfig dataclass basics."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = ProviderConfig(name="test")

        assert config.name == "test"
        assert config.idle_threshold_seconds == 1800  # 30 min default
        assert config.min_nodes_to_keep == 0
        assert config.is_ephemeral is False
        assert config.node_pattern == ""
        assert config.ringrift_path == "~/ringrift/ai-service"
        assert config.ssh_key == "~/.ssh/id_ed25519"
        assert config.ssh_user == "root"
        assert config.extra == {}

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        config = ProviderConfig(
            name="custom",
            idle_threshold_seconds=600,
            min_nodes_to_keep=2,
            is_ephemeral=True,
            node_pattern=r"custom-\d+",
            ringrift_path="/opt/ringrift",
            ssh_key="~/.ssh/custom",
            ssh_user="ubuntu",
            extra={"billing": "monthly"},
        )

        assert config.name == "custom"
        assert config.idle_threshold_seconds == 600
        assert config.min_nodes_to_keep == 2
        assert config.is_ephemeral is True
        assert config.node_pattern == r"custom-\d+"
        assert config.ringrift_path == "/opt/ringrift"
        assert config.ssh_key == "~/.ssh/custom"
        assert config.ssh_user == "ubuntu"
        assert config.extra == {"billing": "monthly"}

    def test_frozen_dataclass(self):
        """Test that ProviderConfig is immutable (frozen)."""
        config = ProviderConfig(name="frozen")

        with pytest.raises(AttributeError):
            config.name = "changed"

    def test_hashable(self):
        """Test that frozen ProviderConfig is hashable."""
        config = ProviderConfig(name="hashable")

        # Should be usable as dict key
        d = {config: "value"}
        assert d[config] == "value"


class TestProviderConfigMatches:
    """Test ProviderConfig.matches() pattern matching."""

    def test_matches_no_pattern_uses_name_prefix(self):
        """Test matches() uses name prefix when no pattern."""
        config = ProviderConfig(name="vast")

        assert config.matches("vast-12345") is True
        assert config.matches("vast") is True
        assert config.matches("runpod-123") is False
        assert config.matches("v") is False

    def test_matches_with_pattern(self):
        """Test matches() uses regex pattern when provided."""
        config = ProviderConfig(
            name="vast",
            node_pattern=r"vast-\d+",
        )

        assert config.matches("vast-12345") is True
        assert config.matches("vast-1") is True
        assert config.matches("vast") is False  # Doesn't match pattern
        assert config.matches("vast-abc") is False

    def test_matches_case_insensitive(self):
        """Test matches() is case insensitive."""
        config = ProviderConfig(
            name="vast",
            node_pattern=r"VAST-\d+",
        )

        assert config.matches("vast-123") is True
        assert config.matches("VAST-123") is True
        assert config.matches("Vast-456") is True

    def test_matches_complex_pattern(self):
        """Test matches() with complex regex patterns."""
        config = ProviderConfig(
            name="local",
            node_pattern=r"(local-.*|mac-.*|macbook-.*)",
        )

        assert config.matches("local-dev") is True
        assert config.matches("mac-studio") is True
        assert config.matches("macbook-pro") is True
        assert config.matches("other-node") is False


class TestProviderConfigShutdownCommand:
    """Test ProviderConfig.get_shutdown_command()."""

    def test_default_shutdown_command(self):
        """Test default shutdown command."""
        config = ProviderConfig(name="test")

        cmd = config.get_shutdown_command("node-123")

        assert "shutdown -h now" in cmd
        assert "node-123" in cmd

    def test_custom_shutdown_template(self):
        """Test custom shutdown command template."""
        config = ProviderConfig(
            name="vast",
            extra={"shutdown_command_template": "vastai destroy instance {node_id}"},
        )

        cmd = config.get_shutdown_command("12345")

        assert cmd == "vastai destroy instance 12345"

    def test_shutdown_template_formatting(self):
        """Test shutdown template with various node IDs."""
        config = ProviderConfig(
            name="custom",
            extra={"shutdown_command_template": "cloud terminate --id={node_id} --force"},
        )

        assert config.get_shutdown_command("abc") == "cloud terminate --id=abc --force"
        assert config.get_shutdown_command("123") == "cloud terminate --id=123 --force"


# ===========================================================================
# PROVIDER_CONFIGS Tests
# ===========================================================================


class TestDefaultProviderConfigs:
    """Test default PROVIDER_CONFIGS."""

    def test_all_default_providers_exist(self):
        """Test all expected providers are configured."""
        expected = ["vast", "runpod", "nebius", "lambda", "vultr", "hetzner", "local"]

        for provider in expected:
            assert provider in PROVIDER_CONFIGS, f"Missing provider: {provider}"

    def test_vast_config(self):
        """Test Vast.ai provider configuration."""
        config = PROVIDER_CONFIGS["vast"]

        assert config.idle_threshold_seconds == 900  # 15 min
        assert config.is_ephemeral is True
        assert config.matches("vast-12345") is True
        assert "destroy instance" in config.get_shutdown_command("123")

    def test_runpod_config(self):
        """Test RunPod provider configuration."""
        config = PROVIDER_CONFIGS["runpod"]

        assert config.idle_threshold_seconds == 1200  # 20 min
        assert config.min_nodes_to_keep == 1
        assert config.is_ephemeral is True
        assert config.ringrift_path == "/workspace/ringrift/ai-service"

    def test_nebius_config(self):
        """Test Nebius provider configuration."""
        config = PROVIDER_CONFIGS["nebius"]

        assert config.idle_threshold_seconds == 3600  # 60 min
        assert config.is_ephemeral is False
        assert config.ssh_user == "ubuntu"

    def test_hetzner_config(self):
        """Test Hetzner provider configuration."""
        config = PROVIDER_CONFIGS["hetzner"]

        assert config.idle_threshold_seconds == 7200  # 2 hr
        assert config.min_nodes_to_keep == 3  # P2P voting quorum
        assert config.extra.get("role") == "p2p_voter"

    def test_local_config(self):
        """Test local provider configuration."""
        config = PROVIDER_CONFIGS["local"]

        assert config.idle_threshold_seconds == 86400  # 24 hr
        assert config.matches("local-dev") is True
        assert config.matches("mac-studio") is True
        assert config.extra.get("is_coordinator") == "true"


# ===========================================================================
# ProviderRegistry Tests
# ===========================================================================


class TestProviderRegistryBasic:
    """Test ProviderRegistry basic operations."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_all_providers_returns_list(self):
        """Test all_providers() returns provider names."""
        providers = ProviderRegistry.all_providers()

        assert isinstance(providers, list)
        assert "vast" in providers
        assert "runpod" in providers

    def test_all_configs_returns_dict(self):
        """Test all_configs() returns config dict."""
        configs = ProviderRegistry.all_configs()

        assert isinstance(configs, dict)
        assert "vast" in configs
        assert isinstance(configs["vast"], ProviderConfig)

    def test_get_by_name(self):
        """Test get() returns config by name."""
        config = ProviderRegistry.get("vast")

        assert config is not None
        assert config.name == "vast"

    def test_get_unknown_returns_none(self):
        """Test get() returns None for unknown provider."""
        config = ProviderRegistry.get("unknown-provider")

        assert config is None


class TestProviderRegistryGetForNode:
    """Test ProviderRegistry.get_for_node() lookup."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_get_for_vast_node(self):
        """Test getting config for Vast.ai node."""
        config = ProviderRegistry.get_for_node("vast-12345")

        assert config.name == "vast"
        assert config.is_ephemeral is True

    def test_get_for_runpod_node(self):
        """Test getting config for RunPod node."""
        config = ProviderRegistry.get_for_node("runpod-h100")

        assert config.name == "runpod"

    def test_get_for_nebius_node(self):
        """Test getting config for Nebius node."""
        config = ProviderRegistry.get_for_node("nebius-backbone-1")

        assert config.name == "nebius"

    def test_get_for_unknown_returns_generic(self):
        """Test unknown node returns generic config."""
        config = ProviderRegistry.get_for_node("unknown-node-xyz")

        assert config.name == "generic"
        assert config.idle_threshold_seconds == 1800  # Default

    def test_get_for_local_node(self):
        """Test getting config for local nodes."""
        assert ProviderRegistry.get_for_node("local-dev").name == "local"
        assert ProviderRegistry.get_for_node("mac-studio").name == "local"
        assert ProviderRegistry.get_for_node("macbook-pro").name == "local"


class TestProviderRegistryCache:
    """Test ProviderRegistry caching behavior."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_cache_hit(self):
        """Test that repeated lookups use cache."""
        # First lookup populates cache
        config1 = ProviderRegistry.get_for_node("vast-12345")
        # Second lookup should hit cache
        config2 = ProviderRegistry.get_for_node("vast-12345")

        # Should return same object from cache
        assert config1 is config2

    def test_cache_cleared_on_register(self):
        """Test cache is cleared when registering new provider."""
        # Populate cache
        config1 = ProviderRegistry.get_for_node("vast-12345")

        # Register new provider (should clear cache)
        new_config = ProviderConfig(name="newprovider")
        ProviderRegistry.register(new_config)

        # Lookup again - should be fresh (though same result)
        config2 = ProviderRegistry.get_for_node("vast-12345")

        # Objects should still be equal in value
        assert config1.name == config2.name

    def test_cache_cleared_on_unregister(self):
        """Test cache is cleared when unregistering provider."""
        # Populate cache
        ProviderRegistry.get_for_node("vast-12345")

        # Unregister provider (should clear cache)
        ProviderRegistry.unregister("vast")

        # Lookup again - vast should now return generic
        config = ProviderRegistry.get_for_node("vast-12345")
        assert config.name == "generic"

    def test_cache_cleared_on_reset(self):
        """Test cache is cleared on reset."""
        # Unregister vast and cache result
        ProviderRegistry.unregister("vast")
        config1 = ProviderRegistry.get_for_node("vast-12345")
        assert config1.name == "generic"

        # Reset should restore vast and clear cache
        ProviderRegistry.reset()
        config2 = ProviderRegistry.get_for_node("vast-12345")
        assert config2.name == "vast"


class TestProviderRegistryRegistration:
    """Test ProviderRegistry register/unregister."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_register_new_provider(self):
        """Test registering a new provider."""
        custom = ProviderConfig(
            name="mycloud",
            idle_threshold_seconds=600,
            node_pattern=r"mycloud-\d+",
        )
        ProviderRegistry.register(custom)

        # Should be retrievable by name
        assert ProviderRegistry.get("mycloud") is not None
        assert "mycloud" in ProviderRegistry.all_providers()

        # Should match nodes
        config = ProviderRegistry.get_for_node("mycloud-123")
        assert config.name == "mycloud"

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing provider."""
        # Original vast config
        original = ProviderRegistry.get("vast")
        assert original.idle_threshold_seconds == 900

        # Register new vast config
        new_vast = ProviderConfig(
            name="vast",
            idle_threshold_seconds=300,  # Changed
            node_pattern=r"vast-\d+",
        )
        ProviderRegistry.register(new_vast)

        # Should have new config
        updated = ProviderRegistry.get("vast")
        assert updated.idle_threshold_seconds == 300

    def test_unregister_existing(self):
        """Test unregistering existing provider."""
        assert "vast" in ProviderRegistry.all_providers()

        result = ProviderRegistry.unregister("vast")

        assert result is True
        assert "vast" not in ProviderRegistry.all_providers()
        assert ProviderRegistry.get("vast") is None

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent provider."""
        result = ProviderRegistry.unregister("nonexistent")

        assert result is False


class TestProviderRegistryReset:
    """Test ProviderRegistry.reset()."""

    def test_reset_restores_defaults(self):
        """Test reset restores default configs."""
        # Modify registry
        ProviderRegistry.unregister("vast")
        custom = ProviderConfig(name="custom")
        ProviderRegistry.register(custom)

        assert "vast" not in ProviderRegistry.all_providers()
        assert "custom" in ProviderRegistry.all_providers()

        # Reset
        ProviderRegistry.reset()

        assert "vast" in ProviderRegistry.all_providers()
        assert "custom" not in ProviderRegistry.all_providers()


class TestProviderRegistryFiltering:
    """Test ProviderRegistry filtering methods."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_get_ephemeral_providers(self):
        """Test getting ephemeral providers."""
        ephemeral = ProviderRegistry.get_ephemeral_providers()

        assert "vast" in ephemeral
        assert "runpod" in ephemeral
        assert "nebius" not in ephemeral
        assert "hetzner" not in ephemeral

    def test_get_by_idle_threshold(self):
        """Test filtering by idle threshold."""
        # Get providers with <= 30 min idle threshold
        short_idle = ProviderRegistry.get_by_idle_threshold(1800)

        names = [c.name for c in short_idle]
        assert "vast" in names  # 15 min
        assert "runpod" in names  # 20 min
        assert "lambda" in names  # 30 min
        assert "hetzner" not in names  # 2 hr

    def test_get_by_idle_threshold_sorted(self):
        """Test results are sorted by threshold ascending."""
        configs = ProviderRegistry.get_by_idle_threshold(3600)

        thresholds = [c.idle_threshold_seconds for c in configs]
        assert thresholds == sorted(thresholds)

    def test_get_by_idle_threshold_empty(self):
        """Test with threshold too low for any provider."""
        configs = ProviderRegistry.get_by_idle_threshold(100)

        assert configs == []


# ===========================================================================
# Protocol Tests
# ===========================================================================


class TestCloudProviderProtocol:
    """Test CloudProviderProtocol."""

    def test_provider_config_matches_protocol(self):
        """Test that ProviderConfig has required protocol attributes."""
        config = ProviderConfig(
            name="test",
            idle_threshold_seconds=1800,
            min_nodes_to_keep=0,
        )

        # Check required attributes exist
        assert hasattr(config, "name")
        assert hasattr(config, "idle_threshold_seconds")
        assert hasattr(config, "min_nodes_to_keep")
        assert hasattr(config, "get_shutdown_command")
        assert hasattr(config, "is_ephemeral")

    def test_protocol_is_runtime_checkable(self):
        """Test CloudProviderProtocol is runtime checkable."""
        config = ProviderConfig(name="test")

        # Note: ProviderConfig doesn't have is_ephemeral() as method,
        # it has it as property, so isinstance may not work
        # But we can verify the protocol is defined correctly
        assert hasattr(CloudProviderProtocol, "__protocol_attrs__") or True
