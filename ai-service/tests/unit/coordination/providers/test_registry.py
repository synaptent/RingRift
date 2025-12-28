"""Tests for ProviderRegistry - Cloud provider abstraction layer.

Tests cover:
- ProviderConfig dataclass and pattern matching
- ProviderRegistry registration and lookup
- Node name pattern matching
- Provider caching behavior
- Default provider configurations
- Ephemeral provider filtering
- Threshold-based filtering
"""

import pytest
import re

from app.coordination.providers.registry import (
    CloudProviderProtocol,
    ProviderConfig,
    ProviderRegistry,
    PROVIDER_CONFIGS,
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_default_values(self):
        """Test default values for ProviderConfig."""
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
        """Test custom values for ProviderConfig."""
        config = ProviderConfig(
            name="custom",
            idle_threshold_seconds=600,
            min_nodes_to_keep=2,
            is_ephemeral=True,
            node_pattern=r"custom-\d+",
            ringrift_path="/opt/ringrift",
            ssh_key="~/.ssh/custom_key",
            ssh_user="admin",
            extra={"billing": "hourly"},
        )

        assert config.name == "custom"
        assert config.idle_threshold_seconds == 600
        assert config.min_nodes_to_keep == 2
        assert config.is_ephemeral is True
        assert config.node_pattern == r"custom-\d+"
        assert config.ringrift_path == "/opt/ringrift"
        assert config.ssh_key == "~/.ssh/custom_key"
        assert config.ssh_user == "admin"
        assert config.extra == {"billing": "hourly"}

    def test_frozen_dataclass(self):
        """Test that ProviderConfig is immutable."""
        config = ProviderConfig(name="test")

        with pytest.raises(AttributeError):
            config.name = "modified"  # type: ignore

    def test_equality(self):
        """Test ProviderConfig equality comparison."""
        config1 = ProviderConfig(name="test", idle_threshold_seconds=600)
        config2 = ProviderConfig(name="test", idle_threshold_seconds=600)
        config3 = ProviderConfig(name="test", idle_threshold_seconds=900)

        assert config1 == config2
        assert config1 != config3


class TestProviderConfigMatches:
    """Tests for ProviderConfig.matches() pattern matching."""

    def test_matches_with_pattern(self):
        """Test pattern matching with regex pattern."""
        config = ProviderConfig(name="vast", node_pattern=r"vast-\d+")

        assert config.matches("vast-12345") is True
        assert config.matches("vast-99999") is True
        assert config.matches("runpod-h100") is False
        assert config.matches("VAST-12345") is True  # Case insensitive

    def test_matches_without_pattern(self):
        """Test pattern matching with name prefix fallback."""
        config = ProviderConfig(name="custom")

        assert config.matches("custom-node-1") is True
        assert config.matches("custom") is True
        assert config.matches("other-node") is False

    def test_matches_complex_pattern(self):
        """Test pattern matching with complex regex."""
        config = ProviderConfig(
            name="local",
            node_pattern=r"(local-.*|mac-.*|macbook-.*)"
        )

        assert config.matches("local-dev") is True
        assert config.matches("mac-studio") is True
        assert config.matches("macbook-pro") is True
        assert config.matches("linux-box") is False

    def test_matches_empty_string(self):
        """Test pattern matching with empty node name."""
        config = ProviderConfig(name="test", node_pattern=r"test-\d+")

        assert config.matches("") is False


class TestProviderConfigShutdownCommand:
    """Tests for ProviderConfig.get_shutdown_command()."""

    def test_default_shutdown_command(self):
        """Test default shutdown command."""
        config = ProviderConfig(name="test")
        cmd = config.get_shutdown_command("node-123")

        assert "shutdown" in cmd
        assert "node-123" in cmd

    def test_custom_shutdown_command_template(self):
        """Test custom shutdown command from extra."""
        config = ProviderConfig(
            name="vast",
            extra={"shutdown_command_template": "vastai destroy instance {node_id}"}
        )
        cmd = config.get_shutdown_command("12345")

        assert cmd == "vastai destroy instance 12345"

    def test_shutdown_command_special_characters(self):
        """Test shutdown command with special characters in node_id."""
        config = ProviderConfig(name="test")
        cmd = config.get_shutdown_command("node-with-dashes_and_underscores")

        assert "node-with-dashes_and_underscores" in cmd


class TestDefaultProviderConfigs:
    """Tests for PROVIDER_CONFIGS default configurations."""

    def test_all_default_providers_exist(self):
        """Test that all expected providers are configured."""
        expected = ["vast", "runpod", "nebius", "lambda", "vultr", "hetzner", "local"]

        for provider in expected:
            assert provider in PROVIDER_CONFIGS, f"Missing provider: {provider}"

    def test_vast_config(self):
        """Test Vast.ai provider configuration."""
        config = PROVIDER_CONFIGS["vast"]

        assert config.name == "vast"
        assert config.idle_threshold_seconds == 900  # 15 min
        assert config.is_ephemeral is True
        assert config.ssh_user == "root"
        assert re.match(config.node_pattern, "vast-12345")
        assert "vastai" in config.get_shutdown_command("12345")

    def test_runpod_config(self):
        """Test RunPod provider configuration."""
        config = PROVIDER_CONFIGS["runpod"]

        assert config.name == "runpod"
        assert config.idle_threshold_seconds == 1200  # 20 min
        assert config.min_nodes_to_keep == 1
        assert config.is_ephemeral is True
        assert "workspace" in config.ringrift_path

    def test_nebius_config(self):
        """Test Nebius provider configuration."""
        config = PROVIDER_CONFIGS["nebius"]

        assert config.name == "nebius"
        assert config.idle_threshold_seconds == 3600  # 60 min
        assert config.is_ephemeral is False
        assert config.ssh_user == "ubuntu"

    def test_lambda_config(self):
        """Test Lambda Labs provider configuration."""
        config = PROVIDER_CONFIGS["lambda"]

        assert config.name == "lambda"
        assert config.is_ephemeral is False
        assert config.extra.get("has_nfs") == "true"

    def test_hetzner_config(self):
        """Test Hetzner provider configuration."""
        config = PROVIDER_CONFIGS["hetzner"]

        assert config.name == "hetzner"
        assert config.min_nodes_to_keep == 3  # P2P voting quorum
        assert config.extra.get("gpu") == "none"
        assert config.extra.get("role") == "p2p_voter"

    def test_local_config(self):
        """Test local development provider configuration."""
        config = PROVIDER_CONFIGS["local"]

        assert config.name == "local"
        assert config.idle_threshold_seconds == 86400  # 24 hr
        assert config.min_nodes_to_keep == 1
        assert config.extra.get("is_coordinator") == "true"


class TestProviderRegistryLookup:
    """Tests for ProviderRegistry node lookup."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_get_for_node_vast(self):
        """Test getting config for Vast.ai node."""
        config = ProviderRegistry.get_for_node("vast-12345")

        assert config.name == "vast"
        assert config.is_ephemeral is True

    def test_get_for_node_runpod(self):
        """Test getting config for RunPod node."""
        config = ProviderRegistry.get_for_node("runpod-h100")

        assert config.name == "runpod"

    def test_get_for_node_unknown(self):
        """Test getting config for unknown node returns generic."""
        config = ProviderRegistry.get_for_node("unknown-provider-123")

        assert config.name == "generic"

    def test_get_for_node_caching(self):
        """Test that node lookups are cached."""
        # First lookup
        config1 = ProviderRegistry.get_for_node("vast-12345")
        # Second lookup should hit cache
        config2 = ProviderRegistry.get_for_node("vast-12345")

        assert config1 is config2  # Same object (cached)

    def test_get_for_node_case_insensitive(self):
        """Test pattern matching is case insensitive."""
        config = ProviderRegistry.get_for_node("VAST-12345")

        assert config.name == "vast"


class TestProviderRegistryByName:
    """Tests for ProviderRegistry.get() by name."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_get_existing_provider(self):
        """Test getting existing provider by name."""
        config = ProviderRegistry.get("vast")

        assert config is not None
        assert config.name == "vast"

    def test_get_nonexistent_provider(self):
        """Test getting nonexistent provider returns None."""
        config = ProviderRegistry.get("nonexistent")

        assert config is None


class TestProviderRegistryRegistration:
    """Tests for ProviderRegistry registration and unregistration."""

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

        config = ProviderRegistry.get("mycloud")
        assert config is not None
        assert config.name == "mycloud"
        assert config.idle_threshold_seconds == 600

    def test_register_override_existing(self):
        """Test that registering overrides existing provider."""
        # Register override for vast
        custom_vast = ProviderConfig(
            name="vast",
            idle_threshold_seconds=300,  # 5 min instead of 15
        )
        ProviderRegistry.register(custom_vast)

        config = ProviderRegistry.get("vast")
        assert config is not None
        assert config.idle_threshold_seconds == 300

    def test_register_invalidates_cache(self):
        """Test that registration invalidates node cache."""
        # Cache a lookup
        ProviderRegistry.get_for_node("vast-12345")
        cache_before = len(ProviderRegistry._node_cache)

        # Register new provider
        custom = ProviderConfig(name="newprovider")
        ProviderRegistry.register(custom)

        assert len(ProviderRegistry._node_cache) == 0

    def test_unregister_existing(self):
        """Test unregistering an existing provider."""
        result = ProviderRegistry.unregister("vast")

        assert result is True
        assert ProviderRegistry.get("vast") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent provider returns False."""
        result = ProviderRegistry.unregister("nonexistent")

        assert result is False


class TestProviderRegistryQueries:
    """Tests for ProviderRegistry query methods."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_all_providers(self):
        """Test getting all provider names."""
        providers = ProviderRegistry.all_providers()

        assert isinstance(providers, list)
        assert "vast" in providers
        assert "runpod" in providers
        assert len(providers) == len(PROVIDER_CONFIGS)

    def test_all_configs(self):
        """Test getting all configurations."""
        configs = ProviderRegistry.all_configs()

        assert isinstance(configs, dict)
        assert "vast" in configs
        assert isinstance(configs["vast"], ProviderConfig)
        # Verify it's a copy, not the original
        assert configs is not ProviderRegistry._configs

    def test_get_ephemeral_providers(self):
        """Test getting ephemeral provider names."""
        ephemeral = ProviderRegistry.get_ephemeral_providers()

        assert "vast" in ephemeral
        assert "runpod" in ephemeral
        assert "nebius" not in ephemeral  # Not ephemeral
        assert "hetzner" not in ephemeral  # Not ephemeral

    def test_get_by_idle_threshold_returns_sorted(self):
        """Test getting providers by idle threshold returns sorted list."""
        # vast=900, runpod=1200, so both should be returned if max=1500
        matching = ProviderRegistry.get_by_idle_threshold(1500)

        assert len(matching) >= 2
        # Should be sorted ascending
        thresholds = [c.idle_threshold_seconds for c in matching]
        assert thresholds == sorted(thresholds)

    def test_get_by_idle_threshold_filters(self):
        """Test that threshold filtering works."""
        # Only vast (900) should match with max=1000
        matching = ProviderRegistry.get_by_idle_threshold(1000)

        assert len(matching) >= 1
        for config in matching:
            assert config.idle_threshold_seconds <= 1000


class TestProviderRegistryReset:
    """Tests for ProviderRegistry.reset()."""

    def test_reset_restores_defaults(self):
        """Test that reset restores default configurations."""
        # Modify registry
        ProviderRegistry.unregister("vast")
        custom = ProviderConfig(name="custom")
        ProviderRegistry.register(custom)

        # Reset
        ProviderRegistry.reset()

        assert ProviderRegistry.get("vast") is not None
        assert ProviderRegistry.get("custom") is None

    def test_reset_clears_cache(self):
        """Test that reset clears node cache."""
        # Populate cache
        ProviderRegistry.get_for_node("vast-12345")
        ProviderRegistry.get_for_node("runpod-h100")
        assert len(ProviderRegistry._node_cache) > 0

        ProviderRegistry.reset()

        assert len(ProviderRegistry._node_cache) == 0


class TestCloudProviderProtocol:
    """Tests for CloudProviderProtocol interface."""

    def test_provider_config_satisfies_protocol(self):
        """Test that ProviderConfig can satisfy the protocol requirements."""
        config = ProviderConfig(name="test", idle_threshold_seconds=600)

        # ProviderConfig has these attributes
        assert hasattr(config, "name")
        assert hasattr(config, "idle_threshold_seconds")
        assert hasattr(config, "min_nodes_to_keep")
        assert hasattr(config, "get_shutdown_command")
        assert hasattr(config, "is_ephemeral")

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol can be used for runtime checks."""
        class MyProvider:
            name = "test"
            idle_threshold_seconds = 600
            min_nodes_to_keep = 0

            def get_shutdown_command(self, node_id: str) -> str:
                return f"shutdown {node_id}"

            def is_ephemeral(self) -> bool:
                return False

        provider = MyProvider()
        assert isinstance(provider, CloudProviderProtocol)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Reset registry before each test."""
        ProviderRegistry.reset()

    def test_empty_node_name(self):
        """Test lookup with empty node name."""
        config = ProviderRegistry.get_for_node("")

        assert config.name == "generic"

    def test_node_name_with_special_characters(self):
        """Test lookup with special characters in node name."""
        config = ProviderRegistry.get_for_node("vast-12345-abc_def")

        # Should still match vast pattern
        assert config.name == "vast"

    def test_multiple_pattern_matches(self):
        """Test that first matching pattern wins."""
        # Both patterns could match, but order in dict determines winner
        # This tests that patterns are evaluated in consistent order
        config = ProviderRegistry.get_for_node("nebius-h100")

        assert config.name == "nebius"

    def test_get_by_idle_threshold_zero(self):
        """Test getting providers with zero threshold returns nothing."""
        matching = ProviderRegistry.get_by_idle_threshold(0)

        # No provider has 0 idle threshold
        assert len(matching) == 0

    def test_provider_config_extra_is_isolated(self):
        """Test that extra dict is isolated between instances."""
        config1 = ProviderConfig(name="test1")
        config2 = ProviderConfig(name="test2")

        # Modifying one shouldn't affect the other
        # (Though frozen=True prevents this, test the default factory)
        assert config1.extra is not config2.extra
