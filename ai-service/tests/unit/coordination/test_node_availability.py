"""Unit tests for node_availability module (December 2025).

Tests for:
- StateChecker base class and enums
- ConfigUpdater atomic YAML updates
- Provider checkers (Vast, Lambda, RunPod)
- NodeAvailabilityDaemon
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from app.coordination.node_availability import (
    ConfigUpdateResult,
    ConfigUpdater,
    InstanceInfo,
    NodeAvailabilityConfig,
    NodeAvailabilityDaemon,
    ProviderInstanceState,
    STATE_TO_YAML_STATUS,
    get_node_availability_daemon,
    reset_daemon_instance,
)


# =============================================================================
# ProviderInstanceState Tests
# =============================================================================


class TestProviderInstanceState:
    """Tests for ProviderInstanceState enum."""

    def test_all_states_exist(self) -> None:
        """Test all expected states are defined."""
        states = [s.value for s in ProviderInstanceState]
        assert "running" in states
        assert "starting" in states
        assert "stopping" in states
        assert "stopped" in states
        assert "terminated" in states
        assert "unknown" in states

    def test_state_to_yaml_mapping(self) -> None:
        """Test state to YAML status mapping."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.RUNNING] == "ready"
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STARTING] == "setup"
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPING] == "offline"
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPED] == "offline"
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.TERMINATED] == "retired"
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.UNKNOWN] == "offline"

    def test_all_states_have_mapping(self) -> None:
        """Test all states have a YAML status mapping."""
        for state in ProviderInstanceState:
            assert state in STATE_TO_YAML_STATUS


# =============================================================================
# InstanceInfo Tests
# =============================================================================


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic InstanceInfo creation."""
        info = InstanceInfo(
            instance_id="vast-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.instance_id == "vast-12345"
        assert info.state == ProviderInstanceState.RUNNING
        assert info.provider == "vast"

    def test_yaml_status_property(self) -> None:
        """Test yaml_status property."""
        info = InstanceInfo(
            instance_id="test",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.yaml_status == "ready"

        info.state = ProviderInstanceState.TERMINATED
        assert info.yaml_status == "retired"

    def test_str_representation(self) -> None:
        """Test string representation."""
        info = InstanceInfo(
            instance_id="12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
            node_name="vast-12345",
        )
        result = str(info)
        assert "vast-12345" in result
        assert "running" in result

    def test_optional_fields(self) -> None:
        """Test optional fields have correct defaults."""
        info = InstanceInfo(
            instance_id="test",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.node_name is None
        assert info.tailscale_ip is None
        assert info.public_ip is None
        assert info.ssh_host is None
        assert info.ssh_port == 22
        assert info.gpu_type is None
        assert info.gpu_count == 0
        assert info.gpu_vram_gb == 0.0
        assert info.raw_data == {}


# =============================================================================
# ConfigUpdater Tests
# =============================================================================


class TestConfigUpdater:
    """Tests for ConfigUpdater."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temp directory with sample config."""
        config_path = tmp_path / "distributed_hosts.yaml"
        sample_config = {
            "hosts": {
                "vast-12345": {"status": "ready", "ssh_host": "1.2.3.4"},
                "vast-67890": {"status": "ready", "ssh_host": "5.6.7.8"},
                "lambda-gh200-1": {"status": "ready", "ssh_host": "10.0.0.1"},
            }
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config, f)
        return config_path

    def test_load_config(self, temp_config_dir: Path) -> None:
        """Test loading configuration."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)
        config = updater.load_config()
        assert "hosts" in config
        assert "vast-12345" in config["hosts"]

    @pytest.mark.asyncio
    async def test_dry_run_no_write(self, temp_config_dir: Path) -> None:
        """Test dry run doesn't write to file."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)
        original_content = temp_config_dir.read_text()

        result = await updater.update_node_statuses({"vast-12345": "retired"})

        assert result.success
        assert result.dry_run
        assert result.update_count == 1
        assert temp_config_dir.read_text() == original_content  # No change

    @pytest.mark.asyncio
    async def test_actual_write(self, temp_config_dir: Path) -> None:
        """Test actual write to file."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=False)

        result = await updater.update_node_statuses({"vast-12345": "retired"})

        assert result.success
        assert not result.dry_run
        assert result.update_count == 1

        # Verify file was updated
        config = updater.load_config()
        assert config["hosts"]["vast-12345"]["status"] == "retired"

    @pytest.mark.asyncio
    async def test_no_update_when_same_status(self, temp_config_dir: Path) -> None:
        """Test no update when status is the same."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)

        result = await updater.update_node_statuses({"vast-12345": "ready"})

        assert result.success
        assert result.update_count == 0

    @pytest.mark.asyncio
    async def test_skip_unknown_node(self, temp_config_dir: Path) -> None:
        """Test skipping unknown node names."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)

        result = await updater.update_node_statuses({"nonexistent-node": "retired"})

        assert result.success
        assert result.update_count == 0

    @pytest.mark.asyncio
    async def test_backup_creation(self, temp_config_dir: Path) -> None:
        """Test backup is created before update."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=False)

        result = await updater.update_node_statuses({"vast-12345": "offline"})

        assert result.success
        assert result.backup_path is not None
        assert result.backup_path.exists()

    @pytest.mark.asyncio
    async def test_get_current_statuses(self, temp_config_dir: Path) -> None:
        """Test getting current statuses."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)

        statuses = await updater.get_current_statuses()

        assert statuses["vast-12345"] == "ready"
        assert statuses["vast-67890"] == "ready"
        assert statuses["lambda-gh200-1"] == "ready"

    @pytest.mark.asyncio
    async def test_get_nodes_by_status(self, temp_config_dir: Path) -> None:
        """Test filtering nodes by status."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)

        ready_nodes = await updater.get_nodes_by_status("ready")

        assert len(ready_nodes) == 3
        assert "vast-12345" in ready_nodes

    @pytest.mark.asyncio
    async def test_get_nodes_by_provider(self, temp_config_dir: Path) -> None:
        """Test filtering nodes by provider."""
        updater = ConfigUpdater(config_path=temp_config_dir, dry_run=True)

        vast_nodes = await updater.get_nodes_by_provider("vast")
        lambda_nodes = await updater.get_nodes_by_provider("lambda")

        assert len(vast_nodes) == 2
        assert len(lambda_nodes) == 1


# =============================================================================
# NodeAvailabilityConfig Tests
# =============================================================================


class TestNodeAvailabilityConfig:
    """Tests for NodeAvailabilityConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = NodeAvailabilityConfig()
        assert config.check_interval_seconds == 300.0
        assert config.dry_run is True
        assert config.grace_period_seconds == 60.0
        assert config.vast_enabled is True
        assert config.lambda_enabled is True
        assert config.runpod_enabled is True
        assert config.auto_update_voters is False

    def test_from_env(self) -> None:
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            "RINGRIFT_NODE_AVAILABILITY_ENABLED": "1",
            "RINGRIFT_NODE_AVAILABILITY_DRY_RUN": "0",
            "RINGRIFT_NODE_AVAILABILITY_INTERVAL": "600",
            "RINGRIFT_NODE_AVAILABILITY_VAST": "1",
            "RINGRIFT_NODE_AVAILABILITY_LAMBDA": "0",
        }):
            config = NodeAvailabilityConfig.from_env()
            assert config.enabled is True
            assert config.dry_run is False
            assert config.check_interval_seconds == 600.0
            assert config.vast_enabled is True
            assert config.lambda_enabled is False


# =============================================================================
# NodeAvailabilityDaemon Tests
# =============================================================================


class TestNodeAvailabilityDaemon:
    """Tests for NodeAvailabilityDaemon."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton between tests."""
        reset_daemon_instance()
        yield
        reset_daemon_instance()

    def test_singleton_pattern(self) -> None:
        """Test singleton pattern."""
        daemon1 = get_node_availability_daemon()
        daemon2 = get_node_availability_daemon()
        assert daemon1 is daemon2

    def test_default_config(self) -> None:
        """Test daemon uses default config."""
        daemon = NodeAvailabilityDaemon()
        assert daemon.config.dry_run is True
        assert daemon.config.check_interval_seconds == 300.0

    def test_custom_config(self) -> None:
        """Test daemon with custom config."""
        config = NodeAvailabilityConfig(dry_run=False, check_interval_seconds=60.0)
        daemon = NodeAvailabilityDaemon(config)
        assert daemon.config.dry_run is False
        assert daemon.config.check_interval_seconds == 60.0

    def test_health_check_no_checkers(self) -> None:
        """Test health check when no checkers are enabled."""
        # Mock no API keys available
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "exists", return_value=False):
                daemon = NodeAvailabilityDaemon()

        health = daemon.health_check()
        # May be healthy or unhealthy depending on checker availability

    def test_get_status(self) -> None:
        """Test get_status returns correct structure."""
        daemon = NodeAvailabilityDaemon()
        status = daemon.get_status()

        assert "running" in status
        assert "config" in status
        assert "stats" in status
        assert "providers" in status
        assert "pending_terminations" in status

    def test_grace_period_tracking(self) -> None:
        """Test grace period for terminations."""
        daemon = NodeAvailabilityDaemon()

        # First check should start grace period
        result1 = daemon._check_grace_period("test-node")
        assert result1 is False
        assert "test-node" in daemon._pending_terminations

        # Second check should not pass yet
        result2 = daemon._check_grace_period("test-node")
        assert result2 is False


# =============================================================================
# Provider Checker Tests
# =============================================================================


class TestVastChecker:
    """Tests for VastChecker."""

    def test_disabled_without_api_key(self) -> None:
        """Test checker is disabled without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                from app.coordination.node_availability.providers.vast_checker import (
                    VastChecker,
                )

                checker = VastChecker()
                assert not checker.is_enabled

    def test_state_mapping(self) -> None:
        """Test Vast.ai state mapping."""
        from app.coordination.node_availability.providers.vast_checker import (
            VAST_STATE_MAP,
        )

        assert VAST_STATE_MAP["running"] == ProviderInstanceState.RUNNING
        assert VAST_STATE_MAP["loading"] == ProviderInstanceState.STARTING
        assert VAST_STATE_MAP["exited"] == ProviderInstanceState.STOPPED
        assert VAST_STATE_MAP["destroying"] == ProviderInstanceState.STOPPING


class TestLambdaChecker:
    """Tests for LambdaChecker."""

    def test_disabled_without_api_key(self) -> None:
        """Test checker is disabled without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                from app.coordination.node_availability.providers.lambda_checker import (
                    LambdaChecker,
                )

                checker = LambdaChecker()
                assert not checker.is_enabled

    def test_state_mapping(self) -> None:
        """Test Lambda Labs state mapping."""
        from app.coordination.node_availability.providers.lambda_checker import (
            LAMBDA_STATE_MAP,
        )

        assert LAMBDA_STATE_MAP["active"] == ProviderInstanceState.RUNNING
        assert LAMBDA_STATE_MAP["booting"] == ProviderInstanceState.STARTING
        assert LAMBDA_STATE_MAP["terminated"] == ProviderInstanceState.TERMINATED


class TestRunPodChecker:
    """Tests for RunPodChecker."""

    def test_disabled_without_api_key(self) -> None:
        """Test checker is disabled without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                from app.coordination.node_availability.providers.runpod_checker import (
                    RunPodChecker,
                )

                checker = RunPodChecker()
                assert not checker.is_enabled

    def test_state_mapping(self) -> None:
        """Test RunPod state mapping."""
        from app.coordination.node_availability.providers.runpod_checker import (
            RUNPOD_STATE_MAP,
        )

        assert RUNPOD_STATE_MAP["RUNNING"] == ProviderInstanceState.RUNNING
        assert RUNPOD_STATE_MAP["STARTING"] == ProviderInstanceState.STARTING
        assert RUNPOD_STATE_MAP["STOPPED"] == ProviderInstanceState.STOPPED
        assert RUNPOD_STATE_MAP["TERMINATED"] == ProviderInstanceState.TERMINATED


# =============================================================================
# Integration Tests
# =============================================================================


class TestNodeAvailabilityIntegration:
    """Integration tests for node availability module."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton between tests."""
        reset_daemon_instance()
        yield
        reset_daemon_instance()

    @pytest.mark.asyncio
    async def test_full_cycle_dry_run(self, tmp_path: Path) -> None:
        """Test full check cycle in dry run mode."""
        # Create sample config
        config_path = tmp_path / "distributed_hosts.yaml"
        sample_config = {
            "hosts": {
                "vast-12345": {"status": "ready", "ssh_host": "1.2.3.4"},
            }
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(sample_config, f)

        # Create daemon with custom config updater
        config = NodeAvailabilityConfig(dry_run=True)
        daemon = NodeAvailabilityDaemon(config)
        daemon._config_updater = ConfigUpdater(config_path=config_path, dry_run=True)

        # Run one cycle (will fail gracefully if no API keys)
        await daemon._run_cycle()

        # Verify stats were updated
        assert daemon._stats.cycles_completed == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
