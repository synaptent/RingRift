"""Unit tests for state_checker.py.

Tests the base classes and data structures for cloud provider state checking.

Created: Dec 29, 2025
Phase 4: Test coverage for critical untested modules.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    STATE_TO_YAML_STATUS,
    StateChecker,
)


class TestProviderInstanceState:
    """Tests for ProviderInstanceState enum."""

    def test_all_states_defined(self):
        """Test all expected states are defined."""
        expected_states = ["RUNNING", "STARTING", "STOPPING", "STOPPED", "TERMINATED", "UNKNOWN"]
        for state_name in expected_states:
            assert hasattr(ProviderInstanceState, state_name)

    def test_state_values(self):
        """Test state values are lowercase strings."""
        assert ProviderInstanceState.RUNNING.value == "running"
        assert ProviderInstanceState.STARTING.value == "starting"
        assert ProviderInstanceState.STOPPING.value == "stopping"
        assert ProviderInstanceState.STOPPED.value == "stopped"
        assert ProviderInstanceState.TERMINATED.value == "terminated"
        assert ProviderInstanceState.UNKNOWN.value == "unknown"

    def test_state_count(self):
        """Test total number of states."""
        assert len(ProviderInstanceState) == 6

    def test_state_from_value(self):
        """Test creating state from string value."""
        state = ProviderInstanceState("running")
        assert state == ProviderInstanceState.RUNNING

    def test_invalid_state_value_raises(self):
        """Test invalid value raises ValueError."""
        with pytest.raises(ValueError):
            ProviderInstanceState("invalid_state")


class TestStateToYamlStatusMapping:
    """Tests for STATE_TO_YAML_STATUS mapping."""

    def test_all_states_mapped(self):
        """Test all states have yaml status mappings."""
        for state in ProviderInstanceState:
            assert state in STATE_TO_YAML_STATUS

    def test_running_maps_to_ready(self):
        """Test RUNNING maps to ready."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.RUNNING] == "ready"

    def test_starting_maps_to_setup(self):
        """Test STARTING maps to setup."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STARTING] == "setup"

    def test_stopping_maps_to_offline(self):
        """Test STOPPING maps to offline."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPING] == "offline"

    def test_stopped_maps_to_offline(self):
        """Test STOPPED maps to offline."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPED] == "offline"

    def test_terminated_maps_to_retired(self):
        """Test TERMINATED maps to retired."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.TERMINATED] == "retired"

    def test_unknown_maps_to_offline(self):
        """Test UNKNOWN maps to offline."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.UNKNOWN] == "offline"

    def test_mapping_completeness(self):
        """Test mapping has exactly 6 entries."""
        assert len(STATE_TO_YAML_STATUS) == 6


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_required_fields(self):
        """Test required fields are enforced."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.instance_id == "i-12345"
        assert info.state == ProviderInstanceState.RUNNING
        assert info.provider == "vast"

    def test_optional_fields_defaults(self):
        """Test optional fields have correct defaults."""
        info = InstanceInfo(
            instance_id="i-12345",
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
        assert info.hostname is None
        assert info.created_at is None
        assert info.last_seen is None
        assert info.raw_data == {}

    def test_all_optional_fields_set(self):
        """Test all optional fields can be set."""
        now = datetime.now()
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="lambda",
            node_name="lambda-h100-1",
            tailscale_ip="100.100.100.1",
            public_ip="1.2.3.4",
            ssh_host="lambda-h100-1.example.com",
            ssh_port=2222,
            gpu_type="H100",
            gpu_count=8,
            gpu_vram_gb=80.0,
            hostname="compute-node-1",
            created_at=now,
            last_seen=now,
            raw_data={"extra": "data"},
        )
        assert info.node_name == "lambda-h100-1"
        assert info.tailscale_ip == "100.100.100.1"
        assert info.public_ip == "1.2.3.4"
        assert info.ssh_host == "lambda-h100-1.example.com"
        assert info.ssh_port == 2222
        assert info.gpu_type == "H100"
        assert info.gpu_count == 8
        assert info.gpu_vram_gb == 80.0
        assert info.hostname == "compute-node-1"
        assert info.created_at == now
        assert info.last_seen == now
        assert info.raw_data == {"extra": "data"}

    def test_yaml_status_property_running(self):
        """Test yaml_status property for RUNNING state."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.yaml_status == "ready"

    def test_yaml_status_property_starting(self):
        """Test yaml_status property for STARTING state."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.STARTING,
            provider="vast",
        )
        assert info.yaml_status == "setup"

    def test_yaml_status_property_stopped(self):
        """Test yaml_status property for STOPPED state."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.STOPPED,
            provider="vast",
        )
        assert info.yaml_status == "offline"

    def test_yaml_status_property_terminated(self):
        """Test yaml_status property for TERMINATED state."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.TERMINATED,
            provider="vast",
        )
        assert info.yaml_status == "retired"

    def test_str_with_node_name(self):
        """Test __str__ when node_name is set."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
            node_name="vast-worker-1",
        )
        result = str(info)
        assert "vast-worker-1" in result
        assert "vast" in result
        assert "running" in result

    def test_str_without_node_name(self):
        """Test __str__ when node_name is not set (falls back to instance_id)."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.STOPPED,
            provider="lambda",
        )
        result = str(info)
        assert "i-12345" in result
        assert "lambda" in result
        assert "stopped" in result

    def test_yaml_status_fallback_for_unknown_state(self):
        """Test yaml_status returns 'offline' for unmapped states."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.UNKNOWN,
            provider="vast",
        )
        assert info.yaml_status == "offline"


class TestStateChecker:
    """Tests for StateChecker abstract base class."""

    def test_init_with_provider_name(self):
        """Test initialization with provider name."""
        # Create a concrete implementation for testing
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("test_provider")
        assert checker.provider_name == "test_provider"
        assert checker._enabled is True
        assert checker._last_check is None
        assert checker._last_error is None

    def test_is_enabled_property(self):
        """Test is_enabled property returns internal state."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("test")
        assert checker.is_enabled is True

        checker._enabled = False
        assert checker.is_enabled is False

    def test_disable_with_reason(self):
        """Test disable method sets flag and logs."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("test")
        assert checker.is_enabled is True

        with patch("app.coordination.node_availability.state_checker.logger") as mock_logger:
            checker.disable("missing API key")

        assert checker.is_enabled is False
        mock_logger.warning.assert_called_once()
        assert "missing API key" in str(mock_logger.warning.call_args)

    def test_enable(self):
        """Test enable method sets flag and logs."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("test")
        checker._enabled = False
        assert checker.is_enabled is False

        with patch("app.coordination.node_availability.state_checker.logger") as mock_logger:
            checker.enable()

        assert checker.is_enabled is True
        mock_logger.info.assert_called_once()

    def test_get_status_with_no_checks(self):
        """Test get_status with no checks performed."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("vast")
        status = checker.get_status()

        assert status["provider"] == "vast"
        assert status["enabled"] is True
        assert status["last_check"] is None
        assert status["last_error"] is None

    def test_get_status_with_last_check(self):
        """Test get_status with last_check set."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("lambda")
        now = datetime.now()
        checker._last_check = now
        checker._last_error = "Connection timeout"

        status = checker.get_status()

        assert status["provider"] == "lambda"
        assert status["enabled"] is True
        assert status["last_check"] == now.isoformat()
        assert status["last_error"] == "Connection timeout"

    def test_get_status_when_disabled(self):
        """Test get_status when checker is disabled."""
        class TestChecker(StateChecker):
            async def get_instance_states(self):
                return []

            async def check_api_availability(self):
                return True

            def correlate_with_config(self, instances, config_hosts):
                return instances

        checker = TestChecker("runpod")
        checker._enabled = False

        status = checker.get_status()

        assert status["provider"] == "runpod"
        assert status["enabled"] is False

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods cannot be instantiated directly."""
        # This should raise TypeError because abstract methods aren't implemented
        with pytest.raises(TypeError):
            StateChecker("test")  # type: ignore


class TestInstanceInfoProviders:
    """Tests for InstanceInfo with different provider scenarios."""

    def test_vast_instance(self):
        """Test InstanceInfo for Vast.ai instance."""
        info = InstanceInfo(
            instance_id="12345678",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
            node_name="vast-12345678",
            gpu_type="RTX 4090",
            gpu_count=1,
            gpu_vram_gb=24.0,
        )
        assert info.yaml_status == "ready"
        assert "vast" in str(info)

    def test_lambda_instance(self):
        """Test InstanceInfo for Lambda Labs instance."""
        info = InstanceInfo(
            instance_id="abc-123",
            state=ProviderInstanceState.STARTING,
            provider="lambda",
            node_name="lambda-gh200-1",
            gpu_type="GH200",
            gpu_count=1,
            gpu_vram_gb=96.0,
        )
        assert info.yaml_status == "setup"
        assert "lambda" in str(info)

    def test_runpod_instance(self):
        """Test InstanceInfo for RunPod instance."""
        info = InstanceInfo(
            instance_id="xyz-789",
            state=ProviderInstanceState.STOPPED,
            provider="runpod",
            node_name="runpod-a100-1",
            gpu_type="A100",
            gpu_count=1,
            gpu_vram_gb=80.0,
        )
        assert info.yaml_status == "offline"
        assert "runpod" in str(info)

    def test_terminated_instance_clears_correlation(self):
        """Test that terminated instances typically lose correlation."""
        info = InstanceInfo(
            instance_id="old-instance",
            state=ProviderInstanceState.TERMINATED,
            provider="vast",
            # No node_name - correlation lost
        )
        assert info.yaml_status == "retired"
        # Falls back to instance_id in string repr
        assert "old-instance" in str(info)


class TestStateCheckerSubclass:
    """Tests for concrete StateChecker subclass behavior."""

    def create_mock_checker(self):
        """Create a mock StateChecker for testing."""
        class MockChecker(StateChecker):
            def __init__(self):
                super().__init__("mock")
                self.instances = []
                self.api_available = True
                self.correlated = []

            async def get_instance_states(self):
                return self.instances

            async def check_api_availability(self):
                return self.api_available

            def correlate_with_config(self, instances, config_hosts):
                self.correlated = instances
                return instances

        return MockChecker()

    @pytest.mark.asyncio
    async def test_get_instance_states_returns_list(self):
        """Test get_instance_states returns list of InstanceInfo."""
        checker = self.create_mock_checker()
        checker.instances = [
            InstanceInfo("i-1", ProviderInstanceState.RUNNING, "mock"),
            InstanceInfo("i-2", ProviderInstanceState.STOPPED, "mock"),
        ]

        result = await checker.get_instance_states()
        assert len(result) == 2
        assert all(isinstance(i, InstanceInfo) for i in result)

    @pytest.mark.asyncio
    async def test_check_api_availability(self):
        """Test check_api_availability returns boolean."""
        checker = self.create_mock_checker()
        checker.api_available = True

        result = await checker.check_api_availability()
        assert result is True

        checker.api_available = False
        result = await checker.check_api_availability()
        assert result is False

    def test_correlate_with_config(self):
        """Test correlate_with_config processes instances."""
        checker = self.create_mock_checker()
        instances = [
            InstanceInfo("i-1", ProviderInstanceState.RUNNING, "mock"),
        ]
        config_hosts = {"mock-node": {"status": "ready"}}

        result = checker.correlate_with_config(instances, config_hosts)
        assert result == instances
        assert checker.correlated == instances
