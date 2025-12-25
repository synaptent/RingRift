"""Tests for providers base classes."""

import pytest
from datetime import datetime

from app.providers.base import (
    Provider,
    InstanceState,
    ProviderInstance,
    HealthCheckResult,
    RecoveryResult,
)


class TestProvider:
    """Tests for Provider enum."""

    def test_all_providers_exist(self):
        """All expected providers are defined."""
        assert Provider.LAMBDA.value == "lambda"
        assert Provider.VAST.value == "vast"
        assert Provider.HETZNER.value == "hetzner"
        assert Provider.AWS.value == "aws"
        assert Provider.LOCAL.value == "local"

    def test_provider_is_string_enum(self):
        """Provider values are strings."""
        for provider in Provider:
            assert isinstance(provider.value, str)


class TestInstanceState:
    """Tests for InstanceState enum."""

    def test_all_states_exist(self):
        """All expected states are defined."""
        expected = ["running", "starting", "stopping", "stopped", "terminated", "error", "unknown"]
        actual = [s.value for s in InstanceState]
        for state in expected:
            assert state in actual

    def test_state_is_string_enum(self):
        """State values are strings."""
        for state in InstanceState:
            assert isinstance(state.value, str)


class TestProviderInstance:
    """Tests for ProviderInstance dataclass."""

    def test_minimal_instance(self):
        """Can create instance with minimal fields."""
        inst = ProviderInstance(
            instance_id="test-123",
            provider=Provider.LAMBDA,
            name="test-node",
        )
        assert inst.instance_id == "test-123"
        assert inst.provider == Provider.LAMBDA
        assert inst.name == "test-node"
        assert inst.state == InstanceState.UNKNOWN

    def test_full_instance(self):
        """Can create instance with all fields."""
        inst = ProviderInstance(
            instance_id="inst-456",
            provider=Provider.VAST,
            name="vast-gpu-node",
            public_ip="1.2.3.4",
            private_ip="10.0.0.1",
            tailscale_ip="100.1.2.3",
            ssh_port=22,
            state=InstanceState.RUNNING,
            gpu_type="RTX 4090",
            gpu_count=4,
            gpu_memory_gb=96,
            cpu_count=64,
            memory_gb=256,
            hourly_cost=2.50,
            metadata={"region": "us-west"},
        )
        assert inst.public_ip == "1.2.3.4"
        assert inst.gpu_count == 4
        assert inst.hourly_cost == 2.50

    def test_ssh_host_prefers_tailscale(self):
        """ssh_host property prefers Tailscale IP."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test",
            public_ip="1.2.3.4",
            private_ip="10.0.0.1",
            tailscale_ip="100.1.2.3",
        )
        assert inst.ssh_host == "100.1.2.3"

    def test_ssh_host_falls_back_to_private(self):
        """ssh_host falls back to private IP."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test",
            public_ip="1.2.3.4",
            private_ip="10.0.0.1",
        )
        assert inst.ssh_host == "10.0.0.1"

    def test_ssh_host_falls_back_to_public(self):
        """ssh_host falls back to public IP."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test",
            public_ip="1.2.3.4",
        )
        assert inst.ssh_host == "1.2.3.4"

    def test_ssh_host_returns_none(self):
        """ssh_host returns None if no IPs available."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test",
        )
        assert inst.ssh_host is None

    def test_str_representation(self):
        """String representation is human-readable."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="my-node",
            gpu_type="H100",
            gpu_count=2,
            state=InstanceState.RUNNING,
        )
        s = str(inst)
        assert "my-node" in s
        assert "lambda" in s
        assert "2x H100" in s
        assert "running" in s

    def test_str_cpu_only(self):
        """String shows CPU for non-GPU instances."""
        inst = ProviderInstance(
            instance_id="test",
            provider=Provider.HETZNER,
            name="cpu-node",
            state=InstanceState.RUNNING,
        )
        s = str(inst)
        assert "CPU" in s


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_healthy_result(self):
        """Can create healthy result."""
        result = HealthCheckResult(
            healthy=True,
            check_type="ssh",
            message="SSH connectivity OK",
            latency_ms=45.2,
        )
        assert result.healthy is True
        assert result.check_type == "ssh"
        assert result.latency_ms == 45.2

    def test_unhealthy_result(self):
        """Can create unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            check_type="p2p",
            message="P2P daemon not responding",
            details={"error": "connection refused"},
        )
        assert result.healthy is False
        assert "error" in result.details

    def test_timestamp_auto_set(self):
        """Timestamp is automatically set."""
        result = HealthCheckResult(
            healthy=True,
            check_type="test",
            message="test",
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_successful_recovery(self):
        """Can create successful recovery result."""
        result = RecoveryResult(
            success=True,
            action="restart_p2p",
            message="P2P daemon restarted successfully",
        )
        assert result.success is True
        assert result.action == "restart_p2p"

    def test_failed_recovery(self):
        """Can create failed recovery result."""
        result = RecoveryResult(
            success=False,
            action="reboot",
            message="Instance reboot failed",
            details={"error_code": 500},
        )
        assert result.success is False
        assert result.details["error_code"] == 500
