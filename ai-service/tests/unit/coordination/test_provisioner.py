"""Unit tests for Provisioner daemon.

Tests the auto-provisioning infrastructure for cluster capacity management.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.provisioner import (
    ClusterCapacity,
    ProvisionerConfig,
    ProvisionResult,
    Provisioner,
    get_provisioner,
    reset_provisioner,
)


class TestProvisionResult:
    """Tests for ProvisionResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = ProvisionResult(success=True, instances_created=0)
        assert result.success is True
        assert result.instances_created == 0
        assert result.instance_ids == []
        assert result.error is None
        assert result.provider == ""
        assert result.gpu_type == ""
        assert result.cost_per_hour == 0.0
        assert isinstance(result.timestamp, datetime)

    def test_successful_provision(self):
        """Test successful provisioning result."""
        result = ProvisionResult(
            success=True,
            instances_created=2,
            instance_ids=["inst-1", "inst-2"],
            provider="lambda",
            gpu_type="H100_80GB",
            cost_per_hour=5.50,
        )
        assert result.success
        assert result.instances_created == 2
        assert len(result.instance_ids) == 2

    def test_failed_provision(self):
        """Test failed provisioning result."""
        result = ProvisionResult(
            success=False,
            instances_created=0,
            error="Quota exceeded",
            provider="vast",
        )
        assert not result.success
        assert result.error == "Quota exceeded"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ProvisionResult(
            success=True,
            instances_created=1,
            instance_ids=["inst-1"],
            provider="lambda",
            gpu_type="A100_80GB",
            cost_per_hour=3.25,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["instances_created"] == 1
        assert d["instance_ids"] == ["inst-1"]
        assert d["provider"] == "lambda"
        assert d["gpu_type"] == "A100_80GB"
        assert d["cost_per_hour"] == 3.25
        assert "timestamp" in d


class TestProvisionerConfig:
    """Tests for ProvisionerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProvisionerConfig()
        assert config.cycle_interval_seconds == 300.0
        assert config.min_gpu_capacity == 4
        assert config.target_gpu_capacity == 10
        assert config.max_provision_per_cycle == 2
        assert config.preferred_provider == "lambda"
        assert "vast" in config.fallback_providers
        assert "runpod" in config.fallback_providers
        assert "GH200_96GB" in config.preferred_gpu_types
        assert config.dry_run is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProvisionerConfig(
            min_gpu_capacity=2,
            max_provision_per_cycle=5,
            preferred_provider="vast",
            dry_run=True,
        )
        assert config.min_gpu_capacity == 2
        assert config.max_provision_per_cycle == 5
        assert config.preferred_provider == "vast"
        assert config.dry_run is True


class TestClusterCapacity:
    """Tests for ClusterCapacity dataclass."""

    def test_default_values(self):
        """Test default capacity values."""
        capacity = ClusterCapacity()
        assert capacity.total_gpu_nodes == 0
        assert capacity.active_gpu_nodes == 0
        assert capacity.healthy_gpu_nodes == 0
        assert capacity.total_gpus == 0
        assert capacity.gpu_utilization == 0.0
        assert capacity.providers == {}

    def test_with_values(self):
        """Test capacity with actual values."""
        capacity = ClusterCapacity(
            total_gpu_nodes=10,
            active_gpu_nodes=8,
            healthy_gpu_nodes=6,
            total_gpus=10,
            gpu_utilization=0.75,
            providers={"lambda": 4, "vast": 4},
        )
        assert capacity.total_gpu_nodes == 10
        assert capacity.healthy_gpu_nodes == 6


class TestProvisioner:
    """Tests for Provisioner daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_provisioner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_provisioner()

    def test_init_default_config(self):
        """Test initialization with default config."""
        provisioner = Provisioner()
        assert provisioner.config is not None
        assert provisioner.config.min_gpu_capacity == 4
        assert provisioner._provision_history == []
        assert provisioner._pending_provisions == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ProvisionerConfig(
            min_gpu_capacity=2,
            dry_run=True,
        )
        provisioner = Provisioner(config)
        assert provisioner.config.min_gpu_capacity == 2
        assert provisioner.config.dry_run is True

    def test_event_subscriptions(self):
        """Test event subscription configuration."""
        provisioner = Provisioner()
        subs = provisioner._get_event_subscriptions()
        assert "CAPACITY_LOW" in subs
        assert "NODE_FAILED_PERMANENTLY" in subs

    @pytest.mark.asyncio
    async def test_on_capacity_low(self):
        """Test CAPACITY_LOW event handler."""
        provisioner = Provisioner()
        event = {"payload": {"needed_gpus": 3}}
        await provisioner._on_capacity_low(event)
        assert provisioner._pending_provisions == 3

    @pytest.mark.asyncio
    async def test_on_capacity_low_max(self):
        """Test CAPACITY_LOW takes max of pending."""
        provisioner = Provisioner()
        provisioner._pending_provisions = 5
        event = {"payload": {"needed_gpus": 3}}
        await provisioner._on_capacity_low(event)
        assert provisioner._pending_provisions == 5  # Keeps max

    @pytest.mark.asyncio
    async def test_on_node_failed(self):
        """Test NODE_FAILED_PERMANENTLY event handler."""
        provisioner = Provisioner()
        event = {"payload": {"node_id": "vast-12345"}}
        # Should just log, not trigger immediate provisioning
        await provisioner._on_node_failed(event)
        assert provisioner._pending_provisions == 0

    @pytest.mark.asyncio
    async def test_run_cycle_capacity_ok(self):
        """Test cycle when capacity is sufficient."""
        config = ProvisionerConfig(min_gpu_capacity=2)
        provisioner = Provisioner(config)

        # Mock capacity check
        with patch.object(
            provisioner,
            "_get_cluster_capacity",
            new_callable=AsyncMock,
            return_value=ClusterCapacity(healthy_gpu_nodes=5),
        ):
            await provisioner._run_cycle()
            # Should not attempt provisioning
            assert len(provisioner._provision_history) == 0

    @pytest.mark.asyncio
    async def test_run_cycle_capacity_low_dry_run(self):
        """Test cycle when capacity is low (dry run)."""
        config = ProvisionerConfig(min_gpu_capacity=4, dry_run=True)
        provisioner = Provisioner(config)

        with patch.object(
            provisioner,
            "_get_cluster_capacity",
            new_callable=AsyncMock,
            return_value=ClusterCapacity(healthy_gpu_nodes=2),
        ):
            with patch.object(
                provisioner,
                "_check_budget",
                new_callable=AsyncMock,
                return_value=True,
            ):
                await provisioner._run_cycle()
                # Should record dry run attempt
                assert len(provisioner._provision_history) == 1
                assert provisioner._provision_history[0].error == "dry_run"

    @pytest.mark.asyncio
    async def test_run_cycle_budget_exceeded(self):
        """Test cycle when budget is exceeded."""
        config = ProvisionerConfig(min_gpu_capacity=4)
        provisioner = Provisioner(config)

        with patch.object(
            provisioner,
            "_get_cluster_capacity",
            new_callable=AsyncMock,
            return_value=ClusterCapacity(healthy_gpu_nodes=2),
        ):
            with patch.object(
                provisioner,
                "_check_budget",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch.object(
                    provisioner,
                    "_emit_budget_exceeded",
                    new_callable=AsyncMock,
                ) as mock_emit:
                    await provisioner._run_cycle()
                    mock_emit.assert_called_once()
                    # Should not provision
                    assert len(provisioner._provision_history) == 0

    @pytest.mark.asyncio
    async def test_provision_nodes_dry_run(self):
        """Test dry run provisioning."""
        config = ProvisionerConfig(dry_run=True)
        provisioner = Provisioner(config)

        result = await provisioner._provision_nodes(2)
        assert result.success is True
        assert result.instances_created == 0
        assert result.error == "dry_run"

    @pytest.mark.asyncio
    async def test_provision_from_provider_not_configured(self):
        """Test provisioning from unconfigured provider."""
        provisioner = Provisioner()

        with patch(
            "app.coordination.availability.provisioner.get_provider",
            return_value=None,
        ):
            result = await provisioner._provision_from_provider("fake_provider", 1)
            assert result.success is False
            assert "not configured" in result.error

    def test_get_gpu_vram(self):
        """Test GPU VRAM mapping."""
        provisioner = Provisioner()

        # Create mock GPU types
        class MockGPU:
            def __init__(self, value):
                self.value = value

        assert provisioner._get_gpu_vram(MockGPU("H100_80GB")) == 80
        assert provisioner._get_gpu_vram(MockGPU("A100_80GB")) == 80
        assert provisioner._get_gpu_vram(MockGPU("RTX_4090")) == 24
        assert provisioner._get_gpu_vram(MockGPU("GH200_96GB")) == 96
        assert provisioner._get_gpu_vram(None) == 0
        assert provisioner._get_gpu_vram(MockGPU("UNKNOWN")) == 24  # Default

    def test_generate_node_name(self):
        """Test node name generation."""
        config = ProvisionerConfig(preferred_provider="lambda")
        provisioner = Provisioner(config)

        class MockInstance:
            def __init__(self, inst_id):
                self.id = inst_id

        name = provisioner._generate_node_name(MockInstance("inst123456789"))
        assert name.startswith("lambda-")
        assert "inst1234" in name  # Truncated to 8 chars

        name_short = provisioner._generate_node_name(MockInstance("abc"))
        assert "abc" in name_short

    def test_health_check_healthy(self):
        """Test health check when healthy."""
        provisioner = Provisioner()
        # Add some successful provisions
        for _ in range(3):
            provisioner._provision_history.append(
                ProvisionResult(success=True, instances_created=1)
            )

        health = provisioner.health_check()
        assert health["healthy"] is True
        assert "pending" in health["message"].lower()

    def test_health_check_unhealthy(self):
        """Test health check when unhealthy."""
        provisioner = Provisioner()
        # Add many failures
        for _ in range(6):
            provisioner._provision_history.append(
                ProvisionResult(success=False, instances_created=0, error="Failed")
            )

        health = provisioner.health_check()
        assert health["healthy"] is False
        assert health["details"]["recent_failures"] == 6

    def test_get_provision_history(self):
        """Test getting provision history."""
        provisioner = Provisioner()

        # Add some history
        for i in range(25):
            provisioner._provision_history.append(
                ProvisionResult(success=True, instances_created=i)
            )

        # Default limit
        history = provisioner.get_provision_history()
        assert len(history) == 20

        # Custom limit
        history = provisioner.get_provision_history(limit=5)
        assert len(history) == 5


class TestProvisionerSingleton:
    """Tests for Provisioner singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_provisioner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_provisioner()

    def test_get_provisioner(self):
        """Test get_provisioner returns singleton."""
        p1 = get_provisioner()
        p2 = get_provisioner()
        assert p1 is p2

    def test_reset_provisioner(self):
        """Test reset_provisioner clears singleton."""
        p1 = get_provisioner()
        reset_provisioner()
        p2 = get_provisioner()
        assert p1 is not p2


class TestProvisionerIntegration:
    """Integration-style tests for Provisioner."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_provisioner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_provisioner()

    @pytest.mark.asyncio
    async def test_full_provisioning_cycle(self):
        """Test complete provisioning cycle."""
        config = ProvisionerConfig(
            min_gpu_capacity=4,
            max_provision_per_cycle=2,
            dry_run=True,
        )
        provisioner = Provisioner(config)

        # Simulate capacity low
        await provisioner._on_capacity_low({"payload": {"needed_gpus": 3}})
        assert provisioner._pending_provisions == 3

        # Mock capacity check
        with patch.object(
            provisioner,
            "_get_cluster_capacity",
            new_callable=AsyncMock,
            return_value=ClusterCapacity(healthy_gpu_nodes=2),
        ):
            with patch.object(
                provisioner,
                "_check_budget",
                new_callable=AsyncMock,
                return_value=True,
            ):
                await provisioner._run_cycle()

        # Should record dry run
        assert len(provisioner._provision_history) == 1
        result = provisioner._provision_history[0]
        assert result.error == "dry_run"
