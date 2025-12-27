"""Tests for DaemonAdapter classes.

December 2025: Unit tests for daemon_adapters.py which provides wrappers
for integrating various daemons with the DaemonManager lifecycle.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.daemon_adapters import (
    DaemonAdapter,
    DaemonAdapterConfig,
    ConfigurableDaemonAdapter,
    DistillationDaemonAdapter,
    PromotionDaemonAdapter,
    ExternalDriveSyncAdapter,
    VastCpuPipelineAdapter,
    ClusterDataSyncAdapter,
    AutoSyncDaemonAdapter,
    NPZDistributionDaemonAdapter,
    OrphanDetectionDaemonAdapter,
    DataCleanupDaemonAdapter,
    get_daemon_adapter,
    get_available_adapters,
    register_adapter_class,
)
from app.coordination.daemon_types import DaemonType
from app.coordination.orchestrator_registry import OrchestratorRole


class TestDaemonAdapterConfig:
    """Test DaemonAdapterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DaemonAdapterConfig()

        assert config.acquire_role is True
        assert config.role_timeout_seconds == 300.0
        assert config.health_check_interval == 60.0
        assert config.unhealthy_threshold == 3
        assert config.auto_restart is True
        assert config.max_restarts == 5
        assert config.restart_delay_seconds == 5.0
        assert config.poll_interval_seconds == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DaemonAdapterConfig(
            acquire_role=False,
            role_timeout_seconds=60.0,
            health_check_interval=30.0,
            unhealthy_threshold=5,
            auto_restart=False,
            max_restarts=10,
            restart_delay_seconds=10.0,
            poll_interval_seconds=15.0,
        )

        assert config.acquire_role is False
        assert config.role_timeout_seconds == 60.0
        assert config.health_check_interval == 30.0
        assert config.unhealthy_threshold == 5
        assert config.auto_restart is False
        assert config.max_restarts == 10
        assert config.restart_delay_seconds == 10.0
        assert config.poll_interval_seconds == 15.0


class TestDaemonAdapterBase:
    """Test DaemonAdapter base class functionality."""

    def test_initial_state(self):
        """Test adapter initializes with correct state."""
        adapter = DistillationDaemonAdapter()

        assert adapter._running is False
        assert adapter._healthy is True
        assert adapter._unhealthy_count == 0
        assert adapter._start_time == 0.0
        assert adapter._daemon_instance is None

    def test_config_defaults(self):
        """Test adapter uses default config when none provided."""
        adapter = DistillationDaemonAdapter()

        assert adapter.config.acquire_role is True
        assert adapter.config.health_check_interval == 60.0

    def test_config_custom(self):
        """Test adapter uses custom config when provided."""
        config = DaemonAdapterConfig(health_check_interval=30.0)
        adapter = DistillationDaemonAdapter(config)

        assert adapter.config.health_check_interval == 30.0

    def test_get_status_not_running(self):
        """Test get_status when adapter is not running."""
        adapter = DistillationDaemonAdapter()
        status = adapter.get_status()

        assert status["daemon_type"] == "distillation"
        assert status["running"] is False
        assert status["healthy"] is True
        assert status["uptime_seconds"] == 0
        assert status["has_instance"] is False

    def test_health_check_not_running(self):
        """Test health_check when adapter is not running."""
        adapter = DistillationDaemonAdapter()
        result = adapter.health_check()

        # Stopped adapters are considered healthy (not unhealthy)
        assert result.healthy is False  # Not running means not healthy
        assert "stopped" in result.status.value.lower() or "error" in result.status.value.lower()


class TestDistillationDaemonAdapter:
    """Test DistillationDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = DistillationDaemonAdapter()
        assert adapter.daemon_type == DaemonType.DISTILLATION

    def test_role(self):
        """Test role property."""
        adapter = DistillationDaemonAdapter()
        assert adapter.role == OrchestratorRole.DISTILLATION_LEADER

    def test_depends_on(self):
        """Test depends_on property."""
        adapter = DistillationDaemonAdapter()
        assert adapter.depends_on == []

    @pytest.mark.asyncio
    async def test_create_daemon_import_error(self):
        """Test _create_daemon handles ImportError gracefully."""
        adapter = DistillationDaemonAdapter()

        with patch.dict("sys.modules", {"app.training.distillation_daemon": None}):
            with patch(
                "app.coordination.daemon_adapters.DistillationDaemonAdapter._create_daemon",
                new_callable=AsyncMock,
                return_value=None
            ):
                daemon = await adapter._create_daemon()
                assert daemon is None


class TestPromotionDaemonAdapter:
    """Test PromotionDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = PromotionDaemonAdapter()
        assert adapter.daemon_type == DaemonType.UNIFIED_PROMOTION

    def test_role(self):
        """Test role property."""
        adapter = PromotionDaemonAdapter()
        assert adapter.role == OrchestratorRole.PROMOTION_LEADER


class TestExternalDriveSyncAdapter:
    """Test ExternalDriveSyncAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = ExternalDriveSyncAdapter()
        assert adapter.daemon_type == DaemonType.EXTERNAL_DRIVE_SYNC

    def test_role(self):
        """Test role property."""
        adapter = ExternalDriveSyncAdapter()
        assert adapter.role == OrchestratorRole.EXTERNAL_SYNC_LEADER


class TestVastCpuPipelineAdapter:
    """Test VastCpuPipelineAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = VastCpuPipelineAdapter()
        assert adapter.daemon_type == DaemonType.VAST_CPU_PIPELINE

    def test_role(self):
        """Test role property."""
        adapter = VastCpuPipelineAdapter()
        assert adapter.role == OrchestratorRole.VAST_PIPELINE_LEADER


class TestClusterDataSyncAdapter:
    """Test ClusterDataSyncAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = ClusterDataSyncAdapter()
        assert adapter.daemon_type == DaemonType.CLUSTER_DATA_SYNC

    def test_role(self):
        """Test role property."""
        adapter = ClusterDataSyncAdapter()
        assert adapter.role == OrchestratorRole.CLUSTER_DATA_SYNC_LEADER

    @pytest.mark.asyncio
    async def test_health_check_no_instance(self):
        """Test _health_check when no daemon instance exists."""
        adapter = ClusterDataSyncAdapter()
        result = await adapter._health_check()
        assert result is False


class TestAutoSyncDaemonAdapter:
    """Test AutoSyncDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = AutoSyncDaemonAdapter()
        assert adapter.daemon_type == DaemonType.AUTO_SYNC

    def test_role_is_none(self):
        """Test role property returns None (no exclusive role)."""
        adapter = AutoSyncDaemonAdapter()
        assert adapter.role is None

    @pytest.mark.asyncio
    async def test_health_check_no_instance(self):
        """Test _health_check when no daemon instance exists."""
        adapter = AutoSyncDaemonAdapter()
        result = await adapter._health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_running_instance(self):
        """Test _health_check when daemon is running."""
        adapter = AutoSyncDaemonAdapter()

        # Mock daemon instance with is_running method
        mock_daemon = MagicMock()
        mock_daemon.is_running.return_value = True
        adapter._daemon_instance = mock_daemon

        result = await adapter._health_check()
        assert result is True


class TestNPZDistributionDaemonAdapter:
    """Test NPZDistributionDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = NPZDistributionDaemonAdapter()
        assert adapter.daemon_type == DaemonType.NPZ_DISTRIBUTION

    def test_role_is_none(self):
        """Test role property returns None (deprecated adapter)."""
        adapter = NPZDistributionDaemonAdapter()
        # No exclusive role since this is deprecated and delegates to unified daemon
        assert adapter.role is None

    def test_depends_on(self):
        """Test depends_on property."""
        adapter = NPZDistributionDaemonAdapter()
        assert adapter.depends_on == []


class TestOrphanDetectionDaemonAdapter:
    """Test OrphanDetectionDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = OrphanDetectionDaemonAdapter()
        assert adapter.daemon_type == DaemonType.ORPHAN_DETECTION

    def test_role_is_none(self):
        """Test role property returns None (runs on all nodes)."""
        adapter = OrphanDetectionDaemonAdapter()
        assert adapter.role is None

    def test_depends_on(self):
        """Test depends_on property."""
        adapter = OrphanDetectionDaemonAdapter()
        assert adapter.depends_on == []


class TestDataCleanupDaemonAdapter:
    """Test DataCleanupDaemonAdapter specifically."""

    def test_daemon_type(self):
        """Test daemon_type property."""
        adapter = DataCleanupDaemonAdapter()
        assert adapter.daemon_type == DaemonType.DATA_CLEANUP

    def test_role_is_none(self):
        """Test role property returns None (runs on all nodes)."""
        adapter = DataCleanupDaemonAdapter()
        assert adapter.role is None

    def test_depends_on(self):
        """Test depends_on property."""
        adapter = DataCleanupDaemonAdapter()
        assert adapter.depends_on == []


class TestAdapterRegistry:
    """Test adapter registry functions."""

    def test_get_available_adapters(self):
        """Test get_available_adapters returns all registered types."""
        adapters = get_available_adapters()

        assert DaemonType.DISTILLATION in adapters
        assert DaemonType.UNIFIED_PROMOTION in adapters
        assert DaemonType.EXTERNAL_DRIVE_SYNC in adapters
        assert DaemonType.VAST_CPU_PIPELINE in adapters
        assert DaemonType.CLUSTER_DATA_SYNC in adapters
        assert DaemonType.AUTO_SYNC in adapters
        assert DaemonType.NPZ_DISTRIBUTION in adapters
        assert DaemonType.ORPHAN_DETECTION in adapters
        assert DaemonType.DATA_CLEANUP in adapters
        assert len(adapters) == 9

    def test_get_daemon_adapter_distillation(self):
        """Test get_daemon_adapter returns correct adapter for DISTILLATION.

        December 2025: DISTILLATION now uses ConfigurableDaemonAdapter
        via ADAPTER_SPECS instead of legacy DistillationDaemonAdapter.
        """
        adapter = get_daemon_adapter(DaemonType.DISTILLATION)

        assert adapter is not None
        # Note: Now uses ConfigurableDaemonAdapter for ADAPTER_SPECS entries
        assert isinstance(adapter, (ConfigurableDaemonAdapter, DistillationDaemonAdapter))
        assert adapter.daemon_type == DaemonType.DISTILLATION

    def test_get_daemon_adapter_with_config(self):
        """Test get_daemon_adapter passes config to adapter."""
        config = DaemonAdapterConfig(health_check_interval=15.0)
        adapter = get_daemon_adapter(DaemonType.AUTO_SYNC, config)

        assert adapter is not None
        assert adapter.config.health_check_interval == 15.0

    def test_get_daemon_adapter_unknown_type(self):
        """Test get_daemon_adapter returns None for unknown types."""
        # Use a daemon type that doesn't have an adapter
        adapter = get_daemon_adapter(DaemonType.EVENT_ROUTER)

        assert adapter is None

    def test_register_adapter_class(self):
        """Test register_adapter_class adds custom adapter."""
        # Create a mock adapter class
        class MockAdapter(DaemonAdapter):
            @property
            def daemon_type(self) -> DaemonType:
                return DaemonType.MAINTENANCE

            async def _create_daemon(self):
                return MagicMock()

            async def _run_daemon(self, daemon):
                pass

        # Register the mock adapter
        register_adapter_class(DaemonType.MAINTENANCE, MockAdapter)

        # Verify it's registered
        adapter = get_daemon_adapter(DaemonType.MAINTENANCE)
        assert adapter is not None
        assert isinstance(adapter, MockAdapter)

        # Cleanup: remove from registry to avoid affecting other tests
        from app.coordination.daemon_adapters import _ADAPTER_CLASSES
        del _ADAPTER_CLASSES[DaemonType.MAINTENANCE]


class TestAdapterHealthCheck:
    """Test adapter health check functionality."""

    def test_health_check_result_structure(self):
        """Test health_check returns proper HealthCheckResult structure."""
        adapter = DistillationDaemonAdapter()
        result = adapter.health_check()

        # Check result has required fields
        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")

    def test_health_check_running_healthy(self):
        """Test health_check when adapter is running and healthy."""
        adapter = DistillationDaemonAdapter()
        adapter._running = True
        adapter._healthy = True
        adapter._daemon_instance = MagicMock()

        result = adapter.health_check()

        assert result.healthy is True
        assert "running" in result.status.value.lower()

    def test_health_check_running_unhealthy(self):
        """Test health_check when adapter is running but unhealthy."""
        adapter = DistillationDaemonAdapter()
        adapter._running = True
        adapter._healthy = False
        adapter._unhealthy_count = 5
        adapter._daemon_instance = MagicMock()

        result = adapter.health_check()

        assert result.healthy is False
        assert "error" in result.status.value.lower()


class TestAdapterLifecycle:
    """Test adapter lifecycle (run, stop, etc.)."""

    @pytest.mark.asyncio
    async def test_run_without_role_acquisition(self):
        """Test run() when role acquisition is disabled."""
        config = DaemonAdapterConfig(acquire_role=False)
        adapter = AutoSyncDaemonAdapter(config)

        # Mock the daemon creation and run
        mock_daemon = MagicMock()
        mock_daemon.start = AsyncMock()
        mock_daemon.is_running = MagicMock(side_effect=[True, False])

        with patch.object(adapter, "_create_daemon", new_callable=AsyncMock, return_value=mock_daemon):
            # Run should complete without trying to acquire a role
            task = asyncio.create_task(adapter.run())
            await asyncio.sleep(0.1)  # Let it start
            adapter._running = False  # Stop the loop
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_run_daemon_creation_failure(self):
        """Test run() when daemon creation fails."""
        config = DaemonAdapterConfig(acquire_role=False)
        adapter = AutoSyncDaemonAdapter(config)

        with patch.object(adapter, "_create_daemon", new_callable=AsyncMock, return_value=None):
            await adapter.run()

            # Should set running to False after failure
            assert adapter._running is False
