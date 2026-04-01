"""Tests for UnifiedDataPlaneDaemon.

December 28, 2025 - Phase 4 of Unified Data Plane implementation tests.

This module tests the consolidated data synchronization daemon that replaces
fragmented sync infrastructure (~4,514 LOC consolidated).
"""

from __future__ import annotations

import asyncio
import pytest
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_data_catalog():
    """Create mock DataCatalog."""
    catalog = MagicMock()
    catalog.get_entries = MagicMock(return_value=[])
    catalog.get_entry = MagicMock(return_value=None)
    catalog.register = MagicMock()
    catalog.update = MagicMock()
    catalog.get_stats = MagicMock(return_value={"entries": 0})
    return catalog


@pytest.fixture
def mock_sync_planner():
    """Create mock SyncPlanner."""
    planner = MagicMock()
    planner.start = AsyncMock()
    planner.stop = AsyncMock()
    planner.create_plan = MagicMock(return_value=None)
    planner.get_pending_syncs = MagicMock(return_value=[])
    return planner


@pytest.fixture
def mock_transport_manager():
    """Create mock TransportManager."""
    transport = MagicMock()
    transport.transfer = AsyncMock(return_value=True)
    transport.get_available_transports = MagicMock(return_value=["ssh", "http"])
    transport.get_stats = MagicMock(return_value={"transfers": 0})
    return transport


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


# =============================================================================
# DataPlaneConfig Tests
# =============================================================================


class TestDataPlaneConfig:
    """Tests for DataPlaneConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.unified_data_plane_daemon import DataPlaneConfig

        config = DataPlaneConfig()

        assert config.catalog_refresh_interval == 60.0
        assert config.replication_check_interval == 300.0
        assert config.s3_backup_interval == 3600.0
        assert config.manifest_broadcast_interval == 120.0
        assert config.min_replication_factor == 3
        assert config.target_replication_factor == 5
        assert config.max_concurrent_syncs == 1  # Feb 2026: reduced to prevent OOM
        assert config.s3_enabled is False
        assert config.owc_enabled is False

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.coordination.unified_data_plane_daemon import DataPlaneConfig

        config = DataPlaneConfig(
            catalog_refresh_interval=30.0,
            min_replication_factor=2,
            s3_enabled=True,
            s3_bucket="test-bucket",
        )

        assert config.catalog_refresh_interval == 30.0
        assert config.min_replication_factor == 2
        assert config.s3_enabled is True
        assert config.s3_bucket == "test-bucket"

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        from app.coordination.unified_data_plane_daemon import DataPlaneConfig

        with patch.dict("os.environ", {}, clear=True):
            config = DataPlaneConfig.from_env()

        assert config.catalog_refresh_interval == 60.0
        assert config.s3_enabled is False
        assert config.owc_enabled is False

    def test_from_env_with_values(self):
        """Test from_env with environment variables set."""
        from app.coordination.unified_data_plane_daemon import DataPlaneConfig

        env = {
            "RINGRIFT_CATALOG_REFRESH_INTERVAL": "30",
            "RINGRIFT_REPLICATION_CHECK_INTERVAL": "120",
            "RINGRIFT_S3_BACKUP_ENABLED": "true",
            "RINGRIFT_S3_BUCKET": "my-bucket",
            "RINGRIFT_MIN_REPLICATION": "2",
            "RINGRIFT_OWC_ENABLED": "true",
            "RINGRIFT_OWC_HOST": "my-host",
        }

        with patch.dict("os.environ", env, clear=True):
            config = DataPlaneConfig.from_env()

        assert config.catalog_refresh_interval == 30.0
        assert config.replication_check_interval == 120.0
        assert config.s3_enabled is True
        assert config.s3_bucket == "my-bucket"
        assert config.min_replication_factor == 2
        assert config.owc_enabled is True
        assert config.owc_host == "my-host"


# =============================================================================
# DataPlaneStats Tests
# =============================================================================


class TestDataPlaneStats:
    """Tests for DataPlaneStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        from app.coordination.unified_data_plane_daemon import DataPlaneStats

        stats = DataPlaneStats()

        assert stats.events_received == 0
        assert stats.events_processed == 0
        assert stats.events_failed == 0
        assert stats.syncs_initiated == 0
        assert stats.syncs_completed == 0
        assert stats.syncs_failed == 0
        assert stats.bytes_synced == 0
        assert stats.start_time == 0.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        from app.coordination.unified_data_plane_daemon import DataPlaneStats

        stats = DataPlaneStats(
            events_received=100,
            events_processed=95,
            events_failed=5,
            syncs_initiated=10,
            syncs_completed=8,
            bytes_synced=1024000,
            start_time=time.time() - 3600,
        )

        result = stats.to_dict()

        assert result["events"]["received"] == 100
        assert result["events"]["processed"] == 95
        assert result["events"]["failed"] == 5
        assert result["syncs"]["initiated"] == 10
        assert result["syncs"]["completed"] == 8
        assert result["syncs"]["bytes"] == 1024000
        assert result["uptime_seconds"] >= 3599

    def test_to_dict_with_zero_start_time(self):
        """Test to_dict with zero start time."""
        from app.coordination.unified_data_plane_daemon import DataPlaneStats

        stats = DataPlaneStats(start_time=0.0)
        result = stats.to_dict()

        assert result["uptime_seconds"] == 0

    def test_mutation(self):
        """Test that stats can be mutated."""
        from app.coordination.unified_data_plane_daemon import DataPlaneStats

        stats = DataPlaneStats()
        stats.events_received += 1
        stats.syncs_completed += 1

        assert stats.events_received == 1
        assert stats.syncs_completed == 1


# =============================================================================
# EventBridge Tests
# =============================================================================


class TestEventBridge:
    """Tests for EventBridge class."""

    def test_initialization(self):
        """Test EventBridge initialization."""
        from app.coordination.unified_data_plane_daemon import EventBridge

        handler = MagicMock()
        bridge = EventBridge(handler)

        assert bridge._on_event is handler
        assert bridge._running is False
        assert len(bridge._subscriptions) == 0
        assert len(bridge._subscribed_events) > 0

    def test_subscribed_events_list(self):
        """Test that expected events are in subscription list."""
        from app.coordination.unified_data_plane_daemon import EventBridge

        bridge = EventBridge(MagicMock())

        expected_events = [
            "SELFPLAY_COMPLETE",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "MODEL_PROMOTED",
            "ORPHAN_GAMES_DETECTED",
        ]

        for event in expected_events:
            assert event in bridge._subscribed_events

    @pytest.mark.asyncio
    async def test_start_subscribes_to_events(self, mock_event_bus):
        """Test that start() subscribes to events."""
        from app.coordination.unified_data_plane_daemon import EventBridge

        bridge = EventBridge(MagicMock())

        # EventBridge.start() now uses the router subscribe helper directly.
        with patch(
            "app.coordination.event_router.subscribe",
            side_effect=mock_event_bus.subscribe,
        ):
            await bridge.start()

        assert bridge._running is True
        # Verify subscribe was called for expected events
        assert mock_event_bus.subscribe.call_count > 0

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, mock_event_bus):
        """Test that calling start() twice is safe."""
        from app.coordination.unified_data_plane_daemon import EventBridge

        bridge = EventBridge(MagicMock())
        bridge._running = True

        await bridge.start()

        assert not mock_event_bus.subscribe.called

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self, mock_event_bus):
        """Test that stop() unsubscribes from events."""
        from app.coordination.unified_data_plane_daemon import EventBridge

        bridge = EventBridge(MagicMock())
        bridge._running = True
        bridge._subscriptions = ["SELFPLAY_COMPLETE", "TRAINING_COMPLETED"]

        with patch(
            "app.coordination.event_router.unsubscribe",
            side_effect=mock_event_bus.unsubscribe,
        ):
            await bridge.stop()

        assert bridge._running is False
        assert len(bridge._subscriptions) == 0


# =============================================================================
# UnifiedDataPlaneDaemon Tests
# =============================================================================


class TestUnifiedDataPlaneDaemonInit:
    """Tests for UnifiedDataPlaneDaemon initialization."""

    def test_default_initialization(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test daemon initializes with default config."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        with patch(
            "app.coordination.unified_data_plane_daemon.get_data_catalog",
            return_value=mock_data_catalog,
        ):
            with patch(
                "app.coordination.unified_data_plane_daemon.get_sync_planner",
                return_value=mock_sync_planner,
            ):
                with patch(
                    "app.coordination.unified_data_plane_daemon.get_transport_manager",
                    return_value=mock_transport_manager,
                ):
                    daemon = UnifiedDataPlaneDaemon()

        assert daemon._running is False
        assert daemon._config is not None
        assert daemon._catalog is mock_data_catalog
        assert daemon._planner is mock_sync_planner
        assert daemon._transport is mock_transport_manager

    def test_custom_config_initialization(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test daemon initializes with custom config."""
        from app.coordination.unified_data_plane_daemon import (
            UnifiedDataPlaneDaemon,
            DataPlaneConfig,
        )

        config = DataPlaneConfig(
            catalog_refresh_interval=30.0,
            s3_enabled=True,
        )

        daemon = UnifiedDataPlaneDaemon(
            config=config,
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        assert daemon._config.catalog_refresh_interval == 30.0
        assert daemon._config.s3_enabled is True

    def test_stats_initialized(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that stats are initialized."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        assert daemon._stats.events_received == 0
        assert daemon._stats.syncs_initiated == 0


class TestUnifiedDataPlaneDaemonLifecycle:
    """Tests for UnifiedDataPlaneDaemon lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that start() sets running flag."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        with patch.object(daemon._event_bridge, "start", new_callable=AsyncMock):
            with patch(
                "app.coordination.unified_data_plane_daemon.register_coordinator"
            ):
                await daemon.start()
                await asyncio.sleep(0.01)

        assert daemon._running is True
        assert daemon._start_time > 0

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that calling start() twice is safe."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        await daemon.start()

        mock_sync_planner.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_clears_running(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that stop() clears running flag."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        with patch.object(daemon._event_bridge, "stop", new_callable=AsyncMock):
            with patch(
                "app.coordination.unified_data_plane_daemon.unregister_coordinator"
            ):
                await daemon.stop()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that calling stop() twice is safe."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = False

        await daemon.stop()

        mock_sync_planner.stop.assert_not_called()


class TestUnifiedDataPlaneDaemonHealth:
    """Tests for UnifiedDataPlaneDaemon health check."""

    def test_health_check_when_not_running(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test health check when daemon is not running."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon
        from app.coordination.protocols import CoordinatorStatus

        # Set up mock health checks to return healthy results
        mock_data_catalog.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_sync_planner.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_transport_manager.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = False
        # Status remains INITIALIZING when not started

        result = daemon.health_check()

        # Healthy is False because status != RUNNING
        assert result.healthy is False
        # Details should show running=False
        assert result.details.get("running") is False

    def test_health_check_when_running(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test health check when daemon is running."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon
        from app.coordination.protocols import CoordinatorStatus

        # Set up mock health checks to return healthy results
        mock_data_catalog.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_sync_planner.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_transport_manager.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True
        daemon._status = CoordinatorStatus.RUNNING  # Must set status to RUNNING
        daemon._start_time = time.time() - 60
        daemon._stats.syncs_completed = 5
        daemon._stats.events_processed = 10

        result = daemon.health_check()

        assert result.healthy is True

    def test_health_check_details(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test health check includes proper details."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon
        from app.coordination.protocols import CoordinatorStatus

        # Set up mock health checks
        mock_data_catalog.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_sync_planner.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_transport_manager.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True
        daemon._status = CoordinatorStatus.RUNNING
        daemon._start_time = time.time()
        daemon._stats.syncs_completed = 3
        daemon._stats.syncs_failed = 1
        daemon._stats.events_processed = 10

        result = daemon.health_check()

        assert result.healthy is True
        assert hasattr(result, "details")
        assert isinstance(result.details, dict)
        assert "running" in result.details
        assert "uptime" in result.details
        assert "stats" in result.details


class TestUnifiedDataPlaneDaemonSingleton:
    """Tests for singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_data_plane_daemon(self):
        """Test get_data_plane_daemon returns singleton."""
        from app.coordination.unified_data_plane_daemon import (
            get_data_plane_daemon,
            reset_data_plane_daemon,
        )

        # Reset requires an event loop because it creates a task
        reset_data_plane_daemon()
        await asyncio.sleep(0.01)  # Allow task to be scheduled

        with patch(
            "app.coordination.unified_data_plane_daemon.get_data_catalog"
        ) as mock_cat:
            with patch(
                "app.coordination.unified_data_plane_daemon.get_sync_planner"
            ) as mock_plan:
                with patch(
                    "app.coordination.unified_data_plane_daemon.get_transport_manager"
                ) as mock_trans:
                    mock_cat.return_value = MagicMock()
                    mock_plan.return_value = MagicMock()
                    mock_trans.return_value = MagicMock()

                    daemon1 = get_data_plane_daemon()
                    daemon2 = get_data_plane_daemon()

        assert daemon1 is daemon2

        reset_data_plane_daemon()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_reset_data_plane_daemon(self):
        """Test reset_data_plane_daemon clears singleton."""
        from app.coordination.unified_data_plane_daemon import (
            get_data_plane_daemon,
            reset_data_plane_daemon,
        )

        # Reset requires an event loop because it creates a task
        reset_data_plane_daemon()
        await asyncio.sleep(0.01)

        with patch(
            "app.coordination.unified_data_plane_daemon.get_data_catalog"
        ) as mock_cat:
            with patch(
                "app.coordination.unified_data_plane_daemon.get_sync_planner"
            ) as mock_plan:
                with patch(
                    "app.coordination.unified_data_plane_daemon.get_transport_manager"
                ) as mock_trans:
                    mock_cat.return_value = MagicMock()
                    mock_plan.return_value = MagicMock()
                    mock_trans.return_value = MagicMock()

                    daemon1 = get_data_plane_daemon()
                    reset_data_plane_daemon()
                    await asyncio.sleep(0.01)
                    daemon2 = get_data_plane_daemon()

        assert daemon1 is not daemon2

        reset_data_plane_daemon()
        await asyncio.sleep(0.01)


class TestUnifiedDataPlaneDaemonMethods:
    """Tests for UnifiedDataPlaneDaemon utility methods."""

    def test_get_status(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test get_status returns proper structure."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True
        daemon._start_time = time.time()

        status = daemon.get_status()

        assert status is not None
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_trigger_priority_sync(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test trigger_priority_sync method."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        # Mock catalog to return entries
        mock_entry = MagicMock()
        mock_entry.locations = ["vast-12345"]
        mock_data_catalog.get_by_config = MagicMock(return_value=[mock_entry])

        # Mock submit_plan on the planner
        mock_sync_planner.submit_plan = AsyncMock()

        # Mock cluster config to return target nodes
        # The import happens inside trigger_priority_sync from app.config.cluster_config
        mock_node = MagicMock()
        mock_node.is_gpu_node = True
        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value={"nebius-h100": mock_node, "runpod-a100": mock_node},
        ):
            result = await daemon.trigger_priority_sync(
                config_key="hex8_2p",
                source_node="vast-12345",
            )

        assert result is True
        # Verify submit_plan was called
        assert mock_sync_planner.submit_plan.called


class TestUnifiedDataPlaneDaemonEventHandling:
    """Tests for event handling."""

    def test_on_event_increments_counter(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that _on_event increments event counter."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        initial_count = daemon._stats.events_received
        daemon._on_event("SELFPLAY_COMPLETE", {"config_key": "hex8_2p"})

        assert daemon._stats.events_received == initial_count + 1

    def test_on_event_handles_unknown_event(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that _on_event handles unknown events gracefully."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        daemon._on_event("UNKNOWN_EVENT_TYPE", {"data": "value"})

        assert daemon._stats.events_received > 0


class TestUnifiedDataPlaneDaemonEventHandlingAdvanced:
    """Advanced tests for event handling."""

    def test_on_event_processes_selfplay_complete(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test handling of SELFPLAY_COMPLETE event."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_sync_planner.plan_for_event = MagicMock(return_value=[])

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        daemon._on_event("SELFPLAY_COMPLETE", {"config_key": "hex8_2p", "games": 100})

        assert daemon._stats.events_received == 1
        assert daemon._stats.events_processed == 1
        mock_sync_planner.plan_for_event.assert_called_once()

    def test_on_event_processes_training_completed(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test handling of TRAINING_COMPLETED event."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_sync_planner.plan_for_event = MagicMock(return_value=[])

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        daemon._on_event("TRAINING_COMPLETED", {"config_key": "hex8_2p", "model_path": "/path/to/model.pth"})

        assert daemon._stats.events_received == 1
        mock_sync_planner.plan_for_event.assert_called_with(
            "TRAINING_COMPLETED", {"config_key": "hex8_2p", "model_path": "/path/to/model.pth"}
        )

    def test_on_event_increments_failed_on_error(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test that _on_event increments failed counter on error."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_sync_planner.plan_for_event = MagicMock(side_effect=Exception("Plan failed"))

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        daemon._on_event("SELFPLAY_COMPLETE", {"config_key": "hex8_2p"})

        assert daemon._stats.events_received == 1
        assert daemon._stats.events_failed == 1

    def test_on_event_with_empty_payload(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test handling event with empty payload."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_sync_planner.plan_for_event = MagicMock(return_value=[])

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )

        daemon._on_event("SYNC_REQUEST", {})

        assert daemon._stats.events_received == 1


class TestUnifiedDataPlaneDaemonHealthAdvanced:
    """Advanced health check tests."""

    def test_health_check_with_unhealthy_catalog(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test health check when catalog is unhealthy."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon
        from app.coordination.protocols import CoordinatorStatus

        mock_data_catalog.health_check = MagicMock(
            return_value=MagicMock(healthy=False)
        )
        mock_sync_planner.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_transport_manager.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True
        daemon._status = CoordinatorStatus.RUNNING

        result = daemon.health_check()

        # Should be degraded due to unhealthy catalog
        assert result.healthy is False
        assert "catalog" in result.message

    def test_health_check_with_unhealthy_planner(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test health check when planner is unhealthy."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon
        from app.coordination.protocols import CoordinatorStatus

        mock_data_catalog.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )
        mock_sync_planner.health_check = MagicMock(
            return_value=MagicMock(healthy=False)
        )
        mock_transport_manager.health_check = MagicMock(
            return_value=MagicMock(healthy=True)
        )

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True
        daemon._status = CoordinatorStatus.RUNNING

        result = daemon.health_check()

        assert result.healthy is False
        assert "planner" in result.message


class TestUnifiedDataPlaneDaemonTriggerSync:
    """Tests for trigger_priority_sync edge cases."""

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_no_entries(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test trigger_priority_sync returns False when no entries."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_data_catalog.get_by_config = MagicMock(return_value=[])

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        result = await daemon.trigger_priority_sync(
            config_key="nonexistent_config",
            source_node="vast-12345",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_no_targets(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test trigger_priority_sync returns False when no target nodes."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_entry = MagicMock()
        mock_entry.locations = ["vast-12345"]
        mock_data_catalog.get_by_config = MagicMock(return_value=[mock_entry])

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        with patch(
            "app.config.cluster_config.get_cluster_nodes",
            return_value={},
        ):
            result = await daemon.trigger_priority_sync(
                config_key="hex8_2p",
                source_node="vast-12345",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_with_explicit_targets(
        self, mock_data_catalog, mock_sync_planner, mock_transport_manager
    ):
        """Test trigger_priority_sync with explicit target nodes."""
        from app.coordination.unified_data_plane_daemon import UnifiedDataPlaneDaemon

        mock_entry = MagicMock()
        mock_entry.locations = ["vast-12345"]
        mock_data_catalog.get_by_config = MagicMock(return_value=[mock_entry])
        mock_sync_planner.submit_plan = AsyncMock()

        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_data_catalog,
            planner=mock_sync_planner,
            transport=mock_transport_manager,
        )
        daemon._running = True

        result = await daemon.trigger_priority_sync(
            config_key="hex8_2p",
            source_node="vast-12345",
            target_nodes=["nebius-h100", "runpod-a100"],
        )

        assert result is True
        mock_sync_planner.submit_plan.assert_called_once()


class TestDaemonRegistryIntegration:
    """Tests for daemon registry integration."""

    def test_daemon_type_exists(self):
        """Test that UNIFIED_DATA_PLANE DaemonType exists."""
        from app.coordination.daemon_types import DaemonType

        assert hasattr(DaemonType, "UNIFIED_DATA_PLANE")
        assert DaemonType.UNIFIED_DATA_PLANE.value == "unified_data_plane"

    def test_registry_entry_exists(self):
        """Test that registry entry exists for UNIFIED_DATA_PLANE."""
        from app.coordination.daemon_types import DaemonType
        from app.coordination.daemon_registry import DAEMON_REGISTRY

        assert DaemonType.UNIFIED_DATA_PLANE in DAEMON_REGISTRY

        spec = DAEMON_REGISTRY[DaemonType.UNIFIED_DATA_PLANE]
        assert spec.runner_name == "create_unified_data_plane"
        assert spec.category == "sync"
        assert DaemonType.EVENT_ROUTER in spec.depends_on

    def test_runner_function_exists(self):
        """Test that runner function exists."""
        from app.coordination.daemon_types import DaemonType
        from app.coordination.daemon_runners import get_runner

        runner = get_runner(DaemonType.UNIFIED_DATA_PLANE)
        assert runner is not None
        assert callable(runner)

    def test_registry_validates(self):
        """Test that registry validates without errors."""
        from app.coordination.daemon_registry import validate_registry

        errors = validate_registry()

        our_errors = [e for e in errors if "UNIFIED_DATA_PLANE" in e]
        assert len(our_errors) == 0, f"Validation errors: {our_errors}"
