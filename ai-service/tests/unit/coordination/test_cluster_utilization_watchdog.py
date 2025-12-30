"""Tests for cluster_utilization_watchdog.py - Cluster Utilization Monitoring.

December 30, 2025: Test coverage for GPU utilization monitoring daemon.
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.cluster_utilization_watchdog import (
    ClusterUtilizationWatchdog,
    UtilizationLevel,
    UtilizationWatchdogConfig,
    get_utilization_watchdog,
    reset_utilization_watchdog,
)
from app.coordination.contracts import CoordinatorStatus


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return UtilizationWatchdogConfig(
        enabled=True,
        check_interval_seconds=30,
        warning_threshold=0.6,
        critical_threshold=0.8,
        idle_util_threshold=10.0,
        idle_duration_trigger_seconds=180,
        recovery_threshold=0.5,
        emit_events=True,
        p2p_status_endpoint="http://localhost:8770/status",
    )


@pytest.fixture
def watchdog(config):
    """Create a test watchdog instance."""
    ClusterUtilizationWatchdog.reset_instance()
    watchdog = ClusterUtilizationWatchdog(config=config)
    yield watchdog
    ClusterUtilizationWatchdog.reset_instance()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    ClusterUtilizationWatchdog.reset_instance()
    yield
    ClusterUtilizationWatchdog.reset_instance()


# ============================================================================
# UtilizationLevel Tests
# ============================================================================


class TestUtilizationLevel:
    """Tests for UtilizationLevel enum."""

    def test_healthy_value(self):
        """Test HEALTHY value."""
        assert UtilizationLevel.HEALTHY.value == "healthy"

    def test_warning_value(self):
        """Test WARNING value."""
        assert UtilizationLevel.WARNING.value == "warning"

    def test_critical_value(self):
        """Test CRITICAL value."""
        assert UtilizationLevel.CRITICAL.value == "critical"

    def test_enum_members(self):
        """Test all enum members exist."""
        assert len(UtilizationLevel) == 3


# ============================================================================
# Configuration Tests
# ============================================================================


class TestUtilizationWatchdogConfig:
    """Tests for UtilizationWatchdogConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        with patch.dict("os.environ", {}, clear=True):
            config = UtilizationWatchdogConfig()

        assert config.check_interval_seconds == 30
        assert config.warning_threshold == 0.6
        assert config.critical_threshold == 0.8
        assert config.idle_util_threshold == 10.0
        assert config.idle_duration_trigger_seconds == 180
        assert config.recovery_threshold == 0.5
        assert config.emit_events is True
        assert config.p2p_status_endpoint == "http://localhost:8770/status"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = UtilizationWatchdogConfig(
            enabled=False,
            check_interval_seconds=60,
            warning_threshold=0.5,
            critical_threshold=0.7,
            idle_util_threshold=5.0,
            idle_duration_trigger_seconds=300,
            recovery_threshold=0.3,
            emit_events=False,
            p2p_status_endpoint="http://custom:8770/status",
        )

        assert config.enabled is False
        assert config.check_interval_seconds == 60
        assert config.warning_threshold == 0.5
        assert config.critical_threshold == 0.7
        assert config.idle_util_threshold == 5.0
        assert config.idle_duration_trigger_seconds == 300
        assert config.recovery_threshold == 0.3
        assert config.emit_events is False

    def test_env_var_enabled(self):
        """Test enabled config from environment variable."""
        with patch.dict("os.environ", {"RINGRIFT_UTILIZATION_WATCHDOG_ENABLED": "true"}):
            config = UtilizationWatchdogConfig()
            assert config.enabled is True

        with patch.dict("os.environ", {"RINGRIFT_UTILIZATION_WATCHDOG_ENABLED": "false"}):
            config = UtilizationWatchdogConfig()
            assert config.enabled is False

    def test_env_var_check_interval(self):
        """Test check_interval_seconds from environment variable."""
        with patch.dict("os.environ", {"RINGRIFT_UTILIZATION_CHECK_INTERVAL": "45"}):
            config = UtilizationWatchdogConfig()
            assert config.check_interval_seconds == 45

    def test_validation_interval_positive(self):
        """Test validation rejects non-positive interval."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            UtilizationWatchdogConfig(check_interval_seconds=0)

        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            UtilizationWatchdogConfig(check_interval_seconds=-1)

    def test_validation_warning_threshold(self):
        """Test validation of warning threshold."""
        with pytest.raises(ValueError, match="warning_threshold must be between 0 and 1"):
            UtilizationWatchdogConfig(warning_threshold=0)

        with pytest.raises(ValueError, match="warning_threshold must be between 0 and 1"):
            UtilizationWatchdogConfig(warning_threshold=1.0)

    def test_validation_critical_threshold(self):
        """Test validation of critical threshold."""
        with pytest.raises(ValueError, match="critical_threshold must be between 0 and 1"):
            UtilizationWatchdogConfig(critical_threshold=0)

    def test_validation_threshold_order(self):
        """Test validation that warning < critical."""
        with pytest.raises(ValueError, match="warning_threshold must be < critical_threshold"):
            UtilizationWatchdogConfig(warning_threshold=0.8, critical_threshold=0.6)

        with pytest.raises(ValueError, match="warning_threshold must be < critical_threshold"):
            UtilizationWatchdogConfig(warning_threshold=0.5, critical_threshold=0.5)


# ============================================================================
# Watchdog Initialization Tests
# ============================================================================


class TestWatchdogInit:
    """Tests for ClusterUtilizationWatchdog initialization."""

    def test_init_with_config(self, config):
        """Test initialization with explicit config."""
        watchdog = ClusterUtilizationWatchdog(config=config)

        assert watchdog.config == config
        assert watchdog._current_level == UtilizationLevel.HEALTHY
        assert watchdog._underutilization_start is None
        assert watchdog._stats_checks == 0

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        with patch.dict("os.environ", {}, clear=True):
            watchdog = ClusterUtilizationWatchdog()

        assert watchdog.config is not None
        assert watchdog.config.check_interval_seconds == 30

    def test_init_sets_cycle_interval(self, config):
        """Test cycle interval is set from config."""
        config.check_interval_seconds = 45
        watchdog = ClusterUtilizationWatchdog(config=config)

        assert watchdog._cycle_interval == 45.0


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_instance(self):
        """Test get_instance returns singleton."""
        watchdog1 = ClusterUtilizationWatchdog.get_instance()
        watchdog2 = ClusterUtilizationWatchdog.get_instance()

        assert watchdog1 is watchdog2

    def test_reset_instance(self):
        """Test reset_instance clears singleton."""
        watchdog1 = ClusterUtilizationWatchdog.get_instance()
        ClusterUtilizationWatchdog.reset_instance()
        watchdog2 = ClusterUtilizationWatchdog.get_instance()

        assert watchdog1 is not watchdog2

    def test_get_utilization_watchdog(self):
        """Test module-level accessor."""
        watchdog = get_utilization_watchdog()
        assert isinstance(watchdog, ClusterUtilizationWatchdog)

    def test_reset_utilization_watchdog(self):
        """Test module-level reset."""
        watchdog1 = get_utilization_watchdog()
        reset_utilization_watchdog()
        watchdog2 = get_utilization_watchdog()

        assert watchdog1 is not watchdog2


# ============================================================================
# Run Cycle Tests
# ============================================================================


class TestRunCycle:
    """Tests for the main run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self, config):
        """Test run cycle skips when disabled."""
        config.enabled = False
        watchdog = ClusterUtilizationWatchdog(config=config)

        with patch.object(watchdog, "_get_cluster_metrics") as mock_metrics:
            await watchdog._run_cycle()
            mock_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_calls_get_metrics(self, watchdog):
        """Test run cycle calls get_cluster_metrics."""
        with patch.object(
            watchdog, "_get_cluster_metrics", return_value=None
        ) as mock_metrics:
            await watchdog._run_cycle()
            mock_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_updates_tracking(self, watchdog):
        """Test run cycle updates tracking state."""
        mock_metrics = {
            "total_gpu_nodes": 10,
            "idle_gpu_nodes": 2,
            "active_gpu_nodes": 8,
            "idle_fraction": 0.2,
        }

        with patch.object(watchdog, "_get_cluster_metrics", return_value=mock_metrics):
            await watchdog._run_cycle()

        assert watchdog._total_gpu_nodes == 10
        assert watchdog._idle_gpu_nodes == 2
        assert watchdog._active_gpu_nodes == 8
        assert watchdog._stats_checks == 1

    @pytest.mark.asyncio
    async def test_run_cycle_handles_none_metrics(self, watchdog):
        """Test run cycle handles None metrics gracefully."""
        with patch.object(watchdog, "_get_cluster_metrics", return_value=None):
            await watchdog._run_cycle()

        assert watchdog._stats_checks == 0

    @pytest.mark.asyncio
    async def test_run_cycle_handles_exception(self, watchdog):
        """Test run cycle handles exceptions gracefully."""
        with patch.object(
            watchdog, "_get_cluster_metrics", side_effect=Exception("Test error")
        ):
            # Should not raise
            await watchdog._run_cycle()


# ============================================================================
# Cluster Metrics Tests
# ============================================================================


class TestGetClusterMetrics:
    """Tests for _get_cluster_metrics method."""

    @pytest.mark.asyncio
    async def test_get_cluster_metrics_success(self, watchdog):
        """Test successful metrics retrieval."""
        mock_status = {
            "peers": {
                "node1": {
                    "state": "alive",
                    "gpu_type": "RTX4090",
                    "active_jobs": 1,
                    "gpu_utilization": 80.0,
                },
                "node2": {
                    "state": "alive",
                    "gpu_memory_gb": 24,
                    "active_jobs": 0,
                    "gpu_utilization": 5.0,
                },
                "node3": {
                    "state": "dead",
                    "gpu_type": "RTX4090",
                },
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_status)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            metrics = await watchdog._get_cluster_metrics()

        # Should have 2 alive GPU nodes (node1 active, node2 idle)
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_get_cluster_metrics_no_aiohttp(self, watchdog):
        """Test metrics retrieval when aiohttp not available."""
        with patch.dict("sys.modules", {"aiohttp": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                metrics = await watchdog._get_cluster_metrics()

        assert metrics is None

    @pytest.mark.asyncio
    async def test_get_cluster_metrics_http_error(self, watchdog):
        """Test metrics retrieval on HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            metrics = await watchdog._get_cluster_metrics()

        assert metrics is None


# ============================================================================
# Level Transition Tests
# ============================================================================


class TestLevelTransition:
    """Tests for _handle_level_transition method."""

    @pytest.mark.asyncio
    async def test_transition_healthy_to_healthy(self, watchdog):
        """Test staying healthy."""
        watchdog._current_level = UtilizationLevel.HEALTHY

        with patch.object(watchdog, "_emit_underutilization_event") as mock_emit:
            await watchdog._handle_level_transition(UtilizationLevel.HEALTHY)

            mock_emit.assert_not_called()
            assert watchdog._current_level == UtilizationLevel.HEALTHY

    @pytest.mark.asyncio
    async def test_transition_healthy_to_warning(self, watchdog):
        """Test transition from healthy to warning."""
        watchdog._current_level = UtilizationLevel.HEALTHY

        # First transition - starts timer but doesn't emit yet
        await watchdog._handle_level_transition(UtilizationLevel.WARNING)

        assert watchdog._underutilization_start is not None
        assert watchdog._current_level == UtilizationLevel.WARNING

    @pytest.mark.asyncio
    async def test_transition_emits_after_duration(self, watchdog):
        """Test event emission after duration trigger."""
        watchdog._current_level = UtilizationLevel.HEALTHY
        watchdog._underutilization_start = time.time() - 200  # More than 180s ago

        with patch.object(watchdog, "_emit_underutilization_event") as mock_emit:
            await watchdog._handle_level_transition(UtilizationLevel.WARNING)

            mock_emit.assert_called_once()
            assert watchdog._stats_warnings == 1

    @pytest.mark.asyncio
    async def test_transition_to_critical(self, watchdog):
        """Test transition to critical level."""
        watchdog._current_level = UtilizationLevel.WARNING
        watchdog._underutilization_start = time.time() - 200

        with patch.object(watchdog, "_emit_underutilization_event") as mock_emit:
            await watchdog._handle_level_transition(UtilizationLevel.CRITICAL)

            mock_emit.assert_called_once()
            args, kwargs = mock_emit.call_args
            assert args[0] == UtilizationLevel.CRITICAL
            assert watchdog._stats_criticals == 1

    @pytest.mark.asyncio
    async def test_transition_recovery(self, watchdog):
        """Test recovery from underutilization."""
        watchdog._current_level = UtilizationLevel.WARNING
        watchdog._underutilization_start = time.time() - 200

        with patch.object(watchdog, "_emit_recovery_event") as mock_emit:
            await watchdog._handle_level_transition(UtilizationLevel.HEALTHY)

            mock_emit.assert_called_once()
            assert watchdog._current_level == UtilizationLevel.HEALTHY
            assert watchdog._underutilization_start is None
            assert watchdog._stats_recoveries == 1


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Tests for event emission methods."""

    @pytest.mark.asyncio
    async def test_emit_underutilization_event(self, watchdog):
        """Test CLUSTER_UNDERUTILIZED event emission."""
        watchdog._total_gpu_nodes = 10
        watchdog._idle_gpu_nodes = 8
        watchdog._active_gpu_nodes = 2

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            await watchdog._emit_underutilization_event(UtilizationLevel.WARNING, 200.0)

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["level"] == "warning"
            assert call_kwargs["total_gpu_nodes"] == 10
            assert call_kwargs["idle_gpu_nodes"] == 8

    @pytest.mark.asyncio
    async def test_emit_underutilization_events_disabled(self, config):
        """Test event not emitted when events disabled."""
        config.emit_events = False
        watchdog = ClusterUtilizationWatchdog(config=config)

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            await watchdog._emit_underutilization_event(UtilizationLevel.WARNING, 200.0)
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_recovery_event(self, watchdog):
        """Test CLUSTER_UTILIZATION_RECOVERED event emission."""
        watchdog._underutilization_start = time.time() - 300
        watchdog._total_gpu_nodes = 10
        watchdog._active_gpu_nodes = 8

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            await watchdog._emit_recovery_event()

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["total_gpu_nodes"] == 10
            assert call_kwargs["active_gpu_nodes"] == 8
            assert call_kwargs["recovery_duration_seconds"] >= 300

    @pytest.mark.asyncio
    async def test_emit_event_handles_import_error(self, watchdog):
        """Test event emission handles import errors gracefully."""
        with patch(
            "app.distributed.data_events.emit_data_event",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise
            await watchdog._emit_underutilization_event(UtilizationLevel.WARNING, 200.0)


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check method."""

    def test_health_check_healthy(self, watchdog):
        """Test health check when healthy."""
        watchdog._current_level = UtilizationLevel.HEALTHY

        result = watchdog.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.details["current_level"] == "healthy"

    def test_health_check_warning(self, watchdog):
        """Test health check when at warning level."""
        watchdog._current_level = UtilizationLevel.WARNING

        result = watchdog.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.details["current_level"] == "warning"

    def test_health_check_critical(self, watchdog):
        """Test health check when at critical level."""
        watchdog._current_level = UtilizationLevel.CRITICAL

        result = watchdog.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.details["current_level"] == "critical"

    def test_health_check_includes_stats(self, watchdog):
        """Test health check includes statistics."""
        watchdog._stats_checks = 100
        watchdog._stats_warnings = 5
        watchdog._stats_criticals = 2
        watchdog._stats_recoveries = 3
        watchdog._total_gpu_nodes = 20
        watchdog._idle_gpu_nodes = 4

        result = watchdog.health_check()

        assert result.details["stats_checks"] == 100
        assert result.details["stats_warnings"] == 5
        assert result.details["stats_criticals"] == 2
        assert result.details["stats_recoveries"] == 3
        assert result.details["total_gpu_nodes"] == 20
        assert result.details["idle_gpu_nodes"] == 4


# ============================================================================
# Accessor Method Tests
# ============================================================================


class TestAccessorMethods:
    """Tests for accessor methods."""

    def test_get_current_level(self, watchdog):
        """Test get_current_level method."""
        assert watchdog.get_current_level() == UtilizationLevel.HEALTHY

        watchdog._current_level = UtilizationLevel.WARNING
        assert watchdog.get_current_level() == UtilizationLevel.WARNING

    def test_get_stats(self, watchdog):
        """Test get_stats method."""
        watchdog._stats_checks = 50
        watchdog._stats_warnings = 3
        watchdog._stats_criticals = 1
        watchdog._stats_recoveries = 2
        watchdog._total_gpu_nodes = 15
        watchdog._idle_gpu_nodes = 5

        stats = watchdog.get_stats()

        assert stats["checks"] == 50
        assert stats["warnings"] == 3
        assert stats["criticals"] == 1
        assert stats["recoveries"] == 2
        assert stats["current_level"] == "healthy"
        assert stats["total_gpu_nodes"] == 15
        assert stats["idle_gpu_nodes"] == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests combining multiple operations."""

    @pytest.mark.asyncio
    async def test_full_underutilization_cycle(self, watchdog):
        """Test full cycle from healthy to warning to recovery."""
        # Start healthy
        assert watchdog._current_level == UtilizationLevel.HEALTHY

        # Simulate underutilization detection
        mock_metrics_underutilized = {
            "total_gpu_nodes": 10,
            "idle_gpu_nodes": 7,  # 70% idle > 60% warning threshold
            "active_gpu_nodes": 3,
            "idle_fraction": 0.7,
        }

        with patch.object(
            watchdog, "_get_cluster_metrics", return_value=mock_metrics_underutilized
        ):
            # First cycle - starts timer
            await watchdog._run_cycle()
            assert watchdog._current_level == UtilizationLevel.WARNING
            assert watchdog._underutilization_start is not None

        # Simulate recovery
        mock_metrics_healthy = {
            "total_gpu_nodes": 10,
            "idle_gpu_nodes": 3,  # 30% idle < 60% threshold
            "active_gpu_nodes": 7,
            "idle_fraction": 0.3,
        }

        with patch.object(watchdog, "_get_cluster_metrics", return_value=mock_metrics_healthy):
            with patch.object(watchdog, "_emit_recovery_event") as mock_emit:
                await watchdog._run_cycle()

                mock_emit.assert_called_once()
                assert watchdog._current_level == UtilizationLevel.HEALTHY
                assert watchdog._underutilization_start is None

    @pytest.mark.asyncio
    async def test_escalation_to_critical(self, watchdog):
        """Test escalation from warning to critical."""
        # Set up warning state
        watchdog._current_level = UtilizationLevel.WARNING
        watchdog._underutilization_start = time.time() - 200  # Past trigger

        mock_metrics_critical = {
            "total_gpu_nodes": 10,
            "idle_gpu_nodes": 9,  # 90% idle > 80% critical threshold
            "active_gpu_nodes": 1,
            "idle_fraction": 0.9,
        }

        with patch.object(watchdog, "_get_cluster_metrics", return_value=mock_metrics_critical):
            with patch.object(watchdog, "_emit_underutilization_event") as mock_emit:
                await watchdog._run_cycle()

                mock_emit.assert_called_once()
                args = mock_emit.call_args[0]
                assert args[0] == UtilizationLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_no_event_before_duration(self, watchdog):
        """Test no event emission before duration trigger."""
        mock_metrics_underutilized = {
            "total_gpu_nodes": 10,
            "idle_gpu_nodes": 7,
            "active_gpu_nodes": 3,
            "idle_fraction": 0.7,
        }

        with patch.object(
            watchdog, "_get_cluster_metrics", return_value=mock_metrics_underutilized
        ):
            with patch.object(watchdog, "_emit_underutilization_event") as mock_emit:
                # First cycle - within duration trigger
                await watchdog._run_cycle()

                # Event should not be emitted yet (duration not exceeded)
                mock_emit.assert_not_called()
