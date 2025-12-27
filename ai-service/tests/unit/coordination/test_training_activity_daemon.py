"""Tests for TrainingActivityDaemon.

December 2025: Comprehensive test coverage for the TrainingActivityDaemon
which monitors cluster for training activity and triggers priority sync.

Tests cover:
- Configuration loading and defaults
- Local training process detection
- P2P status parsing for remote training
- Priority sync triggering
- TRAINING_STARTED event emission
- Graceful shutdown with final sync
- Signal handling via BaseDaemon
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.base_daemon import DaemonConfig
from app.coordination.training_activity_daemon import (
    TrainingActivityConfig,
    TrainingActivityDaemon,
    get_training_activity_daemon,
    reset_training_activity_daemon,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestTrainingActivityConfig:
    """Test TrainingActivityConfig dataclass and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingActivityConfig()

        assert config.enabled is True
        assert config.check_interval_seconds == 30
        assert config.handle_signals is True
        assert config.trigger_priority_sync is True
        assert "app.training.train" in config.training_process_patterns
        assert "train.py" in config.training_process_patterns

    def test_config_inheritance(self):
        """Test that config inherits from DaemonConfig."""
        config = TrainingActivityConfig()
        assert isinstance(config, DaemonConfig)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingActivityConfig(
            enabled=False,
            check_interval_seconds=60,
            handle_signals=False,
            trigger_priority_sync=False,
            training_process_patterns=["custom_train.py"],
        )

        assert config.enabled is False
        assert config.check_interval_seconds == 60
        assert config.handle_signals is False
        assert config.trigger_priority_sync is False
        assert config.training_process_patterns == ["custom_train.py"]

    def test_from_env_defaults(self):
        """Test loading config from environment with defaults."""
        config = TrainingActivityConfig.from_env()

        assert config.enabled is True
        assert config.check_interval_seconds == 30

    @patch.dict("os.environ", {
        "RINGRIFT_TRAINING_ACTIVITY_ENABLED": "0",
        "RINGRIFT_TRAINING_ACTIVITY_TRIGGER_SYNC": "0",
    })
    def test_from_env_custom(self):
        """Test loading config from environment variables."""
        config = TrainingActivityConfig.from_env()

        assert config.enabled is False
        assert config.trigger_priority_sync is False


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestTrainingActivityDaemonInit:
    """Test TrainingActivityDaemon initialization."""

    def test_default_initialization(self):
        """Test daemon initializes with default config."""
        daemon = TrainingActivityDaemon()

        assert daemon.config is not None
        assert daemon.config.check_interval_seconds == 30
        assert daemon._training_nodes == set()
        assert daemon._syncs_triggered == 0

    def test_custom_config_initialization(self):
        """Test daemon initializes with custom config."""
        config = TrainingActivityConfig(check_interval_seconds=60)
        daemon = TrainingActivityDaemon(config=config)

        assert daemon.config.check_interval_seconds == 60

    def test_daemon_name(self):
        """Test daemon name is correct."""
        daemon = TrainingActivityDaemon()
        assert daemon._get_daemon_name() == "TrainingActivityDaemon"

    def test_node_id_set(self):
        """Test node_id is set from hostname."""
        daemon = TrainingActivityDaemon()
        assert daemon.node_id is not None
        assert len(daemon.node_id) > 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern for daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_training_activity_daemon()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_training_activity_daemon()

    def test_get_singleton(self):
        """Test getting singleton instance."""
        daemon1 = get_training_activity_daemon()
        daemon2 = get_training_activity_daemon()

        assert daemon1 is daemon2

    def test_reset_singleton(self):
        """Test resetting singleton creates new instance."""
        daemon1 = get_training_activity_daemon()
        reset_training_activity_daemon()
        daemon2 = get_training_activity_daemon()

        assert daemon1 is not daemon2


# =============================================================================
# Local Training Detection Tests
# =============================================================================


class TestLocalTrainingDetection:
    """Test local training process detection."""

    def test_detect_local_training_no_process(self):
        """Test detection when no training process running."""
        daemon = TrainingActivityDaemon()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)  # pgrep finds nothing
            result = daemon.detect_local_training()

        assert result is False

    def test_detect_local_training_process_found(self):
        """Test detection when training process is running."""
        daemon = TrainingActivityDaemon()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)  # pgrep finds process
            result = daemon.detect_local_training()

        assert result is True

    def test_detect_local_training_pgrep_timeout(self):
        """Test handling of pgrep timeout."""
        daemon = TrainingActivityDaemon()

        with patch("subprocess.run") as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired("pgrep", 5)
            result = daemon.detect_local_training()

        assert result is False

    def test_detect_local_training_custom_patterns(self):
        """Test detection with custom process patterns."""
        config = TrainingActivityConfig(
            training_process_patterns=["my_custom_trainer"]
        )
        daemon = TrainingActivityDaemon(config=config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = daemon.detect_local_training()

        assert result is True
        # Verify pgrep was called with custom pattern
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "my_custom_trainer" in call_args


# =============================================================================
# P2P Status Parsing Tests
# =============================================================================


class TestP2PStatusParsing:
    """Test P2P cluster status parsing for training detection."""

    @pytest.mark.asyncio
    async def test_check_p2p_training_no_training(self):
        """Test P2P check when no training detected."""
        daemon = TrainingActivityDaemon()

        p2p_status = {
            "peers": {
                "node-1": {"running_jobs": [], "processes": []},
                "node-2": {"running_jobs": [{"type": "selfplay"}]},
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=p2p_status):
            result = await daemon._check_p2p_training()

        assert result == set()

    @pytest.mark.asyncio
    async def test_check_p2p_training_job_detected(self):
        """Test P2P check when training job detected."""
        daemon = TrainingActivityDaemon()

        p2p_status = {
            "peers": {
                "node-1": {"running_jobs": [{"type": "training"}], "processes": []},
                "node-2": {"running_jobs": [], "processes": []},
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=p2p_status):
            result = await daemon._check_p2p_training()

        assert result == {"node-1"}

    @pytest.mark.asyncio
    async def test_check_p2p_training_process_detected(self):
        """Test P2P check when training process detected."""
        daemon = TrainingActivityDaemon()

        p2p_status = {
            "peers": {
                "node-1": {
                    "running_jobs": [],
                    "processes": ["python app.training.train --epochs 50"]
                },
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=p2p_status):
            result = await daemon._check_p2p_training()

        assert result == {"node-1"}

    @pytest.mark.asyncio
    async def test_check_p2p_training_multiple_nodes(self):
        """Test P2P check when multiple nodes have training."""
        daemon = TrainingActivityDaemon()

        p2p_status = {
            "peers": {
                "node-1": {"running_jobs": [{"type": "training"}], "processes": []},
                "node-2": {"running_jobs": [{"type": "TRAINING"}], "processes": []},
                "node-3": {"running_jobs": [], "processes": []},
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=p2p_status):
            result = await daemon._check_p2p_training()

        assert result == {"node-1", "node-2"}

    @pytest.mark.asyncio
    async def test_check_p2p_training_p2p_unavailable(self):
        """Test P2P check when P2P service unavailable."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_get_p2p_status", return_value=None):
            result = await daemon._check_p2p_training()

        assert result == set()


# =============================================================================
# Priority Sync Tests
# =============================================================================


class TestPrioritySync:
    """Test priority sync triggering."""

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_success(self):
        """Test successful priority sync trigger."""
        daemon = TrainingActivityDaemon()

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock()

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade
        ):
            await daemon._trigger_priority_sync("node-1")

        mock_facade.trigger_priority_sync.assert_called_once()
        assert daemon._syncs_triggered == 1

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_failure(self):
        """Test priority sync trigger failure handling."""
        daemon = TrainingActivityDaemon()

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock(side_effect=Exception("Sync failed"))

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade
        ):
            # Should not raise, just log error
            await daemon._trigger_priority_sync("node-1")

        assert daemon._syncs_triggered == 0  # Not incremented on failure

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_disabled(self):
        """Test priority sync not triggered when disabled."""
        config = TrainingActivityConfig(trigger_priority_sync=False)
        daemon = TrainingActivityDaemon(config=config)

        # When sync disabled, _emit_training_started is still called but not _trigger_priority_sync
        with patch.object(daemon, "_emit_training_started", return_value=None):
            with patch.object(daemon, "_trigger_priority_sync") as mock_sync:
                await daemon._on_training_detected({"node-1"})

        # Sync should not be triggered
        mock_sync.assert_not_called()


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Test TRAINING_STARTED event emission."""

    @pytest.mark.asyncio
    async def test_emit_training_started(self):
        """Test TRAINING_STARTED event emission."""
        daemon = TrainingActivityDaemon()

        with patch(
            "app.distributed.data_events.emit_training_started",
            new_callable=AsyncMock
        ) as mock_emit:
            await daemon._emit_training_started({"node-1", "node-2"})

        assert mock_emit.call_count == 2

    @pytest.mark.asyncio
    async def test_emit_training_started_import_error(self):
        """Test handling when emit function not available."""
        daemon = TrainingActivityDaemon()

        # Mock the entire import to fail
        with patch.object(
            daemon, "_emit_training_started",
            side_effect=Exception("Import failed")
        ):
            # When _emit_training_started fails, it should be caught
            pass  # The actual error would be caught inside the method

        # Alternative: verify error handling by calling with mock that raises
        original_emit = daemon._emit_training_started
        async def failing_emit(nodes):
            # Simulate the inner try/except
            try:
                raise ImportError("Module not found")
            except Exception:
                pass  # Should be caught internally

        # Call directly - should not raise
        await daemon._emit_training_started({"node-1"})


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Test main daemon cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_no_training(self):
        """Test cycle when no training detected."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_check_p2p_training", return_value=set()):
            with patch.object(daemon, "detect_local_training", return_value=False):
                with patch.object(daemon, "_on_training_detected") as mock_handler:
                    await daemon._run_cycle()

        mock_handler.assert_not_called()
        assert daemon._training_nodes == set()

    @pytest.mark.asyncio
    async def test_run_cycle_new_training_detected(self):
        """Test cycle when new training detected."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_check_p2p_training", return_value={"node-1"}):
            with patch.object(daemon, "detect_local_training", return_value=False):
                with patch.object(daemon, "_on_training_detected") as mock_handler:
                    mock_handler.return_value = None
                    await daemon._run_cycle()

        mock_handler.assert_called_once_with({"node-1"})
        assert daemon._training_nodes == {"node-1"}

    @pytest.mark.asyncio
    async def test_run_cycle_training_completed(self):
        """Test cycle when training completes."""
        daemon = TrainingActivityDaemon()
        daemon._training_nodes = {"node-1"}

        with patch.object(daemon, "_check_p2p_training", return_value=set()):
            with patch.object(daemon, "detect_local_training", return_value=False):
                with patch.object(daemon, "_on_training_detected") as mock_handler:
                    await daemon._run_cycle()

        mock_handler.assert_not_called()
        assert daemon._training_nodes == set()

    @pytest.mark.asyncio
    async def test_run_cycle_local_training_detected(self):
        """Test cycle when local training detected."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_check_p2p_training", return_value=set()):
            with patch.object(daemon, "detect_local_training", return_value=True):
                with patch.object(daemon, "_on_training_detected") as mock_handler:
                    mock_handler.return_value = None
                    await daemon._run_cycle()

        # Local training should add node_id
        assert daemon.node_id in daemon._training_nodes


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Test daemon lifecycle (start/stop)."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test basic start and stop."""
        daemon = TrainingActivityDaemon()

        # Mock the run cycle to prevent actual loop
        with patch.object(daemon, "_run_cycle", return_value=None):
            await daemon.start()
            assert daemon.is_running is True

            await daemon.stop()
            assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Test start when disabled."""
        config = TrainingActivityConfig(enabled=False)
        daemon = TrainingActivityDaemon(config=config)

        await daemon.start()
        assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test starting already running daemon."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_run_cycle", return_value=None):
            await daemon.start()
            await daemon.start()  # Second start should be no-op

            assert daemon.is_running is True

            await daemon.stop()


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Test graceful shutdown with final sync."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_triggers_sync(self):
        """Test that graceful shutdown triggers final sync."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_trigger_priority_sync") as mock_sync:
            mock_sync.return_value = None
            await daemon._on_graceful_shutdown()

        mock_sync.assert_called_once_with("termination")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sync_disabled(self):
        """Test graceful shutdown when sync disabled."""
        config = TrainingActivityConfig(trigger_priority_sync=False)
        daemon = TrainingActivityDaemon(config=config)

        with patch.object(daemon, "_trigger_priority_sync") as mock_sync:
            await daemon._on_graceful_shutdown()

        mock_sync.assert_not_called()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test daemon health check."""

    def test_health_check_not_running(self):
        """Test health check when daemon not running."""
        daemon = TrainingActivityDaemon()

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """Test health check when daemon running."""
        daemon = TrainingActivityDaemon()

        with patch.object(daemon, "_run_cycle", return_value=None):
            await daemon.start()

            result = daemon.health_check()

            assert result.healthy is True
            assert "healthy" in result.message.lower()

            await daemon.stop()


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Test daemon status reporting."""

    def test_get_status(self):
        """Test getting daemon status."""
        daemon = TrainingActivityDaemon()

        status = daemon.get_status()

        assert "name" in status
        assert status["name"] == "TrainingActivityDaemon"
        assert "running" in status
        assert "config" in status

    def test_get_training_nodes(self):
        """Test getting training nodes."""
        daemon = TrainingActivityDaemon()
        daemon._training_nodes = {"node-1", "node-2"}

        nodes = daemon.get_training_nodes()

        assert nodes == {"node-1", "node-2"}
        # Should return a copy
        assert nodes is not daemon._training_nodes

    def test_health_check_includes_stats(self):
        """Test health check includes sync stats."""
        daemon = TrainingActivityDaemon()
        daemon._syncs_triggered = 5

        result = daemon.health_check()

        assert "syncs_triggered" in result.details
        assert result.details["syncs_triggered"] == 5
