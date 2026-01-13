"""Tests for background_selfplay.py - Background selfplay manager.

Tests process lifecycle, subprocess management, event emission, and singleton patterns.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from app.training.background_selfplay import (
    BackgroundSelfplayManager,
    BackgroundSelfplayTask,
    get_background_selfplay_manager,
    reset_background_selfplay_manager,
)


# =============================================================================
# BackgroundSelfplayTask Tests
# =============================================================================


class TestBackgroundSelfplayTask:
    """Tests for BackgroundSelfplayTask dataclass."""

    def test_init_default_values(self):
        """Test task initialization with default values."""
        task = BackgroundSelfplayTask(iteration=1)

        assert task.iteration == 1
        assert task.process is None
        assert task.staging_db_path is None
        assert task.games_requested == 0
        assert task.board_type == "square8"
        assert task.num_players == 2
        assert task.start_time > 0

    def test_init_custom_values(self):
        """Test task initialization with custom values."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        staging_path = Path("/tmp/test.db")

        task = BackgroundSelfplayTask(
            iteration=5,
            process=mock_process,
            staging_db_path=staging_path,
            games_requested=100,
            board_type="hex8",
            num_players=4,
        )

        assert task.iteration == 5
        assert task.process is mock_process
        assert task.staging_db_path == staging_path
        assert task.games_requested == 100
        assert task.board_type == "hex8"
        assert task.num_players == 4

    def test_is_running_no_process(self):
        """Test is_running returns False when no process."""
        task = BackgroundSelfplayTask(iteration=1)
        assert task.is_running() is False

    def test_is_running_process_still_running(self):
        """Test is_running returns True when process is running."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # None means still running

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        assert task.is_running() is True
        mock_process.poll.assert_called_once()

    def test_is_running_process_completed(self):
        """Test is_running returns False when process completed."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0  # 0 means completed

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        assert task.is_running() is False

    def test_wait_no_process(self):
        """Test wait returns success when no process."""
        task = BackgroundSelfplayTask(iteration=1)
        success, code = task.wait()

        assert success is True
        assert code == 0

    def test_wait_process_success(self):
        """Test wait returns success for successful process."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.returncode = 0

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        success, code = task.wait(timeout=10)

        assert success is True
        assert code == 0
        mock_process.wait.assert_called_once_with(timeout=10)

    def test_wait_process_failure(self):
        """Test wait returns failure for failed process."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.returncode = 1

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        success, code = task.wait()

        assert success is False
        assert code == 1

    def test_wait_timeout_expired(self):
        """Test wait handles timeout."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        success, code = task.wait(timeout=5)

        assert success is False
        assert code == -1

    def test_terminate_no_process(self):
        """Test terminate does nothing when no process."""
        task = BackgroundSelfplayTask(iteration=1)
        task.terminate()  # Should not raise

    def test_terminate_not_running(self):
        """Test terminate does nothing when process not running."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0  # Already completed

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        task.terminate()

        mock_process.terminate.assert_not_called()

    def test_terminate_running_graceful(self):
        """Test terminate gracefully stops running process."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # Still running

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        task.terminate()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_terminate_force_kill_on_timeout(self):
        """Test terminate force kills on timeout."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        task.terminate()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_elapsed_time(self):
        """Test elapsed_time calculation."""
        start = time.time()
        task = BackgroundSelfplayTask(iteration=1, start_time=start)

        time.sleep(0.1)
        elapsed = task.elapsed_time()

        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be quick


# =============================================================================
# BackgroundSelfplayManager Tests
# =============================================================================


class TestBackgroundSelfplayManager:
    """Tests for BackgroundSelfplayManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create manager with temp directory."""
        return BackgroundSelfplayManager(ai_service_root=temp_dir)

    def test_init_default_root(self):
        """Test manager initialization with default root."""
        manager = BackgroundSelfplayManager()
        assert manager._ai_service_root is not None
        assert manager._current_task is None
        assert manager._history == []

    def test_init_custom_root(self, temp_dir):
        """Test manager initialization with custom root."""
        manager = BackgroundSelfplayManager(ai_service_root=temp_dir)
        assert manager._ai_service_root == temp_dir

    def test_get_current_task_none(self, manager):
        """Test get_current_task when no task."""
        assert manager.get_current_task() is None

    def test_has_pending_task_false_no_task(self, manager):
        """Test has_pending_task returns False when no task."""
        assert manager.has_pending_task() is False

    def test_has_pending_task_false_completed(self, manager):
        """Test has_pending_task returns False when task completed."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0  # Completed

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        manager._current_task = task

        assert manager.has_pending_task() is False

    def test_has_pending_task_true_running(self, manager):
        """Test has_pending_task returns True when task running."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # Still running

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        manager._current_task = task

        assert manager.has_pending_task() is True

    @mock.patch("subprocess.Popen")
    def test_start_background_selfplay_basic(self, mock_popen, manager, temp_dir):
        """Test starting background selfplay with basic config."""
        mock_process = mock.MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        config = {
            "board": "hex8",
            "players": 2,
            "games_per_iter": 50,
        }

        task = manager.start_background_selfplay(config, iteration=1)

        assert task is not None
        assert task.iteration == 1
        assert task.board_type == "hex8"
        assert task.num_players == 2
        assert task.games_requested == 50
        assert task.staging_db_path is not None
        assert manager._current_task is task

    @mock.patch("subprocess.Popen")
    def test_start_background_selfplay_cancels_previous(self, mock_popen, manager):
        """Test starting new selfplay cancels previous task."""
        mock_process1 = mock.MagicMock()
        mock_process1.poll.return_value = None  # Still running
        mock_process1.pid = 111

        mock_process2 = mock.MagicMock()
        mock_process2.pid = 222
        mock_popen.side_effect = [mock_process1, mock_process2]

        config = {"board": "square8", "players": 2, "games_per_iter": 10}

        # Start first task
        task1 = manager.start_background_selfplay(config, iteration=1)
        assert task1 is not None

        # Start second task - should cancel first
        task2 = manager.start_background_selfplay(config, iteration=2)

        mock_process1.terminate.assert_called_once()
        assert manager._current_task is task2

    @mock.patch("subprocess.Popen")
    def test_start_background_selfplay_creates_staging_dir(self, mock_popen, manager, temp_dir):
        """Test that staging directory is created."""
        mock_popen.return_value = mock.MagicMock(pid=123)

        config = {
            "board": "square8",
            "players": 2,
            "games_per_iter": 10,
            "staging_db_dir": "data/staging/test",
        }

        task = manager.start_background_selfplay(config, iteration=1)

        expected_dir = temp_dir / "data/staging/test"
        assert expected_dir.exists()

    @mock.patch("subprocess.Popen")
    def test_start_background_selfplay_with_optional_params(self, mock_popen, manager):
        """Test starting selfplay with optional parameters."""
        mock_popen.return_value = mock.MagicMock(pid=123)

        config = {
            "board": "hex8",
            "players": 4,
            "games_per_iter": 100,
            "selfplay_difficulty_band": "medium",
            "selfplay_engine_mode": "gumbel",
            "selfplay_nn_pool_size": 3,
            "selfplay_nn_pool_dir": "models/pool",
        }

        manager.start_background_selfplay(config, iteration=1)

        # Check that Popen was called with extended arguments
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert "--difficulty-band" in cmd
        assert "medium" in cmd
        assert "--engine-mode" in cmd
        assert "gumbel" in cmd
        assert "--nn-pool-size" in cmd
        assert "3" in cmd
        assert "--nn-pool-dir" in cmd

    @mock.patch("subprocess.Popen")
    def test_start_background_selfplay_popen_failure(self, mock_popen, manager):
        """Test handling Popen failure."""
        mock_popen.side_effect = OSError("Failed to start process")

        config = {"board": "square8", "players": 2, "games_per_iter": 10}

        task = manager.start_background_selfplay(config, iteration=1)

        assert task is None
        assert manager._current_task is None

    def test_wait_for_current_no_task(self, manager):
        """Test wait_for_current when no task."""
        success, path, games = manager.wait_for_current()

        assert success is True
        assert path is None
        assert games == 0

    def test_wait_for_current_success(self, manager):
        """Test wait_for_current for successful task."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.returncode = 0
        staging_path = Path("/tmp/test.db")

        task = BackgroundSelfplayTask(
            iteration=1,
            process=mock_process,
            staging_db_path=staging_path,
            games_requested=100,
        )
        manager._current_task = task

        with mock.patch.object(manager, "_emit_selfplay_complete"):
            success, path, games = manager.wait_for_current()

        assert success is True
        assert path == staging_path
        assert games == 100
        assert manager._current_task is None
        assert task in manager._history

    def test_wait_for_current_failure(self, manager):
        """Test wait_for_current for failed task."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.returncode = 1

        task = BackgroundSelfplayTask(
            iteration=1,
            process=mock_process,
            staging_db_path=Path("/tmp/test.db"),
            games_requested=100,
        )
        manager._current_task = task

        with mock.patch.object(manager, "_emit_selfplay_complete"):
            success, path, games = manager.wait_for_current()

        assert success is False
        assert path is None
        assert games == 0

    def test_wait_for_current_with_timeout(self, manager):
        """Test wait_for_current with timeout."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        manager._current_task = task

        with mock.patch.object(manager, "_emit_selfplay_complete"):
            success, path, games = manager.wait_for_current(timeout=5)

        assert success is False
        mock_process.wait.assert_called_with(timeout=5)

    def test_cancel_current_no_task(self, manager):
        """Test cancel_current when no task."""
        manager.cancel_current()  # Should not raise
        assert manager._current_task is None

    def test_cancel_current_terminates_task(self, manager):
        """Test cancel_current terminates running task."""
        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # Running

        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        manager._current_task = task

        manager.cancel_current()

        mock_process.terminate.assert_called_once()
        assert manager._current_task is None

    def test_get_statistics_empty(self, manager):
        """Test get_statistics with no history."""
        stats = manager.get_statistics()

        assert stats["total_runs"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["current_running"] is False
        assert stats["current_iteration"] is None
        assert stats["avg_elapsed_time"] == 0.0

    def test_get_statistics_with_history(self, manager):
        """Test get_statistics with completed tasks."""
        # Create completed tasks
        mock_success = mock.MagicMock()
        mock_success.returncode = 0
        task1 = BackgroundSelfplayTask(iteration=1, process=mock_success, start_time=time.time() - 10)

        mock_fail = mock.MagicMock()
        mock_fail.returncode = 1
        task2 = BackgroundSelfplayTask(iteration=2, process=mock_fail, start_time=time.time() - 5)

        manager._history = [task1, task2]

        # Add running task
        mock_running = mock.MagicMock()
        mock_running.poll.return_value = None
        manager._current_task = BackgroundSelfplayTask(iteration=3, process=mock_running)

        stats = manager.get_statistics()

        assert stats["total_runs"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["current_running"] is True
        assert stats["current_iteration"] == 3
        assert stats["avg_elapsed_time"] > 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton management functions."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_background_selfplay_manager()

    def test_get_background_selfplay_manager_creates_singleton(self):
        """Test get_background_selfplay_manager creates singleton."""
        manager1 = get_background_selfplay_manager()
        manager2 = get_background_selfplay_manager()

        assert manager1 is manager2

    def test_get_background_selfplay_manager_with_root(self):
        """Test get_background_selfplay_manager with custom root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = get_background_selfplay_manager(ai_service_root=Path(tmpdir))
            assert manager._ai_service_root == Path(tmpdir)

    def test_reset_background_selfplay_manager(self):
        """Test reset_background_selfplay_manager resets singleton."""
        manager1 = get_background_selfplay_manager()
        reset_background_selfplay_manager()
        manager2 = get_background_selfplay_manager()

        assert manager1 is not manager2

    def test_reset_cancels_current_task(self):
        """Test reset cancels any current task."""
        manager = get_background_selfplay_manager()

        mock_process = mock.MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None  # Running
        task = BackgroundSelfplayTask(iteration=1, process=mock_process)
        manager._current_task = task

        reset_background_selfplay_manager()

        mock_process.terminate.assert_called_once()


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for selfplay completion event emission."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield BackgroundSelfplayManager(ai_service_root=Path(tmpdir))

    def test_emit_selfplay_complete_success(self, manager):
        """Test successful event emission."""
        task = BackgroundSelfplayTask(
            iteration=1,
            board_type="hex8",
            num_players=2,
        )

        with mock.patch("app.coordination.selfplay_orchestrator.emit_selfplay_completion") as mock_emit:
            with mock.patch("asyncio.get_running_loop", side_effect=RuntimeError):
                with mock.patch("asyncio.run") as mock_run:
                    manager._emit_selfplay_complete(task, success=True, games_generated=100)
                    mock_run.assert_called_once()

    def test_emit_selfplay_complete_import_error(self, manager):
        """Test event emission handles ImportError gracefully."""
        task = BackgroundSelfplayTask(iteration=1)

        # This should not raise
        with mock.patch.dict("sys.modules", {"app.coordination.selfplay_orchestrator": None}):
            manager._emit_selfplay_complete(task, success=True, games_generated=50)


# =============================================================================
# Coordination Integration Tests
# =============================================================================


class TestCoordinationIntegration:
    """Tests for coordination system integration."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield BackgroundSelfplayManager(ai_service_root=Path(tmpdir))

    @mock.patch("subprocess.Popen")
    @mock.patch("app.training.background_selfplay.can_spawn_safe")
    def test_coordination_check_allowed(self, mock_can_spawn, mock_popen, manager):
        """Test coordination check when spawning is allowed."""
        mock_can_spawn.return_value = (True, "allowed")
        mock_popen.return_value = mock.MagicMock(pid=123)

        config = {"board": "square8", "players": 2, "games_per_iter": 10}
        task = manager.start_background_selfplay(config, iteration=1)

        assert task is not None

    @mock.patch("subprocess.Popen")
    @mock.patch("app.training.background_selfplay.can_spawn_safe")
    def test_coordination_check_warning_proceeds(self, mock_can_spawn, mock_popen, manager):
        """Test coordination warning doesn't block spawning."""
        mock_can_spawn.return_value = (False, "max tasks reached")
        mock_popen.return_value = mock.MagicMock(pid=123)

        config = {"board": "square8", "players": 2, "games_per_iter": 10}
        task = manager.start_background_selfplay(config, iteration=1)

        # Should still proceed (advisory only)
        assert task is not None

    @mock.patch("subprocess.Popen")
    @mock.patch("app.training.background_selfplay.can_spawn_safe")
    def test_coordination_check_exception_handled(self, mock_can_spawn, mock_popen, manager):
        """Test coordination check exception is handled."""
        mock_can_spawn.side_effect = Exception("coordination error")
        mock_popen.return_value = mock.MagicMock(pid=123)

        config = {"board": "square8", "players": 2, "games_per_iter": 10}
        task = manager.start_background_selfplay(config, iteration=1)

        # Should still proceed
        assert task is not None
