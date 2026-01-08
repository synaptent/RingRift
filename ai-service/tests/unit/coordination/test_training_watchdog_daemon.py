"""Tests for TrainingWatchdogDaemon - monitors stuck training processes.

These tests verify:
1. Configuration (TrainingWatchdogConfig, from_env)
2. Data classes (TrainingProcessInfo)
3. TrainingWatchdogDaemon initialization and lifecycle
4. Database operations (register, heartbeat, mark_completed)
5. Stale process detection
6. Process killing logic
7. Event handlers
8. Health check
9. Singleton pattern
10. Helper functions

January 7, 2026 - Sprint 17: Added as part of Phase 3 test coverage.
"""

import asyncio
import os
import signal
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_watchdog_daemon import (
    DEFAULT_CHECK_INTERVAL_SECONDS,
    DEFAULT_DB_PATH,
    DEFAULT_KILL_GRACE_PERIOD_SECONDS,
    DEFAULT_STALE_THRESHOLD_SECONDS,
    TrainingProcessInfo,
    TrainingWatchdogConfig,
    TrainingWatchdogDaemon,
    get_training_watchdog_daemon,
    reset_training_watchdog_daemon,
    send_training_heartbeat,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestTrainingWatchdogConfig:
    """Tests for TrainingWatchdogConfig dataclass."""

    def test_default_values(self):
        """TrainingWatchdogConfig should have sensible defaults."""
        config = TrainingWatchdogConfig()
        assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL_SECONDS
        assert config.stale_threshold_seconds == DEFAULT_STALE_THRESHOLD_SECONDS
        assert config.kill_grace_period_seconds == DEFAULT_KILL_GRACE_PERIOD_SECONDS
        assert config.db_path == DEFAULT_DB_PATH
        assert config.enable_process_kill is True
        assert config.release_locks_on_kill is True

    def test_custom_values(self):
        """TrainingWatchdogConfig should accept custom values."""
        custom_path = Path("/custom/path/db.sqlite")
        config = TrainingWatchdogConfig(
            check_interval_seconds=120,
            stale_threshold_seconds=3600,
            kill_grace_period_seconds=60,
            db_path=custom_path,
            enable_process_kill=False,
            release_locks_on_kill=False,
        )
        assert config.check_interval_seconds == 120
        assert config.stale_threshold_seconds == 3600
        assert config.kill_grace_period_seconds == 60
        assert config.db_path == custom_path
        assert config.enable_process_kill is False
        assert config.release_locks_on_kill is False

    def test_from_env_defaults(self):
        """from_env should use defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = TrainingWatchdogConfig.from_env()
            assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL_SECONDS
            assert config.stale_threshold_seconds == DEFAULT_STALE_THRESHOLD_SECONDS
            assert config.kill_grace_period_seconds == DEFAULT_KILL_GRACE_PERIOD_SECONDS
            assert config.db_path == DEFAULT_DB_PATH

    def test_from_env_with_env_vars(self):
        """from_env should read from environment variables."""
        env_vars = {
            "RINGRIFT_TRAINING_WATCHDOG_INTERVAL": "120",
            "RINGRIFT_TRAINING_WATCHDOG_STALE_THRESHOLD": "3600",
            "RINGRIFT_TRAINING_WATCHDOG_KILL_GRACE": "60",
            "RINGRIFT_TRAINING_WATCHDOG_DB_PATH": "/custom/path.db",
            "RINGRIFT_TRAINING_WATCHDOG_ENABLE_KILL": "false",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = TrainingWatchdogConfig.from_env()
            assert config.check_interval_seconds == 120
            assert config.stale_threshold_seconds == 3600
            assert config.kill_grace_period_seconds == 60
            assert config.db_path == Path("/custom/path.db")
            assert config.enable_process_kill is False

    def test_from_env_with_invalid_values(self):
        """from_env should ignore invalid values and use defaults."""
        env_vars = {
            "RINGRIFT_TRAINING_WATCHDOG_INTERVAL": "not_a_number",
            "RINGRIFT_TRAINING_WATCHDOG_STALE_THRESHOLD": "also_not_a_number",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = TrainingWatchdogConfig.from_env()
            # Should fall back to defaults for invalid values
            assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL_SECONDS
            assert config.stale_threshold_seconds == DEFAULT_STALE_THRESHOLD_SECONDS


# =============================================================================
# Data Class Tests
# =============================================================================


class TestTrainingProcessInfo:
    """Tests for TrainingProcessInfo dataclass."""

    def test_creation(self):
        """TrainingProcessInfo should accept all parameters."""
        now = time.time()
        info = TrainingProcessInfo(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
            started_at=now - 100,
            last_heartbeat=now,
            status="running",
        )
        assert info.config_key == "hex8_2p"
        assert info.pid == 12345
        assert info.node_id == "worker-1"
        assert info.started_at == now - 100
        assert info.last_heartbeat == now
        assert info.status == "running"

    def test_default_status(self):
        """TrainingProcessInfo should default status to 'running'."""
        now = time.time()
        info = TrainingProcessInfo(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
            started_at=now,
            last_heartbeat=now,
        )
        assert info.status == "running"


# =============================================================================
# TrainingWatchdogDaemon Initialization Tests
# =============================================================================


class TestTrainingWatchdogDaemonInit:
    """Tests for TrainingWatchdogDaemon initialization."""

    def test_default_config(self):
        """TrainingWatchdogDaemon should use default config if none provided."""
        daemon = TrainingWatchdogDaemon()
        assert daemon._daemon_config is not None
        assert daemon._daemon_config.stale_threshold_seconds == DEFAULT_STALE_THRESHOLD_SECONDS

    def test_custom_config(self):
        """TrainingWatchdogDaemon should use provided config."""
        config = TrainingWatchdogConfig(stale_threshold_seconds=1800)
        daemon = TrainingWatchdogDaemon(config=config)
        assert daemon._daemon_config.stale_threshold_seconds == 1800

    def test_initial_statistics(self):
        """TrainingWatchdogDaemon should have zeroed statistics initially."""
        daemon = TrainingWatchdogDaemon()
        assert daemon._processes_registered == 0
        assert daemon._processes_killed == 0
        assert daemon._heartbeats_received == 0
        assert daemon._locks_released == 0

    def test_config_property(self):
        """config property should return daemon configuration."""
        config = TrainingWatchdogConfig(check_interval_seconds=90)
        daemon = TrainingWatchdogDaemon(config=config)
        assert daemon.config.check_interval_seconds == 90


# =============================================================================
# Database Operations Tests
# =============================================================================


class TestTrainingWatchdogDatabaseOps:
    """Tests for TrainingWatchdogDaemon database operations."""

    @pytest.fixture
    def temp_db_daemon(self, tmp_path):
        """Create daemon with temporary database."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            enable_process_kill=False,
        )
        daemon = TrainingWatchdogDaemon(config=config)
        return daemon

    def test_ensure_db_initialized_creates_tables(self, temp_db_daemon):
        """_ensure_db_initialized should create required tables."""
        temp_db_daemon._ensure_db_initialized()

        # Verify table exists
        with sqlite3.connect(str(temp_db_daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='training_processes'"
            )
            assert cursor.fetchone() is not None

    def test_ensure_db_initialized_idempotent(self, temp_db_daemon):
        """_ensure_db_initialized should be safe to call multiple times."""
        temp_db_daemon._ensure_db_initialized()
        temp_db_daemon._ensure_db_initialized()
        assert temp_db_daemon._db_initialized is True

    def test_register_training_process(self, temp_db_daemon):
        """register_training_process should insert process into database."""
        temp_db_daemon.register_training_process(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        assert temp_db_daemon._processes_registered == 1

        # Verify record in database
        with sqlite3.connect(str(temp_db_daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT config_key, pid, node_id, status FROM training_processes"
            )
            row = cursor.fetchone()
            assert row == ("hex8_2p", 12345, "worker-1", "running")

    def test_register_training_process_default_node_id(self, temp_db_daemon):
        """register_training_process should use hostname as default node_id."""
        with patch("socket.gethostname", return_value="test-host"):
            temp_db_daemon._node_id = "test-host"
            temp_db_daemon.register_training_process(
                config_key="hex8_2p",
                pid=12345,
            )

        with sqlite3.connect(str(temp_db_daemon._db_path)) as conn:
            cursor = conn.execute("SELECT node_id FROM training_processes")
            row = cursor.fetchone()
            assert row[0] == "test-host"

    def test_heartbeat_updates_timestamp(self, temp_db_daemon):
        """heartbeat should update last_heartbeat timestamp."""
        # First register
        temp_db_daemon.register_training_process(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        # Wait a tiny bit then heartbeat
        time.sleep(0.01)
        result = temp_db_daemon.heartbeat(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        assert result is True
        assert temp_db_daemon._heartbeats_received == 1

    def test_heartbeat_auto_registers(self, temp_db_daemon):
        """heartbeat should auto-register if process not found."""
        result = temp_db_daemon.heartbeat(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        assert result is True
        # Should have registered + received heartbeat
        assert temp_db_daemon._processes_registered == 1

    def test_mark_completed(self, temp_db_daemon):
        """mark_completed should update status to completed."""
        temp_db_daemon.register_training_process(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        temp_db_daemon.mark_completed(config_key="hex8_2p", node_id="worker-1")

        with sqlite3.connect(str(temp_db_daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM training_processes WHERE config_key = ? AND node_id = ?",
                ("hex8_2p", "worker-1"),
            )
            row = cursor.fetchone()
            assert row[0] == "completed"


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestTrainingWatchdogEventHandlers:
    """Tests for TrainingWatchdogDaemon event handlers."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with temporary database."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            enable_process_kill=False,
        )
        return TrainingWatchdogDaemon(config=config)

    def test_get_event_subscriptions(self, daemon):
        """_get_event_subscriptions should return training event handlers."""
        subs = daemon._get_event_subscriptions()
        assert "training_heartbeat" in subs
        assert "training_lock_acquired" in subs
        assert "training_completed" in subs

    @pytest.mark.asyncio
    async def test_on_training_heartbeat(self, daemon):
        """_on_training_heartbeat should call heartbeat method."""
        event = {
            "config_key": "hex8_2p",
            "pid": 12345,
            "node_id": "worker-1",
        }

        await daemon._on_training_heartbeat(event)

        # Should have auto-registered via heartbeat
        assert daemon._processes_registered == 1

    @pytest.mark.asyncio
    async def test_on_training_heartbeat_empty_event(self, daemon):
        """_on_training_heartbeat should handle empty event gracefully."""
        event = {}

        # Should not raise
        await daemon._on_training_heartbeat(event)
        assert daemon._processes_registered == 0

    @pytest.mark.asyncio
    async def test_on_training_lock_acquired(self, daemon):
        """_on_training_lock_acquired should register process."""
        event = {
            "config_key": "hex8_2p",
            "pid": 12345,
            "node_id": "worker-1",
        }

        await daemon._on_training_lock_acquired(event)

        assert daemon._processes_registered == 1

    @pytest.mark.asyncio
    async def test_on_training_completed(self, daemon):
        """_on_training_completed should mark process completed."""
        # First register
        daemon.register_training_process(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        event = {
            "config_key": "hex8_2p",
            "node_id": "worker-1",
        }

        await daemon._on_training_completed(event)

        # Verify marked completed
        with sqlite3.connect(str(daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM training_processes WHERE config_key = ?",
                ("hex8_2p",),
            )
            row = cursor.fetchone()
            assert row[0] == "completed"


# =============================================================================
# Stale Process Detection Tests
# =============================================================================


class TestTrainingWatchdogStaleDetection:
    """Tests for stale process detection."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with short stale threshold."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            stale_threshold_seconds=60,  # 1 minute
            enable_process_kill=False,
        )
        return TrainingWatchdogDaemon(config=config)

    def test_find_stale_processes_empty(self, daemon):
        """_find_stale_processes should return empty list when no stale processes."""
        daemon._ensure_db_initialized()
        stale = daemon._find_stale_processes()
        assert stale == []

    def test_find_stale_processes_none_stale(self, daemon):
        """_find_stale_processes should not return fresh processes."""
        daemon.register_training_process(
            config_key="hex8_2p",
            pid=12345,
            node_id="worker-1",
        )

        stale = daemon._find_stale_processes()
        assert stale == []

    def test_find_stale_processes_returns_stale(self, daemon):
        """_find_stale_processes should return stale processes."""
        # Insert process with old heartbeat
        daemon._ensure_db_initialized()
        old_time = time.time() - 120  # 2 minutes ago (stale threshold is 60s)

        with sqlite3.connect(str(daemon._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO training_processes
                (config_key, pid, node_id, started_at, last_heartbeat, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                ("hex8_2p", 12345, "worker-1", old_time, old_time),
            )
            conn.commit()

        stale = daemon._find_stale_processes()
        assert len(stale) == 1
        assert stale[0].config_key == "hex8_2p"
        assert stale[0].pid == 12345


# =============================================================================
# Process Killing Tests
# =============================================================================


class TestTrainingWatchdogProcessKilling:
    """Tests for process killing functionality."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with temporary database."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            enable_process_kill=True,
            kill_grace_period_seconds=0.1,  # Fast for testing
        )
        daemon = TrainingWatchdogDaemon(config=config)
        daemon._node_id = "test-node"
        return daemon

    def test_process_exists_false_for_nonexistent(self, daemon):
        """_process_exists should return False for nonexistent PID."""
        # Use a very high PID that's unlikely to exist
        assert daemon._process_exists(99999999) is False

    def test_process_exists_true_for_current(self, daemon):
        """_process_exists should return True for current process."""
        assert daemon._process_exists(os.getpid()) is True

    @pytest.mark.asyncio
    async def test_kill_local_process_disabled(self, tmp_path):
        """_kill_local_process should skip when kill disabled."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            enable_process_kill=False,
        )
        daemon = TrainingWatchdogDaemon(config=config)
        daemon._node_id = "test-node"

        proc = TrainingProcessInfo(
            config_key="hex8_2p",
            pid=12345,
            node_id="test-node",
            started_at=time.time() - 100,
            last_heartbeat=time.time() - 100,
        )

        # Register process first
        daemon.register_training_process("hex8_2p", 12345, "test-node")

        await daemon._kill_local_process(proc)

        # Should not increment kill count (just marked)
        assert daemon._processes_killed == 0

    @pytest.mark.asyncio
    async def test_kill_local_process_not_exists(self, daemon):
        """_kill_local_process should handle nonexistent process."""
        proc = TrainingProcessInfo(
            config_key="hex8_2p",
            pid=99999999,  # Very high PID
            node_id="test-node",
            started_at=time.time() - 100,
            last_heartbeat=time.time() - 100,
        )

        # Register first
        daemon.register_training_process("hex8_2p", 99999999, "test-node")

        await daemon._kill_local_process(proc)

        # Should mark as killed even though process didn't exist
        with sqlite3.connect(str(daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM training_processes WHERE config_key = ?",
                ("hex8_2p",),
            )
            row = cursor.fetchone()
            assert row[0] == "killed"

    def test_mark_process_killed(self, daemon):
        """_mark_process_killed should update status in database."""
        daemon.register_training_process("hex8_2p", 12345, "test-node")

        proc = TrainingProcessInfo(
            config_key="hex8_2p",
            pid=12345,
            node_id="test-node",
            started_at=time.time(),
            last_heartbeat=time.time(),
        )

        daemon._mark_process_killed(proc)

        with sqlite3.connect(str(daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM training_processes WHERE config_key = ?",
                ("hex8_2p",),
            )
            row = cursor.fetchone()
            assert row[0] == "killed"


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestTrainingWatchdogRunCycle:
    """Tests for the main watchdog cycle."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with short thresholds for testing."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(
            db_path=db_path,
            stale_threshold_seconds=1,  # Very short for testing
            enable_process_kill=False,
            release_locks_on_kill=False,
        )
        daemon = TrainingWatchdogDaemon(config=config)
        daemon._node_id = "test-node"
        return daemon

    @pytest.mark.asyncio
    async def test_run_cycle_no_stale(self, daemon):
        """_run_cycle should do nothing when no stale processes."""
        daemon._ensure_db_initialized()

        # Should not raise
        await daemon._run_cycle()

    @pytest.mark.asyncio
    async def test_run_cycle_handles_stale(self, daemon):
        """_run_cycle should handle stale processes."""
        # Insert stale process
        daemon._ensure_db_initialized()
        old_time = time.time() - 10  # Stale (threshold is 1s)

        with sqlite3.connect(str(daemon._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO training_processes
                (config_key, pid, node_id, started_at, last_heartbeat, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                ("hex8_2p", 99999999, "remote-node", old_time, old_time),
            )
            conn.commit()

        await daemon._run_cycle()

        # Should have marked as killed (remote process)
        with sqlite3.connect(str(daemon._db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM training_processes WHERE config_key = ?",
                ("hex8_2p",),
            )
            row = cursor.fetchone()
            assert row[0] == "killed"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestTrainingWatchdogHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with temporary database."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(db_path=db_path)
        return TrainingWatchdogDaemon(config=config)

    def test_health_check_stopped(self, daemon):
        """health_check should return STOPPED when not running."""
        daemon._running = False

        result = daemon.health_check()

        assert result.status.name == "STOPPED"
        assert "not running" in result.message.lower()

    def test_health_check_running(self, daemon):
        """health_check should return RUNNING when healthy."""
        daemon._running = True
        daemon._start_time = time.time() - 60

        result = daemon.health_check()

        assert result.status.name == "RUNNING"
        assert result.healthy is True

    def test_health_check_degraded_high_kills(self, daemon):
        """health_check should return DEGRADED when many processes killed."""
        daemon._running = True
        daemon._start_time = time.time() - 60
        daemon._processes_killed = 15  # > 10 threshold

        result = daemon.health_check()

        assert result.status.name == "DEGRADED"
        assert "15 processes" in result.message

    def test_health_check_includes_details(self, daemon):
        """health_check should include statistics in details."""
        daemon._running = True
        daemon._start_time = time.time() - 60
        daemon._processes_registered = 5
        daemon._heartbeats_received = 100

        result = daemon.health_check()

        assert result.details["processes_registered"] == 5
        assert result.details["heartbeats_received"] == 100


# =============================================================================
# Status and Active Processes Tests
# =============================================================================


class TestTrainingWatchdogStatus:
    """Tests for get_status and get_active_training_processes."""

    @pytest.fixture
    def daemon(self, tmp_path):
        """Create daemon with temporary database."""
        db_path = tmp_path / "test_heartbeats.db"
        config = TrainingWatchdogConfig(db_path=db_path)
        return TrainingWatchdogDaemon(config=config)

    def test_get_status(self, daemon):
        """get_status should return comprehensive status dict."""
        daemon._running = True
        daemon._start_time = time.time() - 60

        status = daemon.get_status()

        assert status["name"] == "TrainingWatchdogDaemon"
        assert status["running"] is True
        assert "uptime_seconds" in status
        assert "config" in status
        assert "health" in status

    def test_get_active_training_processes_empty(self, daemon):
        """get_active_training_processes should return empty list when none."""
        daemon._ensure_db_initialized()

        active = daemon.get_active_training_processes()

        assert active == []

    def test_get_active_training_processes_returns_running(self, daemon):
        """get_active_training_processes should return running processes."""
        daemon.register_training_process("hex8_2p", 12345, "worker-1")
        daemon.register_training_process("hex8_3p", 12346, "worker-2")

        active = daemon.get_active_training_processes()

        assert len(active) == 2
        config_keys = {p.config_key for p in active}
        assert config_keys == {"hex8_2p", "hex8_3p"}

    def test_get_active_training_processes_excludes_completed(self, daemon):
        """get_active_training_processes should exclude completed processes."""
        daemon.register_training_process("hex8_2p", 12345, "worker-1")
        daemon.mark_completed("hex8_2p", "worker-1")

        active = daemon.get_active_training_processes()

        assert len(active) == 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestTrainingWatchdogSingleton:
    """Tests for singleton pattern."""

    def test_get_returns_singleton(self):
        """get_training_watchdog_daemon should return same instance."""
        reset_training_watchdog_daemon()

        daemon1 = get_training_watchdog_daemon()
        daemon2 = get_training_watchdog_daemon()

        assert daemon1 is daemon2

    def test_reset_creates_new_instance(self):
        """reset_training_watchdog_daemon should create new instance."""
        daemon1 = get_training_watchdog_daemon()
        reset_training_watchdog_daemon()
        daemon2 = get_training_watchdog_daemon()

        assert daemon1 is not daemon2


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestSendTrainingHeartbeat:
    """Tests for send_training_heartbeat helper function."""

    def test_send_heartbeat_via_event_bus(self):
        """send_training_heartbeat should emit event via safe_emit_event."""
        with patch(
            "app.coordination.training_watchdog_daemon.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            send_training_heartbeat("hex8_2p", pid=12345)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        # safe_emit_event args: (event_name, payload, source=, context=)
        assert call_args[0][0] == "TRAINING_HEARTBEAT"
        payload = call_args[0][1]
        assert payload["config_key"] == "hex8_2p"
        assert payload["pid"] == 12345

    def test_send_heartbeat_default_pid(self):
        """send_training_heartbeat should use current PID by default."""
        with patch(
            "app.coordination.training_watchdog_daemon.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            send_training_heartbeat("hex8_2p")

        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert payload["pid"] == os.getpid()

    def test_send_heartbeat_fallback_on_error(self):
        """send_training_heartbeat should fallback to direct daemon call on error."""
        # Simulate safe_emit_event returning False (failure)
        with patch(
            "app.coordination.training_watchdog_daemon.safe_emit_event",
            return_value=False,
        ):
            # Should not raise, just silently fall back to direct daemon call
            send_training_heartbeat("hex8_2p", pid=12345)


# =============================================================================
# Cleanup Fixture
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_training_watchdog_daemon()
    yield
    reset_training_watchdog_daemon()
