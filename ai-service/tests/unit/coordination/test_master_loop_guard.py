"""Unit tests for master_loop_guard.py (December 2025).

Tests the master loop guard utilities that ensure the master loop
is running for full automation operations.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.master_loop_guard import (
    PID_FILE_PATH,
    check_or_warn,
    ensure_master_loop_running,
    is_master_loop_running,
)


class TestPidFilePath:
    """Tests for PID file path configuration."""

    def test_pid_file_path_is_pathlib_path(self):
        """Test that PID_FILE_PATH is a Path object."""
        assert isinstance(PID_FILE_PATH, Path)

    def test_pid_file_path_points_to_data_coordination(self):
        """Test that PID file is in data/coordination directory."""
        assert "data" in str(PID_FILE_PATH)
        assert "coordination" in str(PID_FILE_PATH)
        assert PID_FILE_PATH.name == "master_loop.pid"


class TestIsMasterLoopRunning:
    """Tests for is_master_loop_running() function."""

    def test_returns_false_when_pid_file_does_not_exist(self, tmp_path):
        """Test returns False when PID file doesn't exist."""
        pid_file = tmp_path / "nonexistent.pid"
        assert is_master_loop_running(pid_file) is False

    def test_returns_true_when_process_exists(self, tmp_path):
        """Test returns True when PID file contains current process PID."""
        pid_file = tmp_path / "master_loop.pid"
        current_pid = os.getpid()
        pid_file.write_text(str(current_pid))

        assert is_master_loop_running(pid_file) is True

    def test_returns_false_and_cleans_stale_pid(self, tmp_path):
        """Test returns False and removes stale PID file when process doesn't exist."""
        pid_file = tmp_path / "master_loop.pid"
        # Use a PID that definitely doesn't exist (very high number)
        stale_pid = 999999999
        pid_file.write_text(str(stale_pid))

        # Mock os.kill to raise OSError (process doesn't exist)
        with patch("os.kill", side_effect=OSError("No such process")):
            result = is_master_loop_running(pid_file)

        assert result is False
        # Stale PID file should be removed
        assert not pid_file.exists()

    def test_handles_invalid_pid_file_content(self, tmp_path):
        """Test handles non-integer content in PID file gracefully."""
        pid_file = tmp_path / "master_loop.pid"
        pid_file.write_text("not-a-number")

        assert is_master_loop_running(pid_file) is False

    def test_handles_empty_pid_file(self, tmp_path):
        """Test handles empty PID file gracefully."""
        pid_file = tmp_path / "master_loop.pid"
        pid_file.write_text("")

        assert is_master_loop_running(pid_file) is False

    def test_handles_io_error(self, tmp_path):
        """Test handles IOError when reading PID file."""
        pid_file = tmp_path / "master_loop.pid"
        pid_file.write_text("12345")

        with patch("builtins.open", side_effect=IOError("Read error")):
            assert is_master_loop_running(pid_file) is False

    def test_uses_default_pid_file_path(self):
        """Test uses default PID_FILE_PATH when no argument provided."""
        # This test just verifies the function accepts no arguments
        result = is_master_loop_running()
        # Result depends on whether master loop is actually running
        assert isinstance(result, bool)

    def test_handles_permission_error_on_stale_cleanup(self, tmp_path):
        """Test handles permission error when trying to clean stale PID file."""
        pid_file = tmp_path / "master_loop.pid"
        pid_file.write_text("999999999")

        with patch("os.kill", side_effect=OSError("No such process")):
            with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                result = is_master_loop_running(pid_file)

        # Should still return False even if cleanup fails
        assert result is False


class TestEnsureMasterLoopRunning:
    """Tests for ensure_master_loop_running() function."""

    def test_raises_runtime_error_when_not_running(self, tmp_path):
        """Test raises RuntimeError when master loop is not running."""
        pid_file = tmp_path / "master_loop.pid"
        # No PID file exists

        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_master_loop_running()

        assert "Master loop must be running" in str(exc_info.value)
        assert "python scripts/master_loop.py" in str(exc_info.value)

    def test_does_not_raise_when_running(self):
        """Test does not raise when master loop is running."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=True,
        ):
            # Should not raise
            ensure_master_loop_running()

    def test_does_not_raise_when_require_for_automation_is_false(self):
        """Test does not raise when require_for_automation is False."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            # Should not raise even if not running
            ensure_master_loop_running(require_for_automation=False)

    def test_includes_operation_name_in_error(self):
        """Test error message includes operation name."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_master_loop_running(operation_name="cluster sync")

        assert "cluster sync" in str(exc_info.value)

    def test_error_message_includes_skip_env_var(self):
        """Test error message mentions RINGRIFT_SKIP_MASTER_LOOP_CHECK."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_master_loop_running()

        assert "RINGRIFT_SKIP_MASTER_LOOP_CHECK" in str(exc_info.value)


class TestCheckOrWarn:
    """Tests for check_or_warn() function."""

    def test_returns_true_when_running(self):
        """Test returns True when master loop is running."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=True,
        ):
            assert check_or_warn() is True

    def test_returns_false_and_logs_warning_when_not_running(self, caplog):
        """Test returns False and logs warning when not running."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            result = check_or_warn("data sync")

        assert result is False
        # Check that warning was logged
        assert any("Master loop is not running" in record.message for record in caplog.records)

    def test_includes_operation_name_in_warning(self, caplog):
        """Test warning includes operation name."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            check_or_warn("tournament scheduling")

        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("tournament scheduling" in msg for msg in warning_messages)

    def test_includes_startup_command_in_warning(self, caplog):
        """Test warning includes command to start master loop."""
        with patch(
            "app.coordination.master_loop_guard.is_master_loop_running",
            return_value=False,
        ):
            check_or_warn()

        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("python scripts/master_loop.py" in msg for msg in warning_messages)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_are_accessible(self):
        """Test all items in __all__ are importable."""
        from app.coordination import master_loop_guard

        for name in master_loop_guard.__all__:
            assert hasattr(master_loop_guard, name), f"Missing export: {name}"

    def test_exports_include_all_public_functions(self):
        """Test __all__ includes all public functions."""
        from app.coordination import master_loop_guard

        expected = ["is_master_loop_running", "ensure_master_loop_running", "check_or_warn", "PID_FILE_PATH"]
        for name in expected:
            assert name in master_loop_guard.__all__


class TestIntegration:
    """Integration tests for master loop guard."""

    def test_full_workflow_with_temporary_pid_file(self, tmp_path):
        """Test complete workflow: create PID, check, cleanup."""
        pid_file = tmp_path / "master_loop.pid"

        # Initially not running
        assert is_master_loop_running(pid_file) is False

        # Write current process PID
        pid_file.write_text(str(os.getpid()))

        # Now running
        assert is_master_loop_running(pid_file) is True

        # Remove PID file
        pid_file.unlink()

        # Not running again
        assert is_master_loop_running(pid_file) is False

    def test_concurrent_check_safety(self, tmp_path):
        """Test that concurrent checks don't cause issues."""
        pid_file = tmp_path / "master_loop.pid"
        pid_file.write_text(str(os.getpid()))

        import threading

        results = []

        def check():
            results.append(is_master_loop_running(pid_file))

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should return True
        assert all(results)
        assert len(results) == 10
