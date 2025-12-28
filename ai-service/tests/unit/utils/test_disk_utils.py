"""Unit tests for disk space utilities.

Tests the disk space checking functionality that prevents data loss
when disk fills up.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.errors import DiskSpaceError
from app.utils.disk_utils import (
    DEFAULT_MIN_BYTES,
    check_disk_space_available,
    ensure_disk_space,
    get_available_disk_space,
)


class TestGetAvailableDiskSpace:
    """Tests for get_available_disk_space()."""

    def test_returns_positive_for_valid_path(self, tmp_path: Path) -> None:
        """Test that available space is positive for a valid path."""
        available = get_available_disk_space(tmp_path)
        assert available > 0

    def test_works_with_file_path(self, tmp_path: Path) -> None:
        """Test that it works with a path to a non-existent file."""
        file_path = tmp_path / "nonexistent.db"
        available = get_available_disk_space(file_path)
        assert available > 0

    def test_works_with_nested_nonexistent_path(self, tmp_path: Path) -> None:
        """Test with deeply nested non-existent path."""
        nested_path = tmp_path / "a" / "b" / "c" / "file.db"
        available = get_available_disk_space(nested_path)
        assert available > 0

    def test_handles_paths_with_missing_intermediate_dirs(self, tmp_path: Path) -> None:
        """Test that it handles paths with missing intermediate directories."""
        # Even a deeply nested non-existent path should work if parent exists
        nested_path = tmp_path / "deeply" / "nested" / "path" / "file.db"
        available = get_available_disk_space(nested_path)
        assert available > 0  # Should resolve to tmp_path


class TestCheckDiskSpaceAvailable:
    """Tests for check_disk_space_available()."""

    def test_returns_true_with_reasonable_threshold(self, tmp_path: Path) -> None:
        """Test returns True when sufficient space exists."""
        # 1 byte threshold should always be available
        assert check_disk_space_available(tmp_path, min_bytes=1) is True

    def test_returns_false_with_impossible_threshold(self, tmp_path: Path) -> None:
        """Test returns False when threshold is impossibly high."""
        # 100 petabytes should never be available
        impossible_bytes = 100 * 1024**5
        assert check_disk_space_available(tmp_path, min_bytes=impossible_bytes) is False

    def test_uses_default_threshold(self, tmp_path: Path) -> None:
        """Test that default threshold is 100MB."""
        assert DEFAULT_MIN_BYTES == 100 * 1024 * 1024
        # Default check should work on any normal system
        result = check_disk_space_available(tmp_path)
        assert isinstance(result, bool)

    def test_returns_false_when_mocked_to_fail(self, tmp_path: Path) -> None:
        """Test returns False (with warning) on error."""
        # Mock the statvfs to simulate failure
        with patch("app.utils.disk_utils.get_available_disk_space") as mock:
            mock.side_effect = OSError("Mocked disk error")
            result = check_disk_space_available(tmp_path)
            assert result is False

    def test_works_with_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work."""
        result = check_disk_space_available(str(tmp_path), min_bytes=1)
        assert result is True


class TestEnsureDiskSpace:
    """Tests for ensure_disk_space()."""

    def test_succeeds_with_reasonable_threshold(self, tmp_path: Path) -> None:
        """Test succeeds when sufficient space exists."""
        # Should not raise
        ensure_disk_space(tmp_path, min_bytes=1)

    def test_raises_disk_space_error_when_insufficient(self, tmp_path: Path) -> None:
        """Test raises DiskSpaceError when threshold not met."""
        impossible_bytes = 100 * 1024**5  # 100 petabytes
        with pytest.raises(DiskSpaceError) as exc_info:
            ensure_disk_space(tmp_path, min_bytes=impossible_bytes, operation="test")

        error = exc_info.value
        assert "Insufficient disk space" in str(error)
        assert error.details["operation"] == "test"
        assert error.details["path"] == str(tmp_path)
        assert "available_bytes" in error.details
        assert "required_bytes" in error.details

    def test_raises_on_disk_check_failure(self, tmp_path: Path) -> None:
        """Test raises DiskSpaceError when disk check fails."""
        with patch("app.utils.disk_utils.get_available_disk_space") as mock:
            mock.side_effect = OSError("Mocked disk error")
            with pytest.raises(DiskSpaceError) as exc_info:
                ensure_disk_space(tmp_path, operation="test")

            error = exc_info.value
            assert "Cannot verify disk space" in str(error)

    def test_includes_operation_in_error_message(self, tmp_path: Path) -> None:
        """Test that operation name is included in error."""
        with pytest.raises(DiskSpaceError) as exc_info:
            ensure_disk_space(
                tmp_path,
                min_bytes=100 * 1024**5,
                operation="store game",
            )
        assert "store game" in str(exc_info.value)

    def test_uses_default_operation_name(self, tmp_path: Path) -> None:
        """Test default operation name is 'write'."""
        with pytest.raises(DiskSpaceError) as exc_info:
            ensure_disk_space(tmp_path, min_bytes=100 * 1024**5)
        assert "write" in str(exc_info.value)


class TestDiskSpaceErrorDetails:
    """Tests for DiskSpaceError context/details."""

    def test_error_has_path_context(self, tmp_path: Path) -> None:
        """Test error includes path in details."""
        try:
            ensure_disk_space(tmp_path, min_bytes=100 * 1024**5)
        except DiskSpaceError as e:
            assert "path" in e.details
            assert e.details["path"] == str(tmp_path)

    def test_error_has_mb_values(self, tmp_path: Path) -> None:
        """Test error includes MB values for readability."""
        try:
            ensure_disk_space(tmp_path, min_bytes=100 * 1024**5)
        except DiskSpaceError as e:
            assert "available_mb" in e.details
            assert "required_mb" in e.details
            assert isinstance(e.details["available_mb"], float)
            assert isinstance(e.details["required_mb"], float)

    def test_error_is_retryable(self) -> None:
        """Test that DiskSpaceError inherits retryable from ResourceError."""
        from app.errors import DiskSpaceError, ResourceError

        assert issubclass(DiskSpaceError, ResourceError)
        # DiskError (parent of DiskSpaceError) inherits retryable=True from ResourceError


class TestIntegrationWithGameReplay:
    """Test disk space checks in game_replay.py context."""

    def test_game_replay_imports_disk_utils(self) -> None:
        """Test that game_replay.py correctly imports ensure_disk_space."""
        from app.db import game_replay

        assert hasattr(game_replay, "ensure_disk_space")

    def test_store_methods_have_disk_check(self) -> None:
        """Test that store methods reference ensure_disk_space."""
        from app.db.game_replay import GameReplayDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = GameReplayDB(str(Path(tmpdir) / "test.db"))

            # Check bytecode for ensure_disk_space calls
            store_move_code = db._store_move_conn.__code__
            store_snapshot_code = db._store_snapshot_conn.__code__
            store_game_code = db.store_game.__code__

            assert "ensure_disk_space" in str(store_move_code.co_names)
            assert "ensure_disk_space" in str(store_snapshot_code.co_names)
            assert "ensure_disk_space" in str(store_game_code.co_names)


class TestMockedDiskSpace:
    """Tests with mocked disk space for edge cases."""

    def test_check_returns_false_when_space_below_threshold(
        self, tmp_path: Path
    ) -> None:
        """Test check returns False when available < min_bytes."""
        with patch(
            "app.utils.disk_utils.get_available_disk_space", return_value=50 * 1024**2
        ):  # 50 MB
            result = check_disk_space_available(
                tmp_path, min_bytes=100 * 1024**2
            )  # 100 MB
            assert result is False

    def test_ensure_raises_when_space_below_threshold(self, tmp_path: Path) -> None:
        """Test ensure raises when available < min_bytes."""
        with patch(
            "app.utils.disk_utils.get_available_disk_space", return_value=50 * 1024**2
        ):  # 50 MB
            with pytest.raises(DiskSpaceError) as exc_info:
                ensure_disk_space(tmp_path, min_bytes=100 * 1024**2)  # 100 MB

            error = exc_info.value
            assert error.details["available_bytes"] == 50 * 1024**2
            assert error.details["required_bytes"] == 100 * 1024**2
