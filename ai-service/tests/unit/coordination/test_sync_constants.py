"""Tests for sync_constants module.

December 27, 2025: Added to improve test coverage for critical coordination modules.
"""

import pytest

from app.coordination.sync_constants import (
    SyncState,
    SyncPriority,
    SyncDirection,
    SyncTarget,
    SyncResult,
)


class TestSyncState:
    """Tests for SyncState enum."""

    def test_all_states_defined(self):
        """Verify all expected states exist."""
        expected = {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "SKIPPED"}
        actual = {s.name for s in SyncState}
        assert expected == actual

    def test_terminal_states(self):
        """Verify terminal states are correct."""
        terminal = SyncState.terminal_states()
        assert SyncState.COMPLETED in terminal
        assert SyncState.FAILED in terminal
        assert SyncState.CANCELLED in terminal
        assert SyncState.SKIPPED in terminal
        assert SyncState.PENDING not in terminal
        assert SyncState.IN_PROGRESS not in terminal

    def test_is_terminal(self):
        """Test is_terminal method."""
        assert SyncState.COMPLETED.is_terminal()
        assert SyncState.FAILED.is_terminal()
        assert not SyncState.PENDING.is_terminal()
        assert not SyncState.IN_PROGRESS.is_terminal()

    def test_state_values(self):
        """Test state string values."""
        assert SyncState.PENDING.value == "pending"
        assert SyncState.IN_PROGRESS.value == "in_progress"
        assert SyncState.COMPLETED.value == "completed"


class TestSyncPriority:
    """Tests for SyncPriority enum."""

    def test_priority_ordering(self):
        """Verify priorities are correctly ordered."""
        assert SyncPriority.CRITICAL > SyncPriority.HIGH
        assert SyncPriority.HIGH > SyncPriority.NORMAL
        assert SyncPriority.NORMAL > SyncPriority.LOW
        assert SyncPriority.LOW > SyncPriority.BACKGROUND

    def test_priority_values(self):
        """Verify priority numeric values."""
        assert SyncPriority.CRITICAL.value == 100
        assert SyncPriority.HIGH.value == 75
        assert SyncPriority.NORMAL.value == 50
        assert SyncPriority.LOW.value == 25
        assert SyncPriority.BACKGROUND.value == 10

    def test_priority_comparison_lt(self):
        """Test less-than comparison."""
        assert SyncPriority.LOW < SyncPriority.HIGH
        assert not SyncPriority.HIGH < SyncPriority.LOW

    def test_priority_comparison_gt(self):
        """Test greater-than comparison."""
        assert SyncPriority.HIGH > SyncPriority.LOW
        assert not SyncPriority.LOW > SyncPriority.HIGH


class TestSyncDirection:
    """Tests for SyncDirection enum."""

    def test_directions_defined(self):
        """Verify all directions exist."""
        expected = {"PUSH", "PULL", "BIDIRECTIONAL"}
        actual = {d.name for d in SyncDirection}
        assert expected == actual

    def test_direction_values(self):
        """Test direction string values."""
        assert SyncDirection.PUSH.value == "push"
        assert SyncDirection.PULL.value == "pull"
        assert SyncDirection.BIDIRECTIONAL.value == "bidirectional"


class TestSyncTarget:
    """Tests for SyncTarget dataclass."""

    def test_basic_creation(self):
        """Test creating a SyncTarget with minimal args."""
        target = SyncTarget(host="example.com", path="/data/games")
        assert target.host == "example.com"
        assert target.path == "/data/games"
        assert target.port == 22
        assert target.user == "ubuntu"
        assert target.ssh_key is None

    def test_full_creation(self):
        """Test creating a SyncTarget with all args."""
        target = SyncTarget(
            host="10.0.0.1",
            path="/data",
            port=2222,
            user="root",
            ssh_key="/path/to/key",
        )
        assert target.host == "10.0.0.1"
        assert target.path == "/data"
        assert target.port == 2222
        assert target.user == "root"
        assert target.ssh_key == "/path/to/key"

    def test_ssh_spec(self):
        """Test ssh_spec property."""
        target = SyncTarget(host="example.com", path="/data", user="admin")
        assert target.ssh_spec == "admin@example.com"


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = SyncResult(
            success=True,
            files_synced=10,
            bytes_transferred=1024 * 1024,
        )
        assert result.success
        assert result.files_synced == 10
        assert result.bytes_transferred == 1024 * 1024
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = SyncResult(
            success=False,
            error="Connection timeout",
        )
        assert not result.success
        assert result.error == "Connection timeout"
        assert result.files_synced == 0

    def test_default_values(self):
        """Test default values."""
        result = SyncResult(success=True)
        assert result.files_synced == 0
        assert result.bytes_transferred == 0
        assert result.error is None
        assert result.duration_seconds == 0.0
