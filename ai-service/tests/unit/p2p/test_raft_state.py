"""Tests for raft_state.py module.

Comprehensive tests for Raft-based replicated state machines:
- WorkItem dataclass
- JobAssignment dataclass
- Configuration helpers
- ReplicatedWorkQueue (with mocked PySyncObj)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.p2p.raft_state import (
    JobAssignment,
    WorkItem,
    load_raft_partner_addresses,
    get_self_raft_address,
    PYSYNCOBJ_AVAILABLE,
)


# =============================================================================
# WorkItem Tests
# =============================================================================


class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        item = WorkItem(work_id="test-001")
        assert item.work_id == "test-001"
        assert item.work_type == "selfplay"
        assert item.priority == 50
        assert item.status == "pending"
        assert item.claimed_by == ""
        assert item.attempts == 0
        assert item.max_attempts == 3
        assert item.timeout_seconds == 3600.0

    def test_custom_values(self):
        """Should accept custom values."""
        item = WorkItem(
            work_id="test-002",
            work_type="training",
            priority=80,
            status="running",
            claimed_by="node-1",
            attempts=2,
        )
        assert item.work_id == "test-002"
        assert item.work_type == "training"
        assert item.priority == 80
        assert item.status == "running"
        assert item.claimed_by == "node-1"
        assert item.attempts == 2

    def test_to_dict(self):
        """Should convert to dictionary."""
        item = WorkItem(
            work_id="test-003",
            work_type="eval",
            priority=60,
        )
        d = item.to_dict()
        assert d["work_id"] == "test-003"
        assert d["work_type"] == "eval"
        assert d["priority"] == 60
        assert "created_at" in d
        assert "status" in d

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "work_id": "test-004",
            "work_type": "selfplay",
            "priority": 75,
            "status": "claimed",
            "claimed_by": "node-2",
            "attempts": 1,
            "max_attempts": 3,
            "timeout_seconds": 1800.0,
            "config": {"board": "hex8"},
            "created_at": time.time(),
            "claimed_at": 0.0,
            "started_at": 0.0,
            "completed_at": 0.0,
            "result": {},
            "error": "",
        }
        item = WorkItem.from_dict(data)
        assert item.work_id == "test-004"
        assert item.priority == 75
        assert item.status == "claimed"
        assert item.config == {"board": "hex8"}


class TestWorkItemIsClaimable:
    """Tests for WorkItem.is_claimable()."""

    def test_pending_item_is_claimable(self):
        """Pending item with attempts left should be claimable."""
        item = WorkItem(work_id="test", status="pending", attempts=0)
        assert item.is_claimable() is True

    def test_claimed_item_not_claimable(self):
        """Claimed item should not be claimable."""
        item = WorkItem(work_id="test", status="claimed", attempts=0)
        assert item.is_claimable() is False

    def test_running_item_not_claimable(self):
        """Running item should not be claimable."""
        item = WorkItem(work_id="test", status="running", attempts=0)
        assert item.is_claimable() is False

    def test_completed_item_not_claimable(self):
        """Completed item should not be claimable."""
        item = WorkItem(work_id="test", status="completed", attempts=0)
        assert item.is_claimable() is False

    def test_max_attempts_reached(self):
        """Item at max attempts should not be claimable."""
        item = WorkItem(work_id="test", status="pending", attempts=3, max_attempts=3)
        assert item.is_claimable() is False

    def test_under_max_attempts(self):
        """Item under max attempts should be claimable."""
        item = WorkItem(work_id="test", status="pending", attempts=2, max_attempts=3)
        assert item.is_claimable() is True


class TestWorkItemIsTimedOut:
    """Tests for WorkItem.is_timed_out()."""

    def test_pending_item_not_timed_out(self):
        """Pending item should not be timed out."""
        item = WorkItem(work_id="test", status="pending")
        assert item.is_timed_out() is False

    def test_completed_item_not_timed_out(self):
        """Completed item should not be timed out."""
        item = WorkItem(work_id="test", status="completed")
        assert item.is_timed_out() is False

    def test_claimed_not_expired(self):
        """Recently claimed item should not be timed out."""
        item = WorkItem(
            work_id="test",
            status="claimed",
            claimed_at=time.time(),
            timeout_seconds=3600.0,
        )
        assert item.is_timed_out() is False

    def test_claimed_expired(self):
        """Old claimed item should be timed out."""
        item = WorkItem(
            work_id="test",
            status="claimed",
            claimed_at=time.time() - 7200,  # 2 hours ago
            timeout_seconds=3600.0,  # 1 hour timeout
        )
        assert item.is_timed_out() is True

    def test_running_expired(self):
        """Old running item should be timed out."""
        item = WorkItem(
            work_id="test",
            status="running",
            claimed_at=time.time() - 4000,  # More than timeout
            timeout_seconds=3600.0,
        )
        assert item.is_timed_out() is True

    def test_claimed_at_zero_not_timed_out(self):
        """Item with claimed_at=0 should not be timed out."""
        item = WorkItem(
            work_id="test",
            status="claimed",
            claimed_at=0.0,
        )
        assert item.is_timed_out() is False


# =============================================================================
# JobAssignment Tests
# =============================================================================


class TestJobAssignment:
    """Tests for JobAssignment dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        job = JobAssignment(job_id="job-001", node_id="node-1")
        assert job.job_id == "job-001"
        assert job.node_id == "node-1"
        assert job.job_type == "selfplay"
        assert job.board_type == "square8"
        assert job.num_players == 2
        assert job.status == "assigned"

    def test_custom_values(self):
        """Should accept custom values."""
        job = JobAssignment(
            job_id="job-002",
            node_id="node-2",
            job_type="training",
            board_type="hex8",
            num_players=4,
            status="running",
        )
        assert job.job_type == "training"
        assert job.board_type == "hex8"
        assert job.num_players == 4
        assert job.status == "running"

    def test_to_dict(self):
        """Should convert to dictionary."""
        job = JobAssignment(job_id="job-003", node_id="node-3")
        d = job.to_dict()
        assert d["job_id"] == "job-003"
        assert d["node_id"] == "node-3"
        assert "assigned_at" in d
        assert "status" in d

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "job_id": "job-004",
            "node_id": "node-4",
            "job_type": "eval",
            "board_type": "hexagonal",
            "num_players": 3,
            "assigned_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
            "status": "assigned",
            "result": {},
            "error": "",
        }
        job = JobAssignment.from_dict(data)
        assert job.job_id == "job-004"
        assert job.job_type == "eval"
        assert job.num_players == 3


# =============================================================================
# Configuration Helper Tests
# =============================================================================


class TestLoadRaftPartnerAddresses:
    """Tests for load_raft_partner_addresses()."""

    def test_with_yaml_file(self):
        """Should load partners from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
hosts:
  node-1:
    ssh_host: 192.168.1.1
  node-2:
    ssh_host: 192.168.1.2
  node-3:
    ssh_host: 192.168.1.3
p2p_voters:
  - node-1
  - node-2
  - node-3
""")
            config_path = Path(f.name)

        try:
            # Mock cluster_config to force YAML fallback
            with patch("app.p2p.raft_state.HAS_CLUSTER_CONFIG", False):
                partners = load_raft_partner_addresses(
                    "node-1",
                    config_path=config_path,
                    bind_port=4321,
                )
            # Should exclude self (node-1) and include others
            assert "192.168.1.2:4321" in partners
            assert "192.168.1.3:4321" in partners
            assert "192.168.1.1:4321" not in partners
        finally:
            config_path.unlink()

    def test_missing_config_file(self):
        """Should return empty list for missing file."""
        with patch("app.p2p.raft_state.HAS_CLUSTER_CONFIG", False):
            partners = load_raft_partner_addresses(
                "node-1",
                config_path=Path("/nonexistent/config.yaml"),
            )
        assert partners == []

    def test_excludes_localhost(self):
        """Should exclude localhost addresses."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
hosts:
  node-1:
    ssh_host: localhost
  node-2:
    ssh_host: 127.0.0.1
  node-3:
    ssh_host: 192.168.1.3
p2p_voters:
  - node-1
  - node-2
  - node-3
""")
            config_path = Path(f.name)

        try:
            with patch("app.p2p.raft_state.HAS_CLUSTER_CONFIG", False):
                partners = load_raft_partner_addresses(
                    "other-node",
                    config_path=config_path,
                )
            # Should only include non-localhost
            assert len(partners) == 1
            assert "192.168.1.3:4321" in partners
        finally:
            config_path.unlink()

    def test_with_cluster_config(self):
        """Should use cluster_config when available."""
        mock_nodes = {
            "node-1": MagicMock(best_ip="10.0.0.1"),
            "node-2": MagicMock(best_ip="10.0.0.2"),
            "node-3": MagicMock(best_ip="10.0.0.3"),
        }

        with patch("app.p2p.raft_state.HAS_CLUSTER_CONFIG", True), \
             patch("app.p2p.raft_state.get_raft_members", return_value=["node-1", "node-2", "node-3"]), \
             patch("app.p2p.raft_state.get_cluster_nodes", return_value=mock_nodes):
            partners = load_raft_partner_addresses("node-1", bind_port=4321)

        assert "10.0.0.2:4321" in partners
        assert "10.0.0.3:4321" in partners
        assert "10.0.0.1:4321" not in partners  # Self excluded


class TestGetSelfRaftAddress:
    """Tests for get_self_raft_address()."""

    def test_returns_address_format(self):
        """Should return address in host:port format if determinable."""
        with patch("socket.gethostname", return_value="test-host"), \
             patch("socket.gethostbyname", return_value="10.0.0.1"):
            addr = get_self_raft_address(bind_port=4321)

        if addr is not None:
            assert ":" in addr
            host, port = addr.rsplit(":", 1)
            assert port == "4321"

    def test_excludes_localhost(self):
        """Should not return localhost address."""
        with patch("socket.gethostname", return_value="localhost"), \
             patch("socket.gethostbyname", return_value="127.0.0.1"), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            addr = get_self_raft_address()

        # Should either return None or a non-localhost address
        if addr is not None:
            assert "127.0.0.1" not in addr
            assert "localhost" not in addr


# =============================================================================
# ReplicatedWorkQueue Tests (Mocked)
# =============================================================================


@pytest.mark.skipif(not PYSYNCOBJ_AVAILABLE, reason="pysyncobj not installed")
class TestReplicatedWorkQueue:
    """Tests for ReplicatedWorkQueue class."""

    def test_init_requires_pysyncobj(self):
        """Should require pysyncobj to be installed."""
        # This test verifies the import check
        assert PYSYNCOBJ_AVAILABLE is True


class TestReplicatedWorkQueueWithoutPySyncObj:
    """Tests for the PySyncObj fallback surfaces."""

    def test_repl_dict_surface(self):
        """ReplDict should expose the expected local/test surface."""
        from app.p2p import raft_state

        if raft_state.PYSYNCOBJ_AVAILABLE:
            assert raft_state.ReplDict.__module__.startswith("pysyncobj")
            return

        d = raft_state.ReplDict()
        d["key1"] = "value1"
        assert d["key1"] == "value1"
        assert "key1" in d
        assert d.get("key2") is None
        assert d.get("key2", "default") == "default"

    def test_repl_lock_manager_surface(self):
        """ReplLockManager should be available in both real and stub modes."""
        from app.p2p import raft_state

        lm = raft_state.ReplLockManager(autoUnlockTime=60)

        assert lm is not None


# =============================================================================
# RAFT Constants Tests
# =============================================================================


class TestRaftConstants:
    """Tests for Raft-related constants."""

    def test_constants_are_defined(self):
        """Should have Raft constants defined."""
        from app.p2p.constants import (
            RAFT_BIND_PORT,
            RAFT_ENABLED,
            RAFT_AUTO_UNLOCK_TIME,
            RAFT_COMPACTION_MIN_ENTRIES,
        )

        assert isinstance(RAFT_BIND_PORT, int)
        assert RAFT_BIND_PORT > 0
        assert isinstance(RAFT_ENABLED, bool)
        assert isinstance(RAFT_AUTO_UNLOCK_TIME, (int, float))
        assert isinstance(RAFT_COMPACTION_MIN_ENTRIES, int)
