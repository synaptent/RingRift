"""Tests for StateManager: SQLite persistence for P2P orchestrator state."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.managers.state_manager import (
    PersistedLeaderState,
    PersistedState,
    StateManager,
)


# Mock NodeInfo for testing
@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""

    node_id: str
    host: str = "localhost"
    port: int = 8770
    last_heartbeat: float = 0.0
    _extra: dict = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "last_heartbeat": self.last_heartbeat,
            **(self._extra or {}),
        }


# Mock ClusterJob for testing
@dataclass
class MockClusterJob:
    """Mock ClusterJob for testing."""

    job_id: str
    job_type: str = "selfplay"
    node_id: str = "node-1"
    board_type: str = "hex8"
    num_players: int = 2
    engine_mode: str = "gumbel-mcts"
    pid: int = 12345
    started_at: float = 0.0
    status: str = "running"


class TestPersistedLeaderState:
    """Test PersistedLeaderState dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        state = PersistedLeaderState()

        assert state.leader_id == ""
        assert state.leader_lease_id == ""
        assert state.leader_lease_expires == 0.0
        assert state.last_lease_renewal == 0.0
        assert state.role == "follower"
        assert state.voter_grant_leader_id == ""
        assert state.voter_grant_lease_id == ""
        assert state.voter_grant_expires == 0.0
        assert state.voter_node_ids == []
        assert state.voter_config_source == ""

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        state = PersistedLeaderState(
            leader_id="node-1",
            leader_lease_id="lease-123",
            leader_lease_expires=1000.0,
            last_lease_renewal=900.0,
            role="leader",
            voter_grant_leader_id="node-2",
            voter_grant_lease_id="grant-456",
            voter_grant_expires=1100.0,
            voter_node_ids=["node-1", "node-2", "node-3"],
            voter_config_source="cluster",
        )

        assert state.leader_id == "node-1"
        assert state.leader_lease_id == "lease-123"
        assert state.leader_lease_expires == 1000.0
        assert state.last_lease_renewal == 900.0
        assert state.role == "leader"
        assert state.voter_grant_leader_id == "node-2"
        assert state.voter_grant_lease_id == "grant-456"
        assert state.voter_grant_expires == 1100.0
        assert state.voter_node_ids == ["node-1", "node-2", "node-3"]
        assert state.voter_config_source == "cluster"


class TestPersistedState:
    """Test PersistedState dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        state = PersistedState()

        assert state.peers == {}
        assert state.jobs == []
        assert isinstance(state.leader_state, PersistedLeaderState)

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        peers = {"node-1": {"host": "10.0.0.1"}}
        jobs = [{"job_id": "job-1"}]
        leader_state = PersistedLeaderState(leader_id="node-1", role="leader")

        state = PersistedState(
            peers=peers,
            jobs=jobs,
            leader_state=leader_state,
        )

        assert state.peers == peers
        assert state.jobs == jobs
        assert state.leader_state.leader_id == "node-1"
        assert state.leader_state.role == "leader"


class TestStateManagerInit:
    """Test StateManager initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates parent directory."""
        db_path = tmp_path / "subdir" / "test.db"

        manager = StateManager(db_path)

        assert db_path.parent.exists()

    def test_init_sets_attributes(self, tmp_path):
        """Test that initialization sets attributes correctly."""
        db_path = tmp_path / "test.db"

        manager = StateManager(db_path, verbose=True)

        assert manager.db_path == db_path
        assert manager.verbose is True
        assert manager._cluster_epoch == 0


class TestStateManagerDatabaseInit:
    """Test StateManager database initialization."""

    def test_init_database_creates_tables(self, tmp_path):
        """Test that init_database creates all required tables."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)

        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "peers",
            "jobs",
            "state",
            "metrics_history",
            "ab_tests",
            "ab_test_games",
            "peer_cache",
            "config",
        }
        assert expected_tables.issubset(tables)

        conn.close()

    def test_init_database_creates_indexes(self, tmp_path):
        """Test that init_database creates all required indexes."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)

        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_metrics_type_time",
            "idx_metrics_config",
            "idx_ab_games_test",
            "idx_peer_cache_reputation",
        }
        assert expected_indexes.issubset(indexes)

        conn.close()

    def test_init_database_sets_wal_mode(self, tmp_path):
        """Test that init_database sets WAL mode."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)

        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"

        conn.close()


class TestStateManagerLoadState:
    """Test StateManager load_state method."""

    def test_load_state_empty_database(self, tmp_path):
        """Test loading state from empty database."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        state = manager.load_state("self-node")

        assert state.peers == {}
        assert state.jobs == []
        assert state.leader_state.leader_id == ""

    def test_load_state_excludes_self(self, tmp_path):
        """Test that loading state excludes self node from peers."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        # Insert peers including self
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO peers VALUES (?, ?, ?, ?, ?)",
            ("self-node", "localhost", 8770, time.time(), '{"node_id": "self-node"}'),
        )
        cursor.execute(
            "INSERT INTO peers VALUES (?, ?, ?, ?, ?)",
            ("other-node", "10.0.0.1", 8770, time.time(), '{"node_id": "other-node"}'),
        )
        conn.commit()
        conn.close()

        state = manager.load_state("self-node")

        assert "self-node" not in state.peers
        assert "other-node" in state.peers

    def test_load_state_loads_running_jobs(self, tmp_path):
        """Test that loading state loads only running jobs."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-1", "selfplay", "node-1", "hex8", 2, "gumbel", 123, time.time(), "running"),
        )
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-2", "training", "node-2", "square8", 4, "mcts", 456, time.time(), "completed"),
        )
        conn.commit()
        conn.close()

        state = manager.load_state("self-node")

        assert len(state.jobs) == 1
        assert state.jobs[0]["job_id"] == "job-1"
        assert state.jobs[0]["status"] == "running"

    def test_load_state_loads_leader_state(self, tmp_path):
        """Test that loading state loads leader election state."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO state VALUES (?, ?)",
            [
                ("leader_id", "leader-node"),
                ("leader_lease_id", "lease-123"),
                ("leader_lease_expires", "1000.0"),
                ("role", "leader"),
                ("voter_node_ids", '["node-1", "node-2", "node-3"]'),
            ],
        )
        conn.commit()
        conn.close()

        state = manager.load_state("self-node")

        assert state.leader_state.leader_id == "leader-node"
        assert state.leader_state.leader_lease_id == "lease-123"
        assert state.leader_state.leader_lease_expires == 1000.0
        assert state.leader_state.role == "leader"
        assert state.leader_state.voter_node_ids == ["node-1", "node-2", "node-3"]

    def test_load_state_handles_voter_config_csv(self, tmp_path):
        """Test that loading state handles CSV voter config."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO state VALUES (?, ?)", ("voter_node_ids", "node-1, node-2, node-3"))
        conn.commit()
        conn.close()

        state = manager.load_state("self-node")

        assert state.leader_state.voter_node_ids == ["node-1", "node-2", "node-3"]


class TestStateManagerSaveState:
    """Test StateManager save_state method."""

    def test_save_state_saves_peers(self, tmp_path):
        """Test that save_state saves peer information."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        peers = {
            "peer-1": MockNodeInfo("peer-1", "10.0.0.1", 8770),
            "peer-2": MockNodeInfo("peer-2", "10.0.0.2", 8770),
        }
        leader_state = PersistedLeaderState()

        manager.save_state("self-node", peers, {}, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT node_id FROM peers ORDER BY node_id")
        saved_peers = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert saved_peers == ["peer-1", "peer-2"]

    def test_save_state_excludes_self_from_peers(self, tmp_path):
        """Test that save_state excludes self node from peers."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        peers = {
            "self-node": MockNodeInfo("self-node", "localhost", 8770),
            "peer-1": MockNodeInfo("peer-1", "10.0.0.1", 8770),
        }
        leader_state = PersistedLeaderState()

        manager.save_state("self-node", peers, {}, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT node_id FROM peers ORDER BY node_id")
        saved_peers = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert saved_peers == ["peer-1"]

    def test_save_state_saves_jobs(self, tmp_path):
        """Test that save_state saves job information."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        jobs = {
            "job-1": MockClusterJob("job-1", "selfplay", "node-1", "hex8", 2),
            "job-2": MockClusterJob("job-2", "training", "node-2", "square8", 4),
        }
        leader_state = PersistedLeaderState()

        manager.save_state("self-node", {}, jobs, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, job_type FROM jobs ORDER BY job_id")
        saved_jobs = cursor.fetchall()
        conn.close()

        assert len(saved_jobs) == 2
        assert saved_jobs[0] == ("job-1", "selfplay")
        assert saved_jobs[1] == ("job-2", "training")

    def test_save_state_saves_leader_state(self, tmp_path):
        """Test that save_state saves leader election state."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        leader_state = PersistedLeaderState(
            leader_id="leader-node",
            leader_lease_id="lease-123",
            leader_lease_expires=1000.0,
            role="leader",
            voter_node_ids=["node-1", "node-2"],
        )

        manager.save_state("self-node", {}, {}, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM state WHERE key = 'leader_id'")
        leader_id = cursor.fetchone()[0]
        cursor.execute("SELECT value FROM state WHERE key = 'role'")
        role = cursor.fetchone()[0]
        cursor.execute("SELECT value FROM state WHERE key = 'voter_node_ids'")
        voters = json.loads(cursor.fetchone()[0])
        conn.close()

        assert leader_id == "leader-node"
        assert role == "leader"
        assert voters == ["node-1", "node-2"]

    def test_save_state_handles_dict_peers(self, tmp_path):
        """Test that save_state handles dict-based peers."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        peers = {
            "peer-1": {"node_id": "peer-1", "host": "10.0.0.1", "port": 8770, "last_heartbeat": time.time()},
        }
        leader_state = PersistedLeaderState()

        manager.save_state("self-node", peers, {}, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT node_id, host FROM peers")
        saved = cursor.fetchone()
        conn.close()

        assert saved == ("peer-1", "10.0.0.1")

    def test_save_state_handles_dict_jobs(self, tmp_path):
        """Test that save_state handles dict-based jobs."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        jobs = {
            "job-1": {
                "job_id": "job-1",
                "job_type": "selfplay",
                "node_id": "node-1",
                "board_type": "hex8",
                "num_players": 2,
            },
        }
        leader_state = PersistedLeaderState()

        manager.save_state("self-node", {}, jobs, leader_state)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, job_type FROM jobs")
        saved = cursor.fetchone()
        conn.close()

        assert saved == ("job-1", "selfplay")


class TestClusterEpoch:
    """Test cluster epoch management."""

    def test_get_cluster_epoch_default(self, tmp_path):
        """Test that default epoch is 0."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)

        assert manager.get_cluster_epoch() == 0

    def test_load_cluster_epoch(self, tmp_path):
        """Test loading cluster epoch from database."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO config VALUES ('cluster_epoch', '42')")
        conn.commit()
        conn.close()

        epoch = manager.load_cluster_epoch()

        assert epoch == 42
        assert manager.get_cluster_epoch() == 42

    def test_save_cluster_epoch(self, tmp_path):
        """Test saving cluster epoch to database."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        manager._cluster_epoch = 100
        manager.save_cluster_epoch()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = 'cluster_epoch'")
        saved = cursor.fetchone()[0]
        conn.close()

        assert saved == "100"

    def test_increment_cluster_epoch(self, tmp_path):
        """Test incrementing cluster epoch."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        initial = manager.get_cluster_epoch()
        new_epoch = manager.increment_cluster_epoch()

        assert new_epoch == initial + 1
        assert manager.get_cluster_epoch() == initial + 1

    def test_set_cluster_epoch(self, tmp_path):
        """Test setting cluster epoch to specific value."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        manager.set_cluster_epoch(500)

        assert manager.get_cluster_epoch() == 500


class TestJobOperations:
    """Test job-related database operations."""

    def test_delete_job(self, tmp_path):
        """Test deleting a job from database."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-1", "selfplay", "node-1", "hex8", 2, "gumbel", 123, time.time(), "running"),
        )
        conn.commit()
        conn.close()

        manager.delete_job("job-1")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE job_id = 'job-1'")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_delete_nonexistent_job(self, tmp_path):
        """Test deleting a nonexistent job doesn't error."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        # Should not raise
        manager.delete_job("nonexistent")

    def test_update_job_status(self, tmp_path):
        """Test updating a job's status."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-1", "selfplay", "node-1", "hex8", 2, "gumbel", 123, time.time(), "running"),
        )
        conn.commit()
        conn.close()

        manager.update_job_status("job-1", "completed")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM jobs WHERE job_id = 'job-1'")
        status = cursor.fetchone()[0]
        conn.close()

        assert status == "completed"

    def test_clear_stale_jobs(self, tmp_path):
        """Test clearing non-running jobs."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-1", "selfplay", "node-1", "hex8", 2, "gumbel", 123, time.time(), "running"),
        )
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-2", "training", "node-2", "square8", 4, "mcts", 456, time.time(), "completed"),
        )
        cursor.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-3", "selfplay", "node-3", "hex8", 2, "gumbel", 789, time.time(), "failed"),
        )
        conn.commit()
        conn.close()

        cleared = manager.clear_stale_jobs()

        assert cleared == 2

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT job_id FROM jobs")
        remaining = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert remaining == ["job-1"]


class TestThreadSafety:
    """Test thread safety of StateManager operations."""

    def test_concurrent_epoch_increment(self, tmp_path):
        """Test that concurrent epoch increments are safe."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        results = []

        def incrementer():
            for _ in range(10):
                epoch = manager.increment_cluster_epoch()
                results.append(epoch)

        threads = [threading.Thread(target=incrementer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be unique (each increment should be atomic)
        assert len(results) == 50
        # Final epoch should be 50
        assert manager.get_cluster_epoch() == 50

    def test_save_state_with_locks(self, tmp_path):
        """Test save_state with thread locks."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        peers_lock = threading.Lock()
        jobs_lock = threading.Lock()

        peers = {"peer-1": MockNodeInfo("peer-1", "10.0.0.1")}
        jobs = {"job-1": MockClusterJob("job-1")}
        leader_state = PersistedLeaderState()

        # Should not deadlock
        manager.save_state(
            "self-node",
            peers,
            jobs,
            leader_state,
            peers_lock=peers_lock,
            jobs_lock=jobs_lock,
        )

        state = manager.load_state("self-node")
        assert "peer-1" in state.peers
        assert len(state.jobs) == 1


class TestErrorHandling:
    """Test error handling in StateManager."""

    def test_load_state_handles_malformed_json(self, tmp_path):
        """Test that load_state handles malformed JSON gracefully."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO peers VALUES (?, ?, ?, ?, ?)",
            ("peer-1", "10.0.0.1", 8770, time.time(), "not valid json"),
        )
        conn.commit()
        conn.close()

        # Should not raise, should log error and continue
        state = manager.load_state("self-node")

        # Malformed peer should be skipped
        assert "peer-1" not in state.peers

    def test_load_state_handles_database_error(self, tmp_path):
        """Test that load_state handles database errors gracefully."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        # Don't init database - will cause errors

        # Should not raise, should return empty state
        state = manager.load_state("self-node")

        assert state.peers == {}
        assert state.jobs == []


class TestDatabaseConnection:
    """Test database connection handling."""

    def test_db_connect_settings(self, tmp_path):
        """Test that _db_connect sets proper settings."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        manager.init_database()

        conn = manager._db_connect()

        # Check timeout is set
        cursor = conn.cursor()
        cursor.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]

        assert timeout == 30000  # 30 seconds

        conn.close()

    def test_db_timeout_constant(self):
        """Test that DB_TIMEOUT constant is set."""
        assert StateManager.DB_TIMEOUT == 30.0
