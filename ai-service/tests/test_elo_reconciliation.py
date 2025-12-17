"""Tests for EloReconciler.

Tests drift detection, sync result handling, and reconciliation
reports without requiring network access.
"""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.training.elo_reconciliation import (
    ConflictResolution,
    DriftHistory,
    EloDrift,
    EloReconciler,
    ReconciliationReport,
    SyncResult,
    check_elo_drift,
)


class TestEloDrift:
    """Test EloDrift dataclass."""

    def test_basic_creation(self):
        """Test basic drift creation."""
        drift = EloDrift(
            source="/path/to/local.db",
            target="/path/to/remote.db",
            checked_at="2024-01-01T12:00:00",
            participants_in_source=100,
            participants_in_target=95,
            participants_in_both=90,
        )
        assert drift.participants_in_source == 100
        assert drift.participants_in_target == 95
        assert drift.participants_in_both == 90

    def test_max_rating_diff_empty(self):
        """Test max_rating_diff with no diffs."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=0, participants_in_target=0, participants_in_both=0,
        )
        assert drift.max_rating_diff == 0.0

    def test_max_rating_diff_with_diffs(self):
        """Test max_rating_diff calculation."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"player1": 30.0, "player2": -45.0, "player3": 20.0},
        )
        assert drift.max_rating_diff == 45.0

    def test_avg_rating_diff(self):
        """Test avg_rating_diff calculation."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"player1": 30.0, "player2": -30.0, "player3": 60.0},
        )
        # Average of absolute values: (30 + 30 + 60) / 3 = 40
        assert drift.avg_rating_diff == 40.0

    def test_is_significant_false(self):
        """Test insignificant drift."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"player1": 10.0, "player2": -5.0},
        )
        assert not drift.is_significant

    def test_is_significant_max_threshold(self):
        """Test significant drift based on max threshold."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"player1": 55.0},  # > 50 threshold
        )
        assert drift.is_significant

    def test_is_significant_avg_threshold(self):
        """Test significant drift based on avg threshold."""
        drift = EloDrift(
            source="a", target="b", checked_at="now",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"player1": 30.0, "player2": 30.0, "player3": 30.0},  # avg=30 > 25
        )
        assert drift.is_significant

    def test_to_dict(self):
        """Test serialization to dict."""
        drift = EloDrift(
            source="local.db", target="remote.db",
            checked_at="2024-01-01T12:00:00",
            participants_in_source=100,
            participants_in_target=95,
            participants_in_both=90,
            rating_diffs={"player1": 25.0},
        )
        d = drift.to_dict()

        assert d["source"] == "local.db"
        assert d["target"] == "remote.db"
        assert d["participants_in_both"] == 90
        assert d["max_rating_diff"] == 25.0
        assert d["is_significant"] is False


class TestSyncResult:
    """Test SyncResult dataclass."""

    def test_basic_creation(self):
        """Test basic sync result."""
        result = SyncResult(
            remote_host="192.168.1.100",
            synced_at="2024-01-01T12:00:00",
            matches_added=50,
            matches_skipped=10,
            matches_conflict=2,
            participants_added=5,
        )
        assert result.matches_added == 50
        assert result.matches_skipped == 10
        assert result.matches_conflict == 2
        assert result.error is None

    def test_with_error(self):
        """Test sync result with error."""
        result = SyncResult(
            remote_host="192.168.1.100",
            synced_at="2024-01-01T12:00:00",
            matches_added=0,
            matches_skipped=0,
            matches_conflict=0,
            participants_added=0,
            error="Connection refused",
        )
        assert result.error == "Connection refused"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = SyncResult(
            remote_host="host1",
            synced_at="now",
            matches_added=10,
            matches_skipped=5,
            matches_conflict=1,
            participants_added=3,
        )
        d = result.to_dict()

        assert d["remote_host"] == "host1"
        assert d["matches_added"] == 10
        assert d["error"] is None


class TestReconciliationReport:
    """Test ReconciliationReport dataclass."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = ReconciliationReport(
            started_at="2024-01-01T12:00:00",
            completed_at="2024-01-01T12:05:00",
            nodes_synced=["node1", "node2"],
            nodes_failed=["node3"],
            total_matches_added=100,
            total_matches_skipped=20,
            total_conflicts=5,
            total_resolved=3,
            drift_detected=False,
            max_drift=15.0,
        )
        assert len(report.nodes_synced) == 2
        assert len(report.nodes_failed) == 1
        assert report.total_matches_added == 100
        assert report.total_resolved == 3

    def test_summary(self):
        """Test human-readable summary."""
        report = ReconciliationReport(
            started_at="2024-01-01T12:00:00",
            completed_at="2024-01-01T12:05:00",
            nodes_synced=["node1"],
            nodes_failed=[],
            total_matches_added=50,
            total_matches_skipped=10,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=False,
            max_drift=10.0,
        )
        summary = report.summary()

        assert "Reconciliation Report" in summary
        assert "Nodes synced: 1" in summary
        assert "Matches added: 50" in summary
        assert "Resolved: 0" in summary

    def test_summary_with_drift(self):
        """Test summary shows drift warning."""
        report = ReconciliationReport(
            started_at="now",
            completed_at="now",
            nodes_synced=[],
            nodes_failed=[],
            total_matches_added=0,
            total_matches_skipped=0,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=True,
            max_drift=60.0,
        )
        summary = report.summary()

        assert "DRIFT DETECTED" in summary

    def test_summary_with_failures(self):
        """Test summary shows failed nodes."""
        report = ReconciliationReport(
            started_at="now",
            completed_at="now",
            nodes_synced=["node1"],
            nodes_failed=["node2", "node3"],
            total_matches_added=10,
            total_matches_skipped=0,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=False,
            max_drift=0.0,
        )
        summary = report.summary()

        assert "Failed nodes:" in summary
        assert "node2" in summary


class TestEloReconciler:
    """Test EloReconciler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.local_db = Path(self.temp_dir) / "local_elo.db"
        self.remote_db = Path(self.temp_dir) / "remote_elo.db"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_elo_db(self, path: Path, participants: dict):
        """Create a test Elo database."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                last_update TEXT,
                board_type TEXT,
                num_players INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE match_history (
                match_id TEXT PRIMARY KEY,
                player1_id TEXT NOT NULL,
                player2_id TEXT NOT NULL,
                winner_id TEXT,
                player1_rating_before REAL,
                player2_rating_before REAL,
                player1_rating_after REAL,
                player2_rating_after REAL,
                board_type TEXT,
                num_players INTEGER,
                game_length INTEGER,
                timestamp TEXT,
                source TEXT
            )
        """)

        for participant_id, rating in participants.items():
            cursor.execute(
                "INSERT INTO participants (participant_id, rating) VALUES (?, ?)",
                (participant_id, rating)
            )

        conn.commit()
        conn.close()

    def test_initialization(self):
        """Test reconciler initialization."""
        reconciler = EloReconciler(local_db_path=self.local_db)
        assert reconciler.local_db_path == self.local_db
        assert reconciler.ssh_timeout == 30

    def test_check_drift_no_local_db(self):
        """Test drift check when local DB doesn't exist."""
        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift()

        assert drift.participants_in_source == 0

    def test_check_drift_no_remote_db(self):
        """Test drift check when remote DB doesn't exist."""
        self._create_elo_db(self.local_db, {"player1": 1500, "player2": 1600})

        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift(remote_db_path=self.remote_db)

        assert drift.participants_in_source == 2
        assert drift.participants_in_target == 0

    def test_check_drift_identical_dbs(self):
        """Test drift check with identical databases."""
        participants = {"player1": 1500, "player2": 1600}
        self._create_elo_db(self.local_db, participants)
        self._create_elo_db(self.remote_db, participants)

        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift(remote_db_path=self.remote_db)

        assert drift.participants_in_both == 2
        assert drift.max_rating_diff == 0.0
        assert not drift.is_significant

    def test_check_drift_with_differences(self):
        """Test drift check detects rating differences."""
        self._create_elo_db(self.local_db, {"player1": 1500, "player2": 1600})
        self._create_elo_db(self.remote_db, {"player1": 1550, "player2": 1550})

        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift(remote_db_path=self.remote_db)

        assert drift.participants_in_both == 2
        assert len(drift.rating_diffs) == 2
        assert drift.max_rating_diff == 50.0  # player1: -50, player2: +50

    def test_import_matches_empty(self):
        """Test importing empty match list."""
        self._create_elo_db(self.local_db, {})

        reconciler = EloReconciler(local_db_path=self.local_db)
        result = reconciler._import_matches("host", "now", [])

        assert result.matches_added == 0
        assert result.matches_skipped == 0

    def test_import_matches_new(self):
        """Test importing new matches."""
        self._create_elo_db(self.local_db, {})

        matches = [
            {
                "match_id": "match1",
                "player1_id": "p1",
                "player2_id": "p2",
                "winner_id": "p1",
                "timestamp": "2024-01-01T12:00:00",
            },
            {
                "match_id": "match2",
                "player1_id": "p1",
                "player2_id": "p3",
                "winner_id": "p3",
                "timestamp": "2024-01-01T12:01:00",
            },
        ]

        reconciler = EloReconciler(local_db_path=self.local_db)
        result = reconciler._import_matches("host", "now", matches)

        assert result.matches_added == 2
        assert result.matches_skipped == 0
        assert result.participants_added == 3  # p1, p2, p3

    def test_import_matches_duplicates(self):
        """Test importing duplicate matches."""
        self._create_elo_db(self.local_db, {})

        # First import
        matches = [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"}]
        reconciler = EloReconciler(local_db_path=self.local_db)
        result1 = reconciler._import_matches("host", "now", matches)
        assert result1.matches_added == 1

        # Second import (duplicate)
        result2 = reconciler._import_matches("host", "now", matches)
        assert result2.matches_added == 0
        assert result2.matches_skipped == 1

    def test_import_matches_conflict(self):
        """Test importing conflicting matches."""
        self._create_elo_db(self.local_db, {})

        # First import
        match1 = [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"}]
        reconciler = EloReconciler(local_db_path=self.local_db)
        reconciler._import_matches("host", "now", match1)

        # Conflicting import (same ID, different winner)
        match2 = [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2"}]
        result = reconciler._import_matches("host", "now", match2)

        assert result.matches_conflict == 1
        assert result.matches_added == 0

    def test_reconcile_all_no_hosts(self):
        """Test reconciliation with no hosts."""
        reconciler = EloReconciler(local_db_path=self.local_db)
        report = reconciler.reconcile_all(hosts=[])

        assert len(report.nodes_synced) == 0
        assert len(report.nodes_failed) == 0
        assert report.total_matches_added == 0

    def test_drift_history_tracking_enabled(self):
        """Test that drift history is recorded when enabled."""
        participants = {"player1": 1500, "player2": 1600}
        self._create_elo_db(self.local_db, participants)
        self._create_elo_db(self.remote_db, {"player1": 1550, "player2": 1550})

        reconciler = EloReconciler(local_db_path=self.local_db, track_history=True)

        # First drift check
        reconciler.check_drift(remote_db_path=self.remote_db, board_type="square8", num_players=2)
        # Second drift check
        reconciler.check_drift(remote_db_path=self.remote_db, board_type="square8", num_players=2)

        history = reconciler.get_drift_history("square8_2")
        assert history is not None
        assert len(history.snapshots) == 2

    def test_drift_history_tracking_disabled(self):
        """Test that drift history is not recorded when disabled."""
        participants = {"player1": 1500, "player2": 1600}
        self._create_elo_db(self.local_db, participants)

        reconciler = EloReconciler(local_db_path=self.local_db, track_history=False)
        reconciler.check_drift(board_type="square8", num_players=2)

        history = reconciler.get_drift_history("square8_2")
        assert history is None

    def test_get_all_drift_histories(self):
        """Test getting all drift histories."""
        self._create_elo_db(self.local_db, {"player1": 1500})

        reconciler = EloReconciler(local_db_path=self.local_db, track_history=True)

        # Check drift for different configs
        reconciler.check_drift(board_type="square8", num_players=2)
        reconciler.check_drift(board_type="square10", num_players=4)
        reconciler.check_drift()  # All configs

        histories = reconciler.get_all_drift_histories()
        assert len(histories) == 3
        assert "square8_2" in histories
        assert "square10_4" in histories
        assert "all_all" in histories

    def test_drift_history_persistence(self):
        """Test that drift history is persisted to disk and reloaded."""
        self._create_elo_db(self.local_db, {"player1": 1500, "player2": 1600})
        self._create_elo_db(self.remote_db, {"player1": 1550, "player2": 1550})

        # Create reconciler and check drift (should persist)
        reconciler1 = EloReconciler(
            local_db_path=self.local_db,
            track_history=True,
            persist_history=True,
        )
        reconciler1.check_drift(remote_db_path=self.remote_db, board_type="square8", num_players=2)
        reconciler1.check_drift(remote_db_path=self.remote_db, board_type="square8", num_players=2)

        # Verify file was created
        history_path = self.local_db.parent / "elo_drift_history.json"
        assert history_path.exists()

        # Create new reconciler - should load history
        reconciler2 = EloReconciler(
            local_db_path=self.local_db,
            track_history=True,
            persist_history=True,
        )

        # Verify history was loaded
        history = reconciler2.get_drift_history("square8_2")
        assert history is not None
        assert len(history.snapshots) == 2

    def test_drift_history_no_persistence(self):
        """Test that drift history is not persisted when disabled."""
        self._create_elo_db(self.local_db, {"player1": 1500})

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            track_history=True,
            persist_history=False,
        )
        reconciler.check_drift(board_type="square8", num_players=2)

        # Verify file was NOT created
        history_path = self.local_db.parent / "elo_drift_history.json"
        assert not history_path.exists()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "elo.db"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_elo_drift_returns_drift_object(self):
        """Test check_elo_drift returns EloDrift object."""
        # Create a test DB with correct schema
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0
            )
        """)
        cursor.execute("INSERT INTO participants VALUES ('player1', 1500)")
        conn.commit()
        conn.close()

        # Use custom reconciler with test DB
        reconciler = EloReconciler(local_db_path=self.db_path)
        drift = reconciler.check_drift()

        assert isinstance(drift, EloDrift)
        assert drift.participants_in_source == 1


class TestDriftHistory:
    """Test DriftHistory class for historical drift tracking."""

    def test_basic_creation(self):
        """Test basic drift history creation."""
        history = DriftHistory(config_key="square8_2p")
        assert history.config_key == "square8_2p"
        assert history.snapshots == []
        assert history.max_snapshots == 100

    def test_add_snapshot(self):
        """Test adding a drift snapshot."""
        history = DriftHistory(config_key="square8_2p")
        drift = EloDrift(
            source="local.db", target="remote.db",
            checked_at="2024-01-01T12:00:00",
            participants_in_source=100, participants_in_target=95, participants_in_both=90,
            rating_diffs={"player1": 30.0, "player2": -20.0},
        )
        history.add_snapshot(drift)

        assert len(history.snapshots) == 1
        assert history.snapshots[0]["checked_at"] == "2024-01-01T12:00:00"
        assert history.snapshots[0]["max_rating_diff"] == 30.0
        assert history.snapshots[0]["avg_rating_diff"] == 25.0
        assert history.snapshots[0]["is_significant"] is False
        assert history.snapshots[0]["participants"] == 90

    def test_max_snapshots_limit(self):
        """Test that snapshots are pruned when exceeding max_snapshots."""
        history = DriftHistory(config_key="square8_2p", max_snapshots=5)

        for i in range(10):
            drift = EloDrift(
                source="local.db", target="remote.db",
                checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=100, participants_in_target=100, participants_in_both=100,
                rating_diffs={"player1": float(i)},
            )
            history.add_snapshot(drift)

        # Should keep only the last 5 snapshots
        assert len(history.snapshots) == 5
        # First snapshot should be from hour 5
        assert history.snapshots[0]["checked_at"] == "2024-01-01T05:00:00"
        assert history.snapshots[-1]["checked_at"] == "2024-01-01T09:00:00"

    def test_trend_unknown_insufficient_data(self):
        """Test trend returns 'unknown' with insufficient data."""
        history = DriftHistory(config_key="square8_2p")
        assert history.trend == "unknown"

        # Add only 2 snapshots
        for i in range(2):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(i * 10)},
            )
            history.add_snapshot(drift)

        assert history.trend == "unknown"

    def test_trend_improving(self):
        """Test trend detection when drift is improving (decreasing)."""
        history = DriftHistory(config_key="square8_2p")

        # Add snapshots with decreasing drift
        drift_values = [100, 90, 80, 40, 20]  # Clearly decreasing
        for i, val in enumerate(drift_values):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(val)},
            )
            history.add_snapshot(drift)

        assert history.trend == "improving"

    def test_trend_worsening(self):
        """Test trend detection when drift is worsening (increasing)."""
        history = DriftHistory(config_key="square8_2p")

        # Add snapshots with increasing drift
        drift_values = [20, 40, 60, 90, 100]  # Clearly increasing
        for i, val in enumerate(drift_values):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(val)},
            )
            history.add_snapshot(drift)

        assert history.trend == "worsening"

    def test_trend_stable(self):
        """Test trend detection when drift is stable."""
        history = DriftHistory(config_key="square8_2p")

        # Add snapshots with similar drift values
        drift_values = [30, 32, 28, 31, 29]  # Stable values
        for i, val in enumerate(drift_values):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(val)},
            )
            history.add_snapshot(drift)

        assert history.trend == "stable"

    def test_persistent_drift_false(self):
        """Test persistent_drift is False with insufficient data."""
        history = DriftHistory(config_key="square8_2p")
        assert history.persistent_drift is False

        # Add only 2 snapshots
        for i in range(2):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": 100.0},  # Significant
            )
            history.add_snapshot(drift)

        assert history.persistent_drift is False

    def test_persistent_drift_true(self):
        """Test persistent_drift is True when last 3 checks are significant."""
        history = DriftHistory(config_key="square8_2p")

        # Add 3 significant drift snapshots
        for i in range(3):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": 100.0},  # > 50 threshold = significant
            )
            history.add_snapshot(drift)

        assert history.persistent_drift is True

    def test_persistent_drift_mixed(self):
        """Test persistent_drift is False when not all last 3 are significant."""
        history = DriftHistory(config_key="square8_2p")

        # Add mixed significant/non-significant snapshots
        drifts = [100.0, 100.0, 10.0]  # Last one is not significant
        for i, val in enumerate(drifts):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": val},
            )
            history.add_snapshot(drift)

        assert history.persistent_drift is False

    def test_avg_drift_last_hour_empty(self):
        """Test avg_drift_last_hour with no snapshots."""
        history = DriftHistory(config_key="square8_2p")
        assert history.avg_drift_last_hour == 0.0

    def test_avg_drift_last_hour_single(self):
        """Test avg_drift_last_hour with single snapshot."""
        history = DriftHistory(config_key="square8_2p")
        drift = EloDrift(
            source="a", target="b", checked_at="2024-01-01T12:00:00",
            participants_in_source=10, participants_in_target=10, participants_in_both=10,
            rating_diffs={"p1": 50.0},
        )
        history.add_snapshot(drift)

        assert history.avg_drift_last_hour == 50.0

    def test_avg_drift_last_hour_multiple(self):
        """Test avg_drift_last_hour averages last 2 snapshots."""
        history = DriftHistory(config_key="square8_2p")

        # Add 5 snapshots - should only use last 2 for "last hour"
        drift_values = [10, 20, 30, 40, 60]
        for i, val in enumerate(drift_values):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(val)},
            )
            history.add_snapshot(drift)

        # Average of last 2: (40 + 60) / 2 = 50
        assert history.avg_drift_last_hour == 50.0

    def test_to_dict(self):
        """Test serialization to dict."""
        history = DriftHistory(config_key="square8_2p")

        # Add some snapshots
        for i in range(5):
            drift = EloDrift(
                source="a", target="b", checked_at=f"2024-01-01T{i:02d}:00:00",
                participants_in_source=10, participants_in_target=10, participants_in_both=10,
                rating_diffs={"p1": float(i * 10)},
            )
            history.add_snapshot(drift)

        d = history.to_dict()

        assert d["config_key"] == "square8_2p"
        assert d["trend"] == "worsening"  # 0, 10, 20, 30, 40 - increasing
        assert d["snapshot_count"] == 5
        assert "recent_snapshots" in d
        assert len(d["recent_snapshots"]) == 5


class TestConflictResolution:
    """Test conflict resolution strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.local_db = Path(self.temp_dir) / "local_elo.db"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_elo_db(self, path: Path):
        """Create empty Elo database with correct schema."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
                match_id TEXT PRIMARY KEY,
                player1_id TEXT,
                player2_id TEXT,
                winner_id TEXT,
                player1_rating_before REAL,
                player2_rating_before REAL,
                player1_rating_after REAL,
                player2_rating_after REAL,
                board_type TEXT,
                num_players INTEGER,
                game_length INTEGER,
                timestamp TEXT,
                source TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0
            )
        """)
        conn.commit()
        conn.close()

    def _insert_match(self, path: Path, match_id: str, winner_id: str, timestamp: str):
        """Insert a match into the database."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO match_history (match_id, player1_id, player2_id, winner_id, timestamp) "
            "VALUES (?, 'p1', 'p2', ?, ?)",
            (match_id, winner_id, timestamp),
        )
        conn.commit()
        conn.close()

    def _get_match_winner(self, path: Path, match_id: str) -> str:
        """Get the winner of a match."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT winner_id FROM match_history WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def test_conflict_resolution_skip(self):
        """Test SKIP conflict resolution (default) - keeps existing, counts conflict."""
        self._create_elo_db(self.local_db)
        self._insert_match(self.local_db, "match1", "p1", "2024-01-01T12:00:00")

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            conflict_resolution=ConflictResolution.SKIP,
        )

        # Try to import conflicting match
        result = reconciler._import_matches(
            "host", "now",
            [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2", "timestamp": "2024-01-01T13:00:00"}],
        )

        assert result.matches_conflict == 1
        assert result.matches_resolved == 0
        # Original winner should be preserved
        assert self._get_match_winner(self.local_db, "match1") == "p1"

    def test_conflict_resolution_first_write_wins(self):
        """Test FIRST_WRITE_WINS - keeps existing record, marks as resolved."""
        self._create_elo_db(self.local_db)
        self._insert_match(self.local_db, "match1", "p1", "2024-01-01T12:00:00")

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            conflict_resolution=ConflictResolution.FIRST_WRITE_WINS,
        )

        result = reconciler._import_matches(
            "host", "now",
            [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2", "timestamp": "2024-01-01T13:00:00"}],
        )

        assert result.matches_conflict == 0
        assert result.matches_resolved == 1
        # Original winner preserved
        assert self._get_match_winner(self.local_db, "match1") == "p1"

    def test_conflict_resolution_last_write_wins_newer(self):
        """Test LAST_WRITE_WINS with newer incoming timestamp - updates match."""
        self._create_elo_db(self.local_db)
        self._insert_match(self.local_db, "match1", "p1", "2024-01-01T12:00:00")

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
        )

        # Import match with newer timestamp
        result = reconciler._import_matches(
            "host", "now",
            [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2", "timestamp": "2024-01-01T13:00:00"}],
        )

        assert result.matches_conflict == 0
        assert result.matches_resolved == 1
        # Winner should be updated to incoming
        assert self._get_match_winner(self.local_db, "match1") == "p2"

    def test_conflict_resolution_last_write_wins_older(self):
        """Test LAST_WRITE_WINS with older incoming timestamp - keeps existing."""
        self._create_elo_db(self.local_db)
        self._insert_match(self.local_db, "match1", "p1", "2024-01-01T13:00:00")  # Newer existing

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
        )

        # Import match with older timestamp
        result = reconciler._import_matches(
            "host", "now",
            [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2", "timestamp": "2024-01-01T12:00:00"}],
        )

        assert result.matches_conflict == 0
        assert result.matches_resolved == 1
        # Original winner preserved (existing is newer)
        assert self._get_match_winner(self.local_db, "match1") == "p1"

    def test_conflict_resolution_raise(self):
        """Test RAISE conflict resolution - raises ValueError on conflict."""
        self._create_elo_db(self.local_db)
        self._insert_match(self.local_db, "match1", "p1", "2024-01-01T12:00:00")

        reconciler = EloReconciler(
            local_db_path=self.local_db,
            conflict_resolution=ConflictResolution.RAISE,
        )

        with pytest.raises(ValueError) as exc_info:
            reconciler._import_matches(
                "host", "now",
                [{"match_id": "match1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2"}],
            )

        assert "Match conflict" in str(exc_info.value)
        assert "match1" in str(exc_info.value)

    def test_timestamp_comparison_iso_format(self):
        """Test timestamp comparison with ISO format timestamps."""
        self._create_elo_db(self.local_db)
        reconciler = EloReconciler(local_db_path=self.local_db)

        # Test various ISO formats
        assert reconciler._is_newer_timestamp("2024-01-02T12:00:00", "2024-01-01T12:00:00") is True
        assert reconciler._is_newer_timestamp("2024-01-01T12:00:00", "2024-01-02T12:00:00") is False
        assert reconciler._is_newer_timestamp("2024-01-01T13:00:00", "2024-01-01T12:00:00") is True

    def test_timestamp_comparison_with_none(self):
        """Test timestamp comparison with None values."""
        self._create_elo_db(self.local_db)
        reconciler = EloReconciler(local_db_path=self.local_db)

        # None incoming - don't update
        assert reconciler._is_newer_timestamp(None, "2024-01-01T12:00:00") is False
        # None existing - accept incoming
        assert reconciler._is_newer_timestamp("2024-01-01T12:00:00", None) is True

    def test_timestamp_comparison_utc_suffix(self):
        """Test timestamp comparison with Z (UTC) suffix."""
        self._create_elo_db(self.local_db)
        reconciler = EloReconciler(local_db_path=self.local_db)

        assert reconciler._is_newer_timestamp("2024-01-02T12:00:00Z", "2024-01-01T12:00:00Z") is True
        assert reconciler._is_newer_timestamp("2024-01-01T12:00:00Z", "2024-01-02T12:00:00Z") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
