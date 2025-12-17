"""Integration tests for PromotionController and EloReconciler workflow.

Tests the full promotion and Elo reconciliation workflow including:
- PromotionController evaluation with real-ish data
- Metrics emission integration
- EloReconciler with promotion events
- End-to-end promotion flow

These tests use local temp databases and don't require network access.
"""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.elo_reconciliation import (
    EloDrift,
    EloReconciler,
    ReconciliationReport,
    SyncResult,
)
from app.training.promotion_controller import (
    PromotionController,
    PromotionCriteria,
    PromotionDecision,
    PromotionType,
)


class TestPromotionWithEloIntegration:
    """Test PromotionController with Elo data integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_elo.db"
        self._create_test_db()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_db(self):
        """Create a test Elo database with realistic data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create tables matching production schema
        cursor.execute("""
            CREATE TABLE elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                rating_deviation REAL DEFAULT 350.0,
                last_update REAL,
                PRIMARY KEY (participant_id, board_type, num_players)
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

        # Insert test models with various ratings
        models = [
            ("model_baseline", "square8", 2, 1500.0, 100),
            ("model_improved", "square8", 2, 1560.0, 80),  # +60 Elo
            ("model_insufficient_games", "square8", 2, 1600.0, 20),  # Not enough games
            ("model_regression", "square8", 2, 1450.0, 100),  # Below baseline
        ]

        for model_id, board, players, rating, games in models:
            cursor.execute(
                """
                INSERT INTO elo_ratings
                (participant_id, board_type, num_players, rating, games_played)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, board, players, rating, games),
            )

        conn.commit()
        conn.close()

    def test_promotion_evaluation_with_elo_data(self):
        """Test promotion evaluation uses Elo data correctly."""
        # Create mock Elo service that queries our test DB
        mock_elo = MagicMock()

        # Mock rating for improved model
        mock_rating = MagicMock()
        mock_rating.rating = 1560.0
        mock_rating.games_played = 80
        mock_rating.win_rate = 0.58

        # Mock baseline rating
        mock_baseline = MagicMock()
        mock_baseline.rating = 1500.0

        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        controller = PromotionController(
            criteria=PromotionCriteria(min_elo_improvement=25.0, min_games_played=50),
            elo_service=mock_elo,
        )

        decision = controller.evaluate_promotion(
            model_id="model_improved",
            board_type="square8",
            num_players=2,
            promotion_type=PromotionType.PRODUCTION,
            baseline_model_id="model_baseline",
        )

        assert decision.should_promote is True
        assert decision.elo_improvement == 60.0
        assert "Meets all criteria" in decision.reason

    def test_promotion_rejection_insufficient_games(self):
        """Test promotion is rejected for insufficient games."""
        mock_elo = MagicMock()

        mock_rating = MagicMock()
        mock_rating.rating = 1600.0
        mock_rating.games_played = 20  # Below min_games_played
        mock_rating.win_rate = 0.65

        mock_elo.get_rating.return_value = mock_rating

        controller = PromotionController(
            criteria=PromotionCriteria(min_games_played=50),
            elo_service=mock_elo,
        )

        decision = controller.evaluate_promotion(
            model_id="model_insufficient_games",
            promotion_type=PromotionType.PRODUCTION,
        )

        assert decision.should_promote is False
        assert "Insufficient games" in decision.reason

    def test_promotion_rejection_low_elo_improvement(self):
        """Test promotion is rejected for insufficient Elo improvement."""
        mock_elo = MagicMock()

        mock_rating = MagicMock()
        mock_rating.rating = 1510.0
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.52

        mock_baseline = MagicMock()
        mock_baseline.rating = 1500.0  # Only +10 improvement

        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        controller = PromotionController(
            criteria=PromotionCriteria(min_elo_improvement=25.0),
            elo_service=mock_elo,
        )

        decision = controller.evaluate_promotion(
            model_id="model_low_improvement",
            baseline_model_id="model_baseline",
            promotion_type=PromotionType.PRODUCTION,
        )

        assert decision.should_promote is False
        assert "Insufficient Elo improvement" in decision.reason


class TestEloReconcilerIntegration:
    """Test EloReconciler full workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.local_db = Path(self.temp_dir) / "local_elo.db"
        self.remote_db = Path(self.temp_dir) / "remote_elo.db"

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_elo_db(self, path: Path, ratings: dict, matches: list = None):
        """Create a test Elo database."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL DEFAULT 'square8',
                num_players INTEGER NOT NULL DEFAULT 2,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                PRIMARY KEY (participant_id, board_type, num_players)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_history (
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

        for participant_id, rating in ratings.items():
            cursor.execute(
                "INSERT INTO elo_ratings (participant_id, rating) VALUES (?, ?)",
                (participant_id, rating),
            )

        if matches:
            for match in matches:
                cursor.execute(
                    """
                    INSERT INTO match_history (match_id, player1_id, player2_id, winner_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        match["match_id"],
                        match["player1_id"],
                        match["player2_id"],
                        match.get("winner_id"),
                        match.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    ),
                )

        conn.commit()
        conn.close()

    def test_drift_detection_workflow(self):
        """Test full drift detection workflow."""
        # Create local DB with some ratings
        self._create_elo_db(
            self.local_db,
            {"model_a": 1500, "model_b": 1600, "model_c": 1550},
        )

        # Create remote DB with different ratings (drift)
        self._create_elo_db(
            self.remote_db,
            {"model_a": 1500, "model_b": 1650, "model_c": 1520},  # b: +50, c: -30
        )

        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift(remote_db_path=self.remote_db)

        assert drift.participants_in_both == 3
        assert drift.max_rating_diff == 50.0  # model_b diff
        assert len(drift.rating_diffs) == 2  # b and c have meaningful diffs
        assert drift.is_significant is True  # max > 50 threshold

    def test_drift_detection_no_significant_drift(self):
        """Test drift detection with minor differences."""
        self._create_elo_db(
            self.local_db,
            {"model_a": 1500, "model_b": 1600},
        )

        self._create_elo_db(
            self.remote_db,
            {"model_a": 1505, "model_b": 1595},  # Only ±5 diff
        )

        reconciler = EloReconciler(local_db_path=self.local_db)
        drift = reconciler.check_drift(remote_db_path=self.remote_db)

        assert drift.max_rating_diff == 5.0
        assert drift.is_significant is False

    def test_match_import_workflow(self):
        """Test match import workflow."""
        # Create local DB (empty)
        self._create_elo_db(self.local_db, {})

        reconciler = EloReconciler(local_db_path=self.local_db)

        # Import some matches
        matches = [
            {"match_id": "m1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"},
            {"match_id": "m2", "player1_id": "p1", "player2_id": "p3", "winner_id": "p3"},
            {"match_id": "m3", "player1_id": "p2", "player2_id": "p3", "winner_id": "p2"},
        ]

        result = reconciler._import_matches("test_host", "now", matches)

        assert result.matches_added == 3
        assert result.matches_skipped == 0
        assert result.participants_added == 3  # p1, p2, p3

        # Verify matches are in DB
        conn = sqlite3.connect(str(self.local_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM match_history")
        assert cursor.fetchone()[0] == 3
        conn.close()

    def test_duplicate_match_handling(self):
        """Test duplicate match detection during import."""
        # Create local DB with existing match
        self._create_elo_db(
            self.local_db,
            {},
            matches=[{"match_id": "m1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"}],
        )

        reconciler = EloReconciler(local_db_path=self.local_db)

        # Try to import same match again
        matches = [
            {"match_id": "m1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"},  # Duplicate
            {"match_id": "m2", "player1_id": "p1", "player2_id": "p3", "winner_id": "p3"},  # New
        ]

        result = reconciler._import_matches("test_host", "now", matches)

        assert result.matches_added == 1
        assert result.matches_skipped == 1

    def test_conflict_detection(self):
        """Test conflict detection during import."""
        # Create local DB with match
        self._create_elo_db(
            self.local_db,
            {},
            matches=[{"match_id": "m1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p1"}],
        )

        reconciler = EloReconciler(local_db_path=self.local_db)

        # Try to import same match_id with different winner (conflict!)
        matches = [
            {"match_id": "m1", "player1_id": "p1", "player2_id": "p2", "winner_id": "p2"},  # Different winner
        ]

        result = reconciler._import_matches("test_host", "now", matches)

        assert result.matches_added == 0
        assert result.matches_conflict == 1


class TestMetricsEmission:
    """Test that metrics are properly emitted during promotion and reconciliation."""

    def test_promotion_decision_emits_metrics(self):
        """Test promotion decision emits Prometheus metrics."""
        with patch("app.metrics.record_promotion_decision") as mock_record:
            mock_elo = MagicMock()
            mock_rating = MagicMock()
            mock_rating.rating = 1550
            mock_rating.games_played = 100
            mock_rating.win_rate = 0.55
            mock_baseline = MagicMock()
            mock_baseline.rating = 1500
            mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

            controller = PromotionController(elo_service=mock_elo)
            controller.evaluate_promotion(
                model_id="test_model",
                promotion_type=PromotionType.PRODUCTION,
                baseline_model_id="baseline",
            )

            mock_record.assert_called_once()
            call_args = mock_record.call_args
            assert call_args.kwargs["promotion_type"] == "production"
            assert call_args.kwargs["approved"] is True
            assert call_args.kwargs["elo_improvement"] == 50.0

    def test_elo_drift_emits_metrics(self):
        """Test Elo drift check emits Prometheus metrics."""
        with patch("app.metrics.record_elo_drift") as mock_record:
            temp_dir = tempfile.mkdtemp()
            try:
                local_db = Path(temp_dir) / "local.db"

                # Create test DB with elo_ratings table
                conn = sqlite3.connect(str(local_db))
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE elo_ratings (
                        participant_id TEXT PRIMARY KEY,
                        board_type TEXT DEFAULT 'square8',
                        num_players INTEGER DEFAULT 2,
                        rating REAL DEFAULT 1500.0
                    )
                """)
                cursor.execute("INSERT INTO elo_ratings (participant_id, rating) VALUES ('p1', 1500)")
                conn.commit()
                conn.close()

                reconciler = EloReconciler(local_db_path=local_db)
                reconciler.check_drift(board_type="square8", num_players=2)

                mock_record.assert_called_once()
                call_args = mock_record.call_args
                assert call_args.kwargs["board_type"] == "square8"
                assert call_args.kwargs["num_players"] == 2
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestFullPromotionWorkflow:
    """End-to-end tests for the complete promotion workflow."""

    def test_evaluate_and_execute_promotion(self):
        """Test full promotion workflow from evaluation to execution."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.rating = 1560
        mock_rating.games_played = 100
        mock_rating.win_rate = 0.58
        mock_baseline = MagicMock()
        mock_baseline.rating = 1500
        mock_elo.get_rating.side_effect = [mock_rating, mock_baseline]

        mock_registry = MagicMock()
        mock_registry.promote_model.return_value = True

        controller = PromotionController(
            elo_service=mock_elo,
            model_registry=mock_registry,
        )

        # Evaluate
        decision = controller.evaluate_promotion(
            model_id="model_v2",
            promotion_type=PromotionType.STAGING,
            baseline_model_id="model_v1",
        )

        assert decision.should_promote is True

        # Execute
        success = controller.execute_promotion(decision)

        assert success is True
        mock_registry.promote_model.assert_called_once()

    def test_promotion_dry_run(self):
        """Test dry run promotion doesn't actually promote."""
        mock_registry = MagicMock()

        controller = PromotionController(model_registry=mock_registry)

        decision = PromotionDecision(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="Test promotion",
        )

        success = controller.execute_promotion(decision, dry_run=True)

        assert success is True
        mock_registry.promote_model.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
