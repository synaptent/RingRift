"""Tests for distributed_gauntlet.py module.

Comprehensive tests for O(n) gauntlet evaluation:
- GauntletConfig dataclass
- GauntletResult dataclass
- GameTask and GameResult dataclasses
- DistributedNNGauntlet class
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tournament.distributed_gauntlet import (
    CONFIG_KEYS,
    MAX_MOVES,
    DistributedNNGauntlet,
    GameResult,
    GameTask,
    GauntletConfig,
    GauntletResult,
)


# =============================================================================
# GauntletConfig Tests
# =============================================================================


class TestGauntletConfig:
    """Tests for GauntletConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = GauntletConfig()
        assert config.games_per_matchup == 10
        assert config.num_baselines == 4
        assert config.reserved_workers == 2
        assert config.parallel_games == 8
        assert config.timeout_seconds == 300
        assert config.min_games_for_rating == 5
        assert config.stale_run_timeout == 3600
        assert config.max_distributed_wait == 1800

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = GauntletConfig(
            games_per_matchup=20,
            num_baselines=6,
            timeout_seconds=600,
        )
        assert config.games_per_matchup == 20
        assert config.num_baselines == 6
        assert config.timeout_seconds == 600
        # Other values should be defaults
        assert config.reserved_workers == 2


# =============================================================================
# GauntletResult Tests
# =============================================================================


class TestGauntletResult:
    """Tests for GauntletResult dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        result = GauntletResult(
            run_id="test-run",
            config_key="square8_2p",
            started_at=time.time(),
        )
        assert result.completed_at is None
        assert result.models_evaluated == 0
        assert result.total_games == 0
        assert result.status == "pending"
        assert result.model_results == {}

    def test_full_result(self):
        """Should store all fields."""
        now = time.time()
        result = GauntletResult(
            run_id="test-123",
            config_key="hex8_3p",
            started_at=now,
            completed_at=now + 60,
            models_evaluated=10,
            total_games=100,
            status="completed",
            model_results={"model_a": {"wins": 5, "losses": 3}},
        )
        assert result.run_id == "test-123"
        assert result.config_key == "hex8_3p"
        assert result.completed_at == now + 60
        assert result.status == "completed"
        assert "model_a" in result.model_results


# =============================================================================
# GameTask Tests
# =============================================================================


class TestGameTask:
    """Tests for GameTask dataclass."""

    def test_task_fields(self):
        """Should store all task fields."""
        task = GameTask(
            task_id="task-1",
            model_id="model_a",
            baseline_id="random_ai",
            config_key="square8_2p",
            game_num=0,
        )
        assert task.task_id == "task-1"
        assert task.model_id == "model_a"
        assert task.baseline_id == "random_ai"
        assert task.config_key == "square8_2p"
        assert task.game_num == 0


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_model_win(self):
        """Should represent a model win."""
        result = GameResult(
            task_id="task-1",
            model_id="model_a",
            baseline_id="random_ai",
            model_won=True,
            baseline_won=False,
            draw=False,
            game_length=50,
            duration_sec=5.0,
        )
        assert result.model_won is True
        assert result.baseline_won is False
        assert result.draw is False

    def test_baseline_win(self):
        """Should represent a baseline win."""
        result = GameResult(
            task_id="task-1",
            model_id="model_a",
            baseline_id="random_ai",
            model_won=False,
            baseline_won=True,
            draw=False,
            game_length=30,
            duration_sec=3.0,
        )
        assert result.model_won is False
        assert result.baseline_won is True

    def test_draw(self):
        """Should represent a draw."""
        result = GameResult(
            task_id="task-1",
            model_id="model_a",
            baseline_id="model_b",
            model_won=False,
            baseline_won=False,
            draw=True,
            game_length=100,
            duration_sec=10.0,
        )
        assert result.draw is True
        assert result.model_won is False
        assert result.baseline_won is False


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_config_keys(self):
        """CONFIG_KEYS should cover all 12 board/player combinations."""
        assert len(CONFIG_KEYS) == 12
        assert "square8_2p" in CONFIG_KEYS
        assert "hexagonal_4p" in CONFIG_KEYS
        assert "square19_3p" in CONFIG_KEYS
        assert "square19_4p" in CONFIG_KEYS

    def test_max_moves(self):
        """MAX_MOVES should define limits for all 12 supported configs."""
        assert len(MAX_MOVES) == 12
        assert MAX_MOVES["square8_2p"] == 500
        assert MAX_MOVES["hexagonal_4p"] == 4000
        assert MAX_MOVES["square19_2p"] == 1500
        assert MAX_MOVES["square19_4p"] == 2500


# =============================================================================
# DistributedNNGauntlet Tests
# =============================================================================


class TestDistributedNNGauntletInit:
    """Tests for DistributedNNGauntlet initialization."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_default_config(self, temp_db, temp_model_dir):
        """Should initialize with default config."""
        gauntlet = DistributedNNGauntlet(temp_db, temp_model_dir)
        assert gauntlet.elo_db_path == temp_db
        assert gauntlet.model_dir == temp_model_dir
        assert isinstance(gauntlet.config, GauntletConfig)
        assert gauntlet.config.games_per_matchup == 10

    def test_init_custom_config(self, temp_db, temp_model_dir):
        """Should accept custom config."""
        config = GauntletConfig(games_per_matchup=50)
        gauntlet = DistributedNNGauntlet(temp_db, temp_model_dir, config=config)
        assert gauntlet.config.games_per_matchup == 50

    def test_init_empty_state(self, temp_db, temp_model_dir):
        """Should start with empty state."""
        gauntlet = DistributedNNGauntlet(temp_db, temp_model_dir)
        assert gauntlet._current_run is None
        assert gauntlet._reserved_workers == set()


class TestDistributedNNGauntletDatabase:
    """Tests for database operations."""

    @pytest.fixture
    def gauntlet(self):
        """Create gauntlet with temp database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            g._init_gauntlet_tables()
            yield g
        if db_path.exists():
            db_path.unlink()

    def test_init_gauntlet_tables(self, gauntlet):
        """Should create gauntlet tables."""
        conn = gauntlet._get_db_connection()
        try:
            # Check gauntlet_runs table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='gauntlet_runs'"
            )
            assert cursor.fetchone() is not None

            # Check gauntlet_results table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='gauntlet_results'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_cleanup_stale_runs(self, gauntlet):
        """Should cleanup stale runs."""
        conn = gauntlet._get_db_connection()
        try:
            # Insert a stale run (started 2 hours ago)
            stale_time = time.time() - 7200  # 2 hours ago
            conn.execute("""
                INSERT INTO gauntlet_runs (run_id, config_key, started_at, status)
                VALUES (?, ?, ?, ?)
            """, ("stale-run", "square8_2p", stale_time, "running"))
            conn.commit()
        finally:
            conn.close()

        # Cleanup should find and mark the stale run
        cleaned = gauntlet._cleanup_stale_runs("square8_2p")
        assert cleaned == 1

        # Verify run is now marked as timeout
        conn = gauntlet._get_db_connection()
        try:
            cursor = conn.execute(
                "SELECT status FROM gauntlet_runs WHERE run_id = ?",
                ("stale-run",)
            )
            row = cursor.fetchone()
            assert row[0] == "timeout"
        finally:
            conn.close()

    def test_cleanup_stale_runs_respects_timeout(self, gauntlet):
        """Should not cleanup recent runs."""
        conn = gauntlet._get_db_connection()
        try:
            # Insert a recent run (started 5 minutes ago)
            recent_time = time.time() - 300  # 5 minutes ago
            conn.execute("""
                INSERT INTO gauntlet_runs (run_id, config_key, started_at, status)
                VALUES (?, ?, ?, ?)
            """, ("recent-run", "square8_2p", recent_time, "running"))
            conn.commit()
        finally:
            conn.close()

        # Cleanup should not touch recent runs
        cleaned = gauntlet._cleanup_stale_runs("square8_2p")
        assert cleaned == 0


class TestDistributedNNGauntletModels:
    """Tests for model discovery and registration."""

    @pytest.fixture
    def gauntlet_with_models(self):
        """Create gauntlet with temp database containing models."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Initialize database with elo_ratings table
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
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
            );
        """)

        # Insert some models
        now = time.time()
        conn.executemany("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played, last_update)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            ("model_best", "square8", 2, 1800.0, 50, now),
            ("model_mid", "square8", 2, 1500.0, 30, now),
            ("model_low", "square8", 2, 1200.0, 20, now),
            ("model_unrated", "square8", 2, 1200.0, 2, now),  # < min_games
        ])
        conn.commit()
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            g._init_gauntlet_tables()
            yield g

        if db_path.exists():
            db_path.unlink()

    def test_get_models_by_elo(self, gauntlet_with_models):
        """Should return models sorted by Elo."""
        models = gauntlet_with_models.get_models_by_elo("square8_2p")
        assert len(models) == 4
        # Should be sorted descending
        assert models[0][0] == "model_best"
        assert models[0][1] == 1800.0
        assert models[-1][0] == "model_unrated"

    def test_get_unrated_models(self, gauntlet_with_models):
        """Should return models with < min_games."""
        unrated = gauntlet_with_models.get_unrated_models("square8_2p")
        assert len(unrated) == 1
        assert "model_unrated" in unrated
        assert "model_best" not in unrated

    def test_count_models(self, gauntlet_with_models):
        """Should count all models for config."""
        count = gauntlet_with_models.count_models("square8_2p")
        assert count == 4


class TestDistributedNNGauntletBaselines:
    """Tests for baseline selection."""

    @pytest.fixture
    def gauntlet_with_many_models(self):
        """Create gauntlet with many models for baseline selection."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                PRIMARY KEY (participant_id, board_type, num_players)
            );
        """)

        # Insert 10 models with varying Elo
        models = [(f"model_{i:02d}", "square8", 2, 1000 + i * 100, 20)
                  for i in range(10)]
        conn.executemany("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played)
            VALUES (?, ?, ?, ?, ?)
        """, models)
        conn.commit()
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            g._init_gauntlet_tables()
            yield g

        if db_path.exists():
            db_path.unlink()

    def test_select_baselines_many_models(self, gauntlet_with_many_models):
        """Should select canonical, model, heuristic, and random baselines."""
        baselines = gauntlet_with_many_models.select_baselines("square8_2p")

        assert len(baselines) == 5
        assert baselines[0] == "ringrift_best_square8_2p"
        # Best model should be model_09 (rating 1900)
        assert baselines[1] == "model_09"
        assert "heuristic" in baselines
        # Should include random_ai
        assert "random_ai" in baselines

    def test_select_baselines_no_models(self):
        """Should return the default mixed baselines if no models."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Empty database
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                PRIMARY KEY (participant_id, board_type, num_players)
            );
        """)
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            g._init_gauntlet_tables()
            baselines = g.select_baselines("square8_2p")

        assert baselines == [
            "ringrift_best_square8_2p",
            "ringrift_best_square8_2p:policy_only:t0p3",
            "heuristic",
            "random_ai",
        ]
        db_path.unlink()


class TestDistributedNNGauntletTasks:
    """Tests for task creation."""

    @pytest.fixture
    def gauntlet(self):
        """Create basic gauntlet."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            yield g
        if db_path.exists():
            db_path.unlink()

    def test_create_game_tasks(self, gauntlet):
        """Should create tasks for all model-baseline combinations."""
        unrated = ["model_a", "model_b"]
        baselines = ["baseline_1", "baseline_2", "random_ai"]

        tasks = gauntlet.create_game_tasks(unrated, baselines, "square8_2p")

        # 2 models * 3 baselines * 10 games = 60 tasks
        assert len(tasks) == 60

        # Check task structure
        task = tasks[0]
        assert task.model_id in unrated
        assert task.baseline_id in baselines
        assert task.config_key == "square8_2p"
        assert 0 <= task.game_num < 10

    def test_create_game_tasks_skips_self(self, gauntlet):
        """Should skip when model is its own baseline."""
        unrated = ["model_a"]
        baselines = ["model_a", "random_ai"]  # model_a is both unrated and baseline

        tasks = gauntlet.create_game_tasks(unrated, baselines, "square8_2p")

        # Should only create tasks against random_ai, not against itself
        assert len(tasks) == 10  # 1 model * 1 baseline * 10 games
        for task in tasks:
            assert task.baseline_id == "random_ai"

    def test_create_game_tasks_empty_inputs(self, gauntlet):
        """Should handle empty inputs."""
        tasks = gauntlet.create_game_tasks([], ["baseline"], "square8_2p")
        assert tasks == []

        tasks = gauntlet.create_game_tasks(["model"], [], "square8_2p")
        assert tasks == []


class TestDistributedNNGauntletRun:
    """Tests for gauntlet run execution."""

    @pytest.fixture
    def gauntlet_ready(self):
        """Create gauntlet with database ready for run."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                PRIMARY KEY (participant_id, board_type, num_players)
            );
        """)
        # Insert one unrated model
        conn.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played)
            VALUES (?, ?, ?, ?, ?)
        """, ("model_test", "square8", 2, 1200.0, 2))
        conn.commit()
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            yield g

        if db_path.exists():
            db_path.unlink()

    @pytest.mark.asyncio
    async def test_run_gauntlet_no_unrated(self):
        """Should return early if no unrated models."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # All models have enough games
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                PRIMARY KEY (participant_id, board_type, num_players)
            );
        """)
        conn.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played)
            VALUES (?, ?, ?, ?, ?)
        """, ("model_rated", "square8", 2, 1500.0, 50))  # 50 games > min
        conn.commit()
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            g = DistributedNNGauntlet(db_path, Path(tmpdir))
            result = await g.run_gauntlet("square8_2p")

        assert result.status == "no_work"
        assert result.models_evaluated == 0

        if db_path.exists():
            db_path.unlink()

    @pytest.mark.asyncio
    async def test_run_gauntlet_creates_run_record(self, gauntlet_ready):
        """Should create run record in database."""
        # Mock _execute_tasks_local to avoid actual game execution
        gauntlet_ready._execute_tasks_local = AsyncMock(return_value=[])
        gauntlet_ready._aggregate_results = MagicMock()

        result = await gauntlet_ready.run_gauntlet("square8_2p")

        # Should have created a run record
        conn = gauntlet_ready._get_db_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM gauntlet_runs WHERE run_id = ?",
                (result.run_id,)
            )
            count = cursor.fetchone()[0]
            assert count == 1
        finally:
            conn.close()


class TestDistributedNNGauntletModelDiscovery:
    """Tests for model discovery from filesystem."""

    @pytest.fixture
    def gauntlet_with_files(self):
        """Create gauntlet with model files in directory."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
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
            );
        """)
        conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            # Create some model files
            (model_dir / "sq8_2p_model_a.pth").touch()
            (model_dir / "sq8_2p_model_b.pth").touch()
            (model_dir / "hex8_2p_model_c.pth").touch()  # Different config
            (model_dir / "not_a_model.txt").touch()  # Not a .pth file

            g = DistributedNNGauntlet(db_path, model_dir)
            g._init_gauntlet_tables()
            yield g

        if db_path.exists():
            db_path.unlink()

    def test_discover_and_register_models(self, gauntlet_with_files):
        """Should discover and register new models."""
        count = gauntlet_with_files.discover_and_register_models("square8_2p")

        assert count == 2  # sq8_2p_model_a and sq8_2p_model_b

        # Verify models are in database
        models = gauntlet_with_files.get_models_by_elo("square8_2p")
        model_ids = [m[0] for m in models]
        assert "sq8_2p_model_a" in model_ids
        assert "sq8_2p_model_b" in model_ids

    def test_discover_does_not_duplicate(self, gauntlet_with_files):
        """Should not duplicate already registered models."""
        # First discovery
        count1 = gauntlet_with_files.discover_and_register_models("square8_2p")
        assert count1 == 2

        # Second discovery should find nothing new
        count2 = gauntlet_with_files.discover_and_register_models("square8_2p")
        assert count2 == 0
