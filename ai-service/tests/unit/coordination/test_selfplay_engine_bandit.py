"""Unit tests for SelfplayEngineBandit.

Tests Thompson Sampling-based engine selection for selfplay data generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.coordination.selfplay_engine_bandit import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    MIN_GAMES_FOR_EXPLOITATION,
    EngineStats,
    SelfplayEngineBandit,
    get_selfplay_engine_bandit,
    reset_selfplay_engine_bandit,
)


class TestEngineStats:
    """Tests for EngineStats dataclass."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        stats = EngineStats(engine="gumbel-mcts", config_key="hex8_2p")

        assert stats.engine == "gumbel-mcts"
        assert stats.config_key == "hex8_2p"
        assert stats.alpha == 1.0
        assert stats.beta == 1.0
        assert stats.total_games == 0
        assert stats.total_elo_gain == 0.0
        assert stats.observation_count == 0
        assert stats.elo_gains == []
        assert stats.timestamps == []

    def test_mean_elo_per_game_zero_games(self):
        """Test mean Elo with zero games returns 0."""
        stats = EngineStats(engine="test", config_key="test")
        assert stats.mean_elo_per_game == 0.0

    def test_mean_elo_per_game_with_data(self):
        """Test mean Elo calculation with data."""
        stats = EngineStats(
            engine="test",
            config_key="test",
            total_games=100,
            total_elo_gain=50.0,
        )
        assert stats.mean_elo_per_game == 0.5

    def test_success_rate_calculation(self):
        """Test success rate from alpha/beta."""
        stats = EngineStats(engine="test", config_key="test", alpha=3.0, beta=1.0)
        assert stats.success_rate == 0.75

    def test_success_rate_default_uninformative(self):
        """Test default success rate is 0.5 (uninformative prior)."""
        stats = EngineStats(engine="test", config_key="test")
        assert stats.success_rate == 0.5

    def test_sample_thompson_returns_probability(self):
        """Test Thompson sample is in [0, 1]."""
        stats = EngineStats(engine="test", config_key="test")

        for _ in range(100):
            sample = stats.sample_thompson()
            assert 0.0 <= sample <= 1.0

    def test_update_positive_elo_increases_alpha(self):
        """Test positive Elo gain increases alpha."""
        stats = EngineStats(engine="test", config_key="test")
        initial_alpha = stats.alpha
        initial_beta = stats.beta

        stats.update(elo_gain=10.0, games=100)

        assert stats.alpha > initial_alpha
        assert stats.beta == initial_beta  # Should not change
        assert stats.total_games == 100
        assert stats.total_elo_gain == 10.0
        assert stats.observation_count == 1

    def test_update_negative_elo_increases_beta(self):
        """Test negative Elo gain increases beta."""
        stats = EngineStats(engine="test", config_key="test")
        initial_alpha = stats.alpha
        initial_beta = stats.beta

        stats.update(elo_gain=-15.0, games=50)

        assert stats.alpha == initial_alpha  # Should not change
        assert stats.beta > initial_beta
        assert stats.total_games == 50
        assert stats.total_elo_gain == -15.0

    def test_update_caps_alpha_increase(self):
        """Test alpha increase is capped at +5 per observation."""
        stats = EngineStats(engine="test", config_key="test")
        initial_alpha = stats.alpha

        stats.update(elo_gain=1000.0, games=10)  # Very large gain

        # Max increase should be 5.0 (capped)
        assert stats.alpha <= initial_alpha + 5.0

    def test_update_caps_beta_increase(self):
        """Test beta increase is capped at +5 per observation."""
        stats = EngineStats(engine="test", config_key="test")
        initial_beta = stats.beta

        stats.update(elo_gain=-1000.0, games=10)  # Very large loss

        # Max increase should be 5.0 (capped)
        assert stats.beta <= initial_beta + 5.0

    def test_update_stores_history(self):
        """Test update stores elo_gains and timestamps."""
        stats = EngineStats(engine="test", config_key="test")

        stats.update(elo_gain=5.0, games=100)
        stats.update(elo_gain=-3.0, games=50)

        assert len(stats.elo_gains) == 2
        assert stats.elo_gains == [5.0, -3.0]
        assert len(stats.timestamps) == 2

    def test_update_limits_history_to_50(self):
        """Test history is trimmed to last 50 observations."""
        stats = EngineStats(engine="test", config_key="test")

        for i in range(60):
            stats.update(elo_gain=float(i), games=10)

        assert len(stats.elo_gains) == 50
        assert len(stats.timestamps) == 50
        # Should keep last 50 (indices 10-59)
        assert stats.elo_gains[0] == 10.0
        assert stats.elo_gains[-1] == 59.0

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        stats = EngineStats(
            engine="gumbel-mcts",
            config_key="hex8_2p",
            alpha=2.0,
            beta=1.5,
            total_games=200,
            total_elo_gain=30.0,
            observation_count=5,
            elo_gains=[5.0, 10.0, -2.0],
            timestamps=[1000.0, 2000.0, 3000.0],
        )

        data = stats.to_dict()

        assert data["engine"] == "gumbel-mcts"
        assert data["config_key"] == "hex8_2p"
        assert data["alpha"] == 2.0
        assert data["beta"] == 1.5
        assert data["total_games"] == 200
        assert data["total_elo_gain"] == 30.0
        assert data["observation_count"] == 5
        assert data["elo_gains"] == [5.0, 10.0, -2.0]

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "engine": "mcts",
            "config_key": "square8_2p",
            "alpha": 3.0,
            "beta": 2.0,
            "total_games": 500,
            "total_elo_gain": 75.0,
            "observation_count": 10,
            "elo_gains": [1.0, 2.0],
            "timestamps": [100.0, 200.0],
        }

        stats = EngineStats.from_dict(data)

        assert stats.engine == "mcts"
        assert stats.config_key == "square8_2p"
        assert stats.alpha == 3.0
        assert stats.beta == 2.0
        assert stats.total_games == 500
        assert stats.total_elo_gain == 75.0

    def test_from_dict_handles_missing_optional_fields(self):
        """Test deserialization handles missing optional fields."""
        data = {
            "engine": "heuristic-only",
            "config_key": "hex8_4p",
        }

        stats = EngineStats.from_dict(data)

        assert stats.engine == "heuristic-only"
        assert stats.alpha == 1.0  # Default
        assert stats.beta == 1.0  # Default
        assert stats.total_games == 0
        assert stats.elo_gains == []

    def test_roundtrip_serialization(self):
        """Test serialization then deserialization preserves data."""
        original = EngineStats(
            engine="mixed",
            config_key="hexagonal_3p",
            alpha=4.0,
            beta=2.5,
            total_games=1000,
            total_elo_gain=150.0,
            observation_count=20,
        )

        data = original.to_dict()
        restored = EngineStats.from_dict(data)

        assert restored.engine == original.engine
        assert restored.config_key == original.config_key
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.total_games == original.total_games
        assert restored.total_elo_gain == original.total_elo_gain


class TestSelfplayEngineBandit:
    """Tests for SelfplayEngineBandit class."""

    @pytest.fixture
    def temp_state_path(self):
        """Create a temporary state file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "engine_bandit.json"

    @pytest.fixture
    def bandit(self, temp_state_path):
        """Create a fresh bandit instance."""
        return SelfplayEngineBandit(state_path=temp_state_path)

    def test_initialization(self, bandit):
        """Test bandit initializes correctly."""
        assert bandit._stats == {}
        assert bandit._exploration_bonus == 0.1

    def test_initialization_custom_params(self, temp_state_path):
        """Test bandit with custom parameters."""
        bandit = SelfplayEngineBandit(
            state_path=temp_state_path,
            exploration_bonus=0.2,
            decay_rate=0.002,
        )

        assert bandit._exploration_bonus == 0.2
        assert bandit._decay_rate == 0.002

    def test_select_engine_returns_valid_engine(self, bandit):
        """Test select_engine returns a valid engine."""
        engine = bandit.select_engine("hex8_2p")
        assert engine in AVAILABLE_ENGINES

    def test_select_engine_with_custom_available(self, bandit):
        """Test select_engine with custom available engines."""
        engine = bandit.select_engine("hex8_2p", available_engines=["heuristic-only", "mcts"])
        assert engine in ["heuristic-only", "mcts"]

    def test_select_engine_explores_all_engines(self, bandit):
        """Test that select_engine explores all engines over many calls."""
        engines_seen = set()

        # With Thompson Sampling and exploration bonus, should see all engines
        for _ in range(100):
            engine = bandit.select_engine("hex8_2p")
            engines_seen.add(engine)

        # Should have explored at least 3 different engines
        assert len(engines_seen) >= 3

    def test_select_engine_creates_stats(self, bandit):
        """Test that select_engine creates stats for new configs."""
        assert "new_config" not in bandit._stats

        bandit.select_engine("new_config")

        assert "new_config" in bandit._stats
        # Should have stats for at least the selected engine
        assert len(bandit._stats["new_config"]) >= 1

    def test_record_feedback_updates_stats(self, bandit):
        """Test record_feedback updates engine statistics."""
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=10.0, games=500)

        stats = bandit._stats["hex8_2p"]["gumbel-mcts"]
        assert stats.total_games == 500
        assert stats.total_elo_gain == 10.0
        assert stats.observation_count == 1

    def test_record_feedback_accumulates(self, bandit):
        """Test record_feedback accumulates over multiple calls."""
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=10.0, games=500)
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=5.0, games=300)

        stats = bandit._stats["hex8_2p"]["gumbel-mcts"]
        assert stats.total_games == 800
        assert stats.total_elo_gain == 15.0
        assert stats.observation_count == 2

    def test_record_feedback_saves_state(self, bandit, temp_state_path):
        """Test record_feedback persists state to disk."""
        bandit.record_feedback("hex8_2p", "mcts", elo_gain=20.0, games=1000)

        # Check file was created
        assert temp_state_path.exists()

        # Check content
        with open(temp_state_path) as f:
            data = json.load(f)

        assert "hex8_2p" in data["stats"]
        assert "mcts" in data["stats"]["hex8_2p"]

    def test_load_state_on_init(self, temp_state_path):
        """Test state is loaded on initialization."""
        # Create initial bandit and add data
        bandit1 = SelfplayEngineBandit(state_path=temp_state_path)
        bandit1.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=15.0, games=200)

        # Create new bandit - should load state
        bandit2 = SelfplayEngineBandit(state_path=temp_state_path)

        stats = bandit2._stats.get("hex8_2p", {}).get("gumbel-mcts")
        assert stats is not None
        assert stats.total_games == 200
        assert stats.total_elo_gain == 15.0

    def test_get_stats_empty(self, bandit):
        """Test get_stats returns empty dict for unknown config."""
        stats = bandit.get_stats("unknown_config")
        assert stats == {}

    def test_get_stats_with_data(self, bandit):
        """Test get_stats returns correct data."""
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=10.0, games=500)
        bandit.record_feedback("hex8_2p", "heuristic-only", elo_gain=5.0, games=1000)

        stats = bandit.get_stats("hex8_2p")

        assert "gumbel-mcts" in stats
        assert "heuristic-only" in stats
        assert stats["gumbel-mcts"]["total_games"] == 500
        assert stats["heuristic-only"]["total_games"] == 1000

    def test_get_best_engine_no_data(self, bandit):
        """Test get_best_engine returns None with no data."""
        assert bandit.get_best_engine("unknown_config") is None

    def test_get_best_engine_insufficient_data(self, bandit):
        """Test get_best_engine returns None with insufficient data."""
        # Add data but below MIN_GAMES_FOR_EXPLOITATION
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=10.0, games=50)

        assert bandit.get_best_engine("hex8_2p") is None

    def test_get_best_engine_with_sufficient_data(self, bandit):
        """Test get_best_engine returns best engine by mean Elo."""
        # Add sufficient data for both engines
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=50.0, games=MIN_GAMES_FOR_EXPLOITATION)
        bandit.record_feedback("hex8_2p", "heuristic-only", elo_gain=10.0, games=MIN_GAMES_FOR_EXPLOITATION)

        # gumbel-mcts has higher mean Elo per game (0.5 vs 0.1)
        assert bandit.get_best_engine("hex8_2p") == "gumbel-mcts"

    def test_get_summary(self, bandit):
        """Test get_summary returns correct structure."""
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=10.0, games=500)
        bandit.record_feedback("square8_2p", "mcts", elo_gain=5.0, games=200)

        summary = bandit.get_summary()

        assert "configs" in summary
        assert "total_observations" in summary
        assert "total_games" in summary
        assert summary["total_observations"] == 2
        assert summary["total_games"] == 700
        assert "hex8_2p" in summary["configs"]
        assert "square8_2p" in summary["configs"]

    def test_exploitation_after_sufficient_data(self, bandit):
        """Test that bandit exploits best engine after sufficient data."""
        # Give gumbel-mcts very high success rate
        bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=100.0, games=200)
        # Give heuristic-only lower success rate
        bandit.record_feedback("hex8_2p", "heuristic-only", elo_gain=5.0, games=200)

        # Over many trials, gumbel-mcts should be selected more often
        selections = {"gumbel-mcts": 0, "heuristic-only": 0}
        for _ in range(100):
            engine = bandit.select_engine("hex8_2p", available_engines=["gumbel-mcts", "heuristic-only"])
            selections[engine] += 1

        # gumbel-mcts should be selected significantly more often
        assert selections["gumbel-mcts"] > selections["heuristic-only"]

    def test_thread_safety_select_engine(self, bandit):
        """Test select_engine is thread-safe."""
        import threading

        errors = []

        def select_many():
            try:
                for _ in range(50):
                    bandit.select_engine("hex8_2p")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=select_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_thread_safety_record_feedback(self, bandit):
        """Test record_feedback is thread-safe."""
        import threading

        errors = []

        def record_many():
            try:
                for i in range(20):
                    bandit.record_feedback("hex8_2p", "gumbel-mcts", elo_gain=float(i), games=10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have accumulated all updates
        stats = bandit._stats["hex8_2p"]["gumbel-mcts"]
        assert stats.observation_count == 100  # 5 threads * 20 updates


class TestSingletonPattern:
    """Tests for singleton pattern functions."""

    def test_get_selfplay_engine_bandit_returns_instance(self):
        """Test get_selfplay_engine_bandit returns an instance."""
        reset_selfplay_engine_bandit()

        bandit = get_selfplay_engine_bandit()
        assert isinstance(bandit, SelfplayEngineBandit)

    def test_get_selfplay_engine_bandit_returns_same_instance(self):
        """Test get_selfplay_engine_bandit returns the same instance."""
        reset_selfplay_engine_bandit()

        bandit1 = get_selfplay_engine_bandit()
        bandit2 = get_selfplay_engine_bandit()

        assert bandit1 is bandit2

    def test_reset_selfplay_engine_bandit(self):
        """Test reset_selfplay_engine_bandit clears the singleton."""
        bandit1 = get_selfplay_engine_bandit()
        reset_selfplay_engine_bandit()
        bandit2 = get_selfplay_engine_bandit()

        assert bandit1 is not bandit2


class TestConstants:
    """Tests for module constants."""

    def test_available_engines_not_empty(self):
        """Test AVAILABLE_ENGINES is not empty."""
        assert len(AVAILABLE_ENGINES) > 0

    def test_default_engine_in_available(self):
        """Test DEFAULT_ENGINE is in AVAILABLE_ENGINES."""
        assert DEFAULT_ENGINE in AVAILABLE_ENGINES

    def test_min_games_for_exploitation_positive(self):
        """Test MIN_GAMES_FOR_EXPLOITATION is positive."""
        assert MIN_GAMES_FOR_EXPLOITATION > 0

    def test_expected_engines_present(self):
        """Test expected engine modes are present."""
        expected = ["heuristic-only", "gumbel-mcts", "mcts"]
        for engine in expected:
            assert engine in AVAILABLE_ENGINES
