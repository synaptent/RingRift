"""Tests for app.ai.gumbel_search_engine module.

Tests the unified Gumbel MCTS search engine:
- SearchMode enum
- SearchConfig dataclass and factory methods
- SearchResult and BatchSearchResult dataclasses
- GumbelSearchEngine class and factory methods
- Convenience factory functions

Created Dec 2025 as part of Phase 3 test coverage improvement.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ai.gumbel_common import (
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_THROUGHPUT,
)
from app.ai.gumbel_search_engine import (
    BatchSearchResult,
    GumbelSearchEngine,
    SearchConfig,
    SearchMode,
    SearchResult,
    create_evaluation_engine,
    create_play_engine,
    create_selfplay_engine,
)


# =============================================================================
# SearchMode Enum Tests
# =============================================================================


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_all_modes_exist(self):
        """Verify all expected search modes are defined."""
        expected = [
            "SINGLE_GAME",
            "SINGLE_GAME_FAST",
            "MULTI_GAME_BATCH",
            "MULTI_GAME_PARALLEL",
            "AUTO",
        ]
        for name in expected:
            assert hasattr(SearchMode, name), f"Missing SearchMode.{name}"

    def test_mode_values(self):
        """Verify search mode values are strings."""
        assert SearchMode.SINGLE_GAME.value == "single_game"
        assert SearchMode.SINGLE_GAME_FAST.value == "single_game_fast"
        assert SearchMode.MULTI_GAME_BATCH.value == "multi_batch"
        assert SearchMode.MULTI_GAME_PARALLEL.value == "multi_parallel"
        assert SearchMode.AUTO.value == "auto"

    def test_modes_are_unique(self):
        """Verify all modes have unique values."""
        values = [m.value for m in SearchMode]
        assert len(values) == len(set(values)), "Duplicate mode values"


# =============================================================================
# SearchConfig Tests
# =============================================================================


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SearchConfig()
        assert config.simulation_budget == GUMBEL_BUDGET_STANDARD
        assert config.num_sampled_actions == 16
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25

    def test_for_throughput(self):
        """Test throughput-optimized config."""
        config = SearchConfig.for_throughput()
        assert config.simulation_budget == GUMBEL_BUDGET_THROUGHPUT
        assert config.num_sampled_actions == 8
        assert config.temperature == 1.0

    def test_for_quality(self):
        """Test quality-optimized config."""
        config = SearchConfig.for_quality()
        assert config.simulation_budget == GUMBEL_BUDGET_QUALITY
        assert config.num_sampled_actions == 16
        assert config.temperature == 0.0  # Deterministic

    def test_for_balanced(self):
        """Test balanced config."""
        config = SearchConfig.for_balanced()
        assert config.simulation_budget == GUMBEL_BUDGET_STANDARD
        assert config.num_sampled_actions == 16
        assert config.temperature == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            simulation_budget=200,
            num_sampled_actions=32,
            temperature=0.5,
            c_puct=2.0,
        )
        assert config.simulation_budget == 200
        assert config.num_sampled_actions == 32
        assert config.temperature == 0.5
        assert config.c_puct == 2.0


# =============================================================================
# SearchResult Tests
# =============================================================================


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self):
        """Test creating a search result."""
        mock_move = MagicMock()
        result = SearchResult(
            move=mock_move,
            visit_counts={"a1": 100, "b2": 50},
            value=0.75,
            policy={"a1": 0.6, "b2": 0.4},
        )
        assert result.move is mock_move
        assert result.visit_counts["a1"] == 100
        assert result.value == 0.75
        assert result.policy["a1"] == 0.6


class TestBatchSearchResult:
    """Tests for BatchSearchResult dataclass."""

    def test_create_batch_result(self):
        """Test creating a batch search result."""
        mock_moves = [MagicMock(), MagicMock()]
        game_results = [
            {"game_idx": 0, "winner": 1},
            {"game_idx": 1, "winner": 2},
        ]
        result = BatchSearchResult(moves=mock_moves, game_results=game_results)
        assert len(result.moves) == 2
        assert len(result.game_results) == 2
        assert result.game_results[0]["winner"] == 1


# =============================================================================
# GumbelSearchEngine Initialization Tests
# =============================================================================


class TestGumbelSearchEngineInit:
    """Tests for GumbelSearchEngine initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        engine = GumbelSearchEngine()
        assert engine.neural_net is None
        assert engine.mode == SearchMode.AUTO
        assert engine.device == "cuda"
        assert engine.num_games == 64
        assert engine.num_players == 2

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        mock_nn = MagicMock()
        engine = GumbelSearchEngine(
            neural_net=mock_nn,
            mode=SearchMode.SINGLE_GAME,
            device="cpu",
            num_games=32,
            num_players=4,
        )
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.SINGLE_GAME
        assert engine.device == "cpu"
        assert engine.num_games == 32
        assert engine.num_players == 4

    def test_init_with_custom_config(self):
        """Test initialization with custom search config."""
        config = SearchConfig(simulation_budget=500)
        engine = GumbelSearchEngine(config=config)
        assert engine.config.simulation_budget == 500

    def test_init_uses_balanced_config_by_default(self):
        """Test that default config is balanced."""
        engine = GumbelSearchEngine()
        assert engine.config.simulation_budget == GUMBEL_BUDGET_STANDARD


# =============================================================================
# GumbelSearchEngine Factory Methods Tests
# =============================================================================


class TestGumbelSearchEngineFactory:
    """Tests for GumbelSearchEngine factory methods."""

    def test_for_selfplay(self):
        """Test for_selfplay factory method."""
        mock_nn = MagicMock()
        engine = GumbelSearchEngine.for_selfplay(
            neural_net=mock_nn,
            board_type="hex8",
            num_players=2,
            num_games=128,
            device="cuda",
        )
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.MULTI_GAME_PARALLEL
        assert engine.config.simulation_budget == GUMBEL_BUDGET_QUALITY  # Jan 2026: selfplay uses quality budget
        assert engine.num_games == 128
        assert engine.device == "cuda"

    def test_for_evaluation(self):
        """Test for_evaluation factory method."""
        mock_nn = MagicMock()
        engine = GumbelSearchEngine.for_evaluation(neural_net=mock_nn)
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.SINGLE_GAME
        assert engine.config.simulation_budget == GUMBEL_BUDGET_QUALITY
        assert engine.config.temperature == 0.0  # Deterministic

    def test_for_play(self):
        """Test for_play factory method."""
        mock_nn = MagicMock()
        engine = GumbelSearchEngine.for_play(neural_net=mock_nn)
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.SINGLE_GAME
        assert engine.config.simulation_budget == GUMBEL_BUDGET_STANDARD


# =============================================================================
# GumbelSearchEngine Search Methods Tests
# =============================================================================


class TestGumbelSearchEngineMethods:
    """Tests for GumbelSearchEngine search methods."""

    @patch("app.ai.gumbel_search_engine.GumbelSearchEngine._get_single_game_ai")
    def test_search_calls_backend(self, mock_get_ai):
        """Test search method calls backend correctly."""
        mock_ai = MagicMock()
        mock_move = MagicMock()
        mock_ai.select_move.return_value = mock_move
        mock_get_ai.return_value = mock_ai

        mock_state = MagicMock()
        engine = GumbelSearchEngine()
        result = engine.search(mock_state, move_number=5)

        mock_ai.select_move.assert_called_once()
        assert result is mock_move

    @patch("app.ai.gumbel_search_engine.GumbelSearchEngine._get_single_game_ai")
    def test_search_uses_temperature_threshold(self, mock_get_ai):
        """Test search switches to greedy after threshold."""
        mock_ai = MagicMock()
        mock_ai.select_move.return_value = MagicMock()
        mock_get_ai.return_value = mock_ai

        mock_state = MagicMock()
        config = SearchConfig(temperature=1.0, temperature_threshold=20)
        engine = GumbelSearchEngine(config=config)

        # Before threshold - use temperature
        engine.search(mock_state, move_number=10)
        args, kwargs = mock_ai.select_move.call_args
        assert kwargs.get("temperature") == 1.0

        # After threshold - greedy (temp=0)
        engine.search(mock_state, move_number=25)
        args, kwargs = mock_ai.select_move.call_args
        assert kwargs.get("temperature") == 0.0

    @patch("app.ai.gumbel_search_engine.GumbelSearchEngine._get_single_game_ai")
    def test_search_with_details(self, mock_get_ai):
        """Test search_with_details returns SearchResult."""
        mock_ai = MagicMock()
        mock_move = MagicMock()
        mock_ai.select_move_with_details.return_value = (
            mock_move,
            {"visit_counts": {"a1": 50}, "value": 0.8, "policy": {"a1": 0.9}},
        )
        mock_get_ai.return_value = mock_ai

        mock_state = MagicMock()
        engine = GumbelSearchEngine()
        result = engine.search_with_details(mock_state)

        assert isinstance(result, SearchResult)
        assert result.move is mock_move
        assert result.visit_counts == {"a1": 50}
        assert result.value == 0.8
        assert result.policy == {"a1": 0.9}

    @patch("app.ai.gumbel_search_engine.GumbelSearchEngine._get_multi_game_runner")
    def test_search_batch(self, mock_get_runner):
        """Test search_batch method."""
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.game_idx = 0
        mock_result.winner = 1
        mock_result.move_count = 30
        mock_result.moves = [MagicMock()]
        mock_runner.run_batch.return_value = [mock_result]
        mock_get_runner.return_value = mock_runner

        engine = GumbelSearchEngine()
        result = engine.search_batch(num_games=10)

        mock_runner.run_batch.assert_called_once_with(num_games=10, initial_states=None)
        assert isinstance(result, BatchSearchResult)
        assert len(result.game_results) == 1

    @patch("app.ai.gumbel_search_engine.GumbelSearchEngine._get_multi_game_runner")
    def test_run_selfplay(self, mock_get_runner):
        """Test run_selfplay method."""
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.game_idx = 0
        mock_result.winner = 2
        mock_result.status = "completed"
        mock_result.move_count = 45
        mock_result.moves = []
        mock_result.duration_ms = 1500
        mock_runner.run_batch.return_value = [mock_result]
        mock_get_runner.return_value = mock_runner

        engine = GumbelSearchEngine(num_games=50)
        results = engine.run_selfplay()

        mock_runner.run_batch.assert_called_once_with(num_games=50)
        assert len(results) == 1
        assert results[0]["winner"] == 2
        assert results[0]["duration_ms"] == 1500


# =============================================================================
# Convenience Factory Function Tests
# =============================================================================


class TestConvenienceFactories:
    """Tests for convenience factory functions."""

    def test_create_selfplay_engine(self):
        """Test create_selfplay_engine function."""
        mock_nn = MagicMock()
        engine = create_selfplay_engine(
            neural_net=mock_nn,
            board_type="square8",
            num_players=4,
            num_games=100,
            device="cpu",
        )
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.MULTI_GAME_PARALLEL
        assert engine.num_games == 100
        assert engine.device == "cpu"

    def test_create_evaluation_engine(self):
        """Test create_evaluation_engine function."""
        mock_nn = MagicMock()
        engine = create_evaluation_engine(mock_nn)
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.SINGLE_GAME
        assert engine.config.temperature == 0.0

    def test_create_play_engine(self):
        """Test create_play_engine function."""
        mock_nn = MagicMock()
        engine = create_play_engine(mock_nn)
        assert engine.neural_net is mock_nn
        assert engine.mode == SearchMode.SINGLE_GAME
        assert engine.config.simulation_budget == GUMBEL_BUDGET_STANDARD


# =============================================================================
# Backend Loading Tests
# =============================================================================


class TestBackendLoading:
    """Tests for lazy backend loading."""

    def test_backends_none_on_init(self):
        """Test backends are None on initialization."""
        engine = GumbelSearchEngine()
        assert engine._single_game_ai is None
        assert engine._multi_game_runner is None

    def test_single_game_ai_lazy_loaded(self):
        """Test single game AI is lazy loaded."""
        engine = GumbelSearchEngine()

        # Not loaded yet
        assert engine._single_game_ai is None

        # Load on first access - use patch on the method
        with patch.object(engine, "_get_single_game_ai") as mock_get:
            mock_ai = MagicMock()
            mock_ai.select_move.return_value = MagicMock()
            mock_get.return_value = mock_ai
            mock_state = MagicMock()
            engine.search(mock_state)
            mock_get.assert_called_once()


# =============================================================================
# Import Compatibility Tests
# =============================================================================


class TestImports:
    """Tests for module imports."""

    def test_import_module(self):
        """Test that the module can be imported."""
        from app.ai import gumbel_search_engine
        assert gumbel_search_engine is not None

    def test_import_public_classes(self):
        """Test that public classes can be imported."""
        from app.ai.gumbel_search_engine import (
            BatchSearchResult,
            GumbelSearchEngine,
            SearchConfig,
            SearchMode,
            SearchResult,
        )
        assert SearchMode is not None
        assert SearchConfig is not None
        assert SearchResult is not None
        assert BatchSearchResult is not None
        assert GumbelSearchEngine is not None

    def test_import_factory_functions(self):
        """Test that factory functions can be imported."""
        from app.ai.gumbel_search_engine import (
            create_evaluation_engine,
            create_play_engine,
            create_selfplay_engine,
        )
        assert create_selfplay_engine is not None
        assert create_evaluation_engine is not None
        assert create_play_engine is not None
