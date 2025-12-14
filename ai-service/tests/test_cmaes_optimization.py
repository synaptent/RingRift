"""Tests for CMA-ES heuristic weight optimization.

This module tests the core functionality of the CMA-ES optimization script,
including weight flattening/unflattening, fitness evaluation, and the overall
CMA-ES integration.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Allow imports from app/
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.ai.heuristic_weights import (  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HeuristicWeights,
)
from app.models import BoardType  # noqa: E402

# Import the functions to test from the script
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
    ),
)

# We need to mock cma before importing the optimization module
cma_mock = MagicMock()
sys.modules["cma"] = cma_mock

from run_cmaes_optimization import (  # noqa: E402
    WEIGHT_KEYS,
    array_to_weights,
    create_game_state,
    create_heuristic_ai_with_weights,
    evaluate_fitness,
    load_weights_from_file,
    play_single_game,
    save_weights_to_file,
    weights_to_array,
)


class TestWeightConversion:
    """Tests for weight flattening and unflattening."""

    def test_weights_to_array_preserves_order(self):
        """Test that weights_to_array produces values in WEIGHT_KEYS order."""
        weights = dict(BASE_V1_BALANCED_WEIGHTS)
        arr = weights_to_array(weights)

        assert len(arr) == len(WEIGHT_KEYS)
        for i, key in enumerate(WEIGHT_KEYS):
            assert arr[i] == weights[key], f"Mismatch for {key}"

    def test_array_to_weights_reconstructs_dict(self):
        """Test that array_to_weights produces valid HeuristicWeights."""
        arr = np.array([float(i) for i in range(len(WEIGHT_KEYS))])
        weights = array_to_weights(arr)

        assert len(weights) == len(WEIGHT_KEYS)
        for i, key in enumerate(WEIGHT_KEYS):
            assert weights[key] == float(i), f"Mismatch for {key}"

    def test_roundtrip_conversion(self):
        """Test that converting to array and back preserves values."""
        original_weights = dict(BASE_V1_BALANCED_WEIGHTS)
        arr = weights_to_array(original_weights)
        recovered_weights = array_to_weights(arr)

        assert set(recovered_weights.keys()) == set(original_weights.keys())
        for key in WEIGHT_KEYS:
            assert recovered_weights[key] == pytest.approx(
                original_weights[key]
            ), f"Mismatch for {key}"

    def test_roundtrip_with_custom_values(self):
        """Test roundtrip with non-default weight values."""
        custom_weights: HeuristicWeights = {
            key: float(i * 2.5 + 0.1) for i, key in enumerate(WEIGHT_KEYS)
        }
        arr = weights_to_array(custom_weights)
        recovered_weights = array_to_weights(arr)

        for key in WEIGHT_KEYS:
            assert recovered_weights[key] == pytest.approx(custom_weights[key])


class TestWeightPersistence:
    """Tests for saving and loading weights to/from files."""

    def test_save_and_load_weights(self, tmp_path):
        """Test saving weights to JSON and loading them back."""
        weights = dict(BASE_V1_BALANCED_WEIGHTS)
        path = tmp_path / "test_weights.json"

        save_weights_to_file(weights, str(path), generation=5, fitness=0.75)

        # Verify file content
        with open(path, "r") as f:
            data = json.load(f)

        assert "weights" in data
        assert data["generation"] == 5
        assert data["fitness"] == 0.75
        assert "timestamp" in data

        # Load back
        loaded = load_weights_from_file(str(path))
        for key in WEIGHT_KEYS:
            assert loaded[key] == weights[key]

    def test_load_flat_format(self, tmp_path):
        """Test loading weights from flat JSON format (no 'weights' key)."""
        weights = dict(BASE_V1_BALANCED_WEIGHTS)
        path = tmp_path / "flat_weights.json"

        with open(path, "w") as f:
            json.dump(weights, f)

        loaded = load_weights_from_file(str(path))
        for key in WEIGHT_KEYS:
            assert loaded[key] == weights[key]


class TestGameStateCreation:
    """Tests for game state creation."""

    def test_create_game_state_square8(self):
        """Test creating game state for Square8 board."""
        state = create_game_state(BoardType.SQUARE8)

        assert state.board_type == BoardType.SQUARE8
        assert state.board.size == 8
        assert len(state.players) == 2
        assert state.current_player == 1

    def test_create_game_state_square19(self):
        """Test creating game state for Square19 board."""
        state = create_game_state(BoardType.SQUARE19)

        assert state.board_type == BoardType.SQUARE19
        assert state.board.size == 19

    def test_create_game_state_hexagonal(self):
        """Test creating game state for Hexagonal board."""
        state = create_game_state(BoardType.HEXAGONAL)

        assert state.board_type == BoardType.HEXAGONAL
        assert state.board.size == 11


class TestHeuristicAICreation:
    """Tests for creating HeuristicAI instances with custom weights."""

    def test_create_ai_with_custom_weights(self):
        """Test that custom weights are applied to the AI."""
        custom_weights: HeuristicWeights = {
            key: 99.0 for key in WEIGHT_KEYS
        }

        ai = create_heuristic_ai_with_weights(1, custom_weights, difficulty=5)

        assert ai.player_number == 1
        # Check that weights were applied
        for key in WEIGHT_KEYS:
            assert getattr(ai, key) == 99.0


class TestFitnessEvaluation:
    """Tests for fitness evaluation with mock game results."""

    @patch("run_cmaes_optimization.play_single_game")
    def test_fitness_all_wins(self, mock_play):
        """Test fitness calculation when candidate wins all games."""
        # Candidate wins every game in a small, fixed number of moves.
        mock_play.return_value = (1, 10)

        fitness = evaluate_fitness(
            dict(BASE_V1_BALANCED_WEIGHTS),
            dict(BASE_V1_BALANCED_WEIGHTS),
            games_per_eval=4,
        )

        assert fitness == 1.0
        assert mock_play.call_count == 4

    @patch("run_cmaes_optimization.play_single_game")
    def test_fitness_all_losses(self, mock_play):
        """Test fitness calculation when candidate loses all games."""
        mock_play.return_value = (-1, 10)  # Baseline wins

        fitness = evaluate_fitness(
            dict(BASE_V1_BALANCED_WEIGHTS),
            dict(BASE_V1_BALANCED_WEIGHTS),
            games_per_eval=4,
        )

        assert fitness == 0.0
        assert mock_play.call_count == 4

    @patch("run_cmaes_optimization.play_single_game")
    def test_fitness_all_draws(self, mock_play):
        """Test fitness calculation when all games are draws."""
        mock_play.return_value = (0, 10)  # Draw

        fitness = evaluate_fitness(
            dict(BASE_V1_BALANCED_WEIGHTS),
            dict(BASE_V1_BALANCED_WEIGHTS),
            games_per_eval=4,
        )

        # Draws count as 0.5
        assert fitness == 0.5
        assert mock_play.call_count == 4

    @patch("run_cmaes_optimization.play_single_game")
    def test_fitness_mixed_results(self, mock_play):
        """Test fitness with mixed win/loss/draw results."""
        # 2 wins, 1 loss, 1 draw = (2 + 0.5) / 4 = 0.625
        mock_play.side_effect = [
            (1, 10),
            (-1, 20),
            (1, 30),
            (0, 40),
        ]

        fitness = evaluate_fitness(
            dict(BASE_V1_BALANCED_WEIGHTS),
            dict(BASE_V1_BALANCED_WEIGHTS),
            games_per_eval=4,
        )

        assert fitness == pytest.approx(0.625)
        assert mock_play.call_count == 4

    @patch("run_cmaes_optimization.play_single_game")
    def test_fitness_alternates_first_player_and_passes_max_moves(
        self, mock_play
    ):
        """Test that candidate alternates playing first and max_moves is wired."""
        mock_play.return_value = (0, 10)

        evaluate_fitness(
            dict(BASE_V1_BALANCED_WEIGHTS),
            dict(BASE_V1_BALANCED_WEIGHTS),
            games_per_eval=4,
            max_moves=123,
        )

        calls = mock_play.call_args_list
        # Check alternating first player (candidate_plays_first arg)
        assert calls[0][0][2] is True   # Game 0: candidate first
        assert calls[1][0][2] is False  # Game 1: baseline first
        assert calls[2][0][2] is True   # Game 2: candidate first
        assert calls[3][0][2] is False  # Game 3: baseline first

        # All calls should receive the configured max_moves.
        assert all(call[0][4] == 123 for call in calls)

    @patch("run_cmaes_optimization.play_single_game")
    def test_multi_opponent_schedule_split(self, mock_play):
        """Test baseline-plus-incumbent mode splits games between opponents."""
        mock_play.return_value = (1, 5)

        baseline = {"id": "baseline"}  # Sentinel object for identity checks
        incumbent = {"id": "incumbent"}

        games_per_eval = 6
        evaluate_fitness(
            candidate_weights=baseline,
            baseline_weights=baseline,
            games_per_eval=games_per_eval,
            opponent_mode="baseline-plus-incumbent",
            incumbent_weights=incumbent,
        )

        assert mock_play.call_count == games_per_eval
        calls = mock_play.call_args_list

        # 2:1 split => for 6 games: 4 vs baseline, 2 vs incumbent.
        for call in calls[:4]:
            # Opponent weights are the second positional argument.
            assert call[0][1] is baseline
        for call in calls[4:]:
            assert call[0][1] is incumbent

    @patch("run_cmaes_optimization.play_single_game")
    def test_multi_opponent_falls_back_to_baseline_when_no_incumbent(
        self, mock_play
    ):
        """Test that baseline-plus-incumbent falls back to baseline when no incumbent."""
        mock_play.return_value = (0, 5)

        baseline = {"id": "baseline"}

        games_per_eval = 5
        evaluate_fitness(
            candidate_weights=baseline,
            baseline_weights=baseline,
            games_per_eval=games_per_eval,
            opponent_mode="baseline-plus-incumbent",
            incumbent_weights=None,
        )

        assert mock_play.call_count == games_per_eval
        # All opponents should be the baseline when incumbent is None.
        for call in mock_play.call_args_list:
            assert call[0][1] is baseline


class TestCMAESIntegration:
    """Integration tests for CMA-ES optimization."""

    def test_cmaes_with_mocked_es(self):
        """Test CMA-ES integration with mocked evolution strategy."""
        # This test verifies the optimization loop structure
        # without running actual games

        from run_cmaes_optimization import (
            CMAESConfig,
            run_cmaes_optimization,
        )

        # Create a mock CMAEvolutionStrategy
        mock_es_instance = MagicMock()
        mock_es_instance.ask.return_value = [
            weights_to_array(BASE_V1_BALANCED_WEIGHTS).tolist(),
            weights_to_array(BASE_V1_BALANCED_WEIGHTS).tolist(),
        ]

        cma_mock.CMAEvolutionStrategy.return_value = mock_es_instance

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "output.json")
            checkpoint_dir = os.path.join(tmp_dir, "checkpoints")

            config = CMAESConfig(
                generations=2,
                population_size=2,
                games_per_eval=2,
                sigma=0.5,
                output_path=output_path,
                baseline_path=None,
                board_type=BoardType.SQUARE8,
                checkpoint_dir=checkpoint_dir,
                run_id="integration_test_run",
            )

            with patch(
                "run_cmaes_optimization.evaluate_fitness"
            ) as mock_fitness:
                mock_fitness.return_value = 0.6

                run_cmaes_optimization(config)

            # Verify CMAEvolutionStrategy was initialized correctly
            cma_mock.CMAEvolutionStrategy.assert_called_once()
            call_args = cma_mock.CMAEvolutionStrategy.call_args
            # Options are passed as 3rd positional argument (a dict)
            options_dict = call_args[0][2]
            assert options_dict["popsize"] == 2
            assert options_dict["maxiter"] == 2

            # Verify tell was called with fitness values
            assert mock_es_instance.tell.call_count == 2

            # Verify output file was created
            assert os.path.exists(output_path)

            # Verify checkpoint was created
            assert os.path.exists(checkpoint_dir)

    def test_run_writes_run_meta_and_generation_summary(self, tmp_path):
        """Tiny run writes run_meta.json and per-generation summaries."""
        from run_cmaes_optimization import (
            CMAESConfig,
            run_cmaes_optimization,
        )

        mock_es_instance = MagicMock()
        mock_es_instance.ask.return_value = [
            weights_to_array(BASE_V1_BALANCED_WEIGHTS).tolist(),
            weights_to_array(BASE_V1_BALANCED_WEIGHTS).tolist(),
        ]
        cma_mock.CMAEvolutionStrategy.return_value = mock_es_instance

        output_path = os.path.join(tmp_path, "output.json")
        checkpoint_dir = os.path.join(tmp_path, "checkpoints")
        run_id = "meta_test_run"

        config = CMAESConfig(
            generations=1,
            population_size=2,
            games_per_eval=1,
            sigma=0.5,
            output_path=output_path,
            baseline_path=None,
            board_type=BoardType.SQUARE8,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
        )

        with patch("run_cmaes_optimization.evaluate_fitness") as mock_fitness:
            mock_fitness.return_value = 0.5
            run_cmaes_optimization(config)

        # Derive run directory using the same convention as the driver.
        output_dir = os.path.dirname(output_path) or "logs/cmaes"
        run_dir = os.path.join(output_dir, "runs", run_id)

        run_meta_path = os.path.join(run_dir, "run_meta.json")
        assert os.path.exists(run_meta_path)

        with open(run_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["run_id"] == run_id
        assert meta["generations"] == 1
        assert meta["population_size"] == 2
        assert meta["games_per_eval"] == 1
        assert meta["checkpoint_dir"] == checkpoint_dir

        generations_dir = os.path.join(run_dir, "generations")
        assert os.path.isdir(generations_dir)
        files = os.listdir(generations_dir)
        assert any(name.startswith("generation_001") for name in files)

        gen_path = os.path.join(generations_dir, "generation_001.json")
        with open(gen_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        assert gen_data["generation"] == 1
        assert gen_data["population_size"] == 2
        assert gen_data["games_per_eval"] == 1
        assert "mean_fitness" in gen_data
        assert "best_candidate" in gen_data


class TestPlaySingleGame:
    """Tests for single game play (integration with game engine)."""

    def test_game_terminates(self):
        """Test that a game eventually terminates or hits move limit."""
        # This is a smoke test - we just want to make sure the game
        # doesn't hang and produces a valid result

        weights = dict(BASE_V1_BALANCED_WEIGHTS)

        # Use a very low max_moves to ensure quick termination
        result, move_count = play_single_game(
            weights,
            weights,
            candidate_plays_first=True,
            board_type=BoardType.SQUARE8,
            max_moves=10,
        )

        # Result should be -1, 0, or 1 and move_count should be non-negative.
        assert result in [-1, 0, 1]
        assert move_count >= 0

    def test_game_swaps_player_order(self):
        """Test that candidate_plays_first affects player assignment."""
        weights = dict(BASE_V1_BALANCED_WEIGHTS)

        # Just verify games run without error in both configurations
        result1, moves1 = play_single_game(
            weights,
            weights,
            candidate_plays_first=True,
            board_type=BoardType.SQUARE8,
            max_moves=5,
        )
        result2, moves2 = play_single_game(
            weights,
            weights,
            candidate_plays_first=False,
            board_type=BoardType.SQUARE8,
            max_moves=5,
        )

        assert result1 in [-1, 0, 1]
        assert result2 in [-1, 0, 1]
        assert moves1 >= 0
        assert moves2 >= 0


class TestCLIWiring:
    """Tests for CLI argument wiring into CMAESConfig."""

    def test_cli_passes_new_fields_to_config(self, monkeypatch):
        from run_cmaes_optimization import (
            CMAESConfig,
            main as cmaes_main,
        )

        test_argv = [
            "run_cmaes_optimization.py",
            "--generations",
            "3",
            "--population-size",
            "5",
            "--games-per-eval",
            "7",
            "--sigma",
            "0.8",
            "--output",
            "logs/cmaes/cli_output.json",
            "--baseline",
            "baseline.json",
            "--board",
            "square8",
            "--checkpoint-dir",
            "logs/cmaes/cli_checkpoints",
            "--seed",
            "123",
            "--max-moves",
            "222",
            "--run-id",
            "cli_run",
            "--resume-from",
            "logs/cmaes/runs/existing_run",
            "--opponent-mode",
            "baseline-plus-incumbent",
        ]
        monkeypatch.setattr(sys, "argv", test_argv)

        captured = {}

        def fake_run(config):
            captured["config"] = config

        monkeypatch.setattr(
            "run_cmaes_optimization.run_cmaes_optimization", fake_run
        )

        cmaes_main()

        config = captured["config"]
        assert isinstance(config, CMAESConfig)
        assert config.generations == 3
        assert config.population_size == 5
        assert config.games_per_eval == 7
        assert config.sigma == pytest.approx(0.8)
        assert config.output_path == "logs/cmaes/cli_output.json"
        assert config.baseline_path == "baseline.json"
        assert config.board_type == BoardType.SQUARE8
        assert config.checkpoint_dir == "logs/cmaes/cli_checkpoints"
        assert config.seed == 123
        assert config.max_moves == 222
        assert config.opponent_mode == "baseline-plus-incumbent"
        assert config.run_id == "cli_run"
        assert config.run_dir == "logs/cmaes/runs/existing_run"
        assert config.resume_from == "logs/cmaes/runs/existing_run"
        # New progress-related defaults
        assert config.progress_interval_sec == pytest.approx(10.0)
        assert config.enable_eval_progress is True

    def test_cli_directory_output_sets_run_dir_and_output_path(self, monkeypatch):
        from run_cmaes_optimization import (
            CMAESConfig,
            main as cmaes_main,
        )

        test_argv = [
            "run_cmaes_optimization.py",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--games-per-eval",
            "3",
            "--sigma",
            "0.5",
            "--output",
            "logs/cmaes/dir_mode",
            "--run-id",
            "cli_run",
        ]
        monkeypatch.setattr(sys, "argv", test_argv)

        captured = {}

        def fake_run(config):
            captured["config"] = config

        monkeypatch.setattr(
            "run_cmaes_optimization.run_cmaes_optimization", fake_run
        )

        cmaes_main()

        config = captured["config"]
        assert isinstance(config, CMAESConfig)

        expected_run_dir = os.path.join("logs/cmaes/dir_mode", "runs", "cli_run")
        assert config.run_dir == expected_run_dir
        assert config.output_path == os.path.join(
            expected_run_dir, "best_weights.json"
        )
        assert config.checkpoint_dir == os.path.join(
            expected_run_dir, "checkpoints"
        )
        assert config.max_moves == 200
        assert config.opponent_mode == "baseline-only"
        # Default wiring: eval progress enabled unless explicitly disabled
        assert config.enable_eval_progress is True

    def test_cli_disable_eval_progress_flag(self, monkeypatch):
        from run_cmaes_optimization import (
            CMAESConfig,
            main as cmaes_main,
        )

        test_argv = [
            "run_cmaes_optimization.py",
            "--generations",
            "1",
            "--population-size",
            "2",
            "--games-per-eval",
            "3",
            "--sigma",
            "0.5",
            "--output",
            "logs/cmaes/dir_mode",
            "--run-id",
            "cli_run",
            "--disable-eval-progress",
        ]
        monkeypatch.setattr(sys, "argv", test_argv)

        captured = {}

        def fake_run(config):
            captured["config"] = config

        monkeypatch.setattr(
            "run_cmaes_optimization.run_cmaes_optimization", fake_run
        )

        cmaes_main()

        config = captured["config"]
        assert isinstance(config, CMAESConfig)
        assert config.enable_eval_progress is False
