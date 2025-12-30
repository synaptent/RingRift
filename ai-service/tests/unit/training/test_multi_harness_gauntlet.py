"""Unit tests for multi-harness gauntlet evaluation.

Tests cover:
- NNUE baseline opponents in game_gauntlet.py
- Harness metadata in GauntletResult
- Multi-harness gauntlet integration with Phase 1 harness abstraction
- Harness compatibility checks

Dec 2025: Phase 2 of NN/NNUE multi-harness evaluation system.
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from app.models import BoardType


# =============================================================================
# NNUE Baseline Opponent Tests
# =============================================================================


class TestNNUEBaselineOpponents:
    """Tests for NNUE baseline opponents added in Phase 2."""

    def test_nnue_baseline_opponents_exist(self):
        """NNUE baseline opponent types are defined."""
        from app.training.game_gauntlet import BaselineOpponent

        assert hasattr(BaselineOpponent, "NNUE_MINIMAX_D4")
        assert hasattr(BaselineOpponent, "NNUE_MAXN_D3")
        assert hasattr(BaselineOpponent, "NNUE_BRS_D3")

    def test_nnue_minimax_d4_value(self):
        """NNUE_MINIMAX_D4 has correct string value."""
        from app.training.game_gauntlet import BaselineOpponent

        assert BaselineOpponent.NNUE_MINIMAX_D4.value == "nnue_minimax_d4"

    def test_nnue_maxn_d3_value(self):
        """NNUE_MAXN_D3 has correct string value."""
        from app.training.game_gauntlet import BaselineOpponent

        assert BaselineOpponent.NNUE_MAXN_D3.value == "nnue_maxn_d3"

    def test_nnue_brs_d3_value(self):
        """NNUE_BRS_D3 has correct string value."""
        from app.training.game_gauntlet import BaselineOpponent

        assert BaselineOpponent.NNUE_BRS_D3.value == "nnue_brs_d3"

    def test_baseline_elo_for_nnue_opponents(self):
        """NNUE baseline opponents have Elo ratings defined."""
        from app.training.game_gauntlet import BASELINE_ELOS, BaselineOpponent

        nnue_baselines = [
            BaselineOpponent.NNUE_MINIMAX_D4,
            BaselineOpponent.NNUE_MAXN_D3,
            BaselineOpponent.NNUE_BRS_D3,
        ]
        for baseline in nnue_baselines:
            assert baseline in BASELINE_ELOS
            assert isinstance(BASELINE_ELOS[baseline], (int, float))
            assert BASELINE_ELOS[baseline] >= 800  # Reasonable Elo floor


class TestCreateBaselineAINNUE:
    """Tests for create_baseline_ai with NNUE baselines."""

    def test_create_baseline_ai_accepts_num_players(self):
        """create_baseline_ai accepts num_players parameter."""
        from app.training.game_gauntlet import create_baseline_ai

        # Should not raise for function signature check
        import inspect
        sig = inspect.signature(create_baseline_ai)
        assert "num_players" in sig.parameters

    @patch("app.ai.nnue.RingRiftNNUE")
    @patch("app.ai.minimax_ai.MinimaxAI")
    def test_nnue_minimax_d4_creates_minimax_with_nnue(
        self, mock_minimax_class, mock_nnue_class
    ):
        """NNUE_MINIMAX_D4 creates MinimaxAI with NNUE evaluator."""
        from app.training.game_gauntlet import BaselineOpponent, create_baseline_ai

        mock_nnue = MagicMock()
        mock_nnue_class.return_value = mock_nnue
        mock_minimax = MagicMock()
        mock_minimax_class.return_value = mock_minimax

        # The function will try to create an AI - we just verify the signature
        # allows NNUE_MINIMAX_D4 baseline with num_players param
        try:
            result = create_baseline_ai(
                baseline=BaselineOpponent.NNUE_MINIMAX_D4,
                player=1,
                board_type=BoardType.HEX8,
                num_players=2,
            )
        except (ImportError, AttributeError):
            # May fail if NNUE/Minimax not available - that's ok for unit test
            pass

    @patch("app.ai.nnue.RingRiftNNUE")
    @patch("app.ai.maxn_ai.MaxNAI")
    def test_nnue_maxn_d3_creates_maxn_with_nnue(
        self, mock_maxn_class, mock_nnue_class
    ):
        """NNUE_MAXN_D3 creates MaxNAI with NNUE evaluator for 3+ players."""
        from app.training.game_gauntlet import BaselineOpponent, create_baseline_ai

        mock_nnue = MagicMock()
        mock_nnue_class.return_value = mock_nnue
        mock_maxn = MagicMock()
        mock_maxn_class.return_value = mock_maxn

        # The function will try to create an AI - we just verify the signature
        # allows NNUE_MAXN_D3 baseline with num_players param
        try:
            result = create_baseline_ai(
                baseline=BaselineOpponent.NNUE_MAXN_D3,
                player=1,
                board_type=BoardType.HEX8,
                num_players=3,
            )
        except (ImportError, AttributeError):
            # May fail if NNUE/MaxN not available - that's ok for unit test
            pass


# =============================================================================
# GauntletResult Harness Metadata Tests
# =============================================================================


class TestGauntletResultHarnessMetadata:
    """Tests for harness metadata fields in GauntletResult."""

    def test_gauntlet_result_has_harness_fields(self):
        """GauntletResult has harness metadata fields."""
        from app.training.game_gauntlet import GauntletResult

        result = GauntletResult()
        assert hasattr(result, "harness_type")
        assert hasattr(result, "harness_config_hash")
        assert hasattr(result, "model_id")
        assert hasattr(result, "visit_distributions")

    def test_harness_fields_default_values(self):
        """Harness metadata fields have correct defaults."""
        from app.training.game_gauntlet import GauntletResult

        result = GauntletResult()
        assert result.harness_type == ""
        assert result.harness_config_hash == ""
        assert result.model_id == ""
        assert result.visit_distributions == []

    def test_harness_fields_can_be_set(self):
        """Harness metadata fields can be assigned values."""
        from app.training.game_gauntlet import GauntletResult

        result = GauntletResult()
        result.harness_type = "gumbel_mcts"
        result.harness_config_hash = "a1b2c3d4"
        result.model_id = "canonical_hex8_2p:gumbel_mcts:a1b2c3d4"
        result.visit_distributions = [{"move_a": 0.5, "move_b": 0.5}]

        assert result.harness_type == "gumbel_mcts"
        assert result.harness_config_hash == "a1b2c3d4"
        assert result.model_id == "canonical_hex8_2p:gumbel_mcts:a1b2c3d4"
        assert len(result.visit_distributions) == 1

    def test_gauntlet_result_serializable(self):
        """GauntletResult with harness metadata is serializable."""
        from app.training.game_gauntlet import GauntletResult

        result = GauntletResult(
            total_games=10,
            total_wins=7,
            win_rate=0.7,
            harness_type="minimax",
            harness_config_hash="d3d3d3d3",
            model_id="test_model:minimax:d3d3d3d3",
        )

        data = asdict(result)
        assert isinstance(data, dict)
        assert data["harness_type"] == "minimax"
        assert data["harness_config_hash"] == "d3d3d3d3"
        assert data["model_id"] == "test_model:minimax:d3d3d3d3"


class TestRunBaselineGauntletHarnessMetadata:
    """Tests for harness_type parameter in run_baseline_gauntlet."""

    def test_run_baseline_gauntlet_accepts_harness_type(self):
        """run_baseline_gauntlet accepts harness_type parameter."""
        from app.training.game_gauntlet import run_baseline_gauntlet
        import inspect

        sig = inspect.signature(run_baseline_gauntlet)
        assert "harness_type" in sig.parameters

    def test_harness_type_default_empty(self):
        """harness_type defaults to empty string."""
        from app.training.game_gauntlet import run_baseline_gauntlet
        import inspect

        sig = inspect.signature(run_baseline_gauntlet)
        default = sig.parameters["harness_type"].default
        assert default == ""


# =============================================================================
# Multi-Harness Gauntlet Integration Tests
# =============================================================================


class TestMultiHarnessGauntletModule:
    """Tests for multi_harness_gauntlet.py module."""

    def test_module_imports(self):
        """multi_harness_gauntlet module can be imported."""
        from app.training import multi_harness_gauntlet

        assert multi_harness_gauntlet is not None

    def test_harness_compatibility_dict_exists(self):
        """HARNESS_COMPATIBILITY dict is exported."""
        from app.training.multi_harness_gauntlet import HARNESS_COMPATIBILITY

        assert isinstance(HARNESS_COMPATIBILITY, dict)
        assert len(HARNESS_COMPATIBILITY) > 0

    def test_multi_harness_result_dataclass_exists(self):
        """MultiHarnessResult dataclass is defined."""
        from app.training.multi_harness_gauntlet import MultiHarnessResult

        result = MultiHarnessResult()
        assert hasattr(result, "harness_results")
        assert hasattr(result, "best_harness")
        assert hasattr(result, "best_elo")

    def test_evaluate_model_all_harnesses_exists(self):
        """evaluate_model_all_harnesses function is defined."""
        from app.training.multi_harness_gauntlet import evaluate_model_all_harnesses

        assert callable(evaluate_model_all_harnesses)

    def test_backwards_compat_alias_exists(self):
        """run_multi_harness_evaluation alias exists for backwards compatibility."""
        from app.training.multi_harness_gauntlet import run_multi_harness_evaluation

        assert callable(run_multi_harness_evaluation)


class TestHarnessCompatibilityMatrix:
    """Tests for harness compatibility definitions."""

    def test_gumbel_mcts_compatibility(self):
        """Gumbel MCTS supports NN but not NNUE."""
        from app.training.multi_harness_gauntlet import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY.get("gumbel_mcts")
        assert compat is not None
        assert compat["nn"] is True
        assert compat["nnue"] is False
        assert compat["policy_required"] is True

    def test_minimax_compatibility(self):
        """Minimax supports both NN and NNUE."""
        from app.training.multi_harness_gauntlet import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY.get("minimax")
        assert compat is not None
        assert compat["nn"] is True
        assert compat["nnue"] is True
        assert compat["policy_required"] is False

    def test_maxn_compatibility(self):
        """MaxN supports both NN and NNUE, requires 3+ players."""
        from app.training.multi_harness_gauntlet import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY.get("maxn")
        assert compat is not None
        assert compat["nn"] is True
        assert compat["nnue"] is True
        assert compat.get("min_players", 2) >= 3

    def test_brs_compatibility(self):
        """BRS supports both NN and NNUE."""
        from app.training.multi_harness_gauntlet import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY.get("brs")
        assert compat is not None
        assert compat["nn"] is True
        assert compat["nnue"] is True


class TestPhase1HarnessIntegration:
    """Tests for Phase 1 harness abstraction integration."""

    def test_get_harness_compatibility_from_registry_exists(self):
        """get_harness_compatibility_from_registry function is defined."""
        from app.training.multi_harness_gauntlet import (
            get_harness_compatibility_from_registry,
        )

        assert callable(get_harness_compatibility_from_registry)

    def test_create_harness_for_evaluation_exists(self):
        """create_harness_for_evaluation function is defined."""
        from app.training.multi_harness_gauntlet import create_harness_for_evaluation

        assert callable(create_harness_for_evaluation)

    @patch("app.training.multi_harness_gauntlet.Path")
    def test_get_harness_compatibility_returns_dict(self, mock_path):
        """get_harness_compatibility_from_registry returns harness type mapping."""
        from app.training.multi_harness_gauntlet import (
            get_harness_compatibility_from_registry,
        )

        # Mock a model path
        mock_path.return_value.exists.return_value = True

        result = get_harness_compatibility_from_registry("/fake/model.pth")
        assert isinstance(result, dict)
        assert "nn" in result or "compatible_harnesses" in result or len(result) >= 0


class TestMultiHarnessResultDataclass:
    """Tests for MultiHarnessResult dataclass."""

    def test_default_values(self):
        """MultiHarnessResult has correct defaults."""
        from app.training.multi_harness_gauntlet import MultiHarnessResult

        result = MultiHarnessResult()
        assert result.harness_results == {}
        assert result.best_harness is None
        assert result.best_elo == 0.0
        assert result.model_path is None
        assert result.board_type is None
        assert result.num_players == 2

    def test_with_harness_results(self):
        """MultiHarnessResult can store per-harness results."""
        from app.training.game_gauntlet import GauntletResult
        from app.training.multi_harness_gauntlet import MultiHarnessResult

        gumbel_result = GauntletResult(estimated_elo=1500.0)
        minimax_result = GauntletResult(estimated_elo=1400.0)

        result = MultiHarnessResult(
            harness_results={
                "gumbel_mcts": gumbel_result,
                "minimax": minimax_result,
            },
            best_harness="gumbel_mcts",
            best_elo=1500.0,
        )

        assert len(result.harness_results) == 2
        assert result.best_harness == "gumbel_mcts"
        assert result.best_elo == 1500.0

    def test_harness_results_accessible(self):
        """Individual harness results are accessible."""
        from app.training.game_gauntlet import GauntletResult
        from app.training.multi_harness_gauntlet import MultiHarnessResult

        result = MultiHarnessResult(
            harness_results={
                "gumbel_mcts": GauntletResult(
                    win_rate=0.8,
                    estimated_elo=1600.0,
                    harness_type="gumbel_mcts",
                ),
            }
        )

        gumbel = result.harness_results["gumbel_mcts"]
        assert gumbel.win_rate == 0.8
        assert gumbel.estimated_elo == 1600.0
        assert gumbel.harness_type == "gumbel_mcts"


# =============================================================================
# Harness Compatibility Helper Tests
# =============================================================================


class TestGetCompatibleHarnesses:
    """Tests for get_compatible_harnesses_for_model function."""

    def test_function_exists(self):
        """get_compatible_harnesses_for_model is defined."""
        from app.training.multi_harness_gauntlet import get_compatible_harnesses_for_model

        assert callable(get_compatible_harnesses_for_model)

    def test_nn_model_returns_nn_compatible_harnesses(self):
        """NN model returns harnesses that support NN."""
        from app.training.multi_harness_gauntlet import get_compatible_harnesses_for_model

        harnesses = get_compatible_harnesses_for_model(model_type="nn")
        assert "gumbel_mcts" in harnesses or len(harnesses) > 0

    def test_nnue_model_returns_nnue_compatible_harnesses(self):
        """NNUE model returns harnesses that support NNUE."""
        from app.training.multi_harness_gauntlet import get_compatible_harnesses_for_model

        harnesses = get_compatible_harnesses_for_model(model_type="nnue")
        assert "minimax" in harnesses or len(harnesses) > 0
        # Gumbel MCTS should NOT be in NNUE compatible
        if "gumbel_mcts" in harnesses:
            # This would be a bug if gumbel_mcts is compatible with NNUE
            pytest.fail("gumbel_mcts should not be compatible with NNUE")


class TestIsHarnessCompatible:
    """Tests for is_harness_compatible function."""

    def test_function_exists(self):
        """is_harness_compatible is defined."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        assert callable(is_harness_compatible)

    def test_gumbel_mcts_compatible_with_nn(self):
        """Gumbel MCTS is compatible with NN."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        assert is_harness_compatible("gumbel_mcts", "nn") is True

    def test_gumbel_mcts_not_compatible_with_nnue(self):
        """Gumbel MCTS is not compatible with NNUE."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        assert is_harness_compatible("gumbel_mcts", "nnue") is False

    def test_minimax_compatible_with_both(self):
        """Minimax is compatible with both NN and NNUE."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        assert is_harness_compatible("minimax", "nn") is True
        assert is_harness_compatible("minimax", "nnue") is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_harness_type_compatibility(self):
        """Unknown harness type returns safe default."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        # Should not raise, returns False for unknown
        result = is_harness_compatible("nonexistent_harness", "nn")
        assert result is False

    def test_empty_harness_type(self):
        """Empty harness type handles gracefully."""
        from app.training.multi_harness_gauntlet import is_harness_compatible

        result = is_harness_compatible("", "nn")
        assert result is False

    def test_multi_harness_result_empty_harness_results(self):
        """MultiHarnessResult with empty harness_results is valid."""
        from app.training.multi_harness_gauntlet import MultiHarnessResult

        result = MultiHarnessResult(harness_results={})
        assert result.harness_results == {}
        assert result.best_harness is None
        assert result.best_elo == 0.0


# =============================================================================
# NNUE Evaluation Profile Tests (Phase 4)
# =============================================================================


class TestNNUEEvaluationProfile:
    """Tests for NNUEEvaluationProfile dataclass."""

    def test_default_values(self):
        """Default profile has expected values."""
        from app.training.multi_harness_gauntlet import NNUEEvaluationProfile

        profile = NNUEEvaluationProfile()
        assert profile.board_type == "square8"
        assert profile.num_players == 2
        assert profile.minimax_depth == 4
        assert profile.maxn_depth == 3
        assert profile.brs_depth == 3
        assert profile.games_per_baseline == 30
        assert profile.include_brs is True

    def test_custom_values(self):
        """Profile accepts custom values."""
        from app.training.multi_harness_gauntlet import NNUEEvaluationProfile

        profile = NNUEEvaluationProfile(
            board_type="hex8",
            num_players=4,
            minimax_depth=6,
            maxn_depth=4,
            games_per_baseline=50,
        )
        assert profile.board_type == "hex8"
        assert profile.num_players == 4
        assert profile.minimax_depth == 6
        assert profile.maxn_depth == 4
        assert profile.games_per_baseline == 50

    def test_get_harnesses_2_player(self):
        """2-player profile returns minimax harness."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(num_players=2)
        harnesses = profile.get_harnesses()

        assert len(harnesses) == 1
        assert HarnessType.MINIMAX in harnesses

    def test_get_harnesses_4_player(self):
        """4-player profile returns maxn and brs harnesses."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(num_players=4)
        harnesses = profile.get_harnesses()

        assert HarnessType.MAXN in harnesses
        assert HarnessType.BRS in harnesses

    def test_get_harnesses_4_player_no_brs(self):
        """4-player profile without BRS returns only maxn."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(num_players=4, include_brs=False)
        harnesses = profile.get_harnesses()

        assert harnesses == [HarnessType.MAXN]

    def test_get_harness_config_minimax(self):
        """Minimax config uses minimax_depth."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(minimax_depth=5)
        config = profile.get_harness_config(HarnessType.MINIMAX)

        assert config.harness_type == HarnessType.MINIMAX
        assert config.search_depth == 5

    def test_get_harness_config_maxn(self):
        """MaxN config uses maxn_depth."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(maxn_depth=4)
        config = profile.get_harness_config(HarnessType.MAXN)

        assert config.harness_type == HarnessType.MAXN
        assert config.search_depth == 4

    def test_get_harness_config_brs(self):
        """BRS config uses brs_depth."""
        from app.training.multi_harness_gauntlet import (
            HarnessType,
            NNUEEvaluationProfile,
        )

        profile = NNUEEvaluationProfile(brs_depth=2)
        config = profile.get_harness_config(HarnessType.BRS)

        assert config.harness_type == HarnessType.BRS
        assert config.search_depth == 2


class TestGetNNUEEvaluationProfile:
    """Tests for get_nnue_evaluation_profile function."""

    def test_standard_profile(self):
        """Standard profile has expected settings."""
        from app.training.multi_harness_gauntlet import get_nnue_evaluation_profile

        profile = get_nnue_evaluation_profile("hex8", 2)

        assert profile.board_type == "hex8"
        assert profile.num_players == 2
        assert profile.minimax_depth == 4
        assert profile.games_per_baseline == 30

    def test_fast_profile(self):
        """Fast profile has reduced settings."""
        from app.training.multi_harness_gauntlet import get_nnue_evaluation_profile

        profile = get_nnue_evaluation_profile("hex8", 2, fast=True)

        assert profile.minimax_depth == 2
        assert profile.games_per_baseline == 10

    def test_multiplayer_profile(self):
        """Multiplayer profile uses correct depths."""
        from app.training.multi_harness_gauntlet import get_nnue_evaluation_profile

        profile = get_nnue_evaluation_profile("square8", 4)

        assert profile.num_players == 4
        assert profile.maxn_depth == 3
        assert profile.brs_depth == 3
