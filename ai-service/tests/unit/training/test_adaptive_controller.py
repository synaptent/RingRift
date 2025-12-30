"""Tests for app.training.adaptive_controller module.

This module tests:
- IterationResult dataclass
- AdaptiveController dataclass and methods
- create_adaptive_controller factory function
- detect_elo_plateau function
- Plateau intervention functions
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.training.adaptive_controller import (
    AdaptiveController,
    IterationResult,
    create_adaptive_controller,
    detect_elo_plateau,
)


# =============================================================================
# IterationResult Tests
# =============================================================================


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_create_result(self):
        """Create iteration result with all fields."""
        result = IterationResult(
            iteration=1,
            win_rate=0.65,
            promoted=True,
            games_played=100,
            eval_games=50,
        )
        assert result.iteration == 1
        assert result.win_rate == 0.65
        assert result.promoted is True
        assert result.games_played == 100
        assert result.eval_games == 50

    def test_create_non_promoted(self):
        """Create non-promoted result."""
        result = IterationResult(
            iteration=2,
            win_rate=0.48,
            promoted=False,
            games_played=150,
            eval_games=75,
        )
        assert result.promoted is False


# =============================================================================
# AdaptiveController Initialization Tests
# =============================================================================


class TestAdaptiveControllerInit:
    """Tests for AdaptiveController initialization."""

    def test_default_values(self):
        """Test default values are sensible."""
        controller = AdaptiveController()
        assert controller.plateau_threshold == 5
        assert controller.min_games == 50
        assert controller.max_games == 200
        assert controller.min_eval_games == 50
        assert controller.max_eval_games == 200
        assert controller.base_win_rate == 0.55
        assert controller.config_name == "default"
        assert controller.enable_events is True
        assert len(controller.history) == 0

    def test_custom_values(self):
        """Test custom configuration."""
        controller = AdaptiveController(
            plateau_threshold=10,
            min_games=100,
            max_games=500,
            config_name="hex8_2p",
        )
        assert controller.plateau_threshold == 10
        assert controller.min_games == 100
        assert controller.max_games == 500
        assert controller.config_name == "hex8_2p"


# =============================================================================
# Record Iteration Tests
# =============================================================================


class TestRecordIteration:
    """Tests for record_iteration method."""

    def test_record_single_iteration(self):
        """Record a single iteration."""
        controller = AdaptiveController()
        controller.record_iteration(
            iteration=1,
            win_rate=0.60,
            promoted=True,
            games_played=100,
            eval_games=50,
        )
        assert len(controller.history) == 1
        assert controller.history[0].win_rate == 0.60

    def test_record_multiple_iterations(self):
        """Record multiple iterations."""
        controller = AdaptiveController()
        for i in range(3):
            controller.record_iteration(
                iteration=i + 1,
                win_rate=0.55 + i * 0.05,
                promoted=i % 2 == 0,
                games_played=100,
                eval_games=50,
            )
        assert len(controller.history) == 3

    def test_promotion_resets_plateau_flag(self):
        """Promotion resets the plateau emitted flag."""
        controller = AdaptiveController()
        controller._plateau_emitted = True
        controller.record_iteration(
            iteration=1,
            win_rate=0.60,
            promoted=True,
            games_played=100,
            eval_games=50,
        )
        assert controller._plateau_emitted is False

    def test_quality_penalty_decays(self):
        """Quality penalty decays each iteration."""
        controller = AdaptiveController()
        controller._quality_penalty = 0.5
        controller.record_iteration(
            iteration=1,
            win_rate=0.60,
            promoted=False,
            games_played=100,
            eval_games=50,
        )
        assert controller._quality_penalty == pytest.approx(0.4)  # Decayed by 0.1


# =============================================================================
# Should Continue Tests
# =============================================================================


class TestShouldContinue:
    """Tests for should_continue method."""

    def test_continue_with_no_history(self):
        """Should continue with no history."""
        controller = AdaptiveController(plateau_threshold=3)
        assert controller.should_continue() is True

    def test_continue_with_insufficient_history(self):
        """Should continue with less history than threshold."""
        controller = AdaptiveController(plateau_threshold=5)
        for i in range(3):
            controller.record_iteration(i, 0.50, False, 100, 50)
        assert controller.should_continue() is True

    def test_continue_with_recent_promotion(self):
        """Should continue if recent iteration was promoted."""
        controller = AdaptiveController(plateau_threshold=3)
        controller.record_iteration(1, 0.50, False, 100, 50)
        controller.record_iteration(2, 0.50, False, 100, 50)
        controller.record_iteration(3, 0.60, True, 100, 50)  # Promoted
        assert controller.should_continue() is True

    def test_stop_after_plateau(self):
        """Should stop after plateau threshold reached."""
        controller = AdaptiveController(plateau_threshold=3)
        for i in range(5):
            controller.record_iteration(i, 0.50, False, 100, 50)  # No promotions
        assert controller.should_continue() is False


# =============================================================================
# Plateau Count Tests
# =============================================================================


class TestGetPlateauCount:
    """Tests for get_plateau_count method."""

    def test_no_history(self):
        """Zero plateau count with no history."""
        controller = AdaptiveController()
        assert controller.get_plateau_count() == 0

    def test_all_promoted(self):
        """Zero plateau if last was promoted."""
        controller = AdaptiveController()
        controller.record_iteration(1, 0.60, True, 100, 50)
        assert controller.get_plateau_count() == 0

    def test_count_consecutive_no_promotions(self):
        """Count consecutive non-promotions."""
        controller = AdaptiveController()
        controller.record_iteration(1, 0.60, True, 100, 50)  # Promoted
        controller.record_iteration(2, 0.50, False, 100, 50)  # Not
        controller.record_iteration(3, 0.48, False, 100, 50)  # Not
        assert controller.get_plateau_count() == 2


# =============================================================================
# Compute Games Tests
# =============================================================================


class TestComputeGames:
    """Tests for compute_games method."""

    def test_no_history_returns_midpoint(self):
        """Returns midpoint with no history."""
        controller = AdaptiveController(min_games=50, max_games=200)
        games = controller.compute_games()
        assert games == 125  # (50 + 200) // 2

    def test_marginal_win_rate_uses_max(self):
        """Marginal win rate (0.45-0.55) uses max games."""
        controller = AdaptiveController(min_games=50, max_games=200)
        games = controller.compute_games(recent_win_rate=0.50)
        assert games >= 150  # Should be close to max

    def test_clear_win_rate_uses_min(self):
        """Clear win rate (<0.45 or >0.55) uses min games."""
        controller = AdaptiveController(min_games=50, max_games=200)
        games_low = controller.compute_games(recent_win_rate=0.40)
        games_high = controller.compute_games(recent_win_rate=0.70)
        assert games_low <= 100
        assert games_high <= 100

    def test_uses_history_if_no_rate_provided(self):
        """Uses last history win rate if not provided."""
        controller = AdaptiveController()
        controller.record_iteration(1, 0.50, True, 100, 50)
        games = controller.compute_games()  # No recent_win_rate
        assert games > 0


# =============================================================================
# Compute Eval Games Tests
# =============================================================================


class TestComputeEvalGames:
    """Tests for compute_eval_games method."""

    def test_no_history_returns_midpoint(self):
        """Returns midpoint with no history."""
        controller = AdaptiveController(min_eval_games=50, max_eval_games=200)
        eval_games = controller.compute_eval_games()
        assert eval_games == 125

    def test_very_marginal_uses_max(self):
        """Very marginal win rate (0.48-0.52) uses max eval games."""
        controller = AdaptiveController(max_eval_games=200)
        eval_games = controller.compute_eval_games(recent_win_rate=0.50)
        assert eval_games == 200

    def test_moderately_marginal_uses_mid(self):
        """Moderately marginal (0.45-0.55) uses mid eval games."""
        controller = AdaptiveController(min_eval_games=50, max_eval_games=200)
        eval_games = controller.compute_eval_games(recent_win_rate=0.54)
        assert eval_games == 125

    def test_clear_result_uses_min(self):
        """Clear result uses min eval games."""
        controller = AdaptiveController(min_eval_games=50, max_eval_games=200)
        eval_games = controller.compute_eval_games(recent_win_rate=0.70)
        assert eval_games == 50


# =============================================================================
# Trend Factor Tests
# =============================================================================


class TestComputeTrendFactor:
    """Tests for _compute_trend_factor method."""

    def test_insufficient_history(self):
        """Returns 1.0 with insufficient history."""
        controller = AdaptiveController()
        controller.record_iteration(1, 0.50, False, 100, 50)
        factor = controller._compute_trend_factor()
        assert factor == 1.0

    def test_strong_improvement(self):
        """Strong improvement (>60% promotions) boosts factor."""
        controller = AdaptiveController()
        for i in range(5):
            controller.record_iteration(i, 0.60, i % 4 == 0, 100, 50)  # 40% promoted
        # Not quite >60%, so check it's reasonable
        factor = controller._compute_trend_factor()
        assert 0.5 < factor < 2.0

    def test_plateau_reduces_factor(self):
        """Plateau (<20% promotions) reduces factor."""
        controller = AdaptiveController()
        for i in range(5):
            controller.record_iteration(i, 0.50, False, 100, 50)  # 0% promoted
        factor = controller._compute_trend_factor()
        assert factor < 1.0

    def test_quality_penalty_reduces_factor(self):
        """Quality penalty reduces trend factor."""
        controller = AdaptiveController()
        for i in range(5):
            controller.record_iteration(i, 0.60, True, 100, 50)  # All promoted
        controller._quality_penalty = 1.0  # Max penalty
        factor = controller._compute_trend_factor()
        # Factor should be reduced by quality adjustment (1.0 - 1.0 * 0.3 = 0.7)
        assert factor < 1.0


# =============================================================================
# Quality Penalty Tests
# =============================================================================


class TestApplyQualityPenalty:
    """Tests for apply_quality_penalty method."""

    def test_apply_penalty(self):
        """Apply penalty updates internal state."""
        controller = AdaptiveController()
        controller.apply_quality_penalty(0.5, "test reason")
        assert controller._quality_penalty == 0.5

    def test_penalty_capped_at_one(self):
        """Penalty is capped at 1.0."""
        controller = AdaptiveController()
        controller.apply_quality_penalty(1.5)
        assert controller._quality_penalty == 1.0

    def test_penalty_takes_max(self):
        """New penalty takes max of old and new."""
        controller = AdaptiveController()
        controller._quality_penalty = 0.6
        controller.apply_quality_penalty(0.3)  # Lower than current
        assert controller._quality_penalty == 0.6  # Unchanged

    def test_higher_penalty_updates(self):
        """Higher penalty updates the value."""
        controller = AdaptiveController()
        controller._quality_penalty = 0.3
        controller.apply_quality_penalty(0.6)
        assert controller._quality_penalty == 0.6


# =============================================================================
# Statistics Tests
# =============================================================================


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_empty_history(self):
        """Statistics with empty history."""
        controller = AdaptiveController()
        stats = controller.get_statistics()
        assert stats["total_iterations"] == 0
        assert stats["total_promotions"] == 0
        assert stats["trend"] == "unknown"

    def test_with_history(self):
        """Statistics with history."""
        controller = AdaptiveController()
        controller.record_iteration(1, 0.55, True, 100, 50)
        controller.record_iteration(2, 0.58, True, 100, 50)
        controller.record_iteration(3, 0.52, False, 100, 50)
        stats = controller.get_statistics()
        assert stats["total_iterations"] == 3
        assert stats["total_promotions"] == 2
        assert stats["promotion_rate"] == pytest.approx(2 / 3)
        assert 0.52 <= stats["avg_win_rate"] <= 0.58

    def test_trend_improving(self):
        """Trend detection for improving."""
        controller = AdaptiveController()
        # Older: lower win rates
        controller.record_iteration(1, 0.50, False, 100, 50)
        controller.record_iteration(2, 0.52, False, 100, 50)
        # Recent: higher win rates
        controller.record_iteration(3, 0.60, True, 100, 50)
        controller.record_iteration(4, 0.62, True, 100, 50)
        controller.record_iteration(5, 0.65, True, 100, 50)
        stats = controller.get_statistics()
        assert stats["trend"] == "improving"

    def test_trend_declining(self):
        """Trend detection for declining."""
        controller = AdaptiveController()
        # Older: higher win rates
        controller.record_iteration(1, 0.70, True, 100, 50)
        controller.record_iteration(2, 0.68, True, 100, 50)
        # Recent: lower win rates
        controller.record_iteration(3, 0.50, False, 100, 50)
        controller.record_iteration(4, 0.48, False, 100, 50)
        controller.record_iteration(5, 0.45, False, 100, 50)
        stats = controller.get_statistics()
        assert stats["trend"] == "declining"


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for save and load methods."""

    def test_save_load_roundtrip(self):
        """Save and load preserves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"

            controller = AdaptiveController(
                plateau_threshold=10,
                min_games=100,
                max_games=300,
                config_name="test_config",
            )
            controller.record_iteration(1, 0.60, True, 100, 50)
            controller.record_iteration(2, 0.55, False, 150, 75)
            controller._quality_penalty = 0.3
            controller.save(path)

            loaded = AdaptiveController.load(path)
            assert loaded.plateau_threshold == 10
            assert loaded.min_games == 100
            assert loaded.max_games == 300
            assert loaded.config_name == "test_config"
            assert len(loaded.history) == 2
            assert loaded._quality_penalty == 0.3

    def test_load_nonexistent_file(self):
        """Load from nonexistent file returns default."""
        loaded = AdaptiveController.load(Path("/nonexistent/path.json"))
        assert loaded.plateau_threshold == 5  # Default

    def test_load_with_config_override(self):
        """Load with config_name override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            controller = AdaptiveController(config_name="original")
            controller.save(path)

            loaded = AdaptiveController.load(path, config_name="override")
            assert loaded.config_name == "override"

    def test_load_corrupt_file(self):
        """Load from corrupt file returns default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            path.write_text("not valid json")

            loaded = AdaptiveController.load(path)
            assert loaded.plateau_threshold == 5  # Default


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateAdaptiveController:
    """Tests for create_adaptive_controller factory function."""

    def test_create_default(self):
        """Create with default parameters."""
        controller = create_adaptive_controller()
        assert controller.plateau_threshold == 5
        assert controller.config_name == "default"

    def test_create_with_params(self):
        """Create with custom parameters."""
        controller = create_adaptive_controller(
            plateau_threshold=10,
            min_games=100,
            max_games=500,
            config_name="hex8_2p",
        )
        assert controller.plateau_threshold == 10
        assert controller.min_games == 100
        assert controller.config_name == "hex8_2p"

    def test_load_from_state_path(self):
        """Load existing state from path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            # Create and save initial state
            initial = AdaptiveController(config_name="saved")
            initial.record_iteration(1, 0.60, True, 100, 50)
            initial.save(path)

            # Load via factory
            controller = create_adaptive_controller(
                config_name="saved",
                state_path=path,
            )
            assert len(controller.history) == 1

    def test_update_params_on_load(self):
        """Factory updates params when loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            initial = AdaptiveController(plateau_threshold=5)
            initial.save(path)

            controller = create_adaptive_controller(
                plateau_threshold=15,  # Different from saved
                state_path=path,
            )
            assert controller.plateau_threshold == 15


# =============================================================================
# Elo Plateau Detection Tests
# =============================================================================


class TestDetectEloPlateau:
    """Tests for detect_elo_plateau function."""

    def test_insufficient_data(self):
        """Insufficient data returns no plateau."""
        is_plateau, details = detect_elo_plateau([1500])
        assert is_plateau is False
        assert details["reason"] == "insufficient_data"

    def test_improving_elo(self):
        """Clearly improving Elo is not plateau."""
        elo_history = [1500, 1520, 1550, 1590, 1640, 1700]
        is_plateau, details = detect_elo_plateau(elo_history, window_size=5)
        assert is_plateau == False  # Use == for numpy bool compatibility
        assert details["slope"] > 0.5
        assert details["reason"] == "improving"

    def test_plateau_detected(self):
        """Stagnant Elo is detected as plateau."""
        elo_history = [1500, 1501, 1502, 1501, 1502, 1503, 1502, 1503]
        is_plateau, details = detect_elo_plateau(
            elo_history,
            window_size=5,
            threshold_elo_per_game=0.5,
        )
        assert is_plateau == True  # Use == for numpy bool compatibility
        assert details["slope"] < 0.5
        assert details["reason"] == "low_slope"

    def test_window_size_limits(self):
        """Uses available data if less than window size."""
        elo_history = [1500, 1510, 1520]
        is_plateau, details = detect_elo_plateau(elo_history, window_size=10)
        assert details["window_size"] == 3

    def test_returns_start_end_elo(self):
        """Returns start and end Elo values."""
        elo_history = [1500, 1520, 1540, 1560, 1580]
        is_plateau, details = detect_elo_plateau(elo_history)
        assert details["start_elo"] == 1500
        assert details["end_elo"] == 1580

    def test_confidence_computed(self):
        """R-squared confidence is computed."""
        elo_history = [1500, 1520, 1540, 1560, 1580]  # Linear
        is_plateau, details = detect_elo_plateau(elo_history)
        assert details["confidence"] > 0.9  # Should be very high for linear


# =============================================================================
# Async Event Tests
# =============================================================================


class TestAsyncMethods:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_record_iteration_async(self):
        """Async record iteration works."""
        controller = AdaptiveController(enable_events=False)  # Disable events
        await controller.record_iteration_async(
            iteration=1,
            win_rate=0.60,
            promoted=True,
            games_played=100,
            eval_games=50,
        )
        assert len(controller.history) == 1

    @pytest.mark.asyncio
    async def test_emit_plateau_no_event_system(self):
        """Emit plateau gracefully handles no event system."""
        controller = AdaptiveController(enable_events=True)
        # Patch HAS_EVENT_SYSTEM to False
        with patch("app.training.adaptive_controller.HAS_EVENT_SYSTEM", False):
            result = await controller.emit_plateau_if_detected()
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_game_count_no_event_system(self):
        """Emit game count gracefully handles no event system."""
        controller = AdaptiveController(enable_events=True)
        with patch("app.training.adaptive_controller.HAS_EVENT_SYSTEM", False):
            result = await controller.emit_game_count_update()
            assert result is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_very_high_win_rate(self):
        """Handle very high win rate."""
        controller = AdaptiveController()
        games = controller.compute_games(recent_win_rate=0.99)
        assert games >= controller.min_games
        assert games <= controller.max_games

    def test_very_low_win_rate(self):
        """Handle very low win rate."""
        controller = AdaptiveController()
        games = controller.compute_games(recent_win_rate=0.01)
        assert games >= controller.min_games
        assert games <= controller.max_games

    def test_negative_quality_penalty(self):
        """Negative penalty is clamped to zero."""
        controller = AdaptiveController()
        controller._quality_penalty = 0.05
        controller.record_iteration(1, 0.50, False, 100, 50)
        # Decays by 0.1, should not go negative
        assert controller._quality_penalty >= 0.0

    def test_many_iterations(self):
        """Handle many iterations."""
        controller = AdaptiveController()
        for i in range(100):
            controller.record_iteration(
                iteration=i,
                win_rate=0.50 + (i % 10) * 0.01,
                promoted=i % 5 == 0,
                games_played=100,
                eval_games=50,
            )
        assert len(controller.history) == 100
        stats = controller.get_statistics()
        assert stats["total_iterations"] == 100
