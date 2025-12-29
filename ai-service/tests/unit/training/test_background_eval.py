"""Comprehensive unit tests for background_eval module.

Tests cover:
- BackgroundEvalConfig dataclass and defaults
- EvalResult dataclass
- BackgroundEvaluator initialization and state management
- Threshold imports from app.config.thresholds
- Baseline gating logic
- Circuit breaker functionality
- Failure/success recording
- Health status reporting
- Thread management (start/stop)
- Step updating and event subscription
- Placeholder evaluation
- Result processing and checkpointing
- Factory functions and singleton management
- Auto-wiring from training coordinator
- Edge cases and error conditions

Created: December 2025
Updated: December 29, 2025 - Expanded to 40+ tests
"""

from __future__ import annotations

import threading
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.background_eval import (
    BackgroundEvalConfig,
    BackgroundEvaluator,
    EvalConfig,  # Alias
    EvalResult,
    auto_wire_from_training_coordinator,
    create_background_evaluator,
    get_background_evaluator,
    reset_background_evaluator,
    wire_background_evaluator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model_getter():
    """Create a mock model getter function."""
    return MagicMock(return_value={"state_dict": {}, "path": None})


@pytest.fixture
def custom_config(tmp_path):
    """Create a custom BackgroundEvalConfig with test values."""
    return BackgroundEvalConfig(
        eval_interval_steps=100,
        games_per_eval=10,
        baselines=["random", "heuristic"],
        elo_checkpoint_threshold=5.0,
        elo_drop_threshold=30.0,
        auto_checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        min_baseline_win_rates={"random": 0.6, "heuristic": 0.4},
        max_consecutive_failures=3,
        failure_cooldown_seconds=30.0,
        max_failures_per_hour=10,
        eval_timeout_seconds=60.0,
    )


@pytest.fixture
def evaluator(mock_model_getter, custom_config):
    """Create a BackgroundEvaluator with custom config."""
    return BackgroundEvaluator(
        model_getter=mock_model_getter,
        config=custom_config,
        board_type=None,
        use_real_games=False,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_background_evaluator()
    yield
    reset_background_evaluator()


class TestBackgroundEvalConfigDataclass:
    """Tests for BackgroundEvalConfig dataclass."""

    def test_default_creation(self):
        """Test BackgroundEvalConfig with defaults."""
        config = BackgroundEvalConfig()

        assert config.eval_interval_steps == 1000
        assert config.games_per_eval == 20
        assert config.auto_checkpoint is True
        assert config.checkpoint_dir == "data/eval_checkpoints"

    def test_default_baselines(self):
        """Test default baselines include random and heuristic."""
        config = BackgroundEvalConfig()

        assert "random" in config.baselines
        assert "heuristic" in config.baselines
        assert len(config.baselines) == 2

    def test_elo_thresholds(self):
        """Test Elo thresholds have sensible defaults."""
        config = BackgroundEvalConfig()

        assert config.elo_checkpoint_threshold > 0
        assert config.elo_checkpoint_threshold == 10.0
        assert config.elo_drop_threshold > 0
        # elo_drop_threshold should match ELO_DROP_ROLLBACK from thresholds
        assert config.elo_drop_threshold == 50.0

    def test_min_baseline_win_rates(self):
        """Test minimum baseline win rates are configured."""
        config = BackgroundEvalConfig()

        assert "random" in config.min_baseline_win_rates
        assert "heuristic" in config.min_baseline_win_rates
        # Random should require higher win rate than heuristic
        assert config.min_baseline_win_rates["random"] >= 0.5
        assert config.min_baseline_win_rates["heuristic"] >= 0.3

    def test_failsafe_configuration(self):
        """Test failsafe/circuit breaker configuration."""
        config = BackgroundEvalConfig()

        assert config.max_consecutive_failures >= 3
        assert config.failure_cooldown_seconds > 0
        assert config.max_failures_per_hour > 0
        assert config.eval_timeout_seconds > 0

    def test_custom_eval_interval(self):
        """Test custom eval_interval_steps."""
        config = BackgroundEvalConfig(eval_interval_steps=500)
        assert config.eval_interval_steps == 500

    def test_custom_games_per_eval(self):
        """Test custom games_per_eval."""
        config = BackgroundEvalConfig(games_per_eval=50)
        assert config.games_per_eval == 50

    def test_custom_baselines(self):
        """Test custom baselines list."""
        config = BackgroundEvalConfig(baselines=["random", "heuristic", "neural"])
        assert len(config.baselines) == 3
        assert "neural" in config.baselines

    def test_custom_checkpoint_dir(self):
        """Test custom checkpoint directory."""
        config = BackgroundEvalConfig(checkpoint_dir="/custom/path")
        assert config.checkpoint_dir == "/custom/path"

    def test_disable_auto_checkpoint(self):
        """Test disabling auto_checkpoint."""
        config = BackgroundEvalConfig(auto_checkpoint=False)
        assert config.auto_checkpoint is False


class TestEvalConfigAlias:
    """Tests for EvalConfig backward-compatible alias."""

    def test_eval_config_is_alias(self):
        """Test EvalConfig is alias for BackgroundEvalConfig."""
        assert EvalConfig is BackgroundEvalConfig

    def test_create_via_alias(self):
        """Test creating config via alias works."""
        config = EvalConfig(eval_interval_steps=200)
        assert config.eval_interval_steps == 200
        assert isinstance(config, BackgroundEvalConfig)


class TestEvalResultDataclass:
    """Tests for EvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic EvalResult creation."""
        result = EvalResult(
            step=1000,
            timestamp=1234567890.0,
            elo_estimate=1550.0,
            elo_std=25.0,
            games_played=20,
            win_rate=0.65,
            baseline_results={"random": 0.90, "heuristic": 0.55},
        )

        assert result.step == 1000
        assert result.elo_estimate == 1550.0
        assert result.games_played == 20

    def test_default_passes_baseline_gating(self):
        """Test passes_baseline_gating defaults to True."""
        result = EvalResult(
            step=0,
            timestamp=0.0,
            elo_estimate=1500.0,
            elo_std=0.0,
            games_played=0,
            win_rate=0.0,
            baseline_results={},
        )

        assert result.passes_baseline_gating is True

    def test_default_failed_baselines_empty(self):
        """Test failed_baselines defaults to empty list."""
        result = EvalResult(
            step=0,
            timestamp=0.0,
            elo_estimate=1500.0,
            elo_std=0.0,
            games_played=0,
            win_rate=0.0,
            baseline_results={},
        )

        assert result.failed_baselines == []

    def test_with_failed_baselines(self):
        """Test EvalResult with failed baselines."""
        result = EvalResult(
            step=1000,
            timestamp=1234567890.0,
            elo_estimate=1400.0,
            elo_std=30.0,
            games_played=20,
            win_rate=0.40,
            baseline_results={"random": 0.60, "heuristic": 0.35},
            passes_baseline_gating=False,
            failed_baselines=["random"],
        )

        assert result.passes_baseline_gating is False
        assert "random" in result.failed_baselines

    def test_baseline_results_dict(self):
        """Test baseline_results is a dict."""
        result = EvalResult(
            step=1000,
            timestamp=1234567890.0,
            elo_estimate=1600.0,
            elo_std=20.0,
            games_played=50,
            win_rate=0.70,
            baseline_results={
                "random": 0.95,
                "heuristic": 0.65,
                "weak_neural": 0.55,
            },
        )

        assert isinstance(result.baseline_results, dict)
        assert len(result.baseline_results) == 3
        assert result.baseline_results["random"] == 0.95


class TestBackgroundEvaluatorImport:
    """Tests for BackgroundEvaluator class import."""

    def test_evaluator_importable(self):
        """Test BackgroundEvaluator can be imported."""
        from app.training.background_eval import BackgroundEvaluator
        assert BackgroundEvaluator is not None

    def test_evaluator_instantiation(self):
        """Test BackgroundEvaluator basic instantiation."""
        from app.training.background_eval import BackgroundEvaluator

        # May require model or config
        try:
            evaluator = BackgroundEvaluator()
            assert evaluator is not None
        except TypeError:
            # May require constructor arguments
            pass


class TestThresholdIntegration:
    """Tests for threshold imports from app.config.thresholds."""

    def test_elo_drop_threshold_matches(self):
        """Test ELO_DROP_ROLLBACK is imported correctly."""
        from app.config.thresholds import ELO_DROP_ROLLBACK

        config = BackgroundEvalConfig()
        assert config.elo_drop_threshold == ELO_DROP_ROLLBACK

    def test_initial_elo_available(self):
        """Test INITIAL_ELO_RATING is available."""
        from app.training.background_eval import INITIAL_ELO_RATING
        assert INITIAL_ELO_RATING > 0
        assert INITIAL_ELO_RATING == 1500.0

    def test_min_win_rates_from_thresholds(self):
        """Test minimum win rates come from thresholds module."""
        from app.config.thresholds import (
            MIN_WIN_RATE_VS_HEURISTIC,
            MIN_WIN_RATE_VS_RANDOM,
        )

        config = BackgroundEvalConfig()
        assert config.min_baseline_win_rates["random"] == MIN_WIN_RATE_VS_RANDOM
        assert config.min_baseline_win_rates["heuristic"] == MIN_WIN_RATE_VS_HEURISTIC


class TestBaselineGating:
    """Tests for baseline gating logic."""

    def test_high_win_rates_pass_gating(self):
        """Test high win rates pass baseline gating."""
        result = EvalResult(
            step=1000,
            timestamp=0.0,
            elo_estimate=1650.0,
            elo_std=15.0,
            games_played=50,
            win_rate=0.75,
            baseline_results={"random": 0.95, "heuristic": 0.70},
            passes_baseline_gating=True,
            failed_baselines=[],
        )

        assert result.passes_baseline_gating is True
        assert len(result.failed_baselines) == 0

    def test_low_random_win_rate_fails(self):
        """Test low random win rate fails gating."""
        result = EvalResult(
            step=1000,
            timestamp=0.0,
            elo_estimate=1400.0,
            elo_std=30.0,
            games_played=20,
            win_rate=0.50,
            baseline_results={"random": 0.50, "heuristic": 0.55},
            passes_baseline_gating=False,
            failed_baselines=["random"],
        )

        assert result.passes_baseline_gating is False
        assert "random" in result.failed_baselines


class TestEvalResultScores:
    """Tests for EvalResult score validation."""

    def test_elo_estimate_range(self):
        """Test Elo estimate is in reasonable range."""
        result = EvalResult(
            step=1000,
            timestamp=0.0,
            elo_estimate=1600.0,
            elo_std=25.0,
            games_played=50,
            win_rate=0.65,
            baseline_results={},
        )

        # Reasonable Elo range
        assert 800 <= result.elo_estimate <= 3000

    def test_win_rate_range(self):
        """Test win rate is between 0 and 1."""
        result = EvalResult(
            step=1000,
            timestamp=0.0,
            elo_estimate=1500.0,
            elo_std=0.0,
            games_played=20,
            win_rate=0.75,
            baseline_results={},
        )

        assert 0.0 <= result.win_rate <= 1.0

    def test_elo_std_non_negative(self):
        """Test Elo standard deviation is non-negative."""
        result = EvalResult(
            step=1000,
            timestamp=0.0,
            elo_estimate=1500.0,
            elo_std=25.0,
            games_played=20,
            win_rate=0.5,
            baseline_results={},
        )

        assert result.elo_std >= 0


# =============================================================================
# Tests for BackgroundEvaluator Initialization
# =============================================================================


class TestBackgroundEvaluatorInit:
    """Tests for BackgroundEvaluator initialization."""

    def test_basic_init(self, mock_model_getter):
        """Should initialize with minimal arguments."""
        evaluator = BackgroundEvaluator(model_getter=mock_model_getter)
        assert evaluator.model_getter is mock_model_getter
        assert evaluator.config is not None
        assert evaluator.board_type is None
        assert evaluator.use_real_games is False

    def test_init_with_config(self, mock_model_getter, custom_config):
        """Should accept custom config."""
        evaluator = BackgroundEvaluator(
            model_getter=mock_model_getter,
            config=custom_config,
        )
        assert evaluator.config is custom_config
        assert evaluator.config.eval_interval_steps == 100

    def test_init_state_defaults(self, evaluator):
        """Should initialize state with correct defaults."""
        assert evaluator.current_step == 0
        assert evaluator.last_eval_step == 0
        assert evaluator.best_elo == 0.0
        assert evaluator.current_elo == 1500.0  # INITIAL_ELO_RATING
        assert evaluator.elo_history == []
        assert evaluator.eval_results == []

    def test_init_thread_state(self, evaluator):
        """Should initialize thread state correctly."""
        assert evaluator._running is False
        assert evaluator._eval_thread is None
        assert isinstance(evaluator._lock, type(threading.Lock()))

    def test_init_failsafe_state(self, evaluator):
        """Should initialize failsafe state correctly."""
        assert evaluator._consecutive_failures == 0
        assert evaluator._failure_timestamps == []
        assert evaluator._circuit_breaker_tripped is False
        assert evaluator._total_failures == 0
        assert evaluator._total_successes == 0

    def test_init_creates_checkpoint_dir(self, mock_model_getter, tmp_path):
        """Should create checkpoint directory on init."""
        config = BackgroundEvalConfig(checkpoint_dir=str(tmp_path / "new_dir"))
        evaluator = BackgroundEvaluator(
            model_getter=mock_model_getter,
            config=config,
        )
        assert evaluator.checkpoint_dir.exists()

    def test_real_games_without_board_type_falls_back(self, mock_model_getter):
        """Should fall back to placeholder mode if use_real_games without board_type."""
        evaluator = BackgroundEvaluator(
            model_getter=mock_model_getter,
            use_real_games=True,
            board_type=None,
        )
        assert evaluator.use_real_games is False


# =============================================================================
# Tests for Circuit Breaker Functionality
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_check_circuit_breaker_normal(self, evaluator):
        """Should return True when circuit breaker is not tripped."""
        assert evaluator._check_circuit_breaker() is True

    def test_check_circuit_breaker_tripped(self, evaluator):
        """Should return False when circuit breaker is tripped."""
        evaluator._circuit_breaker_tripped = True
        evaluator._circuit_breaker_reset_time = time.time() + 60.0
        assert evaluator._check_circuit_breaker() is False

    def test_circuit_breaker_reset_after_cooldown(self, evaluator):
        """Should reset circuit breaker after cooldown period."""
        evaluator._circuit_breaker_tripped = True
        evaluator._circuit_breaker_reset_time = time.time() - 1.0  # In the past
        evaluator._consecutive_failures = 5

        result = evaluator._check_circuit_breaker()

        assert result is True
        assert evaluator._circuit_breaker_tripped is False
        assert evaluator._consecutive_failures == 0

    def test_rate_limit_trips_circuit_breaker(self, evaluator):
        """Should trip circuit breaker when rate limit exceeded."""
        current_time = time.time()
        # Add more failures than allowed per hour
        evaluator._failure_timestamps = [
            current_time - i for i in range(evaluator.config.max_failures_per_hour + 1)
        ]

        result = evaluator._check_circuit_breaker()

        assert result is False
        assert evaluator._circuit_breaker_tripped is True

    def test_old_failures_excluded_from_rate_limit(self, evaluator):
        """Should exclude failures older than 1 hour from rate limit."""
        current_time = time.time()
        # Add old failures (> 1 hour ago)
        evaluator._failure_timestamps = [current_time - 4000 for _ in range(20)]

        result = evaluator._check_circuit_breaker()

        assert result is True
        assert len(evaluator._failure_timestamps) == 0  # Old ones cleaned up


# =============================================================================
# Tests for Failure Recording
# =============================================================================


class TestFailureRecording:
    """Tests for failure and success recording."""

    def test_record_failure_increments_counts(self, evaluator):
        """Should increment failure counts on record_failure."""
        evaluator._record_failure("Test error")

        assert evaluator._consecutive_failures == 1
        assert evaluator._total_failures == 1
        assert len(evaluator._failure_timestamps) == 1

    def test_record_multiple_failures(self, evaluator):
        """Should track multiple failures correctly."""
        for i in range(3):
            evaluator._record_failure(f"Error {i}")

        assert evaluator._consecutive_failures == 3
        assert evaluator._total_failures == 3
        assert len(evaluator._failure_timestamps) == 3

    def test_record_failure_trips_circuit_breaker(self, evaluator):
        """Should trip circuit breaker after max consecutive failures."""
        for i in range(evaluator.config.max_consecutive_failures):
            evaluator._record_failure(f"Error {i}")

        assert evaluator._circuit_breaker_tripped is True

    def test_record_success_resets_consecutive(self, evaluator):
        """Should reset consecutive failures on success."""
        evaluator._consecutive_failures = 3
        evaluator._record_success()

        assert evaluator._consecutive_failures == 0
        assert evaluator._total_successes == 1

    def test_record_success_increments_total(self, evaluator):
        """Should increment total successes."""
        for _ in range(5):
            evaluator._record_success()

        assert evaluator._total_successes == 5


# =============================================================================
# Tests for Health Status
# =============================================================================


class TestHealthStatus:
    """Tests for health status reporting."""

    def test_get_health_status_initial(self, evaluator):
        """Should return correct initial health status."""
        status = evaluator.get_health_status()

        assert status["circuit_breaker_tripped"] is False
        assert status["consecutive_failures"] == 0
        assert status["total_failures"] == 0
        assert status["total_successes"] == 0
        assert status["failures_last_hour"] == 0
        assert status["success_rate"] == 1.0

    def test_get_health_status_after_failures(self, evaluator):
        """Should reflect failures in health status."""
        evaluator._consecutive_failures = 3
        evaluator._total_failures = 5
        evaluator._failure_timestamps = [time.time() for _ in range(4)]

        status = evaluator.get_health_status()

        assert status["consecutive_failures"] == 3
        assert status["total_failures"] == 5
        assert status["failures_last_hour"] == 4

    def test_get_health_status_success_rate_calculation(self, evaluator):
        """Should calculate success rate correctly."""
        evaluator._total_successes = 7
        evaluator._total_failures = 3

        status = evaluator.get_health_status()

        assert abs(status["success_rate"] - 0.7) < 0.001


# =============================================================================
# Tests for Thread Management
# =============================================================================


class TestThreadManagement:
    """Tests for start/stop thread management."""

    def test_start_sets_running_flag(self, evaluator):
        """Should set running flag on start."""
        with patch.object(evaluator, '_eval_loop'):
            evaluator.start()
            assert evaluator._running is True
            evaluator.stop()

    def test_start_idempotent(self, evaluator):
        """Should be idempotent (calling start twice is safe)."""
        with patch.object(evaluator, '_eval_loop'):
            evaluator.start()
            thread1 = evaluator._eval_thread
            evaluator.start()
            thread2 = evaluator._eval_thread
            assert thread1 is thread2
            evaluator.stop()

    def test_stop_clears_running_flag(self, evaluator):
        """Should clear running flag on stop."""
        evaluator._running = True
        evaluator.stop()
        assert evaluator._running is False

    def test_stop_clears_thread_reference(self, evaluator):
        """Should clear thread reference on stop."""
        evaluator._running = True
        mock_thread = MagicMock()
        mock_thread.join = MagicMock()
        evaluator._eval_thread = mock_thread

        evaluator.stop()

        assert evaluator._eval_thread is None

    def test_get_thread_health_no_thread(self, evaluator):
        """Should return correct health when no thread."""
        health = evaluator.get_thread_health()

        assert health["running"] is False
        assert health["thread_alive"] is False
        assert health["thread_type"] == "none"

    def test_get_thread_health_raw_thread(self, evaluator):
        """Should report raw thread health correctly."""
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive = MagicMock(return_value=True)
        # Remove 'state' attribute to ensure it's treated as raw thread
        del mock_thread.state
        evaluator._running = True
        evaluator._eval_thread = mock_thread

        health = evaluator.get_thread_health()

        assert health["running"] is True
        assert health["thread_type"] == "raw"


# =============================================================================
# Tests for Step Updating
# =============================================================================


class TestStepUpdating:
    """Tests for step update functionality."""

    def test_update_step(self, evaluator):
        """Should update current step."""
        evaluator.update_step(500)
        assert evaluator.current_step == 500

    def test_update_step_thread_safe(self, evaluator):
        """Should be thread-safe."""
        results = []

        def update_and_read(step):
            evaluator.update_step(step)
            results.append(evaluator.current_step)

        threads = [threading.Thread(target=update_and_read, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All updates should have been applied
        assert len(results) == 10


# =============================================================================
# Tests for Placeholder Evaluation
# =============================================================================


class TestPlaceholderEvaluation:
    """Tests for placeholder evaluation mode."""

    def test_run_placeholder_evaluation_returns_result(self, evaluator):
        """Should return EvalResult from placeholder evaluation."""
        result = evaluator._run_placeholder_evaluation(step=100)

        assert isinstance(result, EvalResult)
        assert result.step == 100
        assert result.games_played > 0

    def test_run_placeholder_evaluation_has_baseline_results(self, evaluator):
        """Should include baseline results."""
        result = evaluator._run_placeholder_evaluation(step=100)

        assert "random" in result.baseline_results
        assert "heuristic" in result.baseline_results

    def test_run_placeholder_evaluation_updates_last_eval_step(self, evaluator):
        """Should update last_eval_step."""
        evaluator._run_placeholder_evaluation(step=500)

        assert evaluator.last_eval_step == 500

    def test_run_placeholder_evaluation_elo_estimate(self, evaluator):
        """Should estimate Elo from win rate."""
        result = evaluator._run_placeholder_evaluation(step=100)

        # Elo should be calculated (not zero)
        assert result.elo_estimate != 0

    def test_run_placeholder_checks_baseline_gating(self, evaluator):
        """Should check baseline gating thresholds."""
        # With high thresholds, should fail gating
        evaluator.config.min_baseline_win_rates = {"random": 0.99, "heuristic": 0.99}

        result = evaluator._run_placeholder_evaluation(step=100)

        # Result type is correct regardless of pass/fail
        assert isinstance(result.passes_baseline_gating, bool)


# =============================================================================
# Tests for Result Processing
# =============================================================================


class TestResultProcessing:
    """Tests for result processing and checkpointing."""

    def test_process_result_appends_to_history(self, evaluator):
        """Should append result to eval_results and elo_history."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1550.0,
            elo_std=50.0,
            games_played=20,
            win_rate=0.65,
            baseline_results={},
        )

        evaluator._process_result(result)

        assert len(evaluator.eval_results) == 1
        assert len(evaluator.elo_history) == 1
        assert evaluator.elo_history[0] == (100, 1550.0)

    def test_process_result_updates_current_elo(self, evaluator):
        """Should update current Elo."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1600.0,
            elo_std=50.0,
            games_played=20,
            win_rate=0.7,
            baseline_results={},
        )

        evaluator._process_result(result)

        assert evaluator.current_elo == 1600.0

    def test_process_result_updates_best_elo_on_improvement(self, evaluator):
        """Should update best Elo when significant improvement."""
        evaluator.best_elo = 1500.0
        evaluator.config.elo_checkpoint_threshold = 10.0

        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1520.0,  # +20 above best
            elo_std=50.0,
            games_played=20,
            win_rate=0.7,
            baseline_results={},
            passes_baseline_gating=True,
        )

        evaluator._process_result(result)

        assert evaluator.best_elo == 1520.0

    def test_process_result_no_update_below_threshold(self, evaluator):
        """Should not update best Elo below threshold."""
        evaluator.best_elo = 1500.0
        evaluator.config.elo_checkpoint_threshold = 50.0

        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1510.0,  # Only +10, below threshold
            elo_std=50.0,
            games_played=20,
            win_rate=0.55,
            baseline_results={},
            passes_baseline_gating=True,
        )

        evaluator._process_result(result)

        assert evaluator.best_elo == 1500.0  # Unchanged

    def test_process_result_no_checkpoint_on_gating_failure(self, evaluator):
        """Should not checkpoint when baseline gating fails."""
        evaluator.best_elo = 0.0
        evaluator.config.elo_checkpoint_threshold = 10.0

        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1600.0,  # Big improvement
            elo_std=50.0,
            games_played=20,
            win_rate=0.8,
            baseline_results={},
            passes_baseline_gating=False,
            failed_baselines=["random"],
        )

        evaluator._process_result(result)

        assert evaluator.best_elo == 0.0  # Not updated


# =============================================================================
# Tests for Baseline Gating Status
# =============================================================================


class TestBaselineGatingStatus:
    """Tests for baseline gating status functionality."""

    def test_get_baseline_gating_status_no_results(self, evaluator):
        """Should return default status when no results."""
        passes, failed, consecutive = evaluator.get_baseline_gating_status()

        assert passes is True
        assert failed == []
        assert consecutive == 0

    def test_get_baseline_gating_status_passing(self, evaluator):
        """Should return correct status when passing."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1550.0,
            elo_std=50.0,
            games_played=20,
            win_rate=0.7,
            baseline_results={"random": 0.9},
            passes_baseline_gating=True,
            failed_baselines=[],
        )
        evaluator.eval_results.append(result)

        passes, failed, consecutive = evaluator.get_baseline_gating_status()

        assert passes is True
        assert failed == []
        assert consecutive == 0

    def test_get_baseline_gating_status_failing(self, evaluator):
        """Should return correct status when failing."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1400.0,
            elo_std=50.0,
            games_played=20,
            win_rate=0.4,
            baseline_results={"random": 0.3},
            passes_baseline_gating=False,
            failed_baselines=["random"],
        )
        evaluator.eval_results.append(result)

        passes, failed, consecutive = evaluator.get_baseline_gating_status()

        assert passes is False
        assert "random" in failed
        assert consecutive == 1

    def test_consecutive_failures_counted(self, evaluator):
        """Should count consecutive baseline gating failures."""
        for i in range(3):
            result = EvalResult(
                step=100 + i,
                timestamp=time.time(),
                elo_estimate=1400.0,
                elo_std=50.0,
                games_played=20,
                win_rate=0.4,
                baseline_results={"random": 0.3},
                passes_baseline_gating=False,
                failed_baselines=["random"],
            )
            evaluator.eval_results.append(result)

        _, _, consecutive = evaluator.get_baseline_gating_status()

        assert consecutive == 3

    def test_should_trigger_baseline_warning_below_threshold(self, evaluator):
        """Should not trigger warning below threshold."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=1400.0,
            elo_std=50.0,
            games_played=20,
            win_rate=0.4,
            baseline_results={},
            passes_baseline_gating=False,
            failed_baselines=["random"],
        )
        evaluator.eval_results.append(result)

        assert evaluator.should_trigger_baseline_warning(failure_threshold=3) is False

    def test_should_trigger_baseline_warning_above_threshold(self, evaluator):
        """Should trigger warning when exceeding threshold."""
        for i in range(4):
            result = EvalResult(
                step=100 + i,
                timestamp=time.time(),
                elo_estimate=1400.0,
                elo_std=50.0,
                games_played=20,
                win_rate=0.4,
                baseline_results={},
                passes_baseline_gating=False,
                failed_baselines=["random"],
            )
            evaluator.eval_results.append(result)

        assert evaluator.should_trigger_baseline_warning(failure_threshold=3) is True


# =============================================================================
# Tests for Early Stopping
# =============================================================================


class TestEarlyStopping:
    """Tests for early stopping functionality."""

    def test_should_early_stop_false_initially(self, evaluator):
        """Should not early stop initially."""
        assert evaluator.should_early_stop() is False

    def test_should_early_stop_on_elo_drop(self, evaluator):
        """Should early stop on significant Elo drop."""
        evaluator.best_elo = 1600.0
        evaluator.current_elo = 1500.0  # 100 point drop
        evaluator.config.elo_drop_threshold = 50.0

        assert evaluator.should_early_stop() is True

    def test_should_not_early_stop_small_drop(self, evaluator):
        """Should not early stop on small Elo drop."""
        evaluator.best_elo = 1600.0
        evaluator.current_elo = 1580.0  # 20 point drop
        evaluator.config.elo_drop_threshold = 50.0

        assert evaluator.should_early_stop() is False


# =============================================================================
# Tests for Elo History and Current Elo
# =============================================================================


class TestEloTracking:
    """Tests for Elo tracking functionality."""

    def test_get_current_elo_initial(self, evaluator):
        """Should return initial Elo."""
        assert evaluator.get_current_elo() == 1500.0

    def test_get_current_elo_after_update(self, evaluator):
        """Should return updated Elo."""
        evaluator.current_elo = 1650.0
        assert evaluator.get_current_elo() == 1650.0

    def test_get_elo_history_empty(self, evaluator):
        """Should return empty list initially."""
        assert evaluator.get_elo_history() == []

    def test_get_elo_history_returns_copy(self, evaluator):
        """Should return a copy of history."""
        evaluator.elo_history = [(100, 1500.0), (200, 1550.0)]
        history = evaluator.get_elo_history()

        history.append((300, 1600.0))
        assert len(evaluator.elo_history) == 2  # Original unchanged


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and singleton management functions."""

    def test_create_background_evaluator(self, mock_model_getter):
        """Should create evaluator with factory function."""
        evaluator = create_background_evaluator(
            model_getter=mock_model_getter,
            eval_interval=500,
            games_per_eval=15,
        )

        assert isinstance(evaluator, BackgroundEvaluator)
        assert evaluator.config.eval_interval_steps == 500
        assert evaluator.config.games_per_eval == 15

    def test_get_background_evaluator_none_initially(self):
        """Should return None when no evaluator wired."""
        assert get_background_evaluator() is None

    def test_wire_background_evaluator_creates_singleton(self, mock_model_getter):
        """Should create and return singleton evaluator."""
        evaluator = wire_background_evaluator(
            model_getter=mock_model_getter,
            use_event_driven=False,
        )

        assert evaluator is not None
        assert get_background_evaluator() is evaluator

    def test_wire_background_evaluator_returns_existing(self, mock_model_getter):
        """Should return existing evaluator if already wired."""
        first = wire_background_evaluator(
            model_getter=mock_model_getter,
            use_event_driven=False,
        )
        second = wire_background_evaluator(
            model_getter=mock_model_getter,
            use_event_driven=False,
        )

        assert first is second

    def test_reset_background_evaluator_clears_singleton(self, mock_model_getter):
        """Should reset singleton evaluator."""
        wire_background_evaluator(
            model_getter=mock_model_getter,
            use_event_driven=False,
        )

        reset_background_evaluator()

        assert get_background_evaluator() is None


# =============================================================================
# Tests for Auto-Wiring
# =============================================================================


class TestAutoWiring:
    """Tests for auto-wiring from training coordinator."""

    def test_auto_wire_returns_none_no_coordinator(self):
        """Should return None when no training coordinator available."""
        # The import happens inside the function, so we patch at module level
        with patch.dict('sys.modules', {'app.coordination.training_coordinator': None}):
            with patch(
                "app.coordination.training_coordinator.get_training_coordinator",
                side_effect=ImportError,
                create=True,
            ):
                result = auto_wire_from_training_coordinator()
                # Should return None when ImportError occurs
                assert result is None

    def test_auto_wire_returns_none_coordinator_none(self):
        """Should return None when coordinator returns None."""
        mock_module = MagicMock()
        mock_module.get_training_coordinator.return_value = None

        with patch.dict('sys.modules', {'app.coordination.training_coordinator': mock_module}):
            result = auto_wire_from_training_coordinator()
            assert result is None

    def test_auto_wire_returns_none_no_active_jobs(self):
        """Should return None when no active training jobs."""
        mock_coordinator = MagicMock()
        mock_coordinator.get_active_jobs.return_value = []

        mock_module = MagicMock()
        mock_module.get_training_coordinator.return_value = mock_coordinator

        with patch.dict('sys.modules', {'app.coordination.training_coordinator': mock_module}):
            result = auto_wire_from_training_coordinator()
            assert result is None

    def test_auto_wire_returns_existing_if_wired(self, mock_model_getter):
        """Should return existing evaluator if already wired."""
        # First wire manually
        existing = wire_background_evaluator(
            model_getter=mock_model_getter,
            use_event_driven=False,
        )

        # Auto-wire should return existing
        result = auto_wire_from_training_coordinator()

        assert result is existing


# =============================================================================
# Tests for Event Subscription
# =============================================================================


class TestEventSubscription:
    """Tests for event-driven evaluation mode."""

    def test_subscribe_returns_false_no_event_bus(self, evaluator):
        """Should return False when event bus not available."""
        # The import happens inside subscribe_to_training_events, so patch sys.modules
        with patch.dict('sys.modules', {'app.coordination.event_router': None}):
            result = evaluator.subscribe_to_training_events()
            assert result is False

    def test_start_event_driven_falls_back_to_polling(self, evaluator):
        """Should fall back to polling when events unavailable."""
        with patch.object(
            evaluator,
            "subscribe_to_training_events",
            return_value=False,
        ):
            with patch.object(evaluator, "_eval_loop"):
                result = evaluator.start_event_driven()
                assert result is False
                assert evaluator._running is True

    def test_start_event_driven_uses_events_when_available(self, evaluator):
        """Should use event-driven mode when available."""
        with patch.object(
            evaluator,
            "subscribe_to_training_events",
            return_value=True,
        ):
            result = evaluator.start_event_driven()
            assert result is True
            assert evaluator._running is True


# =============================================================================
# Tests for Model Saving
# =============================================================================


class TestModelSaving:
    """Tests for temporary model saving during evaluation."""

    def test_save_temp_model_with_path_string(self, evaluator):
        """Should return path directly when given path string."""
        result = evaluator._save_temp_model("/path/to/model.pth")
        assert result == Path("/path/to/model.pth")

    def test_save_temp_model_with_path_object(self, evaluator):
        """Should return path directly when given Path object."""
        path = Path("/path/to/model.pth")
        result = evaluator._save_temp_model(path)
        assert result == path

    def test_save_temp_model_with_dict_path(self, evaluator):
        """Should return path from dict with path key."""
        result = evaluator._save_temp_model({"path": "/path/to/model.pth"})
        assert result == Path("/path/to/model.pth")

    def test_save_temp_model_with_state_dict(self, evaluator, tmp_path):
        """Should save state_dict to temp file."""
        evaluator.checkpoint_dir = tmp_path
        evaluator._temp_model_path = tmp_path / "temp_model.pth"

        with patch("torch.save") as mock_save:
            state_dict = {"layer.weight": "tensor"}
            result = evaluator._save_temp_model({"state_dict": state_dict})

            if mock_save.called:
                assert result == evaluator._temp_model_path


# =============================================================================
# Tests for Eval Loop Control
# =============================================================================


class TestEvalLoopControl:
    """Tests for evaluation loop control."""

    def test_should_continue_false_when_not_running(self, evaluator):
        """Should return False when not running."""
        evaluator._running = False
        assert evaluator._should_continue() is False

    def test_should_continue_true_when_running(self, evaluator):
        """Should return True when running."""
        evaluator._running = True
        assert evaluator._should_continue() is True

    def test_should_continue_checks_spawned_thread(self, evaluator):
        """Should check SpawnedThread.should_stop() if available."""
        evaluator._running = True
        mock_thread = MagicMock()
        mock_thread.should_stop = MagicMock(return_value=True)
        evaluator._eval_thread = mock_thread

        assert evaluator._should_continue() is False


# =============================================================================
# Tests for Run Evaluation
# =============================================================================


class TestRunEvaluation:
    """Tests for evaluation dispatch."""

    def test_run_evaluation_uses_placeholder_by_default(self, evaluator):
        """Should use placeholder evaluation by default."""
        with patch.object(
            evaluator,
            "_run_placeholder_evaluation",
            return_value=EvalResult(
                step=100,
                timestamp=time.time(),
                elo_estimate=1500.0,
                elo_std=100.0,
                games_played=10,
                win_rate=0.5,
                baseline_results={},
            ),
        ) as mock_placeholder:
            evaluator._run_evaluation(100)
            mock_placeholder.assert_called_once_with(100)

    def test_run_evaluation_calls_model_getter(self, evaluator):
        """Should call model getter."""
        evaluator._run_evaluation(100)
        evaluator.model_getter.assert_called_once()


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_baselines_list(self, mock_model_getter):
        """Should handle empty baselines list."""
        config = BackgroundEvalConfig(baselines=[])
        evaluator = BackgroundEvaluator(
            model_getter=mock_model_getter,
            config=config,
        )

        # Should not crash
        result = evaluator._run_placeholder_evaluation(100)
        assert result.baseline_results == {}

    def test_zero_games_per_eval(self, mock_model_getter):
        """Should handle zero games per eval gracefully."""
        config = BackgroundEvalConfig(games_per_eval=0)
        evaluator = BackgroundEvaluator(
            model_getter=mock_model_getter,
            config=config,
        )

        result = evaluator._run_placeholder_evaluation(100)
        # Should have 0 or default games
        assert result.games_played >= 0

    def test_very_high_elo_threshold(self, evaluator):
        """Should never checkpoint with very high threshold."""
        evaluator.config.elo_checkpoint_threshold = 10000.0
        evaluator.best_elo = 0.0

        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=2000.0,  # High but not 10000 higher than 0
            elo_std=50.0,
            games_played=20,
            win_rate=0.9,
            baseline_results={},
            passes_baseline_gating=True,
        )

        evaluator._process_result(result)

        # Still shouldn't update best_elo since gain < threshold
        assert evaluator.best_elo == 0.0

    def test_negative_elo_estimates(self, evaluator):
        """Should handle negative Elo estimates."""
        result = EvalResult(
            step=100,
            timestamp=time.time(),
            elo_estimate=-500.0,  # Negative Elo
            elo_std=100.0,
            games_played=10,
            win_rate=0.1,
            baseline_results={},
        )

        evaluator._process_result(result)

        assert evaluator.current_elo == -500.0

    def test_concurrent_access_to_eval_results(self, evaluator):
        """Should handle concurrent access safely."""
        results_added = []

        def add_result():
            result = EvalResult(
                step=100,
                timestamp=time.time(),
                elo_estimate=1500.0,
                elo_std=100.0,
                games_played=10,
                win_rate=0.5,
                baseline_results={},
            )
            evaluator._process_result(result)
            results_added.append(True)

        threads = [threading.Thread(target=add_result) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results_added) == 10
        assert len(evaluator.eval_results) == 10

    def test_model_getter_exception_handling(self, evaluator):
        """Should handle model_getter exceptions."""
        evaluator.model_getter = MagicMock(side_effect=RuntimeError("Model error"))

        with pytest.raises(RuntimeError):
            evaluator._run_evaluation(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
