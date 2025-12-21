"""Tests for Selfplay-Signal Integration (Feedback Loop).

Tests the integration between selfplay prioritization and training signals,
ensuring that:
- Regressing configs get priority boost for more selfplay data
- Plateau/overfit detection triggers diversity engine selection
- Signal computer integration works correctly

This validates the feedback loop plan implementation.
"""

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockConfigState:
    """Mock config state for testing."""
    games_since_training: int = 0
    last_training_time: float = field(default_factory=time.time)
    curriculum_weight: float = 1.0
    current_elo: float = 1500.0


@dataclass
class MockLoopState:
    """Mock unified loop state for testing."""
    configs: dict[str, MockConfigState] = field(default_factory=dict)


@dataclass
class MockTrainingSignals:
    """Mock training signals for testing."""
    elo_regression_detected: bool = False
    elo_trend: float = 0.0
    urgency: str = "NORMAL"


class MockSignalComputer:
    """Mock signal computer for testing."""

    def __init__(self, signals_by_config: dict[str, MockTrainingSignals] = None):
        self._signals = signals_by_config or {}
        self._default = MockTrainingSignals()

    def compute_signals(self, current_games: int, current_elo: float, config_key: str):
        return self._signals.get(config_key, self._default)


class MockTrainingScheduler:
    """Mock training scheduler for testing."""

    def __init__(self, threshold: int = 1000, quality_by_config: dict = None):
        self._threshold = threshold
        self._quality = quality_by_config or {}

    def _get_dynamic_threshold(self, config_key: str) -> int:
        return self._threshold

    def get_training_quality(self, config_key: str) -> dict:
        return self._quality.get(config_key, {})


# =============================================================================
# Priority Tests - ELO Regression Detection
# =============================================================================


class TestPriorityWithSignals:
    """Tests for selfplay priority with signal computer integration."""

    def _create_selfplay_generator(self, configs: dict, signal_computer=None, scheduler=None):
        """Create a SelfplayGenerator with mocked dependencies."""
        # Import here to avoid circular imports
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={k: MockConfigState(**v) for k, v in configs.items()})
        config = SelfplayConfig()
        event_bus = MagicMock()

        generator = SelfplayGenerator(config, state, event_bus)
        generator._signal_computer = signal_computer
        generator._training_scheduler = scheduler

        return generator

    def test_regressing_config_gets_priority_boost(self):
        """Regressing config should get +0.20 priority boost."""
        # Setup: Two configs with same base priority, one regressing
        configs = {
            "square8_2p": {"games_since_training": 500, "current_elo": 1500},
            "square19_2p": {"games_since_training": 500, "current_elo": 1400},
        }

        signal_computer = MockSignalComputer({
            "square8_2p": MockTrainingSignals(elo_regression_detected=True),
            "square19_2p": MockTrainingSignals(elo_regression_detected=False),
        })

        scheduler = MockTrainingScheduler(threshold=1000)
        generator = self._create_selfplay_generator(configs, signal_computer, scheduler)

        priorities = generator.get_config_priorities()

        # Regressing config should have higher priority
        assert priorities["square8_2p"] > priorities["square19_2p"]
        # The boost should be approximately 0.20
        diff = priorities["square8_2p"] - priorities["square19_2p"]
        assert 0.15 <= diff <= 0.25, f"Expected ~0.20 boost, got {diff}"

    def test_declining_config_gets_partial_boost(self):
        """Declining (negative trend) config should get +0.10 partial boost."""
        configs = {
            "square8_2p": {"games_since_training": 500},
            "square19_2p": {"games_since_training": 500},
        }

        signal_computer = MockSignalComputer({
            "square8_2p": MockTrainingSignals(elo_trend=-5.0),  # Declining
            "square19_2p": MockTrainingSignals(elo_trend=0.0),  # Flat
        })

        scheduler = MockTrainingScheduler(threshold=1000)
        generator = self._create_selfplay_generator(configs, signal_computer, scheduler)

        priorities = generator.get_config_priorities()

        # Declining config should have higher priority
        assert priorities["square8_2p"] > priorities["square19_2p"]
        diff = priorities["square8_2p"] - priorities["square19_2p"]
        assert 0.05 <= diff <= 0.15, f"Expected ~0.10 boost, got {diff}"

    def test_improving_config_gets_slight_reduction(self):
        """Config with high positive ELO trend should get -0.05 reduction."""
        configs = {
            "square8_2p": {"games_since_training": 500},
            "square19_2p": {"games_since_training": 500},
        }

        signal_computer = MockSignalComputer({
            "square8_2p": MockTrainingSignals(elo_trend=25.0),  # Strong improvement
            "square19_2p": MockTrainingSignals(elo_trend=0.0),  # Flat
        })

        scheduler = MockTrainingScheduler(threshold=1000)
        generator = self._create_selfplay_generator(configs, signal_computer, scheduler)

        priorities = generator.get_config_priorities()

        # Improving config should have lower priority (let others catch up)
        assert priorities["square8_2p"] < priorities["square19_2p"]

    def test_no_signal_computer_fallback(self):
        """Without signal computer, priority should work without Factor 4."""
        configs = {
            "square8_2p": {"games_since_training": 800},
            "square19_2p": {"games_since_training": 200},
        }

        scheduler = MockTrainingScheduler(threshold=1000)
        generator = self._create_selfplay_generator(configs, None, scheduler)

        priorities = generator.get_config_priorities()

        # Config closer to threshold should still have higher priority
        assert priorities["square8_2p"] > priorities["square19_2p"]

    def test_get_prioritized_config_returns_regressing_first(self):
        """get_prioritized_config should return the regressing config."""
        configs = {
            "square8_2p": {"games_since_training": 100},  # Far from threshold
            "square19_2p": {"games_since_training": 100},  # Far from threshold
        }

        signal_computer = MockSignalComputer({
            "square8_2p": MockTrainingSignals(elo_regression_detected=True),
            "square19_2p": MockTrainingSignals(elo_regression_detected=False),
        })

        scheduler = MockTrainingScheduler(threshold=1000)
        generator = self._create_selfplay_generator(configs, signal_computer, scheduler)

        prioritized = generator.get_prioritized_config()
        assert prioritized == "square8_2p"


# =============================================================================
# Diversity Need Tests
# =============================================================================


class TestDiversityNeed:
    """Tests for diversity need calculation from training quality."""

    def _create_generator(self, scheduler=None):
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={"square8_2p": MockConfigState()})
        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._training_scheduler = scheduler
        return generator

    def test_overfit_detected_returns_high_diversity(self):
        """Overfit detection should return 0.9 diversity need."""
        scheduler = MockTrainingScheduler(
            quality_by_config={"square8_2p": {"overfit_detected": True}}
        )
        generator = self._create_generator(scheduler)

        diversity = generator._get_diversity_need("square8_2p")
        assert diversity == 0.9

    def test_loss_plateau_returns_moderate_diversity(self):
        """Loss plateau should return 0.6 diversity need."""
        scheduler = MockTrainingScheduler(
            quality_by_config={"square8_2p": {"loss_plateau": True}}
        )
        generator = self._create_generator(scheduler)

        diversity = generator._get_diversity_need("square8_2p")
        assert diversity == 0.6

    def test_no_issues_returns_zero_diversity(self):
        """No training issues should return 0.0 diversity need."""
        scheduler = MockTrainingScheduler(
            quality_by_config={"square8_2p": {}}
        )
        generator = self._create_generator(scheduler)

        diversity = generator._get_diversity_need("square8_2p")
        assert diversity == 0.0

    def test_no_scheduler_returns_zero(self):
        """Without scheduler, should return 0.0."""
        generator = self._create_generator(None)

        diversity = generator._get_diversity_need("square8_2p")
        assert diversity == 0.0


# =============================================================================
# Adaptive Engine Selection Tests
# =============================================================================


class TestAdaptiveEngineSelection:
    """Tests for adaptive engine selection based on feedback signals."""

    def _create_generator(self, scheduler=None, signal_computer=None):
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={"square8_2p": MockConfigState()})
        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._training_scheduler = scheduler
        generator._signal_computer = signal_computer
        return generator

    def test_high_diversity_need_selects_mcts(self):
        """High diversity need (>0.7) should select MCTS for exploration."""
        scheduler = MockTrainingScheduler(
            quality_by_config={"square8_2p": {"overfit_detected": True}}  # 0.9 diversity
        )
        generator = self._create_generator(scheduler)

        engine = generator.get_adaptive_engine("square8_2p")
        assert engine == "mcts"

    def test_high_priority_selects_gumbel(self):
        """High priority (>=0.7) without diversity need should select gumbel."""
        # High games = high priority (close to threshold)
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={
            "square8_2p": MockConfigState(games_since_training=900)  # 90% of threshold
        })
        scheduler = MockTrainingScheduler(threshold=1000)

        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._training_scheduler = scheduler

        engine = generator.get_adaptive_engine("square8_2p")
        assert engine == "gumbel"

    def test_low_priority_selects_descent(self):
        """Low priority should select descent for throughput."""
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={
            "square8_2p": MockConfigState(games_since_training=100)  # 10% of threshold
        })
        scheduler = MockTrainingScheduler(threshold=1000)

        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._training_scheduler = scheduler

        engine = generator.get_adaptive_engine("square8_2p")
        assert engine == "descent"

    def test_moderate_diversity_uses_priority_fallback(self):
        """Moderate diversity (<=0.7) should fall back to priority-based selection."""
        scheduler = MockTrainingScheduler(
            threshold=1000,
            quality_by_config={"square8_2p": {"loss_plateau": True}}  # 0.6 diversity
        )

        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={
            "square8_2p": MockConfigState(games_since_training=100)  # Low priority
        })

        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._training_scheduler = scheduler

        engine = generator.get_adaptive_engine("square8_2p")
        # Should use descent (low priority) not mcts (diversity < 0.7)
        assert engine == "descent"


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeedbackLoopIntegration:
    """Integration tests for the complete feedback loop."""

    def test_regression_triggers_both_priority_boost_and_engine_adaptation(self):
        """Regression should boost priority AND may trigger engine change."""
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        # Config with regression and overfit (full feedback loop)
        state = MockLoopState(configs={
            "square8_2p": MockConfigState(games_since_training=500)
        })

        signal_computer = MockSignalComputer({
            "square8_2p": MockTrainingSignals(elo_regression_detected=True)
        })

        scheduler = MockTrainingScheduler(
            threshold=1000,
            quality_by_config={"square8_2p": {"overfit_detected": True}}
        )

        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._signal_computer = signal_computer
        generator._training_scheduler = scheduler

        # Priority should be boosted (baseline ~0.5 proximity, +0.20 regression)
        priorities = generator.get_config_priorities()
        assert priorities["square8_2p"] >= 0.4  # At least base + boost

        # Engine should be MCTS due to overfit
        engine = generator.get_adaptive_engine("square8_2p")
        assert engine == "mcts"

    def test_full_loop_prioritizes_struggling_config(self):
        """In a multi-config scenario, struggling config should win priority."""
        from scripts.unified_loop.selfplay import SelfplayGenerator
        from scripts.unified_loop.config import SelfplayConfig

        state = MockLoopState(configs={
            "square8_2p": MockConfigState(games_since_training=400),   # 40% proximity
            "square19_2p": MockConfigState(games_since_training=600),  # 60% proximity
            "hexagonal_2p": MockConfigState(games_since_training=300), # 30% proximity + regression
        })

        signal_computer = MockSignalComputer({
            "hexagonal_2p": MockTrainingSignals(elo_regression_detected=True),
        })

        scheduler = MockTrainingScheduler(threshold=1000)

        generator = SelfplayGenerator(SelfplayConfig(), state, MagicMock())
        generator._signal_computer = signal_computer
        generator._training_scheduler = scheduler

        # Despite lower proximity, hexagonal should win due to regression boost
        prioritized = generator.get_prioritized_config()
        assert prioritized == "hexagonal_2p"
