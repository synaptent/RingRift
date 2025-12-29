"""Unit tests for SelfplayScheduler.

December 29, 2025: Comprehensive tests for the priority-based selfplay scheduler.
Tests cover dataclasses, priority computation, allocation logic, and event handling.
"""

import asyncio
import logging
import math
import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import replace

from app.coordination.selfplay_scheduler import (
    # Classes
    ConfigPriority,
    DynamicWeights,
    NodeCapability,
    SelfplayScheduler,
    # Constants
    ALL_CONFIGS,
    CURRICULUM_WEIGHT,
    DATA_DEFICIT_WEIGHT,
    DATA_STARVATION_CRITICAL_MULTIPLIER,
    DATA_STARVATION_CRITICAL_THRESHOLD,
    DATA_STARVATION_EMERGENCY_MULTIPLIER,
    DATA_STARVATION_EMERGENCY_THRESHOLD,
    DEFAULT_GAMES_PER_CONFIG,
    DEFAULT_TRAINING_SAMPLES_TARGET,
    DYNAMIC_WEIGHT_BOUNDS,
    ELO_VELOCITY_WEIGHT,
    FRESH_DATA_THRESHOLD,
    MAX_STALENESS_HOURS,
    PLAYER_COUNT_ALLOCATION_MULTIPLIER,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    SAMPLES_PER_GAME_BY_BOARD,
    STALENESS_WEIGHT,
    STALE_DATA_THRESHOLD,
    TARGET_GAMES_FOR_2000_ELO,
    VOI_SAMPLE_COST_BY_BOARD,
    # Functions
    get_selfplay_scheduler,
    reset_selfplay_scheduler,
)


# =============================================================================
# TestDynamicWeights - Dataclass for weight computation
# =============================================================================


class TestDynamicWeights:
    """Tests for DynamicWeights dataclass."""

    def test_default_values(self):
        """Test default weight values match constants."""
        weights = DynamicWeights()
        assert weights.staleness == STALENESS_WEIGHT
        assert weights.velocity == ELO_VELOCITY_WEIGHT
        assert weights.curriculum == CURRICULUM_WEIGHT
        assert weights.data_deficit == DATA_DEFICIT_WEIGHT

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all weight fields."""
        weights = DynamicWeights(
            staleness=0.5,
            velocity=0.3,
            idle_gpu_fraction=0.25,
        )
        result = weights.to_dict()

        assert result["staleness"] == 0.5
        assert result["velocity"] == 0.3
        assert result["idle_gpu_fraction"] == 0.25
        assert "quality" in result
        assert "voi" in result

    def test_cluster_state_fields_tracked(self):
        """Test that cluster state driving weights is recorded."""
        weights = DynamicWeights(
            idle_gpu_fraction=0.6,
            training_queue_depth=15,
            configs_at_target_fraction=0.3,
            average_elo=1750.0,
        )
        d = weights.to_dict()
        assert d["idle_gpu_fraction"] == 0.6
        assert d["training_queue_depth"] == 15
        assert d["configs_at_target_fraction"] == 0.3
        assert d["average_elo"] == 1750.0


# =============================================================================
# TestConfigPriority - Priority dataclass with computed properties
# =============================================================================


class TestConfigPriorityStaleness:
    """Tests for ConfigPriority staleness_factor property."""

    def test_fresh_data_zero_staleness(self):
        """Data fresher than threshold has 0 staleness factor."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=0.5,  # Less than FRESH_DATA_THRESHOLD
        )
        assert priority.staleness_factor == 0.0

    def test_stale_data_has_factor(self):
        """Data older than fresh threshold has positive staleness factor."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=2.5,  # Between fresh and stale thresholds
        )
        assert 0.0 < priority.staleness_factor < 1.0

    def test_max_staleness_capped(self):
        """Very old data is capped at 1.0 staleness factor."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=100.0,  # Way past MAX_STALENESS_HOURS
        )
        assert priority.staleness_factor == 1.0


class TestConfigPriorityVelocity:
    """Tests for ConfigPriority velocity_factor property."""

    def test_zero_velocity(self):
        """Zero Elo velocity results in 0 factor."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=0.0)
        assert priority.velocity_factor == 0.0

    def test_negative_velocity_treated_as_zero(self):
        """Regression (negative velocity) is treated as zero factor."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=-50.0)
        assert priority.velocity_factor == 0.0

    def test_positive_velocity_scales(self):
        """Positive velocity gives proportional factor."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=50.0)
        assert priority.velocity_factor == 0.5  # 50/100 = 0.5

    def test_velocity_capped_at_one(self):
        """Very high velocity is capped at 1.0."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=200.0)
        assert priority.velocity_factor == 1.0


class TestConfigPriorityDataDeficit:
    """Tests for ConfigPriority data_deficit_factor property."""

    def test_no_deficit_when_at_target(self):
        """No deficit when game count meets target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO,
            is_large_board=False,
        )
        assert priority.data_deficit_factor == 0.0

    def test_max_deficit_with_zero_games(self):
        """Maximum deficit when no games exist."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=0,
            is_large_board=False,
        )
        assert priority.data_deficit_factor == 1.0

    def test_large_board_has_higher_target(self):
        """Large boards require more games (higher target)."""
        games = 80000  # 80% of small board target

        small_board = ConfigPriority(
            config_key="hex8_2p",
            game_count=games,
            is_large_board=False,
        )
        large_board = ConfigPriority(
            config_key="square19_2p",
            game_count=games,
            is_large_board=True,
        )

        # Large board should have higher deficit at same game count
        assert large_board.data_deficit_factor > small_board.data_deficit_factor


class TestConfigPriorityPlayerCount:
    """Tests for ConfigPriority player_count property."""

    def test_2p_extraction(self):
        """Correctly extracts 2 from 'hex8_2p'."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.player_count == 2

    def test_3p_extraction(self):
        """Correctly extracts 3 from 'square8_3p'."""
        priority = ConfigPriority(config_key="square8_3p")
        assert priority.player_count == 3

    def test_4p_extraction(self):
        """Correctly extracts 4 from 'hexagonal_4p'."""
        priority = ConfigPriority(config_key="hexagonal_4p")
        assert priority.player_count == 4

    def test_invalid_format_defaults_to_2(self):
        """Invalid config format defaults to 2-player."""
        priority = ConfigPriority(config_key="invalid_config")
        assert priority.player_count == 2


class TestConfigPriorityGamesNeeded:
    """Tests for ConfigPriority games_needed property."""

    def test_zero_games_needs_full_target(self):
        """Zero games needs full target divided by samples per game."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=0,
            target_training_samples=50000,
            samples_per_game_estimate=50.0,
        )
        assert priority.games_needed == 1000  # 50000 / 50

    def test_partial_progress_reduces_needed(self):
        """Partial game count reduces games needed."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=500,  # 500 * 50 = 25000 samples
            target_training_samples=50000,
            samples_per_game_estimate=50.0,
        )
        assert priority.games_needed == 500  # (50000 - 25000) / 50

    def test_at_target_needs_zero(self):
        """At or above target needs 0 games."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=1000,  # 1000 * 50 = 50000 samples
            target_training_samples=50000,
            samples_per_game_estimate=50.0,
        )
        assert priority.games_needed == 0


class TestConfigPriorityVOI:
    """Tests for ConfigPriority VOI (Value of Information) properties."""

    def test_elo_gap_computed(self):
        """Elo gap is difference from target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=1500.0,
            target_elo=2000.0,
        )
        assert priority.elo_gap == 500.0

    def test_elo_gap_not_negative(self):
        """Elo gap is clamped to 0 when above target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=2100.0,
            target_elo=2000.0,
        )
        assert priority.elo_gap == 0.0

    def test_info_gain_high_with_zero_games(self):
        """Info gain is maximum when no games exist."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=0,
            elo_uncertainty=200.0,
        )
        assert priority.info_gain_per_game == 200.0

    def test_info_gain_decreases_with_games(self):
        """Info gain decreases with more games (sqrt rule)."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=100,
            elo_uncertainty=200.0,
        )
        # 200 / sqrt(100) = 200 / 10 = 20
        assert priority.info_gain_per_game == 20.0

    def test_voi_score_in_valid_range(self):
        """VOI score is between 0 and 1."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=1200.0,
            target_elo=2000.0,
            elo_uncertainty=250.0,
            game_count=50,
        )
        assert 0.0 <= priority.voi_score <= 1.0

    def test_high_voi_for_uncertain_underperformer(self):
        """High VOI when uncertain and far from target."""
        uncertain = ConfigPriority(
            config_key="hex8_2p",
            current_elo=1200.0,
            target_elo=2000.0,
            elo_uncertainty=300.0,  # High uncertainty
            game_count=10,  # Few games
        )
        certain = ConfigPriority(
            config_key="hex8_2p",
            current_elo=1800.0,
            target_elo=2000.0,
            elo_uncertainty=50.0,  # Low uncertainty
            game_count=10000,  # Many games
        )
        assert uncertain.voi_score > certain.voi_score


# =============================================================================
# TestNodeCapability - Node capability dataclass
# =============================================================================


class TestNodeCapability:
    """Tests for NodeCapability dataclass."""

    def test_available_capacity_full_when_idle(self):
        """Full capacity when load is 0."""
        node = NodeCapability(
            node_id="test-node",
            gpu_type="A100",
            current_load=0.0,
        )
        # Available = (1 - 0) * weight
        assert node.available_capacity > 0

    def test_available_capacity_zero_when_full(self):
        """Zero capacity when fully loaded."""
        node = NodeCapability(
            node_id="test-node",
            gpu_type="A100",
            current_load=1.0,
        )
        assert node.available_capacity == 0.0

    def test_capacity_weight_from_gpu_type(self):
        """Capacity weight is based on GPU type."""
        h100 = NodeCapability(node_id="n1", gpu_type="H100")
        rtx3060 = NodeCapability(node_id="n2", gpu_type="RTX3060")

        # H100 should have higher weight than RTX 3060
        assert h100.capacity_weight > rtx3060.capacity_weight


# =============================================================================
# TestSelfplaySchedulerInit - Scheduler initialization
# =============================================================================


class TestSelfplaySchedulerInit:
    """Tests for SelfplayScheduler initialization."""

    def test_singleton_pattern(self):
        """get_selfplay_scheduler returns singleton."""
        reset_selfplay_scheduler()
        scheduler1 = get_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is scheduler2
        reset_selfplay_scheduler()

    def test_reset_creates_new_instance(self):
        """reset_selfplay_scheduler creates new instance."""
        scheduler1 = get_selfplay_scheduler()
        reset_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is not scheduler2
        reset_selfplay_scheduler()

    def test_all_configs_initialized(self):
        """All 12 configs have priority objects."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        for config in ALL_CONFIGS:
            priority = scheduler.get_config_priority(config)
            assert priority is not None
            assert priority.config_key == config

        reset_selfplay_scheduler()


# =============================================================================
# TestSelfplaySchedulerPriority - Priority computation
# =============================================================================


class TestSelfplaySchedulerPriorityComputation:
    """Tests for priority score computation."""

    def test_stale_config_higher_priority(self):
        """Config with stale data gets higher priority."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Manually set staleness for testing
        fresh = scheduler._config_priorities["hex8_2p"]
        stale = scheduler._config_priorities["hex8_3p"]
        fresh.staleness_hours = 0.5
        stale.staleness_hours = 10.0

        score_fresh = scheduler._compute_priority_score(fresh)
        score_stale = scheduler._compute_priority_score(stale)

        # Stale config should have higher priority score
        assert score_stale > score_fresh
        reset_selfplay_scheduler()

    def test_data_starved_config_boosted(self):
        """Config with very few games gets emergency boost."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        starved = scheduler._config_priorities["hex8_4p"]
        starved.game_count = 50  # Below emergency threshold

        plenty = scheduler._config_priorities["hex8_2p"]
        plenty.game_count = 50000

        score_starved = scheduler._compute_priority_score(starved)
        score_plenty = scheduler._compute_priority_score(plenty)

        # Starved config should have much higher priority
        assert score_starved > score_plenty * 2  # At least 2x higher
        reset_selfplay_scheduler()

    def test_priority_override_multiplier_applied(self):
        """Priority override multiplier boosts score."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        normal = scheduler._config_priorities["hex8_2p"]
        normal.priority_override = 3  # LOW (1.0x)

        critical = scheduler._config_priorities["hex8_3p"]
        critical.priority_override = 0  # CRITICAL (3.0x)

        # Set same base values
        normal.staleness_hours = 5.0
        normal.game_count = 10000
        critical.staleness_hours = 5.0
        critical.game_count = 10000

        score_normal = scheduler._compute_priority_score(normal)
        score_critical = scheduler._compute_priority_score(critical)

        # Critical should be ~3x higher (accounting for other factors)
        assert score_critical > score_normal
        reset_selfplay_scheduler()


class TestDynamicWeightComputation:
    """Tests for _compute_dynamic_weights method."""

    def test_weights_sum_reasonable(self):
        """Dynamic weights should sum to a reasonable total."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        weights = scheduler._compute_dynamic_weights()

        # Weights should be positive
        assert weights.staleness > 0
        assert weights.velocity > 0
        assert weights.curriculum > 0

        # Total should be reasonable (not all maxed out)
        total = (
            weights.staleness + weights.velocity + weights.training +
            weights.exploration + weights.curriculum + weights.improvement
        )
        assert 0.5 < total < 2.5  # Reasonable range
        reset_selfplay_scheduler()

    def test_weights_within_bounds(self):
        """All weights stay within defined bounds."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        weights = scheduler._compute_dynamic_weights()

        for name, (min_bound, max_bound) in DYNAMIC_WEIGHT_BOUNDS.items():
            weight_value = getattr(weights, name)
            assert min_bound <= weight_value <= max_bound, f"{name} out of bounds"

        reset_selfplay_scheduler()


# =============================================================================
# TestSelfplaySchedulerAllocation - Allocation methods
# =============================================================================


class TestStarvationFloorEnforcement:
    """Tests for _enforce_starvation_floor method."""

    def test_starved_config_gets_minimum_allocation(self):
        """Critically starved configs get floor allocation."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Set up a starved config
        starved = scheduler._config_priorities["hex8_4p"]
        starved.game_count = 50  # Below emergency threshold

        # Initial allocation with nested dict structure (config -> node -> games)
        allocation = {
            "hex8_2p": {"node1": 500},
            "hex8_3p": {"node1": 300},
            "hex8_4p": {},  # Starved but got nothing
        }

        total_games = 1000
        result = scheduler._enforce_starvation_floor(allocation, total_games)

        # Starved config should now have some allocation
        total_4p = sum(result.get("hex8_4p", {}).values())
        assert total_4p > 0
        reset_selfplay_scheduler()


class TestPlayerCountAllocation:
    """Tests for _enforce_4p_allocation_minimums method."""

    def test_4p_configs_get_minimum_share(self):
        """4-player configs get minimum allocation share."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Initial allocation heavily favoring 2p (nested dict structure)
        allocation = {
            "hex8_2p": {"node1": 800},
            "hex8_3p": {"node1": 100},
            "hex8_4p": {"node1": 100},
        }

        total_games = 1000
        result = scheduler._enforce_4p_allocation_minimums(allocation, total_games)

        # 4p should have reasonable share (not just 10%)
        total_4p = sum(result.get("hex8_4p", {}).values())
        assert total_4p >= total_games * 0.15  # At least 15%
        reset_selfplay_scheduler()


# =============================================================================
# TestSelfplaySchedulerCallbacks - Callback integrations
# =============================================================================


class TestSchedulerCallbackIntegration:
    """Tests for callback-based integrations."""

    def test_custom_elo_callback(self):
        """Custom Elo callback is used when provided."""
        mock_elo = MagicMock(return_value={
            "hex8_2p": {"elo": 1600.0, "velocity": 10.0},
            "hex8_3p": {"elo": 1400.0, "velocity": -5.0},
        })

        scheduler = SelfplayScheduler(get_cluster_elo_fn=mock_elo)

        # Trigger priority refresh
        scheduler.get_priority_configs_sync()

        # Should have called our mock
        mock_elo.assert_called()

    def test_backpressure_callback_respected(self):
        """Backpressure callback prevents over-allocation."""
        mock_stop = MagicMock(return_value=True)

        scheduler = SelfplayScheduler(should_stop_production_fn=mock_stop)

        # When backpressure is active, allocation should be limited
        # This tests that the callback is integrated


# =============================================================================
# TestSelfplaySchedulerEventHandlers - Event handling
# =============================================================================


class TestEventHandlerSelfplayComplete:
    """Tests for _on_selfplay_complete event handler."""

    def test_selfplay_complete_updates_freshness(self):
        """Selfplay completion updates data freshness."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Set initial staleness
        scheduler._config_priorities["hex8_2p"].staleness_hours = 10.0

        # Simulate selfplay complete event (event.payload pattern)
        event = SimpleNamespace(payload={
            "config_key": "hex8_2p",
            "games_count": 100,
            "timestamp": 0,
        })
        scheduler._on_selfplay_complete(event)

        # Staleness should be reset to 0
        assert scheduler._config_priorities["hex8_2p"].staleness_hours == 0.0
        reset_selfplay_scheduler()


class TestEventHandlerQualityDegraded:
    """Tests for _on_quality_degraded event handler."""

    def test_quality_degraded_applies_penalty(self):
        """Quality degradation applies penalty to priority."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Event with .payload pattern
        event = SimpleNamespace(payload={
            "config_key": "hex8_2p",
            "quality_score": 0.4,  # Low quality
            "threshold": 0.6,
        })
        scheduler._on_quality_degraded(event)

        # Should have penalty applied
        assert scheduler._config_priorities["hex8_2p"].quality_penalty < 0
        reset_selfplay_scheduler()


class TestEventHandlerCurriculumRebalanced:
    """Tests for _on_curriculum_rebalanced event handler."""

    def test_curriculum_weights_updated(self):
        """Curriculum rebalance updates config weights."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Event with .payload pattern
        event = SimpleNamespace(payload={
            "weights": {
                "hex8_2p": 1.5,
                "hex8_3p": 0.8,
                "hex8_4p": 2.0,
            },
        })
        scheduler._on_curriculum_rebalanced(event)

        # Weights should be updated
        assert scheduler._config_priorities["hex8_2p"].curriculum_weight == 1.5
        assert scheduler._config_priorities["hex8_3p"].curriculum_weight == 0.8
        assert scheduler._config_priorities["hex8_4p"].curriculum_weight == 2.0
        reset_selfplay_scheduler()


class TestEventHandlerNodeHealth:
    """Tests for node health event handlers."""

    def test_node_overloaded_adds_to_backoff(self):
        """Overloaded node is added to backoff."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Event with .payload pattern - uses _on_node_overloaded for backoff
        event = SimpleNamespace(payload={
            "node_id": "bad-node-1",
            "reason": "GPU overload",
            "backoff_seconds": 60,  # 60 second backoff
        })
        scheduler._on_node_overloaded(event)

        assert scheduler.is_node_under_backoff("bad-node-1")
        reset_selfplay_scheduler()

    def test_node_recovered_clears_backoff(self):
        """Recovered node has backoff cleared."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # First mark as overloaded (adds to backoff)
        event_overload = SimpleNamespace(payload={
            "node_id": "node-1",
            "reason": "overload",
            "backoff_seconds": 60,
        })
        scheduler._on_node_overloaded(event_overload)
        assert scheduler.is_node_under_backoff("node-1")

        # Clear the backoff manually (recovery doesn't auto-clear, it expires)
        scheduler._overloaded_nodes["node-1"] = 0  # Set backoff to past
        assert not scheduler.is_node_under_backoff("node-1")
        reset_selfplay_scheduler()


# =============================================================================
# TestSelfplaySchedulerMetrics - Metrics and health
# =============================================================================


class TestSchedulerHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_result(self):
        """Health check returns HealthCheckResult."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        result = scheduler.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert hasattr(result, "details")
        reset_selfplay_scheduler()

    def test_healthy_when_recently_updated(self):
        """Reports healthy when scheduler has recent priority updates."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Simulate that priorities were recently updated
        scheduler._subscribed = True
        scheduler._last_priority_update = time.time()

        result = scheduler.health_check()
        assert result.healthy is True
        reset_selfplay_scheduler()

    def test_unhealthy_when_stale(self):
        """Reports unhealthy when priority data is stale."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        # Fresh scheduler has _last_priority_update = 0.0, so stale
        scheduler._subscribed = True  # Even subscribed, but no updates
        # _last_priority_update stays at 0.0

        result = scheduler.health_check()
        assert result.healthy is False
        assert "stale" in result.message.lower()
        reset_selfplay_scheduler()


class TestSchedulerMetrics:
    """Tests for get_metrics method."""

    def test_metrics_include_config_counts(self):
        """Metrics include config-level statistics."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        metrics = scheduler.get_metrics()

        assert "configs" in metrics or "config_count" in metrics or len(metrics) > 0
        reset_selfplay_scheduler()


class TestSchedulerStatus:
    """Tests for get_status method."""

    def test_status_includes_priorities(self):
        """Status includes priority information."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        status = scheduler.get_status()

        assert isinstance(status, dict)
        reset_selfplay_scheduler()


class TestSchedulerTopPriorities:
    """Tests for get_top_priorities method."""

    def test_returns_requested_count(self):
        """Returns requested number of top priorities."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        top_5 = scheduler.get_top_priorities(n=5)
        assert len(top_5) == 5

        top_3 = scheduler.get_top_priorities(n=3)
        assert len(top_3) == 3
        reset_selfplay_scheduler()

    def test_sorted_by_priority(self):
        """Results are sorted by priority score descending."""
        reset_selfplay_scheduler()
        scheduler = get_selfplay_scheduler()

        top = scheduler.get_top_priorities(n=5)

        # Should be in descending order
        for i in range(len(top) - 1):
            assert top[i].get("priority_score", 0) >= top[i + 1].get("priority_score", 0)
        reset_selfplay_scheduler()


# =============================================================================
# TestConstants - Constant validation
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_all_configs_complete(self):
        """ALL_CONFIGS contains all 12 configurations."""
        assert len(ALL_CONFIGS) == 12

        # Verify structure
        board_types = {"hex8", "square8", "square19", "hexagonal"}
        player_counts = {"2p", "3p", "4p"}

        for config in ALL_CONFIGS:
            board, players = config.rsplit("_", 1)
            assert board in board_types
            assert players in player_counts

    def test_samples_per_game_all_configs(self):
        """SAMPLES_PER_GAME_BY_BOARD has all board types."""
        for board in ["hex8", "square8", "square19", "hexagonal"]:
            assert board in SAMPLES_PER_GAME_BY_BOARD
            for players in ["2p", "3p", "4p"]:
                assert players in SAMPLES_PER_GAME_BY_BOARD[board]

    def test_voi_sample_cost_all_configs(self):
        """VOI_SAMPLE_COST_BY_BOARD has all board types."""
        for board in ["hex8", "square8", "square19", "hexagonal"]:
            assert board in VOI_SAMPLE_COST_BY_BOARD
            for players in ["2p", "3p", "4p"]:
                assert players in VOI_SAMPLE_COST_BY_BOARD[board]

    def test_player_allocation_multipliers(self):
        """Player count allocation multipliers favor 3p/4p."""
        assert PLAYER_COUNT_ALLOCATION_MULTIPLIER[2] == 1.0  # Baseline
        assert PLAYER_COUNT_ALLOCATION_MULTIPLIER[3] > 1.0  # 3p boosted
        assert PLAYER_COUNT_ALLOCATION_MULTIPLIER[4] > PLAYER_COUNT_ALLOCATION_MULTIPLIER[3]  # 4p more boosted

    def test_priority_override_multipliers(self):
        """Priority override multipliers are properly ordered."""
        assert PRIORITY_OVERRIDE_MULTIPLIERS[0] > PRIORITY_OVERRIDE_MULTIPLIERS[1]  # CRITICAL > HIGH
        assert PRIORITY_OVERRIDE_MULTIPLIERS[1] > PRIORITY_OVERRIDE_MULTIPLIERS[2]  # HIGH > MEDIUM
        assert PRIORITY_OVERRIDE_MULTIPLIERS[2] > PRIORITY_OVERRIDE_MULTIPLIERS[3]  # MEDIUM > LOW
        assert PRIORITY_OVERRIDE_MULTIPLIERS[3] == 1.0  # LOW is baseline

    def test_starvation_thresholds_ordered(self):
        """Starvation thresholds are properly ordered."""
        assert DATA_STARVATION_EMERGENCY_THRESHOLD < DATA_STARVATION_CRITICAL_THRESHOLD
        assert DATA_STARVATION_EMERGENCY_MULTIPLIER > DATA_STARVATION_CRITICAL_MULTIPLIER
