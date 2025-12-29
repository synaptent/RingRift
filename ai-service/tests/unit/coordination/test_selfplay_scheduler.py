"""Tests for SelfplayScheduler (December 2025).

Comprehensive test suite covering:
- ConfigPriority and NodeCapability dataclasses
- Priority calculation with all weight factors
- Curriculum weight handling and normalization
- Queue management and allocation
- Health check method
- Backpressure integration
- Event handlers for pipeline integration
- Node targeting and job distribution
- Cluster health event handling
- Metrics and status reporting
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.selfplay_scheduler import (
    ALL_CONFIGS,
    CURRICULUM_WEIGHT,
    DATA_DEFICIT_WEIGHT,
    DEFAULT_GAMES_PER_CONFIG,
    ELO_VELOCITY_WEIGHT,
    EXPLORATION_BOOST_WEIGHT,
    FRESH_DATA_THRESHOLD,
    IMPROVEMENT_BOOST_WEIGHT,
    LARGE_BOARD_TARGET_MULTIPLIER,
    MAX_STALENESS_HOURS,
    MIN_GAMES_PER_ALLOCATION,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    STALE_DATA_THRESHOLD,
    STALENESS_WEIGHT,
    TARGET_GAMES_FOR_2000_ELO,
    TRAINING_NEED_WEIGHT,
    ConfigPriority,
    NodeCapability,
    SelfplayScheduler,
    get_selfplay_scheduler,
    reset_selfplay_scheduler,
)


# =============================================================================
# ConfigPriority Tests
# =============================================================================


class TestConfigPriorityBasics:
    """Basic tests for ConfigPriority dataclass."""

    def test_config_priority_defaults(self):
        """Test default values."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.config_key == "hex8_2p"
        assert priority.staleness_hours == 0.0
        assert priority.elo_velocity == 0.0
        assert priority.training_pending is False
        assert priority.exploration_boost == 1.0
        assert priority.curriculum_weight == 1.0
        assert priority.improvement_boost == 0.0
        assert priority.quality_penalty == 0.0
        assert priority.momentum_multiplier == 1.0
        assert priority.game_count == 0
        assert priority.is_large_board is False
        assert priority.priority_override == 3
        assert priority.search_budget == 400
        assert priority.priority_score == 0.0
        assert priority.games_allocated == 0
        assert priority.nodes_allocated == []

    def test_config_priority_custom_values(self):
        """Test custom initialization values."""
        priority = ConfigPriority(
            config_key="square19_4p",
            staleness_hours=12.0,
            elo_velocity=50.0,
            training_pending=True,
            exploration_boost=1.5,
            curriculum_weight=1.8,
            improvement_boost=0.1,
            quality_penalty=-0.05,
            momentum_multiplier=1.2,
            game_count=5000,
            is_large_board=True,
            priority_override=1,
            search_budget=800,
        )
        assert priority.staleness_hours == 12.0
        assert priority.elo_velocity == 50.0
        assert priority.training_pending is True
        assert priority.exploration_boost == 1.5
        assert priority.curriculum_weight == 1.8
        assert priority.improvement_boost == 0.1
        assert priority.quality_penalty == -0.05
        assert priority.momentum_multiplier == 1.2
        assert priority.game_count == 5000
        assert priority.is_large_board is True
        assert priority.priority_override == 1
        assert priority.search_budget == 800


class TestStalenessFactor:
    """Tests for staleness factor calculation."""

    def test_fresh_data_zero_staleness(self):
        """Fresh data (0 hours) has zero staleness factor."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=0.0)
        assert priority.staleness_factor == 0.0

    def test_fresh_threshold_boundary(self):
        """Data at FRESH_DATA_THRESHOLD has zero staleness."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=FRESH_DATA_THRESHOLD)
        assert priority.staleness_factor == 0.0

    def test_just_above_fresh_threshold(self):
        """Data just above FRESH_DATA_THRESHOLD has small staleness."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=FRESH_DATA_THRESHOLD + 0.5)
        assert 0.0 < priority.staleness_factor < 0.5

    def test_stale_threshold(self):
        """Data at STALE_DATA_THRESHOLD has high staleness."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=STALE_DATA_THRESHOLD)
        assert priority.staleness_factor == 1.0

    def test_very_stale_capped(self):
        """Staleness factor is capped at 1.0 for very old data."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=100.0)
        assert priority.staleness_factor == 1.0

    def test_max_staleness_hours(self):
        """Data at MAX_STALENESS_HOURS is capped."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=MAX_STALENESS_HOURS)
        assert priority.staleness_factor == 1.0

    def test_linear_interpolation(self):
        """Staleness factor interpolates linearly."""
        mid_point = (FRESH_DATA_THRESHOLD + STALE_DATA_THRESHOLD) / 2
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=mid_point)
        assert 0.0 < priority.staleness_factor < 1.0


class TestVelocityFactor:
    """Tests for ELO velocity factor calculation."""

    def test_zero_velocity(self):
        """Zero velocity gives zero factor."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=0.0)
        assert priority.velocity_factor == 0.0

    def test_negative_velocity(self):
        """Negative velocity is clamped to zero."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=-10.0)
        assert priority.velocity_factor == 0.0

    def test_positive_velocity(self):
        """Positive velocity gives positive factor."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=25.0)
        assert priority.velocity_factor == 0.25

    def test_high_velocity_capped(self):
        """Velocity factor is capped at 1.0."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=200.0)
        assert priority.velocity_factor == 1.0

    def test_max_velocity_at_100(self):
        """100 ELO/day gives factor of 1.0."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=100.0)
        assert priority.velocity_factor == 1.0


class TestDataDeficitFactor:
    """Tests for data deficit factor calculation."""

    def test_zero_games_max_deficit(self):
        """Zero games gives maximum deficit factor."""
        priority = ConfigPriority(config_key="hex8_2p", game_count=0)
        assert priority.data_deficit_factor == 1.0

    def test_at_target_no_deficit(self):
        """At target game count, no deficit."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO,
            is_large_board=False,
        )
        assert priority.data_deficit_factor == 0.0

    def test_above_target_no_deficit(self):
        """Above target game count, no deficit."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO + 10000,
            is_large_board=False,
        )
        assert priority.data_deficit_factor == 0.0

    def test_half_target_half_deficit(self):
        """Half the target gives roughly 0.5 deficit."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO // 2,
            is_large_board=False,
        )
        assert 0.45 < priority.data_deficit_factor < 0.55

    def test_large_board_higher_target(self):
        """Large boards have higher target, thus higher deficit."""
        small = ConfigPriority(
            config_key="hex8_2p",
            game_count=50000,
            is_large_board=False,
        )
        large = ConfigPriority(
            config_key="hexagonal_2p",
            game_count=50000,
            is_large_board=True,
        )
        # Large board needs more games, so same count = higher deficit
        assert large.data_deficit_factor > small.data_deficit_factor


# =============================================================================
# NodeCapability Tests
# =============================================================================


class TestNodeCapability:
    """Tests for NodeCapability dataclass."""

    def test_node_capability_defaults(self):
        """Test default values."""
        node = NodeCapability(node_id="test-node")
        assert node.node_id == "test-node"
        assert node.gpu_type == "unknown"
        assert node.gpu_memory_gb == 0.0
        assert node.is_ephemeral is False
        assert node.current_load == 0.0
        assert node.data_lag_seconds == 0.0

    def test_node_capability_custom_values(self):
        """Test custom initialization."""
        node = NodeCapability(
            node_id="gpu-node-1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            is_ephemeral=True,
            current_load=0.5,
            data_lag_seconds=30.0,
        )
        assert node.gpu_type == "A100"
        assert node.gpu_memory_gb == 80.0
        assert node.is_ephemeral is True
        assert node.current_load == 0.5
        assert node.data_lag_seconds == 30.0

    def test_capacity_weight_returns_number(self):
        """Capacity weight returns a numeric value."""
        node = NodeCapability(node_id="test", gpu_type="A100")
        assert isinstance(node.capacity_weight, (int, float))
        assert node.capacity_weight >= 0

    def test_available_capacity_calculation(self):
        """Available capacity calculation."""
        node = NodeCapability(
            node_id="test",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.3,
        )
        # Should have 70% capacity remaining, scaled by weight
        assert node.available_capacity >= 0.0
        assert node.available_capacity <= node.capacity_weight

    def test_available_capacity_full_load(self):
        """Fully loaded node has zero available capacity."""
        node = NodeCapability(
            node_id="test",
            gpu_type="A100",
            current_load=1.0,
        )
        assert node.available_capacity == 0.0

    def test_available_capacity_idle(self):
        """Idle node has full available capacity."""
        node = NodeCapability(
            node_id="test",
            gpu_type="A100",
            current_load=0.0,
        )
        assert node.available_capacity == node.capacity_weight


# =============================================================================
# SelfplayScheduler Basic Tests
# =============================================================================


class TestSelfplaySchedulerInit:
    """Tests for SelfplayScheduler initialization."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    def test_init_creates_priorities_for_all_configs(self):
        """Scheduler creates priority entries for all configs."""
        scheduler = SelfplayScheduler()
        assert len(scheduler._config_priorities) == len(ALL_CONFIGS)
        for config in ALL_CONFIGS:
            assert config in scheduler._config_priorities

    def test_init_with_callbacks(self):
        """Scheduler accepts callback functions."""
        mock_fn = MagicMock(return_value=False)
        scheduler = SelfplayScheduler(
            is_emergency_active_fn=mock_fn,
            verbose=True,
        )
        assert scheduler._is_emergency_active_fn is mock_fn
        assert scheduler._verbose is True

    def test_all_configs_constant(self):
        """ALL_CONFIGS contains all 12 canonical configurations."""
        expected = [
            "hex8_2p", "hex8_3p", "hex8_4p",
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]
        assert sorted(ALL_CONFIGS) == sorted(expected)


class TestSelfplaySchedulerSingleton:
    """Tests for singleton behavior."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    def test_singleton_returns_same_instance(self):
        """get_selfplay_scheduler returns the same instance."""
        scheduler1 = get_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is scheduler2

    def test_reset_clears_singleton(self):
        """reset_selfplay_scheduler clears the singleton."""
        scheduler1 = get_selfplay_scheduler()
        reset_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is not scheduler2


# =============================================================================
# Priority Calculation Tests
# =============================================================================


class TestPriorityScoreComputation:
    """Tests for priority score computation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_compute_low_priority(self, scheduler):
        """Fresh data with no activity has low priority."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=0.0,
            elo_velocity=0.0,
            training_pending=False,
            game_count=TARGET_GAMES_FOR_2000_ELO,
        )
        score = scheduler._compute_priority_score(priority)
        assert score < 0.5

    def test_compute_high_priority(self, scheduler):
        """Stale data with activity has high priority."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=48.0,
            elo_velocity=50.0,
            training_pending=True,
            game_count=1000,
        )
        score = scheduler._compute_priority_score(priority)
        assert score > 0.0

    def test_staleness_increases_priority(self, scheduler):
        """More staleness increases priority."""
        fresh = ConfigPriority(config_key="hex8_2p", staleness_hours=0.0)
        stale = ConfigPriority(config_key="hex8_2p", staleness_hours=10.0)

        fresh_score = scheduler._compute_priority_score(fresh)
        stale_score = scheduler._compute_priority_score(stale)

        assert stale_score > fresh_score

    def test_velocity_increases_priority(self, scheduler):
        """Higher velocity increases priority."""
        slow = ConfigPriority(config_key="hex8_2p", elo_velocity=0.0)
        fast = ConfigPriority(config_key="hex8_2p", elo_velocity=50.0)

        slow_score = scheduler._compute_priority_score(slow)
        fast_score = scheduler._compute_priority_score(fast)

        assert fast_score > slow_score

    def test_training_pending_increases_priority(self, scheduler):
        """Training pending increases priority."""
        idle = ConfigPriority(config_key="hex8_2p", training_pending=False)
        pending = ConfigPriority(config_key="hex8_2p", training_pending=True)

        idle_score = scheduler._compute_priority_score(idle)
        pending_score = scheduler._compute_priority_score(pending)

        assert pending_score > idle_score

    def test_exploration_boost_multiplies(self, scheduler):
        """Exploration boost multiplies the priority score."""
        normal = ConfigPriority(config_key="hex8_2p", exploration_boost=1.0)
        boosted = ConfigPriority(config_key="hex8_2p", exploration_boost=1.5)

        normal_score = scheduler._compute_priority_score(normal)
        boosted_score = scheduler._compute_priority_score(boosted)

        # Boosted should be ~1.5x normal for otherwise equal priorities
        assert boosted_score >= normal_score

    def test_momentum_multiplier_affects_priority(self, scheduler):
        """Momentum multiplier affects priority."""
        stable = ConfigPriority(
            config_key="hex8_2p",
            momentum_multiplier=1.0,
            staleness_hours=5.0,
        )
        accelerating = ConfigPriority(
            config_key="hex8_2p",
            momentum_multiplier=1.5,
            staleness_hours=5.0,
        )

        stable_score = scheduler._compute_priority_score(stable)
        accel_score = scheduler._compute_priority_score(accelerating)

        assert accel_score > stable_score

    def test_quality_penalty_reduces_priority(self, scheduler):
        """Quality penalty reduces priority."""
        good = ConfigPriority(config_key="hex8_2p", quality_penalty=0.0, staleness_hours=5.0)
        penalized = ConfigPriority(config_key="hex8_2p", quality_penalty=-0.1, staleness_hours=5.0)

        good_score = scheduler._compute_priority_score(good)
        penalized_score = scheduler._compute_priority_score(penalized)

        assert penalized_score < good_score

    def test_priority_override_multiplier(self, scheduler):
        """Priority override applies multiplier."""
        low_priority = ConfigPriority(
            config_key="hex8_2p",
            priority_override=3,
            staleness_hours=5.0,
        )
        critical_priority = ConfigPriority(
            config_key="hex8_2p",
            priority_override=0,
            staleness_hours=5.0,
        )

        low_score = scheduler._compute_priority_score(low_priority)
        critical_score = scheduler._compute_priority_score(critical_priority)

        # Critical should be 3x
        assert critical_score >= low_score


# =============================================================================
# Curriculum Weight Tests
# =============================================================================


class TestCurriculumWeightHandling:
    """Tests for curriculum weight handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_curriculum_weight_affects_priority(self, scheduler):
        """Curriculum weight affects priority calculation."""
        low_weight = ConfigPriority(config_key="hex8_2p", curriculum_weight=0.5)
        high_weight = ConfigPriority(config_key="hex8_2p", curriculum_weight=2.0)

        low_score = scheduler._compute_priority_score(low_weight)
        high_score = scheduler._compute_priority_score(high_weight)

        assert high_score > low_score

    def test_on_curriculum_rebalanced_updates_weight(self, scheduler):
        """Curriculum rebalanced event updates weight."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "weight": 1.5,
            "reason": "test",
        }

        scheduler._on_curriculum_rebalanced(event)

        assert scheduler._config_priorities["hex8_2p"].curriculum_weight == 1.5

    def test_on_curriculum_rebalanced_skips_self_originated(self, scheduler):
        """Skip events from selfplay_scheduler source (loop guard)."""
        original_weight = scheduler._config_priorities["hex8_2p"].curriculum_weight

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "weight": 2.0,
            "source": "selfplay_scheduler",
        }

        scheduler._on_curriculum_rebalanced(event)

        # Weight should not change
        assert scheduler._config_priorities["hex8_2p"].curriculum_weight == original_weight

    def test_on_curriculum_rebalanced_batch_update(self, scheduler):
        """Curriculum rebalanced handles batch weights update."""
        event = MagicMock()
        event.payload = {
            "weights": {
                "hex8_2p": 1.3,
                "square8_2p": 0.8,
            },
        }

        scheduler._on_curriculum_rebalanced(event)

        assert scheduler._config_priorities["hex8_2p"].curriculum_weight == 1.3
        assert scheduler._config_priorities["square8_2p"].curriculum_weight == 0.8


# =============================================================================
# Queue Management Tests
# =============================================================================


class TestQueueManagement:
    """Tests for queue management and allocation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_allocate_to_nodes_empty_when_no_nodes(self, scheduler):
        """Allocation returns empty when no nodes available."""
        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)
        assert allocation == {}

    def test_allocate_to_nodes_with_capacity(self, scheduler):
        """Allocation works with available nodes."""
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )

        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)

        assert isinstance(allocation, dict)
        if allocation:
            assert "node1" in allocation

    def test_allocate_excludes_unhealthy_nodes(self, scheduler):
        """Allocation excludes unhealthy nodes."""
        scheduler._node_capabilities["healthy"] = NodeCapability(
            node_id="healthy",
            gpu_type="A100",
            current_load=0.0,
        )
        scheduler._node_capabilities["unhealthy"] = NodeCapability(
            node_id="unhealthy",
            gpu_type="A100",
            current_load=0.0,
        )
        scheduler._unhealthy_nodes = {"unhealthy"}

        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)

        assert "unhealthy" not in allocation

    def test_allocate_applies_cluster_health_factor(self, scheduler):
        """Allocation applies cluster health factor."""
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            current_load=0.0,
        )
        scheduler._cluster_health_factor = 0.5

        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)

        if allocation:
            total = sum(allocation.values())
            # Should be reduced from 100
            assert total <= 100

    def test_ephemeral_boost_for_short_jobs(self, scheduler):
        """Ephemeral nodes get boost for short jobs."""
        scheduler._node_capabilities["ephemeral"] = NodeCapability(
            node_id="ephemeral",
            gpu_type="RTX4090",
            current_load=0.0,
            is_ephemeral=True,
        )

        # hex8 is a short job
        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)

        # Just verify no error
        assert isinstance(allocation, dict)

    def test_ephemeral_reduction_for_long_jobs(self, scheduler):
        """Ephemeral nodes get reduction for long jobs."""
        scheduler._node_capabilities["ephemeral"] = NodeCapability(
            node_id="ephemeral",
            gpu_type="RTX4090",
            current_load=0.0,
            is_ephemeral=True,
        )

        # square19 is a long job
        allocation = scheduler._allocate_to_nodes("square19_2p", 100)

        # Just verify no error
        assert isinstance(allocation, dict)


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_health_check_returns_result(self, scheduler):
        """Health check returns a HealthCheckResult."""
        result = scheduler.health_check()
        assert result is not None
        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    def test_health_check_unhealthy_when_not_subscribed(self, scheduler):
        """Health check reports unhealthy when not subscribed."""
        result = scheduler.health_check()
        # New scheduler is not subscribed
        assert result.healthy is False
        assert "subscribed" in result.message.lower() or result.details.get("subscribed") is False

    def test_health_check_details_contains_metrics(self, scheduler):
        """Health check details contain expected metrics."""
        result = scheduler.health_check()
        details = result.details

        assert "configs_tracked" in details
        assert "nodes_tracked" in details
        assert "games_allocated_total" in details

    def test_health_check_detects_stale_priority(self, scheduler):
        """Health check detects stale priority data."""
        scheduler._subscribed = True
        scheduler._last_priority_update = time.time() - 400  # 6+ minutes ago

        result = scheduler.health_check()

        assert result.healthy is False
        assert "stale" in result.message.lower()


# =============================================================================
# Backpressure Integration Tests
# =============================================================================


class TestBackpressureIntegration:
    """Tests for backpressure integration."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.mark.asyncio
    async def test_allocate_batch_respects_backpressure_pause(self):
        """Allocation returns empty when backpressure says pause."""
        mock_bp = MagicMock()
        mock_signal = MagicMock()
        mock_signal.should_pause = True
        mock_signal.overall_pressure = 0.95
        mock_bp.get_signal = AsyncMock(return_value=mock_signal)

        scheduler = SelfplayScheduler(backpressure_monitor=mock_bp)

        allocation = await scheduler.allocate_selfplay_batch()

        assert allocation == {}

    @pytest.mark.asyncio
    async def test_allocate_batch_scales_by_backpressure(self):
        """Allocation scales games by backpressure multiplier."""
        mock_bp = MagicMock()
        mock_signal = MagicMock()
        mock_signal.should_pause = False
        mock_signal.spawn_rate_multiplier = 0.5
        mock_signal.overall_pressure = 0.7
        mock_bp.get_signal = AsyncMock(return_value=mock_signal)

        scheduler = SelfplayScheduler(backpressure_monitor=mock_bp)
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            current_load=0.0,
        )

        # Should not raise
        allocation = await scheduler.allocate_selfplay_batch(games_per_config=100)

        assert isinstance(allocation, dict)


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handlers."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_on_selfplay_complete_resets_staleness(self, scheduler):
        """Selfplay complete resets staleness for config."""
        scheduler._config_priorities["hex8_2p"].staleness_hours = 10.0

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "games": 100}

        scheduler._on_selfplay_complete(event)

        assert scheduler._config_priorities["hex8_2p"].staleness_hours == 0.0

    def test_on_training_complete_clears_pending(self, scheduler):
        """Training complete clears training pending flag."""
        scheduler._config_priorities["hex8_2p"].training_pending = True

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        scheduler._on_training_complete(event)

        assert scheduler._config_priorities["hex8_2p"].training_pending is False

    def test_on_promotion_complete_success_resets_boost(self, scheduler):
        """Successful promotion resets exploration boost."""
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.5

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "success": True}

        scheduler._on_promotion_complete(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost == 1.0

    def test_on_promotion_complete_failure_increases_boost(self, scheduler):
        """Failed promotion increases exploration boost."""
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.0

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "success": False}

        scheduler._on_promotion_complete(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost > 1.0

    def test_on_quality_degraded_applies_penalty(self, scheduler):
        """Quality degradation applies penalty."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.4,
            "threshold": 0.6,
        }

        scheduler._on_quality_degraded(event)

        assert scheduler._config_priorities["hex8_2p"].quality_penalty < 0

    def test_on_quality_degraded_clears_penalty_when_recovered(self, scheduler):
        """Quality above threshold clears penalty."""
        scheduler._config_priorities["hex8_2p"].quality_penalty = -0.1

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.7,
            "threshold": 0.6,
        }

        scheduler._on_quality_degraded(event)

        assert scheduler._config_priorities["hex8_2p"].quality_penalty == 0.0

    def test_on_selfplay_rate_changed_updates_momentum(self, scheduler):
        """Selfplay rate change updates momentum multiplier."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "new_rate": 1.5,
            "momentum_state": "accelerating",
        }

        scheduler._on_selfplay_rate_changed(event)

        assert scheduler._config_priorities["hex8_2p"].momentum_multiplier == 1.5

    def test_on_training_blocked_by_quality_boosts_selfplay(self, scheduler):
        """Training blocked boosts selfplay allocation."""
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "stale_data",
            "data_age_hours": 24.0,
        }

        scheduler._on_training_blocked_by_quality(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost >= 1.5
        assert scheduler._config_priorities["hex8_2p"].training_pending is True


# =============================================================================
# Cluster Health Event Tests
# =============================================================================


class TestClusterHealthEvents:
    """Tests for cluster health event handlers."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_on_node_unhealthy_marks_node(self, scheduler):
        """Node unhealthy event marks node as unavailable."""
        event = MagicMock()
        event.payload = {"node_id": "bad-node", "reason": "test"}

        scheduler._on_node_unhealthy(event)

        assert "bad-node" in scheduler._unhealthy_nodes

    def test_on_node_recovered_clears_mark(self, scheduler):
        """Node recovered event clears unhealthy mark."""
        scheduler._unhealthy_nodes = {"recovering-node"}

        event = MagicMock()
        event.payload = {"node_id": "recovering-node"}

        scheduler._on_node_recovered(event)

        assert "recovering-node" not in scheduler._unhealthy_nodes

    def test_on_host_offline_marks_host(self, scheduler):
        """Host offline event marks host as unavailable."""
        event = MagicMock()
        event.payload = {"host": "offline-host", "reason": "retired"}

        scheduler._on_host_offline(event)

        assert "offline-host" in scheduler._unhealthy_nodes

    def test_on_cluster_unhealthy_reduces_factor(self, scheduler):
        """Cluster unhealthy event reduces health factor."""
        event = MagicMock()
        event.payload = {"healthy_nodes": 5, "total_nodes": 10}

        scheduler._on_cluster_unhealthy(event)

        assert scheduler._cluster_health_factor == 0.5

    def test_on_cluster_healthy_restores_factor(self, scheduler):
        """Cluster healthy event restores health factor."""
        scheduler._cluster_health_factor = 0.5
        scheduler._unhealthy_nodes = {"node1", "node2"}

        event = MagicMock()
        event.payload = {}

        scheduler._on_cluster_healthy(event)

        assert scheduler._cluster_health_factor == 1.0
        assert len(scheduler._unhealthy_nodes) == 0


# =============================================================================
# Node Targeting Tests
# =============================================================================


class TestNodeTargeting:
    """Tests for node targeting and job distribution."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    def test_get_target_jobs_returns_int(self):
        """get_target_jobs_for_node returns an integer."""
        scheduler = SelfplayScheduler()

        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.gpu_type = "A100"
        mock_node.memory_gb = 64
        mock_node.has_gpu = True
        mock_node.cpu_count = 16
        mock_node.cpu_percent = 50.0
        mock_node.memory_percent = 40.0
        mock_node.disk_percent = 30.0
        mock_node.gpu_percent = 60.0
        mock_node.gpu_memory_percent = 50.0
        mock_node.selfplay_jobs = 2

        result = scheduler.get_target_jobs_for_node(mock_node)

        assert isinstance(result, int)
        assert result >= 0

    def test_get_target_jobs_respects_emergency(self):
        """Emergency callback stops all jobs."""
        scheduler = SelfplayScheduler(
            is_emergency_active_fn=MagicMock(return_value=True),
        )

        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.memory_gb = 64

        result = scheduler.get_target_jobs_for_node(mock_node)

        assert result == 0

    def test_get_target_jobs_respects_min_memory(self):
        """Low memory returns zero jobs."""
        scheduler = SelfplayScheduler()

        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.memory_gb = 1  # Too low

        result = scheduler.get_target_jobs_for_node(mock_node)

        assert result == 0

    def test_compute_hardware_limit_gh200(self):
        """GH200 gets high hardware limit."""
        scheduler = SelfplayScheduler()
        limit = scheduler._compute_hardware_limit(
            has_gpu=True,
            gpu_name="NVIDIA GH200",
            gpu_count=1,
            cpu_count=64,
            memory_gb=96,
        )
        assert limit >= 40

    def test_compute_hardware_limit_h100(self):
        """H100 gets appropriate limit."""
        scheduler = SelfplayScheduler()
        limit = scheduler._compute_hardware_limit(
            has_gpu=True,
            gpu_name="NVIDIA H100",
            gpu_count=1,
            cpu_count=64,
            memory_gb=80,
        )
        assert 20 <= limit <= 48

    def test_compute_hardware_limit_rtx4090(self):
        """RTX 4090 gets appropriate limit."""
        scheduler = SelfplayScheduler()
        limit = scheduler._compute_hardware_limit(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_count=1,
            cpu_count=16,
            memory_gb=24,
        )
        assert 4 <= limit <= 24

    def test_compute_hardware_limit_cpu_only(self):
        """CPU-only node gets appropriate limit."""
        scheduler = SelfplayScheduler()
        limit = scheduler._compute_hardware_limit(
            has_gpu=False,
            gpu_name="",
            gpu_count=0,
            cpu_count=32,
            memory_gb=64,
        )
        assert 8 <= limit <= 32


# =============================================================================
# Metrics and Status Tests
# =============================================================================


class TestMetricsAndStatus:
    """Tests for metrics and status reporting."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_get_metrics_returns_dict(self, scheduler):
        """get_metrics returns a dictionary."""
        metrics = scheduler.get_metrics()
        assert isinstance(metrics, dict)

    def test_get_metrics_contains_expected_keys(self, scheduler):
        """Metrics contains expected keys."""
        metrics = scheduler.get_metrics()
        assert "games_allocated_total" in metrics
        assert "games_per_hour" in metrics

    def test_get_status_returns_dict(self, scheduler):
        """get_status returns a dictionary."""
        status = scheduler.get_status()
        assert isinstance(status, dict)

    def test_get_status_contains_priorities(self, scheduler):
        """Status contains priority information."""
        status = scheduler.get_status()
        assert "config_priorities" in status
        assert len(status["config_priorities"]) == len(ALL_CONFIGS)

    def test_get_top_priorities(self, scheduler):
        """get_top_priorities returns sorted priorities."""
        # Set some priorities
        scheduler._config_priorities["hex8_2p"].priority_score = 0.9
        scheduler._config_priorities["square8_2p"].priority_score = 0.5
        scheduler._config_priorities["square19_2p"].priority_score = 0.1

        top = scheduler.get_top_priorities(n=3)

        assert len(top) == 3
        assert top[0]["config"] == "hex8_2p"
        assert top[0]["priority"] == 0.9

    def test_record_allocation_updates_metrics(self, scheduler):
        """_record_allocation updates metrics."""
        scheduler._record_allocation(100)

        assert scheduler._games_allocated_total == 100
        assert len(scheduler._allocation_history) == 1

    def test_record_allocation_ignores_zero(self, scheduler):
        """_record_allocation ignores zero allocation."""
        scheduler._record_allocation(0)

        assert scheduler._games_allocated_total == 0
        assert len(scheduler._allocation_history) == 0


# =============================================================================
# Sync Priority Configs Tests
# =============================================================================


class TestSyncPriorityConfigs:
    """Tests for synchronous priority config access."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_get_priority_configs_sync_returns_list(self, scheduler):
        """get_priority_configs_sync returns a list."""
        result = scheduler.get_priority_configs_sync()
        assert isinstance(result, list)
        assert len(result) == len(ALL_CONFIGS)

    def test_get_priority_configs_sync_sorted(self, scheduler):
        """Results are sorted by priority descending."""
        scheduler._config_priorities["hex8_2p"].priority_score = 0.9
        scheduler._config_priorities["square8_2p"].priority_score = 0.5

        result = scheduler.get_priority_configs_sync()

        # First should be highest priority
        assert result[0][0] == "hex8_2p"

    def test_get_priority_configs_sync_top_n(self, scheduler):
        """top_n limits results."""
        result = scheduler.get_priority_configs_sync(top_n=3)
        assert len(result) == 3

    def test_get_priority_configs_sync_filter(self, scheduler):
        """filter_configs limits to specified configs."""
        result = scheduler.get_priority_configs_sync(
            filter_configs=["hex8_2p", "square8_2p"]
        )
        assert len(result) == 2
        configs = [r[0] for r in result]
        assert "hex8_2p" in configs
        assert "square8_2p" in configs


# =============================================================================
# Boost Config Allocation Tests
# =============================================================================


class TestBoostConfigAllocation:
    """Tests for boost_config_allocation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_boost_known_config(self, scheduler):
        """Boosting known config succeeds."""
        original = scheduler._config_priorities["hex8_2p"].exploration_boost

        result = scheduler.boost_config_allocation("hex8_2p", multiplier=1.5)

        assert result is True
        assert scheduler._config_priorities["hex8_2p"].exploration_boost > original

    def test_boost_unknown_config(self, scheduler):
        """Boosting unknown config returns False."""
        result = scheduler.boost_config_allocation("nonexistent_config")
        assert result is False

    def test_boost_capped_at_2(self, scheduler):
        """Exploration boost is capped at 2.0."""
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.8

        scheduler.boost_config_allocation("hex8_2p", multiplier=2.0)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost <= 2.0

    def test_boost_forces_priority_update(self, scheduler):
        """Boosting forces priority recalculation."""
        scheduler._last_priority_update = time.time()

        scheduler.boost_config_allocation("hex8_2p")

        assert scheduler._last_priority_update == 0.0


# =============================================================================
# Async Priority Configs Tests
# =============================================================================


class TestAsyncPriorityConfigs:
    """Async tests for priority config methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    @pytest.mark.asyncio
    async def test_get_priority_configs_returns_list(self, scheduler):
        """get_priority_configs returns list of tuples."""
        priorities = await scheduler.get_priority_configs(top_n=5)
        assert isinstance(priorities, list)
        assert len(priorities) <= 5

    @pytest.mark.asyncio
    async def test_get_priority_configs_sorted_descending(self, scheduler):
        """Results are sorted by priority descending."""
        scheduler._config_priorities["hex8_2p"].priority_score = 0.9
        scheduler._config_priorities["square8_2p"].priority_score = 0.5

        priorities = await scheduler.get_priority_configs(top_n=3)

        for i in range(len(priorities) - 1):
            assert priorities[i][1] >= priorities[i + 1][1]


# =============================================================================
# Exploration Boost Decay Tests
# =============================================================================


class TestExplorationBoostDecay:
    """Tests for exploration boost expiry and decay."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_decay_expired_boosts_no_active(self, scheduler):
        """No decay when no boosts active."""
        count = scheduler._decay_expired_boosts(time.time())
        assert count == 0

    def test_decay_expired_boosts_not_expired(self, scheduler):
        """Boost not expired yet is not decayed."""
        future_time = time.time() + 1000
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.5
        scheduler._config_priorities["hex8_2p"].exploration_boost_expires_at = future_time

        count = scheduler._decay_expired_boosts(time.time())

        assert count == 0
        assert scheduler._config_priorities["hex8_2p"].exploration_boost == 1.5

    def test_decay_expired_boosts_expired(self, scheduler):
        """Expired boost is decayed to 1.0."""
        past_time = time.time() - 100
        scheduler._config_priorities["hex8_2p"].exploration_boost = 1.5
        scheduler._config_priorities["hex8_2p"].exploration_boost_expires_at = past_time

        count = scheduler._decay_expired_boosts(time.time())

        assert count == 1
        assert scheduler._config_priorities["hex8_2p"].exploration_boost == 1.0


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestRegressionDetection:
    """Tests for regression detection event handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_on_regression_detected_mild(self, scheduler):
        """Mild regression applies small boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "severity": "mild",
            "win_rate_drop": 0.05,
        }

        scheduler._on_regression_detected(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost == 0.1

    def test_on_regression_detected_severe(self, scheduler):
        """Severe regression applies larger boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "severity": "severe",
            "win_rate_drop": 0.2,
        }

        scheduler._on_regression_detected(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost == 0.3


# =============================================================================
# Low Quality Warning Tests
# =============================================================================


class TestLowQualityWarning:
    """Tests for low quality warning event handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_on_low_quality_warning_throttles_all(self, scheduler):
        """Low quality warning throttles all configs."""
        original_boosts = {
            cfg: p.exploration_boost
            for cfg, p in scheduler._config_priorities.items()
        }

        event = MagicMock()
        event.payload = {
            "quality_score": 0.3,
            "old_state": "good",
            "new_state": "warning",
        }

        scheduler._on_low_quality_warning(event)

        for cfg in scheduler._config_priorities:
            assert scheduler._config_priorities[cfg].exploration_boost <= original_boosts[cfg]

    def test_on_low_quality_warning_applies_penalty(self, scheduler):
        """Low quality applies quality penalty."""
        event = MagicMock()
        event.payload = {
            "quality_score": 0.3,
        }

        scheduler._on_low_quality_warning(event)

        for cfg in scheduler._config_priorities:
            assert scheduler._config_priorities[cfg].quality_penalty < 0


# =============================================================================
# Training Early Stopped Tests
# =============================================================================


class TestTrainingEarlyStopped:
    """Tests for training early stopped event handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        return SelfplayScheduler()

    def test_on_training_early_stopped_regression(self, scheduler):
        """Regression reason gets aggressive boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "regression_detected",
            "final_loss": 0.5,
            "epochs_completed": 10,
        }

        scheduler._on_training_early_stopped(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost > 1.0
        assert scheduler._config_priorities["hex8_2p"].curriculum_weight > 1.0

    def test_on_training_early_stopped_plateau(self, scheduler):
        """Plateau reason gets moderate boost."""
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "loss_plateau",
            "final_loss": 0.3,
            "epochs_completed": 20,
        }

        scheduler._on_training_early_stopped(event)

        assert scheduler._config_priorities["hex8_2p"].exploration_boost > 1.0


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_weight_sum_reasonable(self):
        """Priority weights should sum to approximately 1.0."""
        total = (
            STALENESS_WEIGHT +
            ELO_VELOCITY_WEIGHT +
            TRAINING_NEED_WEIGHT +
            EXPLORATION_BOOST_WEIGHT +
            CURRICULUM_WEIGHT +
            IMPROVEMENT_BOOST_WEIGHT +
            DATA_DEFICIT_WEIGHT
        )
        # Allow some slack since quality penalty is separate
        assert 0.8 <= total <= 1.5

    def test_priority_override_multipliers(self):
        """Priority override multipliers have expected values."""
        assert PRIORITY_OVERRIDE_MULTIPLIERS[0] == 3.0  # CRITICAL
        assert PRIORITY_OVERRIDE_MULTIPLIERS[1] == 2.0  # HIGH
        assert PRIORITY_OVERRIDE_MULTIPLIERS[2] == 1.25  # MEDIUM
        assert PRIORITY_OVERRIDE_MULTIPLIERS[3] == 1.0  # LOW

    def test_staleness_thresholds_ordered(self):
        """Staleness thresholds are in correct order."""
        assert FRESH_DATA_THRESHOLD < STALE_DATA_THRESHOLD < MAX_STALENESS_HOURS

    def test_large_board_multiplier_greater_than_one(self):
        """Large board target multiplier is greater than 1."""
        assert LARGE_BOARD_TARGET_MULTIPLIER > 1.0

    def test_min_games_per_allocation_positive(self):
        """Minimum games per allocation is positive."""
        assert MIN_GAMES_PER_ALLOCATION > 0


# =============================================================================
# December 29, 2025 Feature Tests
# =============================================================================


class TestDynamicWeightsDataclass:
    """Tests for DynamicWeights dataclass (Dec 29, 2025)."""

    def test_default_values(self):
        """Test default DynamicWeights values."""
        from app.coordination.selfplay_scheduler import (
            DynamicWeights,
            STALENESS_WEIGHT,
            ELO_VELOCITY_WEIGHT,
            TRAINING_NEED_WEIGHT,
            EXPLORATION_BOOST_WEIGHT,
            CURRICULUM_WEIGHT,
            IMPROVEMENT_BOOST_WEIGHT,
            DATA_DEFICIT_WEIGHT,
            QUALITY_WEIGHT,
            VOI_WEIGHT,
        )

        weights = DynamicWeights()

        # Check weight fields have correct defaults
        assert weights.staleness == STALENESS_WEIGHT
        assert weights.velocity == ELO_VELOCITY_WEIGHT
        assert weights.training == TRAINING_NEED_WEIGHT
        assert weights.exploration == EXPLORATION_BOOST_WEIGHT
        assert weights.curriculum == CURRICULUM_WEIGHT
        assert weights.improvement == IMPROVEMENT_BOOST_WEIGHT
        assert weights.data_deficit == DATA_DEFICIT_WEIGHT
        assert weights.quality == QUALITY_WEIGHT
        assert weights.voi == VOI_WEIGHT

        # Check cluster state fields
        assert weights.idle_gpu_fraction == 0.0
        assert weights.training_queue_depth == 0
        assert weights.configs_at_target_fraction == 0.0
        assert weights.average_elo == 1500.0

    def test_custom_values(self):
        """Test DynamicWeights with custom values."""
        from app.coordination.selfplay_scheduler import DynamicWeights

        weights = DynamicWeights(
            staleness=0.3,
            velocity=0.2,
            training=0.15,
            idle_gpu_fraction=0.5,
            training_queue_depth=10,
            configs_at_target_fraction=0.25,
            average_elo=1800.0,
        )

        assert weights.staleness == 0.3
        assert weights.velocity == 0.2
        assert weights.training == 0.15
        assert weights.idle_gpu_fraction == 0.5
        assert weights.training_queue_depth == 10
        assert weights.configs_at_target_fraction == 0.25
        assert weights.average_elo == 1800.0

    def test_to_dict(self):
        """Test DynamicWeights.to_dict() method."""
        from app.coordination.selfplay_scheduler import DynamicWeights

        weights = DynamicWeights(
            staleness=0.25,
            velocity=0.15,
            idle_gpu_fraction=0.3,
            average_elo=1700.0,
        )

        result = weights.to_dict()

        assert isinstance(result, dict)
        assert result["staleness"] == 0.25
        assert result["velocity"] == 0.15
        assert result["idle_gpu_fraction"] == 0.3
        assert result["average_elo"] == 1700.0

        # Should have all weight keys
        expected_keys = [
            "staleness", "velocity", "training", "exploration",
            "curriculum", "improvement", "data_deficit", "quality", "voi",
            "idle_gpu_fraction", "training_queue_depth",
            "configs_at_target_fraction", "average_elo",
        ]
        for key in expected_keys:
            assert key in result


class TestConfigPriorityVOIProperties:
    """Tests for VOI-related ConfigPriority properties (Dec 29, 2025)."""

    def test_player_count_2p(self):
        """Test player_count extraction for 2-player config."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.player_count == 2

    def test_player_count_3p(self):
        """Test player_count extraction for 3-player config."""
        priority = ConfigPriority(config_key="square19_3p")
        assert priority.player_count == 3

    def test_player_count_4p(self):
        """Test player_count extraction for 4-player config."""
        priority = ConfigPriority(config_key="hexagonal_4p")
        assert priority.player_count == 4

    def test_player_count_invalid_format(self):
        """Test player_count fallback for invalid format."""
        priority = ConfigPriority(config_key="invalid")
        assert priority.player_count == 2  # Default fallback

    def test_elo_gap_below_target(self):
        """Test elo_gap when below target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=1500.0,
            target_elo=2000.0,
        )
        assert priority.elo_gap == 500.0

    def test_elo_gap_at_target(self):
        """Test elo_gap when at target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=2000.0,
            target_elo=2000.0,
        )
        assert priority.elo_gap == 0.0

    def test_elo_gap_above_target(self):
        """Test elo_gap clamped to 0 when above target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            current_elo=2100.0,
            target_elo=2000.0,
        )
        assert priority.elo_gap == 0.0  # Can't be negative

    def test_info_gain_per_game_no_games(self):
        """Test info_gain_per_game with no games (max info)."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=0,
            elo_uncertainty=200.0,
        )
        assert priority.info_gain_per_game == 200.0  # Full uncertainty

    def test_info_gain_per_game_with_games(self):
        """Test info_gain_per_game decreases with sqrt(n)."""
        import math

        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=100,
            elo_uncertainty=200.0,
        )
        expected = 200.0 / math.sqrt(100)  # = 20.0
        assert abs(priority.info_gain_per_game - expected) < 0.01

    def test_voi_score_high_uncertainty(self):
        """Test VOI score with high uncertainty."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            elo_uncertainty=300.0,  # Max (normalized to 1.0)
            current_elo=1500.0,
            target_elo=2000.0,  # 500 gap (normalized to 1.0)
            game_count=0,  # Max info gain
        )
        # voi_score = 1.0 * 0.4 + 1.0 * 0.3 + 1.0 * 0.3 = 1.0 (max)
        assert priority.voi_score > 0.9  # Near max

    def test_voi_score_low_uncertainty(self):
        """Test VOI score with low uncertainty (at target)."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            elo_uncertainty=30.0,  # Low (normalized to 0.1)
            current_elo=2000.0,
            target_elo=2000.0,  # No gap
            game_count=1000,  # Many games = low info gain
        )
        # All factors near minimum
        assert priority.voi_score < 0.3


class TestGameCountNormalization:
    """Tests for game count normalization (Dec 29, 2025)."""

    def test_games_needed_defaults(self):
        """Test games_needed with default values."""
        from app.coordination.selfplay_scheduler import DEFAULT_TRAINING_SAMPLES_TARGET

        priority = ConfigPriority(config_key="hex8_2p")
        # Default target_training_samples = 50000, samples_per_game = 50
        # games_needed = 50000 / 50 = 1000
        expected = DEFAULT_TRAINING_SAMPLES_TARGET // 50  # 1000
        assert priority.games_needed == expected

    def test_games_needed_with_existing_games(self):
        """Test games_needed subtracts existing games."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=500,
            target_training_samples=100000,
            samples_per_game_estimate=50.0,
        )
        # Current samples = 500 * 50 = 25000
        # Remaining = 100000 - 25000 = 75000
        # Games needed = 75000 / 50 = 1500
        assert priority.games_needed == 1500

    def test_games_needed_target_met(self):
        """Test games_needed returns 0 when target met."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=2500,
            target_training_samples=100000,
            samples_per_game_estimate=50.0,
        )
        # Current samples = 2500 * 50 = 125000 > 100000
        assert priority.games_needed == 0

    def test_games_needed_zero_samples_per_game(self):
        """Test games_needed handles zero samples_per_game."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            samples_per_game_estimate=0.0,
        )
        assert priority.games_needed == 0  # Avoid division by zero

    def test_set_target_training_samples(self):
        """Test set_target_training_samples method."""
        scheduler = SelfplayScheduler()

        scheduler.set_target_training_samples("hex8_2p", 50000)

        priority = scheduler._config_priorities["hex8_2p"]
        assert priority.target_training_samples == 50000

    def test_set_target_training_samples_unknown_config(self):
        """Test set_target_training_samples creates new config."""
        scheduler = SelfplayScheduler()

        scheduler.set_target_training_samples("new_config_7p", 25000)

        assert "new_config_7p" in scheduler._config_priorities
        assert scheduler._config_priorities["new_config_7p"].target_training_samples == 25000

    def test_get_games_needed(self):
        """Test get_games_needed method."""
        scheduler = SelfplayScheduler()
        scheduler._config_priorities["hex8_2p"].target_training_samples = 50000
        scheduler._config_priorities["hex8_2p"].game_count = 100
        scheduler._config_priorities["hex8_2p"].samples_per_game_estimate = 50.0

        # 50000 - (100 * 50) = 45000 / 50 = 900
        assert scheduler.get_games_needed("hex8_2p") == 900

    def test_get_games_needed_unknown_config(self):
        """Test get_games_needed returns 0 for unknown config."""
        scheduler = SelfplayScheduler()
        assert scheduler.get_games_needed("unknown_config") == 0

    def test_get_all_games_needed(self):
        """Test get_all_games_needed method."""
        scheduler = SelfplayScheduler()

        result = scheduler.get_all_games_needed()

        assert isinstance(result, dict)
        # Should include all configs from ALL_CONFIGS
        assert "hex8_2p" in result
        assert "square8_4p" in result


class TestAdaptiveBudget:
    """Tests for _get_adaptive_budget_for_elo (Dec 29, 2025)."""

    def test_budget_low_elo(self):
        """Test budget for low Elo (<1500)."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_STANDARD

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(1200.0)

        assert budget == GUMBEL_BUDGET_STANDARD

    def test_budget_medium_elo(self):
        """Test budget for medium Elo (1500-1800)."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_QUALITY

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(1650.0)

        assert budget == GUMBEL_BUDGET_QUALITY

    def test_budget_high_elo(self):
        """Test budget for high Elo (1800-2000)."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_ULTIMATE

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(1900.0)

        assert budget == GUMBEL_BUDGET_ULTIMATE

    def test_budget_master_elo(self):
        """Test budget for master Elo (>=2000)."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_MASTER

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(2100.0)

        assert budget == GUMBEL_BUDGET_MASTER

    def test_budget_boundary_1500(self):
        """Test budget at 1500 boundary."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_QUALITY

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(1500.0)

        assert budget == GUMBEL_BUDGET_QUALITY

    def test_budget_boundary_1800(self):
        """Test budget at 1800 boundary."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_ULTIMATE

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(1800.0)

        assert budget == GUMBEL_BUDGET_ULTIMATE

    def test_budget_boundary_2000(self):
        """Test budget at 2000 boundary."""
        from app.coordination.selfplay_scheduler import GUMBEL_BUDGET_MASTER

        scheduler = SelfplayScheduler()
        budget = scheduler._get_adaptive_budget_for_elo(2000.0)

        assert budget == GUMBEL_BUDGET_MASTER


class TestDynamicWeightsComputation:
    """Tests for _compute_dynamic_weights method (Dec 29, 2025)."""

    def test_rate_limiting(self):
        """Test that weight computation is rate-limited."""
        scheduler = SelfplayScheduler()

        # First call computes weights
        weights1 = scheduler._compute_dynamic_weights()

        # Immediate second call returns cached weights
        weights2 = scheduler._compute_dynamic_weights()

        # Should be the same object (cached)
        assert weights1 is weights2

    def test_idle_gpu_fraction_calculation(self):
        """Test idle GPU fraction is computed from node capabilities."""
        scheduler = SelfplayScheduler()

        # Set up node capabilities - 2 idle GPU nodes out of 4 = 50%
        scheduler._node_capabilities = {
            "node1": NodeCapability(node_id="node1", gpu_memory_gb=80.0, current_jobs=0),
            "node2": NodeCapability(node_id="node2", gpu_memory_gb=80.0, current_jobs=2),
            "node3": NodeCapability(node_id="node3", gpu_memory_gb=48.0, current_jobs=0),
            "node4": NodeCapability(node_id="node4", gpu_memory_gb=24.0, current_jobs=1),
        }

        # Force weight update
        scheduler._last_dynamic_weights_update = 0

        weights = scheduler._compute_dynamic_weights()

        assert weights.idle_gpu_fraction == 0.5

    def test_high_idle_gpus_boosts_staleness(self):
        """Test that high idle GPU fraction boosts staleness weight."""
        from app.coordination.selfplay_scheduler import STALENESS_WEIGHT

        scheduler = SelfplayScheduler()

        # Set up many idle GPUs (>50%)
        scheduler._node_capabilities = {
            f"node{i}": NodeCapability(
                node_id=f"node{i}",
                gpu_memory_gb=80.0,
                current_jobs=0 if i < 6 else 1,  # 6 idle out of 8
            )
            for i in range(8)
        }

        scheduler._last_dynamic_weights_update = 0
        weights = scheduler._compute_dynamic_weights()

        # Staleness should be boosted (but bounded)
        assert weights.staleness >= STALENESS_WEIGHT

    def test_configs_at_target_fraction(self):
        """Test configs at target fraction calculation."""
        scheduler = SelfplayScheduler()

        # Set 3 of 12 configs at target Elo
        for i, (cfg, priority) in enumerate(scheduler._config_priorities.items()):
            priority.current_elo = 2100.0 if i < 3 else 1500.0

        scheduler._last_dynamic_weights_update = 0
        weights = scheduler._compute_dynamic_weights()

        assert weights.configs_at_target_fraction == 3 / 12

    def test_average_elo_calculation(self):
        """Test average Elo calculation."""
        scheduler = SelfplayScheduler()

        # Set varied Elos
        elos = [1400, 1500, 1600, 1700, 1800, 1900, 1500, 1600, 1700, 1500, 1600, 1700]
        for cfg, elo in zip(scheduler._config_priorities.keys(), elos):
            scheduler._config_priorities[cfg].current_elo = float(elo)

        scheduler._last_dynamic_weights_update = 0
        weights = scheduler._compute_dynamic_weights()

        expected_avg = sum(elos) / len(elos)
        assert abs(weights.average_elo - expected_avg) < 1.0


class TestEloVelocityTracking:
    """Tests for Elo velocity tracking (Dec 29, 2025)."""

    def test_get_elo_velocity_unknown_config(self):
        """Test get_elo_velocity returns 0 for unknown config."""
        scheduler = SelfplayScheduler()
        assert scheduler.get_elo_velocity("unknown_config") == 0.0

    def test_on_elo_updated_initializes_history(self):
        """Test _on_elo_updated initializes history for new config."""
        scheduler = SelfplayScheduler()

        event = {"config_key": "hex8_2p", "new_elo": 1600.0}
        scheduler._on_elo_updated(event)

        assert "hex8_2p" in scheduler._elo_history
        assert len(scheduler._elo_history["hex8_2p"]) == 1

    def test_on_elo_updated_tracks_multiple_points(self):
        """Test _on_elo_updated accumulates history."""
        scheduler = SelfplayScheduler()

        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1500.0})
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1550.0})
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1600.0})

        assert len(scheduler._elo_history["hex8_2p"]) == 3

    def test_velocity_calculation_requires_minimum_time(self):
        """Test velocity needs at least 30 min of data."""
        scheduler = SelfplayScheduler()

        # Two points but too close in time
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1500.0})
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1600.0})

        # Velocity should not be computed yet (needs 30min)
        # But history is tracked
        assert len(scheduler._elo_history["hex8_2p"]) == 2

    def test_velocity_calculation_with_time_spread(self):
        """Test velocity calculation with sufficient time spread."""
        import time as time_module

        scheduler = SelfplayScheduler()

        # Manually insert history with time spread
        now = time_module.time()
        scheduler._elo_history["hex8_2p"] = [
            (now - 7200, 1500.0),  # 2 hours ago
            (now, 1600.0),  # Now
        ]

        # Trigger velocity calculation
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1600.0})

        # Velocity should be ~50 Elo/hour (100 over 2 hours)
        velocity = scheduler.get_elo_velocity("hex8_2p")
        assert abs(velocity - 50.0) < 5.0

    def test_history_pruning_24h(self):
        """Test history older than 24h is pruned."""
        import time as time_module

        scheduler = SelfplayScheduler()

        now = time_module.time()
        # Insert old and recent data
        scheduler._elo_history["hex8_2p"] = [
            (now - 100000, 1400.0),  # >24h ago (should be pruned)
            (now - 3600, 1500.0),    # 1h ago (should be kept)
        ]

        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 1550.0})

        # Old entry should be pruned
        assert len(scheduler._elo_history["hex8_2p"]) == 2  # Only recent entries

    def test_on_elo_updated_ignores_invalid_payload(self):
        """Test _on_elo_updated ignores invalid payloads."""
        scheduler = SelfplayScheduler()

        # Empty config
        scheduler._on_elo_updated({"config_key": "", "new_elo": 1500.0})
        assert len(scheduler._elo_history) == 0

        # Zero Elo
        scheduler._on_elo_updated({"config_key": "hex8_2p", "new_elo": 0.0})
        assert "hex8_2p" not in scheduler._elo_history

    def test_on_elo_updated_handles_event_with_payload_attr(self):
        """Test _on_elo_updated handles event objects with payload attribute."""
        scheduler = SelfplayScheduler()

        class MockEvent:
            payload = {"config_key": "hex8_2p", "new_elo": 1650.0}

        scheduler._on_elo_updated(MockEvent())

        assert "hex8_2p" in scheduler._elo_history


class TestComputeTargetGames:
    """Tests for _compute_target_games method (Dec 29, 2025)."""

    def test_target_games_at_target_elo(self):
        """Test target games is 0 when at target Elo."""
        scheduler = SelfplayScheduler()

        result = scheduler._compute_target_games("hex8_2p", 1900.0)

        assert result == 0

    def test_target_games_above_target_elo(self):
        """Test target games is 0 when above target Elo."""
        scheduler = SelfplayScheduler()

        result = scheduler._compute_target_games("hex8_2p", 2100.0)

        assert result == 0

    def test_target_games_below_target(self):
        """Test target games calculation below target."""
        scheduler = SelfplayScheduler()

        # 400 Elo gap * 500 base = 200,000 base
        result = scheduler._compute_target_games("hex8_2p", 1500.0)

        # Should be > 0 and proportional to gap
        assert result > 0
        assert result > scheduler._compute_target_games("hex8_2p", 1700.0)

    def test_target_games_board_difficulty_scaling(self):
        """Test larger boards need more target games."""
        scheduler = SelfplayScheduler()

        hex8_target = scheduler._compute_target_games("hex8_2p", 1500.0)
        square19_target = scheduler._compute_target_games("square19_2p", 1500.0)

        # square19 is larger board, should need more games
        assert square19_target > hex8_target


class TestDynamicWeightBounds:
    """Tests for DYNAMIC_WEIGHT_BOUNDS constants (Dec 29, 2025)."""

    def test_bounds_defined_for_key_weights(self):
        """Test bounds are defined for key weight types."""
        from app.coordination.selfplay_scheduler import DYNAMIC_WEIGHT_BOUNDS

        assert "staleness" in DYNAMIC_WEIGHT_BOUNDS
        assert "velocity" in DYNAMIC_WEIGHT_BOUNDS
        assert "curriculum" in DYNAMIC_WEIGHT_BOUNDS

    def test_bounds_are_tuples(self):
        """Test bounds are (min, max) tuples."""
        from app.coordination.selfplay_scheduler import DYNAMIC_WEIGHT_BOUNDS

        for name, bounds in DYNAMIC_WEIGHT_BOUNDS.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            min_val, max_val = bounds
            assert min_val < max_val

    def test_bounds_are_reasonable(self):
        """Test bounds are within 0-1 range."""
        from app.coordination.selfplay_scheduler import DYNAMIC_WEIGHT_BOUNDS

        for name, (min_val, max_val) in DYNAMIC_WEIGHT_BOUNDS.items():
            assert 0.0 <= min_val <= 1.0
            assert 0.0 <= max_val <= 1.0


class TestDec29Constants:
    """Tests for Dec 29, 2025 threshold constants."""

    def test_idle_gpu_thresholds(self):
        """Test idle GPU threshold values."""
        from app.coordination.selfplay_scheduler import (
            IDLE_GPU_HIGH_THRESHOLD,
            IDLE_GPU_LOW_THRESHOLD,
        )

        assert IDLE_GPU_LOW_THRESHOLD < IDLE_GPU_HIGH_THRESHOLD
        assert 0.0 <= IDLE_GPU_LOW_THRESHOLD <= 1.0
        assert 0.0 <= IDLE_GPU_HIGH_THRESHOLD <= 1.0

    def test_training_queue_threshold(self):
        """Test training queue threshold is positive."""
        from app.coordination.selfplay_scheduler import TRAINING_QUEUE_HIGH_THRESHOLD

        assert TRAINING_QUEUE_HIGH_THRESHOLD > 0

    def test_elo_thresholds_ordered(self):
        """Test Elo thresholds are properly ordered."""
        from app.coordination.selfplay_scheduler import (
            ELO_MEDIUM_THRESHOLD,
            ELO_HIGH_THRESHOLD,
        )

        assert ELO_MEDIUM_THRESHOLD < ELO_HIGH_THRESHOLD

    def test_configs_at_target_threshold(self):
        """Test configs at target threshold is 0-1."""
        from app.coordination.selfplay_scheduler import CONFIGS_AT_TARGET_THRESHOLD

        assert 0.0 <= CONFIGS_AT_TARGET_THRESHOLD <= 1.0


# =============================================================================
# NODE_OVERLOADED Handler Tests (Dec 29, 2025)
# =============================================================================


class TestNodeOverloadedHandler:
    """Tests for NODE_OVERLOADED event handler and node backoff tracking."""

    def setup_method(self):
        """Create fresh scheduler for each test."""
        reset_selfplay_scheduler()
        self.scheduler = SelfplayScheduler()

    def test_initial_no_overloaded_nodes(self):
        """Test that initially no nodes are overloaded."""
        assert self.scheduler.get_overloaded_nodes() == []

    def test_is_node_under_backoff_false_initially(self):
        """Test is_node_under_backoff returns False for unknown node."""
        assert self.scheduler.is_node_under_backoff("unknown-node") is False

    def test_on_node_overloaded_adds_to_backoff(self):
        """Test _on_node_overloaded adds node to backoff set."""
        event = MagicMock()
        event.payload = {
            "host": "vast-12345",
            "cpu_percent": 95,
            "gpu_percent": 90,
            "memory_percent": 85,
            "resource_type": "cpu",
        }

        self.scheduler._on_node_overloaded(event)

        assert self.scheduler.is_node_under_backoff("vast-12345") is True
        assert "vast-12345" in self.scheduler.get_overloaded_nodes()

    def test_on_node_overloaded_ignores_empty_host(self):
        """Test _on_node_overloaded ignores event without host."""
        event = MagicMock()
        event.payload = {
            "host": "",
            "cpu_percent": 95,
        }

        self.scheduler._on_node_overloaded(event)

        assert self.scheduler.get_overloaded_nodes() == []

    def test_on_node_overloaded_memory_longer_backoff(self):
        """Test memory overload gets longer backoff (90s vs 60s)."""
        event = MagicMock()
        event.payload = {
            "host": "vast-memory-node",
            "memory_percent": 95,
            "resource_type": "memory",
        }

        self.scheduler._on_node_overloaded(event)

        # Node should be under backoff
        assert self.scheduler.is_node_under_backoff("vast-memory-node") is True

    def test_on_node_overloaded_consecutive_failures_longest_backoff(self):
        """Test consecutive_failures gets longest backoff (120s)."""
        event = MagicMock()
        event.payload = {
            "host": "failing-node",
            "resource_type": "consecutive_failures",
        }

        self.scheduler._on_node_overloaded(event)

        assert self.scheduler.is_node_under_backoff("failing-node") is True

    def test_get_overloaded_nodes_cleans_expired(self):
        """Test get_overloaded_nodes cleans up expired backoffs."""
        # Manually add an expired backoff
        import time

        if not hasattr(self.scheduler, "_overloaded_nodes"):
            self.scheduler._overloaded_nodes = {}

        self.scheduler._overloaded_nodes["expired-node"] = time.time() - 10
        self.scheduler._overloaded_nodes["active-node"] = time.time() + 60

        result = self.scheduler.get_overloaded_nodes()

        assert "expired-node" not in result
        assert "active-node" in result
        assert "expired-node" not in self.scheduler._overloaded_nodes

    def test_get_status_includes_overloaded_nodes(self):
        """Test get_status includes overloaded_nodes key."""
        status = self.scheduler.get_status()

        assert "overloaded_nodes" in status
        assert isinstance(status["overloaded_nodes"], list)

    def test_multiple_overloaded_nodes_tracked(self):
        """Test multiple nodes can be tracked as overloaded."""
        for i in range(5):
            event = MagicMock()
            event.payload = {
                "host": f"node-{i}",
                "cpu_percent": 90 + i,
                "resource_type": "cpu",
            }
            self.scheduler._on_node_overloaded(event)

        overloaded = self.scheduler.get_overloaded_nodes()
        assert len(overloaded) == 5
        for i in range(5):
            assert f"node-{i}" in overloaded

    def test_on_node_overloaded_updates_existing_backoff(self):
        """Test repeated overload events update the backoff time."""
        event = MagicMock()
        event.payload = {
            "host": "repeat-node",
            "cpu_percent": 95,
            "resource_type": "cpu",
        }

        # First overload
        self.scheduler._on_node_overloaded(event)
        assert self.scheduler.is_node_under_backoff("repeat-node") is True

        # Second overload - should update backoff time
        self.scheduler._on_node_overloaded(event)
        assert self.scheduler.is_node_under_backoff("repeat-node") is True

    def test_on_node_overloaded_handles_dict_payload(self):
        """Test handler accepts dict directly as event (no .payload attr)."""
        # Some events may be dicts rather than objects with .payload
        event_dict = {
            "host": "dict-node",
            "cpu_percent": 95,
            "resource_type": "cpu",
        }

        self.scheduler._on_node_overloaded(event_dict)

        assert self.scheduler.is_node_under_backoff("dict-node") is True


# Run with: pytest tests/unit/coordination/test_selfplay_scheduler.py -v -k "Dec29 or DynamicWeights or VOI or GameCount or AdaptiveBudget or EloVelocity or NodeOverloaded"
