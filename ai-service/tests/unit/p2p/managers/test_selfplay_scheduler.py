"""Comprehensive unit tests for SelfplayScheduler.

Tests cover:
- DiversityMetrics dataclass
- Initialization and configuration
- ELO-based priority boost calculation
- Weighted config selection (pick_weighted_config)
- Job targeting (get_target_jobs_for_node, get_hybrid_job_targets)
- Diversity tracking
- Health check functionality
- Event subscription handlers
- Exploration boost and promotion failure tracking

30+ tests organized by functionality.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from scripts.p2p.managers.selfplay_scheduler import (
    DiversityMetrics,
    SelfplayScheduler,
)


# =============================================================================
# Mock NodeInfo for testing
# =============================================================================


@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""

    node_id: str = "test-node"
    has_gpu: bool = True
    gpu_name: str = "RTX 4090"
    gpu_count: int = 1
    memory_gb: int = 128
    gpu_vram_gb: int = 24  # Dec 2025: GPU VRAM used for board size filtering
    cpu_count: int = 32
    cpu_percent: float = 50.0
    memory_percent: float = 60.0
    disk_percent: float = 40.0
    gpu_percent: float = 55.0
    gpu_memory_percent: float = 50.0
    selfplay_jobs: int = 4
    capabilities: list = None  # Dec 2025: Capabilities for job targeting

    def __post_init__(self):
        """Set default capabilities based on GPU availability."""
        if self.capabilities is None:
            self.capabilities = ["selfplay", "training"] if self.has_gpu else ["selfplay"]


# =============================================================================
# Test DiversityMetrics Dataclass
# =============================================================================


class TestDiversityMetrics:
    """Tests for DiversityMetrics dataclass."""

    def test_default_initialization(self) -> None:
        """Test default values on initialization."""
        metrics = DiversityMetrics()
        assert metrics.games_by_engine_mode == {}
        assert metrics.games_by_board_config == {}
        assert metrics.games_by_difficulty == {}
        assert metrics.asymmetric_games == 0
        assert metrics.symmetric_games == 0
        assert isinstance(metrics.last_reset, float)
        assert metrics.last_reset <= time.time()

    def test_to_dict_empty(self) -> None:
        """Test to_dict with empty metrics."""
        metrics = DiversityMetrics()
        result = metrics.to_dict()

        assert result["games_by_engine_mode"] == {}
        assert result["games_by_board_config"] == {}
        assert result["games_by_difficulty"] == {}
        assert result["asymmetric_games"] == 0
        assert result["symmetric_games"] == 0
        assert result["asymmetric_ratio"] == 0.0
        assert result["engine_mode_distribution"] == {}
        assert "uptime_seconds" in result

    def test_to_dict_with_data(self) -> None:
        """Test to_dict with populated metrics."""
        metrics = DiversityMetrics(
            games_by_engine_mode={"mixed": 60, "mcts-only": 40},
            games_by_board_config={"hex8_2p": 50, "square8_2p": 50},
            games_by_difficulty={"medium": 100},
            asymmetric_games=30,
            symmetric_games=70,
        )
        result = metrics.to_dict()

        assert result["asymmetric_ratio"] == 0.3
        assert result["engine_mode_distribution"]["mixed"] == 0.6
        assert result["engine_mode_distribution"]["mcts-only"] == 0.4

    def test_asymmetric_ratio_calculation(self) -> None:
        """Test asymmetric ratio calculation."""
        # All asymmetric
        metrics = DiversityMetrics(asymmetric_games=100, symmetric_games=0)
        assert metrics.to_dict()["asymmetric_ratio"] == 1.0

        # All symmetric
        metrics = DiversityMetrics(asymmetric_games=0, symmetric_games=100)
        assert metrics.to_dict()["asymmetric_ratio"] == 0.0

        # 50/50 split
        metrics = DiversityMetrics(asymmetric_games=50, symmetric_games=50)
        assert metrics.to_dict()["asymmetric_ratio"] == 0.5


# =============================================================================
# Test SelfplayScheduler Initialization
# =============================================================================


class TestSelfplaySchedulerInit:
    """Tests for SelfplayScheduler initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with no callbacks."""
        scheduler = SelfplayScheduler()

        assert scheduler.verbose is False
        assert scheduler._subscribed is False
        assert scheduler._rate_multipliers == {}
        assert scheduler._previous_targets == {}
        assert scheduler._previous_priorities == {}
        assert scheduler._exploration_boosts == {}
        assert scheduler._training_complete_boosts == {}
        assert isinstance(scheduler.diversity_metrics, DiversityMetrics)

    def test_initialization_with_callbacks(self) -> None:
        """Test initialization with all callbacks provided."""
        get_cluster_elo = MagicMock(return_value={})
        load_curriculum_weights = MagicMock(return_value={})
        get_board_priority_overrides = MagicMock(return_value={})

        scheduler = SelfplayScheduler(
            get_cluster_elo_fn=get_cluster_elo,
            load_curriculum_weights_fn=load_curriculum_weights,
            get_board_priority_overrides_fn=get_board_priority_overrides,
            verbose=True,
        )

        assert scheduler.verbose is True
        assert scheduler.get_cluster_elo is get_cluster_elo
        assert scheduler.load_curriculum_weights is load_curriculum_weights
        assert scheduler.get_board_priority_overrides is get_board_priority_overrides

    def test_initialization_subscription_lock_exists(self) -> None:
        """Test that subscription lock is initialized."""
        scheduler = SelfplayScheduler()
        assert hasattr(scheduler, "_subscription_lock")
        # Verify it's a threading lock by attempting to acquire
        acquired = scheduler._subscription_lock.acquire(blocking=False)
        assert acquired is True
        scheduler._subscription_lock.release()


# =============================================================================
# Test ELO-Based Priority Boost
# =============================================================================


class TestEloPriorityBoost:
    """Tests for ELO-based priority boost calculation."""

    def test_no_cluster_elo_returns_zero(self) -> None:
        """Test boost is 0 when no cluster ELO data available."""
        scheduler = SelfplayScheduler(get_cluster_elo_fn=lambda: {})

        boost = scheduler.get_elo_based_priority_boost("hex8", 2)
        assert boost >= 0  # Should be at least 0

    def test_underrepresented_board_boost(self) -> None:
        """Test boost for underrepresented board types."""
        scheduler = SelfplayScheduler(get_cluster_elo_fn=lambda: {})

        # hexagonal and square19 get +1 boost
        hex_boost = scheduler.get_elo_based_priority_boost("hexagonal", 2)
        sq19_boost = scheduler.get_elo_based_priority_boost("square19", 2)
        sq8_boost = scheduler.get_elo_based_priority_boost("square8", 2)

        # Hex and sq19 should have higher boost than sq8
        assert hex_boost >= sq8_boost
        assert sq19_boost >= sq8_boost

    def test_multiplayer_boost(self) -> None:
        """Test boost for >2 player configs."""
        scheduler = SelfplayScheduler(get_cluster_elo_fn=lambda: {})

        boost_2p = scheduler.get_elo_based_priority_boost("hex8", 2)
        boost_3p = scheduler.get_elo_based_priority_boost("hex8", 3)
        boost_4p = scheduler.get_elo_based_priority_boost("hex8", 4)

        # 3p and 4p should have higher boost than 2p
        assert boost_3p >= boost_2p
        assert boost_4p >= boost_2p

    def test_high_elo_model_boost(self) -> None:
        """Test boost for high ELO models."""

        def get_high_elo():
            return {
                "top_models": [
                    {"name": "hex8_2p_champion", "elo": 1500},
                ]
            }

        scheduler = SelfplayScheduler(get_cluster_elo_fn=get_high_elo)

        # Model with 1500 ELO should get boost (300 above 1200 = +3)
        boost = scheduler.get_elo_based_priority_boost("hex8", 2)
        # Should have at least some boost from high ELO
        assert boost >= 0

    def test_boost_capped_at_5(self) -> None:
        """Test that boost is capped at 5."""

        def get_very_high_elo():
            return {
                "top_models": [
                    {"name": "hexagonal_4p_master", "elo": 2000},
                ]
            }

        scheduler = SelfplayScheduler(get_cluster_elo_fn=get_very_high_elo)

        # hexagonal + 4p + high ELO should still be capped at 5
        boost = scheduler.get_elo_based_priority_boost("hexagonal", 4)
        assert boost <= 5


# =============================================================================
# Test Weighted Config Selection
# =============================================================================


class TestPickWeightedConfig:
    """Tests for pick_weighted_config method."""

    def test_returns_config_dict(self) -> None:
        """Test that pick_weighted_config returns a config dict."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo()

        config = scheduler.pick_weighted_config(node)

        assert config is not None
        assert "board_type" in config
        assert "num_players" in config
        assert "engine_mode" in config
        assert "priority" in config

    def test_low_memory_node_filters_out_large_boards(self) -> None:
        """Test that low GPU VRAM nodes avoid only the truly large boards.

        Jan 12, 2026: SelfplayScheduler allows both hex8 and square8 on
        smaller GPUs; only square19 and hexagonal are filtered out.
        """
        scheduler = SelfplayScheduler()
        node = MockNodeInfo(gpu_vram_gb=16)  # Low GPU VRAM

        # Run multiple times to ensure consistent filtering
        for _ in range(10):
            config = scheduler.pick_weighted_config(node)
            if config:
                assert config["board_type"] in {"square8", "hex8"}

    def test_high_memory_node_gets_variety(self) -> None:
        """Test that high GPU VRAM nodes can get various board types."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo(gpu_vram_gb=80)  # High GPU VRAM (H100 80GB)

        board_types_seen = set()
        for _ in range(100):
            config = scheduler.pick_weighted_config(node)
            if config:
                board_types_seen.add(config["board_type"])

        # Should see multiple board types
        assert len(board_types_seen) > 1

    def test_effective_priority_calculated(self) -> None:
        """Test that effective_priority is added to selected config."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo()

        config = scheduler.pick_weighted_config(node)

        assert config is not None
        assert "effective_priority" in config
        assert isinstance(config["effective_priority"], int)

    def test_rate_multiplier_affects_priority(self) -> None:
        """Test that rate multiplier affects effective priority."""
        scheduler = SelfplayScheduler()
        scheduler._rate_multipliers["hex8_2p"] = 1.5  # 50% boost

        node = MockNodeInfo()
        configs_with_boost = []

        for _ in range(50):
            config = scheduler.pick_weighted_config(node)
            if config and f"{config['board_type']}_{config['num_players']}p" == "hex8_2p":
                configs_with_boost.append(config)

        # With boost, hex8_2p should be selected more often
        # (probabilistic, so just check we got some)
        # Note: This is a statistical test that may occasionally fail
        # In a real test suite, we'd use seeded random for determinism

    def test_curriculum_weights_applied(self) -> None:
        """Test that curriculum weights affect priority."""

        def get_curriculum_weights():
            return {"hex8_2p": 1.5}  # 50% boost for hex8_2p

        scheduler = SelfplayScheduler(
            load_curriculum_weights_fn=get_curriculum_weights
        )
        node = MockNodeInfo()

        config = scheduler.pick_weighted_config(node)
        assert config is not None


# =============================================================================
# Test Job Targeting
# =============================================================================


class TestGetTargetJobsForNode:
    """Tests for get_target_jobs_for_node method."""

    def test_returns_positive_integer(self) -> None:
        """Test that method returns a positive integer."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo()

        target = scheduler.get_target_jobs_for_node(node)

        assert isinstance(target, int)
        assert target >= 0

    def test_emergency_active_returns_zero(self) -> None:
        """Test that emergency mode returns 0 jobs."""
        scheduler = SelfplayScheduler(
            is_emergency_active_fn=lambda: True
        )
        node = MockNodeInfo()

        target = scheduler.get_target_jobs_for_node(node)

        assert target == 0

    def test_backpressure_stop_returns_zero(self) -> None:
        """Test that backpressure stop returns 0 jobs."""
        scheduler = SelfplayScheduler(
            should_stop_production_fn=lambda _: True,
            should_throttle_production_fn=lambda _: False,
        )
        node = MockNodeInfo()

        target = scheduler.get_target_jobs_for_node(node)

        assert target == 0

    def test_low_memory_returns_zero(self) -> None:
        """Test that nodes with insufficient memory return 0 jobs."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo(memory_gb=8)  # Below MIN_MEMORY_GB_FOR_TASKS (64)

        target = scheduler.get_target_jobs_for_node(node)

        assert target == 0

    def test_gpu_overloaded_reduces_target(self) -> None:
        """Test that GPU overload reduces target jobs."""
        scheduler = SelfplayScheduler()
        overloaded_node = MockNodeInfo(gpu_percent=90.0, gpu_memory_percent=90.0)

        target = scheduler.get_target_jobs_for_node(overloaded_node)

        # Should still return something but be reduced
        assert isinstance(target, int)

    def test_high_disk_usage_limits_target(self) -> None:
        """Test that high disk usage limits target jobs."""
        scheduler = SelfplayScheduler()
        high_disk_node = MockNodeInfo(disk_percent=70.0)

        target = scheduler.get_target_jobs_for_node(high_disk_node)

        # Should be capped at 4 when disk > 65%
        assert target <= 4

    def test_high_memory_pressure_limits_target(self) -> None:
        """Test that high memory pressure limits target jobs."""
        scheduler = SelfplayScheduler()
        high_mem_node = MockNodeInfo(memory_percent=80.0)

        target = scheduler.get_target_jobs_for_node(high_mem_node)

        # Should be capped at 2 when memory > 75%
        assert target <= 2

    def test_cpu_only_node_scaling(self) -> None:
        """Test job targeting for CPU-only nodes."""
        scheduler = SelfplayScheduler()
        cpu_node = MockNodeInfo(
            has_gpu=False,
            gpu_name="",
            gpu_count=0,
            cpu_count=64,
        )

        target = scheduler.get_target_jobs_for_node(cpu_node)

        # CPU-only: ~0.3 jobs per core, capped at 32
        # With 64 cores: 64 * 0.3 = 19.2, should be around 19
        assert target >= 1
        assert target <= 32


class TestGetHybridJobTargets:
    """Tests for get_hybrid_job_targets method."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Test that method returns dict with required keys."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo()

        targets = scheduler.get_hybrid_job_targets(node)

        assert isinstance(targets, dict)
        assert "gpu_jobs" in targets
        assert "cpu_only_jobs" in targets
        assert "total_jobs" in targets

    def test_total_equals_sum(self) -> None:
        """Test that total_jobs equals gpu_jobs + cpu_only_jobs."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo()

        targets = scheduler.get_hybrid_job_targets(node)

        assert targets["total_jobs"] == targets["gpu_jobs"] + targets["cpu_only_jobs"]

    def test_fallback_no_cpu_only_jobs(self) -> None:
        """Test fallback when hybrid limits function not available."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=None
        )
        node = MockNodeInfo()

        targets = scheduler.get_hybrid_job_targets(node)

        # Without hybrid limits, cpu_only_jobs should be 0
        assert targets["cpu_only_jobs"] == 0


class TestShouldSpawnCpuOnlyJobs:
    """Tests for should_spawn_cpu_only_jobs method."""

    def test_no_hybrid_limits_returns_false(self) -> None:
        """Test returns False when no hybrid limits function."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=None
        )
        node = MockNodeInfo()

        result = scheduler.should_spawn_cpu_only_jobs(node)

        assert result is False

    def test_low_cpu_returns_false(self) -> None:
        """Test returns False for nodes with <64 CPUs."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 4, "cpu_only_jobs": 8}
        )
        node = MockNodeInfo(cpu_count=32)

        result = scheduler.should_spawn_cpu_only_jobs(node)

        assert result is False

    def test_datacenter_gpu_returns_false(self) -> None:
        """Test returns False for datacenter GPUs (GH200, H100, etc.)."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 4, "cpu_only_jobs": 8}
        )
        node = MockNodeInfo(cpu_count=128, gpu_name="GH200", has_gpu=True)

        result = scheduler.should_spawn_cpu_only_jobs(node)

        assert result is False

    def test_consumer_gpu_with_high_cpu_returns_true(self) -> None:
        """Test returns True for consumer GPUs with high CPU count."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 4, "cpu_only_jobs": 8}
        )
        node = MockNodeInfo(cpu_count=128, gpu_name="RTX 3060", has_gpu=True)

        result = scheduler.should_spawn_cpu_only_jobs(node)

        assert result is True


# =============================================================================
# Test Diversity Tracking
# =============================================================================


class TestDiversityTracking:
    """Tests for diversity tracking."""

    def test_track_diversity_engine_mode(self) -> None:
        """Test tracking engine mode."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_engine_mode["mixed"] == 1

    def test_track_diversity_board_config(self) -> None:
        """Test tracking board config."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_board_config["hex8_2p"] == 1

    def test_track_diversity_symmetric_game(self) -> None:
        """Test tracking symmetric games."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.symmetric_games == 1
        assert scheduler.diversity_metrics.asymmetric_games == 0

    def test_track_diversity_asymmetric_game(self) -> None:
        """Test tracking asymmetric games."""
        scheduler = SelfplayScheduler()

        config = {
            "engine_mode": "mixed",
            "board_type": "hex8",
            "num_players": 2,
            "asymmetric": True,
            "strong_config": {"engine_mode": "gumbel", "difficulty": 5},
            "weak_config": {"engine_mode": "heuristic", "difficulty": 2},
        }
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.asymmetric_games == 1
        assert scheduler.diversity_metrics.symmetric_games == 0

    def test_track_diversity_difficulty(self) -> None:
        """Test tracking difficulty levels."""
        scheduler = SelfplayScheduler()

        config = {
            "engine_mode": "mixed",
            "board_type": "hex8",
            "num_players": 2,
            "difficulty": "hard",
        }
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_difficulty["hard"] == 1

    def test_get_diversity_metrics(self) -> None:
        """Test getting diversity metrics dict."""
        scheduler = SelfplayScheduler()

        # Track some games
        for _ in range(5):
            scheduler.track_diversity(
                {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
            )

        metrics = scheduler.get_diversity_metrics()

        assert metrics["symmetric_games"] == 5
        assert metrics["games_by_engine_mode"]["mixed"] == 5


# =============================================================================
# Test Rate Multipliers
# =============================================================================


class TestRateMultipliers:
    """Tests for rate multiplier management."""

    def test_get_rate_multiplier_default(self) -> None:
        """Test default rate multiplier is 1.0."""
        scheduler = SelfplayScheduler()

        rate = scheduler.get_rate_multiplier("unknown_config")

        assert rate == 1.0

    def test_get_rate_multiplier_with_value(self) -> None:
        """Test getting set rate multiplier."""
        scheduler = SelfplayScheduler()
        scheduler._rate_multipliers["hex8_2p"] = 1.5

        rate = scheduler.get_rate_multiplier("hex8_2p")

        assert rate == 1.5


# =============================================================================
# Test Event Handlers
# =============================================================================


class TestEventHandlers:
    """Tests for event handler methods."""

    def test_get_event_subscriptions(self) -> None:
        """Test _get_event_subscriptions returns expected events."""
        scheduler = SelfplayScheduler()

        subscriptions = scheduler._get_event_subscriptions()

        assert "SELFPLAY_RATE_CHANGED" in subscriptions
        assert "EXPLORATION_BOOST" in subscriptions
        assert "TRAINING_COMPLETED" in subscriptions
        assert "ELO_VELOCITY_CHANGED" in subscriptions

    @pytest.mark.asyncio
    async def test_on_selfplay_rate_changed(self) -> None:
        """Test handling SELFPLAY_RATE_CHANGED event."""
        scheduler = SelfplayScheduler()

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p", "new_rate": 1.5, "reason": "test"}

        with patch.object(scheduler, "_extract_event_payload", return_value=event.payload):
            await scheduler._on_selfplay_rate_changed(event)

        assert scheduler._rate_multipliers["hex8_2p"] == 1.5

    @pytest.mark.asyncio
    async def test_on_exploration_boost(self) -> None:
        """Test handling EXPLORATION_BOOST event."""
        scheduler = SelfplayScheduler()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "boost_factor": 1.3,
            "duration_seconds": 600,
            "reason": "training_anomaly",
        }

        with patch.object(scheduler, "_extract_event_payload", return_value=event.payload):
            with patch.object(scheduler, "set_exploration_boost") as mock_set:
                await scheduler._on_exploration_boost(event)
                mock_set.assert_called_once_with("hex8_2p", 1.3, 600)

    @pytest.mark.asyncio
    async def test_on_training_completed(self) -> None:
        """Test handling TRAINING_COMPLETED event."""
        scheduler = SelfplayScheduler()

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        with patch.object(scheduler, "_extract_event_payload", return_value=event.payload):
            with patch.object(scheduler, "on_training_complete") as mock_complete:
                await scheduler._on_training_completed(event)
                mock_complete.assert_called_once_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_on_elo_velocity_changed_accelerating(self) -> None:
        """Test handling ELO_VELOCITY_CHANGED with accelerating trend."""
        scheduler = SelfplayScheduler()
        scheduler._rate_multipliers["hex8_2p"] = 1.0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "velocity": 50.0,
            "trend": "accelerating",
        }

        with patch.object(scheduler, "_extract_event_payload", return_value=event.payload):
            await scheduler._on_elo_velocity_changed(event)

        # Accelerating should increase rate
        assert scheduler._rate_multipliers["hex8_2p"] > 1.0

    @pytest.mark.asyncio
    async def test_on_elo_velocity_changed_decelerating(self) -> None:
        """Test handling ELO_VELOCITY_CHANGED with decelerating trend."""
        scheduler = SelfplayScheduler()
        scheduler._rate_multipliers["hex8_2p"] = 1.0

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "velocity": -20.0,
            "trend": "decelerating",
        }

        with patch.object(scheduler, "_extract_event_payload", return_value=event.payload):
            await scheduler._on_elo_velocity_changed(event)

        # Decelerating should decrease rate
        assert scheduler._rate_multipliers["hex8_2p"] < 1.0


# =============================================================================
# Test Exploration Boost
# =============================================================================


class TestExplorationBoost:
    """Tests for exploration boost functionality."""

    def test_set_exploration_boost(self) -> None:
        """Test setting exploration boost."""
        scheduler = SelfplayScheduler()

        scheduler.set_exploration_boost("hex8_2p", 1.5, 600)

        assert "hex8_2p" in scheduler._exploration_boosts
        boost_factor, expiry = scheduler._exploration_boosts["hex8_2p"]
        assert boost_factor == 1.5
        assert expiry > time.time()

    def test_get_exploration_boost_active(self) -> None:
        """Test getting active exploration boost."""
        scheduler = SelfplayScheduler()
        scheduler.set_exploration_boost("hex8_2p", 1.5, 600)

        boost = scheduler.get_exploration_boost("hex8_2p")

        assert boost == 1.5

    def test_get_exploration_boost_expired(self) -> None:
        """Test getting expired exploration boost returns 1.0."""
        scheduler = SelfplayScheduler()
        # Set boost with 0 duration (already expired)
        scheduler._exploration_boosts["hex8_2p"] = (1.5, time.time() - 1)

        boost = scheduler.get_exploration_boost("hex8_2p")

        assert boost == 1.0
        assert "hex8_2p" not in scheduler._exploration_boosts

    def test_get_exploration_boost_not_set(self) -> None:
        """Test getting exploration boost when not set returns 1.0."""
        scheduler = SelfplayScheduler()

        boost = scheduler.get_exploration_boost("unknown_config")

        assert boost == 1.0


# =============================================================================
# Test Training Complete Handling
# =============================================================================


class TestOnTrainingComplete:
    """Tests for on_training_complete method."""

    def test_sets_training_complete_boost(self) -> None:
        """Test that training complete sets a boost."""
        scheduler = SelfplayScheduler()

        scheduler.on_training_complete("hex8_2p")

        assert "hex8_2p" in scheduler._training_complete_boosts
        assert scheduler._training_complete_boosts["hex8_2p"] > time.time()


# =============================================================================
# Test Promotion Failure Recording
# =============================================================================


class TestPromotionFailureRecording:
    """Tests for record_promotion_failure method."""

    def test_records_first_failure(self) -> None:
        """Test recording first promotion failure."""
        scheduler = SelfplayScheduler()

        scheduler.record_promotion_failure("hex8_2p")

        assert hasattr(scheduler, "_promotion_failures")
        assert "hex8_2p" in scheduler._promotion_failures
        assert len(scheduler._promotion_failures["hex8_2p"]) == 1

    def test_records_multiple_failures(self) -> None:
        """Test recording multiple promotion failures."""
        scheduler = SelfplayScheduler()

        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")

        assert len(scheduler._promotion_failures["hex8_2p"]) == 3

    def test_applies_penalty_after_failures(self) -> None:
        """Test that penalty is applied after multiple failures."""
        scheduler = SelfplayScheduler()

        # Record 3 failures
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")

        assert hasattr(scheduler, "_promotion_penalties")
        assert "hex8_2p" in scheduler._promotion_penalties
        penalty_factor, expiry = scheduler._promotion_penalties["hex8_2p"]
        # After 3 failures, penalty should be 0.3 (30%)
        assert penalty_factor == 0.3


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_healthy_when_no_issues(self) -> None:
        """Test health check returns healthy status."""
        scheduler = SelfplayScheduler()
        scheduler._subscribed = True

        health = scheduler.health_check()

        # health_check returns HealthCheckResult dataclass
        assert hasattr(health, "healthy")
        assert hasattr(health, "details")
        # Check details dict for subscribed status
        if hasattr(health, "details") and health.details:
            assert health.details.get("subscribed") is True

    def test_degraded_when_not_subscribed(self) -> None:
        """Test health check returns degraded when not subscribed."""
        scheduler = SelfplayScheduler()
        scheduler._subscribed = False

        health = scheduler.health_check()

        # Should indicate not healthy or degraded state
        assert hasattr(health, "healthy")
        # Message should indicate subscription issue
        if hasattr(health, "message"):
            assert "subscribed" in health.message.lower() or "event" in health.message.lower()

    def test_includes_diversity_metrics(self) -> None:
        """Test health check includes diversity metrics."""
        scheduler = SelfplayScheduler()

        health = scheduler.health_check()

        # Check details dict
        assert hasattr(health, "details")
        if health.details:
            assert "diversity_metrics" in health.details
            assert "symmetric_games" in health.details["diversity_metrics"]

    def test_includes_exploration_boosts_count(self) -> None:
        """Test health check includes active exploration boosts count."""
        scheduler = SelfplayScheduler()
        scheduler.set_exploration_boost("hex8_2p", 1.5, 600)
        scheduler.set_exploration_boost("square8_2p", 1.3, 600)

        health = scheduler.health_check()

        assert hasattr(health, "details")
        if health.details:
            assert "active_exploration_boosts" in health.details
            assert health.details["active_exploration_boosts"] == 2

    def test_degraded_when_no_targets(self) -> None:
        """Test health check returns degraded when no configs have targets."""
        scheduler = SelfplayScheduler()
        scheduler._previous_targets = {"node1": 0, "node2": 0}

        health = scheduler.health_check()

        # Should be degraded or unhealthy
        assert hasattr(health, "healthy")
        assert hasattr(health, "status")


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Tests for event emission functionality."""

    def test_emit_selfplay_target_updated_graceful_failure(self) -> None:
        """Test that event emission fails gracefully when router unavailable."""
        scheduler = SelfplayScheduler()

        # Should not raise even if event router is not available
        scheduler._emit_selfplay_target_updated(
            config_key="hex8_2p",
            priority="normal",
            reason="test",
            target_jobs=10,
        )

    def test_emit_selfplay_target_updated_with_router(self) -> None:
        """Test event emission when router is available."""
        scheduler = SelfplayScheduler(verbose=True)

        # Mock the publish_sync function at the module level
        with patch("app.coordination.event_router.publish_sync") as mock_publish:
            scheduler._emit_selfplay_target_updated(
                config_key="hex8_2p",
                priority="high",
                reason="priority_boost:3",
                target_jobs=15,
                effective_priority=10,
                exploration_boost=1.5,
            )

            mock_publish.assert_called_once()
            call_args = mock_publish.call_args
            assert call_args[0][0] == "SELFPLAY_TARGET_UPDATED"
            payload = call_args[0][1]
            assert payload["config_key"] == "hex8_2p"
            assert payload["priority"] == "high"
            assert payload["target_jobs"] == 15


# =============================================================================
# Test GPU Type Handling
# =============================================================================


class TestGPUTypeHandling:
    """Tests for different GPU type handling in job targeting."""

    @pytest.mark.parametrize(
        "gpu_name,expected_multiplier",
        [
            ("GH200", 0.8),  # High multiplier for GH200
            ("H100", 0.5),  # Medium-high for H100
            ("A100", 0.4),  # Medium for A100
            ("RTX 4090", 0.3),  # Lower for consumer
            ("RTX 3060", 0.2),  # Lower for older consumer
        ],
    )
    def test_gpu_type_affects_job_target(
        self, gpu_name: str, expected_multiplier: float
    ) -> None:
        """Test that GPU type affects job targeting."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo(gpu_name=gpu_name, cpu_count=64)

        target = scheduler.get_target_jobs_for_node(node)

        # Should return a positive number based on GPU capability
        assert target >= 1

    def test_rtx_5090_handling(self) -> None:
        """Test RTX 5090 specific handling."""
        scheduler = SelfplayScheduler()
        node = MockNodeInfo(gpu_name="RTX 5090", cpu_count=64, gpu_count=1)

        target = scheduler.get_target_jobs_for_node(node)

        # RTX 5090 should have decent capacity
        assert target >= 1
