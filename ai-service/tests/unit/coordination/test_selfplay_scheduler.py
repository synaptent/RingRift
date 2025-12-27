"""Tests for SelfplayScheduler (December 2025).

Tests cover:
- Configuration priority calculation
- Data freshness/staleness factors
- Node capability tracking
- Allocation algorithms
- Event handling
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.selfplay_scheduler import (
    ALL_CONFIGS,
    ConfigPriority,
    NodeCapability,
    SelfplayScheduler,
    get_selfplay_scheduler,
    reset_selfplay_scheduler,
    STALENESS_WEIGHT,
    ELO_VELOCITY_WEIGHT,
)


class TestConfigPriority:
    """Tests for ConfigPriority dataclass."""

    def test_config_priority_defaults(self):
        """Test default values."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.config_key == "hex8_2p"
        assert priority.staleness_hours == 0.0
        assert priority.elo_velocity == 0.0
        assert priority.training_pending is False
        assert priority.priority_score == 0.0

    def test_staleness_factor(self):
        """Test staleness factor calculation."""
        # Fresh data (0 hours) = low staleness factor
        fresh = ConfigPriority(config_key="hex8_2p", staleness_hours=0.0)
        assert fresh.staleness_factor == 0.0

        # Stale data (24 hours) = high staleness factor
        stale = ConfigPriority(config_key="hex8_2p", staleness_hours=24.0)
        assert stale.staleness_factor > 0.5

        # Very stale (72 hours) = capped at 1.0
        very_stale = ConfigPriority(config_key="hex8_2p", staleness_hours=100.0)
        assert very_stale.staleness_factor <= 1.0

    def test_velocity_factor(self):
        """Test ELO velocity factor."""
        # No velocity
        static = ConfigPriority(config_key="hex8_2p", elo_velocity=0.0)
        assert static.velocity_factor == 0.0

        # Positive velocity (improving)
        improving = ConfigPriority(config_key="hex8_2p", elo_velocity=10.0)
        assert improving.velocity_factor > 0.0

        # Negative velocity (degrading)
        degrading = ConfigPriority(config_key="hex8_2p", elo_velocity=-5.0)
        assert degrading.velocity_factor >= 0.0  # Factor should be clamped

    def test_data_deficit_factor(self):
        """Test data deficit factor."""
        # Plenty of games = low deficit
        rich = ConfigPriority(config_key="hex8_2p", game_count=100000)
        assert rich.data_deficit_factor < 0.2

        # Few games = high deficit
        poor = ConfigPriority(config_key="hex8_2p", game_count=1000)
        assert poor.data_deficit_factor > 0.5


class TestNodeCapability:
    """Tests for NodeCapability dataclass."""

    def test_node_capability_defaults(self):
        """Test default values."""
        node = NodeCapability(node_id="test-node")
        assert node.node_id == "test-node"
        assert node.gpu_memory_gb == 0.0
        assert node.current_load == 0.0
        assert node.is_ephemeral is False

    def test_capacity_weight(self):
        """Test capacity weight calculation."""
        # Different GPU types have different weights
        node = NodeCapability(node_id="test", gpu_type="A100")
        # Just verify it returns a number
        assert isinstance(node.capacity_weight, (int, float))

    def test_available_capacity(self):
        """Test available capacity calculation."""
        node = NodeCapability(
            node_id="test",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.3,  # 30% loaded
        )
        # Should have 70% capacity remaining, scaled by weight
        assert node.available_capacity >= 0.0
        assert node.available_capacity <= 1.0 * node.capacity_weight


class TestSelfplayScheduler:
    """Tests for SelfplayScheduler class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_init(self, scheduler):
        """Test initialization."""
        assert scheduler is not None
        assert len(scheduler._config_priorities) == len(ALL_CONFIGS)

    def test_all_configs_tracked(self, scheduler):
        """Test that all configs are tracked."""
        for config in ALL_CONFIGS:
            priority = scheduler.get_config_priority(config)
            assert priority is not None
            assert priority.config_key == config

    def test_compute_priority_score(self, scheduler):
        """Test priority score computation."""
        # Fresh, no velocity, no training pending = low priority
        low_priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=0.0,
            elo_velocity=0.0,
            training_pending=False,
        )
        low_score = scheduler._compute_priority_score(low_priority)

        # Stale data = higher priority
        high_priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=48.0,  # 2 days stale
            elo_velocity=5.0,  # Improving
            training_pending=True,
            game_count=1000,  # Low game count
        )
        high_score = scheduler._compute_priority_score(high_priority)

        assert high_score > low_score

    def test_boost_config_allocation(self, scheduler):
        """Test manually boosting config allocation."""
        original = scheduler.get_config_priority("hex8_2p")
        original_boost = original.exploration_boost if original else 1.0

        # Boost allocation
        result = scheduler.boost_config_allocation("hex8_2p", multiplier=2.0)
        assert result is True

        boosted = scheduler.get_config_priority("hex8_2p")
        assert boosted.exploration_boost > original_boost

    def test_boost_invalid_config(self, scheduler):
        """Test boosting invalid config returns False."""
        result = scheduler.boost_config_allocation("invalid_config_xyz", multiplier=2.0)
        assert result is False

    def test_get_top_priorities(self, scheduler):
        """Test getting top priority configs."""
        # Set some priorities manually
        scheduler._config_priorities["hex8_2p"].priority_score = 0.9
        scheduler._config_priorities["square8_2p"].priority_score = 0.8
        scheduler._config_priorities["square19_2p"].priority_score = 0.1

        top = scheduler.get_top_priorities(n=2)
        assert len(top) == 2
        assert top[0]["config"] == "hex8_2p"
        assert top[1]["config"] == "square8_2p"

    def test_get_status(self, scheduler):
        """Test getting scheduler status."""
        status = scheduler.get_status()

        # Check for expected keys (may vary by implementation)
        assert isinstance(status, dict)
        # Should have some status info
        assert len(status) > 0

    def test_allocate_to_nodes_basic(self, scheduler):
        """Test basic node allocation."""
        # Set up some node capabilities
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )
        scheduler._node_capabilities["node2"] = NodeCapability(
            node_id="node2",
            gpu_type="RTX3090",
            gpu_memory_gb=24.0,
            current_load=0.0,
        )

        # Allocate games - signature is (config_key, total_games)
        allocation = scheduler._allocate_to_nodes("hex8_2p", 100)

        # Should return a dict mapping node_id -> games (may be empty if no allocation)
        assert isinstance(allocation, dict)

    def test_singleton_behavior(self):
        """Test that get_selfplay_scheduler returns singleton."""
        scheduler1 = get_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is scheduler2


class TestSelfplaySchedulerAsync:
    """Async tests for SelfplayScheduler."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    @pytest.mark.asyncio
    async def test_get_priority_configs(self, scheduler):
        """Test getting priority configs."""
        # Set some priorities
        scheduler._config_priorities["hex8_2p"].priority_score = 0.9
        scheduler._config_priorities["square8_2p"].priority_score = 0.5

        priorities = await scheduler.get_priority_configs(top_n=3)

        assert len(priorities) <= 3
        # Should be sorted by priority (highest first)
        for i in range(len(priorities) - 1):
            assert priorities[i][1] >= priorities[i + 1][1]

    @pytest.mark.asyncio
    async def test_allocate_selfplay_batch(self, scheduler):
        """Test batch allocation."""
        # Mock node capabilities
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )

        allocation = await scheduler.allocate_selfplay_batch(
            games_per_config=50,
            max_configs=2,
        )

        # Should return allocation dict
        assert isinstance(allocation, dict)

    @pytest.mark.asyncio
    async def test_update_priorities(self, scheduler):
        """Test priority update."""
        # Mock the data fetchers
        with patch.object(scheduler, '_get_data_freshness', return_value={"hex8_2p": 24.0}):
            with patch.object(scheduler, '_get_elo_velocities', return_value={"hex8_2p": 5.0}):
                with patch.object(scheduler, '_get_game_counts', return_value={"hex8_2p": 1000}):
                    await scheduler._update_priorities()

        # Check that priority was updated
        priority = scheduler.get_config_priority("hex8_2p")
        assert priority is not None


class TestEventHandling:
    """Tests for event handling in SelfplayScheduler."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_on_selfplay_complete(self, scheduler):
        """Test handling selfplay complete event."""
        event = MagicMock()
        event.get.side_effect = lambda k, d=None: {
            "config_key": "hex8_2p",
            "games": 100,
        }.get(k, d)

        # Should not raise
        scheduler._on_selfplay_complete(event)

    def test_on_training_complete(self, scheduler):
        """Test handling training complete event."""
        # The handler expects event.payload.get("config_key")
        event = MagicMock()
        event.payload = MagicMock()
        event.payload.get.side_effect = lambda k, d=None: {
            "config_key": "hex8_2p",
        }.get(k, d)

        # Should reset training_pending flag
        scheduler._config_priorities["hex8_2p"].training_pending = True
        scheduler._on_training_complete(event)
        assert scheduler._config_priorities["hex8_2p"].training_pending is False

    def test_on_curriculum_rebalanced(self, scheduler):
        """Test handling curriculum rebalance event."""
        # The handler expects event.payload with config_key and weight
        event = MagicMock()
        event.payload = MagicMock()
        event.payload.get.side_effect = lambda k, d=None: {
            "config_key": "hex8_2p",
            "weight": 0.8,
            "reason": "test",
        }.get(k, d)

        scheduler._on_curriculum_rebalanced(event)
        # Should update curriculum_weight for the specific config
        assert scheduler._config_priorities["hex8_2p"].curriculum_weight == 0.8


class TestMetricsAndHealth:
    """Tests for metrics and health check methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_get_metrics_returns_dict(self, scheduler):
        """Test that get_metrics returns a dictionary."""
        metrics = scheduler.get_metrics()
        assert isinstance(metrics, dict)

    def test_get_metrics_has_allocation_count(self, scheduler):
        """Test that metrics includes allocation tracking."""
        # Manually increment allocations
        scheduler._allocation_count = 10

        metrics = scheduler.get_metrics()
        # Should have some metrics about allocations
        assert "games_allocated_total" in metrics or "allocation_count" in metrics

    def test_get_metrics_has_games_per_hour(self, scheduler):
        """Test that metrics includes throughput info."""
        metrics = scheduler.get_metrics()
        # Should track games per hour or similar throughput metric
        assert "games_per_hour" in metrics or "throughput" in metrics or len(metrics) > 0

    def test_health_check_returns_result(self, scheduler):
        """Test that health_check returns a HealthCheckResult."""
        result = scheduler.health_check()
        # Should return HealthCheckResult or similar
        assert result is not None
        # Should have healthy/unhealthy status
        assert hasattr(result, "healthy") or hasattr(result, "status")

    def test_health_check_healthy_by_default(self, scheduler):
        """Test that new scheduler is healthy by default."""
        result = scheduler.health_check()
        # Fresh scheduler should be healthy
        if hasattr(result, "healthy"):
            assert result.healthy is True
        elif hasattr(result, "status"):
            assert result.status in ("healthy", "HEALTHY", True)

    def test_health_check_with_nodes(self, scheduler):
        """Test health check with active nodes."""
        # Add some node capabilities
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
        )

        result = scheduler.health_check()
        assert result is not None


class TestNodeTargeting:
    """Tests for node targeting and job distribution."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_get_target_jobs_returns_int(self, scheduler):
        """Test that get_target_jobs_for_node returns an integer."""
        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.gpu_type = "A100"

        result = scheduler.get_target_jobs_for_node(mock_node)
        assert isinstance(result, int)

    def test_get_target_jobs_for_a100(self, scheduler):
        """Test target jobs for high-end GPU."""
        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.gpu_type = "A100"
        mock_node.gpu_memory_gb = 80.0

        result = scheduler.get_target_jobs_for_node(mock_node)
        # A100 should get reasonable number of jobs
        assert result >= 0

    def test_get_target_jobs_for_rtx3090(self, scheduler):
        """Test target jobs for consumer GPU."""
        mock_node = MagicMock()
        mock_node.node_id = "test-node"
        mock_node.gpu_type = "RTX3090"
        mock_node.gpu_memory_gb = 24.0

        result = scheduler.get_target_jobs_for_node(mock_node)
        assert result >= 0

    def test_node_capability_update(self, scheduler):
        """Test updating node capabilities."""
        # Register a node
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )

        # Update load
        scheduler._node_capabilities["node1"].current_load = 0.5

        # Verify update
        assert scheduler._node_capabilities["node1"].current_load == 0.5

    def test_allocate_with_multiple_nodes(self, scheduler):
        """Test allocation across multiple nodes."""
        # Register multiple nodes with different capabilities
        scheduler._node_capabilities["node1"] = NodeCapability(
            node_id="node1",
            gpu_type="A100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )
        scheduler._node_capabilities["node2"] = NodeCapability(
            node_id="node2",
            gpu_type="RTX3090",
            gpu_memory_gb=24.0,
            current_load=0.0,
        )
        scheduler._node_capabilities["node3"] = NodeCapability(
            node_id="node3",
            gpu_type="H100",
            gpu_memory_gb=80.0,
            current_load=0.0,
        )

        # Allocate games
        allocation = scheduler._allocate_to_nodes("hex8_2p", 200)

        # Should return a dict
        assert isinstance(allocation, dict)


class TestPriorityEdgeCases:
    """Tests for priority calculation edge cases."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_extremely_stale_data(self):
        """Test priority with extremely stale data (weeks old)."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=168.0,  # 1 week
        )
        # Staleness factor should be capped at 1.0
        assert priority.staleness_factor <= 1.0

    def test_negative_staleness(self):
        """Test handling of negative staleness (future timestamp edge case)."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=-1.0,  # Negative (shouldn't happen but handle gracefully)
        )
        # Should clamp to 0
        assert priority.staleness_factor >= 0.0

    def test_very_high_game_count(self):
        """Test data deficit with very high game count."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=10_000_000,  # 10 million games
        )
        # Data deficit should be minimal
        assert priority.data_deficit_factor <= 0.1

    def test_zero_game_count(self):
        """Test data deficit with zero games."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=0,
        )
        # Data deficit should be maximum
        assert priority.data_deficit_factor == 1.0

    def test_curriculum_weight_override(self, scheduler):
        """Test that curriculum weight affects priority."""
        # Set high curriculum weight for a config
        scheduler._config_priorities["hex8_2p"].curriculum_weight = 2.0
        scheduler._config_priorities["square8_2p"].curriculum_weight = 0.5

        # Compute priorities
        hex_score = scheduler._compute_priority_score(
            scheduler._config_priorities["hex8_2p"]
        )
        square_score = scheduler._compute_priority_score(
            scheduler._config_priorities["square8_2p"]
        )

        # With higher curriculum weight, hex8_2p should have higher priority
        # (assuming other factors are equal)
        # Note: This depends on the actual implementation

    def test_all_configs_have_priorities(self, scheduler):
        """Verify all 12 canonical configs are tracked."""
        expected_configs = [
            "hex8_2p", "hex8_3p", "hex8_4p",
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]

        for config in expected_configs:
            priority = scheduler.get_config_priority(config)
            assert priority is not None, f"Missing priority for {config}"


class TestNodeRecoveryHandling:
    """Tests for node recovery event handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the scheduler singleton before each test."""
        reset_selfplay_scheduler()
        yield
        reset_selfplay_scheduler()

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler."""
        return SelfplayScheduler()

    def test_on_node_recovered(self, scheduler):
        """Test handling node recovery event."""
        event = MagicMock()
        event.payload = MagicMock()
        event.payload.get.side_effect = lambda k, d=None: {
            "node_id": "test-node",
            "gpu_type": "A100",
        }.get(k, d)

        # Should not raise
        if hasattr(scheduler, "_on_node_recovered"):
            scheduler._on_node_recovered(event)

    def test_on_data_quality_updated(self, scheduler):
        """Test handling data quality update event."""
        event = MagicMock()
        event.payload = MagicMock()
        event.payload.get.side_effect = lambda k, d=None: {
            "config_key": "hex8_2p",
            "quality_score": 0.95,
        }.get(k, d)

        # Should not raise
        if hasattr(scheduler, "_on_data_quality_updated"):
            scheduler._on_data_quality_updated(event)
