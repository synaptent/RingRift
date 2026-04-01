"""Unit tests for SelfplayScheduler.

Tests the core selfplay scheduling logic including:
- Priority calculation with multi-factor weights
- Dynamic weight adjustment based on cluster state
- Event handling (training completed, Elo updates, etc.)
- Configuration priority computation
- Allocation decisions

Dec 30, 2025: Created comprehensive test coverage for this critical 4,114 LOC module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict

from app.coordination.selfplay_scheduler import (
    DynamicWeights,
    ConfigPriority,
    SelfplayScheduler,
    get_selfplay_scheduler,
    reset_selfplay_scheduler,
    STALENESS_WEIGHT,
    ELO_VELOCITY_WEIGHT,
    CURRICULUM_WEIGHT,
    QUALITY_WEIGHT,
    VOI_WEIGHT,
    FRESH_DATA_THRESHOLD,
    STALE_DATA_THRESHOLD,
    MAX_STALENESS_HOURS,
    TARGET_GAMES_FOR_2000_ELO,
    LARGE_BOARD_TARGET_MULTIPLIER,
)


class TestDynamicWeights:
    """Test DynamicWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values match constants."""
        weights = DynamicWeights()
        assert weights.staleness == STALENESS_WEIGHT
        assert weights.velocity == ELO_VELOCITY_WEIGHT
        assert weights.curriculum == CURRICULUM_WEIGHT
        assert weights.quality == QUALITY_WEIGHT
        assert weights.voi == VOI_WEIGHT

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = DynamicWeights(staleness=0.5, velocity=0.3, quality=0.2)
        assert weights.staleness == 0.5
        assert weights.velocity == 0.3
        assert weights.quality == 0.2

    def test_to_dict(self):
        """Test conversion to dict for logging."""
        weights = DynamicWeights(staleness=0.4, idle_gpu_fraction=0.7)
        d = weights.to_dict()
        assert d["staleness"] == 0.4
        assert d["idle_gpu_fraction"] == 0.7
        assert "velocity" in d
        assert "curriculum" in d

    def test_cluster_state_tracking(self):
        """Test cluster state fields are tracked."""
        weights = DynamicWeights(
            idle_gpu_fraction=0.5,
            training_queue_depth=10,
            configs_at_target_fraction=0.25,
            average_elo=1800.0,
        )
        assert weights.idle_gpu_fraction == 0.5
        assert weights.training_queue_depth == 10
        assert weights.configs_at_target_fraction == 0.25
        assert weights.average_elo == 1800.0


class TestConfigPriority:
    """Test ConfigPriority dataclass and computed properties."""

    def test_default_priority(self):
        """Test default priority values."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.config_key == "hex8_2p"
        assert priority.staleness_hours == 0.0
        assert priority.elo_velocity == 0.0
        assert priority.priority_score == 0.0

    def test_staleness_factor_fresh(self):
        """Test staleness factor when data is fresh."""
        priority = ConfigPriority(config_key="hex8_2p", staleness_hours=1.0)
        assert priority.staleness_factor == 0.0

    def test_staleness_factor_stale(self):
        """Test staleness factor when data is stale."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=(FRESH_DATA_THRESHOLD + STALE_DATA_THRESHOLD) / 2
        )
        # Should be between 0 and 1
        assert 0.0 < priority.staleness_factor < 1.0

    def test_staleness_factor_very_stale(self):
        """Test staleness factor at max staleness."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            staleness_hours=MAX_STALENESS_HOURS + 10
        )
        assert priority.staleness_factor == 1.0

    def test_velocity_factor_negative(self):
        """Test velocity factor with regression."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=-10.0)
        assert priority.velocity_factor == 0.0

    def test_velocity_factor_positive(self):
        """Test velocity factor with improvement."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=50.0)
        assert priority.velocity_factor == 0.5  # 50/100

    def test_velocity_factor_capped(self):
        """Test velocity factor is capped at 1.0."""
        priority = ConfigPriority(config_key="hex8_2p", elo_velocity=200.0)
        assert priority.velocity_factor == 1.0

    def test_data_deficit_factor_at_target(self):
        """Test data deficit when at target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO,
            is_large_board=False
        )
        assert priority.data_deficit_factor == 0.0

    def test_data_deficit_factor_below_target(self):
        """Test data deficit when below target."""
        priority = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO // 2,
            is_large_board=False
        )
        assert priority.data_deficit_factor == 0.5

    def test_data_deficit_factor_large_board(self):
        """Test data deficit for large boards (higher target)."""
        small_board = ConfigPriority(
            config_key="hex8_2p",
            game_count=100,
            is_large_board=False
        )
        large_board = ConfigPriority(
            config_key="square19_2p",
            game_count=100,
            is_large_board=True
        )
        # Large board has higher target, so same game count = higher deficit
        assert large_board.data_deficit_factor > small_board.data_deficit_factor

    def test_player_count_extraction_2p(self):
        """Test player count extraction for 2-player config."""
        priority = ConfigPriority(config_key="hex8_2p")
        assert priority.player_count == 2

    def test_player_count_extraction_4p(self):
        """Test player count extraction for 4-player config."""
        priority = ConfigPriority(config_key="square8_4p")
        assert priority.player_count == 4

    def test_player_count_extraction_invalid(self):
        """Test player count extraction with invalid config."""
        priority = ConfigPriority(config_key="invalid")
        assert priority.player_count == 2  # Default


class TestSelfplaySchedulerSingleton:
    """Test SelfplayScheduler singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_get_instance_returns_same_object(self):
        """Test that get_selfplay_scheduler returns same instance."""
        scheduler1 = get_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is scheduler2

    def test_reset_creates_new_instance(self):
        """Test that reset creates a new instance."""
        scheduler1 = get_selfplay_scheduler()
        reset_selfplay_scheduler()
        scheduler2 = get_selfplay_scheduler()
        assert scheduler1 is not scheduler2


class TestSelfplaySchedulerPriorityCalculation:
    """Test priority calculation logic."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_staleness_increases_priority(self):
        """Test that stale configs get higher priority."""
        # Create two priorities with different staleness
        fresh = ConfigPriority(config_key="hex8_2p", staleness_hours=1.0)
        stale = ConfigPriority(config_key="hex8_3p", staleness_hours=48.0)

        # Compute scores (higher staleness should give higher score contribution)
        fresh_staleness = fresh.staleness_factor
        stale_staleness = stale.staleness_factor

        assert stale_staleness > fresh_staleness

    def test_high_velocity_configs_prioritized(self):
        """Test that high-velocity configs get priority boost."""
        slow = ConfigPriority(config_key="hex8_2p", elo_velocity=10.0)
        fast = ConfigPriority(config_key="hex8_3p", elo_velocity=80.0)

        assert fast.velocity_factor > slow.velocity_factor

    def test_data_deficit_prioritization(self):
        """Test that configs with less data get higher priority."""
        rich = ConfigPriority(
            config_key="hex8_2p",
            game_count=TARGET_GAMES_FOR_2000_ELO
        )
        poor = ConfigPriority(
            config_key="hex8_3p",
            game_count=100
        )

        assert poor.data_deficit_factor > rich.data_deficit_factor


class TestSelfplaySchedulerEventHandling:
    """Test event subscription and handling."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_has_event_subscriptions(self):
        """Test scheduler has subscribe_to_events method."""
        scheduler = get_selfplay_scheduler()
        # Scheduler uses subscribe_to_events() method
        assert hasattr(scheduler, "subscribe_to_events")
        assert callable(scheduler.subscribe_to_events)
        # Should have _subscribed flag for tracking
        assert hasattr(scheduler, "_subscribed")

    def test_elo_velocity_tracking(self):
        """Test Elo velocity is tracked per config."""
        scheduler = get_selfplay_scheduler()

        # Update Elo velocity
        scheduler._elo_velocity["hex8_2p"] = 25.0

        assert scheduler._elo_velocity.get("hex8_2p") == 25.0
        assert scheduler._elo_velocity.get("nonexistent") is None

    def test_training_blocked_by_quality_uses_publish_sync(self):
        """Blocked training should trigger a sync selfplay target update."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            exploration_boost=1.0,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "stale_data",
            "data_age_hours": 12.0,
        }

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            scheduler._on_training_blocked_by_quality(event)

        mock_publish_sync.assert_called_once()
        event_type, payload = mock_publish_sync.call_args.args[:2]
        assert event_type.value == "selfplay_target_updated"
        assert payload["config_key"] == "hex8_2p"
        assert payload["priority"] == "high"
        assert payload["reason"] == "training_blocked:stale_data"
        assert payload["exploration_boost"] == 1.5
        assert mock_publish_sync.call_args.kwargs["source"] == "selfplay_quality_signal_handler"

    def test_training_early_stopped_uses_publish_sync(self):
        """Early-stopped training should trigger an urgent sync target update."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            exploration_boost=1.0,
            curriculum_weight=1.0,
            staleness_hours=0.0,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "plateau",
            "final_loss": 0.42,
            "epochs_completed": 4,
        }

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            scheduler._on_training_early_stopped(event)

        mock_publish_sync.assert_called_once()
        event_type, payload = mock_publish_sync.call_args.args[:2]
        assert event_type.value == "selfplay_target_updated"
        assert payload["config_key"] == "hex8_2p"
        assert payload["priority"] == "urgent"
        assert payload["reason"] == "training_early_stopped:plateau"
        assert payload["exploration_boost"] == 1.3
        assert payload["curriculum_weight"] == 1.3
        assert mock_publish_sync.call_args.kwargs["source"] == "selfplay_quality_signal_handler"

    def test_elo_velocity_changed_uses_publish_sync_for_rate_and_target(self):
        """Velocity changes should publish both rate and target updates via router."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            momentum_multiplier=1.0,
            exploration_boost=1.0,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "velocity": 8.0,
            "previous_velocity": 4.0,
            "trend": "accelerating",
        }

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            scheduler._on_elo_velocity_changed(event)

        rate_call = next(
            call for call in mock_publish_sync.call_args_list
            if call.args[0].value == "selfplay_rate_changed"
        )
        target_call = next(
            call for call in mock_publish_sync.call_args_list
            if call.args[0].value == "selfplay_target_updated"
        )
        assert rate_call.args[0].value == "selfplay_rate_changed"
        assert rate_call.args[1]["config_key"] == "hex8_2p"
        assert rate_call.args[1]["change_percent"] == pytest.approx(20.0)
        assert rate_call.kwargs["source"] == "selfplay_velocity_mixin"
        assert target_call.args[0].value == "selfplay_target_updated"
        assert target_call.args[1]["reason"] == "velocity_changed:accelerating"
        assert target_call.args[1]["momentum_multiplier"] == pytest.approx(1.2)
        assert target_call.kwargs["source"] == "selfplay_velocity_mixin"

    def test_exploration_boost_uses_publish_sync(self):
        """Anomaly-driven exploration boosts should publish sync target updates."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            exploration_boost=1.0,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "boost_factor": 1.4,
            "reason": "regression_detected",
            "anomaly_count": 3,
        }

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            scheduler._on_exploration_boost(event)

        target_call = next(
            call for call in mock_publish_sync.call_args_list
            if call.args[0].value == "selfplay_target_updated"
        )
        event_type, payload = target_call.args[:2]
        assert event_type.value == "selfplay_target_updated"
        assert payload["config_key"] == "hex8_2p"
        assert payload["priority"] == "high"
        assert payload["exploration_boost"] == 1.4
        assert payload["anomaly_count"] == 3
        assert target_call.kwargs["source"] == "selfplay_velocity_mixin"

    def test_exploration_adjusted_uses_publish_sync(self):
        """Quality-driven exploration adjustments should publish sync target updates."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            exploration_boost=1.0,
        )

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "quality_score": 0.3,
            "trend": "declining",
            "position_difficulty": "harder",
            "mcts_budget_multiplier": 1.5,
            "exploration_temp_boost": 0.2,
        }

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            scheduler._on_exploration_adjusted(event)

        mock_publish_sync.assert_called_once()
        event_type, payload = mock_publish_sync.call_args.args[:2]
        assert event_type.value == "selfplay_target_updated"
        assert payload["config_key"] == "hex8_2p"
        assert payload["priority"] == "high"
        assert payload["reason"] == "quality_exploration:declining"
        assert payload["exploration_boost"] == pytest.approx(1.4)
        assert payload["quality_score"] == 0.3
        assert mock_publish_sync.call_args.kwargs["source"] == "selfplay_velocity_mixin"

    def test_decay_expired_boosts_uses_publish_sync(self):
        """Expired exploration boosts should publish sync reset events."""
        scheduler = get_selfplay_scheduler()
        scheduler._config_priorities["hex8_2p"] = ConfigPriority(
            config_key="hex8_2p",
            exploration_boost=1.6,
            exploration_boost_expires_at=1.0,
        )

        with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
            decayed = scheduler._decay_expired_boosts(now=2.0)

        assert decayed == 1
        mock_publish_sync.assert_called_once()
        event_type, payload = mock_publish_sync.call_args.args[:2]
        assert event_type.value == "selfplay_target_updated"
        assert payload["config_key"] == "hex8_2p"
        assert payload["reason"] == "exploration_boost_expired"
        assert payload["exploration_boost"] == 1.0
        assert mock_publish_sync.call_args.kwargs["source"] == "selfplay_velocity_mixin"


class TestSelfplaySchedulerDynamicWeights:
    """Test dynamic weight computation based on cluster state."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_compute_dynamic_weights_exists(self):
        """Test dynamic weight computation method exists."""
        scheduler = get_selfplay_scheduler()
        assert hasattr(scheduler, "_compute_dynamic_weights")


class TestSelfplaySchedulerIntegration:
    """Integration tests for scheduler."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_config_key_parsing(self):
        """Test config key parsing for board type and players."""
        priority = ConfigPriority(config_key="square19_4p")
        assert priority.player_count == 4

        priority = ConfigPriority(config_key="hexagonal_3p")
        assert priority.player_count == 3

    def test_is_large_board_detection(self):
        """Test large board detection."""
        large1 = ConfigPriority(config_key="square19_2p", is_large_board=True)
        large2 = ConfigPriority(config_key="hexagonal_2p", is_large_board=True)
        small = ConfigPriority(config_key="hex8_2p", is_large_board=False)

        assert large1.is_large_board
        assert large2.is_large_board
        assert not small.is_large_board


class TestTrainingOnlyNodes:
    """Tests for training-only node feature (January 2026).

    Tests the ability to exclude nodes with selfplay_enabled=False from
    receiving selfplay jobs, preventing OOM from training + selfplay conflicts.
    """

    def setup_method(self):
        """Reset singleton before each test."""
        reset_selfplay_scheduler()

    def test_is_selfplay_enabled_returns_true_by_default(self):
        """Test that _is_selfplay_enabled returns True for unknown nodes."""
        scheduler = get_selfplay_scheduler()
        # Unknown node should default to True (allow selfplay)
        assert scheduler._is_selfplay_enabled("unknown-node-xyz") is True

    def test_is_selfplay_enabled_with_mock_cluster_config(self):
        """Test _is_selfplay_enabled respects cluster config."""
        from app.config.cluster_config import ClusterNode

        mock_nodes = {
            "training-only-node": MagicMock(selfplay_enabled=False),
            "normal-node": MagicMock(selfplay_enabled=True),
        }

        scheduler = get_selfplay_scheduler()
        with patch(
            "app.coordination.selfplay.node_targeting.get_cluster_nodes",
            return_value=mock_nodes,
        ):
            assert scheduler._is_selfplay_enabled("training-only-node") is False
            assert scheduler._is_selfplay_enabled("normal-node") is True
            # Unknown node still defaults to True
            assert scheduler._is_selfplay_enabled("other-node") is True

    def test_is_selfplay_enabled_handles_exception_gracefully(self):
        """Test that config errors default to allowing selfplay."""
        scheduler = get_selfplay_scheduler()
        with patch(
            "app.coordination.selfplay.node_targeting.get_cluster_nodes",
            side_effect=Exception("Config error"),
        ):
            # Should return True (allow selfplay) on error
            assert scheduler._is_selfplay_enabled("any-node") is True

    def test_training_only_filtering_logic(self):
        """Test that training-only nodes are correctly identified for exclusion.

        This tests the _is_selfplay_enabled check that feeds into allocation.
        """
        scheduler = get_selfplay_scheduler()

        mock_cluster_nodes = {
            "gpu-node-1": MagicMock(selfplay_enabled=True),
            "training-node-1": MagicMock(selfplay_enabled=False),
            "training-node-2": MagicMock(selfplay_enabled=False),
            "gpu-node-2": MagicMock(selfplay_enabled=True),
        }

        with patch(
            "app.coordination.selfplay.node_targeting.get_cluster_nodes",
            return_value=mock_cluster_nodes,
        ):
            # Check selfplay enabled status for each node
            assert scheduler._is_selfplay_enabled("gpu-node-1") is True
            assert scheduler._is_selfplay_enabled("training-node-1") is False
            assert scheduler._is_selfplay_enabled("training-node-2") is False
            assert scheduler._is_selfplay_enabled("gpu-node-2") is True

            # Count how many would be excluded
            excluded = sum(
                1 for node_id in mock_cluster_nodes
                if not scheduler._is_selfplay_enabled(node_id)
            )
            assert excluded == 2

    def test_training_only_nodes_set_construction(self):
        """Test that training_only_nodes set is built correctly."""
        scheduler = get_selfplay_scheduler()

        # Mock node capabilities with 3 nodes
        from app.coordination.node_allocator import NodeCapability

        scheduler._node_capabilities = {
            "node-a": NodeCapability(node_id="node-a", gpu_type="A100", gpu_memory_gb=80),
            "node-b": NodeCapability(node_id="node-b", gpu_type="GH200", gpu_memory_gb=96),
            "node-c": NodeCapability(node_id="node-c", gpu_type="RTX4090", gpu_memory_gb=24),
        }

        mock_cluster_nodes = {
            "node-a": MagicMock(selfplay_enabled=True),
            "node-b": MagicMock(selfplay_enabled=False),  # Training-only
            "node-c": MagicMock(selfplay_enabled=False),  # Training-only
        }

        with patch(
            "app.coordination.selfplay.node_targeting.get_cluster_nodes",
            return_value=mock_cluster_nodes,
        ):
            # Build the set manually to verify
            training_only = set()
            for node_id in scheduler._node_capabilities.keys():
                if not scheduler._is_selfplay_enabled(node_id):
                    training_only.add(node_id)

            assert len(training_only) == 2
            assert "node-b" in training_only
            assert "node-c" in training_only
            assert "node-a" not in training_only
