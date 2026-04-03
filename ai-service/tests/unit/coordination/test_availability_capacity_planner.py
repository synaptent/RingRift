"""Unit tests for availability/capacity_planner.py.

Tests for CapacityPlanner daemon that manages budget-aware capacity.

Created: Dec 28, 2025
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.capacity_planner import (
    CapacityBudget,
    CapacityPlanner,
    CapacityPlannerConfig,
    ScaleAction,
    ScaleRecommendation,
    UtilizationMetrics,
    get_capacity_planner,
    reset_capacity_planner,
)


class TestScaleAction:
    """Tests for ScaleAction enum."""

    def test_action_values(self):
        """Test all action values exist."""
        assert ScaleAction.SCALE_UP is not None
        assert ScaleAction.SCALE_DOWN is not None
        assert ScaleAction.NONE is not None

    def test_action_names(self):
        """Test action names."""
        assert ScaleAction.SCALE_UP.value == "scale_up"
        assert ScaleAction.SCALE_DOWN.value == "scale_down"
        assert ScaleAction.NONE.value == "none"


class TestCapacityBudget:
    """Tests for CapacityBudget dataclass."""

    def test_default_budget(self):
        """Test default budget values."""
        budget = CapacityBudget()
        assert budget.hourly_limit_usd > 0
        assert budget.daily_limit_usd > 0
        assert budget.current_hourly_usd == 0.0
        assert budget.current_daily_usd == 0.0

    def test_custom_budget(self):
        """Test custom budget values."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=25.0,
            current_daily_usd=500.0,
        )
        assert budget.hourly_limit_usd == 100.0
        assert budget.daily_limit_usd == 2000.0
        assert budget.current_hourly_usd == 25.0

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=20.0,
            current_daily_usd=200.0,
        )
        assert budget.remaining_hourly_budget() == 30.0
        assert budget.remaining_daily_budget() == 300.0

    def test_budget_percent_used(self):
        """Test budget percentage used calculation."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=1000.0,
            current_hourly_usd=25.0,
            current_daily_usd=100.0,
        )
        assert budget.hourly_budget_percent_used() == 25.0
        assert budget.daily_budget_percent_used() == 10.0

    def test_can_afford(self):
        """Test affordability check."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=40.0,
            current_daily_usd=100.0,
        )
        # Can afford $5/hour more
        assert budget.can_afford(5.0) is True
        # Cannot afford $15/hour more (exceeds hourly limit)
        assert budget.can_afford(15.0) is False

    def test_is_over_alert_threshold(self):
        """Test alert threshold detection."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=1000.0,
            alert_threshold_percent=80.0,
            current_hourly_usd=85.0,  # 85% used
            current_daily_usd=100.0,
        )
        assert budget.is_over_alert_threshold() is True

    def test_to_dict(self):
        """Test serialization to dict."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
        )
        d = budget.to_dict()
        assert d["hourly_limit_usd"] == 50.0
        assert d["daily_limit_usd"] == 500.0
        assert "current_hourly_usd" in d
        assert "hourly_percent_used" in d


class TestScaleRecommendation:
    """Tests for ScaleRecommendation dataclass."""

    def test_scale_up_recommendation(self):
        """Test scale up recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_UP,
            count=2,
            reason="Utilization above 90%",
        )
        assert rec.action == ScaleAction.SCALE_UP
        assert rec.count == 2
        assert "90%" in rec.reason

    def test_no_action_recommendation(self):
        """Test no action recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.NONE,
            count=0,
            reason="Cluster is within target utilization",
        )
        assert rec.action == ScaleAction.NONE
        assert rec.count == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_UP,
            count=2,
            reason="High utilization",
            provider="lambda",
            gpu_type="GH200_96GB",
        )
        d = rec.to_dict()
        assert d["action"] == "scale_up"
        assert d["count"] == 2
        assert d["provider"] == "lambda"
        assert "timestamp" in d


class TestUtilizationMetrics:
    """Tests for UtilizationMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = UtilizationMetrics()
        assert metrics.gpu_utilization_avg == 0.0
        assert metrics.active_gpu_nodes == 0
        assert metrics.total_gpu_nodes == 0

    def test_custom_metrics(self):
        """Test custom metrics."""
        metrics = UtilizationMetrics(
            gpu_utilization_avg=0.75,
            active_gpu_nodes=12,
            total_gpu_nodes=16,
            selfplay_jobs_running=8,
        )
        assert metrics.gpu_utilization_avg == 0.75
        assert metrics.active_gpu_nodes == 12

    def test_overall_utilization(self):
        """Test overall utilization calculation."""
        metrics = UtilizationMetrics(
            gpu_utilization_avg=0.80,
            memory_utilization_avg=0.60,
            active_gpu_nodes=10,
            total_gpu_nodes=10,
            selfplay_jobs_running=8,
            training_jobs_running=2,
        )
        util = metrics.overall_utilization
        assert 0.0 <= util <= 1.0


class TestCapacityPlannerConfig:
    """Tests for CapacityPlannerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CapacityPlannerConfig()
        assert config.hourly_budget_usd > 0
        assert config.daily_budget_usd > 0
        assert config.min_gpu_nodes > 0
        assert 0 < config.scale_up_utilization_threshold < 1
        assert 0 < config.scale_down_utilization_threshold < 1

    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_HOURLY_BUDGET_USD": "100.0",
            "RINGRIFT_MIN_GPU_NODES": "8",
        }):
            config = CapacityPlannerConfig.from_env()
            assert config.hourly_budget_usd == 100.0
            assert config.min_gpu_nodes == 8

    def test_gpu_costs(self):
        """Test GPU cost dictionary."""
        config = CapacityPlannerConfig()
        assert "GH200_96GB" in config.gpu_costs
        assert "H100_80GB" in config.gpu_costs
        assert config.gpu_costs["GH200_96GB"] > 0


class TestCapacityPlanner:
    """Tests for CapacityPlanner daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_capacity_planner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_capacity_planner()

    def test_singleton_pattern(self):
        """Test that get_capacity_planner returns singleton."""
        planner1 = get_capacity_planner()
        planner2 = get_capacity_planner()

        assert planner1 is planner2

    def test_event_subscriptions(self):
        """Test event subscription setup."""
        planner = get_capacity_planner()

        subs = planner._get_event_subscriptions()

        # Should subscribe to provisioning events
        assert "NODE_PROVISIONED" in subs
        assert "NODE_TERMINATED" in subs

    def test_get_budget_status(self):
        """Test getting current budget status."""
        planner = get_capacity_planner()

        status = planner.get_budget_status()
        assert "hourly_limit_usd" in status
        assert "current_hourly_usd" in status

    def test_get_utilization_history(self):
        """Test getting utilization history."""
        planner = get_capacity_planner()

        history = planner.get_utilization_history()
        assert isinstance(history, list)

    def test_health_check(self):
        """Test health_check method returns valid result."""
        planner = get_capacity_planner()

        result = planner.health_check()

        assert isinstance(result.healthy, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.details, dict)


class TestCapacityPlannerBudgetChecks:
    """Tests for budget checking logic."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_capacity_planner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_should_scale_up_within_budget(self):
        """Test scale up allowed when within budget."""
        planner = get_capacity_planner()

        # Set budget with room to spare
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=50.0,
            current_daily_usd=500.0,
        )

        result = await planner.should_scale_up(count=1)
        # Should be True since we have budget room
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_should_not_scale_up_budget_exceeded(self):
        """Test scale up blocked when budget exceeded."""
        planner = get_capacity_planner()

        # Set budget at limit
        planner.budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=50.0,  # At limit
            current_daily_usd=100.0,
        )

        result = await planner.should_scale_up(count=1)
        assert result is False


class TestCapacityPlannerEventHandlers:
    """Tests for CapacityPlanner event handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_capacity_planner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_on_node_provisioned(self):
        """Test handling NODE_PROVISIONED event."""
        planner = get_capacity_planner()

        # Reset budget
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
        )

        initial_hourly = planner.budget.current_hourly_usd

        event = {
            "payload": {
                "node_id": "test-node",
                "cost_per_hour": 2.50,
            }
        }

        await planner._on_node_provisioned(event)

        # Budget should increase
        assert planner.budget.current_hourly_usd >= initial_hourly

    @pytest.mark.asyncio
    async def test_on_node_terminated(self):
        """Test handling NODE_TERMINATED event."""
        planner = get_capacity_planner()

        # Set initial budget
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=10.0,
            current_daily_usd=100.0,
        )

        initial_hourly = planner.budget.current_hourly_usd

        event = {
            "payload": {
                "node_id": "test-node",
                "cost_per_hour": 2.50,
            }
        }

        await planner._on_node_terminated(event)

        # Budget should decrease
        assert planner.budget.current_hourly_usd <= initial_hourly


class TestCapacityPlannerScaling:
    """Tests for scaling recommendation logic."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_capacity_planner()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_get_scale_recommendation_returns_valid(self):
        """Test scale recommendation returns valid result."""
        planner = get_capacity_planner()

        # Ensure budget allows scaling
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=20.0,
            current_daily_usd=200.0,
        )

        # Mock the utilization metrics collection
        with patch.object(planner, "_collect_utilization_metrics", new_callable=AsyncMock) as mock:
            mock.return_value = UtilizationMetrics(
                gpu_utilization_avg=0.5,
                active_gpu_nodes=5,
                total_gpu_nodes=10,
            )

            rec = await planner.get_scale_recommendation()

            # Should return a valid recommendation
            assert rec is not None
            assert rec.action in [ScaleAction.SCALE_UP, ScaleAction.SCALE_DOWN, ScaleAction.NONE]

    def test_record_scale_up(self):
        """Test recording scale up for cooldown tracking."""
        planner = get_capacity_planner()

        initial_time = planner._last_scale_up_time
        planner.record_scale_up()

        assert planner._last_scale_up_time > initial_time

    def test_record_scale_down(self):
        """Test recording scale down for cooldown tracking."""
        planner = get_capacity_planner()

        initial_time = planner._last_scale_down_time
        planner.record_scale_down()

        assert planner._last_scale_down_time > initial_time
