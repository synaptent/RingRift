"""Comprehensive tests for capacity_planner.py.

December 29, 2025: Created to test budget-aware capacity planning.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.capacity_planner import (
    CapacityBudget,
    CapacityPlanner,
    CapacityPlannerConfig,
    ScaleRecommendation,
    UtilizationMetrics,
    get_capacity_planner,
    reset_capacity_planner,
)
from app.coordination.enums import ScaleAction


# =============================================================================
# CapacityBudget Tests
# =============================================================================


class TestCapacityBudget:
    """Tests for CapacityBudget dataclass."""

    def test_default_values(self) -> None:
        """Test default budget limits."""
        budget = CapacityBudget()
        assert budget.hourly_limit_usd == 50.0
        assert budget.daily_limit_usd == 500.0
        assert budget.alert_threshold_percent == 80.0
        assert budget.current_hourly_usd == 0.0
        assert budget.current_daily_usd == 0.0

    def test_remaining_hourly_budget(self) -> None:
        """Test remaining_hourly_budget calculation."""
        budget = CapacityBudget(hourly_limit_usd=100.0, current_hourly_usd=30.0)
        assert budget.remaining_hourly_budget() == 70.0

    def test_remaining_hourly_budget_negative_clamps_to_zero(self) -> None:
        """Test that remaining budget never goes negative."""
        budget = CapacityBudget(hourly_limit_usd=50.0, current_hourly_usd=100.0)
        assert budget.remaining_hourly_budget() == 0.0

    def test_remaining_daily_budget(self) -> None:
        """Test remaining_daily_budget calculation."""
        budget = CapacityBudget(daily_limit_usd=1000.0, current_daily_usd=400.0)
        assert budget.remaining_daily_budget() == 600.0

    def test_remaining_daily_budget_negative_clamps_to_zero(self) -> None:
        """Test that remaining daily budget never goes negative."""
        budget = CapacityBudget(daily_limit_usd=100.0, current_daily_usd=200.0)
        assert budget.remaining_daily_budget() == 0.0

    def test_hourly_budget_percent_used(self) -> None:
        """Test hourly_budget_percent_used calculation."""
        budget = CapacityBudget(hourly_limit_usd=100.0, current_hourly_usd=40.0)
        assert budget.hourly_budget_percent_used() == 40.0

    def test_hourly_budget_percent_used_zero_limit(self) -> None:
        """Test that zero limit returns 100%."""
        budget = CapacityBudget(hourly_limit_usd=0.0, current_hourly_usd=0.0)
        assert budget.hourly_budget_percent_used() == 100.0

    def test_hourly_budget_percent_used_negative_limit(self) -> None:
        """Test that negative limit returns 100%."""
        budget = CapacityBudget(hourly_limit_usd=-10.0, current_hourly_usd=0.0)
        assert budget.hourly_budget_percent_used() == 100.0

    def test_daily_budget_percent_used(self) -> None:
        """Test daily_budget_percent_used calculation."""
        budget = CapacityBudget(daily_limit_usd=500.0, current_daily_usd=250.0)
        assert budget.daily_budget_percent_used() == 50.0

    def test_daily_budget_percent_used_zero_limit(self) -> None:
        """Test that zero daily limit returns 100%."""
        budget = CapacityBudget(daily_limit_usd=0.0, current_daily_usd=0.0)
        assert budget.daily_budget_percent_used() == 100.0

    def test_is_over_alert_threshold_hourly(self) -> None:
        """Test alert threshold detection for hourly budget."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            current_hourly_usd=85.0,  # 85% used
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is True

    def test_is_over_alert_threshold_daily(self) -> None:
        """Test alert threshold detection for daily budget."""
        budget = CapacityBudget(
            daily_limit_usd=500.0,
            current_daily_usd=450.0,  # 90% used
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is True

    def test_is_over_alert_threshold_false(self) -> None:
        """Test alert threshold when under limit."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=500.0,
            current_hourly_usd=50.0,  # 50% used
            current_daily_usd=200.0,  # 40% used
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is False

    def test_can_afford_success(self) -> None:
        """Test can_afford when budget allows."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=500.0,
            current_hourly_usd=10.0,
            current_daily_usd=100.0,
        )
        # Additional $5/hr: hourly OK (10+5=15 < 100), daily OK (100+5*24=220 < 500)
        assert budget.can_afford(5.0) is True

    def test_can_afford_fails_hourly(self) -> None:
        """Test can_afford fails when hourly limit exceeded."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=1000.0,
            current_hourly_usd=45.0,
        )
        # Additional $10/hr: hourly fails (45+10=55 > 50)
        assert budget.can_afford(10.0) is False

    def test_can_afford_fails_daily(self) -> None:
        """Test can_afford fails when daily limit exceeded."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=100.0,  # Low daily limit
            current_hourly_usd=0.0,
            current_daily_usd=50.0,
        )
        # Additional $5/hr: hourly OK, daily fails (50 + 5*24=170 > 100)
        assert budget.can_afford(5.0) is False

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=500.0,
            current_hourly_usd=40.0,
            current_daily_usd=200.0,
            alert_threshold_percent=80.0,
        )
        d = budget.to_dict()
        assert d["hourly_limit_usd"] == 100.0
        assert d["daily_limit_usd"] == 500.0
        assert d["current_hourly_usd"] == 40.0
        assert d["current_daily_usd"] == 200.0
        assert d["hourly_percent_used"] == 40.0
        assert d["daily_percent_used"] == 40.0
        assert d["alert_threshold_percent"] == 80.0


# =============================================================================
# ScaleRecommendation Tests
# =============================================================================


class TestScaleRecommendation:
    """Tests for ScaleRecommendation dataclass."""

    def test_default_values(self) -> None:
        """Test default values for scale recommendation."""
        rec = ScaleRecommendation(action=ScaleAction.NONE)
        assert rec.action == ScaleAction.NONE
        assert rec.count == 0
        assert rec.reason == ""
        assert rec.provider == ""
        assert rec.gpu_type == ""
        assert rec.estimated_hourly_cost == 0.0
        assert rec.utilization == 0.0
        assert isinstance(rec.timestamp, datetime)

    def test_scale_up_recommendation(self) -> None:
        """Test scale up recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_UP,
            count=2,
            reason="High utilization (85%)",
            provider="lambda",
            gpu_type="GH200_96GB",
            estimated_hourly_cost=4.98,
            utilization=0.85,
        )
        assert rec.action == ScaleAction.SCALE_UP
        assert rec.count == 2
        assert "High utilization" in rec.reason

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_DOWN,
            count=1,
            reason="Low utilization",
            utilization=0.25,
        )
        d = rec.to_dict()
        assert d["action"] == "scale_down"
        assert d["count"] == 1
        assert d["reason"] == "Low utilization"
        assert d["utilization"] == 0.25
        assert "timestamp" in d


# =============================================================================
# CapacityPlannerConfig Tests
# =============================================================================


class TestCapacityPlannerConfig:
    """Tests for CapacityPlannerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CapacityPlannerConfig()
        assert config.check_interval_seconds == 60
        assert config.scale_up_utilization_threshold == 0.85
        assert config.scale_down_utilization_threshold == 0.30
        assert config.min_scale_up_interval_seconds == 300.0
        assert config.min_scale_down_interval_seconds == 600.0
        assert config.hourly_budget_usd == 50.0
        assert config.daily_budget_usd == 500.0
        assert config.min_gpu_nodes == 4

    def test_gpu_costs_default(self) -> None:
        """Test default GPU cost mapping."""
        config = CapacityPlannerConfig()
        assert "GH200_96GB" in config.gpu_costs
        assert "H100_80GB" in config.gpu_costs
        assert "A100_80GB" in config.gpu_costs
        assert config.gpu_costs["GH200_96GB"] == 2.49

    def test_from_env(self) -> None:
        """Test configuration from environment variables."""
        env_vars = {
            "RINGRIFT_CAPACITY_CYCLE_INTERVAL": "120",
            "RINGRIFT_SCALE_UP_THRESHOLD": "0.90",
            "RINGRIFT_SCALE_DOWN_THRESHOLD": "0.20",
            "RINGRIFT_HOURLY_BUDGET_USD": "100",
            "RINGRIFT_DAILY_BUDGET_USD": "1000",
            "RINGRIFT_MIN_GPU_NODES": "8",
        }
        with patch.dict(os.environ, env_vars):
            config = CapacityPlannerConfig.from_env()
            assert config.check_interval_seconds == 120
            assert config.scale_up_utilization_threshold == 0.90
            assert config.scale_down_utilization_threshold == 0.20
            assert config.hourly_budget_usd == 100.0
            assert config.daily_budget_usd == 1000.0
            assert config.min_gpu_nodes == 8

    def test_from_env_defaults(self) -> None:
        """Test from_env with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = CapacityPlannerConfig.from_env()
            assert config.check_interval_seconds == 60
            assert config.scale_up_utilization_threshold == 0.85


# =============================================================================
# UtilizationMetrics Tests
# =============================================================================


class TestUtilizationMetrics:
    """Tests for UtilizationMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default utilization values."""
        metrics = UtilizationMetrics()
        assert metrics.total_gpu_nodes == 0
        assert metrics.active_gpu_nodes == 0
        assert metrics.gpu_utilization_avg == 0.0
        assert metrics.memory_utilization_avg == 0.0
        assert metrics.selfplay_jobs_running == 0
        assert metrics.training_jobs_running == 0
        assert metrics.pending_work_items == 0
        assert isinstance(metrics.timestamp, datetime)

    def test_overall_utilization_zero_nodes(self) -> None:
        """Test overall utilization when no GPU nodes."""
        metrics = UtilizationMetrics(total_gpu_nodes=0)
        assert metrics.overall_utilization == 0.0

    def test_overall_utilization_weighted_average(self) -> None:
        """Test overall utilization weighted calculation.

        Formula: 0.5 * GPU + 0.3 * job_util + 0.2 * memory
        """
        metrics = UtilizationMetrics(
            total_gpu_nodes=10,
            active_gpu_nodes=8,
            gpu_utilization_avg=0.8,
            memory_utilization_avg=0.6,
            selfplay_jobs_running=4,
            training_jobs_running=2,
        )
        # job_util = min(1.0, (4+2) / max(1, 8)) = min(1.0, 0.75) = 0.75
        # overall = 0.5 * 0.8 + 0.3 * 0.75 + 0.2 * 0.6
        #         = 0.4 + 0.225 + 0.12 = 0.745
        expected = 0.5 * 0.8 + 0.3 * 0.75 + 0.2 * 0.6
        assert abs(metrics.overall_utilization - expected) < 0.001

    def test_overall_utilization_job_capped_at_one(self) -> None:
        """Test that job utilization is capped at 1.0."""
        metrics = UtilizationMetrics(
            total_gpu_nodes=4,
            active_gpu_nodes=2,
            gpu_utilization_avg=0.5,
            memory_utilization_avg=0.5,
            selfplay_jobs_running=10,  # Way more than active nodes
            training_jobs_running=5,
        )
        # job_util capped at 1.0
        expected = 0.5 * 0.5 + 0.3 * 1.0 + 0.2 * 0.5
        assert abs(metrics.overall_utilization - expected) < 0.001


# =============================================================================
# CapacityPlanner Tests
# =============================================================================


class TestCapacityPlanner:
    """Tests for CapacityPlanner class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()
        yield
        reset_capacity_planner()

    def test_init_creates_budget(self) -> None:
        """Test that init creates budget from config."""
        config = CapacityPlannerConfig(
            hourly_budget_usd=100.0,
            daily_budget_usd=1000.0,
        )
        planner = CapacityPlanner(config=config)
        assert planner.budget.hourly_limit_usd == 100.0
        assert planner.budget.daily_limit_usd == 1000.0

    def test_get_daemon_name(self) -> None:
        """Test daemon name."""
        planner = CapacityPlanner()
        assert planner._get_daemon_name() == "CapacityPlanner"

    def test_get_default_config(self) -> None:
        """Test default config creation."""
        planner = CapacityPlanner()
        config = planner._get_default_config()
        assert isinstance(config, CapacityPlannerConfig)

    def test_get_event_subscriptions(self) -> None:
        """Test event subscriptions."""
        planner = CapacityPlanner()
        subs = planner._get_event_subscriptions()
        assert "NODE_PROVISIONED" in subs
        assert "NODE_TERMINATED" in subs
        assert "HOURLY_TICK" in subs

    @pytest.mark.asyncio
    async def test_on_node_provisioned(self) -> None:
        """Test budget update on node provisioned."""
        planner = CapacityPlanner()
        event = {"payload": {"cost_per_hour": 2.49}}

        await planner._on_node_provisioned(event)

        assert planner.budget.current_hourly_usd == 2.49
        assert planner.budget.current_daily_usd == 2.49

    @pytest.mark.asyncio
    async def test_on_node_terminated(self) -> None:
        """Test budget update on node terminated."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 10.0
        event = {"payload": {"cost_per_hour": 2.49}}

        await planner._on_node_terminated(event)

        assert planner.budget.current_hourly_usd == 7.51

    @pytest.mark.asyncio
    async def test_on_node_terminated_clamps_to_zero(self) -> None:
        """Test that budget doesn't go negative on termination."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 1.0
        event = {"payload": {"cost_per_hour": 5.0}}

        await planner._on_node_terminated(event)

        assert planner.budget.current_hourly_usd == 0.0

    @pytest.mark.asyncio
    async def test_should_scale_up_budget_allows(self) -> None:
        """Test should_scale_up when budget allows."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 10.0
        planner._last_scale_up_time = 0  # No cooldown

        result = await planner.should_scale_up(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_scale_up_budget_exceeded(self) -> None:
        """Test should_scale_up when budget exceeded."""
        config = CapacityPlannerConfig(hourly_budget_usd=10.0, daily_budget_usd=50.0)
        planner = CapacityPlanner(config=config)
        planner.budget.current_hourly_usd = 9.0

        # Need to afford ~1.3/hr avg but only 1$ left
        result = await planner.should_scale_up(1)

        assert result is False

    @pytest.mark.asyncio
    async def test_should_scale_up_cooldown_active(self) -> None:
        """Test should_scale_up during cooldown."""
        planner = CapacityPlanner()
        planner._last_scale_up_time = time.time()  # Just scaled up

        result = await planner.should_scale_up(1)

        assert result is False

    def test_record_scale_up(self) -> None:
        """Test scale up recording."""
        planner = CapacityPlanner()
        before = planner._last_scale_up_time

        planner.record_scale_up()

        assert planner._last_scale_up_time > before

    def test_record_scale_down(self) -> None:
        """Test scale down recording."""
        planner = CapacityPlanner()
        before = planner._last_scale_down_time

        planner.record_scale_down()

        assert planner._last_scale_down_time > before

    def test_get_budget_status(self) -> None:
        """Test get_budget_status returns dict."""
        planner = CapacityPlanner()
        status = planner.get_budget_status()

        assert "hourly_limit_usd" in status
        assert "daily_limit_usd" in status
        assert "hourly_percent_used" in status

    def test_get_utilization_history_empty(self) -> None:
        """Test utilization history when empty."""
        planner = CapacityPlanner()
        history = planner.get_utilization_history()

        assert history == []

    def test_get_utilization_history_with_data(self) -> None:
        """Test utilization history with data."""
        planner = CapacityPlanner()
        metrics = UtilizationMetrics(total_gpu_nodes=10, active_gpu_nodes=8)
        planner._utilization_history.append(metrics)

        history = planner.get_utilization_history()

        assert len(history) == 1
        assert history[0]["total_gpu_nodes"] == 10
        assert history[0]["active_gpu_nodes"] == 8

    def test_get_utilization_history_respects_limit(self) -> None:
        """Test utilization history limit parameter."""
        planner = CapacityPlanner()
        for i in range(100):
            metrics = UtilizationMetrics(total_gpu_nodes=i)
            planner._utilization_history.append(metrics)

        history = planner.get_utilization_history(limit=10)

        # Should return last 10
        assert len(history) == 10
        assert history[-1]["total_gpu_nodes"] == 99

    def test_health_check_healthy(self) -> None:
        """Test health_check when budget is OK."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 10.0

        health = planner.health_check()

        assert health.healthy is True
        assert "CapacityPlanner: Budget within limits" in health.message

    def test_health_check_unhealthy(self) -> None:
        """Test health_check when budget exceeded."""
        config = CapacityPlannerConfig(hourly_budget_usd=100.0)
        planner = CapacityPlanner(config=config)
        planner.budget.current_hourly_usd = 90.0  # 90% used, over 80% threshold

        health = planner.health_check()

        assert health.healthy is False
        assert "90%" in health.message

    def test_health_check_details(self) -> None:
        """Test health_check returns expected details."""
        planner = CapacityPlanner()
        metrics = UtilizationMetrics(active_gpu_nodes=5, total_gpu_nodes=10)
        planner._utilization_history.append(metrics)

        health = planner.health_check()

        assert "hourly_budget_percent" in health.details
        assert "daily_budget_percent" in health.details
        assert health.details["active_gpu_nodes"] == 5


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()
        yield
        reset_capacity_planner()

    def test_get_capacity_planner_creates_singleton(self) -> None:
        """Test singleton creation."""
        planner1 = get_capacity_planner()
        planner2 = get_capacity_planner()

        assert planner1 is planner2

    def test_reset_capacity_planner_clears_singleton(self) -> None:
        """Test singleton reset."""
        planner1 = get_capacity_planner()
        reset_capacity_planner()
        planner2 = get_capacity_planner()

        assert planner1 is not planner2


# =============================================================================
# Scale Recommendation Tests
# =============================================================================


class TestGetScaleRecommendation:
    """Tests for get_scale_recommendation method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()
        yield
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_scale_up_high_utilization(self) -> None:
        """Test scale up recommendation on high utilization."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 0.0  # Plenty of budget
        planner._last_scale_up_time = 0  # No cooldown

        # Mock utilization metrics to return high utilization
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.9,
                memory_utilization_avg=0.9,
                selfplay_jobs_running=10,
            ),
        ):
            rec = await planner.get_scale_recommendation()

        assert rec.action == ScaleAction.SCALE_UP
        assert rec.count == 2
        assert "High utilization" in rec.reason

    @pytest.mark.asyncio
    async def test_scale_down_low_utilization(self) -> None:
        """Test scale down recommendation on low utilization."""
        config = CapacityPlannerConfig(min_gpu_nodes=2)
        planner = CapacityPlanner(config=config)
        planner._last_scale_down_time = 0  # No cooldown

        # Mock low utilization
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=8,
                gpu_utilization_avg=0.1,
                memory_utilization_avg=0.1,
                selfplay_jobs_running=0,
            ),
        ):
            rec = await planner.get_scale_recommendation()

        assert rec.action == ScaleAction.SCALE_DOWN
        assert rec.count == 1
        assert "Low utilization" in rec.reason

    @pytest.mark.asyncio
    async def test_no_scale_normal_utilization(self) -> None:
        """Test no scaling on normal utilization."""
        planner = CapacityPlanner()

        # Mock moderate utilization
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.5,
                memory_utilization_avg=0.5,
                selfplay_jobs_running=5,
            ),
        ):
            rec = await planner.get_scale_recommendation()

        assert rec.action == ScaleAction.NONE
        assert "within acceptable range" in rec.reason

    @pytest.mark.asyncio
    async def test_no_scale_up_budget_exceeded(self) -> None:
        """Test no scale up when budget exceeded despite high util."""
        config = CapacityPlannerConfig(hourly_budget_usd=10.0)
        planner = CapacityPlanner(config=config)
        planner.budget.current_hourly_usd = 9.5  # Almost at limit
        planner._last_scale_up_time = 0  # No cooldown

        # Need high utilization (>= 85% overall) to trigger scale-up attempt
        # overall = 0.5 * gpu + 0.3 * job_util + 0.2 * memory
        # With gpu=0.95, jobs=10/10=1.0, memory=0.95:
        # overall = 0.5 * 0.95 + 0.3 * 1.0 + 0.2 * 0.95 = 0.475 + 0.3 + 0.19 = 0.965
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.95,
                memory_utilization_avg=0.95,
                selfplay_jobs_running=10,  # Need jobs to increase utilization
            ),
        ):
            rec = await planner.get_scale_recommendation()

        assert rec.action == ScaleAction.NONE
        assert "budget exceeded" in rec.reason

    @pytest.mark.asyncio
    async def test_no_scale_down_below_minimum(self) -> None:
        """Test no scale down when at minimum nodes."""
        config = CapacityPlannerConfig(min_gpu_nodes=8)
        planner = CapacityPlanner(config=config)
        planner._last_scale_down_time = 0

        # Low utilization but at minimum nodes
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=UtilizationMetrics(
                total_gpu_nodes=8,
                active_gpu_nodes=8,  # At minimum
                gpu_utilization_avg=0.1,
            ),
        ):
            rec = await planner.get_scale_recommendation()

        # Should NOT recommend scale down even with low util
        assert rec.action == ScaleAction.NONE


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for _run_cycle method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()
        yield
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_run_cycle_collects_metrics(self) -> None:
        """Test that run cycle collects utilization metrics."""
        planner = CapacityPlanner()

        mock_metrics = UtilizationMetrics(total_gpu_nodes=10)
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            await planner._run_cycle()

        assert len(planner._utilization_history) == 1

    @pytest.mark.asyncio
    async def test_run_cycle_prunes_old_history(self) -> None:
        """Test that run cycle removes old history entries."""
        planner = CapacityPlanner()

        # Add old metric
        old_metric = UtilizationMetrics()
        old_metric.timestamp = datetime.now() - timedelta(hours=2)
        planner._utilization_history.append(old_metric)

        mock_metrics = UtilizationMetrics()
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            await planner._run_cycle()

        # Old entry should be removed
        assert len(planner._utilization_history) == 1
        assert planner._utilization_history[0] is mock_metrics

    @pytest.mark.asyncio
    async def test_run_cycle_emits_budget_alert(self) -> None:
        """Test budget alert emission when over threshold."""
        config = CapacityPlannerConfig(hourly_budget_usd=100.0)
        planner = CapacityPlanner(config=config)
        planner.budget.current_hourly_usd = 90.0  # 90% used

        mock_metrics = UtilizationMetrics()
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ), patch.object(
            planner,
            "_emit_budget_alert",
            new_callable=AsyncMock,
        ) as mock_alert:
            await planner._run_cycle()

        mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_daily_reset(self) -> None:
        """Test daily budget reset on new day."""
        planner = CapacityPlanner()
        planner.budget.current_daily_usd = 100.0
        planner.budget.daily_reset_time = datetime.now() - timedelta(days=1)

        mock_metrics = UtilizationMetrics()
        with patch.object(
            planner,
            "_collect_utilization_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            await planner._run_cycle()

        assert planner.budget.current_daily_usd == 0.0
