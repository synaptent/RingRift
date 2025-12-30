"""Tests for capacity_planner.py - budget-aware capacity planning.

Tests cover:
- CapacityBudget dataclass and budget calculations
- ScaleRecommendation dataclass
- CapacityPlannerConfig with env var loading
- UtilizationMetrics and overall utilization calculation
- CapacityPlanner daemon lifecycle and scaling logic
- Singleton pattern and health checks

December 2025 - Test coverage for critical untested module.
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
        """Test default budget values."""
        budget = CapacityBudget()
        assert budget.hourly_limit_usd == 50.0
        assert budget.daily_limit_usd == 500.0
        assert budget.alert_threshold_percent == 80.0
        assert budget.current_hourly_usd == 0.0
        assert budget.current_daily_usd == 0.0

    def test_custom_values(self) -> None:
        """Test custom budget values."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=1000.0,
            alert_threshold_percent=90.0,
        )
        assert budget.hourly_limit_usd == 100.0
        assert budget.daily_limit_usd == 1000.0
        assert budget.alert_threshold_percent == 90.0

    def test_remaining_hourly_budget(self) -> None:
        """Test remaining hourly budget calculation."""
        budget = CapacityBudget(hourly_limit_usd=50.0, current_hourly_usd=30.0)
        assert budget.remaining_hourly_budget() == 20.0

    def test_remaining_hourly_budget_over_limit(self) -> None:
        """Test remaining budget when over limit."""
        budget = CapacityBudget(hourly_limit_usd=50.0, current_hourly_usd=60.0)
        assert budget.remaining_hourly_budget() == 0.0

    def test_remaining_daily_budget(self) -> None:
        """Test remaining daily budget calculation."""
        budget = CapacityBudget(daily_limit_usd=500.0, current_daily_usd=200.0)
        assert budget.remaining_daily_budget() == 300.0

    def test_hourly_budget_percent_used(self) -> None:
        """Test hourly budget percentage used."""
        budget = CapacityBudget(hourly_limit_usd=50.0, current_hourly_usd=25.0)
        assert budget.hourly_budget_percent_used() == 50.0

    def test_hourly_budget_percent_zero_limit(self) -> None:
        """Test hourly percentage with zero limit."""
        budget = CapacityBudget(hourly_limit_usd=0.0)
        assert budget.hourly_budget_percent_used() == 100.0

    def test_daily_budget_percent_used(self) -> None:
        """Test daily budget percentage used."""
        budget = CapacityBudget(daily_limit_usd=500.0, current_daily_usd=400.0)
        assert budget.daily_budget_percent_used() == 80.0

    def test_daily_budget_percent_zero_limit(self) -> None:
        """Test daily percentage with zero limit."""
        budget = CapacityBudget(daily_limit_usd=0.0)
        assert budget.daily_budget_percent_used() == 100.0

    def test_is_over_alert_threshold_hourly(self) -> None:
        """Test alert threshold detection for hourly."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            current_hourly_usd=45.0,  # 90%
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is True

    def test_is_over_alert_threshold_daily(self) -> None:
        """Test alert threshold detection for daily."""
        budget = CapacityBudget(
            daily_limit_usd=500.0,
            current_daily_usd=450.0,  # 90%
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is True

    def test_is_under_alert_threshold(self) -> None:
        """Test under alert threshold."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=10.0,  # 20%
            current_daily_usd=100.0,  # 20%
            alert_threshold_percent=80.0,
        )
        assert budget.is_over_alert_threshold() is False

    def test_can_afford_within_budget(self) -> None:
        """Test can_afford when within budget."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=10.0,
        )
        assert budget.can_afford(5.0) is True

    def test_can_afford_exceeds_hourly(self) -> None:
        """Test can_afford when exceeds hourly limit."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            current_hourly_usd=48.0,
        )
        assert budget.can_afford(5.0) is False

    def test_can_afford_exceeds_daily(self) -> None:
        """Test can_afford when exceeds daily limit."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=100.0,  # Low daily limit
            current_hourly_usd=0.0,
            current_daily_usd=95.0,
        )
        # Daily impact = 5 * 24 = 120, which exceeds remaining daily budget
        assert budget.can_afford(5.0) is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=25.0,
            current_daily_usd=250.0,
        )
        d = budget.to_dict()

        assert d["hourly_limit_usd"] == 50.0
        assert d["daily_limit_usd"] == 500.0
        assert d["current_hourly_usd"] == 25.0
        assert d["current_daily_usd"] == 250.0
        assert d["hourly_percent_used"] == 50.0
        assert d["daily_percent_used"] == 50.0


# =============================================================================
# ScaleRecommendation Tests
# =============================================================================


class TestScaleRecommendation:
    """Tests for ScaleRecommendation dataclass."""

    def test_default_values(self) -> None:
        """Test default recommendation values."""
        rec = ScaleRecommendation(action=ScaleAction.NONE)
        assert rec.action == ScaleAction.NONE
        assert rec.count == 0
        assert rec.reason == ""
        assert rec.provider == ""
        assert rec.gpu_type == ""
        assert rec.estimated_hourly_cost == 0.0

    def test_scale_up_recommendation(self) -> None:
        """Test scale-up recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_UP,
            count=2,
            reason="High utilization",
            provider="lambda",
            gpu_type="GH200_96GB",
            estimated_hourly_cost=4.98,
            utilization=0.92,
        )
        assert rec.action == ScaleAction.SCALE_UP
        assert rec.count == 2
        assert rec.estimated_hourly_cost == 4.98

    def test_to_dict(self) -> None:
        """Test recommendation serialization."""
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
    """Tests for CapacityPlannerConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = CapacityPlannerConfig()
        assert config.check_interval_seconds == 60
        assert config.scale_up_utilization_threshold == 0.85
        assert config.scale_down_utilization_threshold == 0.30
        assert config.hourly_budget_usd == 50.0
        assert config.daily_budget_usd == 500.0
        assert config.min_gpu_nodes == 4

    def test_gpu_costs_default(self) -> None:
        """Test default GPU cost estimates."""
        config = CapacityPlannerConfig()
        assert "GH200_96GB" in config.gpu_costs
        assert "H100_80GB" in config.gpu_costs
        assert config.gpu_costs["GH200_96GB"] == 2.49

    def test_from_env(self) -> None:
        """Test loading config from environment."""
        with patch.dict(os.environ, {
            "RINGRIFT_CAPACITY_CYCLE_INTERVAL": "120",
            "RINGRIFT_SCALE_UP_THRESHOLD": "0.90",
            "RINGRIFT_SCALE_DOWN_THRESHOLD": "0.20",
            "RINGRIFT_HOURLY_BUDGET_USD": "100",
            "RINGRIFT_DAILY_BUDGET_USD": "1000",
            "RINGRIFT_MIN_GPU_NODES": "6",
        }):
            config = CapacityPlannerConfig.from_env()

            assert config.check_interval_seconds == 120
            assert config.scale_up_utilization_threshold == 0.90
            assert config.scale_down_utilization_threshold == 0.20
            assert config.hourly_budget_usd == 100.0
            assert config.daily_budget_usd == 1000.0
            assert config.min_gpu_nodes == 6


# =============================================================================
# UtilizationMetrics Tests
# =============================================================================


class TestUtilizationMetrics:
    """Tests for UtilizationMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        metrics = UtilizationMetrics()
        assert metrics.total_gpu_nodes == 0
        assert metrics.active_gpu_nodes == 0
        assert metrics.gpu_utilization_avg == 0.0
        assert metrics.selfplay_jobs_running == 0
        assert metrics.training_jobs_running == 0

    def test_overall_utilization_empty(self) -> None:
        """Test overall utilization with no nodes."""
        metrics = UtilizationMetrics(total_gpu_nodes=0)
        assert metrics.overall_utilization == 0.0

    def test_overall_utilization_weighted(self) -> None:
        """Test overall utilization weighting."""
        metrics = UtilizationMetrics(
            total_gpu_nodes=10,
            active_gpu_nodes=10,
            gpu_utilization_avg=0.80,
            memory_utilization_avg=0.60,
            selfplay_jobs_running=8,
            training_jobs_running=2,
        )
        # 0.5 * 0.80 + 0.3 * min(1.0, 10/10) + 0.2 * 0.60
        # = 0.40 + 0.30 + 0.12 = 0.82
        assert metrics.overall_utilization == pytest.approx(0.82)

    def test_overall_utilization_partial_jobs(self) -> None:
        """Test overall utilization with partial job fill."""
        metrics = UtilizationMetrics(
            total_gpu_nodes=10,
            active_gpu_nodes=10,
            gpu_utilization_avg=0.50,
            memory_utilization_avg=0.40,
            selfplay_jobs_running=3,
            training_jobs_running=2,  # 50% job utilization
        )
        # 0.5 * 0.50 + 0.3 * 0.50 + 0.2 * 0.40
        # = 0.25 + 0.15 + 0.08 = 0.48
        assert metrics.overall_utilization == pytest.approx(0.48)


# =============================================================================
# CapacityPlanner Tests
# =============================================================================


class TestCapacityPlannerInit:
    """Tests for CapacityPlanner initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    def test_init_default(self) -> None:
        """Test default initialization."""
        planner = CapacityPlanner()
        assert planner.budget.hourly_limit_usd == 50.0
        assert planner.budget.daily_limit_usd == 500.0

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        config = CapacityPlannerConfig(
            hourly_budget_usd=100.0,
            daily_budget_usd=1000.0,
        )
        planner = CapacityPlanner(config)
        assert planner.budget.hourly_limit_usd == 100.0
        assert planner.budget.daily_limit_usd == 1000.0

    def test_daemon_name(self) -> None:
        """Test daemon name."""
        planner = CapacityPlanner()
        assert planner._get_daemon_name() == "CapacityPlanner"


class TestCapacityPlannerEventHandlers:
    """Tests for CapacityPlanner event handlers."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_on_node_provisioned(self) -> None:
        """Test node provisioned event updates budget."""
        planner = CapacityPlanner()
        assert planner.budget.current_hourly_usd == 0.0

        await planner._on_node_provisioned({"cost_per_hour": 2.49})

        assert planner.budget.current_hourly_usd == 2.49
        assert planner.budget.current_daily_usd == 2.49
        assert len(planner._cost_history) == 1

    @pytest.mark.asyncio
    async def test_on_node_terminated(self) -> None:
        """Test node terminated event reduces budget."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 5.0

        await planner._on_node_terminated({"cost_per_hour": 2.0})

        assert planner.budget.current_hourly_usd == 3.0

    @pytest.mark.asyncio
    async def test_on_node_terminated_not_negative(self) -> None:
        """Test node terminated doesn't go negative."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 1.0

        await planner._on_node_terminated({"cost_per_hour": 5.0})

        assert planner.budget.current_hourly_usd == 0.0

    @pytest.mark.asyncio
    async def test_on_hourly_tick(self) -> None:
        """Test hourly tick resets hourly budget."""
        planner = CapacityPlanner()

        with patch.object(
            planner, "_calculate_current_hourly_cost", new_callable=AsyncMock
        ) as mock_calc:
            mock_calc.return_value = 10.0

            await planner._on_hourly_tick({})

            assert planner.budget.current_hourly_usd == 10.0


class TestCapacityPlannerScaling:
    """Tests for CapacityPlanner scaling logic."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_should_scale_up_within_budget(self) -> None:
        """Test scale up allowed within budget."""
        planner = CapacityPlanner()
        # Budget allows scaling
        planner.budget.current_hourly_usd = 10.0

        can_scale = await planner.should_scale_up(1)
        assert can_scale is True

    @pytest.mark.asyncio
    async def test_should_scale_up_exceeds_budget(self) -> None:
        """Test scale up blocked when exceeds budget."""
        planner = CapacityPlanner()
        # Budget nearly exhausted
        planner.budget.current_hourly_usd = 49.0

        can_scale = await planner.should_scale_up(1)
        assert can_scale is False

    @pytest.mark.asyncio
    async def test_should_scale_up_cooldown(self) -> None:
        """Test scale up blocked during cooldown."""
        planner = CapacityPlanner()
        planner._last_scale_up_time = time.time()  # Just scaled up

        can_scale = await planner.should_scale_up(1)
        assert can_scale is False

    def test_record_scale_up(self) -> None:
        """Test recording scale up time."""
        planner = CapacityPlanner()
        before = planner._last_scale_up_time

        planner.record_scale_up()

        assert planner._last_scale_up_time > before

    def test_record_scale_down(self) -> None:
        """Test recording scale down time."""
        planner = CapacityPlanner()
        before = planner._last_scale_down_time

        planner.record_scale_down()

        assert planner._last_scale_down_time > before


class TestCapacityPlannerRecommendation:
    """Tests for CapacityPlanner recommendations."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    @pytest.mark.asyncio
    async def test_recommendation_scale_up(self) -> None:
        """Test scale up recommendation at high utilization."""
        planner = CapacityPlanner()

        with patch.object(planner, "_collect_utilization_metrics", new_callable=AsyncMock) as mock_metrics:
            # Need jobs running to get overall utilization above 0.85
            # Overall = 0.5 * gpu_util + 0.3 * job_util + 0.2 * mem_util
            # With gpu=0.95, jobs=10/10=1.0, mem=0.80: 0.475 + 0.3 + 0.16 = 0.935
            mock_metrics.return_value = UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.95,
                memory_utilization_avg=0.80,
                selfplay_jobs_running=8,
                training_jobs_running=2,
            )

            rec = await planner.get_scale_recommendation()

            assert rec.action == ScaleAction.SCALE_UP
            assert rec.count == 2

    @pytest.mark.asyncio
    async def test_recommendation_scale_down(self) -> None:
        """Test scale down recommendation at low utilization."""
        config = CapacityPlannerConfig(min_gpu_nodes=2)
        planner = CapacityPlanner(config)

        with patch.object(planner, "_collect_utilization_metrics", new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.10,
                memory_utilization_avg=0.10,
            )

            rec = await planner.get_scale_recommendation()

            assert rec.action == ScaleAction.SCALE_DOWN
            assert rec.count == 1

    @pytest.mark.asyncio
    async def test_recommendation_none(self) -> None:
        """Test no scaling recommendation at normal utilization."""
        planner = CapacityPlanner()

        with patch.object(planner, "_collect_utilization_metrics", new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.60,
                memory_utilization_avg=0.50,
            )

            rec = await planner.get_scale_recommendation()

            assert rec.action == ScaleAction.NONE

    @pytest.mark.asyncio
    async def test_recommendation_budget_exceeded(self) -> None:
        """Test no scale up when budget exceeded."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 49.0  # Near limit

        with patch.object(planner, "_collect_utilization_metrics", new_callable=AsyncMock) as mock_metrics:
            # High utilization that would trigger scale up if budget allowed
            mock_metrics.return_value = UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=10,
                gpu_utilization_avg=0.95,
                memory_utilization_avg=0.80,
                selfplay_jobs_running=8,
                training_jobs_running=2,
            )

            rec = await planner.get_scale_recommendation()

            assert rec.action == ScaleAction.NONE
            assert "budget" in rec.reason.lower()


class TestCapacityPlannerHealth:
    """Tests for CapacityPlanner health check."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    def test_health_check_healthy(self) -> None:
        """Test health check when budget within limits."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 10.0  # 20%

        health = planner.health_check()

        assert health.healthy is True
        assert "within limits" in health.message

    def test_health_check_unhealthy(self) -> None:
        """Test health check when budget exceeded."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 45.0  # 90%

        health = planner.health_check()

        assert health.healthy is False
        assert "90%" in health.message

    def test_health_check_includes_details(self) -> None:
        """Test health check includes utilization details."""
        planner = CapacityPlanner()
        planner._utilization_history.append(
            UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=8,
            )
        )

        health = planner.health_check()

        assert "active_gpu_nodes" in health.details
        assert health.details["active_gpu_nodes"] == 8


class TestCapacityPlannerSingleton:
    """Tests for CapacityPlanner singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    def test_get_capacity_planner_singleton(self) -> None:
        """Test singleton returns same instance."""
        p1 = get_capacity_planner()
        p2 = get_capacity_planner()
        assert p1 is p2

    def test_reset_capacity_planner(self) -> None:
        """Test reset clears singleton."""
        p1 = get_capacity_planner()
        reset_capacity_planner()
        p2 = get_capacity_planner()
        assert p1 is not p2


class TestCapacityPlannerStatusMethods:
    """Tests for CapacityPlanner status methods."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_capacity_planner()

    def test_get_budget_status(self) -> None:
        """Test get_budget_status returns dict."""
        planner = CapacityPlanner()
        planner.budget.current_hourly_usd = 25.0

        status = planner.get_budget_status()

        assert status["current_hourly_usd"] == 25.0
        assert status["hourly_percent_used"] == 50.0

    def test_get_utilization_history_empty(self) -> None:
        """Test utilization history when empty."""
        planner = CapacityPlanner()
        history = planner.get_utilization_history()
        assert history == []

    def test_get_utilization_history_with_data(self) -> None:
        """Test utilization history with data."""
        planner = CapacityPlanner()
        planner._utilization_history.append(
            UtilizationMetrics(
                total_gpu_nodes=10,
                active_gpu_nodes=8,
                gpu_utilization_avg=0.75,
            )
        )

        history = planner.get_utilization_history()

        assert len(history) == 1
        assert history[0]["total_gpu_nodes"] == 10
        assert history[0]["gpu_utilization"] == 0.75

    def test_get_utilization_history_limit(self) -> None:
        """Test utilization history respects limit."""
        planner = CapacityPlanner()

        # Add 100 entries
        for i in range(100):
            planner._utilization_history.append(
                UtilizationMetrics(total_gpu_nodes=i)
            )

        history = planner.get_utilization_history(limit=10)
        assert len(history) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
