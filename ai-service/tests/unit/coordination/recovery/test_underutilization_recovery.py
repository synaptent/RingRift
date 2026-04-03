"""Tests for UnderutilizationRecoveryHandler.

Tests cover:
- Configuration loading
- Event subscriptions (CLUSTER_UNDERUTILIZED, WORK_QUEUE_EXHAUSTED)
- Work injection logic
- Recovery cooldown handling
- Event emission (UTILIZATION_RECOVERY_*)
- Statistics tracking
- Health check reporting
- Config key parsing and work item creation
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.underutilization_recovery_handler import (
    DEFAULT_CHECK_INTERVAL_SECONDS,
    DEFAULT_HIGH_PRIORITY,
    DEFAULT_INJECTION_BATCH_SIZE,
    DEFAULT_MIN_IDLE_PERCENT,
    DEFAULT_RECOVERY_COOLDOWN_SECONDS,
    RecoveryStats,
    UnderutilizationConfig,
    UnderutilizationRecoveryHandler,
    get_underutilization_handler,
)


class TestUnderutilizationConfig:
    """Tests for UnderutilizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = UnderutilizationConfig()

        assert config.injection_batch_size == DEFAULT_INJECTION_BATCH_SIZE
        assert config.recovery_cooldown_seconds == DEFAULT_RECOVERY_COOLDOWN_SECONDS
        assert config.high_priority == DEFAULT_HIGH_PRIORITY
        assert config.min_idle_percent == DEFAULT_MIN_IDLE_PERCENT
        assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL_SECONDS
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = UnderutilizationConfig(
            injection_batch_size=50,
            recovery_cooldown_seconds=600.0,
            high_priority=200,
            min_idle_percent=0.7,
            check_interval_seconds=120.0,
            enabled=False,
        )

        assert config.injection_batch_size == 50
        assert config.recovery_cooldown_seconds == 600.0
        assert config.high_priority == 200
        assert config.min_idle_percent == 0.7
        assert config.check_interval_seconds == 120.0
        assert config.enabled is False

    def test_from_env_defaults(self) -> None:
        """Test config creation from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = UnderutilizationConfig.from_env()

        assert config.injection_batch_size == DEFAULT_INJECTION_BATCH_SIZE
        assert config.recovery_cooldown_seconds == DEFAULT_RECOVERY_COOLDOWN_SECONDS
        assert config.enabled is True

    def test_from_env_custom(self) -> None:
        """Test config creation from environment variables."""
        env_vars = {
            "RINGRIFT_UNDERUTIL_BATCH_SIZE": "50",
            "RINGRIFT_UNDERUTIL_COOLDOWN": "600",
            "RINGRIFT_UNDERUTIL_PRIORITY": "200",
            "RINGRIFT_UNDERUTIL_IDLE_THRESHOLD": "0.7",
            "RINGRIFT_UNDERUTIL_CHECK_INTERVAL": "120",
            "RINGRIFT_UNDERUTIL_ENABLED": "false",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = UnderutilizationConfig.from_env()

        assert config.injection_batch_size == 50
        assert config.recovery_cooldown_seconds == 600.0
        assert config.high_priority == 200
        assert config.min_idle_percent == 0.7
        assert config.check_interval_seconds == 120.0
        assert config.enabled is False


class TestRecoveryStats:
    """Tests for RecoveryStats dataclass."""

    def test_default_values(self) -> None:
        """Test default stats values."""
        stats = RecoveryStats()

        assert stats.total_recoveries == 0
        assert stats.successful_recoveries == 0
        assert stats.failed_recoveries == 0
        assert stats.items_injected == 0
        assert stats.last_recovery_time == 0.0
        assert stats.last_recovery_reason == ""
        assert stats.configs_targeted == {}

    def test_custom_values(self) -> None:
        """Test custom stats values."""
        stats = RecoveryStats(
            total_recoveries=10,
            successful_recoveries=8,
            failed_recoveries=2,
            items_injected=100,
            last_recovery_time=1234567890.0,
            last_recovery_reason="work_queue_exhausted",
            configs_targeted={"hex8_2p": 50},
        )

        assert stats.total_recoveries == 10
        assert stats.successful_recoveries == 8
        assert stats.failed_recoveries == 2
        assert stats.items_injected == 100
        assert stats.last_recovery_time == 1234567890.0
        assert stats.last_recovery_reason == "work_queue_exhausted"
        assert stats.configs_targeted == {"hex8_2p": 50}


class TestUnderutilizationHandlerInit:
    """Tests for handler initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_default_initialization(self) -> None:
        """Test default handler initialization."""
        handler = UnderutilizationRecoveryHandler()

        assert handler._work_queue is None
        assert handler._selfplay_scheduler is None
        assert handler._stats.total_recoveries == 0
        assert handler._recovery_in_progress is False

    def test_custom_config_initialization(self) -> None:
        """Test handler with custom config."""
        config = UnderutilizationConfig(
            injection_batch_size=50,
            enabled=False,
        )
        handler = UnderutilizationRecoveryHandler(config=config)

        assert handler._config.injection_batch_size == 50
        assert handler._config.enabled is False

    def test_with_work_queue(self) -> None:
        """Test handler with work queue injection."""
        mock_queue = MagicMock()
        handler = UnderutilizationRecoveryHandler(work_queue=mock_queue)

        assert handler._work_queue is mock_queue

    def test_with_selfplay_scheduler(self) -> None:
        """Test handler with selfplay scheduler injection."""
        mock_scheduler = MagicMock()
        handler = UnderutilizationRecoveryHandler(selfplay_scheduler=mock_scheduler)

        assert handler._selfplay_scheduler is mock_scheduler

    def test_event_subscriptions(self) -> None:
        """Test event subscriptions."""
        handler = UnderutilizationRecoveryHandler()
        subscriptions = handler._get_event_subscriptions()

        assert "CLUSTER_UNDERUTILIZED" in subscriptions
        assert "WORK_QUEUE_EXHAUSTED" in subscriptions
        assert callable(subscriptions["CLUSTER_UNDERUTILIZED"])
        assert callable(subscriptions["WORK_QUEUE_EXHAUSTED"])


class TestUnderutilizationSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_get_instance_returns_same(self) -> None:
        """Test get_instance returns same instance."""
        handler1 = UnderutilizationRecoveryHandler.get_instance()
        handler2 = UnderutilizationRecoveryHandler.get_instance()

        assert handler1 is handler2

    def test_reset_instance_clears(self) -> None:
        """Test reset_instance clears singleton."""
        handler1 = UnderutilizationRecoveryHandler.get_instance()
        UnderutilizationRecoveryHandler.reset_instance()
        handler2 = UnderutilizationRecoveryHandler.get_instance()

        assert handler1 is not handler2

    def test_get_underutilization_handler(self) -> None:
        """Test module-level accessor function."""
        handler = get_underutilization_handler()
        assert isinstance(handler, UnderutilizationRecoveryHandler)


class TestUnderutilizationHealthCheck:
    """Tests for health check functionality."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_health_check_healthy(self) -> None:
        """Test health check when healthy."""
        handler = UnderutilizationRecoveryHandler()

        result = handler.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "enabled" in result.details

    def test_health_check_recovery_in_progress(self) -> None:
        """Test health check during recovery."""
        handler = UnderutilizationRecoveryHandler()
        handler._recovery_in_progress = True

        result = handler.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.details["recovery_in_progress"] is True

    def test_health_check_disabled(self) -> None:
        """Test health check when disabled."""
        config = UnderutilizationConfig(enabled=False)
        handler = UnderutilizationRecoveryHandler(config=config)

        result = handler.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED
        assert result.details["enabled"] is False


class TestUnderutilizationGetStats:
    """Tests for stats retrieval."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_get_stats_initial(self) -> None:
        """Test initial stats values."""
        handler = UnderutilizationRecoveryHandler()

        stats = handler.get_stats()

        assert stats["total_recoveries"] == 0
        assert stats["successful_recoveries"] == 0
        assert stats["failed_recoveries"] == 0
        assert stats["items_injected"] == 0
        assert stats["recovery_in_progress"] is False
        assert stats["enabled"] is True

    def test_get_stats_after_recovery(self) -> None:
        """Test stats after recovery operations."""
        handler = UnderutilizationRecoveryHandler()
        handler._stats.total_recoveries = 5
        handler._stats.successful_recoveries = 4
        handler._stats.failed_recoveries = 1
        handler._stats.items_injected = 80
        handler._stats.configs_targeted = {"hex8_2p": 40, "square8_4p": 40}

        stats = handler.get_stats()

        assert stats["total_recoveries"] == 5
        assert stats["successful_recoveries"] == 4
        assert stats["failed_recoveries"] == 1
        assert stats["items_injected"] == 80
        assert stats["configs_targeted"] == {"hex8_2p": 40, "square8_4p": 40}


class TestUnderutilizationCooldown:
    """Tests for recovery cooldown."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_can_attempt_recovery_initial(self) -> None:
        """Test can attempt recovery initially."""
        handler = UnderutilizationRecoveryHandler()

        assert handler._can_attempt_recovery() is True

    def test_cannot_attempt_during_recovery(self) -> None:
        """Test cannot attempt while recovery in progress."""
        handler = UnderutilizationRecoveryHandler()
        handler._recovery_in_progress = True

        assert handler._can_attempt_recovery() is False

    def test_cannot_attempt_during_cooldown(self) -> None:
        """Test cannot attempt during cooldown period."""
        config = UnderutilizationConfig(recovery_cooldown_seconds=60.0)
        handler = UnderutilizationRecoveryHandler(config=config)
        handler._last_recovery_attempt = time.time()  # Just attempted

        assert handler._can_attempt_recovery() is False

    def test_can_attempt_after_cooldown(self) -> None:
        """Test can attempt after cooldown expires."""
        config = UnderutilizationConfig(recovery_cooldown_seconds=1.0)
        handler = UnderutilizationRecoveryHandler(config=config)
        handler._last_recovery_attempt = time.time() - 2.0  # Cooldown expired

        assert handler._can_attempt_recovery() is True


class TestUnderutilizationEventHandling:
    """Tests for event handling."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_on_cluster_underutilized_disabled(self) -> None:
        """Test event handling when disabled."""
        config = UnderutilizationConfig(enabled=False)
        handler = UnderutilizationRecoveryHandler(config=config)

        event = {"idle_percent": 0.7, "idle_nodes": ["node1", "node2"]}

        await handler._on_cluster_underutilized(event)

        assert handler._pending_recovery_event is None

    @pytest.mark.asyncio
    async def test_on_cluster_underutilized_queues_recovery(self) -> None:
        """Test event queues recovery when cooldown allows."""
        handler = UnderutilizationRecoveryHandler()

        event = {"idle_percent": 0.7, "idle_nodes": ["node1", "node2"]}

        await handler._on_cluster_underutilized(event)

        assert handler._pending_recovery_event is not None
        assert handler._pending_recovery_event["reason"] == "cluster_underutilized"
        assert handler._pending_recovery_event["idle_percent"] == 0.7

    @pytest.mark.asyncio
    async def test_on_work_queue_exhausted_disabled(self) -> None:
        """Test work queue event handling when disabled."""
        config = UnderutilizationConfig(enabled=False)
        handler = UnderutilizationRecoveryHandler(config=config)

        event = {"queue_depth": 0}

        await handler._on_work_queue_exhausted(event)

        assert handler._pending_recovery_event is None

    @pytest.mark.asyncio
    async def test_on_work_queue_exhausted_queues_recovery(self) -> None:
        """Test work queue event queues recovery."""
        handler = UnderutilizationRecoveryHandler()

        event = {"queue_depth": 0}

        await handler._on_work_queue_exhausted(event)

        assert handler._pending_recovery_event is not None
        assert handler._pending_recovery_event["reason"] == "work_queue_exhausted"
        assert handler._pending_recovery_event["queue_depth"] == 0


class TestUnderutilizationRunCycle:
    """Tests for run cycle logic."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self) -> None:
        """Test run cycle does nothing when disabled."""
        config = UnderutilizationConfig(enabled=False)
        handler = UnderutilizationRecoveryHandler(config=config)
        handler._pending_recovery_event = {"reason": "test"}

        await handler._run_cycle()

        # Pending event should remain
        assert handler._pending_recovery_event is not None

    @pytest.mark.asyncio
    async def test_run_cycle_no_pending(self) -> None:
        """Test run cycle with no pending recovery."""
        handler = UnderutilizationRecoveryHandler()

        with patch.object(handler, "_execute_recovery", new_callable=AsyncMock) as mock:
            await handler._run_cycle()
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_executes_pending(self) -> None:
        """Test run cycle executes pending recovery."""
        handler = UnderutilizationRecoveryHandler()
        handler._pending_recovery_event = {"reason": "test", "timestamp": time.time()}

        with patch.object(handler, "_execute_recovery", new_callable=AsyncMock) as mock:
            await handler._run_cycle()
            mock.assert_called_once()


class TestUnderutilizationRecovery:
    """Tests for recovery execution."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_execute_recovery_already_in_progress(self) -> None:
        """Test recovery skipped when already in progress."""
        handler = UnderutilizationRecoveryHandler()
        handler._recovery_in_progress = True

        with patch.object(handler, "_get_underserved_configs", new_callable=AsyncMock) as mock:
            await handler._execute_recovery({"reason": "test"})
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_recovery_no_configs(self) -> None:
        """Test recovery fails when no configs found."""
        handler = UnderutilizationRecoveryHandler()

        with patch.object(handler, "_get_underserved_configs", new_callable=AsyncMock, return_value=[]):
            with patch.object(handler, "_emit_recovery_started"):
                await handler._execute_recovery({"reason": "test"})

        assert handler._stats.failed_recoveries == 1
        assert handler._recovery_in_progress is False

    @pytest.mark.asyncio
    async def test_execute_recovery_success(self) -> None:
        """Test successful recovery execution."""
        handler = UnderutilizationRecoveryHandler()

        with patch.object(handler, "_get_underserved_configs", new_callable=AsyncMock, return_value=["hex8_2p"]):
            with patch.object(handler, "_inject_work_items", new_callable=AsyncMock, return_value=10):
                with patch.object(handler, "_emit_recovery_started"):
                    with patch.object(handler, "_emit_recovery_completed"):
                        await handler._execute_recovery({"reason": "test"})

        assert handler._stats.successful_recoveries == 1
        assert handler._stats.items_injected == 10
        assert handler._recovery_in_progress is False

    @pytest.mark.asyncio
    async def test_execute_recovery_no_items_injected(self) -> None:
        """Test recovery fails when no items injected."""
        handler = UnderutilizationRecoveryHandler()

        with patch.object(handler, "_get_underserved_configs", new_callable=AsyncMock, return_value=["hex8_2p"]):
            with patch.object(handler, "_inject_work_items", new_callable=AsyncMock, return_value=0):
                with patch.object(handler, "_emit_recovery_started"):
                    with patch.object(handler, "_emit_recovery_failed"):
                        await handler._execute_recovery({"reason": "test"})

        assert handler._stats.failed_recoveries == 1


class TestUnderutilizationGetUnderservedConfigs:
    """Tests for getting underserved configurations."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_get_underserved_from_scheduler(self) -> None:
        """Test getting configs from selfplay scheduler."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_underserved_configs.return_value = ["hex8_2p", "square8_4p"]

        handler = UnderutilizationRecoveryHandler(selfplay_scheduler=mock_scheduler)

        configs = await handler._get_underserved_configs()

        assert "hex8_2p" in configs
        assert "square8_4p" in configs

    @pytest.mark.asyncio
    async def test_get_underserved_fallback_to_priorities(self) -> None:
        """Test fallback to config priorities."""
        mock_scheduler = MagicMock()
        mock_scheduler.get_config_priorities.return_value = {
            "hex8_2p": 100,
            "square8_4p": 80,
            "hex8_3p": 60,
        }
        # No get_underserved_configs method
        del mock_scheduler.get_underserved_configs

        handler = UnderutilizationRecoveryHandler(selfplay_scheduler=mock_scheduler)

        configs = await handler._get_underserved_configs()

        # Should return highest priority configs
        assert "hex8_2p" in configs

    @pytest.mark.asyncio
    async def test_get_underserved_fallback_to_canonical(self) -> None:
        """Test fallback to canonical configs when no scheduler."""
        handler = UnderutilizationRecoveryHandler()

        configs = await handler._get_underserved_configs()

        # Should return canonical configs
        assert len(configs) > 0
        # All configs should be valid format
        for config in configs:
            assert "_" in config
            assert config.endswith("p")


class TestUnderutilizationInjectWorkItems:
    """Tests for work item injection."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    @pytest.mark.asyncio
    async def test_inject_no_queue(self) -> None:
        """Test injection fails without work queue."""
        handler = UnderutilizationRecoveryHandler()

        count = await handler._inject_work_items(["hex8_2p"])

        assert count == 0

    @pytest.mark.asyncio
    async def test_inject_with_push_queue(self) -> None:
        """Test injection with push-based queue."""
        mock_queue = MagicMock()
        mock_queue.push = AsyncMock()

        handler = UnderutilizationRecoveryHandler(work_queue=mock_queue)

        count = await handler._inject_work_items(["hex8_2p"])

        assert count > 0
        mock_queue.push.assert_called()

    @pytest.mark.asyncio
    async def test_inject_with_put_queue(self) -> None:
        """Test injection with put-based queue."""
        mock_queue = MagicMock()
        mock_queue.put = AsyncMock()
        # Remove push to force put
        del mock_queue.push

        handler = UnderutilizationRecoveryHandler(work_queue=mock_queue)

        count = await handler._inject_work_items(["hex8_2p"])

        assert count > 0
        mock_queue.put.assert_called()

    @pytest.mark.asyncio
    async def test_inject_with_add_item_queue(self) -> None:
        """Test injection with add_item-based queue."""
        mock_queue = MagicMock(spec=[])  # Empty spec - no methods by default
        mock_queue.add_item = MagicMock()

        handler = UnderutilizationRecoveryHandler(work_queue=mock_queue)

        count = await handler._inject_work_items(["hex8_2p"])

        assert count > 0

    @pytest.mark.asyncio
    async def test_inject_tracks_configs(self) -> None:
        """Test injection tracks targeted configs."""
        mock_queue = MagicMock()
        mock_queue.push = AsyncMock()

        handler = UnderutilizationRecoveryHandler(work_queue=mock_queue)

        await handler._inject_work_items(["hex8_2p", "square8_4p"])

        assert "hex8_2p" in handler._stats.configs_targeted
        assert "square8_4p" in handler._stats.configs_targeted


class TestUnderutilizationCreateWorkItem:
    """Tests for work item creation."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_create_work_item_hex8_2p(self) -> None:
        """Test work item creation for hex8_2p."""
        handler = UnderutilizationRecoveryHandler()

        item = handler._create_work_item("hex8_2p", 0)

        assert item["config_key"] == "hex8_2p"
        assert item["board_type"] == "hex8"
        assert item["num_players"] == 2
        assert item["work_type"] == "selfplay"
        assert item["source"] == "underutilization_recovery"
        assert "work_id" in item
        assert "created_at" in item

    def test_create_work_item_square8_4p(self) -> None:
        """Test work item creation for square8_4p."""
        handler = UnderutilizationRecoveryHandler()

        item = handler._create_work_item("square8_4p", 1)

        assert item["config_key"] == "square8_4p"
        assert item["board_type"] == "square8"
        assert item["num_players"] == 4

    def test_create_work_item_hexagonal_3p(self) -> None:
        """Test work item creation for hexagonal_3p."""
        handler = UnderutilizationRecoveryHandler()

        item = handler._create_work_item("hexagonal_3p", 2)

        assert item["config_key"] == "hexagonal_3p"
        assert item["board_type"] == "hexagonal"
        assert item["num_players"] == 3

    def test_create_work_item_priority(self) -> None:
        """Test work item has high priority."""
        config = UnderutilizationConfig(high_priority=200)
        handler = UnderutilizationRecoveryHandler(config=config)

        item = handler._create_work_item("hex8_2p", 0)

        assert item["priority"] == 200


class TestUnderutilizationSetters:
    """Tests for late-binding setters."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_set_work_queue(self) -> None:
        """Test setting work queue after init."""
        handler = UnderutilizationRecoveryHandler()
        assert handler._work_queue is None

        mock_queue = MagicMock()
        handler.set_work_queue(mock_queue)

        assert handler._work_queue is mock_queue

    def test_set_selfplay_scheduler(self) -> None:
        """Test setting selfplay scheduler after init."""
        handler = UnderutilizationRecoveryHandler()
        assert handler._selfplay_scheduler is None

        mock_scheduler = MagicMock()
        handler.set_selfplay_scheduler(mock_scheduler)

        assert handler._selfplay_scheduler is mock_scheduler


class TestUnderutilizationEventEmission:
    """Tests for recovery event emission."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        UnderutilizationRecoveryHandler.reset_instance()

    def test_emit_recovery_started(self) -> None:
        """Test recovery started event emission."""
        handler = UnderutilizationRecoveryHandler()
        handler._stats.total_recoveries = 1

        event = {"reason": "test_reason"}

        # Patch at the source module where it's imported from
        with patch("app.coordination.event_emission_helpers.safe_emit_event") as mock:
            handler._emit_recovery_started(event)
            mock.assert_called_once()

    def test_emit_recovery_completed(self) -> None:
        """Test recovery completed event emission."""
        handler = UnderutilizationRecoveryHandler()
        handler._stats.total_recoveries = 1

        event = {"reason": "test_reason"}

        # Patch at the source module where it's imported from
        with patch("app.coordination.event_emission_helpers.safe_emit_event") as mock:
            handler._emit_recovery_completed(event, 10, ["hex8_2p"])
            mock.assert_called_once()

    def test_emit_recovery_failed(self) -> None:
        """Test recovery failed event emission."""
        handler = UnderutilizationRecoveryHandler()
        handler._stats.total_recoveries = 1

        event = {"reason": "test_reason"}

        # Patch at the source module where it's imported from
        with patch("app.coordination.event_emission_helpers.safe_emit_event") as mock:
            handler._emit_recovery_failed(event, "no_items")
            mock.assert_called_once()
