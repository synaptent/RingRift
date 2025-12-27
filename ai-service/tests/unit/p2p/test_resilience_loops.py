"""Tests for Resilience Loops - SelfHealingLoop and PredictiveMonitoringLoop.

December 2025: Tests for the P2P resilience infrastructure including:
- SelfHealingLoop: Stuck job detection and stale process cleanup
- PredictiveMonitoringLoop: Proactive monitoring and alerting
"""

import asyncio
import pytest
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.p2p.loops.resilience_loops import (
    SelfHealingLoop,
    PredictiveMonitoringLoop,
    SelfHealingConfig,
    PredictiveMonitoringConfig,
    STALE_PROCESS_CHECK_INTERVAL,
)
from scripts.p2p.loops.base import LoopStats


# =============================================================================
# Mock Classes
# =============================================================================


class MockHealthManager:
    """Mock implementation of HealthManagerProtocol."""

    def __init__(self):
        self.work_queue = None
        self.stuck_jobs_to_return: list[tuple[Any, float]] = []
        self.recover_stuck_job_result = MagicMock()
        self.recover_stuck_job_result.value = "success"

    def set_work_queue(self, work_queue: Any) -> None:
        self.work_queue = work_queue

    def find_stuck_jobs(self, work_items: list[Any]) -> list[tuple[Any, float]]:
        return self.stuck_jobs_to_return

    async def recover_stuck_job(self, work_item: Any, expected_timeout: float) -> Any:
        return self.recover_stuck_job_result


class MockWorkQueue:
    """Mock implementation of WorkQueueProtocol."""

    def __init__(self):
        self.queue_status = {
            "running": [],
            "by_status": {"pending": 0, "running": 0, "completed": 0},
        }

    def get_queue_status(self) -> dict[str, Any]:
        return self.queue_status


class MockPeer:
    """Mock implementation of PeerProtocol."""

    def __init__(
        self,
        node_id: str,
        alive: bool = True,
        disk_percent: float = 50.0,
        mem_percent: float = 40.0,
    ):
        self.node_id = node_id
        self._alive = alive
        self.disk_percent = disk_percent
        self.mem_percent = mem_percent

    def is_alive(self) -> bool:
        return self._alive


class MockAlertManager:
    """Mock implementation of AlertManagerProtocol."""

    def __init__(self):
        self.disk_usages: dict[str, float] = {}
        self.memory_usages: dict[str, float] = {}
        self.queue_depths: list[int] = []
        self.alerts_to_return: list[Any] = []

    def record_disk_usage(self, node_id: str, pct: float) -> None:
        self.disk_usages[node_id] = pct

    def record_memory_usage(self, node_id: str, pct: float) -> None:
        self.memory_usages[node_id] = pct

    def record_queue_depth(self, depth: int) -> None:
        self.queue_depths.append(depth)

    async def run_all_checks(
        self,
        node_ids: list[str],
        model_ids: list[str],
        last_training_time: float,
    ) -> list[Any]:
        return self.alerts_to_return


class MockNotifier:
    """Mock implementation of NotifierProtocol."""

    def __init__(self):
        self.notifications: list[dict[str, Any]] = []

    async def send(
        self,
        title: str,
        message: str,
        level: str,
        fields: dict[str, str],
        node_id: str,
    ) -> None:
        self.notifications.append({
            "title": title,
            "message": message,
            "level": level,
            "fields": fields,
            "node_id": node_id,
        })


@dataclass
class MockWorkItem:
    """Mock work item for stuck job testing."""

    work_id: str
    claimed_by: str
    started_at: float = 0.0


@dataclass
class MockAlert:
    """Mock alert for predictive monitoring testing."""

    alert_type: Any = None
    message: str = "Test alert"
    severity: Any = None
    action: str = "test_action"
    target_id: str = "test_node"


class MockAlertType:
    value = "test_alert"


class MockSeverity:
    value = "warning"


# =============================================================================
# SelfHealingConfig Tests
# =============================================================================


class TestSelfHealingConfig:
    """Tests for SelfHealingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SelfHealingConfig()
        assert config.healing_interval_seconds == 60.0
        assert config.stale_process_check_interval_seconds == 300.0
        assert config.initial_delay_seconds == 45.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SelfHealingConfig(
            healing_interval_seconds=30.0,
            stale_process_check_interval_seconds=120.0,
            initial_delay_seconds=10.0,
        )
        assert config.healing_interval_seconds == 30.0
        assert config.stale_process_check_interval_seconds == 120.0
        assert config.initial_delay_seconds == 10.0


# =============================================================================
# PredictiveMonitoringConfig Tests
# =============================================================================


class TestPredictiveMonitoringConfig:
    """Tests for PredictiveMonitoringConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PredictiveMonitoringConfig()
        assert config.monitoring_interval_seconds == 300.0
        assert config.initial_delay_seconds == 90.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PredictiveMonitoringConfig(
            monitoring_interval_seconds=60.0,
            initial_delay_seconds=30.0,
        )
        assert config.monitoring_interval_seconds == 60.0
        assert config.initial_delay_seconds == 30.0


# =============================================================================
# SelfHealingLoop Tests
# =============================================================================


class TestSelfHealingLoop:
    """Tests for SelfHealingLoop."""

    def test_initialization(self):
        """Test loop initializes correctly."""
        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=lambda: 0,
        )
        assert loop.name == "self_healing"
        assert loop.interval == 60.0
        assert loop._stale_processes_cleaned == 0
        assert loop._stuck_jobs_recovered == 0

    def test_initialization_with_custom_config(self):
        """Test loop with custom config."""
        config = SelfHealingConfig(healing_interval_seconds=30.0)
        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=lambda: 0,
            config=config,
        )
        assert loop.interval == 30.0

    @pytest.mark.asyncio
    async def test_run_once_as_follower(self):
        """Test that followers skip job recovery but do stale cleanup."""
        cleanup_called = False
        cleanup_count = 3

        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            return cleanup_count

        loop = SelfHealingLoop(
            is_leader=lambda: False,  # Not leader
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=mock_cleanup,
            config=SelfHealingConfig(stale_process_check_interval_seconds=0),
        )

        await loop._run_once()

        # Cleanup should have been called
        assert cleanup_called
        assert loop._stale_processes_cleaned == cleanup_count

    @pytest.mark.asyncio
    async def test_run_once_as_leader_no_stuck_jobs(self):
        """Test leader execution with no stuck jobs."""
        health_manager = MockHealthManager()
        work_queue = MockWorkQueue()

        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: health_manager,
            get_work_queue=lambda: work_queue,
            cleanup_stale_processes=lambda: 0,
            config=SelfHealingConfig(stale_process_check_interval_seconds=0),
        )

        await loop._run_once()

        # Health manager should have work queue wired
        assert health_manager.work_queue is work_queue
        assert loop._stuck_jobs_recovered == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_stale_check_within_interval(self):
        """Test that stale process check is throttled."""
        cleanup_call_count = 0

        def mock_cleanup():
            nonlocal cleanup_call_count
            cleanup_call_count += 1
            return 0

        loop = SelfHealingLoop(
            is_leader=lambda: False,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=mock_cleanup,
            config=SelfHealingConfig(
                stale_process_check_interval_seconds=300,
                healing_interval_seconds=1,
            ),
        )

        # First run should check
        await loop._run_once()
        assert cleanup_call_count == 1

        # Second run within interval should skip
        await loop._run_once()
        assert cleanup_call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_run_once_without_health_manager(self):
        """Test graceful handling when health manager unavailable."""
        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: None,  # Returns None
            get_work_queue=lambda: MockWorkQueue(),
            cleanup_stale_processes=lambda: 0,
            config=SelfHealingConfig(stale_process_check_interval_seconds=0),
        )

        # Should not raise
        await loop._run_once()

    @pytest.mark.asyncio
    async def test_run_once_without_work_queue(self):
        """Test graceful handling when work queue unavailable."""
        health_manager = MockHealthManager()

        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: health_manager,
            get_work_queue=lambda: None,  # Returns None
            cleanup_stale_processes=lambda: 0,
            config=SelfHealingConfig(stale_process_check_interval_seconds=0),
        )

        # Should not raise
        await loop._run_once()

    def test_get_healing_stats(self):
        """Test stats retrieval."""
        loop = SelfHealingLoop(
            is_leader=lambda: True,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=lambda: 0,
        )
        loop._stale_processes_cleaned = 5
        loop._stuck_jobs_recovered = 2
        loop._last_stale_check = 1000.0

        stats = loop.get_healing_stats()

        assert stats["stale_processes_cleaned"] == 5
        assert stats["stuck_jobs_recovered"] == 2
        assert stats["last_stale_check"] == 1000.0
        assert "name" in stats  # From base loop stats


# =============================================================================
# PredictiveMonitoringLoop Tests
# =============================================================================


class TestPredictiveMonitoringLoop:
    """Tests for PredictiveMonitoringLoop."""

    def test_initialization(self):
        """Test loop initializes correctly."""
        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: None,
            get_work_queue=lambda: None,
            get_peers=lambda: [],
            get_notifier=lambda: None,
        )
        assert loop.name == "predictive_monitoring"
        assert loop.interval == 300.0
        assert loop._alerts_sent == 0
        assert loop._checks_performed == 0

    def test_initialization_with_custom_config(self):
        """Test loop with custom config."""
        config = PredictiveMonitoringConfig(monitoring_interval_seconds=60.0)
        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: None,
            get_work_queue=lambda: None,
            get_peers=lambda: [],
            get_notifier=lambda: None,
            config=config,
        )
        assert loop.interval == 60.0

    @pytest.mark.asyncio
    async def test_run_once_as_follower(self):
        """Test that followers skip monitoring."""
        alert_manager = MockAlertManager()

        loop = PredictiveMonitoringLoop(
            is_leader=lambda: False,  # Not leader
            get_alert_manager=lambda: alert_manager,
            get_work_queue=lambda: MockWorkQueue(),
            get_peers=lambda: [],
            get_notifier=lambda: None,
        )

        await loop._run_once()

        # No checks should have been performed
        assert loop._checks_performed == 0

    @pytest.mark.asyncio
    async def test_run_once_records_peer_metrics(self):
        """Test that peer disk/memory metrics are recorded."""
        alert_manager = MockAlertManager()
        peers = [
            MockPeer("node-1", disk_percent=75.0, mem_percent=60.0),
            MockPeer("node-2", disk_percent=50.0, mem_percent=40.0),
            MockPeer("node-3", alive=False),  # Dead peer - should be skipped
        ]

        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: alert_manager,
            get_work_queue=lambda: MockWorkQueue(),
            get_peers=lambda: peers,
            get_notifier=lambda: None,
        )

        await loop._run_once()

        # Check that metrics were recorded for alive peers
        assert alert_manager.disk_usages["node-1"] == 75.0
        assert alert_manager.disk_usages["node-2"] == 50.0
        assert "node-3" not in alert_manager.disk_usages

        assert alert_manager.memory_usages["node-1"] == 60.0
        assert alert_manager.memory_usages["node-2"] == 40.0
        assert "node-3" not in alert_manager.memory_usages

    @pytest.mark.asyncio
    async def test_run_once_records_queue_depth(self):
        """Test that queue depth is recorded."""
        alert_manager = MockAlertManager()
        work_queue = MockWorkQueue()
        work_queue.queue_status = {
            "running": [],
            "by_status": {"pending": 25, "running": 5, "completed": 100},
        }

        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: alert_manager,
            get_work_queue=lambda: work_queue,
            get_peers=lambda: [],
            get_notifier=lambda: None,
        )

        await loop._run_once()

        assert 25 in alert_manager.queue_depths

    @pytest.mark.asyncio
    async def test_run_once_sends_alerts(self):
        """Test that alerts are sent via notifier."""
        alert = MockAlert()
        alert.alert_type = MockAlertType()
        alert.severity = MockSeverity()

        alert_manager = MockAlertManager()
        alert_manager.alerts_to_return = [alert]
        notifier = MockNotifier()

        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: alert_manager,
            get_work_queue=lambda: MockWorkQueue(),
            get_peers=lambda: [],
            get_notifier=lambda: notifier,
        )

        await loop._run_once()

        assert loop._alerts_sent == 1
        assert len(notifier.notifications) == 1
        assert "Proactive Alert" in notifier.notifications[0]["title"]

    @pytest.mark.asyncio
    async def test_run_once_without_alert_manager(self):
        """Test graceful handling when alert manager unavailable."""
        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: None,  # Returns None
            get_work_queue=lambda: MockWorkQueue(),
            get_peers=lambda: [],
            get_notifier=lambda: None,
        )

        # Should not raise
        await loop._run_once()
        assert loop._checks_performed == 0

    @pytest.mark.asyncio
    async def test_run_once_with_custom_model_getter(self):
        """Test that custom model getter is used."""
        alert_manager = MockAlertManager()

        def custom_model_getter():
            return (["model-1", "model-2"], time.time() - 1800)

        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: alert_manager,
            get_work_queue=lambda: MockWorkQueue(),
            get_peers=lambda: [],
            get_notifier=lambda: None,
            get_production_models=custom_model_getter,
        )

        await loop._run_once()

        assert loop._checks_performed == 1

    def test_get_monitoring_stats(self):
        """Test stats retrieval."""
        loop = PredictiveMonitoringLoop(
            is_leader=lambda: True,
            get_alert_manager=lambda: None,
            get_work_queue=lambda: None,
            get_peers=lambda: [],
            get_notifier=lambda: None,
        )
        loop._alerts_sent = 10
        loop._checks_performed = 5

        stats = loop.get_monitoring_stats()

        assert stats["alerts_sent"] == 10
        assert stats["checks_performed"] == 5
        assert "name" in stats  # From base loop stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestResilienceLoopsIntegration:
    """Integration tests for resilience loops."""

    @pytest.mark.asyncio
    async def test_self_healing_start_stop(self):
        """Test SelfHealingLoop start/stop lifecycle."""
        loop = SelfHealingLoop(
            is_leader=lambda: False,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=lambda: 0,
            config=SelfHealingConfig(
                initial_delay_seconds=0.1,
                healing_interval_seconds=0.1,
            ),
        )

        # Start in background
        task = loop.start_background()
        await asyncio.sleep(0.3)

        assert loop.running is True

        # Stop
        loop.stop()
        await asyncio.sleep(0.2)

        assert loop.running is False

    @pytest.mark.asyncio
    async def test_predictive_monitoring_start_stop(self):
        """Test PredictiveMonitoringLoop start/stop lifecycle."""
        loop = PredictiveMonitoringLoop(
            is_leader=lambda: False,
            get_alert_manager=lambda: None,
            get_work_queue=lambda: None,
            get_peers=lambda: [],
            get_notifier=lambda: None,
            config=PredictiveMonitoringConfig(
                initial_delay_seconds=0.1,
                monitoring_interval_seconds=0.1,
            ),
        )

        # Start in background
        task = loop.start_background()
        await asyncio.sleep(0.3)

        assert loop.running is True

        # Stop
        loop.stop()
        await asyncio.sleep(0.2)

        assert loop.running is False

    @pytest.mark.asyncio
    async def test_error_recovery_with_backoff(self):
        """Test that loops recover from errors with backoff."""
        call_count = 0

        def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated failure")
            return 0

        loop = SelfHealingLoop(
            is_leader=lambda: False,
            get_health_manager=lambda: None,
            get_work_queue=lambda: None,
            cleanup_stale_processes=failing_cleanup,
            config=SelfHealingConfig(
                initial_delay_seconds=0,
                healing_interval_seconds=0.05,
                stale_process_check_interval_seconds=0,
            ),
        )

        # Start and let it run with errors
        task = loop.start_background()
        await asyncio.sleep(0.5)

        # Should have attempted multiple runs
        assert call_count >= 2

        loop.stop()
        await asyncio.sleep(0.1)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_stale_process_check_interval(self):
        """Test that constant has reasonable value."""
        assert STALE_PROCESS_CHECK_INTERVAL >= 60  # At least 1 minute
        assert STALE_PROCESS_CHECK_INTERVAL <= 600  # At most 10 minutes
