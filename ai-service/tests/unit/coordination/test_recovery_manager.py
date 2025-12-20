"""Tests for RecoveryManager.

Tests core functionality:
- Job recovery state tracking
- Node recovery state tracking
- Recovery attempt limits
- Escalation behavior
- Event recording
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.coordination.recovery_manager import (
    JobRecoveryState,
    NodeRecoveryState,
    RecoveryAction,
    RecoveryConfig,
    RecoveryManager,
    RecoveryResult,
    load_recovery_config_from_yaml,
)


@pytest.fixture
def config():
    """Provide a test configuration with short cooldowns."""
    return RecoveryConfig(
        enabled=True,
        stuck_job_timeout_multiplier=1.5,
        max_recovery_attempts_per_node=3,
        max_recovery_attempts_per_job=2,
        recovery_attempt_cooldown=1,  # Short for testing
        consecutive_failures_for_escalation=3,
        escalation_cooldown=1,  # Short for testing
        node_unhealthy_after_failures=3,
        node_recovery_timeout=5,
    )


@pytest.fixture
def manager(config):
    """Provide a fresh RecoveryManager for each test."""
    return RecoveryManager(config=config)


class TestRecoveryConfig:
    """Tests for RecoveryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RecoveryConfig()

        assert config.enabled is True
        assert config.stuck_job_timeout_multiplier == 1.5
        assert config.max_recovery_attempts_per_node == 3
        assert config.max_recovery_attempts_per_job == 2

    def test_load_from_yaml(self):
        """Test loading config from YAML dict."""
        yaml_config = {
            "self_healing": {
                "enabled": False,
                "stuck_job_timeout_multiplier": 2.0,
                "max_recovery_attempts_per_node": 5,
            }
        }

        config = load_recovery_config_from_yaml(yaml_config)

        assert config.enabled is False
        assert config.stuck_job_timeout_multiplier == 2.0
        assert config.max_recovery_attempts_per_node == 5
        # Defaults for unspecified values
        assert config.max_recovery_attempts_per_job == 2


class TestJobRecoveryState:
    """Tests for job recovery state tracking."""

    def test_get_job_state_creates_new(self, manager):
        """Test getting job state creates new state."""
        state = manager._get_job_state("job-123")

        assert isinstance(state, JobRecoveryState)
        assert state.work_id == "job-123"
        assert state.recovery_attempts == 0

    def test_get_job_state_returns_existing(self, manager):
        """Test getting job state returns existing state."""
        state1 = manager._get_job_state("job-123")
        state1.recovery_attempts = 5

        state2 = manager._get_job_state("job-123")

        assert state2.recovery_attempts == 5
        assert state1 is state2

    def test_can_attempt_job_recovery(self, manager):
        """Test job recovery attempt checking."""
        # Fresh job should allow recovery
        assert manager._can_attempt_job_recovery("job-123") is True

        # After max attempts, should deny
        state = manager._get_job_state("job-123")
        state.recovery_attempts = manager.config.max_recovery_attempts_per_job

        assert manager._can_attempt_job_recovery("job-123") is False

    def test_reset_job_state(self, manager):
        """Test resetting job state."""
        state = manager._get_job_state("job-123")
        state.recovery_attempts = 5

        manager.reset_job_state("job-123")

        # State should be removed, getting it creates fresh one
        assert "job-123" not in manager._job_states


class TestNodeRecoveryState:
    """Tests for node recovery state tracking."""

    def test_get_node_state_creates_new(self, manager):
        """Test getting node state creates new state."""
        state = manager._get_node_state("node-1")

        assert isinstance(state, NodeRecoveryState)
        assert state.node_id == "node-1"
        assert state.recovery_attempts == 0
        assert state.is_escalated is False

    def test_can_attempt_node_recovery_respects_limit(self, manager):
        """Test node recovery respects attempt limit."""
        # Fresh node should allow recovery
        assert manager._can_attempt_node_recovery("node-1") is True

        # After max attempts, should deny
        state = manager._get_node_state("node-1")
        state.recovery_attempts = manager.config.max_recovery_attempts_per_node

        assert manager._can_attempt_node_recovery("node-1") is False

    def test_can_attempt_node_recovery_respects_escalation(self, manager):
        """Test node recovery respects escalation state."""
        state = manager._get_node_state("node-1")
        state.is_escalated = True
        state.last_escalation_time = time.time()

        # Should be denied during escalation cooldown
        assert manager._can_attempt_node_recovery("node-1") is False

    def test_can_attempt_node_recovery_after_cooldown(self, manager):
        """Test node recovery allowed after escalation cooldown."""
        state = manager._get_node_state("node-1")
        state.is_escalated = True
        state.last_escalation_time = time.time() - 10  # Past cooldown

        # Should be allowed after cooldown (config has 1s cooldown)
        assert manager._can_attempt_node_recovery("node-1") is True
        # And escalation flag should be reset
        assert state.is_escalated is False

    def test_reset_node_state(self, manager):
        """Test resetting node state."""
        state = manager._get_node_state("node-1")
        state.recovery_attempts = 5
        state.consecutive_failures = 3
        state.is_escalated = True

        manager.reset_node_state("node-1")

        # State should be reset
        new_state = manager._get_node_state("node-1")
        assert new_state.recovery_attempts == 0
        assert new_state.consecutive_failures == 0
        assert new_state.is_escalated is False


class TestRecoverStuckJob:
    """Tests for stuck job recovery."""

    @pytest.mark.asyncio
    async def test_recover_stuck_job_disabled(self, config):
        """Test stuck job recovery when disabled."""
        config.enabled = False
        manager = RecoveryManager(config=config)

        work_item = MagicMock()
        work_item.work_id = "job-123"
        work_item.claimed_by = "node-1"

        result = await manager.recover_stuck_job(work_item, 60.0)

        assert result == RecoveryResult.SKIPPED

    @pytest.mark.asyncio
    async def test_recover_stuck_job_max_attempts(self, manager):
        """Test stuck job recovery at max attempts."""
        work_item = MagicMock()
        work_item.work_id = "job-123"
        work_item.claimed_by = "node-1"

        # Set max attempts
        state = manager._get_job_state("job-123")
        state.recovery_attempts = manager.config.max_recovery_attempts_per_job

        result = await manager.recover_stuck_job(work_item, 60.0)

        assert result == RecoveryResult.ESCALATED

    @pytest.mark.asyncio
    async def test_recover_stuck_job_success(self, manager):
        """Test successful stuck job recovery."""
        work_item = MagicMock()
        work_item.work_id = "job-123"
        work_item.claimed_by = "node-1"

        # Set up mock callbacks
        kill_callback = AsyncMock()
        manager.set_kill_job_callback(kill_callback)

        work_queue = MagicMock()
        manager.set_work_queue(work_queue)

        result = await manager.recover_stuck_job(work_item, 60.0)

        assert result == RecoveryResult.SUCCESS
        kill_callback.assert_called_once()
        work_queue.fail_work.assert_called_once()


class TestRecoverUnhealthyNode:
    """Tests for unhealthy node recovery."""

    @pytest.mark.asyncio
    async def test_recover_node_disabled(self, config):
        """Test node recovery when disabled."""
        config.enabled = False
        manager = RecoveryManager(config=config)

        result = await manager.recover_unhealthy_node("node-1", "test reason")

        assert result == RecoveryResult.SKIPPED

    @pytest.mark.asyncio
    async def test_recover_node_escalates_at_limit(self, manager):
        """Test node recovery escalates at attempt limit."""
        # Set max attempts
        state = manager._get_node_state("node-1")
        state.recovery_attempts = manager.config.max_recovery_attempts_per_node

        notifier = AsyncMock()
        manager.set_notifier(notifier)

        result = await manager.recover_unhealthy_node("node-1", "test reason")

        assert result == RecoveryResult.ESCALATED

    @pytest.mark.asyncio
    async def test_recover_node_success(self, manager):
        """Test successful node recovery."""
        restart_callback = AsyncMock(return_value=True)
        manager.set_restart_services_callback(restart_callback)

        result = await manager.recover_unhealthy_node("node-1", "test reason")

        assert result == RecoveryResult.SUCCESS
        restart_callback.assert_called_once_with("node-1")

        # State should be reset on success
        state = manager._get_node_state("node-1")
        assert state.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_recover_node_failure_tracks_consecutive(self, manager):
        """Test node recovery failure tracks consecutive failures."""
        restart_callback = AsyncMock(return_value=False)
        manager.set_restart_services_callback(restart_callback)

        result = await manager.recover_unhealthy_node("node-1", "test reason")

        assert result == RecoveryResult.FAILED

        state = manager._get_node_state("node-1")
        assert state.consecutive_failures == 1


class TestEventRecording:
    """Tests for recovery event recording."""

    def test_record_event(self, manager):
        """Test event recording."""
        manager._record_event(
            action=RecoveryAction.RESTART_NODE_SERVICES,
            target_type="node",
            target_id="node-1",
            result=RecoveryResult.SUCCESS,
            reason="test reason",
            duration=1.5,
        )

        assert len(manager._recovery_history) == 1
        event = manager._recovery_history[0]
        assert event.action == RecoveryAction.RESTART_NODE_SERVICES
        assert event.target_id == "node-1"
        assert event.result == RecoveryResult.SUCCESS

    def test_event_history_limit(self, manager):
        """Test event history is limited to 500."""
        for i in range(600):
            manager._record_event(
                action=RecoveryAction.NONE,
                target_type="test",
                target_id=f"target-{i}",
                result=RecoveryResult.SUCCESS,
                reason="test",
            )

        assert len(manager._recovery_history) == 500


class TestStats:
    """Tests for recovery statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test async stats retrieval."""
        # Record some events
        manager._record_event(
            action=RecoveryAction.RESTART_NODE_SERVICES,
            target_type="node",
            target_id="node-1",
            result=RecoveryResult.SUCCESS,
            reason="test",
        )

        stats = await manager.get_stats()

        assert stats["name"] == "RecoveryManager"
        assert stats["enabled"] is True
        assert "recoveries_last_hour" in stats
        assert stats["recoveries_last_hour"]["success"] == 1

    def test_get_recovery_stats_sync(self, manager):
        """Test sync stats retrieval."""
        stats = manager.get_recovery_stats()

        assert stats["name"] == "RecoveryManager"
        assert "status" in stats
        assert "recoveries_last_hour" in stats
        assert "nodes_tracked" in stats
        assert "jobs_tracked" in stats


class TestFindStuckJobs:
    """Tests for stuck job detection."""

    def test_find_stuck_jobs_empty(self, manager):
        """Test finding stuck jobs with empty list."""
        stuck = manager.find_stuck_jobs([])
        assert stuck == []

    def test_find_stuck_jobs_none_stuck(self, manager):
        """Test finding stuck jobs when none are stuck."""
        item = MagicMock()
        item.work_id = "job-1"
        item.timeout_seconds = 60
        item.started_at = time.time()  # Just started

        stuck = manager.find_stuck_jobs([item])
        assert stuck == []

    def test_find_stuck_jobs_detects_stuck(self, manager):
        """Test finding stuck jobs detects stuck items."""
        item = MagicMock()
        item.work_id = "job-1"
        item.timeout_seconds = 60
        # Started 2 minutes ago (past 1.5x timeout)
        item.started_at = time.time() - 120

        stuck = manager.find_stuck_jobs([item])

        assert len(stuck) == 1
        assert stuck[0][0] == item
        assert stuck[0][1] == 60  # Expected timeout

    def test_find_stuck_jobs_custom_multiplier(self, manager):
        """Test finding stuck jobs with custom multiplier."""
        item = MagicMock()
        item.work_id = "job-1"
        item.timeout_seconds = 60
        item.started_at = time.time() - 90  # 1.5x timeout

        # With 2x multiplier, should not be stuck
        stuck = manager.find_stuck_jobs([item], timeout_multiplier=2.0)
        assert stuck == []

        # With 1.0x multiplier, should be stuck
        stuck = manager.find_stuck_jobs([item], timeout_multiplier=1.0)
        assert len(stuck) == 1


class TestDependencyInjection:
    """Tests for dependency injection via setters."""

    def test_set_work_queue(self, manager):
        """Test setting work queue."""
        work_queue = MagicMock()
        manager.set_work_queue(work_queue)

        assert manager._work_queue is work_queue

    def test_set_notifier(self, manager):
        """Test setting notifier."""
        notifier = MagicMock()
        manager.set_notifier(notifier)

        assert manager._notifier is notifier

    def test_set_callbacks(self, manager):
        """Test setting callback functions."""
        kill_cb = AsyncMock()
        restart_cb = AsyncMock()
        reboot_cb = AsyncMock()

        manager.set_kill_job_callback(kill_cb)
        manager.set_restart_services_callback(restart_cb)
        manager.set_reboot_node_callback(reboot_cb)

        assert manager._kill_job_callback is kill_cb
        assert manager._restart_services_callback is restart_cb
        assert manager._reboot_node_callback is reboot_cb
