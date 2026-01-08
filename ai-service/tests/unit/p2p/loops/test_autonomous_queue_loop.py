"""Tests for AutonomousQueuePopulationLoop.

January 2026: Comprehensive tests for autonomous queue population, activation/deactivation
logic, local queue management, and work item claiming.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loops.autonomous_queue_loop import (
    AutonomousQueueConfig,
    AutonomousQueuePopulationLoop,
    AutonomousQueueState,
    DEFAULT_CHECK_INTERVAL_SECONDS,
    DEFAULT_HEALTHY_QUEUE_DEPTH,
    DEFAULT_LEADER_STABLE_DURATION_SECONDS,
    DEFAULT_LOCAL_QUEUE_TARGET,
    DEFAULT_NO_LEADER_THRESHOLD_SECONDS,
    DEFAULT_POPULATION_BATCH_SIZE,
    DEFAULT_QUEUE_STARVATION_DURATION_SECONDS,
    DEFAULT_QUEUE_STARVATION_THRESHOLD,
)


# =============================================================================
# Mock Orchestrator Factory
# =============================================================================


def create_mock_orchestrator(
    *,
    leader_id: str | None = "leader-node-1",
    node_id: str = "test-node-1",
    queue_depth: int = 100,
) -> MagicMock:
    """Create a mock orchestrator with configurable state."""
    mock = MagicMock()
    mock.leader_id = leader_id
    mock.node_id = node_id
    mock.selfplay_scheduler = None
    mock._cluster_config = None
    mock._role = "worker"

    # Mock work queue
    mock._work_queue = MagicMock()
    mock._work_queue.qsize.return_value = queue_depth

    return mock


# =============================================================================
# AutonomousQueueConfig Tests
# =============================================================================


class TestAutonomousQueueConfig:
    """Tests for AutonomousQueueConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AutonomousQueueConfig()

        assert config.no_leader_threshold_seconds == DEFAULT_NO_LEADER_THRESHOLD_SECONDS
        assert config.queue_starvation_threshold == DEFAULT_QUEUE_STARVATION_THRESHOLD
        assert config.queue_starvation_duration_seconds == DEFAULT_QUEUE_STARVATION_DURATION_SECONDS
        assert config.healthy_queue_depth == DEFAULT_HEALTHY_QUEUE_DEPTH
        assert config.leader_stable_duration_seconds == DEFAULT_LEADER_STABLE_DURATION_SECONDS
        assert config.local_queue_target == DEFAULT_LOCAL_QUEUE_TARGET
        assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL_SECONDS
        assert config.population_batch_size == DEFAULT_POPULATION_BATCH_SIZE
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AutonomousQueueConfig(
            no_leader_threshold_seconds=120.0,
            queue_starvation_threshold=5,
            queue_starvation_duration_seconds=60.0,
            healthy_queue_depth=100,
            leader_stable_duration_seconds=30.0,
            local_queue_target=10,
            check_interval_seconds=15.0,
            population_batch_size=3,
            enabled=False,
        )

        assert config.no_leader_threshold_seconds == 120.0
        assert config.queue_starvation_threshold == 5
        assert config.queue_starvation_duration_seconds == 60.0
        assert config.healthy_queue_depth == 100
        assert config.leader_stable_duration_seconds == 30.0
        assert config.local_queue_target == 10
        assert config.check_interval_seconds == 15.0
        assert config.population_batch_size == 3
        assert config.enabled is False

    def test_from_env(self) -> None:
        """Test config creation from environment variables."""
        with patch.dict("os.environ", {
            "RINGRIFT_AUTONOMOUS_QUEUE_NO_LEADER_THRESHOLD": "180",
            "RINGRIFT_AUTONOMOUS_QUEUE_STARVATION_THRESHOLD": "15",
            "RINGRIFT_AUTONOMOUS_QUEUE_ENABLED": "false",
        }):
            config = AutonomousQueueConfig.from_env()

        assert config.no_leader_threshold_seconds == 180.0
        assert config.queue_starvation_threshold == 15
        assert config.enabled is False


# =============================================================================
# AutonomousQueueState Tests
# =============================================================================


class TestAutonomousQueueState:
    """Tests for AutonomousQueueState dataclass."""

    def test_initial_state(self) -> None:
        """Test initial state values."""
        state = AutonomousQueueState()

        assert state.activated is False
        assert state.activation_time is None
        assert state.activation_reason == ""
        assert state.queue_starved_since is None
        assert state.items_populated == 0
        assert state.deactivation_count == 0

    def test_activate(self) -> None:
        """Test activation sets proper state."""
        state = AutonomousQueueState()

        state.activate("test_reason")

        assert state.activated is True
        assert state.activation_time is not None
        assert state.activation_reason == "test_reason"

    def test_deactivate(self) -> None:
        """Test deactivation clears state."""
        state = AutonomousQueueState()
        state.activate("test_reason")
        state.items_populated = 10

        state.deactivate("recovery")

        assert state.activated is False
        assert state.activation_time is None
        assert state.activation_reason == ""
        assert state.items_populated == 0
        assert state.deactivation_count == 1

    def test_deactivate_increments_count(self) -> None:
        """Test deactivation increments deactivation count."""
        state = AutonomousQueueState()

        state.activate("reason1")
        state.deactivate("recovery1")
        state.activate("reason2")
        state.deactivate("recovery2")

        assert state.deactivation_count == 2

    def test_deactivate_when_not_active_does_not_increment(self) -> None:
        """Test deactivation when not active doesn't increment count."""
        state = AutonomousQueueState()

        state.deactivate("reason")

        assert state.deactivation_count == 0

    def test_record_leader_seen(self) -> None:
        """Test recording leader seen updates timestamp."""
        state = AutonomousQueueState()
        old_time = state.last_leader_seen

        time.sleep(0.01)  # Small delay
        state.record_leader_seen()

        assert state.last_leader_seen > old_time

    def test_record_queue_healthy_clears_starvation(self) -> None:
        """Test recording healthy queue clears starvation timer."""
        state = AutonomousQueueState()
        state.record_queue_starved()
        assert state.queue_starved_since is not None

        state.record_queue_healthy()

        assert state.queue_starved_since is None

    def test_record_queue_starved_sets_timer(self) -> None:
        """Test recording queue starved sets timer."""
        state = AutonomousQueueState()
        assert state.queue_starved_since is None

        state.record_queue_starved()

        assert state.queue_starved_since is not None

    def test_record_queue_starved_does_not_reset_timer(self) -> None:
        """Test repeated starvation recording doesn't reset timer."""
        state = AutonomousQueueState()
        state.record_queue_starved()
        first_time = state.queue_starved_since

        time.sleep(0.01)
        state.record_queue_starved()

        assert state.queue_starved_since == first_time

    def test_get_no_leader_duration(self) -> None:
        """Test calculating no leader duration."""
        state = AutonomousQueueState()
        state.last_leader_seen = time.time() - 60  # 60 seconds ago

        duration = state.get_no_leader_duration()

        assert 59 < duration < 62  # Allow some timing slack

    def test_get_starvation_duration_when_not_starved(self) -> None:
        """Test starvation duration is 0 when not starved."""
        state = AutonomousQueueState()

        duration = state.get_starvation_duration()

        assert duration == 0.0

    def test_get_starvation_duration_when_starved(self) -> None:
        """Test starvation duration calculation."""
        state = AutonomousQueueState()
        state.queue_starved_since = time.time() - 30

        duration = state.get_starvation_duration()

        assert 29 < duration < 32


# =============================================================================
# AutonomousQueuePopulationLoop Initialization Tests
# =============================================================================


class TestLoopInitialization:
    """Tests for loop initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        orchestrator = create_mock_orchestrator()

        loop = AutonomousQueuePopulationLoop(orchestrator)

        assert loop.name == "autonomous_queue_population"
        assert loop.interval == DEFAULT_CHECK_INTERVAL_SECONDS
        assert loop.enabled is True
        assert loop.is_activated is False
        assert loop.local_queue_depth == 0

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(
            check_interval_seconds=15.0,
            enabled=False,
        )

        loop = AutonomousQueuePopulationLoop(orchestrator, config)

        assert loop.interval == 15.0
        assert loop.enabled is False

    def test_initialization_creates_fresh_state(self) -> None:
        """Test that initialization creates fresh state."""
        orchestrator = create_mock_orchestrator()

        loop = AutonomousQueuePopulationLoop(orchestrator)

        assert loop._state.activated is False
        assert loop._state.items_populated == 0


# =============================================================================
# Activation Logic Tests
# =============================================================================


class TestActivationLogic:
    """Tests for activation condition checking."""

    def test_activates_on_no_leader(self) -> None:
        """Test activation when no leader for too long."""
        orchestrator = create_mock_orchestrator(leader_id=None)
        config = AutonomousQueueConfig(no_leader_threshold_seconds=5.0)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.last_leader_seen = time.time() - 10  # 10 seconds without leader
        loop._grace_period_complete = True

        loop._check_should_activate(None, 100)

        assert loop._state.activated is True
        assert "no_leader" in loop._state.activation_reason

    def test_activates_on_queue_starvation(self) -> None:
        """Test activation when queue starved for too long."""
        orchestrator = create_mock_orchestrator(queue_depth=5)
        config = AutonomousQueueConfig(
            queue_starvation_threshold=10,
            queue_starvation_duration_seconds=5.0,
        )
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.queue_starved_since = time.time() - 10  # 10 seconds starved
        loop._grace_period_complete = True

        loop._check_should_activate("leader-1", 5)

        assert loop._state.activated is True
        assert "starved" in loop._state.activation_reason

    def test_does_not_activate_with_leader_and_healthy_queue(self) -> None:
        """Test no activation when leader present and queue healthy."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.last_leader_seen = time.time()
        loop._grace_period_complete = True

        loop._check_should_activate("leader-1", 100)

        assert loop._state.activated is False


# =============================================================================
# Deactivation Logic Tests
# =============================================================================


class TestDeactivationLogic:
    """Tests for deactivation condition checking."""

    def test_deactivates_when_leader_stable_and_queue_healthy(self) -> None:
        """Test deactivation when leader is stable and queue is healthy."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(
            leader_stable_duration_seconds=5.0,
            healthy_queue_depth=50,
        )
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")
        loop._state.last_leader_seen = time.time()  # Leader just seen

        loop._check_should_deactivate("leader-1", 100)

        assert loop._state.activated is False

    def test_deactivates_when_queue_very_healthy(self) -> None:
        """Test deactivation when queue is very healthy (2x threshold)."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(healthy_queue_depth=50)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")

        loop._check_should_deactivate(None, 150)  # 3x threshold

        assert loop._state.activated is False

    def test_does_not_deactivate_with_low_queue(self) -> None:
        """Test no deactivation when queue is still low."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(healthy_queue_depth=50)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")

        loop._check_should_deactivate("leader-1", 30)  # Below healthy threshold

        assert loop._state.activated is True


# =============================================================================
# Local Queue Population Tests
# =============================================================================


class TestLocalQueuePopulation:
    """Tests for local queue population."""

    @pytest.mark.asyncio
    async def test_populates_local_queue_when_activated(self) -> None:
        """Test that local queue is populated when activated."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(
            local_queue_target=10,
            population_batch_size=3,
        )
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")

        await loop._populate_local_queue()

        assert loop.local_queue_depth == 3

    @pytest.mark.asyncio
    async def test_does_not_overfill_queue(self) -> None:
        """Test that queue is not filled beyond target."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(
            local_queue_target=5,
            population_batch_size=10,
        )
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")

        await loop._populate_local_queue()

        assert loop.local_queue_depth <= config.local_queue_target

    @pytest.mark.asyncio
    async def test_skips_population_when_at_target(self) -> None:
        """Test that population is skipped when at target."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(local_queue_target=3)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._state.activate("test")
        loop._local_queue = [{}, {}, {}]  # Already at target

        await loop._populate_local_queue()

        assert loop.local_queue_depth == 3  # Unchanged


# =============================================================================
# Work Item Creation Tests
# =============================================================================


class TestWorkItemCreation:
    """Tests for work item creation."""

    def test_creates_valid_work_item(self) -> None:
        """Test that created work items have required fields."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        item = loop._create_work_item()

        assert item is not None
        assert "work_id" in item
        assert "work_type" in item
        assert item["work_type"] == "selfplay"
        assert "config_key" in item
        assert "board_type" in item
        assert "num_players" in item
        assert item["source"] == "autonomous_queue"

    def test_work_items_rotate_through_configs(self) -> None:
        """Test that work items rotate through configs."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        items = []
        for _ in range(3):
            item = loop._create_work_item()
            loop._state.items_populated += 1
            items.append(item)

        config_keys = [item["config_key"] for item in items]
        # Should have different configs
        assert len(set(config_keys)) > 1 or config_keys == ["hex8_2p", "hex8_3p", "hex8_4p"][:len(config_keys)]

    def test_uses_scheduler_priorities_when_available(self) -> None:
        """Test that scheduler priorities are used when available."""
        orchestrator = create_mock_orchestrator()
        orchestrator.selfplay_scheduler = MagicMock()
        orchestrator.selfplay_scheduler.get_config_priorities.return_value = {
            "square8_4p": 100,
            "hex8_2p": 50,
        }
        loop = AutonomousQueuePopulationLoop(orchestrator)

        item = loop._create_work_item()

        assert item["config_key"] == "square8_4p"  # Highest priority


# =============================================================================
# Work Claiming Tests
# =============================================================================


class TestWorkClaiming:
    """Tests for work item claiming."""

    @pytest.mark.asyncio
    async def test_pop_local_work_returns_item(self) -> None:
        """Test popping work from local queue."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test")
        loop._local_queue = [{"work_id": "test-1"}, {"work_id": "test-2"}]

        item = await loop.pop_local_work()

        assert item is not None
        assert item["work_id"] == "test-1"
        assert loop.local_queue_depth == 1

    @pytest.mark.asyncio
    async def test_pop_local_work_returns_none_when_empty(self) -> None:
        """Test popping from empty queue returns None."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test")

        item = await loop.pop_local_work()

        assert item is None

    @pytest.mark.asyncio
    async def test_pop_local_work_returns_none_when_not_activated(self) -> None:
        """Test popping when not activated returns None."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._local_queue = [{"work_id": "test-1"}]

        item = await loop.pop_local_work()

        assert item is None

    @pytest.mark.asyncio
    async def test_claim_local_work_marks_claimed(self) -> None:
        """Test claiming work adds claimed metadata."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test")
        loop._local_queue = [{"work_id": "test-1"}]

        item = await loop.claim_local_work("worker-1")

        assert item is not None
        assert item["claimed_by"] == "worker-1"
        assert "claimed_at" in item

    @pytest.mark.asyncio
    async def test_claim_local_work_checks_capabilities(self) -> None:
        """Test claiming respects capability requirements."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test")
        loop._local_queue = [
            {"work_id": "test-1", "required_capabilities": ["cuda"]},
            {"work_id": "test-2", "required_capabilities": []},
        ]

        # Worker without cuda capability
        item = await loop.claim_local_work("worker-1", capabilities=["cpu"])

        assert item is not None
        assert item["work_id"] == "test-2"  # Got the one without cuda requirement


# =============================================================================
# Run Once Tests
# =============================================================================


class TestRunOnce:
    """Tests for _run_once method."""

    @pytest.mark.asyncio
    async def test_run_once_skips_during_grace_period(self) -> None:
        """Test that run_once skips during grace period."""
        orchestrator = create_mock_orchestrator(leader_id=None)
        config = AutonomousQueueConfig(no_leader_threshold_seconds=0)  # Would activate immediately
        loop = AutonomousQueuePopulationLoop(orchestrator, config)
        loop._startup_time = time.time()  # Just started

        await loop._run_once()

        assert loop._state.activated is False

    @pytest.mark.asyncio
    async def test_run_once_populates_when_activated(self) -> None:
        """Test that run_once populates queue when activated."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._grace_period_complete = True
        loop._state.activate("test")

        # Mock low queue depth so it stays activated and populates
        with patch.object(loop, "_get_p2p_queue_depth", return_value=5):
            # Also mock _is_selfplay_enabled_for_node to return True
            with patch.object(loop, "_is_selfplay_enabled_for_node", return_value=True):
                await loop._run_once()

        assert loop.local_queue_depth > 0

    @pytest.mark.asyncio
    async def test_run_once_checks_selfplay_enabled(self) -> None:
        """Test that run_once checks if selfplay is enabled for node."""
        orchestrator = create_mock_orchestrator()
        orchestrator._role = "coordinator"
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._grace_period_complete = True
        loop._state.activate("test")

        await loop._run_once()

        # Coordinator role should disable selfplay
        assert loop._state.activated is False


# =============================================================================
# Status and Health Check Tests
# =============================================================================


class TestStatusAndHealth:
    """Tests for status and health check methods."""

    def test_get_status_includes_all_fields(self) -> None:
        """Test that get_status includes all expected fields."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        status = loop.get_status()

        assert "enabled" in status
        assert "running" in status
        assert "activated" in status
        assert "activation_reason" in status
        assert "local_queue_depth" in status
        assert "items_populated_total" in status
        assert "config" in status

    def test_get_status_reflects_activation(self) -> None:
        """Test that status reflects activation state."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test_reason")
        loop._state.items_populated = 5

        status = loop.get_status()

        assert status["activated"] is True
        assert status["activation_reason"] == "test_reason"
        assert status["items_populated_total"] == 5

    def test_health_check_healthy_when_running(self) -> None:
        """Test health check returns healthy when running."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._running = True

        health = loop.health_check()

        assert health["healthy"] is True

    def test_health_check_healthy_when_disabled(self) -> None:
        """Test health check returns healthy when disabled."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(enabled=False)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)

        health = loop.health_check()

        assert health["healthy"] is True

    def test_health_check_includes_details(self) -> None:
        """Test health check includes status details."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        health = loop.health_check()

        assert "details" in health


# =============================================================================
# P2P Queue Depth Tests
# =============================================================================


class TestP2PQueueDepth:
    """Tests for P2P queue depth retrieval."""

    def test_gets_depth_from_work_queue(self) -> None:
        """Test getting depth from orchestrator's work queue."""
        orchestrator = create_mock_orchestrator(queue_depth=75)
        loop = AutonomousQueuePopulationLoop(orchestrator)

        depth = loop._get_p2p_queue_depth()

        assert depth == 75

    def test_returns_zero_on_error(self) -> None:
        """Test returning 0 when queue access fails."""
        orchestrator = create_mock_orchestrator()
        orchestrator._work_queue.qsize.side_effect = Exception("Queue error")
        loop = AutonomousQueuePopulationLoop(orchestrator)

        depth = loop._get_p2p_queue_depth()

        assert depth == 0

    def test_tries_state_manager_fallback(self) -> None:
        """Test falling back to state manager."""
        orchestrator = create_mock_orchestrator()
        orchestrator._work_queue = None
        orchestrator.state_manager = MagicMock()
        orchestrator.state_manager.get_work_queue_depth.return_value = 42
        loop = AutonomousQueuePopulationLoop(orchestrator)

        depth = loop._get_p2p_queue_depth()

        assert depth == 42


# =============================================================================
# Selfplay Enabled Check Tests
# =============================================================================


class TestSelfplayEnabledCheck:
    """Tests for selfplay enabled checking."""

    def test_returns_false_for_coordinator_role(self) -> None:
        """Test returns False when node has coordinator role."""
        orchestrator = create_mock_orchestrator()
        orchestrator._role = "coordinator"
        loop = AutonomousQueuePopulationLoop(orchestrator)

        result = loop._is_selfplay_enabled_for_node()

        assert result is False

    def test_returns_true_by_default(self) -> None:
        """Test returns True when config can't be determined."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        result = loop._is_selfplay_enabled_for_node()

        assert result is True

    def test_caches_result(self) -> None:
        """Test that result is cached after first check."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        # First call
        result1 = loop._is_selfplay_enabled_for_node()
        # Second call (should use cache)
        result2 = loop._is_selfplay_enabled_for_node()

        assert result1 == result2
        assert loop._selfplay_enabled_checked is True


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_activation_event_handles_import_error(self) -> None:
        """Test activation event handles import errors gracefully."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        # Should not raise even if imports fail
        loop._emit_activation_event("test_reason")

    def test_emit_deactivation_event_handles_import_error(self) -> None:
        """Test deactivation event handles import errors gracefully."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activation_time = time.time()

        # Should not raise even if imports fail
        loop._emit_deactivation_event("test_reason")


# =============================================================================
# Background Start Tests
# =============================================================================


class TestBackgroundStart:
    """Tests for start_background method."""

    def test_start_background_returns_none_when_disabled(self) -> None:
        """Test start_background returns None when disabled."""
        orchestrator = create_mock_orchestrator()
        config = AutonomousQueueConfig(enabled=False)
        loop = AutonomousQueuePopulationLoop(orchestrator, config)

        result = loop.start_background()

        assert result is None

    @pytest.mark.asyncio
    async def test_start_background_returns_task_when_enabled(self) -> None:
        """Test start_background returns task when enabled."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        task = loop.start_background()

        assert task is not None
        assert isinstance(task, asyncio.Task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# =============================================================================
# On Stop Tests
# =============================================================================


class TestOnStop:
    """Tests for _on_stop method."""

    @pytest.mark.asyncio
    async def test_on_stop_deactivates_if_active(self) -> None:
        """Test _on_stop deactivates if active."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)
        loop._state.activate("test")

        await loop._on_stop()

        assert loop._state.activated is False

    @pytest.mark.asyncio
    async def test_on_stop_does_nothing_if_not_active(self) -> None:
        """Test _on_stop does nothing if not active."""
        orchestrator = create_mock_orchestrator()
        loop = AutonomousQueuePopulationLoop(orchestrator)

        await loop._on_stop()

        assert loop._state.activated is False
        assert loop._state.deactivation_count == 0
