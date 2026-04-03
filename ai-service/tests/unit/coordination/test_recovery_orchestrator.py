"""Tests for RecoveryOrchestrator.

Tests cover:
- Recovery action escalation
- Cooldown tracking
- Circuit breaker behavior
- Rate limiting
- Node recovery state
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.recovery_orchestrator import (
    NodeRecoveryState,
    RecoveryAction,
    RecoveryResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def node_state():
    """Create a fresh NodeRecoveryState for each test."""
    return NodeRecoveryState(node_id="test-node")


# =============================================================================
# RecoveryAction Enum Tests
# =============================================================================


class TestRecoveryAction:
    """Test RecoveryAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        assert RecoveryAction.RESTART_P2P.value == "restart_p2p"
        assert RecoveryAction.RESTART_TAILSCALE.value == "restart_tailscale"
        assert RecoveryAction.SOFT_REBOOT.value == "soft_reboot"
        assert RecoveryAction.HARD_REBOOT.value == "hard_reboot"
        assert RecoveryAction.REPROVISION.value == "reprovision"
        assert RecoveryAction.ALERT_HUMAN.value == "alert_human"

    def test_action_is_string_enum(self):
        """Test actions are string-compatible."""
        assert str(RecoveryAction.RESTART_P2P).endswith(".RESTART_P2P")
        assert RecoveryAction.RESTART_P2P.value == "restart_p2p"


# =============================================================================
# NodeRecoveryState Tests
# =============================================================================


class TestNodeRecoveryState:
    """Test NodeRecoveryState dataclass."""

    def test_init_default_values(self, node_state):
        """Test default initialization."""
        assert node_state.node_id == "test-node"
        assert node_state.last_action is None
        assert node_state.attempts_this_hour == 0
        assert node_state.consecutive_failures == 0
        assert node_state.circuit_open is False

    def test_is_action_on_cooldown_not_set(self, node_state):
        """Test cooldown check when action not yet tried."""
        assert not node_state.is_action_on_cooldown(RecoveryAction.RESTART_P2P, 5)

    def test_is_action_on_cooldown_recent(self, node_state):
        """Test cooldown check when action was recent."""
        node_state.action_cooldowns[RecoveryAction.RESTART_P2P] = time.time()

        assert node_state.is_action_on_cooldown(RecoveryAction.RESTART_P2P, 5)

    def test_is_action_on_cooldown_expired(self, node_state):
        """Test cooldown check when cooldown expired."""
        node_state.action_cooldowns[RecoveryAction.RESTART_P2P] = time.time() - 600

        assert not node_state.is_action_on_cooldown(RecoveryAction.RESTART_P2P, 5)

    def test_record_attempt_success(self, node_state):
        """Test recording a successful attempt."""
        node_state.record_attempt(RecoveryAction.RESTART_P2P, success=True)

        assert node_state.last_action == RecoveryAction.RESTART_P2P
        assert node_state.attempts_this_hour == 1
        assert node_state.consecutive_failures == 0

    def test_record_attempt_failure(self, node_state):
        """Test recording a failed attempt."""
        node_state.record_attempt(RecoveryAction.RESTART_P2P, success=False)

        assert node_state.consecutive_failures == 1

    def test_record_attempt_resets_failures_on_success(self, node_state):
        """Test success resets consecutive failure count."""
        node_state.consecutive_failures = 5

        node_state.record_attempt(RecoveryAction.RESTART_P2P, success=True)

        assert node_state.consecutive_failures == 0

    def test_circuit_breaker_opens_after_failures(self, node_state):
        """Test circuit breaker opens after 3 consecutive failures."""
        for _ in range(3):
            node_state.record_attempt(RecoveryAction.RESTART_P2P, success=False)

        assert node_state.circuit_open is True

    def test_is_circuit_open_when_closed(self, node_state):
        """Test is_circuit_open when circuit is closed."""
        assert not node_state.is_circuit_open()

    def test_is_circuit_open_when_open(self, node_state):
        """Test is_circuit_open when circuit is open."""
        node_state.circuit_open = True
        node_state.circuit_open_until = time.time() + 1800

        assert node_state.is_circuit_open()

    def test_is_circuit_open_auto_closes(self, node_state):
        """Test circuit breaker auto-closes after timeout."""
        node_state.circuit_open = True
        node_state.circuit_open_until = time.time() - 1  # Expired

        assert not node_state.is_circuit_open()
        assert node_state.circuit_open is False  # Updated

    def test_hourly_counter_reset(self, node_state):
        """Test hourly counter resets after an hour."""
        node_state.attempts_this_hour = 5
        node_state.hour_start = time.time() - 3700  # Over an hour ago

        node_state.record_attempt(RecoveryAction.RESTART_P2P, success=True)

        assert node_state.attempts_this_hour == 1  # Reset to 1


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Test RecoveryResult dataclass."""

    def test_result_success(self):
        """Test successful recovery result."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=True,
            message="P2P restarted successfully",
        )

        assert result.success is True
        assert result.action == RecoveryAction.RESTART_P2P

    def test_result_failure_with_next_action(self):
        """Test failed recovery result with next action."""
        result = RecoveryResult(
            node_id="test-node",
            action=RecoveryAction.RESTART_P2P,
            success=False,
            message="Failed to restart P2P",
            next_action=RecoveryAction.RESTART_TAILSCALE,
        )

        assert result.success is False
        assert result.next_action == RecoveryAction.RESTART_TAILSCALE


# =============================================================================
# RecoveryOrchestrator Tests (with mocks)
# =============================================================================


class TestRecoveryOrchestratorLogic:
    """Test RecoveryOrchestrator business logic."""

    def test_escalation_order(self):
        """Test recovery actions are in escalation order."""
        from app.coordination.recovery_orchestrator import RecoveryOrchestrator

        expected_order = [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.SOFT_REBOOT,
            RecoveryAction.HARD_REBOOT,
            RecoveryAction.REPROVISION,
            RecoveryAction.ALERT_HUMAN,
        ]

        assert RecoveryOrchestrator.ESCALATION_ORDER == expected_order

    def test_cooldown_values(self):
        """Test cooldown values are defined."""
        from app.coordination.recovery_orchestrator import RecoveryOrchestrator

        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.RESTART_P2P] == 5
        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.RESTART_TAILSCALE] == 10
        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.SOFT_REBOOT] == 15
        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.HARD_REBOOT] == 30
        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.REPROVISION] == 60
        assert RecoveryOrchestrator.ACTION_COOLDOWNS[RecoveryAction.ALERT_HUMAN] == 120

    def test_max_attempts_per_hour(self):
        """Test max attempts per hour is set."""
        from app.coordination.recovery_orchestrator import RecoveryOrchestrator

        assert RecoveryOrchestrator.MAX_ATTEMPTS_PER_HOUR == 6


class TestRecoveryOrchestratorIntegration:
    """Integration tests for RecoveryOrchestrator (with mocked dependencies)."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        with patch("app.coordination.recovery_orchestrator.get_health_orchestrator") as mock_health, \
             patch("app.coordination.recovery_orchestrator.LambdaManager"), \
             patch("app.coordination.recovery_orchestrator.VastManager"), \
             patch("app.coordination.recovery_orchestrator.HetznerManager"), \
             patch("app.coordination.recovery_orchestrator.AWSManager"), \
             patch("app.coordination.recovery_orchestrator.TailscaleManager"):

            # Setup mock health orchestrator
            mock_health_instance = MagicMock()
            mock_health.return_value = mock_health_instance

            from app.coordination.recovery_orchestrator import RecoveryOrchestrator

            orchestrator = RecoveryOrchestrator(
                health_orchestrator=mock_health_instance,
                slack_webhook_url=None,
            )

            yield orchestrator, mock_health_instance

    def test_get_node_state_creates_new(self, mock_orchestrator):
        """Test getting node state creates new if not exists."""
        orchestrator, _ = mock_orchestrator

        state = orchestrator._get_node_state("new-node")

        assert state.node_id == "new-node"
        assert "new-node" in orchestrator.node_states

    def test_get_node_state_returns_existing(self, mock_orchestrator):
        """Test getting node state returns existing."""
        orchestrator, _ = mock_orchestrator

        state1 = orchestrator._get_node_state("test-node")
        state1.attempts_this_hour = 5

        state2 = orchestrator._get_node_state("test-node")

        assert state2.attempts_this_hour == 5

    def test_get_recovery_stats(self, mock_orchestrator):
        """Test getting recovery statistics."""
        orchestrator, _ = mock_orchestrator

        # Add some state
        state = orchestrator._get_node_state("test-node")
        state.attempts_this_hour = 3

        stats = orchestrator.get_recovery_stats()

        assert stats["total_nodes_tracked"] == 1
        assert stats["total_attempts_this_hour"] == 3


class TestRecoveryOrchestratorNextAction:
    """Test next action selection logic."""

    @pytest.fixture
    def orchestrator_for_action_tests(self):
        """Create orchestrator for action selection tests."""
        with patch("app.coordination.recovery_orchestrator.get_health_orchestrator") as mock_health, \
             patch("app.coordination.recovery_orchestrator.LambdaManager"), \
             patch("app.coordination.recovery_orchestrator.VastManager"), \
             patch("app.coordination.recovery_orchestrator.HetznerManager"), \
             patch("app.coordination.recovery_orchestrator.AWSManager"), \
             patch("app.coordination.recovery_orchestrator.TailscaleManager"):

            mock_health_instance = MagicMock()
            mock_health.return_value = mock_health_instance

            from app.coordination.recovery_orchestrator import RecoveryOrchestrator
            from app.providers import Provider

            orchestrator = RecoveryOrchestrator(
                health_orchestrator=mock_health_instance,
            )

            # Create mock health details
            mock_health_details = MagicMock()
            mock_health_details.provider = Provider.LAMBDA

            yield orchestrator, mock_health_details

    def test_next_action_first_try(self, orchestrator_for_action_tests):
        """Test first action is RESTART_P2P."""
        orchestrator, health = orchestrator_for_action_tests

        node_state = NodeRecoveryState(node_id="test-node")

        action = orchestrator._get_next_action(node_state, health)

        assert action == RecoveryAction.RESTART_P2P

    def test_next_action_after_cooldown(self, orchestrator_for_action_tests):
        """Test next action skips cooled down actions."""
        orchestrator, health = orchestrator_for_action_tests

        node_state = NodeRecoveryState(node_id="test-node")
        node_state.action_cooldowns[RecoveryAction.RESTART_P2P] = time.time()

        action = orchestrator._get_next_action(node_state, health)

        assert action == RecoveryAction.RESTART_TAILSCALE

    def test_next_action_rate_limited(self, orchestrator_for_action_tests):
        """Test rate limiting triggers ALERT_HUMAN."""
        orchestrator, health = orchestrator_for_action_tests

        node_state = NodeRecoveryState(node_id="test-node")
        node_state.attempts_this_hour = 10
        node_state.hour_start = time.time()

        action = orchestrator._get_next_action(node_state, health)

        assert action == RecoveryAction.ALERT_HUMAN

    def test_next_action_circuit_open(self, orchestrator_for_action_tests):
        """Test circuit breaker triggers ALERT_HUMAN."""
        orchestrator, health = orchestrator_for_action_tests

        node_state = NodeRecoveryState(node_id="test-node")
        node_state.circuit_open = True
        node_state.circuit_open_until = time.time() + 1800

        action = orchestrator._get_next_action(node_state, health)

        assert action == RecoveryAction.ALERT_HUMAN

    def test_next_action_skips_reprovision_for_non_vast(self, orchestrator_for_action_tests):
        """Test REPROVISION is skipped for non-Vast providers."""
        orchestrator, health = orchestrator_for_action_tests

        from app.providers import Provider

        health.provider = Provider.LAMBDA

        node_state = NodeRecoveryState(node_id="test-node")
        # Put all actions except REPROVISION and ALERT_HUMAN on cooldown
        for action in [
            RecoveryAction.RESTART_P2P,
            RecoveryAction.RESTART_TAILSCALE,
            RecoveryAction.SOFT_REBOOT,
            RecoveryAction.HARD_REBOOT,
        ]:
            node_state.action_cooldowns[action] = time.time()

        action = orchestrator._get_next_action(node_state, health)

        # Should skip REPROVISION and go to ALERT_HUMAN
        assert action == RecoveryAction.ALERT_HUMAN

    def test_next_action_all_on_cooldown(self, orchestrator_for_action_tests):
        """Test returns None when all actions on cooldown."""
        orchestrator, health = orchestrator_for_action_tests

        node_state = NodeRecoveryState(node_id="test-node")
        # Put all actions on cooldown
        for action in RecoveryAction:
            node_state.action_cooldowns[action] = time.time()

        action = orchestrator._get_next_action(node_state, health)

        assert action is None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_recovery_orchestrator_returns_singleton(self):
        """Test get_recovery_orchestrator returns same instance."""
        with patch("app.coordination.recovery_orchestrator.get_health_orchestrator") as mock_health, \
             patch("app.coordination.recovery_orchestrator.LambdaManager"), \
             patch("app.coordination.recovery_orchestrator.VastManager"), \
             patch("app.coordination.recovery_orchestrator.HetznerManager"), \
             patch("app.coordination.recovery_orchestrator.AWSManager"), \
             patch("app.coordination.recovery_orchestrator.TailscaleManager"):

            mock_health.return_value = MagicMock()

            import app.coordination.recovery_orchestrator as ro
            ro._recovery_orchestrator = None

            from app.coordination.recovery_orchestrator import get_recovery_orchestrator

            coord1 = get_recovery_orchestrator()
            coord2 = get_recovery_orchestrator()

            assert coord1 is coord2
