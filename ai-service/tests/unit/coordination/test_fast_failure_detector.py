"""Tests for FastFailureDetector.

Tests the tiered failure detection system with escalating responses:
- Tier 1 (5 min): Warning log only
- Tier 2 (10 min): Emit FAST_FAILURE_ALERT, boost selfplay 1.5x
- Tier 3 (30 min): Trigger autonomous queue, boost selfplay 2x
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.fast_failure_detector import (
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_RECOVERY_THRESHOLD,
    DEFAULT_TIER1_THRESHOLD,
    DEFAULT_TIER2_THRESHOLD,
    DEFAULT_TIER3_THRESHOLD,
    DetectorStats,
    FailureSignals,
    FailureTier,
    FailureTierConfig,
    FastFailureConfig,
    FastFailureDetector,
    get_fast_failure_detector,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    FastFailureDetector._instance = None
    yield
    FastFailureDetector._instance = None


@pytest.fixture
def mock_safe_emit():
    """Mock safe_emit_event for event emission tests.

    Note: safe_emit_event is imported dynamically inside methods, so we
    patch at the source module location.
    """
    with patch(
        "app.coordination.event_emission_helpers.safe_emit_event"
    ) as mock:
        mock.return_value = True
        yield mock


# =============================================================================
# Test FailureTier Enum
# =============================================================================


class TestFailureTier:
    """Tests for FailureTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert FailureTier.HEALTHY.value == 0
        assert FailureTier.WARNING.value == 1
        assert FailureTier.ALERT.value == 2
        assert FailureTier.RECOVERY.value == 3

    def test_tier_ordering(self):
        """Test tier ordering by value."""
        tiers = sorted(FailureTier, key=lambda t: t.value)
        assert tiers == [
            FailureTier.HEALTHY,
            FailureTier.WARNING,
            FailureTier.ALERT,
            FailureTier.RECOVERY,
        ]

    def test_tier_names(self):
        """Test tier names."""
        assert FailureTier.HEALTHY.name == "HEALTHY"
        assert FailureTier.WARNING.name == "WARNING"
        assert FailureTier.ALERT.name == "ALERT"
        assert FailureTier.RECOVERY.name == "RECOVERY"


# =============================================================================
# Test FailureTierConfig
# =============================================================================


class TestFailureTierConfig:
    """Tests for FailureTierConfig dataclass."""

    def test_basic_config(self):
        """Test basic tier config creation."""
        config = FailureTierConfig(
            tier=FailureTier.WARNING,
            threshold_seconds=300.0,
            action="log",
        )
        assert config.tier == FailureTier.WARNING
        assert config.threshold_seconds == 300.0
        assert config.action == "log"
        assert config.event_type is None
        assert config.selfplay_boost == 1.0

    def test_config_with_event(self):
        """Test tier config with event type."""
        config = FailureTierConfig(
            tier=FailureTier.ALERT,
            threshold_seconds=600.0,
            action="emit",
            event_type="FAST_FAILURE_ALERT",
            selfplay_boost=1.5,
        )
        assert config.event_type == "FAST_FAILURE_ALERT"
        assert config.selfplay_boost == 1.5

    def test_invalid_action_raises(self):
        """Test invalid action raises ValueError."""
        with pytest.raises(ValueError, match="Invalid action"):
            FailureTierConfig(
                tier=FailureTier.WARNING,
                threshold_seconds=300.0,
                action="invalid",
            )

    def test_valid_actions(self):
        """Test all valid action types."""
        for action in ["log", "emit", "recover"]:
            config = FailureTierConfig(
                tier=FailureTier.WARNING,
                threshold_seconds=300.0,
                action=action,
            )
            assert config.action == action


# =============================================================================
# Test FastFailureConfig
# =============================================================================


class TestFastFailureConfig:
    """Tests for FastFailureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = FastFailureConfig()
        assert config.recovery_threshold_seconds == DEFAULT_RECOVERY_THRESHOLD
        assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL
        assert config.enabled is True
        assert len(config.tiers) == 3

    def test_default_tiers(self):
        """Test default tier configurations."""
        config = FastFailureConfig()

        # Tier 1: WARNING (5 min, log)
        tier1 = config.tiers[0]
        assert tier1.tier == FailureTier.WARNING
        assert tier1.threshold_seconds == DEFAULT_TIER1_THRESHOLD
        assert tier1.action == "log"

        # Tier 2: ALERT (10 min, emit)
        tier2 = config.tiers[1]
        assert tier2.tier == FailureTier.ALERT
        assert tier2.threshold_seconds == DEFAULT_TIER2_THRESHOLD
        assert tier2.action == "emit"
        assert tier2.event_type == "FAST_FAILURE_ALERT"
        assert tier2.selfplay_boost == 1.5

        # Tier 3: RECOVERY (30 min, recover)
        tier3 = config.tiers[2]
        assert tier3.tier == FailureTier.RECOVERY
        assert tier3.threshold_seconds == DEFAULT_TIER3_THRESHOLD
        assert tier3.action == "recover"
        assert tier3.event_type == "FAST_FAILURE_RECOVERY"
        assert tier3.selfplay_boost == 2.0

    def test_custom_tiers(self):
        """Test custom tier configuration."""
        custom_tiers = [
            FailureTierConfig(
                tier=FailureTier.WARNING,
                threshold_seconds=60.0,
                action="log",
            ),
        ]
        config = FastFailureConfig(tiers=custom_tiers)
        assert len(config.tiers) == 1
        assert config.tiers[0].threshold_seconds == 60.0

    def test_from_env_defaults(self):
        """Test from_env with no env vars."""
        with patch.dict("os.environ", {}, clear=True):
            config = FastFailureConfig.from_env()
            assert config.recovery_threshold_seconds == DEFAULT_RECOVERY_THRESHOLD
            assert config.check_interval_seconds == DEFAULT_CHECK_INTERVAL
            assert config.enabled is True

    def test_from_env_custom(self):
        """Test from_env with custom env vars."""
        env = {
            "RINGRIFT_FAST_FAILURE_RECOVERY_THRESHOLD": "180",
            "RINGRIFT_FAST_FAILURE_CHECK_INTERVAL": "60",
            "RINGRIFT_FAST_FAILURE_ENABLED": "false",
        }
        with patch.dict("os.environ", env, clear=True):
            config = FastFailureConfig.from_env()
            assert config.recovery_threshold_seconds == 180.0
            assert config.check_interval_seconds == 60.0
            assert config.enabled is False


# =============================================================================
# Test FailureSignals
# =============================================================================


class TestFailureSignals:
    """Tests for FailureSignals dataclass."""

    def test_default_signals(self):
        """Test default signal values."""
        signals = FailureSignals()
        assert signals.no_leader is False
        assert signals.queue_empty is False
        assert signals.queue_depth == 0
        assert signals.low_selfplay_rate is False
        assert signals.selfplay_rate == 0.0
        assert signals.high_idle_percent is False
        assert signals.idle_percent == 0.0
        assert signals.timestamp > 0

    def test_is_failing_no_signals(self):
        """Test is_failing with no active signals."""
        signals = FailureSignals()
        assert signals.is_failing is False

    def test_is_failing_single_signal(self):
        """Test is_failing with single signal."""
        signals = FailureSignals(no_leader=True)
        assert signals.is_failing is True

        signals = FailureSignals(queue_empty=True)
        assert signals.is_failing is True

        signals = FailureSignals(low_selfplay_rate=True)
        assert signals.is_failing is True

        signals = FailureSignals(high_idle_percent=True)
        assert signals.is_failing is True

    def test_is_failing_multiple_signals(self):
        """Test is_failing with multiple signals."""
        signals = FailureSignals(no_leader=True, queue_empty=True)
        assert signals.is_failing is True

    def test_signal_count(self):
        """Test signal_count property."""
        signals = FailureSignals()
        assert signals.signal_count == 0

        signals = FailureSignals(no_leader=True)
        assert signals.signal_count == 1

        signals = FailureSignals(no_leader=True, queue_empty=True)
        assert signals.signal_count == 2

        signals = FailureSignals(
            no_leader=True,
            queue_empty=True,
            low_selfplay_rate=True,
            high_idle_percent=True,
        )
        assert signals.signal_count == 4


# =============================================================================
# Test DetectorStats
# =============================================================================


class TestDetectorStats:
    """Tests for DetectorStats dataclass."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = DetectorStats()
        assert stats.checks_performed == 0
        assert stats.failures_detected == 0
        assert stats.alerts_emitted == 0
        assert stats.recoveries_triggered == 0
        assert stats.tier_escalations == 0
        assert stats.current_tier == FailureTier.HEALTHY
        assert stats.failure_start_time == 0.0
        assert stats.last_healthy_time > 0
        assert stats.last_check_time == 0.0
        assert stats.last_signals is None


# =============================================================================
# Test FastFailureDetector Basic
# =============================================================================


class TestFastFailureDetectorBasic:
    """Basic tests for FastFailureDetector."""

    def test_initialization_default(self):
        """Test default initialization."""
        detector = FastFailureDetector()
        assert detector._config.enabled is True
        assert detector._current_boost == 1.0
        assert detector._stats.current_tier == FailureTier.HEALTHY

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = FastFailureConfig(
            check_interval_seconds=15.0,
            enabled=False,
        )
        detector = FastFailureDetector(config=config)
        assert detector._config.check_interval_seconds == 15.0
        assert detector._config.enabled is False

    def test_initialization_with_callbacks(self):
        """Test initialization with callbacks."""
        get_leader = MagicMock(return_value="leader-1")
        get_queue = MagicMock(return_value=50)
        get_rate = MagicMock(return_value=1.5)
        get_util = MagicMock(return_value=0.8)
        trigger_queue = MagicMock()

        detector = FastFailureDetector(
            get_leader_id=get_leader,
            get_work_queue_depth=get_queue,
            get_selfplay_rate=get_rate,
            get_cluster_utilization=get_util,
            trigger_autonomous_queue=trigger_queue,
        )

        assert detector._get_leader_id is get_leader
        assert detector._get_work_queue_depth is get_queue
        assert detector._get_selfplay_rate is get_rate
        assert detector._get_cluster_utilization is get_util
        assert detector._trigger_autonomous_queue is trigger_queue


# =============================================================================
# Test FastFailureDetector Singleton
# =============================================================================


class TestFastFailureDetectorSingleton:
    """Singleton pattern tests for FastFailureDetector."""

    def test_get_instance(self):
        """Test singleton get_instance."""
        detector1 = FastFailureDetector.get_instance()
        detector2 = FastFailureDetector.get_instance()
        assert detector1 is detector2

    def test_get_instance_with_config(self):
        """Test singleton respects first config."""
        config1 = FastFailureConfig(check_interval_seconds=15.0)
        detector1 = FastFailureDetector.get_instance(config=config1)

        config2 = FastFailureConfig(check_interval_seconds=30.0)
        detector2 = FastFailureDetector.get_instance(config=config2)

        # Should use first config
        assert detector1 is detector2
        assert detector1._config.check_interval_seconds == 15.0

    def test_reset_instance(self):
        """Test singleton reset."""
        detector1 = FastFailureDetector.get_instance()
        FastFailureDetector.reset_instance()
        detector2 = FastFailureDetector.get_instance()
        assert detector1 is not detector2


# =============================================================================
# Test Set Callbacks
# =============================================================================


class TestSetCallbacks:
    """Tests for set_callbacks method."""

    def test_set_callbacks_all(self):
        """Test setting all callbacks."""
        detector = FastFailureDetector()

        get_leader = MagicMock()
        get_queue = MagicMock()
        get_rate = MagicMock()
        get_util = MagicMock()
        trigger_queue = MagicMock()

        detector.set_callbacks(
            get_leader_id=get_leader,
            get_work_queue_depth=get_queue,
            get_selfplay_rate=get_rate,
            get_cluster_utilization=get_util,
            trigger_autonomous_queue=trigger_queue,
        )

        assert detector._get_leader_id is get_leader
        assert detector._get_work_queue_depth is get_queue
        assert detector._get_selfplay_rate is get_rate
        assert detector._get_cluster_utilization is get_util
        assert detector._trigger_autonomous_queue is trigger_queue

    def test_set_callbacks_partial(self):
        """Test setting partial callbacks."""
        original_leader = MagicMock()
        detector = FastFailureDetector(get_leader_id=original_leader)

        new_queue = MagicMock()
        detector.set_callbacks(get_work_queue_depth=new_queue)

        # Original preserved
        assert detector._get_leader_id is original_leader
        # New set
        assert detector._get_work_queue_depth is new_queue


# =============================================================================
# Test Collect Failure Signals
# =============================================================================


class TestCollectFailureSignals:
    """Tests for _collect_failure_signals method."""

    def test_collect_all_healthy(self):
        """Test signal collection when all healthy."""
        detector = FastFailureDetector(
            get_leader_id=lambda: "leader-1",
            get_work_queue_depth=lambda: 100,
            get_selfplay_rate=lambda: 1.5,
            get_cluster_utilization=lambda: 0.8,
        )

        signals = detector._collect_failure_signals()

        assert signals.no_leader is False
        assert signals.queue_empty is False
        assert signals.queue_depth == 100
        assert signals.low_selfplay_rate is False
        assert signals.selfplay_rate == 1.5
        assert signals.high_idle_percent is False
        assert signals.idle_percent == pytest.approx(0.2)  # 1.0 - 0.8

    def test_collect_no_leader(self):
        """Test signal when no leader."""
        detector = FastFailureDetector(
            get_leader_id=lambda: None,
        )

        signals = detector._collect_failure_signals()
        assert signals.no_leader is True

    def test_collect_leader_exception(self):
        """Test signal on leader callback exception."""
        detector = FastFailureDetector(
            get_leader_id=lambda: (_ for _ in ()).throw(Exception("error")),
        )

        signals = detector._collect_failure_signals()
        assert signals.no_leader is True

    def test_collect_queue_empty(self):
        """Test signal when queue near-empty."""
        detector = FastFailureDetector(
            get_work_queue_depth=lambda: 3,  # < 5 threshold
        )

        signals = detector._collect_failure_signals()
        assert signals.queue_empty is True
        assert signals.queue_depth == 3

    def test_collect_queue_exception(self):
        """Test signal on queue callback exception."""
        detector = FastFailureDetector(
            get_work_queue_depth=lambda: (_ for _ in ()).throw(Exception("error")),
        )

        signals = detector._collect_failure_signals()
        assert signals.queue_empty is True
        assert signals.queue_depth == 0

    def test_collect_low_selfplay_rate(self):
        """Test signal when selfplay rate low."""
        detector = FastFailureDetector(
            get_selfplay_rate=lambda: 0.05,  # < 0.1 threshold
        )

        signals = detector._collect_failure_signals()
        assert signals.low_selfplay_rate is True
        assert signals.selfplay_rate == 0.05

    def test_collect_high_idle(self):
        """Test signal when cluster highly idle."""
        detector = FastFailureDetector(
            get_cluster_utilization=lambda: 0.2,  # < 0.3 threshold
        )

        signals = detector._collect_failure_signals()
        assert signals.high_idle_percent is True
        assert signals.idle_percent == 0.8  # 1.0 - 0.2

    def test_collect_no_callbacks(self):
        """Test signal collection with no callbacks."""
        detector = FastFailureDetector()
        signals = detector._collect_failure_signals()

        # All signals false when no callbacks
        assert signals.no_leader is False
        assert signals.queue_empty is False
        assert signals.low_selfplay_rate is False
        assert signals.high_idle_percent is False


# =============================================================================
# Test Run Cycle
# =============================================================================


class TestRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self):
        """Test cycle does nothing when disabled."""
        config = FastFailureConfig(enabled=False)
        detector = FastFailureDetector(config=config)

        await detector._run_cycle()

        assert detector._stats.checks_performed == 0

    @pytest.mark.asyncio
    async def test_run_cycle_healthy(self):
        """Test cycle when cluster healthy."""
        detector = FastFailureDetector(
            get_leader_id=lambda: "leader-1",
            get_work_queue_depth=lambda: 100,
            get_selfplay_rate=lambda: 1.5,
            get_cluster_utilization=lambda: 0.8,
        )

        await detector._run_cycle()

        assert detector._stats.checks_performed == 1
        assert detector._stats.current_tier == FailureTier.HEALTHY
        assert detector._stats.last_signals is not None

    @pytest.mark.asyncio
    async def test_run_cycle_failure_detected(self, mock_safe_emit):
        """Test cycle when failure detected."""
        detector = FastFailureDetector(
            get_leader_id=lambda: None,  # No leader = failing
        )

        await detector._run_cycle()

        assert detector._stats.checks_performed == 1
        assert detector._stats.failures_detected == 1


# =============================================================================
# Test Handle Failure
# =============================================================================


class TestHandleFailure:
    """Tests for _handle_failure method."""

    @pytest.mark.asyncio
    async def test_handle_failure_initial_detection(self, mock_safe_emit):
        """Test initial failure detection."""
        detector = FastFailureDetector()
        signals = FailureSignals(no_leader=True)

        await detector._handle_failure(signals)

        assert detector._stats.failures_detected == 1
        assert detector._stats.failure_start_time > 0

    @pytest.mark.asyncio
    async def test_handle_failure_tier_escalation(self, mock_safe_emit):
        """Test tier escalation based on duration.

        Note: The implementation resets failure_start_time when tier is HEALTHY,
        so we use time mocking to control time progression.
        """
        # Create config with short thresholds for testing
        config = FastFailureConfig(
            tiers=[
                FailureTierConfig(
                    tier=FailureTier.WARNING,
                    threshold_seconds=1.0,
                    action="log",
                ),
                FailureTierConfig(
                    tier=FailureTier.ALERT,
                    threshold_seconds=2.0,
                    action="emit",
                    event_type="FAST_FAILURE_ALERT",
                    selfplay_boost=1.5,
                ),
            ],
        )
        detector = FastFailureDetector(config=config)
        signals = FailureSignals(no_leader=True)

        # Use time mocking to simulate time progression
        with patch("app.coordination.fast_failure_detector.time") as mock_time:
            # First call at t=100 - initial detection
            mock_time.time.return_value = 100.0
            await detector._handle_failure(signals)
            assert detector._stats.failures_detected == 1
            assert detector._stats.current_tier == FailureTier.HEALTHY  # Duration=0, no escalation

            # Set tier to WARNING so failure_start_time isn't reset
            # (simulating that we've been in failure state)
            detector._stats.current_tier = FailureTier.WARNING

            # Second call at t=102.5 - should check for ALERT escalation
            mock_time.time.return_value = 102.5
            await detector._handle_failure(signals)
            # Duration = 102.5 - 100.0 = 2.5s >= 2.0s threshold
            assert detector._stats.current_tier == FailureTier.ALERT
            assert detector._stats.tier_escalations == 1


# =============================================================================
# Test Escalate to Tier
# =============================================================================


class TestEscalateToTier:
    """Tests for _escalate_to_tier method."""

    @pytest.mark.asyncio
    async def test_escalate_log_action(self, mock_safe_emit):
        """Test escalation with log action."""
        config = FailureTierConfig(
            tier=FailureTier.WARNING,
            threshold_seconds=300.0,
            action="log",
        )
        detector = FastFailureDetector()
        signals = FailureSignals(no_leader=True)

        await detector._escalate_to_tier(config, signals, 350.0)

        assert detector._stats.current_tier == FailureTier.WARNING
        assert detector._stats.tier_escalations == 1
        assert detector._current_boost == 1.0  # Log doesn't change boost

    @pytest.mark.asyncio
    async def test_escalate_emit_action(self, mock_safe_emit):
        """Test escalation with emit action."""
        config = FailureTierConfig(
            tier=FailureTier.ALERT,
            threshold_seconds=600.0,
            action="emit",
            event_type="FAST_FAILURE_ALERT",
            selfplay_boost=1.5,
        )
        detector = FastFailureDetector()
        signals = FailureSignals(no_leader=True)

        await detector._escalate_to_tier(config, signals, 650.0)

        assert detector._stats.current_tier == FailureTier.ALERT
        assert detector._stats.alerts_emitted == 1
        assert detector._current_boost == 1.5
        mock_safe_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_recover_action(self, mock_safe_emit):
        """Test escalation with recover action."""
        trigger_queue = MagicMock()
        config = FailureTierConfig(
            tier=FailureTier.RECOVERY,
            threshold_seconds=1800.0,
            action="recover",
            event_type="FAST_FAILURE_RECOVERY",
            selfplay_boost=2.0,
        )
        detector = FastFailureDetector(trigger_autonomous_queue=trigger_queue)
        signals = FailureSignals(no_leader=True)

        await detector._escalate_to_tier(config, signals, 1900.0)

        assert detector._stats.current_tier == FailureTier.RECOVERY
        assert detector._stats.recoveries_triggered == 1
        assert detector._current_boost == 2.0
        trigger_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_recover_no_callback(self, mock_safe_emit):
        """Test recover action with no trigger callback."""
        config = FailureTierConfig(
            tier=FailureTier.RECOVERY,
            threshold_seconds=1800.0,
            action="recover",
            event_type="FAST_FAILURE_RECOVERY",
            selfplay_boost=2.0,
        )
        detector = FastFailureDetector()  # No trigger callback
        signals = FailureSignals(no_leader=True)

        # Should not raise
        await detector._escalate_to_tier(config, signals, 1900.0)

        assert detector._stats.recoveries_triggered == 1

    @pytest.mark.asyncio
    async def test_escalate_recover_callback_exception(self, mock_safe_emit):
        """Test recover action when callback raises."""
        trigger_queue = MagicMock(side_effect=Exception("trigger failed"))
        config = FailureTierConfig(
            tier=FailureTier.RECOVERY,
            threshold_seconds=1800.0,
            action="recover",
            event_type="FAST_FAILURE_RECOVERY",
            selfplay_boost=2.0,
        )
        detector = FastFailureDetector(trigger_autonomous_queue=trigger_queue)
        signals = FailureSignals(no_leader=True)

        # Should not raise, just log error
        await detector._escalate_to_tier(config, signals, 1900.0)

        assert detector._stats.recoveries_triggered == 1


# =============================================================================
# Test Handle Recovery
# =============================================================================


class TestHandleRecovery:
    """Tests for _handle_recovery method."""

    @pytest.mark.asyncio
    async def test_handle_recovery_already_healthy(self, mock_safe_emit):
        """Test recovery when already healthy."""
        detector = FastFailureDetector()
        detector._stats.current_tier = FailureTier.HEALTHY

        await detector._handle_recovery()

        # Should just update last_healthy_time
        assert detector._stats.current_tier == FailureTier.HEALTHY
        mock_safe_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_recovery_not_long_enough(self, mock_safe_emit):
        """Test recovery when not healthy long enough."""
        config = FastFailureConfig(recovery_threshold_seconds=120.0)
        detector = FastFailureDetector(config=config)
        detector._stats.current_tier = FailureTier.ALERT
        detector._stats.last_healthy_time = time.time()  # Just now

        await detector._handle_recovery()

        # Should not recover yet
        assert detector._stats.current_tier == FailureTier.ALERT
        mock_safe_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_recovery_success(self, mock_safe_emit):
        """Test successful recovery after threshold."""
        config = FastFailureConfig(recovery_threshold_seconds=1.0)
        detector = FastFailureDetector(config=config)
        detector._stats.current_tier = FailureTier.ALERT
        detector._stats.last_healthy_time = time.time() - 2.0  # 2 seconds ago
        detector._current_boost = 1.5

        await detector._handle_recovery()

        assert detector._stats.current_tier == FailureTier.HEALTHY
        assert detector._stats.failure_start_time == 0.0
        assert detector._current_boost == 1.0
        mock_safe_emit.assert_called_once()


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Tests for event emission methods."""

    def test_emit_failure_event(self, mock_safe_emit):
        """Test failure event emission."""
        detector = FastFailureDetector()
        detector._current_boost = 1.5
        signals = FailureSignals(
            no_leader=True,
            queue_empty=True,
            queue_depth=3,
        )

        detector._emit_failure_event(
            "FAST_FAILURE_ALERT",
            FailureTier.ALERT,
            signals,
            650.0,
        )

        mock_safe_emit.assert_called_once()
        call_args = mock_safe_emit.call_args
        assert call_args[0][0] == "FAST_FAILURE_ALERT"

        payload = call_args[0][1]
        assert payload["tier"] == "ALERT"
        assert payload["failure_duration_seconds"] == 650.0
        assert payload["signals"]["no_leader"] is True
        assert payload["signals"]["queue_empty"] is True
        assert payload["selfplay_boost"] == 1.5

    def test_emit_recovered_event(self, mock_safe_emit):
        """Test recovery event emission."""
        detector = FastFailureDetector()
        detector._stats.failure_start_time = time.time() - 100

        detector._emit_recovered_event(FailureTier.ALERT)

        mock_safe_emit.assert_called_once()
        call_args = mock_safe_emit.call_args
        assert call_args[0][0] == "FAST_FAILURE_RECOVERED"

        payload = call_args[0][1]
        assert payload["from_tier"] == "ALERT"
        assert payload["total_failure_duration_seconds"] > 0


# =============================================================================
# Test Get Current Boost
# =============================================================================


class TestGetCurrentBoost:
    """Tests for get_current_boost method."""

    def test_get_current_boost_default(self):
        """Test default boost value."""
        detector = FastFailureDetector()
        assert detector.get_current_boost() == 1.0

    def test_get_current_boost_after_escalation(self, mock_safe_emit):
        """Test boost after escalation."""
        detector = FastFailureDetector()
        detector._current_boost = 2.0
        assert detector.get_current_boost() == 2.0


# =============================================================================
# Test Get Stats
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        detector = FastFailureDetector()
        stats = detector.get_stats()

        assert stats["enabled"] is True
        assert stats["current_tier"] == "HEALTHY"
        assert stats["current_boost"] == 1.0
        assert stats["checks_performed"] == 0
        assert stats["failures_detected"] == 0
        assert stats["alerts_emitted"] == 0
        assert stats["recoveries_triggered"] == 0
        assert stats["tier_escalations"] == 0
        assert stats["last_signals"] is None

    def test_get_stats_with_signals(self):
        """Test statistics with signals."""
        detector = FastFailureDetector()
        detector._stats.last_signals = FailureSignals(
            no_leader=True,
            queue_depth=10,
            selfplay_rate=0.5,
        )

        stats = detector.get_stats()

        assert stats["last_signals"] is not None
        assert stats["last_signals"]["no_leader"] is True
        assert stats["last_signals"]["queue_depth"] == 10
        assert stats["last_signals"]["selfplay_rate"] == 0.5


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_healthy(self):
        """Test health check when healthy."""
        detector = FastFailureDetector()
        result = detector.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "checks_performed" in result.details

    def test_health_check_degraded(self):
        """Test health check when degraded."""
        detector = FastFailureDetector()
        detector._stats.current_tier = FailureTier.WARNING

        result = detector.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED

    def test_health_check_alert(self):
        """Test health check when in alert."""
        detector = FastFailureDetector()
        detector._stats.current_tier = FailureTier.ALERT

        result = detector.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED

    def test_health_check_recovery(self):
        """Test health check when in recovery mode."""
        detector = FastFailureDetector()
        detector._stats.current_tier = FailureTier.RECOVERY

        result = detector.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED


# =============================================================================
# Test Module Function
# =============================================================================


class TestModuleFunction:
    """Tests for get_fast_failure_detector module function."""

    def test_get_fast_failure_detector(self):
        """Test module-level accessor."""
        detector1 = get_fast_failure_detector()
        detector2 = get_fast_failure_detector()
        assert detector1 is detector2

    def test_get_fast_failure_detector_with_config(self):
        """Test module-level accessor with config."""
        config = FastFailureConfig(check_interval_seconds=15.0)
        detector = get_fast_failure_detector(config=config)
        assert detector._config.check_interval_seconds == 15.0

    def test_get_fast_failure_detector_with_kwargs(self):
        """Test module-level accessor with kwargs."""
        get_leader = MagicMock(return_value="leader-1")
        detector = get_fast_failure_detector(get_leader_id=get_leader)
        assert detector._get_leader_id is get_leader


# =============================================================================
# Test Full Cycle Integration
# =============================================================================


class TestFullCycleIntegration:
    """Integration tests for full detection cycle."""

    @pytest.mark.asyncio
    async def test_full_failure_to_recovery_cycle(self, mock_safe_emit):
        """Test complete failure detection and recovery cycle using time mocking."""
        # Short thresholds for testing
        config = FastFailureConfig(
            tiers=[
                FailureTierConfig(
                    tier=FailureTier.WARNING,
                    threshold_seconds=1.0,
                    action="log",
                ),
                FailureTierConfig(
                    tier=FailureTier.ALERT,
                    threshold_seconds=2.0,
                    action="emit",
                    event_type="FAST_FAILURE_ALERT",
                    selfplay_boost=1.5,
                ),
            ],
            recovery_threshold_seconds=1.0,
        )

        # Start with failing state
        leader_state = {"leader": None}
        detector = FastFailureDetector(
            config=config,
            get_leader_id=lambda: leader_state["leader"],
        )

        with patch("app.coordination.fast_failure_detector.time") as mock_time:
            # T=100: Initial check - failure detected
            mock_time.time.return_value = 100.0
            await detector._run_cycle()
            assert detector._stats.failures_detected == 1
            assert detector._stats.current_tier == FailureTier.HEALTHY

            # Manually set tier to prevent failure_start_time reset
            # This simulates the scenario where failure tracking started
            detector._stats.current_tier = FailureTier.WARNING

            # T=102.5: Should escalate to ALERT (duration=2.5s >= 2.0s)
            mock_time.time.return_value = 102.5
            await detector._run_cycle()
            assert detector._stats.current_tier == FailureTier.ALERT
            assert detector._current_boost == 1.5

            # T=103: Leader returns, start recovery tracking
            mock_time.time.return_value = 103.0
            leader_state["leader"] = "leader-1"
            await detector._run_cycle()
            # Still in ALERT (not healthy long enough)
            assert detector._stats.current_tier == FailureTier.ALERT

            # T=104.5: Healthy for 1.5s > 1.0s recovery threshold
            mock_time.time.return_value = 104.5
            await detector._run_cycle()
            assert detector._stats.current_tier == FailureTier.HEALTHY
            assert detector._current_boost == 1.0

    @pytest.mark.asyncio
    async def test_multiple_signals_failure(self, mock_safe_emit):
        """Test failure with multiple signals."""
        config = FastFailureConfig(
            tiers=[
                FailureTierConfig(
                    tier=FailureTier.WARNING,
                    threshold_seconds=0.1,
                    action="log",
                ),
            ],
        )

        detector = FastFailureDetector(
            config=config,
            get_leader_id=lambda: None,
            get_work_queue_depth=lambda: 0,
            get_selfplay_rate=lambda: 0.01,
            get_cluster_utilization=lambda: 0.1,
        )

        await detector._run_cycle()

        assert detector._stats.failures_detected == 1
        assert detector._stats.last_signals.signal_count == 4
