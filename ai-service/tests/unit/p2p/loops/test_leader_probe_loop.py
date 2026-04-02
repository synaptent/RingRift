"""Tests for LeaderProbeLoop.

January 2026: Comprehensive tests for leader health probing, election triggering,
split-brain detection, and backup candidate management.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loops.leader_probe_loop import LeaderProbeLoop


# =============================================================================
# Mock Orchestrator Factory
# =============================================================================


def create_mock_orchestrator(
    *,
    is_leader: bool = False,
    leader_id: str | None = "leader-node-1",
    node_id: str = "test-node-1",
    voter_node_ids: list[str] | None = None,
) -> MagicMock:
    """Create a mock orchestrator with configurable state."""
    mock = MagicMock()
    mock.is_leader = is_leader
    mock.leader_id = leader_id
    mock.node_id = node_id
    mock.voter_node_ids = voter_node_ids or ["voter-1", "voter-2", "voter-3"]

    # Mock _urls_for_peer to return health URLs
    def urls_for_peer(peer_id: str, endpoint: str) -> list[str]:
        return [f"http://{peer_id}:8770/{endpoint}"]

    mock._urls_for_peer = urls_for_peer

    # Mock election methods
    mock._start_election = AsyncMock()
    mock._clear_leader = MagicMock()
    mock._check_quorum_health = MagicMock(return_value=None)
    mock._gossip_mixin = None

    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLeaderProbeLoopInit:
    """Tests for LeaderProbeLoop initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default parameters."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        assert loop.name == "leader_probe"
        assert loop.interval == 10.0
        assert loop._failure_threshold == 6
        assert loop._probe_timeout == 5.0
        assert loop._consecutive_failures == 0
        assert loop._election_triggered_recently is False
        assert loop._probe_backup_candidates is True

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(
            orchestrator,
            probe_interval=5.0,
            failure_threshold=3,
            probe_timeout=2.0,
            probe_backup_candidates=False,
        )

        assert loop.interval == 5.0
        assert loop._failure_threshold == 3
        assert loop._probe_timeout == 2.0
        assert loop._probe_backup_candidates is False

    def test_initialization_sets_up_latency_tracking(self) -> None:
        """Test that latency history is initialized."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        assert hasattr(loop, "_latency_history")
        assert len(loop._latency_history) == 0
        assert loop._latency_warning_emitted is False

    def test_initialization_sets_up_split_brain_tracking(self) -> None:
        """Test that split-brain detection is initialized."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        assert loop._split_brain_detected is False
        assert loop._split_brain_check_interval == 3


# =============================================================================
# Leader Detection Tests
# =============================================================================


class TestLeaderDetection:
    """Tests for _is_leader and _get_leader_id methods."""

    def test_is_leader_when_leader(self) -> None:
        """Test _is_leader returns True when node is leader."""
        orchestrator = create_mock_orchestrator(is_leader=True)
        loop = LeaderProbeLoop(orchestrator)

        assert loop._is_leader() is True

    def test_is_leader_when_not_leader(self) -> None:
        """Test _is_leader returns False when node is not leader."""
        orchestrator = create_mock_orchestrator(is_leader=False)
        loop = LeaderProbeLoop(orchestrator)

        assert loop._is_leader() is False

    def test_is_leader_handles_missing_attribute(self) -> None:
        """Test _is_leader handles orchestrator without is_leader."""
        orchestrator = MagicMock(spec=[])  # No attributes
        loop = LeaderProbeLoop(orchestrator)

        assert loop._is_leader() is False

    def test_get_leader_id_returns_leader(self) -> None:
        """Test _get_leader_id returns the leader ID."""
        orchestrator = create_mock_orchestrator(leader_id="leader-42")
        loop = LeaderProbeLoop(orchestrator)

        assert loop._get_leader_id() == "leader-42"

    def test_get_leader_id_returns_none_when_no_leader(self) -> None:
        """Test _get_leader_id returns None when no leader."""
        orchestrator = create_mock_orchestrator(leader_id=None)
        loop = LeaderProbeLoop(orchestrator)

        assert loop._get_leader_id() is None

    def test_get_leader_id_handles_missing_attribute(self) -> None:
        """Test _get_leader_id handles orchestrator without leader_id."""
        orchestrator = MagicMock(spec=[])
        loop = LeaderProbeLoop(orchestrator)

        assert loop._get_leader_id() is None


# =============================================================================
# Probe Execution Tests
# =============================================================================


class TestProbeExecution:
    """Tests for _run_once and probe execution."""

    @pytest.mark.asyncio
    async def test_run_once_skips_when_leader(self) -> None:
        """Test that probing is skipped when this node is the leader."""
        orchestrator = create_mock_orchestrator(is_leader=True)
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 5  # Set high to ensure it gets reset

        await loop._run_once()

        # Failures should be reset when we're the leader
        assert loop._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_when_no_leader(self) -> None:
        """Test that probing is skipped when no leader is known."""
        orchestrator = create_mock_orchestrator(leader_id=None)
        loop = LeaderProbeLoop(orchestrator)

        # This should not raise and should return early
        await loop._run_once()

    @pytest.mark.asyncio
    async def test_run_once_increments_cycle_counter(self) -> None:
        """Test that cycle counter increments each run."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        initial_counter = loop._probe_cycle_counter

        with patch.object(loop, "_probe_leader", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = True
            await loop._run_once()

        assert loop._probe_cycle_counter == initial_counter + 1


# =============================================================================
# Probe Success/Failure Tests
# =============================================================================


class TestProbeResults:
    """Tests for probe success and failure handling."""

    def test_on_probe_success_resets_failures(self) -> None:
        """Test that successful probe resets consecutive failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 3
        loop._election_triggered_recently = True

        loop._on_probe_success()

        assert loop._consecutive_failures == 0
        assert loop._election_triggered_recently is False

    def test_on_probe_success_emits_recovery_event(self) -> None:
        """Test that recovery event is emitted after failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 5
        loop._last_success_time = time.time() - 60

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._on_probe_success()

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "LEADER_PROBE_RECOVERED"
        assert call_args[0][1]["failures_before_recovery"] == 5

    def test_on_probe_success_no_event_when_no_prior_failures(self) -> None:
        """Test that no event is emitted when there were no prior failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 0

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._on_probe_success()

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_probe_failure_increments_failures(self) -> None:
        """Test that probe failure increments consecutive failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 2

        with patch.object(loop, "_emit_event"):
            await loop._on_probe_failure("leader-1")

        assert loop._consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_on_probe_failure_emits_event(self) -> None:
        """Test that probe failure emits event."""
        orchestrator = create_mock_orchestrator()
        # Disable startup grace period so event is emitted immediately
        loop = LeaderProbeLoop(orchestrator, startup_grace_period=0)

        with patch.object(loop, "_emit_event") as mock_emit:
            await loop._on_probe_failure("leader-1")

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "LEADER_PROBE_FAILED"
        assert call_args[0][1]["leader_id"] == "leader-1"

    @pytest.mark.asyncio
    async def test_on_probe_failure_triggers_election_at_threshold(self) -> None:
        """Test that election is triggered when threshold reached."""
        orchestrator = create_mock_orchestrator()
        # Disable startup grace period so election can be triggered
        loop = LeaderProbeLoop(orchestrator, failure_threshold=3, startup_grace_period=0)
        loop._consecutive_failures = 2  # Will become 3

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock) as mock_election:
                await loop._on_probe_failure("leader-1")

        mock_election.assert_called_once_with("leader-1")

    @pytest.mark.asyncio
    async def test_on_probe_failure_no_election_below_threshold(self) -> None:
        """Test that election is not triggered below threshold."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        loop._consecutive_failures = 2  # Will become 3 (below 6)

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock) as mock_election:
                await loop._on_probe_failure("leader-1")

        mock_election.assert_not_called()


# =============================================================================
# Election Triggering Tests
# =============================================================================


class TestElectionTriggering:
    """Tests for election triggering logic."""

    @pytest.mark.asyncio
    async def test_trigger_election_sets_cooldown(self) -> None:
        """Test that triggering election sets cooldown flag."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._election_triggered_recently = False

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_verify_elected_leader_after_delay", new_callable=AsyncMock):
                await loop._trigger_election("leader-1")

        assert loop._election_triggered_recently is True

    @pytest.mark.asyncio
    async def test_trigger_election_skips_during_cooldown(self) -> None:
        """Test that election is skipped during cooldown."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._election_triggered_recently = True

        await loop._trigger_election("leader-1")

        # Election method should not be called
        orchestrator._start_election.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_election_resets_failure_counter(self) -> None:
        """Test that triggering election resets failure counter."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._consecutive_failures = 10

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_verify_elected_leader_after_delay", new_callable=AsyncMock):
                await loop._trigger_election("leader-1")

        assert loop._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_trigger_election_calls_start_election(self) -> None:
        """Test that _start_election is called on orchestrator."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_verify_elected_leader_after_delay", new_callable=AsyncMock):
                await loop._trigger_election("leader-1")

        orchestrator._start_election.assert_called_once_with(reason="leader_unreachable_probe")


# =============================================================================
# Latency Trending Tests
# =============================================================================


class TestLatencyTrending:
    """Tests for latency trend detection."""

    def test_check_latency_trend_needs_minimum_samples(self) -> None:
        """Test that trend check requires minimum samples."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # Add only 3 samples (need 6)
        loop._latency_history.extend([0.1, 0.1, 0.1])

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._check_latency_trend()

        # Should not emit any event
        mock_emit.assert_not_called()

    def test_check_latency_trend_detects_increase(self) -> None:
        """Test that latency increase is detected."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # Add samples: first half low, second half high (more than 2x)
        loop._latency_history.extend([0.05, 0.05, 0.05, 0.2, 0.25, 0.3])

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._check_latency_trend()

        assert loop._latency_warning_emitted is True
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "LEADER_LATENCY_WARNING"

    def test_check_latency_trend_detects_recovery(self) -> None:
        """Test that latency recovery is detected."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._latency_warning_emitted = True

        # Add samples: all low
        loop._latency_history.extend([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._check_latency_trend()

        assert loop._latency_warning_emitted is False
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "LEADER_LATENCY_RECOVERED"

    def test_check_latency_trend_ignores_small_baseline(self) -> None:
        """Test that very low baseline latency is ignored."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # Add samples with <10ms baseline (should ignore)
        loop._latency_history.extend([0.005, 0.005, 0.005, 0.01, 0.01, 0.01])

        with patch.object(loop, "_emit_event") as mock_emit:
            loop._check_latency_trend()

        # No warning should be emitted for such low latencies
        assert loop._latency_warning_emitted is False


# =============================================================================
# Backup Candidate Tests
# =============================================================================


class TestBackupCandidates:
    """Tests for backup candidate probing."""

    def test_get_backup_candidates_excludes_leader(self) -> None:
        """Test that backup candidates exclude current leader."""
        orchestrator = create_mock_orchestrator(
            voter_node_ids=["leader-1", "voter-2", "voter-3", "voter-4"],
            node_id="test-node",
        )
        loop = LeaderProbeLoop(orchestrator)

        candidates = loop._get_backup_candidates("leader-1")

        assert "leader-1" not in candidates

    def test_get_backup_candidates_excludes_self(self) -> None:
        """Test that backup candidates exclude this node."""
        orchestrator = create_mock_orchestrator(
            voter_node_ids=["voter-1", "test-node", "voter-3"],
            node_id="test-node",
        )
        loop = LeaderProbeLoop(orchestrator)

        candidates = loop._get_backup_candidates("voter-1")

        assert "test-node" not in candidates

    def test_get_backup_candidates_limits_count(self) -> None:
        """Test that backup candidates are limited."""
        orchestrator = create_mock_orchestrator(
            voter_node_ids=["v1", "v2", "v3", "v4", "v5", "v6"],
            node_id="test-node",
        )
        loop = LeaderProbeLoop(orchestrator)

        candidates = loop._get_backup_candidates("v1", limit=2)

        assert len(candidates) == 2

    def test_get_warm_backup_candidates_returns_reachable(self) -> None:
        """Test that only reachable candidates are returned."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._backup_candidate_status = {
            "node-1": True,
            "node-2": False,
            "node-3": True,
        }

        warm = loop.get_warm_backup_candidates()

        assert "node-1" in warm
        assert "node-2" not in warm
        assert "node-3" in warm


# =============================================================================
# Split-Brain Detection Tests
# =============================================================================


class TestSplitBrainDetection:
    """Tests for split-brain detection."""

    @pytest.mark.asyncio
    async def test_check_split_brain_no_issue_when_leader_claims(self) -> None:
        """Test no split-brain when leader claims leadership."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_get_node_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {"is_leader": True, "leader_id": "leader-1"}
            await loop._check_for_split_brain("leader-1")

        assert loop._split_brain_detected is False

    @pytest.mark.asyncio
    async def test_check_split_brain_detects_discrepancy(self) -> None:
        """Test split-brain detected when leader doesn't claim leadership."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_get_node_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {"is_leader": False, "leader_id": "other-node"}
            with patch.object(loop, "_on_split_brain_detected", new_callable=AsyncMock) as mock_handler:
                await loop._check_for_split_brain("claimed-leader")

        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_split_brain_resolved_clears_flag(self) -> None:
        """Test that resolved split-brain clears flag."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._split_brain_detected = True

        with patch.object(loop, "_get_node_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {"is_leader": True, "leader_id": "leader-1"}
            with patch.object(loop, "_emit_event") as mock_emit:
                await loop._check_for_split_brain("leader-1")

        assert loop._split_brain_detected is False
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "SPLIT_BRAIN_RESOLVED"

    @pytest.mark.asyncio
    async def test_on_split_brain_triggers_election(self) -> None:
        """Test that split-brain detection triggers election."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock) as mock_election:
                await loop._on_split_brain_detected("claimed-leader", {"is_leader": False})

        assert loop._split_brain_detected is True
        mock_election.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_split_brain_clears_leader(self) -> None:
        """Test that split-brain detection clears local leader reference."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock):
                await loop._on_split_brain_detected("claimed-leader", {"is_leader": False})

        orchestrator._clear_leader.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_split_brain_only_triggers_once(self) -> None:
        """Test that split-brain only triggers once per detection."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._split_brain_detected = True  # Already detected

        with patch.object(loop, "_emit_event") as mock_emit:
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock) as mock_election:
                await loop._on_split_brain_detected("claimed-leader", {"is_leader": False})

        # Should not trigger again
        mock_emit.assert_not_called()
        mock_election.assert_not_called()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check() method."""

    def test_health_check_healthy_when_leader(self) -> None:
        """Test health check returns healthy when node is leader."""
        orchestrator = create_mock_orchestrator(is_leader=True)
        loop = LeaderProbeLoop(orchestrator)
        loop._running = True

        result = loop.health_check()

        assert result.healthy is True
        assert "leader" in result.message.lower()

    def test_health_check_healthy_when_no_failures(self) -> None:
        """Test health check returns healthy when no failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._running = True
        loop._consecutive_failures = 0

        result = loop.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_degraded_when_some_failures(self) -> None:
        """Test health check returns degraded when some failures."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        loop._running = True
        loop._consecutive_failures = 3

        result = loop.health_check()

        assert result.healthy is True
        assert "degraded" in result.message.lower() or "failures" in result.message.lower()

    def test_health_check_error_at_threshold(self) -> None:
        """Test health check returns error at failure threshold."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        loop._running = True
        loop._consecutive_failures = 6

        result = loop.health_check()

        assert result.healthy is False
        assert "unreachable" in result.message.lower()

    def test_health_check_error_on_split_brain(self) -> None:
        """Test health check returns error on split-brain."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._running = True
        loop._split_brain_detected = True

        result = loop.health_check()

        assert result.healthy is False
        assert "split" in result.message.lower()

    def test_health_check_stopped_when_not_running(self) -> None:
        """Test health check returns stopped status."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._running = False

        result = loop.health_check()

        assert "stopped" in result.message.lower()


# =============================================================================
# Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status() method."""

    def test_get_status_includes_basic_info(self) -> None:
        """Test status includes basic loop info."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._running = True

        status = loop.get_status()

        assert status["name"] == "leader_probe"
        assert status["running"] is True
        assert status["enabled"] is True

    def test_get_status_includes_failure_info(self) -> None:
        """Test status includes failure tracking."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=10)
        loop._consecutive_failures = 3

        status = loop.get_status()

        assert status["consecutive_failures"] == 3
        assert status["failure_threshold"] == 10

    def test_get_status_includes_latency_stats(self) -> None:
        """Test status includes latency statistics."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._latency_history.extend([0.05, 0.06, 0.07])

        status = loop.get_status()

        assert "avg_latency_ms" in status
        assert "recent_latency_ms" in status
        assert status["latency_samples"] == 3

    def test_get_status_includes_split_brain_flag(self) -> None:
        """Test status includes split-brain detection flag."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)
        loop._split_brain_detected = True

        status = loop.get_status()

        assert status["split_brain_detected"] is True


# =============================================================================
# URL Probing Tests
# =============================================================================


class TestUrlProbing:
    """Tests for parallel URL probing."""

    @pytest.mark.asyncio
    async def test_probe_leader_success_with_mocked_parallel(self) -> None:
        """Test probe_leader returns True when _probe_urls_parallel succeeds."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_probe_urls_parallel", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = True
            result = await loop._probe_leader("leader-1")

        assert result is True
        mock_probe.assert_called_once()

    @pytest.mark.asyncio
    async def test_probe_leader_failure_with_mocked_parallel(self) -> None:
        """Test probe_leader returns False when _probe_urls_parallel fails."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_probe_urls_parallel", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = False
            result = await loop._probe_leader("leader-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_leader_returns_true_when_no_urls(self) -> None:
        """Test probe_leader returns True (assumes healthy) when no URLs available."""
        orchestrator = create_mock_orchestrator()
        orchestrator._urls_for_peer = MagicMock(return_value=[])
        loop = LeaderProbeLoop(orchestrator)

        result = await loop._probe_leader("leader-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_leader_returns_true_when_no_urls_method(self) -> None:
        """Test probe_leader returns True when urls_for_peer method is missing."""
        orchestrator = create_mock_orchestrator()
        orchestrator._urls_for_peer = None
        loop = LeaderProbeLoop(orchestrator)

        result = await loop._probe_leader("leader-1")

        assert result is True


# =============================================================================
# Consensus Verification Tests
# =============================================================================


class TestConsensusVerification:
    """Tests for elected leader consensus verification."""

    @pytest.mark.asyncio
    async def test_verify_consensus_returns_true_when_enough_voters_agree(self) -> None:
        """Test consensus verification returns True with 3+ agreeing voters."""
        orchestrator = create_mock_orchestrator(
            leader_id="leader-1",
            voter_node_ids=["voter-1", "voter-2", "voter-3", "voter-4", "voter-5"],
            node_id="voter-1",
        )
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_probe_voter_for_leader", new_callable=AsyncMock) as mock_probe:
            # 4 voters agree on leader-1 (including self)
            mock_probe.side_effect = ["leader-1", "leader-1", "leader-1", "leader-1"]

            result = await loop._verify_elected_leader_consensus()

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_consensus_returns_false_when_insufficient_agreement(self) -> None:
        """Test consensus verification returns False with <3 agreeing voters."""
        orchestrator = create_mock_orchestrator(
            leader_id="leader-1",
            voter_node_ids=["voter-1", "voter-2", "voter-3", "voter-4"],
            node_id="voter-1",
        )
        loop = LeaderProbeLoop(orchestrator)

        with patch.object(loop, "_probe_voter_for_leader", new_callable=AsyncMock) as mock_probe:
            # Only 2 voters agree (self + 1)
            mock_probe.side_effect = ["leader-1", "other-leader", "other-leader"]

            result = await loop._verify_elected_leader_consensus()

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_consensus_returns_false_when_no_leader(self) -> None:
        """Test consensus verification returns False when no leader."""
        orchestrator = create_mock_orchestrator(leader_id=None)
        loop = LeaderProbeLoop(orchestrator)

        result = await loop._verify_elected_leader_consensus()

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_consensus_returns_true_when_no_voter_list(self) -> None:
        """Test consensus returns True when voter list is unavailable."""
        orchestrator = create_mock_orchestrator()
        # Remove voter_node_ids attribute to simulate unavailable list
        del orchestrator.voter_node_ids
        loop = LeaderProbeLoop(orchestrator)

        result = await loop._verify_elected_leader_consensus()

        # Should return True when can't get voter list (assumes consensus)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_consensus_requires_minimum_agreement(self) -> None:
        """Test consensus requires at least 3 agreeing voters."""
        orchestrator = create_mock_orchestrator(
            leader_id="leader-1",
            voter_node_ids=[],  # Empty list
        )
        loop = LeaderProbeLoop(orchestrator)

        result = await loop._verify_elected_leader_consensus()

        # With no voters, can't meet 3 voter requirement, returns False
        assert result is False


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_event_includes_source(self) -> None:
        """Test that events include source field."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # Mock the emit_event import inside _emit_event method
        mock_emit = MagicMock()
        with patch.dict(
            "sys.modules",
            {"app.coordination.event_router": MagicMock(emit_event=mock_emit)},
        ):
            loop._emit_event("TEST_EVENT", {"data": "value"})

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert payload["source"] == "leader_probe_loop"
        assert payload["data"] == "value"
        assert "timestamp" in payload

    def test_emit_event_handles_import_error(self) -> None:
        """Test that event emission handles missing module gracefully."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # Create a mock module that raises ImportError when emit_event is accessed
        mock_module = MagicMock()
        mock_module.emit_event.side_effect = Exception("Test error")

        with patch.dict(
            "sys.modules",
            {"app.coordination.event_router": mock_module},
        ):
            # Should not raise
            loop._emit_event("TEST_EVENT", {"data": "value"})

    def test_emit_event_graceful_on_exception(self) -> None:
        """Test that event emission catches and logs exceptions."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator)

        # The _emit_event method has try/except, so it should handle any errors
        # Just verify it doesn't raise
        loop._emit_event("TEST_EVENT", {"data": "value"})


# =============================================================================
# Dynamic Threshold Tests (P2.1 - Jan 13, 2026)
# =============================================================================


class TestDynamicThreshold:
    """Tests for dynamic failure threshold scaling."""

    def test_dynamic_threshold_default_when_disabled(self) -> None:
        """Test that base threshold is used when dynamic scaling is disabled."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        loop._dynamic_threshold_enabled = False

        threshold = loop._compute_dynamic_failure_threshold()

        assert threshold == 6

    def test_dynamic_threshold_reduces_for_healthy_quorum(self) -> None:
        """Test threshold reduces when quorum is HEALTHY."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)

        # Mock healthy quorum
        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.HEALTHY)

        threshold = loop._compute_dynamic_failure_threshold()

        # Should reduce by 2 (but clamped to min)
        assert threshold <= 6
        assert threshold >= loop._min_failure_threshold

    def test_dynamic_threshold_increases_for_degraded_quorum(self) -> None:
        """Test threshold increases when quorum is DEGRADED."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.DEGRADED)

        threshold = loop._compute_dynamic_failure_threshold()

        # Should increase by 2
        assert threshold >= 6
        assert threshold <= loop._max_failure_threshold

    def test_dynamic_threshold_increases_more_for_minimum_quorum(self) -> None:
        """Test threshold increases significantly when quorum is at MINIMUM."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.MINIMUM)

        threshold = loop._compute_dynamic_failure_threshold()

        # Should increase by 4
        assert threshold >= 8
        assert threshold <= loop._max_failure_threshold

    def test_dynamic_threshold_increases_for_high_latency(self) -> None:
        """Test threshold increases when latency is high."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        orchestrator._check_quorum_health = MagicMock(return_value=None)

        # Add very high latency samples (>5s)
        loop._latency_history.extend([5.5, 5.8, 6.0])

        threshold = loop._compute_dynamic_failure_threshold()

        # Should increase by 3 for very high latency
        assert threshold >= 9

    def test_dynamic_threshold_moderate_increase_for_moderate_latency(self) -> None:
        """Test threshold increases moderately for moderate latency."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        orchestrator._check_quorum_health = MagicMock(return_value=None)

        # Add elevated latency samples (>3s)
        loop._latency_history.extend([3.2, 3.4, 3.1])

        threshold = loop._compute_dynamic_failure_threshold()

        # Should increase by 1
        assert threshold >= 7

    def test_dynamic_threshold_reduces_for_low_latency(self) -> None:
        """Test threshold reduces when latency is very low."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        orchestrator._check_quorum_health = MagicMock(return_value=None)

        # Add very low latency samples (<100ms)
        loop._latency_history.extend([0.05, 0.04, 0.06])

        threshold = loop._compute_dynamic_failure_threshold()

        # Should reduce by 1 (but clamped to min)
        assert threshold <= 6
        assert threshold >= loop._min_failure_threshold

    def test_dynamic_threshold_clamped_to_min(self) -> None:
        """Test threshold is clamped to minimum value."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=3)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.HEALTHY)
        loop._latency_history.extend([0.01, 0.01, 0.01])

        threshold = loop._compute_dynamic_failure_threshold()

        assert threshold >= loop._min_failure_threshold

    def test_dynamic_threshold_clamped_to_max(self) -> None:
        """Test threshold is clamped to maximum value."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=10)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.MINIMUM)
        loop._latency_history.extend([5.0, 6.0, 7.0])

        threshold = loop._compute_dynamic_failure_threshold()

        assert threshold <= loop._max_failure_threshold

    def test_dynamic_threshold_combines_factors(self) -> None:
        """Test that quorum health and latency factors combine."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.DEGRADED)
        loop._latency_history.extend([1.5, 1.6, 1.4])  # Moderate latency

        threshold = loop._compute_dynamic_failure_threshold()

        # DEGRADED (+2) + relay-range latency (+0) = 8 under the relaxed Jan 2026 thresholds
        assert threshold >= 8

    def test_get_status_includes_dynamic_threshold(self) -> None:
        """Test that get_status() includes dynamic threshold info."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6)
        loop._running = True

        status = loop.get_status()

        assert "dynamic_threshold" in status
        assert "dynamic_timeout_seconds" in status
        assert "dynamic_threshold_enabled" in status

    @pytest.mark.asyncio
    async def test_probe_failure_uses_dynamic_threshold(self) -> None:
        """Test that _on_probe_failure uses dynamic threshold for election."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6, startup_grace_period=0)

        from scripts.p2p.leader_election import QuorumHealthLevel
        orchestrator._check_quorum_health = MagicMock(return_value=QuorumHealthLevel.HEALTHY)
        loop._latency_history.extend([0.05, 0.04, 0.06])  # Low latency

        # Dynamic threshold should be ~4 (healthy quorum -2, low latency -1)
        # Set failures just below dynamic threshold
        loop._consecutive_failures = 3  # Will become 4

        with patch.object(loop, "_emit_event"):
            with patch.object(loop, "_trigger_election", new_callable=AsyncMock) as mock_election:
                await loop._on_probe_failure("leader-1")

        # Should trigger election at dynamic threshold (4), not base (6)
        mock_election.assert_called_once()

    @pytest.mark.asyncio
    async def test_probe_failure_event_includes_dynamic_threshold(self) -> None:
        """Test that LEADER_PROBE_FAILED event includes dynamic threshold info."""
        orchestrator = create_mock_orchestrator()
        loop = LeaderProbeLoop(orchestrator, failure_threshold=6, startup_grace_period=0)

        with patch.object(loop, "_emit_event") as mock_emit:
            await loop._on_probe_failure("leader-1")

        call_args = mock_emit.call_args
        payload = call_args[0][1]
        assert "failure_threshold" in payload
        assert "base_threshold" in payload
        assert "dynamic_threshold_enabled" in payload
