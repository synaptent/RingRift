"""Unit tests for leadership_state_machine.py.

Tests the unified leadership state machine for P2P cluster leader management.
Created as part of the ULSM initiative (Jan 2026).
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.leadership_state_machine import (
    VALID_TRANSITIONS,
    LeadershipStateMachine,
    LeaderState,
    QuorumHealth,
    TransitionReason,
)


# =============================================================================
# QuorumHealth Tests
# =============================================================================


class TestQuorumHealth:
    """Tests for the QuorumHealth dataclass."""

    def test_initial_state(self):
        """QuorumHealth starts with zero failures."""
        qh = QuorumHealth()
        assert qh.consecutive_failures == 0
        assert qh.failure_threshold == 5
        assert qh.voters_seen_last_check == 0

    def test_record_failure_increments_count(self):
        """record_failure increments consecutive_failures."""
        qh = QuorumHealth()
        qh.record_failure(voters_alive=2)
        assert qh.consecutive_failures == 1
        assert qh.voters_seen_last_check == 2

    def test_record_failure_threshold_not_exceeded(self):
        """record_failure returns False when under threshold."""
        qh = QuorumHealth(failure_threshold=5)
        for i in range(4):
            result = qh.record_failure(voters_alive=1)
            assert result is False
        assert qh.consecutive_failures == 4

    def test_record_failure_threshold_exceeded(self):
        """record_failure returns True when threshold exceeded."""
        qh = QuorumHealth(failure_threshold=3)
        qh.record_failure(voters_alive=1)
        qh.record_failure(voters_alive=1)
        result = qh.record_failure(voters_alive=0)
        assert result is True
        assert qh.consecutive_failures == 3

    def test_record_success_resets_failures(self):
        """record_success resets consecutive_failures to 0."""
        qh = QuorumHealth()
        qh.record_failure(voters_alive=1)
        qh.record_failure(voters_alive=1)
        assert qh.consecutive_failures == 2

        qh.record_success(voters_alive=5)
        assert qh.consecutive_failures == 0
        assert qh.voters_seen_last_check == 5
        assert qh.last_success_time > 0

    def test_reset_clears_failures(self):
        """reset() clears consecutive_failures."""
        qh = QuorumHealth()
        qh.consecutive_failures = 10
        qh.reset()
        assert qh.consecutive_failures == 0

    def test_to_dict_serialization(self):
        """to_dict returns proper structure for /status endpoint."""
        qh = QuorumHealth(consecutive_failures=2, failure_threshold=5)
        qh.voters_seen_last_check = 3
        qh.last_success_time = time.time() - 10

        result = qh.to_dict()
        assert result["consecutive_failures"] == 2
        assert result["failure_threshold"] == 5
        assert result["voters_seen_last_check"] == 3
        assert 9 < result["seconds_since_success"] < 12


# =============================================================================
# LeaderState Tests
# =============================================================================


class TestLeaderState:
    """Tests for the LeaderState enum."""

    def test_all_states_defined(self):
        """All expected states are defined."""
        assert LeaderState.FOLLOWER.value == "follower"
        assert LeaderState.CANDIDATE.value == "candidate"
        assert LeaderState.PROVISIONAL_LEADER.value == "provisional_leader"
        assert LeaderState.LEADER.value == "leader"
        assert LeaderState.STEPPING_DOWN.value == "stepping_down"

    def test_stepping_down_is_intermediate(self):
        """STEPPING_DOWN can only go to FOLLOWER."""
        valid = VALID_TRANSITIONS[LeaderState.STEPPING_DOWN]
        assert valid == {LeaderState.FOLLOWER}

    def test_leader_must_step_down(self):
        """LEADER cannot go directly to FOLLOWER."""
        valid = VALID_TRANSITIONS[LeaderState.LEADER]
        assert LeaderState.FOLLOWER not in valid
        assert LeaderState.STEPPING_DOWN in valid


# =============================================================================
# TransitionReason Tests
# =============================================================================


class TestTransitionReason:
    """Tests for the TransitionReason enum."""

    def test_leadership_gain_reasons(self):
        """Reasons for gaining leadership are defined."""
        assert TransitionReason.ELECTION_WON.value == "election_won"
        assert TransitionReason.PROVISIONAL_CLAIM.value == "provisional_claim"
        assert TransitionReason.PROVISIONAL_PROMOTED.value == "provisional_promoted"

    def test_leadership_loss_reasons(self):
        """Reasons for losing leadership are defined."""
        assert TransitionReason.LEASE_EXPIRED.value == "lease_expired"
        assert TransitionReason.QUORUM_LOST.value == "quorum_lost"
        assert TransitionReason.ARBITER_OVERRIDE.value == "arbiter_override"
        assert TransitionReason.HIGHER_EPOCH_SEEN.value == "higher_epoch_seen"

    def test_recovery_reasons(self):
        """Reasons for recovery are defined."""
        assert TransitionReason.RESTART_HEALING.value == "restart_healing"
        assert TransitionReason.ELECTION_LOST.value == "election_lost"


# =============================================================================
# VALID_TRANSITIONS Tests
# =============================================================================


class TestValidTransitions:
    """Tests for the transition matrix."""

    def test_follower_can_become_candidate(self):
        """FOLLOWER can transition to CANDIDATE."""
        assert LeaderState.CANDIDATE in VALID_TRANSITIONS[LeaderState.FOLLOWER]

    def test_follower_can_become_provisional(self):
        """FOLLOWER can transition to PROVISIONAL_LEADER."""
        assert LeaderState.PROVISIONAL_LEADER in VALID_TRANSITIONS[LeaderState.FOLLOWER]

    def test_candidate_can_become_leader(self):
        """CANDIDATE can transition to LEADER."""
        assert LeaderState.LEADER in VALID_TRANSITIONS[LeaderState.CANDIDATE]

    def test_candidate_can_fall_back(self):
        """CANDIDATE can transition back to FOLLOWER."""
        assert LeaderState.FOLLOWER in VALID_TRANSITIONS[LeaderState.CANDIDATE]

    def test_leader_requires_stepping_down(self):
        """LEADER must go through STEPPING_DOWN."""
        valid = VALID_TRANSITIONS[LeaderState.LEADER]
        assert valid == {LeaderState.STEPPING_DOWN}

    def test_provisional_can_be_promoted(self):
        """PROVISIONAL_LEADER can be promoted to LEADER."""
        assert LeaderState.LEADER in VALID_TRANSITIONS[LeaderState.PROVISIONAL_LEADER]

    def test_stepping_down_only_to_follower(self):
        """STEPPING_DOWN only goes to FOLLOWER."""
        valid = VALID_TRANSITIONS[LeaderState.STEPPING_DOWN]
        assert valid == {LeaderState.FOLLOWER}


# =============================================================================
# LeadershipStateMachine Initialization Tests
# =============================================================================


class TestLeadershipStateMachineInit:
    """Tests for LeadershipStateMachine initialization."""

    def test_default_initialization(self):
        """Default initialization sets correct defaults."""
        sm = LeadershipStateMachine(node_id="node-1")
        assert sm.node_id == "node-1"
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 0
        assert sm.leader_id is None
        assert sm.is_leader is False
        assert sm.is_provisional_leader is False

    def test_custom_initial_state(self):
        """Can initialize with custom state."""
        sm = LeadershipStateMachine(
            node_id="node-1",
            initial_state=LeaderState.CANDIDATE,
            initial_epoch=5,
        )
        assert sm.state == LeaderState.CANDIDATE
        assert sm.epoch == 5

    def test_leader_state_sets_leader_id(self):
        """Initializing as LEADER sets leader_id to self."""
        sm = LeadershipStateMachine(
            node_id="node-1",
            initial_state=LeaderState.LEADER,
        )
        assert sm.leader_id == "node-1"
        assert sm.is_leader is True

    def test_provisional_leader_sets_leader_id(self):
        """Initializing as PROVISIONAL_LEADER sets leader_id to self."""
        sm = LeadershipStateMachine(
            node_id="node-1",
            initial_state=LeaderState.PROVISIONAL_LEADER,
        )
        assert sm.leader_id == "node-1"
        assert sm.is_provisional_leader is True


# =============================================================================
# LeadershipStateMachine Transition Tests
# =============================================================================


class TestLeadershipStateMachineTransitions:
    """Tests for transition_to() method."""

    @pytest.fixture
    def sm(self):
        """Create a fresh state machine for each test."""
        return LeadershipStateMachine(node_id="test-node")

    @pytest.mark.asyncio
    async def test_valid_follower_to_candidate(self, sm):
        """FOLLOWER -> CANDIDATE is valid."""
        result = await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        assert result is True
        assert sm.state == LeaderState.CANDIDATE

    @pytest.mark.asyncio
    async def test_valid_candidate_to_leader(self, sm):
        """CANDIDATE -> LEADER is valid."""
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        result = await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        assert result is True
        assert sm.state == LeaderState.LEADER
        assert sm.leader_id == "test-node"
        assert sm.is_leader is True

    @pytest.mark.asyncio
    async def test_invalid_follower_to_leader(self, sm):
        """FOLLOWER -> LEADER is invalid (must go through CANDIDATE)."""
        result = await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        assert result is False
        assert sm.state == LeaderState.FOLLOWER

    @pytest.mark.asyncio
    async def test_invalid_follower_to_stepping_down(self, sm):
        """FOLLOWER -> STEPPING_DOWN is invalid."""
        result = await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)
        assert result is False
        assert sm.state == LeaderState.FOLLOWER

    @pytest.mark.asyncio
    async def test_leader_to_stepping_down_increments_epoch(self, sm):
        """LEADER -> STEPPING_DOWN increments epoch."""
        # Become leader first
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        initial_epoch = sm.epoch

        # Step down
        result = await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)
        assert result is True
        assert sm.epoch == initial_epoch + 1

    @pytest.mark.asyncio
    async def test_leader_to_stepping_down_sets_invalidation(self, sm):
        """LEADER -> STEPPING_DOWN sets invalidation window."""
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)

        assert sm.in_invalidation_window is True
        assert sm.invalidation_remaining_seconds > 0

    @pytest.mark.asyncio
    async def test_stepping_down_to_follower(self, sm):
        """STEPPING_DOWN -> FOLLOWER clears leader_id."""
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)

        result = await sm.transition_to(LeaderState.FOLLOWER, TransitionReason.STEP_DOWN_COMPLETE)
        assert result is True
        assert sm.state == LeaderState.FOLLOWER
        assert sm.leader_id is None
        assert sm.is_leader is False

    @pytest.mark.asyncio
    async def test_transition_resets_quorum_health(self, sm):
        """Any transition resets quorum health."""
        sm.quorum_health.consecutive_failures = 10
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        assert sm.quorum_health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_transition_increments_count(self, sm):
        """Transitions increment transition_count."""
        initial_count = sm._transition_count
        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        assert sm._transition_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_broadcast_called_on_stepping_down(self, sm):
        """Broadcast callback is called when stepping down."""
        mock_callback = AsyncMock()
        sm._broadcast_callback = mock_callback

        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        assert call_args[0] == "stepping_down"
        assert call_args[2] == TransitionReason.QUORUM_LOST

    @pytest.mark.asyncio
    async def test_broadcast_failure_does_not_block_transition(self, sm):
        """Transition succeeds even if broadcast fails."""
        mock_callback = AsyncMock(side_effect=Exception("Network error"))
        sm._broadcast_callback = mock_callback

        await sm.transition_to(LeaderState.CANDIDATE, TransitionReason.ELECTION_WON)
        await sm.transition_to(LeaderState.LEADER, TransitionReason.ELECTION_WON)

        # Should still succeed despite broadcast failure
        result = await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)
        assert result is True
        assert sm.state == LeaderState.STEPPING_DOWN


# =============================================================================
# LeadershipStateMachine validate_leader_claim Tests
# =============================================================================


class TestValidateLeaderClaim:
    """Tests for validate_leader_claim() method."""

    @pytest.fixture
    def sm(self):
        """Create a follower state machine."""
        return LeadershipStateMachine(node_id="test-node")

    def test_valid_claim_accepted(self, sm):
        """Valid leader claim is accepted."""
        future = time.time() + 60
        result = sm.validate_leader_claim(
            claimed_leader="leader-1",
            claimed_epoch=1,
            lease_expires=future,
        )
        assert result is True

    def test_stale_epoch_rejected(self, sm):
        """Claim with lower epoch is rejected."""
        sm._epoch = 5
        result = sm.validate_leader_claim(
            claimed_leader="leader-1",
            claimed_epoch=3,
            lease_expires=time.time() + 60,
        )
        assert result is False

    def test_expired_lease_rejected(self, sm):
        """Claim with expired lease is rejected."""
        past = time.time() - 10
        result = sm.validate_leader_claim(
            claimed_leader="leader-1",
            claimed_epoch=1,
            lease_expires=past,
        )
        assert result is False

    def test_invalidation_window_rejects_claims(self, sm):
        """Claims during invalidation window are rejected."""
        sm._invalidation_until = time.time() + 30

        result = sm.validate_leader_claim(
            claimed_leader="leader-1",
            claimed_epoch=10,
            lease_expires=time.time() + 60,
        )
        assert result is False

    def test_existing_leader_same_epoch_rejected(self, sm):
        """Claim at same epoch when we have a leader is rejected."""
        sm._leader_id = "existing-leader"
        sm._epoch = 5

        result = sm.validate_leader_claim(
            claimed_leader="new-leader",
            claimed_epoch=5,
            lease_expires=time.time() + 60,
        )
        assert result is False

    def test_higher_epoch_claim_with_existing_leader(self, sm):
        """Claim at higher epoch is accepted even with existing leader."""
        sm._leader_id = "existing-leader"
        sm._epoch = 5

        result = sm.validate_leader_claim(
            claimed_leader="new-leader",
            claimed_epoch=6,
            lease_expires=time.time() + 60,
        )
        assert result is True


# =============================================================================
# LeadershipStateMachine set_leader/clear_leader Tests
# =============================================================================


class TestSetClearLeader:
    """Tests for set_leader() and clear_leader() methods."""

    def test_set_leader_updates_leader_id(self):
        """set_leader updates leader_id."""
        sm = LeadershipStateMachine(node_id="test-node")
        sm.set_leader("leader-1", epoch=5)
        assert sm.leader_id == "leader-1"

    def test_set_leader_updates_epoch_if_higher(self):
        """set_leader updates epoch if claimed is higher."""
        sm = LeadershipStateMachine(node_id="test-node", initial_epoch=3)
        sm.set_leader("leader-1", epoch=5)
        assert sm.epoch == 5

    def test_set_leader_does_not_lower_epoch(self):
        """set_leader does not lower epoch."""
        sm = LeadershipStateMachine(node_id="test-node", initial_epoch=10)
        sm.set_leader("leader-1", epoch=5)
        assert sm.epoch == 10

    def test_clear_leader_clears_leader_id(self):
        """clear_leader sets leader_id to None."""
        sm = LeadershipStateMachine(node_id="test-node")
        sm.set_leader("leader-1", epoch=1)
        assert sm.leader_id == "leader-1"

        sm.clear_leader()
        assert sm.leader_id is None


# =============================================================================
# LeadershipStateMachine from_persisted_state Tests
# =============================================================================


class TestFromPersistedState:
    """Tests for from_persisted_state() factory method."""

    def test_basic_restoration(self):
        """Basic state restoration works."""
        state_dict = {
            "state": "follower",
            "epoch": 5,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.node_id == "node-1"
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 5

    def test_leader_state_healed_to_follower(self):
        """LEADER state is healed to FOLLOWER on restart."""
        state_dict = {
            "state": "leader",
            "epoch": 5,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 6  # Incremented for healing

    def test_provisional_leader_healed_to_follower(self):
        """PROVISIONAL_LEADER is healed to FOLLOWER on restart."""
        state_dict = {
            "state": "provisional_leader",
            "epoch": 3,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 4

    def test_stepping_down_healed_to_follower(self):
        """STEPPING_DOWN is healed to FOLLOWER on restart."""
        state_dict = {
            "state": "stepping_down",
            "epoch": 7,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 8

    def test_candidate_not_healed(self):
        """CANDIDATE state is preserved (not leader claim)."""
        state_dict = {
            "state": "candidate",
            "epoch": 5,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.state == LeaderState.CANDIDATE
        assert sm.epoch == 5

    def test_follower_not_healed(self):
        """FOLLOWER state is preserved."""
        state_dict = {
            "state": "follower",
            "epoch": 5,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 5

    def test_invalidation_window_restored(self):
        """Active invalidation window is restored."""
        future = time.time() + 30
        state_dict = {
            "state": "follower",
            "epoch": 5,
            "invalidation_until": future,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.in_invalidation_window is True
        assert sm.invalidation_remaining_seconds > 25

    def test_expired_invalidation_not_restored(self):
        """Expired invalidation window is not restored."""
        past = time.time() - 30
        state_dict = {
            "state": "follower",
            "epoch": 5,
            "invalidation_until": past,
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.in_invalidation_window is False

    def test_quorum_health_threshold_restored(self):
        """Quorum health failure_threshold is restored."""
        state_dict = {
            "state": "follower",
            "epoch": 5,
            "quorum_health": {
                "failure_threshold": 10,
            },
        }
        sm = LeadershipStateMachine.from_persisted_state("node-1", state_dict)
        assert sm.quorum_health.failure_threshold == 10

    def test_empty_state_dict(self):
        """Empty state dict uses defaults."""
        sm = LeadershipStateMachine.from_persisted_state("node-1", {})
        assert sm.state == LeaderState.FOLLOWER
        assert sm.epoch == 0


# =============================================================================
# LeadershipStateMachine Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for to_dict() and to_persistence_dict() methods."""

    @pytest.fixture
    def sm(self):
        """Create a leader state machine for serialization tests."""
        sm = LeadershipStateMachine(
            node_id="test-node",
            initial_state=LeaderState.LEADER,
            initial_epoch=5,
        )
        return sm

    def test_to_dict_has_all_fields(self, sm):
        """to_dict includes all status fields."""
        result = sm.to_dict()

        assert "state" in result
        assert "epoch" in result
        assert "leader_id" in result
        assert "node_id" in result
        assert "is_leader" in result
        assert "is_provisional_leader" in result
        assert "in_invalidation_window" in result
        assert "quorum_health" in result
        assert "last_transition_time" in result
        assert "transition_count" in result

    def test_to_dict_values(self, sm):
        """to_dict has correct values."""
        result = sm.to_dict()

        assert result["state"] == "leader"
        assert result["epoch"] == 5
        assert result["leader_id"] == "test-node"
        assert result["node_id"] == "test-node"
        assert result["is_leader"] is True
        assert result["is_provisional_leader"] is False

    def test_to_persistence_dict_minimal(self, sm):
        """to_persistence_dict has minimal footprint."""
        result = sm.to_persistence_dict()

        # Should have these fields
        assert "state" in result
        assert "epoch" in result
        assert "invalidation_until" in result
        assert "quorum_health" in result

        # Should NOT have transient fields
        assert "is_leader" not in result
        assert "last_transition_time" not in result
        assert "transition_count" not in result

    def test_round_trip_persistence(self, sm):
        """Can persist and restore state."""
        # Persist
        persisted = sm.to_persistence_dict()

        # Restore
        restored = LeadershipStateMachine.from_persisted_state("test-node", persisted)

        # Leader state gets healed to follower
        assert restored.state == LeaderState.FOLLOWER
        assert restored.epoch == sm.epoch + 1  # Incremented for healing


# =============================================================================
# LeadershipStateMachine Properties Tests
# =============================================================================


class TestProperties:
    """Tests for computed properties."""

    def test_is_leader_requires_leader_state_and_self_id(self):
        """is_leader is True only when LEADER and leader_id == node_id."""
        sm = LeadershipStateMachine(node_id="test-node", initial_state=LeaderState.LEADER)
        assert sm.is_leader is True

        sm._leader_id = "other-node"
        assert sm.is_leader is False

        sm._leader_id = "test-node"
        sm._state = LeaderState.FOLLOWER
        assert sm.is_leader is False

    def test_is_provisional_leader(self):
        """is_provisional_leader is True only for PROVISIONAL_LEADER state."""
        sm = LeadershipStateMachine(node_id="test-node", initial_state=LeaderState.PROVISIONAL_LEADER)
        assert sm.is_provisional_leader is True

        sm._state = LeaderState.LEADER
        assert sm.is_provisional_leader is False

    def test_is_stepping_down(self):
        """is_stepping_down is True only for STEPPING_DOWN state."""
        sm = LeadershipStateMachine(node_id="test-node")
        assert sm.is_stepping_down is False

        sm._state = LeaderState.STEPPING_DOWN
        assert sm.is_stepping_down is True

    def test_in_invalidation_window(self):
        """in_invalidation_window tracks time correctly."""
        sm = LeadershipStateMachine(node_id="test-node")
        assert sm.in_invalidation_window is False

        sm._invalidation_until = time.time() + 10
        assert sm.in_invalidation_window is True

        sm._invalidation_until = time.time() - 10
        assert sm.in_invalidation_window is False

    def test_invalidation_remaining_seconds(self):
        """invalidation_remaining_seconds returns correct value."""
        sm = LeadershipStateMachine(node_id="test-node")
        assert sm.invalidation_remaining_seconds == 0

        sm._invalidation_until = time.time() + 30
        remaining = sm.invalidation_remaining_seconds
        assert 28 < remaining <= 30


# =============================================================================
# LeadershipStateMachine __repr__ Tests
# =============================================================================


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_format(self):
        """__repr__ returns informative string."""
        sm = LeadershipStateMachine(
            node_id="test-node",
            initial_state=LeaderState.LEADER,
            initial_epoch=5,
        )
        repr_str = repr(sm)
        assert "LeadershipStateMachine" in repr_str
        assert "test-node" in repr_str
        assert "leader" in repr_str
        assert "epoch=5" in repr_str
