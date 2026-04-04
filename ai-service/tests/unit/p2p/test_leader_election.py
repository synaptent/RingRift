"""Unit tests for leader_election.py.

Tests the leader election mixin for P2P cluster consensus.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Mock NodeInfo dataclass
@dataclass
class MockNodeInfo:
    node_id: str
    host: str = ""
    port: int = 8770
    last_heartbeat: float = 0.0
    leader_id: str | None = None

    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 90


# Mock NodeRole enum
class MockNodeRole(str, Enum):
    LEADER = "leader"
    FOLLOWER = "follower"


_P2P_TEST_MODULES = {
    "scripts.p2p.types": MagicMock(NodeRole=MockNodeRole),
    "scripts.p2p.models": MagicMock(NodeInfo=MockNodeInfo),
}

# Import the module under test with lightweight P2P type shims installed.
with patch.dict("sys.modules", _P2P_TEST_MODULES):
    from scripts.p2p.leader_election import (
        VOTER_MIN_QUORUM,
        LeaderElectionMixin,
        check_quorum,
    )


@pytest.fixture(autouse=True)
def patch_runtime_p2p_modules():
    """Keep lightweight P2P shims available for lazy imports during each test."""
    with patch.dict("sys.modules", _P2P_TEST_MODULES):
        yield


class TestableLeaderElection(LeaderElectionMixin):
    """Concrete implementation for testing the mixin."""

    __test__ = False

    def __init__(self, node_id: str = "test-node-1"):
        self.node_id = node_id
        self.role = MockNodeRole.FOLLOWER
        self.leader_id: str | None = None
        self.leader_lease_id = ""
        self.leader_lease_expires = 0.0
        self.last_lease_renewal = 0.0
        self.voter_node_ids: list[str] = []
        self.voter_grant_leader_id = ""
        self.voter_grant_lease_id = ""
        self.voter_grant_expires = 0.0
        self.peers_lock = threading.RLock()
        self.peers: dict[str, MockNodeInfo] = {}
        self._events_emitted: list[tuple[str, dict]] = []
        self._logs: list[tuple[str, str]] = []

    def _save_state(self) -> None:
        pass  # No-op for tests

    def _safe_emit_event(self, event_type: str, data: dict) -> None:
        self._events_emitted.append((event_type, data))

    def _log_warning(self, msg: str) -> None:
        self._logs.append(("warning", msg))

    def _log_error(self, msg: str) -> None:
        self._logs.append(("error", msg))

    def _log_info(self, msg: str) -> None:
        self._logs.append(("info", msg))

    def _count_alive_peers(self, node_ids: list[str]) -> int:
        """Count how many of the given node IDs are currently alive."""
        if not node_ids:
            return 0

        alive = 0
        with self.peers_lock:
            peers = dict(self.peers)

        for nid in node_ids:
            if nid == self.node_id:
                alive += 1
                continue
            peer = peers.get(nid)
            if peer and peer.is_alive():
                alive += 1

        return alive

    def _get_alive_peer_list(self, node_ids: list[str]) -> list[str]:
        """Get list of node IDs that are currently alive."""
        if not node_ids:
            return []

        alive_list = []
        with self.peers_lock:
            peers = dict(self.peers)

        for nid in node_ids:
            if nid == self.node_id:
                alive_list.append(nid)
                continue
            peer = peers.get(nid)
            if peer and peer.is_alive():
                alive_list.append(nid)

        return alive_list


class TestVoterMinQuorum:
    """Test the VOTER_MIN_QUORUM constant."""

    def test_voter_min_quorum_value(self):
        """Test that VOTER_MIN_QUORUM matches the current 2-voter minimum."""
        assert VOTER_MIN_QUORUM == 2


class TestHasVoterQuorum:
    """Test _has_voter_quorum method."""

    def test_returns_true_when_no_voters(self):
        """Test that quorum is met when no voters configured."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        assert election._has_voter_quorum() is True

    def test_returns_true_when_quorum_met(self):
        """Test that quorum is met with enough alive voters."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3", "voter-4", "voter-5"]

        # Add 3 alive voters - comfortably above the current minimum quorum.
        for i in range(1, 4):
            election.peers[f"voter-{i}"] = MockNodeInfo(
                node_id=f"voter-{i}",
                last_heartbeat=time.time(),
            )

        assert election._has_voter_quorum() is True

    def test_returns_true_with_dynamic_quorum_when_two_voters_alive(self):
        """Dynamic quorum allows progress with two currently alive voters."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3", "voter-4", "voter-5"]

        # Jan 2026: dynamic quorum uses majority-of-alive voters, so two alive
        # voters can still form quorum even if five voters are configured.
        for i in range(1, 3):
            election.peers[f"voter-{i}"] = MockNodeInfo(
                node_id=f"voter-{i}",
                last_heartbeat=time.time(),
            )

        assert election._has_voter_quorum() is True

    def test_returns_false_when_no_voters_are_alive(self):
        """Quorum is lost when no configured voters are currently alive."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3", "voter-4", "voter-5"]

        assert election._has_voter_quorum() is False

    def test_uses_min_of_quorum_and_total_voters(self):
        """Test that quorum is min(VOTER_MIN_QUORUM, total_voters)."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2"]  # Only 2 voters

        # Add both voters alive - should meet quorum since min(2, 2) = 2
        for i in range(1, 3):
            election.peers[f"voter-{i}"] = MockNodeInfo(
                node_id=f"voter-{i}",
                last_heartbeat=time.time(),
            )

        assert election._has_voter_quorum() is True

    def test_counts_self_as_voter_if_in_list(self):
        """Test that self is counted as a voter if in the voter list."""
        election = TestableLeaderElection(node_id="voter-1")
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]

        # Add 2 other alive voters
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
        )

        # Self (voter-1) + 2 peers = 3 voters = quorum met
        assert election._has_voter_quorum() is True


class TestReleaseVoterGrantIfSelf:
    """Test _release_voter_grant_if_self method."""

    def test_releases_grant_when_self_is_leader(self):
        """Test that voter grant is released when self is the leader."""
        election = TestableLeaderElection()
        election.voter_grant_leader_id = election.node_id
        election.voter_grant_lease_id = "some-lease-id"
        election.voter_grant_expires = time.time() + 60

        election._release_voter_grant_if_self()

        assert election.voter_grant_leader_id == ""
        assert election.voter_grant_lease_id == ""
        assert election.voter_grant_expires == 0.0

    def test_does_not_release_when_other_is_leader(self):
        """Test that voter grant is not released when other node is leader."""
        election = TestableLeaderElection()
        election.voter_grant_leader_id = "other-node"
        election.voter_grant_lease_id = "some-lease-id"
        election.voter_grant_expires = time.time() + 60

        election._release_voter_grant_if_self()

        assert election.voter_grant_leader_id == "other-node"
        assert election.voter_grant_lease_id == "some-lease-id"


class TestGetVoterQuorumStatus:
    """Test _get_voter_quorum_status method."""

    def test_returns_empty_when_no_voters(self):
        """Test that empty status is returned when no voters."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        status = election._get_voter_quorum_status()

        assert status["voters"] == []
        assert status["alive"] == 0
        assert status["total"] == 0
        assert status["quorum_met"] is True

    def test_returns_detailed_status(self):
        """Test that detailed status reflects the current quorum threshold."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]

        # Add 2 alive voters
        election.peers["voter-1"] = MockNodeInfo(
            node_id="voter-1",
            last_heartbeat=time.time(),
        )
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
        )

        status = election._get_voter_quorum_status()

        assert status["total"] == 3
        assert status["alive"] == 2
        assert status["quorum_required"] == min(VOTER_MIN_QUORUM, 3)
        assert status["quorum_met"] is True
        assert "voter-1" in status["alive_list"]
        assert "voter-2" in status["alive_list"]


class TestCheckLeaderConsistency:
    """Test _check_leader_consistency method."""

    def test_consistent_when_leader_matches_role(self):
        """Test that state is consistent when leader_id and role match."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.LEADER
        election.leader_id = election.node_id

        is_consistent, reason = election._check_leader_consistency()

        assert is_consistent is True
        assert reason == "consistent"

    def test_consistent_when_follower_with_other_leader(self):
        """Test that state is consistent when follower with other leader."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.FOLLOWER
        election.leader_id = "other-leader"

        is_consistent, reason = election._check_leader_consistency()

        assert is_consistent is True
        assert reason == "consistent"

    def test_inconsistent_when_leader_id_self_but_role_follower(self):
        """Test inconsistency when leader_id=self but role=follower."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.FOLLOWER
        election.leader_id = election.node_id

        is_consistent, reason = election._check_leader_consistency()

        assert is_consistent is False
        assert "leader_id=self but role!=leader" in reason

    def test_inconsistent_when_role_leader_but_leader_id_other(self):
        """Test inconsistency when role=leader but leader_id is other node."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.LEADER
        election.leader_id = "other-node"

        is_consistent, reason = election._check_leader_consistency()

        assert is_consistent is False
        assert "role=leader but leader_id!=self" in reason


class TestHasVoterConsensusOnLeader:
    """Test _has_voter_consensus_on_leader method."""

    def test_returns_true_when_no_voters(self):
        """Test that consensus is met when no voters configured."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        assert election._has_voter_consensus_on_leader("any-leader") is True

    def test_returns_true_when_quorum_agrees(self):
        """Test that consensus is met when quorum agrees."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"  # Must be in voter list to count self's vote
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_id = "proposed-leader"

        # Add 2 peers that agree on the leader
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="proposed-leader",
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="proposed-leader",
        )

        # Self (voter-1) + 2 agreeing peers = 3 votes = quorum met
        assert election._has_voter_consensus_on_leader("proposed-leader") is True

    def test_returns_false_when_no_consensus(self):
        """Test that consensus fails when voters disagree."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_id = "proposed-leader"

        # Add peers with different leaders
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="other-leader",  # Disagrees
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="other-leader",  # Disagrees
        )

        # Only self agrees - not enough for quorum
        assert election._has_voter_consensus_on_leader("proposed-leader") is False

    def test_counts_self_vote(self):
        """Test that self's vote is counted if in voter list."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_id = "proposed-leader"

        # Add 2 peers that agree
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="proposed-leader",
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="proposed-leader",
        )

        # Self + 2 peers = 3 votes = quorum met
        assert election._has_voter_consensus_on_leader("proposed-leader") is True


class TestCountVotesForLeader:
    """Test _count_votes_for_leader method."""

    def test_returns_1_when_no_voters(self):
        """Test that 1 is returned when no voters configured."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        assert election._count_votes_for_leader("any-leader") == 1

    def test_counts_agreeing_voters(self):
        """Test that agreeing voters are counted."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3", "voter-4"]
        election.leader_id = "target-leader"

        # Add 2 peers that agree
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="target-leader",
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="target-leader",
        )
        # One peer disagrees
        election.peers["voter-4"] = MockNodeInfo(
            node_id="voter-4",
            last_heartbeat=time.time(),
            leader_id="other-leader",
        )

        # Self + 2 agreeing peers = 3 votes
        assert election._count_votes_for_leader("target-leader") == 3


class TestDetectSplitBrain:
    """Test _detect_split_brain method."""

    def test_returns_none_when_no_voters(self):
        """Test that None is returned when no voters configured."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        assert election._detect_split_brain() is None

    def test_returns_none_when_no_split_brain(self):
        """Test that None is returned when all voters agree."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_id = "the-leader"

        # All peers agree on the same leader
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="the-leader",
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="the-leader",
        )

        assert election._detect_split_brain() is None

    def test_detects_split_brain(self):
        """Test that split-brain is detected when voters disagree."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_id = "leader-A"

        # Peers disagree on leader
        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="leader-B",  # Different leader
        )
        election.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            last_heartbeat=time.time(),
            leader_id="leader-A",
        )

        result = election._detect_split_brain()

        assert result is not None
        assert "leader-A" in result["leaders_seen"]
        assert "leader-B" in result["leaders_seen"]

    def test_emits_event_on_split_brain(self):
        """Test that SPLIT_BRAIN_DETECTED event is emitted."""
        election = TestableLeaderElection()
        election.node_id = "voter-1"
        election.voter_node_ids = ["voter-1", "voter-2"]
        election.leader_id = "leader-A"

        election.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            last_heartbeat=time.time(),
            leader_id="leader-B",
        )

        election._detect_split_brain()

        assert any(e[0] == "SPLIT_BRAIN_DETECTED" for e in election._events_emitted)


class TestResolveSplitBrain:
    """Test _resolve_split_brain method."""

    def test_does_nothing_if_not_leader(self):
        """Test that nothing happens if node is not a leader."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.FOLLOWER

        leaders_seen = {"leader-A": ["voter-1"], "leader-B": ["voter-2"]}

        election._resolve_split_brain(leaders_seen)

        # Should still be follower
        assert election.role == MockNodeRole.FOLLOWER

    def test_steps_down_if_not_canonical_leader(self):
        """Test that node steps down if not the canonical leader."""
        election = TestableLeaderElection()
        election.node_id = "leader-B"
        election.role = MockNodeRole.LEADER
        election.leader_id = election.node_id
        election.leader_lease_id = "some-lease"

        # leader-A has more votes
        leaders_seen = {
            "leader-A": ["voter-1", "voter-2", "voter-3"],  # 3 votes
            "leader-B": ["voter-4"],  # 1 vote
        }

        election._resolve_split_brain(leaders_seen)

        assert election.role == MockNodeRole.FOLLOWER
        assert election.leader_id == "leader-A"
        assert election.leader_lease_id == ""

    def test_stays_leader_if_canonical(self):
        """Test that node stays leader if it is the canonical leader."""
        election = TestableLeaderElection()
        election.node_id = "leader-A"
        election.role = MockNodeRole.LEADER
        election.leader_id = election.node_id

        # leader-A has more votes
        leaders_seen = {
            "leader-A": ["voter-1", "voter-2", "voter-3"],
            "leader-B": ["voter-4"],
        }

        election._resolve_split_brain(leaders_seen)

        assert election.role == MockNodeRole.LEADER
        assert election.leader_id == election.node_id

    def test_uses_node_id_as_tiebreaker(self):
        """Test that lower node_id wins on tie."""
        election = TestableLeaderElection()
        election.node_id = "leader-B"  # Higher alphabetically
        election.role = MockNodeRole.LEADER

        # Equal votes - lower node_id wins
        leaders_seen = {
            "leader-A": ["voter-1", "voter-2"],  # Same votes, but lower ID
            "leader-B": ["voter-3", "voter-4"],  # Same votes
        }

        election._resolve_split_brain(leaders_seen)

        assert election.role == MockNodeRole.FOLLOWER
        assert election.leader_id == "leader-A"

    def test_emits_event_on_resolution(self):
        """Test that SPLIT_BRAIN_RESOLVED event is emitted."""
        election = TestableLeaderElection()
        election.node_id = "leader-B"
        election.role = MockNodeRole.LEADER

        leaders_seen = {
            "leader-A": ["voter-1", "voter-2", "voter-3"],
            "leader-B": ["voter-4"],
        }

        election._resolve_split_brain(leaders_seen)

        assert any(e[0] == "SPLIT_BRAIN_RESOLVED" for e in election._events_emitted)


class TestElectionHealthCheck:
    """Test election_health_check method."""

    def test_returns_health_status(self):
        """Test that health status is returned."""
        election = TestableLeaderElection()
        election.role = MockNodeRole.LEADER
        election.leader_id = election.node_id
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        election.leader_lease_expires = time.time() + 60

        # Add alive voters
        for i in range(1, 4):
            election.peers[f"voter-{i}"] = MockNodeInfo(
                node_id=f"voter-{i}",
                last_heartbeat=time.time(),
            )

        health = election.election_health_check()

        assert "is_healthy" in health
        assert "role" in health
        assert "leader_id" in health
        assert "has_quorum" in health
        assert health["voter_count"] == 3
        assert health["alive_voters"] >= 0
        assert "lease_remaining_seconds" in health

    def test_healthy_when_quorum_met(self):
        """Test that node is healthy when quorum is met."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]

        for i in range(1, 4):
            election.peers[f"voter-{i}"] = MockNodeInfo(
                node_id=f"voter-{i}",
                last_heartbeat=time.time(),
            )

        health = election.election_health_check()

        assert health["is_healthy"] is True
        assert health["has_quorum"] is True

    def test_unhealthy_when_no_quorum(self):
        """Test that node is unhealthy when quorum is not met."""
        election = TestableLeaderElection()
        election.voter_node_ids = ["voter-1", "voter-2", "voter-3"]
        # No alive peers

        health = election.election_health_check()

        assert health["is_healthy"] is False
        assert health["has_quorum"] is False

    def test_healthy_with_no_voters(self):
        """Test that node is healthy when no voters configured."""
        election = TestableLeaderElection()
        election.voter_node_ids = []

        health = election.election_health_check()

        assert health["is_healthy"] is True


class TestCheckQuorumFunction:
    """Test the standalone check_quorum function."""

    def test_returns_true_when_no_voters(self):
        """Test that quorum is met when no voters configured."""
        result = check_quorum(
            voters=[],
            alive_peers={},
            self_node_id="self",
        )

        assert result is True

    def test_returns_true_when_quorum_met(self):
        """Test that quorum is met with enough alive voters."""
        voters = ["voter-1", "voter-2", "voter-3", "voter-4"]
        alive_peers = {
            "voter-2": MockNodeInfo(node_id="voter-2", last_heartbeat=time.time()),
            "voter-3": MockNodeInfo(node_id="voter-3", last_heartbeat=time.time()),
            "voter-4": MockNodeInfo(node_id="voter-4", last_heartbeat=time.time()),
        }

        result = check_quorum(
            voters=voters,
            alive_peers=alive_peers,
            self_node_id="voter-1",  # Self counts too
        )

        assert result is True

    def test_returns_false_when_quorum_not_met(self):
        """Test that quorum fails with too few voters."""
        voters = ["voter-1", "voter-2", "voter-3", "voter-4"]
        alive_peers = {
            "voter-2": MockNodeInfo(node_id="voter-2", last_heartbeat=time.time()),
        }

        result = check_quorum(
            voters=voters,
            alive_peers=alive_peers,
            self_node_id="self",  # Not a voter
        )

        assert result is False

    def test_counts_self_if_in_voters(self):
        """Test that self is counted if in voter list."""
        voters = ["voter-1", "voter-2", "voter-3"]
        alive_peers = {
            "voter-2": MockNodeInfo(node_id="voter-2", last_heartbeat=time.time()),
            "voter-3": MockNodeInfo(node_id="voter-3", last_heartbeat=time.time()),
        }

        result = check_quorum(
            voters=voters,
            alive_peers=alive_peers,
            self_node_id="voter-1",  # Self is a voter
        )

        # Self + 2 peers = 3 = quorum met
        assert result is True

    def test_uses_min_quorum(self):
        """Test that min(VOTER_MIN_QUORUM, len(voters)) is used."""
        voters = ["voter-1", "voter-2"]  # Only 2 voters
        alive_peers = {
            "voter-1": MockNodeInfo(node_id="voter-1", last_heartbeat=time.time()),
            "voter-2": MockNodeInfo(node_id="voter-2", last_heartbeat=time.time()),
        }

        result = check_quorum(
            voters=voters,
            alive_peers=alive_peers,
            self_node_id="self",
        )

        # min(VOTER_MIN_QUORUM, 2) = 2, and we have 2 alive = quorum met
        assert result is True
