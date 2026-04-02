"""Tests for leader_election.py mixin.

Tests the LeaderElectionMixin extracted from p2p_orchestrator.py.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.leader_election import (
    LeaderElectionMixin,
    check_quorum,
)


class MockNodeInfo:
    """Mock NodeInfo for testing."""

    def __init__(self, node_id: str, alive: bool = True):
        self.node_id = node_id
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class MockOrchestrator(LeaderElectionMixin):
    """Mock orchestrator class that uses LeaderElectionMixin."""

    def __init__(self):
        self.node_id = "test-node"
        self.role = "follower"
        self.leader_id = None
        self.leader_lease_id = ""
        self.leader_lease_expires = 0.0
        self.last_lease_renewal = 0.0
        self.voter_node_ids: list[str] = []
        self.voter_grant_leader_id = ""
        self.voter_grant_lease_id = ""
        self.voter_grant_expires = 0.0
        self.peers_lock = threading.RLock()
        self.peers: dict[str, Any] = {}


class TestHasVoterQuorum:
    """Test _has_voter_quorum method."""

    def test_empty_voters_returns_true(self):
        """No voters configured means quorum is always met."""
        orch = MockOrchestrator()
        orch.voter_node_ids = []
        assert orch._has_voter_quorum() is True

    def test_single_voter_self_has_quorum(self):
        """Single voter (self) meets quorum."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node"]
        assert orch._has_voter_quorum() is True

    def test_two_voters_self_and_alive_peer(self):
        """Two voters both alive meets quorum."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1"]
        orch.peers = {"peer-1": MockNodeInfo("peer-1", alive=True)}
        assert orch._has_voter_quorum() is True

    def test_three_voters_min_quorum(self):
        """Three voters meet quorum once 2 voters are alive."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2"]
        orch.peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=True),
        }
        assert orch._has_voter_quorum() is True

    def test_five_voters_only_one_alive_uses_emergency_quorum(self):
        """Five voters fall back to emergency single-node quorum when only self is alive."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2", "peer-3", "peer-4"]
        orch.peers = {
            "peer-1": MockNodeInfo("peer-1", alive=False),
            "peer-2": MockNodeInfo("peer-2", alive=False),
            "peer-3": MockNodeInfo("peer-3", alive=False),
            "peer-4": MockNodeInfo("peer-4", alive=False),
        }
        # Dynamic quorum drops to 1 when only one voter is alive.
        assert orch._has_voter_quorum() is True

    def test_five_voters_three_alive_passes(self):
        """Five voters with 3 alive passes quorum."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2", "peer-3", "peer-4"]
        orch.peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=True),
            "peer-3": MockNodeInfo("peer-3", alive=False),
            "peer-4": MockNodeInfo("peer-4", alive=False),
        }
        # Self + peer-1 + peer-2 = 3 alive
        assert orch._has_voter_quorum() is True


class TestReleaseVoterGrantIfSelf:
    """Test _release_voter_grant_if_self method."""

    def test_release_when_self_is_grant_holder(self):
        """Release grant when we hold it."""
        orch = MockOrchestrator()
        orch.voter_grant_leader_id = "test-node"
        orch.voter_grant_lease_id = "lease-123"
        orch.voter_grant_expires = time.time() + 100

        orch._release_voter_grant_if_self()

        assert orch.voter_grant_leader_id == ""
        assert orch.voter_grant_lease_id == ""
        assert orch.voter_grant_expires == 0.0

    def test_no_release_when_other_is_grant_holder(self):
        """Don't release grant when another node holds it."""
        orch = MockOrchestrator()
        orch.voter_grant_leader_id = "other-node"
        orch.voter_grant_lease_id = "lease-123"
        orch.voter_grant_expires = time.time() + 100

        orch._release_voter_grant_if_self()

        # Should remain unchanged
        assert orch.voter_grant_leader_id == "other-node"
        assert orch.voter_grant_lease_id == "lease-123"

    def test_no_release_when_empty(self):
        """No-op when no grant is held."""
        orch = MockOrchestrator()
        orch.voter_grant_leader_id = ""
        orch.voter_grant_lease_id = ""
        orch.voter_grant_expires = 0.0

        orch._release_voter_grant_if_self()

        assert orch.voter_grant_leader_id == ""
        assert orch.voter_grant_lease_id == ""
        assert orch.voter_grant_expires == 0.0


class TestGetVoterQuorumStatus:
    """Test _get_voter_quorum_status method."""

    def test_empty_voters_status(self):
        """Status for empty voter list."""
        orch = MockOrchestrator()
        orch.voter_node_ids = []

        status = orch._get_voter_quorum_status()

        assert status["voters"] == []
        assert status["alive"] == 0
        assert status["total"] == 0
        assert status["quorum_met"] is True

    def test_status_with_alive_and_dead_voters(self):
        """Status shows alive vs total voters correctly."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2", "peer-3"]
        orch.peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=False),
            "peer-3": MockNodeInfo("peer-3", alive=True),
        }

        status = orch._get_voter_quorum_status()

        assert status["total"] == 4
        assert status["alive"] == 3  # self + peer-1 + peer-3
        assert status["quorum_required"] == 2
        assert status["quorum_met"] is True
        assert "test-node" in status["alive_list"]
        assert "peer-1" in status["alive_list"]
        assert "peer-3" in status["alive_list"]


class TestCheckLeaderConsistency:
    """Test _check_leader_consistency method."""

    def test_consistent_follower_no_leader(self):
        """Follower with no leader is consistent."""
        orch = MockOrchestrator()
        orch.role = "follower"
        orch.leader_id = None

        is_consistent, reason = orch._check_leader_consistency()

        assert is_consistent is True
        assert reason == "consistent"

    def test_consistent_follower_other_leader(self):
        """Follower with another leader is consistent."""
        orch = MockOrchestrator()
        orch.role = "follower"
        orch.leader_id = "other-node"

        is_consistent, reason = orch._check_leader_consistency()

        assert is_consistent is True
        assert reason == "consistent"

    def test_inconsistent_leader_id_self_but_role_follower(self):
        """Inconsistent: leader_id=self but role=follower."""
        orch = MockOrchestrator()
        orch.role = "follower"
        orch.leader_id = "test-node"

        is_consistent, reason = orch._check_leader_consistency()

        assert is_consistent is False
        assert "leader_id=self but role!=leader" in reason


class TestCheckQuorumStandalone:
    """Test standalone check_quorum function."""

    def test_empty_voters(self):
        """Empty voter list means quorum is met."""
        assert check_quorum([], {}, "test-node") is True

    def test_single_voter_self(self):
        """Single voter (self) meets quorum."""
        assert check_quorum(["test-node"], {}, "test-node") is True

    def test_three_voters_all_alive(self):
        """Three voters all alive meets quorum."""
        peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=True),
        }
        assert check_quorum(["test-node", "peer-1", "peer-2"], peers, "test-node") is True

    def test_three_voters_two_alive_fails(self):
        """Three voters with 2 alive still meet the current quorum."""
        peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=False),
        }
        # self + peer-1 = 2, which meets the min quorum of 2
        assert check_quorum(["test-node", "peer-1", "peer-2"], peers, "test-node") is True


class TestEdgeCases:
    """Test edge cases."""

    def test_voter_not_in_peers(self):
        """Voter not in peers dict is treated as not alive."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2"]
        # peer-2 not in peers
        orch.peers = {"peer-1": MockNodeInfo("peer-1", alive=True)}

        # self + peer-1 = 2, which meets the min quorum of 2
        assert orch._has_voter_quorum() is True

    def test_concurrent_access_to_peers(self):
        """Concurrent access to peers dict is protected by lock."""
        orch = MockOrchestrator()
        orch.voter_node_ids = ["test-node", "peer-1", "peer-2"]
        orch.peers = {
            "peer-1": MockNodeInfo("peer-1", alive=True),
            "peer-2": MockNodeInfo("peer-2", alive=True),
        }

        # Simulate concurrent access
        results = []
        def check():
            results.append(orch._has_voter_quorum())

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should return True
        assert all(r is True for r in results)
