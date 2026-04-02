"""Tests for P2P script mixins.

Comprehensive unit tests for:
- LeaderElectionMixin (leader_election.py)
- GossipProtocolMixin (gossip_protocol.py)
- PeerManagerMixin (peer_manager.py)
- ConsensusMixin (consensus_mixin.py)
- MembershipMixin (membership_mixin.py)

December 28, 2025: Created for P2P mixin test coverage.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mixin classes
from scripts.p2p.leader_election import (
    LeaderElectionMixin,
    check_quorum,
    VOTER_MIN_QUORUM,
)
from scripts.p2p.peer_manager import (
    PeerManagerMixin,
    PEER_CACHE_MAX_ENTRIES,
    PEER_CACHE_TTL_SECONDS,
    PEER_REPUTATION_ALPHA,
)
from scripts.p2p.gossip_protocol import (
    GossipProtocolMixin,
    calculate_compression_ratio,
)
from scripts.p2p.consensus_mixin import (
    ConsensusMixin,
    RAFT_ENABLED,
    CONSENSUS_MODE,
    PYSYNCOBJ_AVAILABLE,
)
from scripts.p2p.membership_mixin import (
    MembershipMixin,
    SWIM_ENABLED,
    MEMBERSHIP_MODE,
)


# =============================================================================
# Test Fixtures and Mock Classes
# =============================================================================


class MockPeer:
    """Mock peer for testing."""

    def __init__(
        self,
        node_id: str = "peer1",
        alive: bool = True,
        leader_id: str | None = None,
        endpoint: str | None = None,
    ):
        self.node_id = node_id
        self._alive = alive
        self.leader_id = leader_id
        self.endpoint = endpoint or f"{node_id}:8770"
        self.host = node_id
        self.port = 8770
        self.tailscale_ip = ""
        self.last_heartbeat = time.time() if alive else 0
        self.consecutive_failures = 0 if alive else 5
        self.last_failure_time = 0.0
        self.retired = False

    def is_alive(self) -> bool:
        return self._alive


class MockNodeRole:
    """Mock NodeRole enum that matches the actual implementation."""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "value"):
            return self.value == other.value
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.value


# Create instances that match the actual NodeRole enum
class NodeRoleEnum:
    LEADER = MockNodeRole("leader")
    FOLLOWER = MockNodeRole("follower")


# =============================================================================
# LeaderElectionMixin Tests
# =============================================================================


class TestLeaderElectionMixinQuorum:
    """Test voter quorum logic in LeaderElectionMixin."""

    def _create_mixin(
        self,
        node_id: str = "self",
        voters: list[str] | None = None,
        peers: dict[str, MockPeer] | None = None,
    ) -> LeaderElectionMixin:
        """Create a LeaderElectionMixin instance for testing."""

        class TestMixin(LeaderElectionMixin):
            MIXIN_TYPE = "test_leader"

        mixin = TestMixin()
        mixin.node_id = node_id
        mixin.voter_node_ids = voters or []
        mixin.peers = peers or {}
        mixin.peers_lock = threading.RLock()
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.leader_id = None
        mixin.leader_lease_id = ""
        mixin.leader_lease_expires = 0.0
        mixin.last_lease_renewal = 0.0
        mixin.voter_grant_leader_id = ""
        mixin.voter_grant_lease_id = ""
        mixin.voter_grant_expires = 0.0
        return mixin

    def test_has_voter_quorum_no_voters(self) -> None:
        """Test quorum returns True when no voters configured."""
        mixin = self._create_mixin()
        assert mixin._has_voter_quorum() is True

    def test_has_voter_quorum_empty_voters(self) -> None:
        """Test quorum returns True with empty voter list."""
        mixin = self._create_mixin(voters=[])
        assert mixin._has_voter_quorum() is True

    def test_has_voter_quorum_self_only(self) -> None:
        """Test quorum with self as only voter."""
        mixin = self._create_mixin(node_id="self", voters=["self"])
        assert mixin._has_voter_quorum() is True

    def test_has_voter_quorum_three_alive(self) -> None:
        """Test quorum with 3 alive voters (meets minimum quorum)."""
        voters = ["self", "peer1", "peer2"]
        peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }
        mixin = self._create_mixin(node_id="self", voters=voters, peers=peers)
        assert mixin._has_voter_quorum() is True

    def test_has_voter_quorum_two_alive_of_five(self) -> None:
        """Test dynamic quorum accepts 2 alive voters out of 5."""
        voters = ["self", "peer1", "peer2", "peer3", "peer4"]
        peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=False),
            "peer3": MockPeer("peer3", alive=False),
            "peer4": MockPeer("peer4", alive=False),
        }
        mixin = self._create_mixin(node_id="self", voters=voters, peers=peers)
        # Dynamic quorum drops to 2 when only 2 voters are alive.
        assert mixin._has_voter_quorum() is True

    def test_has_voter_quorum_uses_fixed_minimum(self) -> None:
        """Test dynamic quorum stays below majority when only 3 voters are alive."""
        # With 7 configured voters and 3 alive, dynamic quorum is 2 rather than majority-of-total.
        voters = ["self", "peer1", "peer2", "peer3", "peer4", "peer5", "peer6"]
        peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
            "peer3": MockPeer("peer3", alive=False),
            "peer4": MockPeer("peer4", alive=False),
            "peer5": MockPeer("peer5", alive=False),
            "peer6": MockPeer("peer6", alive=False),
        }
        mixin = self._create_mixin(node_id="self", voters=voters, peers=peers)
        # 3 alive (self + peer1 + peer2) exceeds the dynamic quorum requirement.
        assert mixin._has_voter_quorum() is True


class TestLeaderElectionMixinConsistency:
    """Test leader consistency checks."""

    def _create_mixin(self) -> LeaderElectionMixin:
        """Create a LeaderElectionMixin instance for testing."""

        class TestMixin(LeaderElectionMixin):
            MIXIN_TYPE = "test_leader"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.voter_node_ids = []
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.leader_id = None
        mixin.leader_lease_id = ""
        mixin.leader_lease_expires = 0.0
        mixin.last_lease_renewal = 0.0
        mixin.voter_grant_leader_id = ""
        mixin.voter_grant_lease_id = ""
        mixin.voter_grant_expires = 0.0
        return mixin

    def test_check_leader_consistency_as_follower(self) -> None:
        """Test consistency check passes when follower with no leader_id=self."""
        mixin = self._create_mixin()
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.leader_id = "other_node"
        is_consistent, reason = mixin._check_leader_consistency()
        assert is_consistent is True
        assert reason == "consistent"

    def test_check_leader_consistency_as_leader(self) -> None:
        """Test consistency check passes when leader with leader_id=self."""
        mixin = self._create_mixin()
        mixin.role = NodeRoleEnum.LEADER
        mixin.leader_id = "self"
        is_consistent, reason = mixin._check_leader_consistency()
        assert is_consistent is True
        assert reason == "consistent"

    def test_check_leader_consistency_inconsistent_leader_id(self) -> None:
        """Test inconsistency when leader_id=self but role!=leader."""
        mixin = self._create_mixin()
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.leader_id = "self"
        is_consistent, reason = mixin._check_leader_consistency()
        assert is_consistent is False
        assert "leader_id=self but role!=leader" in reason

    def test_check_leader_consistency_inconsistent_role(self) -> None:
        """Test inconsistency when role=leader but leader_id!=self."""
        mixin = self._create_mixin()
        mixin.role = NodeRoleEnum.LEADER
        mixin.leader_id = "other_node"
        is_consistent, reason = mixin._check_leader_consistency()
        assert is_consistent is False
        assert "role=leader but leader_id!=self" in reason


class TestLeaderElectionMixinSplitBrain:
    """Test split-brain detection and resolution."""

    def _create_mixin(
        self,
        peers: dict[str, MockPeer] | None = None,
        voters: list[str] | None = None,
    ) -> LeaderElectionMixin:
        """Create a LeaderElectionMixin instance for testing."""

        class TestMixin(LeaderElectionMixin):
            MIXIN_TYPE = "test_leader"

            def _save_state(self) -> None:
                pass

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.voter_node_ids = voters or []
        mixin.peers = peers or {}
        mixin.peers_lock = threading.RLock()
        mixin.role = NodeRoleEnum.LEADER
        mixin.leader_id = "self"
        mixin.leader_lease_id = ""
        mixin.leader_lease_expires = 0.0
        mixin.last_lease_renewal = 0.0
        mixin.voter_grant_leader_id = ""
        mixin.voter_grant_lease_id = ""
        mixin.voter_grant_expires = 0.0
        return mixin

    def test_detect_split_brain_no_voters(self) -> None:
        """Test no split-brain detected with no voters."""
        mixin = self._create_mixin()
        result = mixin._detect_split_brain()
        assert result is None

    def test_detect_split_brain_single_leader(self) -> None:
        """Test no split-brain when all agree on same leader."""
        voters = ["self", "peer1", "peer2"]
        peers = {
            "peer1": MockPeer("peer1", alive=True, leader_id="self"),
            "peer2": MockPeer("peer2", alive=True, leader_id="self"),
        }
        mixin = self._create_mixin(peers=peers, voters=voters)
        mixin.leader_id = "self"
        result = mixin._detect_split_brain()
        assert result is None

    def test_detect_split_brain_multiple_leaders(self) -> None:
        """Test split-brain detected when voters disagree on leader."""
        voters = ["self", "peer1", "peer2", "peer3"]
        peers = {
            "peer1": MockPeer("peer1", alive=True, leader_id="self"),
            "peer2": MockPeer("peer2", alive=True, leader_id="other_leader"),
            "peer3": MockPeer("peer3", alive=True, leader_id="other_leader"),
        }
        mixin = self._create_mixin(peers=peers, voters=voters)
        mixin.leader_id = "self"
        result = mixin._detect_split_brain()
        assert result is not None
        assert len(result["leaders_seen"]) == 2
        assert "self" in result["leaders_seen"]
        assert "other_leader" in result["leaders_seen"]

    def test_count_votes_for_leader(self) -> None:
        """Test _count_votes_for_leader counts correctly."""
        voters = ["self", "peer1", "peer2", "peer3"]
        peers = {
            "peer1": MockPeer("peer1", alive=True, leader_id="leader1"),
            "peer2": MockPeer("peer2", alive=True, leader_id="leader1"),
            "peer3": MockPeer("peer3", alive=True, leader_id="leader2"),
        }
        mixin = self._create_mixin(peers=peers, voters=voters)
        mixin.leader_id = "leader1"

        votes = mixin._count_votes_for_leader("leader1")
        assert votes == 3  # self + peer1 + peer2

        votes = mixin._count_votes_for_leader("leader2")
        assert votes == 1  # peer3 only


class TestLeaderElectionMixinHealthCheck:
    """Test health check methods."""

    def _create_mixin(self) -> LeaderElectionMixin:
        """Create a LeaderElectionMixin instance for testing."""

        class TestMixin(LeaderElectionMixin):
            MIXIN_TYPE = "test_leader"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.voter_node_ids = ["self", "peer1", "peer2"]
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }
        mixin.peers_lock = threading.RLock()
        mixin.role = NodeRoleEnum.LEADER
        mixin.leader_id = "self"
        mixin.leader_lease_id = "lease123"
        mixin.leader_lease_expires = time.time() + 30
        return mixin

    def test_election_health_check(self) -> None:
        """Test election_health_check returns correct structure."""
        mixin = self._create_mixin()
        health = mixin.election_health_check()

        assert "is_healthy" in health
        assert "role" in health
        assert "leader_id" in health
        assert "has_quorum" in health
        assert "voter_count" in health
        assert "alive_voters" in health
        assert "lease_remaining_seconds" in health

    def test_election_health_check_healthy(self) -> None:
        """Test election_health_check reports healthy with quorum."""
        mixin = self._create_mixin()
        health = mixin.election_health_check()

        assert health["is_healthy"] is True
        assert health["has_quorum"] is True
        assert health["voter_count"] == 3
        assert health["alive_voters"] == 3

    def test_health_check_wrapper(self) -> None:
        """Test health_check returns standard format."""
        mixin = self._create_mixin()
        health = mixin.health_check()

        assert "healthy" in health
        assert "message" in health
        assert "details" in health
        assert health["healthy"] is True


class TestStandaloneQuorumCheck:
    """Test standalone check_quorum function."""

    def test_check_quorum_no_voters(self) -> None:
        """Test check_quorum returns True with no voters."""
        assert check_quorum([], {}, "self") is True

    def test_check_quorum_self_counts(self) -> None:
        """Test check_quorum counts self as alive."""
        voters = ["self"]
        assert check_quorum(voters, {}, "self") is True

    def test_check_quorum_enough_alive(self) -> None:
        """Test check_quorum with enough alive peers."""
        voters = ["self", "peer1", "peer2"]
        alive_peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }
        assert check_quorum(voters, alive_peers, "self") is True

    def test_check_quorum_not_enough_alive(self) -> None:
        """Test check_quorum with insufficient alive peers."""
        voters = ["self", "peer1", "peer2", "peer3", "peer4"]
        alive_peers = {
            "peer1": MockPeer("peer1", alive=False),
            "peer2": MockPeer("peer2", alive=False),
            "peer3": MockPeer("peer3", alive=False),
            "peer4": MockPeer("peer4", alive=False),
        }
        # Only self alive = 1 < 3 required
        assert check_quorum(voters, alive_peers, "self") is False


# =============================================================================
# PeerManagerMixin Tests
# =============================================================================


class TestPeerManagerMixinReputation:
    """Test peer reputation management."""

    def _create_mixin(self, tmp_path: Path) -> PeerManagerMixin:
        """Create a PeerManagerMixin instance with database."""
        db_path = tmp_path / "p2p.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                tailscale_ip TEXT,
                reputation_score REAL DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_seen REAL,
                is_bootstrap_seed INTEGER DEFAULT 0
            )
        """
        )
        conn.commit()
        conn.close()

        class TestMixin(PeerManagerMixin):
            MIXIN_TYPE = "test_peer"

        mixin = TestMixin()
        mixin.db_path = db_path
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.bootstrap_seeds = []
        mixin.verbose = True
        return mixin

    def test_update_peer_reputation_success(self, tmp_path: Path) -> None:
        """Test reputation increases on success."""
        mixin = self._create_mixin(tmp_path)

        # First interaction - success
        mixin._update_peer_reputation("peer1", success=True)

        # Check result
        result = mixin._execute_db_query(
            "SELECT reputation_score, success_count FROM peer_cache WHERE node_id = ?",
            ("peer1",),
            fetch=True,
        )
        assert result is not None
        assert len(result) == 1
        # Score should be higher than initial 0.5
        score = result[0][0]
        # EMA with alpha=0.2: 0.2 * 1.0 + 0.8 * 0.5 = 0.6
        assert score == pytest.approx(0.6, rel=0.1)
        assert result[0][1] == 1  # success_count

    def test_update_peer_reputation_failure(self, tmp_path: Path) -> None:
        """Test reputation decreases on failure."""
        mixin = self._create_mixin(tmp_path)

        # First interaction - failure
        mixin._update_peer_reputation("peer1", success=False)

        result = mixin._execute_db_query(
            "SELECT reputation_score, failure_count FROM peer_cache WHERE node_id = ?",
            ("peer1",),
            fetch=True,
        )
        assert result is not None
        assert len(result) == 1
        # EMA with alpha=0.2: 0.2 * 0.0 + 0.8 * 0.5 = 0.4
        score = result[0][0]
        assert score == pytest.approx(0.4, rel=0.1)
        assert result[0][1] == 1  # failure_count


class TestPeerManagerMixinCache:
    """Test peer cache management."""

    def _create_mixin(self, tmp_path: Path) -> PeerManagerMixin:
        """Create a PeerManagerMixin instance with database."""
        db_path = tmp_path / "p2p.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                tailscale_ip TEXT,
                reputation_score REAL DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_seen REAL,
                is_bootstrap_seed INTEGER DEFAULT 0
            )
        """
        )
        conn.commit()
        conn.close()

        class TestMixin(PeerManagerMixin):
            MIXIN_TYPE = "test_peer"

        mixin = TestMixin()
        mixin.db_path = db_path
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.bootstrap_seeds = ["seed1.example.com:8770"]
        mixin.verbose = True
        return mixin

    def test_save_peer_to_cache(self, tmp_path: Path) -> None:
        """Test saving peer to cache."""
        mixin = self._create_mixin(tmp_path)
        mixin._save_peer_to_cache("peer1", "192.168.1.1", 8770, "100.1.2.3")

        result = mixin._execute_db_query(
            "SELECT host, port, tailscale_ip FROM peer_cache WHERE node_id = ?",
            ("peer1",),
            fetch=True,
        )
        assert result is not None
        assert result[0] == ("192.168.1.1", 8770, "100.1.2.3")

    def test_save_peer_to_cache_skips_self(self, tmp_path: Path) -> None:
        """Test saving self is skipped."""
        mixin = self._create_mixin(tmp_path)
        mixin._save_peer_to_cache("self", "127.0.0.1", 8770)

        result = mixin._execute_db_query(
            "SELECT * FROM peer_cache WHERE node_id = ?",
            ("self",),
            fetch=True,
        )
        assert result == []

    def test_get_cached_peer_count(self, tmp_path: Path) -> None:
        """Test getting cached peer count."""
        mixin = self._create_mixin(tmp_path)
        mixin._save_peer_to_cache("peer1", "192.168.1.1", 8770)
        mixin._save_peer_to_cache("peer2", "192.168.1.2", 8770)

        count = mixin._get_cached_peer_count()
        assert count == 2

    def test_clear_peer_cache(self, tmp_path: Path) -> None:
        """Test clearing non-seed peers from cache."""
        mixin = self._create_mixin(tmp_path)
        mixin._save_peer_to_cache("peer1", "192.168.1.1", 8770)
        mixin._save_peer_to_cache("peer2", "192.168.1.2", 8770)

        cleared = mixin._clear_peer_cache()
        assert cleared >= 0

        count = mixin._get_cached_peer_count()
        assert count == 0


class TestPeerManagerMixinHealthScore:
    """Test peer health scoring."""

    def _create_mixin(self, tmp_path: Path) -> PeerManagerMixin:
        """Create a PeerManagerMixin instance."""
        db_path = tmp_path / "p2p.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                tailscale_ip TEXT,
                reputation_score REAL DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_seen REAL,
                is_bootstrap_seed INTEGER DEFAULT 0
            )
        """
        )
        conn.commit()
        conn.close()

        class TestMixin(PeerManagerMixin):
            MIXIN_TYPE = "test_peer"

        mixin = TestMixin()
        mixin.db_path = db_path
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.bootstrap_seeds = []
        mixin.verbose = True
        mixin._peer_reputations = {}
        mixin._p2p_sync_results = {}
        return mixin

    def test_get_peer_health_score_default(self, tmp_path: Path) -> None:
        """Test default health score for unknown peer."""
        mixin = self._create_mixin(tmp_path)
        score = mixin._get_peer_health_score("unknown_peer")
        # 0.7 * 0.5 (default reputation) + 0.3 * 0.5 (neutral sync) = 0.5
        assert score == pytest.approx(0.5, rel=0.1)

    def test_get_peer_health_score_with_history(self, tmp_path: Path) -> None:
        """Test health score with sync history."""
        mixin = self._create_mixin(tmp_path)
        mixin._peer_reputations["peer1"] = 0.8
        mixin._p2p_sync_results["peer1"] = [True, True, True, False]

        score = mixin._get_peer_health_score("peer1")
        # 0.7 * 0.8 + 0.3 * 0.75 = 0.56 + 0.225 = 0.785
        assert score == pytest.approx(0.785, rel=0.1)

    def test_record_p2p_sync_result(self, tmp_path: Path) -> None:
        """Test recording sync results."""
        mixin = self._create_mixin(tmp_path)
        mixin._record_p2p_sync_result("peer1", True)
        mixin._record_p2p_sync_result("peer1", True)
        mixin._record_p2p_sync_result("peer1", False)

        assert len(mixin._p2p_sync_results["peer1"]) == 3
        assert mixin._p2p_sync_results["peer1"] == [True, True, False]


class TestPeerManagerMixinHealthCheck:
    """Test health check for peer manager."""

    def _create_mixin(self, tmp_path: Path) -> PeerManagerMixin:
        """Create a PeerManagerMixin instance."""
        db_path = tmp_path / "p2p.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT, port INTEGER, tailscale_ip TEXT,
                reputation_score REAL, success_count INTEGER,
                failure_count INTEGER, last_seen REAL, is_bootstrap_seed INTEGER
            )
        """
        )
        conn.commit()
        conn.close()

        class TestMixin(PeerManagerMixin):
            MIXIN_TYPE = "test_peer"

        mixin = TestMixin()
        mixin.db_path = db_path
        mixin.node_id = "self"
        mixin.peers = {"peer1": MockPeer("peer1")}
        mixin.peers_lock = threading.RLock()
        mixin.bootstrap_seeds = []
        mixin.verbose = False
        mixin._nat_blocked_peers = set()
        return mixin

    def test_peer_health_check(self, tmp_path: Path) -> None:
        """Test peer_health_check returns correct structure."""
        mixin = self._create_mixin(tmp_path)
        health = mixin.peer_health_check()

        assert "is_healthy" in health
        assert "cached_peers" in health
        assert "active_peers" in health
        assert "bootstrap_seeds" in health

    def test_health_check_wrapper(self, tmp_path: Path) -> None:
        """Test health_check returns standard format."""
        mixin = self._create_mixin(tmp_path)
        health = mixin.health_check()

        assert "healthy" in health
        assert "message" in health
        assert "details" in health


# =============================================================================
# GossipProtocolMixin Tests
# =============================================================================


class TestGossipProtocolMixinInit:
    """Test gossip protocol initialization."""

    def _create_mixin(self) -> GossipProtocolMixin:
        """Create a GossipProtocolMixin instance."""

        class TestMixin(GossipProtocolMixin):
            MIXIN_TYPE = "test_gossip"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.verbose = True
        return mixin

    def test_init_gossip_protocol(self) -> None:
        """Test _init_gossip_protocol initializes state."""
        mixin = self._create_mixin()
        mixin._init_gossip_protocol()

        assert hasattr(mixin, "_gossip_peer_states")
        assert hasattr(mixin, "_gossip_peer_manifests")
        assert hasattr(mixin, "_gossip_learned_endpoints")
        assert hasattr(mixin, "_gossip_metrics")
        assert hasattr(mixin, "_gossip_compression_stats")

    def test_init_gossip_protocol_idempotent(self) -> None:
        """Test _init_gossip_protocol is idempotent."""
        mixin = self._create_mixin()
        mixin._init_gossip_protocol()
        mixin._gossip_peer_states["peer1"] = {"data": "test"}
        mixin._init_gossip_protocol()
        # Should not reset existing data
        assert "peer1" in mixin._gossip_peer_states


class TestGossipProtocolMixinMetrics:
    """Test gossip metrics tracking."""

    def _create_mixin(self) -> GossipProtocolMixin:
        """Create a GossipProtocolMixin instance."""

        class TestMixin(GossipProtocolMixin):
            MIXIN_TYPE = "test_gossip"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.verbose = True
        mixin._init_gossip_protocol()
        return mixin

    def test_record_gossip_metrics_sent(self) -> None:
        """Test recording sent message metrics."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("sent", "peer1")

        assert mixin._gossip_metrics["message_sent"] == 1

    def test_record_gossip_metrics_received(self) -> None:
        """Test recording received message metrics."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("received", "peer1")

        assert mixin._gossip_metrics["message_received"] == 1

    def test_record_gossip_metrics_update(self) -> None:
        """Test recording state update metrics."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("update", "peer1")

        assert mixin._gossip_metrics["state_updates"] == 1

    def test_record_gossip_metrics_latency(self) -> None:
        """Test recording latency metrics."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("latency", "peer1", 50.5)
        mixin._record_gossip_metrics("latency", "peer1", 100.5)

        assert len(mixin._gossip_metrics["propagation_delay_ms"]) == 2
        assert 50.5 in mixin._gossip_metrics["propagation_delay_ms"]
        assert 100.5 in mixin._gossip_metrics["propagation_delay_ms"]

    def test_record_gossip_compression(self) -> None:
        """Test recording compression metrics."""
        mixin = self._create_mixin()
        mixin._record_gossip_compression(1000, 300)

        stats = mixin._gossip_compression_stats
        assert stats["total_original_bytes"] == 1000
        assert stats["total_compressed_bytes"] == 300
        assert stats["messages_compressed"] == 1

    def test_get_gossip_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("sent", "peer1")
        mixin._record_gossip_metrics("received", "peer2")
        mixin._record_gossip_metrics("latency", "peer1", 50.0)
        mixin._record_gossip_compression(1000, 300)

        summary = mixin._get_gossip_metrics_summary()

        assert summary["message_sent"] == 1
        assert summary["message_received"] == 1
        assert summary["avg_latency_ms"] == pytest.approx(50.0)
        assert summary["compression_ratio"] == pytest.approx(0.7, rel=0.01)


class TestGossipProtocolMixinHealthStatus:
    """Test gossip health status methods."""

    def _create_mixin(self) -> GossipProtocolMixin:
        """Create a GossipProtocolMixin instance."""

        class TestMixin(GossipProtocolMixin):
            MIXIN_TYPE = "test_gossip"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.verbose = True
        mixin._init_gossip_protocol()
        return mixin

    def test_get_gossip_health_status_healthy(self) -> None:
        """Test health status when gossip is healthy."""
        mixin = self._create_mixin()
        # Record some activity
        for _ in range(15):
            mixin._record_gossip_metrics("sent", "peer1")
            mixin._record_gossip_metrics("latency", "peer1", 50.0)

        health = mixin._get_gossip_health_status()

        assert health["is_healthy"] is True
        assert len(health["warnings"]) == 0

    def test_get_gossip_health_status_high_latency(self) -> None:
        """Test health status with high latency."""
        mixin = self._create_mixin()
        for _ in range(10):
            mixin._record_gossip_metrics("latency", "peer1", 1500.0)

        health = mixin._get_gossip_health_status()

        assert health["is_healthy"] is False
        assert any("latency" in w.lower() for w in health["warnings"])

    def test_health_check_wrapper(self) -> None:
        """Test health_check returns standard format."""
        mixin = self._create_mixin()
        mixin._record_gossip_metrics("sent", "peer1")

        health = mixin.health_check()

        assert "healthy" in health
        assert "message" in health
        assert "details" in health


class TestGossipProtocolMixinStateManagement:
    """Test gossip state management."""

    def _create_mixin(self) -> GossipProtocolMixin:
        """Create a GossipProtocolMixin instance."""

        class TestMixin(GossipProtocolMixin):
            MIXIN_TYPE = "test_gossip"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.verbose = True
        mixin._init_gossip_protocol()
        return mixin

    def test_get_gossip_peer_states(self) -> None:
        """Test getting peer states returns shallow copy of dict."""
        mixin = self._create_mixin()
        mixin._gossip_peer_states["peer1"] = {"key": "value"}

        states = mixin.get_gossip_peer_states()

        assert "peer1" in states
        assert states["peer1"]["key"] == "value"
        # The dict itself is a copy (adding new key doesn't affect original)
        states["peer2"] = {"new": "peer"}
        assert "peer2" not in mixin._gossip_peer_states

    def test_get_gossip_learned_endpoints(self) -> None:
        """Test getting learned endpoints returns shallow copy of dict."""
        mixin = self._create_mixin()
        mixin._gossip_learned_endpoints["peer1"] = {"host": "192.168.1.1"}

        endpoints = mixin.get_gossip_learned_endpoints()

        assert "peer1" in endpoints
        assert endpoints["peer1"]["host"] == "192.168.1.1"
        # The dict itself is a copy
        endpoints["peer2"] = {"host": "10.0.0.1"}
        assert "peer2" not in mixin._gossip_learned_endpoints


class TestCalculateCompressionRatio:
    """Test standalone compression ratio function."""

    def test_calculate_compression_ratio_positive(self) -> None:
        """Test compression ratio with positive compression."""
        ratio = calculate_compression_ratio(1000, 300)
        assert ratio == pytest.approx(0.7, rel=0.01)

    def test_calculate_compression_ratio_zero_original(self) -> None:
        """Test compression ratio with zero original size."""
        ratio = calculate_compression_ratio(0, 100)
        assert ratio == 0.0

    def test_calculate_compression_ratio_no_compression(self) -> None:
        """Test compression ratio with no compression."""
        ratio = calculate_compression_ratio(1000, 1000)
        assert ratio == pytest.approx(0.0, rel=0.01)

    def test_calculate_compression_ratio_expansion(self) -> None:
        """Test compression ratio when data expands."""
        ratio = calculate_compression_ratio(100, 150)
        assert ratio < 0  # Negative indicates expansion


# =============================================================================
# ConsensusMixin Tests
# =============================================================================


class TestConsensusMixinInit:
    """Test consensus mixin initialization."""

    def _create_mixin(self) -> ConsensusMixin:
        """Create a ConsensusMixin instance."""

        class TestMixin(ConsensusMixin):
            MIXIN_TYPE = "test_consensus"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.voter_node_ids = []
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.advertise_host = "localhost"
        mixin.advertise_port = 8770
        return mixin

    def test_init_raft_consensus_disabled(self) -> None:
        """Test Raft init returns False when disabled."""
        mixin = self._create_mixin()
        # Raft is disabled by default in production
        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = mixin._init_raft_consensus()
        # Should return False when RAFT_ENABLED is False
        assert result is False

    def test_should_use_raft_bully_mode(self) -> None:
        """Test _should_use_raft returns False in bully mode."""
        mixin = self._create_mixin()
        mixin._raft_initialized = True

        with patch("scripts.p2p.consensus_mixin.CONSENSUS_MODE", "bully"):
            assert mixin._should_use_raft() is False

    def test_get_raft_status(self) -> None:
        """Test get_raft_status returns correct structure."""
        mixin = self._create_mixin()
        mixin._raft_initialized = False
        mixin._raft_init_error = None

        status = mixin.get_raft_status()

        assert "raft_enabled" in status
        assert "raft_available" in status
        assert "raft_initialized" in status
        assert "consensus_mode" in status


class TestConsensusMixinHealthCheck:
    """Test consensus health check."""

    def _create_mixin(self) -> ConsensusMixin:
        """Create a ConsensusMixin instance."""

        class TestMixin(ConsensusMixin):
            MIXIN_TYPE = "test_consensus"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.voter_node_ids = []
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.advertise_host = "localhost"
        mixin.advertise_port = 8770
        mixin._raft_initialized = False
        mixin._raft_init_error = None
        return mixin

    def test_consensus_health_check_raft_disabled(self) -> None:
        """Test health check when Raft is disabled."""
        mixin = self._create_mixin()

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            health = mixin.consensus_health_check()

        assert health["is_healthy"] is True
        assert health["raft_enabled"] is False

    def test_health_check_wrapper(self) -> None:
        """Test health_check returns standard format."""
        mixin = self._create_mixin()

        health = mixin.health_check()

        assert "healthy" in health
        assert "message" in health
        assert "details" in health


# =============================================================================
# MembershipMixin Tests
# =============================================================================


class TestMembershipMixinInit:
    """Test membership mixin initialization."""

    def _create_mixin(self) -> MembershipMixin:
        """Create a MembershipMixin instance."""

        class TestMixin(MembershipMixin):
            MIXIN_TYPE = "test_membership"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        return mixin

    def test_init_swim_membership_disabled(self) -> None:
        """Test SWIM init returns False when disabled."""
        mixin = self._create_mixin()

        with patch("scripts.p2p.membership_mixin.SWIM_ENABLED", False):
            result = mixin._init_swim_membership()
        assert result is False


class TestMembershipMixinPeerAlive:
    """Test hybrid peer alive checking."""

    def _create_mixin(self) -> MembershipMixin:
        """Create a MembershipMixin instance."""

        class TestMixin(MembershipMixin):
            MIXIN_TYPE = "test_membership"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=False),
        }
        mixin.peers_lock = threading.RLock()
        mixin._swim_manager = None
        mixin._swim_started = False
        return mixin

    def test_is_peer_alive_hybrid_http_fallback(self) -> None:
        """Test is_peer_alive_hybrid uses HTTP when SWIM not started."""
        mixin = self._create_mixin()

        assert mixin.is_peer_alive_hybrid("peer1") is True
        assert mixin.is_peer_alive_hybrid("peer2") is False
        assert mixin.is_peer_alive_hybrid("unknown") is False

    def test_get_alive_peers_hybrid_http_fallback(self) -> None:
        """Test get_alive_peers_hybrid uses HTTP when SWIM not started."""
        mixin = self._create_mixin()

        alive = mixin.get_alive_peers_hybrid()

        assert "peer1" in alive
        assert "peer2" not in alive


class TestMembershipMixinEventHandlers:
    """Test membership event handlers."""

    def _create_mixin(self) -> MembershipMixin:
        """Create a MembershipMixin instance."""

        class TestMixin(MembershipMixin):
            MIXIN_TYPE = "test_membership"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=False),
        }
        mixin.peers_lock = threading.RLock()
        mixin._swim_manager = None
        mixin._swim_started = False
        return mixin

    def test_on_swim_member_alive(self) -> None:
        """Test _on_swim_member_alive updates peer state."""
        mixin = self._create_mixin()
        old_heartbeat = mixin.peers["peer1"].last_heartbeat

        mixin._on_swim_member_alive("peer1")

        assert mixin.peers["peer1"].last_heartbeat > old_heartbeat
        assert mixin.peers["peer1"].consecutive_failures == 0

    def test_on_swim_member_failed(self) -> None:
        """Test _on_swim_member_failed updates peer state."""
        mixin = self._create_mixin()
        mixin.peers["peer1"].consecutive_failures = 0

        mixin._on_swim_member_failed("peer1")

        assert mixin.peers["peer1"].consecutive_failures == 1
        assert mixin.peers["peer1"].last_failure_time > 0


class TestMembershipMixinSummary:
    """Test membership summary methods."""

    def _create_mixin(self) -> MembershipMixin:
        """Create a MembershipMixin instance."""

        class TestMixin(MembershipMixin):
            MIXIN_TYPE = "test_membership"

        mixin = TestMixin()
        mixin.node_id = "self"
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin._swim_manager = None
        mixin._swim_started = False
        return mixin

    def test_get_swim_membership_summary_no_manager(self) -> None:
        """Test summary when no SWIM manager."""
        mixin = self._create_mixin()

        summary = mixin.get_swim_membership_summary()

        assert summary["swim_enabled"] == SWIM_ENABLED
        assert summary["swim_started"] is False
        assert summary["membership_mode"] == MEMBERSHIP_MODE

    def test_membership_health_check_http_mode(self) -> None:
        """Test health check in HTTP mode."""
        mixin = self._create_mixin()
        mixin.peers = {"peer1": MockPeer("peer1", alive=True)}

        health = mixin.membership_health_check()

        # Healthy with at least one peer in HTTP mode
        assert health["is_healthy"] is True

    def test_health_check_wrapper(self) -> None:
        """Test health_check returns standard format."""
        mixin = self._create_mixin()

        health = mixin.health_check()

        assert "healthy" in health
        assert "message" in health
        assert "details" in health


# =============================================================================
# Integration Tests
# =============================================================================


class TestMixinInteraction:
    """Test interactions between different mixins."""

    def test_leader_election_with_gossip_state(self) -> None:
        """Test leader election can use gossip-learned peer state."""

        class CombinedMixin(LeaderElectionMixin, GossipProtocolMixin):
            MIXIN_TYPE = "combined"

        mixin = CombinedMixin()
        mixin.node_id = "self"
        mixin.voter_node_ids = ["self", "peer1"]
        mixin.peers = {"peer1": MockPeer("peer1", alive=True)}
        mixin.peers_lock = threading.RLock()
        mixin.role = NodeRoleEnum.FOLLOWER
        mixin.leader_id = None
        mixin.leader_lease_id = ""
        mixin.leader_lease_expires = 0.0
        mixin.last_lease_renewal = 0.0
        mixin.voter_grant_leader_id = ""
        mixin.voter_grant_lease_id = ""
        mixin.voter_grant_expires = 0.0

        # Initialize gossip
        mixin._init_gossip_protocol()

        # Verify both work together
        assert mixin._has_voter_quorum() is True
        assert hasattr(mixin, "_gossip_peer_states")

    def test_peer_manager_with_membership(self, tmp_path: Path) -> None:
        """Test peer manager can work with membership mixin."""
        db_path = tmp_path / "p2p.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT, port INTEGER, tailscale_ip TEXT,
                reputation_score REAL, success_count INTEGER,
                failure_count INTEGER, last_seen REAL, is_bootstrap_seed INTEGER
            )
        """
        )
        conn.commit()
        conn.close()

        class CombinedMixin(PeerManagerMixin, MembershipMixin):
            MIXIN_TYPE = "combined"

        mixin = CombinedMixin()
        mixin.node_id = "self"
        mixin.db_path = db_path
        mixin.peers = {"peer1": MockPeer("peer1", alive=True)}
        mixin.peers_lock = threading.RLock()
        mixin.bootstrap_seeds = []
        mixin.verbose = True
        mixin._swim_manager = None
        mixin._swim_started = False

        # Test both work together
        mixin._save_peer_to_cache("peer1", "192.168.1.1", 8770)
        assert mixin.is_peer_alive_hybrid("peer1") is True
        assert mixin._get_cached_peer_count() == 1
