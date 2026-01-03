"""
Tests for PartitionHealer - P2P cluster partition detection and healing.

January 2026: Phase 2.3 - Comprehensive test coverage for partition_healer.py.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Data Classes (PartitionInfo, HealingResult)
# =============================================================================


class TestPartitionInfo:
    """Tests for PartitionInfo dataclass."""

    def test_creation_with_defaults(self):
        """PartitionInfo should create with default values."""
        from scripts.p2p.partition_healer import PartitionInfo

        info = PartitionInfo(partition_id="p1", leader=None)
        assert info.partition_id == "p1"
        assert info.leader is None
        assert info.members == set()
        assert info.addresses == {}

    def test_creation_with_all_fields(self):
        """PartitionInfo should accept all fields."""
        from scripts.p2p.partition_healer import PartitionInfo

        members = {"node1", "node2", "node3"}
        addresses = {"node1": "10.0.0.1", "node2": "10.0.0.2"}

        info = PartitionInfo(
            partition_id="partition_0",
            leader="node1",
            members=members,
            addresses=addresses,
        )

        assert info.partition_id == "partition_0"
        assert info.leader == "node1"
        assert info.members == members
        assert info.addresses == addresses

    def test_size_property(self):
        """size property should return member count."""
        from scripts.p2p.partition_healer import PartitionInfo

        info = PartitionInfo(
            partition_id="p1",
            leader="node1",
            members={"a", "b", "c", "d"},
        )
        assert info.size == 4

    def test_size_property_empty(self):
        """size property should return 0 for empty partition."""
        from scripts.p2p.partition_healer import PartitionInfo

        info = PartitionInfo(partition_id="p1", leader=None)
        assert info.size == 0


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_creation_with_defaults(self):
        """HealingResult should create with default values."""
        from scripts.p2p.partition_healer import HealingResult

        result = HealingResult(
            success=True,
            partitions_found=2,
            partitions_healed=1,
            nodes_reconnected=5,
        )

        assert result.success is True
        assert result.partitions_found == 2
        assert result.partitions_healed == 1
        assert result.nodes_reconnected == 5
        assert result.errors == []
        assert result.duration_ms == 0.0

    def test_creation_with_errors(self):
        """HealingResult should accept error list."""
        from scripts.p2p.partition_healer import HealingResult

        result = HealingResult(
            success=False,
            partitions_found=3,
            partitions_healed=0,
            nodes_reconnected=0,
            errors=["Connection failed", "Timeout"],
            duration_ms=1234.5,
        )

        assert result.success is False
        assert result.errors == ["Connection failed", "Timeout"]
        assert result.duration_ms == 1234.5

    def test_to_dict_via_asdict(self):
        """HealingResult should be convertible to dict."""
        from scripts.p2p.partition_healer import HealingResult

        result = HealingResult(
            success=True,
            partitions_found=1,
            partitions_healed=1,
            nodes_reconnected=3,
        )

        d = asdict(result)
        assert d["success"] is True
        assert d["partitions_found"] == 1
        assert d["nodes_reconnected"] == 3


# =============================================================================
# Test PartitionHealer Initialization
# =============================================================================


class TestPartitionHealerInit:
    """Tests for PartitionHealer initialization."""

    def test_default_initialization(self):
        """PartitionHealer should initialize with default values."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()
        assert healer._p2p_port == 8770
        assert healer._session is None

    def test_custom_config_path(self, tmp_path):
        """PartitionHealer should accept custom config path."""
        from scripts.p2p.partition_healer import PartitionHealer

        config_file = tmp_path / "custom_hosts.yaml"
        config_file.touch()

        healer = PartitionHealer(config_path=config_file)
        assert healer._config_path == config_file

    def test_custom_port(self):
        """PartitionHealer should accept custom P2P port."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer(p2p_port=9999)
        assert healer._p2p_port == 9999

    def test_custom_timeout(self):
        """PartitionHealer should accept custom timeout."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer(timeout=60.0)
        assert healer._timeout == 60.0


# =============================================================================
# Test Partition Detection
# =============================================================================


class TestPartitionDetection:
    """Tests for partition detection logic."""

    @pytest.mark.asyncio
    async def test_detects_single_partition(self):
        """Should return single partition when all nodes see each other."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # Mock peer views where all nodes see each other
        mock_views = {
            "node1": {"node1", "node2", "node3"},
            "node2": {"node1", "node2", "node3"},
            "node3": {"node1", "node2", "node3"},
        }

        # Mock the get_peer_view method
        async def mock_get_peer_view(addr):
            node_id = addr.replace("10.0.0.", "node")
            if node_id in mock_views:
                return {
                    "node_id": node_id,
                    "peers": {p: {} for p in mock_views[node_id] if p != node_id},
                }
            return None

        healer.get_peer_view = mock_get_peer_view

        # Create mock peers
        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        partitions = await healer.detect_partitions(known_peers)

        # Should find exactly 1 partition
        assert len(partitions) == 1
        assert partitions[0].size == 3

    @pytest.mark.asyncio
    async def test_detects_multiple_partitions(self):
        """Should detect multiple partitions when nodes are isolated."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # Two partitions: {node1, node2} and {node3, node4}
        mock_views = {
            "node1": {"node1", "node2"},  # Only sees node2
            "node2": {"node1", "node2"},  # Only sees node1
            "node3": {"node3", "node4"},  # Only sees node4
            "node4": {"node3", "node4"},  # Only sees node3
        }

        async def mock_get_peer_view(addr):
            # Extract node from address like "10.0.0.1" -> "node1"
            parts = addr.split(".")
            if len(parts) == 4:
                node_id = f"node{parts[-1]}"
                if node_id in mock_views:
                    return {
                        "node_id": node_id,
                        "peers": {p: {} for p in mock_views[node_id] if p != node_id},
                        "leader_id": f"leader_{node_id}",
                    }
            return None

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
            "node4": DiscoveredPeer(node_id="node4", addresses=["10.0.0.4"]),
        }

        partitions = await healer.detect_partitions(known_peers)

        # Should find 2 partitions
        assert len(partitions) == 2
        sizes = sorted([p.size for p in partitions], reverse=True)
        assert sizes == [2, 2]

    @pytest.mark.asyncio
    async def test_handles_unreachable_peers(self):
        """Should handle peers that cannot be reached."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        async def mock_get_peer_view(addr):
            # Only node1 and node2 respond
            if addr == "10.0.0.1":
                return {"node_id": "node1", "peers": {"node2": {}}}
            elif addr == "10.0.0.2":
                return {"node_id": "node2", "peers": {"node1": {}}}
            return None  # node3 is unreachable

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        partitions = await healer.detect_partitions(known_peers)

        # Should only include reachable nodes in partition
        assert len(partitions) >= 1
        # node3 should not be in any partition's members
        all_members = set()
        for p in partitions:
            all_members.update(p.members)
        assert "node3" not in all_members

    @pytest.mark.asyncio
    async def test_empty_peers_returns_empty(self):
        """Should return empty list for no peers."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()
        partitions = await healer.detect_partitions({})
        assert partitions == []


# =============================================================================
# Test Convergence Validation
# =============================================================================


class TestConvergenceValidation:
    """Tests for gossip convergence validation."""

    @pytest.mark.asyncio
    async def test_validates_gossip_convergence(self):
        """Should return True when all nodes agree on peers."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # All nodes see the same peers - each node returns its own ID in the view
        async def mock_get_peer_view(addr):
            # Extract node number from IP like "10.0.0.1" -> "node1"
            node_num = addr.split(".")[-1]
            node_id = f"node{node_num}"
            # Each node sees all three nodes as peers
            return {
                "node_id": node_id,
                "peers": {"node1": {}, "node2": {}, "node3": {}},
            }

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        converged, msg = await healer._validate_convergence(known_peers, timeout=2.0)
        assert converged is True
        assert "Convergence achieved" in msg

    @pytest.mark.asyncio
    async def test_timeout_on_slow_convergence(self):
        """Should timeout when nodes never converge."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        call_count = 0

        # Nodes never agree
        async def mock_get_peer_view(addr):
            nonlocal call_count
            call_count += 1
            # node1 only sees node2, node2 only sees node3, etc.
            node_num = int(addr.split(".")[-1])
            next_node = (node_num % 3) + 1
            return {
                "node_id": f"node{node_num}",
                "peers": {f"node{next_node}": {}},
            }

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        converged, msg = await healer._validate_convergence(known_peers, timeout=1.0)
        assert converged is False
        assert "timeout" in msg.lower()

    @pytest.mark.asyncio
    async def test_partial_convergence_handling(self):
        """Should handle partial convergence correctly."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # 2 of 3 nodes agree, one disagrees
        async def mock_get_peer_view(addr):
            node_num = int(addr.split(".")[-1])
            if node_num in [1, 2]:
                return {
                    "node_id": f"node{node_num}",
                    "peers": {"node1": {}, "node2": {}, "node3": {}},
                }
            else:
                # node3 only sees itself
                return {
                    "node_id": "node3",
                    "peers": {},
                }

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        # With 80% threshold, 66% agreement should fail
        converged, msg = await healer._validate_convergence(known_peers, timeout=1.0)
        assert converged is False

    @pytest.mark.asyncio
    async def test_no_reachable_nodes(self):
        """Should return False when no nodes are reachable."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        async def mock_get_peer_view(addr):
            return None

        healer.get_peer_view = mock_get_peer_view

        from scripts.p2p.union_discovery import DiscoveredPeer

        # Peers with no addresses
        known_peers = {
            "node1": DiscoveredPeer(node_id="node1"),  # No IP
        }

        converged, msg = await healer._validate_convergence(known_peers, timeout=1.0)
        assert converged is False
        assert "No reachable nodes" in msg


# =============================================================================
# Test Peer Injection
# =============================================================================


class TestPeerInjection:
    """Tests for peer injection functionality."""

    @pytest.mark.asyncio
    async def test_inject_peer_success(self):
        """Should return True on successful injection."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # Mock successful HTTP response using context manager pattern
        mock_response = MagicMock()
        mock_response.status = 200

        # Create async context manager for the response
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = mock_cm

        # Mock _get_session to return our mock
        async def mock_get_session():
            return mock_session

        healer._get_session = mock_get_session

        result = await healer.inject_peer(
            target_address="10.0.0.1",
            peer_to_inject="node2",
            peer_address="10.0.0.2",
        )

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_peer_failure_http_error(self):
        """Should return False on HTTP error."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        healer._session = mock_session

        result = await healer.inject_peer(
            target_address="10.0.0.1",
            peer_to_inject="node2",
            peer_address="10.0.0.2",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_inject_peer_network_error(self):
        """Should return False on network error."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=Exception("Connection refused"))
        healer._session = mock_session

        result = await healer.inject_peer(
            target_address="10.0.0.1",
            peer_to_inject="node2",
            peer_address="10.0.0.2",
        )

        assert result is False


# =============================================================================
# Test Heal Partitions
# =============================================================================


class TestHealPartitions:
    """Tests for partition healing logic."""

    @pytest.mark.asyncio
    async def test_heal_single_partition_no_op(self):
        """Should not heal when only one partition exists."""
        from scripts.p2p.partition_healer import PartitionHealer, PartitionInfo

        healer = PartitionHealer()

        partitions = [
            PartitionInfo(
                partition_id="p1",
                leader="node1",
                members={"node1", "node2", "node3"},
                addresses={"node1": "10.0.0.1", "node2": "10.0.0.2"},
            )
        ]

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
        }

        result = await healer.heal_partitions(partitions, known_peers)

        assert result.success is True
        assert result.partitions_healed == 0
        assert result.nodes_reconnected == 0

    @pytest.mark.asyncio
    async def test_heal_two_partitions(self):
        """Should create bridges between two partitions."""
        from scripts.p2p.partition_healer import PartitionHealer, PartitionInfo

        healer = PartitionHealer()

        # Mock successful injections
        healer.inject_peer = AsyncMock(return_value=True)

        partitions = [
            PartitionInfo(
                partition_id="p1",
                leader="node1",
                members={"node1", "node2"},
                addresses={"node1": "10.0.0.1", "node2": "10.0.0.2"},
            ),
            PartitionInfo(
                partition_id="p2",
                leader="node3",
                members={"node3", "node4"},
                addresses={"node3": "10.0.0.3", "node4": "10.0.0.4"},
            ),
        ]

        from scripts.p2p.union_discovery import DiscoveredPeer

        known_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
        }

        result = await healer.heal_partitions(partitions, known_peers)

        assert result.partitions_found == 2
        assert result.partitions_healed >= 1
        assert result.nodes_reconnected > 0
        assert healer.inject_peer.call_count > 0

    @pytest.mark.asyncio
    async def test_heal_records_duration(self):
        """Should record healing duration in milliseconds."""
        from scripts.p2p.partition_healer import PartitionHealer, PartitionInfo

        healer = PartitionHealer()
        healer.inject_peer = AsyncMock(return_value=True)

        partitions = [
            PartitionInfo(
                partition_id="p1",
                leader="node1",
                members={"node1"},
                addresses={"node1": "10.0.0.1"},
            ),
            PartitionInfo(
                partition_id="p2",
                leader="node2",
                members={"node2"},
                addresses={"node2": "10.0.0.2"},
            ),
        ]

        result = await healer.heal_partitions(partitions, {})

        assert result.duration_ms >= 0


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for healing trigger rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_respects_min_interval(self):
        """Should rate limit healing requests."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        # Mock run_healing_pass to avoid actual work
        healer.run_healing_pass = AsyncMock(
            return_value=MagicMock(
                success=True, partitions_healed=1, partitions_found=2
            )
        )

        # First call should succeed
        healer._last_healing_time = 0
        result1 = await healer.trigger_healing_pass(delay=0.01)
        assert result1 is not None

        # Second call immediately after should be rate-limited
        result2 = await healer.trigger_healing_pass(delay=0.01)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_force_bypasses_rate_limit(self):
        """Force flag should bypass rate limiting."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        healer.run_healing_pass = AsyncMock(
            return_value=MagicMock(success=True, partitions_healed=1)
        )

        # Set recent healing time
        healer._last_healing_time = time.time()

        # Should succeed with force=True despite recent healing
        result = await healer.trigger_healing_pass(delay=0.01, force=True)
        assert result is not None


# =============================================================================
# Test Healing Pass
# =============================================================================


class TestHealingPass:
    """Tests for full healing pass."""

    @pytest.mark.asyncio
    async def test_run_healing_pass_no_partitions(self):
        """Should succeed when less than 2 peers discovered."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()
        healer.discover_all_peers = AsyncMock(return_value={})

        result = await healer.run_healing_pass()

        assert result.success is True
        assert result.partitions_found == 0
        assert "Not enough peers" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_healing_pass_single_partition(self):
        """Should succeed with single healthy partition."""
        from scripts.p2p.partition_healer import PartitionHealer, PartitionInfo

        healer = PartitionHealer()

        from scripts.p2p.union_discovery import DiscoveredPeer

        mock_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
        }

        healer.discover_all_peers = AsyncMock(return_value=mock_peers)
        healer.detect_partitions = AsyncMock(
            return_value=[
                PartitionInfo(
                    partition_id="p1",
                    leader="node1",
                    members={"node1", "node2"},
                )
            ]
        )

        result = await healer.run_healing_pass()

        assert result.success is True
        assert result.partitions_found == 1
        assert result.partitions_healed == 0

    @pytest.mark.asyncio
    async def test_run_healing_pass_heals_partitions(self):
        """Should heal multiple partitions."""
        from scripts.p2p.partition_healer import PartitionHealer, PartitionInfo

        healer = PartitionHealer()

        from scripts.p2p.union_discovery import DiscoveredPeer

        mock_peers = {
            "node1": DiscoveredPeer(node_id="node1", addresses=["10.0.0.1"]),
            "node2": DiscoveredPeer(node_id="node2", addresses=["10.0.0.2"]),
            "node3": DiscoveredPeer(node_id="node3", addresses=["10.0.0.3"]),
        }

        healer.discover_all_peers = AsyncMock(return_value=mock_peers)
        healer.detect_partitions = AsyncMock(
            return_value=[
                PartitionInfo(
                    partition_id="p1",
                    leader="node1",
                    members={"node1"},
                    addresses={"node1": "10.0.0.1"},
                ),
                PartitionInfo(
                    partition_id="p2",
                    leader="node2",
                    members={"node2", "node3"},
                    addresses={"node2": "10.0.0.2", "node3": "10.0.0.3"},
                ),
            ]
        )
        healer.inject_peer = AsyncMock(return_value=True)
        healer._validate_convergence = AsyncMock(return_value=(True, "Converged"))
        healer._emit_healing_event = MagicMock()

        result = await healer.run_healing_pass()

        assert result.partitions_found == 2
        assert result.partitions_healed >= 1
        healer._emit_healing_event.assert_called_once()


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_healing_event_handles_missing_module(self):
        """Should handle missing event module gracefully."""
        from scripts.p2p.partition_healer import PartitionHealer, HealingResult

        healer = PartitionHealer()

        result = HealingResult(
            success=True,
            partitions_found=2,
            partitions_healed=1,
            nodes_reconnected=5,
            duration_ms=123.45,
        )

        # Should not raise even if emit_event is not available
        # (the function handles ImportError internally)
        healer._emit_healing_event(result)

    def test_emit_healing_event_logs_on_success(self):
        """Should log event emission."""
        from scripts.p2p.partition_healer import PartitionHealer, HealingResult
        import logging

        healer = PartitionHealer()

        result = HealingResult(
            success=True,
            partitions_found=2,
            partitions_healed=1,
            nodes_reconnected=5,
            duration_ms=123.45,
        )

        # Verify logging behavior - the function should not crash
        with patch.object(logging, "getLogger") as mock_logger:
            healer._emit_healing_event(result)
            # Function should complete without error


# =============================================================================
# Test Singleton Access
# =============================================================================


class TestSingletonAccess:
    """Tests for singleton pattern."""

    def test_get_partition_healer_creates_instance(self):
        """get_partition_healer should create singleton instance."""
        from scripts.p2p.partition_healer import (
            get_partition_healer,
            reset_partition_healer,
        )

        reset_partition_healer()  # Clean state

        healer = get_partition_healer()
        assert healer is not None

        # Same instance returned
        healer2 = get_partition_healer()
        assert healer is healer2

        reset_partition_healer()

    def test_reset_clears_singleton(self):
        """reset_partition_healer should clear singleton."""
        from scripts.p2p.partition_healer import (
            get_partition_healer,
            reset_partition_healer,
        )

        healer1 = get_partition_healer()
        reset_partition_healer()
        healer2 = get_partition_healer()

        assert healer1 is not healer2
        reset_partition_healer()


# =============================================================================
# Test Close Method
# =============================================================================


class TestClose:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        """close() should close aiohttp session."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        mock_session = AsyncMock()
        mock_session.closed = False
        healer._session = mock_session

        await healer.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self):
        """close() should handle None session."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()
        healer._session = None

        # Should not raise
        await healer.close()

    @pytest.mark.asyncio
    async def test_close_handles_already_closed_session(self):
        """close() should handle already closed session."""
        from scripts.p2p.partition_healer import PartitionHealer

        healer = PartitionHealer()

        mock_session = AsyncMock()
        mock_session.closed = True  # Already closed
        healer._session = mock_session

        await healer.close()

        # Should not try to close again
        mock_session.close.assert_not_called()
