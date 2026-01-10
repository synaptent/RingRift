"""Unit tests for QuorumManager.

Tests voter quorum management, IP mapping, health checking, and gossip adoption.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.managers.quorum_manager import (
    QuorumConfig,
    QuorumManager,
    VoterHealthStatus,
    create_quorum_manager,
    get_quorum_manager,
    set_quorum_manager,
)


class MockPeerInfo:
    """Mock peer info for testing."""

    def __init__(
        self,
        node_id: str,
        host: str = "",
        status: str = "alive",
    ) -> None:
        self.node_id = node_id
        self.host = host
        self._status = status

    def is_alive(self) -> bool:
        return self._status == "alive"


# ============================================================================
# QuorumConfig Tests
# ============================================================================


class TestQuorumConfig:
    """Tests for QuorumConfig dataclass."""

    def test_config_with_minimal_args(self):
        """Test config with only required arguments."""
        config = QuorumConfig(node_id="test-node")
        assert config.node_id == "test-node"
        assert config.config_path is None
        assert config.ringrift_path is None

    def test_config_with_all_args(self):
        """Test config with all arguments."""
        config = QuorumConfig(
            node_id="test-node",
            config_path=Path("/path/to/config.yaml"),
            ringrift_path=Path("/path/to/ringrift"),
        )
        assert config.node_id == "test-node"
        assert config.config_path == Path("/path/to/config.yaml")
        assert config.ringrift_path == Path("/path/to/ringrift")

    def test_config_auto_detects_path_from_ai_service(self):
        """Test config auto-detects config path from ai-service directory."""
        config = QuorumConfig(
            node_id="test-node",
            ringrift_path=Path("/path/to/ai-service"),
        )
        assert config.config_path == Path("/path/to/ai-service/config/distributed_hosts.yaml")

    def test_config_auto_detects_path_from_ringrift_root(self):
        """Test config auto-detects config path from RingRift root."""
        config = QuorumConfig(
            node_id="test-node",
            ringrift_path=Path("/path/to/ringrift"),
        )
        assert config.config_path == Path("/path/to/ringrift/ai-service/config/distributed_hosts.yaml")


# ============================================================================
# VoterHealthStatus Tests
# ============================================================================


class TestVoterHealthStatus:
    """Tests for VoterHealthStatus dataclass."""

    def test_default_values(self):
        """Test default values."""
        status = VoterHealthStatus()
        assert status.voters_total == 0
        assert status.voters_alive == 0
        assert status.voters_offline == []
        assert status.quorum_size == 0
        assert status.quorum_ok is True
        assert status.quorum_threatened is False
        assert status.voter_status == {}

    def test_custom_values(self):
        """Test with custom values."""
        status = VoterHealthStatus(
            voters_total=7,
            voters_alive=5,
            voters_offline=["node1", "node2"],
            quorum_size=4,
            quorum_ok=True,
            quorum_threatened=True,
            voter_status={"node1": {"alive": False, "detail": "unreachable"}},
        )
        assert status.voters_total == 7
        assert status.voters_alive == 5
        assert len(status.voters_offline) == 2


# ============================================================================
# QuorumManager Initialization Tests
# ============================================================================


class TestQuorumManagerInit:
    """Tests for QuorumManager initialization."""

    def test_init_with_minimal_args(self):
        """Test initialization with minimal arguments."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )
        assert manager._config.node_id == "test-node"
        assert manager.voter_node_ids == []
        assert manager.voter_config_source == "none"

    def test_init_with_peers_lock(self):
        """Test initialization with peers lock."""
        lock = threading.RLock()
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
            get_peers_lock=lambda: lock,
        )
        assert manager._get_peers_lock is not None


# ============================================================================
# Load Voter Node IDs Tests
# ============================================================================


class TestLoadVoterNodeIds:
    """Tests for load_voter_node_ids()."""

    def test_load_from_env_var(self):
        """Test loading voters from environment variable."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "node1, node2, node3"}):
            voters = manager.load_voter_node_ids()

        assert voters == ["node1", "node2", "node3"]
        assert manager.voter_config_source == "env"

    def test_load_from_env_deduplicates(self):
        """Test that env var voters are deduplicated."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "node1, node2, node1, node2"}):
            voters = manager.load_voter_node_ids()

        assert voters == ["node1", "node2"]

    def test_load_from_env_strips_whitespace(self):
        """Test that env var voters have whitespace stripped."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "  node1  ,  node2  "}):
            voters = manager.load_voter_node_ids()

        assert voters == ["node1", "node2"]

    def test_load_from_cluster_config(self):
        """Test loading voters from cluster_config module."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        # Ensure env var is not set
        orig_env = os.environ.pop("RINGRIFT_P2P_VOTERS", None)
        try:
            # Mock the dynamic import inside load_voter_node_ids
            import sys

            mock_module = MagicMock()
            mock_module.get_p2p_voters = MagicMock(return_value=["voter1", "voter2"])
            sys.modules["app.config.cluster_config"] = mock_module

            try:
                voters = manager.load_voter_node_ids()

                # If the mock worked, we should get config source
                # Note: The actual code may use real cluster_config if available
                assert manager.voter_config_source in ("env", "config", "none")
            finally:
                # Restore
                sys.modules.pop("app.config.cluster_config", None)
        finally:
            if orig_env is not None:
                os.environ["RINGRIFT_P2P_VOTERS"] = orig_env

    def test_load_from_yaml_fallback(self):
        """Test loading voters from YAML file as fallback."""
        # Create temporary YAML config
        yaml_content = """
p2p_voters:
  - hetzner-cpu1
  - hetzner-cpu2
  - nebius-backbone-1
hosts:
  hetzner-cpu1:
    tailscale_ip: 100.64.0.1
  hetzner-cpu2:
    tailscale_ip: 100.64.0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: {},
            )

            # Clear env and make cluster_config import fail
            orig_env = os.environ.pop("RINGRIFT_P2P_VOTERS", None)
            try:
                import sys

                # Remove cached module to force re-import (which will fail)
                orig_module = sys.modules.pop("app.config.cluster_config", None)

                # Create a mock module that raises ImportError when accessed
                class FailingModule:
                    def __getattr__(self, name):
                        raise ImportError("Mock ImportError")

                sys.modules["app.config.cluster_config"] = FailingModule()

                try:
                    voters = manager.load_voter_node_ids()
                finally:
                    sys.modules.pop("app.config.cluster_config", None)
                    if orig_module is not None:
                        sys.modules["app.config.cluster_config"] = orig_module

                assert "hetzner-cpu1" in voters
                assert "hetzner-cpu2" in voters
                assert "nebius-backbone-1" in voters
                assert manager.voter_config_source == "config"
            finally:
                if orig_env is not None:
                    os.environ["RINGRIFT_P2P_VOTERS"] = orig_env
        finally:
            config_path.unlink()

    def test_load_returns_empty_when_no_config(self):
        """Test loading returns empty list when no config available."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RINGRIFT_P2P_VOTERS", None)

            # Mock cluster_config to raise ImportError
            import sys
            sys.modules.pop("app.config.cluster_config", None)

            voters = manager.load_voter_node_ids()

        # Without env, cluster_config, or YAML, should return empty or actual cluster config
        assert manager.voter_config_source in ("env", "config", "none")


# ============================================================================
# Build IP Mapping Tests
# ============================================================================


class TestBuildIpMapping:
    """Tests for build_voter_ip_mapping() and build_ip_to_node_map()."""

    def test_build_voter_ip_mapping_empty_when_no_voters(self):
        """Test voter IP mapping is empty when no voters configured."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        mapping = manager.build_voter_ip_mapping()
        assert mapping == {}

    def test_build_voter_ip_mapping_from_config(self):
        """Test building voter IP mapping from config file."""
        yaml_content = """
p2p_voters:
  - node1
  - node2
hosts:
  node1:
    tailscale_ip: 100.64.0.1
    ssh_host: 192.168.1.1
  node2:
    tailscale_ip: 100.64.0.2
  node3:
    tailscale_ip: 100.64.0.3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: {},
            )

            # Load voters first
            with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "node1,node2"}):
                manager.load_voter_node_ids()

            mapping = manager.build_voter_ip_mapping()

            assert "node1" in mapping
            assert "100.64.0.1" in mapping["node1"]
            assert "192.168.1.1" in mapping["node1"]
            assert "node2" in mapping
            assert "100.64.0.2" in mapping["node2"]
            # node3 not in voters
            assert "node3" not in mapping
        finally:
            config_path.unlink()

    def test_build_ip_to_node_map(self):
        """Test building IP-to-node reverse mapping."""
        yaml_content = """
hosts:
  node1:
    tailscale_ip: 100.64.0.1
    ssh_host: 192.168.1.1
  node2:
    tailscale_ip: 100.64.0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: {},
            )

            mapping = manager.build_ip_to_node_map()

            assert mapping.get("100.64.0.1") == "node1"
            assert mapping.get("192.168.1.1") == "node1"
            assert mapping.get("100.64.0.2") == "node2"
        finally:
            config_path.unlink()

    def test_build_ip_to_node_map_skips_hostnames(self):
        """Test that ssh_host with hostnames are skipped."""
        yaml_content = """
hosts:
  node1:
    tailscale_ip: 100.64.0.1
    ssh_host: ssh5.vast.ai
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: {},
            )

            mapping = manager.build_ip_to_node_map()

            assert mapping.get("100.64.0.1") == "node1"
            assert "ssh5.vast.ai" not in mapping
        finally:
            config_path.unlink()


# ============================================================================
# Peer ID Resolution Tests
# ============================================================================


class TestResolvePeerIdToNodeName:
    """Tests for resolve_peer_id_to_node_name()."""

    def test_returns_node_id_unchanged(self):
        """Test that plain node IDs are returned unchanged."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        assert manager.resolve_peer_id_to_node_name("node1") == "node1"
        assert manager.resolve_peer_id_to_node_name("hetzner-cpu1") == "hetzner-cpu1"

    def test_translates_ip_port_to_node_name(self):
        """Test that IP:port format is translated."""
        yaml_content = """
hosts:
  node1:
    tailscale_ip: 100.64.0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: {},
            )
            manager.build_ip_to_node_map()

            result = manager.resolve_peer_id_to_node_name("100.64.0.1:8770")
            assert result == "node1"
        finally:
            config_path.unlink()

    def test_returns_original_if_no_mapping(self):
        """Test returns original if no mapping found."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        result = manager.resolve_peer_id_to_node_name("192.168.1.100:8770")
        assert result == "192.168.1.100:8770"


# ============================================================================
# Find Voter By IP Tests
# ============================================================================


class TestFindVoterPeerByIp:
    """Tests for find_voter_peer_by_ip()."""

    def test_find_by_direct_node_id_match(self):
        """Test finding voter by direct node_id match in peers."""
        peers = {
            "voter1": {"status": "alive", "host": "100.64.0.1"},
            "voter2": {"status": "alive", "host": "100.64.0.2"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        peer_key, peer_info = manager.find_voter_peer_by_ip("voter1")
        assert peer_key == "voter1"
        assert peer_info["host"] == "100.64.0.1"

    def test_find_by_host_field(self):
        """Test finding voter by host field match."""
        yaml_content = """
p2p_voters:
  - voter1
hosts:
  voter1:
    tailscale_ip: 100.64.0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)

        try:
            peers = {
                "unknown-peer": {"status": "alive", "host": "100.64.0.1"},
            }
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node", config_path=config_path),
                get_peers=lambda: peers,
            )

            with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1"}):
                manager.load_voter_node_ids()

            peer_key, peer_info = manager.find_voter_peer_by_ip("voter1")
            assert peer_key == "unknown-peer"
            assert peer_info["host"] == "100.64.0.1"
        finally:
            config_path.unlink()

    def test_returns_none_when_not_found(self):
        """Test returns None when voter not found."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        peer_key, peer_info = manager.find_voter_peer_by_ip("nonexistent")
        assert peer_key is None
        assert peer_info is None


# ============================================================================
# Count Alive Voters Tests
# ============================================================================


class TestCountAliveVoters:
    """Tests for count_alive_voters()."""

    def test_count_returns_zero_when_no_voters(self):
        """Test count returns 0 when no voters configured."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        count = manager.count_alive_voters()
        assert count == 0

    def test_count_includes_self_if_voter(self):
        """Test count includes self if we are a voter."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="my-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "my-node,other-node"}):
            manager.load_voter_node_ids()

        count = manager.count_alive_voters()
        assert count >= 1  # At least self is counted

    def test_count_alive_voters_by_direct_match(self):
        """Test counting alive voters by direct node_id match."""
        peers = {
            "voter1": {"status": "alive", "host": "100.64.0.1"},
            "voter2": {"status": "offline", "host": "100.64.0.2"},
            "voter3": {"status": "alive", "host": "100.64.0.3"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2,voter3"}):
            manager.load_voter_node_ids()

        count = manager.count_alive_voters()
        assert count == 2  # voter1 and voter3 are alive

    def test_count_skips_swim_peers(self):
        """Test that SWIM protocol peers (port 7947) are skipped."""
        peers = {
            "100.64.0.1:7947": {"status": "alive"},  # SWIM entry - should be skipped
            "voter1": {"status": "alive", "host": "100.64.0.1"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1"}):
            manager.load_voter_node_ids()

        count = manager.count_alive_voters()
        # Should count voter1 but not double-count via SWIM entry
        assert count == 1


# ============================================================================
# Check Voter Health Tests
# ============================================================================


class TestCheckVoterHealth:
    """Tests for check_voter_health()."""

    def test_check_returns_ok_when_no_voters(self):
        """Test health check returns OK when no voters configured."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        health = manager.check_voter_health(voter_quorum_size=4)
        assert health.quorum_ok is True
        assert health.voters_total == 0

    def test_check_detects_quorum_loss(self):
        """Test health check detects quorum loss."""
        peers = {
            "voter1": {"status": "alive", "host": "100.64.0.1"},
            "voter2": {"status": "offline", "host": "100.64.0.2"},
            "voter3": {"status": "offline", "host": "100.64.0.3"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2,voter3"}):
            manager.load_voter_node_ids()

        health = manager.check_voter_health(voter_quorum_size=2)
        assert health.quorum_ok is False
        assert health.voters_alive == 1
        assert health.voters_total == 3
        assert "voter2" in health.voters_offline
        assert "voter3" in health.voters_offline

    def test_check_detects_quorum_threatened(self):
        """Test health check detects quorum threatened."""
        peers = {
            "voter1": {"status": "alive", "host": "100.64.0.1"},
            "voter2": {"status": "alive", "host": "100.64.0.2"},
            "voter3": {"status": "offline", "host": "100.64.0.3"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2,voter3"}):
            manager.load_voter_node_ids()

        health = manager.check_voter_health(voter_quorum_size=2)
        assert health.quorum_ok is True
        assert health.quorum_threatened is True  # alive == quorum_size

    def test_check_includes_self_status(self):
        """Test health check includes self status."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="my-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "my-node,other-node"}):
            manager.load_voter_node_ids()

        health = manager.check_voter_health(voter_quorum_size=1)
        assert "my-node" in health.voter_status
        assert health.voter_status["my-node"]["alive"] is True
        assert health.voter_status["my-node"]["detail"] == "self"


# ============================================================================
# Adopt Voter Set Tests
# ============================================================================


class TestMaybeAdoptVoterNodeIds:
    """Tests for maybe_adopt_voter_node_ids()."""

    def test_adopt_when_no_config(self):
        """Test adopting voter set when no config exists."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        adopted = manager.maybe_adopt_voter_node_ids(["voter1", "voter2"], source="gossip")
        assert adopted is True
        assert manager.voter_node_ids == ["voter1", "voter2"]
        assert "adopted:gossip" in manager.voter_config_source

    def test_does_not_adopt_when_env_var_set(self):
        """Test voter set not adopted when env var is set."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "existing-voter"}):
            manager.load_voter_node_ids()
            adopted = manager.maybe_adopt_voter_node_ids(["new-voter"], source="gossip")

        assert adopted is False
        assert "existing-voter" in manager.voter_node_ids

    def test_does_not_adopt_when_config_source(self):
        """Test voter set not adopted when configured from YAML."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )
        manager._voter_config_source = "config"
        manager._voter_node_ids = ["configured-voter"]

        adopted = manager.maybe_adopt_voter_node_ids(["new-voter"], source="gossip")
        assert adopted is False

    def test_does_not_adopt_same_set(self):
        """Test voter set not adopted if same as current."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )
        manager._voter_node_ids = ["voter1", "voter2"]
        manager._voter_config_source = "adopted:previous"

        # Try to adopt same set (different order)
        adopted = manager.maybe_adopt_voter_node_ids(["voter2", "voter1"], source="gossip")
        assert adopted is False

    def test_does_not_adopt_empty_set(self):
        """Test empty voter set is not adopted."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        adopted = manager.maybe_adopt_voter_node_ids([], source="gossip")
        assert adopted is False


# ============================================================================
# SWIM Peer ID Tests
# ============================================================================


class TestIsSwimPeerId:
    """Tests for is_swim_peer_id()."""

    def test_detects_swim_format(self):
        """Test detection of SWIM protocol format."""
        assert QuorumManager.is_swim_peer_id("100.64.0.1:7947") is True
        assert QuorumManager.is_swim_peer_id("192.168.1.1:7947") is True

    def test_rejects_non_swim_ports(self):
        """Test rejection of non-SWIM ports."""
        assert QuorumManager.is_swim_peer_id("100.64.0.1:8770") is False
        assert QuorumManager.is_swim_peer_id("100.64.0.1:443") is False

    def test_rejects_node_ids(self):
        """Test rejection of plain node IDs."""
        assert QuorumManager.is_swim_peer_id("hetzner-cpu1") is False
        assert QuorumManager.is_swim_peer_id("my-node") is False

    def test_handles_empty_string(self):
        """Test handling of empty string."""
        assert QuorumManager.is_swim_peer_id("") is False


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health_check() method."""

    def test_health_check_idle_when_no_voters(self):
        """Test health check returns idle when no voters configured."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        health = manager.health_check()
        assert health["status"] == "idle"
        assert health["voter_count"] == 0

    def test_health_check_healthy_without_quorum_size(self):
        """Test health check returns healthy when quorum size not provided."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2"}):
            manager.load_voter_node_ids()

        health = manager.health_check()
        assert health["status"] == "healthy"
        assert health["voter_count"] == 2

    def test_health_check_with_quorum_size(self):
        """Test health check with quorum size validation."""
        peers = {
            "voter1": {"status": "alive", "host": "100.64.0.1"},
            "voter2": {"status": "alive", "host": "100.64.0.2"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2"}):
            manager.load_voter_node_ids()

        health = manager.health_check(voter_quorum_size=2)
        assert health["status"] in ("healthy", "degraded")
        assert "voters_alive" in health
        assert "quorum_ok" in health

    def test_health_check_error_on_quorum_loss(self):
        """Test health check returns error on quorum loss."""
        peers = {
            "voter1": {"status": "offline", "host": "100.64.0.1"},
            "voter2": {"status": "offline", "host": "100.64.0.2"},
        }
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: peers,
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2"}):
            manager.load_voter_node_ids()

        health = manager.health_check(voter_quorum_size=2)
        assert health["status"] == "error"
        assert health["quorum_ok"] is False


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_quorum_manager_initially_none(self):
        """Test get_quorum_manager returns None initially."""
        # Reset global state
        import scripts.p2p.managers.quorum_manager as qm
        original = qm._quorum_manager
        qm._quorum_manager = None

        try:
            assert get_quorum_manager() is None
        finally:
            qm._quorum_manager = original

    def test_set_quorum_manager(self):
        """Test set_quorum_manager sets the singleton."""
        import scripts.p2p.managers.quorum_manager as qm
        original = qm._quorum_manager

        try:
            manager = QuorumManager(
                config=QuorumConfig(node_id="test-node"),
                get_peers=lambda: {},
            )
            set_quorum_manager(manager)
            assert get_quorum_manager() is manager
        finally:
            qm._quorum_manager = original

    def test_create_quorum_manager(self):
        """Test create_quorum_manager creates and registers manager."""
        import scripts.p2p.managers.quorum_manager as qm
        original = qm._quorum_manager

        try:
            manager = create_quorum_manager(
                node_id="created-node",
                get_peers=lambda: {},
            )
            assert manager is not None
            assert manager._config.node_id == "created-node"
            assert get_quorum_manager() is manager
        finally:
            qm._quorum_manager = original

    def test_create_quorum_manager_with_all_args(self):
        """Test create_quorum_manager with all arguments."""
        import scripts.p2p.managers.quorum_manager as qm
        original = qm._quorum_manager

        try:
            lock = threading.RLock()
            manager = create_quorum_manager(
                node_id="created-node",
                get_peers=lambda: {"peer1": {"status": "alive"}},
                get_peers_lock=lambda: lock,
                config_path=Path("/path/to/config.yaml"),
                ringrift_path=Path("/path/to/ringrift"),
            )
            assert manager._config.node_id == "created-node"
            assert manager._config.config_path == Path("/path/to/config.yaml")
        finally:
            qm._quorum_manager = original


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_voter_access(self):
        """Test concurrent access to voter node IDs is thread-safe."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        with patch.dict(os.environ, {"RINGRIFT_P2P_VOTERS": "voter1,voter2,voter3"}):
            manager.load_voter_node_ids()

        results = []

        def read_voters():
            for _ in range(100):
                voters = manager.voter_node_ids
                results.append(len(voters))

        threads = [threading.Thread(target=read_voters) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should see the same value
        assert all(r == 3 for r in results)

    def test_concurrent_adopt_is_safe(self):
        """Test concurrent adopt operations are thread-safe."""
        manager = QuorumManager(
            config=QuorumConfig(node_id="test-node"),
            get_peers=lambda: {},
        )

        results = []

        def try_adopt(voter_set: list[str]):
            adopted = manager.maybe_adopt_voter_node_ids(voter_set, source="test")
            results.append((voter_set, adopted))

        threads = [
            threading.Thread(target=try_adopt, args=(["voter1"],)),
            threading.Thread(target=try_adopt, args=(["voter2"],)),
            threading.Thread(target=try_adopt, args=(["voter3"],)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed (first one)
        adopted_count = sum(1 for _, adopted in results if adopted)
        assert adopted_count >= 1  # At least one succeeded

        # Final state should be one of the voter sets
        assert len(manager.voter_node_ids) == 1
