"""Tests for scripts.lib.hosts module.

Tests the unified cluster host configuration module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
from scripts.lib.hosts import (
    HostConfig,
    HostsManager,
    EloSyncConfig,
    get_hosts,
    get_host,
    get_host_names,
    get_active_hosts,
    get_training_hosts,
    get_selfplay_hosts,
    get_p2p_voters,
    get_elo_sync_config,
    get_hosts_by_group,
)


class TestHostConfig:
    """Tests for HostConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic HostConfig."""
        host = HostConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            ssh_user="ubuntu",
        )
        assert host.name == "test-node"
        assert host.ssh_host == "192.168.1.1"
        assert host.ssh_user == "ubuntu"

    def test_effective_ssh_host_with_tailscale(self):
        """Test that tailscale_ip is preferred over ssh_host."""
        host = HostConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            tailscale_ip="100.1.2.3",
        )
        assert host.effective_ssh_host == "100.1.2.3"

    def test_effective_ssh_host_without_tailscale(self):
        """Test fallback to ssh_host when no tailscale_ip."""
        host = HostConfig(
            name="test-node",
            ssh_host="192.168.1.1",
        )
        assert host.effective_ssh_host == "192.168.1.1"

    def test_is_vast(self):
        """Test Vast.ai instance detection."""
        vast_host = HostConfig(
            name="vast-12345",
            ssh_host="ssh.vast.ai",
            vast_instance_id="12345",
        )
        assert vast_host.is_vast is True

        regular_host = HostConfig(
            name="lambda-gh200-a",
            ssh_host="192.168.1.1",
        )
        assert regular_host.is_vast is False

    def test_is_lambda(self):
        """Test Lambda Labs instance detection."""
        lambda_host = HostConfig(
            name="lambda-gh200-a",
            ssh_host="192.168.1.1",
        )
        assert lambda_host.is_lambda is True

        other_host = HostConfig(
            name="aws-staging",
            ssh_host="192.168.1.1",
        )
        assert other_host.is_lambda is False

    def test_has_role(self):
        """Test role checking."""
        host = HostConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            role="nn_training",
            roles=["selfplay", "benchmark"],
        )
        assert host.has_role("nn_training") is True
        assert host.has_role("selfplay") is True
        assert host.has_role("benchmark") is True
        assert host.has_role("gauntlet") is False

    def test_all_roles(self):
        """Test getting all roles."""
        host = HostConfig(
            name="test-node",
            ssh_host="192.168.1.1",
            role="primary",
            roles=["selfplay", "training"],
        )
        all_roles = host.all_roles
        assert "primary" in all_roles
        assert "selfplay" in all_roles
        assert "training" in all_roles


class TestHostsManager:
    """Tests for HostsManager class."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config file."""
        config_content = """
hosts:
  test-node-1:
    ssh_host: 192.168.1.1
    tailscale_ip: 100.1.1.1
    ssh_user: ubuntu
    role: selfplay
    status: ready
    p2p_voter: true
  test-node-2:
    ssh_host: 192.168.1.2
    ssh_user: root
    role: nn_training
    status: ready
  test-node-offline:
    ssh_host: 192.168.1.3
    status: offline
elo_sync:
  coordinator: test-node-1
  sync_port: 8766
  sync_interval: 300
"""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_load_hosts_from_config(self, mock_config):
        """Test loading hosts from config file."""
        manager = HostsManager(distributed_hosts_path=mock_config)
        hosts = manager.get_hosts()

        assert len(hosts) == 3
        names = [h.name for h in hosts]
        assert "test-node-1" in names
        assert "test-node-2" in names

    def test_filter_by_role(self, mock_config):
        """Test filtering hosts by role."""
        manager = HostsManager(distributed_hosts_path=mock_config)

        # Note: hosts without explicit role default to "selfplay"
        selfplay_hosts = manager.get_hosts(role="selfplay")
        assert len(selfplay_hosts) >= 1
        assert any(h.name == "test-node-1" for h in selfplay_hosts)

        training_hosts = manager.get_hosts(role="nn_training")
        assert len(training_hosts) == 1
        assert training_hosts[0].name == "test-node-2"

    def test_filter_by_status(self, mock_config):
        """Test filtering hosts by status."""
        manager = HostsManager(distributed_hosts_path=mock_config)

        ready_hosts = manager.get_hosts(status="ready")
        assert len(ready_hosts) == 2

        offline_hosts = manager.get_hosts(status="offline")
        assert len(offline_hosts) == 1
        assert offline_hosts[0].name == "test-node-offline"

    def test_filter_by_p2p_voter(self, mock_config):
        """Test filtering P2P voters."""
        manager = HostsManager(distributed_hosts_path=mock_config)

        voters = manager.get_hosts(p2p_voter=True)
        assert len(voters) == 1
        assert voters[0].name == "test-node-1"

    def test_get_host_by_name(self, mock_config):
        """Test getting a specific host."""
        manager = HostsManager(distributed_hosts_path=mock_config)

        host = manager.get_host("test-node-1")
        assert host is not None
        assert host.name == "test-node-1"
        assert host.tailscale_ip == "100.1.1.1"

        missing = manager.get_host("nonexistent")
        assert missing is None

    def test_get_elo_sync_config(self, mock_config):
        """Test loading ELO sync configuration."""
        manager = HostsManager(distributed_hosts_path=mock_config)

        elo_config = manager.get_elo_sync_config()
        assert elo_config.coordinator == "test-node-1"
        assert elo_config.sync_port == 8766
        assert elo_config.sync_interval == 300


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_hosts_returns_list(self):
        """Test that get_hosts returns a list."""
        hosts = get_hosts()
        assert isinstance(hosts, list)

    def test_get_host_names_returns_list(self):
        """Test that get_host_names returns a list of strings."""
        names = get_host_names()
        assert isinstance(names, list)
        if names:
            assert all(isinstance(n, str) for n in names)

    def test_get_active_hosts(self):
        """Test getting active hosts."""
        hosts = get_active_hosts()
        assert isinstance(hosts, list)
        # All returned hosts should have ready or active status
        for h in hosts:
            assert h.status in ("ready", "active")

    def test_get_training_hosts(self):
        """Test getting training hosts."""
        hosts = get_training_hosts()
        assert isinstance(hosts, list)
        # All returned hosts should have training role
        for h in hosts:
            assert h.has_role("nn_training") or "training" in h.role

    def test_get_p2p_voters(self):
        """Test getting P2P voters."""
        voters = get_p2p_voters()
        assert isinstance(voters, list)
        # All returned hosts should be voters
        for h in voters:
            assert h.p2p_voter is True


class TestClusterYamlFallback:
    """Tests for cluster.yaml fallback when distributed_hosts.yaml missing."""

    @pytest.fixture
    def cluster_yaml_config(self, tmp_path):
        """Create a mock cluster.yaml file."""
        config_content = """
cluster:
  name: test-cluster

nodes:
  node-a:
    host: node-a
    tailscale_ip: '100.1.1.1'
    ssh_user: ubuntu
    gpu_type: GH200
    roles: [selfplay, training]
    status: active
  node-b:
    host: node-b
    ssh_user: ubuntu
    gpu_type: H100
    roles: [selfplay]
    status: active

groups:
  primary:
    description: Primary training nodes
    nodes: [node-a]
"""
        config_file = tmp_path / "cluster.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_fallback_to_cluster_yaml(self, cluster_yaml_config, tmp_path):
        """Test fallback when distributed_hosts.yaml doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        manager = HostsManager(
            distributed_hosts_path=nonexistent,
            cluster_yaml_path=cluster_yaml_config,
        )

        hosts = manager.get_hosts()
        assert len(hosts) == 2
        assert any(h.name == "node-a" for h in hosts)

    def test_get_hosts_by_group(self, cluster_yaml_config, tmp_path):
        """Test getting hosts by group name."""
        nonexistent = tmp_path / "nonexistent.yaml"
        manager = HostsManager(
            distributed_hosts_path=nonexistent,
            cluster_yaml_path=cluster_yaml_config,
        )

        primary_hosts = manager.get_hosts_by_group("primary")
        assert len(primary_hosts) == 1
        assert primary_hosts[0].name == "node-a"

        missing_group = manager.get_hosts_by_group("nonexistent")
        assert len(missing_group) == 0
