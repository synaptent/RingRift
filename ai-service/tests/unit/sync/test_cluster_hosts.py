"""Tests for cluster_hosts.py - cluster host configuration and discovery utilities."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.sync.cluster_hosts import (
    DATA_SYNC_PORT,
    ELO_SYNC_PORT,
    MODEL_SYNC_PORT,
    ClusterNode,
    EloSyncConfig,
    get_active_nodes,
    get_cluster_nodes,
    get_coordinator_address,
    get_coordinator_node,
    get_elo_sync_config,
    load_hosts_config,
)


def load_hosts_config_with_warning():
    """Call deprecated load_hosts_config() while consuming its expected warning."""
    with pytest.deprecated_call(match="load_hosts_config\\(\\) is deprecated"):
        return load_hosts_config()


class TestClusterNode:
    """Test ClusterNode dataclass."""

    def test_default_initialization(self):
        """Test ClusterNode with minimal required fields."""
        node = ClusterNode(name="test-node")

        assert node.name == "test-node"
        assert node.tailscale_ip is None
        assert node.ssh_host is None
        assert node.ssh_user == "ubuntu"
        assert node.ssh_key is None
        assert node.ssh_port == 22
        assert node.ringrift_path == "~/ringrift/ai-service"
        assert node.status == "unknown"
        assert node.role == "unknown"
        assert node.memory_gb == 0
        assert node.cpus == 0
        assert node.gpu == ""
        assert node.data_server_port == DATA_SYNC_PORT
        assert node.data_server_url is None

    def test_full_initialization(self):
        """Test ClusterNode with all fields provided."""
        node = ClusterNode(
            name="runpod-h100",
            tailscale_ip="100.10.20.30",
            ssh_host="102.210.171.65",
            ssh_user="root",
            ssh_key="~/.ssh/id_ed25519",
            ssh_port=30178,
            ringrift_path="/workspace/ringrift/ai-service",
            status="active",
            role="training",
            memory_gb=80,
            cpus=16,
            gpu="H100",
            data_server_port=8765,
            data_server_url="http://custom-url:8765",
        )

        assert node.name == "runpod-h100"
        assert node.tailscale_ip == "100.10.20.30"
        assert node.ssh_host == "102.210.171.65"
        assert node.ssh_user == "root"
        assert node.ssh_key == "~/.ssh/id_ed25519"
        assert node.ssh_port == 30178
        assert node.ringrift_path == "/workspace/ringrift/ai-service"
        assert node.status == "active"
        assert node.role == "training"
        assert node.memory_gb == 80
        assert node.cpus == 16
        assert node.gpu == "H100"
        assert node.data_server_port == 8765
        assert node.data_server_url == "http://custom-url:8765"

    def test_best_ip_prefers_tailscale(self):
        """Test that best_ip prefers Tailscale IP over SSH host."""
        node = ClusterNode(
            name="test",
            tailscale_ip="100.10.20.30",
            ssh_host="192.168.1.100",
        )

        assert node.best_ip == "100.10.20.30"

    def test_best_ip_falls_back_to_ssh_host(self):
        """Test that best_ip falls back to SSH host when no Tailscale IP."""
        node = ClusterNode(
            name="test",
            tailscale_ip=None,
            ssh_host="192.168.1.100",
        )

        assert node.best_ip == "192.168.1.100"

    def test_best_ip_extracts_host_from_ssh_user_host(self):
        """Test that best_ip extracts host from 'user@host' format."""
        node = ClusterNode(
            name="test",
            tailscale_ip=None,
            ssh_host="ubuntu@192.168.1.100",
        )

        assert node.best_ip == "192.168.1.100"

    def test_best_ip_prefers_tailscale_even_with_ssh_user_host(self):
        """Test that best_ip prefers Tailscale even when SSH has user@host."""
        node = ClusterNode(
            name="test",
            tailscale_ip="100.10.20.30",
            ssh_host="ubuntu@192.168.1.100",
        )

        assert node.best_ip == "100.10.20.30"

    def test_best_ip_handles_empty_strings(self):
        """Test that best_ip handles empty strings correctly."""
        node = ClusterNode(
            name="test",
            tailscale_ip="",
            ssh_host="192.168.1.100",
        )

        assert node.best_ip == "192.168.1.100"

    def test_best_ip_handles_whitespace(self):
        """Test that best_ip strips whitespace."""
        node = ClusterNode(
            name="test",
            tailscale_ip="  ",
            ssh_host="  192.168.1.100  ",
        )

        assert node.best_ip == "192.168.1.100"

    def test_best_ip_returns_none_when_no_ips(self):
        """Test that best_ip returns None when no IPs available."""
        node = ClusterNode(name="test")

        assert node.best_ip is None

    def test_data_server_base_url_uses_custom_url(self):
        """Test that data_server_base_url uses custom URL if provided."""
        node = ClusterNode(
            name="test",
            tailscale_ip="100.10.20.30",
            data_server_url="http://custom:9999",
        )

        assert node.data_server_base_url == "http://custom:9999"

    def test_data_server_base_url_constructs_from_best_ip(self):
        """Test that data_server_base_url constructs URL from best_ip."""
        node = ClusterNode(
            name="test",
            tailscale_ip="100.10.20.30",
            data_server_port=8765,
        )

        assert node.data_server_base_url == "http://100.10.20.30:8765"

    def test_data_server_base_url_uses_ssh_host_when_no_tailscale(self):
        """Test that data_server_base_url uses SSH host as fallback."""
        node = ClusterNode(
            name="test",
            ssh_host="192.168.1.100",
            data_server_port=8766,
        )

        assert node.data_server_base_url == "http://192.168.1.100:8766"

    def test_data_server_base_url_returns_none_when_no_ip(self):
        """Test that data_server_base_url returns None when no IP available."""
        node = ClusterNode(name="test")

        assert node.data_server_base_url is None

    def test_is_active_true_for_active_status(self):
        """Test that is_active returns True for active nodes."""
        node = ClusterNode(name="test", status="active")
        assert node.is_active is True

    def test_is_active_true_for_running_status(self):
        """Test that is_active returns True for running nodes."""
        node = ClusterNode(name="test", status="running")
        assert node.is_active is True

    def test_is_active_false_for_terminated(self):
        """Test that is_active returns False for terminated nodes."""
        node = ClusterNode(name="test", status="terminated")
        assert node.is_active is False

    def test_is_active_false_for_offline(self):
        """Test that is_active returns False for offline nodes."""
        node = ClusterNode(name="test", status="offline")
        assert node.is_active is False

    def test_is_active_false_for_setup(self):
        """Test that is_active returns False for setup nodes."""
        node = ClusterNode(name="test", status="setup")
        assert node.is_active is False

    def test_is_active_true_for_unknown_status(self):
        """Test that is_active returns True for unknown status (conservative)."""
        node = ClusterNode(name="test", status="unknown")
        assert node.is_active is True


class TestEloSyncConfig:
    """Test EloSyncConfig dataclass."""

    def test_default_initialization(self):
        """Test EloSyncConfig with defaults."""
        config = EloSyncConfig()

        assert config.coordinator == "mac-studio"
        assert config.sync_port == 8766
        assert config.sync_interval == 300
        assert config.divergence_threshold == 50
        assert config.transports == ["tailscale", "aria2", "http"]

    def test_custom_initialization(self):
        """Test EloSyncConfig with custom values."""
        config = EloSyncConfig(
            coordinator="custom-node",
            sync_port=9999,
            sync_interval=600,
            divergence_threshold=100,
            transports=["http"],
        )

        assert config.coordinator == "custom-node"
        assert config.sync_port == 9999
        assert config.sync_interval == 600
        assert config.divergence_threshold == 100
        assert config.transports == ["http"]


class TestLoadHostsConfig:
    """Test load_hosts_config function."""

    def test_loads_valid_yaml_with_pyyaml(self, tmp_path, monkeypatch):
        """Test loading valid YAML using PyYAML."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    tailscale_ip: "100.10.20.30"
    ssh_host: "192.168.1.100"
    status: active
  node2:
    ssh_host: "192.168.1.101"
    status: offline

elo_sync:
  coordinator: node1
  sync_port: 8766
""")

        # Mock HOSTS_CONFIG path
        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert "hosts" in config
        assert "node1" in config["hosts"]
        assert config["hosts"]["node1"]["tailscale_ip"] == "100.10.20.30"
        assert config["hosts"]["node1"]["status"] == "active"
        assert "elo_sync" in config
        assert config["elo_sync"]["coordinator"] == "node1"

    def test_loads_nested_structure(self, tmp_path):
        """Test loading nested YAML structure with multiple nodes."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    tailscale_ip: "100.10.20.30"
    ssh_host: "192.168.1.100"
    ssh_port: 22
    status: active
  node2:
    ssh_host: "192.168.1.101"

elo_sync:
  coordinator: node1
  sync_port: 8766
  sync_interval: 300
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert "hosts" in config
        assert "node1" in config["hosts"]
        assert config["hosts"]["node1"]["tailscale_ip"] == "100.10.20.30"
        assert config["hosts"]["node1"]["ssh_port"] == 22
        assert "elo_sync" in config
        assert config["elo_sync"]["sync_port"] == 8766

    def test_handles_missing_config_file(self, tmp_path):
        """Test that missing config file returns empty dict."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", nonexistent):
            config = load_hosts_config_with_warning()

        assert config == {}

    def test_handles_empty_config_file(self, tmp_path):
        """Test that empty config file returns empty dict."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert config == {}

    def test_handles_comments_and_blank_lines(self, tmp_path):
        """Test that parser handles comments and blank lines."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
# This is a comment
hosts:
  # Node 1
  node1:
    tailscale_ip: "100.10.20.30"

  # Node 2
  node2:
    ssh_host: "192.168.1.101"

# Elo sync config
elo_sync:
  coordinator: node1
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert "hosts" in config
        assert len(config["hosts"]) == 2

    def test_handles_integer_values(self, tmp_path):
        """Test that integer values are parsed correctly."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    ssh_port: 22
    memory_gb: 80
    cpus: 16

elo_sync:
  sync_port: 8766
  sync_interval: 300
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert config["hosts"]["node1"]["ssh_port"] == 22
        assert config["hosts"]["node1"]["memory_gb"] == 80
        assert config["elo_sync"]["sync_port"] == 8766

    def test_handles_quoted_strings(self, tmp_path):
        """Test that quoted strings are handled correctly."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    tailscale_ip: "100.10.20.30"
    ssh_host: '192.168.1.100'
    gpu: "H100"
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        assert config["hosts"]["node1"]["tailscale_ip"] == "100.10.20.30"
        assert config["hosts"]["node1"]["ssh_host"] == "192.168.1.100"
        assert config["hosts"]["node1"]["gpu"] == "H100"

    def test_handles_corrupt_yaml(self, tmp_path):
        """Test that corrupt YAML returns empty dict instead of crashing."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("invalid: yaml: syntax: {{{")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = load_hosts_config_with_warning()

        # Should return empty dict on error
        assert config == {}


class TestGetEloSyncConfig:
    """Test get_elo_sync_config function."""

    def test_returns_defaults_when_no_config(self, tmp_path):
        """Test that defaults are used when no config file exists."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", nonexistent):
            config = get_elo_sync_config()

        assert config.coordinator == "mac-studio"
        assert config.sync_port == 8766
        assert config.sync_interval == 300
        assert config.divergence_threshold == 50
        assert config.transports == ["tailscale", "aria2", "http"]

    def test_loads_config_from_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
elo_sync:
  coordinator: custom-node
  sync_port: 9999
  sync_interval: 600
  divergence_threshold: 100
  transports:
    - http
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = get_elo_sync_config()

        assert config.coordinator == "custom-node"
        assert config.sync_port == 9999
        assert config.sync_interval == 600
        assert config.divergence_threshold == 100
        assert config.transports == ["http"]

    def test_merges_partial_config_with_defaults(self, tmp_path):
        """Test that partial config is merged with defaults."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
elo_sync:
  coordinator: partial-node
  sync_port: 7777
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            config = get_elo_sync_config()

        # Overridden values
        assert config.coordinator == "partial-node"
        assert config.sync_port == 7777
        # Default values
        assert config.sync_interval == 300
        assert config.divergence_threshold == 50


class TestGetClusterNodes:
    """Test get_cluster_nodes function."""

    def test_returns_empty_dict_when_no_hosts(self, tmp_path):
        """Test that empty dict is returned when no hosts configured."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            nodes = get_cluster_nodes()

        assert nodes == {}

    def test_creates_cluster_nodes_from_config(self, tmp_path):
        """Test creating ClusterNode objects from config."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    tailscale_ip: "100.10.20.30"
    ssh_host: "192.168.1.100"
    ssh_user: "root"
    ssh_port: 22
    status: active
    role: training
    memory_gb: 80
    cpus: 16
    gpu: "H100"

  node2:
    ssh_host: "192.168.1.101"
    status: offline
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            nodes = get_cluster_nodes()

        assert len(nodes) == 2
        assert "node1" in nodes
        assert "node2" in nodes

        node1 = nodes["node1"]
        assert node1.name == "node1"
        assert node1.tailscale_ip == "100.10.20.30"
        assert node1.ssh_host == "192.168.1.100"
        assert node1.ssh_user == "root"
        assert node1.status == "active"
        assert node1.memory_gb == 80

        node2 = nodes["node2"]
        assert node2.name == "node2"
        assert node2.ssh_host == "192.168.1.101"
        assert node2.status == "offline"

    def test_uses_default_values_for_missing_fields(self, tmp_path):
        """Test that default values are used for missing fields."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  minimal-node:
    ssh_host: "192.168.1.100"
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            nodes = get_cluster_nodes()

        node = nodes["minimal-node"]
        assert node.ssh_user == "ubuntu"  # default
        assert node.ssh_port == 22  # default
        assert node.ringrift_path == "~/ringrift/ai-service"  # default
        assert node.status == "unknown"  # default
        assert node.memory_gb == 0  # default

    def test_uses_config_data_server_port(self, tmp_path):
        """Test that data_server_port from config is used."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    ssh_host: "192.168.1.100"
    data_server_port: 9999
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            nodes = get_cluster_nodes()

        assert nodes["node1"].data_server_port == 9999

    @patch("app.sync.cluster_hosts._get_default_data_server_port")
    def test_uses_default_data_server_port_when_not_in_config(
        self, mock_get_default, tmp_path
    ):
        """Test that default data_server_port is used when not in config."""
        mock_get_default.return_value = 8888

        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    ssh_host: "192.168.1.100"
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            nodes = get_cluster_nodes()

        assert nodes["node1"].data_server_port == 8888
        mock_get_default.assert_called_once()


class TestGetActiveNodes:
    """Test get_active_nodes function."""

    def test_filters_out_terminated_nodes(self, tmp_path):
        """Test that terminated nodes are filtered out."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  active-node:
    ssh_host: "192.168.1.100"
    status: active
  terminated-node:
    ssh_host: "192.168.1.101"
    status: terminated
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            active_nodes = get_active_nodes()

        assert len(active_nodes) == 1
        assert active_nodes[0].name == "active-node"

    def test_filters_out_offline_nodes(self, tmp_path):
        """Test that offline nodes are filtered out."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  active-node:
    ssh_host: "192.168.1.100"
    status: active
  offline-node:
    ssh_host: "192.168.1.101"
    status: offline
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            active_nodes = get_active_nodes()

        assert len(active_nodes) == 1
        assert active_nodes[0].name == "active-node"

    def test_filters_out_setup_nodes(self, tmp_path):
        """Test that setup nodes are filtered out."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  active-node:
    ssh_host: "192.168.1.100"
    status: active
  setup-node:
    ssh_host: "192.168.1.101"
    status: setup
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            active_nodes = get_active_nodes()

        assert len(active_nodes) == 1
        assert active_nodes[0].name == "active-node"

    def test_includes_unknown_status_nodes(self, tmp_path):
        """Test that unknown status nodes are included (conservative)."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  active-node:
    ssh_host: "192.168.1.100"
    status: active
  unknown-node:
    ssh_host: "192.168.1.101"
    status: unknown
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            active_nodes = get_active_nodes()

        assert len(active_nodes) == 2
        names = {n.name for n in active_nodes}
        assert "active-node" in names
        assert "unknown-node" in names


class TestGetCoordinatorNode:
    """Test get_coordinator_node function."""

    def test_returns_coordinator_node(self, tmp_path):
        """Test that coordinator node is returned correctly."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    ssh_host: "192.168.1.100"
  coordinator-node:
    ssh_host: "192.168.1.101"
    status: active

elo_sync:
  coordinator: coordinator-node
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            coordinator = get_coordinator_node()

        assert coordinator is not None
        assert coordinator.name == "coordinator-node"
        assert coordinator.ssh_host == "192.168.1.101"

    def test_returns_none_when_coordinator_not_in_hosts(self, tmp_path):
        """Test that None is returned when coordinator not in hosts."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  node1:
    ssh_host: "192.168.1.100"

elo_sync:
  coordinator: nonexistent-node
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            coordinator = get_coordinator_node()

        assert coordinator is None

    def test_uses_default_coordinator_name(self, tmp_path):
        """Test that default coordinator name is used when not specified."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  mac-studio:
    ssh_host: "192.168.1.100"
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            coordinator = get_coordinator_node()

        assert coordinator is not None
        assert coordinator.name == "mac-studio"


class TestGetCoordinatorAddress:
    """Test get_coordinator_address function."""

    def test_returns_coordinator_ip_and_port(self, tmp_path):
        """Test that coordinator IP and port are returned."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  coordinator-node:
    tailscale_ip: "100.10.20.30"

elo_sync:
  coordinator: coordinator-node
  sync_port: 8766
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            ip, port = get_coordinator_address()

        assert ip == "100.10.20.30"
        assert port == 8766

    def test_uses_ssh_host_when_no_tailscale(self, tmp_path):
        """Test that SSH host is used when no Tailscale IP."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
hosts:
  coordinator-node:
    ssh_host: "192.168.1.100"

elo_sync:
  coordinator: coordinator-node
  sync_port: 8766
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            ip, port = get_coordinator_address()

        assert ip == "192.168.1.100"
        assert port == 8766

    def test_falls_back_to_env_var(self, tmp_path, monkeypatch):
        """Test that environment variable is used as fallback."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
elo_sync:
  sync_port: 8766
""")

        monkeypatch.setenv("RINGRIFT_COORDINATOR_IP", "10.0.0.1")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            ip, port = get_coordinator_address()

        assert ip == "10.0.0.1"
        assert port == 8766

    def test_returns_none_ip_when_no_coordinator(self, tmp_path):
        """Test that None IP is returned when no coordinator configured."""
        config_file = tmp_path / "distributed_hosts.yaml"
        config_file.write_text("""
elo_sync:
  sync_port: 8766
""")

        with patch("app.sync.cluster_hosts.HOSTS_CONFIG", config_file):
            ip, port = get_coordinator_address()

        assert ip is None
        assert port == 8766


class TestPortConstants:
    """Test that port constants have expected values."""

    def test_elo_sync_port(self):
        """Test ELO_SYNC_PORT constant."""
        assert ELO_SYNC_PORT == 8766

    def test_data_sync_port(self):
        """Test DATA_SYNC_PORT constant."""
        assert DATA_SYNC_PORT == 8766

    def test_model_sync_port(self):
        """Test MODEL_SYNC_PORT constant."""
        assert MODEL_SYNC_PORT == 8765
