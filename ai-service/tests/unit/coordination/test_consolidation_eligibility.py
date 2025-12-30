"""Unit tests for ConsolidationEligibilityManager (December 2025).

Tests the consolidation eligibility manager for distributed data pipeline.

Created: December 30, 2025
"""

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.coordination.consolidation_eligibility import (
    ConsolidationConfig,
    ConsolidationEligibilityManager,
    DEFAULT_ELIGIBLE_ROLES,
    EligibilityResult,
    NodeConsolidationInfo,
    get_eligibility_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_root_path() -> Path:
    """Create a temporary root path with config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_dir = root / "config"
        config_dir.mkdir(parents=True)
        yield root


@pytest.fixture
def config() -> ConsolidationConfig:
    """Create a test configuration."""
    return ConsolidationConfig(
        enabled=True,
        min_disk_free_gb=10.0,
        max_cpu_percent=90.0,
        min_games_to_consolidate=20,
        max_games_per_batch=1000,
        eligible_roles=["coordinator", "gpu_training"],
        excluded_nodes=["excluded-node"],
        prefer_nodes_with_most_data=True,
        consolidation_interval_seconds=120,
    )


@pytest.fixture
def manager(temp_root_path: Path, config: ConsolidationConfig) -> ConsolidationEligibilityManager:
    """Create a test manager with reset singleton."""
    ConsolidationEligibilityManager.reset_instance()
    return ConsolidationEligibilityManager(root_path=temp_root_path, config=config)


@pytest.fixture
def sample_hosts_yaml(temp_root_path: Path) -> Path:
    """Create a sample distributed_hosts.yaml file."""
    config_path = temp_root_path / "config" / "distributed_hosts.yaml"

    hosts_config = {
        "consolidation": {
            "enabled": True,
            "min_disk_free_gb": 15.0,
            "max_cpu_percent": 85.0,
            "eligible_roles": ["coordinator", "gpu_training"],
        },
        "hosts": {
            "coordinator-1": {
                "status": "ready",
                "role": "coordinator",
                "tailscale_ip": "100.1.1.1",
                "consolidation_enabled": True,
            },
            "gpu-node-1": {
                "status": "active",
                "role": "gpu_training",
                "tailscale_ip": "100.1.1.2",
            },
            "gpu-node-2": {
                "status": "offline",
                "role": "gpu_training",
                "tailscale_ip": "100.1.1.3",
            },
            "selfplay-node": {
                "status": "ready",
                "role": "gpu_selfplay",
                "tailscale_ip": "100.1.1.4",
            },
            "disabled-node": {
                "status": "ready",
                "role": "coordinator",
                "tailscale_ip": "100.1.1.5",
                "consolidation_enabled": False,
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(hosts_config, f)

    return config_path


@pytest.fixture(autouse=True)
def reset_manager_singleton():
    """Reset manager singleton after each test."""
    yield
    ConsolidationEligibilityManager.reset_instance()


# ============================================================================
# ConsolidationConfig Tests
# ============================================================================


class TestConsolidationConfig:
    """Tests for ConsolidationConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = ConsolidationConfig()

        assert config.enabled is True
        assert config.min_disk_free_gb == 20.0
        assert config.max_cpu_percent == 80.0
        assert config.min_games_to_consolidate == 50
        assert config.max_games_per_batch == 5000
        assert config.eligible_roles == DEFAULT_ELIGIBLE_ROLES
        assert config.excluded_nodes == []
        assert config.prefer_nodes_with_most_data is True
        assert config.consolidation_interval_seconds == 300

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConsolidationConfig(
            enabled=False,
            min_disk_free_gb=30.0,
            max_cpu_percent=70.0,
            min_games_to_consolidate=100,
            excluded_nodes=["node1", "node2"],
        )

        assert config.enabled is False
        assert config.min_disk_free_gb == 30.0
        assert config.max_cpu_percent == 70.0
        assert config.min_games_to_consolidate == 100
        assert config.excluded_nodes == ["node1", "node2"]

    def test_from_yaml_config_full(self):
        """Test loading from YAML config dictionary."""
        yaml_config = {
            "enabled": True,
            "min_disk_free_gb": 25.0,
            "max_cpu_percent": 75.0,
            "min_games_to_consolidate": 75,
            "max_games_per_batch": 2000,
            "eligible_roles": ["coordinator"],
            "excluded_nodes": ["bad-node"],
            "prefer_nodes_with_most_data": False,
            "consolidation_interval_seconds": 600,
        }

        config = ConsolidationConfig.from_yaml_config(yaml_config)

        assert config.enabled is True
        assert config.min_disk_free_gb == 25.0
        assert config.max_cpu_percent == 75.0
        assert config.min_games_to_consolidate == 75
        assert config.max_games_per_batch == 2000
        assert config.eligible_roles == ["coordinator"]
        assert config.excluded_nodes == ["bad-node"]
        assert config.prefer_nodes_with_most_data is False
        assert config.consolidation_interval_seconds == 600

    def test_from_yaml_config_partial(self):
        """Test loading from partial YAML config (uses defaults)."""
        yaml_config = {
            "enabled": False,
            "min_disk_free_gb": 10.0,
        }

        config = ConsolidationConfig.from_yaml_config(yaml_config)

        assert config.enabled is False
        assert config.min_disk_free_gb == 10.0
        # Should use defaults for missing keys
        assert config.max_cpu_percent == 80.0
        assert config.min_games_to_consolidate == 50

    def test_from_yaml_config_empty(self):
        """Test loading from empty YAML config."""
        config = ConsolidationConfig.from_yaml_config({})

        assert config.enabled is True
        assert config.min_disk_free_gb == 20.0


# ============================================================================
# EligibilityResult Tests
# ============================================================================


class TestEligibilityResult:
    """Tests for EligibilityResult dataclass."""

    def test_eligible_result(self):
        """Test creating an eligible result."""
        result = EligibilityResult(
            is_eligible=True,
            reason="All criteria met",
            disk_free_gb=50.0,
            role="coordinator",
        )

        assert result.is_eligible is True
        assert result.reason == "All criteria met"
        assert result.disk_free_gb == 50.0
        assert result.role == "coordinator"

    def test_ineligible_result(self):
        """Test creating an ineligible result."""
        result = EligibilityResult(
            is_eligible=False,
            reason="Node status is offline",
        )

        assert result.is_eligible is False
        assert result.reason == "Node status is offline"
        assert result.disk_free_gb is None
        assert result.cpu_percent is None
        assert result.role is None

    def test_serialization(self):
        """Test result serialization."""
        result = EligibilityResult(
            is_eligible=True,
            reason="OK",
            disk_free_gb=25.5,
        )

        result_dict = asdict(result)
        assert result_dict["is_eligible"] is True
        assert result_dict["reason"] == "OK"
        assert result_dict["disk_free_gb"] == 25.5


# ============================================================================
# NodeConsolidationInfo Tests
# ============================================================================


class TestNodeConsolidationInfo:
    """Tests for NodeConsolidationInfo dataclass."""

    def test_full_info(self):
        """Test creating full node info."""
        info = NodeConsolidationInfo(
            node_id="gpu-node-1",
            is_eligible=True,
            reason="All criteria met",
            disk_free_gb=100.0,
            game_counts={"hex8_2p": 500, "square8_4p": 300},
            role="gpu_training",
        )

        assert info.node_id == "gpu-node-1"
        assert info.is_eligible is True
        assert info.disk_free_gb == 100.0
        assert info.game_counts["hex8_2p"] == 500
        assert info.role == "gpu_training"

    def test_minimal_info(self):
        """Test creating minimal node info."""
        info = NodeConsolidationInfo(
            node_id="test-node",
            is_eligible=False,
            reason="Node offline",
            disk_free_gb=0.0,
            game_counts={},
        )

        assert info.node_id == "test-node"
        assert info.is_eligible is False
        assert info.game_counts == {}
        assert info.role is None


# ============================================================================
# ConsolidationEligibilityManager Initialization Tests
# ============================================================================


class TestManagerInit:
    """Tests for ConsolidationEligibilityManager initialization."""

    def test_init_with_config(self, temp_root_path: Path, config: ConsolidationConfig):
        """Test initialization with custom config."""
        manager = ConsolidationEligibilityManager(
            root_path=temp_root_path,
            config=config,
        )

        assert manager.config == config
        assert manager.root_path == temp_root_path

    def test_init_with_default_root_path(self, config: ConsolidationConfig):
        """Test initialization with default root path."""
        manager = ConsolidationEligibilityManager(config=config)

        # Should use default path (parent.parent of voter_health_daemon.py)
        assert manager.root_path.exists()

    def test_init_loads_from_yaml(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test initialization loads config from YAML."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        # Should have loaded config from YAML
        assert manager.config.min_disk_free_gb == 15.0
        assert manager.config.max_cpu_percent == 85.0

    def test_init_loads_host_configs(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test initialization loads host configs."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        # Should have loaded hosts
        assert "coordinator-1" in manager._host_configs
        assert "gpu-node-1" in manager._host_configs


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_instance(self, temp_root_path: Path, config: ConsolidationConfig):
        """Test get_instance returns singleton."""
        # Create first instance manually
        manager1 = ConsolidationEligibilityManager(
            root_path=temp_root_path,
            config=config,
        )

        # Store as singleton
        ConsolidationEligibilityManager._instance = manager1

        # Should return same instance
        manager2 = ConsolidationEligibilityManager.get_instance()
        assert manager1 is manager2

    def test_reset_instance(self, temp_root_path: Path, config: ConsolidationConfig):
        """Test reset_instance clears singleton."""
        manager1 = ConsolidationEligibilityManager(
            root_path=temp_root_path,
            config=config,
        )
        ConsolidationEligibilityManager._instance = manager1

        ConsolidationEligibilityManager.reset_instance()

        assert ConsolidationEligibilityManager._instance is None

    def test_get_eligibility_manager(self):
        """Test module-level accessor."""
        manager = get_eligibility_manager()
        assert isinstance(manager, ConsolidationEligibilityManager)


# ============================================================================
# Eligibility Checking Tests
# ============================================================================


class TestEligibilityChecking:
    """Tests for eligibility checking."""

    def test_disabled_consolidation(self, manager: ConsolidationEligibilityManager):
        """Test eligibility when consolidation is disabled."""
        manager.config = ConsolidationConfig(enabled=False)

        is_eligible, reason = manager.is_node_eligible("any-node")

        assert is_eligible is False
        assert "disabled globally" in reason.lower()

    def test_excluded_node(self, manager: ConsolidationEligibilityManager):
        """Test eligibility for excluded node."""
        manager.config.excluded_nodes = ["excluded-node"]
        manager._host_configs["excluded-node"] = {"status": "ready", "role": "coordinator"}

        is_eligible, reason = manager.is_node_eligible("excluded-node")

        assert is_eligible is False
        assert "excluded" in reason.lower()

    def test_unknown_node(self, manager: ConsolidationEligibilityManager):
        """Test eligibility for unknown node."""
        is_eligible, reason = manager.is_node_eligible("unknown-node")

        assert is_eligible is False
        assert "not found" in reason.lower()

    def test_offline_node(self, manager: ConsolidationEligibilityManager):
        """Test eligibility for offline node."""
        manager._host_configs["offline-node"] = {"status": "offline", "role": "coordinator"}

        is_eligible, reason = manager.is_node_eligible("offline-node")

        assert is_eligible is False
        assert "offline" in reason.lower()

    def test_disabled_per_node(self, manager: ConsolidationEligibilityManager):
        """Test eligibility when disabled per-node."""
        manager._host_configs["disabled-node"] = {
            "status": "ready",
            "role": "coordinator",
            "consolidation_enabled": False,
        }

        is_eligible, reason = manager.is_node_eligible("disabled-node")

        assert is_eligible is False
        assert "explicitly disabled" in reason.lower()

    def test_ineligible_role(self, manager: ConsolidationEligibilityManager):
        """Test eligibility for ineligible role."""
        manager.config.eligible_roles = ["coordinator"]
        manager._host_configs["worker-node"] = {
            "status": "ready",
            "role": "gpu_selfplay",
        }

        is_eligible, reason = manager.is_node_eligible("worker-node")

        assert is_eligible is False
        assert "role" in reason.lower()

    def test_eligible_role_override(self, manager: ConsolidationEligibilityManager):
        """Test eligibility with role override."""
        manager.config.eligible_roles = ["coordinator"]
        manager._host_configs["worker-node"] = {
            "status": "ready",
            "role": "gpu_selfplay",
            "consolidation_enabled": True,  # Override
        }

        is_eligible, reason = manager.is_node_eligible("worker-node")

        assert is_eligible is True

    def test_eligible_node(self, manager: ConsolidationEligibilityManager):
        """Test eligibility for valid node."""
        manager._host_configs["good-node"] = {
            "status": "ready",
            "role": "coordinator",
        }

        is_eligible, reason = manager.is_node_eligible("good-node")

        assert is_eligible is True
        assert "criteria met" in reason.lower()

    def test_low_disk_space(self, temp_root_path: Path, config: ConsolidationConfig):
        """Test eligibility with low disk space."""
        config.min_disk_free_gb = 1000.0  # Very high threshold
        manager = ConsolidationEligibilityManager(root_path=temp_root_path, config=config)
        manager._host_configs["local-node"] = {
            "status": "ready",
            "role": "coordinator",
        }

        # Mock env.node_id to match our test node
        with patch("app.config.env.env") as mock_env:
            mock_env.node_id = "local-node"

            is_eligible, reason = manager.is_node_eligible("local-node")

            # Should fail disk space check (likely < 1000GB free)
            assert is_eligible is False
            assert "disk" in reason.lower()


# ============================================================================
# Get Eligible Nodes Tests
# ============================================================================


class TestGetEligibleNodes:
    """Tests for get_eligible_nodes method."""

    def test_get_eligible_nodes(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test getting list of eligible nodes."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        eligible = manager.get_eligible_nodes()

        # Should include coordinator-1 and gpu-node-1 (ready/active with eligible roles)
        # Should exclude gpu-node-2 (offline) and disabled-node (consolidation_enabled=False)
        # and selfplay-node (role not eligible)
        assert "coordinator-1" in eligible
        assert "gpu-node-1" in eligible
        assert "gpu-node-2" not in eligible
        assert "disabled-node" not in eligible

    def test_get_eligible_nodes_empty(self, manager: ConsolidationEligibilityManager):
        """Test when no nodes are eligible."""
        manager._host_configs = {}

        eligible = manager.get_eligible_nodes()

        assert eligible == []


# ============================================================================
# Best Consolidation Node Tests
# ============================================================================


class TestGetBestConsolidationNode:
    """Tests for get_best_consolidation_node method."""

    def test_best_node_with_most_data(self, manager: ConsolidationEligibilityManager):
        """Test selecting node with most data."""
        manager._host_configs = {
            "node-a": {"status": "ready", "role": "coordinator"},
            "node-b": {"status": "ready", "role": "coordinator"},
        }
        manager.config.prefer_nodes_with_most_data = True

        # Mock game counts
        with patch.object(manager, "_get_node_game_counts") as mock_counts:
            mock_counts.side_effect = lambda node_id: {
                "node-a": {"hex8_2p": 100},
                "node-b": {"hex8_2p": 500},  # More data
            }.get(node_id, {})

            best = manager.get_best_consolidation_node("hex8_2p")

            # Should prefer node-b (more data)
            assert best == "node-b"

    def test_best_node_by_disk_space(self, manager: ConsolidationEligibilityManager):
        """Test selecting node by disk space when data preference disabled."""
        manager._host_configs = {
            "node-a": {"status": "ready", "role": "coordinator"},
            "node-b": {"status": "ready", "role": "coordinator"},
        }
        manager.config.prefer_nodes_with_most_data = False

        # Best node will be alphabetically first (same disk space in mock)
        with patch.object(manager, "_get_node_game_counts", return_value={}):
            best = manager.get_best_consolidation_node("hex8_2p")

            # Should return one of the nodes
            assert best in ["node-a", "node-b"]

    def test_best_node_with_exclusions(self, manager: ConsolidationEligibilityManager):
        """Test selecting best node with exclusions."""
        manager._host_configs = {
            "node-a": {"status": "ready", "role": "coordinator"},
            "node-b": {"status": "ready", "role": "coordinator"},
        }

        with patch.object(manager, "_get_node_game_counts", return_value={}):
            best = manager.get_best_consolidation_node("hex8_2p", exclude_nodes=["node-a"])

            assert best == "node-b"

    def test_best_node_no_eligible(self, manager: ConsolidationEligibilityManager):
        """Test when no nodes are eligible."""
        manager._host_configs = {}

        best = manager.get_best_consolidation_node("hex8_2p")

        assert best is None


# ============================================================================
# Node Game Counts Tests
# ============================================================================


class TestGetNodeGameCounts:
    """Tests for _get_node_game_counts method."""

    def test_get_game_counts_success(self, manager: ConsolidationEligibilityManager):
        """Test getting game counts from manifest."""
        mock_manifest = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([
            ("hex8_2p", 100),
            ("square8_4p", 200),
        ])

        mock_conn.execute.return_value = mock_cursor
        mock_manifest._connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_manifest._connection.return_value.__exit__ = MagicMock(return_value=None)

        with patch(
            "app.distributed.cluster_manifest.get_cluster_manifest",
            return_value=mock_manifest,
        ):
            counts = manager._get_node_game_counts("test-node")

            assert counts.get("hex8_2p", 0) == 100
            assert counts.get("square8_4p", 0) == 200

    def test_get_game_counts_error(self, manager: ConsolidationEligibilityManager):
        """Test handling errors when getting game counts."""
        with patch(
            "app.distributed.cluster_manifest.get_cluster_manifest",
            side_effect=ImportError("Module not found"),
        ):
            counts = manager._get_node_game_counts("test-node")

            assert counts == {}


# ============================================================================
# Get All Node Info Tests
# ============================================================================


class TestGetAllNodeInfo:
    """Tests for get_all_node_info method."""

    def test_get_all_node_info(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test getting info for all configured nodes."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        with patch.object(manager, "_get_node_game_counts", return_value={"hex8_2p": 50}):
            info_list = manager.get_all_node_info()

        # Should have info for all configured nodes
        node_ids = [info.node_id for info in info_list]
        assert "coordinator-1" in node_ids
        assert "gpu-node-1" in node_ids
        assert "gpu-node-2" in node_ids

        # Check individual node info
        coord_info = next(i for i in info_list if i.node_id == "coordinator-1")
        assert coord_info.is_eligible is True
        assert coord_info.game_counts == {"hex8_2p": 50}

        offline_info = next(i for i in info_list if i.node_id == "gpu-node-2")
        assert offline_info.is_eligible is False

    def test_get_all_node_info_empty(self, manager: ConsolidationEligibilityManager):
        """Test when no nodes configured."""
        manager._host_configs = {}

        info_list = manager.get_all_node_info()

        assert info_list == []


# ============================================================================
# Config Loading Tests
# ============================================================================


class TestConfigLoading:
    """Tests for config loading from YAML."""

    def test_load_config_missing_file(self, temp_root_path: Path):
        """Test loading config when file doesn't exist."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        # Should use default config
        assert manager.config.enabled is True
        assert manager.config.min_disk_free_gb == 20.0

    def test_load_config_invalid_yaml(self, temp_root_path: Path):
        """Test loading config with invalid YAML."""
        config_path = temp_root_path / "config" / "distributed_hosts.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        # Should use default config on error
        assert manager.config.min_disk_free_gb == 20.0

    def test_load_host_configs_missing_file(self, manager: ConsolidationEligibilityManager):
        """Test loading host configs when file doesn't exist."""
        # Already called during init, but hosts should be empty
        # if no file existed at that path
        manager._host_configs = {}  # Reset
        manager._load_host_configs()

        # Should remain empty or have whatever was loaded
        assert isinstance(manager._host_configs, dict)


# ============================================================================
# Disk Space Tests
# ============================================================================


class TestDiskSpace:
    """Tests for disk space checking."""

    def test_get_local_disk_free_non_local(self, manager: ConsolidationEligibilityManager):
        """Test disk space check for non-local node."""
        with patch("app.config.env.env") as mock_env:
            mock_env.node_id = "other-node"

            result = manager._get_local_disk_free_gb("test-node")

            # Should return None for non-local node
            assert result is None

    def test_get_local_disk_free_local(self, manager: ConsolidationEligibilityManager):
        """Test disk space check for local node."""
        with patch("app.config.env.env") as mock_env:
            mock_env.node_id = "test-node"

            result = manager._get_local_disk_free_gb("test-node")

            # Should return actual disk space (positive number)
            assert result is not None
            assert result > 0

    def test_get_local_disk_free_error(self, manager: ConsolidationEligibilityManager):
        """Test disk space check when error occurs."""
        with patch("app.config.env.env") as mock_env:
            mock_env.node_id = "test-node"

            with patch("shutil.disk_usage", side_effect=OSError("Disk error")):
                result = manager._get_local_disk_free_gb("test-node")

                # Should return None on error
                assert result is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_eligibility_workflow(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test full eligibility check workflow."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        # Get all eligible nodes
        eligible = manager.get_eligible_nodes()
        assert len(eligible) > 0

        # Check each eligible node
        for node_id in eligible:
            is_eligible, reason = manager.is_node_eligible(node_id)
            assert is_eligible is True

        # Get all node info
        with patch.object(manager, "_get_node_game_counts", return_value={}):
            all_info = manager.get_all_node_info()

        # Verify eligible nodes match
        eligible_from_info = [i.node_id for i in all_info if i.is_eligible]
        assert set(eligible) == set(eligible_from_info)

    def test_best_node_selection_workflow(self, temp_root_path: Path, sample_hosts_yaml: Path):
        """Test best node selection workflow."""
        manager = ConsolidationEligibilityManager(root_path=temp_root_path)

        game_counts = {
            "coordinator-1": {"hex8_2p": 100},
            "gpu-node-1": {"hex8_2p": 500},
        }

        with patch.object(
            manager,
            "_get_node_game_counts",
            side_effect=lambda node_id: game_counts.get(node_id, {}),
        ):
            best = manager.get_best_consolidation_node("hex8_2p")

            # Should select gpu-node-1 (more data for config)
            assert best == "gpu-node-1"
