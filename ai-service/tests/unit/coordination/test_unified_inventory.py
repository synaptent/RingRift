"""Tests for unified_inventory module (December 2025).

Tests the UnifiedInventory class that discovers nodes from multiple sources
(Vast, Tailscale, Lambda, Hetzner) and maintains a unified registry.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDiscoveredNode:
    """Tests for DiscoveredNode dataclass."""

    def test_basic_creation(self):
        """Test creating a DiscoveredNode with minimal fields."""
        from app.coordination.unified_inventory import DiscoveredNode

        node = DiscoveredNode(
            node_id="vast-12345",
            host="192.168.1.1",
        )
        assert node.node_id == "vast-12345"
        assert node.host == "192.168.1.1"
        assert node.port == 8770  # P2P_DEFAULT_PORT
        assert node.source == "unknown"
        assert node.status == "unknown"

    def test_full_creation(self):
        """Test creating a DiscoveredNode with all fields."""
        from app.coordination.unified_inventory import DiscoveredNode

        node = DiscoveredNode(
            node_id="runpod-h100",
            host="10.0.0.1",
            port=8770,
            source="tailscale",
            ssh_host="10.0.0.1",
            ssh_port=22,
            ssh_user="root",
            tailscale_ip="100.64.1.1",
            gpu_name="H100",
            num_gpus=2,
            memory_gb=128,
            vcpus=32,
            status="running",
            gpu_percent=75.5,
            cpu_percent=45.2,
            selfplay_jobs=3,
            training_jobs=1,
            role="nn_training_primary",
            vast_instance_id="",
            p2p_healthy=True,
            retired=False,
        )
        assert node.gpu_name == "H100"
        assert node.num_gpus == 2
        assert node.memory_gb == 128
        assert node.p2p_healthy is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        from app.coordination.unified_inventory import DiscoveredNode

        node = DiscoveredNode(
            node_id="vast-12345",
            host="192.168.1.1",
            source="vast",
            gpu_name="RTX 4090",
            status="running",
            gpu_percent=50.0,
            selfplay_jobs=2,
            role="nn_training_primary",
        )
        d = node.to_dict()

        assert d["node_id"] == "vast-12345"
        assert d["host"] == "192.168.1.1"
        assert d["source"] == "vast"
        assert d["gpu_name"] == "RTX 4090"
        assert d["gpu_percent"] == 50.0
        assert d["selfplay_jobs"] == 2

    def test_last_seen_default(self):
        """Test last_seen field has a default timestamp."""
        from app.coordination.unified_inventory import DiscoveredNode

        before = time.time()
        node = DiscoveredNode(node_id="test", host="1.1.1.1")
        after = time.time()

        assert before <= node.last_seen <= after


class TestGPURoles:
    """Tests for GPU_ROLES mapping."""

    def test_gpu_roles_exist(self):
        """Test GPU_ROLES contains expected mappings."""
        from app.coordination.unified_inventory import GPU_ROLES

        assert "RTX 4090" not in GPU_ROLES  # Not in the default list
        assert "H100" in GPU_ROLES
        assert "A100" in GPU_ROLES
        assert GPU_ROLES["H100"] == "nn_training_primary"
        assert GPU_ROLES["RTX 3060"] == "gpu_selfplay"

    def test_gpu_roles_coverage(self):
        """Test GPU_ROLES covers main GPU types."""
        from app.coordination.unified_inventory import GPU_ROLES

        # Training-class GPUs
        training_gpus = ["H100", "A100", "A40", "GH200", "RTX 5090"]
        for gpu in training_gpus:
            assert gpu in GPU_ROLES
            assert GPU_ROLES[gpu] == "nn_training_primary"

        # Selfplay-class GPUs
        selfplay_gpus = ["RTX 3060", "RTX 3070", "RTX 2060S"]
        for gpu in selfplay_gpus:
            assert gpu in GPU_ROLES
            assert GPU_ROLES[gpu] == "gpu_selfplay"


class TestUnifiedInventory:
    """Tests for UnifiedInventory class."""

    @pytest.fixture
    def inventory(self):
        """Create a fresh UnifiedInventory instance with mocked config loading."""
        from app.coordination.unified_inventory import UnifiedInventory

        with patch.object(UnifiedInventory, '_load_configs'):
            inv = UnifiedInventory()
            inv._distributed_hosts = {}
            inv._cluster_nodes = {}
            yield inv

    def test_init(self, inventory):
        """Test UnifiedInventory initialization."""
        assert inventory._nodes == {}
        assert inventory._last_discovery == 0.0

    def test_get_all_nodes_empty(self, inventory):
        """Test get_all_nodes returns empty dict initially."""
        assert inventory.get_all_nodes() == {}

    def test_get_node_not_found(self, inventory):
        """Test get_node returns None for unknown node."""
        assert inventory.get_node("unknown-node") is None

    def test_get_node_case_insensitive(self, inventory):
        """Test get_node is case-insensitive."""
        from app.coordination.unified_inventory import DiscoveredNode

        inventory._nodes["vast-12345"] = DiscoveredNode(
            node_id="vast-12345",
            host="1.1.1.1",
        )
        assert inventory.get_node("VAST-12345") is not None
        assert inventory.get_node("vast-12345") is not None

    def test_get_idle_nodes_empty(self, inventory):
        """Test get_idle_nodes returns empty list initially."""
        assert inventory.get_idle_nodes() == []

    def test_get_idle_nodes_filters_correctly(self, inventory):
        """Test get_idle_nodes filters by GPU utilization and jobs."""
        from app.coordination.unified_inventory import DiscoveredNode

        # Idle node - low GPU, no jobs
        idle_node = DiscoveredNode(
            node_id="idle-1",
            host="1.1.1.1",
            gpu_name="RTX 4090",
            gpu_percent=5.0,
            selfplay_jobs=0,
            training_jobs=0,
        )
        # Busy node - high GPU
        busy_node = DiscoveredNode(
            node_id="busy-1",
            host="2.2.2.2",
            gpu_name="RTX 4090",
            gpu_percent=80.0,
            selfplay_jobs=0,
            training_jobs=0,
        )
        # Node with jobs
        working_node = DiscoveredNode(
            node_id="working-1",
            host="3.3.3.3",
            gpu_name="RTX 4090",
            gpu_percent=5.0,
            selfplay_jobs=2,
            training_jobs=0,
        )
        # Retired node
        retired_node = DiscoveredNode(
            node_id="retired-1",
            host="4.4.4.4",
            gpu_name="RTX 4090",
            gpu_percent=5.0,
            selfplay_jobs=0,
            training_jobs=0,
            retired=True,
        )
        # CPU-only node (no GPU name)
        cpu_node = DiscoveredNode(
            node_id="cpu-1",
            host="5.5.5.5",
            gpu_name="",
            gpu_percent=0.0,
            selfplay_jobs=0,
            training_jobs=0,
        )

        inventory._nodes = {
            "idle-1": idle_node,
            "busy-1": busy_node,
            "working-1": working_node,
            "retired-1": retired_node,
            "cpu-1": cpu_node,
        }

        idle = inventory.get_idle_nodes(gpu_threshold=10.0)
        assert len(idle) == 1
        assert idle[0].node_id == "idle-1"

    def test_get_nodes_by_source(self, inventory):
        """Test get_nodes_by_source filters by source."""
        from app.coordination.unified_inventory import DiscoveredNode

        vast_node = DiscoveredNode(node_id="vast-1", host="1.1.1.1", source="vast")
        ts_node = DiscoveredNode(node_id="ts-1", host="2.2.2.2", source="tailscale")

        inventory._nodes = {"vast-1": vast_node, "ts-1": ts_node}

        vast_nodes = inventory.get_nodes_by_source("vast")
        assert len(vast_nodes) == 1
        assert vast_nodes[0].node_id == "vast-1"

        ts_nodes = inventory.get_nodes_by_source("tailscale")
        assert len(ts_nodes) == 1
        assert ts_nodes[0].node_id == "ts-1"

    def test_get_healthy_nodes(self, inventory):
        """Test get_healthy_nodes filters correctly."""
        from app.coordination.unified_inventory import DiscoveredNode

        healthy = DiscoveredNode(node_id="healthy-1", host="1.1.1.1", p2p_healthy=True)
        unhealthy = DiscoveredNode(node_id="unhealthy-1", host="2.2.2.2", p2p_healthy=False)
        retired = DiscoveredNode(node_id="retired-1", host="3.3.3.3", p2p_healthy=True, retired=True)

        inventory._nodes = {
            "healthy-1": healthy,
            "unhealthy-1": unhealthy,
            "retired-1": retired,
        }

        result = inventory.get_healthy_nodes()
        assert len(result) == 1
        assert result[0].node_id == "healthy-1"

    def test_get_status_summary(self, inventory):
        """Test get_status_summary returns correct stats."""
        from app.coordination.unified_inventory import DiscoveredNode

        inventory._nodes = {
            "vast-1": DiscoveredNode(
                node_id="vast-1", host="1.1.1.1", source="vast",
                gpu_percent=5.0, selfplay_jobs=0, p2p_healthy=True
            ),
            "vast-2": DiscoveredNode(
                node_id="vast-2", host="2.2.2.2", source="vast",
                gpu_percent=80.0, selfplay_jobs=2, p2p_healthy=True
            ),
            "ts-1": DiscoveredNode(
                node_id="ts-1", host="3.3.3.3", source="tailscale",
                gpu_percent=5.0, selfplay_jobs=0, p2p_healthy=False
            ),
        }
        inventory._last_discovery = 1234567890.0

        summary = inventory.get_status_summary()

        assert summary["total_nodes"] == 3
        assert summary["by_source"]["vast"] == 2
        assert summary["by_source"]["tailscale"] == 1
        assert summary["idle_nodes"] == 2  # vast-1 and ts-1 have low GPU and no jobs
        assert summary["healthy_nodes"] == 2  # vast-1 and vast-2 are healthy
        assert summary["last_discovery"] == 1234567890.0


class TestMergeNodes:
    """Tests for node merging logic."""

    @pytest.fixture
    def inventory(self):
        """Create a fresh UnifiedInventory instance with mocked config loading."""
        from app.coordination.unified_inventory import UnifiedInventory

        with patch.object(UnifiedInventory, '_load_configs'):
            inv = UnifiedInventory()
            inv._distributed_hosts = {}
            yield inv

    def test_merge_two_nodes_newer_wins(self, inventory):
        """Test _merge_two_nodes prefers newer status info."""
        from app.coordination.unified_inventory import DiscoveredNode

        old = DiscoveredNode(
            node_id="test", host="1.1.1.1",
            gpu_percent=10.0, last_seen=1000.0
        )
        new = DiscoveredNode(
            node_id="test", host="1.1.1.1",
            gpu_percent=50.0, last_seen=2000.0
        )

        result = inventory._merge_two_nodes(old, new)
        assert result.gpu_percent == 50.0
        assert result.last_seen == 2000.0

    def test_merge_two_nodes_fills_missing_info(self, inventory):
        """Test _merge_two_nodes fills in missing info."""
        from app.coordination.unified_inventory import DiscoveredNode

        existing = DiscoveredNode(
            node_id="test", host="1.1.1.1",
            gpu_name="", tailscale_ip="",
            last_seen=2000.0,
        )
        new = DiscoveredNode(
            node_id="test", host="1.1.1.1",
            gpu_name="RTX 4090", tailscale_ip="100.64.1.1",
            last_seen=1000.0,  # Older
        )

        result = inventory._merge_two_nodes(existing, new)
        # Fills in missing info from new even though it's older
        assert result.gpu_name == "RTX 4090"
        assert result.tailscale_ip == "100.64.1.1"
        # Keeps existing status since it's newer
        assert result.last_seen == 2000.0

    def test_merge_nodes_deduplicates(self, inventory):
        """Test _merge_nodes removes duplicates."""
        from app.coordination.unified_inventory import DiscoveredNode

        nodes = [
            DiscoveredNode(node_id="test", host="1.1.1.1", source="vast"),
            DiscoveredNode(node_id="test", host="1.1.1.1", source="tailscale"),
        ]

        result = inventory._merge_nodes(nodes)
        assert len(result) == 1
        assert "test" in result


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def inventory(self):
        """Create a fresh UnifiedInventory instance with mocked config loading."""
        from app.coordination.unified_inventory import UnifiedInventory

        with patch.object(UnifiedInventory, '_load_configs'):
            inv = UnifiedInventory()
            inv._distributed_hosts = {
                "runpod-h100": {
                    "tailscale_ip": "100.64.1.1",
                    "ssh_host": "192.168.1.100",
                    "gpu": "H100",
                    "memory_gb": 80,
                    "role": "nn_training_primary",
                },
                "nebius-l40s": {
                    "tailscale_ip": "100.64.2.2",
                    "gpu": "L40S",
                    "role": "selfplay",
                },
            }
            yield inv

    def test_find_host_by_tailscale_ip_found(self, inventory):
        """Test _find_host_by_tailscale_ip finds matching host."""
        result = inventory._find_host_by_tailscale_ip("100.64.1.1")
        assert result is not None
        assert result["gpu"] == "H100"

    def test_find_host_by_tailscale_ip_not_found(self, inventory):
        """Test _find_host_by_tailscale_ip returns None for unknown IP."""
        result = inventory._find_host_by_tailscale_ip("10.0.0.99")
        assert result is None

    def test_find_host_by_ssh_host(self, inventory):
        """Test _find_host_by_tailscale_ip also matches ssh_host."""
        result = inventory._find_host_by_tailscale_ip("192.168.1.100")
        assert result is not None
        assert result["gpu"] == "H100"

    def test_hostname_to_node_id_from_config(self, inventory):
        """Test _hostname_to_node_id uses config when available."""
        # Should find runpod-h100 by tailscale IP
        result = inventory._hostname_to_node_id("some-hostname", "100.64.1.1")
        assert result == "runpod-h100"

    def test_hostname_to_node_id_gh200_pattern(self, inventory):
        """Test _hostname_to_node_id handles GH200 patterns."""
        result = inventory._hostname_to_node_id("gh200a", "10.0.0.1")
        assert result == "gh200-a"

        result = inventory._hostname_to_node_id("gh200-b", "10.0.0.2")
        assert result == "gh200-b"

    def test_hostname_to_node_id_fallback(self, inventory):
        """Test _hostname_to_node_id falls back to hostname."""
        result = inventory._hostname_to_node_id("unknown-host.tail", "10.0.0.99")
        assert result == "unknown-host"

    def test_hostname_to_node_id_cleans_tailscale_suffix(self, inventory):
        """Test _hostname_to_node_id removes Tailscale suffixes."""
        result = inventory._hostname_to_node_id("myhost.tail.ts.net", "10.0.0.1")
        assert "tail" not in result
        assert "ts.net" not in result


class TestDiscoveryMethods:
    """Tests for async discovery methods."""

    @pytest.fixture
    def inventory(self):
        """Create a fresh UnifiedInventory instance with mocked config loading."""
        from app.coordination.unified_inventory import UnifiedInventory

        with patch.object(UnifiedInventory, '_load_configs'):
            inv = UnifiedInventory()
            inv._distributed_hosts = {}
            yield inv

    @pytest.mark.asyncio
    async def test_discover_vast_success(self, inventory):
        """Test _discover_vast with successful CLI output."""
        mock_instances = [
            {
                "id": 12345,
                "actual_status": "running",
                "ssh_host": "192.168.1.1",
                "ssh_port": 22,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "cpu_ram": 65536,  # 64GB in MB
                "cpu_cores_effective": 16,
            }
        ]

        async def mock_communicate():
            return json.dumps(mock_instances).encode(), b""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mock_communicate

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            nodes = await inventory._discover_vast()

        assert len(nodes) == 1
        assert nodes[0].node_id == "vast-12345"
        assert nodes[0].gpu_name == "RTX 4090"
        assert nodes[0].source == "vast"

    @pytest.mark.asyncio
    async def test_discover_vast_cli_not_found(self, inventory):
        """Test _discover_vast handles missing CLI gracefully."""
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError):
            nodes = await inventory._discover_vast()

        assert nodes == []

    @pytest.mark.asyncio
    async def test_discover_vast_timeout(self, inventory):
        """Test _discover_vast handles timeout gracefully."""
        with patch('asyncio.create_subprocess_exec', side_effect=asyncio.TimeoutError):
            nodes = await inventory._discover_vast()

        assert nodes == []

    @pytest.mark.asyncio
    async def test_discover_tailscale_success(self, inventory):
        """Test _discover_tailscale with successful CLI output."""
        mock_status = {
            "Peer": {
                "peer1": {
                    "TailscaleIPs": ["100.64.1.1"],
                    "HostName": "runpod-h100",
                    "Online": True,
                }
            }
        }

        async def mock_communicate():
            return json.dumps(mock_status).encode(), b""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mock_communicate

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            nodes = await inventory._discover_tailscale()

        assert len(nodes) == 1
        assert nodes[0].tailscale_ip == "100.64.1.1"
        assert nodes[0].source == "tailscale"

    @pytest.mark.asyncio
    async def test_discover_tailscale_filters_offline(self, inventory):
        """Test _discover_tailscale filters offline nodes."""
        mock_status = {
            "Peer": {
                "peer1": {
                    "TailscaleIPs": ["100.64.1.1"],
                    "HostName": "online-node",
                    "Online": True,
                },
                "peer2": {
                    "TailscaleIPs": ["100.64.1.2"],
                    "HostName": "offline-node",
                    "Online": False,
                }
            }
        }

        async def mock_communicate():
            return json.dumps(mock_status).encode(), b""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mock_communicate

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            nodes = await inventory._discover_tailscale()

        assert len(nodes) == 1
        assert nodes[0].node_id == "online-node"

    @pytest.mark.asyncio
    async def test_discover_hetzner_success(self, inventory):
        """Test _discover_hetzner with successful CLI output."""
        mock_servers = [
            {
                "name": "cpu1",
                "status": "running",
                "server_type": {"name": "cx31"},
                "public_net": {
                    "ipv4": {"ip": "192.168.1.1"},
                },
            }
        ]

        async def mock_communicate():
            return json.dumps(mock_servers).encode(), b""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mock_communicate

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            nodes = await inventory._discover_hetzner()

        assert len(nodes) == 1
        assert nodes[0].node_id == "hetzner-cpu1"
        assert nodes[0].source == "hetzner"
        assert nodes[0].role == "cpu_cmaes"  # cx* server type

    @pytest.mark.asyncio
    async def test_discover_all_aggregates(self, inventory):
        """Test discover_all aggregates from multiple sources."""
        from app.coordination.unified_inventory import DiscoveredNode

        # Mock the individual discovery methods
        async def mock_vast():
            return [DiscoveredNode(node_id="vast-1", host="1.1.1.1", source="vast")]

        async def mock_tailscale():
            return [DiscoveredNode(node_id="ts-1", host="2.2.2.2", source="tailscale")]

        async def mock_lambda():
            return []

        async def mock_hetzner():
            return [DiscoveredNode(node_id="hz-1", host="3.3.3.3", source="hetzner")]

        with patch.multiple(
            inventory,
            _discover_vast=mock_vast,
            _discover_tailscale=mock_tailscale,
            _discover_lambda=mock_lambda,
            _discover_hetzner=mock_hetzner,
        ):
            nodes = await inventory.discover_all()

        assert len(nodes) == 3
        assert "vast-1" in nodes
        assert "ts-1" in nodes
        assert "hz-1" in nodes

    @pytest.mark.asyncio
    async def test_discover_all_handles_failures(self, inventory):
        """Test discover_all continues if one source fails."""
        from app.coordination.unified_inventory import DiscoveredNode

        async def mock_vast():
            return [DiscoveredNode(node_id="vast-1", host="1.1.1.1", source="vast")]

        async def mock_tailscale():
            raise RuntimeError("Tailscale failed")

        async def mock_lambda():
            return []

        async def mock_hetzner():
            return []

        with patch.multiple(
            inventory,
            _discover_vast=mock_vast,
            _discover_tailscale=mock_tailscale,
            _discover_lambda=mock_lambda,
            _discover_hetzner=mock_hetzner,
        ):
            nodes = await inventory.discover_all()

        # Should still have vast nodes despite tailscale failure
        assert len(nodes) == 1
        assert "vast-1" in nodes


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_inventory_returns_singleton(self):
        """Test get_inventory returns the same instance."""
        from app.coordination import unified_inventory

        # Reset module state
        unified_inventory._inventory = None

        with patch.object(unified_inventory.UnifiedInventory, '_load_configs'):
            inv1 = unified_inventory.get_inventory()
            inv2 = unified_inventory.get_inventory()

            assert inv1 is inv2

        # Cleanup
        unified_inventory._inventory = None

    def test_module_exports(self):
        """Test __all__ contains expected exports."""
        from app.coordination import unified_inventory

        assert "DiscoveredNode" in unified_inventory.__all__
        assert "UnifiedInventory" in unified_inventory.__all__
        assert "get_inventory" in unified_inventory.__all__
