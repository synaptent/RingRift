"""Tests for app/core/node.py - Unified NodeInfo dataclass.

Tests the NodeInfo class and related dataclasses for representing
cluster node information across the codebase.
"""

from __future__ import annotations

import json
import time

import pytest

from app.core.node import (
    ConnectionInfo,
    GPUInfo,
    HealthStatus,
    JobStatus,
    NodeHealth,
    NodeInfo,
    NodeRole,
    NodeState,
    Provider,
    ProviderInfo,
    ResourceMetrics,
)


class TestEnums:
    """Tests for enum classes."""

    def test_node_role_values(self):
        """NodeRole should have expected values."""
        assert NodeRole.LEADER.value == "leader"
        assert NodeRole.FOLLOWER.value == "follower"
        assert NodeRole.CANDIDATE.value == "candidate"
        assert NodeRole.OFFLINE.value == "offline"

    def test_node_health_values(self):
        """NodeHealth should have expected values."""
        assert NodeHealth.HEALTHY.value == "healthy"
        assert NodeHealth.DEGRADED.value == "degraded"
        assert NodeHealth.UNHEALTHY.value == "unhealthy"
        assert NodeHealth.OFFLINE.value == "offline"
        assert NodeHealth.RETIRED.value == "retired"
        assert NodeHealth.UNKNOWN.value == "unknown"

    def test_node_state_values(self):
        """NodeState should have expected values."""
        assert NodeState.ONLINE.value == "online"
        assert NodeState.DEGRADED.value == "degraded"
        assert NodeState.OFFLINE.value == "offline"
        assert NodeState.UNKNOWN.value == "unknown"

    def test_provider_values(self):
        """Provider should have expected values."""
        assert Provider.LAMBDA.value == "lambda"
        assert Provider.VAST.value == "vast"
        assert Provider.RUNPOD.value == "runpod"
        assert Provider.NEBIUS.value == "nebius"
        assert Provider.VULTR.value == "vultr"
        assert Provider.HETZNER.value == "hetzner"
        assert Provider.LOCAL.value == "local"


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_default_values(self):
        """GPUInfo should have sensible defaults."""
        gpu = GPUInfo()
        assert gpu.has_gpu is False
        assert gpu.gpu_name == ""
        assert gpu.gpu_count == 0
        assert gpu.gpu_type == ""
        assert gpu.memory_total_gb == 0.0
        assert gpu.memory_used_gb == 0.0
        assert gpu.utilization_percent == 0.0

    def test_custom_values(self):
        """GPUInfo should accept custom values."""
        gpu = GPUInfo(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_count=1,
            gpu_type="RTX 4090",
            memory_total_gb=24.0,
            memory_used_gb=8.0,
            utilization_percent=75.5,
        )
        assert gpu.has_gpu is True
        assert gpu.gpu_name == "NVIDIA RTX 4090"
        assert gpu.gpu_count == 1
        assert gpu.memory_total_gb == 24.0
        assert gpu.memory_used_gb == 8.0
        assert gpu.utilization_percent == 75.5

    def test_memory_free_calculation(self):
        """GPUInfo should calculate free memory."""
        gpu = GPUInfo(
            has_gpu=True,
            memory_total_gb=24.0,
            memory_used_gb=8.0,
        )
        assert gpu.memory_free_gb == pytest.approx(16.0, rel=0.1)

    def test_is_cuda_gpu(self):
        """is_cuda_gpu should identify CUDA GPUs."""
        cuda_gpu = GPUInfo(has_gpu=True, gpu_name="NVIDIA RTX 4090")
        apple_gpu = GPUInfo(has_gpu=True, gpu_name="Apple M3 Pro")
        no_gpu = GPUInfo(has_gpu=False)

        assert cuda_gpu.is_cuda_gpu is True
        assert apple_gpu.is_cuda_gpu is False
        assert no_gpu.is_cuda_gpu is False

    def test_power_score_calculation(self):
        """GPUInfo should calculate power score based on GPU type."""
        h100 = GPUInfo(has_gpu=True, gpu_name="H100", gpu_count=1)
        rtx4090 = GPUInfo(has_gpu=True, gpu_name="RTX 4090", gpu_count=1)
        unknown = GPUInfo(has_gpu=True, gpu_name="Some Unknown GPU", gpu_count=1)

        assert h100.power_score > rtx4090.power_score
        assert rtx4090.power_score > unknown.power_score

    def test_from_nvidia_smi(self):
        """from_nvidia_smi should parse nvidia-smi output."""
        output = "45, 8192, 24576"  # util%, mem_used_mb, mem_total_mb
        gpu = GPUInfo.from_nvidia_smi(output)

        assert gpu.has_gpu is True
        assert gpu.utilization_percent == 45.0
        assert gpu.memory_used_gb == pytest.approx(8.0, rel=0.1)
        assert gpu.memory_total_gb == pytest.approx(24.0, rel=0.1)

    def test_to_dict(self):
        """to_dict should serialize GPUInfo."""
        gpu = GPUInfo(has_gpu=True, gpu_name="RTX 4090")
        d = gpu.to_dict()
        assert isinstance(d, dict)
        assert d["has_gpu"] is True
        assert d["gpu_name"] == "RTX 4090"
        # Should not include class variable
        assert "GPU_POWER_RANKINGS" not in d

    def test_from_dict(self):
        """from_dict should deserialize GPUInfo."""
        data = {"has_gpu": True, "gpu_name": "RTX 4090", "gpu_count": 2}
        gpu = GPUInfo.from_dict(data)
        assert gpu.has_gpu is True
        assert gpu.gpu_name == "RTX 4090"
        assert gpu.gpu_count == 2


class TestResourceMetrics:
    """Tests for ResourceMetrics dataclass."""

    def test_default_values(self):
        """ResourceMetrics should have sensible defaults."""
        metrics = ResourceMetrics()
        assert metrics.cpu_count == 0
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_percent == 0.0

    def test_custom_values(self):
        """ResourceMetrics should accept custom values."""
        metrics = ResourceMetrics(
            cpu_count=8,
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            memory_gb_total=32.0,
            memory_gb_available=12.8,
        )
        assert metrics.cpu_count == 8
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.memory_gb_total == 32.0

    def test_load_score(self):
        """load_score should return max of cpu and memory."""
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=75.0)
        assert metrics.load_score == 75.0

    def test_is_overloaded(self):
        """is_overloaded should detect high utilization."""
        normal = ResourceMetrics(cpu_percent=50.0, memory_percent=50.0)
        overloaded_cpu = ResourceMetrics(cpu_percent=85.0, memory_percent=50.0)
        overloaded_mem = ResourceMetrics(cpu_percent=50.0, memory_percent=85.0)

        assert normal.is_overloaded is False
        assert overloaded_cpu.is_overloaded is True
        assert overloaded_mem.is_overloaded is True

    def test_disk_warning_thresholds(self):
        """Disk thresholds should be detected."""
        normal = ResourceMetrics(disk_percent=50.0)
        warning = ResourceMetrics(disk_percent=65.0)
        critical = ResourceMetrics(disk_percent=75.0)

        assert normal.is_disk_warning is False
        assert warning.is_disk_warning is True
        assert critical.is_disk_critical is True


class TestConnectionInfo:
    """Tests for ConnectionInfo dataclass."""

    def test_default_values(self):
        """ConnectionInfo should have sensible defaults."""
        conn = ConnectionInfo()
        assert conn.host == ""
        assert conn.port == 8770
        assert conn.ssh_port == 22
        assert conn.ssh_user == "ubuntu"

    def test_custom_values(self):
        """ConnectionInfo should accept custom values."""
        conn = ConnectionInfo(
            host="node1.example.com",
            port=8770,
            ssh_host="192.168.1.100",
            ssh_port=2222,
            ssh_user="admin",
            tailscale_ip="100.64.1.1",
        )
        assert conn.host == "node1.example.com"
        assert conn.ssh_host == "192.168.1.100"
        assert conn.ssh_port == 2222
        assert conn.ssh_user == "admin"
        assert conn.tailscale_ip == "100.64.1.1"

    def test_endpoint_property(self):
        """endpoint should construct URL."""
        conn = ConnectionInfo(host="192.168.1.100", port=8770, scheme="http")
        assert conn.endpoint == "http://192.168.1.100:8770"

    def test_best_ip_property(self):
        """best_ip should prefer tailscale_ip."""
        conn_with_tailscale = ConnectionInfo(
            host="192.168.1.100",
            tailscale_ip="100.64.1.1",
        )
        conn_without_tailscale = ConnectionInfo(host="192.168.1.100")

        assert conn_with_tailscale.best_ip == "100.64.1.1"
        assert conn_without_tailscale.best_ip == "192.168.1.100"


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_default_values(self):
        """HealthStatus should have sensible defaults."""
        status = HealthStatus()
        assert status.health == NodeHealth.UNKNOWN
        assert status.state == NodeState.UNKNOWN
        assert status.consecutive_failures == 0
        assert status.retired is False

    def test_custom_values(self):
        """HealthStatus should accept custom values."""
        now = time.time()
        status = HealthStatus(
            health=NodeHealth.HEALTHY,
            state=NodeState.ONLINE,
            last_heartbeat=now,
            last_seen=now,
            uptime_seconds=3600,
        )
        assert status.health == NodeHealth.HEALTHY
        assert status.state == NodeState.ONLINE
        assert status.last_heartbeat == now
        assert status.uptime_seconds == 3600

    def test_is_alive(self):
        """is_alive should check heartbeat recency."""
        recent = HealthStatus(last_heartbeat=time.time())
        old = HealthStatus(
            last_heartbeat=time.time() - HealthStatus.PEER_TIMEOUT - 1
        )

        assert recent.is_alive is True
        assert old.is_alive is False

    def test_is_healthy(self):
        """is_healthy should check health status."""
        healthy = HealthStatus(
            health=NodeHealth.HEALTHY,
            last_heartbeat=time.time(),
        )
        retired = HealthStatus(
            health=NodeHealth.HEALTHY,
            last_heartbeat=time.time(),
            retired=True,
        )

        assert healthy.is_healthy is True
        assert retired.is_healthy is False


class TestProviderInfo:
    """Tests for ProviderInfo dataclass."""

    def test_default_values(self):
        """ProviderInfo should have sensible defaults."""
        info = ProviderInfo()
        assert info.provider == Provider.UNKNOWN
        assert info.instance_id is None

    def test_detect_from_node_id(self):
        """detect_from_node_id should identify providers."""
        vast = ProviderInfo.detect_from_node_id("vast-rtx4090-1")
        runpod = ProviderInfo.detect_from_node_id("runpod-h100")
        nebius = ProviderInfo.detect_from_node_id("nebius-backbone-1")

        assert vast.provider == Provider.VAST
        assert runpod.provider == Provider.RUNPOD
        assert nebius.provider == Provider.NEBIUS


class TestJobStatus:
    """Tests for JobStatus dataclass."""

    def test_default_values(self):
        """JobStatus should have sensible defaults."""
        status = JobStatus()
        assert status.active_jobs == 0
        assert status.selfplay_jobs == 0
        assert status.training_active is False

    def test_has_external_work(self):
        """has_external_work should detect gauntlet/tournament."""
        normal = JobStatus()
        with_gauntlet = JobStatus(gauntlet_running=True)
        with_tournament = JobStatus(tournament_running=True)

        assert normal.has_external_work is False
        assert with_gauntlet.has_external_work is True
        assert with_tournament.has_external_work is True


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_minimal_creation(self):
        """NodeInfo should be creatable with just node_id."""
        node = NodeInfo(node_id="test-node")
        assert node.node_id == "test-node"
        assert node.hostname == "test-node"  # Defaults to node_id
        assert node.role == NodeRole.FOLLOWER

    def test_full_creation(self):
        """NodeInfo should accept all fields."""
        node = NodeInfo(
            node_id="test-node",
            hostname="test-host.local",
            role=NodeRole.LEADER,
            gpu=GPUInfo(has_gpu=True, gpu_name="RTX 4090"),
            resources=ResourceMetrics(cpu_percent=50.0),
            health=HealthStatus(
                health=NodeHealth.HEALTHY,
                state=NodeState.ONLINE,
                last_heartbeat=time.time(),
            ),
        )
        assert node.node_id == "test-node"
        assert node.hostname == "test-host.local"
        assert node.role == NodeRole.LEADER
        assert node.resources.cpu_percent == 50.0
        assert node.gpu.gpu_name == "RTX 4090"
        assert node.health.health == NodeHealth.HEALTHY

    def test_to_dict(self):
        """to_dict should serialize NodeInfo."""
        node = NodeInfo(
            node_id="test-node",
            role=NodeRole.LEADER,
            health=HealthStatus(health=NodeHealth.HEALTHY),
        )
        data = node.to_dict()

        assert isinstance(data, dict)
        assert data["node_id"] == "test-node"
        assert data["role"] == "leader"
        assert data["health"]["health"] == "healthy"

    def test_from_dict(self):
        """from_dict should deserialize dictionary to NodeInfo."""
        data = {
            "node_id": "test-node",
            "hostname": "test-host",
            "role": "leader",
            "health": {"health": "healthy", "state": "online"},
            "resources": {"cpu_percent": 50.0, "memory_percent": 60.0},
        }
        node = NodeInfo.from_dict(data)

        assert node.node_id == "test-node"
        assert node.hostname == "test-host"
        assert node.role == NodeRole.LEADER
        assert node.health.health == NodeHealth.HEALTHY

    def test_json_roundtrip(self):
        """NodeInfo should survive JSON serialization roundtrip."""
        original = NodeInfo(
            node_id="test-node",
            hostname="test-host",
            role=NodeRole.LEADER,
            gpu=GPUInfo(has_gpu=True, gpu_name="RTX 4090"),
            health=HealthStatus(health=NodeHealth.HEALTHY),
        )

        # Serialize to JSON
        json_str = original.to_json()

        # Deserialize from JSON
        restored = NodeInfo.from_json(json_str)

        assert restored.node_id == original.node_id
        assert restored.role == original.role
        assert restored.health.health == original.health.health


class TestNodeInfoFactoryMethods:
    """Tests for NodeInfo factory methods."""

    def test_from_p2p_status(self):
        """from_p2p_status should create NodeInfo from P2P peer data."""
        peer_data = {
            "node_id": "runpod-h100",
            "role": "follower",
            "health": "healthy",
            "has_gpu": True,
            "gpu_name": "H100",
            "gpu_count": 1,
            "gpu_memory_total": 80,
            "cpu_percent": 50.0,
            "disk_percent": 30.0,
            "last_heartbeat": time.time(),
        }

        node = NodeInfo.from_p2p_status(peer_data)
        assert node.node_id == "runpod-h100"
        assert node.gpu.has_gpu is True
        assert node.health.health == NodeHealth.HEALTHY

    def test_from_ssh_discovery(self):
        """from_ssh_discovery should create NodeInfo."""
        nvidia_output = "45, 8192, 24576"

        node = NodeInfo.from_ssh_discovery(
            node_id="gpu-node",
            host="192.168.1.100",
            nvidia_smi_output=nvidia_output,
        )

        assert node.node_id == "gpu-node"
        assert node.connection.host == "192.168.1.100"
        assert node.gpu.has_gpu is True

    def test_from_cluster_config(self):
        """from_cluster_config should create NodeInfo from YAML config."""
        config = {
            "ssh_host": "192.168.1.100",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "gpu": "RTX 4090",
            "memory_gb": 64,
            "cpus": 16,
            "status": "ready",
        }

        node = NodeInfo.from_cluster_config("test-node", config)

        assert node.node_id == "test-node"
        assert node.connection.ssh_host == "192.168.1.100"
        assert node.gpu.has_gpu is True
        assert node.resources.memory_gb_total == 64


class TestNodeInfoProperties:
    """Tests for NodeInfo computed properties."""

    def test_is_healthy(self):
        """is_healthy should check health and disk status."""
        healthy_node = NodeInfo(
            node_id="test",
            health=HealthStatus(
                health=NodeHealth.HEALTHY,
                last_heartbeat=time.time(),
            ),
        )
        unhealthy_node = NodeInfo(
            node_id="test",
            health=HealthStatus(
                health=NodeHealth.UNHEALTHY,
                last_heartbeat=time.time(),
            ),
        )

        assert healthy_node.is_healthy is True
        assert unhealthy_node.is_healthy is False

    def test_is_gpu_node(self):
        """is_gpu_node should check for CUDA GPUs."""
        with_gpu = NodeInfo(
            node_id="gpu-node",
            gpu=GPUInfo(has_gpu=True, gpu_name="RTX 4090"),
        )
        without_gpu = NodeInfo(node_id="cpu-node")

        assert with_gpu.is_gpu_node is True
        assert without_gpu.is_gpu_node is False

    def test_gpu_power_score(self):
        """gpu_power_score should return GPU power ranking."""
        h100_node = NodeInfo(
            node_id="h100-node",
            gpu=GPUInfo(has_gpu=True, gpu_name="H100", gpu_count=1),
        )
        rtx_node = NodeInfo(
            node_id="rtx-node",
            gpu=GPUInfo(has_gpu=True, gpu_name="RTX 4090", gpu_count=1),
        )

        assert h100_node.gpu_power_score > rtx_node.gpu_power_score

    def test_endpoint(self):
        """endpoint should return connection endpoint."""
        node = NodeInfo(
            node_id="test",
            connection=ConnectionInfo(host="192.168.1.100", port=8770),
        )
        assert node.endpoint == "http://192.168.1.100:8770"
