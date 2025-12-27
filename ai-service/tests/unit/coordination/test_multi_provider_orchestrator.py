"""Tests for MultiProviderOrchestrator (multi-cloud node management).

Tests cover:
- Provider and NodeRole enums
- ClusterNode dataclass
- MultiProviderOrchestrator state management
- Module functions
"""

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Test Provider Enum
# =============================================================================

class TestProvider:
    """Tests for Provider enum."""

    def test_lambda_value(self):
        """Test LAMBDA provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.LAMBDA.value == "lambda"

    def test_vast_value(self):
        """Test VAST provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.VAST.value == "vast"

    def test_aws_value(self):
        """Test AWS provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.AWS.value == "aws"

    def test_hetzner_value(self):
        """Test HETZNER provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.HETZNER.value == "hetzner"

    def test_local_value(self):
        """Test LOCAL provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.LOCAL.value == "local"

    def test_unknown_value(self):
        """Test UNKNOWN provider value."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert Provider.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        """Test that Provider is a string enum."""
        from app.coordination.multi_provider_orchestrator import Provider
        assert isinstance(Provider.LAMBDA, str)
        assert Provider.LAMBDA == "lambda"


# =============================================================================
# Test NodeRole Enum
# =============================================================================

class TestNodeRole:
    """Tests for NodeRole enum."""

    def test_training_value(self):
        """Test TRAINING role value."""
        from app.coordination.multi_provider_orchestrator import NodeRole
        assert NodeRole.TRAINING.value == "training"

    def test_selfplay_value(self):
        """Test SELFPLAY role value."""
        from app.coordination.multi_provider_orchestrator import NodeRole
        assert NodeRole.SELFPLAY.value == "selfplay"

    def test_coordinator_value(self):
        """Test COORDINATOR role value."""
        from app.coordination.multi_provider_orchestrator import NodeRole
        assert NodeRole.COORDINATOR.value == "coordinator"

    def test_idle_value(self):
        """Test IDLE role value."""
        from app.coordination.multi_provider_orchestrator import NodeRole
        assert NodeRole.IDLE.value == "idle"

    def test_offline_value(self):
        """Test OFFLINE role value."""
        from app.coordination.multi_provider_orchestrator import NodeRole
        assert NodeRole.OFFLINE.value == "offline"


# =============================================================================
# Test ClusterNode Dataclass
# =============================================================================

class TestClusterNode:
    """Tests for ClusterNode dataclass."""

    def test_create_minimal(self):
        """Test creating node with minimal fields."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
        )

        node = ClusterNode(name="gh200-a", provider=Provider.LAMBDA)

        assert node.name == "gh200-a"
        assert node.provider == Provider.LAMBDA

    def test_default_values(self):
        """Test default values are set correctly."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            NodeRole,
            Provider,
        )

        node = ClusterNode(name="test", provider=Provider.LOCAL)

        assert node.tailscale_ip is None
        assert node.public_ip is None
        assert node.ssh_user == "ubuntu"
        assert node.ssh_port == 22
        assert node.ssh_host is None
        assert node.gpu_name is None
        assert node.gpu_count == 1
        assert node.gpu_memory_gb == 0
        assert node.cpu_cores == 0
        assert node.memory_gb == 0
        assert node.is_online is False
        assert node.is_tailscale_connected is False
        assert node.role == NodeRole.IDLE
        assert node.last_seen == 0
        assert node.provider_id is None
        assert node.selfplay_running is False
        assert node.training_running is False
        assert node.current_job is None

    def test_all_fields(self):
        """Test creating node with all fields."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            NodeRole,
            Provider,
        )

        node = ClusterNode(
            name="gh200-training",
            provider=Provider.LAMBDA,
            tailscale_ip="100.64.0.1",
            public_ip="34.56.78.90",
            ssh_user="ubuntu",
            ssh_port=22,
            ssh_host=None,
            gpu_name="GH200",
            gpu_count=1,
            gpu_memory_gb=96.0,
            cpu_cores=72,
            memory_gb=480.0,
            is_online=True,
            is_tailscale_connected=True,
            role=NodeRole.TRAINING,
            last_seen=time.time(),
            provider_id="lambda-123",
            selfplay_running=False,
            training_running=True,
            current_job="train_sq8_2p",
        )

        assert node.name == "gh200-training"
        assert node.gpu_name == "GH200"
        assert node.gpu_memory_gb == 96.0
        assert node.role == NodeRole.TRAINING
        assert node.training_running is True

    def test_ssh_command_with_tailscale(self):
        """Test SSH command generation with Tailscale IP."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
        )

        node = ClusterNode(
            name="test",
            provider=Provider.LAMBDA,
            tailscale_ip="100.64.0.5",
            ssh_user="ubuntu",
        )

        cmd = node.ssh_command("ls -la")
        assert "100.64.0.5" in cmd
        assert "-p 22" in cmd
        assert "ubuntu@" in cmd
        assert "'ls -la'" in cmd

    def test_ssh_command_with_public_ip(self):
        """Test SSH command generation with public IP (no Tailscale)."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
        )

        node = ClusterNode(
            name="test",
            provider=Provider.AWS,
            public_ip="54.123.45.67",
            ssh_user="ec2-user",
            ssh_port=2222,
        )

        cmd = node.ssh_command("whoami")
        assert "54.123.45.67" in cmd
        assert "-p 2222" in cmd
        assert "ec2-user@" in cmd

    def test_ssh_command_with_ssh_host(self):
        """Test SSH command with SSH jump host."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
        )

        node = ClusterNode(
            name="test",
            provider=Provider.HETZNER,
            ssh_host="jump.example.com",
            ssh_user="admin",
        )

        cmd = node.ssh_command("uptime")
        assert "jump.example.com" in cmd


# =============================================================================
# Test MultiProviderOrchestrator
# =============================================================================

class TestMultiProviderOrchestrator:
    """Tests for MultiProviderOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.coordination.multi_provider_orchestrator import (
            MultiProviderOrchestrator,
            reset_orchestrator,
        )

        reset_orchestrator()
        orch = MultiProviderOrchestrator()
        yield orch
        reset_orchestrator()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.nodes == {}
        assert orchestrator._last_discovery == 0
        assert orchestrator._discovery_interval == 60

    def test_add_node(self, orchestrator):
        """Test adding nodes to orchestrator."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
        )

        node = ClusterNode(name="test-node", provider=Provider.LOCAL)
        orchestrator.nodes["test-node"] = node

        assert "test-node" in orchestrator.nodes
        assert orchestrator.nodes["test-node"].provider == Provider.LOCAL

    def test_multiple_nodes(self, orchestrator):
        """Test managing multiple nodes."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            NodeRole,
            Provider,
        )

        # Add nodes from different providers
        orchestrator.nodes["gh200-a"] = ClusterNode(
            name="gh200-a",
            provider=Provider.LAMBDA,
            is_online=True,
            role=NodeRole.TRAINING,
        )
        orchestrator.nodes["vast-rtx3090"] = ClusterNode(
            name="vast-rtx3090",
            provider=Provider.VAST,
            is_online=True,
            role=NodeRole.SELFPLAY,
        )
        orchestrator.nodes["mac-studio"] = ClusterNode(
            name="mac-studio",
            provider=Provider.LOCAL,
            is_online=True,
            role=NodeRole.COORDINATOR,
        )

        assert len(orchestrator.nodes) == 3

        # Filter by role
        training_nodes = [n for n in orchestrator.nodes.values() if n.role == NodeRole.TRAINING]
        assert len(training_nodes) == 1
        assert training_nodes[0].name == "gh200-a"

    @pytest.mark.asyncio
    async def test_discover_all_handles_exceptions(self, orchestrator):
        """Test that discover_all handles discovery failures gracefully."""
        # Mock all discovery methods to fail
        with patch.object(orchestrator, '_discover_tailscale', new_callable=AsyncMock) as mock_ts, \
             patch.object(orchestrator, '_discover_vast', new_callable=AsyncMock) as mock_vast, \
             patch.object(orchestrator, '_discover_aws', new_callable=AsyncMock) as mock_aws, \
             patch.object(orchestrator, '_discover_hetzner', new_callable=AsyncMock) as mock_hetzner, \
             patch.object(orchestrator, '_discover_hosts_from_config', new_callable=AsyncMock) as mock_hosts:

            mock_ts.side_effect = Exception("Tailscale error")
            mock_vast.side_effect = Exception("Vast error")
            mock_aws.return_value = []
            mock_hetzner.return_value = []
            mock_hosts.return_value = []

            # Should not raise, should handle gracefully
            result = await orchestrator.discover_all()
            assert result is not None

    @pytest.mark.asyncio
    async def test_discover_all_updates_timestamp(self, orchestrator):
        """Test that discover_all updates last discovery time."""
        with patch.object(orchestrator, '_discover_tailscale', new_callable=AsyncMock, return_value=[]), \
             patch.object(orchestrator, '_discover_vast', new_callable=AsyncMock, return_value=[]), \
             patch.object(orchestrator, '_discover_aws', new_callable=AsyncMock, return_value=[]), \
             patch.object(orchestrator, '_discover_hetzner', new_callable=AsyncMock, return_value=[]), \
             patch.object(orchestrator, '_discover_hosts_from_config', new_callable=AsyncMock, return_value=[]):

            before = time.time()
            await orchestrator.discover_all()
            after = time.time()

            assert orchestrator._last_discovery >= before
            assert orchestrator._last_discovery <= after


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.coordination.multi_provider_orchestrator import reset_orchestrator

        reset_orchestrator()
        yield
        reset_orchestrator()

    def test_get_orchestrator_creates_singleton(self):
        """Test that get_orchestrator creates a singleton."""
        from app.coordination.multi_provider_orchestrator import (
            MultiProviderOrchestrator,
            get_orchestrator,
        )

        orch1 = get_orchestrator()
        orch2 = get_orchestrator()

        assert orch1 is orch2
        assert isinstance(orch1, MultiProviderOrchestrator)

    def test_reset_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.coordination.multi_provider_orchestrator import (
            get_orchestrator,
            reset_orchestrator,
        )

        orch1 = get_orchestrator()
        reset_orchestrator()
        orch2 = get_orchestrator()

        assert orch1 is not orch2

    @pytest.mark.asyncio
    async def test_discover_all_nodes(self):
        """Test discover_all_nodes convenience function."""
        from app.coordination.multi_provider_orchestrator import (
            discover_all_nodes,
            get_orchestrator,
        )

        orch = get_orchestrator()

        with patch.object(orch, 'discover_all', new_callable=AsyncMock, return_value={}):
            result = await discover_all_nodes()
            assert result == {}

    @pytest.mark.asyncio
    async def test_deploy_to_all_nodes(self):
        """Test deploy_to_all_nodes convenience function."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            Provider,
            deploy_to_all_nodes,
            get_orchestrator,
        )

        orch = get_orchestrator()
        orch.nodes["test"] = ClusterNode(
            name="test",
            provider=Provider.LOCAL,
            is_online=True,
            is_tailscale_connected=True,
        )

        with patch.object(orch, 'deploy_selfplay', new_callable=AsyncMock, return_value=True):
            result = await deploy_to_all_nodes(role="selfplay")
            # Result should be a dict of node_name -> success
            assert isinstance(result, dict)


# =============================================================================
# Integration Tests
# =============================================================================

class TestOrchestratorIntegration:
    """Integration tests for orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.coordination.multi_provider_orchestrator import reset_orchestrator

        reset_orchestrator()
        yield
        reset_orchestrator()

    def test_node_lifecycle(self):
        """Test node lifecycle from discovery to assignment."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            MultiProviderOrchestrator,
            NodeRole,
            Provider,
        )

        orch = MultiProviderOrchestrator()

        # Simulate discovery
        node = ClusterNode(
            name="gh200-a",
            provider=Provider.LAMBDA,
            tailscale_ip="100.64.0.10",
            gpu_name="GH200",
            gpu_memory_gb=96.0,
            is_online=True,
            is_tailscale_connected=True,
        )
        orch.nodes["gh200-a"] = node

        assert node.role == NodeRole.IDLE

        # Assign to training
        node.role = NodeRole.TRAINING
        node.training_running = True
        node.current_job = "train_sq8_2p_v3"

        assert node.role == NodeRole.TRAINING
        assert node.training_running is True

        # Complete training
        node.training_running = False
        node.current_job = None
        node.role = NodeRole.IDLE

        assert node.role == NodeRole.IDLE
        assert node.training_running is False

    def test_filter_nodes_by_capability(self):
        """Test filtering nodes by GPU capability."""
        from app.coordination.multi_provider_orchestrator import (
            ClusterNode,
            MultiProviderOrchestrator,
            Provider,
        )

        orch = MultiProviderOrchestrator()

        # Add various nodes
        orch.nodes["gh200-a"] = ClusterNode(
            name="gh200-a",
            provider=Provider.LAMBDA,
            gpu_name="GH200",
            gpu_memory_gb=96.0,
            is_online=True,
        )
        orch.nodes["h100-1"] = ClusterNode(
            name="h100-1",
            provider=Provider.LAMBDA,
            gpu_name="H100",
            gpu_memory_gb=80.0,
            is_online=True,
        )
        orch.nodes["rtx3090"] = ClusterNode(
            name="rtx3090",
            provider=Provider.VAST,
            gpu_name="RTX 3090",
            gpu_memory_gb=24.0,
            is_online=True,
        )

        # Filter by GPU memory
        large_gpu_nodes = [
            n for n in orch.nodes.values()
            if n.gpu_memory_gb >= 48.0 and n.is_online
        ]
        assert len(large_gpu_nodes) == 2

        # Filter by provider
        lambda_nodes = [
            n for n in orch.nodes.values()
            if n.provider == Provider.LAMBDA
        ]
        assert len(lambda_nodes) == 2
