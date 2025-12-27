"""Unit tests for P2PAutoDeployer (December 2025).

Tests the daemon that ensures P2P network runs on all cluster nodes.

Created: December 27, 2025
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.p2p_auto_deployer import (
    P2PAutoDeployer,
    P2PCoverageReport,
    P2PDeploymentConfig,
    P2PDeploymentResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config() -> P2PDeploymentConfig:
    """Create a test deployment configuration."""
    return P2PDeploymentConfig(
        check_interval_seconds=1.0,  # Fast for tests
        deployment_timeout_seconds=5.0,
        max_concurrent_deployments=2,
        health_check_timeout_seconds=2.0,
        p2p_port=8770,
        min_coverage_percent=50.0,
        excluded_nodes=["local-mac", "mac-studio"],
    )


@pytest.fixture
def mock_hosts() -> dict:
    """Create mock host configurations."""
    return {
        "node-1": {
            "ssh_host": "192.168.1.1",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "ssh_key": "~/.ssh/id_rsa",
            "status": "ready",
            "ringrift_path": "/home/ubuntu/ringrift/ai-service",
        },
        "node-2": {
            "ssh_host": "192.168.1.2",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "status": "ready",
        },
        "retired-node": {
            "ssh_host": "192.168.1.3",
            "status": "retired",
        },
        "local-mac": {
            "ssh_host": "localhost",
            "status": "ready",
        },
    }


@pytest.fixture
def deployer(config: P2PDeploymentConfig, mock_hosts: dict) -> P2PAutoDeployer:
    """Create a P2PAutoDeployer for testing."""
    with patch.object(P2PAutoDeployer, "_load_hosts_config"):
        deployer = P2PAutoDeployer(config)
        deployer._hosts = mock_hosts
        return deployer


# ============================================================================
# P2PDeploymentConfig Tests
# ============================================================================


class TestP2PDeploymentConfig:
    """Tests for P2PDeploymentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = P2PDeploymentConfig()

        assert config.check_interval_seconds == 300.0
        assert config.deployment_timeout_seconds == 120.0
        assert config.max_concurrent_deployments == 5
        assert config.retry_count == 2
        assert config.retry_delay_seconds == 30.0
        assert config.health_check_timeout_seconds == 20.0
        assert config.min_coverage_percent == 90.0

    def test_excluded_nodes_default(self) -> None:
        """Test default excluded nodes."""
        config = P2PDeploymentConfig()

        assert "mac-studio" in config.excluded_nodes
        assert "local-mac" in config.excluded_nodes
        assert "aws-proxy" in config.excluded_nodes

    def test_custom_config(self, config: P2PDeploymentConfig) -> None:
        """Test custom configuration."""
        assert config.check_interval_seconds == 1.0
        assert config.deployment_timeout_seconds == 5.0
        assert config.max_concurrent_deployments == 2


# ============================================================================
# P2PDeploymentResult Tests
# ============================================================================


class TestP2PDeploymentResult:
    """Tests for P2PDeploymentResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful deployment result."""
        result = P2PDeploymentResult(
            node_id="node-1",
            success=True,
            method="ssh_direct",
            message="P2P started successfully",
            duration_seconds=5.5,
        )

        assert result.node_id == "node-1"
        assert result.success is True
        assert result.method == "ssh_direct"
        assert result.duration_seconds == 5.5

    def test_failure_result(self) -> None:
        """Test failed deployment result."""
        result = P2PDeploymentResult(
            node_id="node-2",
            success=False,
            method="none",
            message="No SSH host available",
            duration_seconds=0.1,
        )

        assert result.success is False
        assert result.message == "No SSH host available"


# ============================================================================
# P2PCoverageReport Tests
# ============================================================================


class TestP2PCoverageReport:
    """Tests for P2PCoverageReport dataclass."""

    def test_full_coverage(self) -> None:
        """Test 100% coverage report."""
        report = P2PCoverageReport(
            total_nodes=10,
            nodes_with_p2p=10,
            nodes_without_p2p=0,
            unreachable_nodes=0,
            coverage_percent=100.0,
            nodes_needing_deployment=[],
        )

        assert report.coverage_percent == 100.0
        assert len(report.nodes_needing_deployment) == 0

    def test_partial_coverage(self) -> None:
        """Test partial coverage report."""
        report = P2PCoverageReport(
            total_nodes=10,
            nodes_with_p2p=7,
            nodes_without_p2p=2,
            unreachable_nodes=1,
            coverage_percent=70.0,
            nodes_needing_deployment=["node-8", "node-9", "node-10"],
        )

        assert report.coverage_percent == 70.0
        assert len(report.nodes_needing_deployment) == 3

    def test_timestamp_auto_set(self) -> None:
        """Test timestamp is automatically set."""
        before = time.time()
        report = P2PCoverageReport(
            total_nodes=1,
            nodes_with_p2p=1,
            nodes_without_p2p=0,
            unreachable_nodes=0,
            coverage_percent=100.0,
            nodes_needing_deployment=[],
        )
        after = time.time()

        assert before <= report.timestamp <= after


# ============================================================================
# P2PAutoDeployer Initialization Tests
# ============================================================================


class TestP2PAutoDeployerInit:
    """Tests for P2PAutoDeployer initialization."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default config."""
        with patch.object(P2PAutoDeployer, "_load_hosts_config"):
            deployer = P2PAutoDeployer()

            assert deployer.config is not None
            assert deployer._running is False
            assert deployer._last_check == 0.0
            assert deployer._deployment_history == []
            assert deployer._coverage_history == []

    def test_init_with_custom_config(
        self, config: P2PDeploymentConfig
    ) -> None:
        """Test initialization with custom config."""
        with patch.object(P2PAutoDeployer, "_load_hosts_config"):
            deployer = P2PAutoDeployer(config)

            assert deployer.config is config

    def test_init_with_health_orchestrator(self) -> None:
        """Test initialization with health orchestrator."""
        mock_orchestrator = MagicMock()

        with patch.object(P2PAutoDeployer, "_load_hosts_config"):
            deployer = P2PAutoDeployer(health_orchestrator=mock_orchestrator)

            assert deployer.health_orchestrator is mock_orchestrator


# ============================================================================
# P2PAutoDeployer Host Management Tests
# ============================================================================


class TestP2PAutoDeployerHostManagement:
    """Tests for host configuration management."""

    def test_get_deployable_hosts(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test getting deployable hosts."""
        deployable = deployer._get_deployable_hosts()

        # Should include node-1 and node-2
        assert "node-1" in deployable
        assert "node-2" in deployable

        # Should exclude retired and local nodes
        assert "retired-node" not in deployable
        assert "local-mac" not in deployable

    def test_get_deployable_hosts_excludes_no_ssh(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test that nodes without SSH are excluded."""
        deployer._hosts["no-ssh-node"] = {"status": "ready"}

        deployable = deployer._get_deployable_hosts()

        assert "no-ssh-node" not in deployable

    def test_build_ssh_args(self, deployer: P2PAutoDeployer) -> None:
        """Test building SSH arguments."""
        host_info = {
            "ssh_host": "192.168.1.1",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "ssh_key": "~/.ssh/id_rsa",
        }

        args = deployer._build_ssh_args(host_info, timeout=10.0)

        assert "ssh" in args
        assert "-p" in args
        assert "22" in args
        assert "ubuntu@192.168.1.1" in args
        assert "ConnectTimeout=10" in " ".join(args)


# ============================================================================
# P2PAutoDeployer Health Check Tests
# ============================================================================


class TestP2PAutoDeployerHealthCheck:
    """Tests for P2P health checking."""

    @pytest.mark.asyncio
    async def test_check_p2p_health_with_orchestrator(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test health check using orchestrator."""
        mock_health = MagicMock()
        mock_health.p2p_healthy = True

        mock_orchestrator = MagicMock()
        mock_orchestrator.get_node_health.return_value = mock_health

        deployer.health_orchestrator = mock_orchestrator

        result = await deployer.check_p2p_health("node-1", {"ssh_host": "1.2.3.4"})

        assert result is True
        mock_orchestrator.get_node_health.assert_called_once_with("node-1")

    @pytest.mark.asyncio
    async def test_check_p2p_health_no_ssh_host(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test health check fails without SSH host."""
        deployer.health_orchestrator = None

        result = await deployer.check_p2p_health("node-1", {})

        assert result is False

    @pytest.mark.asyncio
    async def test_check_p2p_health_via_ssh(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test health check via SSH."""
        deployer.health_orchestrator = None

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"healthy": true}',
                b"",
            )
            mock_exec.return_value = mock_proc

            result = await deployer.check_p2p_health(
                "node-1",
                {"ssh_host": "192.168.1.1"},
            )

            assert result is True


# ============================================================================
# P2PAutoDeployer Deployment Tests
# ============================================================================


class TestP2PAutoDeployerDeployment:
    """Tests for P2P deployment functionality."""

    @pytest.mark.asyncio
    async def test_deploy_no_ssh_host(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test deployment fails without SSH host."""
        result = await deployer.deploy_p2p_to_node("node-x", {})

        assert result.success is False
        assert result.method == "none"
        assert "No SSH host" in result.message

    @pytest.mark.asyncio
    async def test_deploy_success(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test successful deployment."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b"P2P_STARTED",
                b"",
            )
            mock_exec.return_value = mock_proc

            with patch.object(
                deployer, "_verify_mesh_join", return_value=True
            ):
                result = await deployer.deploy_p2p_to_node(
                    "node-1",
                    deployer._hosts["node-1"],
                )

                assert result.success is True
                assert result.method == "ssh_direct"

    @pytest.mark.asyncio
    async def test_deploy_timeout(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test deployment timeout handling."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.side_effect = asyncio.TimeoutError()
            mock_exec.return_value = mock_proc

            result = await deployer.deploy_p2p_to_node(
                "node-1",
                deployer._hosts["node-1"],
            )

            assert result.success is False
            assert "timeout" in result.message.lower()


# ============================================================================
# P2PAutoDeployer Mesh Verification Tests
# ============================================================================


class TestP2PAutoDeployerMeshVerify:
    """Tests for mesh join verification."""

    @pytest.mark.asyncio
    async def test_verify_mesh_join_success(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test successful mesh verification."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"alive_peers": 5, "epoch": 42, "leader_id": "node-1"}',
                b"",
            )
            mock_exec.return_value = mock_proc

            result = await deployer._verify_mesh_join(
                "node-1",
                {"ssh_host": "192.168.1.1"},
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_verify_mesh_join_no_peers(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test mesh verification fails with no peers."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"alive_peers": 0, "epoch": 0}',
                b"",
            )
            mock_exec.return_value = mock_proc

            result = await deployer._verify_mesh_join(
                "node-1",
                {"ssh_host": "192.168.1.1"},
                max_retries=1,
                retry_delay=0.01,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_verify_mesh_join_invalid_json(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test mesh verification handles invalid JSON."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b"not json",
                b"",
            )
            mock_exec.return_value = mock_proc

            result = await deployer._verify_mesh_join(
                "node-1",
                {"ssh_host": "192.168.1.1"},
                max_retries=1,
                retry_delay=0.01,
            )

            assert result is False


# ============================================================================
# P2PAutoDeployer Check and Deploy Tests
# ============================================================================


class TestP2PAutoDeployerCheckAndDeploy:
    """Tests for check_and_deploy functionality."""

    @pytest.mark.asyncio
    async def test_check_and_deploy_all_healthy(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test check_and_deploy when all nodes are healthy."""
        with patch.object(
            deployer, "check_p2p_health", return_value=True
        ):
            report = await deployer.check_and_deploy()

            assert report.nodes_with_p2p == 2  # node-1 and node-2
            assert report.nodes_without_p2p == 0
            assert report.coverage_percent == 100.0

    @pytest.mark.asyncio
    async def test_check_and_deploy_some_missing(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test check_and_deploy with some nodes missing P2P."""
        call_count = [0]

        async def mock_health(*args, **kwargs):
            call_count[0] += 1
            return call_count[0] % 2 == 0  # Alternating healthy/missing

        with patch.object(deployer, "check_p2p_health", side_effect=mock_health):
            with patch.object(deployer, "deploy_p2p_to_node") as mock_deploy:
                mock_deploy.return_value = P2PDeploymentResult(
                    node_id="node-1",
                    success=True,
                    method="ssh_direct",
                    message="Success",
                    duration_seconds=1.0,
                )

                report = await deployer.check_and_deploy()

                # Deploy should have been called for missing nodes
                assert mock_deploy.called

    @pytest.mark.asyncio
    async def test_check_and_deploy_stores_history(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test that check_and_deploy stores history."""
        with patch.object(deployer, "check_p2p_health", return_value=True):
            await deployer.check_and_deploy()

            assert len(deployer._coverage_history) == 1
            assert deployer.get_latest_coverage() is not None


# ============================================================================
# P2PAutoDeployer Daemon Tests
# ============================================================================


class TestP2PAutoDeployerDaemon:
    """Tests for daemon functionality."""

    @pytest.mark.asyncio
    async def test_run_daemon_starts_and_stops(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test daemon can start and stop."""
        with patch.object(
            deployer, "check_and_deploy", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = P2PCoverageReport(
                total_nodes=2,
                nodes_with_p2p=2,
                nodes_without_p2p=0,
                unreachable_nodes=0,
                coverage_percent=100.0,
                nodes_needing_deployment=[],
            )

            # Start daemon in background
            task = asyncio.create_task(deployer.run_daemon())

            await asyncio.sleep(0.05)
            assert deployer._running is True

            # Stop daemon
            deployer.stop()
            await asyncio.sleep(0.05)

            assert deployer._running is False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def test_stop(self, deployer: P2PAutoDeployer) -> None:
        """Test stop method."""
        deployer._running = True
        deployer.stop()
        assert deployer._running is False


# ============================================================================
# P2PAutoDeployer Seed Peers Tests
# ============================================================================


class TestP2PAutoDeployerSeedPeers:
    """Tests for seed peer functionality."""

    def test_get_seed_peers_empty(self, deployer: P2PAutoDeployer) -> None:
        """Test getting seed peers when none configured."""
        with patch(
            "app.coordination.p2p_auto_deployer.get_p2p_voters",
            side_effect=ImportError,
        ):
            peers = deployer._get_seed_peers()

            # Should return empty or fallback peers
            assert isinstance(peers, str)

    def test_get_seed_peers_with_voters(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test getting seed peers with voter configuration."""
        deployer._hosts["voter-1"] = {"ssh_host": "192.168.1.100"}
        deployer._hosts["voter-2"] = {"ssh_host": "192.168.1.101"}

        with patch(
            "app.coordination.p2p_auto_deployer.get_p2p_voters",
            return_value=["voter-1", "voter-2"],
        ):
            peers = deployer._get_seed_peers()

            assert "192.168.1.100" in peers
            assert "8770" in peers


# ============================================================================
# P2PAutoDeployer Coverage Report Tests
# ============================================================================


class TestP2PAutoDeployerCoverage:
    """Tests for coverage reporting."""

    def test_get_latest_coverage_empty(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test getting latest coverage when no history."""
        result = deployer.get_latest_coverage()
        assert result is None

    def test_get_latest_coverage_with_history(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test getting latest coverage with history."""
        report = P2PCoverageReport(
            total_nodes=10,
            nodes_with_p2p=8,
            nodes_without_p2p=2,
            unreachable_nodes=0,
            coverage_percent=80.0,
            nodes_needing_deployment=["node-9", "node-10"],
        )
        deployer._coverage_history.append(report)

        result = deployer.get_latest_coverage()

        assert result is report
        assert result.coverage_percent == 80.0


# ============================================================================
# P2PAutoDeployer Load Config Tests
# ============================================================================


class TestP2PAutoDeployerLoadConfig:
    """Tests for configuration loading."""

    def test_load_hosts_from_cluster_config(self) -> None:
        """Test loading hosts from cluster_config."""
        mock_config = MagicMock()
        mock_config.hosts_raw = {
            "test-node": {"ssh_host": "1.2.3.4"}
        }

        with patch(
            "app.coordination.p2p_auto_deployer.load_cluster_config",
            return_value=mock_config,
        ):
            deployer = P2PAutoDeployer()

            assert "test-node" in deployer._hosts

    def test_load_hosts_fallback_yaml(self) -> None:
        """Test loading hosts from YAML fallback."""
        with patch(
            "app.coordination.p2p_auto_deployer.load_cluster_config",
            side_effect=ImportError,
        ):
            with patch("builtins.open", side_effect=FileNotFoundError):
                deployer = P2PAutoDeployer()

                assert deployer._hosts == {}


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestP2PAutoDeployerEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_check_p2p_health_exception(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test health check handles exceptions."""
        deployer.health_orchestrator = None

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("Connection refused"),
        ):
            result = await deployer.check_p2p_health(
                "node-1",
                {"ssh_host": "192.168.1.1"},
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_deploy_exception(
        self, deployer: P2PAutoDeployer
    ) -> None:
        """Test deployment handles exceptions."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=Exception("Unexpected error"),
        ):
            result = await deployer.deploy_p2p_to_node(
                "node-1",
                deployer._hosts["node-1"],
            )

            assert result.success is False
            assert "error" in result.message.lower()

    def test_deployable_hosts_empty_config(self) -> None:
        """Test getting deployable hosts with empty config."""
        with patch.object(P2PAutoDeployer, "_load_hosts_config"):
            deployer = P2PAutoDeployer()
            deployer._hosts = {}

            result = deployer._get_deployable_hosts()

            assert result == {}
