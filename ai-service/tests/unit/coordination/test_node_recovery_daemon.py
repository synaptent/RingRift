"""Tests for Node Recovery Daemon.

Tests the cluster node monitoring and recovery daemon that handles:
- Node state tracking
- Provider detection (Lambda, Vast, RunPod, Hetzner)
- Recovery action determination
- Proactive recovery based on resource trends
- Health check integration

December 27, 2025: Created to address P1 test gap for node_recovery_daemon.py.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_recovery_daemon import (
    NodeInfo,
    NodeProvider,
    NodeRecoveryAction,
    NodeRecoveryConfig,
    NodeRecoveryDaemon,
    RecoveryAction,
    RecoveryStats,
    get_node_recovery_daemon,
    reset_node_recovery_daemon,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestNodeRecoveryAction:
    """Tests for NodeRecoveryAction enum."""

    def test_action_values(self):
        """Should have expected action values."""
        assert NodeRecoveryAction.NONE.value == "none"
        assert NodeRecoveryAction.RESTART.value == "restart"
        assert NodeRecoveryAction.PREEMPTIVE_RESTART.value == "preemptive_restart"
        assert NodeRecoveryAction.NOTIFY.value == "notify"
        assert NodeRecoveryAction.FAILOVER.value == "failover"

    def test_backward_compat_alias(self):
        """Should have RecoveryAction alias for backward compatibility."""
        assert RecoveryAction is NodeRecoveryAction


class TestNodeProvider:
    """Tests for NodeProvider enum."""

    def test_provider_values(self):
        """Should have expected provider values."""
        assert NodeProvider.LAMBDA.value == "lambda"
        assert NodeProvider.VAST.value == "vast"
        assert NodeProvider.RUNPOD.value == "runpod"
        assert NodeProvider.HETZNER.value == "hetzner"
        assert NodeProvider.UNKNOWN.value == "unknown"


# =============================================================================
# Config Tests
# =============================================================================


class TestNodeRecoveryConfig:
    """Tests for NodeRecoveryConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = NodeRecoveryConfig()
        assert config.enabled is True
        # Default from DaemonConfig base class is 300 (5 min)
        assert config.check_interval_seconds == 300
        assert config.max_consecutive_failures == 3
        assert config.recovery_cooldown_seconds == 600
        assert config.memory_exhaustion_threshold == 0.02
        assert config.memory_exhaustion_window_minutes == 30
        assert config.preemptive_recovery_enabled is True

    def test_api_key_defaults(self):
        """Should have empty API key defaults."""
        config = NodeRecoveryConfig()
        assert config.lambda_api_key == ""
        assert config.vast_api_key == ""
        assert config.runpod_api_key == ""

    def test_from_env_reads_enabled(self):
        """Should read enabled flag from environment."""
        with patch.dict("os.environ", {"RINGRIFT_NODE_RECOVERY_ENABLED": "0"}):
            config = NodeRecoveryConfig.from_env()
            assert config.enabled is False

    def test_from_env_reads_interval(self):
        """Should read interval from environment."""
        with patch.dict("os.environ", {"RINGRIFT_NODE_RECOVERY_INTERVAL": "120"}):
            config = NodeRecoveryConfig.from_env()
            assert config.check_interval_seconds == 120

    def test_from_env_reads_api_keys(self):
        """Should read API keys from environment."""
        with patch.dict(
            "os.environ",
            {
                "LAMBDA_API_KEY": "lambda-key",
                "VAST_API_KEY": "vast-key",
                "RUNPOD_API_KEY": "runpod-key",
            },
        ):
            config = NodeRecoveryConfig.from_env()
            assert config.lambda_api_key == "lambda-key"
            assert config.vast_api_key == "vast-key"
            assert config.runpod_api_key == "runpod-key"


# =============================================================================
# NodeInfo Tests
# =============================================================================


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        node = NodeInfo(node_id="test-node", host="192.168.1.1")
        assert node.node_id == "test-node"
        assert node.host == "192.168.1.1"
        assert node.provider == NodeProvider.UNKNOWN
        assert node.status == "unknown"
        assert node.last_seen == 0.0
        assert node.consecutive_failures == 0
        assert node.last_recovery_attempt == 0.0
        assert node.memory_samples == []
        assert node.sample_timestamps == []

    def test_custom_values(self):
        """Should accept custom values."""
        node = NodeInfo(
            node_id="lambda-h100",
            host="10.0.0.1",
            provider=NodeProvider.LAMBDA,
            status="running",
            consecutive_failures=2,
        )
        assert node.provider == NodeProvider.LAMBDA
        assert node.status == "running"
        assert node.consecutive_failures == 2


# =============================================================================
# RecoveryStats Tests
# =============================================================================


class TestRecoveryStats:
    """Tests for RecoveryStats dataclass."""

    def test_default_values(self):
        """Should have zero defaults."""
        stats = RecoveryStats()
        assert stats.jobs_processed == 0
        assert stats.jobs_succeeded == 0
        assert stats.jobs_failed == 0
        assert stats.preemptive_recoveries == 0

    def test_backward_compat_aliases(self):
        """Should have backward compatibility aliases."""
        stats = RecoveryStats()
        stats.jobs_processed = 10
        stats.jobs_succeeded = 5
        stats.jobs_failed = 2

        assert stats.total_checks == 10
        assert stats.nodes_recovered == 5
        assert stats.recovery_failures == 2

    def test_record_check(self):
        """Should record a node check."""
        stats = RecoveryStats()
        stats.record_check()
        assert stats.jobs_processed == 1
        assert stats.last_job_time > 0

    def test_record_recovery_success(self):
        """Should record successful recovery."""
        stats = RecoveryStats()
        stats.record_recovery_success()
        assert stats.jobs_succeeded == 1
        assert stats.preemptive_recoveries == 0

    def test_record_recovery_success_preemptive(self):
        """Should track preemptive recoveries separately."""
        stats = RecoveryStats()
        stats.record_recovery_success(preemptive=True)
        assert stats.jobs_succeeded == 1
        assert stats.preemptive_recoveries == 1

    def test_record_recovery_failure(self):
        """Should record failed recovery."""
        stats = RecoveryStats()
        stats.record_recovery_failure("API timeout")
        assert stats.jobs_failed == 1
        assert stats.last_error == "API timeout"


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestNodeRecoveryDaemonInit:
    """Tests for NodeRecoveryDaemon initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        daemon = NodeRecoveryDaemon()
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._stats is not None
        assert daemon._node_states == {}
        assert daemon._http_session is None

    def test_custom_config(self):
        """Should accept custom config."""
        config = NodeRecoveryConfig(
            enabled=False,
            max_consecutive_failures=5,
        )
        daemon = NodeRecoveryDaemon(config=config)
        assert daemon.config.enabled is False
        assert daemon.config.max_consecutive_failures == 5


# =============================================================================
# Provider Detection Tests
# =============================================================================


class TestProviderDetection:
    """Tests for provider detection logic."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    def test_detect_lambda_from_node_id(self, daemon):
        """Should detect Lambda from node ID."""
        provider = daemon._detect_provider("lambda-h100-1", {})
        assert provider == NodeProvider.LAMBDA

    def test_detect_lambda_from_info(self, daemon):
        """Should detect Lambda from info dict."""
        provider = daemon._detect_provider("node-1", {"provider": "lambda"})
        assert provider == NodeProvider.LAMBDA

    def test_detect_vast_from_node_id(self, daemon):
        """Should detect Vast from node ID."""
        provider = daemon._detect_provider("vast-12345", {})
        assert provider == NodeProvider.VAST

    def test_detect_runpod_from_node_id(self, daemon):
        """Should detect RunPod from node ID."""
        provider = daemon._detect_provider("runpod-abc123", {})
        assert provider == NodeProvider.RUNPOD

    def test_detect_hetzner_from_node_id(self, daemon):
        """Should detect Hetzner from node ID."""
        provider = daemon._detect_provider("hetzner-cpu1", {})
        assert provider == NodeProvider.HETZNER

    def test_unknown_provider(self, daemon):
        """Should return UNKNOWN for unrecognized nodes."""
        provider = daemon._detect_provider("some-random-node", {})
        assert provider == NodeProvider.UNKNOWN


# =============================================================================
# Recovery Action Determination Tests
# =============================================================================


class TestDetermineRecoveryAction:
    """Tests for recovery action determination logic."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    def test_no_action_for_running_node(self, daemon):
        """Should return NONE for running node."""
        node = NodeInfo(node_id="test", host="1.1.1.1", status="running")
        action = daemon._determine_recovery_action(node)
        assert action == NodeRecoveryAction.NONE

    def test_notify_for_few_failures(self, daemon):
        """Should notify when failures below threshold."""
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            status="unreachable",
            consecutive_failures=1,
        )
        action = daemon._determine_recovery_action(node)
        assert action == NodeRecoveryAction.NOTIFY

    def test_restart_for_many_failures(self, daemon):
        """Should restart when failures exceed threshold."""
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            status="unreachable",
            consecutive_failures=3,
        )
        action = daemon._determine_recovery_action(node)
        assert action == NodeRecoveryAction.RESTART

    def test_restart_for_terminated_node(self, daemon):
        """Should restart terminated node with many failures."""
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            status="terminated",
            consecutive_failures=3,
        )
        action = daemon._determine_recovery_action(node)
        assert action == NodeRecoveryAction.RESTART

    def test_cooldown_prevents_action(self, daemon):
        """Should not act during cooldown period."""
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            status="unreachable",
            consecutive_failures=5,
            last_recovery_attempt=time.time(),  # Just now
        )
        action = daemon._determine_recovery_action(node)
        assert action == NodeRecoveryAction.NONE


# =============================================================================
# Resource Trend Analysis Tests
# =============================================================================


class TestResourceTrends:
    """Tests for proactive recovery based on resource trends."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    def test_no_action_with_few_samples(self, daemon):
        """Should return NONE with too few samples."""
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            memory_samples=[50.0, 55.0],
            sample_timestamps=[time.time() - 60, time.time()],
        )
        action = daemon._check_resource_trends(node)
        assert action == NodeRecoveryAction.NONE

    def test_no_action_for_stable_memory(self, daemon):
        """Should return NONE for stable memory usage."""
        now = time.time()
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            memory_samples=[50.0, 50.5, 51.0, 50.8, 51.2],
            sample_timestamps=[now - 300, now - 240, now - 180, now - 120, now - 60],
        )
        action = daemon._check_resource_trends(node)
        assert action == NodeRecoveryAction.NONE

    def test_preemptive_restart_for_memory_leak(self, daemon):
        """Should trigger preemptive restart for memory leak."""
        now = time.time()
        # Simulate memory growing from 50% to 90% in 20 minutes = 2%/min
        node = NodeInfo(
            node_id="test",
            host="1.1.1.1",
            memory_samples=[50.0, 60.0, 70.0, 80.0, 90.0],
            sample_timestamps=[
                now - 1200,
                now - 900,
                now - 600,
                now - 300,
                now,
            ],
        )
        action = daemon._check_resource_trends(node)
        assert action == NodeRecoveryAction.PREEMPTIVE_RESTART


# =============================================================================
# Node State Update Tests
# =============================================================================


class TestNodeStateUpdate:
    """Tests for node state update logic."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    def test_update_creates_new_node(self, daemon):
        """Should create new NodeInfo for unknown nodes."""
        daemon._update_node_info(
            "new-node",
            {"host": "10.0.0.1", "provider": "vast"},
            "running",
        )
        assert "new-node" in daemon._node_states
        assert daemon._node_states["new-node"].host == "10.0.0.1"
        assert daemon._node_states["new-node"].provider == NodeProvider.VAST

    def test_update_existing_node(self, daemon):
        """Should update existing node state."""
        # First update creates the node
        daemon._update_node_info("node-1", {"host": "10.0.0.1"}, "running")
        assert daemon._node_states["node-1"].consecutive_failures == 0

        # Simulate a failure
        daemon._node_states["node-1"].consecutive_failures = 2
        daemon._node_states["node-1"].status = "unreachable"

        # Update back to running should reset failures
        daemon._update_node_info("node-1", {"host": "10.0.0.1"}, "running")
        assert daemon._node_states["node-1"].consecutive_failures == 0
        assert daemon._node_states["node-1"].status == "running"

    def test_update_tracks_memory(self, daemon):
        """Should track memory samples."""
        daemon._update_node_info(
            "node-1",
            {"host": "10.0.0.1", "memory_used_percent": 50.0},
            "running",
        )
        assert len(daemon._node_states["node-1"].memory_samples) == 1
        assert daemon._node_states["node-1"].memory_samples[0] == 50.0


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for daemon health check."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        daemon = NodeRecoveryDaemon()
        daemon._running = True
        return daemon

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, daemon):
        """Should report unhealthy when not running."""
        daemon._running = False
        result = await daemon.health_check()
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_running_healthy(self, daemon):
        """Should report healthy when running normally."""
        daemon._stats.last_job_time = time.time()
        result = await daemon.health_check()
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_health_check_stale_data(self, daemon):
        """Should report unhealthy with stale check data."""
        daemon._stats.last_job_time = time.time() - 3600  # 1 hour ago
        result = await daemon.health_check()
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_all_failures(self, daemon):
        """Should report unhealthy with only failures."""
        daemon._stats.last_job_time = time.time()
        daemon._stats.jobs_failed = 15
        daemon._stats.jobs_succeeded = 0
        result = await daemon.health_check()
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_some_failures_ok(self, daemon):
        """Should be healthy with some failures if also successes."""
        daemon._stats.last_job_time = time.time()
        daemon._stats.jobs_failed = 15
        daemon._stats.jobs_succeeded = 10  # Some successes
        result = await daemon.health_check()
        assert result.healthy is True


# =============================================================================
# Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        daemon = NodeRecoveryDaemon()
        daemon._running = True
        daemon._stats.jobs_processed = 100
        daemon._stats.jobs_succeeded = 5
        daemon._stats.jobs_failed = 2
        daemon._stats.preemptive_recoveries = 1
        return daemon

    def test_get_status_includes_recovery_stats(self, daemon):
        """Should include recovery-specific stats."""
        status = daemon.get_status()

        assert "recovery_stats" in status
        assert status["recovery_stats"]["total_checks"] == 100
        assert status["recovery_stats"]["nodes_recovered"] == 5
        assert status["recovery_stats"]["recovery_failures"] == 2
        assert status["recovery_stats"]["preemptive_recoveries"] == 1

    def test_get_status_includes_tracked_nodes(self, daemon):
        """Should include tracked node count."""
        daemon._update_node_info("node-1", {"host": "1.1.1.1"}, "running")
        daemon._update_node_info("node-2", {"host": "2.2.2.2"}, "running")

        status = daemon.get_status()

        assert status["tracked_nodes"] == 2
        assert "nodes" in status
        assert "node-1" in status["nodes"]


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_node_recovery_daemon()

    def test_get_returns_singleton(self):
        """Should return same instance."""
        daemon1 = get_node_recovery_daemon()
        daemon2 = get_node_recovery_daemon()
        assert daemon1 is daemon2

    def test_reset_clears_singleton(self):
        """Reset should allow new instance."""
        daemon1 = get_node_recovery_daemon()
        reset_node_recovery_daemon()
        daemon2 = get_node_recovery_daemon()
        # After reset, new instance created
        assert daemon2._running is False
        assert daemon2._node_states == {}


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription handling."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    def test_on_nodes_dead_updates_state(self, daemon):
        """Should update node state on death event."""
        # First, create the node
        daemon._update_node_info("node-1", {"host": "1.1.1.1"}, "running")

        # Simulate death event
        event = {"nodes": ["node-1"]}
        daemon._on_nodes_dead(event)

        assert daemon._node_states["node-1"].status == "unreachable"
        assert daemon._node_states["node-1"].consecutive_failures == 1

    def test_on_nodes_dead_with_payload_attr(self, daemon):
        """Should handle event with payload attribute."""
        daemon._update_node_info("node-1", {"host": "1.1.1.1"}, "running")

        # Simulate event with payload attribute
        event = MagicMock()
        event.payload = {"nodes": ["node-1"]}
        daemon._on_nodes_dead(event)

        assert daemon._node_states["node-1"].status == "unreachable"


# =============================================================================
# Recovery Execution Tests
# =============================================================================


class TestRecoveryExecution:
    """Tests for recovery action execution."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    @pytest.mark.asyncio
    async def test_notify_action_succeeds(self, daemon):
        """Should succeed for notify action."""
        node = NodeInfo(node_id="test", host="1.1.1.1")

        with patch.object(daemon, "_emit_recovery_event"):
            result = await daemon._execute_recovery(node, NodeRecoveryAction.NOTIFY)

        assert result is True
        assert node.last_recovery_attempt > 0

    @pytest.mark.asyncio
    async def test_restart_lambda_without_api_key(self, daemon):
        """Should fail Lambda restart without API key."""
        daemon.config.lambda_api_key = ""
        node = NodeInfo(
            node_id="lambda-test",
            host="1.1.1.1",
            provider=NodeProvider.LAMBDA,
            instance_id="i-12345",
        )

        result = await daemon._restart_lambda_node(node)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_lambda_without_instance_id(self, daemon):
        """Should fail Lambda restart without instance ID."""
        daemon.config.lambda_api_key = "test-key"
        node = NodeInfo(
            node_id="lambda-test",
            host="1.1.1.1",
            provider=NodeProvider.LAMBDA,
            instance_id="",  # No instance ID
        )

        result = await daemon._restart_lambda_node(node)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_unknown_provider_fails(self, daemon):
        """Should fail restart for unknown provider."""
        node = NodeInfo(
            node_id="unknown-node",
            host="1.1.1.1",
            provider=NodeProvider.UNKNOWN,
        )

        result = await daemon._restart_node(node)
        assert result is False


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestDaemonLifecycle:
    """Tests for daemon lifecycle methods."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return NodeRecoveryDaemon()

    @pytest.mark.asyncio
    async def test_on_start_logs_config(self, daemon):
        """Should log configuration on start."""
        with patch.object(daemon, "_subscribe_to_events"):
            await daemon._on_start()
        # Just verify no exception raised

    @pytest.mark.asyncio
    async def test_on_stop_closes_http_session(self, daemon):
        """Should close HTTP session on stop."""
        mock_session = AsyncMock()
        daemon._http_session = mock_session

        with patch(
            "app.coordination.event_emitters.emit_coordinator_shutdown",
            new_callable=AsyncMock,
        ):
            await daemon._on_stop()

        mock_session.close.assert_called_once()
