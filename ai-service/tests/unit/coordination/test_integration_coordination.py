"""Tests for coordination integration modules (December 2025).

Tests cover:
- Pipeline actions (StageCompletionResult, ActionConfig)
- Circuit breaker for fault tolerance
- Daemon adapters and DaemonManager integration
- Bandwidth-coordinated rsync
- New DaemonTypes and OrchestratorRoles
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStageCompletionResult:
    """Test StageCompletionResult data class."""

    def test_create_successful_result(self):
        """Test creating a successful stage result."""
        from app.coordination.pipeline_actions import StageCompletionResult

        result = StageCompletionResult(
            success=True,
            stage="data_sync",
            iteration=1,
            duration_seconds=10.5,
            output_path="/data/training/hex8_2p.npz",
        )
        assert result.success is True
        assert result.stage == "data_sync"
        assert result.iteration == 1
        assert result.duration_seconds == 10.5
        assert result.output_path == "/data/training/hex8_2p.npz"
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating a failed stage result."""
        from app.coordination.pipeline_actions import StageCompletionResult

        result = StageCompletionResult(
            success=False,
            stage="training",
            iteration=2,
            error="CUDA out of memory",
            exit_code=1,
        )
        assert result.success is False
        assert result.error == "CUDA out of memory"
        assert result.exit_code == 1

    def test_to_dict_conversion(self):
        """Test converting result to dictionary."""
        from app.coordination.pipeline_actions import StageCompletionResult

        result = StageCompletionResult(
            success=True,
            stage="npz_export",
            iteration=3,
            metadata={"samples_exported": 10000},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["stage"] == "npz_export"
        assert d["metadata"]["samples_exported"] == 10000


class TestActionConfig:
    """Test ActionConfig data class."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.pipeline_actions import ActionConfig

        config = ActionConfig()
        assert config.sync_timeout == 1800.0  # 30 minutes
        assert config.export_timeout == 3600.0  # 1 hour
        assert config.training_timeout == 86400.0  # 24 hours
        assert config.python_executable == "python3"

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.coordination.pipeline_actions import ActionConfig

        config = ActionConfig(
            sync_timeout=600.0,
            export_timeout=1800.0,
            python_executable="/usr/bin/python3.11",
        )
        assert config.sync_timeout == 600.0
        assert config.export_timeout == 1800.0
        assert config.python_executable == "/usr/bin/python3.11"


class TestCircuitBreaker:
    """Test CircuitBreaker for pipeline fault tolerance."""

    def test_initial_state_closed(self, tmp_path):
        """Test circuit starts in closed state."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(db_path=tmp_path / "circuit_breaker.db")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.can_execute() is True

    def test_opens_after_threshold_failures(self, tmp_path):
        """Test circuit opens after reaching failure threshold."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(
            failure_threshold=3,
            db_path=tmp_path / "circuit_breaker.db",
        )

        cb.record_failure("stage1", "Error 1")
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure("stage1", "Error 2")
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure("stage1", "Error 3")
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_in_half_open(self, tmp_path):
        """Test successful execution in half-open resets to closed."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # Fast reset for test
            db_path=tmp_path / "circuit_breaker.db",
        )

        # Trip the circuit
        cb.record_failure("stage1", "Error 1")
        cb.record_failure("stage1", "Error 2")
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for reset timeout
        deadline = time.time() + 0.5
        became_half_open = False
        while time.time() < deadline:
            if cb.can_execute():
                became_half_open = True
                break
            time.sleep(0.02)

        # Recovery is checked on can_execute(), not on raw state reads.
        assert became_half_open is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record success
        cb.record_success("stage1")
        assert cb.state == CircuitBreakerState.CLOSED

    def test_failure_in_half_open_reopens(self, tmp_path):
        """Test failure in half-open immediately reopens circuit."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            db_path=tmp_path / "circuit_breaker.db",
        )

        # Trip the circuit
        cb.record_failure("stage1", "Error 1")
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for half-open
        deadline = time.time() + 0.5
        became_half_open = False
        while time.time() < deadline:
            if cb.can_execute():
                became_half_open = True
                break
            time.sleep(0.02)

        assert became_half_open is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Fail in half-open
        cb.record_failure("stage1", "Error 2")
        assert cb.state == CircuitBreakerState.OPEN

    def test_get_status(self, tmp_path):
        """Test getting circuit breaker status."""
        from app.coordination.data_pipeline_orchestrator import CircuitBreaker

        cb = CircuitBreaker(
            failure_threshold=3,
            db_path=tmp_path / "circuit_breaker.db",
        )
        cb.record_failure("sync", "Error 1")
        cb.record_success("sync")

        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failures_by_stage" in status
        assert status["failures_by_stage"]["sync"] == 1

    def test_manual_reset(self, tmp_path):
        """Test manual circuit reset."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(
            failure_threshold=2,
            db_path=tmp_path / "circuit_breaker.db",
        )
        cb.record_failure("stage1", "Error 1")
        cb.record_failure("stage1", "Error 2")
        assert cb.state == CircuitBreakerState.OPEN

        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True


class TestDaemonTypes:
    """Test new DaemonType enum values."""

    def test_new_daemon_types_exist(self):
        """Test that new daemon types are defined."""
        from app.coordination.daemon_manager import DaemonType

        assert hasattr(DaemonType, "DISTILLATION")
        assert hasattr(DaemonType, "UNIFIED_PROMOTION")
        assert hasattr(DaemonType, "EXTERNAL_DRIVE_SYNC")
        assert hasattr(DaemonType, "VAST_CPU_PIPELINE")

    def test_daemon_type_values(self):
        """Test daemon type string values."""
        from app.coordination.daemon_manager import DaemonType

        assert DaemonType.DISTILLATION.value == "distillation"
        assert DaemonType.UNIFIED_PROMOTION.value == "unified_promotion"
        assert DaemonType.EXTERNAL_DRIVE_SYNC.value == "external_drive_sync"
        assert DaemonType.VAST_CPU_PIPELINE.value == "vast_cpu_pipeline"


class TestOrchestratorRoles:
    """Test new OrchestratorRole enum values."""

    def test_new_roles_exist(self):
        """Test that new orchestrator roles are defined."""
        from app.coordination.orchestrator_registry import OrchestratorRole

        assert hasattr(OrchestratorRole, "DISTILLATION_LEADER")
        assert hasattr(OrchestratorRole, "PROMOTION_LEADER")
        assert hasattr(OrchestratorRole, "EXTERNAL_SYNC_LEADER")
        assert hasattr(OrchestratorRole, "VAST_PIPELINE_LEADER")

    def test_role_values(self):
        """Test orchestrator role string values."""
        from app.coordination.orchestrator_registry import OrchestratorRole

        assert OrchestratorRole.DISTILLATION_LEADER.value == "distillation_leader"
        assert OrchestratorRole.PROMOTION_LEADER.value == "promotion_leader"
        assert OrchestratorRole.EXTERNAL_SYNC_LEADER.value == "external_sync_leader"
        assert OrchestratorRole.VAST_PIPELINE_LEADER.value == "vast_pipeline_leader"


class TestDaemonAdapters:
    """Test daemon adapter classes."""

    def test_adapter_config_defaults(self):
        """Test DaemonAdapterConfig default values."""
        from app.coordination.daemon_adapters import DaemonAdapterConfig

        config = DaemonAdapterConfig()
        assert config.acquire_role is True
        assert config.health_check_interval == 60.0
        assert config.auto_restart is True
        assert config.max_restarts == 5

    def test_get_available_adapters(self):
        """Test getting list of available adapters."""
        from app.coordination.daemon_adapters import get_available_adapters
        from app.coordination.daemon_manager import DaemonType

        adapters = get_available_adapters()
        assert DaemonType.DISTILLATION in adapters
        assert DaemonType.UNIFIED_PROMOTION in adapters
        assert DaemonType.EXTERNAL_DRIVE_SYNC in adapters
        assert DaemonType.VAST_CPU_PIPELINE in adapters

    def test_get_daemon_adapter(self):
        """Test getting adapter instance by daemon type."""
        from app.coordination.daemon_adapters import (
            ConfigurableDaemonAdapter,
            get_daemon_adapter,
        )
        from app.coordination.daemon_manager import DaemonType

        adapter = get_daemon_adapter(DaemonType.DISTILLATION)
        assert adapter is not None
        assert isinstance(adapter, ConfigurableDaemonAdapter)
        assert adapter.daemon_type == DaemonType.DISTILLATION

    def test_adapter_returns_none_for_unknown(self):
        """Test that get_daemon_adapter returns None for unknown types."""
        from app.coordination.daemon_adapters import get_daemon_adapter
        from app.coordination.daemon_manager import DaemonType

        adapter = get_daemon_adapter(DaemonType.SYNC_COORDINATOR)
        assert adapter is None

    def test_adapter_get_status(self):
        """Test getting adapter status."""
        from app.coordination.daemon_adapters import get_daemon_adapter
        from app.coordination.daemon_manager import DaemonType
        from app.coordination.orchestrator_registry import OrchestratorRole

        adapter = get_daemon_adapter(DaemonType.DISTILLATION)
        status = adapter.get_status()

        assert "daemon_type" in status
        assert "role" in status
        assert "running" in status
        assert "healthy" in status
        assert status["daemon_type"] == "distillation"
        assert status["role"] == OrchestratorRole.DISTILLATION_LEADER.value


class TestBandwidthManager:
    """Test BandwidthManager for rsync coordination."""

    def test_bandwidth_config_defaults(self):
        """Test BandwidthConfig default values."""
        from app.coordination.sync_bandwidth import BandwidthConfig

        config = BandwidthConfig()
        assert config.default_bwlimit_kbps == 25000  # 25 MB/s
        assert config.max_concurrent_per_host == 5
        assert config.max_concurrent_total == 12

    def test_bandwidth_allocation_creation(self):
        """Test BandwidthAllocation creation."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation,
            TransferPriority,
        )

        allocation = BandwidthAllocation(
            host="lambda-gh200-a",
            priority=TransferPriority.HIGH,
            bwlimit_kbps=20000,
            transfer_id="transfer_1",
        )
        assert allocation.host == "lambda-gh200-a"
        assert allocation.priority == TransferPriority.HIGH
        assert allocation.bwlimit_kbps == 20000

    def test_allocation_expiration(self):
        """Test allocation expiration check."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation,
            TransferPriority,
        )

        # Non-expired allocation
        allocation = BandwidthAllocation(
            host="host1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() + 3600,
        )
        assert allocation.is_expired is False

        # Expired allocation
        expired_allocation = BandwidthAllocation(
            host="host1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() - 1,
        )
        assert expired_allocation.is_expired is True

    @pytest.mark.asyncio
    async def test_bandwidth_manager_allocation(self):
        """Test requesting and releasing bandwidth allocations."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig,
            BandwidthManager,
            TransferPriority,
        )

        # Create manager with custom config
        config = BandwidthConfig(max_concurrent_per_host=2)
        manager = BandwidthManager(config)

        # Request allocation
        allocation = await manager.request_allocation(
            host="test-host",
            priority=TransferPriority.NORMAL,
            timeout=1.0,
        )

        assert allocation is not None
        assert allocation.host == "test-host"
        assert allocation.bwlimit_kbps > 0

        # Check status
        status = manager.get_status()
        assert status["active_transfers"] == 1
        assert "test-host" in status["per_host"]

        # Release allocation
        await manager.release_allocation(allocation)
        status = manager.get_status()
        assert status["active_transfers"] == 0

    @pytest.mark.asyncio
    async def test_bandwidth_manager_concurrent_limit(self):
        """Test that concurrent limits are enforced."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig,
            BandwidthManager,
            TransferPriority,
        )

        config = BandwidthConfig(
            max_concurrent_per_host=2,
            max_concurrent_total=3,
        )
        manager = BandwidthManager(config)

        # Allocate up to per-host limit
        alloc1 = await manager.request_allocation("host1", TransferPriority.NORMAL, 1.0)
        alloc2 = await manager.request_allocation("host1", TransferPriority.NORMAL, 1.0)
        assert alloc1 is not None
        assert alloc2 is not None

        # Third allocation to same host should timeout (over per-host limit)
        alloc3 = await manager.request_allocation("host1", TransferPriority.NORMAL, 0.1)
        assert alloc3 is None

        # But allocation to different host should work (under total limit)
        alloc4 = await manager.request_allocation("host2", TransferPriority.NORMAL, 1.0)
        assert alloc4 is not None

        # Cleanup
        await manager.release_allocation(alloc1)
        await manager.release_allocation(alloc2)
        await manager.release_allocation(alloc4)


class TestTransferPriority:
    """Test TransferPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        from app.coordination.sync_bandwidth import TransferPriority

        assert TransferPriority.LOW.value == "low"
        assert TransferPriority.NORMAL.value == "normal"
        assert TransferPriority.HIGH.value == "high"
        assert TransferPriority.CRITICAL.value == "critical"


class TestSyncResult:
    """Test SyncResult data class."""

    def test_sync_result_success(self):
        """Test successful sync result."""
        from app.coordination.sync_bandwidth import SyncResult

        result = SyncResult(
            success=True,
            source="/data/games/",
            dest="ubuntu@remote:/data/games/",
            host="lambda-gh200-a",
            bytes_transferred=1024 * 1024 * 100,  # 100 MB
            duration_seconds=10.0,
            bwlimit_kbps=10000,
            effective_rate_kbps=10240.0,
        )
        assert result.success is True
        assert result.bytes_transferred == 104857600
        assert result.effective_rate_kbps == 10240.0

    def test_sync_result_failure(self):
        """Test failed sync result."""
        from app.coordination.sync_bandwidth import SyncResult

        result = SyncResult(
            success=False,
            source="/data/",
            dest="remote:/data/",
            host="offline-host",
            error="Connection refused",
            exit_code=12,
        )
        assert result.success is False
        assert result.error == "Connection refused"
        assert result.exit_code == 12


class TestPipelineConfig:
    """Test PipelineConfig per-stage controls."""

    def test_per_stage_defaults(self):
        """Test per-stage auto-trigger defaults."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        # auto_trigger is True by default (changed in Dec 2025)
        assert config.auto_trigger is True
        assert config.auto_trigger_sync is True
        assert config.auto_trigger_export is True
        assert config.auto_trigger_training is True
        assert config.auto_trigger_evaluation is True
        assert config.auto_trigger_promotion is True

    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        from app.coordination.coordinator_config import PipelineConfig

        config = PipelineConfig()
        assert config.circuit_breaker_enabled is True
        assert config.circuit_breaker_failure_threshold == 3
        assert config.circuit_breaker_reset_timeout_seconds == 300.0


class TestActionPriority:
    """Test ActionPriority enum."""

    def test_priority_values(self):
        """Test action priority enum values."""
        from app.coordination.pipeline_actions import ActionPriority

        assert ActionPriority.LOW.value == "low"
        assert ActionPriority.NORMAL.value == "normal"
        assert ActionPriority.HIGH.value == "high"
        assert ActionPriority.CRITICAL.value == "critical"
