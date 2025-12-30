"""Unit tests for sync_bandwidth module (December 2025).

Tests cover:
- TransferPriority: Priority enum for bandwidth allocation
- BandwidthAllocation: Allocation tracking dataclass
- BandwidthConfig: Configuration dataclass
- BandwidthManager: Bandwidth allocation manager (singleton)
- BandwidthCoordinatedRsync: Rsync with bandwidth coordination
- BatchSyncResult: Result of batch sync operations
- BatchRsync: Batch rsync operations
- Module functions: get_bandwidth_manager, get_coordinated_rsync, etc.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# TransferPriority Tests
# =============================================================================


class TestTransferPriority:
    """Tests for TransferPriority enum."""

    def test_all_priorities_defined(self):
        """Test all expected priorities are defined."""
        from app.coordination.sync_bandwidth import TransferPriority

        expected = ["BACKGROUND", "LOW", "NORMAL", "HIGH", "CRITICAL"]
        for name in expected:
            assert hasattr(TransferPriority, name)

    def test_priority_values(self):
        """Test priority values are strings."""
        from app.coordination.sync_bandwidth import TransferPriority

        assert TransferPriority.NORMAL.value == "normal"
        assert TransferPriority.HIGH.value == "high"
        assert TransferPriority.CRITICAL.value == "critical"

    def test_priority_comparison(self):
        """Test priorities are comparable."""
        from app.coordination.sync_bandwidth import TransferPriority

        assert TransferPriority.NORMAL != TransferPriority.HIGH


# =============================================================================
# BandwidthAllocation Tests
# =============================================================================


class TestBandwidthAllocation:
    """Tests for BandwidthAllocation dataclass."""

    def test_basic_creation(self):
        """Test basic BandwidthAllocation creation."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
        )

        assert alloc.host == "gpu-node-1"
        assert alloc.priority == TransferPriority.NORMAL
        assert alloc.bwlimit_kbps == 10000
        assert alloc.granted is True  # Default

    def test_is_expired_not_expired(self):
        """Test is_expired returns False when not expired."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() + 3600,  # Expires in 1 hour
        )

        assert alloc.is_expired is False

    def test_is_expired_expired(self):
        """Test is_expired returns True when expired."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() - 10,  # Already expired
        )

        assert alloc.is_expired is True

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when no expiry set."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=0.0,  # No expiry
        )

        assert alloc.is_expired is False

    def test_bwlimit_mbps_conversion(self):
        """Test bwlimit_mbps property converts correctly."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=12500,  # 12500 KB/s = 100 Mbps
        )

        assert alloc.bwlimit_mbps == 100.0

    def test_to_dict(self):
        """Test to_dict returns expected structure."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation, TransferPriority
        )

        alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.HIGH,
            bwlimit_kbps=50000,
            transfer_id="test-123",
        )

        result = alloc.to_dict()

        assert result["host"] == "gpu-node-1"
        assert result["priority"] == "high"
        assert result["bwlimit_kbps"] == 50000
        assert result["transfer_id"] == "test-123"
        assert "bwlimit_mbps" in result
        assert "is_expired" in result


# =============================================================================
# BandwidthConfig Tests
# =============================================================================


class TestBandwidthConfig:
    """Tests for BandwidthConfig dataclass."""

    def test_default_values(self):
        """Test BandwidthConfig has sensible defaults."""
        from app.coordination.sync_bandwidth import BandwidthConfig

        config = BandwidthConfig()

        assert config.default_bwlimit_kbps > 0
        assert config.max_bwlimit_kbps > config.default_bwlimit_kbps
        assert config.min_bwlimit_kbps > 0
        assert config.per_host_limit_kbps > 0
        assert config.total_limit_kbps > 0
        assert config.max_concurrent_per_host > 0
        assert config.max_concurrent_total >= config.max_concurrent_per_host

    def test_custom_values(self):
        """Test BandwidthConfig with custom values."""
        from app.coordination.sync_bandwidth import BandwidthConfig

        config = BandwidthConfig(
            default_bwlimit_kbps=5000,
            max_bwlimit_kbps=100000,
            per_host_limit_kbps=25000,
            max_concurrent_per_host=3,
        )

        assert config.default_bwlimit_kbps == 5000
        assert config.max_bwlimit_kbps == 100000
        assert config.per_host_limit_kbps == 25000
        assert config.max_concurrent_per_host == 3

    def test_priority_multipliers(self):
        """Test priority multipliers are set."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig, TransferPriority
        )

        config = BandwidthConfig()

        assert TransferPriority.NORMAL in config.priority_multipliers
        assert TransferPriority.HIGH in config.priority_multipliers
        assert config.priority_multipliers[TransferPriority.NORMAL] == 1.0
        assert config.priority_multipliers[TransferPriority.HIGH] > 1.0


# =============================================================================
# BandwidthManager Tests
# =============================================================================


@pytest.fixture(autouse=True)
def reset_bandwidth_manager_singleton():
    """Reset BandwidthManager singleton between tests."""
    from app.coordination.sync_bandwidth import BandwidthManager

    BandwidthManager.reset_instance()
    yield
    BandwidthManager.reset_instance()


class TestBandwidthManagerInit:
    """Tests for BandwidthManager initialization."""

    def test_initialization_defaults(self):
        """Test BandwidthManager initializes with defaults."""
        from app.coordination.sync_bandwidth import BandwidthManager

        manager = BandwidthManager()

        assert manager.config is not None
        assert isinstance(manager._allocations, dict)
        assert len(manager._allocations) == 0

    def test_initialization_with_config(self):
        """Test BandwidthManager with custom config."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager, BandwidthConfig
        )

        config = BandwidthConfig(max_concurrent_per_host=10)
        manager = BandwidthManager(config=config)

        assert manager.config.max_concurrent_per_host == 10

    def test_singleton_pattern(self):
        """Test get_instance returns singleton."""
        from app.coordination.sync_bandwidth import BandwidthManager

        manager1 = BandwidthManager.get_instance()
        manager2 = BandwidthManager.get_instance()

        assert manager1 is manager2

    def test_reset_instance(self):
        """Test reset_instance clears singleton."""
        from app.coordination.sync_bandwidth import BandwidthManager

        manager1 = BandwidthManager.get_instance()
        BandwidthManager.reset_instance()
        manager2 = BandwidthManager.get_instance()

        assert manager1 is not manager2


class TestBandwidthManagerAllocation:
    """Tests for BandwidthManager allocation methods."""

    @pytest.mark.asyncio
    async def test_request_allocation_success(self):
        """Test successful bandwidth allocation request."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager, TransferPriority
        )

        manager = BandwidthManager()

        allocation = await manager.request_allocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            timeout=5.0,
        )

        assert allocation is not None
        assert allocation.host == "gpu-node-1"
        assert allocation.bwlimit_kbps > 0

    @pytest.mark.asyncio
    async def test_release_allocation(self):
        """Test releasing an allocation."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager, TransferPriority
        )

        manager = BandwidthManager()

        # Get allocation
        allocation = await manager.request_allocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
        )

        assert allocation is not None
        initial_count = len(manager._allocations)

        # Release it
        await manager.release_allocation(allocation)

        assert len(manager._allocations) < initial_count

    def test_can_allocate_within_limits(self):
        """Test _can_allocate returns True within limits."""
        from app.coordination.sync_bandwidth import BandwidthManager, BandwidthConfig

        config = BandwidthConfig(max_concurrent_per_host=5, max_concurrent_total=10)
        manager = BandwidthManager(config=config)

        result = manager._can_allocate("gpu-node-1")

        assert result is True

    def test_can_allocate_at_host_limit(self):
        """Test _can_allocate returns False at per-host limit."""
        from app.coordination.sync_bandwidth import BandwidthManager, BandwidthConfig

        config = BandwidthConfig(max_concurrent_per_host=2)
        manager = BandwidthManager(config=config)

        # Simulate reaching host limit
        manager._host_transfers["gpu-node-1"] = 2

        result = manager._can_allocate("gpu-node-1")

        assert result is False


class TestBandwidthManagerStatus:
    """Tests for BandwidthManager status methods."""

    def test_get_status(self):
        """Test get_status returns expected structure."""
        from app.coordination.sync_bandwidth import BandwidthManager

        manager = BandwidthManager()

        status = manager.get_status()

        assert isinstance(status, dict)
        assert "active_allocations" in status or "allocations" in status


# =============================================================================
# BandwidthCoordinatedRsync Tests
# =============================================================================


class TestBandwidthCoordinatedRsync:
    """Tests for BandwidthCoordinatedRsync class."""

    def test_initialization(self):
        """Test BandwidthCoordinatedRsync initializes correctly."""
        from app.coordination.sync_bandwidth import BandwidthCoordinatedRsync

        rsync = BandwidthCoordinatedRsync()

        assert rsync.manager is not None

    def test_initialization_with_manager(self):
        """Test BandwidthCoordinatedRsync with custom manager."""
        from app.coordination.sync_bandwidth import (
            BandwidthCoordinatedRsync, BandwidthManager
        )

        manager = BandwidthManager()
        rsync = BandwidthCoordinatedRsync(manager=manager)

        assert rsync.manager is manager


# =============================================================================
# BatchSyncResult Tests
# =============================================================================


class TestBatchSyncResult:
    """Tests for BatchSyncResult dataclass."""

    def test_basic_creation(self):
        """Test basic BatchSyncResult creation."""
        from app.coordination.sync_bandwidth import BatchSyncResult

        result = BatchSyncResult(
            success=True,
            source_dir="/data/games",
            dest="gpu-node-1:/data/games",
            host="gpu-node-1",
            files_transferred=100,
        )

        assert result.host == "gpu-node-1"
        assert result.success is True
        assert result.files_transferred == 100
        assert result.source_dir == "/data/games"
        assert result.dest == "gpu-node-1:/data/games"

    def test_with_error(self):
        """Test BatchSyncResult with error."""
        from app.coordination.sync_bandwidth import BatchSyncResult

        result = BatchSyncResult(
            success=False,
            source_dir="/data/games",
            dest="gpu-node-1:/data/games",
            host="gpu-node-1",
            files_transferred=0,
            errors=["Connection refused"],
        )

        assert result.success is False
        assert "Connection refused" in result.errors


# =============================================================================
# BatchRsync Tests
# =============================================================================


class TestBatchRsync:
    """Tests for BatchRsync class."""

    def test_initialization(self):
        """Test BatchRsync initializes correctly."""
        from app.coordination.sync_bandwidth import BatchRsync

        batch = BatchRsync()

        assert batch.manager is not None

    def test_initialization_with_manager(self):
        """Test BatchRsync with custom manager."""
        from app.coordination.sync_bandwidth import BatchRsync, BandwidthManager

        manager = BandwidthManager()
        batch = BatchRsync(manager=manager)

        assert batch.manager is manager


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_bandwidth_manager_singleton(self):
        """Test get_bandwidth_manager returns singleton."""
        from app.coordination.sync_bandwidth import (
            get_bandwidth_manager, BandwidthManager
        )

        manager1 = get_bandwidth_manager()
        manager2 = get_bandwidth_manager()

        assert manager1 is manager2
        assert isinstance(manager1, BandwidthManager)

    def test_get_coordinated_rsync(self):
        """Test get_coordinated_rsync returns instance."""
        from app.coordination.sync_bandwidth import (
            get_coordinated_rsync, BandwidthCoordinatedRsync
        )

        rsync = get_coordinated_rsync()

        assert isinstance(rsync, BandwidthCoordinatedRsync)

    def test_get_batch_rsync(self):
        """Test get_batch_rsync returns instance."""
        from app.coordination.sync_bandwidth import get_batch_rsync, BatchRsync

        batch = get_batch_rsync()

        assert isinstance(batch, BatchRsync)

    def test_reset_bandwidth_manager(self):
        """Test reset_bandwidth_manager clears singleton."""
        from app.coordination.sync_bandwidth import (
            get_bandwidth_manager, reset_bandwidth_manager
        )

        manager1 = get_bandwidth_manager()
        reset_bandwidth_manager()
        manager2 = get_bandwidth_manager()

        assert manager1 is not manager2

    def test_get_bandwidth_stats(self):
        """Test get_bandwidth_stats returns stats dict."""
        from app.coordination.sync_bandwidth import get_bandwidth_stats

        stats = get_bandwidth_stats()

        assert isinstance(stats, dict)


class TestLoadHostBandwidthHints:
    """Tests for load_host_bandwidth_hints function."""

    def test_returns_dict(self):
        """Test load_host_bandwidth_hints returns dict."""
        from app.coordination.sync_bandwidth import load_host_bandwidth_hints

        result = load_host_bandwidth_hints()

        assert isinstance(result, dict)


class TestProviderBandwidthHints:
    """Tests for PROVIDER_BANDWIDTH_HINTS constant."""

    def test_has_common_providers(self):
        """Test PROVIDER_BANDWIDTH_HINTS includes common providers."""
        from app.coordination.sync_bandwidth import PROVIDER_BANDWIDTH_HINTS

        # Check that the dict exists and has some provider entries
        assert isinstance(PROVIDER_BANDWIDTH_HINTS, dict)
        assert len(PROVIDER_BANDWIDTH_HINTS) > 0
        # All values should be positive bandwidth limits
        for provider, limit in PROVIDER_BANDWIDTH_HINTS.items():
            assert limit > 0, f"Provider {provider} has invalid limit {limit}"


# =============================================================================
# BandwidthManager Cleanup Tests
# =============================================================================


class TestBandwidthManagerCleanup:
    """Tests for BandwidthManager cleanup methods."""

    def test_cleanup_expired_allocations(self):
        """Test expired allocations are cleaned up."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager, BandwidthAllocation, TransferPriority
        )

        manager = BandwidthManager()

        # Add an expired allocation directly
        expired_alloc = BandwidthAllocation(
            host="gpu-node-1",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            transfer_id="expired-1",
            expires_at=time.time() - 100,  # Already expired
        )
        manager._allocations["expired-1"] = expired_alloc
        manager._host_usage["gpu-node-1"] = 10000
        manager._host_transfers["gpu-node-1"] = 1

        # Trigger cleanup
        manager._cleanup_expired()

        # Allocation should be removed
        assert "expired-1" not in manager._allocations


# =============================================================================
# BandwidthManager Priority Tests
# =============================================================================


class TestBandwidthManagerPriority:
    """Tests for priority-based bandwidth allocation."""

    @pytest.mark.asyncio
    async def test_high_priority_gets_more_bandwidth(self):
        """Test high priority allocations get more bandwidth."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager, BandwidthConfig, TransferPriority
        )

        config = BandwidthConfig(
            per_host_limit_kbps=100000,
            max_bwlimit_kbps=100000,
        )
        manager = BandwidthManager(config=config)

        normal_alloc = await manager.request_allocation(
            host="node-1",
            priority=TransferPriority.NORMAL,
        )
        
        # Release before getting high priority
        await manager.release_allocation(normal_alloc)

        high_alloc = await manager.request_allocation(
            host="node-2",
            priority=TransferPriority.HIGH,
        )

        assert normal_alloc is not None
        assert high_alloc is not None
        # High priority should get higher bandwidth (with multiplier)
        assert high_alloc.bwlimit_kbps >= normal_alloc.bwlimit_kbps
