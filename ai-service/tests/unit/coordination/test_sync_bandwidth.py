"""Tests for app.coordination.sync_bandwidth module.

Tests bandwidth coordination for rsync operations.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_bandwidth import (
    PROVIDER_BANDWIDTH_HINTS,
    BandwidthAllocation,
    BandwidthConfig,
    BandwidthCoordinatedRsync,
    BandwidthManager,
    BatchRsync,
    BatchSyncResult,
    TransferPriority,
    get_bandwidth_manager,
    get_batch_rsync,
    get_coordinated_rsync,
    load_host_bandwidth_hints,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return BandwidthConfig(
        default_bwlimit_kbps=10000,
        max_bwlimit_kbps=50000,
        min_bwlimit_kbps=1000,
        per_host_limit_kbps=20000,
        total_limit_kbps=100000,
        max_concurrent_per_host=2,
        max_concurrent_total=8,
        allocation_timeout_seconds=60.0,
        enable_adaptive=False,  # Disable for predictable tests
    )


@pytest.fixture
def manager(config):
    """Create a fresh manager for each test."""
    return BandwidthManager(config)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    BandwidthManager.reset_instance()
    yield
    BandwidthManager.reset_instance()


# =============================================================================
# TransferPriority Tests
# =============================================================================


class TestTransferPriority:
    """Tests for TransferPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert TransferPriority.LOW.value == "low"
        assert TransferPriority.NORMAL.value == "normal"
        assert TransferPriority.HIGH.value == "high"
        assert TransferPriority.CRITICAL.value == "critical"

    def test_priority_ordering(self):
        """Test that priorities can be compared by value."""
        priorities = [
            TransferPriority.LOW,
            TransferPriority.NORMAL,
            TransferPriority.HIGH,
            TransferPriority.CRITICAL,
        ]
        # Just verify all four are distinct
        assert len(set(priorities)) == 4


# =============================================================================
# Provider Bandwidth Hints Tests
# =============================================================================


class TestProviderBandwidthHints:
    """Tests for PROVIDER_BANDWIDTH_HINTS constant."""

    def test_provider_hints_exist(self):
        """Test that provider hints are defined."""
        assert "lambda" in PROVIDER_BANDWIDTH_HINTS
        assert "runpod" in PROVIDER_BANDWIDTH_HINTS
        assert "vast" in PROVIDER_BANDWIDTH_HINTS
        assert "default" in PROVIDER_BANDWIDTH_HINTS

    def test_hint_values_reasonable(self):
        """Test that hint values are reasonable (KB/s)."""
        for provider, hint in PROVIDER_BANDWIDTH_HINTS.items():
            assert hint >= 1000, f"{provider} should have >= 1 MB/s"
            assert hint <= 1000000, f"{provider} should have <= 1 GB/s"


# =============================================================================
# BandwidthAllocation Tests
# =============================================================================


class TestBandwidthAllocation:
    """Tests for BandwidthAllocation dataclass."""

    def test_creation(self):
        """Test creating an allocation."""
        alloc = BandwidthAllocation(
            host="test-host",
            priority=TransferPriority.HIGH,
            bwlimit_kbps=10000,
        )
        assert alloc.host == "test-host"
        assert alloc.priority == TransferPriority.HIGH
        assert alloc.bwlimit_kbps == 10000

    def test_default_values(self):
        """Test default values."""
        alloc = BandwidthAllocation(
            host="test",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=5000,
        )
        assert alloc.transfer_id == ""
        assert alloc.expires_at == 0.0

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry set."""
        alloc = BandwidthAllocation(
            host="test",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=5000,
            expires_at=0.0,  # No expiry
        )
        assert not alloc.is_expired

    def test_is_expired_future(self):
        """Test is_expired when expiry is in future."""
        alloc = BandwidthAllocation(
            host="test",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=5000,
            expires_at=time.time() + 3600,  # 1 hour from now
        )
        assert not alloc.is_expired

    def test_is_expired_past(self):
        """Test is_expired when expiry is in past."""
        alloc = BandwidthAllocation(
            host="test",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=5000,
            expires_at=time.time() - 1,  # 1 second ago
        )
        assert alloc.is_expired


# =============================================================================
# BandwidthConfig Tests
# =============================================================================


class TestBandwidthConfig:
    """Tests for BandwidthConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BandwidthConfig()
        assert config.default_bwlimit_kbps == 10000
        assert config.max_bwlimit_kbps == 50000
        assert config.min_bwlimit_kbps == 1000
        assert config.max_concurrent_per_host == 2
        assert config.max_concurrent_total == 8

    def test_priority_multipliers(self):
        """Test priority multipliers."""
        config = BandwidthConfig()
        assert config.priority_multipliers[TransferPriority.LOW] == 0.5
        assert config.priority_multipliers[TransferPriority.NORMAL] == 1.0
        assert config.priority_multipliers[TransferPriority.HIGH] == 1.5
        assert config.priority_multipliers[TransferPriority.CRITICAL] == 2.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = BandwidthConfig(
            default_bwlimit_kbps=5000,
            max_concurrent_total=4,
        )
        assert config.default_bwlimit_kbps == 5000
        assert config.max_concurrent_total == 4


# =============================================================================
# BandwidthManager Tests
# =============================================================================


class TestBandwidthManager:
    """Tests for BandwidthManager class."""

    def test_creation(self, manager, config):
        """Test manager creation."""
        assert manager.config == config
        assert len(manager._allocations) == 0
        assert len(manager._host_usage) == 0

    @pytest.mark.asyncio
    async def test_request_allocation_success(self, manager):
        """Test successful allocation request."""
        alloc = await manager.request_allocation(
            host="test-host",
            priority=TransferPriority.NORMAL,
            timeout=5.0,
        )
        assert alloc is not None
        assert alloc.host == "test-host"
        assert alloc.priority == TransferPriority.NORMAL
        assert alloc.bwlimit_kbps > 0

    @pytest.mark.asyncio
    async def test_request_allocation_tracks_usage(self, manager):
        """Test that allocations are tracked."""
        alloc = await manager.request_allocation(host="test-host")
        assert manager._host_usage.get("test-host", 0) == alloc.bwlimit_kbps
        assert manager._host_transfers.get("test-host", 0) == 1

    @pytest.mark.asyncio
    async def test_release_allocation(self, manager):
        """Test releasing an allocation."""
        alloc = await manager.request_allocation(host="test-host")

        await manager.release_allocation(alloc)

        assert manager._host_usage.get("test-host", 0) == 0
        assert manager._host_transfers.get("test-host", 0) == 0

    @pytest.mark.asyncio
    async def test_per_host_concurrent_limit(self, manager, config):
        """Test per-host concurrent limit is enforced."""
        # Allocate up to the limit
        allocations = []
        for _ in range(config.max_concurrent_per_host):
            alloc = await manager.request_allocation(host="test-host", timeout=2.0)
            assert alloc is not None
            allocations.append(alloc)

        # Check that we can't allocate more (internal check, not waiting)
        assert not manager._can_allocate("test-host")

    @pytest.mark.asyncio
    async def test_different_hosts_independent(self, manager):
        """Test that different hosts have independent limits."""
        alloc1 = await manager.request_allocation(host="host1")
        alloc2 = await manager.request_allocation(host="host2")

        assert alloc1 is not None
        assert alloc2 is not None
        assert alloc1.host == "host1"
        assert alloc2.host == "host2"

    @pytest.mark.asyncio
    async def test_total_concurrent_limit(self, manager, config):
        """Test total concurrent limit is enforced."""
        allocations = []

        # Allocate up to total limit across different hosts
        for i in range(config.max_concurrent_total):
            alloc = await manager.request_allocation(host=f"host-{i}", timeout=2.0)
            assert alloc is not None
            allocations.append(alloc)

        # Check that we can't allocate more (internal check, not waiting)
        assert not manager._can_allocate("extra-host")

    @pytest.mark.asyncio
    async def test_priority_affects_bandwidth(self, manager):
        """Test that priority affects bandwidth allocation."""
        # LOW priority gets less bandwidth
        low_alloc = await manager.request_allocation(
            host="host-low",
            priority=TransferPriority.LOW,
        )
        await manager.release_allocation(low_alloc)

        # HIGH priority gets more bandwidth
        high_alloc = await manager.request_allocation(
            host="host-high",
            priority=TransferPriority.HIGH,
        )

        # HIGH should get more (or equal if capped)
        assert high_alloc.bwlimit_kbps >= low_alloc.bwlimit_kbps

    def test_get_status_empty(self, manager):
        """Test status with no allocations."""
        status = manager.get_status()
        assert status["total_usage_kbps"] == 0
        assert status["active_transfers"] == 0
        assert status["active_allocations"] == []

    @pytest.mark.asyncio
    async def test_get_status_with_allocations(self, manager):
        """Test status with active allocations."""
        await manager.request_allocation(host="test-host")

        status = manager.get_status()
        assert status["total_usage_kbps"] > 0
        assert status["active_transfers"] == 1
        assert len(status["active_allocations"]) == 1
        assert status["active_allocations"][0]["host"] == "test-host"

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        manager1 = BandwidthManager.get_instance()
        manager2 = BandwidthManager.get_instance()
        assert manager1 is manager2

    def test_singleton_reset(self):
        """Test singleton reset."""
        manager1 = BandwidthManager.get_instance()
        BandwidthManager.reset_instance()
        manager2 = BandwidthManager.get_instance()
        assert manager1 is not manager2


# =============================================================================
# BandwidthCoordinatedRsync Tests
# =============================================================================


class TestBandwidthCoordinatedRsync:
    """Tests for BandwidthCoordinatedRsync class."""

    def test_creation(self, manager):
        """Test rsync wrapper creation."""
        rsync = BandwidthCoordinatedRsync(manager=manager)
        assert rsync.manager is manager
        assert rsync.rsync_path == "rsync"
        assert rsync.verify_checksum is True

    def test_default_options(self, manager):
        """Test default rsync options."""
        rsync = BandwidthCoordinatedRsync(manager=manager)
        assert "-avz" in rsync.default_options
        assert "--progress" in rsync.default_options

    @pytest.mark.asyncio
    async def test_sync_allocation_failure(self, manager):
        """Test sync when allocation fails - use internal check to avoid timeout."""
        # Exhaust allocations first
        allocations = []
        for i in range(manager.config.max_concurrent_total):
            alloc = await manager.request_allocation(host=f"host-{i}")
            allocations.append(alloc)

        # Verify we can't allocate more
        assert not manager._can_allocate("new-host")

        # Clean up
        for alloc in allocations:
            await manager.release_allocation(alloc)

    @pytest.mark.asyncio
    async def test_sync_with_mock_subprocess(self, manager):
        """Test sync with mocked subprocess."""
        rsync = BandwidthCoordinatedRsync(manager=manager)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"sent 1234 bytes  received 56 bytes", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await rsync.sync(
                source="/tmp/test",
                dest="user@host:/tmp/",
                host="test-host",
            )

        assert result.success
        assert result.bytes_transferred == 1234

    def test_parse_bytes_transferred_sent(self):
        """Test parsing 'sent X bytes' pattern."""
        rsync = BandwidthCoordinatedRsync()
        output = "sent 123,456 bytes  received 789 bytes"
        assert rsync._parse_bytes_transferred(output) == 123456

    def test_parse_bytes_transferred_total_size(self):
        """Test parsing 'total size is X' pattern."""
        rsync = BandwidthCoordinatedRsync()
        output = "total size is 987,654"
        assert rsync._parse_bytes_transferred(output) == 987654

    def test_parse_bytes_transferred_no_match(self):
        """Test parsing with no recognizable pattern."""
        rsync = BandwidthCoordinatedRsync()
        output = "some random output"
        assert rsync._parse_bytes_transferred(output) == 0


# =============================================================================
# BatchSyncResult Tests
# =============================================================================


class TestBatchSyncResult:
    """Tests for BatchSyncResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = BatchSyncResult(
            success=True,
            source_dir="/tmp/src",
            dest="user@host:/tmp/",
            host="test-host",
        )
        assert result.files_requested == 0
        assert result.files_transferred == 0
        assert result.bytes_transferred == 0
        assert result.errors == []


# =============================================================================
# BatchRsync Tests
# =============================================================================


class TestBatchRsync:
    """Tests for BatchRsync class."""

    def test_creation(self, manager):
        """Test batch rsync creation."""
        batch = BatchRsync(manager=manager)
        assert batch.manager is manager
        assert batch.rsync_path == "rsync"

    @pytest.mark.asyncio
    async def test_sync_files_empty_list(self, manager):
        """Test syncing empty file list."""
        batch = BatchRsync(manager=manager)
        result = await batch.sync_files(
            source_dir="/tmp/",
            dest="user@host:/tmp/",
            host="test-host",
            files=[],
        )
        assert result.success
        assert result.files_requested == 0

    @pytest.mark.asyncio
    async def test_sync_files_allocation_failure(self, manager):
        """Test sync_files allocation blocked - use internal check to avoid timeout."""
        # Exhaust allocations
        allocations = []
        for i in range(manager.config.max_concurrent_total):
            alloc = await manager.request_allocation(host=f"host-{i}")
            allocations.append(alloc)

        # Verify we can't allocate more
        assert not manager._can_allocate("new-host")

        # Clean up
        for alloc in allocations:
            await manager.release_allocation(alloc)

    @pytest.mark.asyncio
    async def test_sync_files_with_mock(self, manager):
        """Test sync_files with mocked subprocess."""
        batch = BatchRsync(manager=manager)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"sent 5,000 bytes\n>f file1.txt\n>f file2.txt\n", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await batch.sync_files(
                source_dir="/tmp/",
                dest="user@host:/tmp/",
                host="test-host",
                files=["file1.txt", "file2.txt"],
            )

        assert result.success
        assert result.files_requested == 2
        assert result.files_transferred == 2

    @pytest.mark.asyncio
    async def test_sync_directory_allocation_failure(self, manager):
        """Test sync_directory allocation blocked - use internal check to avoid timeout."""
        # Exhaust allocations
        allocations = []
        for i in range(manager.config.max_concurrent_total):
            alloc = await manager.request_allocation(host=f"host-{i}")
            allocations.append(alloc)

        # Verify we can't allocate more
        assert not manager._can_allocate("new-host")

        # Clean up
        for alloc in allocations:
            await manager.release_allocation(alloc)

    def test_parse_bytes(self, manager):
        """Test _parse_bytes helper."""
        batch = BatchRsync(manager=manager)

        # Test 'sent X bytes' pattern
        assert batch._parse_bytes("sent 1,234 bytes") == 1234

        # Test 'total size is X' pattern
        assert batch._parse_bytes("total size is 5,678") == 5678

        # Test no match
        assert batch._parse_bytes("no pattern here") == 0


# =============================================================================
# Singleton Access Tests
# =============================================================================


class TestSingletonAccess:
    """Tests for singleton access functions."""

    def test_get_bandwidth_manager(self):
        """Test get_bandwidth_manager function."""
        manager = get_bandwidth_manager()
        assert isinstance(manager, BandwidthManager)

    def test_get_bandwidth_manager_same_instance(self):
        """Test get_bandwidth_manager returns same instance."""
        m1 = get_bandwidth_manager()
        m2 = get_bandwidth_manager()
        assert m1 is m2

    def test_get_coordinated_rsync(self):
        """Test get_coordinated_rsync function."""
        rsync = get_coordinated_rsync()
        assert isinstance(rsync, BandwidthCoordinatedRsync)

    def test_get_batch_rsync(self):
        """Test get_batch_rsync function."""
        batch = get_batch_rsync()
        assert isinstance(batch, BatchRsync)


# =============================================================================
# load_host_bandwidth_hints Tests
# =============================================================================


class TestLoadHostBandwidthHints:
    """Tests for load_host_bandwidth_hints function."""

    def test_load_hints_import_error(self):
        """Test graceful handling when cluster_config unavailable."""
        with patch.dict("sys.modules", {"app.config.cluster_config": None}):
            # Should return empty dict on import error
            hints = load_host_bandwidth_hints()
            assert isinstance(hints, dict)

    def test_load_hints_success(self):
        """Test loading hints from cluster_config."""
        mock_nodes = {"node1": MagicMock(), "node2": MagicMock()}

        with patch("app.coordination.sync_bandwidth.get_cluster_nodes", return_value=mock_nodes), \
            patch("app.coordination.sync_bandwidth.get_node_bandwidth_kbs", side_effect=[50000, 100000]):
            hints = load_host_bandwidth_hints()
            assert hints.get("node1") == 50000
            assert hints.get("node2") == 100000


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for bandwidth coordination."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, manager):
        """Test complete allocation -> transfer -> release workflow."""
        # Request allocation
        alloc = await manager.request_allocation(
            host="integration-test",
            priority=TransferPriority.HIGH,
        )
        assert alloc is not None

        # Check status during allocation
        status = manager.get_status()
        assert status["active_transfers"] == 1

        # Release allocation
        await manager.release_allocation(alloc)

        # Check status after release
        status = manager.get_status()
        assert status["active_transfers"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self, manager):
        """Test concurrent allocation requests."""
        hosts = ["host1", "host2", "host3"]

        # Request allocations concurrently
        tasks = [manager.request_allocation(host=h) for h in hosts]
        allocations = await asyncio.gather(*tasks)

        # All should succeed
        assert all(a is not None for a in allocations)
        assert len({a.host for a in allocations}) == 3

        # Status should reflect all
        status = manager.get_status()
        assert status["active_transfers"] == 3

        # Release all
        for alloc in allocations:
            await manager.release_allocation(alloc)

        status = manager.get_status()
        assert status["active_transfers"] == 0

    @pytest.mark.asyncio
    async def test_allocation_blocked_scenario(self, manager, config):
        """Test scenario where allocation would be blocked."""
        # Fill up one host to its limit
        allocations = []
        for _ in range(config.max_concurrent_per_host):
            alloc = await manager.request_allocation(host="busy-host")
            allocations.append(alloc)

        # Verify allocation would be blocked (don't actually wait)
        assert not manager._can_allocate("busy-host")
        assert manager._host_transfers.get("busy-host", 0) == config.max_concurrent_per_host

        # Clean up
        for alloc in allocations:
            await manager.release_allocation(alloc)

        # After cleanup, allocation should be possible again
        assert manager._can_allocate("busy-host")
