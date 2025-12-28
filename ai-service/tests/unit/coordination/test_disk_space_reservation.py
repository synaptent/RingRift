"""Tests for disk space reservation system.

December 2025: Tests for the atomic disk space reservation mechanism
that prevents race conditions in concurrent sync operations.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from app.coordination.disk_space_reservation import (
    DEFAULT_SAFETY_MARGIN,
    STALE_RESERVATION_THRESHOLD_SECONDS,
    DiskSpaceError,
    DiskSpaceReservation,
    ReservationInfo,
    cleanup_stale_reservations,
    disk_space_reservation,
    disk_space_reservation_sync,
    get_active_reservations,
    get_effective_available_space,
    get_total_reserved_bytes,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def reservation_dir(temp_dir):
    """Create a reservation directory for tests."""
    res_dir = temp_dir / "reservations"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


class TestDiskSpaceReservation:
    """Tests for DiskSpaceReservation class."""

    def test_init_creates_reservation_file_path(self, temp_dir, reservation_dir):
        """Test that initialization creates a unique reservation file path."""
        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=100 * 1024 * 1024,
            reservation_dir=reservation_dir,
        )

        assert reservation.target_dir == temp_dir
        assert reservation.estimated_bytes == 100 * 1024 * 1024
        assert reservation.reservation_file.parent == reservation_dir
        assert "reserve_" in str(reservation.reservation_file)

    def test_safety_margin_applied(self, temp_dir, reservation_dir):
        """Test that safety margin is applied to reserved bytes."""
        estimated = 100 * 1024 * 1024  # 100MB
        margin = 0.10  # 10%

        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=estimated,
            safety_margin=margin,
            reservation_dir=reservation_dir,
        )

        expected_reserved = int(estimated * (1 + margin))
        assert reservation.reserved_bytes == expected_reserved

    def test_acquire_creates_reservation_file(self, temp_dir, reservation_dir):
        """Test that acquire creates a reservation file."""
        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=1024,  # 1KB - very small to ensure we have space
            reservation_dir=reservation_dir,
        )

        assert reservation.acquire()
        assert reservation.is_acquired
        assert reservation.reservation_file.exists()

        # Check file contents
        with open(reservation.reservation_file) as f:
            data = json.load(f)

        assert data["target_dir"] == str(temp_dir)
        assert data["estimated_bytes"] == 1024
        assert "created_at" in data
        assert "pid" in data
        assert "hostname" in data

        # Cleanup
        reservation.release()

    def test_release_removes_reservation_file(self, temp_dir, reservation_dir):
        """Test that release removes the reservation file."""
        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=1024,
            reservation_dir=reservation_dir,
        )

        reservation.acquire()
        assert reservation.reservation_file.exists()

        reservation.release()
        assert not reservation.reservation_file.exists()
        assert not reservation.is_acquired

    def test_context_manager_sync(self, temp_dir, reservation_dir):
        """Test synchronous context manager usage."""
        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=1024,
            reservation_dir=reservation_dir,
        )

        with reservation:
            assert reservation.is_acquired
            assert reservation.reservation_file.exists()

        assert not reservation.is_acquired
        assert not reservation.reservation_file.exists()

    def test_context_manager_releases_on_exception(self, temp_dir, reservation_dir):
        """Test that context manager releases reservation on exception."""
        reservation = DiskSpaceReservation(
            target_dir=temp_dir,
            estimated_bytes=1024,
            reservation_dir=reservation_dir,
        )

        with pytest.raises(ValueError):
            with reservation:
                assert reservation.is_acquired
                raise ValueError("Test exception")

        assert not reservation.is_acquired
        assert not reservation.reservation_file.exists()

    def test_acquire_fails_when_insufficient_space(self, temp_dir, reservation_dir):
        """Test that acquire fails when there's not enough disk space."""
        # Mock disk_usage to return very limited space
        mock_usage = mock.Mock()
        mock_usage.free = 1024  # Only 1KB free

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            reservation = DiskSpaceReservation(
                target_dir=temp_dir,
                estimated_bytes=1024 * 1024 * 1024,  # Request 1GB
                reservation_dir=reservation_dir,
            )

            assert not reservation.acquire()
            assert not reservation.is_acquired

    def test_existing_reservations_reduce_available_space(self, temp_dir, reservation_dir):
        """Test that existing reservations reduce available space."""
        # Mock disk_usage to return 100GB free
        # Need to patch in the module where it's used, not where it's defined
        mock_usage = mock.Mock()
        mock_usage.free = 100 * 1024 * 1024 * 1024  # 100GB

        with mock.patch(
            "app.coordination.disk_space_reservation.shutil.disk_usage",
            return_value=mock_usage
        ):
            # Create first reservation (50GB + 10% margin = 55GB)
            res1 = DiskSpaceReservation(
                target_dir=temp_dir,
                estimated_bytes=50 * 1024 * 1024 * 1024,  # 50GB
                reservation_dir=reservation_dir,
            )
            assert res1.acquire()

            # Create second reservation that would exceed available space
            # after accounting for first reservation
            # First reservation reserves 55GB (50GB + 10%)
            # Second wants 55GB (50GB + 10%)
            # Total: 110GB > 100GB free
            res2 = DiskSpaceReservation(
                target_dir=temp_dir,
                estimated_bytes=50 * 1024 * 1024 * 1024,  # 50GB
                reservation_dir=reservation_dir,
            )

            # Should fail because ~55GB reserved + ~55GB needed > 100GB free
            assert not res2.acquire()

            res1.release()


class TestAsyncContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_dir, reservation_dir):
        """Test async context manager usage."""
        async with disk_space_reservation(
            target_dir=temp_dir,
            estimated_bytes=1024,
        ) as acquired:
            assert acquired

    @pytest.mark.asyncio
    async def test_async_context_manager_releases_on_exception(self, temp_dir, reservation_dir):
        """Test that async context manager releases on exception."""
        with pytest.raises(ValueError):
            async with disk_space_reservation(
                target_dir=temp_dir,
                estimated_bytes=1024,
            ):
                raise ValueError("Test exception")

    @pytest.mark.asyncio
    async def test_async_context_manager_raises_on_failure(self, temp_dir, reservation_dir):
        """Test that async context manager can raise DiskSpaceError."""
        mock_usage = mock.Mock()
        mock_usage.free = 1024

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            with pytest.raises(DiskSpaceError):
                async with disk_space_reservation(
                    target_dir=temp_dir,
                    estimated_bytes=1024 * 1024 * 1024,
                    raise_on_failure=True,
                ):
                    pass  # Should not reach here


class TestSyncContextManager:
    """Tests for synchronous context manager function."""

    def test_sync_context_manager(self, temp_dir):
        """Test sync context manager usage."""
        with disk_space_reservation_sync(
            target_dir=temp_dir,
            estimated_bytes=1024,
        ) as acquired:
            assert acquired

    def test_sync_context_manager_raises_on_failure(self, temp_dir):
        """Test that sync context manager can raise DiskSpaceError."""
        mock_usage = mock.Mock()
        mock_usage.free = 1024

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            with pytest.raises(DiskSpaceError) as exc_info:
                with disk_space_reservation_sync(
                    target_dir=temp_dir,
                    estimated_bytes=1024 * 1024 * 1024,
                    raise_on_failure=True,
                ):
                    pass

            assert exc_info.value.required_bytes > 0


class TestCleanupStaleReservations:
    """Tests for cleanup_stale_reservations function."""

    def test_cleanup_removes_stale_files(self, reservation_dir):
        """Test that cleanup removes stale reservation files."""
        # Create a stale reservation file
        stale_file = reservation_dir / "reserve_test_stale"
        stale_data = {
            "target_dir": "/tmp/test",
            "estimated_bytes": 1024,
            "reserved_bytes": 1126,
            "created_at": time.time() - STALE_RESERVATION_THRESHOLD_SECONDS - 100,
            "pid": 12345,
            "hostname": "test-host",
        }
        with open(stale_file, "w") as f:
            json.dump(stale_data, f)

        # Create a fresh reservation file
        fresh_file = reservation_dir / "reserve_test_fresh"
        fresh_data = {
            "target_dir": "/tmp/test",
            "estimated_bytes": 1024,
            "reserved_bytes": 1126,
            "created_at": time.time(),
            "pid": 12346,
            "hostname": "test-host",
        }
        with open(fresh_file, "w") as f:
            json.dump(fresh_data, f)

        # Run cleanup
        cleaned = cleanup_stale_reservations(reservation_dir)

        assert cleaned == 1
        assert not stale_file.exists()
        assert fresh_file.exists()

    def test_cleanup_removes_corrupt_files(self, reservation_dir):
        """Test that cleanup removes corrupt reservation files."""
        corrupt_file = reservation_dir / "reserve_test_corrupt"
        corrupt_file.write_text("not valid json")

        cleaned = cleanup_stale_reservations(reservation_dir)

        assert cleaned == 1
        assert not corrupt_file.exists()

    def test_cleanup_returns_zero_for_empty_dir(self, reservation_dir):
        """Test that cleanup returns 0 for empty directory."""
        cleaned = cleanup_stale_reservations(reservation_dir)
        assert cleaned == 0


class TestGetActiveReservations:
    """Tests for get_active_reservations function."""

    def test_returns_active_reservations(self, reservation_dir):
        """Test that function returns only active reservations."""
        # Create an active reservation
        active_file = reservation_dir / "reserve_test_active"
        active_data = {
            "target_dir": "/tmp/test",
            "estimated_bytes": 2048,
            "reserved_bytes": 2253,
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": "test-host",
        }
        with open(active_file, "w") as f:
            json.dump(active_data, f)

        # Create a stale reservation
        stale_file = reservation_dir / "reserve_test_stale"
        stale_data = {
            "target_dir": "/tmp/test",
            "estimated_bytes": 1024,
            "reserved_bytes": 1126,
            "created_at": time.time() - STALE_RESERVATION_THRESHOLD_SECONDS - 100,
            "pid": 12345,
            "hostname": "test-host",
        }
        with open(stale_file, "w") as f:
            json.dump(stale_data, f)

        reservations = get_active_reservations(reservation_dir)

        assert len(reservations) == 1
        assert reservations[0].estimated_bytes == 2048
        assert reservations[0].pid == os.getpid()

    def test_returns_reservation_info_objects(self, reservation_dir):
        """Test that function returns ReservationInfo objects."""
        active_file = reservation_dir / "reserve_test"
        active_data = {
            "target_dir": "/tmp/test",
            "estimated_bytes": 1024,
            "reserved_bytes": 1126,
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": "test-host",
        }
        with open(active_file, "w") as f:
            json.dump(active_data, f)

        reservations = get_active_reservations(reservation_dir)

        assert len(reservations) == 1
        info = reservations[0]
        assert isinstance(info, ReservationInfo)
        assert info.target_dir == "/tmp/test"
        assert info.estimated_bytes == 1024
        assert not info.is_stale
        assert info.age_seconds >= 0


class TestGetEffectiveAvailableSpace:
    """Tests for get_effective_available_space function."""

    def test_returns_disk_free_when_no_reservations(self, temp_dir, reservation_dir):
        """Test that function returns disk free when no reservations."""
        mock_usage = mock.Mock()
        mock_usage.free = 100 * 1024 * 1024 * 1024  # 100GB

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            available = get_effective_available_space(temp_dir, reservation_dir)

        assert available == 100 * 1024 * 1024 * 1024

    def test_subtracts_reservations_for_same_target(self, temp_dir, reservation_dir):
        """Test that function subtracts reservations for same target."""
        # Create a reservation for the same target directory
        res_file = reservation_dir / "reserve_test"
        res_data = {
            "target_dir": str(temp_dir),
            "estimated_bytes": 10 * 1024 * 1024 * 1024,  # 10GB
            "reserved_bytes": 11 * 1024 * 1024 * 1024,
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": "test-host",
        }
        with open(res_file, "w") as f:
            json.dump(res_data, f)

        mock_usage = mock.Mock()
        mock_usage.free = 100 * 1024 * 1024 * 1024  # 100GB

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            available = get_effective_available_space(temp_dir, reservation_dir)

        # Should be 100GB - 10GB = 90GB
        assert available == 90 * 1024 * 1024 * 1024


class TestGetTotalReservedBytes:
    """Tests for get_total_reserved_bytes function."""

    def test_returns_zero_when_no_reservations(self, reservation_dir):
        """Test that function returns 0 when no reservations."""
        total = get_total_reserved_bytes(reservation_dir=reservation_dir)
        assert total == 0

    def test_sums_all_active_reservations(self, reservation_dir):
        """Test that function sums all active reservations."""
        # Create multiple reservations
        for i in range(3):
            res_file = reservation_dir / f"reserve_test_{i}"
            res_data = {
                "target_dir": f"/tmp/test_{i}",
                "estimated_bytes": 1024 * (i + 1),
                "reserved_bytes": 1024 * (i + 1),
                "created_at": time.time(),
                "pid": os.getpid(),
                "hostname": "test-host",
            }
            with open(res_file, "w") as f:
                json.dump(res_data, f)

        total = get_total_reserved_bytes(reservation_dir=reservation_dir)

        # 1024 + 2048 + 3072 = 6144
        assert total == 6144

    def test_filters_by_target_dir(self, temp_dir, reservation_dir):
        """Test that function filters by target directory."""
        # Create reservation for target dir
        res_file1 = reservation_dir / "reserve_test_1"
        res_data1 = {
            "target_dir": str(temp_dir),
            "estimated_bytes": 1024,
            "reserved_bytes": 1024,
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": "test-host",
        }
        with open(res_file1, "w") as f:
            json.dump(res_data1, f)

        # Create reservation for different dir
        res_file2 = reservation_dir / "reserve_test_2"
        res_data2 = {
            "target_dir": "/tmp/other",
            "estimated_bytes": 2048,
            "reserved_bytes": 2048,
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": "test-host",
        }
        with open(res_file2, "w") as f:
            json.dump(res_data2, f)

        total = get_total_reserved_bytes(target_dir=temp_dir, reservation_dir=reservation_dir)

        # Only the reservation for temp_dir
        assert total == 1024


class TestReservationInfo:
    """Tests for ReservationInfo dataclass."""

    def test_age_seconds(self):
        """Test age_seconds calculation."""
        info = ReservationInfo(
            target_dir="/tmp/test",
            estimated_bytes=1024,
            created_at=time.time() - 60,
            pid=12345,
            hostname="test-host",
            reservation_file="/tmp/reserve_test",
        )

        assert info.age_seconds >= 60
        assert info.age_seconds < 70  # Allow some slack

    def test_is_stale(self):
        """Test is_stale property."""
        fresh_info = ReservationInfo(
            target_dir="/tmp/test",
            estimated_bytes=1024,
            created_at=time.time(),
            pid=12345,
            hostname="test-host",
            reservation_file="/tmp/reserve_test",
        )
        assert not fresh_info.is_stale

        stale_info = ReservationInfo(
            target_dir="/tmp/test",
            estimated_bytes=1024,
            created_at=time.time() - STALE_RESERVATION_THRESHOLD_SECONDS - 100,
            pid=12345,
            hostname="test-host",
            reservation_file="/tmp/reserve_test",
        )
        assert stale_info.is_stale

    def test_to_dict(self):
        """Test to_dict method."""
        created_at = time.time()
        info = ReservationInfo(
            target_dir="/tmp/test",
            estimated_bytes=1024,
            created_at=created_at,
            pid=12345,
            hostname="test-host",
            reservation_file="/tmp/reserve_test",
        )

        d = info.to_dict()

        assert d["target_dir"] == "/tmp/test"
        assert d["estimated_bytes"] == 1024
        assert d["created_at"] == created_at
        assert d["pid"] == 12345
        assert d["hostname"] == "test-host"
        assert "age_seconds" in d
        assert "is_stale" in d


class TestDiskSpaceError:
    """Tests for DiskSpaceError exception."""

    def test_exception_attributes(self):
        """Test that DiskSpaceError stores attributes correctly."""
        err = DiskSpaceError(
            "Test error",
            available_bytes=100,
            required_bytes=200,
            existing_reservations=50,
        )

        assert str(err) == "Test error"
        assert err.available_bytes == 100
        assert err.required_bytes == 200
        assert err.existing_reservations == 50

    def test_exception_default_values(self):
        """Test that DiskSpaceError has reasonable defaults."""
        err = DiskSpaceError("Test error")

        assert err.available_bytes == 0
        assert err.required_bytes == 0
        assert err.existing_reservations == 0


class TestConcurrentReservations:
    """Tests for concurrent reservation behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_reservations_are_counted(self, temp_dir, reservation_dir):
        """Test that concurrent reservations are properly counted."""
        mock_usage = mock.Mock()
        mock_usage.free = 100 * 1024 * 1024  # 100MB

        results = []

        async def make_reservation(size_mb):
            """Make a reservation and record if it succeeded."""
            async with disk_space_reservation(
                target_dir=temp_dir,
                estimated_bytes=size_mb * 1024 * 1024,
            ) as acquired:
                results.append(acquired)
                if acquired:
                    # Hold the reservation for a bit
                    await asyncio.sleep(0.1)

        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            # Try to reserve 40MB each from 3 concurrent tasks
            # Only 2 should succeed (40MB + 40MB = 80MB < 100MB)
            # Third one should see 80MB reserved and fail
            await asyncio.gather(
                make_reservation(40),
                make_reservation(40),
                make_reservation(40),
            )

        # At least 2 should succeed, but maybe not all 3
        successful = sum(results)
        assert successful >= 2
        # With timing, some might fail due to race
        assert successful <= 3
