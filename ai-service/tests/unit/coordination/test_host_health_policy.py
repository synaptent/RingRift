"""Tests for host_health_policy module.

Tests fast SSH health checks, caching, and healthy host filtering.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, Mock

import pytest

from app.coordination.host_health_policy import (
    # Dataclasses
    HostHealthStatus,
    HealthStatus,  # Backward-compat alias
    # Functions
    check_host_health,
    is_host_healthy,
    get_healthy_hosts,
    get_health_summary,
    clear_health_cache,
    get_cache_status,
    # Internal
    _quick_ssh_check,
    _get_ssh_target,
    _health_cache,
    _cache_lock,
    # Constants
    DEFAULT_SSH_TIMEOUT,
    HEALTH_CACHE_TTL,
    UNHEALTHY_CACHE_TTL,
)


# ============================================================================
# HostHealthStatus Dataclass Tests
# ============================================================================


class TestHostHealthStatus:
    """Tests for HostHealthStatus dataclass."""

    def test_healthy_status(self) -> None:
        """Test creating a healthy status."""
        status = HostHealthStatus(
            host="gpu-server-1",
            healthy=True,
            checked_at=time.time(),
            latency_ms=25.5,
        )
        assert status.host == "gpu-server-1"
        assert status.healthy is True
        assert status.latency_ms == 25.5
        assert status.error is None

    def test_unhealthy_status(self) -> None:
        """Test creating an unhealthy status."""
        status = HostHealthStatus(
            host="dead-server",
            healthy=False,
            checked_at=time.time(),
            error="Connection refused",
        )
        assert status.healthy is False
        assert status.error == "Connection refused"

    def test_age_seconds_property(self) -> None:
        """Test age_seconds property calculates correctly."""
        past_time = time.time() - 30
        status = HostHealthStatus(
            host="test",
            healthy=True,
            checked_at=past_time,
        )
        assert 29 < status.age_seconds < 31

    def test_is_stale_healthy(self) -> None:
        """Test is_stale for healthy hosts uses HEALTH_CACHE_TTL."""
        # Not stale (fresh)
        status = HostHealthStatus(
            host="test",
            healthy=True,
            checked_at=time.time(),
        )
        assert status.is_stale is False

        # Stale (old)
        old_status = HostHealthStatus(
            host="test",
            healthy=True,
            checked_at=time.time() - (HEALTH_CACHE_TTL + 10),
        )
        assert old_status.is_stale is True

    def test_is_stale_unhealthy(self) -> None:
        """Test is_stale for unhealthy hosts uses UNHEALTHY_CACHE_TTL."""
        # Not stale (fresh)
        status = HostHealthStatus(
            host="test",
            healthy=False,
            checked_at=time.time(),
            error="timeout",
        )
        assert status.is_stale is False

        # Stale (old) - unhealthy cache TTL is shorter
        old_status = HostHealthStatus(
            host="test",
            healthy=False,
            checked_at=time.time() - (UNHEALTHY_CACHE_TTL + 10),
            error="timeout",
        )
        assert old_status.is_stale is True

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        status = HostHealthStatus(
            host="server-1",
            healthy=True,
            checked_at=time.time(),
            latency_ms=15.0,
            load_1m=0.5,
            cpu_count=8,
        )
        d = status.to_dict()

        assert d["host"] == "server-1"
        assert d["healthy"] is True
        assert d["latency_ms"] == 15.0
        assert d["load_1m"] == 0.5
        assert d["cpu_count"] == 8
        assert "checked_at" in d
        assert "age_seconds" in d

    def test_backward_compat_alias(self) -> None:
        """Test HealthStatus is an alias for HostHealthStatus."""
        assert HealthStatus is HostHealthStatus


# ============================================================================
# _get_ssh_target Tests
# ============================================================================


class TestGetSshTarget:
    """Tests for _get_ssh_target function."""

    def test_fallback_to_host_directly(self) -> None:
        """Test fallback when host config not found."""
        with patch(
            "app.coordination.host_health_policy.load_remote_hosts",
            side_effect=ImportError,
        ):
            target, key, port = _get_ssh_target("unknown-host")

            assert target == "unknown-host"
            assert key is None
            assert port == 22

    def test_uses_config_when_available(self) -> None:
        """Test uses host config from distributed_hosts."""
        mock_host = MagicMock()
        mock_host.ssh_target = "user@192.168.1.100"
        mock_host.ssh_key_path = "/path/to/key"
        mock_host.ssh_key = True
        mock_host.ssh_port = 2222

        with patch(
            "app.coordination.host_health_policy.load_remote_hosts",
            return_value={"my-host": mock_host},
        ):
            target, key, port = _get_ssh_target("my-host")

            assert target == "user@192.168.1.100"
            assert key == "/path/to/key"
            assert port == 2222


# ============================================================================
# _quick_ssh_check Tests
# ============================================================================


class TestQuickSshCheck:
    """Tests for _quick_ssh_check function."""

    def test_successful_check(self) -> None:
        """Test successful SSH check."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK\n0.50 0.30 0.20 1/100 12345\n8\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch(
                "app.coordination.host_health_policy._get_ssh_target",
                return_value=("user@host", None, 22),
            ):
                status = _quick_ssh_check("test-host")

                assert status.healthy is True
                assert status.host == "test-host"
                assert status.latency_ms is not None
                assert status.load_1m == 0.50
                assert status.cpu_count == 8

    def test_failed_check(self) -> None:
        """Test failed SSH check."""
        mock_result = MagicMock()
        mock_result.returncode = 255
        mock_result.stdout = ""
        mock_result.stderr = "Connection refused"

        with patch("subprocess.run", return_value=mock_result):
            with patch(
                "app.coordination.host_health_policy._get_ssh_target",
                return_value=("user@host", None, 22),
            ):
                status = _quick_ssh_check("test-host")

                assert status.healthy is False
                assert "Connection refused" in status.error

    def test_timeout_check(self) -> None:
        """Test SSH timeout."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ssh"], timeout=5),
        ):
            with patch(
                "app.coordination.host_health_policy._get_ssh_target",
                return_value=("user@host", None, 22),
            ):
                status = _quick_ssh_check("test-host", timeout=5)

                assert status.healthy is False
                assert "timeout" in status.error.lower()

    def test_exception_handling(self) -> None:
        """Test exception handling."""
        with patch(
            "subprocess.run",
            side_effect=OSError("Network unreachable"),
        ):
            with patch(
                "app.coordination.host_health_policy._get_ssh_target",
                return_value=("user@host", None, 22),
            ):
                status = _quick_ssh_check("test-host")

                assert status.healthy is False
                assert "Network unreachable" in status.error


# ============================================================================
# check_host_health Tests
# ============================================================================


class TestCheckHostHealth:
    """Tests for check_host_health function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_localhost_always_healthy(self) -> None:
        """Test localhost is always marked healthy without SSH."""
        status = check_host_health("localhost")

        assert status.healthy is True
        assert status.latency_ms == 0

    def test_local_hostname_healthy(self) -> None:
        """Test local hostname is always healthy."""
        hostname = socket.gethostname()
        status = check_host_health(hostname)

        assert status.healthy is True

    def test_caches_result(self) -> None:
        """Test that results are cached."""
        mock_status = HostHealthStatus(
            host="cached-host",
            healthy=True,
            checked_at=time.time(),
            latency_ms=10,
        )

        with patch(
            "app.coordination.host_health_policy._quick_ssh_check",
            return_value=mock_status,
        ) as mock_check:
            # First call
            status1 = check_host_health("cached-host")
            # Second call should use cache
            status2 = check_host_health("cached-host")

            # Should only call SSH check once
            assert mock_check.call_count == 1
            assert status1 is status2

    def test_force_refresh_bypasses_cache(self) -> None:
        """Test force_refresh bypasses cache."""
        mock_status = HostHealthStatus(
            host="refresh-host",
            healthy=True,
            checked_at=time.time(),
            latency_ms=10,
        )

        with patch(
            "app.coordination.host_health_policy._quick_ssh_check",
            return_value=mock_status,
        ) as mock_check:
            # First call
            check_host_health("refresh-host")
            # Second call with force_refresh
            check_host_health("refresh-host", force_refresh=True)

            # Should call SSH check twice
            assert mock_check.call_count == 2


# ============================================================================
# is_host_healthy Tests
# ============================================================================


class TestIsHostHealthy:
    """Tests for is_host_healthy function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_returns_true_for_healthy(self) -> None:
        """Test returns True for healthy host."""
        mock_status = HostHealthStatus(
            host="healthy-host",
            healthy=True,
            checked_at=time.time(),
        )

        with patch(
            "app.coordination.host_health_policy.check_host_health",
            return_value=mock_status,
        ):
            assert is_host_healthy("healthy-host") is True

    def test_returns_false_for_unhealthy(self) -> None:
        """Test returns False for unhealthy host."""
        mock_status = HostHealthStatus(
            host="unhealthy-host",
            healthy=False,
            checked_at=time.time(),
            error="Connection refused",
        )

        with patch(
            "app.coordination.host_health_policy.check_host_health",
            return_value=mock_status,
        ):
            assert is_host_healthy("unhealthy-host") is False


# ============================================================================
# get_healthy_hosts Tests
# ============================================================================


class TestGetHealthyHosts:
    """Tests for get_healthy_hosts function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_empty_list(self) -> None:
        """Test empty input returns empty output."""
        result = get_healthy_hosts([])
        assert result == []

    def test_filters_unhealthy_sequential(self) -> None:
        """Test filters out unhealthy hosts (sequential)."""

        def mock_healthy(host: str, force: bool = False) -> bool:
            return host in ("host-1", "host-3")

        with patch(
            "app.coordination.host_health_policy.is_host_healthy",
            side_effect=mock_healthy,
        ):
            result = get_healthy_hosts(
                ["host-1", "host-2", "host-3"],
                parallel=False,
            )

            assert result == ["host-1", "host-3"]

    def test_parallel_check(self) -> None:
        """Test parallel checking works."""

        def mock_healthy(host: str, force: bool = False) -> bool:
            time.sleep(0.01)  # Simulate network latency
            return host != "host-2"

        with patch(
            "app.coordination.host_health_policy.is_host_healthy",
            side_effect=mock_healthy,
        ):
            hosts = ["host-1", "host-2", "host-3", "host-4", "host-5"]
            result = get_healthy_hosts(hosts, parallel=True)

            assert "host-2" not in result
            assert len(result) == 4


# ============================================================================
# get_health_summary Tests
# ============================================================================


class TestGetHealthSummary:
    """Tests for get_health_summary function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_summary_statistics(self) -> None:
        """Test summary includes correct statistics."""
        statuses = [
            HostHealthStatus(
                host="host-1",
                healthy=True,
                checked_at=time.time(),
                latency_ms=10,
            ),
            HostHealthStatus(
                host="host-2",
                healthy=True,
                checked_at=time.time(),
                latency_ms=20,
            ),
            HostHealthStatus(
                host="host-3",
                healthy=False,
                checked_at=time.time(),
                error="timeout",
            ),
        ]

        with patch(
            "app.coordination.host_health_policy.check_host_health",
            side_effect=statuses,
        ):
            summary = get_health_summary(["host-1", "host-2", "host-3"])

            assert summary["total"] == 3
            assert summary["healthy"] == 2
            assert summary["unhealthy"] == 1
            assert "healthy_hosts" in summary
            assert "unhealthy_hosts" in summary


# ============================================================================
# Cache Management Tests
# ============================================================================


class TestCacheManagement:
    """Tests for cache management functions."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_clear_health_cache(self) -> None:
        """Test clearing the health cache."""
        # Add something to cache directly via check
        mock_status = HostHealthStatus(
            host="test-host",
            healthy=True,
            checked_at=time.time(),
        )
        with patch(
            "app.coordination.host_health_policy._quick_ssh_check",
            return_value=mock_status,
        ):
            check_host_health("test-host")

        # Cache should have entry
        status = get_cache_status()
        assert status["total_entries"] > 0

        # Clear and verify
        count = clear_health_cache()
        assert count > 0

        status = get_cache_status()
        assert status["total_entries"] == 0

    def test_get_cache_status(self) -> None:
        """Test getting cache status."""
        clear_health_cache()

        # Add test data via check
        mock_status_healthy = HostHealthStatus(
            host="healthy-1",
            healthy=True,
            checked_at=time.time(),
        )
        mock_status_unhealthy = HostHealthStatus(
            host="unhealthy-1",
            healthy=False,
            checked_at=time.time(),
            error="timeout",
        )

        with patch(
            "app.coordination.host_health_policy._quick_ssh_check",
            side_effect=[mock_status_healthy, mock_status_unhealthy],
        ):
            check_host_health("healthy-1")
            check_host_health("unhealthy-1")

        status = get_cache_status()

        assert status["total_entries"] == 2
        assert "entries" in status
        assert len(status["entries"]) == 2


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_concurrent_cache_access(self) -> None:
        """Test concurrent cache access is thread-safe."""
        errors = []

        def check_host(host: str):
            try:
                mock_status = HostHealthStatus(
                    host=host,
                    healthy=True,
                    checked_at=time.time(),
                )
                with _cache_lock:
                    _health_cache[host] = mock_status
                    _ = _health_cache.get(host)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            t = threading.Thread(target=check_host, args=(f"host-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for host health policy."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_health_cache()

    def test_check_then_filter_workflow(self) -> None:
        """Test typical workflow: check hosts then filter."""
        mock_statuses = {
            "gpu-1": True,
            "gpu-2": False,
            "cpu-1": True,
        }

        def mock_check(host: str, force: bool = False) -> HostHealthStatus:
            return HostHealthStatus(
                host=host,
                healthy=mock_statuses.get(host, False),
                checked_at=time.time(),
                error=None if mock_statuses.get(host) else "failed",
            )

        with patch(
            "app.coordination.host_health_policy._quick_ssh_check",
            side_effect=mock_check,
        ):
            hosts = ["gpu-1", "gpu-2", "cpu-1"]

            # First check all
            for h in hosts:
                check_host_health(h)

            # Then filter healthy
            healthy = [h for h in hosts if is_host_healthy(h)]

            assert healthy == ["gpu-1", "cpu-1"]
