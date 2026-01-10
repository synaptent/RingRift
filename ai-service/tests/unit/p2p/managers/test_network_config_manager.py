"""Unit tests for NetworkConfigManager.

Tests IP discovery, validation, health checking, and factory functions.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.managers.network_config_manager import (
    NetworkConfig,
    NetworkConfigManager,
    NetworkState,
    create_network_config_manager,
    get_network_config_manager,
    set_network_config_manager,
)


# ============================================================================
# NetworkConfig Tests
# ============================================================================


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NetworkConfig()
        assert config.prefer_ipv4 is True
        assert config.include_private_ips is False
        assert config.tailscale_ipv4_prefix == "100."
        assert config.tailscale_ipv6_prefix == "fd7a:115c:a1e0:"
        assert config.revalidation_interval == 300.0
        assert config.initial_revalidation_delay == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NetworkConfig(
            prefer_ipv4=False,
            include_private_ips=True,
            tailscale_ipv4_prefix="10.",
            revalidation_interval=60.0,
        )
        assert config.prefer_ipv4 is False
        assert config.include_private_ips is True
        assert config.tailscale_ipv4_prefix == "10."
        assert config.revalidation_interval == 60.0


# ============================================================================
# NetworkState Tests
# ============================================================================


class TestNetworkState:
    """Tests for NetworkState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = NetworkState()
        assert state.advertise_host == ""
        assert state.advertise_port == 8770
        assert state.alternate_ips == set()
        assert state.has_tailscale is False
        assert state.last_validation_time == 0.0

    def test_custom_values(self):
        """Test custom state values."""
        state = NetworkState(
            advertise_host="100.64.0.1",
            advertise_port=9000,
            alternate_ips={"192.168.1.1", "10.0.0.1"},
            has_tailscale=True,
            last_validation_time=12345.0,
        )
        assert state.advertise_host == "100.64.0.1"
        assert state.advertise_port == 9000
        assert len(state.alternate_ips) == 2
        assert state.has_tailscale is True


# ============================================================================
# NetworkConfigManager Initialization Tests
# ============================================================================


class TestNetworkConfigManagerInit:
    """Tests for NetworkConfigManager initialization."""

    def test_init_with_minimal_args(self):
        """Test initialization with minimal arguments."""
        manager = NetworkConfigManager(node_id="test-node")
        assert manager.node_id == "test-node"
        assert manager.advertise_host == ""
        assert manager.advertise_port == 8770
        assert manager.alternate_ips == set()
        assert manager.has_tailscale is False

    def test_init_with_all_args(self):
        """Test initialization with all arguments."""
        callback = MagicMock()
        config = NetworkConfig(prefer_ipv4=False)

        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
            initial_port=9000,
            config=config,
            on_host_changed=callback,
        )

        assert manager.node_id == "test-node"
        assert manager.advertise_host == "100.64.0.1"
        assert manager.advertise_port == 9000


# ============================================================================
# Property Tests
# ============================================================================


class TestProperties:
    """Tests for manager properties."""

    def test_advertise_host_getter(self):
        """Test advertise_host property getter."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
        )
        assert manager.advertise_host == "100.64.0.1"

    def test_advertise_host_setter(self):
        """Test advertise_host property setter."""
        manager = NetworkConfigManager(node_id="test-node")
        manager.advertise_host = "100.64.0.2"
        assert manager.advertise_host == "100.64.0.2"

    def test_alternate_ips_getter(self):
        """Test alternate_ips returns a copy."""
        manager = NetworkConfigManager(node_id="test-node")
        manager._state.alternate_ips = {"10.0.0.1", "10.0.0.2"}

        # Get copy
        alts = manager.alternate_ips
        assert alts == {"10.0.0.1", "10.0.0.2"}

        # Modify copy doesn't affect original
        alts.add("10.0.0.3")
        assert manager.alternate_ips == {"10.0.0.1", "10.0.0.2"}

    def test_alternate_ips_setter(self):
        """Test alternate_ips property setter."""
        manager = NetworkConfigManager(node_id="test-node")
        manager.alternate_ips = {"10.0.0.1", "10.0.0.2"}
        assert manager.alternate_ips == {"10.0.0.1", "10.0.0.2"}

    def test_alternate_ips_setter_with_none(self):
        """Test alternate_ips setter handles None-like values."""
        manager = NetworkConfigManager(node_id="test-node")
        manager._state.alternate_ips = {"10.0.0.1"}
        manager.alternate_ips = set()
        assert manager.alternate_ips == set()


# ============================================================================
# IP Discovery Tests
# ============================================================================


class TestDiscoverAllIps:
    """Tests for discover_all_ips()."""

    def test_discover_excludes_loopback(self):
        """Test that discovery excludes loopback addresses."""
        manager = NetworkConfigManager(node_id="test-node")

        # Mock all internal discovery methods
        with patch.object(manager, "_discover_tailscale_ips", return_value={"100.64.0.1"}):
            with patch.object(manager, "_discover_hostname_ips", return_value={"127.0.0.1", "192.168.1.1"}):
                with patch.object(manager, "_discover_interface_ips", return_value=set()):
                    with patch.object(manager, "_discover_config_ips", return_value=set()):
                        ips = manager.discover_all_ips()

        assert "127.0.0.1" not in ips
        assert "100.64.0.1" in ips
        # Note: 192.168.1.1 may or may not be included depending on private IP settings

    def test_discover_excludes_ipv6_loopback(self):
        """Test that discovery excludes IPv6 loopback."""
        manager = NetworkConfigManager(node_id="test-node")

        with patch.object(manager, "_discover_tailscale_ips", return_value=set()):
            with patch.object(manager, "_discover_hostname_ips", return_value={"::1"}):
                with patch.object(manager, "_discover_interface_ips", return_value=set()):
                    with patch.object(manager, "_discover_config_ips", return_value=set()):
                        ips = manager.discover_all_ips()

        assert "::1" not in ips

    def test_discover_excludes_specified_primary(self):
        """Test that discovery excludes specified primary IP."""
        manager = NetworkConfigManager(node_id="test-node")

        with patch.object(manager, "_discover_tailscale_ips", return_value={"100.64.0.1", "100.64.0.2"}):
            with patch.object(manager, "_discover_hostname_ips", return_value=set()):
                with patch.object(manager, "_discover_interface_ips", return_value=set()):
                    with patch.object(manager, "_discover_config_ips", return_value=set()):
                        ips = manager.discover_all_ips(exclude_primary="100.64.0.1")

        assert "100.64.0.1" not in ips
        assert "100.64.0.2" in ips

    def test_discover_updates_tailscale_flag(self):
        """Test that discovery updates has_tailscale flag."""
        manager = NetworkConfigManager(node_id="test-node")
        assert manager.has_tailscale is False

        with patch.object(manager, "_discover_tailscale_ips", return_value={"100.64.0.1"}):
            with patch.object(manager, "_discover_hostname_ips", return_value=set()):
                with patch.object(manager, "_discover_interface_ips", return_value=set()):
                    with patch.object(manager, "_discover_config_ips", return_value=set()):
                        manager.discover_all_ips()

        assert manager.has_tailscale is True


# ============================================================================
# Primary Host Selection Tests
# ============================================================================


class TestSelectPrimaryHost:
    """Tests for _select_primary_host()."""

    def test_prefers_tailscale_ipv4(self):
        """Test that Tailscale CGNAT IPv4 is preferred."""
        manager = NetworkConfigManager(node_id="test-node")

        all_ips = {
            "100.64.0.1",  # Tailscale IPv4
            "192.168.1.1",  # Private IPv4
            "fd7a:115c:a1e0::1",  # Tailscale IPv6
        }

        primary, alternates = manager._select_primary_host(all_ips)
        assert primary == "100.64.0.1"
        assert "100.64.0.1" not in alternates
        assert len(alternates) == 2

    def test_prefers_ipv4_over_ipv6(self):
        """Test that IPv4 is preferred over IPv6."""
        manager = NetworkConfigManager(node_id="test-node")

        all_ips = {
            "192.168.1.1",  # IPv4
            "2001:db8::1",  # IPv6
        }

        primary, alternates = manager._select_primary_host(all_ips)
        assert primary == "192.168.1.1"
        assert "192.168.1.1" not in alternates

    def test_uses_ipv6_if_no_ipv4(self):
        """Test that IPv6 is used when no IPv4 available."""
        manager = NetworkConfigManager(node_id="test-node")

        all_ips = {
            "2001:db8::1",
            "2001:db8::2",
        }

        primary, alternates = manager._select_primary_host(all_ips)
        assert ":" in primary  # IPv6
        assert primary in all_ips
        assert len(alternates) == 1

    def test_returns_empty_for_no_ips(self):
        """Test that empty is returned when no IPs."""
        manager = NetworkConfigManager(node_id="test-node")
        primary, alternates = manager._select_primary_host(set())
        assert primary == ""
        assert alternates == set()


# ============================================================================
# Validate and Fix Tests
# ============================================================================


class TestValidateAndFix:
    """Tests for validate_and_fix_advertise_host()."""

    def test_fix_returns_false_when_no_ips(self):
        """Test that validation fails when no IPs discovered."""
        manager = NetworkConfigManager(node_id="test-node")

        with patch.object(manager, "_discover_all_ips_internal", return_value=set()):
            result = manager.validate_and_fix_advertise_host()

        assert result is False

    def test_fix_sets_primary_when_no_current_host(self):
        """Test that primary is set when no current host."""
        manager = NetworkConfigManager(node_id="test-node")
        all_ips = {"100.64.0.1", "100.64.0.2"}

        with patch.object(manager, "_discover_all_ips_internal", return_value=all_ips):
            result = manager.validate_and_fix_advertise_host()

        assert result is True
        # Both IPs start with "100." (Tailscale prefix), so either could be selected
        # since set iteration order is non-deterministic
        assert manager.advertise_host in all_ips
        assert manager.alternate_ips == all_ips - {manager.advertise_host}

    def test_fix_keeps_valid_host(self):
        """Test that valid current host is kept."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
        )

        with patch.object(manager, "_discover_all_ips_internal", return_value={"100.64.0.1", "100.64.0.2"}):
            result = manager.validate_and_fix_advertise_host()

        assert result is True
        assert manager.advertise_host == "100.64.0.1"
        assert manager.alternate_ips == {"100.64.0.2"}

    def test_fix_switches_ipv6_to_ipv4_when_available(self):
        """Test that IPv6 is switched to IPv4 when available."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="2001:db8::1",
            config=NetworkConfig(prefer_ipv4=True),
        )

        with patch.object(manager, "_discover_all_ips_internal", return_value={"2001:db8::1", "100.64.0.1"}):
            result = manager.validate_and_fix_advertise_host()

        assert result is True
        assert manager.advertise_host == "100.64.0.1"

    def test_fix_calls_callback_on_host_change(self):
        """Test that callback is called when host changes."""
        callback = MagicMock()
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="192.168.1.1",  # Private, will be replaced
            on_host_changed=callback,
        )

        with patch.object(manager, "_discover_all_ips_internal", return_value={"100.64.0.1"}):
            result = manager.validate_and_fix_advertise_host()

        assert result is True
        callback.assert_called_once_with("192.168.1.1", "100.64.0.1")


# ============================================================================
# Format IP for URL Tests
# ============================================================================


class TestFormatIpForUrl:
    """Tests for format_ip_for_url()."""

    def test_ipv4_unchanged(self):
        """Test that IPv4 addresses are unchanged."""
        assert NetworkConfigManager.format_ip_for_url("192.168.1.1") == "192.168.1.1"
        assert NetworkConfigManager.format_ip_for_url("100.64.0.1") == "100.64.0.1"

    def test_ipv6_bracketed(self):
        """Test that IPv6 addresses are bracketed."""
        assert NetworkConfigManager.format_ip_for_url("2001:db8::1") == "[2001:db8::1]"
        assert NetworkConfigManager.format_ip_for_url("fd7a:115c:a1e0::1") == "[fd7a:115c:a1e0::1]"

    def test_already_bracketed_unchanged(self):
        """Test that already-bracketed IPv6 is unchanged."""
        assert NetworkConfigManager.format_ip_for_url("[2001:db8::1]") == "[2001:db8::1]"


# ============================================================================
# Network State Tests
# ============================================================================


class TestGetNetworkState:
    """Tests for get_network_state()."""

    def test_returns_state_dict(self):
        """Test that get_network_state returns complete state."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
            initial_port=8770,
        )
        manager._state.alternate_ips = {"10.0.0.1"}
        manager._state.has_tailscale = True
        manager._state.last_validation_time = 12345.0

        state = manager.get_network_state()

        assert state["advertise_host"] == "100.64.0.1"
        assert state["advertise_port"] == 8770
        assert state["alternate_ips"] == ["10.0.0.1"]
        assert state["has_tailscale"] is True
        assert state["last_validation_time"] == 12345.0


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health_check()."""

    def test_error_when_no_host(self):
        """Test health check returns error when no host."""
        manager = NetworkConfigManager(node_id="test-node")

        health = manager.health_check()

        assert health["status"] == "error"
        assert "No advertise host" in health["message"]

    def test_healthy_with_tailscale(self):
        """Test health check returns healthy with Tailscale."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
        )
        manager._state.has_tailscale = True

        health = manager.health_check()

        assert health["status"] == "healthy"
        assert "Tailscale available" in health["message"]

    def test_warning_without_tailscale_but_has_alternates(self):
        """Test health check returns warning without Tailscale."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="192.168.1.1",
        )
        manager._state.has_tailscale = False
        manager._state.alternate_ips = {"192.168.1.2"}

        health = manager.health_check()

        assert health["status"] == "warning"
        assert "No Tailscale" in health["message"]

    def test_degraded_without_tailscale_and_no_alternates(self):
        """Test health check returns degraded when no alternates."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="192.168.1.1",
        )
        manager._state.has_tailscale = False
        manager._state.alternate_ips = set()

        health = manager.health_check()

        assert health["status"] == "degraded"
        assert "No Tailscale and no alternate" in health["message"]

    def test_health_includes_details(self):
        """Test health check includes details."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
        )

        health = manager.health_check()

        assert "details" in health
        assert "advertise_host" in health["details"]
        assert "has_tailscale" in health["details"]
        assert "alternate_ips_count" in health["details"]
        assert "revalidation_active" in health["details"]


# ============================================================================
# Async Lifecycle Tests
# ============================================================================


class TestAsyncLifecycle:
    """Tests for async start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_revalidation_loop(self):
        """Test starting revalidation loop."""
        manager = NetworkConfigManager(
            node_id="test-node",
            config=NetworkConfig(initial_revalidation_delay=0.01),
        )

        await manager.start_revalidation_loop()
        assert manager._running is True
        assert manager._revalidation_task is not None

        try:
            await manager.stop_revalidation_loop()
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_revalidation_loop(self):
        """Test stopping revalidation loop."""
        manager = NetworkConfigManager(
            node_id="test-node",
            config=NetworkConfig(initial_revalidation_delay=0.01),
        )

        await manager.start_revalidation_loop()

        try:
            await manager.stop_revalidation_loop()
        except asyncio.CancelledError:
            pass

        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Test that start is idempotent."""
        manager = NetworkConfigManager(
            node_id="test-node",
            config=NetworkConfig(initial_revalidation_delay=0.01),
        )

        await manager.start_revalidation_loop()
        task1 = manager._revalidation_task

        await manager.start_revalidation_loop()
        task2 = manager._revalidation_task

        # Should be same task (not started twice)
        assert task1 is task2

        try:
            await manager.stop_revalidation_loop()
        except asyncio.CancelledError:
            pass


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_network_config_manager_initially_none(self):
        """Test get returns None initially."""
        import scripts.p2p.managers.network_config_manager as ncm
        original = ncm._network_config_manager
        ncm._network_config_manager = None

        try:
            assert get_network_config_manager() is None
        finally:
            ncm._network_config_manager = original

    def test_set_network_config_manager(self):
        """Test set registers the manager."""
        import scripts.p2p.managers.network_config_manager as ncm
        original = ncm._network_config_manager

        try:
            manager = NetworkConfigManager(node_id="test-node")
            set_network_config_manager(manager)
            assert get_network_config_manager() is manager
        finally:
            ncm._network_config_manager = original

    def test_set_network_config_manager_to_none(self):
        """Test set can clear the manager."""
        import scripts.p2p.managers.network_config_manager as ncm
        original = ncm._network_config_manager

        try:
            manager = NetworkConfigManager(node_id="test-node")
            set_network_config_manager(manager)
            set_network_config_manager(None)
            assert get_network_config_manager() is None
        finally:
            ncm._network_config_manager = original

    def test_create_network_config_manager(self):
        """Test create creates and registers manager."""
        import scripts.p2p.managers.network_config_manager as ncm
        original = ncm._network_config_manager

        try:
            manager = create_network_config_manager(
                node_id="created-node",
                initial_host="100.64.0.1",
                initial_port=9000,
            )
            assert manager is not None
            assert manager.node_id == "created-node"
            assert manager.advertise_host == "100.64.0.1"
            assert manager.advertise_port == 9000
            assert get_network_config_manager() is manager
        finally:
            ncm._network_config_manager = original

    def test_create_network_config_manager_with_callback(self):
        """Test create with callback."""
        import scripts.p2p.managers.network_config_manager as ncm
        original = ncm._network_config_manager

        try:
            callback = MagicMock()
            manager = create_network_config_manager(
                node_id="created-node",
                on_host_changed=callback,
            )
            assert manager._on_host_changed is callback
        finally:
            ncm._network_config_manager = original


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_host_access(self):
        """Test concurrent access to advertise_host."""
        manager = NetworkConfigManager(
            node_id="test-node",
            initial_host="100.64.0.1",
        )

        results = []

        def read_host():
            for _ in range(100):
                host = manager.advertise_host
                results.append(host)

        threads = [threading.Thread(target=read_host) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should see the same value
        assert all(r == "100.64.0.1" for r in results)

    def test_concurrent_read_write(self):
        """Test concurrent read and write operations."""
        manager = NetworkConfigManager(node_id="test-node")

        read_results = []
        write_complete = threading.Event()

        def writer():
            for i in range(50):
                manager.advertise_host = f"100.64.0.{i}"
            write_complete.set()

        def reader():
            while not write_complete.is_set():
                host = manager.advertise_host
                read_results.append(host)

        writer_thread = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader) for _ in range(3)]

        for r in readers:
            r.start()
        writer_thread.start()

        writer_thread.join()
        for r in readers:
            r.join()

        # Final value should be the last written
        assert manager.advertise_host == "100.64.0.49"


# ============================================================================
# Discovery Helper Method Tests
# ============================================================================


class TestDiscoveryHelpers:
    """Tests for IP discovery helper methods."""

    def test_discover_tailscale_ips_graceful_failure(self):
        """Test Tailscale discovery handles ImportError gracefully."""
        import sys

        manager = NetworkConfigManager(node_id="test-node")

        # Remove the module from sys.modules to trigger ImportError on next import
        orig_module = sys.modules.pop("scripts.p2p.resource_detector", None)
        try:
            # Use a broken module that raises ImportError on attribute access
            class BrokenModule:
                def __getattr__(self, name):
                    raise ImportError("Mocked ImportError")

            sys.modules["scripts.p2p.resource_detector"] = BrokenModule()
            ips = manager._discover_tailscale_ips()

            # Should return empty set, not crash
            assert ips == set()
        finally:
            # Restore or remove the mock
            sys.modules.pop("scripts.p2p.resource_detector", None)
            if orig_module is not None:
                sys.modules["scripts.p2p.resource_detector"] = orig_module

    def test_discover_hostname_ips_graceful_failure(self):
        """Test hostname discovery handles failure gracefully."""
        manager = NetworkConfigManager(node_id="test-node")

        with patch("socket.gethostname", side_effect=OSError):
            ips = manager._discover_hostname_ips()

        assert ips == set()

    def test_discover_config_ips_graceful_failure(self):
        """Test config discovery handles ImportError gracefully."""
        import sys

        manager = NetworkConfigManager(node_id="test-node")

        # Remove the module from sys.modules to trigger ImportError on next import
        orig_module = sys.modules.pop("app.config.cluster_config", None)
        try:
            # Use a broken module that raises ImportError on attribute access
            class BrokenModule:
                def __getattr__(self, name):
                    raise ImportError("Mocked ImportError")

            sys.modules["app.config.cluster_config"] = BrokenModule()
            ips = manager._discover_config_ips()

            # Should return empty set, not crash
            assert ips == set()
        finally:
            # Restore or remove the mock
            sys.modules.pop("app.config.cluster_config", None)
            if orig_module is not None:
                sys.modules["app.config.cluster_config"] = orig_module
