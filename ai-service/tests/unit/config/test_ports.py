"""Tests for app/config/ports.py - centralized port configuration.

These tests verify the port infrastructure that underpins cluster communication.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest


class TestP2PPorts:
    """Tests for P2P communication ports."""

    def test_p2p_default_port_value(self):
        """Test P2P_DEFAULT_PORT has expected value."""
        from app.config.ports import P2P_DEFAULT_PORT

        assert P2P_DEFAULT_PORT == 8770

    def test_p2p_port_env_override(self):
        """Test P2P port can be overridden via environment."""
        with mock.patch.dict(os.environ, {"RINGRIFT_P2P_PORT": "9999"}):
            # Need to reimport to get new value
            import importlib

            from app.config import ports

            importlib.reload(ports)
            assert ports.P2P_DEFAULT_PORT == 9999

            # Restore original
            os.environ.pop("RINGRIFT_P2P_PORT", None)
            importlib.reload(ports)

    def test_gossip_port_value(self):
        """Test GOSSIP_PORT has expected value."""
        from app.config.ports import GOSSIP_PORT

        assert GOSSIP_PORT == 8771

    def test_swim_port_value(self):
        """Test SWIM_PORT has expected value."""
        from app.config.ports import SWIM_PORT

        assert SWIM_PORT == 7947


class TestHealthPorts:
    """Tests for health and monitoring ports."""

    def test_health_check_port_value(self):
        """Test HEALTH_CHECK_PORT has expected value."""
        from app.config.ports import HEALTH_CHECK_PORT

        assert HEALTH_CHECK_PORT == 8765

    def test_metrics_port_value(self):
        """Test METRICS_PORT has expected value."""
        from app.config.ports import METRICS_PORT

        assert METRICS_PORT == 9090


class TestDataTransferPorts:
    """Tests for data transfer ports."""

    def test_data_server_port_value(self):
        """Test DATA_SERVER_PORT has expected value."""
        from app.config.ports import DATA_SERVER_PORT

        assert DATA_SERVER_PORT == 8766

    def test_distributed_data_port_value(self):
        """Test DISTRIBUTED_DATA_PORT has expected value."""
        from app.config.ports import DISTRIBUTED_DATA_PORT

        assert DISTRIBUTED_DATA_PORT == 8767

    def test_unified_sync_api_port_value(self):
        """Test UNIFIED_SYNC_API_PORT has expected value."""
        from app.config.ports import UNIFIED_SYNC_API_PORT

        assert UNIFIED_SYNC_API_PORT == 8772


class TestServicePorts:
    """Tests for AI service API ports."""

    def test_ai_service_default_port_value(self):
        """Test AI_SERVICE_DEFAULT_PORT has expected value."""
        from app.config.ports import AI_SERVICE_DEFAULT_PORT

        assert AI_SERVICE_DEFAULT_PORT == 8000

    def test_human_eval_port_value(self):
        """Test HUMAN_EVAL_PORT has expected value."""
        from app.config.ports import HUMAN_EVAL_PORT

        assert HUMAN_EVAL_PORT == 8081

    def test_training_dashboard_port_value(self):
        """Test TRAINING_DASHBOARD_PORT has expected value."""
        from app.config.ports import TRAINING_DASHBOARD_PORT

        assert TRAINING_DASHBOARD_PORT == 8080


class TestP2PUrlHelpers:
    """Tests for P2P URL helper functions."""

    def test_get_p2p_status_url_default(self):
        """Test get_p2p_status_url with defaults."""
        from app.config.ports import get_p2p_status_url

        url = get_p2p_status_url()
        assert url == "http://localhost:8770/status"

    def test_get_p2p_status_url_custom_host(self):
        """Test get_p2p_status_url with custom host."""
        from app.config.ports import get_p2p_status_url

        url = get_p2p_status_url(host="10.0.0.1")
        assert url == "http://10.0.0.1:8770/status"

    def test_get_p2p_status_url_custom_port(self):
        """Test get_p2p_status_url with custom port."""
        from app.config.ports import get_p2p_status_url

        url = get_p2p_status_url(port=9999)
        assert url == "http://localhost:9999/status"

    def test_get_p2p_status_url_custom_both(self):
        """Test get_p2p_status_url with custom host and port."""
        from app.config.ports import get_p2p_status_url

        url = get_p2p_status_url(host="192.168.1.100", port=8888)
        assert url == "http://192.168.1.100:8888/status"

    def test_get_p2p_base_url_default(self):
        """Test get_p2p_base_url with defaults."""
        from app.config.ports import get_p2p_base_url

        url = get_p2p_base_url()
        assert url == "http://127.0.0.1:8770"

    def test_get_p2p_base_url_custom_host(self):
        """Test get_p2p_base_url with custom host."""
        from app.config.ports import get_p2p_base_url

        url = get_p2p_base_url(host="nebius-h100")
        assert url == "http://nebius-h100:8770"

    def test_get_p2p_base_url_custom_port(self):
        """Test get_p2p_base_url with custom port."""
        from app.config.ports import get_p2p_base_url

        url = get_p2p_base_url(port=9000)
        assert url == "http://127.0.0.1:9000"


class TestLocalP2PUrl:
    """Tests for get_local_p2p_url function."""

    def test_get_local_p2p_url_default(self):
        """Test get_local_p2p_url returns default when no env vars."""
        from app.config.ports import get_local_p2p_url

        # Clear any P2P URL env vars
        for var in ["RINGRIFT_P2P_URL", "P2P_URL", "P2P_ORCHESTRATOR_URL"]:
            os.environ.pop(var, None)

        url = get_local_p2p_url()
        assert url == "http://127.0.0.1:8770"

    def test_get_local_p2p_url_ringrift_env(self):
        """Test get_local_p2p_url uses RINGRIFT_P2P_URL first."""
        from app.config.ports import get_local_p2p_url

        with mock.patch.dict(
            os.environ,
            {
                "RINGRIFT_P2P_URL": "http://custom:9999",
                "P2P_URL": "http://other:8888",
            },
        ):
            url = get_local_p2p_url()
            assert url == "http://custom:9999"

    def test_get_local_p2p_url_p2p_env(self):
        """Test get_local_p2p_url uses P2P_URL second."""
        from app.config.ports import get_local_p2p_url

        # Clear higher priority var
        os.environ.pop("RINGRIFT_P2P_URL", None)

        with mock.patch.dict(os.environ, {"P2P_URL": "http://fallback:7777"}):
            url = get_local_p2p_url()
            assert url == "http://fallback:7777"

    def test_get_local_p2p_url_orchestrator_env(self):
        """Test get_local_p2p_url uses P2P_ORCHESTRATOR_URL third."""
        from app.config.ports import get_local_p2p_url

        # Clear higher priority vars
        os.environ.pop("RINGRIFT_P2P_URL", None)
        os.environ.pop("P2P_URL", None)

        with mock.patch.dict(
            os.environ, {"P2P_ORCHESTRATOR_URL": "http://orchestrator:6666"}
        ):
            url = get_local_p2p_url()
            assert url == "http://orchestrator:6666"


class TestDataServerUrl:
    """Tests for get_data_server_url function."""

    def test_get_data_server_url_minimal(self):
        """Test get_data_server_url with just host."""
        from app.config.ports import get_data_server_url

        url = get_data_server_url(host="storage-node")
        assert url == "http://storage-node:8766"

    def test_get_data_server_url_custom_port(self):
        """Test get_data_server_url with custom port."""
        from app.config.ports import get_data_server_url

        url = get_data_server_url(host="storage-node", port=9999)
        assert url == "http://storage-node:9999"

    def test_get_data_server_url_with_path(self):
        """Test get_data_server_url with path."""
        from app.config.ports import get_data_server_url

        url = get_data_server_url(host="storage-node", path="/db")
        assert url == "http://storage-node:8766/db"

    def test_get_data_server_url_full_options(self):
        """Test get_data_server_url with all options."""
        from app.config.ports import get_data_server_url

        url = get_data_server_url(host="10.0.0.5", port=8800, path="/models/latest")
        assert url == "http://10.0.0.5:8800/models/latest"


class TestHealthCheckUrl:
    """Tests for get_health_check_url function."""

    def test_get_health_check_url_default_port(self):
        """Test get_health_check_url with default port."""
        from app.config.ports import get_health_check_url

        url = get_health_check_url(host="worker-1")
        assert url == "http://worker-1:8765/health"

    def test_get_health_check_url_custom_port(self):
        """Test get_health_check_url with custom port."""
        from app.config.ports import get_health_check_url

        url = get_health_check_url(host="worker-1", port=9999)
        assert url == "http://worker-1:9999/health"

    def test_get_health_check_url_ip_address(self):
        """Test get_health_check_url with IP address."""
        from app.config.ports import get_health_check_url

        url = get_health_check_url(host="192.168.1.50")
        assert url == "http://192.168.1.50:8765/health"


class TestPortUniqueness:
    """Tests for port uniqueness and avoiding conflicts."""

    def test_all_ports_are_unique(self):
        """Test that all defined ports are unique (no conflicts)."""
        from app.config import ports

        defined_ports = [
            ports.P2P_DEFAULT_PORT,
            ports.GOSSIP_PORT,
            ports.SWIM_PORT,
            ports.HEALTH_CHECK_PORT,
            ports.METRICS_PORT,
            ports.DATA_SERVER_PORT,
            ports.DISTRIBUTED_DATA_PORT,
            ports.UNIFIED_SYNC_API_PORT,
            ports.AI_SERVICE_DEFAULT_PORT,
            ports.HUMAN_EVAL_PORT,
            ports.TRAINING_DASHBOARD_PORT,
            # KEEPALIVE_DASHBOARD_PORT intentionally shares with GOSSIP
        ]

        # Check for duplicates (excluding intentional sharing)
        seen = set()
        for port in defined_ports:
            if port in seen:
                # Allow known shared ports
                if port == 8771:  # GOSSIP and KEEPALIVE share
                    continue
                pytest.fail(f"Duplicate port detected: {port}")
            seen.add(port)

    def test_ports_are_in_valid_range(self):
        """Test all ports are in valid range (1-65535)."""
        from app.config import ports

        port_names = [
            "P2P_DEFAULT_PORT",
            "GOSSIP_PORT",
            "SWIM_PORT",
            "HEALTH_CHECK_PORT",
            "METRICS_PORT",
            "DATA_SERVER_PORT",
            "DISTRIBUTED_DATA_PORT",
            "UNIFIED_SYNC_API_PORT",
            "AI_SERVICE_DEFAULT_PORT",
            "HUMAN_EVAL_PORT",
            "TRAINING_DASHBOARD_PORT",
            "KEEPALIVE_DASHBOARD_PORT",
        ]

        for name in port_names:
            port = getattr(ports, name)
            assert 1 <= port <= 65535, f"{name} has invalid port: {port}"

    def test_ports_are_above_well_known(self):
        """Test all ports are above well-known range (1024+)."""
        from app.config import ports

        port_names = [
            "P2P_DEFAULT_PORT",
            "GOSSIP_PORT",
            "SWIM_PORT",
            "HEALTH_CHECK_PORT",
            "METRICS_PORT",
            "DATA_SERVER_PORT",
            "DISTRIBUTED_DATA_PORT",
            "UNIFIED_SYNC_API_PORT",
            "AI_SERVICE_DEFAULT_PORT",
            "HUMAN_EVAL_PORT",
            "TRAINING_DASHBOARD_PORT",
        ]

        for name in port_names:
            port = getattr(ports, name)
            assert port > 1024, f"{name} uses privileged port: {port}"


class TestPortDocumentation:
    """Tests that port constants have proper documentation."""

    def test_module_has_docstring(self):
        """Test that ports module has a docstring."""
        from app.config import ports

        assert ports.__doc__ is not None
        assert len(ports.__doc__) > 50
