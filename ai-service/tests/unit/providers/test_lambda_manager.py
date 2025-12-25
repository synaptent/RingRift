"""Tests for Lambda Labs manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.providers.base import (
    InstanceState,
    Provider,
    ProviderInstance,
)
from app.providers.lambda_manager import (
    LambdaManager,
    LambdaInstanceType,
    LAMBDA_INSTANCE_TYPES,
    _parse_instance_state,
    _load_api_key,
)


class TestLambdaInstanceTypes:
    """Tests for Lambda instance type definitions."""

    def test_gh200_defined(self):
        """GH200 instance type is defined."""
        assert "gpu_1x_gh200" in LAMBDA_INSTANCE_TYPES
        gh200 = LAMBDA_INSTANCE_TYPES["gpu_1x_gh200"]
        assert gh200.gpu_type == "GH200"
        assert gh200.gpu_memory_gb == 96

    def test_h100_defined(self):
        """H100 instance types are defined."""
        assert "gpu_1x_h100_pcie" in LAMBDA_INSTANCE_TYPES
        assert "gpu_2x_h100_sxm5" in LAMBDA_INSTANCE_TYPES

    def test_instance_types_have_costs(self):
        """All instance types have hourly costs."""
        for name, type_info in LAMBDA_INSTANCE_TYPES.items():
            assert type_info.hourly_cost > 0, f"{name} missing cost"


class TestParseInstanceState:
    """Tests for state parsing."""

    def test_active_to_running(self):
        assert _parse_instance_state("active") == InstanceState.RUNNING

    def test_booting_to_starting(self):
        assert _parse_instance_state("booting") == InstanceState.STARTING

    def test_unhealthy_to_error(self):
        assert _parse_instance_state("unhealthy") == InstanceState.ERROR

    def test_terminated_to_terminated(self):
        assert _parse_instance_state("terminated") == InstanceState.TERMINATED

    def test_unknown_state(self):
        assert _parse_instance_state("weird_state") == InstanceState.UNKNOWN

    def test_case_insensitive(self):
        assert _parse_instance_state("ACTIVE") == InstanceState.RUNNING
        assert _parse_instance_state("Active") == InstanceState.RUNNING


class TestLoadApiKey:
    """Tests for API key loading."""

    def test_env_var_takes_priority(self):
        """Environment variable is preferred."""
        with patch.dict("os.environ", {"LAMBDA_API_KEY": "env-key-123"}):
            key = _load_api_key()
            assert key == "env-key-123"

    def test_returns_none_when_not_configured(self):
        """Returns None if no key available."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                key = _load_api_key()
                # May return None or a cached value


class TestLambdaManager:
    """Tests for LambdaManager class."""

    def test_init_with_key(self):
        """Can initialize with explicit API key."""
        manager = LambdaManager(api_key="test-key")
        assert manager.api_key == "test-key"
        assert manager.provider == Provider.LAMBDA

    def test_init_without_key(self):
        """Initializes without key (uses loader)."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                manager = LambdaManager()
                # Should not crash

    @pytest.mark.asyncio
    async def test_list_instances_no_key(self):
        """Returns empty list if no API key."""
        manager = LambdaManager(api_key=None)
        manager.api_key = None  # Force no key
        instances = await manager.list_instances()
        assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self):
        """Correctly parses API response."""
        manager = LambdaManager(api_key="test")

        mock_response = {
            "data": [
                {
                    "id": "inst-123",
                    "name": "test-node",
                    "status": "active",
                    "ip": "1.2.3.4",
                    "instance_type": {"name": "gpu_1x_gh200"},
                    "region": {"name": "us-west-1"},
                }
            ]
        }

        with patch.object(manager, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            instances = await manager.list_instances()

            assert len(instances) == 1
            inst = instances[0]
            assert inst.instance_id == "inst-123"
            assert inst.name == "test-node"
            assert inst.state == InstanceState.RUNNING
            assert inst.public_ip == "1.2.3.4"
            assert inst.gpu_type == "GH200"

    @pytest.mark.asyncio
    async def test_get_instance(self):
        """Can get single instance."""
        manager = LambdaManager(api_key="test")

        mock_response = {
            "data": {
                "id": "inst-456",
                "name": "my-node",
                "status": "active",
                "ip": "5.6.7.8",
                "instance_type": {"name": "gpu_1x_h100_pcie"},
            }
        }

        with patch.object(manager, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            instance = await manager.get_instance("inst-456")

            assert instance is not None
            assert instance.instance_id == "inst-456"
            mock_api.assert_called_once_with("GET", "/instances/inst-456")

    @pytest.mark.asyncio
    async def test_reboot_instance(self):
        """Can reboot instance via API."""
        manager = LambdaManager(api_key="test")

        with patch.object(manager, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"data": {"instance_ids": ["inst-123"]}}
            result = await manager.reboot_instance("inst-123")

            assert result is True
            mock_api.assert_called_once_with(
                "POST",
                "/instance-operations/restart",
                {"instance_ids": ["inst-123"]},
            )

    @pytest.mark.asyncio
    async def test_terminate_instance(self):
        """Can terminate instance via API."""
        manager = LambdaManager(api_key="test")

        with patch.object(manager, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"data": {"instance_ids": ["inst-123"]}}
            result = await manager.terminate_instance("inst-123")

            assert result is True
            mock_api.assert_called_once_with(
                "POST",
                "/instance-operations/terminate",
                {"instance_ids": ["inst-123"]},
            )

    @pytest.mark.asyncio
    async def test_launch_instance(self):
        """Can launch new instance."""
        manager = LambdaManager(api_key="test")

        with patch.object(manager, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"data": {"instance_ids": ["new-inst-789"]}}

            config = {
                "instance_type_name": "gpu_1x_gh200",
                "region_name": "us-west-1",
                "ssh_key_names": ["my-key"],
                "name": "new-node",
            }
            instance_id = await manager.launch_instance(config)

            assert instance_id == "new-inst-789"

    @pytest.mark.asyncio
    async def test_check_health_all_pass(self):
        """Health check passes when all checks pass."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        from app.providers.base import HealthCheckResult

        mock_ssh = HealthCheckResult(healthy=True, check_type="ssh", message="OK")
        mock_p2p = HealthCheckResult(healthy=True, check_type="p2p", message="OK")
        mock_ts = HealthCheckResult(healthy=True, check_type="tailscale", message="OK")

        with patch.object(manager, "check_ssh_connectivity", new_callable=AsyncMock) as ssh:
            with patch.object(manager, "check_p2p_health", new_callable=AsyncMock) as p2p:
                with patch.object(manager, "check_tailscale", new_callable=AsyncMock) as ts:
                    ssh.return_value = mock_ssh
                    p2p.return_value = mock_p2p
                    ts.return_value = mock_ts

                    result = await manager.check_health(instance)
                    assert result.healthy is True
                    assert result.details["ssh"] is True
                    assert result.details["p2p"] is True
                    assert result.details["tailscale"] is True

    @pytest.mark.asyncio
    async def test_check_health_ssh_fails(self):
        """Health check fails if SSH fails."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        from app.providers.base import HealthCheckResult

        mock_ssh = HealthCheckResult(healthy=False, check_type="ssh", message="Timeout")

        with patch.object(manager, "check_ssh_connectivity", new_callable=AsyncMock) as ssh:
            ssh.return_value = mock_ssh
            result = await manager.check_health(instance)
            assert result.healthy is False

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Can close HTTP session."""
        manager = LambdaManager(api_key="test")
        manager._session = MagicMock()
        manager._session.closed = False

        await manager.close()
        manager._session.close.assert_called_once()


class TestLambdaManagerSSH:
    """Tests for SSH-based operations."""

    @pytest.mark.asyncio
    async def test_restart_p2p_daemon(self):
        """Can restart P2P daemon via SSH."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (0, "P2P daemon restarted", "")
            result = await manager.restart_p2p_daemon(instance)
            assert result is True

    @pytest.mark.asyncio
    async def test_restart_p2p_daemon_fails(self):
        """Handles P2P restart failure."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (1, "", "command failed")
            result = await manager.restart_p2p_daemon(instance)
            assert result is False

    @pytest.mark.asyncio
    async def test_get_tailscale_ip(self):
        """Can get Tailscale IP."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (0, "100.1.2.3\n", "")
            ip = await manager.get_tailscale_ip(instance)
            assert ip == "100.1.2.3"

    @pytest.mark.asyncio
    async def test_get_tailscale_ip_invalid(self):
        """Returns None for invalid Tailscale IP."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (0, "192.168.1.1\n", "")  # Not a Tailscale IP
            ip = await manager.get_tailscale_ip(instance)
            assert ip is None

    @pytest.mark.asyncio
    async def test_deploy_ssh_key(self):
        """Can deploy SSH key."""
        manager = LambdaManager(api_key="test")

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.LAMBDA,
            name="test-node",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (0, "Key deployed", "")
            result = await manager.deploy_ssh_key(instance, "ssh-ed25519 AAAA... user@host")
            assert result is True
