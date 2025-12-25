"""Tests for Vast.ai manager."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from app.providers.base import (
    InstanceState,
    Provider,
    ProviderInstance,
)
from app.providers.vast_manager import (
    VastManager,
    VastOffer,
    GPU_BOARD_MAPPING,
    _parse_instance_state,
)


class TestGpuBoardMapping:
    """Tests for GPU to board type mapping."""

    def test_small_gpus_map_to_hex8(self):
        """Small GPUs should map to hex8."""
        small_gpus = ["RTX 3070", "RTX 2060S", "RTX 3060 Ti", "RTX 2080 Ti"]
        for gpu in small_gpus:
            assert GPU_BOARD_MAPPING.get(gpu) == "hex8", f"{gpu} should map to hex8"

    def test_high_end_gpus_map_to_hexagonal(self):
        """High-end GPUs should map to hexagonal."""
        high_end = ["A40", "RTX 5090", "H100"]
        for gpu in high_end:
            assert GPU_BOARD_MAPPING.get(gpu) == "hexagonal", f"{gpu} should map to hexagonal"


class TestParseInstanceState:
    """Tests for state parsing."""

    def test_running(self):
        assert _parse_instance_state("running") == InstanceState.RUNNING

    def test_loading(self):
        assert _parse_instance_state("loading") == InstanceState.STARTING

    def test_exited(self):
        assert _parse_instance_state("exited") == InstanceState.STOPPED

    def test_unknown(self):
        assert _parse_instance_state("weird") == InstanceState.UNKNOWN


class TestVastOffer:
    """Tests for VastOffer dataclass."""

    def test_create_offer(self):
        """Can create an offer."""
        offer = VastOffer(
            offer_id=12345,
            gpu_name="RTX 4090",
            num_gpus=4,
            gpu_memory_gb=96,
            cpu_cores=64,
            ram_gb=256,
            hourly_cost=2.50,
            reliability=0.98,
            location="US",
        )
        assert offer.offer_id == 12345
        assert offer.num_gpus == 4
        assert offer.hourly_cost == 2.50


class TestVastManager:
    """Tests for VastManager class."""

    def test_init(self):
        """Can initialize manager."""
        manager = VastManager()
        assert manager.provider == Provider.VAST
        assert manager._vastai_cmd is None

    @pytest.mark.asyncio
    async def test_list_instances_no_cli(self):
        """Returns empty list if CLI not available."""
        manager = VastManager()

        with patch.object(manager, "_check_vastai_available", new_callable=AsyncMock) as mock:
            mock.return_value = False
            instances = await manager.list_instances()
            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self):
        """Correctly parses vastai output."""
        manager = VastManager()

        mock_response = [
            {
                "id": 28889766,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "gpu_ram": 24576,  # MB
                "cpu_cores_effective": 16,
                "cpu_ram": 65536,
                "actual_status": "running",
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 12345,
                "dph_total": 0.45,
                "reliability": 0.99,
            }
        ]

        with patch.object(manager, "_run_vastai", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            instances = await manager.list_instances()

            assert len(instances) == 1
            inst = instances[0]
            assert inst.provider == Provider.VAST
            assert inst.state == InstanceState.RUNNING
            assert inst.gpu_type == "RTX 4090"
            assert inst.ssh_port == 12345
            assert inst.hourly_cost == 0.45

    @pytest.mark.asyncio
    async def test_get_instance(self):
        """Can get specific instance."""
        manager = VastManager()

        mock_response = [
            {"id": 111, "gpu_name": "A40", "actual_status": "running", "num_gpus": 1, "gpu_ram": 48000},
            {"id": 222, "gpu_name": "H100", "actual_status": "running", "num_gpus": 2, "gpu_ram": 80000},
        ]

        with patch.object(manager, "_run_vastai", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            instance = await manager.get_instance("222")

            assert instance is not None
            assert instance.instance_id == "222"

    @pytest.mark.asyncio
    async def test_start_instance(self):
        """Can start instance."""
        manager = VastManager()

        with patch.object(manager, "_run_vastai_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.start_instance("12345")

            assert result is True
            mock.assert_called_once_with("start", "instance", "12345")

    @pytest.mark.asyncio
    async def test_stop_instance(self):
        """Can stop instance."""
        manager = VastManager()

        with patch.object(manager, "_run_vastai_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.stop_instance("12345")

            assert result is True
            mock.assert_called_once_with("stop", "instance", "12345")

    @pytest.mark.asyncio
    async def test_terminate_instance(self):
        """Can terminate instance."""
        manager = VastManager()

        with patch.object(manager, "_run_vastai_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.terminate_instance("12345")

            assert result is True
            mock.assert_called_once_with("destroy", "instance", "12345")

    @pytest.mark.asyncio
    async def test_reboot_instance(self):
        """Reboot stops then starts."""
        manager = VastManager()

        with patch.object(manager, "stop_instance", new_callable=AsyncMock) as stop:
            with patch.object(manager, "start_instance", new_callable=AsyncMock) as start:
                stop.return_value = True
                start.return_value = True

                result = await manager.reboot_instance("12345")

                assert result is True
                stop.assert_called_once_with("12345")
                start.assert_called_once_with("12345")

    @pytest.mark.asyncio
    async def test_get_current_hourly_cost(self):
        """Calculates total hourly cost."""
        manager = VastManager()

        mock_response = [
            {"id": 1, "actual_status": "running", "dph_total": 0.50, "gpu_name": "A40", "num_gpus": 1, "gpu_ram": 48000},
            {"id": 2, "actual_status": "running", "dph_total": 0.75, "gpu_name": "A40", "num_gpus": 1, "gpu_ram": 48000},
            {"id": 3, "actual_status": "stopped", "dph_total": 0.25, "gpu_name": "A40", "num_gpus": 1, "gpu_ram": 48000},
        ]

        with patch.object(manager, "_run_vastai", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            cost = await manager.get_current_hourly_cost()

            # Only running instances
            assert cost == 1.25  # 0.50 + 0.75


class TestVastManagerSSH:
    """Tests for SSH operations."""

    @pytest.mark.asyncio
    async def test_check_health_ssh_fails(self):
        """Health check fails if SSH fails."""
        manager = VastManager()

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.VAST,
            name="vast-test",
            metadata={"ssh_host": "ssh1.vast.ai", "ssh_port": 12345},
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.return_value = (1, "", "Connection refused")
            result = await manager.check_health(instance)

            assert result.healthy is False
            assert "SSH failed" in result.message

    @pytest.mark.asyncio
    async def test_check_health_no_workers(self):
        """Health check fails if no workers running."""
        manager = VastManager()

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.VAST,
            name="vast-test",
            metadata={"ssh_host": "ssh1.vast.ai", "ssh_port": 12345},
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            # SSH OK, but no workers
            ssh.side_effect = [
                (0, "ok", ""),  # echo ok
                (0, "0", ""),  # pgrep workers
                (0, "1000", ""),  # games count
            ]
            result = await manager.check_health(instance)

            assert result.healthy is False
            assert "No selfplay workers" in result.message

    @pytest.mark.asyncio
    async def test_check_health_healthy(self):
        """Health check passes with workers running."""
        manager = VastManager()

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.VAST,
            name="vast-test",
            metadata={"ssh_host": "ssh1.vast.ai", "ssh_port": 12345},
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.side_effect = [
                (0, "ok", ""),  # echo ok
                (0, "3", ""),  # 3 workers
                (0, "5000", ""),  # 5000 games
            ]
            result = await manager.check_health(instance)

            assert result.healthy is True
            assert "3 workers" in result.message
            assert "5000 games" in result.message

    @pytest.mark.asyncio
    async def test_restart_workers(self):
        """Can restart workers."""
        manager = VastManager()

        instance = ProviderInstance(
            instance_id="test",
            provider=Provider.VAST,
            name="vast-test",
            metadata={
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 12345,
                "board_type": "hex8",
            },
        )

        with patch.object(manager, "run_ssh_command", new_callable=AsyncMock) as ssh:
            ssh.side_effect = [
                (0, "abc123 Updated to main", ""),  # git pull
                (0, "", ""),  # pkill
                (0, "12345", ""),  # new PID
            ]
            result = await manager.restart_workers(instance)

            assert result is True
