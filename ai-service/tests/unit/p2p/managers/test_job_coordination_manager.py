"""Targeted tests for node-role gating in job_coordination_manager."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.managers.job_coordination_manager import JobCoordinationManager


class _SessionContext:
    """Minimal async context manager for aiohttp session tests."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def manager() -> JobCoordinationManager:
    """Create a coordination manager with a minimal orchestrator stub."""
    peer = MagicMock()
    peer.node_id = "gh200-8"
    peer.ip = "10.0.0.8"
    peer.port = 8770
    peer.is_alive.return_value = True

    orchestrator = MagicMock()
    orchestrator.peers = {"gh200-8": peer}
    orchestrator.peers_lock = threading.RLock()
    orchestrator._url_for_peer.side_effect = (
        lambda peer_obj, path: f"http://{peer_obj.ip}:{peer_obj.port}{path}"
    )
    orchestrator._auth_headers.return_value = {}

    return JobCoordinationManager(orchestrator=orchestrator)


class TestDispatchQueuedWork:
    """Tests for push dispatch policy enforcement."""

    @pytest.mark.asyncio
    async def test_skips_selfplay_push_dispatch_when_role_disallows(
        self,
        manager: JobCoordinationManager,
    ) -> None:
        """Trainer-role nodes must not receive pushed selfplay work."""
        peer = manager._peers["gh200-8"]
        session = MagicMock()
        session.post = AsyncMock()

        with patch(
            "scripts.p2p.network.get_client_session",
            return_value=_SessionContext(session),
        ), patch(
            "app.config.node_roles.node_allows_work_type",
            return_value=False,
        ) as mock_gate:
            dispatched = await manager.dispatch_queued_work(
                peer,
                {
                    "work_id": "work-123",
                    "work_type": "selfplay",
                    "config": {"board_type": "hex8", "num_players": 2},
                },
            )

        assert dispatched is False
        session.post.assert_not_called()
        mock_gate.assert_called_once_with(
            "gh200-8",
            "selfplay",
            config_key="hex8_2p",
        )


class TestScheduleDiverseSelfplay:
    """Tests for leader-side selfplay scheduling policy enforcement."""

    @pytest.mark.asyncio
    async def test_returns_none_when_node_role_disallows_selfplay(
        self,
        manager: JobCoordinationManager,
    ) -> None:
        """Trainer-role nodes must not be targeted for diverse selfplay scheduling."""
        with patch(
            "app.config.node_roles.get_node_workload_policy",
            return_value=MagicMock(role="trainer"),
        ), patch(
            "app.config.node_roles.policy_allows_work_type",
            return_value=False,
        ):
            result = await manager.schedule_diverse_selfplay_on_node("gh200-8")

        assert result is None


class TestLocalSelfplayGating:
    """Tests for local selfplay spawn policy enforcement."""

    @pytest.mark.asyncio
    async def test_start_local_job_returns_none_when_role_disallows_selfplay(self) -> None:
        """Trainer-role nodes must not spawn local selfplay jobs through the manager."""
        orchestrator = MagicMock()
        orchestrator.node_id = "gh200-9"
        orchestrator._start_local_job = AsyncMock()
        manager = JobCoordinationManager(orchestrator=orchestrator)

        with patch(
            "app.config.node_roles.node_allows_work_type",
            return_value=False,
        ) as mock_gate:
            result = await manager._start_local_job(
                "gpu_selfplay",
                board_type="square19",
                num_players=3,
                engine_mode="gpu",
            )

        assert result is None
        orchestrator._start_local_job.assert_not_called()
        mock_gate.assert_called_once_with(
            "gh200-9",
            "selfplay",
            config_key="square19_3p",
        )

    @pytest.mark.asyncio
    async def test_local_gpu_auto_scale_returns_zero_when_role_disallows_selfplay(self) -> None:
        """Trainer-role nodes must skip local GPU autoscale selfplay starts entirely."""
        orchestrator = MagicMock()
        orchestrator.node_id = "gh200-9"
        orchestrator.self_info = SimpleNamespace(
            has_gpu=True,
            disk_percent=0.0,
            training_jobs=0,
            memory_percent=0.0,
            gpu_percent=5.0,
            gpu_name="GH200",
        )
        manager = JobCoordinationManager(orchestrator=orchestrator)

        with patch(
            "app.config.node_roles.node_allows_work_type",
            return_value=False,
        ) as mock_gate, patch.object(
            manager,
            "_start_local_job",
            AsyncMock(),
        ) as mock_start:
            result = await manager.local_gpu_auto_scale()

        assert result == 0
        mock_start.assert_not_called()
        mock_gate.assert_called_once_with(
            "gh200-9",
            "selfplay",
            config_key=None,
        )
