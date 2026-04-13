"""Tests for scripts.p2p.handlers.selfplay_api."""

from __future__ import annotations

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.selfplay_api import SelfplayHandlersMixin


class _TestSelfplayHandler(SelfplayHandlersMixin):
    """Minimal concrete handler for exercising selfplay API mixin methods."""

    def __init__(self) -> None:
        self.node_id = "gh200-8"
        self.is_leader = True
        self.job_manager = MagicMock()
        self.selfplay_scheduler = MagicMock()
        self.manifest_lock = threading.RLock()
        self.cluster_data_manifest = None
        self.local_data_manifest = None
        self.selfplay_stats_history = []
        self._orchestrator = MagicMock(
            peers={},
            peers_lock=threading.RLock(),
        )

    def is_partition_readonly(self) -> bool:
        return False

    def get_partition_status(self) -> dict[str, object]:
        return {"partition_status": "healthy", "health_ratio": 1.0}

    async def _proxy_to_leader(self, request: web.Request) -> web.Response:
        return web.json_response({"proxied": True})

    async def _reduce_local_selfplay_jobs(self, target: int, reason: str) -> dict[str, object]:
        return {"target": target, "reason": reason}

    async def _run_local_canonical_selfplay(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
        seed: int,
    ) -> None:
        return None

    def _is_leader(self) -> bool:
        return bool(self.is_leader)


class TestHandleSelfplayStart:
    """Tests for POST /selfplay/start."""

    @pytest.mark.asyncio
    async def test_rejects_local_node_blocked_by_role_policy(self) -> None:
        """Trainer-role nodes must refuse incoming selfplay start requests."""
        handler = _TestSelfplayHandler()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={"board_type": "hex8", "num_players": 2, "num_games": 100}
        )

        with patch(
            "app.config.node_roles.get_local_node_workload_policy",
            return_value=MagicMock(role="trainer"),
        ), patch(
            "app.config.node_roles.policy_allows_work_type",
            return_value=False,
        ):
            response = await handler.handle_selfplay_start(request)

        assert response.status == 403
        data = json.loads(response.body)
        assert data["success"] is False
        assert "selfplay disallowed by node role policy" in data["error"]


class TestHandleDispatchSelfplay:
    """Tests for POST /dispatch_selfplay."""

    @pytest.mark.asyncio
    async def test_rejects_when_no_eligible_nodes_for_config(self) -> None:
        """Leader should refuse queueing selfplay when manifest allows no workers."""
        handler = _TestSelfplayHandler()
        handler._orchestrator.peers = {
            "gh200-11": MagicMock(is_alive=MagicMock(return_value=True)),
            "gh200-13": MagicMock(is_alive=MagicMock(return_value=True)),
        }

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"board_type": "square19", "num_players": 2, "num_games": 200}
        )

        with patch(
            "app.config.node_roles.node_allows_work_type",
            return_value=False,
        ):
            response = await handler.handle_dispatch_selfplay(request)

        assert response.status == 409
        data = json.loads(response.body)
        assert data["success"] is False
        assert data["config_key"] == "square19_2p"
        assert "no eligible P2P selfplay nodes" in data["error"]
