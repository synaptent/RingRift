"""Tests for scripts.p2p.handlers.tournament module.

Tests cover:
- TournamentHandlersMixin tournament start (leader vs non-leader)
- Tournament match execution
- Tournament status queries
- Tournament result reporting
- Error handling and edge cases

December 2025.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.tournament import TournamentHandlersMixin


# =============================================================================
# Mock Types
# =============================================================================


class MockNodeRole(str, Enum):
    """Mock NodeRole enum that inherits from str like the real one."""

    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class MockPeer:
    """Mock peer for testing."""

    def __init__(self, node_id: str, healthy: bool = True):
        self.node_id = node_id
        self._healthy = healthy

    def is_healthy(self) -> bool:
        return self._healthy


class MockDistributedTournamentState:
    """Mock tournament state for testing."""

    def __init__(
        self,
        job_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        agent_ids: list | None = None,
        games_per_pairing: int = 2,
        total_matches: int = 0,
        pending_matches: list | None = None,
        status: str = "pending",
        started_at: float | None = None,
        last_update: float | None = None,
        completed_matches: int = 0,
        results: list | None = None,
    ):
        self.job_id = job_id
        self.board_type = board_type
        self.num_players = num_players
        self.agent_ids = agent_ids or []
        self.games_per_pairing = games_per_pairing
        self.total_matches = total_matches
        self.pending_matches = pending_matches or []
        self.status = status
        self.started_at = started_at if started_at is not None else time.time()
        self.last_update = last_update if last_update is not None else time.time()
        self.completed_matches = completed_matches
        self.results = results or []
        self.worker_nodes = []

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "agent_ids": self.agent_ids,
            "games_per_pairing": self.games_per_pairing,
            "total_matches": self.total_matches,
            "completed_matches": self.completed_matches,
            "status": self.status,
            "started_at": self.started_at,
            "last_update": self.last_update,
            "results": self.results,
            "worker_nodes": self.worker_nodes,
        }


# =============================================================================
# Test Class Implementation
# =============================================================================


class TournamentHandlersTestClass(TournamentHandlersMixin):
    """Test class that uses the tournament handlers mixin."""

    def __init__(
        self,
        node_id: str = "test-node",
        role: str = "follower",
        peers: dict | None = None,
    ):
        self.node_id = node_id
        self.role = role
        self.peers = peers or {}
        self.peers_lock = threading.Lock()
        self.distributed_tournament_state: dict = {}

        # Track method calls
        self._propose_tournament_calls = []
        self._run_tournament_tasks = []
        self._play_match_tasks = []

    def _propose_tournament(
        self,
        board_type: str,
        num_players: int,
        agent_ids: list,
        games_per_pairing: int,
    ) -> dict:
        """Mock proposal creation."""
        proposal = {
            "proposal_id": f"prop_{len(self._propose_tournament_calls)}",
            "board_type": board_type,
            "num_players": num_players,
            "agent_ids": agent_ids,
            "games_per_pairing": games_per_pairing,
        }
        self._propose_tournament_calls.append(proposal)
        return proposal

    async def _run_distributed_tournament(self, job_id: str) -> None:
        """Mock tournament execution."""
        self._run_tournament_tasks.append(job_id)
        await asyncio.sleep(0.01)

    async def _play_tournament_match(self, job_id: str, match_info: dict) -> None:
        """Mock match execution."""
        self._play_match_tasks.append((job_id, match_info))
        await asyncio.sleep(0.01)


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        json_data: dict | None = None,
        query: dict | None = None,
    ):
        self.headers = headers or {}
        self._json_data = json_data or {}
        self.query = query or {}

    async def json(self):
        return self._json_data


# =============================================================================
# Tournament Start Tests (Leader)
# =============================================================================


class TestHandleTournamentStartLeader:
    """Tests for handle_tournament_start when node is leader."""

    @pytest.fixture
    def handler(self):
        # Create handler as leader with healthy peers
        peers = {
            "worker-1": MockPeer("worker-1", healthy=True),
            "worker-2": MockPeer("worker-2", healthy=True),
        }
        handler = TournamentHandlersTestClass(
            node_id="leader-node",
            role=MockNodeRole.LEADER,
            peers=peers,
        )
        return handler

    @pytest.mark.asyncio
    async def test_leader_starts_tournament_directly(self, handler):
        """Leader starts tournament directly (no proposal)."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "board_type": "hex8",
                        "num_players": 2,
                        "agent_ids": ["agent1", "agent2", "agent3"],
                        "games_per_pairing": 2,
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                assert body["success"] is True
                assert "job_id" in body
                assert body["agents"] == ["agent1", "agent2", "agent3"]
                # 3 agents = 3 pairings, 2 games each = 6 matches
                assert body["total_matches"] == 6

    @pytest.mark.asyncio
    async def test_creates_correct_pairings(self, handler):
        """Creates round-robin pairings correctly."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["a", "b", "c", "d"],
                        "games_per_pairing": 1,
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                # 4 agents: 6 pairings (4 choose 2), 1 game each = 6 matches
                assert body["total_matches"] == 6

    @pytest.mark.asyncio
    async def test_stores_tournament_state(self, handler):
        """Tournament state is stored for tracking."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["agent1", "agent2"],
                        "games_per_pairing": 2,
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                job_id = body["job_id"]
                assert job_id in handler.distributed_tournament_state

    @pytest.mark.asyncio
    async def test_launches_coordinator_task(self, handler):
        """Tournament coordinator task is launched."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["agent1", "agent2"],
                    }
                )

                response = await handler.handle_tournament_start(request)
                # Give task time to start
                await asyncio.sleep(0.05)

                body = json.loads(response.body)
                assert body["job_id"] in handler._run_tournament_tasks

    @pytest.mark.asyncio
    async def test_too_few_agents_returns_400(self, handler):
        """Less than 2 agents returns 400."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            request = MockRequest(
                json_data={
                    "agent_ids": ["only_one"],
                }
            )

            response = await handler.handle_tournament_start(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert "2 agents" in body["error"]

    @pytest.mark.asyncio
    async def test_no_workers_returns_503(self):
        """No healthy workers returns 503."""
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers={},  # No peers
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["agent1", "agent2"],
                    }
                )

                response = await handler.handle_tournament_start(request)

                assert response.status == 503
                body = json.loads(response.body)
                assert "workers" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_includes_worker_list(self, handler):
        """Response includes list of workers."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["agent1", "agent2"],
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                assert "workers" in body
                assert "worker-1" in body["workers"]
                assert "worker-2" in body["workers"]

    @pytest.mark.asyncio
    async def test_only_healthy_workers_included(self):
        """Only healthy workers are included."""
        peers = {
            "healthy-1": MockPeer("healthy-1", healthy=True),
            "unhealthy-1": MockPeer("unhealthy-1", healthy=False),
            "healthy-2": MockPeer("healthy-2", healthy=True),
        }
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers=peers,
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={
                        "agent_ids": ["agent1", "agent2"],
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                assert "healthy-1" in body["workers"]
                assert "healthy-2" in body["workers"]
                assert "unhealthy-1" not in body["workers"]


# =============================================================================
# Tournament Start Tests (Non-Leader)
# =============================================================================


class TestHandleTournamentStartNonLeader:
    """Tests for handle_tournament_start when node is not leader."""

    @pytest.fixture
    def handler(self):
        return TournamentHandlersTestClass(
            node_id="follower-node",
            role=MockNodeRole.FOLLOWER,
        )

    @pytest.mark.asyncio
    async def test_non_leader_creates_proposal(self, handler):
        """Non-leader creates proposal instead of starting directly."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            request = MockRequest(
                json_data={
                    "board_type": "hex8",
                    "num_players": 2,
                    "agent_ids": ["agent1", "agent2"],
                    "games_per_pairing": 2,
                }
            )

            response = await handler.handle_tournament_start(request)

            body = json.loads(response.body)
            assert body["success"] is True
            assert body["mode"] == "proposal"
            assert "proposal_id" in body
            assert "awaiting gossip consensus" in body["status"]

    @pytest.mark.asyncio
    async def test_proposal_includes_agents(self, handler):
        """Proposal response includes agent list."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            request = MockRequest(
                json_data={
                    "agent_ids": ["model1", "model2", "model3"],
                }
            )

            response = await handler.handle_tournament_start(request)

            body = json.loads(response.body)
            assert body["agents"] == ["model1", "model2", "model3"]

    @pytest.mark.asyncio
    async def test_non_leader_too_few_agents_returns_400(self, handler):
        """Non-leader with too few agents returns 400."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            request = MockRequest(
                json_data={
                    "agent_ids": ["only_one"],
                }
            )

            response = await handler.handle_tournament_start(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_proposal_calls_method(self, handler):
        """Proposal creation calls _propose_tournament."""
        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            request = MockRequest(
                json_data={
                    "board_type": "hexagonal",
                    "num_players": 4,
                    "agent_ids": ["a", "b"],
                    "games_per_pairing": 3,
                }
            )

            await handler.handle_tournament_start(request)

            assert len(handler._propose_tournament_calls) == 1
            proposal = handler._propose_tournament_calls[0]
            assert proposal["board_type"] == "hexagonal"
            assert proposal["num_players"] == 4
            assert proposal["games_per_pairing"] == 3


# =============================================================================
# Tournament Match Tests
# =============================================================================


class TestHandleTournamentMatch:
    """Tests for handle_tournament_match endpoint."""

    @pytest.fixture
    def handler(self):
        return TournamentHandlersTestClass()

    @pytest.mark.asyncio
    async def test_match_request_accepted(self, handler):
        """Match request is accepted and started."""
        request = MockRequest(
            json_data={
                "job_id": "tournament_abc123",
                "match": {
                    "agent1": "model-a",
                    "agent2": "model-b",
                    "game_num": 0,
                },
            }
        )

        response = await handler.handle_tournament_match(request)
        await asyncio.sleep(0.05)  # Let task run

        body = json.loads(response.body)
        assert body["success"] is True
        assert body["status"] == "match_started"

    @pytest.mark.asyncio
    async def test_match_launches_task(self, handler):
        """Match request launches _play_tournament_match task."""
        match_info = {"agent1": "a", "agent2": "b", "game_num": 0}
        request = MockRequest(
            json_data={
                "job_id": "job123",
                "match": match_info,
            }
        )

        await handler.handle_tournament_match(request)
        await asyncio.sleep(0.05)

        assert len(handler._play_match_tasks) == 1
        assert handler._play_match_tasks[0] == ("job123", match_info)

    @pytest.mark.asyncio
    async def test_missing_job_id_returns_400(self, handler):
        """Missing job_id returns 400."""
        request = MockRequest(
            json_data={
                "match": {"agent1": "a", "agent2": "b"},
            }
        )

        response = await handler.handle_tournament_match(request)

        assert response.status == 400
        body = json.loads(response.body)
        assert "job_id" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_match_returns_400(self, handler):
        """Missing match info returns 400."""
        request = MockRequest(
            json_data={
                "job_id": "job123",
            }
        )

        response = await handler.handle_tournament_match(request)

        assert response.status == 400
        body = json.loads(response.body)
        assert "match" in body["error"]

    @pytest.mark.asyncio
    async def test_json_error_returns_500(self, handler):
        """JSON parse error returns 500."""
        request = MockRequest()
        request.json = AsyncMock(side_effect=Exception("Invalid JSON"))

        response = await handler.handle_tournament_match(request)

        assert response.status == 500


# =============================================================================
# Tournament Status Tests
# =============================================================================


class TestHandleTournamentStatus:
    """Tests for handle_tournament_status endpoint."""

    @pytest.fixture
    def handler(self):
        handler = TournamentHandlersTestClass()
        # Add some tournament states
        handler.distributed_tournament_state["job1"] = MockDistributedTournamentState(
            job_id="job1",
            agent_ids=["a", "b"],
            total_matches=2,
            completed_matches=1,
            status="running",
        )
        handler.distributed_tournament_state["job2"] = MockDistributedTournamentState(
            job_id="job2",
            agent_ids=["x", "y", "z"],
            total_matches=6,
            completed_matches=6,
            status="completed",
        )
        return handler

    @pytest.mark.asyncio
    async def test_get_specific_tournament(self, handler):
        """Query specific tournament by job_id."""
        request = MockRequest(query={"job_id": "job1"})

        response = await handler.handle_tournament_status(request)

        body = json.loads(response.body)
        assert body["job_id"] == "job1"
        assert body["completed_matches"] == 1
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_all_tournaments(self, handler):
        """Query all tournaments when no job_id specified."""
        request = MockRequest(query={})

        response = await handler.handle_tournament_status(request)

        body = json.loads(response.body)
        assert "job1" in body
        assert "job2" in body
        assert body["job1"]["status"] == "running"
        assert body["job2"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_unknown_tournament_returns_404(self, handler):
        """Unknown job_id returns 404."""
        request = MockRequest(query={"job_id": "nonexistent"})

        response = await handler.handle_tournament_status(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_includes_all_state_fields(self, handler):
        """Response includes all state fields."""
        request = MockRequest(query={"job_id": "job1"})

        response = await handler.handle_tournament_status(request)

        body = json.loads(response.body)
        assert "board_type" in body
        assert "num_players" in body
        assert "agent_ids" in body
        assert "total_matches" in body
        assert "completed_matches" in body
        assert "started_at" in body
        assert "worker_nodes" in body


# =============================================================================
# Tournament Result Tests
# =============================================================================


class TestHandleTournamentResult:
    """Tests for handle_tournament_result endpoint."""

    @pytest.fixture
    def handler(self):
        handler = TournamentHandlersTestClass()
        handler.distributed_tournament_state["job1"] = MockDistributedTournamentState(
            job_id="job1",
            total_matches=10,
            completed_matches=5,
            results=[],
        )
        return handler

    @pytest.mark.asyncio
    async def test_result_recorded(self, handler):
        """Match result is recorded correctly."""
        request = MockRequest(
            json_data={
                "job_id": "job1",
                "result": {
                    "agent1": "model-a",
                    "agent2": "model-b",
                    "winner": "model-a",
                    "game_length": 45,
                },
                "worker_id": "worker-1",
            }
        )

        response = await handler.handle_tournament_result(request)

        body = json.loads(response.body)
        assert body["success"] is True
        assert body["completed"] == 6  # Was 5, now 6

        state = handler.distributed_tournament_state["job1"]
        assert len(state.results) == 1
        assert state.results[0]["winner"] == "model-a"

    @pytest.mark.asyncio
    async def test_updates_completed_count(self, handler):
        """Completed count is incremented."""
        state = handler.distributed_tournament_state["job1"]
        initial_count = state.completed_matches

        request = MockRequest(
            json_data={
                "job_id": "job1",
                "result": {"winner": "a"},
                "worker_id": "worker-1",
            }
        )

        await handler.handle_tournament_result(request)

        assert state.completed_matches == initial_count + 1

    @pytest.mark.asyncio
    async def test_updates_last_update_time(self, handler):
        """Last update time is refreshed."""
        state = handler.distributed_tournament_state["job1"]
        old_time = state.last_update

        # Small delay to ensure time difference
        await asyncio.sleep(0.01)

        request = MockRequest(
            json_data={
                "job_id": "job1",
                "result": {},
                "worker_id": "worker-1",
            }
        )

        await handler.handle_tournament_result(request)

        assert state.last_update > old_time

    @pytest.mark.asyncio
    async def test_unknown_tournament_returns_404(self, handler):
        """Unknown job_id returns 404."""
        request = MockRequest(
            json_data={
                "job_id": "nonexistent",
                "result": {},
            }
        )

        response = await handler.handle_tournament_result(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_includes_progress_in_response(self, handler):
        """Response includes progress info."""
        request = MockRequest(
            json_data={
                "job_id": "job1",
                "result": {},
                "worker_id": "worker-1",
            }
        )

        response = await handler.handle_tournament_result(request)

        body = json.loads(response.body)
        assert body["total"] == 10
        assert body["completed"] == 6

    @pytest.mark.asyncio
    async def test_json_error_returns_500(self, handler):
        """JSON parse error returns 500."""
        request = MockRequest()
        request.json = AsyncMock(side_effect=Exception("Parse error"))

        response = await handler.handle_tournament_result(request)

        assert response.status == 500


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTournamentEdgeCases:
    """Edge case tests for tournament handlers."""

    @pytest.mark.asyncio
    async def test_large_tournament(self):
        """Handles tournament with many agents."""
        peers = {f"worker-{i}": MockPeer(f"worker-{i}") for i in range(10)}
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers=peers,
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                # 10 agents = 45 pairings
                request = MockRequest(
                    json_data={
                        "agent_ids": [f"agent-{i}" for i in range(10)],
                        "games_per_pairing": 2,
                    }
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                assert body["success"] is True
                # 10 choose 2 = 45 pairings, 2 games each = 90 matches
                assert body["total_matches"] == 90

    @pytest.mark.asyncio
    async def test_concurrent_tournaments(self):
        """Multiple tournaments can run concurrently."""
        peers = {"worker-1": MockPeer("worker-1")}
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers=peers,
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                # Start first tournament
                request1 = MockRequest(
                    json_data={"agent_ids": ["a", "b"]}
                )
                response1 = await handler.handle_tournament_start(request1)

                # Start second tournament
                request2 = MockRequest(
                    json_data={"agent_ids": ["x", "y"]}
                )
                response2 = await handler.handle_tournament_start(request2)

                body1 = json.loads(response1.body)
                body2 = json.loads(response2.body)

                assert body1["success"] is True
                assert body2["success"] is True
                assert body1["job_id"] != body2["job_id"]
                assert len(handler.distributed_tournament_state) == 2

    @pytest.mark.asyncio
    async def test_default_board_type(self):
        """Uses default board_type when not specified."""
        peers = {"worker-1": MockPeer("worker-1")}
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers=peers,
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={"agent_ids": ["a", "b"]}
                )

                response = await handler.handle_tournament_start(request)
                await asyncio.sleep(0.05)

                body = json.loads(response.body)
                job_id = body["job_id"]
                state = handler.distributed_tournament_state[job_id]
                assert state.board_type == "square8"  # Default

    @pytest.mark.asyncio
    async def test_default_games_per_pairing(self):
        """Uses default games_per_pairing when not specified."""
        peers = {"worker-1": MockPeer("worker-1")}
        handler = TournamentHandlersTestClass(
            role=MockNodeRole.LEADER,
            peers=peers,
        )

        with patch("scripts.p2p.types.NodeRole", MockNodeRole):
            with patch(
                "scripts.p2p.models.DistributedTournamentState",
                MockDistributedTournamentState,
            ):
                request = MockRequest(
                    json_data={"agent_ids": ["a", "b", "c"]}
                )

                response = await handler.handle_tournament_start(request)

                body = json.loads(response.body)
                # 3 pairings * 2 games (default) = 6 matches
                assert body["total_matches"] == 6

    @pytest.mark.asyncio
    async def test_result_with_empty_result_dict(self):
        """Handles empty result dict."""
        handler = TournamentHandlersTestClass()
        handler.distributed_tournament_state["job1"] = MockDistributedTournamentState(
            job_id="job1",
            total_matches=5,
            completed_matches=0,
        )

        request = MockRequest(
            json_data={
                "job_id": "job1",
                "result": {},
                "worker_id": "worker-1",
            }
        )

        response = await handler.handle_tournament_result(request)

        body = json.loads(response.body)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_empty_tournament_state(self):
        """Status query on empty state returns empty dict."""
        handler = TournamentHandlersTestClass()
        request = MockRequest(query={})

        response = await handler.handle_tournament_status(request)

        body = json.loads(response.body)
        assert body == {}
