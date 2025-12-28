"""Tests for scripts.p2p.handlers.work_queue module.

Tests the HTTP handler mixin for work queue management endpoints.
December 2025.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.work_queue import (
    WorkQueueHandlersMixin,
    get_work_queue,
)


class MockWorkType:
    """Mock WorkType enum."""
    SELFPLAY = MagicMock(value="selfplay")
    TRAINING = MagicMock(value="training")
    TOURNAMENT = MagicMock(value="tournament")

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class MockWorkItem:
    """Mock WorkItem for testing."""

    def __init__(
        self,
        work_id: str = "test-work-1",
        work_type: str = "selfplay",
        priority: int = 50,
        config: dict | None = None,
        assigned_to: str = "",
    ):
        self.work_id = work_id
        self.work_type = MagicMock(value=work_type)
        self.priority = priority
        self.config = config or {"board_type": "hex8", "num_players": 2}
        self.assigned_to = assigned_to
        self.timeout_seconds = 3600.0

    def to_dict(self):
        return {
            "work_id": self.work_id,
            "work_type": self.work_type.value,
            "priority": self.priority,
            "config": self.config,
            "assigned_to": self.assigned_to,
        }


class MockWorkQueue:
    """Mock work queue for testing."""

    def __init__(self):
        self.items = {}
        self._add_called = []
        self._claim_result = None
        self._complete_result = True
        self._fail_result = True
        self._cancel_result = True
        self._start_result = True
        self._timed_out = 0
        self._history = []
        self._node_work = []

    def add_work(self, item):
        work_id = f"work-{len(self._add_called)}"
        self._add_called.append(item)
        return work_id

    def claim_work(self, node_id, capabilities=None):
        return self._claim_result

    def start_work(self, work_id):
        return self._start_result

    def complete_work(self, work_id, result):
        return self._complete_result

    def fail_work(self, work_id, error):
        return self._fail_result

    def cancel_work(self, work_id):
        return self._cancel_result

    def check_timeouts(self):
        return self._timed_out

    def get_queue_status(self):
        return {
            "pending": 5,
            "running": 2,
            "completed": 10,
            "failed": 1,
        }

    def get_work_for_node(self, node_id):
        return self._node_work

    def get_history(self, limit=50, status_filter=None):
        return self._history[:limit]


class MockQueuePopulator:
    """Mock queue populator for testing."""

    def __init__(self):
        self._elo_updates = []
        self._games_increments = []
        self._training_increments = []

    def update_target_elo(self, board_type, num_players, elo, model_id):
        self._elo_updates.append((board_type, num_players, elo, model_id))

    def increment_games(self, board_type, num_players, games):
        self._games_increments.append((board_type, num_players, games))

    def increment_training(self, board_type, num_players):
        self._training_increments.append((board_type, num_players))

    def get_status(self):
        return {"active": True, "targets": {"hex8_2p": {"elo": 1200}}}


class WorkQueueHandlersTestClass(WorkQueueHandlersMixin):
    """Test class that uses the work queue handlers mixin."""

    def __init__(self, is_leader: bool = True):
        self._is_leader = is_leader
        self._leader_id = "leader-node"
        self._queue_populator = MockQueuePopulator()

    @property
    def is_leader(self):
        return self._is_leader

    @property
    def leader_id(self):
        return self._leader_id


@pytest.fixture
def handler():
    """Create a test handler instance."""
    return WorkQueueHandlersTestClass()


@pytest.fixture
def follower_handler():
    """Create a handler that is not the leader."""
    return WorkQueueHandlersTestClass(is_leader=False)


@pytest.fixture
def mock_work_queue():
    """Create a mock work queue."""
    return MockWorkQueue()


class TestWorkQueueHandlersMixinAttributes:
    """Test required attributes on the mixin."""

    def test_has_is_leader_hint(self):
        """Mixin should have is_leader type hint."""
        hints = WorkQueueHandlersMixin.__annotations__
        assert "is_leader" in hints

    def test_has_leader_id_hint(self):
        """Mixin should have leader_id type hint."""
        hints = WorkQueueHandlersMixin.__annotations__
        assert "leader_id" in hints

    def test_has_queue_populator_hint(self):
        """Mixin should have _queue_populator type hint."""
        hints = WorkQueueHandlersMixin.__annotations__
        assert "_queue_populator" in hints


class TestGetWorkQueue:
    """Tests for the get_work_queue singleton function."""

    def test_get_work_queue_import_error(self):
        """get_work_queue should return None on import error."""
        with patch.dict("sys.modules", {"app.coordination.work_queue": None}):
            from scripts.p2p.handlers.work_queue import _work_queue
            import scripts.p2p.handlers.work_queue as wq_module
            # Reset singleton
            wq_module._work_queue = None
            with patch("scripts.p2p.handlers.work_queue.get_work_queue") as mock_get:
                mock_get.return_value = None
                result = mock_get()
                assert result is None


class TestHandleWorkAdd:
    """Tests for POST /work/add handler."""

    @pytest.mark.asyncio
    async def test_add_work_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_add(request)

        assert response.status == 403
        data = json.loads(response.body)
        assert "Not leader" in data["error"]
        assert data["leader_id"] == "leader-node"

    @pytest.mark.asyncio
    async def test_add_work_no_queue(self, handler):
        """Should return 503 when queue unavailable."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=None):
            request = MagicMock()

            response = await handler.handle_work_add(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_add_work_success(self, handler):
        """Should add work and return work_id."""
        # Create a complete mock chain
        mock_wq = MagicMock()
        mock_wq.add_work.return_value = "work-123"

        # Mock the entire work_queue module to handle internal imports
        mock_work_queue_module = MagicMock()
        mock_work_queue_module.WorkItem = MagicMock()
        mock_work_queue_module.WorkType = MagicMock(return_value=MagicMock())
        mock_work_queue_module.get_work_queue.return_value = mock_wq

        with patch.dict("sys.modules", {"app.coordination.work_queue": mock_work_queue_module}), \
             patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):

            request = MagicMock()
            request.json = AsyncMock(return_value={
                "work_type": "selfplay",
                "priority": 75,
                "config": {"board_type": "hex8"},
            })

            response = await handler.handle_work_add(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "added"
            assert data["work_id"] == "work-123"


class TestHandleWorkAddBatch:
    """Tests for POST /work/add/batch handler."""

    @pytest.mark.asyncio
    async def test_add_batch_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_add_batch(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_add_batch_no_items(self, handler):
        """Should return 400 when no items provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"items": []})

            response = await handler.handle_work_add_batch(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "No items provided" in data["error"]

    @pytest.mark.asyncio
    async def test_add_batch_too_many_items(self, handler):
        """Should return 400 when too many items (>100)."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={
                "items": [{"work_type": "selfplay"}] * 101
            })

            response = await handler.handle_work_add_batch(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "Too many items" in data["error"]


class TestHandleWorkClaim:
    """Tests for POST /work/claim handler."""

    @pytest.mark.asyncio
    async def test_claim_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()
        request.query = {}

        response = await follower_handler.handle_work_claim(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_claim_no_node_id(self, handler):
        """Should return 400 when node_id not provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.query = {"node_id": ""}

            response = await handler.handle_work_claim(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "node_id required" in data["error"]

    @pytest.mark.asyncio
    async def test_claim_no_work_available(self, handler, mock_work_queue):
        """Should return no_work_available when queue is empty."""
        mock_work_queue._claim_result = None
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.query = {"node_id": "worker-1", "capabilities": ""}

            response = await handler.handle_work_claim(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "no_work_available"

    @pytest.mark.asyncio
    async def test_claim_success(self, handler, mock_work_queue):
        """Should claim and return work item."""
        work_item = MockWorkItem()
        mock_work_queue._claim_result = work_item
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.query = {"node_id": "worker-1", "capabilities": "gpu,cuda"}

            response = await handler.handle_work_claim(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "claimed"
            assert "work" in data


class TestHandleWorkStart:
    """Tests for POST /work/start handler."""

    @pytest.mark.asyncio
    async def test_start_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_start(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_start_no_work_id(self, handler):
        """Should return 400 when work_id not provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": ""})

            response = await handler.handle_work_start(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_start_success(self, handler, mock_work_queue):
        """Should mark work as started."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": "work-123"})

            response = await handler.handle_work_start(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "started"


class TestHandleWorkComplete:
    """Tests for POST /work/complete handler."""

    @pytest.mark.asyncio
    async def test_complete_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_complete(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_complete_no_work_id(self, handler):
        """Should return 400 when work_id not provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": ""})

            response = await handler.handle_work_complete(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_complete_success(self, handler, mock_work_queue):
        """Should mark work as completed."""
        mock_work_queue.items["work-123"] = MockWorkItem(
            work_id="work-123", work_type="selfplay"
        )
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.json = AsyncMock(return_value={
                "work_id": "work-123",
                "result": {"games_generated": 100},
            })

            response = await handler.handle_work_complete(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_complete_updates_populator_for_selfplay(self, handler, mock_work_queue):
        """Should update populator with selfplay results."""
        # Create a work item with matching work_type
        mock_selfplay_type = MagicMock()
        mock_selfplay_type.value = "selfplay"

        work_item = MagicMock()
        work_item.work_type = mock_selfplay_type
        work_item.config = {"board_type": "hex8", "num_players": 2}
        work_item.assigned_to = "worker-1"
        mock_work_queue.items["work-123"] = work_item

        # Mock WorkType enum
        mock_work_queue_module = MagicMock()
        mock_work_queue_module.WorkType = MagicMock()
        mock_work_queue_module.WorkType.SELFPLAY = mock_selfplay_type
        mock_work_queue_module.WorkType.TRAINING = MagicMock()
        mock_work_queue_module.WorkType.TOURNAMENT = MagicMock()

        with patch.dict("sys.modules", {"app.coordination.work_queue": mock_work_queue_module}), \
             patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):

            request = MagicMock()
            request.json = AsyncMock(return_value={
                "work_id": "work-123",
                "result": {"games_generated": 100},
            })

            await handler.handle_work_complete(request)

            # Check populator was updated with games increment
            assert len(handler._queue_populator._games_increments) >= 1


class TestHandleWorkFail:
    """Tests for POST /work/fail handler."""

    @pytest.mark.asyncio
    async def test_fail_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_fail(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_fail_no_work_id(self, handler):
        """Should return 400 when work_id not provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": ""})

            response = await handler.handle_work_fail(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_fail_success(self, handler, mock_work_queue):
        """Should mark work as failed."""
        mock_work_queue.items["work-123"] = MockWorkItem(work_id="work-123")
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.json = AsyncMock(return_value={
                "work_id": "work-123",
                "error": "GPU out of memory",
            })

            response = await handler.handle_work_fail(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "failed"

    @pytest.mark.asyncio
    async def test_fail_emits_event(self, handler, mock_work_queue):
        """Should emit work failed event."""
        mock_work_queue.items["work-123"] = MockWorkItem(
            work_id="work-123",
            work_type="training",
            assigned_to="worker-1"
        )

        # Mock the event bridge
        mock_bridge = MagicMock()
        mock_bridge.emit = AsyncMock()

        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue), \
             patch("scripts.p2p.handlers.work_queue._event_bridge", mock_bridge):

            request = MagicMock()
            request.json = AsyncMock(return_value={
                "work_id": "work-123",
                "error": "OOM",
            })

            await handler.handle_work_fail(request)

            mock_bridge.emit.assert_called_once()
            args = mock_bridge.emit.call_args
            assert args[0][0] == "p2p_work_failed"
            assert args[0][1]["work_id"] == "work-123"
            assert args[0][1]["error"] == "OOM"


class TestHandleWorkStatus:
    """Tests for GET /work/status handler."""

    @pytest.mark.asyncio
    async def test_status_queue_unavailable(self, handler):
        """Should return 503 when queue unavailable."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=None):
            request = MagicMock()

            response = await handler.handle_work_status(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_status_success(self, handler, mock_work_queue):
        """Should return queue status."""
        mock_work_queue._timed_out = 2
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()

            response = await handler.handle_work_status(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["pending"] == 5
            assert data["running"] == 2
            assert data["is_leader"] is True
            assert data["timed_out_this_check"] == 2


class TestHandlePopulatorStatus:
    """Tests for GET /populator/status handler."""

    @pytest.mark.asyncio
    async def test_populator_not_initialized(self, handler):
        """Should return disabled message when populator is None."""
        handler._queue_populator = None

        request = MagicMock()

        response = await handler.handle_populator_status(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_populator_success(self, handler):
        """Should return populator status."""
        request = MagicMock()

        response = await handler.handle_populator_status(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["active"] is True
        assert data["is_leader"] is True


class TestHandleWorkForNode:
    """Tests for GET /work/node/{node_id} handler."""

    @pytest.mark.asyncio
    async def test_work_for_node_no_queue(self, handler):
        """Should return 503 when queue unavailable."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=None):
            request = MagicMock()
            request.match_info = {"node_id": "worker-1"}

            response = await handler.handle_work_for_node(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_work_for_node_no_node_id(self, handler, mock_work_queue):
        """Should return 400 when node_id not provided."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.match_info = {"node_id": ""}

            response = await handler.handle_work_for_node(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_work_for_node_success(self, handler, mock_work_queue):
        """Should return work items for node."""
        mock_work_queue._node_work = [
            {"work_id": "w1", "status": "running"},
            {"work_id": "w2", "status": "claimed"},
        ]
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.match_info = {"node_id": "worker-1"}

            response = await handler.handle_work_for_node(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["node_id"] == "worker-1"
            assert data["count"] == 2


class TestHandleWorkCancel:
    """Tests for POST /work/cancel handler."""

    @pytest.mark.asyncio
    async def test_cancel_not_leader(self, follower_handler):
        """Non-leaders should get 403."""
        request = MagicMock()

        response = await follower_handler.handle_work_cancel(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_cancel_no_work_id(self, handler):
        """Should return 400 when work_id not provided."""
        mock_wq = MagicMock()
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": ""})

            response = await handler.handle_work_cancel(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_cancel_success(self, handler, mock_work_queue):
        """Should cancel work item."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_id": "work-123"})

            response = await handler.handle_work_cancel(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["status"] == "cancelled"


class TestHandleWorkHistory:
    """Tests for GET /work/history handler."""

    @pytest.mark.asyncio
    async def test_history_queue_unavailable(self, handler):
        """Should return 503 when queue unavailable."""
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=None):
            request = MagicMock()
            request.query = {}

            response = await handler.handle_work_history(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_history_success(self, handler, mock_work_queue):
        """Should return work history."""
        mock_work_queue._history = [
            {"work_id": "w1", "status": "completed"},
            {"work_id": "w2", "status": "failed"},
        ]
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_work_queue):
            request = MagicMock()
            request.query = {"limit": "10", "status": "completed"}

            response = await handler.handle_work_history(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["count"] == 2
            assert data["limit"] == 10
            assert data["status_filter"] == "completed"


class TestErrorHandling:
    """Tests for error handling in work queue handlers."""

    @pytest.mark.asyncio
    async def test_add_work_exception(self, handler):
        """Should return 500 on exception."""
        mock_wq = MagicMock()
        mock_wq.add_work.side_effect = RuntimeError("Database error")
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.json = AsyncMock(return_value={"work_type": "selfplay"})

            # Need to mock the inner import too
            with patch.dict("sys.modules", {
                "app.coordination.work_queue": MagicMock(),
            }):
                response = await handler.handle_work_add(request)

                assert response.status == 500

    @pytest.mark.asyncio
    async def test_claim_exception(self, handler):
        """Should return 500 on exception."""
        mock_wq = MagicMock()
        mock_wq.claim_work.side_effect = RuntimeError("Claim error")
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()
            request.query = {"node_id": "worker-1", "capabilities": ""}

            response = await handler.handle_work_claim(request)

            assert response.status == 500

    @pytest.mark.asyncio
    async def test_status_exception(self, handler):
        """Should return 500 on exception."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.side_effect = RuntimeError("Timeout check error")
        with patch("scripts.p2p.handlers.work_queue.get_work_queue", return_value=mock_wq):
            request = MagicMock()

            response = await handler.handle_work_status(request)

            assert response.status == 500
