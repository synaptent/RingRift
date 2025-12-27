"""Work Queue HTTP Handlers Mixin.

Provides HTTP endpoints for distributed work queue management.
Supports work item claiming, completion reporting, and queue status.

Usage:
    class P2POrchestrator(WorkQueueHandlersMixin, ...):
        pass

Endpoints:
    GET /work/status - Get work queue status (pending/running counts)
    GET /work/pending - List pending work items with priorities
    POST /work/claim - Claim next available work item for this node
    POST /work/complete - Mark claimed work as completed
    POST /work/fail - Mark claimed work as failed (may retry)
    POST /work/add - Add new work item to queue (leader only)
    POST /work/populate - Trigger queue population (leader only)

Work Item States:
    pending -> claimed -> running -> completed
                      └-> failed (may return to pending if retries remain)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from aiohttp import web

if TYPE_CHECKING:
    from app.coordination.unified_queue_populator import UnifiedQueuePopulator as QueuePopulator

logger = logging.getLogger(__name__)

# Event bridge import (with fallback)
try:
    from scripts.p2p.p2p_event_bridge import (
        emit_p2p_work_completed,
        emit_p2p_work_failed,
    )
    HAS_EVENT_BRIDGE = True
except ImportError as e:
    HAS_EVENT_BRIDGE = False
    # Dec 2025: Log import failure so operators know events aren't being emitted
    logger.warning(f"[WorkQueueHandlers] Event bridge not available ({e}), work events will not be emitted")

    async def emit_p2p_work_completed(*args, **kwargs):
        pass

    async def emit_p2p_work_failed(*args, **kwargs):
        pass


# Work queue singleton (lazy import to avoid circular deps)
_work_queue = None


def get_work_queue():
    """Get the work queue singleton (lazy load)."""
    global _work_queue
    if _work_queue is None:
        try:
            from app.coordination.work_queue import get_work_queue as _get_wq

            _work_queue = _get_wq()
        except ImportError:
            _work_queue = None
    return _work_queue


class OrchestratorProtocol(Protocol):
    """Protocol defining the orchestrator interface needed by work queue handlers."""

    @property
    def is_leader(self) -> bool: ...

    @property
    def leader_id(self) -> str: ...

    @property
    def _queue_populator(self) -> "QueuePopulator | None": ...


class WorkQueueHandlersMixin:
    """Mixin providing work queue HTTP handlers.

    Requires the implementing class to have:
    - is_leader: bool property
    - leader_id: str property
    - _queue_populator: QueuePopulator | None
    """

    # Type hint for self to enable IDE support
    is_leader: bool
    leader_id: str
    _queue_populator: "QueuePopulator | None"

    async def handle_work_add(self, request: web.Request) -> web.Response:
        """Add work to the centralized queue (leader only)."""
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            work_type = data.get("work_type", "selfplay")
            priority = data.get("priority", 50)
            config = data.get("config", {})
            timeout = data.get("timeout_seconds", 3600.0)
            depends_on = data.get("depends_on", [])

            from app.coordination.work_queue import WorkItem, WorkType

            item = WorkItem(
                work_type=WorkType(work_type),
                priority=priority,
                config=config,
                timeout_seconds=timeout,
                depends_on=depends_on,
            )
            work_id = wq.add_work(item)

            return web.json_response(
                {
                    "status": "added",
                    "work_id": work_id,
                    "work_type": work_type,
                    "priority": priority,
                }
            )
        except Exception as e:
            logger.error(f"Error adding work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_add_batch(self, request: web.Request) -> web.Response:
        """Add multiple work items to the queue in a single request (leader only).

        Request body:
        {
            "items": [
                {"work_type": "selfplay", "priority": 50, "config": {...}},
                {"work_type": "training", "priority": 80, "config": {...}},
                ...
            ]
        }

        Response:
        {
            "status": "added",
            "count": 2,
            "work_ids": ["abc123", "def456"]
        }
        """
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            items = data.get("items", [])

            if not items:
                return web.json_response({"error": "no_items_provided"}, status=400)

            if len(items) > 100:
                return web.json_response(
                    {"error": "too_many_items", "max": 100}, status=400
                )

            from app.coordination.work_queue import WorkItem, WorkType

            work_ids = []
            errors = []

            for i, item_data in enumerate(items):
                try:
                    work_type = item_data.get("work_type", "selfplay")
                    priority = item_data.get("priority", 50)
                    config = item_data.get("config", {})
                    timeout = item_data.get("timeout_seconds", 3600.0)
                    depends_on = item_data.get("depends_on", [])

                    item = WorkItem(
                        work_type=WorkType(work_type),
                        priority=priority,
                        config=config,
                        timeout_seconds=timeout,
                        depends_on=depends_on,
                    )
                    work_id = wq.add_work(item)
                    work_ids.append(work_id)
                except Exception as e:
                    errors.append({"index": i, "error": str(e)})

            return web.json_response(
                {
                    "status": "added",
                    "count": len(work_ids),
                    "work_ids": work_ids,
                    "errors": errors if errors else None,
                }
            )
        except Exception as e:
            logger.error(f"Error adding batch work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_claim(self, request: web.Request) -> web.Response:
        """Claim available work from the queue."""
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")
            capabilities = (
                [c.strip() for c in capabilities_str.split(",") if c.strip()] or None
            )

            if not node_id:
                return web.json_response({"error": "node_id_required"}, status=400)

            item = wq.claim_work(node_id, capabilities)
            if item is None:
                return web.json_response({"status": "no_work_available"})

            return web.json_response(
                {
                    "status": "claimed",
                    "work": item.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error claiming work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_start(self, request: web.Request) -> web.Response:
        """Mark work as started (running)."""
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            work_id = data.get("work_id", "")
            if not work_id:
                return web.json_response({"error": "work_id_required"}, status=400)

            success = wq.start_work(work_id)
            return web.json_response(
                {"status": "started" if success else "failed", "work_id": work_id}
            )
        except Exception as e:
            logger.error(f"Error starting work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_complete(self, request: web.Request) -> web.Response:
        """Mark work as completed successfully."""
        try:
            from app.coordination.work_queue import WorkType

            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            work_id = data.get("work_id", "")
            result = data.get("result", {})
            if not work_id:
                return web.json_response({"error": "work_id_required"}, status=400)

            # Dec 2025: Fixed race condition - read work item data under lock
            # before calling complete_work() which modifies state
            with wq.lock:
                work_item = wq.items.get(work_id)
                work_type = work_item.work_type if work_item else None
                # Copy config dict to avoid stale reference after lock release
                config = dict(work_item.config) if work_item else {}
                assigned_to = work_item.assigned_to if work_item else ""

            success = wq.complete_work(work_id, result)

            # Emit event to coordination EventRouter
            if success and HAS_EVENT_BRIDGE:
                # Use locally captured assigned_to (already read under lock above)
                await emit_p2p_work_completed(
                    work_id=work_id,
                    work_type=work_type.value if work_type else "unknown",
                    config_key=f"{config.get('board_type', '')}_{config.get('num_players', 0)}p",
                    result=result,
                    node_id=assigned_to,
                    duration_seconds=result.get("duration_seconds", 0.0),
                )

            # Update queue populator with Elo data if applicable
            if success and self._queue_populator is not None:
                board_type = config.get("board_type", "")
                num_players = config.get("num_players", 0)

                if work_type == WorkType.TOURNAMENT:
                    # Tournament results include Elo updates
                    elo = (
                        result.get("best_elo")
                        or result.get("elo")
                        or result.get("winner_elo")
                    )
                    model_id = result.get("best_model") or result.get("winner_model")
                    if elo and board_type and num_players:
                        self._queue_populator.update_target_elo(
                            board_type, num_players, elo, model_id
                        )
                        logger.info(
                            f"Updated populator Elo: {board_type}_{num_players}p = {elo}"
                        )

                elif work_type == WorkType.SELFPLAY:
                    # Selfplay increments games count
                    games = result.get("games_generated", config.get("games", 0))
                    if games and board_type and num_players:
                        self._queue_populator.increment_games(
                            board_type, num_players, games
                        )

                elif work_type == WorkType.TRAINING:
                    # Training increments training runs
                    if board_type and num_players:
                        self._queue_populator.increment_training(board_type, num_players)

            return web.json_response(
                {"status": "completed" if success else "failed", "work_id": work_id}
            )
        except Exception as e:
            logger.error(f"Error completing work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_fail(self, request: web.Request) -> web.Response:
        """Mark work as failed (may retry based on attempts)."""
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            work_id = data.get("work_id", "")
            error = data.get("error", "unknown")
            if not work_id:
                return web.json_response({"error": "work_id_required"}, status=400)

            # Dec 2025: Fixed race condition - read work item data under lock
            # before calling fail_work() which modifies state
            with wq.lock:
                work_item = wq.items.get(work_id)
                work_type = work_item.work_type.value if work_item and work_item.work_type else "unknown"
                # Copy config dict to avoid stale reference after lock release
                config = dict(work_item.config) if work_item else {}
                node_id = work_item.assigned_to if work_item else ""

            success = wq.fail_work(work_id, error)

            # Emit failure event to coordination EventRouter
            if success and HAS_EVENT_BRIDGE:
                await emit_p2p_work_failed(
                    work_id=work_id,
                    work_type=work_type,
                    config_key=f"{config.get('board_type', '')}_{config.get('num_players', 0)}p",
                    error=error,
                    node_id=node_id,
                )

            return web.json_response(
                {"status": "failed" if success else "not_found", "work_id": work_id}
            )
        except Exception as e:
            logger.error(f"Error failing work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_status(self, request: web.Request) -> web.Response:
        """Get work queue status."""
        try:
            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            # Check for timeouts
            timed_out = wq.check_timeouts()

            status = wq.get_queue_status()
            status["is_leader"] = self.is_leader
            status["leader_id"] = self.leader_id
            status["timed_out_this_check"] = timed_out

            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting work status: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_populator_status(self, request: web.Request) -> web.Response:
        """Get queue populator status for monitoring."""
        try:
            if self._queue_populator is None:
                return web.json_response(
                    {
                        "enabled": False,
                        "message": "Queue populator not initialized",
                    }
                )

            status = self._queue_populator.get_status()
            status["is_leader"] = self.is_leader
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting populator status: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_for_node(self, request: web.Request) -> web.Response:
        """Get all work assigned to a specific node."""
        try:
            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            node_id = request.match_info.get("node_id", "")
            if not node_id:
                return web.json_response({"error": "node_id_required"}, status=400)

            work_items = wq.get_work_for_node(node_id)
            return web.json_response(
                {
                    "node_id": node_id,
                    "work_items": work_items,
                    "count": len(work_items),
                }
            )
        except Exception as e:
            logger.error(f"Error getting work for node: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_cancel(self, request: web.Request) -> web.Response:
        """Cancel a pending or claimed work item."""
        try:
            if not self.is_leader:
                return web.json_response(
                    {"error": "not_leader", "leader_id": self.leader_id}, status=403
                )

            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            data = await request.json()
            work_id = data.get("work_id", "")
            if not work_id:
                return web.json_response({"error": "work_id_required"}, status=400)

            success = wq.cancel_work(work_id)
            return web.json_response(
                {
                    "status": "cancelled" if success else "failed",
                    "work_id": work_id,
                }
            )
        except Exception as e:
            logger.error(f"Error cancelling work: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_work_history(self, request: web.Request) -> web.Response:
        """Get work history from the database."""
        try:
            wq = get_work_queue()
            if wq is None:
                return web.json_response(
                    {"error": "work_queue_not_available"}, status=503
                )

            limit = int(request.query.get("limit", "50"))
            status_filter = request.query.get("status", None)

            history = wq.get_history(limit=limit, status_filter=status_filter)
            return web.json_response(
                {
                    "history": history,
                    "count": len(history),
                    "limit": limit,
                    "status_filter": status_filter,
                }
            )
        except Exception as e:
            logger.error(f"Error getting work history: {e}")
            return web.json_response({"error": str(e)}, status=500)
