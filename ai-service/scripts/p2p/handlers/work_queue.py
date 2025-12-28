"""Work Queue HTTP Handlers Mixin.

Provides HTTP endpoints for distributed work queue management.
Supports work item claiming, completion reporting, and queue status.

December 2025: Migrated to use BaseP2PHandler for consistent response formatting
and error handling. Saves ~40 LOC through consolidated patterns.

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

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.handlers_base import get_event_bridge

if TYPE_CHECKING:
    from app.coordination.unified_queue_populator import UnifiedQueuePopulator as QueuePopulator

logger = logging.getLogger(__name__)

# Event bridge manager for safe event emission (Dec 2025 consolidation)
_event_bridge = get_event_bridge()


# Work queue singleton (lazy import to avoid circular deps)
# Dec 2025: Added thread-safe initialization to prevent race conditions
import threading

_work_queue = None
_work_queue_lock = threading.Lock()


def get_work_queue():
    """Get the work queue singleton (lazy load, thread-safe)."""
    global _work_queue
    # Fast path: already initialized
    if _work_queue is not None:
        return _work_queue

    # Slow path: initialize with lock (double-check pattern)
    with _work_queue_lock:
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


class WorkQueueHandlersMixin(BaseP2PHandler):
    """Mixin providing work queue HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - is_leader: bool property
    - leader_id: str property
    - _queue_populator: QueuePopulator | None
    - node_id: str (from BaseP2PHandler)
    - auth_token: str | None (from BaseP2PHandler)
    """

    # Type hint for self to enable IDE support
    is_leader: bool
    leader_id: str
    _queue_populator: "QueuePopulator | None"

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _not_leader_response(self) -> web.Response:
        """Return 403 response for non-leader nodes."""
        return self.error_response(
            "Not leader - forward to leader",
            status=403,
            error_code="NOT_LEADER",
            details={"leader_id": self.leader_id},
        )

    def _work_queue_unavailable(self) -> web.Response:
        """Return 503 response when work queue is not available."""
        return self.error_response(
            "Work queue not available",
            status=503,
            error_code="WORK_QUEUE_UNAVAILABLE",
        )

    async def handle_work_add(self, request: web.Request) -> web.Response:
        """Add work to the centralized queue (leader only)."""
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            work_type = data.get("work_type", "selfplay")
            priority = data.get("priority", 50)
            config = data.get("config", {})
            timeout = data.get("timeout_seconds", 3600.0)
            depends_on = data.get("depends_on", [])
            force = data.get("force", False)  # Dec 28, 2025: Allow bypassing backpressure

            from app.coordination.work_queue import WorkItem, WorkType

            item = WorkItem(
                work_type=WorkType(work_type),
                priority=priority,
                config=config,
                timeout_seconds=timeout,
                depends_on=depends_on,
            )
            work_id = wq.add_work(item, force=force)

            return self.json_response({
                "status": "added",
                "work_id": work_id,
                "work_type": work_type,
                "priority": priority,
            })
        except RuntimeError as e:
            # Dec 28, 2025: Backpressure rejection - return 429 Too Many Requests
            if "BACKPRESSURE" in str(e):
                logger.warning(f"Work rejected due to backpressure: {e}")
                return self.error_response(
                    str(e),
                    status=429,
                    error_code="BACKPRESSURE",
                    details=wq.get_backpressure_status() if wq else {},
                )
            raise
        except Exception as e:
            logger.error(f"Error adding work: {e}")
            return self.error_response(str(e), status=500)

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
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            items = data.get("items", [])

            if not items:
                return self.bad_request("No items provided")

            if len(items) > 100:
                return self.bad_request("Too many items (max 100)")

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

            return self.json_response({
                "status": "added",
                "count": len(work_ids),
                "work_ids": work_ids,
                "errors": errors if errors else None,
            })
        except Exception as e:
            logger.error(f"Error adding batch work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_claim(self, request: web.Request) -> web.Response:
        """Claim available work from the queue."""
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")
            capabilities = (
                [c.strip() for c in capabilities_str.split(",") if c.strip()] or None
            )

            if not node_id:
                return self.bad_request("node_id required")

            item = wq.claim_work(node_id, capabilities)
            if item is None:
                return self.json_response({"status": "no_work_available"})

            return self.json_response({
                "status": "claimed",
                "work": item.to_dict(),
            })
        except Exception as e:
            logger.error(f"Error claiming work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_start(self, request: web.Request) -> web.Response:
        """Mark work as started (running)."""
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            work_id = data.get("work_id", "")
            if not work_id:
                return self.bad_request("work_id required")

            success = wq.start_work(work_id)
            return self.json_response({
                "status": "started" if success else "failed",
                "work_id": work_id,
            })
        except Exception as e:
            logger.error(f"Error starting work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_complete(self, request: web.Request) -> web.Response:
        """Mark work as completed successfully."""
        try:
            from app.coordination.work_queue import WorkType

            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            work_id = data.get("work_id", "")
            result = data.get("result", {})
            if not work_id:
                return self.bad_request("work_id required")

            # Dec 2025: Fixed race condition - read work item data under lock
            # before calling complete_work() which modifies state
            with wq.lock:
                work_item = wq.items.get(work_id)
                work_type = work_item.work_type if work_item else None
                # Copy config dict to avoid stale reference after lock release
                config = dict(work_item.config) if work_item else {}
                assigned_to = work_item.claimed_by if work_item else ""

            success = wq.complete_work(work_id, result)

            # Emit event to coordination EventRouter (Dec 2025 consolidation)
            if success:
                # Use locally captured assigned_to (already read under lock above)
                await _event_bridge.emit("p2p_work_completed", {
                    "work_id": work_id,
                    "work_type": work_type.value if work_type else "unknown",
                    "config_key": f"{config.get('board_type', '')}_{config.get('num_players', 0)}p",
                    "result": result,
                    "node_id": assigned_to,
                    "duration_seconds": result.get("duration_seconds", 0.0),
                })

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

            return self.json_response({
                "status": "completed" if success else "failed",
                "work_id": work_id,
            })
        except Exception as e:
            logger.error(f"Error completing work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_fail(self, request: web.Request) -> web.Response:
        """Mark work as failed (may retry based on attempts)."""
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            work_id = data.get("work_id", "")
            error = data.get("error", "unknown")
            if not work_id:
                return self.bad_request("work_id required")

            # Dec 2025: Fixed race condition - read work item data under lock
            # before calling fail_work() which modifies state
            with wq.lock:
                work_item = wq.items.get(work_id)
                work_type = work_item.work_type.value if work_item and work_item.work_type else "unknown"
                # Copy config dict to avoid stale reference after lock release
                config = dict(work_item.config) if work_item else {}
                node_id = work_item.claimed_by if work_item else ""

            success = wq.fail_work(work_id, error)

            # Emit failure event to coordination EventRouter (Dec 2025 consolidation)
            if success:
                await _event_bridge.emit("p2p_work_failed", {
                    "work_id": work_id,
                    "work_type": work_type,
                    "config_key": f"{config.get('board_type', '')}_{config.get('num_players', 0)}p",
                    "error": error,
                    "node_id": node_id,
                })

            return self.json_response({
                "status": "failed" if success else "not_found",
                "work_id": work_id,
            })
        except Exception as e:
            logger.error(f"Error failing work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_status(self, request: web.Request) -> web.Response:
        """Get work queue status."""
        try:
            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            # Check for timeouts
            timed_out = wq.check_timeouts()

            status = wq.get_queue_status()
            status["is_leader"] = self.is_leader
            status["leader_id"] = self.leader_id
            status["timed_out_this_check"] = timed_out

            # Dec 28, 2025: Include backpressure status
            status["backpressure"] = wq.get_backpressure_status()

            return self.json_response(status)
        except Exception as e:
            logger.error(f"Error getting work status: {e}")
            return self.error_response(str(e), status=500)

    async def handle_populator_status(self, request: web.Request) -> web.Response:
        """Get queue populator status for monitoring."""
        try:
            if self._queue_populator is None:
                return self.json_response({
                    "enabled": False,
                    "message": "Queue populator not initialized",
                })

            status = self._queue_populator.get_status()
            status["is_leader"] = self.is_leader
            return self.json_response(status)
        except Exception as e:
            logger.error(f"Error getting populator status: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_for_node(self, request: web.Request) -> web.Response:
        """Get all work assigned to a specific node."""
        try:
            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            node_id = request.match_info.get("node_id", "")
            if not node_id:
                return self.bad_request("node_id required")

            work_items = wq.get_work_for_node(node_id)
            return self.json_response({
                "node_id": node_id,
                "work_items": work_items,
                "count": len(work_items),
            })
        except Exception as e:
            logger.error(f"Error getting work for node: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_cancel(self, request: web.Request) -> web.Response:
        """Cancel a pending or claimed work item."""
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            work_id = data.get("work_id", "")
            if not work_id:
                return self.bad_request("work_id required")

            success = wq.cancel_work(work_id)
            return self.json_response({
                "status": "cancelled" if success else "failed",
                "work_id": work_id,
            })
        except Exception as e:
            logger.error(f"Error cancelling work: {e}")
            return self.error_response(str(e), status=500)

    async def handle_work_history(self, request: web.Request) -> web.Response:
        """Get work history from the database."""
        try:
            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            limit = int(request.query.get("limit", "50"))
            status_filter = request.query.get("status", None)

            history = wq.get_history(limit=limit, status_filter=status_filter)
            return self.json_response({
                "history": history,
                "count": len(history),
                "limit": limit,
                "status_filter": status_filter,
            })
        except Exception as e:
            logger.error(f"Error getting work history: {e}")
            return self.error_response(str(e), status=500)
