"""Raft Consensus HTTP Handlers Mixin.

Provides HTTP endpoints for Raft consensus status, work queue, job assignments,
and distributed locking.

December 2025: Migrated to use BaseP2PHandler for consistent response formatting.

Usage:
    class P2POrchestrator(RaftHandlersMixin, ...):
        pass

Endpoints:
    GET /raft/status  - Get Raft consensus status (leader, term, commit index)
    GET /raft/work    - Get work queue status (pending/claimed/completed counts)
    GET /raft/jobs    - Get job assignments status
    POST /raft/lock/{name}   - Acquire distributed lock
    DELETE /raft/lock/{name} - Release distributed lock
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_GOSSIP,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import Raft state classes with graceful fallback
try:
    from app.p2p.raft_state import (
        PYSYNCOBJ_AVAILABLE,
        ReplicatedJobAssignments,
        ReplicatedWorkQueue,
    )
except ImportError:
    PYSYNCOBJ_AVAILABLE = False
    ReplicatedWorkQueue = None  # type: ignore[assignment, misc]
    ReplicatedJobAssignments = None  # type: ignore[assignment, misc]

# Import constants with fallbacks
try:
    from scripts.p2p.constants import (
        CONSENSUS_MODE,
        RAFT_BIND_PORT,
        RAFT_ENABLED,
    )
except ImportError:
    RAFT_ENABLED = False
    RAFT_BIND_PORT = 4321
    CONSENSUS_MODE = "bully"


class RaftHandlersMixin(BaseP2PHandler):
    """Mixin providing Raft HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str (from BaseP2PHandler)
    - auth_token: str | None (from BaseP2PHandler)
    - _raft_work_queue: ReplicatedWorkQueue | None
    - _raft_job_assignments: ReplicatedJobAssignments | None
    - _raft_initialized: bool
    - _is_request_authorized(request) method
    - advertise_host: str (optional, for status reporting)
    """

    # Type hints for IDE support
    _raft_work_queue: Any  # Optional[ReplicatedWorkQueue]
    _raft_job_assignments: Any  # Optional[ReplicatedJobAssignments]
    _raft_initialized: bool

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_raft_status(self, request: web.Request) -> web.Response:
        """GET /raft/status - Get Raft consensus status and configuration.

        Returns Raft protocol configuration, current leader, term, and health.
        Does not require authentication for read-only status.

        Response:
            {
                "node_id": "my-node",
                "raft_enabled": true,
                "pysyncobj_available": true,
                "raft_initialized": true,
                "consensus_mode": "hybrid",
                "config": {
                    "bind_port": 4321
                },
                "work_queue": {
                    "is_ready": true,
                    "is_leader": false,
                    "leader_address": "192.168.1.10:4321"
                },
                "job_assignments": {
                    "is_ready": true,
                    "is_leader": false,
                    "leader_address": "192.168.1.10:4322"
                },
                "cluster_health": "healthy",
                "timestamp": 1703500000.0
            }
        """
        try:
            raft_initialized = getattr(self, "_raft_initialized", False)
            work_queue = getattr(self, "_raft_work_queue", None)
            job_assignments = getattr(self, "_raft_job_assignments", None)

            # Build work queue status
            work_queue_status: dict[str, Any] = {}
            if work_queue is not None:
                try:
                    work_queue_status = {
                        "is_ready": work_queue.is_ready,
                        "is_leader": work_queue.is_leader,
                        "leader_address": work_queue.leader_address,
                    }
                except Exception as e:
                    work_queue_status = {"error": str(e)}
            else:
                work_queue_status = {"available": False}

            # Build job assignments status
            job_assignments_status: dict[str, Any] = {}
            if job_assignments is not None:
                try:
                    job_assignments_status = {
                        "is_ready": job_assignments.is_ready,
                        "is_leader": job_assignments.is_leader,
                        "leader_address": job_assignments.leader_address,
                    }
                except Exception as e:
                    job_assignments_status = {"error": str(e)}
            else:
                job_assignments_status = {"available": False}

            # Determine cluster health
            cluster_health = "disabled"
            if RAFT_ENABLED and raft_initialized:
                wq_ready = work_queue_status.get("is_ready", False)
                ja_ready = job_assignments_status.get("is_ready", False)
                if wq_ready and ja_ready:
                    cluster_health = "healthy"
                elif wq_ready or ja_ready:
                    cluster_health = "partial"
                else:
                    cluster_health = "unavailable"

            return self.json_response({
                "node_id": self.node_id,
                "raft_enabled": RAFT_ENABLED,
                "pysyncobj_available": PYSYNCOBJ_AVAILABLE,
                "raft_initialized": raft_initialized,
                "consensus_mode": CONSENSUS_MODE,
                "config": {
                    "bind_port": RAFT_BIND_PORT,
                },
                "work_queue": work_queue_status,
                "job_assignments": job_assignments_status,
                "cluster_health": cluster_health,
                "timestamp": time.time(),
            })

        except Exception as e:
            logger.error(f"Error in handle_raft_status: {e}", exc_info=True)
            return self.error_response(
                str(e),
                status=500,
                details={
                    "raft_enabled": RAFT_ENABLED,
                    "pysyncobj_available": PYSYNCOBJ_AVAILABLE,
                },
            )

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_raft_work_queue(self, request: web.Request) -> web.Response:
        """GET /raft/work - Get work queue status.

        Returns work queue statistics including pending, claimed, and completed
        work item counts. Does not require authentication for read-only status.

        Response:
            {
                "node_id": "my-node",
                "enabled": true,
                "is_ready": true,
                "is_leader": false,
                "leader_address": "192.168.1.10:4321",
                "stats": {
                    "pending": 10,
                    "claimed": 3,
                    "running": 2,
                    "completed": 100,
                    "failed": 5,
                    "total": 120
                },
                "timestamp": 1703500000.0
            }
        """
        try:
            work_queue = getattr(self, "_raft_work_queue", None)
            raft_initialized = getattr(self, "_raft_initialized", False)

            if not RAFT_ENABLED or not raft_initialized or work_queue is None:
                return self.json_response({
                    "node_id": self.node_id,
                    "enabled": False,
                    "is_ready": False,
                    "stats": {},
                    "message": "Raft work queue not enabled or not initialized",
                    "timestamp": time.time(),
                })

            # Get queue statistics
            try:
                stats = work_queue.get_queue_stats()
            except Exception as e:
                logger.warning(f"Error getting work queue stats: {e}")
                stats = {"error": str(e)}

            return self.json_response({
                "node_id": self.node_id,
                "enabled": True,
                "is_ready": work_queue.is_ready,
                "is_leader": work_queue.is_leader,
                "leader_address": work_queue.leader_address,
                "stats": stats,
                "timestamp": time.time(),
            })

        except Exception as e:
            logger.error(f"Error in handle_raft_work_queue: {e}", exc_info=True)
            return self.error_response(
                str(e),
                status=500,
                details={"enabled": RAFT_ENABLED},
            )

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_raft_jobs(self, request: web.Request) -> web.Response:
        """GET /raft/jobs - Get job assignments status.

        Returns job assignment statistics including counts by status and per-node
        breakdowns. Does not require authentication for read-only status.

        Response:
            {
                "node_id": "my-node",
                "enabled": true,
                "is_ready": true,
                "is_leader": false,
                "leader_address": "192.168.1.10:4322",
                "stats": {
                    "assigned": 5,
                    "running": 8,
                    "completed": 50,
                    "failed": 2,
                    "total": 65,
                    "by_node": {
                        "node-1": {"active": 3, "total": 20},
                        "node-2": {"active": 5, "total": 25}
                    }
                },
                "timestamp": 1703500000.0
            }
        """
        try:
            job_assignments = getattr(self, "_raft_job_assignments", None)
            raft_initialized = getattr(self, "_raft_initialized", False)

            if not RAFT_ENABLED or not raft_initialized or job_assignments is None:
                return self.json_response({
                    "node_id": self.node_id,
                    "enabled": False,
                    "is_ready": False,
                    "stats": {},
                    "message": "Raft job assignments not enabled or not initialized",
                    "timestamp": time.time(),
                })

            # Get assignment statistics
            try:
                stats = job_assignments.get_assignment_stats()
            except Exception as e:
                logger.warning(f"Error getting job assignment stats: {e}")
                stats = {"error": str(e)}

            return self.json_response({
                "node_id": self.node_id,
                "enabled": True,
                "is_ready": job_assignments.is_ready,
                "is_leader": job_assignments.is_leader,
                "leader_address": job_assignments.leader_address,
                "stats": stats,
                "timestamp": time.time(),
            })

        except Exception as e:
            logger.error(f"Error in handle_raft_jobs: {e}", exc_info=True)
            return self.error_response(
                str(e),
                status=500,
                details={"enabled": RAFT_ENABLED},
            )

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_raft_lock(self, request: web.Request) -> web.Response:
        """POST /raft/lock/{name} - Acquire a distributed lock.

        Attempts to acquire a named distributed lock. Requires authentication.

        Path Parameters:
            name: Lock name to acquire

        Request Body (optional):
            {
                "timeout": 30.0  // Lock timeout in seconds (default: 300)
            }

        Response:
            {
                "acquired": true,
                "lock_name": "my-lock",
                "holder": "my-node",
                "timestamp": 1703500000.0
            }
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()

            lock_name = request.match_info.get("name", "")
            if not lock_name:
                return self.bad_request("lock name required")

            work_queue = getattr(self, "_raft_work_queue", None)
            raft_initialized = getattr(self, "_raft_initialized", False)

            if not RAFT_ENABLED or not raft_initialized or work_queue is None:
                return self.error_response(
                    "Raft not enabled or not initialized",
                    status=503,
                    error_code="RAFT_UNAVAILABLE",
                    details={"acquired": False, "lock_name": lock_name},
                )

            if not work_queue.is_ready:
                return self.error_response(
                    "Raft cluster not ready",
                    status=503,
                    error_code="CLUSTER_NOT_READY",
                    details={"acquired": False, "lock_name": lock_name},
                )

            # Try to acquire lock via the lock manager
            # Note: The lock manager is attached to the work queue SyncObj
            try:
                # Access the internal lock manager
                lock_manager = getattr(work_queue, "_ReplicatedWorkQueue__lock_manager", None)
                if lock_manager is None:
                    return self.error_response(
                        "Lock manager not available",
                        status=503,
                        error_code="LOCK_MANAGER_UNAVAILABLE",
                        details={"acquired": False, "lock_name": lock_name},
                    )

                acquired = lock_manager.tryAcquire(lock_name, sync=True)

                return self.json_response({
                    "acquired": acquired,
                    "lock_name": lock_name,
                    "holder": self.node_id if acquired else None,
                    "timestamp": time.time(),
                })

            except Exception as e:
                logger.warning(f"Error acquiring lock {lock_name}: {e}")
                return self.error_response(
                    str(e),
                    status=500,
                    details={"acquired": False, "lock_name": lock_name},
                )

        except Exception as e:
            logger.error(f"Error in handle_raft_lock: {e}", exc_info=True)
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_raft_unlock(self, request: web.Request) -> web.Response:
        """DELETE /raft/lock/{name} - Release a distributed lock.

        Releases a named distributed lock. Requires authentication.

        Path Parameters:
            name: Lock name to release

        Response:
            {
                "released": true,
                "lock_name": "my-lock",
                "timestamp": 1703500000.0
            }
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()

            lock_name = request.match_info.get("name", "")
            if not lock_name:
                return self.bad_request("lock name required")

            work_queue = getattr(self, "_raft_work_queue", None)
            raft_initialized = getattr(self, "_raft_initialized", False)

            if not RAFT_ENABLED or not raft_initialized or work_queue is None:
                return self.error_response(
                    "Raft not enabled or not initialized",
                    status=503,
                    error_code="RAFT_UNAVAILABLE",
                    details={"released": False, "lock_name": lock_name},
                )

            if not work_queue.is_ready:
                return self.error_response(
                    "Raft cluster not ready",
                    status=503,
                    error_code="CLUSTER_NOT_READY",
                    details={"released": False, "lock_name": lock_name},
                )

            # Try to release lock via the lock manager
            try:
                lock_manager = getattr(work_queue, "_ReplicatedWorkQueue__lock_manager", None)
                if lock_manager is None:
                    return self.error_response(
                        "Lock manager not available",
                        status=503,
                        error_code="LOCK_MANAGER_UNAVAILABLE",
                        details={"released": False, "lock_name": lock_name},
                    )

                lock_manager.release(lock_name)

                return self.json_response({
                    "released": True,
                    "lock_name": lock_name,
                    "timestamp": time.time(),
                })

            except Exception as e:
                logger.warning(f"Error releasing lock {lock_name}: {e}")
                return self.error_response(
                    str(e),
                    status=500,
                    details={"released": False, "lock_name": lock_name},
                )

        except Exception as e:
            logger.error(f"Error in handle_raft_unlock: {e}", exc_info=True)
            return self.error_response(str(e), status=500)
