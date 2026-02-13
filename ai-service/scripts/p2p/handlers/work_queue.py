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
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_TOURNAMENT
from scripts.p2p.handlers.handlers_base import get_event_bridge

# January 2026: Import capacity thresholds for OOM prevention
try:
    from app.config.thresholds import (
        MIN_MEMORY_GB_FOR_TRAINING,
        MIN_MEMORY_GB_FOR_SELFPLAY,
        MIN_MEMORY_GB_FOR_GAUNTLET,
    )
except ImportError:
    # Fallback defaults if thresholds module not available
    MIN_MEMORY_GB_FOR_TRAINING = 32
    MIN_MEMORY_GB_FOR_SELFPLAY = 16
    MIN_MEMORY_GB_FOR_GAUNTLET = 16

# Work type to minimum VRAM requirements (GB)
# Session 17.32: Added to prevent OOM by checking capacity before claiming
WORK_TYPE_VRAM_REQUIREMENTS: dict[str, float] = {
    "training": float(MIN_MEMORY_GB_FOR_TRAINING),
    "selfplay": float(MIN_MEMORY_GB_FOR_SELFPLAY),
    "gauntlet": float(MIN_MEMORY_GB_FOR_GAUNTLET),
    "evaluation": float(MIN_MEMORY_GB_FOR_GAUNTLET),
    "tournament": float(MIN_MEMORY_GB_FOR_GAUNTLET),
    "sync": 2.0,  # Minimal VRAM for sync operations
    "export": 4.0,  # NPZ export needs some memory
}
DEFAULT_VRAM_REQUIREMENT = 8.0  # Fallback for unknown work types

# Disk usage threshold - refuse work if disk is too full
DISK_USAGE_CRITICAL_THRESHOLD = 85  # Percentage

# GPU utilization threshold - refuse selfplay if GPU is already busy
# Session 17.42 (Jan 20, 2026): Added to prevent OOM when training is running
# Session 17.48 (Jan 27, 2026): Lowered from 90% to 75% - idle nodes report 85-95%
# utilization due to driver overhead, causing mass rejections. 75% is more realistic.
GPU_UTILIZATION_REJECT_THRESHOLD = 75  # Percentage - reject selfplay if GPU% >= this
GPU_UTILIZATION_RECOVERY_THRESHOLD = 50  # Hysteresis band - accept work below this

# GPU name to total VRAM mapping (GB)
# Session 17.32: Used to infer VRAM capacity from gpu_name since NodeInfo
# doesn't expose gpu_vram_gb directly
GPU_VRAM_GB: dict[str, float] = {
    # NVIDIA Datacenter GPUs
    "GH200": 96.0,
    "H100": 80.0,
    "H100 PCIe": 80.0,
    "H100 SXM": 80.0,
    "A100": 80.0,
    "A100 80GB": 80.0,
    "A100 40GB": 40.0,
    "A100-SXM4-80GB": 80.0,
    "A100-SXM4-40GB": 40.0,
    "A100-PCIE-40GB": 40.0,
    "L40S": 48.0,
    "L40": 48.0,
    "A40": 48.0,
    "A30": 24.0,
    "A10": 24.0,
    "A10G": 24.0,
    "V100": 32.0,
    "V100S": 32.0,
    "T4": 16.0,
    # NVIDIA Consumer GPUs
    "RTX 5090": 32.0,
    "RTX 5080": 16.0,
    "RTX 4090": 24.0,
    "RTX 4080": 16.0,
    "RTX 4070 Ti": 12.0,
    "RTX 4060 Ti": 8.0,
    "RTX 4060": 8.0,
    "RTX 3090": 24.0,
    "RTX 3090 Ti": 24.0,
    "RTX 3080": 10.0,
    "RTX 3080 Ti": 12.0,
    "RTX 3070": 8.0,
    "RTX 3060": 12.0,
    "RTX 3060 Ti": 8.0,
    # Vultr fractional vGPU
    "A100 20GB": 20.0,
}

# January 2026: Use centralized timeouts from LoopTimeouts
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    WORK_QUEUE_ITEM_TIMEOUT = LoopTimeouts.WORK_QUEUE_ITEM
except ImportError:
    WORK_QUEUE_ITEM_TIMEOUT = 3600.0  # 1 hour fallback

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

    def _get_queue_populator(self) -> "QueuePopulator | None":
        """Get the queue populator instance.

        Jan 5, 2026 (Session 17.41): Added to check both direct reference and
        the QueuePopulatorLoop's populator. The loop creates its populator
        lazily on first run.

        Returns:
            QueuePopulator instance or None if not initialized
        """
        populator = self._queue_populator
        if populator is None:
            # Try to get from loop reference if available
            loop = getattr(self, "_queue_populator_loop", None)
            if loop is not None:
                populator = getattr(loop, "populator", None)
        return populator

    def _get_node_info(self, node_id: str) -> dict | None:
        """Get NodeInfo for a node from the peers dict.

        Session 17.32 (Jan 5, 2026): Added for capacity checking.

        Args:
            node_id: The node ID to look up

        Returns:
            NodeInfo dict or None if not found
        """
        # Try to access peers dict via orchestrator attributes
        peers = getattr(self, "peers", None)
        peers_lock = getattr(self, "peers_lock", None)

        if peers is None:
            # Try to find via _orchestrator reference
            orchestrator = getattr(self, "_orchestrator", self)
            peers = getattr(orchestrator, "peers", None)
            peers_lock = getattr(orchestrator, "peers_lock", None)

        if peers is None:
            return None

        try:
            if peers_lock:
                with peers_lock:
                    peer = peers.get(node_id)
            else:
                peer = peers.get(node_id)

            if peer is None:
                return None

            # Convert to dict if it has to_dict method
            if hasattr(peer, "to_dict"):
                return peer.to_dict()
            elif hasattr(peer, "__dict__"):
                return dict(peer.__dict__)
            else:
                return None
        except Exception as e:
            logger.debug(f"[_get_node_info] Error getting info for {node_id}: {e}")
            return None

    def _get_gpu_vram_from_name(self, gpu_name: str) -> float:
        """Infer GPU total VRAM from gpu_name string.

        Session 17.32 (Jan 5, 2026): Added to support VRAM capacity checks.
        NodeInfo provides gpu_name (e.g. "RTX 4090") but not total VRAM,
        so we look it up from the GPU_VRAM_GB mapping.

        Args:
            gpu_name: GPU name from NodeInfo (e.g. "RTX 4090", "H100", "GH200")

        Returns:
            Total VRAM in GB, or 0.0 if GPU not recognized
        """
        if not gpu_name:
            return 0.0

        # Try exact match first
        if gpu_name in GPU_VRAM_GB:
            return GPU_VRAM_GB[gpu_name]

        # Try substring matching for partial names
        # e.g. "NVIDIA GeForce RTX 4090" should match "RTX 4090"
        gpu_name_upper = gpu_name.upper()
        for known_gpu, vram in GPU_VRAM_GB.items():
            if known_gpu.upper() in gpu_name_upper:
                return vram

        # Unknown GPU - log for future mapping additions
        logger.debug(f"[capacity] Unknown GPU '{gpu_name}', no VRAM limit applied")
        return 0.0

    def _check_node_capacity(
        self,
        node_id: str,
        work_type: str,
    ) -> tuple[bool, str]:
        """Check if node has sufficient resources for a work item.

        Session 17.32 (Jan 5, 2026): Added to prevent OOM and improve utilization.
        Session 17.42 (Jan 20, 2026): Added GPU utilization and training checks.

        Checks:
        1. Disk usage < 85% (critical threshold)
        2. GPU utilization < 80% for selfplay (prevent OOM during training)
        3. No active training on node when dispatching selfplay
        4. GPU VRAM available >= job requirement

        Args:
            node_id: The node claiming work
            work_type: Type of work (selfplay, training, evaluation, etc.)

        Returns:
            Tuple of (has_capacity: bool, reason: str)
            - (True, "") if capacity is sufficient
            - (False, reason) if capacity is insufficient
        """
        node_info = self._get_node_info(node_id)

        # If we can't get node info, allow the work claim
        # (better to attempt work than block on missing info)
        if node_info is None:
            logger.debug(f"[capacity] No info for {node_id}, allowing work claim")
            return (True, "")

        # Check disk usage
        disk_percent = node_info.get("disk_percent", 0.0)
        if disk_percent >= DISK_USAGE_CRITICAL_THRESHOLD:
            reason = f"disk usage {disk_percent:.1f}% >= {DISK_USAGE_CRITICAL_THRESHOLD}%"
            logger.info(f"[capacity] {node_id} rejected: {reason}")
            return (False, reason)

        # Session 17.42: Check GPU utilization before dispatching selfplay
        # This prevents OOM when training is already running on the node
        gpu_percent = node_info.get("gpu_percent", 0.0)
        if work_type == "selfplay" and gpu_percent >= GPU_UTILIZATION_REJECT_THRESHOLD:
            reason = (
                f"GPU utilization {gpu_percent:.1f}% >= {GPU_UTILIZATION_REJECT_THRESHOLD}% "
                f"threshold (training likely running)"
            )
            logger.info(f"[capacity] {node_id} rejected for selfplay: {reason}")
            return (False, reason)

        # Session 17.42: REMOVED - training_jobs field is never updated (always 0)
        # The GPU utilization check above is sufficient proxy for "training running"
        # Removed in Session 17.48 (Jan 27, 2026) - this dead code was causing
        # confusion about why selfplay was being rejected.

        # Check GPU VRAM
        # NodeInfo has gpu_memory_percent (percentage used) and gpu_name
        # We infer total VRAM from gpu_name using GPU_VRAM_GB mapping
        gpu_memory_percent = node_info.get("gpu_memory_percent", 0.0)
        gpu_name = node_info.get("gpu_name", "")

        # Look up VRAM from GPU name
        gpu_vram_total_gb = self._get_gpu_vram_from_name(gpu_name)

        # If no GPU info available, allow work (CPU-only nodes or unknown GPU)
        if gpu_vram_total_gb <= 0:
            return (True, "")

        # Calculate available VRAM
        available_vram_gb = gpu_vram_total_gb * (1 - gpu_memory_percent / 100)

        # Get VRAM requirement for this work type
        vram_required = WORK_TYPE_VRAM_REQUIREMENTS.get(work_type, DEFAULT_VRAM_REQUIREMENT)

        if available_vram_gb < vram_required:
            reason = (
                f"available VRAM {available_vram_gb:.1f}GB < "
                f"required {vram_required:.1f}GB for {work_type}"
            )
            logger.info(f"[capacity] {node_id} rejected: {reason}")
            return (False, reason)

        return (True, "")

    def _insufficient_capacity_response(self, reason: str) -> web.Response:
        """Return response when node has insufficient capacity.

        Session 17.48 (Jan 27, 2026): Changed from HTTP 200 to HTTP 429.
        Previously, workers treated HTTP 200 as success and stopped retrying.
        HTTP 429 (Too Many Requests) properly signals "node busy, try again later".
        """
        return self.json_response({
            "status": "insufficient_capacity",
            "reason": reason,
        }, status=429)

    async def _forward_to_leader(
        self,
        endpoint: str,
        data: dict,
        method: str = "POST",
    ) -> web.Response | None:
        """Forward a request to the P2P leader.

        Session 17.45 (Jan 27, 2026): Added for reverse sync - allows non-leader
        nodes to forward work completion/failure reports to the leader.

        Args:
            endpoint: The endpoint path (e.g., "/work/complete")
            data: JSON payload to forward
            method: HTTP method (default: POST)

        Returns:
            Response from leader, or None if forwarding failed
        """
        leader_id = getattr(self, "leader_id", None)
        if not leader_id:
            logger.warning("[ForwardToLeader] No leader_id available")
            return None

        # Get leader info from peers
        peers = getattr(self, "peers", None)
        peers_lock = getattr(self, "peers_lock", None)

        if peers is None:
            logger.warning("[ForwardToLeader] No peers dict available")
            return None

        try:
            if peers_lock:
                with peers_lock:
                    leader_info = peers.get(leader_id)
            else:
                leader_info = peers.get(leader_id)

            if leader_info is None:
                logger.warning(f"[ForwardToLeader] Leader {leader_id} not in peers")
                return None

            # Get leader URL - try multiple attributes
            leader_url = (
                getattr(leader_info, "url", None)
                or getattr(leader_info, "api_url", None)
                or getattr(leader_info, "base_url", None)
            )

            if not leader_url:
                # Try to construct from IP
                leader_ip = getattr(leader_info, "tailscale_ip", None) or getattr(
                    leader_info, "ip", None
                )
                leader_port = getattr(leader_info, "port", 8770)
                if leader_ip:
                    leader_url = f"http://{leader_ip}:{leader_port}"

            if not leader_url:
                logger.warning(f"[ForwardToLeader] No URL for leader {leader_id}")
                return None

            import aiohttp

            full_url = f"{leader_url.rstrip('/')}{endpoint}"
            logger.debug(f"[ForwardToLeader] Forwarding {method} to {full_url}")

            async with aiohttp.ClientSession() as session:
                if method == "POST":
                    async with session.post(
                        full_url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        result = await resp.json()
                        return self.json_response(result, status=resp.status)
                else:
                    async with session.get(
                        full_url,
                        params=data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        result = await resp.json()
                        return self.json_response(result, status=resp.status)

        except Exception as e:
            logger.warning(f"[ForwardToLeader] Failed to forward to leader: {e}")
            return None

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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
            timeout = data.get("timeout_seconds", WORK_QUEUE_ITEM_TIMEOUT)
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
            # For other RuntimeErrors, fall through to generic error handler
            logger.error(f"Error adding work: {e}")
            return self.error_response(str(e), status=500)
        except Exception as e:
            logger.error(f"Error adding work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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
                    timeout = item_data.get("timeout_seconds", WORK_QUEUE_ITEM_TIMEOUT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_claim(self, request: web.Request) -> web.Response:
        """Claim available work from the queue.

        Session 17.32 (Jan 5, 2026): Added capacity checks before claiming work
        to prevent OOM and improve cluster utilization.

        Jan 13, 2026: Added autonomous queue fallback for split-brain resilience.
        When not leader, tries local autonomous queue before returning 403.
        This enables work claiming even during leader election failures.
        """
        try:
            # Parse request params first (needed for both leader and fallback paths)
            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")
            capabilities = (
                [c.strip() for c in capabilities_str.split(",") if c.strip()] or None
            )

            if not node_id:
                return self.bad_request("node_id required")

            # Strategy 1: Leader path - claim from centralized work queue
            if self.is_leader:
                wq = get_work_queue()
                if wq is None:
                    return self._work_queue_unavailable()

                # Session 17.32: Check node capacity before claiming work
                # Determine the most demanding work type from capabilities
                check_work_type = "selfplay"  # Default
                if capabilities:
                    # Check against highest-requirement capability
                    for cap in ["training", "gauntlet", "evaluation", "selfplay"]:
                        if cap in capabilities:
                            check_work_type = cap
                            break

                has_capacity, reason = self._check_node_capacity(node_id, check_work_type)
                if not has_capacity:
                    return self._insufficient_capacity_response(reason)

                item = wq.claim_work(node_id, capabilities)
                if item is None:
                    return self.json_response({"status": "no_work_available"})

                return self.json_response({
                    "status": "claimed",
                    "work": item.to_dict(),
                })

            # Strategy 2: Non-leader fallback - try autonomous queue
            # This enables work claiming during leader election failures
            autonomous_loop = getattr(self, "_autonomous_queue_loop", None)
            if autonomous_loop is None:
                # Try to find it on the orchestrator
                orchestrator = getattr(self, "_orchestrator", self)
                autonomous_loop = getattr(orchestrator, "_autonomous_queue_loop", None)

            if autonomous_loop and getattr(autonomous_loop, "is_activated", False):
                local_item = await autonomous_loop.claim_local_work(node_id, capabilities)
                if local_item:
                    logger.info(
                        f"[work_claim] Served {node_id} from autonomous queue (not leader)"
                    )
                    return self.json_response({
                        "status": "claimed",
                        "source": "autonomous_queue",
                        "work": local_item,
                    })

            # Strategy 3: Return not-leader with hint for client-side forwarding
            return self._not_leader_response()
        except Exception as e:
            logger.error(f"Error claiming work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_claim_batch(self, request: web.Request) -> web.Response:
        """Claim multiple work items in a single request for improved utilization.

        Session 17.34 (Jan 5, 2026): Added batch claiming to reduce round-trip
        overhead and improve GPU utilization by +30-40%.

        Session 17.32 (Jan 5, 2026): Added capacity checks before claiming work.

        Query params:
            node_id: The node claiming work (required)
            capabilities: Comma-separated work types (optional)
            max_items: Maximum items to claim, 1-10 (optional, default: 5)

        Response:
        {
            "status": "claimed" | "no_work_available" | "insufficient_capacity",
            "count": 5,
            "items": [{"work_id": "...", ...}, ...]
        }
        """
        try:
            if not self.is_leader:
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")
            max_items_str = request.query.get("max_items", "5")

            capabilities = (
                [c.strip() for c in capabilities_str.split(",") if c.strip()] or None
            )

            if not node_id:
                return self.bad_request("node_id required")

            # Session 17.32: Check node capacity before claiming work
            check_work_type = "selfplay"  # Default
            if capabilities:
                for cap in ["training", "gauntlet", "evaluation", "selfplay"]:
                    if cap in capabilities:
                        check_work_type = cap
                        break

            has_capacity, reason = self._check_node_capacity(node_id, check_work_type)
            if not has_capacity:
                return self._insufficient_capacity_response(reason)

            try:
                max_items = min(max(1, int(max_items_str)), 10)
            except ValueError:
                max_items = 5

            items = wq.claim_work_batch(node_id, max_items, capabilities)
            if not items:
                return self.json_response({"status": "no_work_available", "count": 0})

            return self.json_response({
                "status": "claimed",
                "count": len(items),
                "items": [item.to_dict() for item in items],
            })
        except Exception as e:
            logger.error(f"Error batch claiming work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_claim_training(self, request: web.Request) -> web.Response:
        """Pull-based training job claim - works without leader.

        Jan 4, 2026: Added for Phase 4 of 48-hour autonomous operation.
        Session 17.32 (Jan 5, 2026): Added capacity checks before claiming work.

        GPU nodes call this to claim training work directly. Tries:
        1. Check capacity (must have enough VRAM and disk for training)
        2. Local work queue (if leader)
        3. Autonomous local queue (if activated)
        4. Returns no_work_available if nothing found

        This allows training to continue even during leader elections or partitions.
        """
        try:
            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")

            if not node_id:
                return self.bad_request("node_id required")

            capabilities = [c.strip() for c in capabilities_str.split(",") if c.strip()] or None

            # Session 17.32: Check node capacity for training work
            has_capacity, reason = self._check_node_capacity(node_id, "training")
            if not has_capacity:
                return self._insufficient_capacity_response(reason)

            # Strategy 1: Check local work queue for training jobs (if leader)
            if self.is_leader:
                wq = get_work_queue()
                if wq is not None:
                    # Try to claim training work specifically
                    item = wq.claim_work(node_id, capabilities, work_types=["training"])
                    if item is not None:
                        return self.json_response({
                            "status": "claimed",
                            "source": "local_work_queue",
                            "work": item.to_dict(),
                        })

            # Strategy 2: Try autonomous local queue
            autonomous_loop = getattr(self, "_autonomous_queue_loop", None)
            if autonomous_loop is None:
                # Try to find it on the orchestrator
                orchestrator = getattr(self, "_orchestrator", self)
                autonomous_loop = getattr(orchestrator, "_autonomous_queue_loop", None)

            if autonomous_loop and getattr(autonomous_loop, "is_activated", False):
                local_item = await autonomous_loop.claim_local_work(node_id, capabilities)
                if local_item:
                    return self.json_response({
                        "status": "claimed",
                        "source": "autonomous_queue",
                        "work": local_item,
                    })

            return self.json_response({"status": "no_work_available"})

        except Exception as e:
            logger.error(f"Error claiming training work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_claim_evaluation(self, request: web.Request) -> web.Response:
        """Pull-based evaluation job claim - works without leader for cluster-wide model evaluation.

        Jan 9, 2026: Added for Phase 3 of cluster-wide model evaluation system.

        GPU nodes call this to claim evaluation work from the EvaluationScheduler.
        Jobs evaluate models under various harnesses (Gumbel MCTS, MaxN, BRS, etc.)
        producing fresh Elo rankings.

        Query params:
            node_id: ID of the claiming node (required)
            capabilities: Comma-separated capabilities e.g. "gpu,nn,nnue" (optional)

        Response:
        {
            "status": "claimed" | "no_work_available",
            "source": "evaluation_scheduler",
            "job": {...} | null
        }
        """
        try:
            node_id = request.query.get("node_id", "")
            capabilities_str = request.query.get("capabilities", "")

            if not node_id:
                return self.bad_request("node_id required")

            capabilities = [c.strip() for c in capabilities_str.split(",") if c.strip()] or None

            # Check node capacity for evaluation work
            has_capacity, reason = self._check_node_capacity(node_id, "evaluation")
            if not has_capacity:
                return self._insufficient_capacity_response(reason)

            # Get evaluation scheduler
            try:
                from app.coordination.evaluation_scheduler import get_evaluation_scheduler
                scheduler = get_evaluation_scheduler()
            except ImportError:
                logger.warning("EvaluationScheduler not available")
                return self.json_response({"status": "no_work_available"})

            # Get next job for this node
            job = scheduler.get_next_job(node_id=node_id, capabilities=capabilities)
            if job is None:
                return self.json_response({"status": "no_work_available"})

            # Claim the job
            if scheduler.claim_job(job.job_id, node_id):
                logger.info(f"Evaluation job {job.job_id} claimed by {node_id}")
                return self.json_response({
                    "status": "claimed",
                    "source": "evaluation_scheduler",
                    "job": job.to_dict(),
                })

            # Claim failed (race condition - another node claimed it)
            return self.json_response({"status": "no_work_available"})

        except Exception as e:
            logger.error(f"Error claiming evaluation work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_peer_claim(self, request: web.Request) -> web.Response:
        """Allow peers to claim work from this node's queue for split-brain resilience.

        Session 17.43 (Jan 6, 2026): Added for Phase 4 of split-brain fix.

        Unlike regular /work/claim, this endpoint does NOT require leader status.
        Any node with an active work queue can serve peer claims.

        This enables decentralized work distribution during:
        - Split-brain scenarios where leader is unreachable
        - Network partitions where direct leader access fails
        - Leader transitions where new leader not yet elected

        POST body:
        {
            "node_id": "worker-1",          # Node claiming work (required)
            "capabilities": ["selfplay"],   # Work types to claim (optional)
            "source_node": "peer-2"         # Requesting peer for audit (optional)
        }

        Response:
        {
            "status": "claimed" | "no_work" | "no_queue",
            "work": {...} | null,
            "served_by": "this-node-id"
        }
        """
        try:
            wq = get_work_queue()
            if wq is None:
                # No work queue on this node - not a problem, caller tries next peer
                return self.json_response({
                    "status": "no_queue",
                    "work": None,
                    "served_by": getattr(self, "node_id", "unknown"),
                })

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            node_id = data.get("node_id", "")
            capabilities = data.get("capabilities", ["selfplay"])
            source_node = data.get("source_node", "")

            if not node_id:
                return self.bad_request("node_id required")

            # Log peer claim for debugging split-brain scenarios
            logger.debug(
                f"[PeerClaim] Peer claim request from {source_node or 'unknown'} "
                f"for node {node_id}, capabilities={capabilities}"
            )

            # Claim work from our local queue
            # Note: We serve claims regardless of our leadership status
            work = wq.claim_next(node_id, capabilities)

            if work:
                logger.info(
                    f"[PeerClaim] Served work {work.get('work_id', 'unknown')} "
                    f"to {node_id} (via peer claim from {source_node or 'direct'})"
                )
                return self.json_response({
                    "status": "claimed",
                    "work": work,
                    "served_by": getattr(self, "node_id", "unknown"),
                })

            return self.json_response({
                "status": "no_work",
                "work": None,
                "served_by": getattr(self, "node_id", "unknown"),
            })

        except Exception as e:
            logger.error(f"Error in peer work claim: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_start(self, request: web.Request) -> web.Response:
        """Mark work as started (running).

        Session 17.45 (Jan 27, 2026): Non-leaders now forward to leader instead
        of returning 403. This enables workers to report work start through
        any P2P node, improving resilience during network partitions.
        """
        try:
            # Parse body early so we can forward it if not leader
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            if not self.is_leader:
                # Forward to leader instead of returning 403
                response = await self._forward_to_leader("/work/start", data)
                if response is not None:
                    return response
                # Forwarding failed - fall back to 403
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_complete(self, request: web.Request) -> web.Response:
        """Mark work as completed successfully.

        Session 17.45 (Jan 27, 2026): Non-leaders now forward to leader instead
        of returning 403. This enables workers to report completions through
        any P2P node, improving resilience during network partitions.
        """
        try:
            from app.coordination.work_queue import WorkType

            # Parse body early so we can forward it if not leader
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            if not self.is_leader:
                # Forward to leader instead of returning 403
                response = await self._forward_to_leader("/work/complete", data)
                if response is not None:
                    return response
                # Forwarding failed - fall back to 403
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

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
            # Jan 5, 2026 (Session 17.41): Use helper to check loop's populator
            populator = self._get_queue_populator()
            if success and populator is not None:
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
                        populator.update_target_elo(
                            board_type, num_players, elo, model_id
                        )
                        logger.info(
                            f"Updated populator Elo: {board_type}_{num_players}p = {elo}"
                        )

                elif work_type == WorkType.SELFPLAY:
                    # Selfplay increments games count
                    games = result.get("games_generated", config.get("games", 0))
                    if games and board_type and num_players:
                        populator.increment_games(
                            board_type, num_players, games
                        )

                elif work_type == WorkType.TRAINING:
                    # Training increments training runs
                    if board_type and num_players:
                        populator.increment_training(board_type, num_players)

            # Feb 2026: Emit EVALUATION_COMPLETED for gauntlet work so the
            # auto-promotion pipeline can process results from GPU nodes.
            # Previously gauntlet results were silently dropped because
            # handle_work_complete had no handler for WorkType.GAUNTLET.
            if success and work_type == WorkType.GAUNTLET:
                config_key = f"{config.get('board_type', '')}_{config.get('num_players', 0)}p"
                model_path = config.get("candidate_model", "")
                try:
                    from app.coordination.event_emission_helpers import safe_emit_event
                    safe_emit_event(
                        "EVALUATION_COMPLETED",
                        {
                            "config_key": config_key,
                            "model_path": model_path,
                            "board_type": config.get("board_type", ""),
                            "num_players": config.get("num_players", 0),
                            "success": True,
                            "win_rates": result.get("win_rates", {}),
                            "opponent_results": result.get("opponent_results", {}),
                            "elo": result.get("elo") or result.get("estimated_elo") or result.get("best_elo"),
                            "estimated_elo": result.get("estimated_elo") or result.get("elo") or result.get("best_elo"),
                            "games_played": result.get("games_played", result.get("total_games", 0)),
                            "vs_random_rate": result.get("vs_random_rate"),
                            "vs_heuristic_rate": result.get("vs_heuristic_rate"),
                            "work_id": work_id,
                            "evaluated_by": assigned_to,
                            "source": "distributed_gauntlet",
                        },
                        context="work_queue_gauntlet_complete",
                    )
                    logger.info(
                        f"Emitted EVALUATION_COMPLETED for gauntlet {work_id}: "
                        f"{config_key} model={model_path}"
                    )
                except ImportError:
                    logger.debug("Event emission not available for gauntlet completion")

            return self.json_response({
                "status": "completed" if success else "failed",
                "work_id": work_id,
            })
        except Exception as e:
            logger.error(f"Error completing work: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_fail(self, request: web.Request) -> web.Response:
        """Mark work as failed (may retry based on attempts).

        Session 17.45 (Jan 27, 2026): Non-leaders now forward to leader instead
        of returning 403. This enables workers to report failures through
        any P2P node, improving resilience during network partitions.
        """
        try:
            # Parse body early so we can forward it if not leader
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            if not self.is_leader:
                # Forward to leader instead of returning 403
                response = await self._forward_to_leader("/work/fail", data)
                if response is not None:
                    return response
                # Forwarding failed - fall back to 403
                return self._not_leader_response()

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_work_metrics(self, request: web.Request) -> web.Response:
        """Get detailed work queue metrics broken down by config and work type.

        January 2026 - Phase 3.3 (Automation Hardening):
        Provides real-time queue depth and throughput metrics for monitoring
        dashboards and automation systems.

        Query params:
            config: Optional filter for specific config (e.g., "hex8_2p")
            work_type: Optional filter for work type (e.g., "selfplay")
            include_history: If "true", include completion/failure rates (slower)

        Response:
        {
            "total_pending": 150,
            "total_running": 12,
            "by_config": {
                "hex8_2p": {"pending": 50, "running": 3, "completed_1h": 45},
                "square8_4p": {"pending": 20, "running": 2, "completed_1h": 18},
                ...
            },
            "by_work_type": {
                "selfplay": {"pending": 100, "running": 8},
                "training": {"pending": 5, "running": 2},
                ...
            },
            "throughput": {
                "completions_1h": 180,
                "failures_1h": 5,
                "avg_duration_seconds": 145.3
            },
            "backpressure": {...}
        }
        """
        try:
            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            # Get query params
            config_filter = request.query.get("config", None)
            work_type_filter = request.query.get("work_type", None)
            include_history = request.query.get("include_history", "").lower() == "true"

            # Build metrics
            metrics = {
                "total_pending": 0,
                "total_running": 0,
                "total_claimed": 0,
                "by_config": {},
                "by_work_type": {},
                "is_leader": self.is_leader,
                "leader_id": self.leader_id,
            }

            # Aggregate from work queue items
            with wq.lock:
                for work_id, item in wq.items.items():
                    # Extract config key
                    config = item.config or {}
                    board_type = config.get("board_type", "unknown")
                    num_players = config.get("num_players", 0)
                    config_key = f"{board_type}_{num_players}p" if board_type != "unknown" else "unknown"

                    # Apply filters
                    if config_filter and config_key != config_filter:
                        continue
                    if work_type_filter and item.work_type.value != work_type_filter:
                        continue

                    # Count by status
                    status = item.status.value if hasattr(item.status, "value") else str(item.status)
                    work_type = item.work_type.value if hasattr(item.work_type, "value") else str(item.work_type)

                    # Update totals
                    if status == "pending":
                        metrics["total_pending"] += 1
                    elif status == "running":
                        metrics["total_running"] += 1
                    elif status == "claimed":
                        metrics["total_claimed"] += 1

                    # Update by_config
                    if config_key not in metrics["by_config"]:
                        metrics["by_config"][config_key] = {
                            "pending": 0,
                            "running": 0,
                            "claimed": 0,
                        }
                    if status in ("pending", "running", "claimed"):
                        metrics["by_config"][config_key][status] += 1

                    # Update by_work_type
                    if work_type not in metrics["by_work_type"]:
                        metrics["by_work_type"][work_type] = {
                            "pending": 0,
                            "running": 0,
                            "claimed": 0,
                        }
                    if status in ("pending", "running", "claimed"):
                        metrics["by_work_type"][work_type][status] += 1

            # Include history-based metrics if requested (slower, hits DB)
            if include_history:
                try:
                    import time
                    one_hour_ago = time.time() - 3600

                    # Get recent completions/failures from history
                    history = wq.get_history(limit=500, status_filter=None)
                    completions_1h = 0
                    failures_1h = 0
                    total_duration = 0.0
                    duration_count = 0

                    for item in history:
                        completed_at = item.get("completed_at", 0)
                        if completed_at and completed_at >= one_hour_ago:
                            if item.get("status") == "completed":
                                completions_1h += 1
                                duration = item.get("duration_seconds", 0)
                                if duration:
                                    total_duration += duration
                                    duration_count += 1
                            elif item.get("status") == "failed":
                                failures_1h += 1

                    metrics["throughput"] = {
                        "completions_1h": completions_1h,
                        "failures_1h": failures_1h,
                        "avg_duration_seconds": round(total_duration / duration_count, 1) if duration_count > 0 else 0,
                    }
                except Exception as e:
                    logger.debug(f"Error getting history metrics: {e}")
                    metrics["throughput"] = {"error": str(e)}

            # Add backpressure status
            metrics["backpressure"] = wq.get_backpressure_status()

            return self.json_response(metrics)
        except Exception as e:
            logger.error(f"Error getting work metrics: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_populator_status(self, request: web.Request) -> web.Response:
        """Get queue populator status for monitoring."""
        try:
            # Jan 5, 2026 (Session 17.41): Use helper to check loop's populator
            populator = self._get_queue_populator()
            if populator is None:
                return self.json_response({
                    "enabled": False,
                    "message": "Queue populator not initialized",
                })

            status = populator.get_status()
            status["is_leader"] = self.is_leader
            return self.json_response(status)
        except Exception as e:
            logger.error(f"Error getting populator status: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_dispatch_stats(self, request: web.Request) -> web.Response:
        """Get dispatch/claim rejection statistics for debugging job dispatch issues.

        Jan 2, 2026: Added to diagnose why GPU nodes are idle despite jobs
        being queued. Returns breakdown of why claim_work() is rejecting jobs.

        Query params:
            reset: If "true", reset stats after returning current values
        """
        try:
            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            # Get claim rejection stats
            rejection_stats = wq.get_claim_rejection_stats()

            # Get queue stats for context
            queue_stats = wq.get_queue_stats()

            # Optionally reset stats
            if request.query.get("reset", "").lower() == "true":
                wq.reset_claim_rejection_stats()

            return self.json_response({
                "claim_rejection_stats": rejection_stats,
                "queue_stats": {
                    "pending": queue_stats.get("pending", 0),
                    "claimed": queue_stats.get("claimed", 0),
                    "running": queue_stats.get("running", 0),
                    "completed": queue_stats.get("completed", 0),
                    "failed": queue_stats.get("failed", 0),
                    "total_items": queue_stats.get("total_items", 0),
                },
                "is_leader": self.is_leader,
                "leader_id": self.leader_id,
            })
        except Exception as e:
            logger.error(f"Error getting dispatch stats: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_clear_stale_targets(self, request: web.Request) -> web.Response:
        """Clear target_node from jobs targeted at non-existent nodes.

        Jan 2, 2026: Added to fix jobs stuck on old/renamed node targets.

        POST /queue/clear-stale-targets

        This endpoint clears target_node assignments for pending jobs where
        the target node no longer exists in the cluster. This allows any
        available node to claim those jobs.

        Returns:
            JSON with count of cleared jobs
        """
        try:
            if not self.is_leader:
                return self.error_response(
                    "Only leader can clear stale targets",
                    status=403
                )

            wq = get_work_queue()
            if wq is None:
                return self._work_queue_unavailable()

            # Get valid node IDs from peer manager
            valid_node_ids = set()
            if hasattr(self, 'peer_manager') and self.peer_manager:
                peers = self.peer_manager.get_all_peers()
                valid_node_ids = {p.get('node_id') for p in peers if p.get('node_id')}
            elif hasattr(self, 'all_peers') and self.all_peers:
                valid_node_ids = set(self.all_peers.keys())

            # Add self
            if hasattr(self, 'node_id'):
                valid_node_ids.add(self.node_id)

            if not valid_node_ids:
                return self.error_response(
                    "Could not determine valid node IDs",
                    status=500
                )

            # Clear stale targets
            cleared_count = wq.clear_stale_target_nodes(valid_node_ids)

            return self.json_response({
                "cleared_count": cleared_count,
                "valid_nodes": len(valid_node_ids),
                "is_leader": self.is_leader,
            })

        except Exception as e:
            logger.error(f"Error clearing stale targets: {e}")
            return self.error_response(str(e), status=500)
