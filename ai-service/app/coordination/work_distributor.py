"""Work Distribution Adapter for Cluster-Wide Task Coordination.

This module bridges training/evaluation pipelines to the centralized work queue,
enabling distributed execution across 50+ cluster nodes.

Integration Points:
- Training requests → WorkQueue (WorkType.TRAINING)
- Evaluation requests → WorkQueue (WorkType.TOURNAMENT/GAUNTLET)
- CMAES optimization → WorkQueue (WorkType.GPU_CMAES/CPU_CMAES)
- Model sync → WorkQueue (WorkType.DATA_SYNC)

Event Integration:
- Emits WORK_SUBMITTED when work is queued
- Emits WORK_CLAIMED when work is assigned
- Subscribes to TRAINING_THRESHOLD_REACHED to auto-queue training

Usage:
    from app.coordination.work_distributor import get_work_distributor

    distributor = get_work_distributor()

    # Submit training work
    work_id = await distributor.submit_training(
        board="square8",
        num_players=2,
        epochs=100,
        priority=80,
    )

    # Submit evaluation work
    work_id = await distributor.submit_evaluation(
        candidate_model="nnue_v7",
        baseline_model="nnue_v6",
        games=200,
    )

    # Get status
    status = distributor.get_work_status(work_id)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Jan 29, 2026: Leader forwarding for work items
# When the coordinator (which creates work) is not the leader (which serves claims),
# work items must be forwarded to the leader via HTTP POST /work/add.
_leader_forward_session = None

# Lazy imports to avoid circular dependencies
_work_queue = None
_WorkItem = None
_WorkType = None
_WorkStatus = None


def _get_work_queue():
    """Lazy load WorkQueue using the singleton to avoid database path inconsistencies."""
    global _work_queue, _WorkItem, _WorkType, _WorkStatus

    if _work_queue is None:
        try:
            from app.coordination.work_queue import (
                WorkItem,
                WorkStatus,
                WorkType,
                get_work_queue,
            )

            _WorkItem = WorkItem
            _WorkType = WorkType
            _WorkStatus = WorkStatus

            # Use the singleton instance to ensure consistent database path
            # The singleton respects RINGRIFT_WORK_QUEUE_DB environment variable
            _work_queue = get_work_queue()
            logger.info("Work distributor connected to singleton work queue")

        except ImportError as e:
            logger.warning(f"WorkQueue not available: {e}")
            return None

    return _work_queue


@dataclass
class DistributedWorkConfig:
    """Configuration for distributed work submission."""

    # Node selection
    require_gpu: bool = False
    require_high_memory: bool = False  # For square19/hexagonal
    preferred_nodes: list[str] | None = None

    # Scheduling
    priority: int = 50  # 0-100, higher = more urgent
    timeout_seconds: float = 3600.0  # 1 hour default
    max_attempts: int = 3

    # Dependencies
    depends_on: list[str] | None = None


class WorkDistributor:
    """Distributes work across the cluster via the central work queue."""

    def __init__(self):
        self._queue = None
        self._local_submissions: dict[str, dict[str, Any]] = {}
        self._event_callbacks: list[callable] = []

    def _ensure_queue(self):
        """Ensure work queue is available."""
        if self._queue is None:
            self._queue = _get_work_queue()
        return self._queue is not None

    async def _forward_to_leader(
        self,
        work_type: str,
        priority: int,
        config: dict[str, Any],
        timeout_seconds: float = 3600.0,
        depends_on: list[str] | None = None,
    ) -> str | None:
        """Forward a work item to the leader's work queue via HTTP.

        Jan 29, 2026: Work items are stored locally AND forwarded to the leader.
        This ensures gauntlet/training/selfplay jobs are available for claiming
        even when the coordinator is not the leader.

        Returns:
            Work ID from the leader, or None if forwarding failed.
        """
        try:
            import aiohttp

            # Get leader URL from P2P orchestrator
            leader_url = self._get_leader_url()
            if not leader_url:
                logger.debug("[WorkDistributor] No leader URL, skipping forward")
                return None

            payload = {
                "work_type": work_type,
                "priority": priority,
                "config": config,
                "timeout_seconds": timeout_seconds,
                "depends_on": depends_on or [],
                "force": True,  # Bypass backpressure since coordinator already validated
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{leader_url}/work/add"
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        remote_id = data.get("work_id")
                        logger.info(
                            f"[WorkDistributor] Forwarded {work_type} to leader: {remote_id}"
                        )
                        return remote_id
                    elif resp.status == 403:
                        # Leader changed or not leader anymore
                        logger.debug("[WorkDistributor] Leader rejected forward (403)")
                    elif resp.status == 429:
                        logger.debug("[WorkDistributor] Leader backpressure (429)")
                    else:
                        body = await resp.text()
                        logger.warning(
                            f"[WorkDistributor] Forward failed ({resp.status}): {body[:200]}"
                        )
        except ImportError:
            logger.debug("[WorkDistributor] aiohttp not available for leader forwarding")
        except Exception as e:
            logger.debug(f"[WorkDistributor] Leader forward failed: {e}")

        return None

    def _get_leader_url(self) -> str | None:
        """Get the leader's HTTP base URL from P2P state.

        Jan 29, 2026: Used to forward work items to the leader node.
        Checks local P2P status endpoint for leader info, and skips
        forwarding if this node IS the leader.
        """
        try:
            # Check environment override first
            leader_url = os.environ.get("RINGRIFT_LEADER_URL")
            if leader_url:
                return leader_url
        except Exception:
            pass

        try:
            # Read from local P2P status endpoint
            import urllib.request
            import json

            req = urllib.request.Request(
                "http://localhost:8770/status",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())

                # Don't forward if we ARE the leader
                role = data.get("role", "")
                if role == "leader":
                    return None

                leader_id = data.get("leader_id")
                if not leader_id:
                    return None

                # Get leader's address from peers
                peers = data.get("peers", {})
                if leader_id in peers:
                    peer = peers[leader_id]
                    if isinstance(peer, dict):
                        ip = peer.get("address", peer.get("ip", peer.get("tailscale_ip", "")))
                        port = peer.get("port", 8770)
                        if ip:
                            return f"http://{ip}:{port}"

                # Fallback: try leader_id as hostname
                return f"http://{leader_id}:8770"
        except Exception:
            pass

        return None

    # =========================================================================
    # Training Submission
    # =========================================================================

    async def submit_training(
        self,
        board: str,
        num_players: int,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        checkpoint_path: str | None = None,
        db_paths: list[str] | None = None,
        config: DistributedWorkConfig | None = None,
        model_version: str = "v5",
    ) -> str | None:
        """Submit a training job to the cluster work queue.

        Args:
            board: Board type (square8, square19, hexagonal).
            num_players: Number of players (2, 3, 4).
            epochs: Training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            checkpoint_path: Optional path to resume from.
            db_paths: Optional list of database paths.
            config: Distributed work configuration.
            model_version: Neural network architecture version (v2, v3, v4, v5, v5-heavy-large).

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            logger.warning("Work queue not available, cannot submit training")
            return None

        config = config or DistributedWorkConfig()

        # Determine priority based on data availability
        priority = config.priority

        # Higher priority for underrepresented configs
        if board in ("square19", "hexagonal"):
            priority = min(100, priority + 20)
        if num_players in (3, 4):
            priority = min(100, priority + 10)

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "checkpoint_path": checkpoint_path,
            "db_paths": db_paths or [],
            "require_gpu": config.require_gpu,
            "require_high_memory": config.require_high_memory or board != "square8",
            "model_version": model_version,
        }

        item = _WorkItem(
            work_type=_WorkType.TRAINING,
            priority=priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            depends_on=config.depends_on or [],
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted training work {work_id}: {board}_{num_players}p")

        # Track locally
        self._local_submissions[work_id] = {
            "type": "training",
            "submitted_at": time.time(),
            "config": work_config,
        }

        # Jan 29, 2026: Forward to leader so GPU nodes can claim it
        await self._forward_to_leader(
            work_type="training",
            priority=priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
        )

        # Emit event
        await self._emit_work_submitted(work_id, "training", work_config)

        return work_id

    # =========================================================================
    # Evaluation Submission
    # =========================================================================

    async def submit_evaluation(
        self,
        candidate_model: str,
        baseline_model: str | None = None,
        games: int = 200,
        board: str = "square8",
        num_players: int = 2,
        evaluation_type: str = "gauntlet",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit an evaluation job to the cluster work queue.

        Args:
            candidate_model: Path/ID of candidate model.
            baseline_model: Path/ID of baseline model (optional).
            games: Number of games to play.
            board: Board type.
            num_players: Number of players.
            evaluation_type: 'gauntlet' or 'tournament'.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            logger.warning("Work queue not available, cannot submit evaluation")
            return None

        config = config or DistributedWorkConfig()

        work_type = (
            _WorkType.GAUNTLET if evaluation_type == "gauntlet"
            else _WorkType.TOURNAMENT
        )

        work_config = {
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "games": games,
            "board_type": board,
            "num_players": num_players,
            # Jan 28, 2026: Gauntlet/evaluation requires GPU - prevents coordinator from claiming
            "requires_gpu": config.require_gpu if config.require_gpu else True,
        }

        # Feb 1, 2026: Use graduated timeouts for evaluation based on board size.
        # Large boards (hexagonal, square19) take much longer per game.
        _board_timeouts = {
            "hex8": 3600, "square8": 7200, "square19": 10800, "hexagonal": 14400,
        }
        base_timeout = _board_timeouts.get(board, config.timeout_seconds)
        player_mult = {2: 1.0, 3: 1.5, 4: 2.0}.get(num_players, 1.0)
        eval_timeout = base_timeout * player_mult

        item = _WorkItem(
            work_type=work_type,
            priority=config.priority,
            config=work_config,
            timeout_seconds=eval_timeout,
            max_attempts=config.max_attempts,
            depends_on=config.depends_on or [],
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted {evaluation_type} work {work_id}")

        self._local_submissions[work_id] = {
            "type": evaluation_type,
            "submitted_at": time.time(),
            "config": work_config,
        }

        await self._emit_work_submitted(work_id, evaluation_type, work_config)

        # Jan 29, 2026: Forward to leader so GPU nodes can claim it
        await self._forward_to_leader(
            work_type=work_type.value if hasattr(work_type, "value") else str(work_type),
            priority=config.priority,
            config=work_config,
            timeout_seconds=eval_timeout,
        )

        return work_id

    # =========================================================================
    # CMAES Optimization Submission
    # =========================================================================

    async def submit_cmaes(
        self,
        board: str,
        num_players: int,
        generations: int = 50,
        population_size: int = 20,
        use_gpu: bool = True,
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a CMAES optimization job.

        Args:
            board: Board type.
            num_players: Number of players.
            generations: Number of generations.
            population_size: Population size.
            use_gpu: Whether to use GPU acceleration.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_type = _WorkType.GPU_CMAES if use_gpu else _WorkType.CPU_CMAES

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "generations": generations,
            "population_size": population_size,
        }

        item = _WorkItem(
            work_type=work_type,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted CMAES work {work_id}")

        return work_id

    # =========================================================================
    # Selfplay Submission
    # =========================================================================

    async def submit_selfplay(
        self,
        board: str,
        num_players: int,
        games: int = 1000,
        ai_type: str = "gumbel-mcts",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a selfplay job.

        Args:
            board: Board type.
            num_players: Number of players.
            games: Number of games to generate.
            ai_type: AI type to use.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "games": games,
            "ai_type": ai_type,
        }

        item = _WorkItem(
            work_type=_WorkType.SELFPLAY,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted selfplay work {work_id}: {board}_{num_players}p, {games} games")

        return work_id

    # =========================================================================
    # Data Sync Submission
    # =========================================================================

    async def submit_data_sync(
        self,
        source_path: str,
        target_nodes: list[str] | None = None,
        sync_type: str = "model",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a data sync job to distribute data across cluster.

        Args:
            source_path: Path to data to sync.
            target_nodes: Specific nodes to sync to (None = all).
            sync_type: Type of sync ('model', 'database', 'checkpoint').
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_config = {
            "source_path": source_path,
            "target_nodes": target_nodes,
            "sync_type": sync_type,
        }

        item = _WorkItem(
            work_type=_WorkType.DATA_SYNC,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted data sync work {work_id}: {sync_type}")

        return work_id

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_work_status(self, work_id: str) -> dict[str, Any] | None:
        """Get status of a submitted work item."""
        if not self._ensure_queue():
            return None

        item = self._queue.get_work_item(work_id)
        if item is None:
            return None

        return {
            "work_id": item.work_id,
            "status": item.status.value,
            "work_type": item.work_type.value,
            "priority": item.priority,
            "claimed_by": item.claimed_by,
            "attempts": item.attempts,
            "created_at": item.created_at,
            "result": item.result,
            "error": item.error,
        }

    def get_queue_stats(self) -> dict[str, Any]:
        """Get overall queue statistics."""
        if not self._ensure_queue():
            return {"available": False}

        # Use get_queue_status which exists on WorkQueue
        return self._queue.get_queue_status()

    def get_pending_work(
        self,
        work_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get pending work items."""
        if not self._ensure_queue():
            return []

        items = self._queue.get_pending(limit=limit)

        if work_type:
            items = [i for i in items if i.work_type.value == work_type]

        return [i.to_dict() for i in items]

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def submit_multiconfig_training(
        self,
        configs: list[tuple[str, int]],  # List of (board, num_players)
        epochs: int = 100,
        base_priority: int = 50,
    ) -> list[str]:
        """Submit training for multiple configurations.

        Args:
            configs: List of (board, num_players) tuples.
            epochs: Training epochs.
            base_priority: Base priority (adjusted per config).

        Returns:
            List of work IDs.
        """
        work_ids = []

        for board, num_players in configs:
            work_id = await self.submit_training(
                board=board,
                num_players=num_players,
                epochs=epochs,
                config=DistributedWorkConfig(priority=base_priority),
            )
            if work_id:
                work_ids.append(work_id)

        return work_ids

    async def submit_crossboard_evaluation(
        self,
        candidate_model: str,
        games_per_config: int = 200,
    ) -> list[str]:
        """Submit evaluations for all 9 board/player configurations.

        Args:
            candidate_model: Model to evaluate.
            games_per_config: Games per configuration.

        Returns:
            List of work IDs.
        """
        from app.training.crossboard_strength import ALL_BOARD_CONFIGS

        work_ids = []

        for board, num_players in ALL_BOARD_CONFIGS:
            work_id = await self.submit_evaluation(
                candidate_model=candidate_model,
                games=games_per_config,
                board=board,
                num_players=num_players,
            )
            if work_id:
                work_ids.append(work_id)

        return work_ids

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> "HealthCheckResult":
        """Check health of the work distributor.

        Returns:
            HealthCheckResult indicating distributor health status.
        """
        # Import from contracts (zero-dependency module)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        # Check if queue is available
        queue_available = self._ensure_queue()
        if not queue_available:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="Work queue not available",
                details={"queue_available": False},
            )

        # Get queue stats
        stats = self.get_queue_stats()
        if not stats.get("available", True):
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="Work queue stats unavailable",
                details=stats,
            )

        # Check for warning conditions
        warnings = []
        # get_queue_status returns by_status dict with counts per status
        by_status = stats.get("by_status", {})
        pending_count = by_status.get("pending", 0)
        failed_count = by_status.get("failed", 0)
        total_items = stats.get("total_items", 0)

        if pending_count > 500:
            warnings.append(f"High pending count: {pending_count}")

        if failed_count > 50:
            warnings.append(f"High failure count: {failed_count}")

        is_healthy = len(warnings) == 0
        status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message="; ".join(warnings) if warnings else "Distributor healthy",
            details={
                "queue_available": True,
                "total_items": total_items,
                "local_submissions": len(self._local_submissions),
                "pending_work": pending_count,
                "failed_work": failed_count,
            },
        )

    # =========================================================================
    # Event Integration
    # =========================================================================

    async def _emit_work_submitted(
        self,
        work_id: str,
        work_type: str,
        config: dict[str, Any],
    ) -> None:
        """Emit event when work is submitted."""
        try:
            from app.distributed.event_helpers import emit_event_safe

            await emit_event_safe(
                "WORK_SUBMITTED",
                {
                    "work_id": work_id,
                    "work_type": work_type,
                    "config": config,
                },
                "work_distributor"
            )
        except Exception as e:
            logger.debug(f"Could not emit work submitted event: {e}")


# Global instance
_distributor_instance: WorkDistributor | None = None


def get_work_distributor() -> WorkDistributor:
    """Get the global WorkDistributor instance."""
    global _distributor_instance
    if _distributor_instance is None:
        _distributor_instance = WorkDistributor()
    return _distributor_instance


# Convenience functions for common operations
async def distribute_training(
    board: str,
    num_players: int,
    **kwargs,
) -> str | None:
    """Convenience function to distribute training work."""
    return await get_work_distributor().submit_training(
        board=board,
        num_players=num_players,
        **kwargs,
    )


async def distribute_evaluation(
    candidate_model: str,
    **kwargs,
) -> str | None:
    """Convenience function to distribute evaluation work."""
    return await get_work_distributor().submit_evaluation(
        candidate_model=candidate_model,
        **kwargs,
    )


async def distribute_selfplay(
    board: str,
    num_players: int,
    games: int = 1000,
    **kwargs,
) -> str | None:
    """Convenience function to distribute selfplay work."""
    return await get_work_distributor().submit_selfplay(
        board=board,
        num_players=num_players,
        games=games,
        **kwargs,
    )
