"""Model Validation Loop for P2P Orchestrator.

December 2025: Background loop for automatic model validation scheduling.

Loops:
- ValidationLoop: Finds newly trained models and queues validation work items

Usage:
    from scripts.p2p.loops import ValidationLoop

    validation = ValidationLoop(
        is_leader=lambda: orchestrator.role == NodeRole.LEADER,
        get_model_registry=lambda: ModelRegistry(),
        get_work_queue=lambda: work_queue,
        send_notification=orchestrator.notifier.send,
    )
    await validation.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation loop."""

    check_interval_seconds: float = 300.0  # 5 minutes
    initial_delay_seconds: float = 120.0  # 2 minutes
    max_models_per_cycle: int = 3
    max_unvalidated_per_cycle: int = 2
    default_baselines: tuple[str, ...] = ("mcts_500",)
    default_games_per_matchup: int = 50
    validation_priority: int = 80


class ValidationLoop(BaseLoop):
    """Background loop for automatic model validation scheduling.

    Finds newly trained models that need validation and queues validation
    work items for them. Only runs on the leader node.

    Features:
    - Scans model registry for pending validations
    - Creates validation work items in the work queue
    - Creates validation entries for unvalidated models
    - Sends notifications when validation is queued
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_model_registry: Callable[[], Any | None],
        get_work_queue: Callable[[], Any | None],
        send_notification: Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        config: ValidationConfig | None = None,
    ):
        """Initialize validation loop.

        Args:
            is_leader: Callback returning True if this node is the leader
            get_model_registry: Callback returning ModelRegistry or None
            get_work_queue: Callback returning WorkQueue or None
            send_notification: Optional async callback to send notifications
                               (message, severity, context)
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        super().__init__(
            name="validation",
            interval=self.config.check_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_model_registry = get_model_registry
        self._get_work_queue = get_work_queue
        self._send_notification = send_notification
        self._first_run = True

        # Statistics
        self._validations_queued = 0
        self._validation_entries_created = 0

    async def _on_start(self) -> None:
        """Initial delay before first validation check."""
        if self._first_run:
            self._first_run = False
            logger.info("Validation loop: initial delay before first check")
            await asyncio.sleep(self.config.initial_delay_seconds)
        logger.info("Validation loop started")

    async def _run_once(self) -> None:
        """Check for models needing validation and queue work items."""
        # Only leader performs validation scheduling
        if not self._is_leader():
            return

        # Get model registry
        registry = self._get_model_registry()
        if registry is None:
            logger.debug("Validation loop: registry not available")
            return

        # Get work queue
        work_queue = self._get_work_queue()
        if work_queue is None:
            logger.debug("Validation loop: work queue not available")
            return

        # Find models needing validation
        try:
            models_pending = registry.get_models_needing_validation()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Validation loop: failed to get pending models: {e}")
            return

        logger.debug(f"Found {len(models_pending)} models pending validation")

        # Process pending validations
        for model_info in models_pending[:self.config.max_models_per_cycle]:
            await self._queue_validation(registry, work_queue, model_info)

        # Also check for models without validation entries
        try:
            unvalidated = registry.get_unvalidated_models()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Validation loop: failed to get unvalidated models: {e}")
            return

        for model_info in unvalidated[:self.config.max_unvalidated_per_cycle]:
            model_id = model_info["model_id"]
            version = model_info["version"]
            try:
                registry.create_validation(model_id, version)
                self._validation_entries_created += 1
                logger.info(f"Created validation entry for {model_id}:v{version}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to create validation entry for {model_id}:v{version}: {e}")

    async def _queue_validation(
        self,
        registry: Any,
        work_queue: Any,
        model_info: dict[str, Any],
    ) -> None:
        """Queue a validation work item for a model.

        Args:
            registry: The model registry instance
            work_queue: The work queue instance
            model_info: Model information dict with model_id, version, etc.
        """
        model_id = model_info["model_id"]
        version = model_info["version"]
        baselines = model_info.get("baselines") or list(self.config.default_baselines)
        games_per = model_info.get("games_per_matchup", self.config.default_games_per_matchup)

        # Create validation work item
        try:
            from app.coordination.work_queue import WorkItem, WorkType
        except ImportError:
            logger.debug("Validation loop: WorkItem/WorkType not available")
            return

        work_id = f"validation_{model_id}_v{version}_{int(time.time())}"

        work_item = WorkItem(
            work_id=work_id,
            work_type=WorkType.VALIDATION,
            priority=self.config.validation_priority,
            config={
                "model_id": model_id,
                "version": version,
                "baselines": baselines,
                "games_per_matchup": games_per,
                "file_path": model_info.get("file_path"),
            },
        )

        try:
            work_queue.add_work(work_item)
            registry.set_validation_queued(model_id, version, work_id)
            self._validations_queued += 1
            logger.info(f"Queued validation for {model_id}:v{version} ({work_id})")

            # Send notification if callback provided
            if self._send_notification:
                try:
                    await self._send_notification(
                        f"🔬 Model validation queued: {model_id}:v{version}",
                        "info",
                        {"baselines": baselines, "games_per_matchup": games_per},
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Failed to send validation notification: {e}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to queue validation for {model_id}:v{version}: {e}")

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics.

        Returns:
            Dict with validation stats and base loop stats.
        """
        return {
            "validations_queued": self._validations_queued,
            "validation_entries_created": self._validation_entries_created,
            **self.stats.to_dict(),
        }


__all__ = [
    "ValidationConfig",
    "ValidationLoop",
]
