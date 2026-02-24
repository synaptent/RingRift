"""Selfplay work executor - delegates to JobManager for GPU game simulation.

Extracted from P2POrchestrator._execute_claimed_work (Feb 2026).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from scripts.p2p.managers.job_manager import JobManager
    from scripts.p2p.managers.selfplay_scheduler import SelfplayScheduler

logger = logging.getLogger("p2p_orchestrator")


async def execute_selfplay_work(
    work_item: dict[str, Any],
    config: dict[str, Any],
    job_manager: "JobManager",
    selfplay_scheduler: "SelfplayScheduler",
) -> bool:
    """Execute a selfplay work item via JobManager.

    Args:
        work_item: Full work item dict.
        config: Work config sub-dict (board_type, num_players, etc.).
        job_manager: JobManager for spawning selfplay jobs.
        selfplay_scheduler: SelfplayScheduler for tracking diversity.

    Returns:
        True (selfplay is fire-and-forget via create_task).
    """
    work_id = work_item.get("work_id", "")

    # Prevent coordinator from spawning selfplay
    from scripts.p2p.managers.work_discovery_manager import _is_selfplay_enabled_for_node
    if not _is_selfplay_enabled_for_node():
        logger.info(f"Skipping selfplay work {work_id}: selfplay_enabled=false for this node")
        return True  # "handled" (just skipped)

    board_type = config.get("board_type", "square8")
    num_players = config.get("num_players", 2)
    num_games = config.get("num_games", 500)
    engine_mode = config.get("engine_mode", "mixed")
    engine_extra_args = config.get("engine_extra_args")
    selfplay_model_version = config.get("model_version", "v2")

    # Delegate to JobManager (fire-and-forget with error tracking)
    safe_create_task(
        job_manager.run_gpu_selfplay_job(
            job_id=f"pull-{work_id}",
            board_type=board_type,
            num_players=num_players,
            num_games=num_games,
            engine_mode=engine_mode,
            engine_extra_args=engine_extra_args,
            model_version=selfplay_model_version,
        ),
        name=f"selfplay-{work_id}",
        error_callback=lambda t: logger.error(
            f"Selfplay {work_id} failed: {t.exception()}"
        ),
    )

    # Track diversity metrics for monitoring
    selfplay_scheduler.track_diversity({
        "board_type": board_type,
        "num_players": num_players,
        "engine_mode": engine_mode,
    })
    return True
