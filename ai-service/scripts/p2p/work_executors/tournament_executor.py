"""Tournament work executor - handles distributed tournament evaluation.

Extracted from P2POrchestrator._execute_claimed_work (Feb 2026).
"""

from __future__ import annotations

import asyncio

from app.core.async_context import safe_create_task
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.managers.job_manager import JobManager
    from scripts.p2p.models import DistributedTournamentState

logger = logging.getLogger("p2p_orchestrator")


async def execute_tournament_work(
    work_item: dict[str, Any],
    config: dict[str, Any],
    peers: dict,
    peers_lock: threading.RLock,
    distributed_tournament_state: dict[str, "DistributedTournamentState"],
    job_manager: "JobManager",
) -> bool:
    """Execute a tournament work item.

    Args:
        work_item: Full work item dict.
        config: Work config sub-dict (board_type, num_players, etc.).
        peers: Dict of peer NodeInfo objects.
        peers_lock: Lock for thread-safe peer access.
        distributed_tournament_state: Shared tournament state tracking dict.
        job_manager: JobManager for running distributed tournaments.

    Returns:
        True if tournament started, False if insufficient agents.
    """
    from scripts.p2p.models import DistributedTournamentState as DTS

    work_id = work_item.get("work_id", "")
    board_type = config.get("board_type", "square8")
    num_players = config.get("num_players", 2)
    games_per_pairing = config.get("games", 2)
    job_id = f"tournament-{work_id}"

    # Discover models for this config
    config_key = f"{board_type}_{num_players}p"
    agent_ids: list[str] = []
    try:
        from app.models.discovery import discover_models
        models = discover_models(
            board_type=board_type,
            num_players=num_players,
            model_type="nn",
        )
        models.sort(key=lambda m: m.modified_at or 0, reverse=True)
        agent_ids = [str(m.path) for m in models[:5]] if models else []
    except (ImportError, ValueError, AttributeError, RuntimeError):
        pass

    # Fallback: use canonical model if available
    if len(agent_ids) < 2:
        canonical_path = f"models/canonical_{config_key}.pth"
        if Path(canonical_path).exists():
            agent_ids = [canonical_path]
        agent_ids.append("heuristic")

    if len(agent_ids) < 2:
        logger.warning(
            f"Tournament {work_id}: Not enough agents for {config_key} "
            f"(found {len(agent_ids)}), skipping"
        )
        return False

    logger.info(
        f"Executing tournament work {work_id}: {board_type}/{num_players}p "
        f"with {len(agent_ids)} agents"
    )

    # Create tournament state (matches /tournament/start handler pattern)
    pairings = []
    for i, a1 in enumerate(agent_ids):
        for a2 in agent_ids[i + 1:]:
            for game_num in range(games_per_pairing):
                pairings.append({
                    "agent1": a1,
                    "agent2": a2,
                    "game_num": game_num,
                    "status": "pending",
                })

    state = DTS(
        job_id=job_id,
        board_type=board_type,
        num_players=num_players,
        agent_ids=agent_ids,
        games_per_pairing=games_per_pairing,
        total_matches=len(pairings),
        pending_matches=pairings,
        status="running",
        started_at=time.time(),
        last_update=time.time(),
    )

    # Find available workers
    with peers_lock:
        workers = [p.node_id for p in peers.values() if p.is_healthy()]
    state.worker_nodes = workers

    # Register state before running
    distributed_tournament_state[job_id] = state

    # Run tournament via job manager
    async def _run_tournament_task():
        try:
            await job_manager.run_distributed_tournament(job_id)
        except Exception as e:
            logger.exception(f"Tournament task failed for {job_id}: {e}")

    safe_create_task(_run_tournament_task(), name=f"tournament-{job_id}")
    return True
