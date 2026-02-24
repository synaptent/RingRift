"""Gauntlet work executor - handles baseline opponent evaluation.

Extracted from P2POrchestrator._execute_claimed_work (Feb 2026).

Critical: Gauntlet runs synchronously (via asyncio.to_thread) so results
are available when WorkerPullLoop calls _report_result(work_item, success).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("p2p_orchestrator")


async def execute_gauntlet_work(
    work_item: dict[str, Any],
    config: dict[str, Any],
    node_id: str,
    ringrift_path: str | Path,
) -> bool:
    """Execute a gauntlet evaluation work item.

    Runs run_baseline_gauntlet in a thread to avoid blocking the event loop.
    Results are stored in work_item["result"] for the caller to report.

    Args:
        work_item: Full work item dict (modified in-place to add result data).
        config: Work config sub-dict (board_type, num_players, candidate_model, etc.).
        node_id: This node's identifier.
        ringrift_path: Path to ai-service root.

    Returns:
        True on success, False on failure or if gauntlet is disabled.
    """
    work_id = work_item.get("work_id", "")

    from app.config.env import env
    if not env.gauntlet_enabled:
        logger.warning(f"Rejecting gauntlet work {work_id}: gauntlet_enabled=false for this node")
        return False  # Return False so work can be reassigned to a GPU node

    board_type = config.get("board_type", "square8")
    num_players = config.get("num_players", 2)
    model_path = config.get("candidate_model", "")
    games = config.get("games", 100)

    if not model_path:
        logger.warning(f"Gauntlet work {work_id}: No model_path specified")
        return False

    # Check model exists locally
    if not Path(model_path).exists():
        config_key_check = f"{board_type}_{num_players}p"
        candidate_path = f"models/candidate_{config_key_check}.pth"
        canonical_path = f"models/canonical_{config_key_check}.pth"
        if Path(candidate_path).exists():
            model_path = candidate_path
        elif Path(canonical_path).exists():
            model_path = canonical_path
        else:
            logger.warning(f"Gauntlet work {work_id}: Model not found: {model_path}")
            return False

    logger.info(
        f"Executing gauntlet work {work_id}: {model_path} "
        f"({board_type}/{num_players}p, {games} games)"
    )

    config_key = f"{board_type}_{num_players}p"
    try:
        from app.training.game_gauntlet import (
            run_baseline_gauntlet,
            BaselineOpponent,
        )

        # Run gauntlet in thread to avoid blocking event loop
        gauntlet_result = await asyncio.to_thread(
            run_baseline_gauntlet,
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
            games_per_opponent=max(games // 4, 10),
            opponents=[
                BaselineOpponent.RANDOM,
                BaselineOpponent.HEURISTIC,
            ],
            verbose=False,
            early_stopping=True,
            parallel_opponents=True,
        )

        # Build result dict with full win rate data
        total_games = getattr(gauntlet_result, "total_games", 0)
        gauntlet_passed = getattr(gauntlet_result, "passed", False)

        # Feb 2026: Zero-game evaluations are failures regardless of passed flag.
        # Previously hardcoded success=True even when all games failed silently,
        # causing the entire evaluation pipeline to report success with 0 games.
        if total_games == 0:
            gauntlet_passed = False
            logger.warning(
                f"Gauntlet work {work_id}: 0 games completed - treating as failure. "
                f"Reason: {getattr(gauntlet_result, 'failure_reason', 'unknown')}"
            )

        result_data: dict[str, Any] = {
            "config_key": config_key,
            "board_type": board_type,
            "num_players": num_players,
            "model_path": model_path,
            "work_id": work_id,
            "success": total_games > 0,
            "win_rates": {},
            "opponent_results": {},
            "games_played": total_games,
            "total_games": total_games,
            "estimated_elo": getattr(gauntlet_result, "estimated_elo", 0.0),
            "elo": getattr(gauntlet_result, "estimated_elo", 0.0),
            "passed": gauntlet_passed,
        }

        # Extract per-opponent results
        opponent_results = getattr(gauntlet_result, "opponent_results", {})
        for opp_name, opp_stats in opponent_results.items():
            opp_name_str = str(opp_name)
            if isinstance(opp_stats, dict):
                result_data["win_rates"][opp_name_str] = opp_stats.get("win_rate", 0.0)
                result_data["opponent_results"][opp_name_str] = opp_stats
            if "random" in opp_name_str.lower():
                result_data["vs_random_rate"] = opp_stats.get("win_rate", 0.0) if isinstance(opp_stats, dict) else 0.0
            elif "heuristic" in opp_name_str.lower():
                result_data["vs_heuristic_rate"] = opp_stats.get("win_rate", 0.0) if isinstance(opp_stats, dict) else 0.0

        logger.info(
            f"Gauntlet completed: {model_path} (work_id={work_id}) "
            f"win_rate={getattr(gauntlet_result, 'win_rate', 0):.1%}, "
            f"elo={result_data['estimated_elo']}, "
            f"games={result_data['games_played']}"
        )

        # Store results in work_item so WorkerPullLoop._report_result
        # sends them to the leader via /work/complete
        work_item["result"] = result_data

        # Also emit EVALUATION_COMPLETED locally for any local subscribers
        # Feb 2026: Only emit when games were actually played
        if total_games > 0:
            try:
                from app.coordination.event_emission_helpers import safe_emit_event
                safe_emit_event(
                    "EVALUATION_COMPLETED",
                    result_data,
                    context="gauntlet_worker",
                )
            except ImportError:
                pass

        # Feb 2026: Return False when 0 games completed so work item is
        # marked as failed and can be retried on another node.
        if total_games == 0:
            return False
        return True

    except ImportError as e:
        logger.error(f"Gauntlet modules not available: {e}")
        return False
    except Exception as e:
        logger.exception(f"Gauntlet execution error for {model_path}: {e}")
        return False
