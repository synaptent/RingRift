"""FastAPI router for online learning endpoints.

Provides REST API for triggering online learning from completed games,
particularly human vs AI games where the human wins.

Created: January 2026
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/online-learning", tags=["online-learning"])

# Shadow model directory for online updates
SHADOW_MODEL_DIR = Path("models/shadow")


# =============================================================================
# Request/Response Models
# =============================================================================


class MoveRecord(BaseModel):
    """Move record for online learning.

    Accepts both client format (type, from, to) and server format (move_type, from_pos, to_pos).
    """

    model_config = ConfigDict(populate_by_name=True)

    move_type: str = Field(..., alias="type", description="Move type (place_ring, move_stack, etc.)")
    from_pos: dict[str, int] | None = Field(None, alias="from", description="Source position {x, y}")
    to_pos: dict[str, int] | None = Field(None, alias="to", description="Target position {x, y}")
    player: int | None = Field(None, description="Player number who made the move")
    capture_target: dict[str, int] | None = Field(
        None, alias="captureTarget", description="Capture target position {x, y}"
    )


class LearnFromGameRequest(BaseModel):
    """Request to learn from a completed game."""

    board_type: str = Field(..., description="Board type (e.g., hex8, square8)")
    num_players: int = Field(..., ge=2, le=4, description="Number of players")
    moves: list[MoveRecord] = Field(..., description="List of moves in order")
    winner: int = Field(..., ge=1, le=4, description="Player number who won (1-indexed)")
    human_player: int = Field(..., ge=1, le=4, description="Which player was human")
    human_won: bool = Field(True, description="Whether human won")
    ai_difficulty: int | None = Field(None, ge=1, le=10, description="AI difficulty level")


class LearningMetrics(BaseModel):
    """Metrics from online learning update."""

    total_loss: float
    td_loss: float
    outcome_loss: float
    num_transitions: int
    games_in_buffer: int
    model_updated: bool
    shadow_model_path: str | None = None


class LearnFromGameResponse(BaseModel):
    """Response from learn_from_game endpoint."""

    success: bool
    metrics: LearningMetrics | None = None
    message: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def _get_shadow_model_path(board_type: str, num_players: int) -> Path:
    """Get path for shadow model that receives online updates."""
    SHADOW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return SHADOW_MODEL_DIR / f"{board_type}_{num_players}p_online.pth"


def _get_canonical_model_path(board_type: str, num_players: int) -> Path:
    """Get path for canonical model to copy for shadow model."""
    return Path(f"models/canonical_{board_type}_{num_players}p.pth")


async def _ensure_shadow_model_exists(board_type: str, num_players: int) -> Path:
    """Ensure shadow model exists, copying from canonical if needed."""
    shadow_path = _get_shadow_model_path(board_type, num_players)
    canonical_path = _get_canonical_model_path(board_type, num_players)

    if not shadow_path.exists():
        if not canonical_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Canonical model not found: {canonical_path}",
            )
        # Copy canonical to shadow
        import shutil

        def _copy():
            shutil.copy2(canonical_path, shadow_path)
            logger.info(f"Created shadow model from canonical: {shadow_path}")

        await asyncio.to_thread(_copy)

    return shadow_path


async def _run_online_learning(
    request: LearnFromGameRequest,
    shadow_model_path: Path,
) -> LearningMetrics:
    """Run online learning update on the shadow model.

    Args:
        request: Learning request with game data
        shadow_model_path: Path to shadow model

    Returns:
        Learning metrics from the update
    """

    def _learn() -> LearningMetrics:
        import torch

        from app.ai.ebmo_online_learner import EBMOOnlineLearner, EBMOOnlineConfig
        from app.game_engine import GameEngine
        from app.board_manager import get_board_for_config

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load shadow model
        try:
            from app.utils.torch_utils import safe_load_checkpoint

            checkpoint = safe_load_checkpoint(str(shadow_model_path))
        except Exception as e:
            logger.error(f"Failed to load shadow model: {e}")
            raise

        # Create network from checkpoint
        from app.ai.ebmo_network import EBMONetwork

        # Get board info for network initialization
        board = get_board_for_config(request.board_type)
        board_size = getattr(board, "size", 8)

        network = EBMONetwork(
            board_size=board_size,
            hidden_dim=checkpoint.get("hidden_dim", 256),
        )
        network.load_state_dict(checkpoint["model_state_dict"])

        # Create online learner with config tuned for human games
        config = EBMOOnlineConfig(
            min_games_before_update=1,  # Learn immediately
            learning_rate=1e-5 if request.human_won else 5e-6,  # Higher LR for human wins
            buffer_size=50,
            batch_size=1,
        )
        learner = EBMOOnlineLearner(network, device=device, config=config)

        # Replay game and record transitions
        try:
            # Create initial state
            state = GameEngine.create_initial_state(
                board_type=request.board_type,
                num_players=request.num_players,
            )

            for move_record in request.moves:
                # Convert move record to Move object
                from app.models import Move

                move = Move(
                    move_type=move_record.move_type,
                    from_pos=move_record.from_pos,
                    to_pos=move_record.to_pos,
                )

                # Record transition
                learner.record_transition(
                    state=state,
                    move=move,
                    player=state.current_player,
                    next_state=None,  # Will be set below
                )

                # Apply move
                try:
                    state = GameEngine.apply_move(state, move)
                except Exception as e:
                    logger.warning(f"Failed to apply move {move}: {e}")
                    break

        except Exception as e:
            logger.error(f"Failed to replay game: {e}")
            raise

        # Update with game outcome
        metrics_dict = learner.update_from_game(winner=request.winner)

        # Save updated weights to shadow model
        model_updated = False
        try:
            checkpoint["model_state_dict"] = network.state_dict()
            torch.save(checkpoint, shadow_model_path)
            model_updated = True
            logger.info(f"Updated shadow model: {shadow_model_path}")
        except Exception as e:
            logger.error(f"Failed to save shadow model: {e}")

        return LearningMetrics(
            total_loss=metrics_dict.get("total_loss", 0.0),
            td_loss=metrics_dict.get("td_loss", 0.0),
            outcome_loss=metrics_dict.get("outcome_loss", 0.0),
            num_transitions=len(request.moves),
            games_in_buffer=len(learner.game_buffer),
            model_updated=model_updated,
            shadow_model_path=str(shadow_model_path) if model_updated else None,
        )

    return await asyncio.to_thread(_learn)


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/learn_from_game", response_model=LearnFromGameResponse)
async def learn_from_game(request: LearnFromGameRequest) -> LearnFromGameResponse:
    """Trigger immediate online learning from a completed game.

    This endpoint is called by the TypeScript server when a human wins
    against the AI. The learning update is applied to a shadow model
    (not the canonical model) to prevent catastrophic forgetting.

    Shadow models are periodically validated and merged into canonical
    models if they show improvement.

    Args:
        request: Game data including moves, winner, and human player info

    Returns:
        Learning metrics and success status
    """
    try:
        # Only learn from human wins by default (highest quality signal)
        if not request.human_won:
            return LearnFromGameResponse(
                success=True,
                message="Skipped learning (human did not win)",
            )

        # Ensure shadow model exists
        shadow_model_path = await _ensure_shadow_model_exists(
            request.board_type,
            request.num_players,
        )

        # Run online learning
        metrics = await _run_online_learning(request, shadow_model_path)

        logger.info(
            f"Online learning completed: {request.board_type}_{request.num_players}p, "
            f"loss={metrics.total_loss:.4f}, transitions={metrics.num_transitions}"
        )

        return LearnFromGameResponse(
            success=True,
            metrics=metrics,
            message="Learning update applied to shadow model",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Online learning failed: {e}")
        return LearnFromGameResponse(
            success=False,
            message=f"Learning failed: {str(e)}",
        )


@router.get("/shadow_models")
async def list_shadow_models() -> dict[str, Any]:
    """List all shadow models and their status.

    Returns information about shadow models that have received
    online updates and are candidates for merging.
    """
    models: list[dict[str, Any]] = []

    if SHADOW_MODEL_DIR.exists():
        for model_file in SHADOW_MODEL_DIR.glob("*_online.pth"):
            stat = model_file.stat()
            models.append(
                {
                    "name": model_file.name,
                    "path": str(model_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                }
            )

    return {
        "shadow_model_dir": str(SHADOW_MODEL_DIR),
        "models": models,
        "count": len(models),
    }


@router.post("/validate_merge/{board_type}/{num_players}")
async def validate_shadow_for_merge(
    board_type: str,
    num_players: int,
    games: int = 20,
) -> dict[str, Any]:
    """Validate if shadow model should be merged into canonical.

    Runs a mini-gauntlet between shadow and canonical models.
    Shadow must win >= 55% to be considered for merge.

    Args:
        board_type: Board type (e.g., hex8)
        num_players: Number of players (2-4)
        games: Number of games to play (default: 20)

    Returns:
        Validation results including win rate and merge recommendation
    """
    shadow_path = _get_shadow_model_path(board_type, num_players)
    canonical_path = _get_canonical_model_path(board_type, num_players)

    if not shadow_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Shadow model not found: {shadow_path}",
        )

    if not canonical_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Canonical model not found: {canonical_path}",
        )

    def _run_validation() -> dict[str, Any]:
        # Import gauntlet runner
        try:
            from app.training.game_gauntlet import run_gauntlet_evaluation
        except ImportError:
            return {
                "error": "Gauntlet module not available",
                "shadow_wins": 0,
                "total_games": 0,
                "win_rate": 0.0,
                "should_merge": False,
            }

        # Run mini-gauntlet
        try:
            results = run_gauntlet_evaluation(
                model_path=str(shadow_path),
                board_type=board_type,
                num_players=num_players,
                opponent_model=str(canonical_path),
                num_games=games,
            )

            shadow_wins = results.get("wins", 0)
            total = results.get("total", games)
            win_rate = shadow_wins / total if total > 0 else 0.0

            return {
                "shadow_wins": shadow_wins,
                "total_games": total,
                "win_rate": round(win_rate, 3),
                "should_merge": win_rate >= 0.55,
                "threshold": 0.55,
            }

        except Exception as e:
            logger.error(f"Gauntlet validation failed: {e}")
            return {
                "error": str(e),
                "shadow_wins": 0,
                "total_games": 0,
                "win_rate": 0.0,
                "should_merge": False,
            }

    result = await asyncio.to_thread(_run_validation)
    return result
