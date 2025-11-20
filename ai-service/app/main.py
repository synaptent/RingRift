"""
RingRift AI Service - FastAPI Application
Provides AI move selection and position evaluation endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

from .ai.random_ai import RandomAI
from .ai.heuristic_ai import HeuristicAI
from .ai.neural_net import NeuralNetAI
from .ai.descent_ai import DescentAI
from .models import (
    GameState,
    Move,
    AIConfig,
    AIType,
    LineRewardChoiceRequest,
    LineRewardChoiceResponse,
    LineRewardChoiceOption,
    RingEliminationChoiceRequest,
    RingEliminationChoiceResponse,
    RingEliminationChoiceOption,
    RegionOrderChoiceRequest,
    RegionOrderChoiceResponse,
    RegionOrderChoiceOption,
    Position,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RingRift AI Service",
    description="AI move selection and evaluation service for RingRift",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI instances cache
ai_instances: Dict[str, Any] = {}


class MoveRequest(BaseModel):
    """Request model for AI move selection"""
    game_state: GameState
    player_number: int
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = None


class MoveResponse(BaseModel):
    """Response model for AI move selection"""
    move: Optional[Move]
    evaluation: float
    thinking_time_ms: int
    ai_type: str
    difficulty: int


class EvaluationRequest(BaseModel):
    """Request model for position evaluation"""
    game_state: GameState
    player_number: int


class EvaluationResponse(BaseModel):
    """Response model for position evaluation"""
    score: float
    breakdown: Dict[str, float]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "RingRift AI Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check for container orchestration"""
    return {"status": "healthy"}


@app.post("/ai/move", response_model=MoveResponse)
async def get_ai_move(request: MoveRequest):
    """
    Get AI-selected move for current game state
    
    Args:
        request: MoveRequest containing game state and AI configuration
        
    Returns:
        MoveResponse with selected move and evaluation
    """
    try:
        import time
        start_time = time.time()
        
        # Select AI type based on difficulty if not specified
        ai_type = request.ai_type or _select_ai_type(request.difficulty)
        
        # Get or create AI instance
        ai_key = f"{ai_type.value}-{request.difficulty}-{request.player_number}"
        
        if ai_key not in ai_instances:
            config = AIConfig(
                difficulty=request.difficulty,
                randomness=_get_randomness_for_difficulty(request.difficulty)
            )
            ai_instances[ai_key] = _create_ai_instance(
                ai_type, 
                request.player_number, 
                config
            )
        
        ai = ai_instances[ai_key]
        
        # Get move from AI
        move = ai.select_move(request.game_state)
        
        # Evaluate position
        evaluation = ai.evaluate_position(request.game_state)
        
        thinking_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"AI move: type={ai_type.value}, difficulty={request.difficulty}, "
            f"time={thinking_time}ms, eval={evaluation:.2f}"
        )
        
        return MoveResponse(
            move=move,
            evaluation=evaluation,
            thinking_time_ms=thinking_time,
            ai_type=ai_type.value,
            difficulty=request.difficulty
        )
        
    except Exception as e:
        logger.error(f"Error generating AI move: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/evaluate", response_model=EvaluationResponse)
async def evaluate_position(request: EvaluationRequest):
    """
    Evaluate current position from a player's perspective
    
    Args:
        request: EvaluationRequest with game state and player number
        
    Returns:
        EvaluationResponse with position score and breakdown
    """
    try:
        # Use heuristic AI for evaluation
        config = AIConfig(difficulty=5, randomness=0)
        ai = HeuristicAI(request.player_number, config)
        
        score = ai.evaluate_position(request.game_state)
        breakdown = ai.get_evaluation_breakdown(request.game_state)
        
        return EvaluationResponse(
            score=score,
            breakdown=breakdown
        )
        
    except Exception as e:
        logger.error(f"Error evaluating position: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/choice/line_reward_option", response_model=LineRewardChoiceResponse)
async def choose_line_reward_option(request: LineRewardChoiceRequest):
    """Select a line reward option for an AI-controlled player.

    For now this mirrors the TypeScript AIInteractionHandler heuristic by
    preferring Option 2 (minimum collapse, no elimination) when it is
    available, falling back to the first option. The endpoint is
    intentionally simple but carries enough metadata (difficulty,
    ai_type, optional game_state) to be extended later without breaking
    the contract.
    """
    try:
        ai_type = request.ai_type or _select_ai_type(request.difficulty)

        if not request.options:
            # Default conservatively to Option 2 semantics when no
            # options are provided, mirroring the TypeScript fallback.
            selected = LineRewardChoiceOption.OPTION_2
        elif LineRewardChoiceOption.OPTION_2 in request.options:
            selected = LineRewardChoiceOption.OPTION_2
        else:
            selected = request.options[0]

        return LineRewardChoiceResponse(
            selected_option=selected,
            ai_type=ai_type.value,
            difficulty=request.difficulty,
        )
    except Exception as e:
        logger.error(
            f"Error selecting line reward option: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/choice/ring_elimination", response_model=RingEliminationChoiceResponse)
async def choose_ring_elimination_option(request: RingEliminationChoiceRequest):
    """Select a ring elimination option for an AI-controlled player.

    This endpoint mirrors the TypeScript AIInteractionHandler heuristic
    by preferring the option with the smallest ``capHeight``,
    tie-breaking on ``totalHeight``. The full ``game_state`` is
    available on the request for more advanced heuristics in the
    future, but is not yet used directly.
    """
    try:
        ai_type = request.ai_type or _select_ai_type(request.difficulty)

        if not request.options:
            # No options is a protocol error: the engine should never
            # ask the AI to choose from an empty set of elimination
            # targets. Surface this clearly so the caller can fix the
            # upstream logic.
            raise HTTPException(status_code=400, detail="ring_elimination choice has no options")

        # Prefer the option with the smallest cap_height; if tied,
        # choose the one with the smallest total_height.
        selected = request.options[0]
        for opt in request.options[1:]:
            if opt.cap_height < selected.cap_height:
                selected = opt
            elif opt.cap_height == selected.cap_height and opt.total_height < selected.total_height:
                selected = opt

        return RingEliminationChoiceResponse(
            selected_option=selected,
            ai_type=ai_type.value,
            difficulty=request.difficulty,
        )
    except HTTPException:
        # Re-raise HTTPExceptions unchanged so FastAPI can handle them
        # as intended.
        raise
    except Exception as e:
        logger.error(
            f"Error selecting ring elimination option: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/choice/region_order", response_model=RegionOrderChoiceResponse)
async def choose_region_order_option(request: RegionOrderChoiceRequest):
    """Select a region order option for an AI-controlled player.

    This endpoint prefers regions that are both *large* and
    *strategically relevant* based on the current GameState. The
    scoring heuristic is:

    - Start from region size (larger regions are generally more
      impactful).
    - Add a bonus for nearby enemy-controlled stacks: for each stack
      whose controlling player is not ``request.player_number`` and is
      within a small radius of the region's representativePosition,
      add a decaying score contribution.

    This keeps the logic simple while still leveraging full GameState
    context. If no game_state is provided, the heuristic falls back to
    a size-only comparison.
    """
    try:
        ai_type = request.ai_type or _select_ai_type(request.difficulty)

        if not request.options:
            # No options: synthesize a zero-sized region.
            selected = RegionOrderChoiceOption(
                regionId="0",  # type: ignore[call-arg]
                size=0,
                representativePosition=Position(x=0, y=0),
            )
        else:
            # Precompute enemy player numbers when game_state is
            # available.
            enemy_players = set()
            stacks_by_key: Dict[str, RingEliminationChoiceOption] = {}
            if request.game_state is not None:
                for p in request.game_state.players:
                    if getattr(p, "player_number", None) != request.player_number:
                        enemy_players.add(p.player_number)

            def _manhattan(a: Position, b: Position) -> int:
                az = a.z or 0
                bz = b.z or 0
                return abs(a.x - b.x) + abs(a.y - b.y) + abs(az - bz)

            def score_region(option: RegionOrderChoiceOption) -> float:
                # Base: region size.
                score = float(option.size)

                if request.game_state is None or not enemy_players:
                    return score

                # Bonus: nearby enemy stacks. We treat stacks within a
                # small radius of the representative position as
                # belonging to or strongly influencing this region.
                board = request.game_state.board
                centre = option.representative_position
                radius = 3
                for stack in board.stacks.values():
                    if stack.controlling_player not in enemy_players:
                        continue
                    dist = _manhattan(stack.position, centre)
                    if dist <= radius:
                        # Closer stacks contribute more; ensure we
                        # never divide by zero.
                        score += 2.0 / float(1 + dist)

                return score

            # Select the option with the highest score; break ties by
            # preferring the larger region, then the first encountered.
            selected = request.options[0]
            best_score = score_region(selected)
            for opt in request.options[1:]:
                s = score_region(opt)
                if s > best_score or (s == best_score and opt.size > selected.size):
                    selected = opt
                    best_score = s

        return RegionOrderChoiceResponse(
            selected_option=selected,
            ai_type=ai_type.value,
            difficulty=request.difficulty,
        )
    except Exception as e:
        logger.error(
            f"Error selecting region order option: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ai/cache")
async def clear_ai_cache():
    """Clear cached AI instances"""
    global ai_instances
    ai_instances.clear()
    logger.info("AI cache cleared")
    return {"status": "cache cleared", "instances_removed": len(ai_instances)}


def _select_ai_type(difficulty: int) -> AIType:
    """Auto-select AI type based on difficulty"""
    if difficulty <= 2:
        return AIType.RANDOM
    elif difficulty <= 5:
        return AIType.HEURISTIC
    elif difficulty <= 8:
        return AIType.MINIMAX
    else:
        # Use Neural Network for highest difficulty
        return AIType.MCTS # Placeholder until MCTS is implemented, or use NeuralNetAI here if desired


def _get_randomness_for_difficulty(difficulty: int) -> float:
    """Get randomness factor for difficulty level"""
    randomness_map = {
        1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05,
        6: 0.02, 7: 0.01, 8: 0, 9: 0, 10: 0
    }
    return randomness_map.get(difficulty, 0.1)


def _create_ai_instance(ai_type: AIType, player_number: int, config: AIConfig):
    """Factory function to create AI instances"""
    if ai_type == AIType.RANDOM:
        return RandomAI(player_number, config)
    elif ai_type == AIType.HEURISTIC:
        return HeuristicAI(player_number, config)
    # elif ai_type == AIType.MINIMAX:
    #     return MinimaxAI(player_number, config)
    elif ai_type == AIType.MCTS: # Using MCTS enum for NeuralNetAI temporarily or add NEURAL_NET enum
         return NeuralNetAI(player_number, config)
    elif ai_type == AIType.DESCENT:
        return DescentAI(player_number, config)
    else:
        # Default to heuristic
        return HeuristicAI(player_number, config)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
