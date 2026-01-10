"""
Minimal AI Inference Service for RingRift Production

This is a lightweight FastAPI server that serves AI moves for the game.
It imports from ai-service but exposes only inference endpoints.

NO training, NO distributed coordination, NO daemons.
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add ai-service to path BEFORE importing from it
# This must happen before any ai-service imports
_ai_service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ai-service"))
if _ai_service_path not in sys.path:
    sys.path.insert(0, _ai_service_path)

# Now we can safely import from ai-service/app/
from app.models.core import AIType, AIConfig, GameState
from app.ai.factory import AIFactory
from app.ai.unified_loader import UnifiedModelLoader

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODELS_DIR = os.environ.get(
    "RINGRIFT_MODELS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models")
)

# Simplified difficulty mapping (production only needs a few levels)
DIFFICULTY_PROFILES = {
    1: {"ai_type": AIType.RANDOM, "description": "Random moves"},
    2: {"ai_type": AIType.HEURISTIC, "description": "Basic strategy"},
    3: {"ai_type": AIType.MINIMAX, "use_nnue": True, "description": "Intermediate"},
    4: {"ai_type": AIType.MCTS, "simulations": 200, "description": "Advanced"},
    5: {"ai_type": AIType.GUMBEL_MCTS, "simulations": 400, "description": "Expert"},
}

# Model cache (loaded at startup)
MODEL_CACHE: dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------

def load_models():
    """Pre-load canonical models at startup."""
    loader = UnifiedModelLoader()

    # Standard board/player combinations
    configs = [
        ("hex8", 2), ("hex8", 3), ("hex8", 4),
        ("square8", 2), ("square8", 3), ("square8", 4),
        ("square19", 2), ("square19", 3), ("square19", 4),
        ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
    ]

    for board_type, num_players in configs:
        model_name = f"canonical_{board_type}_{num_players}p.pth"
        model_path = os.path.join(MODELS_DIR, model_name)

        if os.path.exists(model_path):
            try:
                loaded = loader.load(model_path, board_type)
                cache_key = f"{board_type}_{num_players}p"
                MODEL_CACHE[cache_key] = loaded
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        else:
            logger.debug(f"Model not found: {model_path}")

    logger.info(f"Loaded {len(MODEL_CACHE)} models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    logger.info("Starting AI Inference Service...")
    load_models()
    yield
    logger.info("Shutting down AI Inference Service...")
    MODEL_CACHE.clear()


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="RingRift AI Inference",
    description="Minimal AI service for production game server",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------

class MoveRequest(BaseModel):
    """Request for AI move selection."""
    game_state: dict  # Raw game state from client
    player_number: int
    difficulty: int = 3  # 1-5, default intermediate
    seed: int | None = None


class MoveResponse(BaseModel):
    """Response with selected move."""
    move: dict | None
    evaluation: float
    thinking_time_ms: int
    difficulty: int
    ai_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    model_keys: list[str]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=len(MODEL_CACHE),
        model_keys=list(MODEL_CACHE.keys()),
    )


@app.post("/move", response_model=MoveResponse)
async def get_move(request: MoveRequest):
    """
    Get AI move for the given game state.

    This is the main endpoint used by the game server.
    """
    start_time = time.time()

    # Validate difficulty
    difficulty = max(1, min(5, request.difficulty))
    profile = DIFFICULTY_PROFILES[difficulty]

    try:
        # Create AI config
        config = AIConfig(
            difficulty=difficulty,
            use_neural_net=profile.get("use_nnue", False),
            simulations=profile.get("simulations", 100),
            seed=request.seed,
        )

        # Create AI instance
        ai = AIFactory.create(
            ai_type=profile["ai_type"],
            player_number=request.player_number,
            config=config,
        )

        # Convert game state dict to internal format
        game_state = GameState.model_validate(request.game_state)

        # Get move
        move = ai.select_move(game_state)
        evaluation = ai.evaluate_position(game_state)

        thinking_time_ms = int((time.time() - start_time) * 1000)

        return MoveResponse(
            move=move.to_dict() if move else None,
            evaluation=evaluation,
            thinking_time_ms=thinking_time_ms,
            difficulty=difficulty,
            ai_type=profile["ai_type"].value,
        )

    except Exception as e:
        logger.error(f"Error generating move: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "RingRift AI Inference",
        "version": "1.0.0",
        "endpoints": ["/health", "/move"],
        "docs": "/docs",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=False,
    )
