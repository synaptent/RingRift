"""
RingRift AI Service - FastAPI Application
Provides AI move selection and position evaluation endpoints
"""

import logging
import time
import os
from datetime import datetime, timezone
from dataclasses import dataclass
import threading

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, TypedDict

try:  # Python 3.10 compatibility (NotRequired added in 3.11)
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired  # type: ignore

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import (
    AI_INSTANCE_CACHE_LOOKUPS,
    AI_INSTANCE_CACHE_SIZE,
    AI_MOVE_LATENCY,
    AI_MOVE_REQUESTS,
    observe_ai_move_start,
)
from .config.ladder_config import (
    LadderTierConfig,
    get_effective_ladder_config,
)

from .ai.random_ai import RandomAI
from .ai.heuristic_ai import HeuristicAI
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
    RegionOrderChoiceRequest,
    RegionOrderChoiceResponse,
    RegionOrderChoiceOption,
    LineOrderChoiceRequest,
    LineOrderChoiceResponse,
    LineOrderChoiceLine,
    CaptureDirectionChoiceRequest,
    CaptureDirectionChoiceResponse,
    CaptureDirectionChoiceOption,
    Position,
    GameStatus,
)
from .board_manager import BoardManager
from .game_engine import GameEngine
from .rules.default_engine import DefaultRulesEngine
from .routes import replay_router

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

# CORS middleware - allows sandbox UI to access replay API
# In production, restrict allow_origins to specific domains
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount replay router for game database access
app.include_router(replay_router)

# AI instances cache
@dataclass
class CachedAIInstance:
    ai: Any
    created_at: float
    last_access: float


AI_INSTANCE_CACHE_ENABLED = os.getenv("RINGRIFT_AI_INSTANCE_CACHE", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AI_INSTANCE_CACHE_TTL_SEC = int(os.getenv("RINGRIFT_AI_INSTANCE_CACHE_TTL_SEC", "1800"))
AI_INSTANCE_CACHE_MAX = int(os.getenv("RINGRIFT_AI_INSTANCE_CACHE_MAX", "512"))

_ai_cache_lock = threading.Lock()
ai_instances: Dict[str, CachedAIInstance] = {}


def _should_cache_ai(ai_type: AIType, game_state: GameState) -> bool:
    """Return True if this request should reuse a persistent AI instance."""
    if not AI_INSTANCE_CACHE_ENABLED:
        return False
    if ai_type not in (AIType.MINIMAX, AIType.MCTS, AIType.DESCENT):
        return False
    if getattr(game_state, "game_status", None) != GameStatus.ACTIVE:
        return False
    # Requires stable game id so we do not accidentally share state across games.
    return bool(getattr(game_state, "id", None))


def _ai_cache_key(
    game_state: GameState,
    player_number: int,
    ai_type: AIType,
    config: AIConfig,
) -> str:
    # Key includes config so callers can safely A/B compare profiles
    # without cross-contaminating tree state.
    return "|".join(
        [
            str(game_state.id),
            str(getattr(game_state.board_type, "value", game_state.board_type)),
            str(len(game_state.players) if getattr(game_state, "players", None) else 0),
            str(player_number),
            str(ai_type.value),
            str(config.difficulty),
            str(config.think_time or 0),
            str(config.randomness or 0.0),
            str(config.rng_seed if config.rng_seed is not None else ""),
            str(config.heuristic_profile_id or ""),
            str(config.nn_model_id or ""),
            str(bool(config.use_neural_net) if config.use_neural_net is not None else ""),
            str(bool(config.use_incremental_search)),
        ]
    )


def _prune_ai_cache(now: float) -> None:
    """Best-effort pruning to bound memory from cached search trees."""
    if not ai_instances:
        return

    expired = [
        key
        for key, entry in ai_instances.items()
        if now - entry.last_access > AI_INSTANCE_CACHE_TTL_SEC
    ]
    for key in expired:
        ai_instances.pop(key, None)

    if len(ai_instances) <= AI_INSTANCE_CACHE_MAX:
        return

    # Evict least-recently-used entries.
    entries_by_age = sorted(ai_instances.items(), key=lambda kv: kv[1].last_access)
    overflow = len(ai_instances) - AI_INSTANCE_CACHE_MAX
    for key, _entry in entries_by_age[:overflow]:
        ai_instances.pop(key, None)

    AI_INSTANCE_CACHE_SIZE.set(len(ai_instances))


def _get_cached_ai(key: str) -> Any | None:
    now = time.time()
    with _ai_cache_lock:
        _prune_ai_cache(now)
        entry = ai_instances.get(key)
        if entry is None:
            return None
        entry.last_access = now
        AI_INSTANCE_CACHE_SIZE.set(len(ai_instances))
        return entry.ai


def _put_cached_ai(key: str, ai: Any) -> None:
    now = time.time()
    with _ai_cache_lock:
        _prune_ai_cache(now)
        ai_instances[key] = CachedAIInstance(ai=ai, created_at=now, last_access=now)
        AI_INSTANCE_CACHE_SIZE.set(len(ai_instances))


class MoveRequest(BaseModel):
    """Request model for AI move selection"""
    game_state: GameState
    player_number: int
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = None
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=0x7FFFFFFF,
        description="Optional RNG seed for deterministic AI behavior"
    )


class MoveResponse(BaseModel):
    """Response model for AI move selection"""
    move: Optional[Move]
    evaluation: float
    thinking_time_ms: int
    ai_type: str
    difficulty: int
    # Optional model observability fields (backward compatible for TS clients).
    nn_model_id: Optional[str] = None
    nn_checkpoint: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Request model for position evaluation"""
    game_state: GameState
    player_number: int


class EvaluationResponse(BaseModel):
    """Response model for position evaluation"""
    score: float
    breakdown: Dict[str, float]


class PositionEvaluationByPlayer(BaseModel):
    """Per-player evaluation summary for analysis mode."""

    totalEval: float
    territoryEval: float = 0.0
    ringEval: float = 0.0
    winProbability: Optional[float] = None


class PositionEvaluationRequest(BaseModel):
    """Request model for multi-player position evaluation."""

    game_state: GameState
    # Optional label for the underlying engine profile used to evaluate.
    engine_profile: Optional[str] = None
    # Optional explicit RNG seed for deterministic evaluations.
    random_seed: Optional[int] = None


class PositionEvaluationResponse(BaseModel):
    """Response model for multi-player position evaluation.

    Suitable for streaming over analysis and spectator channels.
    """

    engine_profile: str
    board_type: str
    game_id: str
    move_number: int
    per_player: Dict[int, PositionEvaluationByPlayer]
    evaluation_scale: str
    generated_at: str


class RulesEvalRequest(BaseModel):
    """Request model for rules-engine evaluation (parity/shadow mode)."""
    game_state: GameState
    move: Move


class RulesEvalResponse(BaseModel):
    """Response model for rules-engine evaluation."""
    valid: bool
    validation_error: Optional[str] = None
    next_state: Optional[GameState] = None
    state_hash: Optional[str] = None
    s_invariant: Optional[int] = None
    game_status: Optional[GameStatus] = None


def _derive_non_ssot_seed(request: MoveRequest) -> int:
    """
    Derive a deterministic but non-SSOT RNG seed for /ai/move when neither
    the explicit ``seed`` nor ``game_state.rng_seed`` is provided.

    This is used only for ad-hoc or training/diagnostic callers that invoke
    the AI service without threading through the canonical game seed from
    the TypeScript hosts. Production gameplay traffic should always supply
    either ``seed`` or ``game_state.rng_seed``.
    """
    base = (request.difficulty * 1_000_003) ^ (request.player_number * 97_911)
    return int(base & 0x7FFFFFFF)


def _select_effective_seed(request: MoveRequest) -> tuple[int, str]:
    """
    Compute the effective RNG seed and its provenance for an /ai/move call.

    Priority:
      1. request.seed (explicit override from caller)
      2. request.game_state.rng_seed (engine-provided game seed)
      3. Derived non-SSOT fallback (_derive_non_ssot_seed).

    Returns:
        A tuple of (seed, source), where source is one of:
        "explicit", "game_state", or "derived".
    """
    if request.seed is not None:
        return int(request.seed), "explicit"

    rng_seed = getattr(request.game_state, "rng_seed", None)
    if rng_seed is not None:
        return int(rng_seed), "game_state"

    return _derive_non_ssot_seed(request), "derived"


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


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for local/dev observability.

    This exposes the default Prometheus text exposition format so that
    `ai-service` can be scraped by a Prometheus-compatible collector. In
    production this can be wired into a full metrics stack; in local/dev it
    is primarily useful for ad-hoc debugging.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/ai/move", response_model=MoveResponse)
async def get_ai_move(request: MoveRequest):
    """
    Get AI-selected move for current game state.

    Args:
        request: MoveRequest containing game state and AI configuration.

    Returns:
        MoveResponse with selected move and evaluation.
    """
    start_time = time.time()

    # Select AI type and canonical difficulty profile based on difficulty
    # if not explicitly specified on the request. This keeps the Python
    # service, TypeScript backend, and client-facing difficulty ladder
    # aligned. Where a board-aware LadderTierConfig exists for the current
    # (difficulty, board_type, num_players) triple, its settings take
    # precedence so that live play and tier evaluation share the same
    # canonical ladder.
    profile = _get_difficulty_profile(request.difficulty)

    ladder_config: Optional[LadderTierConfig] = None
    try:
        board_type = getattr(request.game_state, "board_type", None)
        players = getattr(request.game_state, "players", None)
        num_players = len(players) if players is not None else None
        if board_type is not None and num_players:
            ladder_config = get_effective_ladder_config(
                request.difficulty,
                board_type,
                num_players,
            )
    except Exception:
        ladder_config = None

    base_ai_type = (
        ladder_config.ai_type
        if ladder_config is not None
        else profile["ai_type"]
    )
    ai_type = request.ai_type or base_ai_type
    labels_ai_type, labels_difficulty = observe_ai_move_start(
        ai_type.value,
        request.difficulty,
    )

    # Compute effective_seed according to the TS-aligned contract:
    #   1. request.seed (explicit override)
    #   2. game_state.rng_seed (engine-provided)
    #   3. derived non-SSOT fallback for ad-hoc callers.
    effective_seed, seed_source = _select_effective_seed(request)

    try:
        heuristic_profile_id: Optional[str] = None
        nn_model_id: Optional[str] = None

        if ladder_config is not None and ladder_config.heuristic_profile_id:
            heuristic_profile_id = ladder_config.heuristic_profile_id
        elif ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")

        if (
            ladder_config is not None
            and ladder_config.model_id
            and ai_type in (AIType.MCTS, AIType.DESCENT)
        ):
            nn_model_id = ladder_config.model_id

        if seed_source == "derived":
            # Mark non-SSOT / non-parity-critical paths explicitly so they
            # are not confused with fully seed-threaded production traffic.
            logger.debug(
                "Using derived non-SSOT RNG seed for /ai/move",
                extra={
                    "player_number": request.player_number,
                    "difficulty": request.difficulty,
                },
            )

        randomness = (
            ladder_config.randomness
            if ladder_config is not None
            else profile["randomness"]
        )
        think_time_ms = (
            ladder_config.think_time_ms
            if ladder_config is not None
            else profile["think_time_ms"]
        )

        config = AIConfig(
            difficulty=request.difficulty,
            randomness=randomness,
            think_time=think_time_ms,
            rngSeed=effective_seed,
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
        )
        ai = None
        cache_key: Optional[str] = None
        if _should_cache_ai(ai_type, request.game_state):
            cache_key = _ai_cache_key(
                request.game_state,
                request.player_number,
                ai_type,
                config,
            )
            ai = _get_cached_ai(cache_key)
            if ai is None:
                AI_INSTANCE_CACHE_LOOKUPS.labels(ai_type.value, "miss").inc()
            else:
                AI_INSTANCE_CACHE_LOOKUPS.labels(ai_type.value, "hit").inc()

        if ai is None:
            ai = _create_ai_instance(
                ai_type,
                request.player_number,
                config,
            )
            if cache_key is not None:
                _put_cached_ai(cache_key, ai)

        # Get move from AI
        move = ai.select_move(request.game_state)

        # Evaluate position
        evaluation = ai.evaluate_position(request.game_state)

        thinking_time = int((time.time() - start_time) * 1000)

        # Record success metrics
        duration_seconds = time.time() - start_time
        AI_MOVE_REQUESTS.labels(
            labels_ai_type,
            labels_difficulty,
            "success",
        ).inc()
        AI_MOVE_LATENCY.labels(
            labels_ai_type,
            labels_difficulty,
        ).observe(duration_seconds)

        logger.info(
            "AI move: type=%s, difficulty=%d, time=%dms, eval=%.2f",
            ai_type.value,
            request.difficulty,
            thinking_time,
            evaluation,
        )

        nn_checkpoint: Optional[str] = None
        try:
            neural_net = getattr(ai, "neural_net", None)
            path = getattr(neural_net, "loaded_checkpoint_path", None) if neural_net is not None else None
            if path:
                nn_checkpoint = os.path.basename(str(path))
        except Exception:
            nn_checkpoint = None

        return MoveResponse(
            move=move,
            evaluation=evaluation,
            thinking_time_ms=thinking_time,
            ai_type=ai_type.value,
            difficulty=request.difficulty,
            nn_model_id=nn_model_id,
            nn_checkpoint=nn_checkpoint,
        )

    except Exception as e:
        # Record error metrics
        duration_seconds = time.time() - start_time
        AI_MOVE_REQUESTS.labels(
            labels_ai_type,
            labels_difficulty,
            "error",
        ).inc()
        AI_MOVE_LATENCY.labels(
            labels_ai_type,
            labels_difficulty,
        ).observe(duration_seconds)
        logger.error("Error generating AI move: %s", str(e), exc_info=True)
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
        # Use heuristic AI for evaluation. We tie this to the same canonical
        # heuristic profile as difficulty 5 so that offline evaluations and
        # online play remain aligned.
        config = AIConfig(
            difficulty=5,
            randomness=0,
            rngSeed=None,
            heuristic_profile_id="v1-heuristic-5",
        )
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


def _select_effective_eval_seed(
    game_state: GameState,
    explicit_seed: Optional[int],
) -> int:
    """
    Compute an effective RNG seed for /ai/evaluate_position.

    Priority mirrors /ai/move semantics:
      1. explicit random_seed on the request
      2. game_state.rng_seed
      3. fixed fallback for ad-hoc callers
    """
    if explicit_seed is not None:
        try:
            return int(explicit_seed) & 0x7FFFFFFF
        except (TypeError, ValueError):
            pass

    rng_seed = getattr(game_state, "rng_seed", None)
    if rng_seed is not None:
        try:
            return int(rng_seed) & 0x7FFFFFFF
        except (TypeError, ValueError):
            pass

    # Stable but arbitrary fallback when no better seed is available.
    return 42


@app.post("/ai/evaluate_position", response_model=PositionEvaluationResponse)
async def evaluate_position_multi(request: PositionEvaluationRequest):
    """
    Evaluate the current position from all players' perspectives.

    This endpoint is intended for analysis and spectator tooling. It returns
    a compact, per-player summary suitable for streaming over WebSockets and
    rendering in an evaluation panel.
    """
    try:
        state = request.game_state

        # For the initial implementation we reuse the heuristic evaluator at a
        # fixed profile roughly aligned with difficulty 5. This keeps the
        # analysis surface consistent with existing /ai/evaluate while
        # remaining fast enough for live usage. Stronger engines
        # (Minimax/Descent+NN) can be plugged in behind the same contract
        # later.
        engine_profile = request.engine_profile or "heuristic_v1_d5"
        base_seed = _select_effective_eval_seed(state, request.random_seed)

        raw_total: Dict[int, float] = {}
        raw_territory: Dict[int, float] = {}
        raw_rings: Dict[int, float] = {}

        for idx, player in enumerate(state.players):
            player_number = getattr(player, "player_number", None)
            if player_number is None:
                continue

            # Derive a per-player seed so that evaluations are deterministic
            # for a given (game_state, base_seed) pair but do not share RNG
            # streams.
            player_seed = (base_seed + (idx + 1) * 100_003) & 0x7FFFFFFF

            config = AIConfig(
                difficulty=5,
                randomness=0.0,
                rngSeed=player_seed,
                heuristic_profile_id="v1-heuristic-5",
            )
            ai = HeuristicAI(player_number, config)

            breakdown = ai.get_evaluation_breakdown(state)
            total = float(breakdown.get("total", ai.evaluate_position(state)))
            territory = float(breakdown.get("territory", 0.0))
            eliminated_rings = float(breakdown.get("eliminated_rings", 0.0))

            raw_total[player_number] = total
            raw_territory[player_number] = territory
            raw_rings[player_number] = eliminated_rings

        if not raw_total:
            raise HTTPException(
                status_code=400,
                detail="No players found in game_state",
            )

        def _to_zero_sum(source: Dict[int, float]) -> Dict[int, float]:
            if not source:
                return {}
            count = float(len(source))
            mean = float(sum(source.values())) / count
            return {p: float(v - mean) for p, v in source.items()}

        total_zero_sum = _to_zero_sum(raw_total)
        territory_zero_sum = _to_zero_sum(raw_territory)
        rings_zero_sum = _to_zero_sum(raw_rings)

        per_player: Dict[int, PositionEvaluationByPlayer] = {}
        for player_number, total in total_zero_sum.items():
            per_player[player_number] = PositionEvaluationByPlayer(
                totalEval=total,
                territoryEval=territory_zero_sum.get(player_number, 0.0),
                ringEval=rings_zero_sum.get(player_number, 0.0),
                winProbability=None,
            )

        move_number = len(getattr(state, "move_history", []) or [])

        board_type = getattr(state, "board_type", "")
        if hasattr(board_type, "value"):
            board_type_str = str(getattr(board_type, "value"))
        else:
            board_type_str = str(board_type)

        response = PositionEvaluationResponse(
            engine_profile=engine_profile,
            board_type=board_type_str,
            game_id=getattr(state, "id", ""),
            move_number=move_number,
            per_player=per_player,
            evaluation_scale="zero_sum_margin",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error evaluating position (multi-player): %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rules/evaluate_move", response_model=RulesEvalResponse)
async def evaluate_move(request: RulesEvalRequest):
    """Rules-engine evaluation endpoint for TS↔Python parity/shadow mode.

    Given a ``GameState`` and a candidate ``Move``, this endpoint:

    - Validates the move using the canonical ``DefaultRulesEngine``
      (which mirrors the shared TS rules engine).
    - If invalid, returns ``valid=False`` with a ``validation_error``.
    - If valid, applies the move via ``GameEngine.apply_move`` and
      returns the resulting state, hash, S-invariant, and game_status.

    This keeps the HTTP surface aligned with the same validation logic
    used by the offline parity harness in
    ``tests/parity/test_rules_parity_fixtures.py`` while avoiding
    over-constraining equality to the exact synthetic moves emitted by
    ``GameEngine.get_valid_moves`` (which may differ in non-semantic
    fields such as ``id``, ``timestamp``, or sentinel ``to`` values).
    """
    try:
        state = request.game_state
        move = request.move

        engine = DefaultRulesEngine()
        is_valid = engine.validate_move(state, move)

        if not is_valid:
            return RulesEvalResponse(
                valid=False,
                validation_error=(
                    "Move rejected by DefaultRulesEngine.validate_move"
                ),
            )

        # Apply the move using the Python GameEngine (TS-aligned semantics).
        next_state = GameEngine.apply_move(state, move)

        # Compute hash and S-invariant using BoardManager helpers.
        state_hash = BoardManager.hash_game_state(next_state)
        progress = BoardManager.compute_progress_snapshot(next_state)

        return RulesEvalResponse(
            valid=True,
            next_state=next_state,
            state_hash=state_hash,
            s_invariant=progress.S,
            game_status=next_state.game_status,
        )
    except Exception as e:
        logger.error(
            "Error in /rules/evaluate_move: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/choice/line_reward_option",
    response_model=LineRewardChoiceResponse,
)
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
            selectedOption=selected,
            aiType=ai_type.value,
            difficulty=request.difficulty,
        )
    except Exception as e:
        logger.error(
            f"Error selecting line reward option: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/choice/ring_elimination",
    response_model=RingEliminationChoiceResponse,
)
async def choose_ring_elimination_option(
    request: RingEliminationChoiceRequest,
):
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
            raise HTTPException(
                status_code=400,
                detail="ring_elimination choice has no options",
            )

        # Prefer the option with the smallest cap_height; if tied,
        # choose the one with the smallest total_height.
        selected = request.options[0]
        for opt in request.options[1:]:
            if opt.cap_height < selected.cap_height:
                selected = opt
            elif (
                opt.cap_height == selected.cap_height
                and opt.total_height < selected.total_height
            ):
                selected = opt

        return RingEliminationChoiceResponse(
            selectedOption=selected,
            aiType=ai_type.value,
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


@app.post(
    "/ai/choice/region_order",
    response_model=RegionOrderChoiceResponse,
)
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
            if request.game_state is not None:
                for p in request.game_state.players:
                    if getattr(
                        p,
                        "player_number",
                        None,
                    ) != request.player_number:
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
                if s > best_score or (
                    s == best_score and opt.size > selected.size
                ):
                    selected = opt
                    best_score = s

        return RegionOrderChoiceResponse(
            selectedOption=selected,
            aiType=ai_type.value,
            difficulty=request.difficulty,
        )
    except Exception as e:
        logger.error(
            f"Error selecting region order option: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/choice/line_order",
    response_model=LineOrderChoiceResponse,
)
async def choose_line_order_option(request: LineOrderChoiceRequest):
    """Select a line order option for an AI-controlled player.

    This endpoint mirrors the TypeScript AIInteractionHandler heuristic
    by preferring the line with the greatest number of markerPositions.
    When marker counts tie, it falls back to the first option. The full
    GameState is accepted for future, more context-aware heuristics but
    is not yet consulted directly.
    """
    try:
        ai_type = request.ai_type or _select_ai_type(request.difficulty)

        if not request.options:
            # No options is a protocol error: the engine should never ask
            # for a line_order choice with an empty option list.
            raise HTTPException(
                status_code=400,
                detail="line_order choice has no options",
            )

        selected: LineOrderChoiceLine = request.options[0]
        best_len = len(selected.marker_positions)

        for opt in request.options[1:]:
            length = len(opt.marker_positions)
            if length > best_len:
                selected = opt
                best_len = length

        return LineOrderChoiceResponse(
            selectedOption=selected,
            aiType=ai_type.value,
            difficulty=request.difficulty,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error selecting line order option: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ai/choice/capture_direction",
    response_model=CaptureDirectionChoiceResponse,
)
async def choose_capture_direction_option(
    request: CaptureDirectionChoiceRequest,
):
    """Select a capture_direction option for an AI-controlled player.

    This mirrors the TypeScript AIInteractionHandler heuristic:

    - Prefer options with the highest capturedCapHeight.
    - Break ties by choosing the landingPosition closest to a rough
      board centre estimate using Manhattan distance.
    """
    try:
        ai_type = request.ai_type or _select_ai_type(request.difficulty)

        if not request.options:
            raise HTTPException(
                status_code=400,
                detail="capture_direction choice has no options",
            )

        def _estimate_centre(reference: Position) -> Position:
            # Simple heuristic centre; we do not currently need full board
            # config here.
            return Position(
                x=reference.x,
                y=reference.y,
                z=getattr(reference, "z", None),
            )

        def _manhattan(a: Position, b: Position) -> int:
            az = getattr(a, "z", 0) or 0
            bz = getattr(b, "z", 0) or 0
            return abs(a.x - b.x) + abs(a.y - b.y) + abs(az - bz)

        centre = _estimate_centre(request.options[0].landing_position)

        selected: CaptureDirectionChoiceOption = request.options[0]
        best_cap = selected.captured_cap_height
        best_dist = _manhattan(selected.landing_position, centre)

        for opt in request.options[1:]:
            cap = opt.captured_cap_height
            if cap > best_cap:
                selected = opt
                best_cap = cap
                best_dist = _manhattan(opt.landing_position, centre)
                continue

            if cap == best_cap:
                dist = _manhattan(opt.landing_position, centre)
                if dist < best_dist:
                    selected = opt
                    best_dist = dist

        return CaptureDirectionChoiceResponse(
            selectedOption=selected,
            aiType=ai_type.value,
            difficulty=request.difficulty,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error selecting capture direction option: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ai/cache")
async def clear_ai_cache():
    """Clear cached AI instances"""
    global ai_instances
    with _ai_cache_lock:
        removed = len(ai_instances)
        ai_instances.clear()
        AI_INSTANCE_CACHE_SIZE.set(0)
    logger.info("AI cache cleared")
    return {"status": "cache cleared", "instances_removed": removed}


@app.get("/ai/cache/stats")
async def ai_cache_stats():
    """Return basic stats about the in-process AI instance cache."""
    now = time.time()
    with _ai_cache_lock:
        _prune_ai_cache(now)
        count = len(ai_instances)
        AI_INSTANCE_CACHE_SIZE.set(count)
    return {
        "enabled": AI_INSTANCE_CACHE_ENABLED,
        "count": count,
        "max": AI_INSTANCE_CACHE_MAX,
        "ttl_sec": AI_INSTANCE_CACHE_TTL_SEC,
    }


@app.get("/ai/models")
async def get_model_versions():
    """Return currently loaded model versions for cache-busting and monitoring.

    This endpoint enables:
    - TypeScript sandbox to detect when models are updated
    - Monitoring dashboards to track deployed model versions
    - Automatic cache invalidation when new models are promoted
    """
    import hashlib
    from pathlib import Path

    models_dir = Path(__file__).parent.parent / "models"
    versions = {}

    # Check NNUE models
    nnue_dir = models_dir / "nnue"
    if nnue_dir.exists():
        for model_file in nnue_dir.glob("*.pt"):
            try:
                stat = model_file.stat()
                with open(model_file, "rb") as f:
                    # Read first 8KB for quick hash
                    content = f.read(8192)
                    hash_prefix = hashlib.sha256(content).hexdigest()[:12]
                versions[model_file.stem] = {
                    "path": str(model_file),
                    "hash": hash_prefix,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": stat.st_mtime,
                }
            except Exception:
                pass

    # Check neural net models
    nn_dir = models_dir / "neural_net"
    if nn_dir.exists():
        for model_file in nn_dir.glob("*.pt"):
            try:
                stat = model_file.stat()
                with open(model_file, "rb") as f:
                    content = f.read(8192)
                    hash_prefix = hashlib.sha256(content).hexdigest()[:12]
                versions[model_file.stem] = {
                    "path": str(model_file),
                    "hash": hash_prefix,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": stat.st_mtime,
                }
            except Exception:
                pass

    return {
        "models": versions,
        "timestamp": time.time(),
    }


class DifficultyProfile(TypedDict):
    """
    Canonical difficulty profile for a single ladder level.

    This mapping is the Python source of truth for how difficulty 1–10 map
    onto concrete AI configurations. TypeScript mirrors the same ladder in
    its own shared module so that lobby UIs, matchmaking, and the server
    all agree on which underlying AI engine and parameters correspond to a
    given difficulty.
    """

    ai_type: AIType
    randomness: float
    think_time_ms: int
    profile_id: str
    use_neural_net: NotRequired[bool]


# v1 canonical ladder. The randomness values intentionally mirror the legacy
# mapping used by _get_randomness_for_difficulty while the think_time_ms
# budget is expressed as a per-move search-time limit rather than a UX delay.
# Search-based engines (Minimax/MCTS/Descent) interpret think_time_ms as a
# hard upper bound on wall-clock search time; simpler engines may ignore it or
# treat it as a soft search-budget hint but must not use it to delay after a
# move has been selected.
_CANONICAL_DIFFICULTY_PROFILES: Dict[int, DifficultyProfile] = {
    1: {
        # Beginner: pure random baseline
        "ai_type": AIType.RANDOM,
        "randomness": 0.5,
        "think_time_ms": 150,
        "profile_id": "v1-random-1",
        "use_neural_net": False,
    },
    2: {
        # Easy: shallow heuristic play with noticeable randomness
        "ai_type": AIType.HEURISTIC,
        "randomness": 0.3,
        "think_time_ms": 200,
        "profile_id": "v1-heuristic-2",
        "use_neural_net": False,
    },
    3: {
        # Lower-mid: minimax with heuristic evaluation only (no neural net)
        "ai_type": AIType.MINIMAX,
        "randomness": 0.15,
        "think_time_ms": 1800,
        "profile_id": "v1-minimax-3",
        "use_neural_net": False,
    },
    4: {
        # Mid: minimax with NNUE neural evaluation
        "ai_type": AIType.MINIMAX,
        "randomness": 0.08,
        "think_time_ms": 2800,
        "profile_id": "v1-minimax-4-nnue",
        "use_neural_net": True,
    },
    5: {
        # Upper-mid: MCTS with heuristic rollouts only (no neural net)
        "ai_type": AIType.MCTS,
        "randomness": 0.05,
        "think_time_ms": 4000,
        "profile_id": "v1-mcts-5",
        "use_neural_net": False,
    },
    6: {
        # High: MCTS with neural value/policy guidance
        "ai_type": AIType.MCTS,
        "randomness": 0.02,
        "think_time_ms": 5500,
        "profile_id": "v1-mcts-6-neural",
        "use_neural_net": True,
    },
    7: {
        # Expert: MCTS with neural guidance and higher budget
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 7500,
        "profile_id": "v1-mcts-7-neural",
        "use_neural_net": True,
    },
    8: {
        # Strong expert: MCTS with neural guidance and large search budget
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 9600,
        "profile_id": "v1-mcts-8-neural",
        "use_neural_net": True,
    },
    9: {
        # Master: Descent/UBFM-style search with NN guidance
        "ai_type": AIType.DESCENT,
        "randomness": 0.0,
        "think_time_ms": 12600,
        "profile_id": "v1-descent-9",
        "use_neural_net": True,
    },
    10: {
        # Grandmaster: strongest Descent configuration
        "ai_type": AIType.DESCENT,
        "randomness": 0.0,
        "think_time_ms": 16000,
        "profile_id": "v1-descent-10",
        "use_neural_net": True,
    },
}


def _get_difficulty_profile(difficulty: int) -> DifficultyProfile:
    """
    Return the canonical difficulty profile for the given ladder level.

    Difficulty is clamped into [1, 10] so that out-of-range values still map
    to a well-defined profile instead of silently diverging between callers.
    """
    if difficulty < 1:
        effective = 1
    elif difficulty > 10:
        effective = 10
    else:
        effective = difficulty

    return _CANONICAL_DIFFICULTY_PROFILES[effective]


def _select_ai_type(difficulty: int) -> AIType:
    """Auto-select AI type based on canonical difficulty mapping."""
    return _get_difficulty_profile(difficulty)["ai_type"]


def _get_randomness_for_difficulty(difficulty: int) -> float:
    """Get randomness factor for difficulty level from canonical profile."""
    return _get_difficulty_profile(difficulty)["randomness"]


def _create_ai_instance(ai_type: AIType, player_number: int, config: AIConfig):
    """Factory function to create AI instances.

    The NEURAL_DEMO branch is reserved for experimental / sandbox use and is
    gated behind the AI_ENGINE_NEURAL_DEMO_ENABLED environment variable so
    that neural-only engines cannot be enabled accidentally on production
    ladders.
    """
    if ai_type == AIType.RANDOM:
        return RandomAI(player_number, config)
    elif ai_type == AIType.HEURISTIC:
        return HeuristicAI(player_number, config)
    elif ai_type == AIType.MINIMAX:
        from .ai.minimax_ai import MinimaxAI

        return MinimaxAI(player_number, config)
    elif ai_type == AIType.MCTS:
        from .ai.mcts_ai import MCTSAI

        return MCTSAI(player_number, config)
    elif ai_type == AIType.DESCENT:
        return DescentAI(player_number, config)
    elif ai_type == AIType.NEURAL_DEMO:
        # Experimental / demo-only neural engine. This is never selected by
        # the canonical difficulty ladder and must be explicitly enabled via
        # AI_ENGINE_NEURAL_DEMO_ENABLED for safety.
        flag = os.getenv("AI_ENGINE_NEURAL_DEMO_ENABLED", "").lower()
        if flag in {"1", "true", "yes", "on"}:
            from .ai.neural_net import NeuralNetAI

            return NeuralNetAI(player_number, config)

        logger.warning(
            "AIType.NEURAL_DEMO requested but AI_ENGINE_NEURAL_DEMO_ENABLED "
            "is not set; falling back to HeuristicAI."
        )
        return HeuristicAI(player_number, config)
    else:
        # Default to heuristic
        return HeuristicAI(player_number, config)


if __name__ == "__main__":
    import uvicorn

    # When run directly (e.g. via `python -m app.main`), bind to 0.0.0.0 and
    # respect AI_SERVICE_PORT if provided so that local runs and containers
    # share the same configuration surface as docker-compose.
    port_str = os.getenv("AI_SERVICE_PORT", "8001")
    try:
        port = int(port_str)
    except ValueError:
        port = 8001

    uvicorn.run(app, host="0.0.0.0", port=port)
