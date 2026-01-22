"""
RingRift AI Service - FastAPI Application
Provides AI move selection and position evaluation endpoints
"""
from __future__ import annotations


import asyncio
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator, validator

try:
    from typing import Self  # Python 3.11+
except ImportError:  # pragma: no cover
    from typing_extensions import Self  # Python 3.10 compatibility

try:  # Python 3.10 compatibility (NotRequired added in 3.11)
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired  # type: ignore

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config.ladder_config import (
    LadderTierConfig,
    get_effective_ladder_config,
    list_ladder_tiers,
)
from .metrics import (
    AI_INSTANCE_CACHE_LOOKUPS,
    AI_INSTANCE_CACHE_SIZE,
    AI_MOVE_LATENCY,
    AI_MOVE_REQUESTS,
    observe_ai_move_start,
)

# Resource cleanup imports
try:
    from .distributed.db_utils import close_all_connections as _close_db_connections
    HAS_DB_UTILS = True
except ImportError:  # pragma: no cover
    HAS_DB_UTILS = False
    _close_db_connections = None  # type: ignore

try:
    from .ai.model_cache import clear_model_cache as _clear_model_cache
    HAS_MODEL_CACHE = True
except ImportError:  # pragma: no cover
    HAS_MODEL_CACHE = False
    _clear_model_cache = None  # type: ignore

from .ai.descent_ai import DescentAI
from .ai.heuristic_ai import HeuristicAI
from .ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES, load_trained_profiles_if_available
from .ai.random_ai import RandomAI
from .board_manager import BoardManager
from .game_engine import GameEngine
from .models import (
    AIConfig,
    AIType,
    CaptureDirectionChoiceOption,
    CaptureDirectionChoiceRequest,
    CaptureDirectionChoiceResponse,
    GameState,
    GameStatus,
    LineOrderChoiceLine,
    LineOrderChoiceRequest,
    LineOrderChoiceResponse,
    LineRewardChoiceOption,
    LineRewardChoiceRequest,
    LineRewardChoiceResponse,
    Move,
    Position,
    RegionOrderChoiceOption,
    RegionOrderChoiceRequest,
    RegionOrderChoiceResponse,
    RingEliminationChoiceRequest,
    RingEliminationChoiceResponse,
)
from .routes import include_all_routes
from .rules.default_engine import DefaultRulesEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Backwards-compatible import surface for tests and callers that patch
# `app.main.get_ladder_tier_config`. We route through the effective ladder
# config so that env overrides remain honoured in production.
def get_ladder_tier_config(
    difficulty: int,
    board_type: Any,
    num_players: int,
) -> LadderTierConfig:
    return get_effective_ladder_config(difficulty, board_type, num_players)

# Best-effort: load trained heuristic profiles (CMA-ES / calibration) when
# RINGRIFT_TRAINED_HEURISTIC_PROFILES points at a JSON bundle. This keeps the
# stable heuristic_profile_id values used by ladder_config.py while allowing
# deployments to override weights without code changes.
_trained_profiles: dict[str, Any] = {}
try:
    _trained_profiles = load_trained_profiles_if_available()
    if _trained_profiles:
        logger.info(
            "Loaded trained heuristic profiles",
            extra={"profile_count": len(_trained_profiles)},
        )
except (ImportError, FileNotFoundError, ValueError, KeyError):  # pragma: no cover - defensive startup path
    logger.warning("Failed to load trained heuristic profiles", exc_info=True)

# Import CoordinatorRegistry for graceful shutdown
try:
    from app.coordination.coordinator_base import (
        get_coordinator_registry,
        shutdown_all_coordinators,
    )
    HAS_COORDINATOR_REGISTRY = True
except ImportError:  # pragma: no cover
    HAS_COORDINATOR_REGISTRY = False
    get_coordinator_registry = None  # type: ignore
    shutdown_all_coordinators = None  # type: ignore

# Import DaemonManager for unified background service management
try:
    from app.coordination.daemon_manager import (
        DaemonManager,
        DaemonType,
        get_daemon_manager,
    )
    HAS_DAEMON_MANAGER = True
except ImportError:  # pragma: no cover
    HAS_DAEMON_MANAGER = False
    DaemonManager = None  # type: ignore
    get_daemon_manager = None  # type: ignore
    DaemonType = None  # type: ignore

# Import event helpers for unified router configuration (December 2025)
try:
    from app.distributed.event_helpers import (
        has_event_router,
        set_use_router_by_default,
    )
    HAS_EVENT_HELPERS = True
except ImportError:  # pragma: no cover
    HAS_EVENT_HELPERS = False
    set_use_router_by_default = None  # type: ignore
    has_event_router = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown.

    - On startup: Validate configs, install signal handlers, start daemon manager
    - On shutdown: Gracefully shutdown daemons, then coordinators
    """
    # Startup

    # Validate all configuration files before proceeding (December 2025)
    # Catches misconfigurations early before expensive operations begin
    try:
        from .config.config_validator import ConfigValidator
        validator = ConfigValidator()
        result = validator.validate_all()
        if not result.valid:
            for error in result.errors:
                logger.error(f"[Config] {error}")
            logger.warning(
                f"[Config] Validation found {len(result.errors)} errors, "
                f"{len(result.warnings)} warnings"
            )
        else:
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"[Config] {warning}")
            logger.info(
                f"[Config] All configurations validated successfully "
                f"({len(result.warnings)} warnings)"
            )
    except ImportError:
        logger.debug("[Config] Config validator not available")
    except Exception as e:
        logger.warning(f"[Config] Validation failed: {e}")

    # Enable unified event router for all event emissions (December 2025)
    # This routes events through EventBus + StageEventBus + CrossProcessEventQueue
    if HAS_EVENT_HELPERS and has_event_router():
        set_use_router_by_default(True)
        logger.info("Enabled unified event router for all event emissions")

    if HAS_COORDINATOR_REGISTRY:
        registry = get_coordinator_registry()
        # Install signal handlers (SIGTERM/SIGINT)
        try:
            registry.install_signal_handlers()
            logger.info("Installed coordinator signal handlers for graceful shutdown")
        except Exception as e:
            logger.warning(f"Could not install signal handlers: {e}")

    # Phase 18: Auto-start critical daemons for full automation (December 2025)
    # EVENT_ROUTER is required for all event-driven daemons
    # FeedbackLoopController orchestrates the entire SELFPLAY→TRAINING→EVALUATION→PROMOTION chain
    #
    # RINGRIFT_INFERENCE_ONLY=true: Skip all coordination daemons (January 2026)
    # Use this on production web servers that only need AI inference endpoints.
    # This prevents starting the full training cluster coordination stack which
    # causes failures when the server can't reach the training cluster.
    inference_only = os.environ.get("RINGRIFT_INFERENCE_ONLY", "").lower() in ("1", "true", "yes")
    if inference_only:
        logger.info("[Startup] INFERENCE_ONLY mode - skipping coordination daemons")

    if HAS_DAEMON_MANAGER and not inference_only:
        try:
            daemon_manager = get_daemon_manager()

            # 18.1: Always start EVENT_ROUTER - required for all event-driven daemons
            try:
                await daemon_manager.start(DaemonType.EVENT_ROUTER)
                logger.info("[Startup] EVENT_ROUTER started")
            except Exception as e:
                logger.warning(f"[Startup] Failed to start EVENT_ROUTER: {e}")

            # 18.2: Initialize FeedbackLoopController - orchestrates the full training loop
            try:
                from app.coordination.feedback_loop_controller import get_feedback_loop_controller
                controller = get_feedback_loop_controller()
                await controller.start()
                logger.info("[Startup] FeedbackLoopController started")
            except ImportError:
                logger.debug("[Startup] FeedbackLoopController not available")
            except Exception as e:
                logger.warning(f"[Startup] Failed to start FeedbackLoopController: {e}")

            # 18.3: Auto-start daemon profile based on node role
            node_role = os.environ.get("RINGRIFT_NODE_ROLE", "")
            if node_role:
                try:
                    from app.coordination.daemon_manager import DAEMON_PROFILES
                    if node_role in DAEMON_PROFILES:
                        results = await daemon_manager.start_all(DAEMON_PROFILES[node_role])
                        started = sum(1 for v in results.values() if v)
                        logger.info(f"[Startup] Started {node_role} profile: {started}/{len(results)} daemons")
                    else:
                        logger.warning(f"[Startup] Unknown node role: {node_role}")
                except Exception as e:
                    logger.warning(f"[Startup] Failed to start {node_role} profile: {e}")

            # Legacy: Start core daemons if RINGRIFT_START_DAEMONS=1
            if os.environ.get("RINGRIFT_START_DAEMONS", "0") == "1":
                core_daemons = [
                    DaemonType.NODE_HEALTH_MONITOR,  # Dec 2025: Unified health monitoring
                    DaemonType.QUEUE_MONITOR,
                ]
                for daemon_type in core_daemons:
                    try:
                        await daemon_manager.start(daemon_type)
                    except Exception as e:
                        logger.warning(f"Failed to start {daemon_type.value}: {e}")
                logger.info("Daemon manager started core daemons")

        except Exception as e:
            logger.warning(f"Could not initialize daemon manager: {e}")

    logger.info("RingRift AI Service started")

    yield  # App runs here

    # Shutdown
    logger.info("RingRift AI Service shutting down")

    # 0. Shutdown daemon manager first (background tasks)
    if HAS_DAEMON_MANAGER:
        try:
            daemon_manager = get_daemon_manager()
            await daemon_manager.shutdown()
            logger.info("Daemon manager shutdown complete")
        except Exception as e:
            logger.warning(f"Error during daemon manager shutdown: {e}")

    # 1. Shutdown coordinators (most complex resources)
    if HAS_COORDINATOR_REGISTRY:
        try:
            results = await shutdown_all_coordinators(timeout=30.0)
            if results:
                succeeded = sum(1 for v in results.values() if v)
                logger.info(f"Coordinator shutdown: {succeeded}/{len(results)} succeeded")
        except Exception as e:
            logger.error(f"Error during coordinator shutdown: {e}")

    # 2. Clear model cache to release GPU/MPS memory
    if HAS_MODEL_CACHE:
        try:
            _clear_model_cache()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing model cache: {e}")

    # 3. Close all database connections (after coordinators to avoid connection errors)
    if HAS_DB_UTILS:
        try:
            _close_db_connections()
            logger.info("Database connections closed")
        except Exception as e:
            logger.warning(f"Error closing database connections: {e}")

    logger.info("RingRift AI Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RingRift AI Service",
    description="AI move selection and evaluation service for RingRift",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allows sandbox UI to access replay API
# In production, restrict allow_origins to specific domains
_cors_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if not cors_origins:
    cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin API key for protected endpoints (cache management, etc.)
# Generate a secure default if not set, but warn in production
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    ADMIN_API_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "ADMIN_API_KEY not set - generated ephemeral key. "
        "Set ADMIN_API_KEY env var for persistent admin access."
    )


async def verify_admin_api_key(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Dependency to verify admin API key for protected endpoints."""
    if not x_admin_key:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


# Error sanitization for production - prevent stack trace leakage
IS_PRODUCTION = os.getenv("RINGRIFT_ENV", "development").lower() == "production"

# Import sanitize_error_detail from utils to avoid circular import
from app.utils.error_utils import sanitize_error_detail

# AI operation timeout (seconds) - prevents hanging requests
AI_OPERATION_TIMEOUT = float(os.getenv("RINGRIFT_AI_TIMEOUT", "30.0"))


# Mount all modular routers (replay, cluster, training, human-games, online-learning)
include_all_routes(app)

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

_ai_cache_lock = threading.RLock()
ai_instances: dict[str, CachedAIInstance] = {}


def _should_cache_ai(ai_type: AIType, game_state: GameState) -> bool:
    """Return True if this request should reuse a persistent AI instance."""
    if not AI_INSTANCE_CACHE_ENABLED:
        return False
    if ai_type not in (AIType.MINIMAX, AIType.MCTS, AIType.DESCENT, AIType.GUMBEL_MCTS):
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
    player_number: int = Field(ge=1, description="Player number (1-indexed)")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: AIType | None = None
    seed: int | None = Field(
        None,
        ge=0,
        le=0x7FFFFFFF,
        description="Optional RNG seed for deterministic AI behavior"
    )

    @root_validator(skip_on_failure=True)
    def validate_player_number(cls, values: dict) -> dict:
        """Ensure player_number is valid for the given game_state."""
        game_state = values.get("game_state")
        player_number = values.get("player_number")
        players = getattr(game_state, "players", None) if game_state else None
        if not players:
            raise ValueError("game_state.players cannot be empty")
        if player_number and player_number > len(players):
            raise ValueError(
                f"player_number {player_number} exceeds number of players ({len(players)})"
            )
        return values


class MoveResponse(BaseModel):
    """Response model for AI move selection"""
    move: Move | None
    evaluation: float
    thinking_time_ms: int
    ai_type: str
    difficulty: int
    # Optional model observability fields (backward compatible for TS clients).
    heuristic_profile_id: str | None = None
    use_neural_net: bool | None = None
    nn_model_id: str | None = None
    nn_checkpoint: str | None = None
    nnue_checkpoint: str | None = None


class BatchMoveRequest(BaseModel):
    """Request model for batch AI move selection (multiple games at once)"""
    game_states: list[GameState] = Field(
        ...,
        description="List of game states to get moves for (1-64 games)"
    )
    player_numbers: list[int] = Field(
        ...,
        description="Player number for each game state (1-indexed)"
    )
    difficulty: int = Field(ge=1, le=10, default=8)
    mode: str = Field(
        default="batch",
        description="MCTS mode: 'batch' (batched NN) or 'tensor' (full GPU tree)"
    )
    device: str = Field(
        default="auto",
        description="Device: 'auto', 'cpu', 'cuda', 'mps'"
    )
    simulation_budget: int = Field(
        ge=10,
        le=1000,
        default=64,
        description="Simulation budget per move"
    )

    @validator("game_states")
    def validate_game_states_length(cls, v):
        if len(v) < 1:
            raise ValueError("game_states must have at least 1 item")
        if len(v) > 64:
            raise ValueError("game_states cannot exceed 64 items")
        return v

    @root_validator(skip_on_failure=True)
    def validate_lengths(cls, values: dict) -> dict:
        """Ensure game_states and player_numbers have matching lengths."""
        game_states = values.get("game_states", [])
        player_numbers = values.get("player_numbers", [])
        if len(game_states) != len(player_numbers):
            raise ValueError(
                f"game_states ({len(game_states)}) and player_numbers "
                f"({len(player_numbers)}) must have the same length"
            )
        return values


class BatchMoveItem(BaseModel):
    """Single move result in batch response"""
    move: Move | None
    evaluation: float
    game_index: int


class BatchMoveResponse(BaseModel):
    """Response model for batch AI move selection"""
    moves: list[BatchMoveItem]
    total_thinking_time_ms: int
    games_processed: int
    mode: str
    throughput_games_per_sec: float


class EvaluationRequest(BaseModel):
    """Request model for position evaluation"""
    game_state: GameState
    player_number: int = Field(ge=1, description="Player number (1-indexed)")

    @root_validator(skip_on_failure=True)
    def validate_player_number(cls, values: dict) -> dict:
        """Ensure player_number is valid for the given game_state."""
        game_state = values.get("game_state")
        player_number = values.get("player_number")
        players = getattr(game_state, "players", None) if game_state else None
        if not players:
            raise ValueError("game_state.players cannot be empty")
        if player_number and player_number > len(players):
            raise ValueError(
                f"player_number {player_number} exceeds number of players ({len(players)})"
            )
        return values


class EvaluationResponse(BaseModel):
    """Response model for position evaluation"""
    score: float
    breakdown: dict[str, float]


class PositionEvaluationByPlayer(BaseModel):
    """Per-player evaluation summary for analysis mode."""

    totalEval: float
    territoryEval: float = 0.0
    ringEval: float = 0.0
    winProbability: float | None = None


class PositionEvaluationRequest(BaseModel):
    """Request model for multi-player position evaluation."""

    game_state: GameState
    # Optional label for the underlying engine profile used to evaluate.
    engine_profile: str | None = None
    # Optional explicit RNG seed for deterministic evaluations.
    random_seed: int | None = None


class PositionEvaluationResponse(BaseModel):
    """Response model for multi-player position evaluation.

    Suitable for streaming over analysis and spectator channels.
    """

    engine_profile: str
    board_type: str
    game_id: str
    move_number: int
    per_player: dict[int, PositionEvaluationByPlayer]
    evaluation_scale: str
    generated_at: str


class RulesEvalRequest(BaseModel):
    """Request model for rules-engine evaluation (parity/shadow mode)."""
    game_state: GameState
    move: Move


class RulesEvalResponse(BaseModel):
    """Response model for rules-engine evaluation."""
    valid: bool
    validation_error: str | None = None
    next_state: GameState | None = None
    state_hash: str | None = None
    s_invariant: int | None = None
    game_status: GameStatus | None = None


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
    """Health check with component status for container orchestration.

    Returns 200 if healthy, 503 if degraded/unhealthy.
    """
    from fastapi.responses import JSONResponse
    from app.distributed.health_registry import health_endpoint_handler

    result = health_endpoint_handler()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe - is service ready to handle requests?

    Returns 200 if ready, 503 if not ready.
    """
    from fastapi.responses import JSONResponse
    from app.distributed.health_registry import readiness_check as check_ready

    is_ready = check_ready()
    return JSONResponse(
        content={"ready": is_ready},
        status_code=200 if is_ready else 503,
    )


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe - is process alive?

    Always returns 200 if the process is running.
    """
    from app.distributed.health_registry import liveness_check as check_live

    return {"live": check_live()}


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


@app.get("/admin/health/coordinators")
async def admin_health_coordinators(
    _: bool = Depends(verify_admin_api_key),
) -> dict[str, Any]:
    """Get health status of all coordinator managers.

    Requires X-Admin-Key header for authentication.

    Returns detailed status of:
    - RecoveryManager: Node/job recovery tracking
    - BandwidthManager: Transfer bandwidth allocation
    - SyncCoordinator: Data synchronization across cluster
    """
    try:
        from app.metrics.coordinator import collect_all_coordinator_metrics_sync

        metrics = collect_all_coordinator_metrics_sync()
        coordinators = metrics.get("coordinators", {})

        # Build detailed response
        status_summary = {}
        all_healthy = True

        for name, stats in coordinators.items():
            coord_status = stats.get("status", "unknown")
            is_healthy = coord_status in ("ready", "running")
            if not is_healthy:
                all_healthy = False

            status_summary[name] = {
                "status": coord_status,
                "healthy": is_healthy,
                "uptime_seconds": stats.get("uptime_seconds", 0),
                "details": {k: v for k, v in stats.items() if k not in ("status", "uptime_seconds")},
            }

        return {
            "healthy": all_healthy,
            "coordinator_count": len(coordinators),
            "coordinators": status_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ImportError:
        return {
            "healthy": True,
            "coordinator_count": 0,
            "coordinators": {},
            "message": "Coordinator metrics module not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to collect coordinator health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect coordinator health: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.get("/admin/health/full")
async def admin_health_full(
    _: bool = Depends(verify_admin_api_key),
) -> dict[str, Any]:
    """Get comprehensive health check of all pipeline components.

    Requires X-Admin-Key header for authentication.

    Includes:
    - Data sync status
    - Training status
    - Evaluation status
    - Coordinator status
    - Coordinator managers status
    - System resources
    """
    try:
        from app.distributed.health_checks import get_health_summary

        summary = get_health_summary()

        return {
            "healthy": summary.healthy,
            "timestamp": summary.timestamp,
            "components": {
                c.name: {
                    "status": c.status,
                    "healthy": c.healthy,
                    "message": c.message,
                    "details": c.details,
                }
                for c in summary.components
            },
            "issues": summary.issues,
            "warnings": summary.warnings,
        }
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health summary: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.get("/admin/sync/status")
async def admin_sync_status(
    _: bool = Depends(verify_admin_api_key),
) -> dict[str, Any]:
    """Get comprehensive data synchronization status.

    Requires X-Admin-Key header for authentication.

    Returns:
    - Storage provider info (NFS, ephemeral, local)
    - Available transports (aria2, SSH, P2P, gossip)
    - Recent sync statistics
    - Gossip daemon status (if enabled)
    - Circuit breaker states
    """
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        coordinator = SyncCoordinator.get_instance()
        status = coordinator.get_status()

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sync": status,
        }
    except ImportError as e:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"SyncCoordinator not available: {e}",
        }
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sync status: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.post("/admin/sync/trigger")
async def admin_sync_trigger(
    _: bool = Depends(verify_admin_api_key),
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Manually trigger a data sync operation.

    Requires X-Admin-Key header for authentication.

    Args:
        categories: List of categories to sync ("games", "training", "models")
                   If None, syncs all categories.

    Returns:
        Sync operation results with files synced per category
    """
    try:
        from app.distributed.sync_coordinator import SyncCategory, SyncCoordinator

        coordinator = SyncCoordinator.get_instance()

        # Map string categories to enum
        if categories is None:
            cats = [SyncCategory.GAMES, SyncCategory.TRAINING, SyncCategory.MODELS]
        else:
            cat_map = {
                "games": SyncCategory.GAMES,
                "training": SyncCategory.TRAINING,
                "training_data": SyncCategory.TRAINING,
                "models": SyncCategory.MODELS,
            }
            cats = [cat_map[c] for c in categories if c in cat_map]

        # Run full sync
        stats = await coordinator.full_cluster_sync(categories=cats)

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                cat: {
                    "files_synced": s.files_synced,
                    "bytes_transferred": s.bytes_transferred,
                    "transport_used": s.transport_used,
                    "duration_seconds": s.duration_seconds,
                }
                for cat, s in stats.categories.items()
            },
            "total_files": stats.total_files_synced,
            "total_bytes": stats.total_bytes_transferred,
        }
    except ImportError as e:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"SyncCoordinator not available: {e}",
        }
    except Exception as e:
        logger.error(f"Failed to trigger sync: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger sync: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.post("/admin/sync/data-server/start")
async def admin_start_data_server(
    _: bool = Depends(verify_admin_api_key),
    port: int = 8766,
) -> dict[str, Any]:
    """Start the aria2 data server for serving files to other nodes.

    Requires X-Admin-Key header for authentication.

    Args:
        port: Port to serve on (default: 8766)

    Returns:
        Server status
    """
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        coordinator = SyncCoordinator.get_instance()
        success = await coordinator.start_data_server(port=port)

        return {
            "status": "started" if success else "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "port": port if success else None,
        }
    except ImportError as e:
        return {
            "status": "unavailable",
            "error": f"SyncCoordinator not available: {e}",
        }
    except Exception as e:
        logger.error(f"Failed to start data server: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start data server: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.post("/admin/sync/data-server/stop")
async def admin_stop_data_server(
    _: bool = Depends(verify_admin_api_key),
) -> dict[str, Any]:
    """Stop the aria2 data server.

    Requires X-Admin-Key header for authentication.
    """
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        coordinator = SyncCoordinator.get_instance()
        await coordinator.stop_data_server()

        return {
            "status": "stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except ImportError as e:
        return {
            "status": "unavailable",
            "error": f"SyncCoordinator not available: {e}",
        }
    except Exception as e:
        logger.error(f"Failed to stop data server: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop data server: {e}" if not IS_PRODUCTION else "Internal error",
        )


@app.get("/admin/velocity")
async def admin_velocity_dashboard(
    _: bool = Depends(verify_admin_api_key),
) -> dict[str, Any]:
    """Get Elo velocity dashboard for all configurations.

    Requires X-Admin-Key header for authentication.

    Returns velocity (Elo points/day) and ETA to 2000 Elo for each config.
    """
    import sqlite3
    from pathlib import Path

    TARGET_ELO = 2000.0

    # Find Elo database
    candidates = [
        Path(__file__).parent.parent / "data" / "unified_elo.db",
        Path("/lambda/nfs/RingRift/elo/unified_elo.db"),
    ]
    db_path = next((c for c in candidates if c.exists()), None)

    if not db_path:
        return {"error": "Elo database not found", "configs": []}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get best Elo per config with history for velocity calculation
        cursor.execute("""
            SELECT board_type, num_players, MAX(rating) as best_elo,
                   participant_id, games_played
            FROM elo_ratings
            WHERE archived_at IS NULL
            GROUP BY board_type, num_players
        """)
        rows = cursor.fetchall()
        conn.close()

        configs = []
        for board_type, num_players, best_elo, model_id, games in rows:
            gap = max(0, TARGET_ELO - best_elo)
            # Velocity would need historical data - for now show current state
            configs.append({
                "config": f"{board_type}_{num_players}p",
                "current_elo": round(best_elo, 1),
                "target_elo": TARGET_ELO,
                "gap": round(gap, 1),
                "games_played": games or 0,
                "best_model": model_id,
                "target_met": best_elo >= TARGET_ELO,
            })

        # Sort by gap (smallest first - closest to target)
        configs.sort(key=lambda x: x["gap"])

        met_count = sum(1 for c in configs if c["target_met"])

        return {
            "target_elo": TARGET_ELO,
            "total_configs": len(configs),
            "configs_met": met_count,
            "configs_unmet": len(configs) - met_count,
            "configs": configs,
        }
    except Exception as e:
        logger.error(f"Failed to get velocity dashboard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get velocity data: {e}" if not IS_PRODUCTION else "Internal error",
        )


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

    ladder_config: LadderTierConfig | None = None
    try:
        board_type = getattr(request.game_state, "board_type", None)
        players = getattr(request.game_state, "players", None)
        num_players = len(players) if players is not None else None
        if board_type is not None and num_players:
            ladder_config = get_ladder_tier_config(
                request.difficulty,
                board_type,
                num_players,
            )
    except (ValueError, KeyError, AttributeError, TypeError):
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
        use_neural_net = (
            ladder_config.use_neural_net
            if ladder_config is not None
            else bool(profile.get("use_neural_net", False))
        )

        heuristic_profile_id: str | None = None
        nn_model_id: str | None = None

        if ladder_config is not None and ladder_config.heuristic_profile_id:
            heuristic_profile_id = ladder_config.heuristic_profile_id
        elif ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")

        if use_neural_net and ladder_config is not None and ladder_config.model_id:
            nn_model_id = ladder_config.model_id

        if ai_type not in (
            AIType.MINIMAX,
            AIType.MCTS,
            AIType.DESCENT,
            AIType.GUMBEL_MCTS,
            AIType.NEURAL_DEMO,
        ):
            # Defensive: ignore model ids for non-neural engines.
            nn_model_id = None
            use_neural_net = False

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

        # Enable GPU tree acceleration for Gumbel MCTS (177x speedup)
        use_gpu_tree = ai_type == AIType.GUMBEL_MCTS
        config = AIConfig(
            difficulty=request.difficulty,
            randomness=randomness,
            think_time=think_time_ms,
            rngSeed=effective_seed,
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
            use_neural_net=use_neural_net,
            use_gpu_tree=use_gpu_tree,
            gpu_tree_eval_mode="hybrid",  # Balance speed and accuracy
        )
        ai = None
        cache_key: str | None = None
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

        # Get move from AI with timeout protection
        try:
            move = await asyncio.wait_for(
                asyncio.to_thread(ai.select_move, request.game_state),
                timeout=AI_OPERATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"AI move selection timed out after {AI_OPERATION_TIMEOUT}s"
            )

        # Evaluate position with timeout protection
        try:
            evaluation = await asyncio.wait_for(
                asyncio.to_thread(ai.evaluate_position, request.game_state),
                timeout=AI_OPERATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Use a default evaluation if timeout - move was already selected
            evaluation = 0.0
            logger.warning("AI position evaluation timed out, using default")

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

        nn_checkpoint: str | None = None
        try:
            neural_net = getattr(ai, "neural_net", None)
            path = getattr(neural_net, "loaded_checkpoint_path", None) if neural_net is not None else None
            if path:
                nn_checkpoint = os.path.basename(str(path))
        except (AttributeError, TypeError, OSError):
            nn_checkpoint = None

        nnue_checkpoint: str | None = None
        try:
            nnue_eval = getattr(ai, "nnue_evaluator", None)
            nnue_model = getattr(nnue_eval, "model", None) if nnue_eval is not None else None
            path = getattr(nnue_model, "loaded_checkpoint_path", None) if nnue_model is not None else None
            if path:
                nnue_checkpoint = os.path.basename(str(path))
        except (AttributeError, TypeError, OSError):
            nnue_checkpoint = None

        # Reflect the effective (not merely requested) neural backend usage in
        # the API response. Search-based agents may gracefully fall back to
        # heuristic evaluation when checkpoints are unavailable, and callers
        # (including the sandbox host) should be able to distinguish that case
        # without inferring it indirectly from checkpoint names.
        effective_use_neural_net = False
        try:
            if ai_type == AIType.MINIMAX:
                effective_use_neural_net = bool(getattr(ai, "use_nnue", False))
            elif ai_type in (AIType.MCTS, AIType.DESCENT, AIType.GUMBEL_MCTS, AIType.NEURAL_DEMO):
                effective_use_neural_net = getattr(ai, "neural_net", None) is not None
            else:
                effective_use_neural_net = False
        except (AttributeError, TypeError):
            # Preserve backward compatible behaviour on unexpected AI types.
            effective_use_neural_net = bool(use_neural_net)

        return MoveResponse(
            move=move,
            evaluation=evaluation,
            thinking_time_ms=thinking_time,
            ai_type=ai_type.value,
            difficulty=request.difficulty,
            heuristic_profile_id=heuristic_profile_id,
            use_neural_net=effective_use_neural_net,
            nn_model_id=nn_model_id,
            nn_checkpoint=nn_checkpoint,
            nnue_checkpoint=nnue_checkpoint,
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
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@app.post("/ai/moves_batch", response_model=BatchMoveResponse)
async def get_ai_moves_batch(request: BatchMoveRequest):
    """
    Get AI-selected moves for multiple game states in a single batched call.

    This endpoint uses GPU-accelerated batch processing for maximum throughput.
    Ideal for:
    - Gauntlet evaluation (running many games in parallel)
    - Tournament simulations
    - Multi-game spectator mode

    Modes:
    - 'batch': BatchedGumbelMCTS - Batched NN evaluation, 3-4x speedup
    - 'tensor': MultiTreeMCTS - Full GPU tree, 6,671x speedup (best for selfplay)

    Args:
        request: BatchMoveRequest with list of game states and configuration.

    Returns:
        BatchMoveResponse with moves for all games and throughput metrics.
    """
    start_time = time.time()

    try:
        from app.ai.factory import create_mcts

        # Determine board type from first game state
        board_type = "square8"  # Default
        if request.game_states:
            first_board = getattr(request.game_states[0], "board", None)
            if first_board:
                board_type_attr = getattr(first_board, "type", None)
                if board_type_attr:
                    board_type = board_type_attr.value if hasattr(board_type_attr, "value") else str(board_type_attr)

        # Determine number of players
        num_players = 2  # Default
        if request.game_states:
            players = getattr(request.game_states[0], "players", None)
            if players:
                num_players = len(players)

        # Validate mode
        mode = request.mode.lower()
        if mode not in ("batch", "tensor"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Must be 'batch' or 'tensor'."
            )

        # Create the appropriate MCTS instance
        mcts = create_mcts(
            board_type=board_type,
            num_players=num_players,
            mode=mode,
            device=request.device,
            simulation_budget=request.simulation_budget,
            batch_size=len(request.game_states),
            eval_mode="heuristic",  # Fast mode for batch processing
        )

        # Get moves based on mode
        if mode == "tensor":
            # MultiTreeMCTS returns (moves, policies)
            moves_result, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    mcts.search_batch,
                    request.game_states,
                    None,  # neural_net=None for heuristic mode
                ),
                timeout=AI_OPERATION_TIMEOUT * 2,  # Allow more time for batch
            )
            moves = moves_result
        else:
            # BatchedGumbelMCTS returns list of moves
            moves = await asyncio.wait_for(
                asyncio.to_thread(
                    mcts.select_moves_batch,
                    request.game_states,
                ),
                timeout=AI_OPERATION_TIMEOUT * 2,
            )

        thinking_time_ms = int((time.time() - start_time) * 1000)
        games_processed = len(request.game_states)
        throughput = games_processed / max(0.001, time.time() - start_time)

        # Build response items
        move_items = []
        for idx, move in enumerate(moves):
            move_items.append(BatchMoveItem(
                move=move,
                evaluation=0.0,  # Batch mode doesn't compute per-game evaluation
                game_index=idx,
            ))

        logger.info(
            "Batch AI moves: mode=%s, games=%d, time=%dms, throughput=%.1f games/sec",
            mode,
            games_processed,
            thinking_time_ms,
            throughput,
        )

        return BatchMoveResponse(
            moves=move_items,
            total_thinking_time_ms=thinking_time_ms,
            games_processed=games_processed,
            mode=mode,
            throughput_games_per_sec=throughput,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Batch AI move selection timed out after {AI_OPERATION_TIMEOUT * 2}s"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating batch AI moves: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
        logger.error(f"Error evaluating position: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


def _select_effective_eval_seed(
    game_state: GameState,
    explicit_seed: int | None,
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

        raw_total: dict[int, float] = {}
        raw_territory: dict[int, float] = {}
        raw_rings: dict[int, float] = {}

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

        def _to_zero_sum(source: dict[int, float]) -> dict[int, float]:
            if not source:
                return {}
            count = float(len(source))
            mean = float(sum(source.values())) / count
            return {p: float(v - mean) for p, v in source.items()}

        total_zero_sum = _to_zero_sum(raw_total)
        territory_zero_sum = _to_zero_sum(raw_territory)
        rings_zero_sum = _to_zero_sum(raw_rings)

        per_player: dict[int, PositionEvaluationByPlayer] = {}
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
            board_type_str = str(board_type.value)
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
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
            f"Error selecting line reward option: {e!s}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
            if opt.cap_height < selected.cap_height or (
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
            f"Error selecting ring elimination option: {e!s}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
            f"Error selecting region order option: {e!s}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


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
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@app.delete("/ai/cache", dependencies=[Depends(verify_admin_api_key)])
async def clear_ai_cache():
    """Clear cached AI instances. Requires X-Admin-Key header."""
    with _ai_cache_lock:
        removed = len(ai_instances)
        ai_instances.clear()
        AI_INSTANCE_CACHE_SIZE.set(0)
    logger.info("AI cache cleared via admin API", extra={"instances_removed": removed})
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
            except (FileNotFoundError, OSError, PermissionError):
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
            except (FileNotFoundError, OSError, PermissionError):
                pass

    return {
        "models": versions,
        "timestamp": time.time(),
    }


def _safe_file_stat(path: "Path") -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "basename": path.name,
            "exists": True,
            "size_bytes": int(stat.st_size),
            "modified": float(stat.st_mtime),
        }
    except FileNotFoundError:
        return {"basename": path.name, "exists": False}
    except Exception as exc:  # pragma: no cover - defensive
        return {"basename": path.name, "exists": False, "error": str(exc)}


def _resolve_latest_checkpoint(models_dir: "Path", model_id: str) -> dict[str, Any]:
    """Resolve the latest matching .pth checkpoint for a model id prefix."""
    patterns = [
        f"{model_id}.pth",
        f"{model_id}_mps.pth",
        f"{model_id}_*.pth",
        f"{model_id}_*_mps.pth",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(models_dir.glob(pattern))

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in matches:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    def _mtime(path: "Path") -> float:
        try:
            return float(path.stat().st_mtime)
        except (FileNotFoundError, OSError, PermissionError):
            return 0.0

    unique.sort(key=_mtime)

    chosen = unique[-1] if unique else None
    return {
        "model_id": model_id,
        "match_count": len(unique),
        "chosen": _safe_file_stat(chosen) if chosen is not None else None,
    }


@app.get("/internal/ladder/health")
async def ladder_health(
    board_type: str | None = None,
    num_players: int | None = None,
    difficulty: int | None = None,
) -> dict[str, Any]:
    """Report effective ladder tiers and artifact availability.

    This endpoint is designed for validating training→promotion→deployment
    wiring. It does not load models; it only reports configuration + file
    presence metadata.
    """
    from pathlib import Path

    from .ai.nnue import get_nnue_model_path

    base_tiers = list_ladder_tiers()

    requested_board = board_type.strip().lower() if board_type else None
    if requested_board:
        allowed = {"square8", "square19", "hexagonal"}
        if requested_board not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported board_type={board_type!r}; expected one of {sorted(allowed)}",
            )

    if num_players is not None and num_players not in (2, 3, 4):
        raise HTTPException(
            status_code=400,
            detail="num_players must be 2, 3, or 4",
        )

    if difficulty is not None and (difficulty < 1 or difficulty > 10):
        raise HTTPException(
            status_code=400,
            detail="difficulty must be between 1 and 10",
        )

    models_dir = Path(__file__).resolve().parent.parent / "models"
    nnue_dir = models_dir / "nnue"

    tiers: list[dict[str, Any]] = []
    overrides_count = 0
    missing_profiles = 0
    missing_nnue = 0
    missing_nn = 0

    combos = sorted({(t.board_type, t.num_players) for t in base_tiers}, key=lambda x: (x[0].value, x[1]))

    def _board_matches(tier_board_value: str) -> bool:
        return requested_board is None or tier_board_value.lower() == requested_board

    def _players_matches(players: int) -> bool:
        return num_players is None or players == num_players

    def _difficulty_matches(level: int) -> bool:
        return difficulty is None or level == difficulty

    # Synthesize D1 entries so callers can validate the full 1–10 ladder surface.
    if _difficulty_matches(1):
        profile = _get_difficulty_profile(1)
        for bt, np in combos:
            if not (_board_matches(bt.value) and _players_matches(np)):
                continue
            tiers.append(
                {
                    "difficulty": 1,
                    "board_type": bt.value,
                    "num_players": np,
                    "ai_type": profile["ai_type"].value,
                    "use_neural_net": bool(profile.get("use_neural_net", False)),
                    "randomness": float(profile["randomness"]),
                    "think_time_ms": int(profile["think_time_ms"]),
                    "model_id": None,
                    "heuristic_profile_id": None,
                    "overridden": False,
                    "artifacts": {},
                }
            )

    for base in base_tiers:
        if not _difficulty_matches(base.difficulty):
            continue
        if not (_board_matches(base.board_type.value) and _players_matches(base.num_players)):
            continue

        effective = get_effective_ladder_config(base.difficulty, base.board_type, base.num_players)
        overridden = (
            effective.model_id != base.model_id
            or effective.heuristic_profile_id != base.heuristic_profile_id
        )
        if overridden:
            overrides_count += 1

        artifacts: dict[str, Any] = {}

        heuristic_profile_id = effective.heuristic_profile_id
        if heuristic_profile_id:
            available = heuristic_profile_id in HEURISTIC_WEIGHT_PROFILES
            if not available:
                missing_profiles += 1
            artifacts["heuristic_profile"] = {
                "id": heuristic_profile_id,
                "available": available,
                "source": "trained" if heuristic_profile_id in _trained_profiles else "built_in",
            }

        if effective.ai_type == AIType.MINIMAX and effective.use_neural_net:
            nnue_model_id = effective.model_id
            nnue_path = get_nnue_model_path(
                board_type=effective.board_type,
                num_players=effective.num_players,
                model_id=nnue_model_id,
            )
            nnue_stat = _safe_file_stat(nnue_path)
            if not nnue_stat.get("exists"):
                missing_nnue += 1
            artifacts["nnue"] = {
                "model_id": nnue_model_id,
                "file": nnue_stat,
                "root_present": nnue_dir.exists(),
            }

        if (
            effective.ai_type in (AIType.MCTS, AIType.DESCENT, AIType.GUMBEL_MCTS)
            and effective.use_neural_net
            and effective.model_id
        ):
            resolved = _resolve_latest_checkpoint(models_dir, effective.model_id)
            chosen = resolved.get("chosen")
            if not chosen or not chosen.get("exists"):
                missing_nn += 1
            artifacts["neural_net"] = resolved

        tiers.append(
            {
                "difficulty": effective.difficulty,
                "board_type": effective.board_type.value,
                "num_players": effective.num_players,
                "ai_type": effective.ai_type.value,
                "use_neural_net": bool(effective.use_neural_net),
                "randomness": float(effective.randomness),
                "think_time_ms": int(effective.think_time_ms),
                "model_id": effective.model_id,
                "heuristic_profile_id": effective.heuristic_profile_id,
                "overridden": overridden,
                "artifacts": artifacts,
            }
        )

    tiers.sort(key=lambda t: (t["board_type"], t["num_players"], t["difficulty"]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "board_type": requested_board,
            "num_players": num_players,
            "difficulty": difficulty,
        },
        "summary": {
            "tiers": len(tiers),
            "overridden_tiers": overrides_count,
            "missing_heuristic_profiles": missing_profiles,
            "missing_nnue_checkpoints": missing_nnue,
            "missing_neural_checkpoints": missing_nn,
        },
        "tiers": tiers,
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
_CANONICAL_DIFFICULTY_PROFILES: dict[int, DifficultyProfile] = {
    1: {
        # Beginner: pure random baseline
        "ai_type": AIType.RANDOM,
        "randomness": 0.5,
        "think_time_ms": 150,
        "profile_id": "v1-random-1",
        "use_neural_net": False,
    },
    2: {
        "ai_type": AIType.HEURISTIC,
        "randomness": 0.3,
        "think_time_ms": 200,
        "profile_id": "v1-heuristic-2",
        "use_neural_net": False,
    },
    3: {
        "ai_type": AIType.MINIMAX,
        "randomness": 0.15,
        "think_time_ms": 1800,
        "profile_id": "v1-minimax-3",
        "use_neural_net": False,
    },
    4: {
        "ai_type": AIType.MINIMAX,
        "randomness": 0.08,
        "think_time_ms": 2800,
        "profile_id": "v1-minimax-4-nnue",
        "use_neural_net": True,
    },
    5: {
        "ai_type": AIType.DESCENT,
        "randomness": 0.05,
        "think_time_ms": 4000,
        "profile_id": "ringrift_best_sq8_2p",
        "use_neural_net": True,
    },
    6: {
        "ai_type": AIType.DESCENT,
        "randomness": 0.02,
        "think_time_ms": 5500,
        "profile_id": "ringrift_best_sq8_2p",
        "use_neural_net": True,
    },
    7: {
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 7500,
        "profile_id": "v1-mcts-7",
        "use_neural_net": False,
    },
    8: {
        "ai_type": AIType.MCTS,
        "randomness": 0.0,
        "think_time_ms": 9600,
        "profile_id": "ringrift_best_sq8_2p",
        "use_neural_net": True,
    },
    9: {
        "ai_type": AIType.GUMBEL_MCTS,
        "randomness": 0.0,
        "think_time_ms": 12600,
        "profile_id": "ringrift_best_sq8_2p",
        "use_neural_net": True,
    },
    10: {
        "ai_type": AIType.GUMBEL_MCTS,
        "randomness": 0.0,
        "think_time_ms": 16000,
        "profile_id": "ringrift_best_sq8_2p",
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
    elif ai_type == AIType.GPU_MINIMAX:
        from .ai.gpu_minimax_ai import GPUMinimaxAI

        return GPUMinimaxAI(player_number, config)
    elif ai_type == AIType.MAXN:
        from .ai.maxn_ai import MaxNAI

        return MaxNAI(player_number, config)
    elif ai_type == AIType.BRS:
        from .ai.maxn_ai import BRSAI

        return BRSAI(player_number, config)
    elif ai_type == AIType.MCTS:
        from .ai.mcts_ai import MCTSAI

        return MCTSAI(player_number, config)
    elif ai_type == AIType.GUMBEL_MCTS:
        from .ai.gumbel_mcts_ai import GumbelMCTSAI

        return GumbelMCTSAI(player_number, config)
    elif ai_type == AIType.DESCENT:
        return DescentAI(player_number, config)
    elif ai_type == AIType.IG_GMO:
        from .ai.ig_gmo import IGGMO

        return IGGMO(player_number, config)
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
