"""FastAPI router for game replay endpoints.

Provides REST API for browsing, querying, and replaying games stored in the
GameReplayDB SQLite database. Used by the sandbox UI replay panel.

See docs/archive/plans/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md for specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.db.game_replay import GameReplayDB
from app.utils.error_utils import sanitize_error_detail

logger = logging.getLogger(__name__)

# Default database path - can be overridden via environment variable
DEFAULT_DB_PATH = os.getenv(
    "GAME_REPLAY_DB_PATH",
    "data/games/selfplay.db",
)

# Singleton DB instance (lazy-loaded)
_db_instance: GameReplayDB | None = None


def get_replay_db() -> GameReplayDB:
    """Get or create the replay database instance."""
    global _db_instance
    if _db_instance is None:
        db_path = os.getenv("GAME_REPLAY_DB_PATH", DEFAULT_DB_PATH)
        logger.info(f"Initializing GameReplayDB at {db_path}")
        _db_instance = GameReplayDB(db_path)
    return _db_instance


def reset_replay_db() -> None:
    """Reset the DB instance (for testing)."""
    global _db_instance
    _db_instance = None


# =============================================================================
# Request/Response Models
# =============================================================================


class PlayerMetadata(BaseModel):
    """Per-player metadata in game list."""

    playerNumber: int
    playerType: str
    aiType: str | None = None
    aiDifficulty: int | None = None
    finalEliminatedRings: int | None = None
    finalTerritorySpaces: int | None = None
    finalRingsInHand: int | None = None


class GameMetadata(BaseModel):
    """Game metadata returned in list/detail endpoints."""

    gameId: str
    boardType: str
    numPlayers: int
    winner: int | None = None
    terminationReason: str | None = None
    totalMoves: int
    totalTurns: int
    createdAt: str
    completedAt: str | None = None
    durationMs: int | None = None
    source: str | None = None
    # v2 fields
    timeControlType: str | None = None
    initialTimeMs: int | None = None
    timeIncrementMs: int | None = None
    # v5+: full recording metadata decoded from games.metadata_json
    metadata: dict[str, Any] | None = None
    # Player details (included when fetching single game)
    players: list[PlayerMetadata] | None = None


class GameListResponse(BaseModel):
    """Response for game list queries."""

    games: list[GameMetadata]
    total: int
    hasMore: bool


class MoveRecord(BaseModel):
    """A single move with all metadata."""

    moveNumber: int
    turnNumber: int
    player: int
    phase: str
    moveType: str
    move: dict[str, Any]
    timestamp: str | None = None
    thinkTimeMs: int | None = None
    # v2 fields
    timeRemainingMs: int | None = None
    engineEval: float | None = None
    engineEvalType: str | None = None
    engineDepth: int | None = None
    engineNodes: int | None = None
    enginePV: list[str] | None = None
    engineTimeMs: int | None = None


class MovesResponse(BaseModel):
    """Response for move list queries."""

    moves: list[MoveRecord]
    hasMore: bool


class ReplayStateResponse(BaseModel):
    """Response for state-at-move queries."""

    gameState: dict[str, Any]
    moveNumber: int
    totalMoves: int
    engineEval: float | None = None
    enginePV: list[str] | None = None


class ChoiceRecord(BaseModel):
    """A player choice record."""

    choiceType: str
    player: int
    options: list[dict[str, Any]]
    selected: dict[str, Any]
    reasoning: str | None = None


class ChoicesResponse(BaseModel):
    """Response for choices queries."""

    choices: list[ChoiceRecord]


class StatsResponse(BaseModel):
    """Database statistics response."""

    totalGames: int
    gamesByBoardType: dict[str, int]
    gamesByStatus: dict[str, int]
    gamesByTermination: dict[str, int]
    totalMoves: int
    schemaVersion: int


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/replay", tags=["replay"])


@router.get("/games", response_model=GameListResponse)
async def list_games(
    board_type: str | None = Query(None, max_length=50, description="Filter by board type"),
    num_players: int | None = Query(None, ge=2, le=4, description="Filter by player count"),
    winner: int | None = Query(None, ge=1, le=4, description="Filter by winning player"),
    termination_reason: str | None = Query(None, max_length=50, description="Filter by termination reason"),
    source: str | None = Query(None, max_length=100, description="Filter by game source"),
    min_moves: int | None = Query(None, ge=0, le=100000, description="Minimum move count"),
    max_moves: int | None = Query(None, ge=0, le=100000, description="Maximum move count"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, le=1000000, description="Offset for pagination"),
):
    """List games with optional filters.

    Returns paginated list of games matching the specified criteria.
    Games are ordered by creation date (newest first).
    """
    try:
        db = get_replay_db()

        # Build filter kwargs
        filters: dict[str, Any] = {}
        if board_type:
            filters["board_type"] = board_type
        if num_players:
            filters["num_players"] = num_players
        if winner is not None:
            filters["winner"] = winner
        if termination_reason:
            filters["termination_reason"] = termination_reason
        if source:
            filters["source"] = source
        if min_moves is not None:
            filters["min_moves"] = min_moves
        if max_moves is not None:
            filters["max_moves"] = max_moves

        # Fetch one extra to determine hasMore
        # Wrap blocking SQLite calls in asyncio.to_thread to prevent event loop blocking
        games = await asyncio.to_thread(
            db.query_games, **filters, limit=limit + 1, offset=offset
        )
        has_more = len(games) > limit
        games = games[:limit]

        # Get total count for this filter set
        total = await asyncio.to_thread(db.get_game_count, **filters)

        # Convert to response format
        game_list = []
        for g in games:
            raw_metadata_json = g.get("metadata_json")
            decoded_metadata: dict[str, Any] | None
            if raw_metadata_json:
                try:
                    decoded_val = json.loads(raw_metadata_json)
                    decoded_metadata = decoded_val if isinstance(decoded_val, dict) else None
                except json.JSONDecodeError:
                    decoded_metadata = None
            else:
                decoded_metadata = None

            game_list.append(
                GameMetadata(
                    gameId=g["game_id"],
                    boardType=g["board_type"],
                    numPlayers=g["num_players"],
                    winner=g.get("winner"),
                    terminationReason=g.get("termination_reason"),
                    totalMoves=g["total_moves"],
                    totalTurns=g["total_turns"],
                    createdAt=g["created_at"],
                    completedAt=g.get("completed_at"),
                    durationMs=g.get("duration_ms"),
                    source=g.get("source"),
                    timeControlType=g.get("time_control_type"),
                    initialTimeMs=g.get("initial_time_ms"),
                    timeIncrementMs=g.get("time_increment_ms"),
                    metadata=decoded_metadata,
                )
            )

        return GameListResponse(games=game_list, total=total, hasMore=has_more)

    except Exception as e:
        logger.error(f"Error listing games: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@router.get("/games/{game_id}", response_model=GameMetadata)
async def get_game(game_id: str):
    """Get detailed metadata for a specific game including player info."""
    try:
        db = get_replay_db()
        # Wrap blocking SQLite call in asyncio.to_thread to prevent event loop blocking
        game = await asyncio.to_thread(db.get_game_with_players, game_id)

        if game is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Convert player data
        players = []
        for p in game.get("players", []):
            players.append(
                PlayerMetadata(
                    playerNumber=p["playerNumber"],
                    playerType=p["playerType"],
                    aiType=p.get("aiType"),
                    aiDifficulty=p.get("aiDifficulty"),
                    finalEliminatedRings=p.get("finalEliminatedRings"),
                    finalTerritorySpaces=p.get("finalTerritorySpaces"),
                    finalRingsInHand=p.get("finalRingsInHand"),
                )
            )

        raw_metadata_json = game.get("metadata_json")
        decoded_metadata: dict[str, Any] | None
        if raw_metadata_json:
            try:
                decoded_val = json.loads(raw_metadata_json)
                decoded_metadata = decoded_val if isinstance(decoded_val, dict) else None
            except json.JSONDecodeError:
                decoded_metadata = None
        else:
            decoded_metadata = None

        return GameMetadata(
            gameId=game["game_id"],
            boardType=game["board_type"],
            numPlayers=game["num_players"],
            winner=game.get("winner"),
            terminationReason=game.get("termination_reason"),
            totalMoves=game["total_moves"],
            totalTurns=game["total_turns"],
            createdAt=game["created_at"],
            completedAt=game.get("completed_at"),
            durationMs=game.get("duration_ms"),
            source=game.get("source"),
            timeControlType=game.get("time_control_type"),
            initialTimeMs=game.get("initial_time_ms"),
            timeIncrementMs=game.get("time_increment_ms"),
            metadata=decoded_metadata,
            players=players,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@router.get("/games/{game_id}/state", response_model=ReplayStateResponse)
async def get_state_at_move(
    game_id: str,
    move_number: int = Query(0, ge=0, description="Move number (0 = initial state)"),
    legacy: bool = Query(
        False,
        description=(
            "Enable legacy replay phase injection for non-canonical records. "
            "Use only for historical/legacy DBs."
        ),
    ),
):
    """Get reconstructed game state at a specific move.

    Uses snapshots for fast reconstruction when available.
    """
    try:
        db = get_replay_db()

        # Check game exists - wrap blocking SQLite call
        meta = await asyncio.to_thread(db.get_game_metadata, game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        total_moves = meta["total_moves"]

        # Validate move number
        if move_number > total_moves:
            raise HTTPException(
                status_code=400,
                detail=f"Move number {move_number} exceeds total moves {total_moves}",
            )

        # Get state at move - wrap blocking SQLite calls
        if move_number == 0:
            state = await asyncio.to_thread(db.get_initial_state, game_id)
        else:
            if legacy:
                state = await asyncio.to_thread(db.get_state_at_move_legacy, game_id, move_number - 1)
            else:
                state = await asyncio.to_thread(db.get_state_at_move, game_id, move_number - 1)

        if state is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reconstruct state at move {move_number}",
            )

        # Get engine eval for this move if available
        engine_eval = None
        engine_pv = None
        if move_number > 0:
            move_records = await asyncio.to_thread(
                db.get_move_records, game_id, start=move_number - 1, end=move_number
            )
            if move_records:
                engine_eval = move_records[0].get("engineEval")
                engine_pv = move_records[0].get("enginePV")

        return ReplayStateResponse(
            gameState=state.model_dump(by_alias=True),
            moveNumber=move_number,
            totalMoves=total_moves,
            engineEval=engine_eval,
            enginePV=engine_pv,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting state for game {game_id} at move {move_number}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@router.get("/games/{game_id}/moves", response_model=MovesResponse)
async def get_moves(
    game_id: str,
    start: int = Query(0, ge=0, description="Start move number (inclusive)"),
    end: int | None = Query(None, ge=0, description="End move number (exclusive)"),
    limit: int = Query(100, ge=1, le=1000, description="Max moves to return"),
):
    """Get moves for a game in a range.

    Returns move records with full metadata including v2 engine evaluation fields.
    """
    try:
        db = get_replay_db()

        # Check game exists - wrap blocking SQLite call
        meta = await asyncio.to_thread(db.get_game_metadata, game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Compute effective end
        effective_end = end if end is not None else start + limit

        # Fetch moves - wrap blocking SQLite call
        move_records = await asyncio.to_thread(
            db.get_move_records, game_id, start=start, end=effective_end
        )

        # Determine if there are more
        has_more = effective_end < meta["total_moves"]

        # Convert to response format
        moves = [MoveRecord(**r) for r in move_records]

        return MovesResponse(moves=moves, hasMore=has_more)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting moves for game {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@router.get("/games/{game_id}/choices", response_model=ChoicesResponse)
async def get_choices(
    game_id: str,
    move_number: int = Query(..., ge=0, description="Move number to get choices for"),
):
    """Get player choices made at a specific move."""
    try:
        db = get_replay_db()

        # Check game exists - wrap blocking SQLite call
        meta = await asyncio.to_thread(db.get_game_metadata, game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Wrap blocking SQLite call
        choices = await asyncio.to_thread(db.get_choices_at_move, game_id, move_number)

        choice_records = [
            ChoiceRecord(
                choiceType=c["choice_type"],
                player=c["player"],
                options=c["options"],
                selected=c["selected"],
                reasoning=c.get("reasoning"),
            )
            for c in choices
        ]

        return ChoicesResponse(choices=choice_records)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting choices for game {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    try:
        db = get_replay_db()
        # Wrap blocking SQLite call
        stats = await asyncio.to_thread(db.get_stats)

        return StatsResponse(
            totalGames=stats["total_games"],
            gamesByBoardType=stats["games_by_board_type"],
            gamesByStatus=stats["games_by_status"],
            gamesByTermination=stats.get("games_by_termination", {}),
            totalMoves=stats["total_moves"],
            schemaVersion=stats.get("schema_version", 1),
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))


# =============================================================================
# POST Endpoint for Storing Games (from sandbox)
# =============================================================================


class StoreGameRequest(BaseModel):
    """Request to store a game from sandbox."""

    gameId: str | None = Field(None, max_length=100, description="Optional game ID (generated if not provided)")
    initialState: dict[str, Any] = Field(..., description="Initial game state")
    finalState: dict[str, Any] = Field(..., description="Final game state")
    moves: list[dict[str, Any]] = Field(..., description="List of moves (max 10000)")
    choices: list[dict[str, Any]] | None = Field(None, description="List of choices (max 1000)")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")

    @field_validator("moves")
    @classmethod
    def validate_moves_length(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(v) > 10000:
            raise ValueError("moves list cannot exceed 10000 items")
        return v

    @field_validator("choices")
    @classmethod
    def validate_choices_length(
        cls,
        v: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if v is not None and len(v) > 1000:
            raise ValueError("choices list cannot exceed 1000 items")
        return v


class StoreGameResponse(BaseModel):
    """Response after storing a game."""

    gameId: str
    totalMoves: int
    success: bool
    message: str | None = None
    acceptedForTraining: bool = False
    parityStatus: str | None = None
    deduplicated: bool = False


def _metadata_player_types(metadata: dict[str, Any]) -> list[str]:
    """Extract player types from explicit client metadata."""
    raw_types = metadata.get("playerTypes")
    if isinstance(raw_types, list):
        return [str(t).lower() for t in raw_types if t is not None]

    players_info = metadata.get("players", [])
    if isinstance(players_info, list) and players_info:
        player_types: list[str] = []
        for player in players_info:
            if not isinstance(player, dict):
                continue
            player_type = player.get("playerType") or player.get("type")
            if player_type is not None:
                player_types.append(str(player_type).lower())
        if player_types:
            return player_types

    return []


def _normalize_store_metadata(metadata: dict[str, Any], final_state: Any) -> dict[str, Any]:
    """Normalize submission metadata so training gates can reason about provenance."""
    normalized = dict(metadata)
    player_types = _metadata_player_types(normalized)
    if player_types:
        normalized.setdefault("playerTypes", player_types)

    has_human = "human" in player_types
    has_ai = "ai" in player_types
    existing_source = str(normalized.get("source") or "").lower()

    if has_human and has_ai:
        if existing_source in ("", "sandbox"):
            normalized["source"] = "human_vs_ai"
        normalized.setdefault("engine_mode", "human_vs_ai")
    elif has_human and existing_source == "":
        normalized["source"] = "human"
        normalized.setdefault("engine_mode", "human")
    else:
        normalized.setdefault("source", "sandbox")

    for player in getattr(final_state, "players", []) or []:
        player_number = getattr(player, "player_number", None)
        if player_number is None:
            continue
        if getattr(player, "type", None) == "ai":
            profile = getattr(player, "ai_profile", None)
            ai_type = getattr(profile, "ai_type", None) if profile is not None else None
            if hasattr(ai_type, "value"):
                ai_type = ai_type.value
            if ai_type:
                normalized.setdefault(f"player_{player_number}_ai_type", str(ai_type))

    return normalized


def _is_human_or_sandbox_source(source: str) -> bool:
    source = source.lower()
    return "human" in source or "sandbox" in source


def _validate_replay_sequence(initial_state: Any, moves: list[Any]) -> tuple[bool, str | None]:
    """Validate that a submitted game can be replayed by the Python canonical engine."""
    from app.game_engine import GameEngine
    from app.rules.history_contract import validate_canonical_move

    state = initial_state
    for move_index, move in enumerate(moves):
        current_phase = getattr(state, "current_phase", None)
        phase_str = current_phase.value if hasattr(current_phase, "value") else str(current_phase)
        move_type = getattr(move, "type", None)
        if hasattr(move_type, "value"):
            move_type = move_type.value
        canonical_check = validate_canonical_move(phase_str, str(move_type))
        if not canonical_check.ok:
            player = getattr(move, "player", None)
            return (
                False,
                (
                    f"move {move_index}: phase={phase_str} type={move_type} "
                    f"player={player}: {canonical_check.reason}"
                ),
            )
        try:
            state = GameEngine.apply_move(state, move, trace_mode=True)
        except Exception as exc:
            player = getattr(move, "player", None)
            return (
                False,
                f"move {move_index}: type={move_type} player={player}: {exc}",
            )
    return True, None


def _accepted_for_training(source: str | None, parity_status: str | None) -> bool:
    source_value = str(source or "").lower()
    if not _is_human_or_sandbox_source(source_value):
        return True
    return str(parity_status or "").lower() in {"passed", "canonical_history_ok"}


@router.post("/games", response_model=StoreGameResponse)
async def store_game(request: StoreGameRequest):
    """Store a game from the sandbox.

    Used by the sandbox UI to persist AI vs AI games to the database.
    """
    try:
        import uuid

        from app.models import GameState, Move

        db = get_replay_db()

        game_id = request.gameId or str(uuid.uuid4())
        existing = await asyncio.to_thread(db.get_game_metadata, game_id)
        if existing is not None:
            source_type = existing.get("source")
            parity_status = existing.get("parity_status")
            return StoreGameResponse(
                gameId=game_id,
                totalMoves=int(existing.get("total_moves") or 0),
                success=True,
                message="Game was already recorded; duplicate submission ignored.",
                acceptedForTraining=_accepted_for_training(source_type, parity_status),
                parityStatus=parity_status,
                deduplicated=True,
            )

        # Parse states. For recording we treat the provided initial state as
        # the start of the stored sequence and rely on the moves list for the
        # full trajectory. To keep replay semantics and parity harnesses
        # consistent, we clear any pre-populated move history here.
        initial_state = GameState.model_validate(request.initialState)
        if initial_state.move_history:
            initial_state = initial_state.model_copy(update={"move_history": []})
        final_state = GameState.model_validate(request.finalState)

        # Parse moves
        moves = [Move.model_validate(m) for m in request.moves]

        # Prepare metadata and classify human/sandbox provenance before validation.
        metadata = _normalize_store_metadata(request.metadata or {}, final_state)

        # Infer a canonical termination reason for completed games when callers
        # omit it. This keeps sandbox callers lightweight while still satisfying
        # the recording quality gate.
        if metadata.get("termination_reason") is None:
            from app.utils.victory_type import derive_victory_type

            final_status = (
                final_state.game_status.value
                if hasattr(final_state.game_status, "value")
                else str(final_state.game_status)
            )
            if final_status in ("completed", "finished"):
                termination_reason, _ = derive_victory_type(final_state)
                metadata["termination_reason"] = termination_reason

        # Validate game quality before storing
        from app.db.unified_recording import RecordingQualityGate

        gate = RecordingQualityGate()
        valid, error = gate.validate(initial_state, final_state, moves, metadata)
        if not valid:
            logger.warning(f"Game {game_id} rejected by quality gate: {error}")
            raise HTTPException(status_code=400, detail=f"Invalid game data: {error}")

        replay_ok, replay_error = _validate_replay_sequence(initial_state, moves)
        source_type = str(metadata.get("source", "sandbox"))
        is_human_or_sandbox = _is_human_or_sandbox_source(source_type)

        store_history_entries = True
        snapshot_interval = 20
        if replay_ok:
            metadata.setdefault(
                "parity_status",
                "canonical_history_ok" if is_human_or_sandbox else "pending",
            )
            metadata.pop("replay_validation_error", None)
        elif is_human_or_sandbox:
            if "quarantine" not in source_type.lower():
                metadata["source"] = f"{source_type}_quarantine"
            metadata["parity_status"] = "non_canonical_history"
            metadata["excluded_from_training"] = True
            metadata["replay_validation_error"] = replay_error
            store_history_entries = False
            snapshot_interval = 0
            logger.warning(
                "Recording non-training replay %s with non-canonical history: %s",
                game_id,
                replay_error,
            )
        else:
            logger.warning(f"Game {game_id} rejected by replay validation: {replay_error}")
            raise HTTPException(status_code=400, detail=f"Non-replayable game data: {replay_error}")

        # Store the game - wrap blocking SQLite call
        await asyncio.to_thread(
            db.store_game,
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=request.choices,
            metadata=metadata,
            store_history_entries=store_history_entries,
            snapshot_interval=snapshot_interval,
        )

        source_type = metadata.get("source", "sandbox")
        logger.info(f"Stored game {game_id} with {len(moves)} moves (source={source_type})")
        parity_status = metadata.get("parity_status")

        return StoreGameResponse(
            gameId=game_id,
            totalMoves=len(moves),
            success=True,
            message=(
                "Game recorded for training review."
                if metadata.get("excluded_from_training")
                else "Game recorded and accepted for training export."
            ),
            acceptedForTraining=_accepted_for_training(str(source_type), str(parity_status)),
            parityStatus=str(parity_status) if parity_status is not None else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing game: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))
