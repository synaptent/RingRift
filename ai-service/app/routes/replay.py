"""FastAPI router for game replay endpoints.

Provides REST API for browsing, querying, and replaying games stored in the
GameReplayDB SQLite database. Used by the sandbox UI replay panel.

See docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md for specification.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

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
        games = db.query_games(**filters, limit=limit + 1, offset=offset)
        has_more = len(games) > limit
        games = games[:limit]

        # Get total count for this filter set
        total = db.get_game_count(**filters)

        # Convert to response format
        game_list = []
        for g in games:
            raw_metadata_json = g.get("metadata_json")
            decoded_metadata: dict[str, Any] | None
            if raw_metadata_json:
                try:
                    decoded_val = json.loads(raw_metadata_json)
                    decoded_metadata = decoded_val if isinstance(decoded_val, dict) else None
                except Exception:
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
        game = db.get_game_with_players(game_id)

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
            except Exception:
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
):
    """Get reconstructed game state at a specific move.

    Uses snapshots for fast reconstruction when available.
    """
    try:
        db = get_replay_db()

        # Check game exists
        meta = db.get_game_metadata(game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        total_moves = meta["total_moves"]

        # Validate move number
        if move_number > total_moves:
            raise HTTPException(
                status_code=400,
                detail=f"Move number {move_number} exceeds total moves {total_moves}",
            )

        # Get state at move
        if move_number == 0:
            state = db.get_initial_state(game_id)
        else:
            state = db.get_state_at_move(game_id, move_number - 1)

        if state is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reconstruct state at move {move_number}",
            )

        # Get engine eval for this move if available
        engine_eval = None
        engine_pv = None
        if move_number > 0:
            move_records = db.get_move_records(game_id, start=move_number - 1, end=move_number)
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

        # Check game exists
        meta = db.get_game_metadata(game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Compute effective end
        effective_end = end if end is not None else start + limit

        # Fetch moves
        move_records = db.get_move_records(game_id, start=start, end=effective_end)

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

        # Check game exists
        meta = db.get_game_metadata(game_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        choices = db.get_choices_at_move(game_id, move_number)

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
        stats = db.get_stats()

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
    moves: list[dict[str, Any]] = Field(..., max_length=10000, description="List of moves (max 10000)")
    choices: list[dict[str, Any]] | None = Field(None, max_length=1000, description="List of choices (max 1000)")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class StoreGameResponse(BaseModel):
    """Response after storing a game."""

    gameId: str
    totalMoves: int
    success: bool


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

        # Prepare metadata
        metadata = request.metadata or {}
        metadata.setdefault("source", "sandbox")

        # Store the game
        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=request.choices,
            metadata=metadata,
        )

        logger.info(f"Stored game {game_id} with {len(moves)} moves from sandbox")

        return StoreGameResponse(
            gameId=game_id,
            totalMoves=len(moves),
            success=True,
        )

    except Exception as e:
        logger.error(f"Error storing game: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_detail(e))
