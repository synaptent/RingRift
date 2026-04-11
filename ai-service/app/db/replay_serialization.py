"""Serialization and hashing helpers for replay database storage."""

from __future__ import annotations

import gzip
import json
import sqlite3

from app.models import GameState, Move, MoveType, Position


def _compress_json(data: str) -> bytes:
    """Compress JSON string using gzip."""
    return gzip.compress(data.encode("utf-8"))


def _decompress_json(data: bytes) -> str:
    """Decompress gzip-compressed JSON."""
    return gzip.decompress(data).decode("utf-8")


def _serialize_state(state: GameState) -> str:
    """Serialize GameState to JSON string."""
    return state.model_dump_json(by_alias=True)


def _deserialize_state(json_str: str) -> GameState:
    """Deserialize JSON string to GameState."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return GameState.model_validate_json(json_str)

    if data.get("gameStatus") == "finished":
        data["gameStatus"] = "completed"
    return GameState.model_validate(data)


def _serialize_move(move: Move) -> str:
    """Serialize Move to JSON string after validating required positions."""
    from app.rules.move_validation import MovePositionError, validate_move_positions

    valid, error = validate_move_positions(move)
    if not valid:
        raise MovePositionError(
            f"Cannot serialize move {move.id} (type={move.type.value}): {error}",
            move_type=move.type.value,
        )
    return move.model_dump_json(by_alias=True)


def _deserialize_move(json_str: str) -> Move:
    """Deserialize JSON string to Move."""
    return Move.model_validate_json(json_str)


def _move_from_columns(row: sqlite3.Row) -> Move | None:
    """Reconstruct a Move object from column-based schema fields."""
    move_type_str = row["move_type"]
    player = row["player"]
    if not move_type_str or player is None:
        return None

    try:
        move_type = MoveType(move_type_str)
    except ValueError:
        return None

    q = row["position_q"]
    r = row["position_r"]
    needs_position = move_type in (
        MoveType.PLACE_RING,
        MoveType.MOVE_STACK,
        MoveType.OVERTAKING_CAPTURE,
        MoveType.CONTINUE_CAPTURE_SEGMENT,
    )
    if needs_position and (q is None or r is None):
        return None

    to_pos = (
        Position(x=q, y=r, z=-(q + r))
        if q is not None and r is not None
        else None
    )
    move_number = row["move_number"]
    return Move(
        id=f"col_{move_number}",
        type=move_type,
        player=player,
        to=to_pos,
    )


def _compute_state_hash(state: GameState) -> str:
    """Compute a deterministic cross-engine state hash."""
    from app.rules.core import hash_game_state

    fingerprint = hash_game_state(state)
    return _simple_hash(fingerprint)


def _simple_hash(value: str) -> str:
    """Simple string hash matching TypeScript simpleHash for parity."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for char in value:
        code = ord(char)
        h1 = ((h1 ^ code) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ code) * 1597334677) & 0xFFFFFFFF

    h1 = (
        ((h1 ^ (h1 >> 16)) * 2246822507)
        ^ ((h2 ^ (h2 >> 13)) * 3266489909)
    ) & 0xFFFFFFFF
    h2 = (
        ((h2 ^ (h2 >> 16)) * 2246822507)
        ^ ((h1 ^ (h1 >> 13)) * 3266489909)
    ) & 0xFFFFFFFF
    combined = (h2 << 32) | h1
    return format(combined, "016x")[:16]


def _fingerprint_state(state: GameState) -> str:
    """Return the readable canonical fingerprint of a game state."""
    from app.rules.core import hash_game_state

    return hash_game_state(state)


__all__ = [
    "_compress_json",
    "_compute_state_hash",
    "_decompress_json",
    "_deserialize_move",
    "_deserialize_state",
    "_fingerprint_state",
    "_move_from_columns",
    "_serialize_move",
    "_serialize_state",
    "_simple_hash",
]
