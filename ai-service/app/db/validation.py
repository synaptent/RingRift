"""Game validation module for cross-validating recorded games.

This module provides tools for validating recorded games by replaying moves
and comparing computed states against stored snapshots. When divergences are
detected, it can automatically export test fixtures for debugging.

Usage:
    from app.db.validation import validate_game, validate_all_games, export_fixture

    # Validate a single game
    result = validate_game(db, game_id)
    if not result.valid:
        print(f"Divergence at move {result.divergence_move}")
        export_fixture(result, output_dir="fixtures/auto")

    # Validate all games in a database
    results = validate_all_games(db_path)
    for r in results:
        if not r.valid:
            print(f"Game {r.game_id}: divergence at move {r.divergence_move}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.db.game_replay import GameReplayDB, _compute_state_hash, _serialize_state
from app.game_engine import GameEngine
from app.models import GameState, Move

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a recorded game."""

    game_id: str
    valid: bool
    total_moves: int
    validated_moves: int
    divergence_move: Optional[int] = None
    expected_hash: Optional[str] = None
    computed_hash: Optional[str] = None
    expected_state: Optional[GameState] = None
    computed_state: Optional[GameState] = None
    move_at_divergence: Optional[Move] = None
    initial_state: Optional[GameState] = None
    moves_up_to_divergence: List[Move] = field(default_factory=list)
    error: Optional[str] = None


def validate_game(
    db: GameReplayDB,
    game_id: str,
    stop_on_first_divergence: bool = True,
) -> ValidationResult:
    """Validate a recorded game by replaying and comparing against snapshots.

    For each move, applies it to the current state and compares the resulting
    state hash against the stored snapshot hash (if available).

    Args:
        db: GameReplayDB instance
        game_id: ID of game to validate
        stop_on_first_divergence: If True, stop at first mismatch

    Returns:
        ValidationResult with validation status and divergence details
    """
    from app.game_engine import GameEngine

    # Get initial state
    initial_state = db.get_initial_state(game_id)
    if initial_state is None:
        return ValidationResult(
            game_id=game_id,
            valid=False,
            total_moves=0,
            validated_moves=0,
            error="Initial state not found",
        )

    # Get all moves
    moves = db.get_moves(game_id)
    if not moves:
        return ValidationResult(
            game_id=game_id,
            valid=True,
            total_moves=0,
            validated_moves=0,
            initial_state=initial_state,
        )

    # Get stored snapshots with hashes
    snapshots = _get_snapshots_with_hashes(db, game_id)

    # Replay moves and validate
    current_state = initial_state
    validated_count = 0
    moves_applied: List[Move] = []

    for i, move in enumerate(moves):
        try:
            # Apply move
            next_state = GameEngine.apply_move(current_state, move)
            computed_hash = _compute_state_hash(next_state)
            moves_applied.append(move)

            # Check against stored snapshot if available
            if i in snapshots:
                stored_hash = snapshots[i]["hash"]
                if stored_hash and stored_hash != computed_hash:
                    # Divergence detected!
                    stored_state = db.get_state_at_move(game_id, i)
                    return ValidationResult(
                        game_id=game_id,
                        valid=False,
                        total_moves=len(moves),
                        validated_moves=validated_count,
                        divergence_move=i,
                        expected_hash=stored_hash,
                        computed_hash=computed_hash,
                        expected_state=stored_state,
                        computed_state=next_state,
                        move_at_divergence=move,
                        initial_state=initial_state,
                        moves_up_to_divergence=moves_applied.copy(),
                    )
                validated_count += 1

            current_state = next_state

        except Exception as e:
            return ValidationResult(
                game_id=game_id,
                valid=False,
                total_moves=len(moves),
                validated_moves=validated_count,
                divergence_move=i,
                move_at_divergence=move,
                initial_state=initial_state,
                moves_up_to_divergence=moves_applied.copy(),
                error=f"Error applying move {i}: {str(e)}",
            )

    return ValidationResult(
        game_id=game_id,
        valid=True,
        total_moves=len(moves),
        validated_moves=validated_count,
        initial_state=initial_state,
        moves_up_to_divergence=moves_applied,
    )


def validate_all_games(
    db_path: str,
    stop_on_first_divergence: bool = True,
    max_games: Optional[int] = None,
) -> List[ValidationResult]:
    """Validate all games in a database.

    Args:
        db_path: Path to SQLite database
        stop_on_first_divergence: If True, stop each game at first mismatch
        max_games: Maximum number of games to validate (None for all)

    Returns:
        List of ValidationResult for each game
    """
    db = GameReplayDB(db_path)
    results = []

    games = db.query_games(limit=max_games or 10000)
    for game_meta in games:
        game_id = game_meta["game_id"]
        result = validate_game(db, game_id, stop_on_first_divergence)
        results.append(result)

        if not result.valid:
            logger.warning(
                f"Game {game_id}: divergence at move {result.divergence_move}"
            )

    return results


def export_fixture(
    result: ValidationResult,
    output_dir: str = "fixtures/auto",
    include_expected: bool = True,
) -> Optional[str]:
    """Export a divergent game as a test fixture.

    Creates a JSON file containing:
    - Initial state
    - Moves up to divergence
    - Expected state (from snapshot)
    - Computed state (from replay)
    - Move that caused divergence

    Args:
        result: ValidationResult with divergence
        output_dir: Directory to write fixture
        include_expected: Include expected state in fixture

    Returns:
        Path to written fixture, or None if no divergence
    """
    if result.valid:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fixture_name = f"divergence_{result.game_id}_move{result.divergence_move}.json"
    fixture_path = output_path / fixture_name

    fixture_data = {
        "game_id": result.game_id,
        "divergence_move": result.divergence_move,
        "total_moves": result.total_moves,
        "expected_hash": result.expected_hash,
        "computed_hash": result.computed_hash,
        "error": result.error,
    }

    if result.initial_state:
        fixture_data["initial_state"] = json.loads(_serialize_state(result.initial_state))

    if result.move_at_divergence:
        fixture_data["move_at_divergence"] = json.loads(
            result.move_at_divergence.model_dump_json(by_alias=True)
        )

    fixture_data["moves_up_to_divergence"] = [
        json.loads(m.model_dump_json(by_alias=True))
        for m in result.moves_up_to_divergence
    ]

    if include_expected and result.expected_state:
        fixture_data["expected_state"] = json.loads(_serialize_state(result.expected_state))

    if result.computed_state:
        fixture_data["computed_state"] = json.loads(_serialize_state(result.computed_state))

    with open(fixture_path, "w") as f:
        json.dump(fixture_data, f, indent=2)

    logger.info(f"Exported fixture to {fixture_path}")
    return str(fixture_path)


def _get_snapshots_with_hashes(db: GameReplayDB, game_id: str) -> dict:
    """Get all snapshots for a game with their hashes.

    Returns dict mapping move_number -> {"hash": state_hash, "compressed": bool}
    """
    with db._get_conn() as conn:
        rows = conn.execute(
            """
            SELECT move_number, state_hash, compressed
            FROM game_state_snapshots
            WHERE game_id = ?
            ORDER BY move_number
            """,
            (game_id,),
        ).fetchall()

        return {
            row["move_number"]: {
                "hash": row["state_hash"],
                "compressed": bool(row["compressed"]),
            }
            for row in rows
        }


def validate_database_summary(db_path: str, sample_size: int = 100) -> dict:
    """Get a validation summary for a database.

    Args:
        db_path: Path to SQLite database
        sample_size: Number of games to validate

    Returns:
        Summary dict with validation statistics
    """
    results = validate_all_games(db_path, max_games=sample_size)

    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count
    error_count = sum(1 for r in results if r.error)

    return {
        "total_validated": len(results),
        "valid_games": valid_count,
        "invalid_games": invalid_count,
        "games_with_errors": error_count,
        "validation_rate": valid_count / len(results) if results else 0,
        "divergent_games": [
            {
                "game_id": r.game_id,
                "divergence_move": r.divergence_move,
                "error": r.error,
            }
            for r in results
            if not r.valid
        ],
    }
