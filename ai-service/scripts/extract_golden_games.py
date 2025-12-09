#!/usr/bin/env python
"""Extract specific games from a GameReplayDB into a golden fixture DB.

This utility copies one or more games from a source GameReplayDB SQLite
file into a new destination DB, preserving the full initial state, move
list, and recording metadata. It is intended as the final step when
promoting candidate games (for example, from find_golden_candidates.py)
into the golden replay suite under ai-service/tests/fixtures/golden_games/.

Usage examples
--------------

From ``ai-service``::

    # Extract a single game into a golden DB
    PYTHONPATH=. python scripts/extract_golden_games.py \\
        --db data/games/selfplay.db \\
        --game-id 7f031908-655b-49af-ad05-f330e9d07488 \\
        --output tests/fixtures/golden_games/golden_line_territory.db

    # Extract multiple games into one golden DB
    PYTHONPATH=. python scripts/extract_golden_games.py \\
        --db data/games/combined.db \\
        --game-id G1 --game-id G2 --game-id G3 \\
        --output tests/fixtures/golden_games/golden_mixed.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB  # noqa: E402
from app.models import GameState, Move  # noqa: E402


def _load_final_state(
    db: GameReplayDB,
    game_id: str,
    meta: Dict[str, Any],
    initial_state: GameState,
    moves: List[Move],
) -> GameState:
    """Best-effort reconstruction of the final state for a game.

    Prefer GameReplayDB.get_state_at_move when total_moves is available,
    falling back to replaying all moves with the current GameEngine if
    necessary.
    """
    total_moves = meta.get("total_moves")

    if isinstance(total_moves, int) and total_moves > 0:
        try:
            state = db.get_state_at_move(game_id, total_moves - 1)
            if state is not None:
                return state
        except Exception:
            # Fall back to replay below
            pass

    # Fallback: replay moves from the initial state using the current engine.
    # We use the stateless GameEngine.apply_move surface here so that we
    # match the semantics used in other training/replay helpers.
    from app.game_engine import GameEngine  # noqa: E402

    state = initial_state
    for move in moves:
        state = GameEngine.apply_move(state, move)
    return state


def extract_games(
    source_db_path: str,
    game_ids: List[str],
    dest_db_path: str,
) -> None:
    """Extract specified games into a fresh destination GameReplayDB."""
    if not game_ids:
        raise SystemExit("At least one --game-id must be provided.")

    src = GameReplayDB(source_db_path)

    # Ensure destination directory exists and initialise an empty DB there.
    dest_path = Path(dest_db_path)
    if dest_path.parent:
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing destination DB to avoid stale data; callers are expected
    # to treat the output as an artefact.
    if dest_path.exists():
        dest_path.unlink()

    dest = GameReplayDB(str(dest_path))

    for gid in game_ids:
        meta = src.get_game_metadata(gid)
        if meta is None:
            print(f"[extract_golden_games] WARNING: game_id {gid!r} not found in {source_db_path}")
            continue

        initial = src.get_initial_state(gid)
        if initial is None:
            print(
                f"[extract_golden_games] WARNING: initial state missing for game_id {gid!r} " f"in {source_db_path}",
            )
            continue

        moves = src.get_moves(gid)
        final_state = _load_final_state(src, gid, meta, initial, moves)

        # Decode existing metadata_json if present so we preserve provenance.
        raw_meta_json = meta.get("metadata_json")
        metadata: Dict[str, Any] = {}
        if raw_meta_json:
            try:
                metadata = json.loads(raw_meta_json)
            except Exception:
                metadata = {}

        # Tag this as part of the golden replay suite for future reference.
        tags = metadata.get("tags")
        if isinstance(tags, list):
            if "golden_candidate" not in tags:
                tags.append("golden_candidate")
        else:
            metadata["tags"] = ["golden_candidate"]

        dest.store_game(
            game_id=gid,
            initial_state=initial,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            store_history_entries=True,
            compress_states=False,
        )

    print(
        f"[extract_golden_games] Extraction complete. " f"Destination DB: {dest_db_path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract specific games into a golden fixture GameReplayDB.",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Source GameReplayDB SQLite file.",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        action="append",
        default=[],
        help="game_id to extract (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination GameReplayDB path (will be created).",
    )

    args = parser.parse_args()
    extract_games(args.db, args.game_id, args.output)


if __name__ == "__main__":
    main()
