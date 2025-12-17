#!/usr/bin/env python
"""Merge multiple GameReplayDB SQLite databases into a single destination DB.

This utility is intended for consolidating per-run self-play / CMA-ES game
databases into a single corpus while preserving rich recording metadata.

Merging strategy
----------------

- For each source DB:
  - Iterate over games via GameReplayDB.iterate_games(), which yields:
    (game_meta, initial_state, moves).
  - Reconstruct the final state using get_state_at_move(game_id, total_moves-1)
    when possible; fall back to replaying moves if needed.
  - Decode the existing metadata_json from the source games row (if present)
    and ensure:
      - metadata["source"] is set (falling back to the row's source column
        or "merged" if absent).
      - metadata["merged_from_db"] records the basename of the source DB.
  - Store the game into the destination DB via dest_db.store_game(...),
    recomputing history entries and snapshots using the current rules engine.

- Game ID conflicts:
  - By default (--on-conflict=skip), any game_id that already exists in the
    destination DB is skipped.
  - With --on-conflict=rename, conflicting game_ids are replaced with a new
    UUID4 while keeping all other metadata and state identical.

- Deduplication (--dedupe-by-game-id):
  - When enabled, tracks all game_ids seen across ALL source DBs and skips
    duplicates even if they don't exist in the destination yet. This is
    useful for incremental syncs where the same games may appear in multiple
    source databases.

Usage examples
--------------

From the ``ai-service`` root::

    # Merge several CMA-ES run DBs into a single corpus
    python scripts/merge_game_dbs.py \\
        --output data/games/cmaes_aggregate.db \\
        --db logs/cmaes/runs/run1/games.db \\
        --db logs/cmaes/runs/run2/games.db

    # Merge with explicit conflict handling and a source label
    python scripts/merge_game_dbs.py \\
        --output data/games/combined.db \\
        --db data/games/selfplay.square8.db \\
        --db data/games/selfplay.square19.db \\
        --on-conflict rename

    # Incremental merge with deduplication across sources
    python scripts/merge_game_dbs.py \\
        --output data/games/merged.db \\
        --db worker1/selfplay.db \\
        --db worker2/selfplay.db \\
        --dedupe-by-game-id
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB  # noqa: E402
from app.models import GameState, Move  # noqa: E402


# Merge state tracking file
MERGE_STATE_FILE = ".merge_state.json"


def _enable_wal_mode(db_path: str) -> None:
    """Enable WAL mode for better concurrency and crash recovery."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.close()
    except Exception as e:
        print(f"[merge_game_dbs] Warning: Could not enable WAL mode: {e}")


def _load_merge_state(output_path: str) -> Dict[str, Any]:
    """Load merge state from previous run for crash recovery."""
    state_file = Path(output_path).parent / MERGE_STATE_FILE
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_dbs": [], "seen_game_ids": []}


def _save_merge_state(output_path: str, state: Dict[str, Any]) -> None:
    """Save merge state for crash recovery."""
    state_file = Path(output_path).parent / MERGE_STATE_FILE
    state["updated_at"] = datetime.now().isoformat()
    try:
        with open(state_file, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[merge_game_dbs] Warning: Could not save merge state: {e}")


def _clear_merge_state(output_path: str) -> None:
    """Clear merge state after successful completion."""
    state_file = Path(output_path).parent / MERGE_STATE_FILE
    try:
        if state_file.exists():
            state_file.unlink()
    except Exception:
        pass


def _load_final_state(
    db: GameReplayDB,
    game_id: str,
    meta: Dict[str, Any],
    initial_state: GameState,
    moves: List[Move],
) -> GameState:
    """Best-effort reconstruction of the final state for a game."""
    total_moves = meta.get("total_moves")

    if isinstance(total_moves, int) and total_moves > 0:
        try:
            state = db.get_state_at_move(game_id, total_moves - 1)
            if state is not None:
                return state
        except Exception:
            # Fall back to replay below
            pass

    # Fallback: replay all moves from the initial state using the current engine
    from app.game_engine import GameEngine  # noqa: E402

    state = initial_state
    for move in moves:
        state = GameEngine.apply_move(state, move)
    return state


def _merge_single_db(
    dest_db: GameReplayDB,
    src_path: str,
    on_conflict: str = "skip",
    seen_game_ids: Set[str] | None = None,
    *,
    store_history_entries: bool,
    compress_states: bool,
) -> Dict[str, int]:
    """Merge all games from a single source DB into the destination DB.

    Args:
        dest_db: Destination database to merge into.
        src_path: Path to source database.
        on_conflict: Policy for game_id conflicts ("skip" or "rename").
        seen_game_ids: Optional set of game_ids already seen across sources.
            If provided, games with IDs in this set will be skipped (dedupe).
            New game_ids will be added to this set.
    """
    src_db = GameReplayDB(src_path)
    stats = {"total": 0, "merged": 0, "skipped_conflict": 0, "skipped_dedupe": 0, "errors": 0}
    src_label = os.path.basename(src_path)

    for game_meta, initial_state, moves in src_db.iterate_games():
        stats["total"] += 1
        game_id = game_meta.get("game_id")
        if not game_id:
            stats["errors"] += 1
            print(f"[merge_game_dbs] WARNING: Game without game_id in {src_path}, skipping")
            continue

        # Cross-source deduplication
        if seen_game_ids is not None:
            if game_id in seen_game_ids:
                stats["skipped_dedupe"] += 1
                continue
            seen_game_ids.add(game_id)

        # Conflict handling (game already in destination)
        existing = dest_db.get_game_metadata(game_id)
        if existing is not None:
            if on_conflict == "skip":
                stats["skipped_conflict"] += 1
                continue
            elif on_conflict == "rename":
                new_game_id = str(uuid.uuid4())
                print(
                    f"[merge_game_dbs] INFO: Renaming conflicting game_id "
                    f"{game_id} -> {new_game_id} from {src_label}"
                )
                game_id = new_game_id
                if seen_game_ids is not None:
                    seen_game_ids.add(game_id)
            else:
                stats["errors"] += 1
                print(
                    f"[merge_game_dbs] ERROR: Unknown on_conflict policy {on_conflict!r}, "
                    f"skipping game {game_id} from {src_label}"
                )
                continue

        # Decode existing metadata_json if present
        raw_meta_json = game_meta.get("metadata_json")
        metadata: Dict[str, Any] = {}
        if raw_meta_json:
            try:
                metadata = json.loads(raw_meta_json)
            except Exception:
                metadata = {}

        # Ensure source and provenance tagging
        if "source" not in metadata:
            metadata["source"] = game_meta.get("source") or "merged"
        # Track origin DB for later analysis
        if "merged_from_db" not in metadata:
            metadata["merged_from_db"] = src_label

        try:
            final_state = _load_final_state(
                src_db,
                game_meta["game_id"],
                game_meta,
                initial_state,
                moves,
            )
            dest_db.store_game(
                game_id=game_id,
                initial_state=initial_state,
                final_state=final_state,
                moves=moves,
                metadata=metadata,
                store_history_entries=bool(store_history_entries),
                compress_states=bool(compress_states),
            )
            stats["merged"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"[merge_game_dbs] ERROR: Failed to merge game {game_meta['game_id']} " f"from {src_label}: {e}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple GameReplayDB SQLite databases into one.")
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        required=True,
        help="Path to a source SQLite database file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination SQLite database path (created if it does not exist).",
    )
    parser.add_argument(
        "--on-conflict",
        type=str,
        choices=["skip", "rename"],
        default="skip",
        help=(
            "Policy when a game_id already exists in the destination DB: "
            "'skip' (default) or 'rename' to assign a new UUID."
        ),
    )
    parser.add_argument(
        "--dedupe-by-game-id",
        action="store_true",
        help=(
            "Enable cross-source deduplication: track all game_ids seen across "
            "source DBs and skip duplicates even if not in destination. "
            "Useful for incremental syncs where games may appear in multiple sources."
        ),
    )
    parser.add_argument(
        "--store-history-entries",
        action="store_true",
        help=(
            "Store full per-move state history entries (state_before/state_after) in the output DB. "
            "This can make databases ~100x larger; default is disabled (lean merge)."
        ),
    )
    parser.add_argument(
        "--compress-states",
        action="store_true",
        default=True,
        help="Compress state JSON blobs in the output DB (default: enabled).",
    )
    parser.add_argument(
        "--no-compress-states",
        action="store_true",
        help="Disable state compression in the output DB.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous interrupted merge (uses state file).",
    )
    parser.add_argument(
        "--wal-mode",
        action="store_true",
        default=True,
        help="Enable WAL mode for better crash recovery (default: enabled).",
    )

    args = parser.parse_args()

    if not args.db:
        print("[merge_game_dbs] ERROR: At least one --db source must be provided.")
        sys.exit(1)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Enable WAL mode for crash recovery
    if args.wal_mode:
        _enable_wal_mode(args.output)

    dest_db = GameReplayDB(args.output)

    total_stats = {"total": 0, "merged": 0, "skipped_conflict": 0, "skipped_dedupe": 0, "errors": 0}

    # Load previous state for crash recovery
    merge_state = _load_merge_state(args.output) if args.resume else {"processed_dbs": [], "seen_game_ids": []}
    processed_dbs = set(merge_state.get("processed_dbs", []))

    # Initialize cross-source deduplication set if enabled
    seen_game_ids: Set[str] | None = None
    if args.dedupe_by_game_id:
        seen_game_ids = set(merge_state.get("seen_game_ids", []))

    print(f"[merge_game_dbs] Merging {len(args.db)} database(s) into {args.output}")
    if args.resume and processed_dbs:
        print(f"[merge_game_dbs] Resuming: {len(processed_dbs)} DBs already processed")
    if args.dedupe_by_game_id:
        print("[merge_game_dbs] Cross-source deduplication enabled")
    if args.wal_mode:
        print("[merge_game_dbs] WAL mode enabled for crash recovery")
    if args.store_history_entries:
        print(
            "[merge_game_dbs] WARNING: --store-history-entries enabled; output DBs can grow very large.",
            file=sys.stderr,
        )

    compress_states = bool(args.compress_states) and not bool(args.no_compress_states)

    for src in args.db:
        # Skip already processed DBs when resuming
        src_norm = os.path.normpath(os.path.abspath(src))
        if src_norm in processed_dbs:
            print(f"[merge_game_dbs] Skipping (already processed): {src}")
            continue
        if not os.path.exists(src):
            print(f"[merge_game_dbs] WARNING: Source DB not found: {src}, skipping")
            continue

        print(f"[merge_game_dbs] Processing source DB: {src}")
        stats = _merge_single_db(
            dest_db,
            src,
            on_conflict=args.on_conflict,
            seen_game_ids=seen_game_ids,
            store_history_entries=bool(args.store_history_entries),
            compress_states=compress_states,
        )
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

        # Save state after each DB for crash recovery
        processed_dbs.add(src_norm)
        _save_merge_state(args.output, {
            "processed_dbs": list(processed_dbs),
            "seen_game_ids": list(seen_game_ids) if seen_game_ids else [],
            "total_stats": total_stats,
        })

    # Clear state file on successful completion
    _clear_merge_state(args.output)

    print("\n[merge_game_dbs] Merge complete.")
    print(f"  Total games seen:     {total_stats['total']}")
    print(f"  Games merged:         {total_stats['merged']}")
    print(f"  Conflicts skipped:    {total_stats['skipped_conflict']}")
    if args.dedupe_by_game_id:
        print(f"  Duplicates skipped:   {total_stats['skipped_dedupe']}")
    print(f"  Errors during merge:  {total_stats['errors']}")


if __name__ == "__main__":
    main()
