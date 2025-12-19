#!/usr/bin/env python3
"""Phase transition parity tests.

These tests verify that phase transitions recorded by the Python engine
match expected patterns and can be validated against TypeScript replays.

The tests use the game_history_entries table which contains:
- phase_before, phase_after: Phase transitions for each move
- state_hash_before, state_hash_after: State hashes for validation
- Full state JSON for detailed debugging when hashes diverge
"""

import pytest
from pathlib import Path
from typing import List, Tuple

from app.db import GameReplayDB
from app.models import GamePhase


# Valid phase transitions according to RingRift rules
VALID_PHASE_TRANSITIONS: List[Tuple[str, str]] = [
    # Ring placement phase
    ("ring_placement", "movement"),       # After placing ring, if player can move/capture
    ("ring_placement", "line_processing"),  # After placing ring, if no moves available (goes to bookkeeping)
    ("ring_placement", "ring_placement"), # After placing ring, player skipped (no moves), next player also ring_placement

    # Movement phase
    ("movement", "line_processing"),      # After movement, check for lines
    ("movement", "capture"),              # After movement that triggers capture
    ("movement", "territory_processing"), # After movement, if no lines
    ("movement", "ring_placement"),       # After movement, next player's turn
    ("movement", "movement"),             # After movement, player skipped, next player has no rings but can move

    # Capture phase
    ("capture", "chain_capture"),         # After capture, if chain continuation available
    ("capture", "line_processing"),       # After capture completes, check for lines
    ("capture", "territory_processing"),  # After capture, if no lines
    ("capture", "ring_placement"),        # After capture, next player's turn
    ("capture", "movement"),              # After capture, next player has no rings but can move

    # Chain capture phase
    ("chain_capture", "chain_capture"),   # Continuing a chain
    ("chain_capture", "line_processing"), # After chain completes, check for lines
    ("chain_capture", "territory_processing"),  # After chain, if no lines
    ("chain_capture", "ring_placement"),  # After chain, next player's turn
    ("chain_capture", "movement"),        # After chain, next player has no rings but can move

    # Line processing phase
    ("line_processing", "territory_processing"),  # After lines processed, check territories
    ("line_processing", "line_processing"),  # Multiple lines to process
    ("line_processing", "ring_placement"),  # After lines, next player's turn
    ("line_processing", "movement"),      # After lines, same player continues (if rings placed earlier)

    # Territory processing phase
    ("territory_processing", "territory_processing"),  # Multiple territories to process
    ("territory_processing", "ring_placement"),  # After territories, next player's turn
    ("territory_processing", "movement"),  # After territories, if player has no rings but can move
]


@pytest.mark.skip(
    reason="TODO-DB-SCHEMA: Recorded game format changed during GameReplayDB schema "
    "migration. The test expects game_history entries with phase transition data in "
    "the old format. Needs update to: (1) use new schema with GameHistoryEntry model, "
    "(2) extract phase transitions from the state field instead of dedicated columns. "
    "See database/replay.py for current schema."
)
def test_valid_phase_transitions_in_recorded_game():
    """Verify that all recorded phase transitions are valid according to rules."""
    # Find a test database with recorded games
    db_paths = [
        Path("logs/soak_full_20251204/square8_4p.db"),
        Path("logs/soak_full_20251204/square8_3p.db"),
        Path("data/games/selfplay.db"),
    ]

    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break

    if db_path is None:
        pytest.skip("No test database found")

    db = GameReplayDB(str(db_path))

    # Query all phase transitions
    with db._get_conn() as conn:
        transitions = conn.execute("""
            SELECT game_id, move_number, phase_before, phase_after
            FROM game_history_entries
            ORDER BY game_id, move_number
        """).fetchall()

    if not transitions:
        pytest.skip("No game history entries found in database")

    # Check each transition
    invalid_transitions = []
    for game_id, move_number, phase_before, phase_after in transitions:
        transition = (phase_before, phase_after)
        if transition not in VALID_PHASE_TRANSITIONS:
            invalid_transitions.append({
                "game_id": game_id,
                "move_number": move_number,
                "transition": f"{phase_before} -> {phase_after}",
            })

    # Report invalid transitions
    if invalid_transitions:
        sample = invalid_transitions[:10]
        msg = f"Found {len(invalid_transitions)} invalid phase transitions:\n"
        for inv in sample:
            msg += f"  Game {inv['game_id'][:8]}... move {inv['move_number']}: {inv['transition']}\n"
        pytest.fail(msg)


def test_state_hash_chain_consistency():
    """Verify that state_hash_after[i] matches state_hash_before[i+1]."""
    db_paths = [
        Path("logs/soak_full_20251204/square8_4p.db"),
        Path("logs/soak_full_20251204/square8_3p.db"),
        Path("data/games/selfplay.db"),
    ]

    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break

    if db_path is None:
        pytest.skip("No test database found")

    db = GameReplayDB(str(db_path))

    # Get all game IDs
    with db._get_conn() as conn:
        game_ids = [row[0] for row in conn.execute(
            "SELECT DISTINCT game_id FROM game_history_entries"
        ).fetchall()]

    if not game_ids:
        pytest.skip("No game history entries found in database")

    # Check hash chain consistency for each game
    broken_chains = []
    for game_id in game_ids:
        with db._get_conn() as conn:
            entries = conn.execute("""
                SELECT move_number, state_hash_before, state_hash_after
                FROM game_history_entries
                WHERE game_id = ?
                ORDER BY move_number
            """, (game_id,)).fetchall()

        for i in range(len(entries) - 1):
            current_after = entries[i][2]
            next_before = entries[i + 1][1]

            if current_after != next_before:
                broken_chains.append({
                    "game_id": game_id,
                    "move": i,
                    "hash_after": current_after,
                    "hash_before_next": next_before,
                })

    if broken_chains:
        sample = broken_chains[:5]
        msg = f"Found {len(broken_chains)} broken hash chains:\n"
        for brk in sample:
            msg += (
                f"  Game {brk['game_id'][:8]}... move {brk['move']}: "
                f"after={brk['hash_after'][:8]}... != before_next={brk['hash_before_next'][:8]}...\n"
            )
        pytest.fail(msg)


def test_phase_transitions_after_placement():
    """Verify phase transitions after ring_placement match expected patterns."""
    db_paths = [
        Path("logs/soak_full_20251204/square8_4p.db"),
        Path("logs/soak_full_20251204/square8_3p.db"),
    ]

    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break

    if db_path is None:
        pytest.skip("No test database found")

    db = GameReplayDB(str(db_path))

    # Query placement moves
    with db._get_conn() as conn:
        placement_transitions = conn.execute("""
            SELECT game_id, move_number, phase_before, phase_after
            FROM game_history_entries
            WHERE phase_before = 'ring_placement'
        """).fetchall()

    if not placement_transitions:
        pytest.skip("No ring_placement phases found")

    # Collect statistics
    transition_counts = {}
    for _, _, _, phase_after in placement_transitions:
        key = f"ring_placement -> {phase_after}"
        transition_counts[key] = transition_counts.get(key, 0) + 1

    # Verify only valid post-placement phases
    # Note: ring_placement -> ring_placement can happen when player is skipped
    # (no valid moves) and next player also starts in ring_placement
    valid_post_placement = {"movement", "line_processing", "ring_placement"}
    for transition, count in transition_counts.items():
        phase_after = transition.split(" -> ")[1]
        if phase_after not in valid_post_placement:
            pytest.fail(
                f"Unexpected post-placement phase transition: {transition} "
                f"(occurred {count} times)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
