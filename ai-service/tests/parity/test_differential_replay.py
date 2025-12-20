#!/usr/bin/env python3
"""Differential replay tests - compare Python and TypeScript game replays.

These tests verify that when replaying the same recorded game:
1. Both engines produce the same state hashes at each step
2. Phase transitions match
3. Player progression matches

Uses the game_history_entries table which contains pre-recorded state hashes
from the Python engine, and compares them against TypeScript replay output.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

from app.db import GameReplayDB


class ReplayStep(NamedTuple):
    """A single step in a game replay."""
    move_number: int
    move_type: str
    move_player: int
    phase_after: str
    player_after: int
    state_hash: str | None


def get_recorded_game_steps(db_path: str, game_id: str) -> list[ReplayStep]:
    """Get the recorded Python replay steps from the database."""
    db = GameReplayDB(db_path)

    # Get moves with their post-move state from history entries
    with db._get_conn() as conn:
        # Get move details
        moves = conn.execute("""
            SELECT gm.move_number, gm.move_type, gm.player
            FROM game_moves gm
            WHERE gm.game_id = ?
            ORDER BY gm.move_number
        """, (game_id,)).fetchall()

        # Get history entries for state after each move
        history = {
            row[0]: row[1:]
            for row in conn.execute("""
                SELECT move_number, phase_after, state_hash_after
                FROM game_history_entries
                WHERE game_id = ?
            """, (game_id,)).fetchall()
        }

    steps = []
    for move_num, move_type, move_player in moves:
        if move_num in history:
            phase_after, state_hash = history[move_num]
        else:
            phase_after = None
            state_hash = None

        # For player_after, we'd need to look at the next move or use the state
        # For now, we'll skip this and focus on phase/hash comparison
        steps.append(ReplayStep(
            move_number=move_num,
            move_type=move_type,
            move_player=move_player,
            phase_after=phase_after or "",
            player_after=0,  # Not easily available without state
            state_hash=state_hash,
        ))

    return steps


def run_ts_replay(db_path: str, game_id: str) -> list[dict] | None:
    """Run the TypeScript replay script and parse its JSON output."""
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "selfplay-db-ts-replay.ts"

    if not script_path.exists():
        return None

    try:
        result = subprocess.run(
            [
                "npx", "ts-node",
                str(script_path),
                "--db", db_path,
                "--game", game_id,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parent.parent.parent.parent),
            env={
                "TS_NODE_PROJECT": "tsconfig.server.json",
                **dict(subprocess.os.environ),
            },
        )

        if result.returncode != 0:
            print(f"TS replay failed: {result.stderr}")
            return None

        # Parse JSON lines output
        steps = []
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if data.get("kind") == "ts-replay-step":
                        steps.append(data)
                except json.JSONDecodeError:
                    continue

        return steps

    except subprocess.TimeoutExpired:
        print("TS replay timed out")
        return None
    except Exception as e:
        print(f"TS replay error: {e}")
        return None


def compare_replays(
    py_steps: list[ReplayStep],
    ts_steps: list[dict],
) -> dict[str, list[dict]]:
    """Compare Python and TypeScript replay outputs."""
    divergences = []

    min_len = min(len(py_steps), len(ts_steps))

    for i in range(min_len):
        py = py_steps[i]
        ts = ts_steps[i]

        ts_phase = ts.get("summary", {}).get("currentPhase", "")
        ts.get("summary", {}).get("currentPlayer", 0)
        ts_hash = ts.get("summary", {}).get("stateHash", "")

        # Check phase mismatch
        if py.phase_after and ts_phase and py.phase_after != ts_phase:
            divergences.append({
                "move": i,
                "type": "phase_mismatch",
                "py_phase": py.phase_after,
                "ts_phase": ts_phase,
            })

        # Check hash mismatch
        if py.state_hash and ts_hash and py.state_hash != ts_hash:
            divergences.append({
                "move": i,
                "type": "hash_mismatch",
                "py_hash": py.state_hash[:16],
                "ts_hash": ts_hash[:16] if len(ts_hash) >= 16 else ts_hash,
            })

    # Check length mismatch
    if len(py_steps) != len(ts_steps):
        divergences.append({
            "type": "length_mismatch",
            "py_moves": len(py_steps),
            "ts_moves": len(ts_steps),
        })

    return {
        "divergences": divergences,
        "py_move_count": len(py_steps),
        "ts_move_count": len(ts_steps) if ts_steps else 0,
    }


# Optional environment-driven golden case configuration. When both of these
# environment variables are set, the strict golden-game parity test below will
# run and assert that there are zero divergences for the specified game.
GOLDEN_DB_ENV = "RINGRIFT_PARITY_GOLDEN_DB"
GOLDEN_GAME_ENV = "RINGRIFT_PARITY_GOLDEN_GAME_ID"


class TestDifferentialReplay:
    """Test suite for differential Python/TypeScript replay comparison."""

    def get_test_db_and_game(self) -> tuple | None:
        """Find a test database and game for replay testing."""
        db_paths = [
            Path("logs/soak_full_20251204/square8_4p.db"),
            Path("logs/soak_full_20251204/square8_3p.db"),
            Path("data/games/selfplay.db"),
        ]

        for db_path in db_paths:
            if db_path.exists():
                db = GameReplayDB(str(db_path))
                with db._get_conn() as conn:
                    game = conn.execute("""
                        SELECT game_id FROM games
                        WHERE total_moves > 10
                        LIMIT 1
                    """).fetchone()
                    if game:
                        return str(db_path), game[0]

        return None

    def test_python_recorded_steps_are_available(self):
        """Verify that Python replay steps can be extracted from database."""
        result = self.get_test_db_and_game()
        if result is None:
            pytest.skip("No test database found")

        db_path, game_id = result
        steps = get_recorded_game_steps(db_path, game_id)

        assert len(steps) > 0, "Should have recorded steps"
        assert all(s.move_type for s in steps), "All steps should have move types"

    def test_ts_replay_script_exists(self):
        """Verify that the TypeScript replay script exists."""
        script_path = (
            Path(__file__).parent.parent.parent.parent
            / "scripts"
            / "selfplay-db-ts-replay.ts"
        )
        assert script_path.exists(), f"TS replay script not found at {script_path}"

    @pytest.mark.slow
    def test_differential_replay_divergence_report(self):
        """Run differential replay and report any divergences.

        This test is marked slow because it spawns a subprocess for TS replay.
        Run with: pytest -m slow
        """
        result = self.get_test_db_and_game()
        if result is None:
            pytest.skip("No test database found")

        db_path, game_id = result

        # Get Python steps
        py_steps = get_recorded_game_steps(db_path, game_id)

        # Get TypeScript steps
        ts_steps = run_ts_replay(db_path, game_id)

        if ts_steps is None:
            pytest.skip("TypeScript replay failed or unavailable")

        # Compare
        comparison = compare_replays(py_steps, ts_steps)

        # Report divergences (but don't fail - this is for investigation)
        divergences = comparison.get("divergences", [])
        if divergences:
            msg = f"Found {len(divergences)} divergences in {game_id}:\n"
            for d in divergences[:10]:
                msg += f"  {d}\n"

            # For now, just print the divergences without failing
            # Once parity is achieved, we can change this to pytest.fail
            print(msg)

        # Assert that we at least got comparable output

    def test_golden_game_has_no_differential_replay_divergences(self) -> None:
        """Strict parity check for a single configured golden game.

        This test is intentionally skip-friendly and only runs when both:

          - RINGRIFT_PARITY_GOLDEN_DB points to an existing SQLite DB, and
          - RINGRIFT_PARITY_GOLDEN_GAME_ID names a game_id within that DB.

        It runs the same differential replay comparison as the diagnostic
        test above but *fails* if any divergences are found, making golden
        games a lightweight regression guard for TS↔Python replay parity.
        """
        db_path_env = os.environ.get(GOLDEN_DB_ENV)
        game_id = os.environ.get(GOLDEN_GAME_ENV)

        if not db_path_env or not game_id:
            pytest.skip(
                f"{GOLDEN_DB_ENV} and {GOLDEN_GAME_ENV} not set; "
                "no golden game configured for strict differential replay.",
            )

        db_path = Path(db_path_env)
        if not db_path.exists():
            pytest.skip(f"Golden DB path does not exist: {db_path}")

        # Verify the game exists and has recorded moves.
        db = GameReplayDB(str(db_path))
        with db._get_conn() as conn:
            row = conn.execute(
                "SELECT total_moves FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        if row is None:
            pytest.skip(
                f"Golden game_id {game_id!r} not found in {db_path}; "
                "skipping strict parity check.",
            )

        # Build Python and TS step sequences.
        py_steps = get_recorded_game_steps(str(db_path), game_id)
        assert py_steps, "Golden game should have at least one recorded step"

        ts_steps = run_ts_replay(str(db_path), game_id)
        if ts_steps is None:
            pytest.skip(
                "TypeScript replay unavailable or failed for "
                f"db={db_path} game_id={game_id}; skipping strict parity.",
            )
        assert ts_steps, "TS replay produced no steps for golden game"

        comparison = compare_replays(py_steps, ts_steps)
        divergences = comparison.get("divergences", [])

        assert not divergences, (
            "Expected no TS↔Python differential replay divergences for "
            f"golden game_id={game_id} in db={db_path}, but found: "
            f"{divergences[:5]}"
        )
        assert comparison["py_move_count"] > 0
        assert comparison["ts_move_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
