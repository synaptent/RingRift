#!/usr/bin/env python3
"""Mine chain-capture tactics puzzles from RingRift replay databases.

Scans recorded games for movement-phase positions where one move initiates a
forced overtaking-capture chain that is strictly better (by captured-ring
count) than every alternative. Emits a JSON puzzle file suitable for a
"daily tactics" product surface and validates its own output.

The margin metric is purely structural — no neural network required:
    score(move) = maximum number of rings captured by the forced chain
                  initiated by `move` (0 for non-capture moves)
A position becomes a puzzle when
    best_score >= --min-chain  AND  best_score - second_best >= --min-margin
which guarantees a unique best first move.

Read-only by design: only GameReplayDB query paths are used. For absolute
safety against journal/schema side effects on live training databases, pass
--copy-to-temp to mine from a temporary copy.

Usage:
    PYTHONPATH=. python scripts/mine_chain_capture_puzzles.py \
        --db data/games/canonical_hex8_2p.db --copy-to-temp \
        --max-puzzles 60 --output /tmp/hex8_2p_puzzles.json

    # Re-validate an existing puzzle file
    PYTHONPATH=. python scripts/mine_chain_capture_puzzles.py \
        --validate /tmp/hex8_2p_puzzles.json

Schema: docs/puzzles/PUZZLE_FORMAT.md (schema_version 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.game_engine import GameEngine
from app.models import (
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
)

logger = logging.getLogger("mine_puzzles")

SCHEMA_VERSION = 1
THEME = "chain_capture"


# ---------------------------------------------------------------------------
# Chain evaluation (structural, engine-only)
# ---------------------------------------------------------------------------

def chain_score(state: GameState, move: Move) -> tuple[int, list[Move]]:
    """Return (rings captured, principal variation) for the forced chain
    initiated by ``move``.

    Chain continuation is mandatory while legal segments exist
    (RR-CANON-R103), but the mover chooses which segment — so the score is
    the max over continuation choices. Non-capture moves score 0.
    """
    if move.type != MoveType.OVERTAKING_CAPTURE:
        return 0, [move]

    mover = move.player

    def dfs(s: GameState, mv: Move) -> tuple[int, list[Move]]:
        ns = GameEngine.apply_move(s, mv, trace_mode=True)
        if (
            ns.game_status == GameStatus.ACTIVE
            and ns.current_phase == GamePhase.CHAIN_CAPTURE
            and ns.current_player == mover
        ):
            conts = [
                c
                for c in GameEngine.get_valid_moves(ns, mover)
                if c.type == MoveType.CONTINUE_CAPTURE_SEGMENT
            ]
            if conts:
                sub_score, sub_pv = max(
                    (dfs(ns, c) for c in conts), key=lambda t: t[0]
                )
                return 1 + sub_score, [mv, *sub_pv]
        return 1, [mv]

    return dfs(state, move)


def score_position(state: GameState) -> tuple[list[tuple[int, Move, list[Move]]], int]:
    """Score every legal move of the acting player by forced-chain captures.

    Returns (scored_moves sorted best-first, number of legal moves).
    """
    moves = GameEngine.get_valid_moves(state, state.current_player)
    scored = []
    for mv in moves:
        score, pv = chain_score(state, mv)
        scored.append((score, mv, pv))
    scored.sort(key=lambda t: -t[0])
    return scored, len(moves)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def _move_to_dict(move: Move) -> dict[str, Any]:
    return json.loads(move.model_dump_json(by_alias=True, exclude_none=True))


def _state_to_dict(state: GameState) -> dict[str, Any]:
    return json.loads(state.model_dump_json(by_alias=True))


def mine_game(
    game_meta: dict[str, Any],
    initial_state: GameState,
    moves: list[Move],
    *,
    min_chain: int = 3,
    min_margin: int = 2,
    min_ply: int = 6,
    max_per_game: int = 2,
    source_db: str = "",
) -> Iterator[dict[str, Any]]:
    """Replay one recorded game, yielding puzzle dicts as they are found."""
    state = initial_state
    found = 0
    for ply, recorded_move in enumerate(moves):
        if found >= max_per_game or state.game_status != GameStatus.ACTIVE:
            break
        if ply >= min_ply and state.current_phase == GamePhase.MOVEMENT:
            scored, num_legal = score_position(state)
            if num_legal >= 2:
                best_score, _best_move, best_pv = scored[0]
                second_score = scored[1][0]
                if best_score >= min_chain and best_score - second_score >= min_margin:
                    game_id = str(game_meta.get("game_id", "unknown"))
                    num_players = len(state.players)
                    yield {
                        "id": f"{state.board.type.value}_{num_players}p_{game_id[:12]}_{ply}",
                        "schema_version": SCHEMA_VERSION,
                        "theme": THEME,
                        "board_type": state.board.type.value,
                        "num_players": num_players,
                        "player_to_move": state.current_player,
                        "state": _state_to_dict(state),
                        "solution": {
                            "moves": [_move_to_dict(m) for m in best_pv],
                            "score": best_score,
                            "second_best_score": second_score,
                            "margin": best_score - second_score,
                        },
                        "source": {
                            "db": source_db,
                            "game_id": game_id,
                            "ply": ply,
                        },
                    }
                    found += 1
        try:
            state = GameEngine.apply_move(state, recorded_move, trace_mode=True)
        except Exception as exc:  # pragma: no cover - malformed history
            logger.warning(
                "Replay diverged in game %s at ply %d: %s",
                game_meta.get("game_id"), ply, exc,
            )
            return


def mine_games(
    games: Iterable[tuple[dict[str, Any], GameState, list[Move]]],
    *,
    max_puzzles: int = 50,
    **mine_kwargs: Any,
) -> list[dict[str, Any]]:
    """Mine puzzles from an iterable of (metadata, initial_state, moves)."""
    puzzles: list[dict[str, Any]] = []
    for game_meta, initial_state, moves in games:
        for puzzle in mine_game(game_meta, initial_state, moves, **mine_kwargs):
            puzzles.append(puzzle)
            if len(puzzles) >= max_puzzles:
                return puzzles
    return puzzles


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_puzzle(puzzle: dict[str, Any]) -> tuple[bool, str]:
    """Recompute the puzzle from its stored state and check the solution.

    Returns (ok, reason). A valid puzzle's stored best move must be the
    unique argmax of the recomputed chain scores with the stated margin.
    """
    try:
        state = GameState.model_validate(puzzle["state"])
    except Exception as exc:
        return False, f"state failed to deserialize: {exc}"
    if state.current_phase != GamePhase.MOVEMENT:
        return False, f"state is in phase {state.current_phase}, expected movement"

    scored, num_legal = score_position(state)
    if num_legal < 2:
        return False, "position has fewer than 2 legal moves"
    best_score, best_move, _pv = scored[0]
    second_score = scored[1][0]

    sol = puzzle["solution"]
    if best_score != sol["score"]:
        return False, f"best score {best_score} != stored {sol['score']}"
    if second_score != sol["second_best_score"]:
        return False, f"second-best {second_score} != stored {sol['second_best_score']}"
    if best_score - second_score < 1:
        return False, "best move is not unique"

    stored_first = sol["moves"][0]
    recomputed_first = _move_to_dict(best_move)
    for key in ("type", "from", "to", "captureTarget"):
        if stored_first.get(key) != recomputed_first.get(key):
            return False, (
                f"solution first move mismatch on {key!r}: "
                f"{stored_first.get(key)} != {recomputed_first.get(key)}"
            )
    return True, "ok"


def validate_file(path: Path) -> int:
    data = json.loads(path.read_text())
    puzzles = data["puzzles"] if isinstance(data, dict) else data
    failures = 0
    for puzzle in puzzles:
        ok, reason = validate_puzzle(puzzle)
        if not ok:
            failures += 1
            print(f"INVALID {puzzle.get('id')}: {reason}")
    print(f"{len(puzzles) - failures}/{len(puzzles)} puzzles valid")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", action="append", default=[], help="Replay DB path (repeatable)")
    parser.add_argument("--copy-to-temp", action="store_true",
                        help="Mine from a temporary copy of each DB (never touches the original)")
    parser.add_argument("--board-type", default=None)
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--limit-games", type=int, default=300, help="Games to scan per DB")
    parser.add_argument("--min-chain", type=int, default=3, help="Minimum captured rings for the best chain")
    parser.add_argument("--min-margin", type=int, default=2, help="Required (best - second best) gap")
    parser.add_argument("--min-ply", type=int, default=6)
    parser.add_argument("--max-per-game", type=int, default=2)
    parser.add_argument("--max-puzzles", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--validate", type=Path, default=None, help="Validate an existing puzzle file and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.validate:
        return 1 if validate_file(args.validate) else 0

    if not args.db or args.output is None:
        parser.error("--db and --output are required for mining")

    from app.db.game_replay import GameReplayDB  # heavy import, keep local

    filters: dict[str, Any] = {"limit": args.limit_games}
    if args.board_type:
        filters["board_type"] = args.board_type
    if args.num_players:
        filters["num_players"] = args.num_players

    puzzles: list[dict[str, Any]] = []
    tempdir = tempfile.mkdtemp(prefix="ringrift_puzzle_mine_") if args.copy_to_temp else None
    try:
        for db_path in args.db:
            src = Path(db_path)
            open_path = src
            if tempdir:
                open_path = Path(tempdir) / src.name
                logger.info("Copying %s -> %s", src, open_path)
                shutil.copy2(src, open_path)
            db = GameReplayDB(str(open_path))
            try:
                remaining = args.max_puzzles - len(puzzles)
                if remaining <= 0:
                    break
                puzzles.extend(
                    mine_games(
                        db.iterate_games(**filters),
                        max_puzzles=remaining,
                        min_chain=args.min_chain,
                        min_margin=args.min_margin,
                        min_ply=args.min_ply,
                        max_per_game=args.max_per_game,
                        source_db=src.name,
                    )
                )
            finally:
                db.close()
            logger.info("%s: %d puzzles so far", src.name, len(puzzles))
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)

    invalid = sum(1 for p in puzzles if not validate_puzzle(p)[0])
    if invalid:
        logger.error("%d/%d mined puzzles failed self-validation", invalid, len(puzzles))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "theme": THEME,
        "count": len(puzzles),
        "puzzles": puzzles,
    }, indent=1))
    logger.info("Wrote %d validated puzzles to %s", len(puzzles), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
