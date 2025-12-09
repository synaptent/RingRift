from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine


def _load_python_state(db_path: Path, game_id: str, py_index: int):
    GameReplayDB.DB_FILE_EXTENSION = ".db"
    db = GameReplayDB(str(db_path))
    state = db.get_state_at_move(game_id, py_index)
    if state is None:
        raise RuntimeError(f"get_state_at_move({game_id}, {py_index}) returned None")
    return state


def _default_ts_dump_path(
    dump_dir: Path,
    db_path: Path,
    game_id: str,
    ts_k: int,
) -> Path:
    base = db_path.name
    name = f"{base}__{game_id}__k{ts_k}.ts_state.json"
    return dump_dir / name


def _load_ts_state(ts_path: Path) -> Dict[str, Any]:
    if not ts_path.exists():
        raise FileNotFoundError(f"TS state dump not found: {ts_path}")
    with ts_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_players_py(state) -> Dict[int, Tuple[int, int, int]]:
    summary: Dict[int, Tuple[int, int, int]] = {}
    for p in state.players:
        summary[p.player_number] = (p.eliminated_rings, p.territory_spaces, p.rings_in_hand)
    return summary


def _summarize_players_ts(ts_state: Dict[str, Any]) -> Dict[int, Tuple[int, int, int]]:
    summary: Dict[int, Tuple[int, int, int]] = {}
    for p in ts_state.get("players", []):
        num = int(p["playerNumber"])
        summary[num] = (
            int(p.get("eliminatedRings", 0)),
            int(p.get("territorySpaces", 0)),
            int(p.get("ringsInHand", 0)),
        )
    return summary


def _summarize_stacks_py(state) -> Dict[str, Tuple[int, int]]:
    board = state.board
    out: Dict[str, Tuple[int, int]] = {}
    for key, stack in (board.stacks or {}).items():
        out[str(key)] = (stack.stack_height, stack.controlling_player)
    return out


def _summarize_stacks_ts(ts_state: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    board = ts_state.get("board", {})
    stacks = board.get("stacks", {}) or {}
    out: Dict[str, Tuple[int, int]] = {}
    for key, stack in stacks.items():
        out[str(key)] = (int(stack.get("stackHeight", 0)), int(stack.get("controllingPlayer", 0)))
    return out


def _summarize_collapsed_py(state) -> Dict[str, int]:
    board = state.board
    out: Dict[str, int] = {}
    for key, owner in (board.collapsed_spaces or {}).items():
        out[str(key)] = int(owner)
    return out


def _summarize_collapsed_ts(ts_state: Dict[str, Any]) -> Dict[str, int]:
    board = ts_state.get("board", {})
    collapsed = board.get("collapsedSpaces", {}) or {}
    out: Dict[str, int] = {}
    for key, owner in collapsed.items():
        out[str(key)] = int(owner)
    return out


def diff_state(
    db_path: Path,
    game_id: str,
    ts_k: int,
    dump_dir: Path,
    ts_state_path: Path | None = None,
) -> None:
    if ts_k < 0:
        raise ValueError("ts_k must be >= 0")
    py_index = ts_k - 1 if ts_k > 0 else 0

    py_state = _load_python_state(db_path, game_id, py_index)
    GameEngine._check_victory(py_state)

    if ts_state_path is None:
        ts_state_path = _default_ts_dump_path(dump_dir, db_path, game_id, ts_k)
    ts_state = _load_ts_state(ts_state_path)

    print(f"Python state @ move_index={py_index}")
    print(f"  phase={py_state.current_phase} " f"player={py_state.current_player} " f"status={py_state.game_status}")
    print(
        f"  stacks={len(py_state.board.stacks)} "
        f"collapsed={len(py_state.board.collapsed_spaces)} "
        f"total_elims={py_state.total_rings_eliminated}"
    )
    for num, (elim, terr, hand) in sorted(_summarize_players_py(py_state).items()):
        print(f"  P{num}: elim={elim} terr={terr} hand={hand}")

    print()
    print(f"TS state @ k={ts_k} from {ts_state_path}")
    print(
        f"  phase={ts_state.get('currentPhase')} "
        f"player={ts_state.get('currentPlayer')} "
        f"status={ts_state.get('gameStatus')}"
    )
    ts_board = ts_state.get("board", {})
    print(
        f"  stacks={len(ts_board.get('stacks', {}) or {})} "
        f"collapsed={len(ts_board.get('collapsedSpaces', {}) or {})} "
        f"total_elims={ts_state.get('totalRingsEliminated')}"
    )
    for num, (elim, terr, hand) in sorted(_summarize_players_ts(ts_state).items()):
        print(f"  P{num}: elim={elim} terr={terr} hand={hand}")

    print("\n=== Player summary diff ===")
    py_players = _summarize_players_py(py_state)
    ts_players = _summarize_players_ts(ts_state)
    all_players = sorted(set(py_players) | set(ts_players))
    for num in all_players:
        py_vals = py_players.get(num)
        ts_vals = ts_players.get(num)
        if py_vals != ts_vals:
            print(f"  P{num}: PY={py_vals} TS={ts_vals}")

    print("\n=== Stack key diff ===")
    py_stacks = _summarize_stacks_py(py_state)
    ts_stacks = _summarize_stacks_ts(ts_state)
    py_keys = set(py_stacks)
    ts_keys = set(ts_stacks)
    only_py = sorted(py_keys - ts_keys)
    only_ts = sorted(ts_keys - py_keys)
    if only_py:
        print(f"  Stacks only in PY ({len(only_py)}): {', '.join(only_py[:20])}")
        if len(only_py) > 20:
            print(f"    ... {len(only_py) - 20} more")
    if only_ts:
        print(f"  Stacks only in TS ({len(only_ts)}): {', '.join(only_ts[:20])}")
        if len(only_ts) > 20:
            print(f"    ... {len(only_ts) - 20} more")

    shared = sorted(py_keys & ts_keys)
    mismatched = []
    for key in shared:
        if py_stacks[key] != ts_stacks[key]:
            mismatched.append(key)
    if mismatched:
        print(f"  Stacks with differing (height,controller) ({len(mismatched)}):")
        for key in mismatched[:20]:
            print(f"    {key}: PY={py_stacks[key]} TS={ts_stacks[key]}")
        if len(mismatched) > 20:
            print(f"    ... {len(mismatched) - 20} more")

    print("\n=== Collapsed territory diff ===")
    py_collapsed = _summarize_collapsed_py(py_state)
    ts_collapsed = _summarize_collapsed_ts(ts_state)
    py_c_keys = set(py_collapsed)
    ts_c_keys = set(ts_collapsed)
    only_py_c = sorted(py_c_keys - ts_c_keys)
    only_ts_c = sorted(ts_c_keys - py_c_keys)
    if only_py_c:
        print(f"  Collapsed only in PY ({len(only_py_c)}): {', '.join(only_py_c[:20])}")
        if len(only_py_c) > 20:
            print(f"    ... {len(only_py_c) - 20} more")
    if only_ts_c:
        print(f"  Collapsed only in TS ({len(only_ts_c)}): {', '.join(only_ts_c[:20])}")
        if len(only_ts_c) > 20:
            print(f"    ... {len(only_ts_c) - 20} more")

    shared_c = sorted(py_c_keys & ts_c_keys)
    mismatched_c = []
    for key in shared_c:
        if py_collapsed[key] != ts_collapsed[key]:
            mismatched_c.append(key)
    if mismatched_c:
        print(f"  Collapsed cells with differing owners ({len(mismatched_c)}):")
        for key in mismatched_c[:20]:
            print(f"    {key}: PY={py_collapsed[key]} TS={ts_collapsed[key]}")
        if len(mismatched_c) > 20:
            print(f"    ... {len(mismatched_c) - 20} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Diff a Python GameReplayDB state against a TS sandbox replay dump " "for a single (db, game, k).")
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to GameReplayDB .db file (same as used by parity harness).",
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game id within the GameReplayDB.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help=("TS replay step index k. k=0 is initial state; " "k>=1 corresponds to Python get_state_at_move(k-1)."),
    )
    parser.add_argument(
        "--dump-dir",
        default="../ts-replay-dumps",
        help=("Directory where TS replay dumps are stored. " "Defaults to ../ts-replay-dumps relative to ai-service/."),
    )
    parser.add_argument(
        "--ts-state-path",
        default=None,
        help=(
            "Optional explicit path to TS state JSON. "
            "When omitted, constructed from dump-dir, db basename, game id, and k."
        ),
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    dump_dir = Path(args.dump_dir).resolve()
    ts_state_path = Path(args.ts_state_path).resolve() if args.ts_state_path else None

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if ts_state_path is not None and not ts_state_path.exists():
        raise FileNotFoundError(f"Explicit TS state path not found: {ts_state_path}")

    print(f"DB:   {db_path}")
    print(f"Game: {args.game}")
    print(f"TS k: {args.k}")
    if ts_state_path is not None:
        print(f"TS state path: {ts_state_path}")
    else:
        print(f"Dump dir: {dump_dir}")
    print()

    diff_state(
        db_path=db_path,
        game_id=args.game,
        ts_k=args.k,
        dump_dir=dump_dir,
        ts_state_path=ts_state_path,
    )


if __name__ == "__main__":
    main()
