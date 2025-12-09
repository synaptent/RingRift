from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from scripts import debug_ts_python_state_diff as dbg  # type: ignore
from app.rules.serialization import deserialize_game_state


def _compute_structural_diffs_from_states(
    py_state: Any,
    ts_state: Dict[str, Any],
) -> Tuple[bool, str]:
    """Compute a concise structural diff summary between Python and TS states."""
    players_py = dbg._summarize_players_py(py_state)  # type: ignore[attr-defined]
    players_ts = dbg._summarize_players_ts(ts_state)  # type: ignore[attr-defined]

    stacks_py = dbg._summarize_stacks_py(py_state)  # type: ignore[attr-defined]
    stacks_ts = dbg._summarize_stacks_ts(ts_state)  # type: ignore[attr-defined]

    collapsed_py = dbg._summarize_collapsed_py(py_state)  # type: ignore[attr-defined]
    collapsed_ts = dbg._summarize_collapsed_ts(ts_state)  # type: ignore[attr-defined]

    reasons = []

    if players_py != players_ts:
        reasons.append("players(elim/territory/hand) differ")

    if stacks_py != stacks_ts:
        only_py = set(stacks_py.keys()) - set(stacks_ts.keys())
        only_ts = set(stacks_ts.keys()) - set(stacks_py.keys())
        mismatched = {
            key for key in (set(stacks_py.keys()) & set(stacks_ts.keys())) if stacks_py[key] != stacks_ts[key]
        }
        if only_py:
            reasons.append(f"{len(only_py)} stacks only in PY")
        if only_ts:
            reasons.append(f"{len(only_ts)} stacks only in TS")
        if mismatched:
            reasons.append(f"{len(mismatched)} stacks with differing (height,controller)")

    if collapsed_py != collapsed_ts:
        only_py_c = set(collapsed_py.keys()) - set(collapsed_ts.keys())
        only_ts_c = set(collapsed_ts.keys()) - set(collapsed_py.keys())
        mismatched_c = {
            key
            for key in (set(collapsed_py.keys()) & set(collapsed_ts.keys()))
            if collapsed_py[key] != collapsed_ts[key]
        }
        if only_py_c:
            reasons.append(f"{len(only_py_c)} collapsed cells only in PY")
        if only_ts_c:
            reasons.append(f"{len(only_ts_c)} collapsed cells only in TS")
        if mismatched_c:
            reasons.append(f"{len(mismatched_c)} collapsed cells with differing owners")

    structural = bool(reasons)
    return structural, "; ".join(reasons) if reasons else "no structural differences"


def diff_state_from_bundle(bundle_path: Path, ts_k: int | None = None) -> None:
    """Load a state_bundle emitted by the parity harness and print a concise diff."""
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    db_path = bundle.get("db_path")
    game_id = bundle.get("game_id")
    diverged_at = int(bundle.get("diverged_at") or 0)
    mismatch_kinds = bundle.get("mismatch_kinds") or []
    mismatch_context = bundle.get("mismatch_context")
    ts_k_values = bundle.get("ts_k_values") or []

    # Choose k: explicit argument wins, then diverged_at, then last k in ts_k_values.
    if ts_k is None:
        if diverged_at > 0:
            ts_k = diverged_at
        elif ts_k_values:
            try:
                ts_k = int(ts_k_values[-1])
            except Exception:
                ts_k = 0
        else:
            ts_k = 0

    py_states = bundle.get("python_states") or {}
    ts_states = bundle.get("ts_states") or {}

    py_raw = py_states.get(str(ts_k))
    ts_raw = ts_states.get(str(ts_k))

    print(f"Bundle: {bundle_path}")
    print(f"DB:     {db_path}")
    print(f"Game:   {game_id}")
    print(f"ts_k:   {ts_k}")
    print(f"diverged_at: {diverged_at}")
    print(f"mismatch_kinds: {mismatch_kinds}")
    print(f"mismatch_context: {mismatch_context}")
    print(f"available_ts_k_values: {ts_k_values}")
    print()

    if py_raw is None:
        print(f"No Python state recorded for ts_k={ts_k} in bundle.")
        return
    if ts_raw is None:
        print(f"No TS state recorded for ts_k={ts_k} in bundle.")
        return

    # Reconstruct Python GameState from serialized JSON and run a concise diff.
    py_state = deserialize_game_state(py_raw)
    ts_state = ts_raw

    print("Python state:")
    print(f"  phase={py_state.current_phase} " f"player={py_state.current_player} " f"status={py_state.game_status}")
    print(
        f"  stacks={len(py_state.board.stacks)} "
        f"collapsed={len(py_state.board.collapsed_spaces)} "
        f"total_elims={py_state.total_rings_eliminated}"
    )

    print("\nTS state:")
    ts_board = ts_state.get("board", {}) or {}
    print(
        f"  phase={ts_state.get('currentPhase')} "
        f"player={ts_state.get('currentPlayer')} "
        f"status={ts_state.get('gameStatus')}"
    )
    print(
        f"  stacks={len(ts_board.get('stacks', {}) or {})} "
        f"collapsed={len(ts_board.get('collapsedSpaces', {}) or {})} "
        f"total_elims={ts_state.get('totalRingsEliminated')}"
    )

    structural, reason = _compute_structural_diffs_from_states(py_state, ts_state)

    print("\n=== Concise structural diff summary ===")
    print(f"  structural_diff={structural}")
    print(f"  reason={reason}")

    # For quick triage, also print player summaries and a small sample of key diffs.
    players_py = dbg._summarize_players_py(py_state)  # type: ignore[attr-defined]
    players_ts = dbg._summarize_players_ts(ts_state)  # type: ignore[attr-defined]
    print("\nPlayers (PY vs TS):")
    all_players = sorted(set(players_py) | set(players_ts))
    for num in all_players:
        print(f"  P{num}: PY={players_py.get(num)} TS={players_ts.get(num)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a parity state_bundle JSON and print a concise TS vs Python "
            "structural diff at a chosen ts_k (default: diverged_at)."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to a *.state_bundle.json file emitted by check_ts_python_replay_parity.py.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "TS replay step index k to diff. When omitted, defaults to diverged_at "
            "from the bundle (or the last entry in ts_k_values)."
        ),
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    diff_state_from_bundle(bundle_path, ts_k=args.k)


if __name__ == "__main__":
    main()
