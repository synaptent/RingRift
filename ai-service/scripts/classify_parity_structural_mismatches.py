"""
Classify parity divergences as structural vs bookkeeping-only.

This script reads parity_summary.latest.json (or a supplied summary),
replays the corresponding games into the TS sandbox to dump TS states
at the reported divergence indices, and then compares:

- stacks (position -> (height, controller))
- collapsed_spaces (position -> owner)
- per-player (eliminated_rings, territory_spaces, rings_in_hand)

If any of those differ between Python and TS at the divergence point,
the entry is tagged as a *structural* mismatch; otherwise it's treated
as a bookkeeping-only mismatch (e.g. phase/turn differences with
identical board and player counts).

The output JSON can be used to decide which games to exclude from
training datasets (e.g., drop any game that has at least one
structural mismatch).

Usage (from ai-service/):

    PYTHONPATH=. python scripts/classify_parity_structural_mismatches.py \\
        --summary parity_summary.latest.json \\
        --output parity_structural_classification.json \\
        --limit-games 50

By default, it assumes TS dumps live under ../ts-replay-dumps and
that scripts/selfplay-db-ts-replay.ts is available at the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

from scripts import debug_ts_python_state_diff as dbg  # type: ignore


@dataclass
class StructuralClassification:
    db_path: str
    game_id: str
    ts_k: int
    structural: bool
    reason: str


def _run_ts_dump(
    repo_root: Path,
    db_path: Path,
    game_id: str,
    ts_k: int,
    dump_dir: Path,
) -> None:
    """
    Invoke the TS replay harness to dump a single TS state at (game_id, ts_k).

    This relies on scripts/selfplay-db-ts-replay.ts honouring the
    RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K and RINGRIFT_TS_REPLAY_DUMP_DIR
    environment variables.
    """
    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")
    env["RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K"] = str(ts_k)
    env["RINGRIFT_TS_REPLAY_DUMP_DIR"] = str(dump_dir)

    cmd = [
        "npx",
        "ts-node",
        "-T",
        "scripts/selfplay-db-ts-replay.ts",
        "--db",
        str(db_path),
        "--game",
        game_id,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay failed for {game_id} @ k={ts_k}: " f"code={proc.returncode}, stderr={proc.stderr.strip()}"
        )


def _summarize_python_state(db_path: Path, game_id: str, ts_k: int) -> Any:
    """
    Load the Python state at the index corresponding to TS k.

    TS k snapshots reflect the state *after* applying move k, while
    GameReplayDB.get_state_at_move(i) returns the state after move i.
    We follow the same mapping as debug_ts_python_state_diff: py_index = k-1 (or 0).
    """
    py_index = ts_k - 1 if ts_k > 0 else 0
    state = dbg._load_python_state(db_path, game_id, py_index)  # type: ignore[attr-defined]
    # Ensure victory side effects are applied (e.g., collapsed spaces/territory)
    from app.game_engine import GameEngine

    GameEngine._check_victory(state)
    return state


def _compute_structural_diffs(
    py_state: Any,
    ts_state: dict[str, Any],
) -> tuple[bool, str]:
    """
    Compare core structural aspects of the two states.

    Returns (is_structural, reason).
    """
    players_py = dbg._summarize_players_py(py_state)  # type: ignore[attr-defined]
    players_ts = dbg._summarize_players_ts(ts_state)  # type: ignore[attr-defined]

    stacks_py = dbg._summarize_stacks_py(py_state)  # type: ignore[attr-defined]
    stacks_ts = dbg._summarize_stacks_ts(ts_state)  # type: ignore[attr-defined]

    collapsed_py = dbg._summarize_collapsed_py(py_state)  # type: ignore[attr-defined]
    collapsed_ts = dbg._summarize_collapsed_ts(ts_state)  # type: ignore[attr-defined]

    reasons: list[str] = []

    if players_py != players_ts:
        reasons.append("players(elim/territory/hand) differ")
    if stacks_py != stacks_ts:
        only_py = set(stacks_py.keys()) - set(stacks_ts.keys())
        only_ts = set(stacks_ts.keys()) - set(stacks_py.keys())
        if only_py:
            reasons.append(f"stacks only in PY: {sorted(only_py)}")
        if only_ts:
            reasons.append(f"stacks only in TS: {sorted(only_ts)}")
        common = set(stacks_py.keys()) & set(stacks_ts.keys())
        mismatched = [key for key in common if stacks_py[key] != stacks_ts[key]]
        if mismatched:
            reasons.append(f"stacks with mismatched geometry: {sorted(mismatched)}")

    if collapsed_py != collapsed_ts:
        only_py = set(collapsed_py.keys()) - set(collapsed_ts.keys())
        only_ts = set(collapsed_ts.keys()) - set(collapsed_py.keys())
        if only_py:
            reasons.append(f"collapsed only in PY: {sorted(only_py)}")
        if only_ts:
            reasons.append(f"collapsed only in TS: {sorted(only_ts)}")
        common = set(collapsed_py.keys()) & set(collapsed_ts.keys())
        mismatched = [key for key in common if collapsed_py[key] != collapsed_ts[key]]
        if mismatched:
            reasons.append(f"collapsed with mismatched owner: {sorted(mismatched)}")

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def classify_structural_mismatches(
    summary_path: Path,
    dump_dir: Path,
    output_path: Path,
    limit_games: int | None = None,
) -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads(summary_path.read_text())

    divergences: list[dict[str, Any]] = data.get("semantic_divergences", [])
    results: list[StructuralClassification] = []

    processed_games: dict[tuple[str, str, int], bool] = {}

    for idx, entry in enumerate(divergences):
        if limit_games is not None and idx >= limit_games:
            break

        db_path = Path(entry["db_path"]).resolve()
        game_id = entry["game_id"]
        ts_k = int(entry["diverged_at"])

        key = (str(db_path), game_id, ts_k)
        if key in processed_games:
            continue

        processed_games[key] = True

        try:
            _run_ts_dump(root, db_path, game_id, ts_k, dump_dir)
        except Exception as e:  # noqa: BLE001
            results.append(
                StructuralClassification(
                    db_path=str(db_path),
                    game_id=game_id,
                    ts_k=ts_k,
                    structural=True,
                    reason=f"TS replay failed: {e}",
                )
            )
            continue

        ts_state_path = dbg._default_ts_dump_path(dump_dir, db_path, game_id, ts_k)  # type: ignore[attr-defined]
        try:
            ts_state = dbg._load_ts_state(ts_state_path)  # type: ignore[attr-defined]
        except FileNotFoundError as e:
            results.append(
                StructuralClassification(
                    db_path=str(db_path),
                    game_id=game_id,
                    ts_k=ts_k,
                    structural=True,
                    reason=str(e),
                )
            )
            continue

        py_state = _summarize_python_state(db_path, game_id, ts_k)
        structural, reason = _compute_structural_diffs(py_state, ts_state)
        results.append(
            StructuralClassification(
                db_path=str(db_path),
                game_id=game_id,
                ts_k=ts_k,
                structural=structural,
                reason=reason,
            )
        )

    # Aggregate per-game structural flags for convenience.
    per_game: dict[tuple[str, str], bool] = {}
    for r in results:
        key = (r.db_path, r.game_id)
        if r.structural:
            per_game[key] = True
        elif key not in per_game:
            per_game[key] = False

    output = {
        "entries": [asdict(r) for r in results],
        "per_game_structural": {f"{db}::{gid}": structural for (db, gid), structural in per_game.items()},
    }
    output_path.write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify parity divergences as structural vs bookkeeping-only.")
    parser.add_argument(
        "--summary",
        type=str,
        default="parity_summary.latest.json",
        help="Path to parity summary JSON (default: parity_summary.latest.json)",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="../ts-replay-dumps",
        help="Directory where TS replay dumps are/will be stored " "(default: ../ts-replay-dumps from ai-service/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parity_structural_classification.json",
        help="Output JSON path for structural classification results",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional limit on number of divergences to classify " "(primarily for quick testing).",
    )

    args = parser.parse_args()
    summary_path = Path(args.summary)
    dump_dir = Path(args.dump_dir).resolve()
    output_path = Path(args.output)

    dump_dir.mkdir(parents=True, exist_ok=True)
    classify_structural_mismatches(
        summary_path=summary_path,
        dump_dir=dump_dir,
        output_path=output_path,
        limit_games=args.limit_games,
    )


if __name__ == "__main__":
    main()
