#!/usr/bin/env python
"""List candidate "golden" games from one or more GameReplayDBs.

This utility scans GameReplayDB SQLite files and prints a JSON report of
games that look promising as golden-game candidates, based on:

- basic structural criteria (minimum moves, winner present, etc.), and
- richer metadata recorded by run_self_play_soak.py and other harnesses,
  including invariant violation counts and pie-rule usage.

It does not perform TS↔Python parity checks itself; those remain the job
of the dedicated parity scripts/tests. Instead it helps triage a large
corpus down to a small list of "worth inspecting" games that can then be
run through the strict differential replay and invariant suites.

Usage examples
--------------

From ``ai-service``::

    # Scan a single DB for clean, sufficiently long games
    PYTHONPATH=. python scripts/find_golden_candidates.py \\
        --db data/games/selfplay.db \\
        --min-moves 40 \\
        --output golden_candidates.json

    # Scan all *.db files under logs/cmaes and only keep games that:
    #   - have no invariant violations, and
    #   - actually exercised the pie rule at least once.
    PYTHONPATH=. python scripts/find_golden_candidates.py \\
        --db-dir logs/cmaes \\
        --require-pie-rule \\
        --output golden_pie_rule_candidates.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB  # noqa: E402


@dataclass
class CandidateGame:
    db_path: str
    game_id: str
    board_type: str
    num_players: int
    winner: Optional[int]
    termination_reason: Optional[str]
    total_moves: int
    source: Optional[str]
    invariant_violations_by_type: Dict[str, int]
    swap_sides_moves: int
    used_pie_rule: bool


def _collect_db_paths(db_args: List[str], db_dir: Optional[str]) -> List[Path]:
    paths: List[Path] = []

    for raw in db_args:
        p = Path(raw)
        if p.exists():
            paths.append(p)
        else:
            print(f"[find_golden_candidates] WARNING: DB not found: {p}", file=sys.stderr)

    if db_dir:
        root = Path(db_dir)
        if not root.exists():
            print(f"[find_golden_candidates] WARNING: db-dir does not exist: {root}", file=sys.stderr)
        else:
            for p in root.rglob("*.db"):
                if p not in paths:
                    paths.append(p)

    return sorted(set(paths))


def _load_metadata_row(game_meta: Dict[str, Any]) -> Dict[str, Any]:
    raw = game_meta.get("metadata_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def find_candidates_in_db(
    db_path: Path,
    *,
    min_moves: int,
    require_clean_invariants: bool,
    require_pie_rule: bool,
    board_type_filter: Optional[str],
    num_players_filter: Optional[int],
    termination_filter: Optional[str],
) -> List[CandidateGame]:
    db = GameReplayDB(str(db_path))
    candidates: List[CandidateGame] = []

    with db._get_conn() as conn:
        rows = conn.execute(
            """
            SELECT game_id,
                   board_type,
                   num_players,
                   winner,
                   termination_reason,
                   total_moves,
                   source,
                   metadata_json
            FROM games
            """,
        ).fetchall()

    for row in rows:
        total_moves = int(row["total_moves"] or 0)
        if total_moves < min_moves:
            continue

        game_id = str(row["game_id"])
        board_type = str(row["board_type"])
        num_players = int(row["num_players"])
        winner = row["winner"]
        termination_reason = row["termination_reason"]
        source = row["source"]

        # Basic structural/metadata filters.
        if board_type_filter and board_type != board_type_filter:
            continue
        if num_players_filter is not None and num_players != num_players_filter:
            continue
        if termination_filter and str(termination_reason or "") != termination_filter:
            continue

        meta = _load_metadata_row(dict(row))
        inv = meta.get("invariant_violations_by_type") or {}
        if not isinstance(inv, dict):
            inv = {}
        # Normalise keys to strings, values to ints
        inv_clean: Dict[str, int] = {}
        for k, v in inv.items():
            try:
                inv_clean[str(k)] = int(v)
            except Exception:
                continue

        has_violations = any(count > 0 for count in inv_clean.values())
        if require_clean_invariants and has_violations:
            continue

        swap_sides_moves = 0
        used_pie_rule = False

        try:
            swap_sides_moves = int(meta.get("swap_sides_moves") or 0)
        except Exception:
            swap_sides_moves = 0
        try:
            used_pie_rule = bool(meta.get("used_pie_rule") or False)
        except Exception:
            used_pie_rule = False

        if require_pie_rule and not used_pie_rule:
            continue

        candidates.append(
            CandidateGame(
                db_path=str(db_path),
                game_id=game_id,
                board_type=board_type,
                num_players=num_players,
                winner=winner,
                termination_reason=termination_reason,
                total_moves=total_moves,
                source=source,
                invariant_violations_by_type=inv_clean,
                swap_sides_moves=swap_sides_moves,
                used_pie_rule=used_pie_rule,
            ),
        )

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List candidate golden games from GameReplayDB SQLite files.",
    )
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        default=[],
        help="Path to a GameReplayDB SQLite file (repeatable).",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Directory to recursively scan for *.db files (optional).",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=20,
        help="Minimum total_moves for a game to be considered (default: 20).",
    )
    parser.add_argument(
        "--no-invariant-filter",
        action="store_true",
        help="Do not require invariant_violations_by_type to be empty.",
    )
    parser.add_argument(
        "--require-pie-rule",
        action="store_true",
        help="Only include games where used_pie_rule metadata is true.",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default=None,
        help="Optional board_type filter (e.g. 'square8', 'square19', 'hexagonal').",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        help="Optional num_players filter (e.g. 2, 3, 4).",
    )
    parser.add_argument(
        "--termination-reason",
        type=str,
        default=None,
        help="Optional termination_reason filter (exact match, e.g. 'env_done_flag').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output path for JSON report (default: '-' for stdout).",
    )

    args = parser.parse_args()

    db_paths = _collect_db_paths(args.db, args.db_dir)
    if not db_paths:
        print("[find_golden_candidates] No databases to scan.", file=sys.stderr)
        sys.exit(1)

    require_clean_invariants = not args.no_invariant_filter

    all_candidates: List[CandidateGame] = []
    for db_path in db_paths:
        print(f"[find_golden_candidates] Scanning {db_path}", file=sys.stderr)
        try:
            cands = find_candidates_in_db(
                db_path,
                min_moves=args.min_moves,
                require_clean_invariants=require_clean_invariants,
                require_pie_rule=args.require_pie_rule,
                board_type_filter=args.board_type,
                num_players_filter=args.num_players,
                termination_filter=args.termination_reason,
            )
            all_candidates.extend(cands)
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"[find_golden_candidates] ERROR while scanning {db_path}: " f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    payload = [asdict(c) for c in all_candidates]

    if args.output == "-" or args.output == "":
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        out_path = Path(args.output)
        out_dir = out_path.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    print(
        f"[find_golden_candidates] Found {len(all_candidates)} candidate game(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
