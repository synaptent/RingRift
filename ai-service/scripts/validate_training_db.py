#!/usr/bin/env python3
"""Validate replay DB readiness for training without ad-hoc SQL.

This is a lightweight operational check. It reports parity-status distribution,
quarantined/excluded games, and whether each game can replay its first N moves
through the current canonical Python engine. The optional --fix mode only marks
games that pass this prefix replay and canonical phase-history check.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db import GameReplayDB
from app.game_engine import GameEngine
from app.rules.history_validation import validate_canonical_history_for_game


QUARANTINE_PARITY_STATUSES = {
    "failed",
    "error",
    "non_canonical_history",
    "quarantined",
    "smoke_test_excluded",
}


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _fetch_game_rows(db_path: str) -> tuple[list[dict[str, Any]], set[str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, "games")
        rows = conn.execute("SELECT * FROM games ORDER BY created_at ASC").fetchall()
        return [dict(row) for row in rows], columns
    finally:
        conn.close()


def _is_quarantined(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    source = str(row.get("source") or metadata.get("source") or "").lower()
    parity_status = str(row.get("parity_status") or metadata.get("parity_status") or "pending").lower()
    return (
        "quarantine" in source
        or parity_status in QUARANTINE_PARITY_STATUSES
        or _truthy(row.get("excluded_from_training", 0))
        or _truthy(metadata.get("excluded_from_training", False))
    )


def _validate_game_prefix(
    db: GameReplayDB,
    game_id: str,
    *,
    replay_moves: int,
) -> tuple[bool, str | None]:
    history_report = validate_canonical_history_for_game(db, game_id)
    if not history_report.is_canonical:
        first_issue = history_report.issues[0] if history_report.issues else None
        if first_issue is not None:
            return False, (
                f"noncanonical_history at move {first_issue.move_number}: "
                f"{first_issue.phase}/{first_issue.move_type} ({first_issue.reason})"
            )
        return False, "noncanonical_history"

    moves = db.get_moves(game_id, start=0, end=replay_moves)
    if not moves:
        return False, "no_moves"

    state = db.get_initial_state(game_id)
    if state is None:
        return False, "missing_initial_state"

    try:
        for move in moves:
            state = GameEngine.apply_move(state, move, trace_mode=True)
    except Exception as exc:
        return False, f"prefix_replay_failed: {type(exc).__name__}: {exc}"

    return True, None


def validate_training_db(
    db_path: str,
    *,
    replay_moves: int = 10,
    fix: bool = False,
    fix_status: str = "canonical_history_ok",
    max_failures: int = 50,
) -> dict[str, Any]:
    rows, columns = _fetch_game_rows(db_path)
    try:
        db = GameReplayDB(db_path)
    except Exception as exc:
        return {
            "db_path": db_path,
            "total_games": len(rows),
            "parity_status_counts": {},
            "pass_count": 0,
            "fail_count": len(rows),
            "quarantine_count": 0,
            "failure_reasons": {"open_or_schema_error": len(rows) or 1},
            "failures_sample": [
                {
                    "game_id": row.get("game_id"),
                    "parity_status": row.get("parity_status", "pending"),
                    "reason": f"open_or_schema_error: {type(exc).__name__}: {exc}",
                }
                for row in rows[:max_failures]
            ],
            "fix_applied": False,
            "fix_status": None,
            "fixed_count": 0,
            "replay_moves_checked": replay_moves,
            "ok": False,
        }

    parity_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    quarantine_count = 0
    fixed_count = 0
    fixed_ids: list[str] = []

    for row in rows:
        game_id = str(row.get("game_id") or "")
        metadata = _parse_metadata(row.get("metadata_json"))
        parity_status = str(row.get("parity_status") or metadata.get("parity_status") or "pending")
        parity_counts[parity_status] += 1

        if _is_quarantined(row, metadata):
            quarantine_count += 1
            continue

        ok, reason = _validate_game_prefix(db, game_id, replay_moves=replay_moves)
        if ok:
            pass_count += 1
            if fix and "parity_status" in columns and parity_status != "passed":
                fixed_ids.append(game_id)
        else:
            fail_count += 1
            failure_key = reason.split(":", 1)[0] if reason else "unknown"
            failure_reasons[failure_key] += 1
            if len(failures) < max_failures:
                failures.append(
                    {
                        "game_id": game_id,
                        "parity_status": parity_status,
                        "reason": reason or "unknown",
                    }
                )

    if fix and fixed_ids:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(db_path)
        try:
            conn.executemany(
                """
                UPDATE games
                SET parity_status = ?, parity_checked_at = ?
                WHERE game_id = ?
                """,
                [(fix_status, now, game_id) for game_id in fixed_ids],
            )
            conn.commit()
            fixed_count = len(fixed_ids)
        finally:
            conn.close()

    return {
        "db_path": db_path,
        "total_games": len(rows),
        "parity_status_counts": dict(sorted(parity_counts.items())),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "quarantine_count": quarantine_count,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "failures_sample": failures,
        "fix_applied": fix,
        "fix_status": fix_status if fix else None,
        "fixed_count": fixed_count,
        "replay_moves_checked": replay_moves,
        "ok": fail_count == 0,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a GameReplayDB for training readiness.")
    parser.add_argument("db_path", type=str, help="Path to GameReplayDB SQLite file.")
    parser.add_argument(
        "--replay-moves",
        type=int,
        default=10,
        help="Number of prefix moves to replay for each non-quarantined game.",
    )
    parser.add_argument("--fix", action="store_true", help="Update parity_status for games that pass.")
    parser.add_argument(
        "--fix-status",
        default="canonical_history_ok",
        help="Status to write for passing games under --fix.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--max-failures", type=int, default=50, help="Maximum failures to include.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.replay_moves < 1:
        raise ValueError("--replay-moves must be >= 1")
    if not Path(args.db_path).exists():
        raise FileNotFoundError(args.db_path)

    report = validate_training_db(
        args.db_path,
        replay_moves=args.replay_moves,
        fix=args.fix,
        fix_status=args.fix_status,
        max_failures=args.max_failures,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DB: {report['db_path']}")
        print(f"Games: {report['total_games']}")
        print(f"Pass: {report['pass_count']}  Fail: {report['fail_count']}  Quarantine: {report['quarantine_count']}")
        print(f"Parity statuses: {report['parity_status_counts']}")
        if report["failure_reasons"]:
            print(f"Failure reasons: {report['failure_reasons']}")
        if args.fix:
            print(f"Fixed: {report['fixed_count']} games -> {report['fix_status']}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
