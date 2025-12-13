#!/usr/bin/env python3
"""Build a canonical training-pool GameReplayDB from many sources.

Policy:
  - Enforce *per-game* canonical-history + TS↔Python replay parity gates.
  - Exclude holdout sources (tournament/eval) from the training pool.
  - Optionally copy holdout games to a separate holdout DB.
  - Optionally quarantine failing games to a separate DB.

This is intended to be the single ingestion point for training data: any
generator (selfplay, CMA-ES, soaks, hybrids) can write to a staging DB, and
only games that pass strict gates are merged into the training pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_SERVICE_ROOT.parent

if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.db.game_replay import GameReplayDB
from app.rules.history_validation import (
    validate_canonical_config_for_game,
    validate_canonical_history_for_game,
)


DEFAULT_HOLDOUT_SOURCE_SUBSTRINGS = (
    "tournament",
    "elo_tournament",
    "evaluation",
    "eval",
)


@dataclass(frozen=True)
class GateFailure:
    kind: str  # structural | semantic | config | history | holdout | filtered_out
    reason: str
    details: Dict[str, Any]


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    if str(AI_SERVICE_ROOT) not in parts:
        parts.insert(0, str(AI_SERVICE_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _connect_sqlite(db_path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        return sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=60.0,
        )
    return sqlite3.connect(str(db_path), timeout=60.0)


def _normalize_source(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _is_holdout_source(source: Optional[str], substrings: Sequence[str]) -> bool:
    norm = _normalize_source(source)
    if not norm:
        return False
    return any(token in norm for token in substrings)


def _list_games(
    db_path: Path,
    *,
    board_type: Optional[str],
    num_players: Optional[int],
    require_completed: bool,
) -> Dict[str, Dict[str, Any]]:
    conn = _connect_sqlite(db_path, readonly=True)
    try:
        query = (
            "SELECT game_id, board_type, num_players, game_status, source "
            "FROM games WHERE 1=1"
        )
        params: List[Any] = []
        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players is not None:
            query += " AND num_players = ?"
            params.append(int(num_players))
        if require_completed:
            query += " AND game_status = 'completed'"
        rows = conn.execute(query, params).fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for game_id, bt, np, status, source in rows:
            result[str(game_id)] = {
                "game_id": str(game_id),
                "board_type": bt,
                "num_players": int(np),
                "game_status": status,
                "source": source,
            }
        return result
    finally:
        conn.close()


def _looks_like_game_replay_db(db_path: Path) -> bool:
    """Heuristic filter so --scan-dir doesn't ingest non-GameReplayDB sqlite files."""
    try:
        conn = _connect_sqlite(db_path, readonly=True)
    except Exception:
        return False

    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {str(r[0]) for r in rows}
        required = {"games", "game_moves", "game_initial_state"}
        return required.issubset(tables)
    except Exception:
        return False
    finally:
        conn.close()


def _run_parity_summary(db_path: Path, *, work_dir: Path) -> Dict[str, Any]:
    summary_path = work_dir / f"parity_summary.{db_path.stem}.{uuid.uuid4().hex}.json"
    cmd = [
        sys.executable,
        str((AI_SERVICE_ROOT / "scripts" / "check_ts_python_replay_parity.py").resolve()),
        "--db",
        str(db_path),
        "--compact",
        "--mode",
        "legacy",
        "--view",
        "post_move",
        "--summary-json",
        str(summary_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(AI_SERVICE_ROOT),
        env=_build_env(),
        text=True,
        capture_output=True,
    )
    # The parity script is allowed to exit non-zero; we still consume the JSON summary.
    if not summary_path.exists():
        raise RuntimeError(
            "Parity harness did not produce a summary JSON.\n"
            f"cmd={' '.join(cmd)}\n"
            f"rc={proc.returncode}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    try:
        return json.loads(summary_path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse parity summary JSON: {summary_path}: {e}") from e


def _extract_gate_failures(parity_summary: Dict[str, Any]) -> Dict[str, GateFailure]:
    failures: Dict[str, GateFailure] = {}

    for entry in parity_summary.get("structural_issues", []) or []:
        game_id = str(entry.get("game_id", ""))
        if not game_id:
            continue
        failures[game_id] = GateFailure(
            kind="structural",
            reason=str(entry.get("structure", "structural_issue")),
            details={
                "structure": entry.get("structure"),
                "structure_reason": entry.get("structure_reason"),
                "db_path": entry.get("db_path"),
            },
        )

    for entry in parity_summary.get("semantic_divergences", []) or []:
        game_id = str(entry.get("game_id", ""))
        if not game_id:
            continue
        failures[game_id] = GateFailure(
            kind="semantic",
            reason="semantic_divergence",
            details={
                "diverged_at": entry.get("diverged_at"),
                "mismatch_kinds": entry.get("mismatch_kinds"),
                "mismatch_context": entry.get("mismatch_context"),
                "db_path": entry.get("db_path"),
            },
        )

    # NOTE: end_of_game_only divergences are acceptable for training and are
    # intentionally not treated as failures.
    return failures


def _extract_noncanonical_config_failures(
    db_path: Path,
    game_ids: Iterable[str],
) -> Dict[str, GateFailure]:
    failures: Dict[str, GateFailure] = {}
    try:
        db = GameReplayDB(str(db_path))
    except Exception as exc:
        # If we cannot open the DB, treat everything as failed (the parity
        # harness would likely have surfaced this too, but be explicit).
        msg = f"{type(exc).__name__}: {exc}"
        for gid in game_ids:
            failures[str(gid)] = GateFailure(
                kind="config",
                reason="config_validation_error",
                details={"error": msg, "db_path": str(db_path)},
            )
        return failures

    for gid in game_ids:
        game_id = str(gid)
        try:
            report = validate_canonical_config_for_game(db, game_id)
        except Exception as exc:
            failures[game_id] = GateFailure(
                kind="config",
                reason="config_validation_error",
                details={"error": f"{type(exc).__name__}: {exc}", "db_path": str(db_path)},
            )
            continue
        if not getattr(report, "is_canonical", False):
            failures[game_id] = GateFailure(
                kind="config",
                reason="non_canonical_config",
                details={"issues": [asdict(issue) for issue in (report.issues or [])], "db_path": str(db_path)},
            )

    return failures


def _extract_noncanonical_history_failures(
    db_path: Path,
    game_ids: Iterable[str],
) -> Dict[str, GateFailure]:
    failures: Dict[str, GateFailure] = {}
    try:
        db = GameReplayDB(str(db_path))
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        for gid in game_ids:
            failures[str(gid)] = GateFailure(
                kind="history",
                reason="history_validation_error",
                details={"error": msg, "db_path": str(db_path)},
            )
        return failures

    for gid in game_ids:
        game_id = str(gid)
        try:
            report = validate_canonical_history_for_game(db, game_id)
        except Exception as exc:
            failures[game_id] = GateFailure(
                kind="history",
                reason="history_validation_error",
                details={"error": f"{type(exc).__name__}: {exc}", "db_path": str(db_path)},
            )
            continue

        if not getattr(report, "is_canonical", False):
            failures[game_id] = GateFailure(
                kind="history",
                reason="non_canonical_history",
                details={
                    "issues": [asdict(issue) for issue in (report.issues or [])],
                    "db_path": str(db_path),
                },
            )

    return failures


def _ensure_db_initialized(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and db_path.stat().st_size > 0:
        return
    GameReplayDB(str(db_path))


def _table_exists(conn: sqlite3.Connection, db_name: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {db_name}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, db_name: str, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA {db_name}.table_info({table})").fetchall()
    # rows: (cid, name, type, notnull, dflt_value, pk)
    return [str(r[1]) for r in rows]


def _copy_game_rows(
    *,
    dest_conn: sqlite3.Connection,
    src_alias: str,
    game_ids: Sequence[str],
) -> Dict[str, int]:
    """Copy rows for the given game_ids from src_alias into main DB.

    Returns per-table inserted row counts (best-effort).
    """
    tables: List[Tuple[str, str]] = [
        ("games", "game_id"),
        ("game_players", "game_id"),
        ("game_initial_state", "game_id"),
        ("game_moves", "game_id"),
        ("game_state_snapshots", "game_id"),
        ("game_choices", "game_id"),
        ("game_history_entries", "game_id"),
    ]

    inserted: Dict[str, int] = {}
    dest_conn.execute("PRAGMA foreign_keys=OFF")

    for table, key_col in tables:
        if not _table_exists(dest_conn, "main", table) or not _table_exists(dest_conn, src_alias, table):
            continue

        dest_cols = _table_columns(dest_conn, "main", table)
        src_cols = _table_columns(dest_conn, src_alias, table)
        common_cols = [c for c in dest_cols if c in set(src_cols)]
        if not common_cols:
            continue

        cols_sql = ", ".join(common_cols)

        total = 0
        batch_size = 250
        for i in range(0, len(game_ids), batch_size):
            batch = list(game_ids[i : i + batch_size])
            placeholders = ", ".join(["?"] * len(batch))
            sql = (
                f"INSERT OR IGNORE INTO {table} ({cols_sql}) "
                f"SELECT {cols_sql} FROM {src_alias}.{table} "
                f"WHERE {key_col} IN ({placeholders})"
            )
            cur = dest_conn.execute(sql, batch)
            # sqlite3 cursor rowcount is best-effort; OK for reporting.
            try:
                total += int(cur.rowcount or 0)
            except Exception:
                pass

        inserted[table] = total

    return inserted


def _attach_copy_detach(
    *,
    dest_db: Path,
    src_db: Path,
    game_ids: Sequence[str],
) -> Dict[str, int]:
    if not game_ids:
        return {}

    conn = _connect_sqlite(dest_db, readonly=False)
    try:
        alias = f"src_{uuid.uuid4().hex}"
        conn.execute("ATTACH DATABASE ? AS " + alias, (str(src_db),))
        try:
            stats = _copy_game_rows(dest_conn=conn, src_alias=alias, game_ids=game_ids)
            conn.commit()
            return stats
        finally:
            conn.execute("DETACH DATABASE " + alias)
    finally:
        conn.close()


def build_pool(
    *,
    input_dbs: Sequence[Path],
    output_db: Path,
    holdout_db: Optional[Path],
    quarantine_db: Optional[Path],
    board_type: Optional[str],
    num_players: Optional[int],
    require_completed: bool,
    holdout_source_substrings: Sequence[str],
    report_json: Optional[Path],
) -> int:
    _ensure_db_initialized(output_db)
    if holdout_db is not None:
        _ensure_db_initialized(holdout_db)
    if quarantine_db is not None:
        _ensure_db_initialized(quarantine_db)

    if holdout_db is not None and holdout_db.resolve() == output_db.resolve():
        raise SystemExit("--holdout-db must be different from --output-db")
    if quarantine_db is not None and quarantine_db.resolve() == output_db.resolve():
        raise SystemExit("--quarantine-db must be different from --output-db")

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "board_type": board_type,
        "num_players": num_players,
        "require_completed": bool(require_completed),
        "holdout_source_substrings": list(holdout_source_substrings),
        "inputs": [],
        "totals": {
            "games_seen": 0,
            "training_candidates": 0,
            "holdout_candidates": 0,
            "training_passed": 0,
            "training_failed": 0,
            "holdout_passed": 0,
            "holdout_failed": 0,
        },
    }

    with tempfile.TemporaryDirectory(prefix="ringrift_pool_gate_") as td:
        work_dir = Path(td)

        for db_path in input_dbs:
            games = _list_games(
                db_path,
                board_type=board_type,
                num_players=num_players,
                require_completed=require_completed,
            )
            if not games:
                continue

            parity_summary = _run_parity_summary(db_path, work_dir=work_dir)
            failures = _extract_gate_failures(parity_summary)
            config_failures = _extract_noncanonical_config_failures(db_path, games.keys())
            for game_id, failure in config_failures.items():
                failures.setdefault(game_id, failure)
            history_failures = _extract_noncanonical_history_failures(db_path, games.keys())
            for game_id, failure in history_failures.items():
                failures.setdefault(game_id, failure)

            training_ids: List[str] = []
            holdout_ids: List[str] = []
            failed_training_ids: List[str] = []
            failed_holdout_ids: List[str] = []

            for game_id, meta in games.items():
                source = meta.get("source")
                is_holdout = _is_holdout_source(source, holdout_source_substrings)

                failure = failures.get(game_id)
                if failure is not None:
                    if is_holdout:
                        failed_holdout_ids.append(game_id)
                    else:
                        failed_training_ids.append(game_id)
                    continue

                if is_holdout:
                    holdout_ids.append(game_id)
                else:
                    training_ids.append(game_id)

            report["inputs"].append(
                {
                    "db_path": str(db_path),
                    "games_considered": len(games),
                    "training_candidates": len(training_ids) + len(failed_training_ids),
                    "holdout_candidates": len(holdout_ids) + len(failed_holdout_ids),
                    "training_passed": len(training_ids),
                    "training_failed": len(failed_training_ids),
                    "holdout_passed": len(holdout_ids),
                    "holdout_failed": len(failed_holdout_ids),
                    "parity_summary": {
                        "games_with_semantic_divergence": parity_summary.get("games_with_semantic_divergence"),
                        "games_with_structural_issues": parity_summary.get("games_with_structural_issues"),
                        "games_with_non_canonical_history": parity_summary.get("games_with_non_canonical_history"),
                        "games_with_non_canonical_config": len(config_failures),
                        "games_with_non_canonical_history_gate": len(history_failures),
                        "passed_canonical_parity_gate": parity_summary.get("passed_canonical_parity_gate"),
                        "total_games_checked": parity_summary.get("total_games_checked"),
                    },
                }
            )

            totals = report["totals"]
            totals["games_seen"] += len(games)
            totals["training_candidates"] += len(training_ids) + len(failed_training_ids)
            totals["holdout_candidates"] += len(holdout_ids) + len(failed_holdout_ids)
            totals["training_passed"] += len(training_ids)
            totals["training_failed"] += len(failed_training_ids)
            totals["holdout_passed"] += len(holdout_ids)
            totals["holdout_failed"] += len(failed_holdout_ids)

            _attach_copy_detach(dest_db=output_db, src_db=db_path, game_ids=training_ids)
            if holdout_db is not None:
                _attach_copy_detach(dest_db=holdout_db, src_db=db_path, game_ids=holdout_ids)
            if quarantine_db is not None:
                _attach_copy_detach(
                    dest_db=quarantine_db,
                    src_db=db_path,
                    game_ids=sorted(set(failed_training_ids + failed_holdout_ids)),
                )

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-db",
        action="append",
        default=[],
        help="Path to an input GameReplayDB (repeatable).",
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=[],
        help=(
            "Recursively scan a directory for *.db / games.db files to ingest "
            "(repeatable)."
        ),
    )
    parser.add_argument(
        "--output-db",
        required=True,
        help="Path to the output training-pool GameReplayDB.",
    )
    parser.add_argument(
        "--holdout-db",
        default=None,
        help="Optional path to write holdout (tournament/eval) games into a separate DB.",
    )
    parser.add_argument(
        "--quarantine-db",
        default=None,
        help="Optional path to copy failing games into a separate DB for debugging.",
    )
    parser.add_argument(
        "--board-type",
        default=None,
        choices=["square8", "square19", "hexagonal"],
        help="Optional board_type filter applied before merging.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        choices=[2, 3, 4],
        help="Optional num_players filter applied before merging.",
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Only ingest games with game_status='completed'.",
    )
    parser.add_argument(
        "--holdout-source-substring",
        action="append",
        default=[],
        help=(
            "Treat any game whose games.source contains this substring (case-insensitive) as holdout. "
            "Repeatable. Default includes: "
            + ", ".join(DEFAULT_HOLDOUT_SOURCE_SUBSTRINGS)
        ),
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a JSON ingestion report (counts + parity summaries).",
    )

    args = parser.parse_args(argv)

    input_dbs: List[Path] = []
    for raw in args.input_db or []:
        input_dbs.append(_resolve_path(raw))

    for raw in args.scan_dir or []:
        base = _resolve_path(raw)
        if not base.exists() or not base.is_dir():
            raise SystemExit(f"--scan-dir is not a directory: {base}")
        for candidate in base.rglob("*.db"):
            candidate = candidate.resolve()
            if _looks_like_game_replay_db(candidate):
                input_dbs.append(candidate)

    if not input_dbs:
        raise SystemExit("No inputs provided. Use --input-db and/or --scan-dir.")

    output_db = _resolve_path(args.output_db)
    holdout_db = _resolve_path(args.holdout_db) if args.holdout_db else None
    quarantine_db = _resolve_path(args.quarantine_db) if args.quarantine_db else None
    report_json = _resolve_path(args.report_json) if args.report_json else None

    # De-dupe and avoid ingesting the output/holdout/quarantine DBs as inputs.
    excluded: Set[Path] = {output_db.resolve()}
    if holdout_db is not None:
        excluded.add(holdout_db.resolve())
    if quarantine_db is not None:
        excluded.add(quarantine_db.resolve())

    uniq_inputs: List[Path] = []
    seen_paths: Set[Path] = set()
    for path in input_dbs:
        resolved = path.resolve()
        if resolved in excluded or resolved in seen_paths:
            continue
        if not resolved.exists():
            raise SystemExit(f"Input DB not found: {resolved}")
        seen_paths.add(resolved)
        uniq_inputs.append(resolved)

    input_dbs = uniq_inputs

    substrings = list(DEFAULT_HOLDOUT_SOURCE_SUBSTRINGS)
    for token in args.holdout_source_substring or []:
        token_norm = (token or "").strip().lower()
        if token_norm:
            substrings.append(token_norm)

    # De-duplicate while preserving order.
    seen: Set[str] = set()
    holdout_source_substrings: List[str] = []
    for s in substrings:
        if s in seen:
            continue
        seen.add(s)
        holdout_source_substrings.append(s)

    return build_pool(
        input_dbs=input_dbs,
        output_db=output_db,
        holdout_db=holdout_db,
        quarantine_db=quarantine_db,
        board_type=args.board_type,
        num_players=args.num_players,
        require_completed=bool(args.require_completed),
        holdout_source_substrings=holdout_source_substrings,
        report_json=report_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
