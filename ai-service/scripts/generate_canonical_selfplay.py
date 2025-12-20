#!/usr/bin/env python
from __future__ import annotations

"""
End-to-end canonical self-play generator and gate.

This script ties together three pieces for a single board type:

  1. Run a small canonical self-play soak to populate a GameReplayDB.
  2. Run the TS↔Python replay parity harness on that DB.
  3. Run the lightweight canonical history validator over every game.

A database is considered "canonical" for training only if:
  - The parity gate passes (no structural issues, no semantic divergence),
  - The canonical history validator reports zero issues for all games,
  - At least one game was recorded, and
  - The FE/territory fixture tests pass for this board type.

Current implementation:
- Uses scripts/run_canonical_selfplay_parity_gate.py for step (1) + (2).
- Uses a strict trace replay validation (GameEngine.apply_move(trace_mode=True))
    over GameReplayDB moves for step (3).
- Validates the initial-state rules parameters match canonical defaults
    (rings per player, victory thresholds, LPS rounds required) so that
    non-canonical per-game overrides are not accidentally promoted.

Typical usage (from ai-service/):

  PYTHONPATH=. python scripts/generate_canonical_selfplay.py \\
    --board-type square19 \\
    --num-games 50 \\
    --db data/games/canonical_square19.db \\
    --summary db_health.canonical_square19.json

The summary JSON includes:
  - board_type, db_path
  - db_stats (games/moves totals, swap_sides counts)
  - parity_gate (raw summary from run_canonical_selfplay_parity_gate)
  - canonical_history (games_checked, non_canonical_games, sample_issues)
  - fe_territory_fixtures_ok (boolean)
  - canonical_ok (boolean)
"""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
# Ensure `app.*` imports resolve when invoked from repo root.
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.selfplay_config import SelfplayConfig, create_argument_parser
from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine
from app.rules.history_validation import validate_canonical_config_for_game


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    # Ensure PYTHONPATH includes ai-service root when invoked from repo root.
    # Prepend absolute AI_SERVICE_ROOT if missing so relative PYTHONPATH values
    # (e.g. "ai-service") don't break when cwd is ai-service/.
    pythonpath = env.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    if str(AI_SERVICE_ROOT) not in parts:
        parts.insert(0, str(AI_SERVICE_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(parts)

    # Keep OpenMP usage conservative by default.
    env.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
    # Ensure progress output from long-running child scripts is not buffered.
    env.setdefault("PYTHONUNBUFFERED", os.environ.get("PYTHONUNBUFFERED", "1"))
    return env


def _run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    *,
    capture_output: bool = True,
    stream_to_stderr: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess:
    env = _build_env()
    stdout = None
    stderr = None
    if stream_to_stderr:
        # Keep stdout clean for the final JSON gate summary while still
        # providing progress output during long runs.
        stdout = sys.stderr
        stderr = sys.stderr
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or AI_SERVICE_ROOT),
            env=env,
            text=True,
            capture_output=capture_output and not stream_to_stderr,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_seconds,
        )
        return proc
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )


def _run_cmd_tee(
    cmd: list[str],
    cwd: Path | None = None,
    *,
    max_output_lines: int = 5000,
) -> subprocess.CompletedProcess:
    """Run a command while streaming output to stderr and capturing it."""
    env = _build_env()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or AI_SERVICE_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", file=sys.stderr, flush=True)
        lines.append(line)
        if len(lines) > max_output_lines:
            lines.pop(0)

    rc = proc.wait()
    stdout = "".join(lines)
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")


def _count_games_in_db_ro(db_path: Path) -> int | None:
    """Best-effort count of games without triggering schema migrations.

    Parity-only mode should never silently run large migrations (or block on
    write locks) just to print a helpful progress message. Use a direct
    read-only SQLite connection instead of GameReplayDB.
    """
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=1.0,
        )
    except Exception:
        return None

    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()
        if row is None:
            return 0
        # sqlite3 row may be a tuple; prefer index access.
        return int(row[0])
    except Exception:
        return None
    finally:
        conn.close()


def collect_db_stats(db_path: Path) -> dict[str, Any]:
    """Collect lightweight aggregate stats from a GameReplayDB.

    This intentionally uses direct SQLite queries (not GameReplayDB replay) so
    it stays fast even for large databases.
    """

    def _row_int(row: object | None, default: int = 0) -> int:
        if row is None:
            return default
        try:
            return int(row[0])  # type: ignore[index]
        except Exception:
            return default

    if not db_path.exists():
        return {"error": "db_missing", "db_path": str(db_path)}

    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
    except Exception as exc:
        return {"error": "failed_to_open_db", "db_path": str(db_path), "message": str(exc)}

    try:
        cur = conn.cursor()
        games_total = _row_int(cur.execute("SELECT COUNT(*) FROM games").fetchone())
        games_completed = _row_int(
            cur.execute("SELECT COUNT(*) FROM games WHERE game_status = 'completed'").fetchone()
        )
        moves_total = _row_int(cur.execute("SELECT COUNT(*) FROM game_moves").fetchone())

        swap_moves = _row_int(
            cur.execute(
                "SELECT COUNT(*) FROM game_moves WHERE move_type = ?",
                ("swap_sides",),
            ).fetchone()
        )
        swap_games = _row_int(
            cur.execute(
                "SELECT COUNT(DISTINCT game_id) FROM game_moves WHERE move_type = ?",
                ("swap_sides",),
            ).fetchone()
        )

        return {
            "games_total": games_total,
            "games_completed": games_completed,
            "moves_total": moves_total,
            "swap_sides": {"moves": swap_moves, "games": swap_games},
        }
    except Exception as exc:
        return {"error": "failed_to_query_db", "db_path": str(db_path), "message": str(exc)}
    finally:
        conn.close()


def run_selfplay_and_parity(
    board_type: str,
    num_games: int,
    db_path: Path,
    num_players: int,
    hosts: str | None = None,
    difficulty_band: str = "light",
    parity_limit_games_per_db: int = 0,
    parity_timeout_seconds: int | None = None,
    include_training_data_jsonl: bool = False,
    distributed_job_timeout_seconds: int = 0,
    distributed_fetch_timeout_seconds: int = 0,
) -> dict[str, Any]:
    """
    Delegate to run_canonical_selfplay_parity_gate.py to:
      - run a small canonical self-play soak, and
      - run TS↔Python parity on the resulting DB.
    """
    summary_path = db_path.with_suffix(db_path.suffix + ".parity_gate.json")

    # If num_games == 0, assume the DB already exists and skip running
    # a new soak. We still run parity on the provided DB path.
    if num_games <= 0:
        if not db_path.exists():
            return {
                "error": "db_missing_for_parity_only_mode",
                "db_path": str(db_path),
                "num_games": num_games,
                "returncode": 1,
            }

        print(
            f"[generate_canonical_selfplay] Skipping soak (num_games={num_games}); running parity on existing DB...",
            file=sys.stderr,
            flush=True,
        )
        num_existing_games = _count_games_in_db_ro(db_path)
        if num_existing_games is not None:
            print(
                f"[generate_canonical_selfplay] Existing DB contains {num_existing_games} game(s): {db_path}",
                file=sys.stderr,
                flush=True,
            )
            if num_existing_games <= 0:
                return {
                    "error": "db_has_no_games_for_parity_only_mode",
                    "db_path": str(db_path),
                    "num_games": num_games,
                    "returncode": 1,
                }
            if (
                num_existing_games > 5000
                and (not parity_limit_games_per_db or parity_limit_games_per_db <= 0)
            ):
                print(
                    "[generate_canonical_selfplay] WARNING: parity-only mode will check ALL games "
                    "in this DB; consider setting --parity-limit-games-per-db for a quick smoke pass.",
                    file=sys.stderr,
                    flush=True,
                )
        cmd = [
            sys.executable,
            "scripts/check_ts_python_replay_parity.py",
            "--db",
            str(db_path),
            "--mode",
            "canonical",
            "--view",
            "post_move",
            "--progress-every",
            "25",
            "--summary-json",
            str(summary_path),
        ]
        if parity_limit_games_per_db and parity_limit_games_per_db > 0:
            cmd += ["--limit-games-per-db", str(parity_limit_games_per_db)]
        proc = _run_cmd(
            cmd,
            cwd=AI_SERVICE_ROOT,
            capture_output=False,
            stream_to_stderr=True,
            timeout_seconds=parity_timeout_seconds,
        )
    else:
        print(
            f"[generate_canonical_selfplay] Running canonical soak ({num_games} games) + parity gate...",
            file=sys.stderr,
            flush=True,
        )
        cmd = [
            sys.executable,
            "scripts/run_canonical_selfplay_parity_gate.py",
            "--board-type",
            board_type,
            "--num-games",
            str(num_games),
            "--num-players",
            str(num_players),
            "--db",
            str(db_path),
            "--summary",
            str(summary_path),
            "--difficulty-band",
            difficulty_band,
        ]
        if include_training_data_jsonl:
            cmd.append("--include-training-data-jsonl")
        if hosts:
            cmd += ["--hosts", hosts]
            if distributed_job_timeout_seconds and distributed_job_timeout_seconds > 0:
                cmd += [
                    "--distributed-job-timeout-seconds",
                    str(int(distributed_job_timeout_seconds)),
                ]
            if distributed_fetch_timeout_seconds and distributed_fetch_timeout_seconds > 0:
                cmd += [
                    "--distributed-fetch-timeout-seconds",
                    str(int(distributed_fetch_timeout_seconds)),
                ]

        proc = _run_cmd(
            cmd,
            cwd=AI_SERVICE_ROOT,
            capture_output=False,
            stream_to_stderr=True,
            timeout_seconds=parity_timeout_seconds,
        )

    parity_summary: dict[str, Any]
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                parity_summary = json.load(f)
        except Exception:
            parity_summary = {
                "error": "failed_to_load_parity_summary_file",
                "summary_path": str(summary_path),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    else:
        # Fallback to parsing stdout if the summary file was not written.
        # If stdout wasn't captured, surface an actionable error instead.
        if not proc.stdout:
            parity_summary = {
                "error": "parity_summary_file_missing",
                "summary_path": str(summary_path),
            }
        else:
            try:
                parity_summary = json.loads(proc.stdout)
            except Exception:
                parity_summary = {
                    "error": "failed_to_parse_parity_summary_stdout",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

    parity_summary["returncode"] = proc.returncode
    return parity_summary


def merge_distributed_dbs(
    source_dbs: list[Path],
    dest_db: Path,
    reset_db: bool,
) -> dict[str, Any]:
    """Merge per-host distributed self-play DBs into *dest_db*.

    When reset_db is True and dest_db exists, we archive the existing DB
    alongside it before merging, rather than deleting it silently.
    """
    if not source_dbs:
        return {"error": "no_source_dbs", "returncode": 1}

    archived_path: str | None = None
    if reset_db and dest_db.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archived = dest_db.with_suffix(dest_db.suffix + f".archived_{ts}")
        dest_db.replace(archived)
        archived_path = str(archived)

    cmd = [
        sys.executable,
        "scripts/merge_game_dbs.py",
        "--output",
        str(dest_db),
        "--on-conflict",
        "skip",
        "--dedupe-by-game-id",
    ]
    for src in source_dbs:
        cmd += ["--db", str(src)]

    proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "sources": [str(p) for p in source_dbs],
        "dest_db": str(dest_db),
        "archived_previous_db": archived_path,
    }


def run_canonical_history_check(db_path: Path) -> dict[str, Any]:
    """Run strict trace-mode replay validation over all games in the DB.

    This is a stronger gate than checking stored (phase, move_type) pairs,
    because it catches silent phase transitions and missing bookkeeping moves
    by replaying the recorded move list through GameEngine.apply_move with the
    strict per-phase move invariant enabled.
    """
    return _run_canonical_history_check(
        db_path=db_path,
        max_games=None,
        stop_after_first_failure=False,
    )


def _run_canonical_history_check(
    *,
    db_path: Path,
    max_games: int | None,
    stop_after_first_failure: bool,
) -> dict[str, Any]:
    """Inner implementation of canonical history check with optional limits."""
    db = GameReplayDB(str(db_path))

    with db._get_conn() as conn:  # type: ignore[attr-defined]
        rows = conn.execute("SELECT game_id FROM games").fetchall()
        game_ids = [row["game_id"] for row in rows]

    if max_games is not None and max_games >= 0:
        game_ids = game_ids[:max_games]

    issues_by_game: dict[str, list[dict[str, Any]]] = {}

    if game_ids:
        print(
            f"[generate_canonical_selfplay] Canonical history (trace replay) check: {len(game_ids)} game(s)",
            file=sys.stderr,
            flush=True,
        )

    for idx, gid in enumerate(game_ids):
        if idx and idx % 25 == 0:
            print(
                f"[generate_canonical_selfplay] Canonical history progress: {idx}/{len(game_ids)}",
                file=sys.stderr,
                flush=True,
            )
        state = db.get_initial_state(gid)
        if state is None:
            issues_by_game[gid] = [
                {
                    "move_number": -1,
                    "phase_before": None,
                    "move_type": None,
                    "player": None,
                    "reason": "missing_initial_state",
                }
            ]
            continue

        moves = db.get_moves(gid)
        for move_idx, mv in enumerate(moves):
            try:
                state = GameEngine.apply_move(state, mv, trace_mode=True)
            except Exception as exc:
                issues_by_game[gid] = [
                    {
                        "move_number": move_idx,
                        "phase_before": getattr(getattr(state, "current_phase", None), "value", None),
                        "move_type": getattr(getattr(mv, "type", None), "value", str(getattr(mv, "type", ""))),
                        "player": getattr(mv, "player", None),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ]
                break
        if stop_after_first_failure and gid in issues_by_game:
            break

    games_checked = len(game_ids)
    non_canonical_games = len(issues_by_game)

    # For brevity, surface at most a few sample issues.
    sample_issues: dict[str, Any] = {}
    for gid, issues in list(issues_by_game.items())[:5]:
        sample_issues[gid] = issues[:5]

    return {
        "games_checked": games_checked,
        "non_canonical_games": non_canonical_games,
        "sample_issues": sample_issues,
    }


def run_canonical_config_check(db_path: Path) -> dict[str, Any]:
    """Validate initial-state canonical configuration for every game."""
    return _run_canonical_config_check(
        db_path=db_path,
        max_games=None,
        stop_after_first_failure=False,
    )


def _run_canonical_config_check(
    *,
    db_path: Path,
    max_games: int | None,
    stop_after_first_failure: bool,
) -> dict[str, Any]:
    """Inner implementation of canonical config check with optional limits."""
    db = GameReplayDB(str(db_path))

    with db._get_conn() as conn:  # type: ignore[attr-defined]
        rows = conn.execute("SELECT game_id FROM games").fetchall()
        game_ids = [row["game_id"] for row in rows]

    if max_games is not None and max_games >= 0:
        game_ids = game_ids[:max_games]

    issues_by_game: dict[str, list[dict[str, Any]]] = {}

    if game_ids:
        print(
            f"[generate_canonical_selfplay] Canonical config check: {len(game_ids)} game(s)",
            file=sys.stderr,
            flush=True,
        )

    for idx, gid in enumerate(game_ids):
        if idx and idx % 100 == 0:
            print(
                f"[generate_canonical_selfplay] Canonical config progress: {idx}/{len(game_ids)}",
                file=sys.stderr,
                flush=True,
            )

        report = validate_canonical_config_for_game(db, gid)
        if report.is_canonical:
            continue

        issues_by_game[gid] = [
            {
                "reason": issue.reason,
                "observed": issue.observed,
                "expected": issue.expected,
            }
            for issue in report.issues[:25]
        ]
        if stop_after_first_failure:
            break

    games_checked = len(game_ids)
    non_canonical_games = len(issues_by_game)

    sample_issues: dict[str, Any] = {}
    for gid, issues in list(issues_by_game.items())[:5]:
        sample_issues[gid] = issues[:5]

    return {
        "games_checked": games_checked,
        "non_canonical_games": non_canonical_games,
        "sample_issues": sample_issues,
    }


def archive_db_in_place(db_path: Path) -> Path:
    """Archive *db_path* in-place and return the archived path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = db_path.with_suffix(db_path.suffix + f".archived_{ts}")
    db_path.replace(archived)
    return archived


def run_fe_territory_fixtures(board_type: str) -> bool:
    """
    Run a small pytest subset that validates FE/territory fixtures for the
    given board_type. Returns True if tests pass, False otherwise.

    For board types without dedicated FE/territory fixtures, this is a no-op
    gate that returns True.
    """
    if board_type == "square8":
        test_args = ["tests/test_territory_fe_edge_fixture.py", "-q"]
    elif board_type in {"hex", "hexagonal"}:
        test_args = ["tests/test_hex_territory_fe_fixtures.py", "-q"]
    else:
        # No additional FE/territory fixtures defined yet for this board type.
        msg = (
            "[generate_canonical_selfplay] No FE/territory fixture tests "
            "defined for "
            f"board_type={board_type!r}; "
            "treating as fe_territory_fixtures_ok=True."
        )
        print(msg, file=sys.stderr, flush=True)
        return True

    cmd = [sys.executable, "-m", "pytest", *test_args]
    print(
        f"[generate_canonical_selfplay] Running FE/territory fixtures: {board_type}",
        file=sys.stderr,
        flush=True,
    )
    proc = _run_cmd_tee(cmd, cwd=AI_SERVICE_ROOT)

    if proc.returncode != 0:
        joined_cmd = " ".join(cmd)
        print(
            "[generate_canonical_selfplay] FE/territory fixture tests FAILED "
            f"for board_type={board_type!r} using command: {joined_cmd}",
            file=sys.stderr,
            flush=True,
        )

        output = (proc.stdout or "") + (proc.stderr or "")
        lines = output.strip().splitlines()
        if lines:
            preview = "\n".join(lines[:20])
            print(
                "[generate_canonical_selfplay] Pytest output (truncated):\n"
                f"{preview}",
                file=sys.stderr,
                flush=True,
            )

        return False

    return True


def run_anm_invariants(board_type: str) -> dict[str, Any]:
    """
    Run the ANM parity + invariant tests for the given board_type.

    This wires the existing ANM-focused pytest suites into the canonical
    self-play gate:

      - tests/parity/test_anm_global_actions_parity.py
      - tests/invariants/test_anm_and_termination_invariants.py

    The implementation is intentionally small and conservative: it reports a
    boolean pass/fail flag and, when possible, a best-effort estimate of the
    total test count and failures by parsing pytest's summary line.
    """
    test_args = [
        "tests/parity/test_anm_global_actions_parity.py",
        "tests/invariants/test_anm_and_termination_invariants.py",
        "-q",
    ]
    cmd = [sys.executable, "-m", "pytest", *test_args]
    print(
        f"[generate_canonical_selfplay] Running ANM invariants: {board_type}",
        file=sys.stderr,
        flush=True,
    )
    proc = _run_cmd_tee(cmd, cwd=AI_SERVICE_ROOT)

    output = (proc.stdout or "") + (proc.stderr or "")
    num_tests: int | None = None
    num_failed: int | None = None

    # Best-effort parse of pytest's final summary line, e.g.:
    #   "6 passed in 0.03s"
    #   "5 passed, 1 failed in 0.10s"
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        if "passed" in text and " in " in text:
            tokens = text.replace(",", " ").split()
            passed_count: int | None = None
            failed_count: int | None = None
            for idx, token in enumerate(tokens):
                if token == "passed":
                    try:
                        passed_count = int(tokens[idx - 1])
                    except (ValueError, IndexError):
                        passed_count = None
                elif token == "failed":
                    try:
                        failed_count = int(tokens[idx - 1])
                    except (ValueError, IndexError):
                        failed_count = None
            if passed_count is not None:
                if failed_count is None:
                    failed_count = 0
                num_tests = passed_count + failed_count
                num_failed = failed_count
            break

    passed_flag = proc.returncode == 0

    result: dict[str, Any] = {
        "board_type": board_type,
        "passed": bool(passed_flag),
        "returncode": int(proc.returncode),
    }
    if num_tests is not None:
        result["num_tests"] = int(num_tests)
    if num_failed is not None:
        result["num_failed"] = int(num_failed)

    # On failure, include a small, truncated preview of pytest output to make
    # debugging easier without bloating the canonical gate summary.
    if not passed_flag and output:
        lines = output.strip().splitlines()
        if lines:
            preview = "\n".join(lines[:20])
            result["output_preview"] = preview

    return result


def main(argv: list[str] | None = None) -> int:
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description=(
            "Generate a canonical self-play GameReplayDB for a board type and "
            "gate it on TS↔Python parity plus canonical history constraints."
        ),
        include_gpu=False,
        include_ramdrive=False,
    )

    # Add canonical selfplay-specific arguments
    # Note: --board is provided by base parser, but we need it to be required
    # We'll handle this in validation below
    parser.add_argument(
        "--parity-limit-games-per-db",
        type=int,
        default=0,
        help="Optional cap on number of games to check in parity-only mode. Default: 0 (check all).",
    )
    parser.add_argument(
        "--parity-timeout-seconds",
        type=int,
        default=0,
        help="Optional timeout (seconds) for parity/soak subprocess. Default: 0 (no timeout).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to the GameReplayDB SQLite file. Defaults to data/games/canonical_<board>.db.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to write the combined canonical summary JSON.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=None,
        help="Optional directory to write analysis artifacts.",
    )
    parser.add_argument(
        "--skip-analyses",
        action="store_true",
        help="Skip running post-soak analysis scripts on the JSONL logs.",
    )
    parser.add_argument(
        "--distributed-job-timeout-seconds",
        type=int,
        default=0,
        help="Per-job timeout for distributed selfplay. Use 0 for script default.",
    )
    parser.add_argument(
        "--distributed-fetch-timeout-seconds",
        type=int,
        default=0,
        help="scp fetch timeout for distributed selfplay. Use 0 for script default.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Archive existing destination DB before generating new games.",
    )

    parsed = parser.parse_args(argv)

    # Validate required --board argument
    if not parsed.board:
        parser.error("--board is required (e.g., --board square8)")

    # Create SelfplayConfig for tracking
    selfplay_config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        difficulty_band=parsed.difficulty_band,
        hosts=parsed.hosts,
        source="generate_canonical_selfplay.py",
        extra_options={
            "db": parsed.db,
            "summary": parsed.summary,
            "analysis_dir": parsed.analysis_dir,
            "skip_analyses": parsed.skip_analyses,
            "parity_limit_games_per_db": parsed.parity_limit_games_per_db,
            "parity_timeout_seconds": parsed.parity_timeout_seconds,
            "distributed_job_timeout_seconds": parsed.distributed_job_timeout_seconds,
            "distributed_fetch_timeout_seconds": parsed.distributed_fetch_timeout_seconds,
            "reset_db": parsed.reset_db,
        },
    )

    # Create backward-compatible args object (uses board_type for legacy code)
    args = type("Args", (), {
        "board_type": selfplay_config.board_type,  # Map from --board to board_type
        "num_games": selfplay_config.num_games,
        "num_players": selfplay_config.num_players,
        "difficulty_band": selfplay_config.difficulty_band,
        "hosts": selfplay_config.hosts,
        "db": selfplay_config.extra_options["db"],
        "summary": selfplay_config.extra_options["summary"],
        "analysis_dir": selfplay_config.extra_options["analysis_dir"],
        "skip_analyses": selfplay_config.extra_options["skip_analyses"],
        "parity_limit_games_per_db": selfplay_config.extra_options["parity_limit_games_per_db"],
        "parity_timeout_seconds": selfplay_config.extra_options["parity_timeout_seconds"],
        "distributed_job_timeout_seconds": selfplay_config.extra_options["distributed_job_timeout_seconds"],
        "distributed_fetch_timeout_seconds": selfplay_config.extra_options["distributed_fetch_timeout_seconds"],
        "reset_db": selfplay_config.extra_options["reset_db"],
    })()

    board_type: str = args.board_type
    num_games: int = args.num_games
    num_players: int = args.num_players
    hosts: str | None = args.hosts
    difficulty_band: str = args.difficulty_band
    parity_limit_games_per_db: int = args.parity_limit_games_per_db
    parity_timeout_seconds: int | None = (
        int(args.parity_timeout_seconds)
        if int(args.parity_timeout_seconds) > 0
        else None
    )
    distributed_job_timeout_seconds: int = int(args.distributed_job_timeout_seconds or 0)
    distributed_fetch_timeout_seconds: int = int(args.distributed_fetch_timeout_seconds or 0)
    run_analyses = bool(not args.skip_analyses)

    if args.db:
        db_path = Path(args.db).resolve()
    else:
        db_name = f"canonical_{board_type}_{num_players}p.db"
        db_path = (AI_SERVICE_ROOT / "data" / "games" / db_name).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    archived_dest_db: str | None = None
    if bool(args.reset_db) and num_games > 0 and db_path.exists() and not hosts:
        archived = archive_db_in_place(db_path)
        archived_dest_db = str(archived)

    try:
        parity_summary = run_selfplay_and_parity(
            board_type,
            num_games,
            db_path,
            num_players,
            hosts,
            difficulty_band=difficulty_band,
            parity_limit_games_per_db=parity_limit_games_per_db,
            parity_timeout_seconds=parity_timeout_seconds,
            include_training_data_jsonl=run_analyses and num_games > 0,
            distributed_job_timeout_seconds=distributed_job_timeout_seconds,
            distributed_fetch_timeout_seconds=distributed_fetch_timeout_seconds,
        )
    except Exception as e:  # pragma: no cover - debug hook
        payload = {
            "board_type": board_type,
            "num_players": num_players,
            "db_path": str(db_path),
            "num_games_requested": num_games,
            "error": str(e),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise

    # Determine if the parity gate itself passed.
    passed_gate = bool(parity_summary.get("passed_canonical_parity_gate"))
    parity_rc = int(parity_summary.get("soak_returncode", 0) or 0)

    merge_result: dict[str, Any] | None = None
    if hosts:
        checked_paths = [
            Path(p)
            for p in (parity_summary.get("db_paths_checked") or [])
            if isinstance(p, str) and p
        ]
        source_dbs = [p for p in checked_paths if p.exists() and p != db_path]
        merge_result = merge_distributed_dbs(
            source_dbs=source_dbs,
            dest_db=db_path,
            reset_db=bool(args.reset_db),
        )

    merge_ok = True
    if hosts:
        merge_ok = (
            merge_result is not None
            and int(merge_result.get("returncode", 1)) == 0
            and db_path.exists()
        )

    base_ok = passed_gate and parity_rc == 0 and db_path.exists() and merge_ok

    canonical_history: dict[str, Any] = {}
    canonical_config: dict[str, Any] = {}
    games_checked = 0
    non_canonical = 0
    config_non_canonical = 0

    # Postchecks (config + canonical history) are still useful when parity
    # fails: they often point directly at the first invalid move/phase pair or
    # a mis-set non-canonical rules override. To keep failure cases debuggable
    # without making worst-case runs prohibitively slow, we run a small sample
    # and stop on the first failure when base_ok is false.
    if db_path.exists():
        postcheck_max_games = None
        stop_after_first_failure = False
        if not base_ok:
            num_games_in_db = _count_games_in_db_ro(db_path)
            if isinstance(num_games_in_db, int) and num_games_in_db > 0:
                postcheck_max_games = min(5, num_games_in_db)
            else:
                postcheck_max_games = 5
            stop_after_first_failure = True

        canonical_config = _run_canonical_config_check(
            db_path=db_path,
            max_games=postcheck_max_games,
            stop_after_first_failure=stop_after_first_failure,
        )
        config_non_canonical = int(canonical_config.get("non_canonical_games", 0) or 0)

        canonical_history = _run_canonical_history_check(
            db_path=db_path,
            max_games=postcheck_max_games,
            stop_after_first_failure=stop_after_first_failure,
        )
        games_checked = int(canonical_history.get("games_checked", 0) or 0)
        non_canonical = int(canonical_history.get("non_canonical_games", 0) or 0)

    fe_territory_fixtures_ok = run_fe_territory_fixtures(board_type)
    anm_invariants = run_anm_invariants(board_type)
    anm_ok = bool(anm_invariants.get("passed"))

    canonical_ok = (
        base_ok
        and config_non_canonical == 0
        and games_checked > 0
        and non_canonical == 0
        and fe_territory_fixtures_ok
        and anm_ok
    )

    db_stats = collect_db_stats(db_path)

    summary: dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
        "db_path": str(db_path),
        "num_games_requested": num_games,
        "db_stats": db_stats,
        "parity_gate": parity_summary,
        "archived_dest_db": archived_dest_db,
        "merge_result": merge_result,
        "canonical_config": canonical_config,
        "canonical_history": canonical_history,
        "fe_territory_fixtures_ok": bool(fe_territory_fixtures_ok),
        "anm_invariants": anm_invariants,
        "anm_ok": bool(anm_ok),
        "canonical_ok": bool(canonical_ok),
    }

    # Optional post-soak analysis (requires JSONL logs with move history).
    analysis_payload: dict[str, Any] | None = None
    if run_analyses and num_games > 0:
        jsonl_path_str = parity_summary.get("soak_log_jsonl_path")
        jsonl_path = (
            Path(jsonl_path_str).resolve()
            if isinstance(jsonl_path_str, str) and jsonl_path_str
            else None
        )
        if jsonl_path is not None and jsonl_path.exists():
            if args.analysis_dir:
                analysis_dir = Path(args.analysis_dir).resolve()
            elif args.summary:
                s = Path(args.summary).resolve()
                analysis_dir = s.parent / f"{s.stem}.analysis"
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                analysis_dir = (
                    AI_SERVICE_ROOT
                    / "logs"
                    / "selfplay"
                    / "analysis"
                    / f"{board_type}_{num_players}p_{ts}"
                )
            analysis_dir.mkdir(parents=True, exist_ok=True)

            analysis_payload = {
                "analysis_dir": str(analysis_dir),
                "log_jsonl": str(jsonl_path),
                "artifacts": {},
            }

            # 1) Overall game statistics (victory types, win rates, lengths).
            stats_base = analysis_dir / "game_statistics"
            cmd = [
                sys.executable,
                "scripts/analyze_game_statistics.py",
                "--jsonl",
                str(jsonl_path),
                "--format",
                "both",
                "--output",
                str(stats_base),
                "--quiet",
            ]
            proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)
            analysis_payload["artifacts"]["game_statistics"] = {
                "returncode": int(proc.returncode),
                "markdown": str(stats_base.with_suffix(".md")),
                "json": str(stats_base.with_suffix(".json")),
            }

            # 2) Recovery eligibility replay analysis (CPU oracle).
            recovery_out = analysis_dir / "recovery_across_games.json"
            cmd = [
                sys.executable,
                "scripts/analyze_recovery_across_games.py",
                "--input-dir",
                str(jsonl_path.parent),
                "--pattern",
                str(jsonl_path.name),
                "--output",
                str(recovery_out),
            ]
            proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)
            analysis_payload["artifacts"]["recovery_across_games"] = {
                "returncode": int(proc.returncode),
                "json": str(recovery_out),
            }

            # 3) Recovery opportunity windows (lightweight replay).
            recovery_opp_out = analysis_dir / "recovery_opportunities.txt"
            cmd = [
                sys.executable,
                "scripts/analyze_recovery_opportunities.py",
                "--dir",
                str(jsonl_path.parent),
            ]
            proc = _run_cmd(cmd, cwd=AI_SERVICE_ROOT)
            output = (proc.stdout or "") + (proc.stderr or "")
            recovery_opp_out.write_text(output, encoding="utf-8")
            analysis_payload["artifacts"]["recovery_opportunities"] = {
                "returncode": int(proc.returncode),
                "text": str(recovery_opp_out),
            }
        else:
            analysis_payload = {
                "skipped": True,
                "reason": "soak_log_jsonl_path_missing_or_not_found",
            }

    if analysis_payload is not None:
        summary["analysis"] = analysis_payload

    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(
            f"[generate_canonical_selfplay] Wrote summary: {summary_path}",
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if canonical_ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
