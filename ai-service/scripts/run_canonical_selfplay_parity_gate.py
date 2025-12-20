#!/usr/bin/env python
"""
Canonical self-play + parity gate driver.

This script automates a minimal end-to-end check for a given board type:

  1. Run a small Python self-play soak using the canonical GameEngine
     and record completed games into a GameReplayDB.
  2. Run the TS↔Python replay parity harness on that DB.
  3. Emit a compact JSON summary describing whether the DB passes the
     "canonical parity gate" (no structural issues, no semantic divergence).

Typical usage (from ai-service/):

  PYTHONPATH=. python scripts/run_canonical_selfplay_parity_gate.py \\
    --board-type square8 \\
    --num-games 20 \\
    --db data/games/selfplay_square8_parity_gate.db \\
    --summary parity_gate.square8.json

Progress / observability:
  - Long-running stages stream progress to **stderr**.
  - When ``--summary`` is set, a heartbeat refreshes the JSON with elapsed time
    and best-effort DB row counts (games + moves). You can watch it with:
      `tail -f parity_gate.square8.json`

The intent is to make it easy to:
  - Generate fresh, canonical self-play DBs per board type, and
  - Gate training pipelines on those DBs passing basic parity checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
# Ensure `app.*` imports resolve when invoked from repo root.
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import BoardType
from app.training.env import get_theoretical_max_moves
from app.training.selfplay_config import SelfplayConfig, create_argument_parser


def _run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    *,
    capture_output: bool = True,
    stream_to_stderr: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess and return the completed process."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    stdout = None
    stderr = None
    if stream_to_stderr:
        # Keep stdout clean for the final JSON gate summary while still
        # providing progress output during long soaks.
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
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=124)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _start_heartbeat(
    summary_path: Path,
    stage_payload: dict[str, Any],
    *,
    heartbeat_seconds: int,
    db_path: Path | None = None,
    stage_label: str,
) -> threading.Event:
    """Periodically update the summary JSON so long-running stages are observable."""
    stop_event = threading.Event()
    started = time.monotonic()

    def _try_read_db_counts(path: Path) -> dict[str, int]:
        """Best-effort progress counters for the GameReplayDB SQLite file."""
        try:
            import sqlite3

            uri = f"file:{path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=0.1)
            cur = conn.cursor()
            counts: dict[str, int] = {}
            for table in ("games", "game_moves"):
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        counts[f"db_{table}_count"] = int(row[0])
                except Exception:
                    continue
            conn.close()
            return counts
        except Exception:
            return {}

    def _beat() -> None:
        while not stop_event.wait(heartbeat_seconds):
            payload = dict(stage_payload)
            payload["heartbeat"] = {
                "stage": stage_label,
                "elapsed_sec": round(time.monotonic() - started, 1),
                "ts_unix": time.time(),
            }
            if db_path is not None:
                try:
                    payload["heartbeat"]["db_size_bytes"] = db_path.stat().st_size
                except OSError:
                    pass
                payload["heartbeat"].update(_try_read_db_counts(db_path))
            _write_json(summary_path, payload)
            print(
                f"[parity-gate] heartbeat: stage={stage_label} "
                f"elapsed={payload['heartbeat']['elapsed_sec']}s "
                f"games={payload['heartbeat'].get('db_games_count', '?')} "
                f"moves={payload['heartbeat'].get('db_game_moves_count', '?')}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event


def run_selfplay_soak(
    board_type: str,
    num_games: int,
    db_path: Path,
    seed: int | None,
    max_moves: int,
    num_players: int,
    difficulty_band: str,
    include_training_data_jsonl: bool = False,
    soak_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run a small Python self-play soak and record games to db_path."""
    logs_dir = AI_SERVICE_ROOT / "logs" / "selfplay"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Use a DB-scoped log path so concurrent parity gates don't contend on the
    # same JSONL lock file (run_self_play_soak.py uses fcntl flock).
    run_tag = f"{db_path.stem}.{board_type}.{num_players}p.{difficulty_band}"
    summary_path = logs_dir / f"soak.{run_tag}.summary.json"
    jsonl_path = logs_dir / f"soak.{run_tag}.jsonl"

    extra_args: list[str] = []
    if board_type in {"square19", "hexagonal"}:
        # Large boards can exhaust memory mid-game; streaming record and intra-game
        # GC bound peak usage without changing rules semantics.
        extra_args += ["--streaming-record", "--intra-game-gc-interval", "50"]

    cmd = [
        sys.executable,
        "scripts/run_self_play_soak.py",
        "--num-games",
        str(num_games),
        "--board-type",
        board_type,
        "--engine-mode",
        "mixed",
        "--num-players",
        str(num_players),
        "--max-moves",
        str(max_moves),
        "--difficulty-band",
        difficulty_band,
        "--record-db",
        str(db_path),
        "--log-jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_path),
        "--fail-on-anomaly",
        *extra_args,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if include_training_data_jsonl:
        cmd.append("--include-training-data")

    # Enable strict invariant by default so soak respects ANM constraints.
    env_overrides = {
        "RINGRIFT_STRICT_NO_MOVE_INVARIANT": "1",
        # Ensure phase/move invariant checks remain enabled for canonical gates.
        "RINGRIFT_SKIP_PHASE_INVARIANT": "0",
        # Enforce TS↔Python parity during recording; abort early on any divergence.
        # This prevents non-canonical games (e.g., actor mismatches) from entering
        # the canonical DB in the first place.
        "RINGRIFT_PARITY_VALIDATION": "strict",
        # Ensure host applies required bookkeeping/no-op moves for the same actor
        # in line/territory phases, matching TS orchestration.
        "RINGRIFT_FORCE_BOOKKEEPING_MOVES": "1",
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        "PYTHONUNBUFFERED": "1",
        # Keep OpenMP usage conservative for long-running soaks and
        # avoid environment-specific SHM issues on some platforms.
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
    }
    # Large-board parity debugging: disable fast territory detection to
    # avoid masking territory-processing divergence during canonical gates.
    if board_type in {"square19", "hexagonal"}:
        env_overrides.setdefault("RINGRIFT_USE_FAST_TERRITORY", "false")
        # Default to the incremental make/unmake evaluator on large boards
        # to avoid long-running heuristic evaluations during canonical gates.
        if "RINGRIFT_USE_MAKE_UNMAKE" not in os.environ:
            env_overrides["RINGRIFT_USE_MAKE_UNMAKE"] = "true"

    print(
        f"[parity-gate] self-play soak: board={board_type} players={num_players} "
        f"games={num_games} difficulty_band={difficulty_band}",
        file=sys.stderr,
        flush=True,
    )
    proc = _run_cmd(
        cmd,
        cwd=AI_SERVICE_ROOT,
        env_overrides=env_overrides,
        capture_output=False,
        stream_to_stderr=True,
        timeout_seconds=soak_timeout_seconds,
    )

    soak_summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                soak_summary = json.load(f)
        except Exception:
            soak_summary = None

    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "summary_path": str(summary_path),
        "summary": soak_summary,
        "log_jsonl_path": str(jsonl_path),
    }
    return result


def run_parity_check(
    db_path: Path,
    *,
    progress_every: int = 200,
    parity_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run the TS↔Python parity harness on a single DB and return the parsed summary.

    This always invokes the parity script in **canonical** mode with
    ``view = post_move`` so that:

      - ``passed_canonical_parity_gate`` in the returned JSON reflects the
        canonical gate status for this DB, and
      - the process return code is non-zero whenever the canonical gate fails.
    """
    summary_path = db_path.with_suffix(db_path.suffix + ".parity_summary.json")
    cmd = [
        sys.executable,
        "scripts/check_ts_python_replay_parity.py",
        "--db",
        str(db_path),
        "--mode",
        "canonical",
        "--view",
        "post_move",
        "--summary-json",
        str(summary_path),
    ]
    if progress_every and progress_every > 0:
        cmd += ["--progress-every", str(progress_every)]
    env_overrides = {
        "PYTHONPATH": str(AI_SERVICE_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    print(
        f"[parity-gate] parity check: db={db_path.name}",
        file=sys.stderr,
        flush=True,
    )
    proc = _run_cmd(
        cmd,
        cwd=AI_SERVICE_ROOT,
        env_overrides=env_overrides,
        capture_output=False,
        stream_to_stderr=True,
        timeout_seconds=parity_timeout_seconds,
    )

    summary: dict[str, Any]
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {
                "error": "failed_to_load_parity_summary_file",
                "summary_path": str(summary_path),
            }
    else:
        summary = {
            "error": "parity_summary_file_missing",
            "summary_path": str(summary_path),
        }

    summary["returncode"] = proc.returncode
    summary["summary_path"] = str(summary_path)
    return summary


def run_parity_checks(db_paths: list[Path]) -> dict[str, Any]:
    """Run parity on multiple DBs and aggregate results."""
    summaries: list[dict[str, Any]] = []
    all_pass = True
    for db_path in db_paths:
        summary = run_parity_check(db_path)
        summaries.append({"db": str(db_path), "summary": summary})
        rc = summary.get("returncode", 1)
        struct = int(summary.get("games_with_structural_issues", 0))
        sem = int(summary.get("games_with_semantic_divergence", 0))
        total_checked = int(summary.get("total_games_checked", 0))
        if rc != 0 or struct > 0 or sem > 0 or total_checked == 0:
            all_pass = False
    return {"all_pass": all_pass, "per_db": summaries}


def main() -> None:
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Run canonical Python self-play for a board type and gate the resulting GameReplayDB on TS↔Python parity.",
        include_gpu=False,
        include_ramdrive=False,
    )

    # Add parity gate-specific arguments
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the GameReplayDB SQLite file to write.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=0,
        help="Maximum moves per game (0 to auto-select from board/players).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to write the parity gate JSON summary.",
    )
    parser.add_argument(
        "--include-training-data-jsonl",
        action="store_true",
        help="Include full move history + initial_state in the JSONL log.",
    )
    parser.add_argument(
        "--parity-progress-every",
        type=int,
        default=200,
        help="Emit parity progress to stderr every N replay steps (0 disables).",
    )
    parser.add_argument(
        "--soak-timeout-seconds",
        type=int,
        default=0,
        help="Optional wall-clock timeout for the self-play soak (0 disables).",
    )
    parser.add_argument(
        "--distributed-job-timeout-seconds",
        type=int,
        default=0,
        help="Per-job timeout for distributed selfplay (0 uses default).",
    )
    parser.add_argument(
        "--distributed-fetch-timeout-seconds",
        type=int,
        default=0,
        help="scp fetch timeout for distributed selfplay (0 uses default).",
    )
    parser.add_argument(
        "--keep-distributed-remote-artifacts",
        action="store_true",
        help="Keep remote DB/JSONL artifacts after fetching.",
    )
    parser.add_argument(
        "--parity-timeout-seconds",
        type=int,
        default=0,
        help="Optional wall-clock timeout for the parity check (0 disables).",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Emit heartbeat to stderr every N seconds (0 disables).",
    )

    parsed = parser.parse_args()

    # Validate required --board argument
    if not parsed.board:
        parser.error("--board is required (e.g., --board square8)")

    # Create SelfplayConfig for tracking
    selfplay_config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        seed=parsed.seed,
        difficulty_band=parsed.difficulty_band,
        hosts=parsed.hosts,
        source="run_canonical_selfplay_parity_gate.py",
        extra_options={
            "db": parsed.db,
            "max_moves": parsed.max_moves,
            "summary": parsed.summary,
            "include_training_data_jsonl": parsed.include_training_data_jsonl,
            "parity_progress_every": parsed.parity_progress_every,
            "soak_timeout_seconds": parsed.soak_timeout_seconds,
            "distributed_job_timeout_seconds": parsed.distributed_job_timeout_seconds,
            "distributed_fetch_timeout_seconds": parsed.distributed_fetch_timeout_seconds,
            "keep_distributed_remote_artifacts": parsed.keep_distributed_remote_artifacts,
            "parity_timeout_seconds": parsed.parity_timeout_seconds,
            "heartbeat_seconds": parsed.heartbeat_seconds,
        },
    )

    # Create backward-compatible args object (uses board_type for legacy code)
    args = type("Args", (), {
        "board_type": selfplay_config.board_type,
        "num_games": selfplay_config.num_games,
        "difficulty_band": selfplay_config.difficulty_band,
        "num_players": selfplay_config.num_players,
        "db": selfplay_config.extra_options["db"],
        "seed": selfplay_config.seed,
        "max_moves": selfplay_config.extra_options["max_moves"],
        "hosts": selfplay_config.hosts,
        "summary": selfplay_config.extra_options["summary"],
        "include_training_data_jsonl": selfplay_config.extra_options["include_training_data_jsonl"],
        "parity_progress_every": selfplay_config.extra_options["parity_progress_every"],
        "soak_timeout_seconds": selfplay_config.extra_options["soak_timeout_seconds"],
        "distributed_job_timeout_seconds": selfplay_config.extra_options["distributed_job_timeout_seconds"],
        "distributed_fetch_timeout_seconds": selfplay_config.extra_options["distributed_fetch_timeout_seconds"],
        "keep_distributed_remote_artifacts": selfplay_config.extra_options["keep_distributed_remote_artifacts"],
        "parity_timeout_seconds": selfplay_config.extra_options["parity_timeout_seconds"],
        "heartbeat_seconds": selfplay_config.extra_options["heartbeat_seconds"],
    })()

    db_path = Path(args.db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    parity_summary: dict[str, Any] | dict[str, Any]
    soak_result: dict[str, Any] = {}
    dbs_to_check: list[Path] = [db_path]

    summary_path: Path | None = None
    if args.summary:
        summary_path = Path(args.summary).resolve()
        _write_json(
            summary_path,
            {
                "stage": "starting",
                "board_type": args.board_type,
                "num_players": args.num_players,
                "db_path": str(db_path),
            },
        )
        print(
            f"[parity-gate] summary JSON: {summary_path} "
            f"(tail -f {summary_path} for heartbeat updates)",
            file=sys.stderr,
            flush=True,
        )

    # Auto-select max_moves when not provided.
    if args.max_moves and args.max_moves > 0:
        max_moves = args.max_moves
    else:
        bt_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_enum = bt_map[args.board_type]
        max_moves = get_theoretical_max_moves(board_enum, args.num_players)

    if args.hosts:
        hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"distributed_soak_{args.board_type}_{args.num_players}p_{ts}"
        output_dir = db_path.parent / "distributed_soak_runs" / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Delegate to distributed soak runner with filters
        try:
            output_dir_arg = str(output_dir.relative_to(AI_SERVICE_ROOT))
        except ValueError:
            output_dir_arg = str(output_dir)

        cmd = [
            sys.executable,
            "scripts/run_distributed_selfplay_soak.py",
            "--games-per-config",
            str(args.num_games),
            "--hosts",
            ",".join(hosts),
            "--output-dir",
            output_dir_arg,
            "--board-types",
            args.board_type,
            "--num-players",
            str(args.num_players),
            "--difficulty-band",
            args.difficulty_band,
            "--max-parallel-per-host",
            "2",
            "--fetch-jsonl",
        ]
        if args.seed is not None:
            cmd += ["--base-seed", str(args.seed)]
        if not args.keep_distributed_remote_artifacts:
            cmd.append("--cleanup-remote")
        if args.distributed_job_timeout_seconds and args.distributed_job_timeout_seconds > 0:
            cmd += ["--job-timeout-seconds", str(args.distributed_job_timeout_seconds)]
        if args.distributed_fetch_timeout_seconds and args.distributed_fetch_timeout_seconds > 0:
            cmd += ["--fetch-timeout-seconds", str(args.distributed_fetch_timeout_seconds)]
        print(
            f"[parity-gate] distributed soak: board={args.board_type} players={args.num_players} "
            f"games_per_config={args.num_games} difficulty_band={args.difficulty_band}",
            file=sys.stderr,
            flush=True,
        )
        proc = _run_cmd(
            cmd,
            cwd=AI_SERVICE_ROOT,
            capture_output=False,
            stream_to_stderr=True,
            timeout_seconds=args.soak_timeout_seconds or None,
        )
        soak_result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "distributed": True,
            "output_dir": str(output_dir),
        }
        # Collect DBs produced for this config
        dbs_to_check = list(output_dir.glob(f"selfplay_{args.board_type}_{args.num_players}p_*.db"))
        if not dbs_to_check:
            parity_summary = {"error": "no_db_produced", "returncode": 1}
        else:
            parity_summary = run_parity_checks(dbs_to_check)
    else:
        selfplay_heartbeat_stop: threading.Event | None = None
        if summary_path is not None:
            selfplay_stage = {
                "stage": "selfplay_running",
                "board_type": args.board_type,
                "num_players": args.num_players,
                "db_path": str(db_path),
                "num_games": args.num_games,
                "difficulty_band": args.difficulty_band,
                "seed": args.seed,
                "max_moves": max_moves,
            }
            _write_json(summary_path, selfplay_stage)
            if args.heartbeat_seconds and args.heartbeat_seconds > 0:
                selfplay_heartbeat_stop = _start_heartbeat(
                    summary_path,
                    selfplay_stage,
                    heartbeat_seconds=args.heartbeat_seconds,
                    db_path=db_path,
                    stage_label="selfplay",
                )
        print(
            f"[parity-gate] starting selfplay: board={args.board_type} players={args.num_players} "
            f"games={args.num_games} max_moves={max_moves} db={db_path}",
            file=sys.stderr,
            flush=True,
        )
        soak_result = run_selfplay_soak(
            args.board_type,
            args.num_games,
            db_path,
            args.seed,
            max_moves,
            args.num_players,
            args.difficulty_band,
            include_training_data_jsonl=bool(args.include_training_data_jsonl),
            soak_timeout_seconds=args.soak_timeout_seconds or None,
        )
        if selfplay_heartbeat_stop is not None:
            selfplay_heartbeat_stop.set()
        print(
            f"[parity-gate] selfplay complete: returncode={soak_result.get('returncode')}",
            file=sys.stderr,
            flush=True,
        )

        parity_heartbeat_stop: threading.Event | None = None
        if summary_path is not None:
            parity_stage = {
                "stage": "parity_running",
                "board_type": args.board_type,
                "num_players": args.num_players,
                "db_path": str(db_path),
                "soak_returncode": soak_result.get("returncode"),
            }
            _write_json(summary_path, parity_stage)
            if args.heartbeat_seconds and args.heartbeat_seconds > 0:
                parity_heartbeat_stop = _start_heartbeat(
                    summary_path,
                    parity_stage,
                    heartbeat_seconds=args.heartbeat_seconds,
                    db_path=db_path,
                    stage_label="parity",
                )
        print(
            f"[parity-gate] starting parity: db={db_path} progress_every={args.parity_progress_every} "
            f"timeout={args.parity_timeout_seconds or 'none'}",
            file=sys.stderr,
            flush=True,
        )
        parity_summary = run_parity_check(
            db_path,
            progress_every=args.parity_progress_every,
            parity_timeout_seconds=args.parity_timeout_seconds or None,
        )
        if parity_heartbeat_stop is not None:
            parity_heartbeat_stop.set()
        print(
            f"[parity-gate] parity complete: returncode={parity_summary.get('returncode')}",
            file=sys.stderr,
            flush=True,
        )

    # Basic gate: soak must succeed and the canonical parity gate must pass.
    #
    # We require:
    #   - soak_returncode == 0 (Python self-play soak did not abort), and
    #   - parity harness reports passed_canonical_parity_gate == True, and
    #   - underlying parity process exits with code 0.
    #
    # For distributed runs (args.hosts), run_parity_checks() aggregates the
    # per-DB parity summaries; we treat all_pass == True as a requirement that
    # every DB passed its canonical parity gate.
    passed = False
    soak_rc = soak_result.get("returncode")
    if soak_rc == 0 and "error" not in parity_summary:
        if args.hosts:
            passed = bool(parity_summary.get("all_pass"))
        else:
            parity_rc = int(parity_summary.get("returncode", 1))
            passed_gate = bool(parity_summary.get("passed_canonical_parity_gate"))
            passed = parity_rc == 0 and passed_gate

    gate_summary: dict[str, Any] = {
        "stage": "complete",
        "board_type": args.board_type,
        "num_players": args.num_players,
        "db_path": str(db_path),
        "db_paths_checked": [str(p) for p in dbs_to_check],
        "num_games": args.num_games,
        "difficulty_band": args.difficulty_band,
        "seed": args.seed,
        "max_moves": max_moves,
        "hosts": args.hosts.split(",") if args.hosts else None,
        "soak_returncode": soak_result.get("returncode"),
        "soak_summary_path": soak_result.get("summary_path"),
        "soak_log_jsonl_path": soak_result.get("log_jsonl_path"),
        "parity_summary": parity_summary,
        "passed_canonical_parity_gate": bool(passed),
    }

    if args.summary:
        _write_json(Path(args.summary).resolve(), gate_summary)

    print(json.dumps(gate_summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
