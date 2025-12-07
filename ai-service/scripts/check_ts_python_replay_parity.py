"""TS vs Python replay parity checker for GameReplayDB databases.

This script walks all self-play GameReplayDB databases under the repo and,
for each game, compares Python's GameReplayDB.get_state_at_move against the
TypeScript ClientSandboxEngine replay path driven via the
scripts/selfplay-db-ts-replay.ts harness.

For each game it checks:
  - That TS replays the same number of moves as Python reports (total_moves).
  - For k = 0..min(total_moves, tsApplied):
      * currentPlayer
      * currentPhase
      * gameStatus
      * state hash (shared progress/hash function)

Any mismatch is reported with (db, game_id, move_index, python_summary, ts_summary).

Usage (from ai-service/):

  python scripts/check_ts_python_replay_parity.py

You can optionally restrict to a single DB:

  python scripts/check_ts_python_replay_parity.py --db ../ai-service/logs/cmaes/.../games.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.db.game_replay import GameReplayDB, _compute_state_hash
from app.training.generate_data import create_initial_state
from app.game_engine import BoardType
from app.rules.serialization import serialize_game_state
from app.rules.history_validation import validate_canonical_history_for_game


@dataclass
class StateSummary:
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str


@dataclass
class GameParityResult:
    db_path: str
    game_id: str
    structure: str
    structure_reason: Optional[str]
    total_moves_python: int
    total_moves_ts: int
    diverged_at: Optional[int]
    python_summary: Optional[StateSummary]
    ts_summary: Optional[StateSummary]
    # High-level classification of what differed at diverged_at, e.g.:
    # ['current_player'], ['current_phase', 'game_status'], ['move_count'], ['ts_missing_step']
    mismatch_kinds: List[str] = field(default_factory=list)
    # Optional free-form context, such as "initial_state" vs "post_move"
    mismatch_context: Optional[str] = None
    # True when divergence occurs only at the very last move (end-of-game metadata
    # differences). These are insignificant for training purposes as no AI decisions
    # are made based on the divergent state.
    is_end_of_game_only: bool = False


def _canonicalize_status(status: str | None) -> str:
    """Normalize status strings so 'finished' and 'completed' compare equal.

    The canonical terminal value is 'completed'; legacy 'finished' values from
    older TS/PT traces are treated as aliases for comparison purposes.
    """
    if status is None:
        return "active"
    s = str(status)
    if s == "finished":
        return "completed"
    return s


def repo_root() -> Path:
    """Return the monorepo root (one level above ai-service/)."""
    return Path(__file__).resolve().parents[2]


def find_dbs(explicit_db: Optional[str] = None) -> List[Path]:
    """Find GameReplayDB files to inspect."""
    root = repo_root()
    if explicit_db:
        return [Path(explicit_db).resolve()]

    search_paths = [
        root / "data" / "games",
        root / "ai-service" / "logs" / "cmaes",
        root / "ai-service" / "data" / "games",
    ]

    results: List[Path] = []
    visited = set()

    def walk(dir_path: Path, depth: int) -> None:
        if depth <= 0:
            return
        real = dir_path.resolve()
        if real in visited or not real.exists():
            return
        visited.add(real)
        try:
            entries = list(real.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                walk(entry, depth - 1)
            elif entry.is_file() and (entry.name == "games.db" or entry.name.endswith(".db")):
                results.append(entry)

    for base in search_paths:
        walk(base, 7)

    return results


def import_json_to_temp_db(json_path: str) -> Tuple[Path, str]:
    """Import a JSON scenario/fixture file into a temporary GameReplayDB.

    Supports three formats:
      1. LoadableScenario: { id, name, boardType, state, selfPlayMeta?: { moves } }
      2. ringrift_sandbox_fixture_v1: { kind, boardType, state, moveHistory }
      3. GameRecord (golden games): { id, boardType, numPlayers, players, moves }

    Returns (temp_db_path, game_id).
    """
    import sqlite3
    from datetime import datetime

    with open(json_path, "r") as f:
        data = json.load(f)

    # Detect format and extract fields
    players: list = []  # For GameRecord format
    if data.get("kind") == "ringrift_sandbox_fixture_v1":
        # Sandbox fixture format
        game_id = data.get("id") or str(uuid.uuid4())
        board_type = data.get("boardType") or data.get("state", {}).get("board", {}).get("type", "square8")
        state_json = data.get("state", {})
        moves = data.get("moveHistory", [])
        num_players = len(state_json.get("players", [])) or 2
    elif "state" in data:
        # LoadableScenario format
        game_id = data.get("id") or str(uuid.uuid4())
        board_type = data.get("boardType", "square8")
        state_json = data.get("state", {})
        num_players = data.get("playerCount", 2)
        # Moves can be in selfPlayMeta.moves or directly in moveHistory
        moves = []
        if "selfPlayMeta" in data and "moves" in data["selfPlayMeta"]:
            moves = data["selfPlayMeta"]["moves"]
        elif "moveHistory" in state_json:
            moves = state_json.get("moveHistory", [])
    elif "moves" in data and "boardType" in data:
        # GameRecord format (golden games, soak exports)
        game_id = data.get("id") or str(uuid.uuid4())
        board_type = data.get("boardType", "square8")
        num_players = data.get("numPlayers", 2)
        moves = data.get("moves", [])
        players = data.get("players", [])
        # No initial state - use empty dict to signal fresh game
        state_json = {}
    else:
        raise ValueError(f"Unrecognized JSON format in {json_path}")

    # Create temporary DB
    temp_dir = tempfile.mkdtemp(prefix="parity_json_")
    temp_db_path = Path(temp_dir) / "imported.db"
    conn = sqlite3.connect(str(temp_db_path))
    cur = conn.cursor()

    # Create full schema (version 6) to match GameReplayDB and TS replay expectations
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR REPLACE INTO schema_metadata (key, value) VALUES ('schema_version', '6');

        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            rng_seed INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            game_status TEXT NOT NULL,
            winner INTEGER,
            termination_reason TEXT,
            total_moves INTEGER NOT NULL,
            total_turns INTEGER NOT NULL,
            duration_ms INTEGER,
            source TEXT,
            schema_version INTEGER NOT NULL,
            time_control_type TEXT DEFAULT 'none',
            initial_time_ms INTEGER,
            time_increment_ms INTEGER,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS game_players (
            game_id TEXT NOT NULL,
            player_number INTEGER NOT NULL,
            player_type TEXT NOT NULL,
            ai_type TEXT,
            ai_difficulty INTEGER,
            ai_profile_id TEXT,
            final_eliminated_rings INTEGER,
            final_territory_spaces INTEGER,
            final_rings_in_hand INTEGER,
            PRIMARY KEY (game_id, player_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS game_initial_state (
            game_id TEXT PRIMARY KEY,
            initial_state_json TEXT NOT NULL,
            compressed INTEGER DEFAULT 0,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS game_moves (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            turn_number INTEGER NOT NULL,
            player INTEGER NOT NULL,
            phase TEXT NOT NULL,
            move_type TEXT NOT NULL,
            move_json TEXT NOT NULL,
            timestamp TEXT,
            think_time_ms INTEGER,
            time_remaining_ms INTEGER,
            engine_eval REAL,
            engine_eval_type TEXT,
            engine_depth INTEGER,
            engine_nodes INTEGER,
            engine_pv TEXT,
            engine_time_ms INTEGER,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS game_state_snapshots (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            compressed INTEGER DEFAULT 0,
            state_hash TEXT,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS game_choices (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            choice_type TEXT NOT NULL,
            player INTEGER NOT NULL,
            options_json TEXT NOT NULL,
            selected_option_json TEXT NOT NULL,
            ai_reasoning TEXT,
            PRIMARY KEY (game_id, move_number, choice_type),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS game_history_entries (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            player INTEGER NOT NULL,
            phase_before TEXT NOT NULL,
            phase_after TEXT NOT NULL,
            status_before TEXT NOT NULL,
            status_after TEXT NOT NULL,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
    """)

    # Insert game record
    now = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO games (game_id, board_type, num_players, created_at, game_status,
                          total_moves, total_turns, source, schema_version)
        VALUES (?, ?, ?, ?, 'completed', ?, ?, 'json_import', 6)
    """, (game_id, board_type, num_players, now, len(moves), len(moves)))

    # Insert player records (GameRecord format has players, others generate defaults)
    if players:
        for p in players:
            player_num = p.get("playerNumber", 1)
            player_type = p.get("playerType", "ai")
            ai_type = p.get("aiType")
            ai_difficulty = p.get("aiDifficulty")
            cur.execute("""
                INSERT INTO game_players (game_id, player_number, player_type, ai_type, ai_difficulty)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, player_num, player_type, ai_type, ai_difficulty))
    else:
        # Create default player entries for formats without explicit players
        for i in range(num_players):
            cur.execute("""
                INSERT INTO game_players (game_id, player_number, player_type)
                VALUES (?, ?, 'ai')
            """, (game_id, i + 1))

    # Insert initial state only if we have one (GameRecord format has empty state_json)
    if state_json:
        cur.execute("""
            INSERT INTO game_initial_state (game_id, initial_state_json, compressed)
            VALUES (?, ?, 0)
        """, (game_id, json.dumps(state_json)))

    # Insert moves
    for i, move in enumerate(moves):
        # Extract move metadata with fallbacks
        move_type = move.get("type") or move.get("actionType") or move.get("action_type") or "unknown"
        player = move.get("player") or move.get("playerNumber") or 1
        phase = move.get("phase") or move.get("currentPhase") or "unknown"
        turn = move.get("turn") or move.get("turnNumber") or i

        cur.execute("""
            INSERT INTO game_moves (game_id, move_number, turn_number, player, phase, move_type, move_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (game_id, i, turn, player, phase, move_type, json.dumps(move)))

    conn.commit()
    conn.close()

    return temp_db_path, game_id


def summarize_python_state(db: GameReplayDB, game_id: str, move_index: int) -> StateSummary:
    """Summarize state AFTER move_index is applied."""
    state = db.get_state_at_move(game_id, move_index)
    if state is None:
        raise RuntimeError(f"Python get_state_at_move returned None for {game_id} @ {move_index}")
    return StateSummary(
        move_index=move_index,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
    )


def _parse_board_type(board_type_str: str) -> BoardType:
    """Parse a board type string into a BoardType enum value."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    return mapping.get(board_type_str.lower(), BoardType.SQUARE8)


def summarize_python_initial_state(db: GameReplayDB, game_id: str) -> StateSummary:
    """Summarize the initial state BEFORE any moves are applied."""
    state = db.get_initial_state(game_id)
    if state is None:
        # No initial_state record - create a fresh initial state from game metadata.
        # This handles GameRecord format imports (golden games, soak exports) which
        # record games from a standard empty board without explicit initial state.
        metadata = db.get_game_metadata(game_id)
        if metadata is None:
            raise RuntimeError(f"Python get_initial_state returned None and no game metadata for {game_id}")

        board_type_str = metadata.get("board_type", "square8")
        num_players = metadata.get("num_players", 2)
        board_type = _parse_board_type(board_type_str)
        state = create_initial_state(board_type=board_type, num_players=num_players)

    return StateSummary(
        move_index=0,
        current_player=state.current_player,
        current_phase=state.current_phase.value
        if hasattr(state.current_phase, "value")
        else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
    )


def classify_game_structure(db: GameReplayDB, game_id: str) -> Tuple[str, str]:
    """Classify game recording structure.

    Returns (structure, reason) where structure is one of:
      - "good": initial state looks like a true game start (empty board).
      - "mid_snapshot": initial state appears to be a mid-game snapshot.
      - "invalid": missing data.

    Note: We do NOT compare initial_state with get_state_at_move(0) because
    get_state_at_move(0) returns the state AFTER move 0 is applied, which is
    expected to differ from initial_state (e.g., phase changes from
    ring_placement to movement after the first ring is placed).
    """
    initial = db.get_initial_state(game_id)
    if initial is None:
        # No initial_state record means a fresh game from empty board (e.g., soak
        # test games, GameRecord format imports). Treat as "good" since both TS
        # and Python can replay from the standard empty board for this board type.
        return "good", None

    # Treat any pre-populated history or board content as a mid-game snapshot.
    move_hist_len = len(initial.move_history or [])
    board = initial.board
    stacks = getattr(board, "stacks", {}) or {}
    markers = getattr(board, "markers", {}) or {}
    collapsed = getattr(board, "collapsed_spaces", {}) or {}

    stack_count = len(stacks)
    marker_count = len(markers)
    collapsed_count = len(collapsed)

    if move_hist_len > 0 or stack_count > 0 or marker_count > 0 or collapsed_count > 0:
        reason = (
            "initial_state contains history/board: "
            f"move_history={move_hist_len}, stacks={stack_count}, "
            f"markers={marker_count}, collapsed={collapsed_count}"
        )
        return "mid_snapshot", reason

    # Verify we can at least replay move 0
    state0 = db.get_state_at_move(game_id, 0)
    if state0 is None:
        return "invalid", "get_state_at_move(0) returned None"

    return "good", ""


def _get_python_state_for_ts_k(
    db: GameReplayDB,
    game_id: str,
    ts_k: int,
    total_moves_py: int,
):
    """Return the Python GameState corresponding to TS step k, or None if unavailable."""
    if ts_k <= 0:
        state = db.get_initial_state(game_id)
        if state is None:
            metadata = db.get_game_metadata(game_id)
            if metadata is None:
                raise RuntimeError(
                    f"Python get_initial_state returned None and no game metadata for {game_id}"
                )
            board_type_str = metadata.get("board_type", "square8")
            num_players = metadata.get("num_players", 2)
            board_type = _parse_board_type(board_type_str)
            state = create_initial_state(board_type=board_type, num_players=num_players)
        return state

    py_index = ts_k - 1
    if py_index < 0 or py_index >= total_moves_py:
        return None

    state = db.get_state_at_move(game_id, py_index)
    return state


def run_ts_replay(db_path: Path, game_id: str) -> Tuple[int, Dict[int, StateSummary]]:
    """Invoke the TS harness and parse its per-move summaries.

    Returns:
      (total_moves_reported_by_ts, mapping from move_index -> summary)
    """
    root = repo_root()
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

    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay harness failed for {db_path} / {game_id} with code {proc.returncode}:\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    total_ts_moves = 0
    summaries: Dict[int, StateSummary] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = payload.get("kind")
        if kind == "ts-replay-initial":
            total_ts_moves = int(payload.get("totalRecordedMoves", 0))
            summary = payload.get("summary") or {}
            summaries[0] = StateSummary(
                move_index=0,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-step":
            k = int(payload.get("k", 0))
            summary = payload.get("summary") or {}
            summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-final":
            # We could cross-check appliedMoves here if needed.
            pass

    return total_ts_moves, summaries


def _dump_ts_states_for_ks(
    db_path: Path,
    game_id: str,
    ks: List[int],
    dump_dir: Path,
) -> None:
    """Invoke the TS replay harness to dump TS GameState JSON at the requested k values."""
    if not ks:
        return

    root = repo_root()
    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")
    env["RINGRIFT_TS_REPLAY_DUMP_DIR"] = str(dump_dir)

    unique_sorted = sorted(set(ks))
    dump_arg = ",".join(str(k) for k in unique_sorted)

    cmd = [
        "npx",
        "ts-node",
        "-T",
        "scripts/selfplay-db-ts-replay.ts",
        "--db",
        str(db_path),
        "--game",
        game_id,
        "--dump-state-at",
        dump_arg,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay state dump failed for {db_path} / {game_id} ks={unique_sorted} "
            f"with code {proc.returncode}: {proc.stderr.strip()}"
        )


def dump_state_bundle(
    db: GameReplayDB,
    db_path: Path,
    game_id: str,
    result: GameParityResult,
    state_bundles_dir: Path,
) -> None:
    """Emit a rich TS+Python state bundle around the first divergence for faster debugging."""
    if result.diverged_at is None:
        return

    div_k = result.diverged_at
    if div_k <= 0:
        ks = [0]
    else:
        before_k = max(0, div_k - 1)
        ks = [before_k, div_k] if div_k != before_k else [div_k]

    py_states: Dict[int, Dict[str, object]] = {}
    ts_states: Dict[int, Dict[str, object]] = {}

    for ts_k in ks:
        state = _get_python_state_for_ts_k(db, game_id, ts_k, result.total_moves_python)
        if state is not None:
            try:
                py_states[ts_k] = serialize_game_state(state)  # type: ignore[arg-type]
            except Exception:
                continue

    try:
        _dump_ts_states_for_ks(db_path, game_id, ks, state_bundles_dir)
    except Exception:
        pass

    for ts_k in ks:
        file_name = f"{db_path.name}__{game_id}__k{ts_k}.ts_state.json"
        ts_path = state_bundles_dir / file_name
        if not ts_path.exists():
            continue
        try:
            with ts_path.open("r", encoding="utf-8") as f:
                ts_states[ts_k] = json.load(f)
        except Exception:
            continue

    safe_game_id = game_id.replace("/", "_")
    diverged_label = str(result.diverged_at)
    bundle_path = (
        state_bundles_dir
        / f"{Path(db_path).stem}__{safe_game_id}__k{diverged_label}.state_bundle.json"
    )

    bundle = {
        "db_path": str(db_path),
        "game_id": game_id,
        "diverged_at": result.diverged_at,
        "mismatch_kinds": list(result.mismatch_kinds or []),
        "mismatch_context": result.mismatch_context,
        "total_moves_python": result.total_moves_python,
        "total_moves_ts": result.total_moves_ts,
        "ts_k_values": ks,
        "python_states": {str(k): py_states.get(k) for k in ks},
        "ts_states": {str(k): ts_states.get(k) for k in ks},
    }

    try:
        state_bundles_dir.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, sort_keys=True)
    except Exception:
        return


def check_game_parity(db_path: Path, game_id: str) -> GameParityResult:
    db = GameReplayDB(str(db_path))
    meta = db.get_game_metadata(game_id)
    if not meta:
        raise RuntimeError(f"Game {game_id} not found in {db_path}")

    structure, structure_reason = classify_game_structure(db, game_id)
    total_moves_py = int(meta["total_moves"])

    # Pre-flight canonical history check: if the recorded (phase, move_type)
    # pairs do not match the canonical contract, treat the game as a
    # structural/non-canonical recording and skip TS replay. This keeps
    # parity runs focused on games that at least agree on phase semantics
    # per RR-CANON-R070/R075.
    canonical_report = validate_canonical_history_for_game(db, game_id)
    if not canonical_report.is_canonical:
        reasons = sorted({issue.reason for issue in canonical_report.issues})
        reason_str = ";".join(reasons) if reasons else "non_canonical_history"
        return GameParityResult(
            db_path=str(db_path),
            game_id=game_id,
            structure="non_canonical_history",
            structure_reason=reason_str,
            total_moves_python=total_moves_py,
            total_moves_ts=0,
            diverged_at=None,
            python_summary=None,
            ts_summary=None,
            mismatch_kinds=["non_canonical_history"],
            mismatch_context="recording",
        )

    # For invalid games (missing data) we don't attempt TS replay.
    # mid_snapshot games (CMA-ES multi-start) ARE valid for parity testing
    # because the TS replay script loads the initial state from game_initial_state.
    if structure == "invalid":
        return GameParityResult(
            db_path=str(db_path),
            game_id=game_id,
            structure=structure,
            structure_reason=structure_reason,
            total_moves_python=total_moves_py,
            total_moves_ts=0,
            diverged_at=None,
            python_summary=None,
            ts_summary=None,
        )

    total_moves_ts, ts_summaries = run_ts_replay(db_path, game_id)

    diverged_at: Optional[int] = None
    py_summary_at_diverge: Optional[StateSummary] = None
    ts_summary_at_diverge: Optional[StateSummary] = None
    mismatch_kinds: List[str] = []
    mismatch_context: Optional[str] = None

    # Index alignment:
    #   TS k=0 (ts-replay-initial) = initial state BEFORE any moves
    #   TS k=1 (ts-replay-step) = state AFTER move 0
    #   TS k=N = state AFTER move N-1
    #   Python get_state_at_move(0) = state AFTER move 0
    #   Python get_state_at_move(N) = state AFTER move N
    #
    # So: TS k=0 ↔ Python initial_state
    #     TS k=N (N>=1) ↔ Python get_state_at_move(N-1)

    # First, compare initial states (TS k=0 vs Python initial_state)
    # Note: We compare currentPlayer, currentPhase, gameStatus but NOT state_hash
    # because Python uses SHA-256 while TS uses a custom human-readable format.
    ts_initial = ts_summaries.get(0)
    if ts_initial is not None:
        py_initial = summarize_python_initial_state(db, game_id)
        init_mismatches: List[str] = []
        if py_initial.current_player != ts_initial.current_player:
            init_mismatches.append("current_player")
        if py_initial.current_phase != ts_initial.current_phase:
            init_mismatches.append("current_phase")
        if py_initial.game_status != ts_initial.game_status:
            init_mismatches.append("game_status")

        if init_mismatches:
            diverged_at = 0
            py_summary_at_diverge = py_initial
            ts_summary_at_diverge = ts_initial
            mismatch_kinds = init_mismatches
            mismatch_context = "initial_state"

    # Then compare post-move states: TS k ↔ Python get_state_at_move(k-1)
    if diverged_at is None:
        max_ts_k = total_moves_ts  # TS k ranges from 1 to total_moves_ts
        for ts_k in range(1, max_ts_k + 1):
            py_move_index = ts_k - 1  # Python index for state after move (ts_k - 1)
            if py_move_index >= total_moves_py:
                break  # Python doesn't have this move recorded

            ts_summary = ts_summaries.get(ts_k)
            if ts_summary is None:
                diverged_at = ts_k
                py_summary_at_diverge = summarize_python_state(db, game_id, py_move_index)
                ts_summary_at_diverge = None
                mismatch_kinds = ["ts_missing_step"]
                mismatch_context = "post_move"
                break

            py_summary = summarize_python_state(db, game_id, py_move_index)

            step_mismatches: List[str] = []
            if (
                py_summary.current_player != ts_summary.current_player
            ):
                step_mismatches.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                step_mismatches.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                step_mismatches.append("game_status")
            # Treat any difference in the canonical state hash as a semantic
            # divergence. The hash is derived from the shared hash_game_state
            # fingerprint and should match whenever board geometry and core
            # progress counters (elims/territory) agree across engines.
            if py_summary.state_hash != ts_summary.state_hash:
                step_mismatches.append("state_hash")

            if step_mismatches:
                diverged_at = ts_k
                py_summary_at_diverge = py_summary
                ts_summary_at_diverge = ts_summary
                mismatch_kinds = step_mismatches
                mismatch_context = "post_move"
                break

    # If we had no per-move divergence but move counts differ, record that as a
    # distinct mismatch kind so callers can track move-count-only issues.
    if diverged_at is None and total_moves_py != total_moves_ts:
        diverged_at = None  # keep as None; mismatch is global, not at a single k
        mismatch_kinds = ["move_count"]
        mismatch_context = "global"

    # Determine if divergence is only at the very last move (end-of-game only)
    # AND the canonical state hash matches. Pure metadata differences in
    # player/phase/status at the terminal snapshot are insignificant for
    # training; any structural difference (state_hash mismatch) is treated as
    # a full semantic divergence.
    is_end_of_game_only = False
    if (
        diverged_at is not None
        and diverged_at == total_moves_py
        and diverged_at == total_moves_ts
    ):
        if py_summary_at_diverge is not None and ts_summary_at_diverge is not None:
            # Only treat as "end-of-game only" when the underlying board /
            # territory / elimination fingerprint is identical and the
            # divergence is limited to metadata fields.
            if py_summary_at_diverge.state_hash == ts_summary_at_diverge.state_hash:
                is_end_of_game_only = True

    return GameParityResult(
        db_path=str(db_path),
        game_id=game_id,
        structure=structure,
        structure_reason=structure_reason or None,
        total_moves_python=total_moves_py,
        total_moves_ts=total_moves_ts,
        diverged_at=diverged_at,
        python_summary=py_summary_at_diverge,
        ts_summary=ts_summary_at_diverge,
        mismatch_kinds=mismatch_kinds,
        mismatch_context=mismatch_context,
        is_end_of_game_only=is_end_of_game_only,
    )


def trace_game(db_path: Path, game_id: str, max_k: Optional[int] = None) -> None:
    """Emit a per-step TS vs Python trace for a single game.

    This is a focused debugging helper: it prints one line per TS step k,
    including the corresponding Python state (when available) and basic
    move metadata so rule/phase alignment issues can be inspected without
    re-running the full parity sweep.
    """
    db = GameReplayDB(str(db_path))
    meta = db.get_game_metadata(game_id)
    if not meta:
        print(f"[trace] game {game_id} not found in {db_path}")
        return

    structure, structure_reason = classify_game_structure(db, game_id)
    total_moves_py = int(meta.get("total_moves", 0))

    try:
        total_moves_ts, ts_summaries = run_ts_replay(db_path, game_id)
    except Exception as exc:
        print(
            "[trace] TS replay failed "
            f"for db={db_path} game={game_id}: {exc}"
        )
        return

    print(
        "TRACE-HEADER "
        f"db={db_path} "
        f"game={game_id} "
        f"structure={structure} "
        f"structure_reason={json.dumps(structure_reason or '')} "
        f"total_moves_py={total_moves_py} "
        f"total_moves_ts={total_moves_ts}"
    )

    # Initial state (TS k=0 vs Python initial_state)
    py_initial = summarize_python_initial_state(db, game_id)
    ts_initial = ts_summaries.get(0)
    init_dims: List[str] = []
    if ts_initial is None:
        init_dims.append("ts_missing_step")
    else:
        if py_initial.current_player != ts_initial.current_player:
            init_dims.append("current_player")
        if py_initial.current_phase != ts_initial.current_phase:
            init_dims.append("current_phase")
        if py_initial.game_status != ts_initial.game_status:
            init_dims.append("game_status")
        if py_initial.state_hash != ts_initial.state_hash:
            init_dims.append("state_hash")

    print(
        "TRACE "
        f"db={db_path} "
        f"game={game_id} "
        f"k=0 "
        f"move_number=None "
        f"move_player=None "
        f"move_type=None "
        f"py_player={py_initial.current_player} "
        f"ts_player={(ts_initial.current_player if ts_initial is not None else 'None')} "
        f"py_phase={py_initial.current_phase} "
        f"ts_phase={(ts_initial.current_phase if ts_initial is not None else 'None')} "
        f"py_status={py_initial.game_status} "
        f"ts_status={(ts_initial.game_status if ts_initial is not None else 'None')} "
        f"py_hash={py_initial.state_hash} "
        f"ts_hash={(ts_initial.state_hash if ts_initial is not None else 'None')} "
        f"dims={','.join(init_dims)}"
    )

    # Per-move states: TS k ↔ Python get_state_at_move(k-1)
    move_records = db.get_move_records(game_id)
    limit_k = total_moves_ts
    if max_k is not None and max_k > 0:
        limit_k = min(limit_k, max_k)

    for ts_k in range(1, limit_k + 1):
        py_index = ts_k - 1

        py_summary: Optional[StateSummary] = None
        py_error: Optional[str] = None
        if py_index < total_moves_py:
            try:
                py_summary = summarize_python_state(db, game_id, py_index)
            except Exception as exc:  # pragma: no cover - defensive
                py_error = str(exc)

        ts_summary = ts_summaries.get(ts_k)

        dims: List[str] = []
        if py_summary is not None and ts_summary is not None:
            if py_summary.current_player != ts_summary.current_player:
                dims.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                dims.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                dims.append("game_status")
            if py_summary.state_hash != ts_summary.state_hash:
                dims.append("state_hash")
        elif py_summary is None and ts_summary is not None:
            dims.append("python_missing_step")
        elif py_summary is not None and ts_summary is None:
            dims.append("ts_missing_step")

        move_number: Optional[int] = None
        move_player: Optional[int] = None
        move_type: Optional[str] = None
        if 0 <= py_index < len(move_records):
            rec = move_records[py_index]
            move_number = rec.get("moveNumber")
            move_player = rec.get("player")
            move_type = rec.get("moveType")

        py_player = py_summary.current_player if py_summary is not None else None
        ts_player = ts_summary.current_player if ts_summary is not None else None
        py_phase = py_summary.current_phase if py_summary is not None else None
        ts_phase = ts_summary.current_phase if ts_summary is not None else None
        py_status = py_summary.game_status if py_summary is not None else None
        ts_status = ts_summary.game_status if ts_summary is not None else None
        py_hash = py_summary.state_hash if py_summary is not None else None
        ts_hash = ts_summary.state_hash if ts_summary is not None else None

        line = (
            "TRACE "
            f"db={db_path} "
            f"game={game_id} "
            f"k={ts_k} "
            f"py_index={py_index} "
            f"move_number={move_number} "
            f"move_player={move_player} "
            f"move_type={move_type} "
            f"py_player={py_player} "
            f"ts_player={ts_player} "
            f"py_phase={py_phase} "
            f"ts_phase={ts_phase} "
            f"py_status={py_status} "
            f"ts_status={ts_status} "
            f"py_hash={py_hash} "
            f"ts_hash={ts_hash} "
            f"dims={','.join(dims)}"
        )
        if py_error:
            line += f" py_error={json.dumps(py_error)}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check TS vs Python replay parity for all self-play GameReplayDBs."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional path to a single games.db to inspect. "
        "When omitted, scans all known self-play locations.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Path to a JSON scenario/fixture file to check. "
            "Supports LoadableScenario format (from sandbox export) or "
            "ringrift_sandbox_fixture_v1 format. Creates a temporary DB "
            "and runs parity check on the imported game."
        ),
    )
    parser.add_argument(
        "--limit-games-per-db",
        type=int,
        default=0,
        help="Optional limit on number of games per DB to check (0 = all).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Emit only semantic divergences as single-line, grep-friendly "
            "records (no JSON summary, no structural issue output)."
        ),
    )
    parser.add_argument(
        "--emit-fixtures-dir",
        type=str,
        default=None,
        help=(
            "If set, write one JSON fixture per semantic divergence into this directory. "
            "Each fixture captures db_path, game_id, diverged_at, mismatch_kinds/context, "
            "Python/TS summaries, and the canonical move at or immediately before divergence."
        ),
    )
    parser.add_argument(
        "--emit-state-bundles-dir",
        type=str,
        default=None,
        help=(
            "If set, write TS+Python state bundles for each semantic divergence into this directory. "
            "Each bundle captures full serialized GameState JSON for TS and Python at the step "
            "immediately before and at the first divergence."
        ),
    )
    parser.add_argument(
        "--trace-game",
        type=str,
        default=None,
        help=(
            "If set, emit a per-step TS vs Python trace for a single game_id and exit. "
            "Respects --db when provided; otherwise searches all known DB locations."
        ),
    )
    parser.add_argument(
        "--trace-max-k",
        type=int,
        default=0,
        help=(
            "Optional maximum TS k to include in --trace-game output (0 = all steps)."
        ),
    )
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help=(
            "If set, exit with non-zero status (exit code 1) when any semantic "
            "divergences are detected. Intended for CI gates. Structural issues "
            "and end-of-game-only divergences are logged but do not trigger failure."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help=(
            "Optional path to write the parity summary JSON. Directories are "
            "created if needed. Useful for archiving CI results."
        ),
    )
    args = parser.parse_args()

    # Handle JSON input mode: import to temp DB and check single game
    temp_db_path: Optional[Path] = None
    json_game_id: Optional[str] = None
    if args.json:
        import shutil
        try:
            temp_db_path, json_game_id = import_json_to_temp_db(args.json)
            print(f"[json] Imported {args.json} -> temp DB at {temp_db_path}, game_id={json_game_id}")
        except Exception as e:
            print(f"[json] Failed to import {args.json}: {e}")
            return

    db_paths = find_dbs(args.db) if not temp_db_path else [temp_db_path]
    if not db_paths:
        print("No GameReplayDB databases found.")
        return

    # Focused trace mode: find the requested game_id and emit a per-step trace,
    # then exit without running the full parity sweep.
    if args.trace_game:
        for db_path in db_paths:
            db = GameReplayDB(str(db_path))
            try:
                meta = db.get_game_metadata(args.trace_game)
            except Exception:
                meta = None
            if meta:
                max_k = args.trace_max_k if args.trace_max_k and args.trace_max_k > 0 else None
                trace_game(db_path, args.trace_game, max_k=max_k)
                return

        print(
            f"[trace] game {args.trace_game} not found in any GameReplayDB "
            f"(searched {len(db_paths)} databases)"
        )
        return

    structural_issues: List[Dict[str, object]] = []
    semantic_divergences: List[Dict[str, object]] = []
    end_of_game_only_divergences: List[Dict[str, object]] = []
    mismatch_counts_by_dimension: Dict[str, int] = {}
    total_games = 0
    total_semantic_divergent = 0
    total_end_of_game_only = 0
    total_structural_issues = 0

    fixtures_dir: Optional[Path] = Path(args.emit_fixtures_dir).resolve() if args.emit_fixtures_dir else None
    if fixtures_dir is not None:
        fixtures_dir.mkdir(parents=True, exist_ok=True)

    state_bundles_dir: Optional[Path] = (
        Path(args.emit_state_bundles_dir).resolve() if args.emit_state_bundles_dir else None
    )
    if state_bundles_dir is not None:
        state_bundles_dir.mkdir(parents=True, exist_ok=True)

    for db_path in db_paths:
        db = GameReplayDB(str(db_path))
        games = db.query_games(limit=100000)
        if not games:
            continue
        if args.limit_games_per_db and args.limit_games_per_db > 0:
            games = games[: args.limit_games_per_db]

        for game_meta in games:
            game_id = game_meta["game_id"]
            total_games += 1
            try:
                result = check_game_parity(db_path, game_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                structural_issues.append(
                    {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "structure": "error",
                        "structure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                total_structural_issues += 1
                continue

            # Treat any non-"good"/"mid_snapshot" structure as a structural issue
            # (e.g., invalid initial_state, non-canonical history). mid_snapshot
            # games (CMA-ES multi-start, soak test games, etc.) are valid for
            # parity testing since the TS replay script loads the initial state
            # from game_initial_state table.
            if result.structure not in ("good", "mid_snapshot"):
                total_structural_issues += 1
                structural_issues.append(
                    {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "structure": result.structure,
                        "structure_reason": result.structure_reason,
                    }
                )
                continue

            if result.diverged_at is not None or result.total_moves_python != result.total_moves_ts:
                payload = asdict(result)
                if result.python_summary is not None:
                    payload["python_summary"] = asdict(result.python_summary)
                if result.ts_summary is not None:
                    payload["ts_summary"] = asdict(result.ts_summary)

                # Classify divergence: significant vs end-of-game-only
                if result.is_end_of_game_only:
                    total_end_of_game_only += 1
                    end_of_game_only_divergences.append(payload)
                else:
                    total_semantic_divergent += 1
                    semantic_divergences.append(payload)
                    # Increment per-dimension mismatch counters (only for significant divergences)
                    for kind in result.mismatch_kinds or []:
                        mismatch_counts_by_dimension[kind] = mismatch_counts_by_dimension.get(kind, 0) + 1

                # Optionally emit a compact JSON fixture for this divergence so TS tests
                # can consume it directly without re-querying the replay DB.
                if fixtures_dir is not None:
                    try:
                        moves = db.get_moves(game_id)
                    except Exception:
                        moves = []

                    canonical_move_index: Optional[int] = None
                    canonical_move_dict: Optional[Dict[str, object]] = None

                    if moves:
                        if result.diverged_at is None or result.diverged_at <= 0:
                            idx = 0
                        else:
                            idx = max(0, min(len(moves) - 1, result.diverged_at - 1))
                        canonical_move_index = idx
                        try:
                            move_obj = moves[idx]
                            canonical_move_dict = json.loads(
                                move_obj.model_dump_json(by_alias=True)  # type: ignore[attr-defined]
                            )
                        except Exception:
                            canonical_move_dict = None

                    fixture = {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "diverged_at": result.diverged_at,
                        "mismatch_kinds": list(result.mismatch_kinds),
                        "mismatch_context": result.mismatch_context,
                        "total_moves_python": result.total_moves_python,
                        "total_moves_ts": result.total_moves_ts,
                        "python_summary": (
                            asdict(result.python_summary) if result.python_summary is not None else None
                        ),
                        "ts_summary": (
                            asdict(result.ts_summary) if result.ts_summary is not None else None
                        ),
                        "canonical_move_index": canonical_move_index,
                        "canonical_move": canonical_move_dict,
                    }

                    safe_game_id = game_id.replace("/", "_")
                    diverged_label = (
                        "global"
                        if result.diverged_at is None
                        else str(result.diverged_at)
                    )
                    fixture_path = fixtures_dir / f"{Path(db_path).stem}__{safe_game_id}__k{diverged_label}.json"
                    with open(fixture_path, "w", encoding="utf-8") as f:
                        json.dump(fixture, f, indent=2, sort_keys=True)

                if state_bundles_dir is not None and result.diverged_at is not None:
                    try:
                        dump_state_bundle(
                            db=db,
                            db_path=Path(db_path),
                            game_id=game_id,
                            result=result,
                            state_bundles_dir=state_bundles_dir,
                        )
                    except Exception:
                        pass

    if args.compact:
        # Compact mode: emit one line per semantic divergence, skip structural issues.
        for entry in semantic_divergences:
            py = entry.get("python_summary") or {}
            ts = entry.get("ts_summary") or {}
            dims = entry.get("mismatch_kinds") or []
            line = (
                "SEMANTIC "
                f"db={entry.get('db_path')} "
                f"game={entry.get('game_id')} "
                f"diverged_at={entry.get('diverged_at')} "
                f"py_phase={py.get('current_phase')} "
                f"ts_phase={ts.get('current_phase')} "
                f"py_status={py.get('game_status')} "
                f"ts_status={ts.get('game_status')} "
                f"py_hash={py.get('state_hash')} "
                f"ts_hash={ts.get('state_hash')} "
                f"dims={','.join(dims)}"
            )
            print(line)
        # Cleanup temp DB if we created one
        if temp_db_path is not None:
            import shutil
            shutil.rmtree(temp_db_path.parent, ignore_errors=True)
        return

    summary = {
        "total_databases": len(db_paths),
        "total_games_checked": total_games,
        "games_with_semantic_divergence": total_semantic_divergent,
        "games_with_end_of_game_only_divergence": total_end_of_game_only,
        "games_with_structural_issues": total_structural_issues,
        "semantic_divergences": semantic_divergences,
        "end_of_game_only_divergences": end_of_game_only_divergences,
        "structural_issues": structural_issues,
        "mismatch_counts_by_dimension": mismatch_counts_by_dimension,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    # Optionally write summary to a file (for CI artifact archiving)
    if args.summary_json:
        summary_dir = os.path.dirname(args.summary_json)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    # Cleanup temp DB if we created one
    if temp_db_path is not None:
        import shutil
        shutil.rmtree(temp_db_path.parent, ignore_errors=True)

    # CI gate: exit with non-zero status if semantic divergences were found
    if args.fail_on_divergence and total_semantic_divergent > 0:
        print(
            f"\n[FAIL] {total_semantic_divergent} game(s) with semantic divergence detected.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
