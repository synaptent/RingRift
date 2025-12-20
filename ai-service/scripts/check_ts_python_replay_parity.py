"""TS vs Python replay parity checker for GameReplayDB databases.

This script walks GameReplayDB databases and, for each game, compares Python's
GameReplayDB.get_state_at_move against the TypeScript ClientSandboxEngine
replay path driven via the scripts/selfplay-db-ts-replay.ts harness.

Two orthogonal controls define how it behaves:

1) **View semantics** (``--view post_move|post_bridge``)

   By default it operates in a **post_move** view:

     - Python: GameReplayDB.get_state_at_move(game_id, n) → post_move[n]
     - TS: ``ts-replay-step`` event at k = n + 1 (state immediately after
       applying DB move n, before any synthesized bookkeeping).

   An optional ``--view post_bridge`` diagnostic mode is also provided:

     - TS: ``ts-replay-db-move-complete`` events (state after any synthesized
       bookkeeping needed to bridge from one recorded DB move to the next).
     - Python: still uses GameReplayDB.get_state_at_move(game_id, n) for now.
       This mode is intended for tooling and future debugging; canonical
       gating and training continue to rely on post_move semantics.

2) **Parity mode** (``--mode canonical|legacy``)

   - ``canonical`` (default):
       * Forces ``--view post_move`` (any other combination is rejected).
       * Treats any structural issue, non-canonical history, or semantic
         divergence as a **hard failure** for the database.
       * The JSON summary includes ``passed_canonical_parity_gate: true`` only
         when:
             - games_with_structural_issues == 0
             - games_with_non_canonical_history == 0
             - games_with_semantic_divergence == 0
             - total_games_checked > 0
       * The CLI exit status is non-zero when the canonical parity gate fails.

   - ``legacy``:
       * Can be combined with ``--view post_move`` or ``--view post_bridge``.
       * Allows non-canonical histories and structural issues but clearly
         records them in the JSON summary.
       * Does **not** enforce a canonical gate by default; exit status remains
         zero unless ``--fail-on-divergence`` is supplied (semantic-only).

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
import sys
import tempfile
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.lib.paths import AI_SERVICE_ROOT, REPO_ROOT
# Ensure `app.*` imports resolve when invoked from repo root.
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.db.game_replay import GameReplayDB, _compute_state_hash
from app.training.generate_data import create_initial_state
from app.game_engine import GameEngine
from app.models import BoardType
from app.rules.serialization import serialize_game_state
from app.rules.history_validation import validate_canonical_history_for_game
from app.rules import global_actions as ga


@dataclass
class StateSummary:
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str
    # Optional per-state ANM classification for the active player. This is
    # populated for both Python and TS summaries whenever the underlying
    # harness exposes an ANM flag.
    is_anm: Optional[bool] = None


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
    # Optional free-form context, such as "initial_state" vs "post_move" / "post_bridge"
    mismatch_context: Optional[str] = None
    # True when divergence occurs only at the very last move (end-of-game metadata
    # differences). These are insignificant for training purposes as no AI decisions
    # are made based on the divergent state.
    is_end_of_game_only: bool = False


@dataclass
class TsEventMetadata:
    """Metadata about a TS replay event used for parity comparison."""

    ts_k: int
    db_move_index: Optional[int]
    view: Optional[str]
    event_kind: str


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


def find_dbs(explicit_db: Optional[str] = None) -> List[Path]:
    """Find GameReplayDB files to inspect."""
    root = REPO_ROOT
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


def _load_game_id_filter(
    *,
    include_game_ids: Optional[List[str]],
    include_game_ids_file: Optional[str],
) -> set[str]:
    """Return a set of game_ids to include (empty set means include all)."""
    game_ids: set[str] = set()
    for raw in include_game_ids or []:
        token = str(raw or "").strip()
        if token:
            game_ids.add(token)

    if include_game_ids_file:
        path = Path(include_game_ids_file).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"--include-game-ids-file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("game_ids"), list):
                values = payload.get("game_ids") or []
            elif isinstance(payload, list):
                values = payload
            else:
                raise ValueError(
                    f"--include-game-ids-file JSON must be a list of game_ids or {{'game_ids': [...]}}: {path}"
                )
            for value in values:
                token = str(value or "").strip()
                if token:
                    game_ids.add(token)
        else:
            for line in text.splitlines():
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                game_ids.add(token)

    return game_ids


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
    cur.executescript(
        """
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
    """
    )

    # Insert game record
    now = datetime.now().isoformat()
    cur.execute(
        """
        INSERT INTO games (game_id, board_type, num_players, created_at, game_status,
                          total_moves, total_turns, source, schema_version)
        VALUES (?, ?, ?, ?, 'completed', ?, ?, 'json_import', 6)
    """,
        (game_id, board_type, num_players, now, len(moves), len(moves)),
    )

    # Insert player records (GameRecord format has players, others generate defaults)
    if players:
        for p in players:
            player_num = p.get("playerNumber", 1)
            player_type = p.get("playerType", "ai")
            ai_type = p.get("aiType")
            ai_difficulty = p.get("aiDifficulty")
            cur.execute(
                """
                INSERT INTO game_players (game_id, player_number, player_type, ai_type, ai_difficulty)
                VALUES (?, ?, ?, ?, ?)
            """,
                (game_id, player_num, player_type, ai_type, ai_difficulty),
            )
    else:
        # Create default player entries for formats without explicit players
        for i in range(num_players):
            cur.execute(
                """
                INSERT INTO game_players (game_id, player_number, player_type)
                VALUES (?, ?, 'ai')
            """,
                (game_id, i + 1),
            )

    # Insert initial state only if we have one (GameRecord format has empty state_json)
    if state_json:
        cur.execute(
            """
            INSERT INTO game_initial_state (game_id, initial_state_json, compressed)
            VALUES (?, ?, 0)
        """,
            (game_id, json.dumps(state_json)),
        )

    # Insert moves
    for i, move in enumerate(moves):
        # Extract move metadata with fallbacks
        move_type = move.get("type") or move.get("actionType") or move.get("action_type") or "unknown"
        player = move.get("player") or move.get("playerNumber") or 1
        phase = move.get("phase") or move.get("currentPhase") or "unknown"
        turn = move.get("turn") or move.get("turnNumber") or i

        cur.execute(
            """
            INSERT INTO game_moves (game_id, move_number, turn_number, player, phase, move_type, move_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (game_id, i, turn, player, phase, move_type, json.dumps(move)),
        )

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
        current_phase=state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
        is_anm=ga.is_anm_state(state),
    )


def replay_python_post_move_summaries(
    db: GameReplayDB,
    game_id: str,
    *,
    limit_moves: Optional[int] = None,
) -> Dict[int, StateSummary]:
    """Replay a game once and return per-move post-move summaries.

    This is an O(N) alternative to calling GameReplayDB.get_state_at_move(k)
    repeatedly (which is O(N^2) when implemented as "replay from initial state").
    """
    progress_every = int(os.environ.get("RINGRIFT_PARITY_PROGRESS_EVERY", "250") or "250")
    if progress_every > 0:
        print(f"[py-replay] start {game_id}", file=sys.stderr, flush=True)

    state = db.get_initial_state(game_id)
    if state is None:
        metadata = db.get_game_metadata(game_id)
        if metadata is None:
            raise RuntimeError(f"Python get_initial_state returned None and no game metadata for {game_id}")

        board_type_str = metadata.get("board_type", "square8")
        num_players = metadata.get("num_players", 2)
        board_type = _parse_board_type(board_type_str)
        state = create_initial_state(board_type=board_type, num_players=num_players)

    moves = db.get_moves(game_id)
    if limit_moves is not None and limit_moves >= 0:
        moves = moves[:limit_moves]
    total_moves = len(moves)

    summaries: Dict[int, StateSummary] = {}
    working_state = state

    for idx, move in enumerate(moves):
        working_state = GameEngine.apply_move(working_state, move, trace_mode=True)

        summaries[idx] = StateSummary(
            move_index=idx,
            current_player=working_state.current_player,
            current_phase=working_state.current_phase.value
            if hasattr(working_state.current_phase, "value")
            else str(working_state.current_phase),
            game_status=_canonicalize_status(
                working_state.game_status.value
                if hasattr(working_state.game_status, "value")
                else str(working_state.game_status)
            ),
            state_hash=_compute_state_hash(working_state),
            is_anm=ga.is_anm_state(working_state),
        )

        if progress_every > 0 and total_moves > 0 and (idx + 1) % progress_every == 0:
            print(f"[py-replay] {game_id} n={idx + 1}/{total_moves}", file=sys.stderr, flush=True)

    return summaries


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
        current_phase=state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        game_status=_canonicalize_status(
            state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status)
        ),
        state_hash=_compute_state_hash(state),
        is_anm=ga.is_anm_state(state),
    )


def classify_game_structure(db: GameReplayDB, game_id: str) -> Tuple[str, Optional[str]]:
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
                raise RuntimeError(f"Python get_initial_state returned None and no game metadata for {game_id}")
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


def run_ts_replay(
    db_path: Path,
    game_id: str,
    view_mode: str = "post_move",
) -> Tuple[int, Dict[int, StateSummary], Dict[int, TsEventMetadata]]:
    """Invoke the TS harness and parse its per-move summaries.

    Returns:
      (total_moves_reported_by_ts,
       mapping from k -> summary for the selected view,
       mapping from k -> TsEventMetadata for the selected view)

    The TS harness emits several event types:
      - ``ts-replay-initial``: Initial state (k=0) before any moves.
      - ``ts-replay-step``: State immediately after applying a DB move
        (TS "post_move" view).
      - ``ts-replay-bridge``: State after applying a synthesized bookkeeping
        move (ignored for parity).
      - ``ts-replay-db-move-complete``: State after all synthesized bridges
        associated with a DB move (TS "post_bridge" view).
      - ``ts-replay-final``: Final state summary.

    In ``view_mode == 'post_move'`` (canonical default) we use
    ``ts-replay-step`` events, which represent the state IMMEDIATELY after
    applying DB move ``n`` without any additional synthesized bookkeeping.
    This matches Python's ``get_state_at_move(n)`` semantics.

    In ``view_mode == 'post_bridge'`` (diagnostic) we instead use
    ``ts-replay-db-move-complete`` events, which include any synthesized
    bookkeeping needed to bridge from one recorded DB move to the next.
    Python remains on ``get_state_at_move(n)`` (post_move) semantics for now;
    this mode is intended for tooling and detailed debugging only.
    """
    root = REPO_ROOT
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
    # Default to minimal TS replay output for parity (skip FSM traces/bridge logs).
    env.setdefault("RINGRIFT_TS_REPLAY_MINIMAL", "1")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    progress_every = int(os.environ.get("RINGRIFT_PARITY_PROGRESS_EVERY", "250") or "250")
    if progress_every > 0:
        print(f"[ts-replay] start {game_id} ({db_path.name})", file=sys.stderr, flush=True)

    total_ts_moves = 0

    # Per-k summaries for each TS view.
    initial_summary: Optional[StateSummary] = None
    post_move_summaries: Dict[int, StateSummary] = {}
    post_bridge_summaries: Dict[int, StateSummary] = {}

    # Per-k metadata for each TS view.
    meta_post_move: Dict[int, TsEventMetadata] = {}
    meta_post_bridge: Dict[int, TsEventMetadata] = {}

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
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
            is_anm_ts = summary.get("is_anm") if isinstance(summary, dict) else None
            initial_summary = StateSummary(
                move_index=0,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
                is_anm=is_anm_ts,
            )
            meta = TsEventMetadata(
                ts_k=0,
                db_move_index=None,
                view=(summary.get("view") if isinstance(summary, dict) else None) or "initial",
                event_kind=kind,
            )
            meta_post_move[0] = meta
            meta_post_bridge[0] = meta
        elif kind == "ts-replay-step":
            # Use ts-replay-step for canonical post_move comparison: this is
            # the state immediately after applying the recorded DB move,
            # BEFORE any synthesized bookkeeping moves.
            k_raw = payload.get("k", 0)
            try:
                k = int(k_raw)
            except Exception:
                continue
            summary = payload.get("summary") or {}
            is_anm_ts = summary.get("is_anm") if isinstance(summary, dict) else None
            post_move_summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
                is_anm=is_anm_ts,
            )

            db_move_index_raw = payload.get("db_move_index")
            try:
                db_move_index = int(db_move_index_raw) if db_move_index_raw is not None else None
            except Exception:
                db_move_index = None

            view = summary.get("view") if isinstance(summary, dict) else None
            meta_post_move[k] = TsEventMetadata(
                ts_k=k,
                db_move_index=db_move_index,
                view=view,
                event_kind=kind,
            )

            if progress_every > 0 and total_ts_moves > 0 and k % progress_every == 0:
                print(f"[ts-replay] {game_id} k={k}/{total_ts_moves}", file=sys.stderr, flush=True)

            # Cross-check the explicit db_move_index emitted by the TS harness
            # against the implicit k->move_index contract (k = db_move_index + 1).
            #
            # If they disagree we treat this as a structural anomaly in the TS
            # trace but do NOT abort parity. The canonical comparison continues
            # to rely on k (TS step index) ↔ Python get_state_at_move(k-1).
            if db_move_index is not None:
                expected = k - 1
                if db_move_index != expected:
                    msg = (
                        "TS replay emitted inconsistent db_move_index="
                        f"{db_move_index} for k={k} (expected {expected}) "
                        f"in {db_path} / {game_id}"
                    )
                    # Emit to stderr for diagnostics without changing parity
                    # behaviour for canonical runs.
                    print(f"[ts-replay-warning] {msg}", file=sys.stderr)
        elif kind == "ts-replay-db-move-complete":
            # Post-bridge state: state after closing out canonical DB move
            # db_move_index (including any synthesized bookkeeping moves).
            db_move_index_raw = payload.get("db_move_index")
            try:
                db_move_index_int = int(db_move_index_raw)
            except Exception:
                continue

            # For db_move_index == n this corresponds to TS step k = n + 1,
            # i.e. the same indexing used for ts-replay-step.
            k = db_move_index_int + 1
            summary = payload.get("summary") or {}
            is_anm_ts = summary.get("is_anm") if isinstance(summary, dict) else None
            post_bridge_summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=_canonicalize_status(summary.get("gameStatus")),
                state_hash=summary.get("stateHash"),
                is_anm=is_anm_ts,
            )

            view = summary.get("view") if isinstance(summary, dict) else None
            meta_post_bridge[k] = TsEventMetadata(
                ts_k=k,
                db_move_index=db_move_index_int,
                view=view,
                event_kind=kind,
            )
        elif kind == "ts-replay-game-ended":
            # Early victory detection - TS terminated before processing all DB moves.
            # Update total_ts_moves to reflect the actual number of moves applied.
            applied_moves = payload.get("appliedMoves")
            if applied_moves is not None:
                total_ts_moves = int(applied_moves)
            # Capture the final summary at the termination point for comparing
            # against Python's state at the same move index.
            summary = payload.get("summary") or {}
            k = total_ts_moves
            if k > 0 and k not in post_move_summaries:
                post_move_summaries[k] = StateSummary(
                    move_index=k,
                    current_player=summary.get("currentPlayer"),
                    current_phase=summary.get("currentPhase"),
                    game_status=_canonicalize_status(summary.get("gameStatus")),
                    state_hash=summary.get("stateHash"),
                    is_anm=summary.get("is_anm") if isinstance(summary, dict) else None,
                )
        else:
            # ts-replay-bridge / ts-replay-final and any other events are
            # ignored for parity; they are still present in stdout for
            # diagnostics but not used for state comparison.
            continue

    stderr = proc.stderr.read() if proc.stderr else ""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay harness failed for {db_path} / {game_id} with code {proc.returncode}:\n"
            f"STDERR:\n{stderr}"
        )

    # Build the view-specific summaries and metadata.
    summaries: Dict[int, StateSummary] = {}
    if initial_summary is not None:
        summaries[0] = initial_summary

    if view_mode == "post_bridge":
        summaries.update(post_bridge_summaries)
        selected_meta = meta_post_bridge
    else:
        # Default / canonical behaviour.
        summaries.update(post_move_summaries)
        selected_meta = meta_post_move

    # Ensure we always have at least some metadata entry for k=0 so state
    # bundles can describe the initial TS snapshot even if the harness did
    # not emit an explicit initial view tag.
    if 0 not in selected_meta:
        selected_meta[0] = TsEventMetadata(
            ts_k=0,
            db_move_index=None,
            view="initial",
            event_kind="ts-replay-initial",
        )

    return total_ts_moves, summaries, selected_meta


def _dump_ts_states_for_ks(
    db_path: Path,
    game_id: str,
    ks: List[int],
    dump_dir: Path,
) -> None:
    """Invoke the TS replay harness to dump TS GameState JSON at the requested k values."""
    if not ks:
        return

    root = REPO_ROOT
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
    *,
    view_mode: str,
    ts_event_metadata: Dict[int, TsEventMetadata] | None = None,
) -> None:
    """Emit a rich TS+Python state bundle around the first divergence for faster debugging.

    The bundle records the selected TS view semantics (``view_mode``) and, for
    each included TS k, basic metadata about the TS replay event that produced
    the state used for comparison (event_kind, db_move_index, view).
    """
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
    meta_for_ks: Dict[str, Dict[str, object]] = {}

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

    if ts_event_metadata is not None:
        for ts_k in ks:
            meta = ts_event_metadata.get(ts_k)
            if meta is None:
                continue
            meta_for_ks[str(ts_k)] = {
                "ts_k": meta.ts_k,
                "db_move_index": meta.db_move_index,
                "view": meta.view,
                "event_kind": meta.event_kind,
            }

    safe_game_id = game_id.replace("/", "_")
    diverged_label = str(result.diverged_at)
    bundle_path = state_bundles_dir / f"{Path(db_path).stem}__{safe_game_id}__k{diverged_label}.state_bundle.json"

    bundle = {
        "db_path": str(db_path),
        "game_id": game_id,
        "diverged_at": result.diverged_at,
        "mismatch_kinds": list(result.mismatch_kinds or []),
        "mismatch_context": result.mismatch_context,
        "total_moves_python": result.total_moves_python,
        "total_moves_ts": result.total_moves_ts,
        "view_mode": view_mode,
        "ts_k_values": ks,
        "python_states": {str(k): py_states.get(k) for k in ks},
        "ts_states": {str(k): ts_states.get(k) for k in ks},
    }
    # Thread through ANM classification at the divergence step when available
    # so downstream diff tooling can inspect ANM parity alongside structural
    # state differences. This is intentionally small and focused: it does not
    # attempt to recompute ANM for every k from JSON alone.
    if result.diverged_at is not None:
        anm_entry: Dict[str, Optional[bool]] = {
            "is_anm_ts": getattr(result.ts_summary, "is_anm", None) if result.ts_summary is not None else None,
            "is_anm_py": getattr(result.python_summary, "is_anm", None) if result.python_summary is not None else None,
        }
        bundle["anm_state"] = {str(result.diverged_at): anm_entry}
    if meta_for_ks:
        bundle["ts_event_metadata"] = meta_for_ks

    try:
        state_bundles_dir.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, sort_keys=True)
    except Exception:
        return


def check_game_parity(
    db_path: Path,
    game_id: str,
    view: str = "post_move",
    state_bundles_dir: Optional[Path] = None,
) -> GameParityResult:
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

    total_moves_ts, ts_summaries, ts_event_meta = run_ts_replay(db_path, game_id, view_mode=view)
    py_post_move_summaries = replay_python_post_move_summaries(db, game_id)
    # Trust the replayed move count over metadata if they drift.
    if len(py_post_move_summaries) != total_moves_py:
        total_moves_py = len(py_post_move_summaries)

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
        else:
            # Optional ANM parity check at the initial state when TS exposes
            # an ANM flag. This is rare (initial states should generally not be
            # ANM), but we keep the logic consistent with per-move checks.
            if ts_initial.is_anm is not None:
                py_is_anm = py_initial.is_anm
                ts_is_anm = ts_initial.is_anm
                if py_is_anm is None or bool(py_is_anm) != bool(ts_is_anm):
                    diverged_at = 0
                    py_summary_at_diverge = py_initial
                    ts_summary_at_diverge = ts_initial
                    mismatch_kinds = ["anm_state"]
                    mismatch_context = "initial_state"

    # Then compare post-move (or post-bridge) states:
    #   TS k ↔ Python get_state_at_move(k-1)
    if diverged_at is None:
        max_ts_k = total_moves_ts  # TS k ranges from 1 to total_moves_ts
        for ts_k in range(1, max_ts_k + 1):
            py_move_index = ts_k - 1  # Python index for state after move (ts_k - 1)
            if py_move_index >= total_moves_py:
                break  # Python doesn't have this move recorded

            ts_summary = ts_summaries.get(ts_k)
            if ts_summary is None:
                diverged_at = ts_k
                py_summary_at_diverge = py_post_move_summaries.get(py_move_index)
                ts_summary_at_diverge = None
                mismatch_kinds = ["ts_missing_step"]
                mismatch_context = view
                break

            py_summary = py_post_move_summaries.get(py_move_index)
            if py_summary is None:
                diverged_at = ts_k
                py_summary_at_diverge = None
                ts_summary_at_diverge = ts_summary
                mismatch_kinds = ["python_missing_step"]
                mismatch_context = view
                break

            step_mismatches: List[str] = []
            if py_summary.current_player != ts_summary.current_player:
                step_mismatches.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                step_mismatches.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                step_mismatches.append("game_status")
            # ANM parity: when TS exposes an ANM flag for this k, compare it
            # against Python's ANM(state) classification for the same step.
            if ts_summary.is_anm is not None:
                if py_summary.is_anm is None or bool(py_summary.is_anm) != bool(ts_summary.is_anm):
                    step_mismatches.append("anm_state")
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
                mismatch_context = view
                break

    # If we had no per-move divergence but move counts differ, check whether
    # TS terminated early due to victory detection. If both engines show the
    # game as "completed" at TS's termination point with matching state hash,
    # treat the move count difference as acceptable (the DB just has extra
    # moves recorded after victory that should not have been recorded).
    if diverged_at is None and total_moves_py != total_moves_ts:
        # Check if TS terminated early with a completed game
        ts_final_summary = ts_summaries.get(total_moves_ts)
        early_victory_acceptable = False

        if (
            ts_final_summary is not None
            and ts_final_summary.game_status == "completed"
            and total_moves_ts < total_moves_py
        ):
            # TS ended early claiming victory - compare against Python's state
            # at the same move index to verify they agree on the outcome.
            py_move_index = total_moves_ts - 1  # Python uses 0-based move indexing
            if py_move_index >= 0:
                py_final_summary = py_post_move_summaries.get(py_move_index)
                # Accept if Python also shows completed AND state hashes match
                if (
                    py_final_summary is not None
                    and py_final_summary.game_status == "completed"
                    and py_final_summary.state_hash == ts_final_summary.state_hash
                ):
                    early_victory_acceptable = True

        if not early_victory_acceptable:
            diverged_at = None  # keep as None; mismatch is global, not at a single k
            mismatch_kinds = ["move_count"]
            mismatch_context = "global"

    # Determine if divergence is only at the very last move (end-of-game only)
    # AND the canonical state hash matches. Pure metadata differences in
    # player/phase/status at the terminal snapshot are insignificant for
    # training; any structural difference (state_hash mismatch) is treated as
    # a full semantic divergence.
    is_end_of_game_only = False
    if diverged_at is not None and diverged_at == total_moves_py and diverged_at == total_moves_ts:
        if py_summary_at_diverge is not None and ts_summary_at_diverge is not None:
            # Only treat as "end-of-game only" when the underlying board /
            # territory / elimination fingerprint is identical and the
            # divergence is limited to metadata fields.
            if py_summary_at_diverge.state_hash == ts_summary_at_diverge.state_hash:
                is_end_of_game_only = True

    result = GameParityResult(
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

    # Optionally emit a TS+Python state bundle for this game when a semantic
    # divergence is found. This now threads through the selected view_mode and
    # TS event metadata so downstream tooling can distinguish post_move vs
    # post_bridge semantics and see which TS event produced each TS state.
    if state_bundles_dir is not None and result.diverged_at is not None:
        try:
            dump_state_bundle(
                db=db,
                db_path=db_path,
                game_id=game_id,
                result=result,
                state_bundles_dir=state_bundles_dir,
                view_mode=view,
                ts_event_metadata=ts_event_meta,
            )
        except Exception:
            # Bundle emission is best-effort; parity classification should not fail.
            pass

    return result


def trace_game(
    db_path: Path,
    game_id: str,
    max_k: Optional[int] = None,
    view: str = "post_move",
) -> None:
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
        total_moves_ts, ts_summaries, _ = run_ts_replay(db_path, game_id, view_mode=view)
    except Exception as exc:
        print("[trace] TS replay failed " f"for db={db_path} game={game_id}: {exc}")
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
        if ts_initial.is_anm is not None:
            if py_initial.is_anm is None or bool(py_initial.is_anm) != bool(ts_initial.is_anm):
                init_dims.append("anm_state")

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

    try:
        py_post_move_summaries = replay_python_post_move_summaries(
            db,
            game_id,
            limit_moves=limit_k,
        )
    except Exception as exc:
        print("[trace] Python replay failed " f"for db={db_path} game={game_id}: {exc}")
        return

    for ts_k in range(1, limit_k + 1):
        py_index = ts_k - 1

        py_summary = py_post_move_summaries.get(py_index)

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
            if ts_summary.is_anm is not None:
                if py_summary.is_anm is None or bool(py_summary.is_anm) != bool(ts_summary.is_anm):
                    dims.append("anm_state")
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
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TS vs Python replay parity for all self-play GameReplayDBs.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional path to a single games.db to inspect. " "When omitted, scans all known self-play locations.",
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
        "--include-game-id",
        action="append",
        default=[],
        help=(
            "Only check these game_id values (repeatable). "
            "Use --include-game-ids-file when passing many IDs."
        ),
    )
    parser.add_argument(
        "--include-game-ids-file",
        type=str,
        default=None,
        help=(
            "Path to a newline-delimited list of game_ids (or a JSON list / {'game_ids': [...]}). "
            "When set, only those games are checked."
        ),
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
        help=("Optional maximum TS k to include in --trace-game output (0 = all steps)."),
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
    parser.add_argument(
        "--view",
        type=str,
        choices=["post_move", "post_bridge"],
        default="post_move",
        help=(
            "Select TS view semantics for parity comparison. "
            "'post_move' (default) compares TS 'ts-replay-step' states to "
            "Python GameReplayDB.get_state_at_move(n). "
            "'post_bridge' uses TS 'ts-replay-db-move-complete' states, which "
            "include synthesized bookkeeping between recorded DB moves, while "
            "Python remains on post_move semantics for now."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["canonical", "legacy"],
        default="canonical",
        help=(
            "Select parity mode. "
            "'canonical' (default) enforces a strict parity gate using post_move "
            "semantics only and exits non-zero on any structural, canonical-"
            "history, or semantic failure. "
            "'legacy' is a diagnostics-only profile that records structural and "
            "semantic issues in the JSON summary but only exits non-zero when "
            "--fail-on-divergence is set."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help=(
            "Emit replay progress to stderr every N TS steps / Python moves "
            "per game (0 disables). Overrides the environment variable "
            "RINGRIFT_PARITY_PROGRESS_EVERY."
        ),
    )
    args = parser.parse_args()
    mode = args.mode

    if args.progress_every is not None:
        os.environ["RINGRIFT_PARITY_PROGRESS_EVERY"] = str(int(args.progress_every))

    # Enforce mode/view compatibility: canonical parity gate is defined only
    # for post_move semantics.
    if mode == "canonical" and args.view != "post_move":
        print(
            "check_ts_python_replay_parity: canonical mode requires --view post_move "
            f"(got --view {args.view!r})",
            file=sys.stderr,
        )
        sys.exit(2)

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
                trace_game(db_path, args.trace_game, max_k=max_k, view=args.view)
                return

        print(f"[trace] game {args.trace_game} not found in any GameReplayDB " f"(searched {len(db_paths)} databases)")
        return

    include_game_ids: set[str] = set()
    try:
        include_game_ids = _load_game_id_filter(
            include_game_ids=list(args.include_game_id or []),
            include_game_ids_file=args.include_game_ids_file,
        )
    except Exception as exc:
        print(f"[parity-filter] {exc}", file=sys.stderr)
        sys.exit(2)

    filtered_found_game_ids: set[str] = set()

    structural_issues: List[Dict[str, object]] = []
    semantic_divergences: List[Dict[str, object]] = []
    end_of_game_only_divergences: List[Dict[str, object]] = []
    mismatch_counts_by_dimension: Dict[str, int] = {}
    total_games = 0
    total_semantic_divergent = 0
    total_end_of_game_only = 0
    total_structural_issues = 0
    games_with_non_canonical_history = 0

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
        if include_game_ids:
            games = [g for g in games if g.get("game_id") in include_game_ids]
            if not games:
                continue
        if args.limit_games_per_db and args.limit_games_per_db > 0:
            games = games[: args.limit_games_per_db]

        for game_meta in games:
            game_id = game_meta["game_id"]
            if include_game_ids:
                filtered_found_game_ids.add(str(game_id))
            total_games += 1
            try:
                result = check_game_parity(
                    db_path,
                    game_id,
                    view=args.view,
                    state_bundles_dir=state_bundles_dir,
                )
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
                if result.structure == "non_canonical_history":
                    games_with_non_canonical_history += 1
                structural_issues.append(
                    {
                        "db_path": str(db_path),
                        "game_id": game_id,
                        "structure": result.structure,
                        "structure_reason": result.structure_reason,
                    }
                )
                continue

            # A game is considered divergent if there's a specific divergence point
            # OR if there's a move count mismatch (indicated by "move_count" in mismatch_kinds).
            # When TS terminates early due to acceptable early victory detection,
            # total_moves may differ but mismatch_kinds will be empty.
            has_move_count_mismatch = "move_count" in (result.mismatch_kinds or [])
            if result.diverged_at is not None or has_move_count_mismatch:
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
                        "ts_summary": (asdict(result.ts_summary) if result.ts_summary is not None else None),
                        "canonical_move_index": canonical_move_index,
                        "canonical_move": canonical_move_dict,
                    }

                    safe_game_id = game_id.replace("/", "_")
                    diverged_label = "global" if result.diverged_at is None else str(result.diverged_at)
                    fixture_path = fixtures_dir / f"{Path(db_path).stem}__{safe_game_id}__k{diverged_label}.json"
                    with open(fixture_path, "w", encoding="utf-8") as f:
                        json.dump(fixture, f, indent=2, sort_keys=True)

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
        # NOTE: In compact mode we still enforce exit semantics (especially for
        # canonical parity gates) and can still emit --summary-json. We only
        # suppress the large JSON stdout summary + structural issue output.

    # Compute canonical parity gate status. In canonical mode we require:
    #   - no structural issues
    #   - no non-canonical histories
    #   - no semantic divergences
    #   - at least one game checked
    is_canonical_mode = mode == "canonical"
    passed_canonical_parity_gate = False
    if is_canonical_mode:
        passed_canonical_parity_gate = (
            total_games > 0
            and total_structural_issues == 0
            and games_with_non_canonical_history == 0
            and total_semantic_divergent == 0
        )

    summary = {
        "total_databases": len(db_paths),
        "total_games_checked": total_games,
        "games_with_semantic_divergence": total_semantic_divergent,
        "games_with_end_of_game_only_divergence": total_end_of_game_only,
        "games_with_structural_issues": total_structural_issues,
        "games_with_non_canonical_history": games_with_non_canonical_history,
        "semantic_divergences": semantic_divergences,
        "end_of_game_only_divergences": end_of_game_only_divergences,
        "structural_issues": structural_issues,
        "mismatch_counts_by_dimension": mismatch_counts_by_dimension,
        "view_mode": args.view,
        "mode": mode,
        "canonical_gate": bool(is_canonical_mode),
        "legacy_mode": bool(mode == "legacy"),
        "passed_canonical_parity_gate": bool(passed_canonical_parity_gate),
    }
    if include_game_ids:
        missing = sorted(include_game_ids.difference(filtered_found_game_ids))
        summary.update(
            {
                "filtered_game_ids_count": len(include_game_ids),
                "filtered_game_ids_found_count": len(filtered_found_game_ids),
                "filtered_game_ids_missing_count": len(missing),
                "filtered_game_ids_missing_sample": missing[:50],
            }
        )
    if not args.compact:
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

    # Exit semantics:
    #
    #   - In canonical mode we ALWAYS treat this as a parity gate:
    #       * non-zero exit when the canonical gate fails
    #       * zero exit only when passed_canonical_parity_gate is True.
    #
    #   - In legacy mode we preserve the old behaviour:
    #       * structural issues and non-canonical histories are reported but
    #         do not affect exit status by default;
    #       * when --fail-on-divergence is supplied, any semantic divergence
    #         (excluding end-of-game-only) triggers a non-zero exit.
    if is_canonical_mode:
        if not passed_canonical_parity_gate:
            print(
                (
                    "\n[FAIL] canonical parity gate failed: "
                    f"games_with_semantic_divergence={total_semantic_divergent}, "
                    f"games_with_structural_issues={total_structural_issues}, "
                    f"games_with_non_canonical_history={games_with_non_canonical_history}, "
                    f"total_games_checked={total_games}"
                ),
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.fail_on_divergence and total_semantic_divergent > 0:
            print(
                f"\n[FAIL] {total_semantic_divergent} game(s) with semantic divergence detected.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
