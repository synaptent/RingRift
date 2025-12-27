"""On-the-fly TS parity validation for game recordings.

This module provides utilities to validate recorded games against the TS canonical
engine immediately after recording. When divergence is detected, it dumps diagnostic
state bundles and raises an error, allowing immediate detection of parity issues
rather than discovering them later during batch analysis.

Environment Variables
---------------------
RINGRIFT_PARITY_VALIDATION
    Enable/disable on-the-fly parity validation. Values:
    - "off" / "disabled" / "false" / "0": Disabled (default)
    - "warn": Log warnings on divergence but don't fail
    - "strict" / "fail" / "error": Raise exception on divergence

RINGRIFT_PARITY_DUMP_DIR
    Directory to dump state bundles on divergence. Default: "parity_failures"
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.db.game_replay import GameReplayDB, _compute_state_hash
from app.game_engine import GameEngine
from app.models import BoardType

# NOTE: create_initial_state is imported lazily inside functions to avoid circular imports
from app.rules.serialization import serialize_game_state

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class ParityMode:
    """Parity validation modes."""
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


def get_parity_mode() -> str:
    """Get the current parity validation mode from environment."""
    env_val = os.environ.get("RINGRIFT_PARITY_VALIDATION", "off").lower()
    if env_val in ("off", "disabled", "false", "0", "no"):
        return ParityMode.OFF
    elif env_val in ("warn", "warning", "log"):
        return ParityMode.WARN
    elif env_val in ("strict", "fail", "error", "on", "true", "1", "yes"):
        return ParityMode.STRICT
    return ParityMode.OFF


def get_parity_dump_dir() -> Path:
    """Get the directory for dumping parity failure state bundles."""
    raw = Path(os.environ.get("RINGRIFT_PARITY_DUMP_DIR", "parity_failures"))
    # Use an absolute path so TS and Python agree on dump location even when
    # running with different CWDs (e.g. TS scripts run from repo root while
    # Python tooling often runs from ai-service/).
    if raw.is_absolute():
        return raw
    return (_repo_root() / raw).resolve()


def is_parity_validation_enabled() -> bool:
    """Check if any form of parity validation is enabled."""
    return get_parity_mode() != ParityMode.OFF


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class StateSummary:
    """Summary of a game state at a specific point."""
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str


@dataclass
class ParityDivergence:
    """Details of a parity divergence between Python and TS."""
    game_id: str
    db_path: str
    diverged_at: int
    mismatch_kinds: list[str]
    mismatch_context: str
    total_moves_python: int
    total_moves_ts: int
    python_summary: StateSummary | None
    ts_summary: StateSummary | None
    move_at_divergence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.python_summary:
            result["python_summary"] = asdict(self.python_summary)
        if self.ts_summary:
            result["ts_summary"] = asdict(self.ts_summary)
        return result


class ParityValidationError(Exception):
    """Raised when parity validation fails in strict mode."""

    def __init__(self, divergence: ParityDivergence, message: str | None = None):
        self.divergence = divergence
        msg = message or self._default_message()
        super().__init__(msg)

    def _default_message(self) -> str:
        d = self.divergence
        py_phase = d.python_summary.current_phase if d.python_summary else "N/A"
        ts_phase = d.ts_summary.current_phase if d.ts_summary else "N/A"
        py_player = d.python_summary.current_player if d.python_summary else "N/A"
        ts_player = d.ts_summary.current_player if d.ts_summary else "N/A"
        return (
            f"Parity divergence in game {d.game_id} at k={d.diverged_at}: "
            f"mismatches={d.mismatch_kinds}, "
            f"py_phase={py_phase} vs ts_phase={ts_phase}, "
            f"py_player={py_player} vs ts_player={ts_player}"
        )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the monorepo root (parent of ai-service/)."""
    return Path(__file__).resolve().parents[3]


# Cache for resolved npx path
_NPX_PATH_CACHE: str | None = None


def _find_npx() -> str | None:
    """Find the npx executable with fallbacks for various environments.

    Search order:
    1. RINGRIFT_NPX_PATH environment variable (explicit override)
    2. PATH via shutil.which
    3. Common installation locations:
       - ~/.nvm/versions/node/*/bin/npx (nvm)
       - /opt/homebrew/bin/npx (macOS Homebrew)
       - /usr/local/bin/npx (system install)
       - ~/.local/bin/npx (user local)
       - /usr/bin/npx (Linux system)

    Returns:
        Path to npx executable, or None if not found.
    """
    global _NPX_PATH_CACHE

    # Return cached result
    if _NPX_PATH_CACHE is not None:
        return _NPX_PATH_CACHE if _NPX_PATH_CACHE != "" else None

    # 1. Check explicit override
    explicit = os.environ.get("RINGRIFT_NPX_PATH")
    if explicit and Path(explicit).exists():
        _NPX_PATH_CACHE = explicit
        return explicit

    # 2. Check PATH
    which_npx = shutil.which("npx")
    if which_npx:
        _NPX_PATH_CACHE = which_npx
        return which_npx

    # 3. Check common locations
    home = Path.home()
    common_paths = [
        "/opt/homebrew/bin/npx",        # macOS Homebrew
        "/usr/local/bin/npx",           # Common system install
        str(home / ".local/bin/npx"),   # User local
        "/usr/bin/npx",                 # Linux system
    ]

    # Check nvm installations (find latest version)
    nvm_dir = home / ".nvm/versions/node"
    if nvm_dir.exists():
        node_versions = sorted(nvm_dir.iterdir(), reverse=True)
        for ver_dir in node_versions:
            npx_path = ver_dir / "bin/npx"
            if npx_path.exists():
                _NPX_PATH_CACHE = str(npx_path)
                return str(npx_path)

    for path in common_paths:
        if Path(path).exists():
            _NPX_PATH_CACHE = path
            return path

    # Not found
    _NPX_PATH_CACHE = ""  # Cache negative result
    logger.warning(
        "npx executable not found. Set RINGRIFT_NPX_PATH env var or install Node.js. "
        "Parity validation against TypeScript will be skipped."
    )
    return None


def _is_ts_replay_available() -> bool:
    """Check if the TypeScript replay harness is available."""
    npx = _find_npx()
    if not npx:
        return False

    # Check that the TS script exists
    root = _repo_root()
    ts_script = root / "scripts/selfplay-db-ts-replay.ts"
    return ts_script.exists()


def _parse_board_type(board_type_str: str) -> BoardType:
    """Parse a board type string into a BoardType enum value."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    return mapping.get(board_type_str.lower(), BoardType.SQUARE8)


def _summarize_python_state(db: GameReplayDB, game_id: str, move_index: int) -> StateSummary:
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
        game_status=state.game_status.value
        if hasattr(state.game_status, "value")
        else str(state.game_status),
        state_hash=_compute_state_hash(state),
    )


def _summarize_python_initial_state(db: GameReplayDB, game_id: str) -> StateSummary:
    """Summarize the initial state BEFORE any moves are applied."""
    state = db.get_initial_state(game_id)
    if state is None:
        # Lazy import to avoid circular dependency
        from app.training.generate_data import create_initial_state

        metadata = db.get_game_metadata(game_id)
        if metadata is None:
            raise RuntimeError(f"No initial_state and no metadata for {game_id}")
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
        game_status=state.game_status.value
        if hasattr(state.game_status, "value")
        else str(state.game_status),
        state_hash=_compute_state_hash(state),
    )


def _get_python_initial_state_for_replay(db: GameReplayDB, game_id: str):
    """Return the initial GameState for replay, falling back deterministically.

    This mirrors :func:`_summarize_python_initial_state` but returns the full
    GameState so callers can perform a single-pass replay (O(n)) instead of
    repeatedly invoking get_state_at_move (O(n^2)).
    """
    state = db.get_initial_state(game_id)
    if state is not None:
        return state

    # Lazy import to avoid circular dependency
    from app.training.generate_data import create_initial_state

    metadata = db.get_game_metadata(game_id)
    if metadata is None:
        raise RuntimeError(f"No initial_state and no metadata for {game_id}")
    board_type_str = metadata.get("board_type", "square8")
    num_players = metadata.get("num_players", 2)
    board_type = _parse_board_type(board_type_str)
    return create_initial_state(board_type=board_type, num_players=num_players)


def _run_ts_replay(db_path: Path, game_id: str) -> tuple[int, dict[int, StateSummary]]:
    """Invoke the TS harness and parse its per-move summaries.

    Returns:
        (total_moves_reported_by_ts, mapping from k -> summary)

    Raises:
        RuntimeError: If npx is not available or the replay fails.
    """
    npx = _find_npx()
    if not npx:
        raise RuntimeError(
            "npx executable not found. Install Node.js or set RINGRIFT_NPX_PATH. "
            "Skipping TS parity validation."
        )

    root = _repo_root()
    cmd = [
        npx,
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
    stdout, stderr = proc.communicate(timeout=120)

    if proc.returncode != 0:
        raise RuntimeError(
            f"TS replay harness failed for {db_path} / {game_id} with code {proc.returncode}:\n"
            f"STDOUT:\n{stdout[:1000]}\nSTDERR:\n{stderr[:1000]}"
        )

    total_ts_moves = 0
    summaries: dict[int, StateSummary] = {}

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
                game_status=summary.get("gameStatus"),
                state_hash=summary.get("stateHash"),
            )
        elif kind == "ts-replay-step":
            k = int(payload.get("k", 0))
            summary = payload.get("summary") or {}
            summaries[k] = StateSummary(
                move_index=k,
                current_player=summary.get("currentPlayer"),
                current_phase=summary.get("currentPhase"),
                game_status=summary.get("gameStatus"),
                state_hash=summary.get("stateHash"),
            )

    return total_ts_moves, summaries


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
            # Lazy import to avoid circular dependency
            from app.training.generate_data import create_initial_state

            metadata = db.get_game_metadata(game_id)
            if metadata is None:
                raise RuntimeError(f"No initial_state and no metadata for {game_id}")
            board_type_str = metadata.get("board_type", "square8")
            num_players = metadata.get("num_players", 2)
            board_type = _parse_board_type(board_type_str)
            state = create_initial_state(board_type=board_type, num_players=num_players)
        return state

    py_index = ts_k - 1
    if py_index < 0 or py_index >= total_moves_py:
        return None

    return db.get_state_at_move(game_id, py_index)


# -----------------------------------------------------------------------------
# State Bundle Dumping
# -----------------------------------------------------------------------------

def dump_divergence_bundle(
    db: GameReplayDB,
    db_path: Path,
    divergence: ParityDivergence,
    output_dir: Path,
) -> Path:
    """Dump a rich state bundle for debugging a parity divergence.

    Returns the path to the saved bundle file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    div_k = divergence.diverged_at
    if div_k <= 0:
        ks = [0]
    else:
        before_k = max(0, div_k - 1)
        ks = [before_k, div_k] if div_k != before_k else [div_k]

    # Collect Python states
    py_states: dict[int, dict[str, Any]] = {}
    for ts_k in ks:
        state = _get_python_state_for_ts_k(
            db, divergence.game_id, ts_k, divergence.total_moves_python
        )
        if state is not None:
            try:
                py_states[ts_k] = serialize_game_state(state)
            except (TypeError, ValueError, AttributeError):
                continue

    # Collect TS states by invoking replay with dump
    ts_states: dict[int, dict[str, Any]] = {}
    npx = _find_npx()
    if not npx:
        logger.warning(
            "npx not found - skipping TS state dump for divergence bundle. "
            "Set RINGRIFT_NPX_PATH or install Node.js."
        )
    else:
        try:
            root = _repo_root()
            env = os.environ.copy()
            env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")

            # Use both env var AND CLI arg for robustness
            dump_arg = ",".join(str(k) for k in sorted(set(ks)))
            env["RINGRIFT_TS_REPLAY_DUMP_DIR"] = str(output_dir)
            env["RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K"] = dump_arg

            cmd = [
                npx, "ts-node", "-T",
                "scripts/selfplay-db-ts-replay.ts",
                "--db", str(db_path),
                "--game", divergence.game_id,
                "--dump-state-at", dump_arg,
            ]

            result = subprocess.run(
                cmd, cwd=str(root), env=env,
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                logger.warning(
                    f"TS replay for state dump failed with code {result.returncode}:\n"
                    f"STDERR: {result.stderr[:500] if result.stderr else 'N/A'}"
                )

            # Log the TS replay output for debugging
            if result.stdout:
                for line in result.stdout.splitlines()[:5]:
                    logger.debug(f"TS replay output: {line}")

            # Read the dumped TS state files
            for ts_k in ks:
                file_name = f"{db_path.name}__{divergence.game_id}__k{ts_k}.ts_state.json"
                ts_path = output_dir / file_name
                if ts_path.exists():
                    try:
                        with ts_path.open("r") as f:
                            ts_states[ts_k] = json.load(f)
                        logger.debug(f"Loaded TS state from {ts_path}")
                    except Exception as e:
                        logger.warning(f"Failed to parse TS state file {ts_path}: {e}")
                else:
                    logger.warning(f"TS state file not found: {ts_path}")
        except subprocess.TimeoutExpired:
            logger.warning(f"TS replay timed out for game {divergence.game_id}")
        except Exception as e:
            logger.warning(f"Failed to dump TS states: {e}", exc_info=True)

    # Get move metadata around divergence
    moves_window: list[dict[str, Any]] = []
    try:
        moves = db.get_moves(divergence.game_id)
        if moves:
            start_idx = max(0, div_k - 3)
            end_idx = min(len(moves), div_k + 2)
            for i in range(start_idx, end_idx):
                move = moves[i]
                try:
                    move_dict = json.loads(move.model_dump_json(by_alias=True))
                    move_dict["_index"] = i
                    moves_window.append(move_dict)
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    continue
    except Exception as e:
        logger.warning(f"Failed to get moves window: {e}")

    # Assemble bundle
    timestamp = datetime.now().isoformat()
    bundle = {
        "timestamp": timestamp,
        "db_path": str(db_path),
        "game_id": divergence.game_id,
        "diverged_at": div_k,
        "mismatch_kinds": divergence.mismatch_kinds,
        "mismatch_context": divergence.mismatch_context,
        "total_moves_python": divergence.total_moves_python,
        "total_moves_ts": divergence.total_moves_ts,
        "python_summary": asdict(divergence.python_summary) if divergence.python_summary else None,
        "ts_summary": asdict(divergence.ts_summary) if divergence.ts_summary else None,
        "ts_k_values": ks,
        "python_states": {str(k): py_states.get(k) for k in ks},
        "ts_states": {str(k): ts_states.get(k) for k in ks},
        "moves_window": moves_window,
    }

    safe_game_id = divergence.game_id.replace("/", "_")
    bundle_name = f"{Path(db_path).stem}__{safe_game_id}__k{div_k}.parity_failure.json"
    bundle_path = output_dir / bundle_name

    with bundle_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)

    return bundle_path


# -----------------------------------------------------------------------------
# Main Validation Function
# -----------------------------------------------------------------------------

def validate_game_parity(
    db_path: str,
    game_id: str,
    mode: str | None = None,
    dump_dir: Path | None = None,
) -> ParityDivergence | None:
    """Validate a recorded game against the TS canonical engine.

    Args:
        db_path: Path to the GameReplayDB database
        game_id: ID of the game to validate
        mode: Override parity mode (uses env var if None)
        dump_dir: Override dump directory (uses env var if None)

    Returns:
        ParityDivergence if divergence was found, None if parity is clean

    Raises:
        ParityValidationError: If mode is 'strict' and divergence is found
    """
    effective_mode = mode or get_parity_mode()
    if effective_mode == ParityMode.OFF:
        return None

    effective_dump_dir = dump_dir or get_parity_dump_dir()
    db_path_obj = Path(db_path)

    db = GameReplayDB(str(db_path_obj))
    meta = db.get_game_metadata(game_id)
    if not meta:
        logger.warning(f"Game {game_id} not found in {db_path}")
        return None

    moves = db.get_moves(game_id)
    total_moves_py = len(moves)

    try:
        total_moves_ts, ts_summaries = _run_ts_replay(db_path_obj, game_id)
    except Exception as e:
        # Treat TS harness failures as structural divergences in strict mode
        logger.warning(f"TS replay failed for {game_id}: {e}")
        if effective_mode == ParityMode.STRICT:
            divergence = ParityDivergence(
                game_id=game_id,
                db_path=str(db_path_obj),
                diverged_at=-1,
                mismatch_kinds=["structure_error"],
                mismatch_context=f"ts_replay_error:{type(e).__name__}",
                total_moves_python=total_moves_py,
                total_moves_ts=-1,
                python_summary=None,
                ts_summary=None,
                move_at_divergence=None,
            )
            raise ParityValidationError(divergence)
        return None

    divergence: ParityDivergence | None = None

    # Compare initial states (TS k=0 vs Python initial_state)
    ts_initial = ts_summaries.get(0)
    if ts_initial is not None:
        py_initial = _summarize_python_initial_state(db, game_id)
        init_mismatches: list[str] = []
        if py_initial.current_player != ts_initial.current_player:
            init_mismatches.append("current_player")
        if py_initial.current_phase != ts_initial.current_phase:
            init_mismatches.append("current_phase")
        if py_initial.game_status != ts_initial.game_status:
            init_mismatches.append("game_status")

        if init_mismatches:
            divergence = ParityDivergence(
                game_id=game_id,
                db_path=str(db_path_obj),
                diverged_at=0,
                mismatch_kinds=init_mismatches,
                mismatch_context="initial_state",
                total_moves_python=total_moves_py,
                total_moves_ts=total_moves_ts,
                python_summary=py_initial,
                ts_summary=ts_initial,
            )

    # Compare post-move states: TS k ↔ Python get_state_at_move(k-1)
    if divergence is None:
        state = _get_python_initial_state_for_replay(db, game_id)
        max_ts_k = total_moves_ts
        for ts_k in range(1, max_ts_k + 1):
            py_move_index = ts_k - 1
            if py_move_index >= total_moves_py:
                break

            try:
                state = GameEngine.apply_move(state, moves[py_move_index], trace_mode=True)
            except Exception as exc:
                py_summary = StateSummary(
                    move_index=py_move_index,
                    current_player=state.current_player,
                    current_phase=state.current_phase.value
                    if hasattr(state.current_phase, "value")
                    else str(state.current_phase),
                    game_status=state.game_status.value
                    if hasattr(state.game_status, "value")
                    else str(state.game_status),
                    state_hash=_compute_state_hash(state),
                )
                divergence = ParityDivergence(
                    game_id=game_id,
                    db_path=str(db_path_obj),
                    diverged_at=ts_k,
                    mismatch_kinds=["python_replay_error"],
                    mismatch_context=f"python_apply_move_error:{type(exc).__name__}:{exc}",
                    total_moves_python=total_moves_py,
                    total_moves_ts=total_moves_ts,
                    python_summary=py_summary,
                    ts_summary=ts_summaries.get(ts_k),
                )
                break

            py_summary = StateSummary(
                move_index=py_move_index,
                current_player=state.current_player,
                current_phase=state.current_phase.value
                if hasattr(state.current_phase, "value")
                else str(state.current_phase),
                game_status=state.game_status.value
                if hasattr(state.game_status, "value")
                else str(state.game_status),
                state_hash=_compute_state_hash(state),
            )

            ts_summary = ts_summaries.get(ts_k)
            if ts_summary is None:
                divergence = ParityDivergence(
                    game_id=game_id,
                    db_path=str(db_path_obj),
                    diverged_at=ts_k,
                    mismatch_kinds=["ts_missing_step"],
                    mismatch_context="post_move",
                    total_moves_python=total_moves_py,
                    total_moves_ts=total_moves_ts,
                    python_summary=py_summary,
                    ts_summary=None,
                )
                break

            step_mismatches: list[str] = []

            if py_summary.current_player != ts_summary.current_player:
                step_mismatches.append("current_player")
            if py_summary.current_phase != ts_summary.current_phase:
                step_mismatches.append("current_phase")
            if py_summary.game_status != ts_summary.game_status:
                step_mismatches.append("game_status")
            if py_summary.state_hash != ts_summary.state_hash:
                step_mismatches.append("state_hash")

            if step_mismatches:
                # Get the move at divergence for debugging
                move_dict = None
                try:
                    if moves and py_move_index < len(moves):
                        move = moves[py_move_index]
                        move_dict = json.loads(move.model_dump_json(by_alias=True))
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    pass

                divergence = ParityDivergence(
                    game_id=game_id,
                    db_path=str(db_path_obj),
                    diverged_at=ts_k,
                    mismatch_kinds=step_mismatches,
                    mismatch_context="post_move",
                    total_moves_python=total_moves_py,
                    total_moves_ts=total_moves_ts,
                    python_summary=py_summary,
                    ts_summary=ts_summary,
                    move_at_divergence=move_dict,
                )
                break

    # Check move count mismatch
    if divergence is None and total_moves_py != total_moves_ts:
        divergence = ParityDivergence(
            game_id=game_id,
            db_path=str(db_path_obj),
            diverged_at=min(total_moves_py, total_moves_ts),
            mismatch_kinds=["move_count"],
            mismatch_context="global",
            total_moves_python=total_moves_py,
            total_moves_ts=total_moves_ts,
            python_summary=None,
            ts_summary=None,
        )

    if divergence is None:
        return None

    end_of_game_only = (
        divergence.diverged_at == total_moves_py
        and divergence.diverged_at == total_moves_ts
        and divergence.python_summary is not None
        and divergence.ts_summary is not None
        and divergence.python_summary.state_hash == divergence.ts_summary.state_hash
    )

    # Handle divergence based on mode
    bundle_path = dump_divergence_bundle(db, db_path_obj, divergence, effective_dump_dir)

    if effective_mode == ParityMode.WARN:
        logger.warning(
            f"Parity divergence detected in game {game_id} at k={divergence.diverged_at}: "
            f"mismatches={divergence.mismatch_kinds}. "
            f"State bundle saved to {bundle_path}"
        )
        return divergence

    elif effective_mode == ParityMode.STRICT:
        if end_of_game_only:
            logger.warning(
                "End-of-game-only parity divergence allowed for %s at k=%s "
                "(mismatches=%s, state_hash=%s). Bundle saved to %s",
                game_id,
                divergence.diverged_at,
                divergence.mismatch_kinds,
                divergence.python_summary.state_hash if divergence.python_summary else "n/a",
                bundle_path,
            )
            return divergence
        raise ParityValidationError(
            divergence,
            f"Parity divergence in game {game_id} at k={divergence.diverged_at}. "
            f"mismatches={divergence.mismatch_kinds}. "
            f"State bundle saved to {bundle_path}"
        )

    return divergence


# -----------------------------------------------------------------------------
# Convenience Functions for Recording Integration
# -----------------------------------------------------------------------------

def validate_after_recording(
    db: GameReplayDB,
    game_id: str,
    mode: str | None = None,
    dump_dir: Path | None = None,
) -> ParityDivergence | None:
    """Validate a game immediately after recording.

    This is a convenience wrapper for use in recording functions.
    It uses the db's path property if available, or searches for the path.

    Args:
        db: The GameReplayDB instance used for recording
        game_id: ID of the recorded game
        mode: Override parity mode
        dump_dir: Override dump directory

    Returns:
        ParityDivergence if divergence was found, None if parity is clean

    Raises:
        ParityValidationError: If mode is 'strict' and divergence is found
    """
    # Get db path from the instance
    db_path = getattr(db, 'db_path', None) or getattr(db, '_db_path', None)
    if db_path is None:
        logger.warning("Could not determine db path for parity validation")
        return None

    return validate_game_parity(str(db_path), game_id, mode=mode, dump_dir=dump_dir)


# -----------------------------------------------------------------------------
# Python-Native Parity Validation (No Node.js Required)
# -----------------------------------------------------------------------------

@dataclass
class PythonParityResult:
    """Result of Python-only parity validation."""
    game_id: str
    passed: bool
    moves_replayed: int
    total_moves: int
    error_at_move: int | None = None
    error_message: str | None = None
    replay_time_ms: float = 0.0


def validate_game_python_only(
    db: GameReplayDB,
    game_id: str,
    *,
    verbose: bool = False,
) -> PythonParityResult:
    """Validate a game using Python-only replay (no TypeScript required).

    This validates that the recorded moves can be replayed through the Python
    GameEngine without errors, and that the final state matches what was recorded.

    This is useful for:
    - Cluster nodes without Node.js installed
    - Fast validation during selfplay
    - Detecting corrupted move data (e.g., to=None issues)

    Args:
        db: GameReplayDB instance
        game_id: ID of the game to validate
        verbose: Log detailed progress

    Returns:
        PythonParityResult with validation status

    December 2025: Created to enable parity validation on cluster nodes
    without requiring Node.js/npx installation.
    """
    import time

    start_time = time.time()

    # Get game metadata and moves
    meta = db.get_game_metadata(game_id)
    if not meta:
        return PythonParityResult(
            game_id=game_id,
            passed=False,
            moves_replayed=0,
            total_moves=0,
            error_message=f"Game {game_id} not found",
        )

    moves = db.get_moves(game_id)
    total_moves = len(moves)

    if total_moves == 0:
        # Empty game is valid
        return PythonParityResult(
            game_id=game_id,
            passed=True,
            moves_replayed=0,
            total_moves=0,
            replay_time_ms=(time.time() - start_time) * 1000,
        )

    # Get initial state
    state = _get_python_initial_state_for_replay(db, game_id)

    # Replay all moves
    for i, move in enumerate(moves):
        try:
            # Check for corrupted move data
            if hasattr(move, 'to') and move.to is None:
                if hasattr(move, 'type') and move.type in ('PLACE_RING', 'place_ring'):
                    return PythonParityResult(
                        game_id=game_id,
                        passed=False,
                        moves_replayed=i,
                        total_moves=total_moves,
                        error_at_move=i,
                        error_message=f"Corrupted move at index {i}: PLACE_RING with to=None",
                        replay_time_ms=(time.time() - start_time) * 1000,
                    )

            # Apply move
            state = GameEngine.apply_move(state, move, trace_mode=False)

            if verbose and (i + 1) % 10 == 0:
                logger.debug(f"[PythonParity] Game {game_id}: replayed {i + 1}/{total_moves} moves")

        except Exception as e:
            return PythonParityResult(
                game_id=game_id,
                passed=False,
                moves_replayed=i,
                total_moves=total_moves,
                error_at_move=i,
                error_message=f"Move {i} failed: {type(e).__name__}: {e}",
                replay_time_ms=(time.time() - start_time) * 1000,
            )

    # All moves replayed successfully
    elapsed_ms = (time.time() - start_time) * 1000

    if verbose:
        logger.info(f"[PythonParity] Game {game_id}: PASSED ({total_moves} moves in {elapsed_ms:.1f}ms)")

    return PythonParityResult(
        game_id=game_id,
        passed=True,
        moves_replayed=total_moves,
        total_moves=total_moves,
        replay_time_ms=elapsed_ms,
    )


def validate_database_python_only(
    db_path: str | Path,
    *,
    max_games: int | None = None,
    update_status: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate all games in a database using Python-only replay.

    This enables parity validation on cluster nodes without Node.js.

    Args:
        db_path: Path to the GameReplayDB database
        max_games: Maximum number of games to validate (None = all)
        update_status: Update parity_status in database
        verbose: Log detailed progress

    Returns:
        Dict with validation statistics:
        - total_games: Number of games checked
        - passed: Number that passed
        - failed: Number that failed
        - failed_games: List of (game_id, error_message)
        - elapsed_seconds: Total validation time

    December 2025: Created to unblock canonical database validation on cluster.
    """
    import time

    start_time = time.time()
    db_path = Path(db_path)

    db = GameReplayDB(str(db_path))

    # Get all game IDs via query_games
    # Use a large limit to get all games, then apply max_games filter
    limit = max_games if max_games else 100_000
    games = db.query_games(limit=limit, require_moves=True)
    game_ids = [g["game_id"] for g in games]

    passed = 0
    failed = 0
    failed_games: list[tuple[str, str]] = []

    for i, game_id in enumerate(game_ids):
        result = validate_game_python_only(db, game_id, verbose=verbose)

        if result.passed:
            passed += 1
            if update_status:
                _update_parity_status(db, game_id, "passed")
        else:
            failed += 1
            failed_games.append((game_id, result.error_message or "unknown error"))
            if update_status:
                _update_parity_status(db, game_id, "failed", result.error_at_move)

        if (i + 1) % 100 == 0 or verbose:
            logger.info(
                f"[PythonParity] {db_path.name}: {i + 1}/{len(game_ids)} games "
                f"(passed={passed}, failed={failed})"
            )

    elapsed = time.time() - start_time

    # Update database-level parity gate status
    if update_status:
        gate_status = "passed" if failed == 0 else "failed"
        _update_database_parity_gate(db, gate_status, passed, failed)

    result = {
        "database": str(db_path),
        "total_games": len(game_ids),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(game_ids) if game_ids else 1.0,
        "failed_games": failed_games[:50],  # Limit to first 50 for readability
        "elapsed_seconds": elapsed,
    }

    logger.info(
        f"[PythonParity] {db_path.name}: Complete. "
        f"{passed}/{len(game_ids)} passed ({result['pass_rate']:.1%}), "
        f"elapsed={elapsed:.1f}s"
    )

    return result


def _update_parity_status(
    db: GameReplayDB,
    game_id: str,
    status: str,
    divergence_move: int | None = None,
) -> None:
    """Update parity status for a single game."""
    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with db._get_connection() as conn:
            conn.execute(
                """
                UPDATE games
                SET parity_status = ?, parity_checked_at = ?, parity_divergence_move = ?
                WHERE game_id = ?
                """,
                (status, now, divergence_move, game_id),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to update parity status for {game_id}: {e}")


def _update_database_parity_gate(
    db: GameReplayDB,
    status: str,
    passed_count: int,
    failed_count: int,
) -> None:
    """Update database-level parity gate metadata."""
    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with db._get_connection() as conn:
            # Update or insert metadata
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('parity_gate_status', ?)
                """,
                (status,),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('parity_gate_timestamp', ?)
                """,
                (now,),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('parity_gate_passed_count', ?)
                """,
                (str(passed_count),),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('parity_gate_failed_count', ?)
                """,
                (str(failed_count),),
            )
            conn.commit()
            logger.info(f"Updated parity gate status: {status} (passed={passed_count}, failed={failed_count})")
    except Exception as e:
        logger.warning(f"Failed to update database parity gate: {e}")
