#!/usr/bin/env python
"""Long self-play soak harness for RingRift (Python rules engine).

This script runs mixed- or descent-only AI self-play games using the
Python GameEngine + RingRiftEnv stack and records basic stability
metrics. It is intended for offline / long-run use (hundreds or
thousands of games), outside of pytest timeouts.

Key properties
==============
- Uses the same RingRiftEnv and AI selection logic as
  :mod:`app.training.generate_territory_dataset`.
- Supports 2–4 players and multiple board types.
- Can optionally enable a strict invariant:

    RINGRIFT_STRICT_NO_MOVE_INVARIANT=1

  which asserts that every ACTIVE state has at least one legal move for
  the current player, mirroring the shared TS TurnLogic contract.
- Writes a JSONL log of per-game summaries (status, reason, length),
  plus an optional aggregate JSON summary.
- Supports on-the-fly parity validation against the TS canonical engine:

    RINGRIFT_PARITY_VALIDATION=strict  # or "warn" or "off" (default)
    RINGRIFT_PARITY_DUMP_DIR=/tmp/parity-bundles

  When enabled, each recorded game is replayed through the TS engine
  after recording. If divergence is detected:
  - "warn" mode: logs a warning but continues
  - "strict" mode: dumps diagnostic state bundles and halts the soak

Example usage
-------------

From ``ai-service/``::

    # 100 mixed-engine 2p games on square8, invariant enabled
    # max-moves auto-derived: 400 for square8/2p, 2000 for hexagonal/2p
    RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
    python scripts/run_self_play_soak.py \
        --num-games 100 \
        --board-type square8 \
        --engine-mode mixed \
        --num-players 2 \
        --seed 42 \
        --log-jsonl logs/selfplay/soak.square8_2p.mixed.jsonl \
        --summary-json logs/selfplay/soak.square8_2p.mixed.summary.json

    # 50 descent-only 3p games on square8
    python scripts/run_self_play_soak.py \
        --num-games 50 \
        --board-type square8 \
        --engine-mode descent-only \
        --num-players 3 \
        --seed 123
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import random
import sys
import gc
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# Ensure `app.*` imports resolve when run from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.main import (  # type: ignore  # noqa: E402
    _create_ai_instance,
    _get_difficulty_profile,
)
from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    AIType,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Position,
)
from app.training.env import (  # type: ignore  # noqa: E402
    TrainingEnvConfig,
    make_env,
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
    get_theoretical_max_moves,
)
from app.game_engine import (  # type: ignore  # noqa: E402
    GameEngine,
    STRICT_NO_MOVE_INVARIANT,
    PhaseRequirementType,
    PhaseRequirement,
)

# GPU imports - lazy imported only when --gpu is used to avoid torch import overhead
GPU_IMPORTS_LOADED = False
GPUSelfPlayGenerator = None  # Populated on first GPU use

# Module-level logger used throughout the soak harness. The wider app stack
# configures logging handlers/levels, so we only need a named logger here.
logger = logging.getLogger(__name__)

# Import coordination for task limits and duration tracking
try:
    from app.coordination import (
        TaskType,
        can_spawn_safe,
        register_running_task,
        record_task_completion,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    TaskType = None
    can_spawn_safe = None


def _load_gpu_imports() -> bool:
    """Lazily load GPU imports to avoid torch import overhead when not using GPU.

    Returns:
        True if imports succeeded, False if GPU is not available.
    """
    global GPU_IMPORTS_LOADED, GPUSelfPlayGenerator

    if GPU_IMPORTS_LOADED:
        return GPUSelfPlayGenerator is not None

    GPU_IMPORTS_LOADED = True

    try:
        from scripts.run_gpu_selfplay import GPUSelfPlayGenerator as _GPUSelfPlayGenerator
        GPUSelfPlayGenerator = _GPUSelfPlayGenerator
        return True
    except ImportError as e:
        logger.warning(f"GPU imports failed: {e}")
        return False


# =============================================================================
# Heuristic Weight Loading
# =============================================================================


def load_weights_from_profile(
    weights_file: str,
    profile_name: str,
) -> Optional[Dict[str, float]]:
    """Load heuristic weights from a profile file.

    Args:
        weights_file: Path to JSON file containing weight profiles
        profile_name: Name of the profile to load

    Returns:
        Dict of weight name -> value, or None if loading fails
    """
    if not os.path.exists(weights_file):
        print(
            f"[heuristic-weights] Warning: Weights file not found: {weights_file}",
            file=sys.stderr,
        )
        return None

    try:
        with open(weights_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(
            f"[heuristic-weights] Warning: Failed to parse {weights_file}: {e}",
            file=sys.stderr,
        )
        return None

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        print(
            f"[heuristic-weights] Warning: Profile '{profile_name}' not found in {weights_file}. "
            f"Available: {list(profiles.keys())}",
            file=sys.stderr,
        )
        return None

    weights = profiles[profile_name].get("weights", {})
    print(
        f"[heuristic-weights] Loaded profile '{profile_name}' with {len(weights)} weights",
        flush=True,
    )
    return weights
from app.metrics import (  # type: ignore  # noqa: E402
    PYTHON_INVARIANT_VIOLATIONS,
)
from app.rules.core import compute_progress_snapshot  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402
from app.rules import global_actions as ga  # type: ignore  # noqa: E402
from app.rules.history_contract import (  # noqa: E402
    validate_canonical_move,
)
from app.utils.progress_reporter import SoakProgressReporter  # noqa: E402
from app.db import (  # noqa: E402
    get_or_create_db,
    record_completed_game_with_parity_check,
    ParityValidationError,
)
from app.ai.neural_net import clear_model_cache  # noqa: E402

# Hot model reload for unified AI loop integration
try:
    from app.distributed.data_events import (
        emit_new_games,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False


class HotModelReloader:
    """Hot model reload manager for continuous selfplay.

    Monitors model files and triggers reload when they change. This allows
    selfplay to automatically pick up newly promoted models without restart.

    Usage:
        reloader = HotModelReloader(board_type="square8", num_players=2)

        for game_num in range(num_games):
            if reloader.should_check(game_num, interval=100):
                if reloader.check_for_updates():
                    print(f"Model updated: {reloader.current_model_path}")
                    clear_model_cache()  # Force reload on next AI creation
            # ... run game ...
    """

    BOARD_ALIAS_TOKENS = {
        "square8": "sq8",
        "square19": "sq19",
        "hexagonal": "hex",
    }

    def __init__(
        self,
        board_type: str,
        num_players: int,
        model_alias_path: Optional[str] = None,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self._models_dir = os.path.join(ROOT, "models")

        # Determine which model path to watch
        if model_alias_path:
            self._watch_path = model_alias_path
        else:
            token = self.BOARD_ALIAS_TOKENS.get(board_type, board_type)
            self._watch_path = os.path.join(
                self._models_dir,
                f"ringrift_best_{token}_{num_players}p.pth"
            )

        # Track model state
        self._current_mtime: Optional[float] = None
        self._current_hash: Optional[str] = None
        self._update_count = 0
        self._last_check_game = 0

        # Initialize by reading current state
        self._read_current_state()

    @property
    def current_model_path(self) -> str:
        """Get the currently watched model path."""
        return self._watch_path

    @property
    def update_count(self) -> int:
        """Number of model updates detected since initialization."""
        return self._update_count

    def _read_current_state(self) -> bool:
        """Read current model file state. Returns True if file exists."""
        if not os.path.exists(self._watch_path):
            return False

        try:
            stat = os.stat(self._watch_path)
            self._current_mtime = stat.st_mtime
            # Quick hash based on size + mtime (faster than full file hash)
            self._current_hash = f"{stat.st_size}:{stat.st_mtime}"
            return True
        except OSError:
            return False

    def should_check(self, game_num: int, interval: int) -> bool:
        """Check if we should look for model updates at this game number."""
        if interval <= 0:
            return False
        if game_num == 0:
            return False  # Already initialized
        if game_num - self._last_check_game >= interval:
            self._last_check_game = game_num
            return True
        return False

    def check_for_updates(self) -> bool:
        """Check if the model file has been updated.

        Returns True if an update was detected, False otherwise.
        If True is returned, callers should clear_model_cache() to force reload.
        """
        if not os.path.exists(self._watch_path):
            return False

        try:
            stat = os.stat(self._watch_path)
            new_hash = f"{stat.st_size}:{stat.st_mtime}"

            if self._current_hash is None:
                # First check after initialization
                self._current_mtime = stat.st_mtime
                self._current_hash = new_hash
                return False

            if new_hash != self._current_hash:
                # Model file changed!
                self._current_mtime = stat.st_mtime
                self._current_hash = new_hash
                self._update_count += 1
                return True

            return False
        except OSError:
            return False

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current model file."""
        if not os.path.exists(self._watch_path):
            return {"exists": False}

        try:
            stat = os.stat(self._watch_path)
            # Try to read companion .meta.json file
            meta_path = self._watch_path.replace(".pth", ".meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)

            return {
                "exists": True,
                "path": self._watch_path,
                "size_mb": stat.st_size / (1024 * 1024),
                "mtime": stat.st_mtime,
                "update_count": self._update_count,
                **meta,
            }
        except (OSError, json.JSONDecodeError):
            return {"exists": True, "path": self._watch_path}
from app.utils.victory_type import derive_victory_type  # noqa: E402


VIOLATION_TYPE_TO_INVARIANT_ID: Dict[str, str] = {
    "S_INVARIANT_DECREASED": "INV-S-MONOTONIC",
    "TOTAL_RINGS_ELIMINATED_DECREASED": "INV-ELIMINATION-MONOTONIC",
    "ACTIVE_NO_MOVES": "INV-ACTIVE-NO-MOVES",
    "ACTIVE_NO_CANDIDATE_MOVES": "INV-ACTIVE-NO-MOVES",
}

MAX_INVARIANT_VIOLATION_SAMPLES = 50

# Last computed timing profile for the most recent soak run. This is
# populated only when --profile-timing is enabled and is consumed by the
# CLI entrypoint when attaching timing data to summary_json payloads.
_LAST_TIMING_PROFILE: Optional[Dict[str, Any]] = None


@dataclass
class GameRecord:
    index: int
    num_players: int
    board_type: str
    engine_mode: str
    seed: Optional[int]
    length: int
    status: str
    winner: Optional[int]
    termination_reason: str
    skipped: bool = False
    invariant_violations_by_type: Dict[str, int] = field(default_factory=dict)
    # Pie-rule diagnostics: how many SWAP_SIDES moves occurred in this game,
    # and whether the pie rule was exercised at least once.
    swap_sides_moves: int = 0
    used_pie_rule: bool = False
    # Standardized victory type categorization per GAME_RECORD_SPEC.md
    victory_type: Optional[str] = None
    stalemate_tiebreaker: Optional[str] = None
    # Training data: moves and initial state for reconstructing games from JSONL
    # These are optional and only included when --include-training-data is set
    moves: Optional[List[Dict[str, Any]]] = None
    initial_state: Optional[Dict[str, Any]] = None
    # Failure diagnostics (populated only for skipped/exception games)
    failure_debug: Optional[Dict[str, Any]] = None
    # Source tracking for data quality auditing
    source: str = "run_self_play_soak.py"
    # DB recording diagnostics (when --record-db is enabled)
    db_recorded: bool = False
    db_game_id: Optional[str] = None
    db_record_error: Optional[str] = None


def _record_invariant_violation(
    violation_type: str,
    state: GameState,
    game_index: int,
    move_index: int,
    per_game_counts: Dict[str, int],
    samples: List[Dict[str, Any]],
    *,
    prev_snapshot: Optional[Dict[str, int]] = None,
    curr_snapshot: Optional[Dict[str, int]] = None,
) -> None:
    """Record a single invariant violation occurrence for soaks.

    This is a non-throwing mirror of the TS soak harness' violation
    accounting: it increments per-game counts keyed by violation type and,
    while under a bounded limit, appends a small diagnostic sample that can
    be serialised in the final soak summary.
    """
    per_game_counts[violation_type] = (
        per_game_counts.get(
            violation_type,
            0,
        )
        + 1
    )

    if len(samples) >= MAX_INVARIANT_VIOLATION_SAMPLES:
        return

    board_type_value = state.board_type.value if hasattr(state.board_type, "value") else state.board_type

    entry: Dict[str, Any] = {
        "type": violation_type,
        "invariant_id": VIOLATION_TYPE_TO_INVARIANT_ID.get(violation_type),
        "game_index": game_index,
        "move_index": move_index,
        "board_type": board_type_value,
        "game_status": state.game_status.value,
        "current_player": state.current_player,
        "current_phase": state.current_phase.value,
    }

    if prev_snapshot is not None:
        entry["before"] = prev_snapshot
    if curr_snapshot is not None:
        entry["after"] = curr_snapshot

    samples.append(entry)

    # Emit a lightweight Prometheus metric for Python-side invariant
    # violations. This mirrors the TS orchestrator invariant metrics and
    # allows dashboards/alerts to slice by invariant_id. Metrics must never
    # break soak runs, so failures are swallowed.
    invariant_id = VIOLATION_TYPE_TO_INVARIANT_ID.get(violation_type)
    if invariant_id:
        try:
            PYTHON_INVARIANT_VIOLATIONS.labels(
                invariant_id=invariant_id,
                type=violation_type,
            ).inc()
        except Exception:
            # Metrics emission is best-effort only.
            pass


def _append_state_to_jsonl(path: str, state: GameState) -> None:
    """Append a single GameState JSON document as one line to a JSONL file.

    The file is opened in append mode so that repeated soak runs can build up
    a larger evaluation pool over time. The directory portion of `path` is
    created if it does not already exist.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    # Use the Pydantic model's JSON serialisation to ensure
    # round-trippable payloads.
    payload = state.model_dump_json()  # type: ignore[attr-defined]
    with open(path, "a", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")


def _validate_history_trace(
    initial_state: GameState,
    moves: List[Move],
) -> Tuple[bool, Optional[str]]:
    """
    Lightweight trace-mode validation of a recorded move list.

    Replays the moves through GameEngine.apply_move(trace_mode=True) to catch
    actor/phase mismatches (e.g., "not your turn") before committing a game
    to the DB. Returns (ok, error_str).
    """
    try:
        state = initial_state
        for mv in moves:
            state = GameEngine.apply_move(state, mv, trace_mode=True)
        return True, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"invalid_history:{type(exc).__name__}:{exc}"


def _parse_board_type(name: str) -> BoardType:
    name = name.lower()
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hexagonal":
        return BoardType.HEXAGONAL
    raise SystemExit(f"Unknown board type: {name!r} " "(expected square8|square19|hexagonal)")


def _canonical_termination_reason(
    state: GameState,
    fallback: str,
    *,
    max_moves: int | None = None,
) -> str:
    """
    Map a completed GameState to a canonical termination reason.
    """
    status_str = state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status)
    if status_str != "completed":
        return f"status:{status_str}" if status_str else fallback

    vtype, _tb = derive_victory_type(state, max_moves)
    # Historical termination_reason taxonomy used "elimination" while the
    # canonical victory_type module uses "ring_elimination".
    if vtype == "ring_elimination":
        vtype = "elimination"
    return f"status:completed:{vtype}" if vtype else fallback


def _resolve_default_nn_model_id(
    board_type: BoardType,
    num_players: int,
) -> Optional[str]:
    """Resolve a safe default nn_model_id prefix for neural-enabled tiers.

    Canonical NN checkpoints currently exist only for square8 2-player games.
    To keep mixed/nn-only soaks robust:
    - Prefer the active improvement-loop best checkpoint when present.
    - Prefer v5 (v3-family) square8 2p checkpoints when present.
    - Prefer v4 (v2-family) square8 2p checkpoints when present.
    - Prefer v3 (legacy) square8 2p checkpoints when present.
    - Fall back to the stable sq8_2p baseline prefix otherwise.
    - Return None for other boards/player-counts so callers can disable NN
      instead of crashing.
    """
    import glob

    models_dir = os.path.join(ROOT, "models")

    # Prefer the improvement loop's active best checkpoint when present so
    # canonical self-play tracks the currently promoted model.
    best_prefix = f"{board_type.value}_{num_players}p_best"
    best_matches = glob.glob(os.path.join(models_dir, f"{best_prefix}*.pth"))
    if any(os.path.getsize(p) > 0 for p in best_matches):
        return best_prefix

    if board_type == BoardType.SQUARE8 and num_players == 2:
        for prefix in (
            "ringrift_v5_sq8_2p_2xh100",
            "ringrift_v4_sq8_2p",
            "ringrift_v3_sq8_2p",
            "sq8_2p_nn_baseline",
        ):
            matches = glob.glob(os.path.join(models_dir, f"{prefix}*.pth"))
            matches = [p for p in matches if os.path.getsize(p) > 0]
            if matches:
                return prefix
        return "sq8_2p_nn_baseline"
    return None


def _scan_recent_nn_pool_model_ids(
    *,
    board_type: BoardType,
    num_players: int,
    pool_dir: Optional[str],
    pool_size: int,
    exclude_ids: Optional[set[str]] = None,
) -> List[str]:
    """Best-effort scan for recent NN checkpoints usable as a diversity pool."""
    if pool_size <= 0:
        return []

    exclude_ids = exclude_ids or set()
    base_dir = Path(pool_dir).expanduser().resolve() if pool_dir else (Path(ROOT) / "models").resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    board_tokens: List[str]
    if board_type == BoardType.SQUARE8:
        board_tokens = ["sq8", "square8"]
    elif board_type == BoardType.SQUARE19:
        board_tokens = ["sq19", "square19", "19x19"]
    else:  # HEXAGONAL
        board_tokens = ["hex", "hexagonal"]

    candidates: List[Path] = []
    for path in base_dir.glob("*.pth"):
        name = path.name.lower()
        if name.endswith("_mps.pth"):
            continue
        if name.startswith("ringrift_best_"):
            continue
        if f"{num_players}p" not in name:
            continue
        if not any(token in name for token in board_tokens):
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        candidates.append(path)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    pool: List[str] = []
    for path in candidates:
        model_id = path.stem
        if model_id in exclude_ids:
            continue
        pool.append(model_id)
        if len(pool) >= pool_size:
            break

    return pool


def _build_mixed_ai_pool(
    game_index: int,
    player_numbers: List[int],
    engine_mode: str,
    base_seed: Optional[int],
    board_type: BoardType,
    difficulty_band: str = "canonical",
    heuristic_weights: Optional[Dict[str, float]] = None,
    nn_pool_size: int = 0,
    nn_pool_dir: Optional[str] = None,
    think_time_override: Optional[int] = None,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """Construct per-player AI instances for a single game.

    For ``engine_mode == 'descent-only'`` we use DescentAI only.

    For ``engine_mode == 'mixed'`` we sample from the canonical difficulty
    ladder while forcing ``think_time=0`` for faster soaks. When
    ``difficulty_band == 'light'``, we restrict the ladder to a lighter
    subset (Random/Heuristic/low-depth Minimax) to reduce memory and
    runtime for long strict-invariant soaks.

    The ``board_type`` argument is used together with
    ``TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD`` to select the appropriate
    heuristic evaluation mode (``"full"`` vs ``"light"``) for any
    HeuristicAI instances in the pool.

    Returns:
        Tuple of (ai_by_player, ai_metadata) where:
        - ai_by_player: Dict mapping player_number -> AI instance
        - ai_metadata: Dict with per-player AI info for DB recording, e.g.:
            player_{pnum}_ai_type, player_{pnum}_difficulty, player_{pnum}_profile_id
    """

    ai_by_player: Dict[int, Any] = {}
    ai_metadata: Dict[str, Any] = {}

    # In soak contexts we want near-zero search budgets for high-tier engines.
    # The core AI implementations treat think_time <= 0 as "use default budget",
    # so we pass a tiny positive value instead.
    #
    # However, for larger boards (square19, hexagonal), the near-zero budget
    # leads to high stalemate rates as the AI can't find winning lines.
    # Use board-specific think times to balance speed vs stalemate reduction.
    if think_time_override is not None:
        soak_think_time_ms = think_time_override
    else:
        soak_think_time_ms = {
            BoardType.SQUARE8: 1,     # Fast, small board
            BoardType.SQUARE19: 50,   # Larger board needs more search time
            BoardType.HEXAGONAL: 50,  # Larger board needs more search time
        }.get(board_type, 1)

    if engine_mode == "descent-only":
        from app.ai.descent_ai import DescentAI  # type: ignore

        for pnum in player_numbers:
            # For soak-style runs, default to heuristic-only Descent unless
            # callers explicitly opt into neural-network evaluation via
            # AIConfig.use_neural_net or environment flags. This keeps
            # long self-play jobs from eagerly loading heavy CNN weights on
            # developer machines while preserving backwards-compatible
            # behaviour for other DescentAI callers.
            cfg = AIConfig(
                difficulty=5,
                think_time=soak_think_time_ms,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=False,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
            # Record AI metadata for this player
            ai_metadata[f"player_{pnum}_ai_type"] = "descent"
            ai_metadata[f"player_{pnum}_difficulty"] = 5
        return ai_by_player, ai_metadata

    if engine_mode == "random-only":
        from app.ai.random_ai import RandomAI  # type: ignore

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=1,
                think_time=soak_think_time_ms,
                randomness=1.0,
                rngSeed=(base_seed or 0) + pnum + game_index,
            )
            ai_by_player[pnum] = RandomAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "random"
            ai_metadata[f"player_{pnum}_difficulty"] = 1
        return ai_by_player, ai_metadata

    if engine_mode == "heuristic-only":
        from app.ai.heuristic_ai import HeuristicAI  # type: ignore
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(
            board_type,
            "full",
        )

        # If custom weights provided, register them as a dynamic profile
        custom_profile_id: Optional[str] = None
        if heuristic_weights:
            custom_profile_id = "_soak_custom_weights"
            HEURISTIC_WEIGHT_PROFILES[custom_profile_id] = heuristic_weights

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=2,
                think_time=soak_think_time_ms,
                randomness=0.15,  # Increased for training diversity
                rngSeed=(base_seed or 0) + pnum + game_index,
                heuristic_eval_mode=heuristic_eval_mode,
                heuristic_profile_id=custom_profile_id,  # Use custom weights if provided
                weight_noise=0.1,  # 10% weight noise for evaluation diversity
            )
            ai_by_player[pnum] = HeuristicAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "heuristic"
            ai_metadata[f"player_{pnum}_difficulty"] = 2
            if custom_profile_id:
                ai_metadata[f"player_{pnum}_heuristic_profile"] = custom_profile_id
        return ai_by_player, ai_metadata

    if engine_mode == "minimax-only":
        from app.ai.minimax_ai import MinimaxAI  # type: ignore

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=5,  # Mid-tier Minimax difficulty
                think_time=soak_think_time_ms,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=False,
            )
            ai_by_player[pnum] = MinimaxAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "minimax"
            ai_metadata[f"player_{pnum}_difficulty"] = 5
        return ai_by_player, ai_metadata

    if engine_mode == "mcts-only":
        from app.ai.mcts_ai import MCTSAI  # type: ignore

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=8,  # MCTS difficulty band
                think_time=soak_think_time_ms,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=False,
            )
            ai_by_player[pnum] = MCTSAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "mcts"
            ai_metadata[f"player_{pnum}_difficulty"] = 8
        return ai_by_player, ai_metadata

    if engine_mode == "nn-only":
        # Neural-net enabled: Descent + MCTS + NNUE Minimax with neural networks
        from app.ai.descent_ai import DescentAI  # type: ignore

        default_nn_model_id = _resolve_default_nn_model_id(
            board_type,
            len(player_numbers),
        )
        if default_nn_model_id is None:
            logger.warning(
                "nn-only mode requested but no NN checkpoint is available for "
                "board=%s players=%s; falling back to heuristic-only Descent.",
                board_type.value,
                len(player_numbers),
            )

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=10,
                think_time=soak_think_time_ms,
                randomness=0.05,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=default_nn_model_id is not None,
                nn_model_id=default_nn_model_id,
                allow_fresh_weights=False,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
            ai_metadata[f"player_{pnum}_difficulty"] = 10
        return ai_by_player, ai_metadata

    if engine_mode == "best-vs-pool":
        # Best model vs a pool of recent checkpoints (evaluation-diverse self-play).
        from app.ai.descent_ai import DescentAI  # type: ignore

        best_model_id = _resolve_default_nn_model_id(board_type, len(player_numbers))
        if best_model_id is None:
            logger.warning(
                "best-vs-pool mode requested but no NN checkpoint is available for "
                "board=%s players=%s; falling back to heuristic-only Descent.",
                board_type.value,
                len(player_numbers),
            )

        pool_ids = _scan_recent_nn_pool_model_ids(
            board_type=board_type,
            num_players=len(player_numbers),
            pool_dir=nn_pool_dir,
            pool_size=int(nn_pool_size or 0),
            exclude_ids={best_model_id} if best_model_id else None,
        )
        if best_model_id is not None and not pool_ids:
            pool_ids = [best_model_id]

        best_seat = player_numbers[game_index % len(player_numbers)] if player_numbers else 1

        for pnum in player_numbers:
            if best_model_id is None:
                use_nn = False
                nn_model_id = None
            else:
                use_nn = True
                if int(pnum) == int(best_seat):
                    nn_model_id = best_model_id
                else:
                    nn_model_id = random.choice(pool_ids) if pool_ids else best_model_id

            cfg = AIConfig(
                difficulty=10,
                think_time=soak_think_time_ms,
                randomness=0.05,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=use_nn,
                nn_model_id=nn_model_id,
                allow_fresh_weights=False,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "descent_best_vs_pool"
            ai_metadata[f"player_{pnum}_difficulty"] = 10
            if nn_model_id:
                ai_metadata[f"player_{pnum}_nn_model_id"] = nn_model_id
            if int(pnum) == int(best_seat) and best_model_id:
                ai_metadata[f"player_{pnum}_is_best"] = True

        return ai_by_player, ai_metadata

    if engine_mode == "nn-vs-mcts":
        # Asymmetric: Neural network player vs MCTS player (tournament-style)
        from app.ai.descent_ai import DescentAI  # type: ignore
        from app.ai.mcts_ai import MCTSAI  # type: ignore

        default_nn_model_id = _resolve_default_nn_model_id(board_type, len(player_numbers))
        if default_nn_model_id is None:
            logger.warning(
                "nn-vs-mcts mode requested but no NN checkpoint available; "
                "falling back to heuristic Descent vs MCTS."
            )

        # Assign NN to first half of players, MCTS to second half
        nn_seat_count = max(1, len(player_numbers) // 2)
        for idx, pnum in enumerate(player_numbers):
            if idx < nn_seat_count:
                # NN player (Descent with neural network)
                cfg = AIConfig(
                    difficulty=10,
                    think_time=soak_think_time_ms,
                    randomness=0.05,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=default_nn_model_id is not None,
                    nn_model_id=default_nn_model_id,
                    allow_fresh_weights=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
                ai_metadata[f"player_{pnum}_difficulty"] = 10
                if default_nn_model_id:
                    ai_metadata[f"player_{pnum}_nn_model_id"] = default_nn_model_id
            else:
                # MCTS player
                cfg = AIConfig(
                    difficulty=8,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = MCTSAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "mcts"
                ai_metadata[f"player_{pnum}_difficulty"] = 8
        return ai_by_player, ai_metadata

    if engine_mode == "nn-vs-minimax":
        # Asymmetric: Neural network player vs Minimax player (tournament-style)
        from app.ai.descent_ai import DescentAI  # type: ignore
        from app.ai.minimax_ai import MinimaxAI  # type: ignore

        default_nn_model_id = _resolve_default_nn_model_id(board_type, len(player_numbers))
        if default_nn_model_id is None:
            logger.warning(
                "nn-vs-minimax mode requested but no NN checkpoint available; "
                "falling back to heuristic Descent vs Minimax."
            )

        # Assign NN to first half of players, Minimax to second half
        nn_seat_count = max(1, len(player_numbers) // 2)
        for idx, pnum in enumerate(player_numbers):
            if idx < nn_seat_count:
                # NN player (Descent with neural network)
                cfg = AIConfig(
                    difficulty=10,
                    think_time=soak_think_time_ms,
                    randomness=0.05,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=default_nn_model_id is not None,
                    nn_model_id=default_nn_model_id,
                    allow_fresh_weights=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
                ai_metadata[f"player_{pnum}_difficulty"] = 10
                if default_nn_model_id:
                    ai_metadata[f"player_{pnum}_nn_model_id"] = default_nn_model_id
            else:
                # Minimax player
                cfg = AIConfig(
                    difficulty=6,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = MinimaxAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "minimax"
                ai_metadata[f"player_{pnum}_difficulty"] = 6
        return ai_by_player, ai_metadata

    if engine_mode == "nn-vs-descent":
        # Asymmetric: Neural network player vs Descent without NN (tournament-style)
        from app.ai.descent_ai import DescentAI  # type: ignore

        default_nn_model_id = _resolve_default_nn_model_id(board_type, len(player_numbers))
        if default_nn_model_id is None:
            logger.warning(
                "nn-vs-descent mode requested but no NN checkpoint available; "
                "falling back to all heuristic Descent."
            )

        # Assign NN to first half of players, heuristic Descent to second half
        nn_seat_count = max(1, len(player_numbers) // 2)
        for idx, pnum in enumerate(player_numbers):
            if idx < nn_seat_count:
                # NN player (Descent with neural network)
                cfg = AIConfig(
                    difficulty=10,
                    think_time=soak_think_time_ms,
                    randomness=0.05,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=default_nn_model_id is not None,
                    nn_model_id=default_nn_model_id,
                    allow_fresh_weights=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
                ai_metadata[f"player_{pnum}_difficulty"] = 10
                if default_nn_model_id:
                    ai_metadata[f"player_{pnum}_nn_model_id"] = default_nn_model_id
            else:
                # Heuristic Descent player (no neural network)
                cfg = AIConfig(
                    difficulty=5,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_heuristic"
                ai_metadata[f"player_{pnum}_difficulty"] = 5
        return ai_by_player, ai_metadata

    if engine_mode == "tournament-varied":
        # Tournament-style: Each player gets a different AI type (max variety)
        from app.ai.descent_ai import DescentAI  # type: ignore
        from app.ai.mcts_ai import MCTSAI  # type: ignore
        from app.ai.minimax_ai import MinimaxAI  # type: ignore

        default_nn_model_id = _resolve_default_nn_model_id(board_type, len(player_numbers))

        # Rotate through AI types: NN-Descent, MCTS, Minimax, heuristic-Descent
        ai_type_rotation = ["nn_descent", "mcts", "minimax", "descent_heuristic"]

        for idx, pnum in enumerate(player_numbers):
            ai_type = ai_type_rotation[idx % len(ai_type_rotation)]

            if ai_type == "nn_descent":
                cfg = AIConfig(
                    difficulty=10,
                    think_time=soak_think_time_ms,
                    randomness=0.05,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=default_nn_model_id is not None,
                    nn_model_id=default_nn_model_id,
                    allow_fresh_weights=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
                ai_metadata[f"player_{pnum}_difficulty"] = 10
                if default_nn_model_id:
                    ai_metadata[f"player_{pnum}_nn_model_id"] = default_nn_model_id
            elif ai_type == "mcts":
                cfg = AIConfig(
                    difficulty=8,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = MCTSAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "mcts"
                ai_metadata[f"player_{pnum}_difficulty"] = 8
            elif ai_type == "minimax":
                cfg = AIConfig(
                    difficulty=6,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = MinimaxAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "minimax"
                ai_metadata[f"player_{pnum}_difficulty"] = 6
            else:  # descent_heuristic
                cfg = AIConfig(
                    difficulty=5,
                    think_time=soak_think_time_ms,
                    randomness=0.1,
                    rngSeed=(base_seed or 0) + pnum + game_index,
                    use_neural_net=False,
                )
                ai_by_player[pnum] = DescentAI(pnum, cfg)
                ai_metadata[f"player_{pnum}_ai_type"] = "descent_heuristic"
                ai_metadata[f"player_{pnum}_difficulty"] = 5
        return ai_by_player, ai_metadata

    # Diverse AI modes - all 11 AI types with weighted distribution
    if engine_mode in ("diverse", "diverse-cpu"):
        try:
            from app.training.diverse_ai_config import (
                get_weighted_ai_type,
                GPU_OPTIMIZED_WEIGHTS,
                CPU_OPTIMIZED_WEIGHTS,
                get_diverse_matchups,
                DiverseAIConfig,
            )
        except ImportError:
            raise SystemExit(
                "Diverse AI mode requires app.training.diverse_ai_config module"
            )

        use_gpu = engine_mode == "diverse"
        weights = GPU_OPTIMIZED_WEIGHTS if use_gpu else CPU_OPTIMIZED_WEIGHTS
        num_players = len(player_numbers)
        diverse_config = DiverseAIConfig(
            board_type=board_type.name.lower(),
            num_players=num_players,
            use_gpu=use_gpu,
        )

        # Get diverse matchups for this game
        matchups = get_diverse_matchups(num_players=num_players, num_matchups=1, config=diverse_config)
        matchup = matchups[0] if matchups else None

        ai_by_player: Dict[int, Any] = {}
        ai_metadata: Dict[str, Any] = {"engine_mode": engine_mode, "use_gpu": use_gpu}

        for pnum in range(1, num_players + 1):
            if matchup:
                ai_types_list = matchup.ai_types
                ai_type_str = ai_types_list[pnum - 1] if pnum <= len(ai_types_list) else get_weighted_ai_type(weights)
            else:
                ai_type_str = get_weighted_ai_type(weights)

            ai_metadata[f"player_{pnum}_ai_type"] = ai_type_str

            # Map string to AIType enum and create AI instance
            ai_type = AIType(ai_type_str)
            difficulty = 5  # Default difficulty

            # Create AI config and instance
            cfg = AIConfig(
                difficulty=difficulty,
                randomness=0.05,
                think_time=500,
                ai_type=ai_type,
            )

            ai = _create_ai_instance(ai_type, pnum, cfg)
            ai_by_player[pnum] = ai
            ai_metadata[f"player_{pnum}_difficulty"] = difficulty

        return ai_by_player, ai_metadata

    # Single AI type modes (new diverse AI types)
    single_ai_mode_map = {
        "gpu-minimax-only": AIType.GPU_MINIMAX,
        "maxn-only": AIType.MAXN,
        "brs-only": AIType.BRS,
        "policy-only": AIType.POLICY_ONLY,
        "gumbel-mcts-only": AIType.GUMBEL_MCTS,
        "neural-demo-only": AIType.NEURAL_DEMO,
    }

    if engine_mode in single_ai_mode_map:
        ai_type = single_ai_mode_map[engine_mode]
        ai_by_player: Dict[int, Any] = {}
        ai_metadata: Dict[str, Any] = {"engine_mode": engine_mode}

        for pnum in range(1, num_players + 1):
            difficulty = 5
            cfg = AIConfig(
                difficulty=difficulty,
                randomness=0.05,
                think_time=500,
                ai_type=ai_type,
            )
            ai = _create_ai_instance(ai_type, pnum, cfg)
            ai_by_player[pnum] = ai
            ai_metadata[f"player_{pnum}_ai_type"] = ai_type.value
            ai_metadata[f"player_{pnum}_difficulty"] = difficulty

        return ai_by_player, ai_metadata

    # mixed mode
    if engine_mode != "mixed":
        raise SystemExit(
            "engine_mode must be one of: descent-only, mixed, random-only, "
            "heuristic-only, minimax-only, mcts-only, nn-only, best-vs-pool, "
            "nn-vs-mcts, nn-vs-minimax, nn-vs-descent, tournament-varied, "
            "diverse, diverse-cpu, gpu-minimax-only, maxn-only, brs-only, "
            "policy-only, gumbel-mcts-only, neural-demo-only; "
            f"got {engine_mode!r}"
        )

    # Difficulty presets chosen to cover the canonical ladder while keeping
    # runtime reasonable on square8.
    difficulty_choices = [
        1,  # Random
        2,  # Heuristic
        4,
        5,
        6,  # Minimax band
        7,
        8,  # MCTS band
        9,
        10,  # Descent band
    ]

    if difficulty_band == "light":
        # Lighter band for memory-/time-conscious soaks: Random,
        # Heuristic, and (for square8 only) low-depth Minimax.
        #
        # Note: square19/hex move generation is expensive enough that even
        # shallow Minimax can dominate runtime. For those boards, keep the
        # "light" band strictly random/heuristic so canonical parity gates
        # and debuggable self-play DB generation remain practical.
        if board_type in {BoardType.SQUARE19, BoardType.HEXAGONAL}:
            difficulty_choices = [1, 2]
        else:
            difficulty_choices = [
                1,
                2,
                4,
                5,
            ]

    if base_seed is not None:
        game_rng = random.Random(base_seed + game_index)
    else:
        game_rng = random.Random()

    for pnum in player_numbers:
        difficulty = game_rng.choice(difficulty_choices)
        profile = _get_difficulty_profile(difficulty)
        ai_type = profile["ai_type"]

        heuristic_profile_id = None
        nn_model_id = None
        heuristic_eval_mode = None
        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")
            heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(
                board_type,
                "full",
            )

        # Neural tiers (D4+ / D6+ / Descent) should use a compatible default NN
        # checkpoint when available. If none exists for this board, disable NN
        # to avoid crashes during lazy NeuralNetAI initialization.
        use_neural_net = bool(profile.get("use_neural_net", False))
        if use_neural_net:
            best_model_id = _resolve_default_nn_model_id(
                board_type,
                len(player_numbers),
            )
            if best_model_id is None:
                use_neural_net = False
                nn_model_id = None
            else:
                nn_model_id = best_model_id
                pool_size = int(nn_pool_size or 0)
                if pool_size > 0 or nn_pool_dir:
                    pool_ids = _scan_recent_nn_pool_model_ids(
                        board_type=board_type,
                        num_players=len(player_numbers),
                        pool_dir=nn_pool_dir,
                        pool_size=pool_size,
                        exclude_ids={best_model_id},
                    )
                    candidates = [best_model_id] + list(pool_ids)
                    # Use the per-game RNG so selection is deterministic for a given seed.
                    nn_model_id = game_rng.choice(candidates) if candidates else best_model_id

        # Apply weight noise for heuristic AIs to increase training diversity
        weight_noise = 0.1 if ai_type == AIType.HEURISTIC else 0.0

        cfg = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=soak_think_time_ms,
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
            heuristic_eval_mode=heuristic_eval_mode,
            use_neural_net=use_neural_net,
            allow_fresh_weights=False,
            weight_noise=weight_noise,
        )
        ai = _create_ai_instance(ai_type, pnum, cfg)
        ai_by_player[pnum] = ai

        # Record AI metadata for this player
        ai_metadata[f"player_{pnum}_ai_type"] = ai_type.value
        ai_metadata[f"player_{pnum}_difficulty"] = difficulty
        if heuristic_profile_id:
            ai_metadata[f"player_{pnum}_profile_id"] = heuristic_profile_id
        if nn_model_id:
            ai_metadata[f"player_{pnum}_nn_model_id"] = nn_model_id

    return ai_by_player, ai_metadata


def _run_intra_game_gc(
    ai_by_player: Dict[int, Any],
    move_count: int,
    verbose: bool = False,
) -> None:
    """Run lightweight intra-game memory cleanup.

    This clears per-evaluation caches in AI instances without destroying
    the AI instances themselves. The goal is to prevent memory
    accumulation during long games (100+ moves) on large boards.

    Trade-offs:
    - Performance: ~5-15% overhead due to cache rebuilding
    - Correctness: None (AI still produces valid moves)
    - Play strength: Negligible (caches are per-evaluation anyway)
    """
    # Clear any per-move caches in AI instances
    for ai in ai_by_player.values():
        # HeuristicAI and similar classes may have clear_cache() methods
        if hasattr(ai, "clear_evaluation_cache"):
            ai.clear_evaluation_cache()
        # Clear internal state caches if present
        if hasattr(ai, "_cached_visible_stacks"):
            ai._cached_visible_stacks = None
        if hasattr(ai, "_visible_stacks_cache"):
            ai._visible_stacks_cache = {}

    # Run garbage collection but only on generation 0 (fast)
    # This reclaims short-lived objects without full GC overhead
    gc.collect(0)

    if verbose:
        print(
            f"[intra-gc] Cleared AI caches at move {move_count}",
            flush=True,
        )


def run_self_play_soak(
    args: argparse.Namespace,
) -> Tuple[List[GameRecord], List[Dict[str, Any]]]:
    board_type = _parse_board_type(args.board_type)
    num_games = args.num_games
    num_players = args.num_players
    # Auto-derive max_moves from board type and player count if not specified
    max_moves = args.max_moves
    if max_moves is None:
        max_moves = get_theoretical_max_moves(board_type, num_players)
    engine_mode = args.engine_mode
    base_seed = args.seed
    difficulty_band = getattr(args, "difficulty_band", "canonical")
    nn_pool_size = int(getattr(args, "nn_pool_size", 0) or 0)
    nn_pool_dir = getattr(args, "nn_pool_dir", None)

    # Load heuristic weights from CLI args if specified
    heuristic_weights: Optional[Dict[str, float]] = None
    heuristic_weights_file = getattr(args, "heuristic_weights_file", None)
    heuristic_profile = getattr(args, "heuristic_profile", None)
    if heuristic_weights_file and heuristic_profile:
        heuristic_weights = load_weights_from_profile(
            heuristic_weights_file,
            heuristic_profile,
        )
        if heuristic_weights and engine_mode == "heuristic-only":
            print(
                f"[heuristic-weights] Using custom weights for heuristic-only mode",
                flush=True,
            )

    gc_interval = getattr(args, "gc_interval", 5)
    profile_timing = getattr(args, "profile_timing", False)

    # Hot model reload for continuous selfplay
    watch_model_updates = getattr(args, "watch_model_updates", False)
    model_reload_interval = getattr(args, "model_reload_interval", 100)
    model_alias_path = getattr(args, "model_alias_path", None)
    emit_events = getattr(args, "emit_events", False)

    hot_reloader: Optional[HotModelReloader] = None
    if watch_model_updates:
        hot_reloader = HotModelReloader(
            board_type=args.board_type,
            num_players=num_players,
            model_alias_path=model_alias_path,
        )
        print(
            f"[hot-reload] Enabled: watching {hot_reloader.current_model_path} "
            f"(check every {model_reload_interval} games)",
            flush=True,
        )

    # Opening diversity options
    opening_random_moves = getattr(args, "opening_random_moves", 4)
    opening_top_k = getattr(args, "opening_top_k", 0)
    if opening_random_moves > 0:
        print(
            f"[opening-diversity] Enabled: first {opening_random_moves} moves will be "
            f"{'randomly selected' if opening_top_k == 0 else f'sampled from top-{opening_top_k} AI moves'}",
            flush=True,
        )

    # Player bias mitigation via swap_sides probability
    swap_sides_probability = getattr(args, "swap_sides_probability", 0.5)
    if swap_sides_probability > 0 and num_players == 2:
        print(
            f"[player-balance] swap_sides probability: {swap_sides_probability:.0%} "
            f"(helps balance P1/P2 win rates)",
            flush=True,
        )

    # Think time override for stalemate reduction
    think_time_override = getattr(args, "think_time_ms", None)
    if think_time_override is not None:
        print(
            f"[think-time] Override: {think_time_override}ms",
            flush=True,
        )

    # Memory management options
    intra_game_gc_interval = getattr(args, "intra_game_gc_interval", 0)
    streaming_record = getattr(args, "streaming_record", False)
    memory_constrained = getattr(args, "memory_constrained", False)

    # Apply memory-constrained mode defaults
    if memory_constrained:
        if intra_game_gc_interval == 0:
            # Auto-set based on board type
            if board_type == BoardType.HEXAGONAL:
                intra_game_gc_interval = 50
            elif board_type == BoardType.SQUARE19:
                intra_game_gc_interval = 40
            else:
                intra_game_gc_interval = 30
        streaming_record = True
        difficulty_band = "light"
        print(
            f"[memory-constrained] Enabled: intra_gc={intra_game_gc_interval}, "
            f"streaming={streaming_record}, difficulty_band={difficulty_band}",
            flush=True,
        )

    # For large boards, auto-enable intra-game GC if not explicitly set
    if intra_game_gc_interval == 0 and board_type in (
        BoardType.HEXAGONAL,
        BoardType.SQUARE19,
    ):
        # Suggest but don't force - let user opt in
        print(
            f"[memory-warning] Large board {board_type.value} detected. "
            f"Consider using --intra-game-gc-interval=50 or "
            f"--memory-constrained for long games.",
            file=sys.stderr,
        )

    # Optional state-pool configuration. getattr() is used so that existing
    # callers that construct an argparse.Namespace manually without these
    # attributes continue to work unchanged.
    square8_state_pool_output = getattr(
        args,
        "square8_state_pool_output",
        None,
    )
    square8_state_pool_max_states = getattr(
        args,
        "square8_state_pool_max_states",
        500,
    )
    square8_state_pool_sampling_interval = getattr(
        args,
        "square8_state_pool_sampling_interval",
        4,
    )
    if square8_state_pool_sampling_interval <= 0:
        # Guard against accidental zero/negative intervals from CLI. For
        # Square8, a non-positive interval falls back to 1 to preserve
        # historical behaviour.
        square8_state_pool_sampling_interval = 1

    square19_state_pool_output = getattr(
        args,
        "square19_state_pool_output",
        None,
    )
    square19_state_pool_max_states = getattr(
        args,
        "square19_state_pool_max_states",
        0,
    )
    square19_state_pool_sampling_interval = getattr(
        args,
        "square19_state_pool_sampling_interval",
        0,
    )

    hex_state_pool_output = getattr(
        args,
        "hex_state_pool_output",
        None,
    )
    hex_state_pool_max_states = getattr(
        args,
        "hex_state_pool_max_states",
        0,
    )
    hex_state_pool_sampling_interval = getattr(
        args,
        "hex_state_pool_sampling_interval",
        0,
    )

    # Global counters across all games in this soak run.
    square8_state_pool_sampled = 0
    square19_state_pool_sampled = 0
    hex_state_pool_sampled = 0

    # Optional lightweight timing profile across the soak run. When
    # profile_timing is False, the dict remains unused and no extra
    # time measurements are taken in the inner loop.
    timing_totals: Dict[str, float] = {
        "env_reset": 0.0,
        "ai_build": 0.0,
        "move_select": 0.0,
        "env_step": 0.0,
        "db_record": 0.0,
    }
    total_moves_across_games = 0

    # Precompute per-board pool-enable flags so that the inner loop can
    # cheaply skip sampling logic when pools are disabled.
    square8_pool_enabled = bool(
        square8_state_pool_output and square8_state_pool_max_states > 0 and square8_state_pool_sampling_interval > 0
    )
    square19_pool_enabled = bool(
        square19_state_pool_output and square19_state_pool_max_states > 0 and square19_state_pool_sampling_interval > 0
    )
    hex_pool_enabled = bool(
        hex_state_pool_output and hex_state_pool_max_states > 0 and hex_state_pool_sampling_interval > 0
    )

    os.makedirs(os.path.dirname(args.log_jsonl) or ".", exist_ok=True)

    # Resume support: count existing lines in JSONL file to determine starting index
    resume_from_jsonl = getattr(args, "resume_from_jsonl", False)
    checkpoint_interval = getattr(args, "checkpoint_interval", 0)
    start_game_idx = 0
    checkpoint_path = args.log_jsonl + ".checkpoint.json"

    if resume_from_jsonl and os.path.exists(args.log_jsonl):
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            start_game_idx = sum(1 for _ in f)
        if start_game_idx > 0:
            logger.info(f"Resuming from game {start_game_idx} (found {start_game_idx} existing games in {args.log_jsonl})")
        if start_game_idx >= num_games:
            logger.info(f"All {num_games} games already completed. Nothing to do.")
            return [], []

    # Initialize optional game recording database
    # --no-record-db flag overrides --record-db to disable recording
    record_db_path = None if getattr(args, "no_record_db", False) else getattr(args, "record_db", None)
    # Disable canonical history validation for selfplay - Python engine validates moves
    # already and training data doesn't need TS phase alignment
    replay_db = get_or_create_db(record_db_path, enforce_canonical_history=False) if record_db_path else None
    games_recorded = 0
    fail_on_anomaly = bool(getattr(args, "fail_on_anomaly", False))

    # Lean DB mode: skip storing full state history for each move (~100x smaller)
    # Default is True (lean enabled), --no-lean-db disables it
    lean_db_enabled = getattr(args, "lean_db", True) and not getattr(args, "no_lean_db", False)

    # Training data (moves + initial_state) is ALWAYS included in JSONL output.
    # This is mandatory to ensure all game data can be converted to NPZ for training.
    # The --no-include-training-data option has been removed (RR-DATA-QUALITY-2024-12).
    include_training_data = True

    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    records: List[GameRecord] = []
    invariant_violation_samples: List[Dict[str, Any]] = []

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=num_games,
        report_interval_sec=10.0,
        context_label=f"{board_type.value}_{engine_mode}_{num_players}p",
    )

    # Host-level flag: when enabled we must synthesize and apply required
    # bookkeeping moves (no_*_action) instead of treating ANM states as fatal.
    force_bookkeeping_moves = os.getenv(
        "RINGRIFT_FORCE_BOOKKEEPING_MOVES",
        "",
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Helper to write checkpoint file
    def _write_checkpoint(
        games_done: int,
        elapsed_sec: float,
        _records_so_far: List[GameRecord],
    ) -> None:
        checkpoint_data = {
            "games_completed": games_done,
            "total_games": num_games,
            "elapsed_seconds": elapsed_sec,
            "board_type": board_type.value,
            "num_players": num_players,
            "engine_mode": engine_mode,
            "seed": base_seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as ckpt_f:
                json.dump(checkpoint_data, ckpt_f)
        except Exception as e:
            logger.warning(f"Failed to write checkpoint: {e}")

    # Open in append mode if resuming, otherwise write mode
    file_mode = "a" if (resume_from_jsonl and start_game_idx > 0) else "w"
    soak_start_time = time.time()

    with open(args.log_jsonl, file_mode, encoding="utf-8") as log_f:
        # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
        try:
            fcntl.flock(log_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error(f"Cannot acquire lock on {args.log_jsonl} - another process is writing")
            sys.exit(1)

        for game_idx in range(start_game_idx, num_games):
            game_start_time = time.time()

            # Hot model reload check
            if hot_reloader and hot_reloader.should_check(game_idx, model_reload_interval):
                if hot_reloader.check_for_updates():
                    meta = hot_reloader.get_model_metadata()
                    print(
                        f"[hot-reload] Model updated! Reloading... "
                        f"(update #{hot_reloader.update_count}, "
                        f"source: {meta.get('source_model_id', 'unknown')})",
                        flush=True,
                    )
                    # Clear the neural net model cache to force reload on next AI creation
                    clear_model_cache()

            # Periodic resource check (every 50 games) - stop early if 80% limits exceeded
            if game_idx > 0 and game_idx % 50 == 0:
                try:
                    from app.utils.resource_guard import check_memory, check_disk_space
                    if not check_memory(required_gb=2.0, log_warning=False):
                        print(
                            f"[resource-guard] Memory pressure detected at game {game_idx}, "
                            f"stopping early to avoid OOM",
                            flush=True,
                        )
                        break
                    if not check_disk_space(required_gb=1.0, log_warning=False):
                        print(
                            f"[resource-guard] Disk space low at game {game_idx}, "
                            f"stopping early to avoid disk full",
                            flush=True,
                        )
                        break
                except ImportError:
                    pass  # Resource guard not available

            game_seed = None if base_seed is None else base_seed + game_idx
            # RNG for opening diversity (deterministic per-game if seed provided)
            game_rng = random.Random(game_seed) if game_seed is not None else random.Random()
            try:
                if profile_timing:
                    t0 = time.time()
                state: GameState = env.reset(seed=game_seed)
                if profile_timing:
                    timing_totals["env_reset"] += time.time() - t0
            except Exception as exc:  # pragma: no cover - defensive
                rec = GameRecord(
                    index=game_idx,
                    num_players=num_players,
                    board_type=board_type.value,
                    engine_mode=engine_mode,
                    seed=game_seed,
                    length=0,
                    status="error_reset",
                    winner=None,
                    termination_reason=f"reset_exception:{type(exc).__name__}",
                )
                log_f.write(json.dumps(asdict(rec)) + "\n")
                records.append(rec)
                # Record error game for progress reporting
                game_duration = time.time() - game_start_time
                progress_reporter.record_game(moves=0, duration_sec=game_duration)
                continue

            player_numbers = [p.player_number for p in state.players]
            if profile_timing:
                t_ai_start = time.time()
            ai_by_player, per_player_ai_metadata = _build_mixed_ai_pool(
                game_idx,
                player_numbers,
                engine_mode,
                base_seed,
                board_type,
                difficulty_band=difficulty_band,
                heuristic_weights=heuristic_weights,
                nn_pool_size=nn_pool_size,
                nn_pool_dir=nn_pool_dir,
                think_time_override=think_time_override,
            )
            if profile_timing:
                timing_totals["ai_build"] += time.time() - t_ai_start

            move_count = 0
            termination_reason = "unknown"
            last_move = None
            failure_debug: Optional[Dict[str, Any]] = None
            per_game_violations: Dict[str, int] = {}
            swap_sides_moves_for_game = 0
            skipped = False  # track games we drop but continue

            # Game recording: capture initial state and collect moves
            # Also track for JSONL training data if include_training_data is enabled
            should_track_game_data = replay_db or include_training_data
            initial_state_for_recording = state.model_copy(deep=True) if should_track_game_data else None
            game_moves_for_recording: List[Any] = []

            # Initialise S-invariant / elimination snapshot for this game.
            prev_snapshot = compute_progress_snapshot(state)
            prev_S = prev_snapshot["S"]
            prev_eliminated = prev_snapshot["eliminated"]

            while True:
                if state.game_status != GameStatus.ACTIVE:
                    termination_reason = f"status:{state.game_status.value}"
                    break

                current_player = state.current_player
                legal_moves = env.legal_moves()
                if state.current_phase == GamePhase.FORCED_ELIMINATION and any(
                    m.type != MoveType.FORCED_ELIMINATION for m in legal_moves
                ):
                    termination_reason = "illegal_moves_in_forced_elimination"
                    skipped = True
                    print(
                        f"[soak-skip] game {game_idx} surfaced non-FE moves in forced_elimination: "
                        f"{[m.type.value for m in legal_moves]}",
                        file=sys.stderr,
                    )
                    break

                move = None

                if not legal_moves:
                    # No legal moves surfaced. If forced bookkeeping is on, try to
                    # synthesize the required no_* action for this actor to keep
                    # the trace canonical instead of aborting.
                    requirement = GameEngine.get_phase_requirement(
                        state,
                        current_player,
                    )
                    if (
                        requirement is None
                        and force_bookkeeping_moves
                        and state.current_phase
                        in (
                            GamePhase.RING_PLACEMENT,
                            GamePhase.MOVEMENT,
                            GamePhase.LINE_PROCESSING,
                            GamePhase.TERRITORY_PROCESSING,
                            GamePhase.FORCED_ELIMINATION,
                        )
                    ):
                        fallback_req_type = {
                            GamePhase.RING_PLACEMENT: PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED,
                            GamePhase.MOVEMENT: PhaseRequirementType.NO_MOVEMENT_ACTION_REQUIRED,
                            GamePhase.LINE_PROCESSING: PhaseRequirementType.NO_LINE_ACTION_REQUIRED,
                            GamePhase.TERRITORY_PROCESSING: PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED,
                            GamePhase.FORCED_ELIMINATION: PhaseRequirementType.FORCED_ELIMINATION_REQUIRED,
                        }.get(state.current_phase)
                        if fallback_req_type is not None:
                            # For FORCED_ELIMINATION, we need eligible positions
                            eligible_positions: List[Position] = []
                            if fallback_req_type == PhaseRequirementType.FORCED_ELIMINATION_REQUIRED:
                                stacks = BoardManager.get_player_stacks(state.board, current_player)
                                eligible_positions = [
                                    stack.position for stack in stacks.values() if stack.cap_height > 0
                                ]
                            requirement = PhaseRequirement(  # type: ignore[attr-defined]
                                type=fallback_req_type,
                                player=current_player,
                                eligible_positions=eligible_positions,
                            )

                    if requirement is not None:
                        move = GameEngine.synthesize_bookkeeping_move(
                            requirement,
                            state,
                        )

                    # With strict invariant enabled, this should be impossible
                    # for ACTIVE states; if it happens anyway we record it
                    # explicitly as an ACTIVE-no-moves violation.
                    if move is None:
                        _record_invariant_violation(
                            "ACTIVE_NO_MOVES",
                            state,
                            game_idx,
                            move_count,
                            per_game_violations,
                            invariant_violation_samples,
                        )
                        termination_reason = "no_legal_moves_for_current_player"
                        break
                else:
                    ai = ai_by_player.get(current_player)
                    if ai is None:
                        termination_reason = "no_ai_for_current_player"
                        skipped = True
                        break

                    move = None  # Initialize for conditional selection below

                    # Player balance: force swap_sides with probability for 2-player games
                    # This helps balance training data between P1 and P2 perspectives
                    swap_sides_move = None
                    if (
                        swap_sides_probability > 0
                        and num_players == 2
                        and current_player == 2
                        and move_count <= 2  # swap_sides only offered in first few moves
                    ):
                        swap_sides_move = next(
                            (m for m in legal_moves if m.type == MoveType.SWAP_SIDES),
                            None
                        )
                        if swap_sides_move and game_rng.random() < swap_sides_probability:
                            move = swap_sides_move
                            # Skip the rest of move selection
                        else:
                            swap_sides_move = None  # Don't force swap

                    # Opening diversity: randomly select from legal moves for first N moves
                    if move is None and opening_random_moves > 0 and move_count < opening_random_moves:
                        if opening_top_k > 0:
                            # Semi-intelligent: sample from top-K AI moves
                            if profile_timing:
                                t_sel_start = time.time()
                            ai_move = ai.select_move(state)
                            if profile_timing:
                                timing_totals["move_select"] += time.time() - t_sel_start

                            # Get AI's move rankings if available
                            ranked_moves = getattr(ai, "get_ranked_moves", None)
                            if ranked_moves and callable(ranked_moves):
                                try:
                                    top_moves = ranked_moves(state, top_k=opening_top_k)
                                    if top_moves:
                                        move = game_rng.choice(top_moves)
                                    else:
                                        move = game_rng.choice(legal_moves)
                                except Exception:
                                    move = game_rng.choice(legal_moves)
                            else:
                                # Fallback: random from legal moves
                                move = game_rng.choice(legal_moves)
                        else:
                            # Purely random: select any legal move
                            move = game_rng.choice(legal_moves)
                    elif move is None:
                        # Normal AI selection
                        if profile_timing:
                            t_sel_start = time.time()
                        move = ai.select_move(state)
                        if profile_timing:
                            timing_totals["move_select"] += time.time() - t_sel_start

                    if not move:
                        termination_reason = "ai_returned_no_move"
                        skipped = True
                        break

                    # Validate that the AI-selected move is in the legal moves list.
                    # This guards against AI bugs where the AI returns a move that
                    # isn't actually legal for the current player (e.g., place_ring
                    # when the player has 0 rings in hand).
                    move_is_legal = any(
                        m.type == move.type
                        and m.player == move.player
                        and getattr(m, "to", None) == getattr(move, "to", None)
                        and getattr(m, "from_pos", None) == getattr(move, "from_pos", None)
                        for m in legal_moves
                    )
                    if not move_is_legal:
                        termination_reason = f"ai_selected_illegal_move:{move.type.value}"
                        _record_invariant_violation(
                            "AI_ILLEGAL_MOVE",
                            state,
                            game_idx,
                            move_count,
                            per_game_violations,
                            invariant_violation_samples,
                        )
                        skipped = True
                        break

                    # NOTE: swap_sides (pie rule) IS a valid move during ring_placement
                    # when offered by the rules engine. The previous check that rejected
                    # these moves was incorrect and caused 91% of heuristic 2p games to
                    # fail after only 2 moves. The rules engine correctly offers swap_sides
                    # as a legal move when the pie rule is enabled.

                    # Guard against movement/capture moves being returned after the host
                    # has already advanced into decision phases. If this happens, drop
                    # the game to avoid recording a structurally invalid trace.
                    movement_phase_ok = state.current_phase in (
                        GamePhase.MOVEMENT,
                        GamePhase.CAPTURE,
                        GamePhase.CHAIN_CAPTURE,
                    )
                    if (
                        move.type
                        in (
                            MoveType.MOVE_STACK,
                            MoveType.OVERTAKING_CAPTURE,
                            MoveType.CONTINUE_CAPTURE_SEGMENT,
                        )
                        and not movement_phase_ok
                    ):
                        termination_reason = f"illegal_move_for_phase:{state.current_phase.value}:{move.type.value}"
                        skipped = True
                        try:
                            failure_dir = os.path.join(
                                os.path.dirname(args.log_jsonl) or ".",
                                "failures",
                            )
                            os.makedirs(failure_dir, exist_ok=True)
                            failure_path = os.path.join(
                                failure_dir,
                                f"failure_{game_idx}_illegal_move_for_phase.json",
                            )
                            with open(failure_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "game_index": game_idx,
                                        "move_index": move_count,
                                        "current_phase": state.current_phase.value,
                                        "move_type": move.type.value,
                                        "player": move.player,
                                        "state_hash": getattr(state, "zobrist_hash", None),
                                    },
                                    f,
                                )
                        except Exception:
                            pass
                        break

                if state.current_phase == GamePhase.FORCED_ELIMINATION and move.type != MoveType.FORCED_ELIMINATION:
                    termination_reason = "ai_move_not_forced_elimination"
                    skipped = True
                    print(
                        f"[soak-skip] game {game_idx} AI proposed {move.type.value} "
                        "during forced_elimination; skipping game",
                        file=sys.stderr,
                    )
                    break

                # Guard against mis-attributed moves: actor must match the
                # current player when the game is ACTIVE. If an AI returns a
                # move attributed to the wrong player, correct it at the host
                # layer (to keep the trace canonical) while still recording
                # the invariant violation for debugging.
                if state.game_status == GameStatus.ACTIVE and move.player != current_player:
                    _record_invariant_violation(
                        "ACTIVE_WRONG_PLAYER_MOVE",
                        state,
                        game_idx,
                        move_count,
                        per_game_violations,
                        invariant_violation_samples,
                    )
                    move = move.model_copy(update={"player": current_player})

                if move.type == MoveType.SWAP_SIDES:
                    swap_sides_moves_for_game += 1

                # Canonical phase/move guard: if the selected move is not legal
                # for the current phase, fail fast and drop the game instead of
                # recording a mis-ordered trace.
                phase_check = validate_canonical_move(
                    state.current_phase.value,
                    move.type.value,
                )
                if not phase_check.ok:
                    termination_reason = f"phase_move_mismatch:{phase_check.reason}"
                    skipped = True
                    # Log the mismatch for debugging and emit a failure snapshot
                    _record_invariant_violation(
                        "PHASE_MOVE_MISMATCH",
                        state,
                        game_idx,
                        move_count,
                        per_game_violations,
                        invariant_violation_samples,
                    )
                    try:
                        failure_dir = os.path.join(
                            os.path.dirname(args.log_jsonl) or ".",
                            "failures",
                        )
                        os.makedirs(failure_dir, exist_ok=True)
                        failure_path = os.path.join(
                            failure_dir,
                            f"failure_{game_idx}_phase_move_mismatch.json",
                        )
                        with open(failure_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "game_index": game_idx,
                                    "move_index": move_count,
                                    "current_phase": state.current_phase.value,
                                    "move_type": move.type.value,
                                    "reason": phase_check.reason,
                                    "player": move.player,
                                },
                                f,
                            )
                    except Exception:
                        # Never let snapshotting crash the soak loop
                        pass
                    break

                try:
                    if profile_timing:
                        t_step_start = time.time()
                    prev_current_player = state.current_player
                    state, _reward, done, step_info = env.step(move)
                    if profile_timing:
                        timing_totals["env_step"] += time.time() - t_step_start
                    last_move = move
                    # Safety: ensure the recorded move actor matches the
                    # pre-step current player. This guards against AI or
                    # host bugs that might produce mis-attributed moves.
                    if move.player != prev_current_player:
                        termination_reason = "recorded_player_mismatch"
                        _record_invariant_violation(
                            "ACTIVE_WRONG_PLAYER_MOVE",
                            state,
                            game_idx,
                            move_count,
                            per_game_violations,
                            invariant_violation_samples,
                        )
                        skipped = True
                        break
                    # Collect move for game recording (DB or JSONL training data).
                    # Also include any bookkeeping moves (e.g., no_territory_action)
                    # that the host/rules stack may have appended based on phase
                    # requirements per RR-CANON-R075/R076. These are critical for
                    # TS↔Python replay parity.
                    if should_track_game_data:
                        # Normalize moveNumber for recording. Some AI paths may
                        # return legal moves with stale move_number metadata.
                        record_idx = len(game_moves_for_recording) + 1
                        if hasattr(move, "model_copy"):
                            try:
                                move = move.model_copy(update={"move_number": record_idx})  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        game_moves_for_recording.append(move)
                        auto_moves = step_info.get("auto_generated_moves", [])
                        if auto_moves:
                            # Auto-generated moves may include bookkeeping moves
                            # for DIFFERENT players after a turn transition. This
                            # is expected and required for canonical recordings
                            # (e.g., new player has 0 rings → no_placement_action).
                            # We no longer reject cross-player auto-generated moves.
                            for auto_move in auto_moves:
                                record_idx = len(game_moves_for_recording) + 1
                                if hasattr(auto_move, "model_copy"):
                                    try:
                                        auto_move = auto_move.model_copy(  # type: ignore[attr-defined]
                                            update={"move_number": record_idx}
                                        )
                                    except Exception:
                                        pass
                                game_moves_for_recording.append(auto_move)
                    if done:
                        # If the env terminated but the rules engine still reports
                        # ACTIVE, treat it as an env-level cutoff to avoid recording
                        # a partial turn.
                        if state.game_status != GameStatus.ACTIVE:
                            termination_reason = _canonical_termination_reason(
                                state,
                                termination_reason or "status:completed",
                                max_moves=max_moves,
                            )
                            # Completed normally; do not mark skipped.
                        else:
                            termination_reason = "env_done_flag"
                            skipped = True
                        break
                except Exception as exc:  # pragma: no cover - defensive
                    import traceback

                    print(f"[DEBUG] Step exception: {exc}")
                    traceback.print_exc()
                    try:
                        failure_debug = {
                            "kind": "step_exception",
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "pre_step": {
                                "phase": getattr(state.current_phase, "value", str(state.current_phase)),
                                "player": getattr(state, "current_player", None),
                                "move_count": move_count,
                            },
                            "selected_move": (
                                move.model_dump(mode="json")  # type: ignore[attr-defined]
                                if move is not None and hasattr(move, "model_dump")
                                else None
                            ),
                            "legal_moves_surface": [
                                {"type": mv.type.value, "player": mv.player} for mv in (legal_moves or [])
                            ],
                            "traceback": traceback.format_exc(limit=50),
                        }
                        try:
                            from app.rules.fsm import FSMValidationError  # type: ignore

                            if isinstance(exc, FSMValidationError):
                                failure_debug["fsm"] = {
                                    "code": getattr(exc, "code", None),
                                    "message": getattr(exc, "message", None),
                                    "phase": getattr(exc, "current_phase", None),
                                    "move_type": getattr(exc, "move_type", None),
                                    "player": getattr(exc, "player", None),
                                }
                        except Exception:
                            pass
                    except Exception:
                        # Never let debug capture crash the soak loop.
                        failure_debug = None
                    termination_reason = f"step_exception:{type(exc).__name__}"
                    skipped = True
                    state = state  # keep last known state
                    break

                move_count += 1
                if profile_timing:
                    total_moves_across_games += 1

                # Progress invariants:
                # INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC
                curr_snapshot = compute_progress_snapshot(state)
                curr_S = curr_snapshot["S"]
                curr_eliminated = curr_snapshot["eliminated"]

                if curr_S < prev_S:
                    _record_invariant_violation(
                        "S_INVARIANT_DECREASED",
                        state,
                        game_idx,
                        move_count,
                        per_game_violations,
                        invariant_violation_samples,
                        prev_snapshot=prev_snapshot,
                        curr_snapshot=curr_snapshot,
                    )

                if curr_eliminated < prev_eliminated:
                    _record_invariant_violation(
                        "TOTAL_RINGS_ELIMINATED_DECREASED",
                        state,
                        game_idx,
                        move_count,
                        per_game_violations,
                        invariant_violation_samples,
                        prev_snapshot=prev_snapshot,
                        curr_snapshot=curr_snapshot,
                    )

                prev_snapshot = curr_snapshot
                prev_S = curr_S
                prev_eliminated = curr_eliminated

                # ACTIVE-no-moves invariant:
                # INV-ACTIVE-NO-MOVES (global actions, R2xx cluster)
                if state.game_status == GameStatus.ACTIVE and ga.is_anm_state(state):
                    _record_invariant_violation(
                        "ACTIVE_NO_CANDIDATE_MOVES",
                        state,
                        game_idx,
                        move_count,
                        per_game_violations,
                        invariant_violation_samples,
                    )

                # Optional state-pool sampling for mid-/late-game snapshots.
                #
                # Recommended soak configuration for generating evaluation pools:
                # use a long max_moves, sample every N moves, and cap outputs.
                if (
                    square8_pool_enabled
                    and state.board_type == BoardType.SQUARE8
                    and square8_state_pool_sampled < square8_state_pool_max_states
                    and move_count % square8_state_pool_sampling_interval == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, square8_state_pool_output),
                            state,
                        )
                        square8_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        print(
                            "[square8-state-pool] Failed to "
                            "serialise/write state "
                            f"for game {game_idx}, move {move_count}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )

                if (
                    square19_pool_enabled
                    and state.board_type == BoardType.SQUARE19
                    and square19_state_pool_sampled < square19_state_pool_max_states
                    and move_count % square19_state_pool_sampling_interval == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, square19_state_pool_output),
                            state,
                        )
                        square19_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        print(
                            "[square19-state-pool] Failed to "
                            "serialise/write state "
                            f"for game {game_idx}, move {move_count}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )

                if (
                    hex_pool_enabled
                    and state.board_type == BoardType.HEXAGONAL
                    and hex_state_pool_sampled < hex_state_pool_max_states
                    and move_count % hex_state_pool_sampling_interval == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, hex_state_pool_output),
                            state,
                        )
                        hex_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        print(
                            "[hex-state-pool] Failed to "
                            "serialise/write state "
                            f"for game {game_idx}, move {move_count}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )

                if move_count >= max_moves:
                    termination_reason = "max_moves_reached"
                    # Log error/warning for games hitting max_moves without a winner
                    theoretical_max = get_theoretical_max_moves(board_type, num_players)
                    if state.winner is None:
                        if move_count >= theoretical_max:
                            print(
                                f"ERROR: GAME_NON_TERMINATION [game {game_idx}] "
                                f"Game exceeded theoretical maximum moves without a winner. "
                                f"board_type={board_type.value}, num_players={num_players}, "
                                f"move_count={move_count}, max_moves={max_moves}, "
                                f"theoretical_max={theoretical_max}, "
                                f"game_status={state.game_status.value}, winner={state.winner}",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"WARNING: GAME_MAX_MOVES_CUTOFF [game {game_idx}] "
                                f"Game hit max_moves limit without a winner. "
                                f"board_type={board_type.value}, num_players={num_players}, "
                                f"move_count={move_count}, max_moves={max_moves}, "
                                f"theoretical_max={theoretical_max}, "
                                f"game_status={state.game_status.value}, winner={state.winner}",
                                file=sys.stderr,
                            )
                    break

                if done:
                    termination_reason = "env_done_flag"
                    break

                # Intra-game memory cleanup for long games on large boards
                # This prevents OOM within a single game
                if intra_game_gc_interval > 0 and move_count % intra_game_gc_interval == 0:
                    _run_intra_game_gc(
                        ai_by_player,
                        move_count,
                        verbose=(args.verbose and args.verbose >= 2),
                    )

            # For problematic terminations, capture a minimal snapshot of the
            # final GameState + last Move so they can be turned into explicit
            # regression fixtures. This now includes env_done_flag skips and
            # other skipped cases so we can inspect phase requirements and
            # legal moves when the host failed to synthesize bookkeeping.
            if (
                termination_reason
                in (
                    "no_legal_moves_for_current_player",
                    "env_done_flag",
                )
                or termination_reason.startswith("step_exception:RuntimeError")
                or skipped
            ):
                try:
                    failure_dir = os.path.join(
                        os.path.dirname(args.log_jsonl) or ".",
                        "failures",
                    )
                    os.makedirs(failure_dir, exist_ok=True)

                    try:
                        state_payload = state.model_dump(
                            mode="json",
                        )  # type: ignore[attr-defined]
                    except Exception:
                        state_payload = None

                    try:
                        last_move_payload = (
                            last_move.model_dump(
                                mode="json",
                            )  # type: ignore[attr-defined]
                            if last_move is not None
                            else None
                        )
                    except Exception:
                        last_move_payload = None

                    # Include phase requirement and legal moves for the active player
                    # to aid bookkeeping debugging without touching core rules.
                    requirement_payload = None
                    legal_moves_payload = []
                    try:
                        requirement = GameEngine.get_phase_requirement(
                            state,
                            getattr(state, "current_player", None),
                        )
                        if requirement is not None:
                            requirement_payload = {
                                "type": requirement.type.value,
                                "player": requirement.player,
                                "eligible_positions": requirement.eligible_positions,
                            }
                        legal_moves_payload = [
                            {
                                "type": mv.type.value,
                                "player": mv.player,
                            }
                            for mv in GameEngine.get_valid_moves(  # type: ignore[arg-type]
                                state,
                                getattr(state, "current_player", None),
                            )
                        ]
                    except Exception:
                        pass

                    failure_path = os.path.join(
                        failure_dir,
                        f"failure_{game_idx}_" f"{termination_reason.replace(':', '_')}.json",
                    )
                    with open(
                        failure_path,
                        "w",
                        encoding="utf-8",
                    ) as failure_f:
                        json.dump(
                            {
                                "game_index": game_idx,
                                "termination_reason": termination_reason,
                                "state": state_payload,
                                "last_move": last_move_payload,
                                "phase_requirement": requirement_payload,
                                "legal_moves": legal_moves_payload,
                            },
                            failure_f,
                        )
                except Exception:
                    # Snapshotting must never break the soak loop.
                    pass

            # Serialize training data for JSONL if enabled
            training_moves = None
            training_initial_state = None
            if include_training_data and initial_state_for_recording is not None:
                # Serialize moves to JSON-compatible dicts
                training_moves = [
                    m.model_dump(mode="json") if hasattr(m, "model_dump") else m
                    for m in game_moves_for_recording
                ]
                # Serialize initial state
                training_initial_state = initial_state_for_recording.model_dump(mode="json")

            # Derive standardized victory type using shared module
            vtype, stalemate_tb = derive_victory_type(state, max_moves)

            rec = GameRecord(
                index=game_idx,
                num_players=num_players,
                board_type=board_type.value,
                engine_mode=engine_mode,
                seed=game_seed,
                length=move_count,
                status=state.game_status.value,
                winner=getattr(state, "winner", None),
                termination_reason=termination_reason,
                skipped=skipped,
                invariant_violations_by_type=per_game_violations,
                swap_sides_moves=swap_sides_moves_for_game,
                used_pie_rule=swap_sides_moves_for_game > 0,
                victory_type=vtype,
                stalemate_tiebreaker=stalemate_tb,
                moves=training_moves,
                initial_state=training_initial_state,
                failure_debug=failure_debug,
            )
            # Record full game to database if enabled.
            if (not skipped) and replay_db and initial_state_for_recording is not None:
                # Only record completed games; skip any partial/aborted runs.
                if state.game_status != GameStatus.COMPLETED:
                    rec.db_record_error = f"not_completed_status:{state.game_status.value}"
                    skipped = True
                else:
                    # Validate recorded history in trace_mode before committing.
                    ok, err = _validate_history_trace(
                        initial_state_for_recording,
                        game_moves_for_recording,
                    )
                    if not ok:
                        rec.db_record_error = f"trace_replay_failure:{err}"
                        skipped = True
                        dump_dir = os.getenv("RINGRIFT_SOAK_FAILURE_DIR")
                        if dump_dir:
                            try:
                                os.makedirs(dump_dir, exist_ok=True)
                                dump_path = os.path.join(
                                    dump_dir,
                                    f"trace_failure_game_{game_idx}.json",
                                )

                                # Build a replay trace with phases/players to aid debugging.
                                replay_trace = []
                                replay_error = None
                                try:
                                    trace_state = initial_state_for_recording
                                    for idx, mv in enumerate(game_moves_for_recording):
                                        replay_trace.append(
                                            {
                                                "idx": idx,
                                                "move_number": getattr(mv, "move_number", None),
                                                "type": mv.type.value,
                                                "player": mv.player,
                                                "phase_before": (
                                                    getattr(trace_state, "current_phase", None).value
                                                    if trace_state and getattr(trace_state, "current_phase", None)
                                                    else None
                                                ),
                                                "current_player": getattr(trace_state, "current_player", None),
                                            }
                                        )
                                        trace_state = GameEngine.apply_move(trace_state, mv, trace_mode=True)  # type: ignore[arg-type]
                                except Exception as rexc:  # pragma: no cover - defensive
                                    replay_error = f"{type(rexc).__name__}:{rexc}"

                                with open(dump_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "error": err,
                                            "game_index": game_idx,
                                            "moves": [
                                                m.model_dump(mode="json")  # type: ignore[attr-defined]
                                                for m in game_moves_for_recording
                                            ],
                                            "initial_state": (
                                                initial_state_for_recording.model_dump(  # type: ignore[attr-defined]
                                                    mode="json"
                                                )
                                                if initial_state_for_recording
                                                else None
                                            ),
                                            "replay_trace": replay_trace,
                                            "replay_error": replay_error,
                                        },
                                        f,
                                    )
                            except Exception:
                                # Best-effort only.
                                pass
                    else:
                        try:
                            if profile_timing:
                                t_db_start = time.time()
                            game_id = record_completed_game_with_parity_check(
                                db=replay_db,
                                initial_state=initial_state_for_recording,
                                final_state=state,
                                moves=game_moves_for_recording,
                                metadata={
                                    "source": "selfplay_soak",
                                    "engine_mode": engine_mode,
                                    "difficulty_band": difficulty_band,
                                    "termination_reason": termination_reason,
                                    "rng_seed": game_seed,
                                    # Golden-game and diagnostics hooks:
                                    # persist invariant violation counts and pie-rule
                                    # usage so downstream tooling can mine interesting
                                    # traces (for example, games that exercised the
                                    # swap rule or violated invariants).
                                    "invariant_violations_by_type": per_game_violations,
                                    "swap_sides_moves": swap_sides_moves_for_game,
                                    "used_pie_rule": swap_sides_moves_for_game > 0,
                                    # Per-player AI metadata for analysis and debugging:
                                    # keys like player_{pnum}_ai_type, player_{pnum}_difficulty
                                    **per_player_ai_metadata,
                                },
                                # Lean mode: skip storing full state history for each move
                                # to reduce DB size ~100x while preserving training data
                                store_history_entries=not lean_db_enabled,
                            )
                            rec.db_recorded = True
                            rec.db_game_id = game_id
                            if profile_timing:
                                timing_totals["db_record"] += time.time() - t_db_start
                            games_recorded += 1
                        except ParityValidationError as pve:
                            # Parity validation failed - skip this game but continue.
                            rec.db_record_error = f"parity_divergence:{pve}"
                            skipped = True
                        except Exception as exc:  # pragma: no cover - defensive
                            # DB recording must never break the soak loop
                            rec.db_record_error = f"{type(exc).__name__}:{exc}"
                            skipped = True

            # Mirror any DB-derived skip status onto the record.
            rec.skipped = bool(rec.skipped or skipped)

            # Emit per-game JSONL record after attempting DB recording so that
            # db_record_error/db_game_id fields reflect reality.
            log_f.write(json.dumps(asdict(rec)) + "\n")
            log_f.flush()
            records.append(rec)

            # Write checkpoint if interval is configured
            if checkpoint_interval > 0 and (game_idx + 1) % checkpoint_interval == 0:
                _write_checkpoint(
                    games_done=game_idx + 1,
                    elapsed_sec=time.time() - soak_start_time,
                    records_so_far=records,
                )

            # Record game completion for progress reporting
            game_duration = time.time() - game_start_time
            progress_reporter.record_game(
                moves=move_count,
                duration_sec=game_duration,
            )
            if skipped:
                detail = rec.db_record_error or rec.termination_reason
                print(
                    f"[soak-skip] game {game_idx} skipped: reason={detail}",
                    file=sys.stderr,
                )

            if args.verbose and (game_idx + 1) % args.verbose == 0:
                print(
                    f"[soak] completed {game_idx + 1}/{num_games} games "
                    f"(last status={rec.status}, "
                    f"reason={rec.termination_reason}, length={rec.length})",
                    flush=True,
                )

            # Optional periodic cache/GC cleanup to keep long soaks
            # memory-bounded. This clears the GameEngine move cache,
            # neural net model cache (releasing GPU/MPS memory), and
            # triggers a full garbage-collection cycle every N games.
            #
            # For large boards (hex/square19), memory pressure is much higher
            # (~7x more cells than square8), so we clear after EVERY game
            # regardless of gc_interval to prevent OOM issues.
            effective_gc_interval = gc_interval
            if board_type in (BoardType.HEXAGONAL, BoardType.SQUARE19):
                effective_gc_interval = 1  # Always clear for large boards
            if effective_gc_interval and (game_idx + 1) % effective_gc_interval == 0:
                GameEngine.clear_cache()
                clear_model_cache()
                gc.collect()

            if skipped:
                if fail_on_anomaly:
                    break
                continue

    # Emit final progress summary
    progress_reporter.finish()

    # Log DB recording summary if enabled
    if replay_db:
        print(
            f"[record-db] Recorded {games_recorded}/{num_games} games " f"to {record_db_path}",
            flush=True,
        )

    if profile_timing and records:
        total_games_run = len(records)
        total_moves = total_moves_across_games

        # Build a structured timing profile so callers (including the CLI
        # entrypoint) can persist this alongside other soak summary data.
        timing_profile: Dict[str, Any] = {
            "total_games": total_games_run,
            "total_moves": total_moves,
            "env_reset": {
                "total_sec": timing_totals["env_reset"],
                "avg_per_game_sec": (timing_totals["env_reset"] / max(total_games_run, 1)),
            },
            "ai_build": {
                "total_sec": timing_totals["ai_build"],
                "avg_per_game_sec": (timing_totals["ai_build"] / max(total_games_run, 1)),
            },
            "db_record": {
                "total_sec": timing_totals["db_record"],
                "avg_per_game_sec": (timing_totals["db_record"] / max(total_games_run, 1)),
            },
        }

        if total_moves > 0:
            timing_profile["move_select"] = {
                "total_sec": timing_totals["move_select"],
                "avg_per_move_sec": (timing_totals["move_select"] / total_moves),
            }
            timing_profile["env_step"] = {
                "total_sec": timing_totals["env_step"],
                "avg_per_move_sec": (timing_totals["env_step"] / total_moves),
            }

        # Stash for consumption by the CLI entrypoint.
        global _LAST_TIMING_PROFILE
        _LAST_TIMING_PROFILE = timing_profile

        # Also emit a human-readable summary for interactive runs.
        print("[profile] Timing summary (seconds):")
        print(
            f"  env.reset:  total={timing_totals['env_reset']:.3f}, "
            f"avg_per_game={timing_totals['env_reset'] / max(total_games_run, 1):.3f}"
        )
        print(
            f"  AI build:   total={timing_totals['ai_build']:.3f}, "
            f"avg_per_game={timing_totals['ai_build'] / max(total_games_run, 1):.3f}"
        )
        if total_moves > 0:
            print(
                f"  select_move: total={timing_totals['move_select']:.3f}, "
                f"avg_per_move={timing_totals['move_select'] / total_moves:.6f}"
            )
            print(
                f"  env.step:    total={timing_totals['env_step']:.3f}, "
                f"avg_per_move={timing_totals['env_step'] / total_moves:.6f}"
            )
        print(
            f"  DB record:  total={timing_totals['db_record']:.3f}, "
            f"avg_per_game={timing_totals['db_record'] / max(total_games_run, 1):.3f}"
        )

    # Emit event for pipeline integration (new games available)
    if emit_events and HAS_EVENT_BUS and len(records) > 0:
        import asyncio
        config_key = f"{args.board_type}_{num_players}p"
        try:
            asyncio.run(emit_new_games(
                host="localhost",
                new_games=len(records),
                total_games=len(records),
                source="run_self_play_soak.py",
            ))
            print(f"[event] Emitted NEW_GAMES_AVAILABLE: {len(records)} games for {config_key}")
        except Exception as e:
            print(f"[event] Failed to emit event: {e}")

    # Log hot reload summary
    if hot_reloader and hot_reloader.update_count > 0:
        print(
            f"[hot-reload] Session summary: {hot_reloader.update_count} model updates detected",
            flush=True,
        )

    return records, invariant_violation_samples


def _summarise(
    records: List[GameRecord],
    invariant_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    total = len(records)
    by_status: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    skipped_by_reason: Dict[str, int] = {}
    lengths: List[int] = []
    completed_games = 0
    max_moves_games = 0
    skipped_games = 0
    violation_counts_by_type: Dict[str, int] = {}
    invariant_violations_by_id: Dict[str, int] = {}
    total_swap_sides_moves = 0
    games_with_swap_sides = 0
    db_recorded_games = 0
    db_record_error_games = 0

    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_reason[r.termination_reason] = (
            by_reason.get(
                r.termination_reason,
                0,
            )
            + 1
        )
        lengths.append(r.length)

        if getattr(r, "skipped", False):
            skipped_games += 1
            skipped_by_reason[r.termination_reason] = (
                skipped_by_reason.get(
                    r.termination_reason,
                    0,
                )
                + 1
            )

        if r.termination_reason.startswith("status:"):
            completed_games += 1
        if r.termination_reason == "max_moves_reached":
            max_moves_games += 1

        # Pie-rule diagnostics: aggregate SWAP_SIDES usage.
        swap_moves = getattr(r, "swap_sides_moves", 0)
        if swap_moves > 0:
            total_swap_sides_moves += swap_moves
            games_with_swap_sides += 1

        if getattr(r, "db_recorded", False):
            db_recorded_games += 1
        if getattr(r, "db_record_error", None):
            db_record_error_games += 1

        for v_type, count in getattr(
            r,
            "invariant_violations_by_type",
            {},
        ).items():
            violation_counts_by_type[v_type] = violation_counts_by_type.get(v_type, 0) + count
            invariant_id = VIOLATION_TYPE_TO_INVARIANT_ID.get(v_type)
            if invariant_id:
                invariant_violations_by_id[invariant_id] = invariant_violations_by_id.get(invariant_id, 0) + count

    lengths_sorted = sorted(lengths) if lengths else [0]

    summary: Dict[str, Any] = {
        "total_games": total,
        "by_status": by_status,
        "by_termination_reason": by_reason,
        "min_length": lengths_sorted[0],
        "max_length": lengths_sorted[-1],
        "avg_length": (sum(lengths) / total) if total else 0.0,
        "completed_games": completed_games,
        "max_moves_games": max_moves_games,
        "skipped_games": skipped_games,
        "skipped_by_reason": skipped_by_reason,
        "invariant_violations_total": sum(
            invariant_violations_by_id.values(),
        ),
        "invariant_violations_by_id": invariant_violations_by_id,
        "violation_counts_by_type": violation_counts_by_type,
        # Pie-rule usage aggregates
        "swap_sides_total_moves": total_swap_sides_moves,
        "swap_sides_games": games_with_swap_sides,
        "swap_sides_games_fraction": (games_with_swap_sides / total) if total else 0.0,
        "avg_swap_sides_moves_per_game": (total_swap_sides_moves / total) if total else 0.0,
        "db_recorded_games": db_recorded_games,
        "db_record_error_games": db_record_error_games,
    }

    if invariant_samples is not None:
        summary["invariant_violation_samples"] = invariant_samples

    return summary


def _build_healthcheck_summary(
    profile: str,
    board_types: List[str],
    engine_pairs: List[str],
    records: List[GameRecord],
    invariant_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct a compact, machine-readable AI health-check summary.

    This helper layers profile/engine metadata and a parity placeholder over
    the generic soak summary produced by :func:`_summarise`. It is intended
    for CI/nightly "AI self-play healthcheck" jobs that need a stable JSON
    shape keyed by invariant IDs (INV-*) rather than low-level violation
    types.
    """
    base_summary = _summarise(records, invariant_samples)

    # Ensure invariant keys are always present, even for zero-violation runs.
    base_summary.setdefault("invariant_violations_by_id", {})
    base_summary.setdefault(
        "invariant_violations_total",
        sum(
            base_summary["invariant_violations_by_id"].values(),
        ),
    )

    health_summary: Dict[str, Any] = {
        "profile": profile,
        "board_types": sorted(set(board_types)),
        "engine_pairs": engine_pairs,
    }
    health_summary.update(base_summary)

    # Parity integration (PARITY-*) for this profile is intentionally left as
    # a future extension; for now we expose a zeroed, structured placeholder
    # and a descriptive note so downstream tooling can distinguish "not
    # implemented" from "no mismatches observed".
    health_summary.setdefault(
        "parity_mismatches",
        {
            "hash": 0,
            "status": 0,
        },
    )
    health_summary.setdefault(
        "parity_notes",
        (
            "PARITY-* checks are not yet wired into the ai-healthcheck "
            "profile. See docs/INVARIANTS_AND_PARITY_FRAMEWORK.md for "
            "future PARITY-* integration points."
        ),
    )

    # Convenience alias: expose invariant samples under a shorter key while
    # retaining the original field name used by existing callers/tests.
    if "invariant_violation_samples" in base_summary:
        health_summary.setdefault(
            "samples",
            base_summary["invariant_violation_samples"],
        )

    return health_summary


def run_ai_healthcheck_profile(
    args: argparse.Namespace,
) -> Tuple[List[GameRecord], Dict[str, Any]]:
    """Run a lightweight multi-board AI self-play health check.

    This profile reuses :func:`run_self_play_soak` to execute a small,
    deterministic batch of mixed-engine self-play games across the canonical
    board set and aggregates invariant statistics into a single summary.

    Invariants enforced via the soak loop:

    - INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC via S/total elimination
      monotonicity checks.
    - INV-ACTIVE-NO-MOVES via ACTIVE_NO_MOVES / ACTIVE_NO_CANDIDATE_MOVES.
    - INV-TERMINATION (soft) via max_moves_games and termination reasons.
    """
    # Canonical board set for health checks: small/medium/hex.
    board_names = ["square8", "square19", "hexagonal"]

    # Single mixed-engine 2p pairing using the "light" difficulty band
    # (Random/Heuristic/low-depth Minimax) to keep runtime bounded while still
    # exercising realistic AI move generation.
    engine_mode = "mixed"
    difficulty_band = "light"
    num_players = 2

    games_per_config_env = os.getenv("RINGRIFT_AI_HEALTHCHECK_GAMES")
    try:
        games_per_config = int(games_per_config_env) if games_per_config_env else 2
    except ValueError:
        games_per_config = 2
    if games_per_config <= 0:
        games_per_config = 1

    base_seed = args.seed if getattr(args, "seed", None) is not None else 1764142864

    all_records: List[GameRecord] = []
    all_samples: List[Dict[str, Any]] = []

    # Derive a base directory for per-board JSONL logs from the user-supplied
    # --log-jsonl path.
    base_log_path = args.log_jsonl
    base_dir = os.path.dirname(base_log_path) or "."
    base_stem, _ext = os.path.splitext(os.path.basename(base_log_path))

    for index, board_name in enumerate(board_names):
        per_board_args = argparse.Namespace(**vars(args))
        per_board_args.board_type = board_name
        per_board_args.engine_mode = engine_mode
        per_board_args.difficulty_band = difficulty_band
        per_board_args.num_players = num_players
        per_board_args.num_games = games_per_config
        per_board_args.seed = base_seed + index * 100000
        # Keep caller-specified max_moves / gc_interval untouched.
        per_board_args.summary_json = None
        per_board_args.log_jsonl = os.path.join(
            base_dir,
            f"{base_stem}.{board_name}.jsonl",
        )

        records, samples = run_self_play_soak(per_board_args)
        all_records.extend(records)
        all_samples.extend(samples)

    health_summary = _build_healthcheck_summary(
        profile="ai-healthcheck",
        board_types=board_names,
        engine_pairs=[f"{engine_mode}_({difficulty_band})_{num_players}p"],
        records=all_records,
        invariant_samples=all_samples,
    )

    # Attach resolved health-check configuration for downstream inspection.
    health_summary.setdefault(
        "config",
        {
            "profile": "ai-healthcheck",
            "board_types": board_names,
            "engine_mode": engine_mode,
            "difficulty_band": difficulty_band,
            "num_players": num_players,
            "games_per_config": games_per_config,
            "max_moves": args.max_moves,
            "base_seed": base_seed,
            "strict_no_move_invariant": bool(STRICT_NO_MOVE_INVARIANT),
        },
    )

    return all_records, health_summary


# =============================================================================
# GPU-Accelerated Self-Play Soak
# =============================================================================


def run_gpu_self_play_soak(
    args: argparse.Namespace,
) -> Tuple[List[GameRecord], List[Dict[str, Any]]]:
    """Run GPU-accelerated self-play games using ParallelGameRunner.

    This function provides a 5-10x speedup on CUDA GPUs and 1.5-3x on Apple MPS
    compared to CPU-only execution. It uses the same heuristic-based AI as the
    CPU path but evaluates many games in parallel on the GPU.

    CONSTRAINTS:
    - Only supports square8 board type (8x8)
    - Only supports 2 players
    - Only supports heuristic-only engine mode

    Args:
        args: Namespace with num_games, gpu_batch_size, max_moves, log_jsonl, seed

    Returns:
        Tuple of (game_records, invariant_samples).
        GPU mode does not perform invariant checking, so invariant_samples is [].
    """
    import time

    # Lazy load GPU imports
    if not _load_gpu_imports():
        raise RuntimeError(
            "GPU imports failed. Ensure PyTorch is installed with CUDA/MPS support. "
            "Install with: pip install torch"
        )

    # Get GPUSelfPlayGenerator from global after lazy load
    global GPUSelfPlayGenerator
    if GPUSelfPlayGenerator is None:
        raise RuntimeError("GPUSelfPlayGenerator not available after import")

    num_games = args.num_games
    batch_size = getattr(args, "gpu_batch_size", 64)
    max_moves = args.max_moves
    seed = args.seed
    log_jsonl = args.log_jsonl

    # Set seeds if provided
    if seed is not None:
        import random
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        random.seed(seed)

    # Default heuristic weights (same as CPU path)
    weights = {
        "material_weight": 1.0,
        "ring_count_weight": 0.5,
        "stack_height_weight": 0.3,
        "center_control_weight": 0.4,
        "territory_weight": 0.8,
        "mobility_weight": 0.2,
        "line_potential_weight": 0.6,
        "defensive_weight": 0.3,
    }

    # Load custom weights if specified
    if getattr(args, "heuristic_weights_file", None) and getattr(args, "heuristic_profile", None):
        loaded = load_weights_from_profile(args.heuristic_weights_file, args.heuristic_profile)
        if loaded:
            weights = loaded
            print(f"GPU: Using weights from profile '{args.heuristic_profile}'")

    print(f"GPU self-play starting: {num_games} games, batch_size={batch_size}, max_moves={max_moves}")

    # Create generator
    generator = GPUSelfPlayGenerator(
        board_size=8,
        num_players=2,
        batch_size=batch_size,
        max_moves=max_moves,
        weights=weights,
    )

    # Generate games
    start_time = time.time()
    gpu_records = generator.generate_games(
        num_games=num_games,
        output_file=log_jsonl,
        progress_interval=max(1, num_games // 20),  # ~5% progress updates
    )
    elapsed = time.time() - start_time

    stats = generator.get_statistics()

    # Convert GPU records to GameRecord format for compatibility
    game_records: List[GameRecord] = []
    for i, gpu_rec in enumerate(gpu_records):
        record = GameRecord(
            index=i,
            num_players=2,
            board_type="square8",
            engine_mode="gpu-heuristic",
            seed=seed,
            length=gpu_rec.get("move_count", 0),
            status="completed",
            winner=gpu_rec.get("winner", 0),
            termination_reason=gpu_rec.get("victory_type", "unknown"),
            victory_type=gpu_rec.get("victory_type"),
            stalemate_tiebreaker=gpu_rec.get("stalemate_tiebreaker"),
        )
        game_records.append(record)

    # Print summary
    throughput = num_games / elapsed if elapsed > 0 else 0
    print(f"\nGPU self-play complete:")
    print(f"  Games: {stats.get('total_games', num_games)}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} games/sec")
    print(f"  Avg moves/game: {stats.get('moves_per_game', 0):.1f}")
    print(f"  Wins: P1={stats.get('wins_by_player', {}).get(1, 0)}, P2={stats.get('wins_by_player', {}).get(2, 0)}")
    print(f"  Draws: {stats.get('draws', 0)}")

    # GPU mode does not run invariant checks (they're CPU-only)
    invariant_samples: List[Dict[str, Any]] = []

    return game_records, invariant_samples


def _has_anomalies(records: List[GameRecord]) -> bool:
    """Return True if any record encodes an invariant/engine anomaly.

    This matches the semantics used by the CLI `--fail-on-anomaly` flag:
    only hard invariants or engine exceptions (not normal terminations such
    as max-moves cutoffs or completed games) are treated as anomalies.
    """
    anomalous_reasons = {
        "unknown",
        "no_legal_moves_for_current_player",
        "illegal_moves_in_forced_elimination",
        "no_ai_for_current_player",
        "ai_returned_no_move",
        "ai_move_not_forced_elimination",
        "recorded_player_mismatch",
    }
    anomalous_prefixes = (
        "step_exception:",
        "error_reset",
        "ai_selected_illegal_move:",
        "illegal_move_for_phase:",
        "phase_move_mismatch:",
    )

    def _is_bad_status(reason: str) -> bool:
        if not reason.startswith("status:"):
            return False
        # status:completed or status:completed:<victory_type> are normal.
        return not reason.startswith("status:completed")

    for rec in records:
        if getattr(rec, "db_record_error", None):
            return True
        if _is_bad_status(rec.termination_reason):
            return True
        if (rec.termination_reason in anomalous_reasons) or rec.termination_reason.startswith(anomalous_prefixes):
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run long self-play soaks using the Python rules engine and " "mixed/descent AI configurations."),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of self-play games to run (default: 100).",
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal", "hex8"],
        default="square8",
        help="Board type for self-play games (default: square8).",
    )
    parser.add_argument(
        "--profile",
        choices=["python-strict", "ai-healthcheck"],
        default=None,
        help=(
            "Optional named soak profile. 'python-strict' configures a small, "
            "deterministic strict-invariant run suitable for CI-like checks. "
            "'ai-healthcheck' runs a lightweight multi-board AI self-play "
            "health check and emits an invariant-focused JSON summary."
        ),
    )
    parser.add_argument(
        "--engine-mode",
        choices=[
            "descent-only",
            "mixed",
            "random-only",
            "heuristic-only",
            "minimax-only",
            "mcts-only",
            "nn-only",
            "best-vs-pool",
            # New diverse AI modes (all 11 AI types)
            "diverse",           # GPU-optimized diverse AI distribution
            "diverse-cpu",       # CPU-optimized diverse AI distribution
            "gpu-minimax-only",  # GPU batched minimax
            "maxn-only",         # Max-N search
            "brs-only",          # Best-Reply Search
            "policy-only",       # Direct NN policy
            "gumbel-mcts-only",  # Gumbel AlphaZero
            "neural-demo-only",  # Experimental neural
        ],
        default="mixed",
        help=(
            "Engine selection strategy: 'descent-only' for pure DescentAI, "
            "'mixed' to sample across the canonical ladder, "
            "'random-only' for pure RandomAI, "
            "'heuristic-only' for pure HeuristicAI, "
            "'minimax-only' for pure MinimaxAI, "
            "'mcts-only' for pure MCTS, "
            "'nn-only' for neural-net enabled Descent+MCTS+NNUE Minimax, "
            "'best-vs-pool' for best NN vs recent checkpoint pool (DescentAI), "
            "'diverse' for GPU-optimized diverse AI with all 11 types "
            "(GUMBEL_MCTS 20%%, POLICY_ONLY 15%%, GPU_MINIMAX 12%%), "
            "'diverse-cpu' for CPU-optimized distribution. "
            "Default: mixed."
        ),
    )
    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=0,
        help=(
            "Include up to N recent NN checkpoints (excluding the chosen best model) "
            "as alternative nn_model_id choices. In 'best-vs-pool' mode this controls "
            "the opponent pool; in 'mixed' mode this diversifies neural tiers. "
            "Default: 0 (only the best alias)."
        ),
    )
    parser.add_argument(
        "--nn-pool-dir",
        type=str,
        default=None,
        help=(
            "Directory to scan for *.pth checkpoints to populate the NN pool "
            "(defaults to ai-service/models/ when unset)."
        ),
    )
    parser.add_argument(
        "--difficulty-band",
        choices=["canonical", "light"],
        default="canonical",
        help=(
            "For engine_mode='mixed', control the AI difficulty band: "
            "'canonical' uses the full ladder (1–10); 'light' restricts "
            "to Random/Heuristic/low-depth Minimax (1,2,4,5) for "
            "memory-conscious strict-invariant soaks. Ignored when "
            "engine_mode='descent-only'."
        ),
    )
    parser.add_argument(
        "--heuristic-weights-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing heuristic weight profiles. "
            "When specified with --heuristic-profile, uses trained weights "
            "instead of defaults for heuristic-only mode. "
            "Example: config/trained_heuristic_profiles.json"
        ),
    )
    parser.add_argument(
        "--heuristic-profile",
        type=str,
        default=None,
        help=(
            "Name of the weight profile to use from --heuristic-weights-file. "
            "Only applies when engine_mode='heuristic-only'. "
            "Example: 'heuristic_v1_2p' or 'cmaes_gen50_best'"
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help=("Number of active players per game (2–4). " "Defaults to 2."),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help=(
            "Maximum moves per game before treating as a cutoff. If not specified, "
            "uses the theoretical max for the board type and player count (e.g., "
            "400 for square8/2p, 2000 for hexagonal/2p). "
            "Note: With canonical recording, each turn generates ~4-5 moves."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base RNG seed for deterministic runs.",
    )
    parser.add_argument(
        "--log-jsonl",
        required=True,
        help=(
            "Path to a JSONL file where one line per game summary will be "
            "written. Directories are created if needed."
        ),
    )
    parser.add_argument(
        "--summary-json",
        help=(
            "Optional path to write an aggregate JSON summary (counts by "
            "status, termination reason, and length statistics)."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help=(
            "If >0, print a progress line every N games with the latest "
            "status/length info. Default: 1 (print after every game); "
            "set to 0 to disable progress output."
        ),
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=5,
        help=(
            "If >0, clear GameEngine move caches, neural net model cache "
            "(releasing GPU/MPS memory), and run gc.collect() every N games "
            "to bound memory usage in long soaks. Default: 5. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--intra-game-gc-interval",
        type=int,
        default=0,
        help=(
            "If >0, run lightweight memory cleanup (AI caches, gc.collect) every N "
            "moves WITHIN each game. This is critical for large boards (hex/square19) "
            "where a single game can exhaust memory. Default: 0 (disabled). "
            "Recommended: 50-100 for hex, 30-50 for square19. "
            "Trade-off: Reduces peak memory at cost of ~5-15%% performance overhead."
        ),
    )
    parser.add_argument(
        "--streaming-record",
        action="store_true",
        help=(
            "Enable streaming move recording: write moves incrementally to temp storage "
            "instead of accumulating in memory. Reduces peak memory for long games but "
            "adds I/O overhead. Recommended for hex/square19 with DB recording enabled."
        ),
    )
    parser.add_argument(
        "--memory-constrained",
        action="store_true",
        help=(
            "Enable memory-constrained mode: combines --intra-game-gc-interval=50, "
            "--streaming-record, and forces --difficulty-band=light. Optimized for "
            "running large board soaks on memory-limited systems. Trade-offs: "
            "~10-20%% slower, lighter AI opponents (reduced play strength diversity)."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help=(
            "If set, collect a lightweight timing profile for env.reset, AI "
            "construction, move selection, env.step, and optional DB writes "
            "across the soak run and print a summary at the end."
        ),
    )
    parser.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        help=(
            "If set, exit with non-zero status if any game terminates with "
            "an invariant/engine anomaly such as "
            "'no_legal_moves_for_current_player' or 'step_exception:...'. "
            "Intended for automated gates or scheduled jobs."
        ),
    )
    parser.add_argument(
        "--square8-state-pool-output",
        type=str,
        nargs="?",
        const="data/eval_pools/square8/pool_v1.jsonl",
        default=None,
        help=(
            "Optional JSONL output path for sampled Square8 GameState "
            "snapshots. If provided without a value, defaults to "
            "'data/eval_pools/square8/pool_v1.jsonl'. When omitted, no "
            "state pool is generated."
        ),
    )
    parser.add_argument(
        "--square8-state-pool-max-states",
        type=int,
        default=500,
        help=("Maximum number of Square8 GameState snapshots to append to the " "state pool JSONL (default: 500)."),
    )
    parser.add_argument(
        "--square8-state-pool-sampling-interval",
        type=int,
        default=4,
        help=("Sample a GameState every N plies for Square8 games " "(default: 4)."),
    )
    parser.add_argument(
        "--square19-state-pool-output",
        type=str,
        default=None,
        help=(
            "Optional path to write a Square19 state pool JSONL file " "(e.g., data/eval_pools/square19/pool_v1.jsonl)."
        ),
    )
    parser.add_argument(
        "--square19-state-pool-max-states",
        type=int,
        default=0,
        help=("If >0, max number of Square19 states to sample into the pool."),
    )
    parser.add_argument(
        "--square19-state-pool-sampling-interval",
        type=int,
        default=0,
        help=("If >0, record every Nth move for Square19 games into the pool."),
    )
    parser.add_argument(
        "--hex-state-pool-output",
        type=str,
        default=None,
        help=("Optional path to write a Hex state pool JSONL file " "(e.g., data/eval_pools/hex/pool_v1.jsonl)."),
    )
    parser.add_argument(
        "--hex-state-pool-max-states",
        type=int,
        default=0,
        help=("If >0, max number of Hex states to sample into the pool."),
    )
    parser.add_argument(
        "--hex-state-pool-sampling-interval",
        type=int,
        default=0,
        help=("If >0, record every Nth move for Hex games into the pool."),
    )
    parser.add_argument(
        "--record-db",
        type=str,
        default="data/games/selfplay.db",
        help=(
            "Path to a SQLite database file for recording full game replays. "
            "Each completed game's initial state, final state, and all moves "
            "are stored in the GameReplayDB schema. Use --no-record-db to disable. "
            "Default: data/games/selfplay.db"
        ),
    )
    parser.add_argument(
        "--no-record-db",
        action="store_true",
        help="Disable game recording to database (overrides --record-db).",
    )
    parser.add_argument(
        "--lean-db",
        action="store_true",
        default=True,
        help=(
            "Enable lean database recording mode (~100x smaller). Skips storing "
            "full before/after state snapshots for each move. Still stores initial "
            "state, moves, and final state needed for training. Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-lean-db",
        action="store_true",
        help="Disable lean recording mode; store full state history for debugging.",
    )
    # Training data (moves + initial_state) is now ALWAYS included in JSONL output.
    # Accept the historical flag for backwards compatibility with older scripts
    # (e.g. canonical parity gates) that still pass it.
    parser.add_argument(
        "--include-training-data",
        action="store_true",
        help=(
            "Deprecated no-op. Training data is always included in JSONL output "
            "(RR-DATA-QUALITY-2024-12)."
        ),
    )
    # NOTE: Training data (moves + initial_state) is ALWAYS included in JSONL output.
    # This is mandatory to ensure all game data can be converted to NPZ for training.
    # The --no-include-training-data option has been removed to prevent generating
    # unusable game data. See RR-DATA-QUALITY-2024-12 for rationale.
    parser.add_argument(
        "--resume-from-jsonl",
        action="store_true",
        help=(
            "Resume from an existing JSONL file. If --log-jsonl file exists, count its "
            "lines to determine how many games have been completed, skip those games, "
            "and append new games to the file. The seed offset is adjusted to maintain "
            "determinism across restarts. Use for crash recovery in long runs."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help=(
            "If >0, write a checkpoint JSON file every N games with current run state "
            "(games completed, elapsed time, stats). The checkpoint file is written to "
            "--log-jsonl path with '.checkpoint.json' suffix. Use --resume-from-jsonl "
            "to automatically resume from the last checkpoint. Default: 0 (disabled)."
        ),
    )
    # GPU acceleration options
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Enable GPU-accelerated game simulation using ParallelGameRunner. "
            "Provides 5-10x speedup on CUDA GPUs, 1.5-3x on MPS (Apple Silicon). "
            "CONSTRAINTS: Only works with --board-type=square8, --num-players=2, "
            "and --engine-mode=heuristic-only. Other configurations will error."
        ),
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size for GPU parallel game simulation. Higher values improve "
            "throughput but use more GPU memory. Default: 64. "
            "Recommended: 32-128 for consumer GPUs, 256-512 for data center GPUs."
        ),
    )

    # Hot model reload options (for unified AI loop integration)
    parser.add_argument(
        "--watch-model-updates",
        action="store_true",
        help=(
            "Enable hot model reload: periodically check for model file changes and "
            "reload without process restart. Useful for continuous selfplay during "
            "training. Checks the promoted model alias for the current board/player config."
        ),
    )
    parser.add_argument(
        "--model-reload-interval",
        type=int,
        default=100,
        help=(
            "How often to check for model updates (in games). Default: 100. "
            "Set to 0 to disable periodic checks (only reload on startup)."
        ),
    )
    parser.add_argument(
        "--model-alias-path",
        type=str,
        default=None,
        help=(
            "Path to the model alias file to watch for hot reload. "
            "If not specified, uses the promoted model alias for the current config "
            "(e.g., models/ringrift_best_sq8_2p.pth for square8 2p)."
        ),
    )
    parser.add_argument(
        "--emit-events",
        action="store_true",
        help=(
            "Emit data events for pipeline integration (NEW_GAMES_AVAILABLE). "
            "Useful for triggering downstream processing in the unified AI loop."
        ),
    )
    parser.add_argument(
        "--opening-random-moves",
        type=int,
        default=4,
        help=(
            "Number of moves at game start to select randomly from legal moves "
            "instead of using AI selection. This increases opening diversity "
            "and prevents games from always starting with identical sequences. "
            "Default: 4. Set to 0 to disable. Recommended: 4-8 for 2p, 6-12 for 3-4p."
        ),
    )
    parser.add_argument(
        "--opening-top-k",
        type=int,
        default=0,
        help=(
            "If > 0, during opening phase select randomly from top-K AI-recommended "
            "moves instead of purely random. This maintains some move quality while "
            "adding diversity. Set to 0 (default) for purely random opening moves. "
            "Recommended: 3-5 if you want semi-intelligent opening diversity."
        ),
    )
    parser.add_argument(
        "--swap-sides-probability",
        type=float,
        default=0.5,
        help=(
            "Probability that Player 2 will use swap_sides (pie rule) when offered. "
            "This helps balance training data by ensuring more games start from both "
            "perspectives. Default: 0.5 (50%% chance). Set to 0 to let AI decide, "
            "set to 1.0 to always swap. Recommended: 0.5 for balanced training."
        ),
    )
    parser.add_argument(
        "--think-time-ms",
        type=int,
        default=None,
        help=(
            "Override AI think time in milliseconds. By default, uses board-specific "
            "values: 1ms for square8 (fast), 50ms for square19/hexagonal (reduce stalemates). "
            "Increase for better move quality at the cost of longer games."
        ),
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_args()

    # Resource guard: Check disk/memory before starting (80% limits)
    try:
        from app.utils.resource_guard import (
            check_disk_space, check_memory, LIMITS
        )
        # Estimate output size: soak tests can generate lots of data
        num_games = getattr(args, "num_games", 1000)
        estimated_output_mb = (num_games * 0.005) + 100  # ~5KB per game + overhead
        if not check_disk_space(required_gb=max(5.0, estimated_output_mb / 1024)):
            print(f"ERROR: Insufficient disk space (limit: {LIMITS.DISK_MAX_PERCENT}%)", file=sys.stderr)
            raise SystemExit(1)
        if not check_memory(required_gb=4.0):
            print(f"ERROR: Insufficient memory (limit: {LIMITS.MEMORY_MAX_PERCENT}%)", file=sys.stderr)
            raise SystemExit(1)
        print(f"Resource check passed: disk/memory within 80% limits")
    except ImportError:
        pass  # Resource guard not available

    # Check coordination before spawning
    task_id = None
    coord_start_time = time.time()
    if HAS_COORDINATION:
        import socket
        node_id = socket.gethostname()
        task_type = TaskType.GPU_SELFPLAY if getattr(args, "gpu", False) else TaskType.SELFPLAY
        allowed, reason = can_spawn_safe(task_type, node_id)
        if not allowed:
            logger.warning(f"[Coordination] Warning: {reason}")
            logger.info("[Coordination] Proceeding anyway (coordination is advisory)")

        # Register task for tracking
        board_type = getattr(args, "board_type", "square8")
        num_players = getattr(args, "num_players", 2)
        task_id = f"selfplay_soak_{board_type}_{num_players}p_{os.getpid()}"
        try:
            register_running_task(task_id, "selfplay", node_id, os.getpid())
            logger.info(f"[Coordination] Registered task {task_id}")
        except Exception as e:
            logger.warning(f"[Coordination] Warning: Failed to register task: {e}")

    # Check for GPU mode
    if getattr(args, "gpu", False):
        # Validate GPU constraints
        board_type = getattr(args, "board_type", "square8")
        num_players = getattr(args, "num_players", 2)

        if board_type != "square8":
            print(
                f"ERROR: GPU mode only supports square8 board type. Got: {board_type}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        if num_players != 2:
            print(
                f"ERROR: GPU mode only supports 2 players. Got: {num_players}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Run GPU self-play
        config_summary = {
            "num_games": args.num_games,
            "board_type": board_type,
            "engine_mode": "gpu-heuristic",
            "num_players": num_players,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "log_jsonl": args.log_jsonl,
            "summary_json": args.summary_json,
            "gpu": True,
            "gpu_batch_size": getattr(args, "gpu_batch_size", 64),
        }

        print("GPU self-play soak harness starting with config:")
        print(json.dumps(config_summary, indent=2, sort_keys=True))

        records, invariant_samples = run_gpu_self_play_soak(args)
        summary = _summarise(records, invariant_samples)
        summary["config"] = config_summary

        print("\n=== GPU self-play soak summary ===")
        print(json.dumps(summary, indent=2, sort_keys=True))

        if args.summary_json:
            os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

        if args.fail_on_anomaly:
            if _has_anomalies(records):
                print(
                    "GPU self-play soak detected anomalies; "
                    "exiting with non-zero status due to --fail-on-anomaly.",
                    file=sys.stderr,
                )
                raise SystemExit(1)

        # Record task completion for duration learning (GPU mode)
        if HAS_COORDINATION and task_id:
            try:
                import socket
                node_id = socket.gethostname()
                config = f"gpu_{board_type}_{num_players}p"
                # Args: task_type, host, started_at, completed_at, success, config
                record_task_completion("selfplay", node_id, coord_start_time, time.time(), True, config)
                logger.info("[Coordination] Recorded task completion (GPU)")
            except Exception as e:
                logger.warning(f"[Coordination] Warning: Failed to record task completion: {e}")

        return  # Exit early for GPU mode

    profile = getattr(args, "profile", None)
    if profile == "python-strict":
        # Light, deterministic strict-invariant profile mirroring the TS
        # short-soak spirit on square8.
        args.num_games = 6
        args.board_type = "square8"
        args.engine_mode = "mixed"
        args.difficulty_band = getattr(args, "difficulty_band", "light")
        args.num_players = 2
        args.max_moves = 10000
        if args.seed is None:
            args.seed = 1764142864
        if getattr(args, "gc_interval", 0) == 0:
            args.gc_interval = 10

        config_summary = {
            "num_games": args.num_games,
            "board_type": args.board_type,
            "engine_mode": args.engine_mode,
            "difficulty_band": getattr(args, "difficulty_band", "canonical"),
            "num_players": args.num_players,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "log_jsonl": args.log_jsonl,
            "summary_json": args.summary_json,
            "gc_interval": args.gc_interval,
            "strict_no_move_invariant": bool(STRICT_NO_MOVE_INVARIANT),
            "profile": profile,
            "profile_timing": getattr(args, "profile_timing", False),
        }

        print("Self-play soak harness starting with config:")
        print(json.dumps(config_summary, indent=2, sort_keys=True))

        records, invariant_samples = run_self_play_soak(args)
        summary = _summarise(records, invariant_samples)
        summary["config"] = config_summary
        if getattr(args, "profile_timing", False) and _LAST_TIMING_PROFILE is not None:
            summary["timing_profile"] = _LAST_TIMING_PROFILE

    elif profile == "ai-healthcheck":
        # Dedicated multi-board AI self-play health-check profile. This variant
        # ignores most CLI tuning flags and instead runs a small, deterministic
        # mixed-engine job across square8, square19, and hexagonal boards.
        if args.seed is None:
            args.seed = 1764142864
        # Health-check runs are short-lived but span multiple board types,
        # so periodic cleanup is still useful to prevent OOM on large boards.
        if getattr(args, "gc_interval", 5) == 0:
            args.gc_interval = 3  # Moderate cleanup for multi-board runs

        print(
            "AI self-play healthcheck starting with profile 'ai-healthcheck'. "
            "This profile runs a bounded mixed-engine self-play job across "
            "square8, square19, and hexagonal boards and aggregates invariant "
            "violations by INV-* id.",
        )

        records, summary = run_ai_healthcheck_profile(args)
        if getattr(args, "profile_timing", False) and _LAST_TIMING_PROFILE is not None:
            summary.setdefault("timing_profile", _LAST_TIMING_PROFILE)

    else:
        config_summary = {
            "num_games": args.num_games,
            "board_type": args.board_type,
            "engine_mode": args.engine_mode,
            "difficulty_band": getattr(args, "difficulty_band", "canonical"),
            "num_players": args.num_players,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "log_jsonl": args.log_jsonl,
            "summary_json": args.summary_json,
            "gc_interval": args.gc_interval,
            "strict_no_move_invariant": bool(STRICT_NO_MOVE_INVARIANT),
            "profile": profile,
            "profile_timing": getattr(args, "profile_timing", False),
        }

        print("Self-play soak harness starting with config:")
        print(json.dumps(config_summary, indent=2, sort_keys=True))

        records, invariant_samples = run_self_play_soak(args)
        summary = _summarise(records, invariant_samples)
        summary["config"] = config_summary
        if getattr(args, "profile_timing", False) and _LAST_TIMING_PROFILE is not None:
            summary["timing_profile"] = _LAST_TIMING_PROFILE

    heading = "AI self-play healthcheck summary" if profile == "ai-healthcheck" else "Self-play soak summary"

    print(f"\n=== {heading} ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    if args.fail_on_anomaly:
        if _has_anomalies(records):
            print(
                "Self-play soak detected invariant/engine anomalies; "
                "exiting with non-zero status due to --fail-on-anomaly.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Record task completion for duration learning (non-GPU mode)
    if HAS_COORDINATION and task_id:
        try:
            import socket
            node_id = socket.gethostname()
            board_type = getattr(args, "board_type", "square8")
            num_players = getattr(args, "num_players", 2)
            config = f"{board_type}_{num_players}p"
            # Args: task_type, host, started_at, completed_at, success, config
            record_task_completion("selfplay", node_id, coord_start_time, time.time(), True, config)
            logger.info("[Coordination] Recorded task completion")
        except Exception as e:
            logger.warning(f"[Coordination] Warning: Failed to record task completion: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
