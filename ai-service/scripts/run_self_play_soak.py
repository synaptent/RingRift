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
import json
import os
import random
import sys
import gc
import time
from dataclasses import asdict, dataclass, field
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


def _canonical_termination_reason(state: GameState, fallback: str) -> str:
    """
    Map a completed GameState to a canonical termination reason.
    """
    status_str = state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status)
    if status_str != "completed":
        return f"status:{status_str}" if status_str else fallback

    winner = getattr(state, "winner", None)
    if winner is None:
        return "status:completed"

    victory_threshold = getattr(state, "victory_threshold", None)
    territory_threshold = getattr(state, "territory_victory_threshold", None)
    winner_player = next(
        (p for p in state.players if getattr(p, "player_number", None) == winner),
        None,
    )

    if (
        victory_threshold is not None
        and winner_player is not None
        and getattr(winner_player, "eliminated_rings", 0) >= victory_threshold
    ):
        return "status:completed:elimination"

    if (
        territory_threshold is not None
        and winner_player is not None
        and getattr(winner_player, "territory_spaces", 0) >= territory_threshold
    ):
        return "status:completed:territory"

    return "status:completed:lps"


def _build_mixed_ai_pool(
    game_index: int,
    player_numbers: List[int],
    engine_mode: str,
    base_seed: Optional[int],
    board_type: BoardType,
    difficulty_band: str = "canonical",
    heuristic_weights: Optional[Dict[str, float]] = None,
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
                think_time=0,
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
                think_time=0,
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
                think_time=0,
                randomness=0.05,
                rngSeed=(base_seed or 0) + pnum + game_index,
                heuristic_eval_mode=heuristic_eval_mode,
                heuristic_profile_id=custom_profile_id,  # Use custom weights if provided
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
                think_time=0,
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
                think_time=0,
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

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=10,
                think_time=0,
                randomness=0.05,
                rngSeed=(base_seed or 0) + pnum + game_index,
                use_neural_net=True,  # Enable neural network evaluation
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
            ai_metadata[f"player_{pnum}_ai_type"] = "descent_nn"
            ai_metadata[f"player_{pnum}_difficulty"] = 10
        return ai_by_player, ai_metadata

    # mixed mode
    if engine_mode != "mixed":
        raise SystemExit(
            "engine_mode must be one of: descent-only, mixed, random-only, "
            "heuristic-only, minimax-only, mcts-only, nn-only; "
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
        # Heuristic, and low-depth Minimax only.
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

        cfg = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=0,  # disable UX delay in soak
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
            heuristic_eval_mode=heuristic_eval_mode,
        )
        ai = _create_ai_instance(ai_type, pnum, cfg)
        ai_by_player[pnum] = ai

        # Record AI metadata for this player
        ai_metadata[f"player_{pnum}_ai_type"] = ai_type.value
        ai_metadata[f"player_{pnum}_difficulty"] = difficulty
        if heuristic_profile_id:
            ai_metadata[f"player_{pnum}_profile_id"] = heuristic_profile_id

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

    # Lean DB mode: skip storing full state history for each move (~100x smaller)
    # Default is True (lean enabled), --no-lean-db disables it
    lean_db_enabled = getattr(args, "lean_db", True) and not getattr(args, "no_lean_db", False)

    # Include training data (moves + initial_state) in JSONL output.
    # Enabled by default; use --no-include-training-data to disable.
    include_training_data = getattr(args, "include_training_data", True) and not getattr(args, "no_include_training_data", False)

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
        records_so_far: List[GameRecord],
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
        for game_idx in range(start_game_idx, num_games):
            game_start_time = time.time()
            game_seed = None if base_seed is None else base_seed + game_idx
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
            )
            if profile_timing:
                timing_totals["ai_build"] += time.time() - t_ai_start

            move_count = 0
            termination_reason = "unknown"
            last_move = None
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
                        game_moves_for_recording.append(move)
                        auto_moves = step_info.get("auto_generated_moves", [])
                        if auto_moves:
                            # Auto-generated moves may include bookkeeping moves
                            # for DIFFERENT players after a turn transition. This
                            # is expected and required for canonical recordings
                            # (e.g., new player has 0 rings → no_placement_action).
                            # We no longer reject cross-player auto-generated moves.
                            game_moves_for_recording.extend(auto_moves)
                    if done:
                        # If the env terminated but the rules engine still reports
                        # ACTIVE, treat it as an env-level cutoff to avoid recording
                        # a partial turn.
                        if state.game_status != GameStatus.ACTIVE:
                            termination_reason = _canonical_termination_reason(
                                state,
                                termination_reason or "status:completed",
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
            )
            log_f.write(json.dumps(asdict(rec)) + "\n")
            # Ensure per-game records are visible to tail/analysis tools even
            # while a long soak is still in progress.
            log_f.flush()
            records.append(rec)

            # Write checkpoint if interval is configured
            if checkpoint_interval > 0 and (game_idx + 1) % checkpoint_interval == 0:
                _write_checkpoint(
                    games_done=game_idx + 1,
                    elapsed_sec=time.time() - soak_start_time,
                    records_so_far=records,
                )

            if skipped:
                print(
                    f"[soak-skip] game {game_idx} skipped: reason={termination_reason}",
                    file=sys.stderr,
                )
                # Skip recording/metrics; continue to next game
                continue

            # Record full game to database if enabled
            if replay_db and initial_state_for_recording is not None:
                # Only record completed games; skip any partial/aborted runs.
                if state.game_status != GameStatus.COMPLETED:
                    print(
                        f"[record-db] Skipping game {game_idx} "
                        f"because game_status={state.game_status.value} "
                        f"termination_reason={termination_reason}",
                        file=sys.stderr,
                    )
                    continue

                # Validate recorded history in trace_mode before committing.
                ok, err = _validate_history_trace(
                    initial_state_for_recording,
                    game_moves_for_recording,
                )
                if not ok:
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
                                state = initial_state_for_recording
                                for idx, mv in enumerate(game_moves_for_recording):
                                    replay_trace.append(
                                        {
                                            "idx": idx,
                                            "move_number": getattr(mv, "move_number", None),
                                            "type": mv.type.value,
                                            "player": mv.player,
                                            "phase_before": (
                                                getattr(state, "current_phase", None).value
                                                if state and getattr(state, "current_phase", None)
                                                else None
                                            ),
                                            "current_player": getattr(state, "current_player", None),
                                        }
                                    )
                                    state = GameEngine.apply_move(state, mv, trace_mode=True)  # type: ignore[arg-type]
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
                    print(
                        f"[record-db] Skipping game {game_idx} due to trace replay failure: {err}",
                        file=sys.stderr,
                    )
                    continue

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
                    if profile_timing:
                        timing_totals["db_record"] += time.time() - t_db_start
                    games_recorded += 1
                except ParityValidationError as pve:
                    # Parity validation failed - skip this game but continue.
                    print(
                        f"[PARITY ERROR] Skipping game {game_idx} due to TS parity divergence: {pve}",
                        file=sys.stderr,
                    )
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    # DB recording must never break the soak loop
                    print(
                        f"[record-db] Failed to record game {game_idx}: " f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

            # Record game completion for progress reporting
            game_duration = time.time() - game_start_time
            progress_reporter.record_game(
                moves=move_count,
                duration_sec=game_duration,
            )
            if skipped:
                # Note: skipped games do not increment games_recorded; loop continues
                continue

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
    from datetime import datetime

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
            game_id=gpu_rec.get("game_id", str(i)),
            board_type="square8",
            num_players=2,
            engine_mode="gpu-heuristic",
            p1_config={"type": "gpu-heuristic", "difficulty": 5},
            p2_config={"type": "gpu-heuristic", "difficulty": 5},
            winner=gpu_rec.get("winner", 0),
            termination_reason=gpu_rec.get("victory_type", "unknown"),
            move_count=gpu_rec.get("move_count", 0),
            duration_seconds=gpu_rec.get("game_time_seconds", 0.0),
            timestamp=gpu_rec.get("timestamp", datetime.now().isoformat()),
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
    anomalous_reasons = {"no_legal_moves_for_current_player"}
    anomalous_prefixes = ("step_exception:", "error_reset")
    return any(
        (rec.termination_reason in anomalous_reasons) or rec.termination_reason.startswith(anomalous_prefixes)
        for rec in records
    )


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
        choices=["square8", "square19", "hexagonal"],
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
        ],
        default="mixed",
        help=(
            "Engine selection strategy: 'descent-only' for pure DescentAI, "
            "'mixed' to sample across the canonical ladder, "
            "'random-only' for pure RandomAI, "
            "'heuristic-only' for pure HeuristicAI, "
            "'minimax-only' for pure MinimaxAI, "
            "'mcts-only' for pure MCTS, "
            "'nn-only' for neural-net enabled Descent+MCTS+NNUE Minimax. "
            "Default: mixed."
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
    parser.add_argument(
        "--include-training-data",
        action="store_true",
        default=True,
        help=(
            "Include training data (moves and initial_state) in JSONL output. "
            "This allows reconstructing full games from JSONL alone without a database. "
            "Enabled by default. Use --no-include-training-data to disable."
        ),
    )
    parser.add_argument(
        "--no-include-training-data",
        action="store_true",
        help=(
            "Disable including training data (moves and initial_state) in JSONL output. "
            "Reduces JSONL file size but makes games unusable for training without a database."
        ),
    )
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
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_args()

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
        args.max_moves = 150
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


if __name__ == "__main__":  # pragma: no cover
    main()
