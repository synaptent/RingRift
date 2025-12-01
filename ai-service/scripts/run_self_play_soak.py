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

Example usage
-------------

From ``ai-service/``::

    # 100 mixed-engine 2p games on square8, invariant enabled
    RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
    python scripts/run_self_play_soak.py \
        --num-games 100 \
        --board-type square8 \
        --engine-mode mixed \
        --num-players 2 \
        --max-moves 200 \
        --seed 42 \
        --log-jsonl logs/selfplay/soak.square8_2p.mixed.jsonl \
        --summary-json logs/selfplay/soak.square8_2p.mixed.summary.json

    # 50 descent-only 3p games on square8
    python scripts/run_self_play_soak.py \
        --num-games 50 \
        --board-type square8 \
        --engine-mode descent-only \
        --num-players 3 \
        --max-moves 250 \
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
    GameState,
    GameStatus,
    MoveType,
)
from app.training.env import (  # type: ignore  # noqa: E402
    RingRiftEnv,
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
    get_theoretical_max_moves,
)
from app.game_engine import (  # type: ignore  # noqa: E402
    GameEngine,
    STRICT_NO_MOVE_INVARIANT,
)
from app.metrics import (  # type: ignore  # noqa: E402
    PYTHON_INVARIANT_VIOLATIONS,
)
from app.rules.core import compute_progress_snapshot  # noqa: E402
from app.rules import global_actions as ga  # type: ignore  # noqa: E402
from app.utils.progress_reporter import SoakProgressReporter  # noqa: E402
from app.db import get_or_create_db, record_completed_game  # noqa: E402


VIOLATION_TYPE_TO_INVARIANT_ID: Dict[str, str] = {
    "S_INVARIANT_DECREASED": "INV-S-MONOTONIC",
    "TOTAL_RINGS_ELIMINATED_DECREASED": "INV-ELIMINATION-MONOTONIC",
    "ACTIVE_NO_MOVES": "INV-ACTIVE-NO-MOVES",
    "ACTIVE_NO_CANDIDATE_MOVES": "INV-ACTIVE-NO-MOVES",
}

MAX_INVARIANT_VIOLATION_SAMPLES = 50


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
    invariant_violations_by_type: Dict[str, int] = field(default_factory=dict)
    # Pie-rule diagnostics: how many SWAP_SIDES moves occurred in this game,
    # and whether the pie rule was exercised at least once.
    swap_sides_moves: int = 0
    used_pie_rule: bool = False


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
    per_game_counts[violation_type] = per_game_counts.get(
        violation_type,
        0,
    ) + 1

    if len(samples) >= MAX_INVARIANT_VIOLATION_SAMPLES:
        return

    board_type_value = (
        state.board_type.value
        if hasattr(state.board_type, "value")
        else state.board_type
    )

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


def _parse_board_type(name: str) -> BoardType:
    name = name.lower()
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hexagonal":
        return BoardType.HEXAGONAL
    raise SystemExit(
        f"Unknown board type: {name!r} "
        "(expected square8|square19|hexagonal)"
    )


def _build_mixed_ai_pool(
    game_index: int,
    player_numbers: List[int],
    engine_mode: str,
    base_seed: Optional[int],
    board_type: BoardType,
    difficulty_band: str = "canonical",
) -> Dict[int, Any]:
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
    """

    ai_by_player: Dict[int, Any] = {}

    if engine_mode == "descent-only":
        from app.ai.descent_ai import DescentAI  # type: ignore

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=5,
                think_time=0,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
        return ai_by_player

    # mixed mode
    if engine_mode != "mixed":
        raise SystemExit(
            "engine_mode must be 'descent-only' or 'mixed', "
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

    return ai_by_player


def run_self_play_soak(
    args: argparse.Namespace,
) -> Tuple[List[GameRecord], List[Dict[str, Any]]]:
    board_type = _parse_board_type(args.board_type)
    num_games = args.num_games
    num_players = args.num_players
    max_moves = args.max_moves
    engine_mode = args.engine_mode
    base_seed = args.seed
    difficulty_band = getattr(args, "difficulty_band", "canonical")
    gc_interval = getattr(args, "gc_interval", 0)

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

    # Precompute per-board pool-enable flags so that the inner loop can
    # cheaply skip sampling logic when pools are disabled.
    square8_pool_enabled = bool(
        square8_state_pool_output
        and square8_state_pool_max_states > 0
        and square8_state_pool_sampling_interval > 0
    )
    square19_pool_enabled = bool(
        square19_state_pool_output
        and square19_state_pool_max_states > 0
        and square19_state_pool_sampling_interval > 0
    )
    hex_pool_enabled = bool(
        hex_state_pool_output
        and hex_state_pool_max_states > 0
        and hex_state_pool_sampling_interval > 0
    )

    os.makedirs(os.path.dirname(args.log_jsonl) or ".", exist_ok=True)

    # Initialize optional game recording database
    # --no-record-db flag overrides --record-db to disable recording
    record_db_path = None if getattr(args, "no_record_db", False) else getattr(args, "record_db", None)
    replay_db = get_or_create_db(record_db_path) if record_db_path else None
    games_recorded = 0

    env = RingRiftEnv(
        board_type=board_type,
        max_moves=max_moves,
        reward_on="terminal",
        num_players=num_players,
    )

    records: List[GameRecord] = []
    invariant_violation_samples: List[Dict[str, Any]] = []

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=num_games,
        report_interval_sec=10.0,
        context_label=f"{board_type.value}_{engine_mode}_{num_players}p",
    )

    with open(args.log_jsonl, "w", encoding="utf-8") as log_f:
        for game_idx in range(num_games):
            game_start_time = time.time()
            game_seed = None if base_seed is None else base_seed + game_idx
            try:
                state: GameState = env.reset(seed=game_seed)
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
            ai_by_player = _build_mixed_ai_pool(
                game_idx,
                player_numbers,
                engine_mode,
                base_seed,
                board_type,
                difficulty_band=difficulty_band,
            )

            move_count = 0
            termination_reason = "unknown"
            last_move = None
            per_game_violations: Dict[str, int] = {}
            swap_sides_moves_for_game = 0

            # Game recording: capture initial state and collect moves
            initial_state_for_recording = state.model_copy(deep=True) if replay_db else None
            game_moves_for_recording: List[Any] = [] if replay_db else []

            # Initialise S-invariant / elimination snapshot for this game.
            prev_snapshot = compute_progress_snapshot(state)
            prev_S = prev_snapshot["S"]
            prev_eliminated = prev_snapshot["eliminated"]

            while True:
                if state.game_status != GameStatus.ACTIVE:
                    termination_reason = f"status:{state.game_status.value}"
                    break

                legal_moves = env.legal_moves()
                if not legal_moves:
                    # With strict invariant enabled, this should be impossible
                    # for ACTIVE states; if it happens anyway we record it
                    # explicitly as an ACTIVE-no-moves violation.
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

                current_player = state.current_player
                ai = ai_by_player.get(current_player)
                if ai is None:
                    termination_reason = "no_ai_for_current_player"
                    break

                move = ai.select_move(state)
                if not move:
                    termination_reason = "ai_returned_no_move"
                    break

                if move.type == MoveType.SWAP_SIDES:
                    swap_sides_moves_for_game += 1

                try:
                    state, _reward, done, _info = env.step(move)
                    last_move = move
                    # Collect move for game recording
                    if replay_db:
                        game_moves_for_recording.append(move)
                except Exception as exc:  # pragma: no cover - defensive
                    termination_reason = f"step_exception:{type(exc).__name__}"
                    state = state  # keep last known state
                    break

                move_count += 1

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
                if state.game_status == GameStatus.ACTIVE:
                    if ga.is_anm_state(state):
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
                # Recommended soak configuration for generating evaluation
                # pools:
                #
                # - Use a reasonably long `max_moves` (e.g. 200+) so that games
                #   reliably reach rich mid- and late-game positions.
                # - Treat `*_state_pool_sampling_interval` as the primary knob
                #   for biasing toward mid/late-game: larger values sample less
                #   frequently and naturally skip over the earliest plies.
                # - Use `*_state_pool_max_states` to cap the total number of
                #   snapshots per board; once this cap is reached, no further
                #   states are written even if the soak continues.
                #
                # The "v1" evaluation pools used by heuristic CMA-ES training
                # are expected to be generated by long mixed-engine soaks with
                # these knobs tuned so that most sampled states come from
                # mid- and late-game rather than symmetric openings.
                if (
                    square8_pool_enabled
                    and state.board_type == BoardType.SQUARE8
                    and (
                        square8_state_pool_sampled
                        < square8_state_pool_max_states
                    )
                    and (
                        move_count
                        % square8_state_pool_sampling_interval
                    )
                    == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, square8_state_pool_output),
                            state,
                        )
                        square8_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        # State export must never break the soak loop.
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
                    and (
                        square19_state_pool_sampled
                        < square19_state_pool_max_states
                    )
                    and (
                        move_count
                        % square19_state_pool_sampling_interval
                    )
                    == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, square19_state_pool_output),
                            state,
                        )
                        square19_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        # State export must never break the soak loop.
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
                    and (
                        hex_state_pool_sampled
                        < hex_state_pool_max_states
                    )
                    and (
                        move_count
                        % hex_state_pool_sampling_interval
                    )
                    == 0
                    and state.game_status == GameStatus.ACTIVE
                ):
                    try:
                        _append_state_to_jsonl(
                            cast(str, hex_state_pool_output),
                            state,
                        )
                        hex_state_pool_sampled += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        # State export must never break the soak loop.
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

            # For problematic terminations, capture a minimal snapshot of the
            # final GameState + last Move so they can be turned into explicit
            # regression fixtures.
            if termination_reason in (
                "no_legal_moves_for_current_player",
            ) or termination_reason.startswith(
                "step_exception:RuntimeError",
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

                    failure_path = os.path.join(
                        failure_dir,
                        f"failure_{game_idx}_"
                        f"{termination_reason.replace(':', '_')}.json",
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
                            },
                            failure_f,
                        )
                except Exception:
                    # Snapshotting must never break the soak loop.
                    pass

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
                invariant_violations_by_type=per_game_violations,
                swap_sides_moves=swap_sides_moves_for_game,
                used_pie_rule=swap_sides_moves_for_game > 0,
            )
            log_f.write(json.dumps(asdict(rec)) + "\n")
            # Ensure per-game records are visible to tail/analysis tools even
            # while a long soak is still in progress.
            log_f.flush()
            records.append(rec)

            # Record full game to database if enabled
            if replay_db and initial_state_for_recording is not None:
                try:
                    game_id = record_completed_game(
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
                        },
                    )
                    games_recorded += 1
                except Exception as exc:  # pragma: no cover - defensive
                    # DB recording must never break the soak loop
                    print(
                        f"[record-db] Failed to record game {game_idx}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

            # Record game completion for progress reporting
            game_duration = time.time() - game_start_time
            progress_reporter.record_game(
                moves=move_count,
                duration_sec=game_duration,
            )

            if args.verbose and (game_idx + 1) % args.verbose == 0:
                print(
                    f"[soak] completed {game_idx + 1}/{num_games} games "
                    f"(last status={rec.status}, "
                    f"reason={rec.termination_reason}, length={rec.length})",
                    flush=True,
                )

            # Optional periodic cache/GC cleanup to keep long soaks
            # memory-bounded. This clears the GameEngine move cache and
            # triggers a full garbage-collection cycle every N games.
            if gc_interval and (game_idx + 1) % gc_interval == 0:
                GameEngine.clear_cache()
                gc.collect()

    # Emit final progress summary
    progress_reporter.finish()

    # Log DB recording summary if enabled
    if replay_db:
        print(
            f"[record-db] Recorded {games_recorded}/{num_games} games "
            f"to {record_db_path}",
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
    lengths: List[int] = []
    completed_games = 0
    max_moves_games = 0
    violation_counts_by_type: Dict[str, int] = {}
    invariant_violations_by_id: Dict[str, int] = {}
    total_swap_sides_moves = 0
    games_with_swap_sides = 0

    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_reason[r.termination_reason] = by_reason.get(
            r.termination_reason,
            0,
        ) + 1
        lengths.append(r.length)

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
            violation_counts_by_type[v_type] = (
                violation_counts_by_type.get(v_type, 0) + count
            )
            invariant_id = VIOLATION_TYPE_TO_INVARIANT_ID.get(v_type)
            if invariant_id:
                invariant_violations_by_id[invariant_id] = (
                    invariant_violations_by_id.get(invariant_id, 0) + count
                )

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
        "invariant_violations_total": sum(
            invariant_violations_by_id.values(),
        ),
        "invariant_violations_by_id": invariant_violations_by_id,
        "violation_counts_by_type": violation_counts_by_type,
        # Pie-rule usage aggregates
        "swap_sides_total_moves": total_swap_sides_moves,
        "swap_sides_games": games_with_swap_sides,
        "swap_sides_games_fraction": (
            games_with_swap_sides / total
        )
        if total
        else 0.0,
        "avg_swap_sides_moves_per_game": (
            total_swap_sides_moves / total
        )
        if total
        else 0.0,
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
        games_per_config = (
            int(games_per_config_env)
            if games_per_config_env
            else 2
        )
    except ValueError:
        games_per_config = 2
    if games_per_config <= 0:
        games_per_config = 1

    base_seed = (
        args.seed
        if getattr(args, "seed", None) is not None
        else 1764142864
    )

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


def _has_anomalies(records: List[GameRecord]) -> bool:
    """Return True if any record encodes an invariant/engine anomaly.

    This matches the semantics used by the CLI `--fail-on-anomaly` flag:
    only hard invariants or engine exceptions (not normal terminations such
    as max-moves cutoffs or completed games) are treated as anomalies.
    """
    anomalous_reasons = {"no_legal_moves_for_current_player"}
    anomalous_prefixes = ("step_exception:", "error_reset")
    return any(
        (rec.termination_reason in anomalous_reasons)
        or rec.termination_reason.startswith(anomalous_prefixes)
        for rec in records
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run long self-play soaks using the Python rules engine and "
            "mixed/descent AI configurations."
        ),
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
        choices=["descent-only", "mixed"],
        default="mixed",
        help=(
            "Engine selection strategy: 'descent-only' for pure DescentAI, "
            "'mixed' to sample across the canonical ladder. Default: mixed."
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
        "--num-players",
        type=int,
        default=2,
        help=(
            "Number of active players per game (2–4). "
            "Defaults to 2."
        ),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help=(
            "Maximum moves per game before treating as a cutoff "
            "(default: 200)."
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
        default=0,
        help=(
            "If >0, clear GameEngine move caches and run gc.collect() "
            "every N games to bound memory usage in long soaks."
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
        help=(
            "Maximum number of Square8 GameState snapshots to append to the "
            "state pool JSONL (default: 500)."
        ),
    )
    parser.add_argument(
        "--square8-state-pool-sampling-interval",
        type=int,
        default=4,
        help=(
            "Sample a GameState every N plies for Square8 games "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--square19-state-pool-output",
        type=str,
        default=None,
        help=(
            "Optional path to write a Square19 state pool JSONL file "
            "(e.g., data/eval_pools/square19/pool_v1.jsonl)."
        ),
    )
    parser.add_argument(
        "--square19-state-pool-max-states",
        type=int,
        default=0,
        help=(
            "If >0, max number of Square19 states to sample into the pool."
        ),
    )
    parser.add_argument(
        "--square19-state-pool-sampling-interval",
        type=int,
        default=0,
        help=(
            "If >0, record every Nth move for Square19 games into the pool."
        ),
    )
    parser.add_argument(
        "--hex-state-pool-output",
        type=str,
        default=None,
        help=(
            "Optional path to write a Hex state pool JSONL file "
            "(e.g., data/eval_pools/hex/pool_v1.jsonl)."
        ),
    )
    parser.add_argument(
        "--hex-state-pool-max-states",
        type=int,
        default=0,
        help=(
            "If >0, max number of Hex states to sample into the pool."
        ),
    )
    parser.add_argument(
        "--hex-state-pool-sampling-interval",
        type=int,
        default=0,
        help=(
            "If >0, record every Nth move for Hex games into the pool."
        ),
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
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_args()

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
        }

        print("Self-play soak harness starting with config:")
        print(json.dumps(config_summary, indent=2, sort_keys=True))

        records, invariant_samples = run_self_play_soak(args)
        summary = _summarise(records, invariant_samples)
        summary["config"] = config_summary

    elif profile == "ai-healthcheck":
        # Dedicated multi-board AI self-play health-check profile. This variant
        # ignores most CLI tuning flags and instead runs a small, deterministic
        # mixed-engine job across square8, square19, and hexagonal boards.
        if args.seed is None:
            args.seed = 1764142864
        if getattr(args, "gc_interval", 0) == 0:
            # Health-check runs are short-lived; explicit GC is usually
            # unnecessary.
            args.gc_interval = 0

        print(
            "AI self-play healthcheck starting with profile 'ai-healthcheck'. "
            "This profile runs a bounded mixed-engine self-play job across "
            "square8, square19, and hexagonal boards and aggregates invariant "
            "violations by INV-* id.",
        )

        records, summary = run_ai_healthcheck_profile(args)

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
        }

        print("Self-play soak harness starting with config:")
        print(json.dumps(config_summary, indent=2, sort_keys=True))

        records, invariant_samples = run_self_play_soak(args)
        summary = _summarise(records, invariant_samples)
        summary["config"] = config_summary

    heading = (
        "AI self-play healthcheck summary"
        if profile == "ai-healthcheck"
        else "Self-play soak summary"
    )

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
