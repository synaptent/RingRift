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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Ensure `app.*` imports resolve when run from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.main import _create_ai_instance, _get_difficulty_profile  # type: ignore  # noqa: E402
from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.env import RingRiftEnv  # type: ignore  # noqa: E402
from app.game_engine import GameEngine  # type: ignore  # noqa: E402


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


def _parse_board_type(name: str) -> BoardType:
    name = name.lower()
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hexagonal":
        return BoardType.HEXAGONAL
    raise SystemExit(f"Unknown board type: {name!r} (expected square8|square19|hexagonal)")


def _build_mixed_ai_pool(
    game_index: int,
    player_numbers: List[int],
    engine_mode: str,
    base_seed: Optional[int],
    difficulty_band: str = "canonical",
) -> Dict[int, Any]:
    """Construct per-player AI instances for a single game.

    For ``engine_mode == 'descent-only'`` we use DescentAI only.

    For ``engine_mode == 'mixed'`` we sample from the canonical difficulty
    ladder while forcing ``think_time=0`` for faster soaks. When
    ``difficulty_band == 'light'``, we restrict the ladder to a lighter
    subset (Random/Heuristic/low-depth Minimax) to reduce memory and
    runtime for long strict-invariant soaks.
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
            f"engine_mode must be 'descent-only' or 'mixed', got {engine_mode!r}"
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
        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")

        cfg = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=0,  # disable UX delay in soak
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            nn_model_id=nn_model_id,
        )
        ai = _create_ai_instance(ai_type, pnum, cfg)
        ai_by_player[pnum] = ai

    return ai_by_player


def run_self_play_soak(args: argparse.Namespace) -> List[GameRecord]:
    board_type = _parse_board_type(args.board_type)
    num_games = args.num_games
    num_players = args.num_players
    max_moves = args.max_moves
    engine_mode = args.engine_mode
    base_seed = args.seed
    difficulty_band = getattr(args, "difficulty_band", "canonical")
    gc_interval = getattr(args, "gc_interval", 0)

    os.makedirs(os.path.dirname(args.log_jsonl) or ".", exist_ok=True)

    env = RingRiftEnv(
        board_type=board_type,
        max_moves=max_moves,
        reward_on="terminal",
        num_players=num_players,
    )

    records: List[GameRecord] = []

    with open(args.log_jsonl, "w", encoding="utf-8") as log_f:
        for game_idx in range(num_games):
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
                continue

            player_numbers = [p.player_number for p in state.players]
            ai_by_player = _build_mixed_ai_pool(
                game_idx,
                player_numbers,
                engine_mode,
                base_seed,
                difficulty_band=difficulty_band,
            )

            move_count = 0
            termination_reason = "unknown"
            last_move = None

            while True:
                if state.game_status != GameStatus.ACTIVE:
                    termination_reason = f"status:{state.game_status.value}"
                    break

                legal_moves = env.legal_moves()
                if not legal_moves:
                    # With strict invariant enabled, this should be impossible
                    # for ACTIVE states; if it happens anyway we record it
                    # explicitly.
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

                try:
                    state, _reward, done, _info = env.step(move)
                    last_move = move
                except Exception as exc:  # pragma: no cover - defensive
                    termination_reason = f"step_exception:{type(exc).__name__}"
                    state = state  # keep last known state
                    break

                move_count += 1

                if move_count >= max_moves:
                    termination_reason = "max_moves_reached"
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
                        state_payload = state.model_dump(mode="json")  # type: ignore[attr-defined]
                    except Exception:
                        state_payload = None

                    try:
                        last_move_payload = (
                            last_move.model_dump(mode="json")  # type: ignore[attr-defined]
                            if last_move is not None
                            else None
                        )
                    except Exception:
                        last_move_payload = None

                    failure_path = os.path.join(
                        failure_dir,
                        f"failure_{game_idx}_{termination_reason.replace(':', '_')}.json",
                    )
                    with open(failure_path, "w", encoding="utf-8") as failure_f:
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
            )
            log_f.write(json.dumps(asdict(rec)) + "\n")
            records.append(rec)


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

    return records


def _summarise(records: List[GameRecord]) -> Dict[str, Any]:
    total = len(records)
    by_status: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    lengths: List[int] = []

    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_reason[r.termination_reason] = by_reason.get(r.termination_reason, 0) + 1
        lengths.append(r.length)

    lengths_sorted = sorted(lengths) if lengths else [0]

    return {
        "total_games": total,
        "by_status": by_status,
        "by_termination_reason": by_reason,
        "min_length": lengths_sorted[0],
        "max_length": lengths_sorted[-1],
        "avg_length": (sum(lengths) / total) if total else 0.0,
    }


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
        help="Maximum moves per game before treating as a cutoff (default: 200).",
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
        default=0,
        help=(
            "If >0, print a progress line every N games with the latest "
            "status/length info."
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
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_args()
    records = run_self_play_soak(args)
    summary = _summarise(records)

    print("\n=== Self-play soak summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover
    main()
