"""Generate a combined territory + elimination dataset for HeuristicAI.

Goal
====

This module generates a JSON Lines (``.jsonl``) dataset in which each
example pairs a *non-terminal* game state with a scalar target equal to
the **final combined advantage** (territory + eliminated rings) for a
given player at end of game.

For a finished game with final GameState ``S_T`` and players with
``territory_spaces`` counts ``T_p`` and ``eliminated_rings`` counts
``E_p``, the target for player ``i`` is::

    territory_margin_i = T_i - max_{j != i} T_j
    elim_margin_i      = E_i - max_{j != i} E_j
    target_i           = territory_margin_i + elim_margin_i

For each state ``S_t`` along a self-play trajectory, we emit up to two
examples (one per player perspective)::

    {
      "game_state": S_t,
      "player_number": i,
      "target": target_i,
      "time_weight": gamma^(T - t)
    }

where ``gamma`` is a discount factor (default 0.99, configurable via
``--gamma`` CLI flag), ``T`` is the
trajectory length in states, and ``t`` is the index of the state within
that trajectory. ``game_state`` is serialized via
``GameState.model_dump(by_alias=True, mode="json")`` so it can be
reloaded later using ``GameState.model_validate``.

The resulting dataset is suitable for use with
``app.training.train_heuristic_weights``, which expects the
``game_state``, ``player_number``, and ``target`` fields (and optionally
consumes ``time_weight`` as a per-example training weight).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from app.ai.descent_ai import DescentAI
from app.main import _create_ai_instance, _get_difficulty_profile
from app.models import AIConfig, AIType, BoardType, GameState, GameStatus
from app.training.env import TrainingEnvConfig, make_env
from app.utils.progress_reporter import SoakProgressReporter


@dataclass
class TerritoryExample:
    """Single combined territory + elimination training example.

    Attributes
    ----------
    game_state:
        Snapshot of the game state at some time ``t`` during the game.
    player_number:
        Player index (1..N) whose perspective the target is defined for.
    target:
        Final combined advantage in spaces + eliminated rings for
        ``player_number`` at the end of the game, i.e.::

            (territory_i - max_{j != i} territory_j)
            + (elim_i - max_{j != i} elim_j)
    """

    game_state: GameState
    player_number: int
    target: float


def _final_combined_margin(final_state: GameState, player_number: int) -> int:
    """Compute final (territory + eliminated rings) advantage.

    Territory component uses ``territory_spaces`` and elimination
    component uses ``eliminated_rings`` on each Player in the final
    GameState.
    """

    my_territory = 0
    max_other_territory = 0
    my_elim = 0
    max_other_elim = 0

    for p in final_state.players:
        if p.player_number == player_number:
            my_territory = p.territory_spaces
            my_elim = p.eliminated_rings
        else:
            if p.territory_spaces > max_other_territory:
                max_other_territory = p.territory_spaces
            if p.eliminated_rings > max_other_elim:
                max_other_elim = p.eliminated_rings

    territory_margin = my_territory - max_other_territory
    elim_margin = my_elim - max_other_elim
    return territory_margin + elim_margin


def generate_territory_dataset(
    num_games: int,
    output_path: str,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 2000,  # Minimum 2000 for all boards
    seed: int | None = None,
    engine_mode: str = "descent-only",
    num_players: int = 2,
    gamma: float = 0.99,
) -> None:
    """Self-play generator for combined territory+elimination data.

    For each game, we:

    * Play a self-play match via :class:`RingRiftEnv` using either
      fixed DescentAI opponents (``engine_mode='descent-only'``) or a
      *mixed* pool of engines selected from the canonical difficulty
      ladder (``engine_mode='mixed'``).
    * Support 2–4 players per game by constructing an N-player
      initial state and one AI instance per player.
    * Record every non-terminal state visited along the trajectory.
    * After the game finishes, compute the final combined margins for
      all players and emit one JSONL example per (state, player).

    The output file will contain one JSON object per line with fields::

        {"game_state": {...}, "player_number": 1, "target": 12.0,
         "time_weight": 0.9, "ai_type_p1": "descent", ...}

    where ``game_state`` uses the camelCase aliases of ``GameState`` and
    JSON-serialisable values so it can be reloaded via
    ``GameState.model_validate``. The additional ``ai_type_*`` and
    ``ai_difficulty_*`` metadata fields are ignored by the training
    scripts but are useful for offline analysis of the corpus.
    """

    if engine_mode not in {"descent-only", "mixed"}:
        raise ValueError(
            "engine_mode must be 'descent-only' or 'mixed', got "
            f"{engine_mode!r}"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    # Base seed used to derive per-game RNG streams for environment and
    # AI configuration. When seed is None we still create a Random
    # instance for engine selection but leave it unseeded so that runs
    # differ between invocations.
    base_seed = seed or 0

    # Reuse DescentAI instances across games in the legacy 2-player
    # descent-only mode so behaviour stays as close as possible to the
    # original implementation. For N>2 players, we construct fresh
    # DescentAI instances per game for simplicity.
    ai1 = None
    ai2 = None

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=num_games,
        report_interval_sec=10.0,
        context_label=f"territory_dataset_{board_type.value}_{engine_mode}",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for game_idx in range(num_games):
            game_start_time = time.time()
            game_seed = None if seed is None else seed + game_idx
            state = env.reset(seed=game_seed)

            # Derive the active player numbers from the state so that any
            # future variants (e.g. observer slots) are handled robustly.
            player_numbers = [p.player_number for p in state.players]

            # Per-game AI selection.
            ai_by_player: dict[int, Any] = {}
            ai_types: dict[int, AIType] = {}
            ai_difficulties: dict[int, int] = {}

            if engine_mode == "descent-only":
                # Preserve the legacy behaviour exactly for 2-player games
                # by reusing ai1/ai2; for N>2 players we create fresh
                # DescentAI instances per player.
                if num_players == 2:
                    if ai1 is None or ai2 is None:
                        ai1 = DescentAI(
                            1,
                            AIConfig(
                                difficulty=5,
                                think_time=500,
                                randomness=0.1,
                                rngSeed=base_seed,
                            ),
                        )
                        ai2 = DescentAI(
                            2,
                            AIConfig(
                                difficulty=5,
                                think_time=500,
                                randomness=0.1,
                                rngSeed=base_seed + 1,
                            ),
                        )

                    ai_by_player = {1: ai1, 2: ai2}
                    ai_types = {1: AIType.DESCENT, 2: AIType.DESCENT}
                    ai_difficulties = {1: 5, 2: 5}
                else:
                    for pnum in player_numbers:
                        cfg = AIConfig(
                            difficulty=5,
                            think_time=500,
                            randomness=0.1,
                            rngSeed=base_seed + pnum,
                        )
                        ai = DescentAI(pnum, cfg)
                        ai_by_player[pnum] = ai
                        ai_types[pnum] = AIType.DESCENT
                        ai_difficulties[pnum] = 5
            else:  # mixed
                # Derive a deterministic but independent RNG stream per
                # game when a base seed is provided; otherwise leave the
                # RNG unseeded for non-deterministic runs.
                if seed is not None:
                    game_rng = random.Random(base_seed + game_idx)
                else:
                    game_rng = random.Random()

                # Difficulty presets chosen to cover the canonical
                # ladder while keeping runtime reasonable on square8:
                #   1  -> Random
                #   2  -> Heuristic
                #   4–6 -> Minimax
                #   7–8 -> MCTS
                #   9–10 -> Descent
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

                def _build_mixed_ai(
                    player_number: int,
                    *,
                    _game_rng: random.Random = game_rng,
                    _difficulty_choices: list = difficulty_choices,
                ):
                    difficulty = _game_rng.choice(_difficulty_choices)
                    profile = _get_difficulty_profile(difficulty)
                    ai_type = profile["ai_type"]

                    heuristic_profile_id = None
                    nn_model_id = None
                    if ai_type == AIType.HEURISTIC:
                        heuristic_profile_id = profile.get("profile_id")

                    config = AIConfig(
                        difficulty=difficulty,
                        randomness=profile["randomness"],
                        think_time=profile["think_time_ms"],
                        rngSeed=_game_rng.randrange(0, 2**31),
                        heuristic_profile_id=heuristic_profile_id,
                        nn_model_id=nn_model_id,
                    )
                    ai = _create_ai_instance(ai_type, player_number, config)
                    return ai_type, difficulty, ai

                for pnum in player_numbers:
                    ai_type, difficulty, ai = _build_mixed_ai(pnum)
                    ai_by_player[pnum] = ai
                    ai_types[pnum] = ai_type
                    ai_difficulties[pnum] = difficulty

            # String labels for metadata logging
            ai_type_labels: dict[int, str] = {
                pnum: ai_types[pnum].value for pnum in player_numbers
            }

            trajectory_states: list[GameState] = []
            move_count = 0

            while (
                state.game_status == GameStatus.ACTIVE
                and move_count < max_moves
            ):
                current_player = state.current_player
                ai = ai_by_player.get(current_player)
                if ai is None:
                    # Defensive: if we ever see a player number without a
                    # configured AI (e.g. spectator), treat as terminal.
                    break

                move = ai.select_move(state)
                if not move:
                    # No legal moves: treat as terminal.
                    break

                # Record the pre-move state snapshot.
                trajectory_states.append(state)

                state, _reward, done, _info = env.step(move)
                move_count += 1

                if done:
                    break

            final_state = state

            # Compute final combined margins once, then emit weighted
            # examples for each recorded state from all players'
            # perspectives.
            margins: dict[int, float] = {
                p.player_number: float(
                    _final_combined_margin(final_state, p.player_number)
                )
                for p in final_state.players
            }

            # Ensure we have a stable, sorted list of player numbers for
            # iteration and metadata emission.
            player_numbers_sorted = sorted(margins.keys())

            T = len(trajectory_states)

            for t, snapshot in enumerate(trajectory_states, start=1):
                # Training-time weight w_t = gamma^(T - t)
                time_weight = float(gamma ** (T - t))
                snapshot_json = snapshot.model_dump(
                    by_alias=True,
                    mode="json",
                )

                # Precompute per-player engine metadata for logging.
                metadata: dict[str, object] = {
                    "engine_mode": engine_mode,
                    "num_players": len(player_numbers_sorted),
                }
                for pnum in player_numbers_sorted:
                    if pnum in ai_type_labels:
                        metadata[f"ai_type_p{pnum}"] = ai_type_labels[pnum]
                    if pnum in ai_difficulties:
                        metadata[f"ai_difficulty_p{pnum}"] = (
                            ai_difficulties[pnum]
                        )

                for player_number in player_numbers_sorted:
                    example = {
                        "game_state": snapshot_json,
                        "player_number": player_number,
                        "target": margins[player_number],
                        "time_weight": time_weight,
                        **metadata,
                    }
                    f.write(json.dumps(example) + "\n")

            total_examples = len(trajectory_states) * len(player_numbers_sorted)
            print(
                "Generated "
                f"{total_examples} examples "
                f"from game {game_idx + 1}/{num_games} "
                f"(num_players={len(player_numbers_sorted)}, "
                f"engine_mode={engine_mode})"
            )

            # Record game completion for progress reporting
            game_duration = time.time() - game_start_time
            progress_reporter.record_game(
                moves=move_count,
                duration_sec=game_duration,
            )

    # Emit final progress summary
    progress_reporter.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a JSONL dataset of (state, player) examples "
            "labelled with final combined territory+elimination advantage."
        ),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of self-play games to generate (default: 10).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type for self-play games (default: square8).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game before forcing termination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base RNG seed for deterministic runs.",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["descent-only", "mixed"],
        default="descent-only",
        help=(
            "Engine selection strategy: 'descent-only' reuses the legacy "
            "DescentAI-vs-DescentAI pairing, while 'mixed' samples AI "
            "types from the canonical ladder (Random/Heuristic/Minimax/"
            "MCTS/Descent) independently for each player per game. "
            "Default: descent-only."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help=(
            "Number of active players per game (2–4). "
            "Defaults to 2 to preserve existing behaviour."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for time weighting (default: 0.99).",
    )
    return parser.parse_args()


def _board_type_from_str(name: str) -> BoardType:
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hexagonal":
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board type: {name!r}")


def main() -> None:
    args = _parse_args()
    board_type = _board_type_from_str(args.board_type)

    # Validate gamma is in valid range
    if not (0.0 <= args.gamma <= 1.0):
        raise ValueError(
            f"--gamma must be between 0.0 and 1.0 inclusive, got {args.gamma}"
        )

    generate_territory_dataset(
        num_games=args.num_games,
        output_path=args.output,
        board_type=board_type,
        max_moves=args.max_moves,
        seed=args.seed,
        engine_mode=args.engine_mode,
        num_players=args.num_players,
        gamma=args.gamma,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
