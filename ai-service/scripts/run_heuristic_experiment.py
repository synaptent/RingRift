#!/usr/bin/env python
"""Experiment harness for HeuristicAI profiles.

This script automates A/B experiments between heuristic weight profiles
produced by :mod:`app.training.train_heuristic_weights`.

Supported modes
===============

1. Baseline vs trained (single trained file):

   - Profile A: an existing baseline id (e.g. ``heuristic_v1_balanced``).
   - Profile B: the same id loaded from a trained profiles JSON file,
     registered under ``"<base_profile_id>_trained"``.

2. Trained vs trained (A/B, two trained files):

   - Profile A: ``"<base_profile_id_a>_A"`` loaded from
     ``--trained-profiles-a``.
   - Profile B: ``"<base_profile_id_b>_B"`` loaded from
     ``--trained-profiles-b``.

The script can sweep over multiple difficulties and board types and emit an
aggregated CSV/JSON report for downstream analysis.

Usage examples
--------------

From the ``ai-service`` root::

    # Baseline vs trained, difficulty 5 on Square8
    python scripts/run_heuristic_experiment.py \
        --mode baseline-vs-trained \
        --trained-profiles-a \
          logs/heuristic/heuristic_profiles.v1.trained.json \
        --base-profile-id-a heuristic_v1_balanced \
        --difficulties 5 \
        --boards Square8 \
        --games-per-match 200 \
        --out-json logs/heuristic/experiments.baseline_vs_trained.json \
        --out-csv  logs/heuristic/experiments.baseline_vs_trained.csv

    # A/B between two different trained files on multiple difficulties/boards
    python scripts/run_heuristic_experiment.py \
        --mode ab-trained \
        --trained-profiles-a logs/heuristic/heuristic_profiles.v1.expA.json \
        --trained-profiles-b logs/heuristic/heuristic_profiles.v1.expB.json \
        --base-profile-id-a heuristic_v1_balanced \
        --base-profile-id-b heuristic_v1_balanced \
        --difficulties 3,5,7 \
        --boards Square8,Square19 \
        --games-per-match 200 \
        --out-json logs/heuristic/experiments.expA_vs_expB.json \
        --out-csv  logs/heuristic/experiments.expA_vs_expB.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List

# Ensure project root (containing ``app`` and ``scripts``) is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    CMAESConfig,
    run_cmaes_optimization,
)

from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)
from app.ai.heuristic_ai import HeuristicAI  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    HEURISTIC_WEIGHT_PROFILES,
    load_trained_profiles_if_available,
)
from app.rules.default_engine import (  # type: ignore  # noqa: E402
    DefaultRulesEngine,
)
from app.training.env import (  # type: ignore  # noqa: E402
    get_two_player_training_kwargs,
)


BOARD_TYPES: Dict[str, BoardType] = {
    "Square8": BoardType.SQUARE8,
    "Square19": BoardType.SQUARE19,
    "Hex": BoardType.HEXAGONAL,
}


@dataclass
class MatchStats:
    """Aggregate results for one (profileA, profileB, difficulty, board)
    configuration.
    """

    mode: str
    difficulty: int
    board: str
    games: int
    profile_a_id: str
    profile_b_id: str
    wins_a: int
    wins_b: int
    draws: int

    @property
    def non_draws(self) -> int:
        return self.wins_a + self.wins_b

    @property
    def winrate_a_excl_draws(self) -> float:
        if self.non_draws == 0:
            return 0.0
        return self.wins_a / self.non_draws

    @property
    def winrate_b_excl_draws(self) -> float:
        if self.non_draws == 0:
            return 0.0
        return self.wins_b / self.non_draws


def create_empty_game_state(
    board_type_str: str,
    p1_label: str,
    p2_label: str,
) -> GameState:
    """Construct a minimal GameState suitable for AI-vs-AI matches.

    This mirrors the helper in ``run_ai_tournament.py`` but is scoped to
    HeuristicAI experiments.
    """

    board_type = BOARD_TYPES.get(board_type_str, BoardType.SQUARE8)

    size = 8
    if board_type == BoardType.SQUARE19:
        size = 19
    elif board_type == BoardType.HEXAGONAL:
        size = 5

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    from datetime import datetime

    now = datetime.now()

    players = [
        Player(
            id="player1",
            username=p1_label,
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=0,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="player2",
            username=p2_label,
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=0,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    return GameState(
        id="heuristic-experiment",
        boardType=board_type,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


def play_single_game(
    profile_p1: str,
    profile_p2: str,
    difficulty: int,
    board_type: str,
    max_moves: int = 300,
) -> int:
    """Run one game and return the winner (1 or 2) or 0 for draw."""

    ai1 = HeuristicAI(
        1,
        AIConfig(
            difficulty=difficulty,
            think_time=0,
            randomness=0.0,
            rngSeed=None,
            heuristic_profile_id=profile_p1,
        ),
    )
    ai2 = HeuristicAI(
        2,
        AIConfig(
            difficulty=difficulty,
            think_time=0,
            randomness=0.0,
            rngSeed=None,
            heuristic_profile_id=profile_p2,
        ),
    )

    game_state = create_empty_game_state(
        board_type,
        p1_label=profile_p1,
        p2_label=profile_p2,
    )
    rules_engine = DefaultRulesEngine()
    move_count = 0

    while (
        game_state.game_status == GameStatus.ACTIVE
        and move_count < max_moves
    ):
        current_player_num = game_state.current_player
        current_ai = ai1 if current_player_num == 1 else ai2
        current_ai.player_number = current_player_num

        move = current_ai.select_move(game_state)
        if not move:
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2 if current_player_num == 1 else 1
            break

        game_state = rules_engine.apply_move(game_state, move)

        if game_state.game_status == GameStatus.FINISHED:
            break

        move_count += 1

    if game_state.game_status == GameStatus.ACTIVE:
        return 0

    return game_state.winner or 0


def run_match(
    mode: str,
    profile_a_id: str,
    profile_b_id: str,
    difficulty: int,
    board: str,
    games: int,
) -> MatchStats:
    """Run a multi-game match with side-swapping and collect stats."""

    wins_a = 0
    wins_b = 0
    draws = 0

    for i in range(games):
        if i % 2 == 0:
            # A plays as Player 1
            winner = play_single_game(
                profile_a_id,
                profile_b_id,
                difficulty,
                board,
            )
            if winner == 1:
                wins_a += 1
            elif winner == 2:
                wins_b += 1
            else:
                draws += 1
        else:
            # B plays as Player 1
            winner = play_single_game(
                profile_b_id,
                profile_a_id,
                difficulty,
                board,
            )
            if winner == 1:
                wins_b += 1
            elif winner == 2:
                wins_a += 1
            else:
                draws += 1

    return MatchStats(
        mode=mode,
        difficulty=difficulty,
        board=board,
        games=games,
        profile_a_id=profile_a_id,
        profile_b_id=profile_b_id,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )


def _parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _write_json(path: str, rows: Iterable[MatchStats]) -> None:
    data = []
    for r in rows:
        d = asdict(r)
        d["non_draws"] = r.non_draws
        d["winrate_a_excl_draws"] = r.winrate_a_excl_draws
        d["winrate_b_excl_draws"] = r.winrate_b_excl_draws
        data.append(d)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Iterable[MatchStats]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "mode",
        "difficulty",
        "board",
        "games",
        "profile_a_id",
        "profile_b_id",
        "wins_a",
        "wins_b",
        "draws",
        "non_draws",
        "winrate_a_excl_draws",
        "winrate_b_excl_draws",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "mode": r.mode,
                    "difficulty": r.difficulty,
                    "board": r.board,
                    "games": r.games,
                    "profile_a_id": r.profile_a_id,
                    "profile_b_id": r.profile_b_id,
                    "wins_a": r.wins_a,
                    "wins_b": r.wins_b,
                    "draws": r.draws,
                    "non_draws": r.non_draws,
                    "winrate_a_excl_draws": r.winrate_a_excl_draws,
                    "winrate_b_excl_draws": r.winrate_b_excl_draws,
                }
            )


def run_experiments(args: argparse.Namespace) -> List[MatchStats]:
    mode = args.mode

    difficulties = _parse_int_list(args.difficulties)
    boards = _parse_list(args.boards)
    games = args.games_per_match

    results: List[MatchStats] = []

    if mode == "baseline-vs-trained":
        if not args.trained_profiles_a:
            raise SystemExit(
                "--trained-profiles-a is required when "
                "mode=baseline-vs-trained"
            )

        # Load trained profiles under *_trained ids for A/B comparison.
        load_trained_profiles_if_available(
            path=args.trained_profiles_a,
            mode="suffix",
            suffix="_trained",
        )

        base_id_a = args.base_profile_id_a
        trained_id = f"{base_id_a}_trained"

        if trained_id not in HEURISTIC_WEIGHT_PROFILES:
            raise SystemExit(
                f"Trained profile {trained_id!r} not found after loading "
                f"{args.trained_profiles_a!r}."
            )

        for d in difficulties:
            for b in boards:
                stats = run_match(
                    mode=mode,
                    profile_a_id=base_id_a,
                    profile_b_id=trained_id,
                    difficulty=d,
                    board=b,
                    games=games,
                )
                results.append(stats)

    elif mode == "ab-trained":
        if not args.trained_profiles_b:
            raise SystemExit(
                "--trained-profiles-b is required when mode=ab-trained"
            )

        # Load A under *_A, B under *_B
        load_trained_profiles_if_available(
            path=args.trained_profiles_a,
            mode="suffix",
            suffix="_A",
        )
        load_trained_profiles_if_available(
            path=args.trained_profiles_b,
            mode="suffix",
            suffix="_B",
        )

        base_id_a = args.base_profile_id_a
        base_id_b = args.base_profile_id_b
        profile_a = f"{base_id_a}_A"
        profile_b = f"{base_id_b}_B"

        for pid in (profile_a, profile_b):
            if pid not in HEURISTIC_WEIGHT_PROFILES:
                raise SystemExit(
                    f"Trained profile {pid!r} not found after loading A/B "
                    "trained files."
                )

        for d in difficulties:
            for b in boards:
                stats = run_match(
                    mode=mode,
                    profile_a_id=profile_a,
                    profile_b_id=profile_b,
                    difficulty=d,
                    board=b,
                    games=games,
                )
                results.append(stats)

    else:
        raise SystemExit(f"Unknown mode: {mode!r}")

    return results


def run_cmaes_train(args: argparse.Namespace) -> None:
    """Orchestrate a CMA-ES training run based on CLI arguments.

    This resolves a baseline heuristic profile to concrete weights,
    writes them to disk for the CMA-ES driver, and then invokes
    :func:`run_cmaes_optimization` with a constructed
    :class:`CMAESConfig`.

    For 2-player heuristic optimisation runs, this helper wires the
    CMA-ES driver into the shared 2-player training preset exposed by
    :mod:`app.training.env`:

    - multi-board evaluation on Square8, Square19, and Hexagonal boards;
    - ``eval_mode='multi-start'`` from the ``'v1'`` state pools; and
    - a small non-zero ``eval_randomness`` for symmetry breaking.

    Callers that need a different evaluation regime (for example,
    single-board experiments) should invoke
    :func:`run_cmaes_optimization` directly with a custom
    :class:`CMAESConfig`.
    """
    # Resolve baseline heuristic profile to raw weights.
    baseline_profile_id = args.cmaes_baseline_profile_id
    if baseline_profile_id not in HEURISTIC_WEIGHT_PROFILES:
        raise SystemExit(
            f"Unknown baseline profile id for CMA-ES: {baseline_profile_id!r}"
        )
    baseline_weights = HEURISTIC_WEIGHT_PROFILES[baseline_profile_id]

    # Derive run identifiers and directories.
    base_output_dir = args.cmaes_output_dir or "logs/cmaes"
    run_id = args.cmaes_run_id or datetime.now().strftime(
        "cmaes_%Y%m%d_%H%M%S",
    )
    run_dir = os.path.join(base_output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Persist baseline weights for the CMA-ES driver.
    baseline_path = os.path.join(run_dir, "baseline_weights.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_weights, f, indent=2, sort_keys=True)

    # CMA-ES output/checkpoint locations.
    output_path = os.path.join(run_dir, "best_weights.json")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    # Map board string to enum used by the CMA-ES driver.
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.cmaes_board]

    # Canonical 2-player training evaluation kwargs. These are threaded into
    # CMAESConfig so that the driver evaluates candidates using the same
    # multi-board, multi-start, light-randomness regime as
    # DEFAULT_TRAINING_EVAL_CONFIG / TWO_PLAYER_TRAINING_PRESET.
    two_player_kwargs = get_two_player_training_kwargs(
        games_per_eval=args.cmaes_games_per_eval,
        seed=args.cmaes_seed or 0,
    )
    eval_boards = two_player_kwargs["boards"]
    eval_mode = two_player_kwargs["eval_mode"]
    state_pool_id = two_player_kwargs["state_pool_id"]
    eval_randomness = two_player_kwargs["eval_randomness"]

    config = CMAESConfig(
        generations=args.cmaes_generations,
        population_size=args.cmaes_population_size,
        games_per_eval=two_player_kwargs["games_per_eval"],
        sigma=args.cmaes_sigma,
        output_path=output_path,
        baseline_path=baseline_path,
        board_type=board_type,
        checkpoint_dir=checkpoint_dir,
        seed=args.cmaes_seed,
        max_moves=args.cmaes_max_moves,
        opponent_mode=args.cmaes_opponent_mode,
        run_id=run_id,
        run_dir=run_dir,
        resume_from=None,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        eval_boards=eval_boards,
        eval_randomness=eval_randomness,
    )

    print(
        "\n=== Starting CMA-ES training via heuristic "
        "experiment harness ==="
    )
    print(f"Baseline profile id: {baseline_profile_id}")
    print(f"Run directory:       {run_dir}")
    print(f"Baseline weights:    {baseline_path}")
    run_cmaes_optimization(config)
    print("=== CMA-ES training run completed ===\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-vs-trained or A/B experiments for HeuristicAI "
            "profiles, or orchestrate CMA-ES training runs, and emit a "
            "summary report."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline-vs-trained", "ab-trained", "cmaes-train"],
        default="baseline-vs-trained",
        help="Experiment mode (default: baseline-vs-trained).",
    )
    parser.add_argument(
        "--trained-profiles-a",
        required=False,
        help=(
            "Path to trained profiles JSON file for experiment A. In "
            "baseline-vs-trained mode this is the single trained file "
            "used to construct the *_trained profile."
        ),
    )
    parser.add_argument(
        "--trained-profiles-b",
        help=(
            "Path to trained profiles JSON file for experiment B (required "
            "when mode=ab-trained)."
        ),
    )
    parser.add_argument(
        "--base-profile-id-a",
        default="heuristic_v1_balanced",
        help=(
            "Baseline profile id for side A (default: heuristic_v1_balanced). "
            "In baseline-vs-trained mode this is the untrained baseline."
        ),
    )
    parser.add_argument(
        "--base-profile-id-b",
        default="heuristic_v1_balanced",
        help=(
            "Baseline profile id for side B (only used in ab-trained mode; "
            "default: heuristic_v1_balanced)."
        ),
    )
    parser.add_argument(
        "--difficulties",
        default="5",
        help=(
            "Comma-separated list of heuristic difficulties to test "
            "(default: '5')."
        ),
    )
    parser.add_argument(
        "--boards",
        default="Square8",
        help=(
            "Comma-separated list of board types to test (choices: "
            "Square8,Square19,Hex). Default: 'Square8'."
        ),
    )
    parser.add_argument(
        "--games-per-match",
        type=int,
        default=200,
        help=(
            "Number of games per (difficulty, board) pairing "
            "(default: 200)."
        ),
    )
    parser.add_argument(
        "--out-json",
        help=(
            "Optional path to write a JSON summary of all experiments. "
            "Directories are created if needed."
        ),
    )
    parser.add_argument(
        "--out-csv",
        help=(
            "Optional path to write a CSV summary of all experiments. "
            "Directories are created if needed."
        ),
    )
    # CMA-ES training options (used when mode=cmaes-train).
    parser.add_argument(
        "--cmaes-generations",
        type=int,
        default=50,
        help=(
            "Number of CMA-ES generations when mode=cmaes-train "
            "(default: 50)."
        ),
    )
    parser.add_argument(
        "--cmaes-population-size",
        type=int,
        default=20,
        help=(
            "Population size per CMA-ES generation when mode=cmaes-train "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--cmaes-games-per-eval",
        type=int,
        default=10,
        help=(
            "Games per candidate evaluation when mode=cmaes-train "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--cmaes-sigma",
        type=float,
        default=0.5,
        help=(
            "Initial CMA-ES step size when mode=cmaes-train "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--cmaes-output-dir",
        default="logs/cmaes",
        help=(
            "Base directory for CMA-ES runs when mode=cmaes-train "
            "(default: logs/cmaes)."
        ),
    )
    parser.add_argument(
        "--cmaes-board",
        type=str,
        choices=["square8", "square19", "hex"],
        default="square8",
        help=(
            "Board type for CMA-ES self-play when mode=cmaes-train "
            "(default: square8)."
        ),
    )
    parser.add_argument(
        "--cmaes-max-moves",
        type=int,
        default=200,
        help=(
            "Maximum moves per CMA-ES self-play game before declaring a "
            "draw when mode=cmaes-train (default: 200)."
        ),
    )
    parser.add_argument(
        "--cmaes-seed",
        type=int,
        default=None,
        help=(
            "Random seed for CMA-ES runs when mode=cmaes-train "
            "(optional)."
        ),
    )
    parser.add_argument(
        "--cmaes-opponent-mode",
        type=str,
        choices=["baseline-only", "baseline-plus-incumbent"],
        default="baseline-only",
        help=(
            "Opponent pool mode for CMA-ES fitness evaluation when "
            "mode=cmaes-train (default: baseline-only)."
        ),
    )
    parser.add_argument(
        "--cmaes-baseline-profile-id",
        type=str,
        default="heuristic_v1_balanced",
        help=(
            "Baseline heuristic profile id whose weights seed CMA-ES and "
            "act as the evaluation baseline when mode=cmaes-train "
            "(default: heuristic_v1_balanced)."
        ),
    )
    parser.add_argument(
        "--cmaes-run-id",
        type=str,
        default=None,
        help=(
            "Logical run identifier used for CMA-ES training when "
            "mode=cmaes-train (optional)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode in ("baseline-vs-trained", "ab-trained"):
        results = run_experiments(args)

        # Human-readable summary
        for r in results:
            print("\n=== Heuristic Experiment Result ===")
            print(f"Mode:              {r.mode}")
            print(f"Difficulty:        {r.difficulty}")
            print(f"Board:             {r.board}")
            print(f"Games:             {r.games}")
            print(f"Profile A:         {r.profile_a_id}")
            print(f"Profile B:         {r.profile_b_id}")
            print("-----------------------------------")
            print(f"A wins:            {r.wins_a}")
            print(f"B wins:            {r.wins_b}")
            print(f"Draws:             {r.draws}")
            print(f"Non-draws:         {r.non_draws}")
            print(
                f"A winrate (no draws): {r.winrate_a_excl_draws:.3f} | "
                f"B winrate: {r.winrate_b_excl_draws:.3f}"
            )
            print("===================================")

        if args.out_json:
            _write_json(args.out_json, results)
        if args.out_csv:
            _write_csv(args.out_csv, results)
    elif args.mode == "cmaes-train":
        run_cmaes_train(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
