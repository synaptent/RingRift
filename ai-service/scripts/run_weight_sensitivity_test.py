#!/usr/bin/env python
"""Quick weight sensitivity test for RingRift heuristics.

Tests each single-weight profile against zero weights to identify which
weights have meaningful impact on gameplay. Much faster than a full
tournament since each weight is only tested against the baseline.

Usage:
    python scripts/run_weight_sensitivity_test.py --games-per-weight 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    Player,
    TimeControl,
)
from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS, HEURISTIC_WEIGHT_KEYS
from app.rules.default_engine import DefaultRulesEngine
from app.db import GameReplayDB, get_or_create_db, record_completed_game_with_parity_check, ParityValidationError
from app.training.env import get_theoretical_max_moves


@dataclass
class WeightResult:
    """Results for a single weight's sensitivity test."""

    weight_name: str
    weight_value: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_games: int = 0
    avg_game_length: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.total_games)

    @property
    def net_wins(self) -> int:
        return self.wins - self.losses


def create_single_weight_profile(weight_key: str, value: float) -> Dict[str, float]:
    """Create a profile with only one weight active."""
    return {k: (value if k == weight_key else 0.0) for k in HEURISTIC_WEIGHT_KEYS}


def create_zero_profile() -> Dict[str, float]:
    """Create a profile with all weights set to zero."""
    return {k: 0.0 for k in HEURISTIC_WEIGHT_KEYS}


BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex": BoardType.HEXAGONAL,
}


def create_game_state(board_type_str: str) -> GameState:
    """Create initial game state for a board type."""
    board_type = BOARD_TYPE_MAP.get(board_type_str, BoardType.SQUARE8)

    if board_type == BoardType.SQUARE8:
        size = 8
        rings_per_player = 18
    elif board_type == BoardType.SQUARE19:
        size = 19
        rings_per_player = 36
    else:
        size = 13  # Canonical hex: size=13, radius=12
        rings_per_player = 48

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    players = [
        Player(
            id=f"player{i}",
            username=f"AI {i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in (1, 2)
    ]

    now = datetime.now()
    time_control = TimeControl(initialTime=600000, increment=0, type="standard")

    return GameState(
        id="sensitivity-test",
        boardType=board_type,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=time_control,
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=rings_per_player * 2,
        totalRingsEliminated=0,
        victoryThreshold=5,
        territoryVictoryThreshold=15,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )


@dataclass
class GameResult:
    """Result of a single game including move history for recording."""

    winner: int
    move_count: int
    initial_state: GameState
    final_state: GameState
    moves: List[Move]


def play_game(
    p1_weights: Dict[str, float],
    p2_weights: Dict[str, float],
    board_type: str,
    max_moves: int,
    seed: int,
    verbose: bool = False,
    progress_interval: float = 10.0,
    use_true_random_baseline: bool = False,
    collect_moves: bool = False,
) -> tuple[int, int] | GameResult:
    """Play a single game. Returns (winner, move_count) or GameResult if collect_moves=True.

    Args:
        use_true_random_baseline: If True, zero-weights player uses uniform
            random selection over all legal moves instead of deterministic
            "first legal move" behavior.
        collect_moves: If True, collect move history and return GameResult
            for database recording.
    """
    import random as stdlib_random

    rules = DefaultRulesEngine()
    state = create_game_state(board_type)
    initial_state = state.model_copy(deep=True) if collect_moves else None

    # Create AIs with the weight profiles
    config1 = AIConfig(difficulty=5, seed=seed)
    config2 = AIConfig(difficulty=5, seed=seed + 1000)

    ai1 = HeuristicAI(1, config1)
    ai2 = HeuristicAI(2, config2)

    # Apply weight profiles
    for k, v in p1_weights.items():
        setattr(ai1, k, v)
    for k, v in p2_weights.items():
        setattr(ai2, k, v)

    # For true random baseline, create seeded RNGs for each player
    # True random = uniform selection from all legal moves at each decision
    p1_is_zero = all(v == 0.0 for v in p1_weights.values())
    p2_is_zero = all(v == 0.0 for v in p2_weights.values())
    rng1 = stdlib_random.Random(seed) if use_true_random_baseline and p1_is_zero else None
    rng2 = stdlib_random.Random(seed + 1000) if use_true_random_baseline and p2_is_zero else None

    move_count = 0
    game_start = time.time()
    last_progress = game_start
    moves_collected: List[Move] = [] if collect_moves else []

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        curr_player = state.current_player
        ai = ai1 if curr_player == 1 else ai2
        rng = rng1 if curr_player == 1 else rng2

        # True random baseline: if RNG is set for this player, pick uniformly
        # from all legal moves instead of using heuristic evaluation
        if rng is not None:
            valid_moves = rules.get_valid_moves(state, curr_player)
            if not valid_moves:
                break
            move = rng.choice(valid_moves)
        else:
            move = ai.select_move(state)
            if move is None:
                break

        if collect_moves:
            moves_collected.append(move)

        state = rules.apply_move(state, move)
        move_count += 1

        # Periodic progress within game
        now = time.time()
        if verbose and (now - last_progress) >= progress_interval:
            elapsed = now - game_start
            print(f"      [game in progress: {move_count} moves, {elapsed:.1f}s]", flush=True)
            last_progress = now

    # Determine winner from state (set by rules engine)
    # state.winner is set when game_status becomes FINISHED
    winner = state.winner if state.winner is not None else 0

    # Log error/warning for games that hit max_moves without a winner
    board_type_enum = BoardType(board_type)
    if move_count >= max_moves and state.winner is None:
        theoretical_max = get_theoretical_max_moves(board_type_enum, 2)  # 2-player games
        if move_count >= theoretical_max:
            print(
                f"ERROR: GAME_NON_TERMINATION "
                f"Game exceeded theoretical maximum moves without a winner. "
                f"board_type={board_type}, move_count={move_count}, "
                f"max_moves={max_moves}, theoretical_max={theoretical_max}, "
                f"game_status={state.game_status.value}, winner={state.winner}",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: GAME_MAX_MOVES_CUTOFF "
                f"Game hit max_moves limit without a winner. "
                f"board_type={board_type}, move_count={move_count}, "
                f"max_moves={max_moves}, theoretical_max={theoretical_max}, "
                f"game_status={state.game_status.value}, winner={state.winner}",
                file=sys.stderr,
            )

    # If game ended due to max_moves or no-move stall (shouldn't happen with
    # proper rules enforcement), we record it as 0 (inconclusive) for analysis.
    # RingRift has no draws by rule - games always end with a winner via:
    # - Ring elimination threshold
    # - Territory control threshold
    # - Last player standing
    # - Stalemate resolution (tiebreakers)

    if collect_moves:
        return GameResult(
            winner=winner,
            move_count=move_count,
            initial_state=initial_state,
            final_state=state,
            moves=moves_collected,
        )
    return winner, move_count


def run_sensitivity_test(
    board_type: str = "square8",
    games_per_weight: int = 10,
    max_moves: int = 150,
    seed: int = 42,
    verbose: bool = True,
    use_true_random_baseline: bool = False,
    replay_db: Optional[GameReplayDB] = None,
) -> List[WeightResult]:
    """Run sensitivity test for all weights.

    Args:
        use_true_random_baseline: If True, the zero-weight baseline player
            uses uniform random move selection instead of deterministic
            "first legal move" behavior.
        replay_db: Optional database for recording game replays.
    """

    results: List[WeightResult] = []
    zero_profile = create_zero_profile()
    games_recorded = 0

    total_weights = len(HEURISTIC_WEIGHT_KEYS)
    total_games = total_weights * games_per_weight
    print(
        f"\nRunning sensitivity test: {total_weights} weights × {games_per_weight} games = {total_games} games",
        flush=True,
    )
    print(f"Board: {board_type}, Max moves: {max_moves}", flush=True)
    if replay_db:
        print(f"Recording games to database", flush=True)
    print("", flush=True)

    global_game_num = 0
    test_start = time.time()

    for i, weight_key in enumerate(HEURISTIC_WEIGHT_KEYS):
        # Use the baseline value as the test magnitude
        base_value = BASE_V1_BALANCED_WEIGHTS.get(weight_key, 1.0)
        test_value = max(abs(base_value), 1.0)  # At least 1.0

        weight_profile = create_single_weight_profile(weight_key, test_value)

        result = WeightResult(
            weight_name=weight_key,
            weight_value=test_value,
        )

        game_lengths = []
        print(f"  Testing weight: {weight_key} (value={test_value:.1f})", flush=True)

        for g in range(games_per_weight):
            game_seed = seed + i * 1000 + g
            global_game_num += 1
            game_start = time.time()

            # Alternate who plays first for fairness
            if g % 2 == 0:
                game_result = play_game(
                    weight_profile,
                    zero_profile,
                    board_type,
                    max_moves,
                    game_seed,
                    verbose=verbose,
                    use_true_random_baseline=use_true_random_baseline,
                    collect_moves=(replay_db is not None),
                )
                if isinstance(game_result, GameResult):
                    winner = game_result.winner
                    moves = game_result.move_count
                else:
                    winner, moves = game_result
                    game_result = None

                if winner == 1:
                    result.wins += 1
                    outcome = "W"
                elif winner == 2:
                    result.losses += 1
                    outcome = "L"
                else:
                    result.draws += 1
                    outcome = "D"
            else:
                game_result = play_game(
                    zero_profile,
                    weight_profile,
                    board_type,
                    max_moves,
                    game_seed,
                    verbose=verbose,
                    use_true_random_baseline=use_true_random_baseline,
                    collect_moves=(replay_db is not None),
                )
                if isinstance(game_result, GameResult):
                    winner = game_result.winner
                    moves = game_result.move_count
                else:
                    winner, moves = game_result
                    game_result = None

                if winner == 2:
                    result.wins += 1
                    outcome = "W"
                elif winner == 1:
                    result.losses += 1
                    outcome = "L"
                else:
                    result.draws += 1
                    outcome = "D"

            # Record game to database if enabled
            if replay_db is not None and game_result is not None:
                try:
                    record_completed_game_with_parity_check(
                        db=replay_db,
                        initial_state=game_result.initial_state,
                        final_state=game_result.final_state,
                        moves=game_result.moves,
                        metadata={
                            "source": "sensitivity_test",
                            "weight_key": weight_key,
                            "weight_value": test_value,
                            "game_index": g,
                            "rng_seed": game_seed,
                        },
                    )
                    games_recorded += 1
                except ParityValidationError as pve:
                    print(
                        f"[PARITY ERROR] Game diverged at k={pve.divergence.move_index}:\n"
                        f"  Bundle: {pve.divergence.bundle_path or 'N/A'}",
                        file=sys.stderr,
                    )
                    raise
                except Exception as exc:
                    print(
                        f"    [record-db] Failed to record game: " f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

            game_lengths.append(moves)
            result.total_games += 1
            game_elapsed = time.time() - game_start

            # Per-game output
            elapsed_total = time.time() - test_start
            games_done = global_game_num
            if games_done > 0:
                eta_seconds = (elapsed_total / games_done) * (total_games - games_done)
                eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds < 3600 else f"{eta_seconds / 3600:.1f}h"
            else:
                eta_str = "?"
            print(
                f"    [{global_game_num:3d}/{total_games}] game {g+1}/{games_per_weight}: {outcome} in {moves} moves ({game_elapsed:.1f}s) | ETA: {eta_str}",
                flush=True,
            )

        result.avg_game_length = sum(game_lengths) / len(game_lengths)
        results.append(result)

        # Weight summary
        status = "+" if result.win_rate > 0.6 else "-" if result.win_rate < 0.4 else "~"
        print(
            f"  => [{i+1:2d}/{total_weights}] {weight_key:40s} | W:{result.wins:2d} L:{result.losses:2d} D:{result.draws:2d} | WR:{result.win_rate:.0%} {status}\n",
            flush=True,
        )

    if replay_db:
        print(f"[record-db] Recorded {games_recorded}/{total_games} games", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick weight sensitivity test")
    parser.add_argument("--board", default="square8", choices=["square8", "square19", "hex"])
    parser.add_argument("--games-per-weight", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="logs/axis_aligned/sensitivity_results.json")
    parser.add_argument(
        "--use-true-random",
        action="store_true",
        help="Use uniform random move selection for zero-weight baseline instead of "
        "deterministic 'first legal move' behavior",
    )
    parser.add_argument(
        "--record-db",
        type=str,
        default="data/games/sensitivity.db",
        help="Path to SQLite database for recording game replays. "
        "Default: data/games/sensitivity.db. Use --no-record-db to disable.",
    )
    parser.add_argument(
        "--no-record-db",
        action="store_true",
        help="Disable game recording to database (overrides --record-db).",
    )
    args = parser.parse_args()

    # Initialize optional game recording database
    # --no-record-db flag overrides --record-db to disable recording
    record_db_path = None if args.no_record_db else args.record_db
    replay_db = get_or_create_db(record_db_path) if record_db_path else None

    start = time.time()
    results = run_sensitivity_test(
        board_type=args.board,
        games_per_weight=args.games_per_weight,
        max_moves=args.max_moves,
        seed=args.seed,
        use_true_random_baseline=args.use_true_random,
        replay_db=replay_db,
    )
    elapsed = time.time() - start

    # Sort by win rate
    results.sort(key=lambda r: r.win_rate, reverse=True)

    print(f"\n{'='*80}")
    print(f"SENSITIVITY RESULTS (sorted by win rate)")
    print(f"{'='*80}")
    print(f"{'Weight':<45} {'WinRate':>8} {'Net':>5} {'AvgLen':>7}")
    print(f"{'-'*80}")

    significant_weights = []
    neutral_weights = []

    for r in results:
        marker = ""
        if r.win_rate >= 0.7:
            marker = "+++"
            significant_weights.append((r.weight_name, r.win_rate, "strong"))
        elif r.win_rate >= 0.6:
            marker = "+"
            significant_weights.append((r.weight_name, r.win_rate, "moderate"))
        elif r.win_rate <= 0.3:
            marker = "---"
            significant_weights.append((r.weight_name, r.win_rate, "negative"))
        elif r.win_rate <= 0.4:
            marker = "-"
            significant_weights.append((r.weight_name, r.win_rate, "weak negative"))
        else:
            neutral_weights.append(r.weight_name)

        print(f"{r.weight_name:<45} {r.win_rate:>7.0%} {r.net_wins:>+5d} {r.avg_game_length:>7.1f} {marker}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(results):.2f}s per weight)")
    print(f"\nSignificant weights ({len(significant_weights)}):")
    for name, wr, strength in significant_weights:
        print(f"  {strength:>15}: {name} ({wr:.0%})")
    print(
        f"\nNeutral weights ({len(neutral_weights)}): {', '.join(neutral_weights[:5])}{'...' if len(neutral_weights) > 5 else ''}"
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "config": {
            "board": args.board,
            "games_per_weight": args.games_per_weight,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "use_true_random": args.use_true_random,
        },
        "results": [
            {
                "weight": r.weight_name,
                "value": r.weight_value,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
                "win_rate": r.win_rate,
                "avg_game_length": r.avg_game_length,
            }
            for r in results
        ],
        "elapsed_seconds": elapsed,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
