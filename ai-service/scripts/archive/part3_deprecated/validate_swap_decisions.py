#!/usr/bin/env python
"""Validate swap decision weights through self-play experiments.

This script empirically tests whether the Opening Position Classifier weights
produce better swap decisions by comparing win rates across different opening
scenarios.

The experiment:
1. P1 places their first ring in a specific position (center, corner, edge, etc.)
2. P2 decides whether to swap based on the classifier score
3. We measure P2's win rate to validate that swap decisions are correct

Key hypothesis to validate:
- P2 should swap MORE often when P1 opens in center (high strength)
- P2 should swap LESS often when P1 opens in corner (low strength)
- Win rate should be higher when P2 makes "correct" swap decisions

Usage:
    python scripts/validate_swap_decisions.py --games 100 --verbose
    python scripts/validate_swap_decisions.py --games 50 --position center
    python scripts/validate_swap_decisions.py --games 50 --sweep  # Test all positions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS, HeuristicWeights
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.rules.default_engine import DefaultRulesEngine


@dataclass
class SwapExperimentResult:
    """Result from a single swap experiment game."""

    p1_opening_position: tuple[int, int]
    opening_strength: float
    p2_swapped: bool
    winner: int | None
    move_count: int
    p2_player_number: int  # After potential swap


@dataclass
class SwapExperimentSummary:
    """Summary statistics for a set of swap experiments."""

    position_type: str
    position: tuple[int, int]
    games_played: int
    avg_opening_strength: float
    swap_rate: float
    p2_win_rate_overall: float
    p2_win_rate_when_swapped: float
    p2_win_rate_when_not_swapped: float
    games_swapped: int
    games_not_swapped: int
    avg_game_length: float


# Position categories for testing
POSITION_CATEGORIES = {
    "center": [(3, 3), (3, 4), (4, 3), (4, 4)],
    "adjacent": [(3, 2), (2, 3), (4, 5), (5, 4)],
    "diagonal": [(2, 2), (2, 5), (5, 2), (5, 5)],
    "edge": [(0, 3), (3, 0), (7, 4), (4, 7)],
    "corner": [(0, 0), (0, 7), (7, 0), (7, 7)],
}


def create_game_with_p1_opening(
    p1_position: tuple[int, int],
    board_type: BoardType = BoardType.SQUARE8,
) -> GameState:
    """Create a game state where P1 has already placed their first ring.

    This simulates the state where P2 must decide whether to swap.
    The state includes:
    - P1's ring placement in move_history (required for swap rule)
    - rules_options with swapRuleEnabled=True
    """
    from app.models import Move

    now = datetime.now()
    x, y = p1_position

    board = BoardState(
        type=board_type,
        size=8,
        stacks={
            f"{x},{y}": RingStack(
                position=Position(x=x, y=y),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            )
        },
        markers={},
        collapsedSpaces={},
        eliminatedRings={"1": 0, "2": 0},
    )

    players = [
        Player(
            id="player1",
            username="P1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=17,  # 18 - 1 placed
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="player2",
            username="P2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    # P1's first move - required for swap rule to be available
    p1_move = Move(
        id="p1-opening",
        type=MoveType.PLACE_RING,
        player=1,
        to=Position(x=x, y=y),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    return GameState(
        id="swap-validation",
        boardType=board_type,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=2,  # P2's turn to respond (possibly swap)
        moveHistory=[p1_move],  # Include P1's move for swap rule
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=1,
        totalRingsEliminated=0,
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
        rulesOptions={"swapRuleEnabled": True},  # Enable pie rule
    )


def create_ai_with_weights(
    player_number: int,
    weights: HeuristicWeights,
    randomness: float = 0.05,
    seed: int | None = None,
) -> HeuristicAI:
    """Create a HeuristicAI instance with custom weights."""
    ai = HeuristicAI(
        player_number,
        AIConfig(
            difficulty=5,
            think_time=0,
            randomness=randomness,
            rngSeed=seed,
        ),
    )
    for name, value in weights.items():
        setattr(ai, name, value)
    return ai


def play_game_from_state(
    initial_state: GameState,
    weights: HeuristicWeights,
    max_moves: int = 400,
    randomness: float = 0.05,
    seed_base: int | None = None,
) -> SwapExperimentResult:
    """Play a game from a given initial state and track swap decision.

    Returns experiment result including whether P2 swapped and game outcome.
    """
    # Find P1's opening position
    p1_stacks = [s for s in initial_state.board.stacks.values() if s.controlling_player == 1]
    if not p1_stacks:
        raise ValueError("No P1 stack found in initial state")

    p1_pos = p1_stacks[0].position
    p1_opening = (p1_pos.x, p1_pos.y)

    # Create AIs
    ai1 = create_ai_with_weights(1, weights, randomness, seed_base if seed_base else None)
    ai2 = create_ai_with_weights(2, weights, randomness, (seed_base + 1) if seed_base else None)

    # Compute opening strength for P1's position
    opening_strength = ai2.compute_opening_strength(p1_pos, initial_state)

    game_state = initial_state
    rules_engine = DefaultRulesEngine()
    move_count = 0
    p2_swapped = False
    p2_final_player_number = 2

    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = game_state.current_player
        current_ai = ai1 if current_player == 1 else ai2
        current_ai.player_number = current_player

        move = current_ai.select_move(game_state)
        if not move:
            game_state.game_status = GameStatus.COMPLETED
            game_state.winner = 2 if current_player == 1 else 1
            break

        # Track if P2 swaps on their first move
        if move_count == 0 and current_player == 2 and move.type == MoveType.SWAP_SIDES:
            p2_swapped = True
            p2_final_player_number = 1  # P2 becomes P1 after swap

        game_state = rules_engine.apply_move(game_state, move)
        move_count += 1

    # Determine winner from P2's perspective
    winner = game_state.winner

    return SwapExperimentResult(
        p1_opening_position=p1_opening,
        opening_strength=opening_strength,
        p2_swapped=p2_swapped,
        winner=winner,
        move_count=move_count,
        p2_player_number=p2_final_player_number,
    )


def run_experiment_for_position(
    position: tuple[int, int],
    num_games: int,
    weights: HeuristicWeights,
    verbose: bool = False,
) -> list[SwapExperimentResult]:
    """Run multiple games with P1 starting at a specific position."""
    results = []

    for i in range(num_games):
        initial_state = create_game_with_p1_opening(position)
        result = play_game_from_state(
            initial_state,
            weights,
            seed_base=i * 100,
        )
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_games} games for position {position}")

    return results


def analyze_results(
    results: list[SwapExperimentResult],
    position_type: str,
    position: tuple[int, int],
) -> SwapExperimentSummary:
    """Analyze experiment results and compute summary statistics."""
    games_played = len(results)
    if games_played == 0:
        return SwapExperimentSummary(
            position_type=position_type,
            position=position,
            games_played=0,
            avg_opening_strength=0.0,
            swap_rate=0.0,
            p2_win_rate_overall=0.0,
            p2_win_rate_when_swapped=0.0,
            p2_win_rate_when_not_swapped=0.0,
            games_swapped=0,
            games_not_swapped=0,
            avg_game_length=0.0,
        )

    avg_strength = sum(r.opening_strength for r in results) / games_played
    games_swapped = sum(1 for r in results if r.p2_swapped)
    games_not_swapped = games_played - games_swapped
    swap_rate = games_swapped / games_played

    # P2 wins when winner == p2_final_player_number
    p2_wins_overall = sum(1 for r in results if r.winner == r.p2_player_number)
    p2_win_rate_overall = p2_wins_overall / games_played

    # Win rate when swapped
    swapped_results = [r for r in results if r.p2_swapped]
    if swapped_results:
        p2_wins_swapped = sum(1 for r in swapped_results if r.winner == r.p2_player_number)
        p2_win_rate_swapped = p2_wins_swapped / len(swapped_results)
    else:
        p2_win_rate_swapped = 0.0

    # Win rate when not swapped
    not_swapped_results = [r for r in results if not r.p2_swapped]
    if not_swapped_results:
        p2_wins_not_swapped = sum(1 for r in not_swapped_results if r.winner == r.p2_player_number)
        p2_win_rate_not_swapped = p2_wins_not_swapped / len(not_swapped_results)
    else:
        p2_win_rate_not_swapped = 0.0

    avg_game_length = sum(r.move_count for r in results) / games_played

    return SwapExperimentSummary(
        position_type=position_type,
        position=position,
        games_played=games_played,
        avg_opening_strength=avg_strength,
        swap_rate=swap_rate,
        p2_win_rate_overall=p2_win_rate_overall,
        p2_win_rate_when_swapped=p2_win_rate_swapped,
        p2_win_rate_when_not_swapped=p2_win_rate_not_swapped,
        games_swapped=games_swapped,
        games_not_swapped=games_not_swapped,
        avg_game_length=avg_game_length,
    )


def print_summary(summary: SwapExperimentSummary) -> None:
    """Print a formatted summary of experiment results."""
    print(f"\n{'='*60}")
    print(f"Position: {summary.position} ({summary.position_type})")
    print(f"{'='*60}")
    print(f"Games played:        {summary.games_played}")
    print(f"Opening strength:    {summary.avg_opening_strength:.3f}")
    print(f"Swap rate:           {summary.swap_rate:.1%} ({summary.games_swapped}/{summary.games_played})")
    print(f"P2 win rate overall: {summary.p2_win_rate_overall:.1%}")
    print(f"P2 win rate (swap):  {summary.p2_win_rate_when_swapped:.1%} (n={summary.games_swapped})")
    print(f"P2 win rate (no swap): {summary.p2_win_rate_when_not_swapped:.1%} (n={summary.games_not_swapped})")
    print(f"Avg game length:     {summary.avg_game_length:.1f} moves")


def run_position_sweep(
    num_games_per_position: int,
    weights: HeuristicWeights,
    verbose: bool = False,
) -> dict[str, list[SwapExperimentSummary]]:
    """Run experiments for all position categories."""
    all_summaries: dict[str, list[SwapExperimentSummary]] = {}

    for category, positions in POSITION_CATEGORIES.items():
        print(f"\n--- Testing {category.upper()} positions ---")
        category_summaries = []

        for pos in positions:
            print(f"Running {num_games_per_position} games for position {pos}...")
            results = run_experiment_for_position(pos, num_games_per_position, weights, verbose)
            summary = analyze_results(results, category, pos)
            category_summaries.append(summary)
            print_summary(summary)

        all_summaries[category] = category_summaries

    return all_summaries


def print_category_comparison(summaries: dict[str, list[SwapExperimentSummary]]) -> None:
    """Print a comparison table across position categories."""
    print("\n" + "=" * 80)
    print("CATEGORY COMPARISON")
    print("=" * 80)
    print(f"{'Category':<12} {'Strength':>10} {'Swap Rate':>12} {'P2 Win':>10} {'Win if Swap':>12} {'Win if No':>10}")
    print("-" * 80)

    for category in ["center", "adjacent", "diagonal", "edge", "corner"]:
        if category not in summaries:
            continue

        cat_summaries = summaries[category]
        if not cat_summaries:
            continue

        # Aggregate across positions in category
        total_games = sum(s.games_played for s in cat_summaries)
        avg_strength = sum(s.avg_opening_strength * s.games_played for s in cat_summaries) / total_games
        total_swaps = sum(s.games_swapped for s in cat_summaries)
        swap_rate = total_swaps / total_games

        total_wins = sum(s.p2_win_rate_overall * s.games_played for s in cat_summaries) / total_games

        # Win rates conditional on swap
        swapped_games = sum(s.games_swapped for s in cat_summaries)
        not_swapped_games = sum(s.games_not_swapped for s in cat_summaries)

        if swapped_games > 0:
            win_if_swap = sum(s.p2_win_rate_when_swapped * s.games_swapped for s in cat_summaries) / swapped_games
        else:
            win_if_swap = 0.0

        if not_swapped_games > 0:
            win_if_no_swap = (
                sum(s.p2_win_rate_when_not_swapped * s.games_not_swapped for s in cat_summaries) / not_swapped_games
            )
        else:
            win_if_no_swap = 0.0

        print(
            f"{category:<12} {avg_strength:>10.3f} {swap_rate:>11.1%} "
            f"{total_wins:>10.1%} {win_if_swap:>11.1%} {win_if_no_swap:>10.1%}"
        )

    print("-" * 80)
    print("\nExpected behavior if weights are correct:")
    print("- Swap rate should be HIGHER for center (high strength) positions")
    print("- Swap rate should be LOWER for corner (low strength) positions")
    print("- Win rate when swapping center should be > win rate when not swapping")
    print("- Win rate when swapping corner should be < win rate when not swapping")


def main():
    parser = argparse.ArgumentParser(description="Validate swap decision weights through self-play")
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per position (default: 20)",
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["center", "adjacent", "diagonal", "edge", "corner"],
        default=None,
        help="Test only a specific position category",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run tests for all position categories",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress updates",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    weights = dict(BASE_V1_BALANCED_WEIGHTS)

    print("=" * 60)
    print("SWAP DECISION VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Games per position: {args.games}")
    print("Using base v1 balanced weights")
    print("New swap weights:")
    print(f"  WEIGHT_SWAP_OPENING_STRENGTH: {weights.get('WEIGHT_SWAP_OPENING_STRENGTH', 'N/A')}")
    print(f"  WEIGHT_SWAP_CORNER_PENALTY: {weights.get('WEIGHT_SWAP_CORNER_PENALTY', 'N/A')}")
    print(f"  WEIGHT_SWAP_EDGE_BONUS: {weights.get('WEIGHT_SWAP_EDGE_BONUS', 'N/A')}")
    print(f"  WEIGHT_SWAP_DIAGONAL_BONUS: {weights.get('WEIGHT_SWAP_DIAGONAL_BONUS', 'N/A')}")

    if args.sweep or args.position is None:
        # Run full sweep
        summaries = run_position_sweep(args.games, weights, args.verbose)
        print_category_comparison(summaries)

        if args.output:
            # Save results to JSON
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "games_per_position": args.games,
                "categories": {
                    cat: [
                        {
                            "position": s.position,
                            "opening_strength": s.avg_opening_strength,
                            "swap_rate": s.swap_rate,
                            "p2_win_rate": s.p2_win_rate_overall,
                            "p2_win_if_swap": s.p2_win_rate_when_swapped,
                            "p2_win_if_no_swap": s.p2_win_rate_when_not_swapped,
                            "games_swapped": s.games_swapped,
                            "games_not_swapped": s.games_not_swapped,
                        }
                        for s in cat_summaries
                    ]
                    for cat, cat_summaries in summaries.items()
                },
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        # Run for specific category
        category = args.position
        positions = POSITION_CATEGORIES[category]
        print(f"\n--- Testing {category.upper()} positions ---")

        for pos in positions:
            print(f"Running {args.games} games for position {pos}...")
            results = run_experiment_for_position(pos, args.games, weights, args.verbose)
            summary = analyze_results(results, category, pos)
            print_summary(summary)


if __name__ == "__main__":
    main()
