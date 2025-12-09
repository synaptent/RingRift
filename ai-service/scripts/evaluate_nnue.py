#!/usr/bin/env python
"""Evaluate NNUE model performance vs heuristic evaluation.

This script runs tournaments comparing:
1. Minimax with NNUE evaluation vs Minimax with heuristic evaluation
2. Minimax+NNUE vs HeuristicAI (greedy heuristic)

Usage:
    # Quick test (10 games per matchup)
    python scripts/evaluate_nnue.py

    # Full evaluation (50 games per matchup)
    python scripts/evaluate_nnue.py --games 50

    # Specific difficulty level
    python scripts/evaluate_nnue.py --difficulty 5 --games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add the parent directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)
from app.rules.default_engine import DefaultRulesEngine


@dataclass
class MatchResult:
    """Result of a single game."""
    winner: Optional[int]  # 1, 2, or None for draw
    moves: int
    duration_sec: float
    p1_swapped: bool  # True if P1 played as player 2


@dataclass
class TournamentResult:
    """Result of a tournament between two AI configurations."""
    name: str
    p1_label: str
    p2_label: str
    p1_wins: int
    p2_wins: int
    draws: int
    total_games: int
    avg_moves: float
    avg_duration_sec: float

    @property
    def p1_win_rate(self) -> float:
        decisive = self.p1_wins + self.p2_wins
        return self.p1_wins / decisive if decisive > 0 else 0.5


def create_game_state(board_type: BoardType = BoardType.SQUARE8) -> GameState:
    """Create a fresh game state."""
    size = 8 if board_type == BoardType.SQUARE8 else 19
    if board_type == BoardType.HEXAGONAL:
        size = 5

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
            id="p1",
            username="AI1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=4,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="AI2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=4,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    return GameState(
        id=str(uuid.uuid4()),
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def run_game(
    ai1,
    ai2,
    board_type: BoardType = BoardType.SQUARE8,
    swap_sides: bool = False,
    max_moves: int = 300,
    verbose: bool = False,
) -> MatchResult:
    """Run a single game between two AI instances.

    Args:
        ai1: First AI instance (plays as P1 unless swap_sides)
        ai2: Second AI instance (plays as P2 unless swap_sides)
        board_type: Board type for the game
        swap_sides: If True, ai1 plays as P2 and ai2 plays as P1
        max_moves: Maximum moves before declaring draw
        verbose: Print game progress

    Returns:
        MatchResult with winner (from ai1's perspective), moves, and duration
    """
    game = create_game_state(board_type)
    engine = DefaultRulesEngine()

    start_time = time.time()
    moves = 0

    # Swap AI assignments if requested
    p1_ai, p2_ai = (ai2, ai1) if swap_sides else (ai1, ai2)

    while game.game_status == GameStatus.ACTIVE and moves < max_moves:
        current_ai = p1_ai if game.current_player == 1 else p2_ai
        current_ai.player_number = game.current_player

        try:
            move = current_ai.select_move(game)
        except Exception as e:
            if verbose:
                print(f"Error in select_move: {e}")
            # Opponent wins on error
            winner = 2 if game.current_player == 1 else 1
            break

        if not move:
            # No valid moves - opponent wins
            winner = 2 if game.current_player == 1 else 1
            break

        game = engine.apply_move(game, move)
        moves += 1

    duration = time.time() - start_time

    # Determine winner
    if game.game_status == GameStatus.ACTIVE:
        winner = None  # Draw due to max moves
    else:
        winner = game.winner

    # Translate winner back to ai1/ai2 perspective
    if swap_sides and winner is not None:
        winner = 3 - winner  # Flip 1<->2

    return MatchResult(
        winner=winner,
        moves=moves,
        duration_sec=duration,
        p1_swapped=swap_sides,
    )


def run_tournament(
    ai1_factory,
    ai2_factory,
    p1_label: str,
    p2_label: str,
    num_games: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    verbose: bool = False,
) -> TournamentResult:
    """Run a tournament between two AI configurations.

    Args:
        ai1_factory: Callable that creates AI1 instance
        ai2_factory: Callable that creates AI2 instance
        p1_label: Label for AI1
        p2_label: Label for AI2
        num_games: Number of games to play
        board_type: Board type for games
        verbose: Print game-by-game results

    Returns:
        TournamentResult with aggregate statistics
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0
    total_duration = 0.0

    for i in range(num_games):
        # Create fresh AI instances for each game
        ai1 = ai1_factory()
        ai2 = ai2_factory()

        # Alternate sides for fairness
        swap = (i % 2 == 1)

        result = run_game(ai1, ai2, board_type, swap_sides=swap, verbose=verbose)

        if result.winner == 1:
            p1_wins += 1
            winner_label = p1_label
        elif result.winner == 2:
            p2_wins += 1
            winner_label = p2_label
        else:
            draws += 1
            winner_label = "Draw"

        total_moves += result.moves
        total_duration += result.duration_sec

        if verbose:
            side_note = " (swapped)" if swap else ""
            print(f"  Game {i+1}/{num_games}: {winner_label} wins in {result.moves} moves ({result.duration_sec:.1f}s){side_note}")

    return TournamentResult(
        name=f"{p1_label} vs {p2_label}",
        p1_label=p1_label,
        p2_label=p2_label,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        total_games=num_games,
        avg_moves=total_moves / num_games if num_games > 0 else 0,
        avg_duration_sec=total_duration / num_games if num_games > 0 else 0,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NNUE model performance vs heuristic evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per matchup (default: 10)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=4,
        help="Difficulty level for Minimax AI (default: 4, minimum for NNUE)",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print game-by-game results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
    }
    board_type = board_map[args.board]
    difficulty = max(4, args.difficulty)  # NNUE requires D4+

    print("=" * 60)
    print("NNUE Evaluation Tournament")
    print("=" * 60)
    print(f"Board type: {args.board}")
    print(f"Difficulty: {difficulty}")
    print(f"Games per matchup: {args.games}")
    print()

    results: List[TournamentResult] = []

    # Tournament 1: Minimax+NNUE vs Minimax+Heuristic (same difficulty)
    print("Tournament 1: Minimax+NNUE vs Minimax+Heuristic")
    print("-" * 60)

    def make_minimax_nnue():
        # Ensure NNUE is enabled
        os.environ.pop("RINGRIFT_DISABLE_NEURAL_NET", None)
        return MinimaxAI(1, AIConfig(difficulty=difficulty, think_time=0, randomness=0))

    def make_minimax_heuristic():
        # Disable NNUE
        os.environ["RINGRIFT_DISABLE_NEURAL_NET"] = "1"
        ai = MinimaxAI(2, AIConfig(difficulty=difficulty, think_time=0, randomness=0))
        # Reset env for next factory call
        os.environ.pop("RINGRIFT_DISABLE_NEURAL_NET", None)
        return ai

    result1 = run_tournament(
        make_minimax_nnue,
        make_minimax_heuristic,
        f"Minimax+NNUE D{difficulty}",
        f"Minimax+Heuristic D{difficulty}",
        num_games=args.games,
        board_type=board_type,
        verbose=args.verbose,
    )
    results.append(result1)

    print()
    print(f"Results: {result1.p1_label} {result1.p1_wins} - {result1.p2_wins} {result1.p2_label} ({result1.draws} draws)")
    print(f"Win rate: {result1.p1_win_rate*100:.1f}% for {result1.p1_label}")
    print(f"Avg moves: {result1.avg_moves:.1f}, Avg duration: {result1.avg_duration_sec:.2f}s")
    print()

    # Tournament 2: Minimax+NNUE vs HeuristicAI (greedy)
    print("Tournament 2: Minimax+NNUE vs HeuristicAI (greedy)")
    print("-" * 60)

    def make_heuristic_ai():
        return HeuristicAI(2, AIConfig(difficulty=difficulty, think_time=0, randomness=0))

    result2 = run_tournament(
        make_minimax_nnue,
        make_heuristic_ai,
        f"Minimax+NNUE D{difficulty}",
        f"HeuristicAI D{difficulty}",
        num_games=args.games,
        board_type=board_type,
        verbose=args.verbose,
    )
    results.append(result2)

    print()
    print(f"Results: {result2.p1_label} {result2.p1_wins} - {result2.p2_wins} {result2.p2_label} ({result2.draws} draws)")
    print(f"Win rate: {result2.p1_win_rate*100:.1f}% for {result2.p1_label}")
    print(f"Avg moves: {result2.avg_moves:.1f}, Avg duration: {result2.avg_duration_sec:.2f}s")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        nnue_better = r.p1_win_rate > 0.5
        assessment = "NNUE is stronger" if nnue_better else "Heuristic is stronger" if r.p1_win_rate < 0.5 else "Equal"
        print(f"{r.name}: {assessment} ({r.p1_win_rate*100:.1f}% win rate)")

    # Save results if requested
    if args.output:
        output_data = {
            "board_type": args.board,
            "difficulty": difficulty,
            "games_per_matchup": args.games,
            "timestamp": datetime.now().isoformat(),
            "tournaments": [
                {
                    "name": r.name,
                    "p1_label": r.p1_label,
                    "p2_label": r.p2_label,
                    "p1_wins": r.p1_wins,
                    "p2_wins": r.p2_wins,
                    "draws": r.draws,
                    "p1_win_rate": r.p1_win_rate,
                    "avg_moves": r.avg_moves,
                    "avg_duration_sec": r.avg_duration_sec,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
