"""
Benchmark script to compare AI engines
"""

import sys
import os
import time
import random
from datetime import datetime

from app.models import GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl, Player, AIConfig
from app.game_engine import GameEngine
from app.rules.default_engine import DefaultRulesEngine
from app.ai.mcts_ai import MCTSAI
from app.ai.descent_ai import DescentAI


def create_initial_state():
    return GameState(
        id="benchmark",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8, stacks={}, markers={}, collapsedSpaces={}, eliminatedRings={}),
        players=[
            Player(
                id="p1",
                username="AI 1",
                type="ai",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=5,
            ),
            Player(
                id="p2",
                username="AI 2",
                type="ai",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=5,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold=33,
    )


def run_benchmark(num_games=10):
    print(f"Running benchmark: MCTS vs Descent ({num_games} games)", flush=True)

    # Use moderate think time for benchmark accuracy
    config = AIConfig(difficulty=5, randomness=0.1, thinkTime=500)

    # MCTS vs Descent
    mcts_wins = 0
    descent_wins = 0
    draws = 0

    rules_engine = DefaultRulesEngine()

    for i in range(num_games):
        # Clear cache between games to prevent state leakage
        GameEngine.clear_cache()

        # Alternate starting player
        if i % 2 == 0:
            p1_ai = MCTSAI(1, config)
            p2_ai = DescentAI(2, config)
            p1_name = "MCTS"
            p2_name = "Descent"
        else:
            p1_ai = DescentAI(1, config)
            p2_ai = MCTSAI(2, config)
            p1_name = "Descent"
            p2_name = "MCTS"

        state = create_initial_state()
        move_count = 0

        print(f"Game {i+1}: {p1_name} (P1) vs {p2_name} (P2)", flush=True)

        while state.game_status == GameStatus.ACTIVE and move_count < 200:
            current_player = state.current_player
            ai = p1_ai if current_player == 1 else p2_ai

            start_time = time.time()
            move = ai.select_move(state)
            duration = time.time() - start_time

            if not move:
                print(
                    f"  No move found for P{current_player} (Phase: {state.current_phase}, MustMove: {state.must_move_from_stack_key})",
                    flush=True,
                )
                # If no move found, current player loses
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.COMPLETED
                break

            # state = GameEngine.apply_move(state, move)
            state = rules_engine.apply_move(state, move)
            move_count += 1

            print(f"  Move {move_count} ({duration:.2f}s) - P{current_player} played {move.type}", flush=True)

        winner = state.winner
        print(f"  Winner: P{winner}", flush=True)

        # Correct win tracking based on player roles
        if winner == 1:
            if p1_name == "MCTS":
                mcts_wins += 1
            else:
                descent_wins += 1
        elif winner == 2:
            if p2_name == "MCTS":
                mcts_wins += 1
            else:
                descent_wins += 1
        else:
            draws += 1

    print("\nResults:")
    print(f"MCTS Wins: {mcts_wins}")
    print(f"Descent Wins: {descent_wins}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    run_benchmark(num_games=20)  # Longer run for statistical significance
