#!/usr/bin/env python3
"""Quick benchmark for AI optimizations."""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment flags for all optimizations
os.environ["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
os.environ["RINGRIFT_USE_MAKE_UNMAKE"] = "true"
os.environ["RINGRIFT_USE_BATCH_EVAL"] = "true"
os.environ["RINGRIFT_BATCH_EVAL_THRESHOLD"] = "50"
os.environ["RINGRIFT_USE_FAST_TERRITORY"] = "true"
os.environ["RINGRIFT_USE_MOVE_CACHE"] = "true"
os.environ["RINGRIFT_EARLY_TERM_THRESHOLD"] = "50"

from app.ai.heuristic_ai import HeuristicAI  # noqa: E402
from app.models import (  # noqa: E402
    AIConfig,
    BoardType,
    GameState,
    GamePhase,
    GameStatus,
    BoardState,
    Player,
    TimeControl,
)


def create_starting_state(board_type: BoardType, board_size: int) -> GameState:
    """Create a starting game state for benchmarking."""
    # Create players
    players = [
        Player(
            id="p1",
            username="AI1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=19,
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
            aiDifficulty=5,
            ringsInHand=19,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    # Create empty board
    board = BoardState(
        type=board_type,
        size=board_size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )

    # Create time control
    time_control = TimeControl(
        initialTime=600000,
        increment=5000,
        type="fischer",
    )

    # Create game state
    now = datetime.now()
    state = GameState(
        id="benchmark-game",
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
        totalRingsInPlay=38,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )

    return state


def run_benchmark(board_type, board_size, num_games=3, max_moves=30):
    """Run self-play benchmark and return stats."""
    config1 = AIConfig(difficulty=5, rng_seed=42)
    config2 = AIConfig(difficulty=5, rng_seed=43)

    total_moves = 0
    total_time = 0.0

    for game_idx in range(num_games):
        state = create_starting_state(board_type, board_size)

        ai1 = HeuristicAI(1, config1)
        ai2 = HeuristicAI(2, config2)

        # Use engine from AI
        engine = ai1.rules_engine

        for move_idx in range(max_moves):
            current_ai = ai1 if state.current_player == 1 else ai2

            start = time.perf_counter()
            move = current_ai.select_move(state)
            elapsed = time.perf_counter() - start

            if move is None:
                break

            total_moves += 1
            total_time += elapsed

            # Apply move
            state = engine.apply_move(state, move)

            if state.game_status == GameStatus.COMPLETED:
                break

    moves_per_sec = total_moves / total_time if total_time > 0 else 0
    avg_ms = (total_time / total_moves * 1000) if total_moves > 0 else 0
    return total_moves, total_time, moves_per_sec, avg_ms


def main():
    print("Running benchmarks with all optimizations enabled...")
    print()

    # 8x8 benchmark
    moves, time_s, mps, avg_ms = run_benchmark(BoardType.SQUARE8, 8, num_games=3, max_moves=30)
    print(f"8x8 Board:")
    print(f"  Total moves: {moves}")
    print(f"  Total time: {time_s:.2f}s")
    print(f"  Moves/sec: {mps:.1f}")
    print(f"  Avg time/move: {avg_ms:.1f}ms")
    print()

    # Hex benchmark
    moves, time_s, mps, avg_ms = run_benchmark(BoardType.HEXAGONAL, 7, num_games=2, max_moves=20)
    print(f"Hex11 Board:")
    print(f"  Total moves: {moves}")
    print(f"  Total time: {time_s:.2f}s")
    print(f"  Moves/sec: {mps:.1f}")
    print(f"  Avg time/move: {avg_ms:.1f}ms")
    print()

    print("Done!")


if __name__ == "__main__":
    main()
