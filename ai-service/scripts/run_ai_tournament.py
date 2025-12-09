import sys
import os
import time
import argparse
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add the parent directory to sys.path to allow imports from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.progress_reporter import SoakProgressReporter

from app.models import GameState, BoardType, GamePhase, GameStatus, Player, TimeControl, BoardState, AIConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.random_ai import RandomAI
from app.rules.default_engine import DefaultRulesEngine

# Map string names to AI classes
AI_CLASSES = {"Random": RandomAI, "Heuristic": HeuristicAI, "Minimax": MinimaxAI, "MCTS": MCTSAI}

BOARD_TYPES = {"Square8": BoardType.SQUARE8, "Square19": BoardType.SQUARE19, "Hex": BoardType.HEXAGONAL}


def create_game_state(board_type_str: str, p1_config: Dict, p2_config: Dict) -> GameState:
    board_type = BOARD_TYPES.get(board_type_str, BoardType.SQUARE8)

    size = 8
    if board_type == BoardType.SQUARE19:
        size = 19
    elif board_type == BoardType.HEXAGONAL:
        size = 5  # Standard hex size (radius 4)

    board = BoardState(type=board_type, size=size, stacks={}, markers={}, collapsedSpaces={}, eliminatedRings={})

    players = [
        Player(
            id="player1",
            username=f"{p1_config['type']}_L{p1_config['difficulty']}",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=p1_config["difficulty"],
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="player2",
            username=f"{p2_config['type']}_L{p2_config['difficulty']}",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=p2_config["difficulty"],
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


def run_game(p1_ai, p2_ai, board_type: str) -> int:
    """
    Run a single game between two AI instances.
    Returns the winner (1 or 2), or 0 for draw.
    """
    # Create fresh game state
    p1_config = {"type": p1_ai.__class__.__name__.replace("AI", ""), "difficulty": p1_ai.config.difficulty}
    p2_config = {"type": p2_ai.__class__.__name__.replace("AI", ""), "difficulty": p2_ai.config.difficulty}

    game_state = create_game_state(board_type, p1_config, p2_config)
    rules_engine = DefaultRulesEngine()

    move_count = 0
    max_moves = 300  # Prevent infinite games

    print(f"Starting game: {p1_ai.__class__.__name__} (P1) vs " f"{p2_ai.__class__.__name__} (P2) on {board_type}")

    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player_num = game_state.current_player
        current_ai = p1_ai if current_player_num == 1 else p2_ai

        # Ensure AI has correct player number assigned
        current_ai.player_number = current_player_num

        try:
            move = current_ai.select_move(game_state)
        except Exception as e:
            print(f"Error in AI select_move: {e}")
            import traceback

            traceback.print_exc()
            return 2 if current_player_num == 1 else 1

        if not move:
            print(f"No valid moves for Player {current_player_num}. Game Over.")
            game_state.game_status = GameStatus.COMPLETED
            game_state.winner = 2 if current_player_num == 1 else 1
            break

        # Apply move using DefaultRulesEngine
        try:
            game_state = rules_engine.apply_move(game_state, move)
        except Exception as e:
            print(f"Error applying move: {e}")
            import traceback

            traceback.print_exc()
            return 2 if current_player_num == 1 else 1

        # Check victory
        if game_state.game_status == GameStatus.COMPLETED:
            break

        move_count += 1

    if game_state.game_status == GameStatus.ACTIVE:
        print("Max moves reached. Draw.")
        return 0

    print(f"Game Over. Winner: Player {game_state.winner}")
    return game_state.winner if game_state.winner is not None else 0


def main():
    parser = argparse.ArgumentParser(description="Run AI Tournament")
    parser.add_argument("--p1", type=str, required=True, choices=AI_CLASSES.keys(), help="Player 1 AI Type")
    parser.add_argument("--p1-diff", type=int, default=5, help="Player 1 Difficulty (1-10)")
    parser.add_argument("--p2", type=str, required=True, choices=AI_CLASSES.keys(), help="Player 2 AI Type")
    parser.add_argument("--p2-diff", type=int, default=5, help="Player 2 Difficulty (1-10)")
    parser.add_argument("--board", type=str, default="Square8", choices=BOARD_TYPES.keys(), help="Board Type")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")

    args = parser.parse_args()

    print(f"Tournament Configuration:")
    print(f"  Player 1: {args.p1} (Difficulty {args.p1_diff})")
    print(f"  Player 2: {args.p2} (Difficulty {args.p2_diff})")
    print(f"  Board: {args.board}")
    print(f"  Games: {args.games}")
    print("-" * 40)

    # Initialize AIs
    # Note: We re-initialize or reset AIs per game if needed, but here we create instances once
    # and update player_number in run_game.

    p1_class = AI_CLASSES[args.p1]
    p2_class = AI_CLASSES[args.p2]

    # Create AI instances
    # We use player number 1 for initialization, but it will be updated in run_game
    ai1 = p1_class(1, AIConfig(difficulty=args.p1_diff, think_time=0, randomness=0, rngSeed=None))
    ai2 = p2_class(2, AIConfig(difficulty=args.p2_diff, think_time=0, randomness=0, rngSeed=None))

    stats = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=args.games,
        report_interval_sec=10.0,
        context_label=f"{args.p1}_vs_{args.p2}_{args.board}",
    )

    for i in range(args.games):
        game_start_time = time.time()
        print(f"\nMatch {i+1}/{args.games}")

        # Swap sides every other game to ensure fairness
        if i % 2 == 0:
            # P1 is Player 1
            winner = run_game(ai1, ai2, args.board)
            if winner == 1:
                stats["p1_wins"] += 1
            elif winner == 2:
                stats["p2_wins"] += 1
            else:
                stats["draws"] += 1
        else:
            # P1 is Player 2 (swap)
            # run_game expects (p1_ai, p2_ai) where p1_ai plays as Player 1
            winner = run_game(ai2, ai1, args.board)
            if winner == 1:
                stats["p2_wins"] += 1  # ai2 (P2 originally) won as Player 1
            elif winner == 2:
                stats["p1_wins"] += 1  # ai1 (P1 originally) won as Player 2
            else:
                stats["draws"] += 1

        # Record game completion for progress reporting
        game_duration = time.time() - game_start_time
        progress_reporter.record_game(
            moves=0,  # Move count not tracked at this level
            duration_sec=game_duration,
        )

    # Emit final progress summary
    progress_reporter.finish()

    print("\n" + "=" * 40)
    print("Final Results:")
    print(f"  {args.p1} (P1): {stats['p1_wins']} wins")
    print(f"  {args.p2} (P2): {stats['p2_wins']} wins")
    print(f"  Draws: {stats['draws']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
