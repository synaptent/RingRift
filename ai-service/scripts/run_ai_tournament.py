import sys
import os
import time
import argparse
import fcntl
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

# Add the parent directory to sys.path to allow imports from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.progress_reporter import SoakProgressReporter
from app.utils.victory_type import derive_victory_type

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


def run_game(p1_ai, p2_ai, board_type: str, max_moves: int = 10000) -> Tuple[int, GameState, List[Dict[str, Any]], int, Dict[str, Any]]:
    """
    Run a single game between two AI instances.
    Returns tuple of (winner, final_game_state, move_history, move_count, initial_state_json).
    Winner is 1 or 2; timeouts are resolved via deterministic tie-break.
    """
    def _timeout_tiebreak_winner(final_state: GameState) -> int:
        """Deterministically select a winner for evaluation-only max-move cutoffs."""
        players = getattr(final_state, "players", None) or []
        if not players:
            return 1

        marker_counts: Dict[int, int] = {int(p.player_number): 0 for p in players}
        try:
            for marker in final_state.board.markers.values():
                owner = int(marker.player)
                marker_counts[owner] = marker_counts.get(owner, 0) + 1
        except Exception:
            pass

        last_actor = None
        try:
            last_actor = final_state.move_history[-1].player if final_state.move_history else None
        except Exception:
            last_actor = None

        best_player: Optional[int] = None
        best_key: Optional[tuple] = None
        for idx, player in enumerate(players):
            player_num = getattr(player, "player_number", None)
            if player_num is None:
                player_num = idx + 1
            pid = int(player_num)
            try:
                eliminated = int(getattr(player, "eliminated_rings", 0) or 0)
            except Exception:
                eliminated = 0
            try:
                territory = int(getattr(player, "territory_spaces", 0) or 0)
            except Exception:
                territory = 0
            markers = int(marker_counts.get(pid, 0))
            last = 1 if last_actor == pid else 0

            key = (territory, eliminated, markers, last, -pid)
            if best_key is None or key > best_key:
                best_key = key
                best_player = pid

        return int(best_player or 1)
    # Create fresh game state
    p1_config = {"type": p1_ai.__class__.__name__.replace("AI", ""), "difficulty": p1_ai.config.difficulty}
    p2_config = {"type": p2_ai.__class__.__name__.replace("AI", ""), "difficulty": p2_ai.config.difficulty}

    game_state = create_game_state(board_type, p1_config, p2_config)
    # Capture initial state for training data (required for NPZ export)
    initial_state_json = game_state.model_dump(mode="json")
    rules_engine = DefaultRulesEngine()

    move_count = 0
    moves_played: List[Dict[str, Any]] = []

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
            winner = 2 if current_player_num == 1 else 1
            return (winner, game_state, moves_played, move_count, initial_state_json)

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
            winner = 2 if current_player_num == 1 else 1
            return (winner, game_state, moves_played, move_count, initial_state_json)

        # Record move for training data
        move_record: Dict[str, Any] = {
            "type": move.type.value if hasattr(move.type, 'value') else str(move.type),
            "player": move.player,
        }
        if hasattr(move, 'to') and move.to is not None:
            move_record["to"] = {"x": move.to.x, "y": move.to.y}
        if hasattr(move, 'from_pos') and move.from_pos is not None:
            move_record["from"] = {"x": move.from_pos.x, "y": move.from_pos.y}
        moves_played.append(move_record)

        # Check victory
        if game_state.game_status == GameStatus.COMPLETED:
            break

        move_count += 1

    if game_state.game_status == GameStatus.ACTIVE:
        winner = _timeout_tiebreak_winner(game_state)
        print(f"Max moves reached (timeout). Tiebreak winner: Player {winner}")
        return (winner, game_state, moves_played, move_count, initial_state_json)

    winner = game_state.winner if game_state.winner is not None else _timeout_tiebreak_winner(game_state)
    print(f"Game Over. Winner: Player {winner}")
    return (winner, game_state, moves_played, move_count, initial_state_json)


def main():
    parser = argparse.ArgumentParser(description="Run AI Tournament")
    parser.add_argument("--p1", type=str, required=True, choices=AI_CLASSES.keys(), help="Player 1 AI Type")
    parser.add_argument("--p1-diff", type=int, default=5, help="Player 1 Difficulty (1-10)")
    parser.add_argument("--p2", type=str, required=True, choices=AI_CLASSES.keys(), help="Player 2 AI Type")
    parser.add_argument("--p2-diff", type=int, default=5, help="Player 2 Difficulty (1-10)")
    parser.add_argument("--board", type=str, default="Square8", choices=BOARD_TYPES.keys(), help="Board Type")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for games.jsonl (optional)")
    parser.add_argument("--max-moves", type=int, default=10000, help="Max moves per game before timeout tie-break")

    args = parser.parse_args()

    print(f"Tournament Configuration:")
    print(f"  Player 1: {args.p1} (Difficulty {args.p1_diff})")
    print(f"  Player 2: {args.p2} (Difficulty {args.p2_diff})")
    print(f"  Board: {args.board}")
    print(f"  Games: {args.games}")
    print(f"  Max moves: {args.max_moves}")
    if args.output_dir:
        print(f"  Output: {args.output_dir}")
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
    victory_type_counts: Dict[str, int] = {}
    stalemate_by_tiebreaker: Dict[str, int] = {}
    game_records: List[Dict[str, Any]] = []

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=args.games,
        report_interval_sec=10.0,
        context_label=f"{args.p1}_vs_{args.p2}_{args.board}",
    )

    # Set up output file if requested
    output_file = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        games_file = os.path.join(args.output_dir, "games.jsonl")
        output_file = open(games_file, "w")
        # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
        try:
            fcntl.flock(output_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ERROR: Cannot acquire lock on {games_file} - another process is writing to it.")
            print("Use a different output file or wait for the other process to finish.")
            output_file.close()
            sys.exit(1)

    for i in range(args.games):
        game_start_time = time.time()
        print(f"\nMatch {i+1}/{args.games}")

        # Swap sides every other game to ensure fairness
        if i % 2 == 0:
            # P1 is Player 1
            winner, final_state, moves_played, move_count, initial_state = run_game(ai1, ai2, args.board, args.max_moves)
            if winner == 1:
                stats["p1_wins"] += 1
            elif winner == 2:
                stats["p2_wins"] += 1
            else:
                stats["draws"] += 1
            p1_seat, p2_seat = 1, 2
        else:
            # P1 is Player 2 (swap)
            # run_game expects (p1_ai, p2_ai) where p1_ai plays as Player 1
            winner, final_state, moves_played, move_count, initial_state = run_game(ai2, ai1, args.board, args.max_moves)
            if winner == 1:
                stats["p2_wins"] += 1  # ai2 (P2 originally) won as Player 1
            elif winner == 2:
                stats["p1_wins"] += 1  # ai1 (P1 originally) won as Player 2
            else:
                stats["draws"] += 1
            p1_seat, p2_seat = 2, 1

        # Derive victory type using shared module
        victory_type, stalemate_tiebreaker = derive_victory_type(final_state, args.max_moves)
        victory_type_counts[victory_type] = victory_type_counts.get(victory_type, 0) + 1
        if stalemate_tiebreaker:
            stalemate_by_tiebreaker[stalemate_tiebreaker] = stalemate_by_tiebreaker.get(stalemate_tiebreaker, 0) + 1

        game_duration = time.time() - game_start_time

        # Build game record with standardized metadata
        game_status = final_state.game_status.value if hasattr(final_state.game_status, 'value') else str(final_state.game_status)
        record: Dict[str, Any] = {
            # === Core game identifiers ===
            "game_id": f"tournament_{args.board.lower()}_2p_{i}_{int(time.time())}",
            "board_type": args.board.lower(),  # square8, square19, hexagonal
            "num_players": 2,
            # === Game outcome ===
            "winner": winner,
            "move_count": move_count,
            "status": game_status,
            "game_status": game_status,  # Alias for compatibility
            "victory_type": victory_type,
            "stalemate_tiebreaker": stalemate_tiebreaker,
            "termination_reason": f"status:{game_status}:{victory_type}",
            # === Engine/opponent metadata ===
            "engine_mode": "ai_vs_ai",
            "opponent_type": "ai_vs_ai",
            "player_types": [args.p1, args.p2],
            "p1_ai": args.p1,
            "p1_difficulty": args.p1_diff,
            "p2_ai": args.p2,
            "p2_difficulty": args.p2_diff,
            "p1_seat": p1_seat,
            "p2_seat": p2_seat,
            # === Training data (required for NPZ export) ===
            "moves": moves_played,
            "initial_state": initial_state,
            # === Timing metadata ===
            "game_time_seconds": game_duration,
            "timestamp": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            # === Source tracking ===
            "source": "run_ai_tournament.py",
        }
        game_records.append(record)

        # Write to JSONL if output dir specified
        if output_file:
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()

        # Record game completion for progress reporting
        progress_reporter.record_game(
            moves=move_count,
            duration_sec=game_duration,
        )

    # Close output file
    if output_file:
        output_file.close()

    # Emit final progress summary
    progress_reporter.finish()

    print("\n" + "=" * 40)
    print("Final Results:")
    print(f"  {args.p1} (P1): {stats['p1_wins']} wins")
    print(f"  {args.p2} (P2): {stats['p2_wins']} wins")
    print(f"  Draws: {stats['draws']}")
    print("")
    print("Victory Types:")
    for vt, count in sorted(victory_type_counts.items()):
        print(f"  {vt}: {count}")
    if stalemate_by_tiebreaker:
        print("")
        print("Stalemate Tiebreakers:")
        for tb, count in sorted(stalemate_by_tiebreaker.items()):
            print(f"  {tb}: {count}")
    print("=" * 40)

    # Write stats.json if output dir specified
    if args.output_dir:
        stats_path = os.path.join(args.output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "total_games": args.games,
                "p1_ai": args.p1,
                "p1_difficulty": args.p1_diff,
                "p2_ai": args.p2,
                "p2_difficulty": args.p2_diff,
                "board": args.board,
                "max_moves": args.max_moves,
                "p1_wins": stats["p1_wins"],
                "p2_wins": stats["p2_wins"],
                "draws": stats["draws"],
                "victory_type_counts": victory_type_counts,
                "stalemate_by_tiebreaker": stalemate_by_tiebreaker,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"\nGame records saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
