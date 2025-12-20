#!/usr/bin/env python3
"""GMO Integration Script - Register GMO in Elo system and run self-play training.

This script integrates GMO into the training infrastructure:
1. Register GMO as a participant in the Elo system
2. Run self-play games to generate training data
3. Train GMO on self-play data
4. Evaluate GMO against baselines

Usage:
    # Register GMO in Elo system
    python scripts/gmo_integration.py register

    # Run self-play games
    python scripts/gmo_integration.py selfplay --num-games 100

    # Train on self-play data
    python scripts/gmo_integration.py train --epochs 10

    # Full pipeline: selfplay + train
    python scripts/gmo_integration.py pipeline --num-games 500 --epochs 20

    # Evaluate against baselines
    python scripts/gmo_integration.py evaluate --num-games 50
"""

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from app.ai.gmo_ai import GMOAI, GMOConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameState, GameStatus
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
GMO_MODEL_DIR = PROJECT_ROOT / "models" / "gmo"
GMO_CHECKPOINT = GMO_MODEL_DIR / "gmo_best.pt"
GMO_SELFPLAY_DATA = PROJECT_ROOT / "data" / "gmo_selfplay"
ELO_DB_PATH = PROJECT_ROOT / "data" / "unified_elo.db"


def register_gmo_in_elo(
    db_path: Path = ELO_DB_PATH,
    board_type: str = "square8",
    num_players: int = 2,
) -> None:
    """Register GMO as a participant in the Elo system."""
    try:
        from app.training.elo_service import EloService
    except ImportError:
        logger.warning("EloService not available, trying unified_elo_db")
        from app.tournament.unified_elo_db import UnifiedEloDatabase

        db = UnifiedEloDatabase(str(db_path))
        db.register_participant(
            participant_id="gmo_v1",
            display_name="GMO v1 (Gradient Move Optimization)",
            model_type="gmo",
            metadata={
                "algorithm": "Gradient Move Optimization",
                "version": "1.0",
                "components": [
                    "SPENs (gradient inference)",
                    "Action embeddings",
                    "MC Dropout uncertainty",
                    "UCB exploration",
                    "Novelty search",
                ],
            },
        )
        logger.info(f"Registered GMO in unified Elo database: {db_path}")
        return

    # Use EloService
    elo = EloService(str(db_path))
    elo.register_participant(
        participant_id="gmo_v1",
        name="GMO v1 (Gradient Move Optimization)",
        ai_type="gmo",
        use_neural_net=True,
        model_path=str(GMO_CHECKPOINT) if GMO_CHECKPOINT.exists() else None,
        metadata={
            "algorithm": "Gradient Move Optimization",
            "version": "1.0",
            "checkpoint": str(GMO_CHECKPOINT),
        },
    )

    # Initialize rating for the config
    elo.get_rating("gmo_v1", board_type, num_players)
    logger.info(f"Registered GMO in Elo system: {db_path}")
    logger.info(f"  Board type: {board_type}, Players: {num_players}")


def create_gmo_ai(
    player_number: int,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> GMOAI:
    """Create a GMO AI instance."""
    # AIConfig for BaseAI (required)
    ai_config = AIConfig(difficulty=5)

    # GMOConfig for GMO-specific parameters
    gmo_config = GMOConfig(
        device=device,
        top_k=5,
        optim_steps=10,
        lr=0.1,
        beta=0.3,
        gamma=0.1,
    )
    ai = GMOAI(player_number=player_number, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path and checkpoint_path.exists():
        ai.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded GMO checkpoint: {checkpoint_path}")

    return ai


def play_game(
    ai1,
    ai2,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 500,
    initial_state: GameState | None = None,
) -> tuple[int | None, list[dict], int]:
    """Play a single game between two AIs.

    Returns:
        Tuple of (winner, game_history, num_moves)
        winner is None for draw, 1 for ai1, 2 for ai2
    """
    state = initial_state or create_initial_state(
        board_type=board_type,
        num_players=num_players,
    )

    history = []
    ais = {1: ai1, 2: ai2}

    for move_num in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break

        current_player = state.current_player
        current_ai = ais[current_player]

        # Select move
        move = current_ai.select_move(state)
        if move is None:
            phase_req = GameEngine.get_phase_requirement(state, current_player)
            if phase_req is None:
                break
            move = GameEngine.synthesize_bookkeeping_move(phase_req, state)

        if hasattr(move, "model_dump"):
            move_payload = move.model_dump(mode="json", by_alias=True)
        else:
            move_payload = str(move)

        # Record history
        history.append({
            "move_num": move_num,
            "player": current_player,
            "move": move_payload,
            "state_hash": state.zobrist_hash if hasattr(state, "zobrist_hash") else None,
        })

        # Apply move using static method
        state = GameEngine.apply_move(state, move)

    # Determine winner
    winner = None
    if state.game_status == GameStatus.COMPLETED and state.winner:
        winner = state.winner

    return winner, history, len(history)


def run_selfplay(
    num_games: int = 100,
    output_dir: Path = GMO_SELFPLAY_DATA,
    checkpoint_path: Path | None = GMO_CHECKPOINT,
    device: str = "cpu",
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> dict:
    """Run GMO self-play games to generate training data.

    Emits JSONL records compatible with app.training.train_gmo
    (initial_state + moves per game).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"gmo_selfplay_{timestamp}.jsonl"

    logger.info(f"Running {num_games} GMO self-play games")
    logger.info(f"Output: {output_file}")

    results = {
        "total_games": num_games,
        "p1_wins": 0,
        "p2_wins": 0,
        "draws": 0,
        "total_moves": 0,
        "samples": 0,
    }

    with open(output_file, "w") as f:
        for game_idx in range(num_games):
            # Create fresh AI instances for each game
            ai1 = create_gmo_ai(1, checkpoint_path, device)
            ai2 = create_gmo_ai(2, checkpoint_path, device)

            initial_state = create_initial_state(
                board_type=board_type,
                num_players=num_players,
            )
            winner, history, num_moves = play_game(
                ai1,
                ai2,
                board_type,
                num_players,
                initial_state=initial_state,
            )

            moves_for_training = [
                entry["move"]
                for entry in history
                if isinstance(entry.get("move"), dict)
            ]

            # Update stats
            results["total_moves"] += num_moves
            if winner == 1:
                results["p1_wins"] += 1
                outcome = 1.0
            elif winner == 2:
                results["p2_wins"] += 1
                outcome = -1.0
            else:
                results["draws"] += 1
                outcome = 0.0

            # Write game record
            game_record = {
                "game_id": f"gmo_selfplay_{timestamp}_{game_idx}",
                "engine": "gmo",
                "board_type": board_type.value,
                "num_players": num_players,
                "num_moves": num_moves,
                "winner": winner,
                "outcome": outcome,
                "initial_state": initial_state.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                "moves": moves_for_training,
                "history": history,
            }
            f.write(json.dumps(game_record) + "\n")
            results["samples"] += num_moves

            if (game_idx + 1) % 10 == 0:
                p1_rate = results["p1_wins"] / (game_idx + 1) * 100
                logger.info(
                    f"Game {game_idx + 1}/{num_games}: "
                    f"P1 wins={results['p1_wins']} ({p1_rate:.1f}%), "
                    f"P2 wins={results['p2_wins']}, "
                    f"Draws={results['draws']}"
                )

    results["output_file"] = str(output_file)
    results["avg_game_length"] = results["total_moves"] / num_games

    logger.info(f"Self-play complete: {results}")
    return results


def train_on_selfplay(
    data_dir: Path = GMO_SELFPLAY_DATA,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    checkpoint_path: Path = GMO_CHECKPOINT,
    device: str = "cpu",
) -> dict:
    """Train GMO on self-play data.

    Filters to records that include initial_state + moves for train_gmo.
    """
    from app.training.train_gmo import train_gmo

    logger.info(f"Training GMO on self-play data from {data_dir}")

    data_files = sorted(data_dir.glob("gmo_selfplay_*.jsonl"))
    if not data_files:
        logger.warning(f"No self-play data found in {data_dir}")
        return {"error": "No training data"}

    logger.info(f"Found {len(data_files)} self-play data files")

    records = []
    for data_file in data_files:
        try:
            with open(data_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    initial_state = record.get("initial_state")
                    moves = record.get("moves")
                    if initial_state and moves:
                        records.append(record)
                        continue
                    if initial_state and record.get("history"):
                        moves = [
                            entry.get("move")
                            for entry in record["history"]
                            if isinstance(entry.get("move"), dict)
                        ]
                        if moves:
                            record["moves"] = moves
                            records.append(record)
        except Exception as exc:
            logger.warning(f"Failed to load {data_file}: {exc}")

    if not records:
        logger.warning("No valid training records found")
        return {"error": "No valid records"}

    output_dir = checkpoint_path.parent
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            dir=data_dir,
        ) as tmp:
            tmp_path = Path(tmp.name)
            for record in records:
                tmp.write(json.dumps(record) + "\n")

        train_gmo(
            data_path=tmp_path,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device_str=device,
        )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    results = {
        "trained": True,
        "records": len(records),
        "source_files": len(data_files),
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "gmo_best.pt"),
    }
    logger.info(f"Training complete: {results}")
    return results


def evaluate_against_baselines(
    num_games: int = 50,
    checkpoint_path: Path | None = GMO_CHECKPOINT,
    device: str = "cpu",
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> dict:
    """Evaluate GMO against baseline AIs."""
    baselines = {
        "random": lambda p: RandomAI(player_number=p, config=AIConfig(difficulty=1)),
        "heuristic": lambda p: HeuristicAI(player_number=p, config=AIConfig()),
    }

    results = {}

    for baseline_name, baseline_factory in baselines.items():
        logger.info(f"Evaluating GMO vs {baseline_name}...")

        wins = 0
        losses = 0
        draws = 0

        games_per_side = num_games // 2

        # GMO as player 1
        for _ in range(games_per_side):
            gmo = create_gmo_ai(1, checkpoint_path, device)
            baseline = baseline_factory(2)

            winner, _, _ = play_game(gmo, baseline, board_type, num_players)

            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        # GMO as player 2
        for _ in range(games_per_side):
            baseline = baseline_factory(1)
            gmo = create_gmo_ai(2, checkpoint_path, device)

            winner, _, _ = play_game(baseline, gmo, board_type, num_players)

            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        total = wins + losses + draws
        win_rate = wins / total * 100 if total > 0 else 0

        results[baseline_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "total_games": total,
        }

        logger.info(f"  vs {baseline_name}: {wins}W/{losses}L/{draws}D ({win_rate:.1f}% win rate)")

    return results


def run_pipeline(
    num_games: int = 500,
    epochs: int = 20,
    iterations: int = 3,
    device: str = "cpu",
) -> dict:
    """Run full GMO self-play training pipeline.

    Iteratively:
    1. Generate self-play games
    2. Train on new data
    3. Evaluate against baselines
    """
    logger.info(f"Starting GMO training pipeline")
    logger.info(f"  Games per iteration: {num_games}")
    logger.info(f"  Epochs per iteration: {epochs}")
    logger.info(f"  Total iterations: {iterations}")

    pipeline_results = []

    for iteration in range(iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{iterations}")
        logger.info(f"{'='*60}")

        # Self-play
        selfplay_results = run_selfplay(
            num_games=num_games,
            device=device,
        )

        # Train
        train_results = train_on_selfplay(
            epochs=epochs,
            device=device,
        )

        # Evaluate
        eval_results = evaluate_against_baselines(
            num_games=50,
            device=device,
        )

        iteration_results = {
            "iteration": iteration + 1,
            "selfplay": selfplay_results,
            "training": train_results,
            "evaluation": eval_results,
        }
        pipeline_results.append(iteration_results)

        # Log progress
        vs_random = eval_results.get("random", {}).get("win_rate", 0)
        vs_heuristic = eval_results.get("heuristic", {}).get("win_rate", 0)
        logger.info(f"Iteration {iteration + 1} results:")
        logger.info(f"  vs Random: {vs_random:.1f}%")
        logger.info(f"  vs Heuristic: {vs_heuristic:.1f}%")

    return {"iterations": pipeline_results}


def main():
    parser = argparse.ArgumentParser(
        description="GMO Integration - Register, train, and evaluate GMO"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser(
        "register", help="Register GMO in Elo system"
    )
    register_parser.add_argument(
        "--db-path", type=Path, default=ELO_DB_PATH,
        help="Path to Elo database"
    )
    register_parser.add_argument(
        "--board-type", type=str, default="square8",
        help="Board type for initial rating"
    )
    register_parser.add_argument(
        "--num-players", type=int, default=2,
        help="Number of players"
    )

    # Self-play command
    selfplay_parser = subparsers.add_parser(
        "selfplay", help="Run GMO self-play games"
    )
    selfplay_parser.add_argument(
        "--num-games", type=int, default=100,
        help="Number of games to play"
    )
    selfplay_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu/cuda)"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train GMO on self-play data"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full self-play training pipeline"
    )
    pipeline_parser.add_argument(
        "--num-games", type=int, default=500,
        help="Games per iteration"
    )
    pipeline_parser.add_argument(
        "--epochs", type=int, default=20,
        help="Epochs per iteration"
    )
    pipeline_parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of pipeline iterations"
    )
    pipeline_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate GMO against baselines"
    )
    eval_parser.add_argument(
        "--num-games", type=int, default=50,
        help="Games per baseline"
    )
    eval_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Auto-detect device
    device = getattr(args, "device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        device = "cpu"

    if args.command == "register":
        register_gmo_in_elo(args.db_path, args.board_type, args.num_players)

    elif args.command == "selfplay":
        run_selfplay(num_games=args.num_games, device=device)

    elif args.command == "train":
        train_on_selfplay(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    elif args.command == "pipeline":
        run_pipeline(
            num_games=args.num_games,
            epochs=args.epochs,
            iterations=args.iterations,
            device=device,
        )

    elif args.command == "evaluate":
        results = evaluate_against_baselines(
            num_games=args.num_games,
            device=device,
        )
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
