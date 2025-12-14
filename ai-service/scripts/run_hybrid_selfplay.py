#!/usr/bin/env python
"""Hybrid GPU-accelerated self-play with full rule fidelity.

This script generates self-play games using the hybrid CPU/GPU approach:
- CPU: Full game rules (move generation, move application, victory checking)
- GPU: Position evaluation (heuristic scoring)

This provides 5-20x speedup while maintaining 100% rule correctness.

Usage:
    # Basic usage - 100 games on square8
    python scripts/run_hybrid_selfplay.py --num-games 100

    # With specific board and player count
    python scripts/run_hybrid_selfplay.py \
        --num-games 500 \
        --board-type square8 \
        --num-players 2 \
        --output-dir data/selfplay/hybrid_sq8_2p

    # Benchmark mode
    python scripts/run_hybrid_selfplay.py --benchmark

Output:
    - games.jsonl: Game records in JSONL format
    - stats.json: Performance statistics
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Disk monitoring thresholds
DISK_WARNING_THRESHOLD = 85  # Pause selfplay
DISK_CRITICAL_THRESHOLD = 95  # Abort selfplay

# =============================================================================
# Default Heuristic Weights (used when no --weights-file is specified)
# =============================================================================

DEFAULT_WEIGHTS = {
    "material_weight": 1.0,
    "ring_count_weight": 0.5,
    "stack_height_weight": 0.3,
    "center_control_weight": 0.4,
    "territory_weight": 0.8,
    "mobility_weight": 0.2,
    "line_potential_weight": 0.6,
    "defensive_weight": 0.3,
}


def load_weights_from_profile(
    weights_file: str,
    profile_name: str,
) -> Dict[str, float]:
    """Load heuristic weights from a CMA-ES profile file.

    Args:
        weights_file: Path to JSON file containing weight profiles
        profile_name: Name of the profile to load

    Returns:
        Dictionary of weight name -> value

    The profile file should have structure:
    {
        "profiles": {
            "profile_name": {
                "weights": { "material_weight": 1.0, ... }
            }
        }
    }
    """
    if not os.path.exists(weights_file):
        logging.getLogger(__name__).warning(
            f"Weights file not found: {weights_file}, using defaults"
        )
        return DEFAULT_WEIGHTS.copy()

    with open(weights_file, "r") as f:
        data = json.load(f)

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        logging.getLogger(__name__).warning(
            f"Profile '{profile_name}' not found in {weights_file}, using defaults"
        )
        return DEFAULT_WEIGHTS.copy()

    return profiles[profile_name].get("weights", DEFAULT_WEIGHTS.copy())


def get_disk_usage_percent(path: str = "/") -> int:
    """Get disk usage percentage for the filesystem containing path."""
    try:
        total, used, free = shutil.disk_usage(path)
        return int((used / total) * 100)
    except Exception:
        return 0


def run_disk_cleanup() -> bool:
    """Run disk cleanup script if available, return True if cleanup was run."""
    cleanup_script = Path(__file__).parent / "disk_monitor.sh"
    if cleanup_script.exists():
        try:
            subprocess.run(
                ["bash", str(cleanup_script)],
                capture_output=True,
                timeout=120,
            )
            return True
        except Exception:
            pass
    return False


def check_disk_space(logger, output_dir: str) -> str:
    """Check disk space and return status: 'ok', 'warning', or 'critical'."""
    usage = get_disk_usage_percent(output_dir)

    if usage >= DISK_CRITICAL_THRESHOLD:
        logger.error(f"CRITICAL: Disk usage at {usage}% - aborting selfplay")
        return "critical"
    elif usage >= DISK_WARNING_THRESHOLD:
        logger.warning(f"WARNING: Disk usage at {usage}% - running cleanup")
        if run_disk_cleanup():
            # Re-check after cleanup
            new_usage = get_disk_usage_percent(output_dir)
            logger.info(f"Disk usage after cleanup: {new_usage}%")
            if new_usage >= DISK_CRITICAL_THRESHOLD:
                return "critical"
            elif new_usage >= DISK_WARNING_THRESHOLD:
                return "warning"
        return "warning"
    return "ok"


# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared victory type module (must be after path setup)
from app.utils.victory_type import derive_victory_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_hybrid_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 100,
    output_dir: str = "data/selfplay/hybrid",
    max_moves: int | None = None,  # Auto-calculated based on board type
    seed: int = 42,
    use_numba: bool = True,
    engine_mode: str = "heuristic-only",
    weights: Optional[Dict[str, float]] = None,
    mix_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Run hybrid GPU-accelerated self-play.

    Args:
        board_type: Board type (square8, square19, hex)
        num_players: Number of players (2-4)
        num_games: Number of games to generate
        output_dir: Output directory
        max_moves: Maximum moves per game
        seed: Random seed
        use_numba: Use Numba JIT-compiled rules
        engine_mode: Engine mode (random-only, heuristic-only, or mixed)
        weights: Heuristic weights dict (from CMA-ES profile or defaults)
        mix_ratio: For mixed mode: probability of heuristic (0.0-1.0). Default 0.8

    Returns:
        Statistics dictionary
    """
    import torch
    from app.ai.hybrid_gpu import (
        HybridGPUEvaluator,
        HybridSelfPlayRunner,
        create_hybrid_evaluator,
    )
    from app.ai.gpu_batch import get_device
    from app.game_engine import GameEngine
    from app.training.generate_data import create_initial_state
    from app.models import BoardType

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    board_type_key = board_type.lower()
    board_size = {"square8": 8, "square19": 19, "hex": 25, "hexagonal": 25}.get(board_type_key, 8)

    # Auto-calculate max_moves based on board type if not specified
    # Larger boards need more moves (multiple actions per turn are counted)
    if max_moves is None:
        max_moves_defaults = {
            "square8": 500,    # 8x8 games typically complete in ~100-300 moves
            "square19": 2500,  # 19x19 needs ~5x more headroom
            "hex": 2500,       # Hex boards also need higher limits
        }
        max_moves = max_moves_defaults.get(board_type.lower(), 2500)
    board_type_enum_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type_enum = board_type_enum_map.get(board_type_key, BoardType.SQUARE8)
    device = get_device()

    logger.info("=" * 60)
    logger.info("HYBRID GPU-ACCELERATED SELF-PLAY")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Max moves: {max_moves}")
    logger.info(f"Engine mode: {engine_mode}")
    if engine_mode == "mixed":
        logger.info(f"Mix ratio: {mix_ratio:.1%} heuristic / {1-mix_ratio:.1%} random")
    logger.info(f"Device: {device}")
    logger.info(f"Numba: {use_numba}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create hybrid evaluator
    evaluator = create_hybrid_evaluator(
        board_type=board_type,
        num_players=num_players,
        prefer_gpu=True,
    )

    # Statistics
    total_games = 0
    total_moves = 0
    total_time = 0.0
    wins_by_player = {i: 0 for i in range(1, num_players + 1)}
    draws = 0
    victory_type_counts: Dict[str, int] = {}  # Track victory type distribution
    stalemate_by_tiebreaker: Dict[str, int] = {}  # Track which tiebreaker resolved stalemates
    game_lengths: List[int] = []  # Track individual game lengths for detailed stats
    game_records = []

    games_file = os.path.join(output_dir, "games.jsonl")

    logger.info(f"Starting {num_games} games...")
    start_time = time.time()

    with open(games_file, "w") as f:
        # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error(f"Cannot acquire lock on {games_file} - another process is writing")
            sys.exit(1)
        for game_idx in range(num_games):
            # Check disk space every 10 games
            if game_idx % 10 == 0:
                disk_status = check_disk_space(logger, output_dir)
                if disk_status == "critical":
                    logger.error(f"Aborting selfplay at game {game_idx} due to critical disk usage")
                    break
                elif disk_status == "warning":
                    logger.warning(f"Disk space low, continuing cautiously at game {game_idx}")

            game_start = time.time()

            # Create initial state
            game_state = create_initial_state(
                board_type=board_type_enum,
                num_players=num_players,
            )
            # Capture initial state for training data export (required for NPZ conversion)
            initial_state_snapshot = game_state.model_dump(mode="json")

            moves_played = []
            move_count = 0

            while game_state.game_status == "active" and move_count < max_moves:
                current_player = game_state.current_player

                # Get valid moves (CPU - full rules)
                valid_moves = GameEngine.get_valid_moves(
                    game_state, current_player
                )

                if not valid_moves:
                    # Check for phase requirements (bookkeeping moves)
                    requirement = GameEngine.get_phase_requirement(
                        game_state, current_player
                    )
                    if requirement is not None:
                        # Use GameEngine to synthesize the appropriate bookkeeping move
                        best_move = GameEngine.synthesize_bookkeeping_move(
                            requirement, game_state
                        )
                        if best_move is None:
                            # Failed to synthesize - check for endgame
                            GameEngine._check_victory(game_state)
                            break
                    else:
                        # No valid moves and no phase requirement - trigger victory check
                        GameEngine._check_victory(game_state)
                        break
                else:
                    # Select move based on engine mode
                    if engine_mode == "random-only":
                        # Uniform random move selection (no evaluation)
                        best_move = valid_moves[np.random.randint(len(valid_moves))]
                    elif engine_mode == "mixed":
                        # Mixed mode: probabilistically choose random vs heuristic
                        if np.random.random() < mix_ratio:
                            # Use heuristic evaluation
                            move_scores = evaluator.evaluate_moves(
                                game_state,
                                valid_moves,
                                current_player,
                                GameEngine,
                            )
                            if move_scores:
                                best_score = max(s for _, s in move_scores)
                                best_moves = [m for m, s in move_scores if s == best_score]
                                best_move = np.random.choice(best_moves) if len(best_moves) > 1 else best_moves[0]
                            else:
                                best_move = valid_moves[0]
                        else:
                            # Use random selection
                            best_move = valid_moves[np.random.randint(len(valid_moves))]
                    else:
                        # heuristic-only: Evaluate moves (hybrid CPU/GPU)
                        move_scores = evaluator.evaluate_moves(
                            game_state,
                            valid_moves,
                            current_player,
                            GameEngine,
                        )

                        # Select best move (with random tie-breaking)
                        if move_scores:
                            best_score = max(s for _, s in move_scores)
                            best_moves = [m for m, s in move_scores if s == best_score]
                            best_move = np.random.choice(best_moves) if len(best_moves) > 1 else best_moves[0]
                        else:
                            best_move = valid_moves[0]

                # Apply move (CPU - full rules)
                game_state = GameEngine.apply_move(game_state, best_move)

                # Record full move data for training
                move_record = {
                    "type": best_move.type.value if hasattr(best_move.type, 'value') else str(best_move.type),
                    "player": best_move.player,
                }
                # Add position data if available
                if hasattr(best_move, 'to') and best_move.to is not None:
                    move_record["to"] = {"x": best_move.to.x, "y": best_move.to.y}
                if hasattr(best_move, 'from_pos') and best_move.from_pos is not None:
                    move_record["from"] = {"x": best_move.from_pos.x, "y": best_move.from_pos.y}
                if hasattr(best_move, 'capture_target') and best_move.capture_target is not None:
                    move_record["capture_target"] = {"x": best_move.capture_target.x, "y": best_move.capture_target.y}
                # Add capture chain for multi-captures
                if hasattr(best_move, 'capture_chain') and best_move.capture_chain:
                    move_record["capture_chain"] = [{"x": p.x, "y": p.y} for p in best_move.capture_chain]
                # Add line/territory data if present
                if hasattr(best_move, 'formed_lines') and best_move.formed_lines:
                    move_record["formed_lines"] = len(best_move.formed_lines)
                if hasattr(best_move, 'claimed_territory') and best_move.claimed_territory:
                    move_record["claimed_territory"] = len(best_move.claimed_territory)

                moves_played.append(move_record)
                move_count += 1

            game_time = time.time() - game_start
            total_time += game_time
            total_moves += move_count
            total_games += 1
            game_lengths.append(move_count)  # Track individual game length

            # Record result
            winner = game_state.winner or 0
            if winner == 0:
                draws += 1
            else:
                wins_by_player[winner] = wins_by_player.get(winner, 0) + 1

            # Derive victory type per GAME_RECORD_SPEC.md
            victory_type, stalemate_tiebreaker = derive_victory_type(game_state, max_moves)
            victory_type_counts[victory_type] = victory_type_counts.get(victory_type, 0) + 1

            # Track stalemate tiebreaker breakdown
            if stalemate_tiebreaker:
                stalemate_by_tiebreaker[stalemate_tiebreaker] = stalemate_by_tiebreaker.get(stalemate_tiebreaker, 0) + 1

            record = {
                # === Core game identifiers ===
                "game_id": f"hybrid_{board_type}_{num_players}p_{game_idx}_{int(datetime.now().timestamp())}",
                "board_type": board_type,  # square8, square19, hexagonal
                "num_players": num_players,
                # === Game outcome ===
                "winner": winner,
                "move_count": move_count,
                "status": game_state.game_status,  # completed, abandoned, etc.
                "game_status": game_state.game_status,  # Alias for compatibility
                "victory_type": victory_type,  # territory, elimination, lps, stalemate, timeout
                "stalemate_tiebreaker": stalemate_tiebreaker,  # territory, ring_elim, or None
                "termination_reason": f"status:{game_state.game_status}:{victory_type}",
                # === Engine/opponent metadata ===
                "engine_mode": engine_mode,  # heuristic-only, random-only, mixed, mcts-only
                "opponent_type": "selfplay",  # selfplay, human, ai_vs_ai
                "player_types": ["hybrid_gpu"] * num_players,  # Type of each player
                "mix_ratio": mix_ratio if engine_mode == "mixed" else None,
                # === Training data (required for NPZ export) ===
                "moves": moves_played,  # Full move history
                "initial_state": initial_state_snapshot,  # For replay/reconstruction
                # === Timing metadata ===
                "game_time_seconds": game_time,
                "timestamp": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                # === Source tracking ===
                "source": "run_hybrid_selfplay.py",
                "device": str(device),
            }
            game_records.append(record)
            f.write(json.dumps(record) + "\n")
            # Flush immediately to minimize data loss on abnormal termination
            f.flush()

            # Progress logging
            if (game_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_idx + 1) / elapsed
                eta = (num_games - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0

                logger.info(
                    f"  Game {game_idx + 1}/{num_games}: "
                    f"{games_per_sec:.2f} g/s, ETA: {eta:.0f}s"
                )

    total_elapsed = time.time() - start_time

    # Get evaluator stats
    eval_stats = evaluator.get_performance_stats()

    # Build statistics
    stats = {
        "total_games": total_games,
        "total_moves": total_moves,
        "total_time_seconds": total_elapsed,
        "games_per_second": total_games / total_elapsed if total_elapsed > 0 else 0,
        "moves_per_game": total_moves / total_games if total_games > 0 else 0,
        "wins_by_player": wins_by_player,
        "draws": draws,
        "draw_rate": draws / total_games if total_games > 0 else 0,
        "victory_type_counts": victory_type_counts,
        "stalemate_by_tiebreaker": stalemate_by_tiebreaker,  # Breakdown of which tiebreaker resolved stalemates
        "board_type": board_type,
        "num_players": num_players,
        "max_moves": max_moves,
        "device": str(device),
        "evaluator_stats": eval_stats,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "game_lengths": game_lengths,  # Individual game lengths for detailed analysis
    }

    # Add win rates
    total_decided = sum(wins_by_player.values())
    for p in range(1, num_players + 1):
        stats[f"p{p}_win_rate"] = wins_by_player.get(p, 0) / total_decided if total_decided > 0 else 0

    # Save statistics
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total games: {stats['total_games']}")
    logger.info(f"Total moves: {stats['total_moves']}")
    logger.info(f"Avg moves/game: {stats['moves_per_game']:.1f}")
    logger.info(f"Total time: {stats['total_time_seconds']:.1f}s")
    logger.info(f"Throughput: {stats['games_per_second']:.2f} games/sec")
    logger.info(f"Draw rate: {stats['draw_rate']:.1%}")
    logger.info("")
    logger.info("Win rates by player:")
    for p in range(1, num_players + 1):
        logger.info(f"  Player {p}: {stats[f'p{p}_win_rate']:.1%}")
    logger.info("")
    logger.info("Evaluator stats:")
    logger.info(f"  Evals: {eval_stats['eval_count']}")
    logger.info(f"  Evals/sec: {eval_stats['evals_per_second']:.0f}")
    logger.info(f"  GPU fraction: {eval_stats['gpu_fraction']:.1%}")
    logger.info("")
    logger.info(f"Games saved to: {games_file}")
    logger.info(f"Stats saved to: {stats_file}")

    return stats


def run_benchmark(board_type: str = "square8", num_players: int = 2):
    """Run benchmark comparing pure CPU vs hybrid evaluation."""
    import torch
    from app.ai.hybrid_gpu import (
        HybridGPUEvaluator,
        create_hybrid_evaluator,
        benchmark_hybrid_evaluation,
    )
    from app.ai.numba_rules import (
        NUMBA_AVAILABLE,
        benchmark_numba_functions,
        BoardArrays,
    )
    from app.ai.gpu_batch import get_device
    from app.game_engine import GameEngine
    from app.training.generate_data import create_initial_state
    from app.models import BoardType

    logger.info("=" * 60)
    logger.info("BENCHMARK: CPU vs Hybrid GPU Evaluation")
    logger.info("=" * 60)

    board_size = {"square8": 8, "square19": 19, "hex": 25}.get(board_type.lower(), 8)
    board_type_enum = getattr(BoardType, board_type.upper(), BoardType.SQUARE8)
    device = get_device()

    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Device: {device}")
    logger.info(f"Numba available: {NUMBA_AVAILABLE}")
    logger.info("")

    # Create test game state
    game_state = create_initial_state(
        board_type=board_type_enum,
        num_players=num_players,
    )

    # Play a few moves to get an interesting state
    for _ in range(10):
        moves = GameEngine.get_valid_moves(game_state, game_state.current_player)
        if moves:
            game_state = GameEngine.apply_move(game_state, moves[0])

    # Benchmark Numba functions
    logger.info("Numba JIT benchmark:")
    numba_results = benchmark_numba_functions(game_state, num_iterations=10000, board_size=board_size)
    for key, value in numba_results.items():
        if key.endswith("_us"):
            logger.info(f"  {key}: {value:.2f} µs")

    # Benchmark hybrid evaluation
    logger.info("")
    logger.info("Hybrid GPU benchmark:")
    evaluator = create_hybrid_evaluator(
        board_type=board_type,
        num_players=num_players,
        prefer_gpu=True,
    )

    hybrid_results = benchmark_hybrid_evaluation(
        evaluator,
        GameEngine,
        num_positions=1000,
    )
    logger.info(f"  Positions evaluated: {hybrid_results['benchmark_positions']}")
    logger.info(f"  Total time: {hybrid_results['benchmark_time']:.2f}s")
    logger.info(f"  Positions/sec: {hybrid_results['positions_per_second']:.0f}")
    logger.info(f"  GPU fraction: {hybrid_results['gpu_fraction']:.1%}")

    # Compare pure CPU vs hybrid for move evaluation
    logger.info("")
    logger.info("Move evaluation comparison:")

    # Get valid moves
    moves = GameEngine.get_valid_moves(game_state, game_state.current_player)
    num_moves = len(moves)
    logger.info(f"  Moves to evaluate: {num_moves}")

    # Pure CPU
    import time
    start = time.perf_counter()
    for _ in range(10):
        for move in moves:
            next_state = GameEngine.apply_move(game_state, move)
            # Simple heuristic eval placeholder
    cpu_time = (time.perf_counter() - start) / 10
    logger.info(f"  Pure CPU: {cpu_time*1000:.1f} ms ({num_moves/cpu_time:.0f} moves/sec)")

    # Hybrid
    start = time.perf_counter()
    for _ in range(10):
        _ = evaluator.evaluate_moves(game_state, moves, game_state.current_player, GameEngine)
    hybrid_time = (time.perf_counter() - start) / 10
    logger.info(f"  Hybrid GPU: {hybrid_time*1000:.1f} ms ({num_moves/hybrid_time:.0f} moves/sec)")

    speedup = cpu_time / hybrid_time if hybrid_time > 0 else 0
    logger.info(f"  Speedup: {speedup:.1f}x")

    return {
        "numba": numba_results,
        "hybrid": hybrid_results,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid GPU-accelerated self-play with full rule fidelity"
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex", "hexagonal"],
        help="Board type (hexagonal is an alias for hex)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum moves per game (default: 500 for square8, 2500 for square19/hex)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay/hybrid",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark only",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT compilation",
    )
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="heuristic-only",
        choices=["random-only", "heuristic-only", "mixed"],
        help="Engine mode: random-only (uniform random), heuristic-only (GPU heuristic), or mixed (probabilistic blend)",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.8,
        help="For mixed mode: probability of using heuristic (0.0=all random, 1.0=all heuristic). Default: 0.8",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        help="Path to CMA-ES heuristic weights JSON file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Profile name in weights file (requires --weights-file)",
    )

    args = parser.parse_args()

    # Normalize board type aliases
    if args.board_type == "hexagonal":
        args.board_type = "hex"

    if args.benchmark:
        run_benchmark(args.board_type, args.num_players)
    else:
        # Load weights from profile if specified
        weights = None
        if args.weights_file and args.profile:
            weights = load_weights_from_profile(args.weights_file, args.profile)
            logger.info(f"Loaded weights from {args.weights_file}:{args.profile}")
        elif args.weights_file or args.profile:
            # Only one was specified - warn user
            logger.warning("Both --weights-file and --profile are required to load custom weights")

        run_hybrid_selfplay(
            board_type=args.board_type,
            num_players=args.num_players,
            num_games=args.num_games,
            output_dir=args.output_dir,
            max_moves=args.max_moves,
            seed=args.seed,
            use_numba=not args.no_numba,
            engine_mode=args.engine_mode,
            weights=weights,
            mix_ratio=args.mix_ratio,
        )


if __name__ == "__main__":
    main()
