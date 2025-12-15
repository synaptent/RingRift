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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Disk monitoring thresholds - 70% limit enforced as of 2025-12-15
DISK_WARNING_THRESHOLD = 65  # Pause selfplay
DISK_CRITICAL_THRESHOLD = 70  # Abort selfplay (consistent with orchestrator MAX_DISK_USAGE_PERCENT)

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
from app.db import (
    get_or_create_db,
    record_completed_game_with_parity_check,
    ParityValidationError,
)

# Import coordination for task limits and duration tracking
try:
    from app.coordination import (
        TaskCoordinator,
        TaskType,
        register_running_task,
        record_task_completion,
    )
    from app.coordination.helpers import can_spawn_safe as can_spawn
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    TaskCoordinator = None
    TaskType = None
    can_spawn = None

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
    record_db: Optional[str] = None,
    lean_db: bool = False,
    enforce_canonical_history: bool = True,
    parity_mode: Optional[str] = None,
    mcts_sims: int = 100,
    nnue_blend: float = 0.5,
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
        engine_mode: Engine mode (random-only, heuristic-only, mixed, nnue-guided, or mcts)
        weights: Heuristic weights dict (from CMA-ES profile or defaults)
        mix_ratio: For mixed mode: probability of heuristic (0.0-1.0). Default 0.8
        mcts_sims: Number of MCTS simulations per move (for mcts mode). Default 100
        nnue_blend: For nnue-guided mode: blend ratio of NNUE vs heuristic. Default 0.5

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
            "square8": 10000,   # Allow games to complete naturally
            "square19": 10000,  # Allow games to complete naturally
            "hex": 10000,       # Allow games to complete naturally
        }
        max_moves = max_moves_defaults.get(board_type.lower(), 10000)
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
    elif engine_mode == "mcts":
        logger.info(f"MCTS simulations: {mcts_sims}")
    elif engine_mode == "nnue-guided":
        logger.info(f"NNUE blend: {nnue_blend:.1%} NNUE / {1-nnue_blend:.1%} heuristic")
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

    # Initialize NNUE evaluator for nnue-guided mode
    nnue_evaluator = None
    if engine_mode == "nnue-guided":
        try:
            from app.ai.nnue import BatchNNUEEvaluator
            nnue_evaluator = BatchNNUEEvaluator(
                board_type=board_type_enum,
                num_players=num_players,
                device=device,
            )
            if nnue_evaluator.available:
                logger.info(f"NNUE evaluator loaded for {board_type_enum.value}")
            else:
                logger.warning("NNUE model not available, falling back to heuristic")
                nnue_evaluator = None
        except ImportError as e:
            logger.warning(f"NNUE not available: {e}. Falling back to heuristic.")
            nnue_evaluator = None

    # Initialize MCTS for mcts mode
    mcts_ai = None
    if engine_mode == "mcts":
        try:
            from app.mcts.improved_mcts import ImprovedMCTS, MCTSConfig
            mcts_config = MCTSConfig(
                num_simulations=mcts_sims,
                exploration_constant=1.414,
                use_dirichlet_noise=True,
            )
            mcts_ai = ImprovedMCTS(config=mcts_config)
            logger.info(f"MCTS initialized with {mcts_sims} simulations per move")
        except ImportError as e:
            logger.warning(f"MCTS not available: {e}. Falling back to heuristic.")
            mcts_ai = None

    # Optional recording to GameReplayDB for downstream training/parity tooling.
    replay_db = get_or_create_db(
        record_db,
        enforce_canonical_history=bool(enforce_canonical_history),
        respect_env_disable=True,
    ) if record_db else None
    store_history_entries = not bool(lean_db)

    games_recorded = 0
    record_failures = 0

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
            initial_state_for_db = game_state
            # Capture initial state for training data export (required for NPZ conversion)
            initial_state_snapshot = game_state.model_dump(mode="json")

            moves_played = []
            moves_for_db = []
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
                    elif engine_mode == "mcts" and mcts_ai is not None:
                        # MCTS mode: Use Monte Carlo Tree Search for move selection
                        try:
                            best_move = mcts_ai.search(game_state)
                            if best_move is None or best_move not in valid_moves:
                                # Fallback to heuristic if MCTS fails
                                move_scores = evaluator.evaluate_moves(
                                    game_state, valid_moves, current_player, GameEngine
                                )
                                if move_scores:
                                    best_score = max(s for _, s in move_scores)
                                    best_moves = [m for m, s in move_scores if s == best_score]
                                    best_move = np.random.choice(best_moves) if len(best_moves) > 1 else best_moves[0]
                                else:
                                    best_move = valid_moves[0]
                        except Exception as e:
                            logger.debug(f"MCTS error: {e}, falling back to heuristic")
                            best_move = valid_moves[np.random.randint(len(valid_moves))]
                    elif engine_mode == "nnue-guided" and nnue_evaluator is not None:
                        # NNUE-guided mode: Blend NNUE and heuristic scores
                        try:
                            # Get heuristic scores
                            heuristic_scores = evaluator.evaluate_moves(
                                game_state, valid_moves, current_player, GameEngine
                            )
                            heuristic_dict = {m: s for m, s in heuristic_scores} if heuristic_scores else {}

                            # Get NNUE scores for each move (evaluate resulting positions)
                            nnue_scores = {}
                            for move in valid_moves:
                                try:
                                    # Apply move to get resulting state
                                    next_state = GameEngine.apply_move(game_state.model_copy(deep=True), move)
                                    nnue_val = nnue_evaluator.evaluate(next_state)
                                    nnue_scores[move] = nnue_val if nnue_val is not None else 0.0
                                except Exception:
                                    nnue_scores[move] = 0.0

                            # Blend scores: nnue_blend * NNUE + (1 - nnue_blend) * heuristic
                            blended_scores = []
                            for move in valid_moves:
                                h_score = heuristic_dict.get(move, 0.0)
                                n_score = nnue_scores.get(move, 0.0)
                                blended = nnue_blend * n_score + (1 - nnue_blend) * h_score
                                blended_scores.append((move, blended))

                            if blended_scores:
                                best_score = max(s for _, s in blended_scores)
                                best_moves = [m for m, s in blended_scores if s == best_score]
                                best_move = np.random.choice(best_moves) if len(best_moves) > 1 else best_moves[0]
                            else:
                                best_move = valid_moves[0]
                        except Exception as e:
                            logger.debug(f"NNUE error: {e}, falling back to heuristic")
                            best_move = valid_moves[np.random.randint(len(valid_moves))]
                    else:
                        # heuristic-only (default): Evaluate moves (hybrid CPU/GPU)
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

                move_timestamp = datetime.now(timezone.utc)
                stamped_move = best_move.model_copy(
                    update={
                        "id": f"move-{move_count + 1}",
                        "timestamp": move_timestamp,
                        "think_time": 0,
                        "move_number": move_count + 1,
                    }
                )

                # Apply move (CPU - full rules)
                game_state = GameEngine.apply_move(game_state, stamped_move)
                moves_for_db.append(stamped_move)

                # Record full move data for training
                move_record = {
                    "type": stamped_move.type.value if hasattr(stamped_move.type, 'value') else str(stamped_move.type),
                    "player": stamped_move.player,
                }
                # Add position data if available
                if hasattr(stamped_move, 'to') and stamped_move.to is not None:
                    move_record["to"] = {"x": stamped_move.to.x, "y": stamped_move.to.y}
                if hasattr(stamped_move, 'from_pos') and stamped_move.from_pos is not None:
                    move_record["from"] = {"x": stamped_move.from_pos.x, "y": stamped_move.from_pos.y}
                if hasattr(stamped_move, 'capture_target') and stamped_move.capture_target is not None:
                    move_record["capture_target"] = {"x": stamped_move.capture_target.x, "y": stamped_move.capture_target.y}
                # Add capture chain for multi-captures
                if hasattr(stamped_move, 'capture_chain') and stamped_move.capture_chain:
                    move_record["capture_chain"] = [{"x": p.x, "y": p.y} for p in stamped_move.capture_chain]
                # Add line/territory data if present
                if hasattr(stamped_move, 'formed_lines') and stamped_move.formed_lines:
                    move_record["formed_lines"] = len(stamped_move.formed_lines)
                if hasattr(stamped_move, 'claimed_territory') and stamped_move.claimed_territory:
                    move_record["claimed_territory"] = len(stamped_move.claimed_territory)

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

            if replay_db is not None:
                try:
                    meta = {
                        "source": "run_hybrid_selfplay.py",
                        "engine_mode": engine_mode,
                        "mix_ratio": mix_ratio if engine_mode == "mixed" else None,
                        "device": str(device),
                    }
                    _ = record_completed_game_with_parity_check(
                        db=replay_db,
                        initial_state=initial_state_for_db,
                        final_state=game_state,
                        moves=moves_for_db,
                        metadata=meta,
                        game_id=str(record.get("game_id") or ""),
                        parity_mode=parity_mode,
                        store_history_entries=store_history_entries,
                    )
                    games_recorded += 1
                except ParityValidationError as exc:
                    record_failures += 1
                    logger.warning(f"[record-db] Parity divergence; skipping game {game_idx}: {exc}")
                except Exception as exc:
                    record_failures += 1
                    logger.warning(f"[record-db] Failed to record game {game_idx}: {type(exc).__name__}: {exc}")

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
        "record_db_path": record_db,
        "games_recorded": games_recorded,
        "record_failures": record_failures,
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
        default=None,  # Auto-generate unique dir to avoid conflicts
        help="Output directory (default: auto-generated unique path)",
    )
    parser.add_argument(
        "--record-db",
        type=str,
        default="",
        help="Optional path to a GameReplayDB SQLite file to record games for training/parity tooling.",
    )
    parser.add_argument(
        "--no-record-db",
        action="store_true",
        help="Disable DB recording even if --record-db is set.",
    )
    parser.add_argument(
        "--lean-db",
        action="store_true",
        help="Store a lean DB (no per-move history snapshots) to reduce disk usage.",
    )
    parser.add_argument(
        "--no-enforce-canonical-history",
        action="store_true",
        help="Allow recording non-canonical move types to DB (not recommended).",
    )
    parser.add_argument(
        "--parity-mode",
        type=str,
        default=None,
        choices=["off", "warn", "strict"],
        help="Override parity validation mode for recorded games (default: env-driven).",
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
        choices=["random-only", "heuristic-only", "mixed", "nnue-guided", "mcts"],
        help="Engine mode: random-only (uniform random), heuristic-only (GPU heuristic), mixed (probabilistic blend), nnue-guided (NNUE neural network), or mcts (Monte Carlo Tree Search)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=100,
        help="Number of MCTS simulations per move (for mcts mode). Default: 100",
    )
    parser.add_argument(
        "--nnue-blend",
        type=float,
        default=0.5,
        help="For nnue-guided mode: blend ratio of NNUE vs heuristic (0.0=heuristic only, 1.0=NNUE only). Default: 0.5",
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

    # Validate GPU/CUDA environment
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("=" * 60)
            logger.warning("WARNING: CUDA is not available!")
            logger.warning("Hybrid selfplay requires GPU acceleration for optimal performance.")
            logger.warning("The script will still run but may fall back to CPU evaluation.")
            logger.warning("")
            logger.warning("To use CPU-only selfplay, run scripts/run_self_play.py instead.")
            logger.warning("=" * 60)
        else:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            logger.info(f"CUDA available: {device_count} GPU(s) detected ({device_name})")
    except ImportError:
        logger.error("PyTorch not installed - hybrid selfplay requires torch with CUDA support")
        sys.exit(1)

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

        # Check coordination before spawning
        task_id = None
        start_time = time.time()
        if HAS_COORDINATION:
            import socket
            node_id = socket.gethostname()
            allowed, reason = can_spawn(TaskType.HYBRID_SELFPLAY, node_id)
            if not allowed:
                logger.warning(f"Coordination denied spawn: {reason}")
                logger.info("Proceeding anyway (coordination is advisory)")

            # Register task for tracking
            task_id = f"hybrid_selfplay_{args.board_type}_{args.num_players}p_{os.getpid()}"
            try:
                register_running_task(task_id, "hybrid_selfplay", node_id, os.getpid())
                logger.info(f"Registered task {task_id} with coordinator")
            except Exception as e:
                logger.warning(f"Failed to register task: {e}")

        # Auto-generate unique output directory if not specified
        output_dir = args.output_dir
        if output_dir is None:
            ts = int(time.time())
            pid = os.getpid()
            output_dir = f"data/selfplay/auto_{ts}/{pid}"
            logger.info(f"Auto-generated output directory: {output_dir}")

        try:
            run_hybrid_selfplay(
                board_type=args.board_type,
                num_players=args.num_players,
                num_games=args.num_games,
                output_dir=output_dir,
                max_moves=args.max_moves,
                seed=args.seed,
                use_numba=not args.no_numba,
                engine_mode=args.engine_mode,
                weights=weights,
                mix_ratio=args.mix_ratio,
                record_db=None if args.no_record_db else (args.record_db or None),
                lean_db=bool(args.lean_db),
                enforce_canonical_history=not bool(args.no_enforce_canonical_history),
                parity_mode=args.parity_mode,
                mcts_sims=args.mcts_sims,
                nnue_blend=args.nnue_blend,
            )
        finally:
            # Record task completion for duration learning
            if HAS_COORDINATION and task_id:
                try:
                    import socket
                    node_id = socket.gethostname()
                    config = f"{args.board_type}_{args.num_players}p"
                    record_task_completion("hybrid_selfplay", config, node_id, start_time, time.time())
                    logger.info(f"Recorded task completion for duration learning")
                except Exception as e:
                    logger.warning(f"Failed to record task completion: {e}")


if __name__ == "__main__":
    main()
