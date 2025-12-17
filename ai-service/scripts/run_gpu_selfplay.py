#!/usr/bin/env python
"""GPU-accelerated self-play data generation.

This script generates training data using GPU parallel game simulation,
achieving 10-100x speedup compared to CPU-based self-play.

The generated data can be used for:
1. Neural network training (policy/value targets)
2. CMA-ES fitness evaluation baselines
3. Game analysis and statistics

Usage:
    # Generate 1000 games on GPU
    python scripts/run_gpu_selfplay.py \\
        --num-games 1000 \\
        --board square8 \\
        --num-players 2 \\
        --output-dir data/selfplay/gpu_square8_2p

    # With specific heuristic weights
    python scripts/run_gpu_selfplay.py \\
        --num-games 500 \\
        --board square8 \\
        --weights-file config/trained_heuristic_profiles.json \\
        --profile heuristic_v1_2p \\
        --output-dir data/selfplay/trained_2p

    # Benchmark mode
    python scripts/run_gpu_selfplay.py --benchmark-only

Output:
    - games.jsonl: Game records in JSONL format
    - stats.json: Aggregated statistics
    - training_data.npz: NumPy arrays ready for NN training (optional)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ramdrive utilities for high-speed I/O
from app.utils.ramdrive import add_ramdrive_args, get_config_from_args, get_games_directory, RamdriveSyncer

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_batch import get_device, clear_gpu_memory
from app.ai.nnue import BatchNNUEEvaluator
from app.ai.gpu_parallel_games import (
    ParallelGameRunner,
    BatchGameState,
    benchmark_parallel_games,
)
from app.models.core import BoardType
from app.models import Move, MoveType, Position, GameState
from app.training.generate_data import create_initial_state
from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine

# Import coordination helpers for task limits and duration tracking
from app.coordination.helpers import (
    has_coordination,
    get_task_types,
    can_spawn_safe,
    register_running_task_safe,
    record_task_completion_safe,
)

# Curriculum feedback for adaptive training weights
try:
    from app.training.curriculum_feedback import record_selfplay_game, get_curriculum_feedback
    HAS_CURRICULUM_FEEDBACK = True
except ImportError:
    HAS_CURRICULUM_FEEDBACK = False
    record_selfplay_game = None
    get_curriculum_feedback = None

# For backwards compatibility
HAS_COORDINATION = has_coordination()
TaskType = get_task_types()

# NOTE: Victory type derivation is now handled internally by the
# BatchGameState class in gpu_parallel_games.py, which derives
# victory_type and stalemate_tiebreaker from the final GPU state.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Default Heuristic Weights
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


def _parse_move(move_dict: Dict[str, Any], move_number: int, timestamp: str) -> Move:
    """Parse a move dict into a Move object."""
    move_type_str = move_dict.get("type", "unknown")

    # Map move type strings to MoveType enum
    # NOTE: Only include move types that exist in the MoveType enum
    move_type_map = {
        "place_ring": MoveType.PLACE_RING,
        "skip_placement": MoveType.SKIP_PLACEMENT,
        "no_placement_action": MoveType.NO_PLACEMENT_ACTION,
        "swap_sides": MoveType.SWAP_SIDES,
        "move_stack": MoveType.MOVE_STACK,
        "move_ring": MoveType.MOVE_RING,
        "no_movement_action": MoveType.NO_MOVEMENT_ACTION,
        "overtaking_capture": MoveType.OVERTAKING_CAPTURE,
        "continue_capture_segment": MoveType.CONTINUE_CAPTURE_SEGMENT,
        "skip_capture": MoveType.SKIP_CAPTURE,
        "choose_line_option": MoveType.CHOOSE_LINE_OPTION,
        "choose_line_reward": MoveType.CHOOSE_LINE_REWARD,
        "process_line": MoveType.PROCESS_LINE,
        "no_line_action": MoveType.NO_LINE_ACTION,
        "choose_territory_option": MoveType.CHOOSE_TERRITORY_OPTION,
        "process_territory_region": MoveType.PROCESS_TERRITORY_REGION,
        "no_territory_action": MoveType.NO_TERRITORY_ACTION,
        "skip_territory_processing": MoveType.SKIP_TERRITORY_PROCESSING,
        "eliminate_rings_from_stack": MoveType.ELIMINATE_RINGS_FROM_STACK,
        "forced_elimination": MoveType.FORCED_ELIMINATION,
        "recovery_slide": MoveType.RECOVERY_SLIDE,
        "skip_recovery": MoveType.SKIP_RECOVERY,
    }

    move_type = move_type_map.get(move_type_str, MoveType.PLACE_RING)

    # Parse positions
    from_dict = move_dict.get("from")
    to_dict = move_dict.get("to")
    capture_dict = move_dict.get("capture_target")

    from_pos = Position(x=from_dict["x"], y=from_dict["y"], z=from_dict.get("z")) if from_dict else None
    to_pos = Position(x=to_dict["x"], y=to_dict["y"], z=to_dict.get("z")) if to_dict else None
    capture_target = Position(x=capture_dict["x"], y=capture_dict["y"], z=capture_dict.get("z")) if capture_dict else None

    # For overtaking captures, compute capture_target if not provided
    # The capture target is the midpoint between from and to (the stack being jumped over)
    if move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT) and capture_target is None:
        if from_pos and to_pos:
            mid_x = (from_pos.x + to_pos.x) // 2
            mid_y = (from_pos.y + to_pos.y) // 2
            capture_target = Position(x=mid_x, y=mid_y)

    return Move(
        id=f"move-{move_number}",
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        timestamp=timestamp,
        think_time=move_dict.get("think_time_ms", 0),
        move_number=move_number,
    )


def _get_board_type(board_str: str) -> BoardType:
    """Convert board type string to BoardType enum."""
    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        # Legacy alias used by some historical JSONLs / embeddings.
        "square25": BoardType.HEXAGONAL,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
    }
    return board_map.get(board_str, BoardType.SQUARE8)


def _expand_gpu_moves_to_canonical(
    moves: List[Move],
    initial_state: GameState,
) -> Tuple[List[Move], GameState]:
    """Expand GPU simplified moves to canonical moves with phase handling.

    GPU moves are coarse-grained (place_ring, move_stack, overtaking_capture, recovery_slide).
    The canonical game engine requires explicit phase handling moves (no_line_action,
    no_territory_action) when those phases are visited but have no actions.

    GPU selfplay automatically processes lines and territory during its phases, but
    doesn't emit the explicit moves. This function:
    1. Detects when the canonical engine expects line/territory processing
    2. Queries the engine for available line/territory moves
    3. Applies those moves (selecting first option) to match GPU's auto-processing
    4. Only inserts NO_*_ACTION when there truly are no actions available

    Returns:
        Tuple of (expanded_moves, final_state)
    """
    from app.models.core import GamePhase

    expanded_moves = []
    current_state = initial_state
    move_num = 0

    for gpu_move in moves:
        # Insert phase handling moves as needed before applying the GPU move
        # Use a safety counter to prevent infinite loops
        safety_counter = 0
        max_phase_iterations = 100  # Prevent runaway loops

        while current_state.game_status.value == "active" and safety_counter < max_phase_iterations:
            safety_counter += 1
            phase = current_state.current_phase
            player = current_state.current_player

            # Check if we need to insert a phase handling move
            if phase == GamePhase.LINE_PROCESSING:
                # GPU auto-processes lines - check if there are lines to process
                if gpu_move.type not in (MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION, MoveType.NO_LINE_ACTION):
                    # Get available line processing moves
                    line_moves = GameEngine._get_line_processing_moves(current_state, player)

                    if line_moves:
                        # There are lines - apply them (GPU already processed these)
                        # First apply PROCESS_LINE moves, then CHOOSE_LINE_OPTION
                        process_moves = [m for m in line_moves if m.type == MoveType.PROCESS_LINE]
                        choose_moves = [m for m in line_moves if m.type == MoveType.CHOOSE_LINE_OPTION]

                        # Apply a CHOOSE_LINE_OPTION if available (this is the actual collapse)
                        if choose_moves:
                            line_move = choose_moves[0]  # Pick first option
                            move_num += 1
                            phase_move = Move(
                                id=f"move-{move_num}",
                                type=line_move.type,
                                player=player,
                                to=line_move.to,
                                formed_lines=line_move.formed_lines,
                                collapsed_markers=line_move.collapsed_markers,
                                timestamp=gpu_move.timestamp,
                                think_time=0,
                                move_number=move_num,
                            )
                            expanded_moves.append(phase_move)
                            current_state = GameEngine.apply_move(current_state, phase_move)
                            continue
                        elif process_moves:
                            # Only PROCESS_LINE available (shouldn't happen often)
                            line_move = process_moves[0]
                            move_num += 1
                            phase_move = Move(
                                id=f"move-{move_num}",
                                type=line_move.type,
                                player=player,
                                to=line_move.to,
                                formed_lines=line_move.formed_lines,
                                timestamp=gpu_move.timestamp,
                                think_time=0,
                                move_number=move_num,
                            )
                            expanded_moves.append(phase_move)
                            current_state = GameEngine.apply_move(current_state, phase_move)
                            continue
                    else:
                        # No lines to process - insert NO_LINE_ACTION
                        move_num += 1
                        phase_move = Move(
                            id=f"move-{move_num}",
                            type=MoveType.NO_LINE_ACTION,
                            player=player,
                            timestamp=gpu_move.timestamp,
                            think_time=0,
                            move_number=move_num,
                        )
                        expanded_moves.append(phase_move)
                        current_state = GameEngine.apply_move(current_state, phase_move)
                        continue

            elif phase == GamePhase.TERRITORY_PROCESSING:
                # GPU auto-processes territory - check if there are territory decisions
                if gpu_move.type not in (MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION, MoveType.NO_TERRITORY_ACTION, MoveType.ELIMINATE_RINGS_FROM_STACK):
                    # Get available territory processing moves
                    territory_moves = GameEngine._get_territory_processing_moves(current_state, player)

                    if territory_moves:
                        # There are territory decisions - apply the first one
                        terr_move = territory_moves[0]
                        move_num += 1
                        phase_move = Move(
                            id=f"move-{move_num}",
                            type=terr_move.type,
                            player=player,
                            to=terr_move.to,
                            timestamp=gpu_move.timestamp,
                            think_time=0,
                            move_number=move_num,
                        )
                        expanded_moves.append(phase_move)
                        current_state = GameEngine.apply_move(current_state, phase_move)
                        continue
                    else:
                        # No territory to process - insert NO_TERRITORY_ACTION
                        move_num += 1
                        phase_move = Move(
                            id=f"move-{move_num}",
                            type=MoveType.NO_TERRITORY_ACTION,
                            player=player,
                            timestamp=gpu_move.timestamp,
                            think_time=0,
                            move_number=move_num,
                        )
                        expanded_moves.append(phase_move)
                        current_state = GameEngine.apply_move(current_state, phase_move)
                        continue

            elif phase == GamePhase.CAPTURE:
                # If next GPU move isn't a capture, insert skip_capture
                if gpu_move.type not in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT, MoveType.SKIP_CAPTURE):
                    move_num += 1
                    phase_move = Move(
                        id=f"move-{move_num}",
                        type=MoveType.SKIP_CAPTURE,
                        player=player,
                        timestamp=gpu_move.timestamp,
                        think_time=0,
                        move_number=move_num,
                    )
                    expanded_moves.append(phase_move)
                    current_state = GameEngine.apply_move(current_state, phase_move)
                    continue

            elif phase == GamePhase.CHAIN_CAPTURE:
                # Chain capture phase - GPU records all captures as overtaking_capture
                # but canonical engine expects CONTINUE_CAPTURE_SEGMENT for chain captures
                if gpu_move.type not in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT, MoveType.SKIP_CAPTURE):
                    # Next GPU move isn't a capture - skip remaining captures
                    move_num += 1
                    phase_move = Move(
                        id=f"move-{move_num}",
                        type=MoveType.SKIP_CAPTURE,
                        player=player,
                        timestamp=gpu_move.timestamp,
                        think_time=0,
                        move_number=move_num,
                    )
                    expanded_moves.append(phase_move)
                    current_state = GameEngine.apply_move(current_state, phase_move)
                    continue

            # No phase move needed, break the while loop
            break

        if safety_counter >= max_phase_iterations:
            logger.warning(f"Phase handling loop exceeded {max_phase_iterations} iterations, breaking")

        # Determine the correct move type for captures based on phase
        # GPU records all captures as "overtaking_capture" but canonical engine expects:
        # - OVERTAKING_CAPTURE in CAPTURE phase (initial capture)
        # - CONTINUE_CAPTURE_SEGMENT in CHAIN_CAPTURE phase (chain captures)
        actual_move_type = gpu_move.type
        if gpu_move.type == MoveType.OVERTAKING_CAPTURE and current_state.current_phase == GamePhase.CHAIN_CAPTURE:
            actual_move_type = MoveType.CONTINUE_CAPTURE_SEGMENT

        # Now apply the actual GPU move with updated move number
        move_num += 1
        gpu_move_updated = Move(
            id=f"move-{move_num}",
            type=actual_move_type,
            player=gpu_move.player,
            from_pos=gpu_move.from_pos,
            to=gpu_move.to,
            capture_target=gpu_move.capture_target,
            timestamp=gpu_move.timestamp,
            think_time=gpu_move.think_time,
            move_number=move_num,
        )
        expanded_moves.append(gpu_move_updated)
        current_state = GameEngine.apply_move(current_state, gpu_move_updated)

    return expanded_moves, current_state


def load_weights_from_profile(
    weights_file: str,
    profile_name: str,
) -> Dict[str, float]:
    """Load heuristic weights from a profile file."""
    if not os.path.exists(weights_file):
        logger.warning(f"Weights file not found: {weights_file}, using defaults")
        return DEFAULT_WEIGHTS.copy()

    with open(weights_file, "r") as f:
        data = json.load(f)

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        logger.warning(f"Profile {profile_name} not found, using defaults")
        return DEFAULT_WEIGHTS.copy()

    return profiles[profile_name].get("weights", DEFAULT_WEIGHTS.copy())


# =============================================================================
# GPU Self-Play Generator
# =============================================================================


class GPUSelfPlayGenerator:
    """Generate self-play games using GPU parallel simulation.

    Shadow Validation (Phase 2):
        When enabled, a subset of GPU-generated moves are validated against
        the canonical CPU rules engine. This catches GPU/CPU divergence early.

        Configuration:
            shadow_validation: Enable shadow validation
            shadow_sample_rate: Fraction of moves to validate (default 0.05 = 5%)
            shadow_threshold: Max divergence rate before error (default 0.001 = 0.1%)

        See GPU_PIPELINE_ROADMAP.md Section 7.4.2 for architecture details.
    """

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        batch_size: int = 256,
        max_moves: int = 10000,
        device: Optional[torch.device] = None,
        weights: Optional[Dict[str, float]] = None,
        engine_mode: str = "heuristic-only",
        shadow_validation: bool = False,
        shadow_sample_rate: float = 0.05,
        shadow_threshold: float = 0.001,
        lps_victory_rounds: int = 3,
        rings_per_player: Optional[int] = None,
        board_type: Optional[str] = None,
        use_heuristic_selection: bool = False,
        weight_noise: float = 0.0,
        use_policy: bool = False,
        policy_model_path: Optional[str] = None,
        temperature: float = 1.0,
        noise_scale: float = 0.1,
        min_game_length: int = 0,
        random_opening_moves: int = 0,
        temperature_mix: Optional[str] = None,
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.device = device or get_device()
        self.engine_mode = engine_mode
        self.shadow_validation = shadow_validation
        self.lps_victory_rounds = lps_victory_rounds
        self.rings_per_player = rings_per_player
        self.board_type = board_type
        self.min_game_length = min_game_length
        self.filtered_short_games = 0  # Track filtered games for logging
        self.temperature_mix = temperature_mix
        self.base_temperature = temperature
        # Temperature levels for difficulty mixing: optimal → random
        self.mix_temperatures = [0.5, 1.0, 2.0, 4.0]
        self._temp_cycle_idx = 0  # Current position in temperature cycle
        # For random-only mode, use None weights (uniform random)
        # For heuristic-only mode, use provided weights or defaults
        # For nnue-guided mode, use heuristic + NNUE evaluation
        if engine_mode == "random-only":
            self.weights = None  # Triggers uniform random selection
        else:
            self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Initialize NNUE evaluator for nnue-guided mode
        # NOTE: Full NNUE-guided move selection is WIP. Currently uses heuristic
        # with NNUE as a secondary scoring signal. Future work:
        # - Evaluate candidate moves with NNUE before selection
        # - Use NNUE scores with softmax for move sampling
        # - See BatchNNUEEvaluator in app/ai/nnue.py for batch evaluation API
        self.nnue_evaluator = None
        if engine_mode == "nnue-guided":
            board_type_map = {
                8: BoardType.SQUARE8,
                19: BoardType.SQUARE19,
                25: BoardType.HEXAGONAL,
            }
            bt = board_type_map.get(board_size, BoardType.SQUARE8)
            self.nnue_evaluator = BatchNNUEEvaluator(
                board_type=bt,
                num_players=num_players,
                device=self.device,
            )
            if self.nnue_evaluator.available:
                logger.info(f"NNUE-guided mode: NNUE evaluator loaded for {bt.value}")
                logger.info("  Note: Currently using heuristic + NNUE logging. Full NNUE-guided WIP.")
            else:
                logger.warning("NNUE-guided mode: NNUE model not available, using heuristic only")

        self.runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=self.device,
            shadow_validation=shadow_validation,
            shadow_sample_rate=shadow_sample_rate,
            shadow_threshold=shadow_threshold,
            lps_victory_rounds=lps_victory_rounds,
            rings_per_player=rings_per_player,
            board_type=board_type,
            use_heuristic_selection=use_heuristic_selection,
            weight_noise=weight_noise,
            temperature=temperature,
            noise_scale=noise_scale,
            random_opening_moves=random_opening_moves,
        )

        # Log shadow validation status
        if shadow_validation:
            logger.info(f"Shadow validation ENABLED: sample_rate={shadow_sample_rate}, threshold={shadow_threshold}")
        else:
            logger.info("Shadow validation disabled")

        # Load policy model if requested
        self.use_policy = use_policy
        if use_policy:
            if self.runner.load_policy_model(policy_model_path):
                logger.info("Policy-based move selection ENABLED")
            else:
                logger.warning("Policy model not available, falling back to heuristic/center-bias")
                self.use_policy = False

        # Statistics
        self.total_games = 0
        self.total_moves = 0
        self.total_time = 0.0
        self.wins_by_player = {i: 0 for i in range(1, num_players + 1)}
        self.draws = 0

        # Pre-compute initial state for training data compatibility
        # All GPU games start from the same initial state with custom rules applied
        board_type_map = {
            8: BoardType.SQUARE8,
            19: BoardType.SQUARE19,
            # Hex boards use a 25×25 embedding in the GPU kernels.
            25: BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board_size, BoardType.SQUARE8)
        self._initial_state = create_initial_state(
            board_type,
            num_players,
            rings_per_player_override=rings_per_player,
            lps_rounds_required=lps_victory_rounds,
        )
        self._initial_state_json = self._initial_state.model_dump(mode="json")

    def _get_next_temperature(self) -> float:
        """Get next temperature for difficulty mixing.

        Returns:
            Temperature value based on mixing mode:
            - 'uniform': cycles through all temps equally
            - 'weighted': biased toward optimal (lower) temps
            - 'random': random selection each batch
        """
        import random as _random

        if self.temperature_mix == "uniform":
            # Cycle through temperatures
            temp = self.mix_temperatures[self._temp_cycle_idx]
            self._temp_cycle_idx = (self._temp_cycle_idx + 1) % len(self.mix_temperatures)
            return temp

        elif self.temperature_mix == "weighted":
            # Weighted toward lower (more optimal) temperatures
            # Weights: [0.4, 0.3, 0.2, 0.1] for [0.5, 1.0, 2.0, 4.0]
            weights = [0.4, 0.3, 0.2, 0.1]
            return _random.choices(self.mix_temperatures, weights=weights)[0]

        elif self.temperature_mix == "random":
            # Uniform random selection
            return _random.choice(self.mix_temperatures)

        else:
            return self.base_temperature

    def generate_batch(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a batch of games.

        Returns dict with:
            - winners: List of winning player numbers (0 = draw)
            - move_counts: List of move counts per game
            - games_per_second: Throughput
            - elapsed_seconds: Wall time
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Apply temperature mixing if enabled
        if self.temperature_mix:
            temp = self._get_next_temperature()
            self.runner.set_temperature(temp)
        else:
            # Restore base temperature if not mixing
            self.runner.set_temperature(self.base_temperature)

        start = time.time()
        # Pass None for random mode (uniform random), weights for heuristic mode
        weights_list = None if self.weights is None else [self.weights] * self.batch_size
        results = self.runner.run_games(
            weights_list=weights_list,
            max_moves=self.max_moves,
        )
        elapsed = time.time() - start

        # Update statistics
        self.total_games += self.batch_size
        self.total_moves += sum(results["move_counts"])
        self.total_time += elapsed

        for winner in results["winners"]:
            if winner == 0:
                self.draws += 1
            else:
                self.wins_by_player[winner] = self.wins_by_player.get(winner, 0) + 1

        # Record results for curriculum feedback (adaptive training weights)
        if HAS_CURRICULUM_FEEDBACK and self.board_type:
            config_key = f"{self.board_type}_{self.num_players}p"
            for winner in results["winners"]:
                # For selfplay, winner=1 means model won, winner=0 means draw
                # We record each game result for curriculum tracking
                outcome = 1 if winner == 1 else (-1 if winner > 1 else 0)
                record_selfplay_game(config_key, outcome)

        return results

    def generate_games(
        self,
        num_games: int,
        output_file: Optional[str] = None,
        output_db: Optional[str] = None,
        progress_interval: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate multiple batches of games.

        Args:
            num_games: Total number of games to generate
            output_file: Optional JSONL file to stream results to
            output_db: Optional SQLite DB file to write games to (canonical format)
            progress_interval: Log progress every N batches

        Returns:
            List of game records
        """
        all_records = []
        num_batches = (num_games + self.batch_size - 1) // self.batch_size

        logger.info(f"Generating {num_games} games in {num_batches} batches...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info("")

        start_time = time.time()
        file_handle = None
        db = None

        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            file_handle = open(output_file, "w")
            # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
            try:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.error(f"Cannot acquire lock on {output_file} - another process is writing")
                file_handle.close()
                sys.exit(1)

        if output_db:
            # NOTE: Canonical DB output is disabled for GPU games.
            # GPU selfplay moves don't map 1:1 to canonical game engine phases.
            # Use scripts/import_gpu_selfplay_to_db.py for post-hoc conversion.
            logger.info(f"  Note: DB output not supported for GPU games (use import_gpu_selfplay_to_db.py)")

        # Buffered write for better I/O performance (flush every N records)
        WRITE_BUFFER_SIZE = 100  # Records before flush - balances throughput vs data safety
        write_buffer: List[str] = []

        def flush_write_buffer():
            """Flush accumulated records to file."""
            nonlocal write_buffer
            if file_handle and write_buffer:
                file_handle.write("\n".join(write_buffer) + "\n")
                file_handle.flush()
                write_buffer = []

        try:
            for batch_idx in range(num_batches):
                # Adjust batch size for last batch
                remaining = num_games - len(all_records)
                actual_batch = min(self.batch_size, remaining)

                # Generate batch
                results = self.generate_batch(seed=batch_idx * 1000)

                # Create game records
                # Use the board_type that was passed in, not inferred from board_size
                board_type_str = self.board_type or {8: "square8", 9: "hex8", 19: "square19", 25: "hexagonal"}.get(self.board_size, "square8")
                for i in range(actual_batch):
                    move_count = int(results["move_counts"][i])

                    # Filter out games that are too short
                    if self.min_game_length > 0 and move_count < self.min_game_length:
                        self.filtered_short_games += 1
                        continue

                    game_idx = len(all_records)
                    vtype = results["victory_types"][i]
                    record = {
                        # === Core game identifiers ===
                        "game_id": f"gpu_{board_type_str}_{self.num_players}p_{game_idx}_{int(datetime.now().timestamp())}",
                        "board_type": board_type_str,  # Standardized: square8, square19, hexagonal
                        "board_size": self.board_size,  # Legacy field for compatibility
                        "num_players": self.num_players,
                        # === Game outcome ===
                        "winner": int(results["winners"][i]),
                        "move_count": move_count,
                        "max_moves": self.max_moves,
                        "status": "completed",
                        "game_status": "completed",
                        "victory_type": vtype,
                        "stalemate_tiebreaker": results["stalemate_tiebreakers"][i],
                        "termination_reason": f"status:completed:{vtype}",
                        # === Engine/opponent metadata ===
                        "engine_mode": "gpu_heuristic",
                        "opponent_type": "selfplay",
                        "player_types": ["gpu_batch"] * self.num_players,
                        "batch_id": batch_idx,
                        # === Training data (required for NPZ export) ===
                        "moves": results["move_histories"][i],
                        "initial_state": self._initial_state_json,
                        # === Timing metadata ===
                        "timestamp": datetime.now().isoformat(),
                        "created_at": datetime.now().isoformat(),
                        # === Source tracking ===
                        "source": "run_gpu_selfplay.py",
                        "device": str(self.device),
                    }
                    all_records.append(record)

                    # Buffered write: accumulate records and flush periodically
                    if file_handle:
                        write_buffer.append(json.dumps(record))
                        if len(write_buffer) >= WRITE_BUFFER_SIZE:
                            flush_write_buffer()

                    # NOTE: DB storage disabled for GPU games.
                    # GPU selfplay uses simplified move semantics that don't map 1:1 to
                    # the canonical game engine phases. The JSONL output contains full
                    # move data sufficient for training. If canonical DB format is needed,
                    # use scripts/import_gpu_selfplay_to_db.py for post-hoc conversion.
                    # See GPU_PIPELINE_ROADMAP.md for GPU/canonical parity details.

                # Flush at end of each batch for safety
                flush_write_buffer()

                # Progress logging
                if (batch_idx + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    games_done = len(all_records)
                    games_per_sec = games_done / elapsed if elapsed > 0 else 0
                    eta = (num_games - games_done) / games_per_sec if games_per_sec > 0 else 0

                    logger.info(
                        f"  Batch {batch_idx + 1}/{num_batches}: "
                        f"{games_done}/{num_games} games, "
                        f"{games_per_sec:.1f} g/s, "
                        f"ETA: {eta:.0f}s"
                    )

        finally:
            # Flush any remaining buffered records before closing
            flush_write_buffer()
            if file_handle:
                file_handle.close()

        return all_records

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        total_decided = sum(self.wins_by_player.values())

        stats = {
            "total_games": self.total_games,
            "total_moves": self.total_moves,
            "total_time_seconds": self.total_time,
            "games_per_second": self.total_games / self.total_time if self.total_time > 0 else 0,
            "moves_per_game": self.total_moves / self.total_games if self.total_games > 0 else 0,
            "wins_by_player": self.wins_by_player,
            "draws": self.draws,
            "draw_rate": self.draws / self.total_games if self.total_games > 0 else 0,
            "board_size": self.board_size,
            "num_players": self.num_players,
            "batch_size": self.batch_size,
            "max_moves": self.max_moves,
            "device": str(self.device),
            "weights": self.weights,
            "min_game_length": self.min_game_length,
            "filtered_short_games": self.filtered_short_games,
        }

        # Add win rates by player
        for p in range(1, self.num_players + 1):
            wins = self.wins_by_player.get(p, 0)
            stats[f"p{p}_win_rate"] = wins / total_decided if total_decided > 0 else 0

        # Add shadow validation stats if enabled
        shadow_report = self.runner.get_shadow_validation_report()
        if shadow_report:
            stats["shadow_validation"] = shadow_report
            logger.info(f"Shadow validation report: {shadow_report['status']}")
            if shadow_report.get("divergence_rate", 0) > 0:
                logger.warning(f"  Divergence rate: {shadow_report['divergence_rate']:.4%}")
                logger.warning(f"  Total divergences: {shadow_report.get('total_divergences', 0)}")

        return stats


# =============================================================================
# Main Entry Point
# =============================================================================


def run_gpu_selfplay(
    board_type: str,
    num_players: int,
    num_games: int,
    output_dir: str,
    batch_size: int = 256,
    max_moves: int = 10000,
    weights: Optional[Dict[str, float]] = None,
    engine_mode: str = "heuristic-only",
    seed: int = 42,
    shadow_validation: bool = False,
    shadow_sample_rate: float = 0.05,
    shadow_threshold: float = 0.001,
    lps_victory_rounds: int = 3,
    rings_per_player: Optional[int] = None,
    output_db: Optional[str] = None,
    use_heuristic_selection: bool = False,
    weight_noise: float = 0.0,
    use_policy: bool = False,
    policy_model_path: Optional[str] = None,
    temperature: float = 1.0,
    noise_scale: float = 0.1,
    min_game_length: int = 0,
    random_opening_moves: int = 0,
    temperature_mix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run GPU-accelerated self-play generation.

    Args:
        board_type: Board type (square8, square19, hex, hexagonal)
        num_players: Number of players
        num_games: Total games to generate
        output_dir: Output directory
        batch_size: GPU batch size
        max_moves: Max moves per game
        weights: Heuristic weights (ignored in random-only mode)
        engine_mode: Engine mode (random-only or heuristic-only)
        seed: Random seed
        shadow_validation: Enable shadow validation (GPU/CPU parity checking)
        shadow_sample_rate: Fraction of moves to validate (default 5%)
        shadow_threshold: Max divergence rate before error (default 0.1%)
        lps_victory_rounds: LPS victory threshold (default 3)
        rings_per_player: Starting rings per player (None = board default)
        output_db: Optional path to SQLite DB for canonical game storage
        use_heuristic_selection: Use heuristic-based move selection instead of center-bias random
        weight_noise: Multiplicative noise factor (0.0-1.0) for heuristic weights diversity

    Returns:
        Statistics dict
    """
    # Create output directory with explicit error handling
    output_dir = os.path.abspath(output_dir)  # Resolve to absolute path
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    np.random.seed(seed)
    torch.manual_seed(seed)

    board_size = {"square8": 8, "square19": 19, "hex": 25, "hexagonal": 25, "hex8": 9}.get(board_type.lower(), 8)

    # Auto-adjust max_moves based on board type and player count if using default
    # Larger boards and more players need more moves to reach natural game end
    if max_moves == 500:  # Default value - auto-adjust
        max_moves_table = {
            # (board_type, num_players): max_moves
            ("square8", 2): 500,
            ("square8", 3): 800,
            ("square8", 4): 1200,
            ("square19", 2): 1200,
            ("square19", 3): 1600,
            ("square19", 4): 2000,
            ("hex", 2): 1200,
            ("hex", 3): 1600,
            ("hex", 4): 2000,
            ("hexagonal", 2): 1200,
            ("hexagonal", 3): 1600,
            ("hexagonal", 4): 2000,
        }
        key = (board_type.lower(), num_players)
        if key in max_moves_table:
            max_moves = max_moves_table[key]
            logger.info(f"Auto-adjusted max_moves to {max_moves} for {board_type} {num_players}p")

    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED SELF-PLAY GENERATION")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Engine mode: {engine_mode}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max moves: {max_moves}")
    logger.info(f"LPS victory rounds: {lps_victory_rounds}")
    logger.info(f"Rings per player: {rings_per_player or 'board default'}")
    logger.info(f"Shadow validation: {shadow_validation}")
    if shadow_validation:
        logger.info(f"  Sample rate: {shadow_sample_rate:.1%}")
        logger.info(f"  Threshold: {shadow_threshold:.2%}")
    if use_policy:
        logger.info(f"Move selection: policy-guided" + (f" ({policy_model_path})" if policy_model_path else ""))
    else:
        logger.info(f"Move selection: {'heuristic-based' if use_heuristic_selection else 'center-bias random'}")
    logger.info(f"Weight noise: {weight_noise:.1%}" if weight_noise > 0 else "Weight noise: disabled")
    if min_game_length > 0:
        logger.info(f"Min game length filter: {min_game_length} moves")
    if random_opening_moves > 0:
        logger.info(f"Random opening moves: {random_opening_moves}")
    if temperature_mix:
        logger.info(f"Temperature mixing: {temperature_mix} (temps: [0.5, 1.0, 2.0, 4.0])")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create generator
    generator = GPUSelfPlayGenerator(
        board_size=board_size,
        num_players=num_players,
        batch_size=batch_size,
        max_moves=max_moves,
        weights=weights,
        engine_mode=engine_mode,
        shadow_validation=shadow_validation,
        shadow_sample_rate=shadow_sample_rate,
        shadow_threshold=shadow_threshold,
        lps_victory_rounds=lps_victory_rounds,
        rings_per_player=rings_per_player,
        board_type=board_type,
        use_heuristic_selection=use_heuristic_selection,
        weight_noise=weight_noise,
        use_policy=use_policy,
        policy_model_path=policy_model_path,
        temperature=temperature,
        noise_scale=noise_scale,
        min_game_length=min_game_length,
        random_opening_moves=random_opening_moves,
        temperature_mix=temperature_mix,
    )

    # Generate games - use unique filename per config to avoid lock contention
    games_file = os.path.join(output_dir, f"games_{board_type}_{num_players}p_{os.getpid()}.jsonl")
    records = generator.generate_games(
        num_games=num_games,
        output_file=games_file,
        progress_interval=10,
    )

    # Get and save statistics
    stats = generator.get_statistics()
    stats["timestamp"] = datetime.now().isoformat()
    stats["seed"] = seed

    stats_file = os.path.join(output_dir, "stats.json")
    # Ensure directory exists right before write (handles race conditions on remote hosts)
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
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
    logger.info(f"Throughput: {stats['games_per_second']:.1f} games/sec")
    logger.info(f"Draw rate: {stats['draw_rate']:.1%}")
    if stats.get('filtered_short_games', 0) > 0:
        logger.info(f"Filtered short games (<{stats['min_game_length']} moves): {stats['filtered_short_games']}")
    logger.info("")
    logger.info("Win rates by player:")
    for p in range(1, num_players + 1):
        logger.info(f"  Player {p}: {stats[f'p{p}_win_rate']:.1%}")
    logger.info("")
    logger.info(f"Games saved to: {games_file}")
    logger.info(f"Stats saved to: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated self-play data generation"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex", "hexagonal", "hex8"],
        help="Board type",
    )
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="heuristic-only",
        choices=["random-only", "heuristic-only", "nnue-guided"],
        help="Engine mode: random-only (uniform random), heuristic-only (GPU heuristic), or nnue-guided (NNUE + heuristic)",
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
        default=1000,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="GPU batch size (games per batch)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay/gpu",
        help="Output directory",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default=None,
        help="Output SQLite DB path for canonical game storage (default: output-dir/games.db)",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        help="Path to heuristic weights JSON file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Profile name in weights file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run GPU benchmark",
    )

    # Shadow validation options
    parser.add_argument(
        "--shadow-validation",
        action="store_true",
        help="Enable shadow validation (GPU/CPU parity checking)",
    )
    parser.add_argument(
        "--shadow-sample-rate",
        type=float,
        default=0.05,
        help="Fraction of moves to validate against CPU (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.001,
        help="Max divergence rate before error (default: 0.001 = 0.1%%)",
    )

    # Game rule configuration
    parser.add_argument(
        "--lps-victory-rounds",
        type=int,
        default=3,
        help="LPS victory threshold in consecutive rounds (default: 3)",
    )
    parser.add_argument(
        "--rings-per-player",
        type=int,
        default=None,
        help="Starting rings per player (default: board default - 18/72/96)",
    )

    # Move selection mode
    parser.add_argument(
        "--use-heuristic",
        action="store_true",
        help="Use heuristic-based move selection (center distance, capture value, line potential) instead of center-bias random",
    )
    parser.add_argument(
        "--weight-noise",
        type=float,
        default=0.0,
        help="Multiplicative noise (0.0-1.0) for heuristic weights to increase training diversity. "
             "Each weight is multiplied by uniform(1-noise, 1+noise). Default: 0.0 (no noise)",
    )
    parser.add_argument(
        "--use-policy",
        action="store_true",
        help="Use policy network for move selection instead of heuristic/center-bias. "
             "Loads the trained policy model from models/nnue/nnue_policy_{board}_{num_players}p.pt",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default=None,
        help="Path to custom policy model (default: auto-detect based on board type)",
    )

    # Curriculum learning parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for move sampling (higher = more random). "
             "Used for curriculum learning. Default: 1.0",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Scale of noise added to move scores. "
             "Used for curriculum learning diversity. Default: 0.1",
    )

    # Data quality filters
    parser.add_argument(
        "--min-game-length",
        type=int,
        default=0,
        help="Minimum move count for a game to be recorded (default: 0 = no filter). "
             "Games shorter than this are discarded to improve training data quality.",
    )
    parser.add_argument(
        "--random-opening-moves",
        type=int,
        default=0,
        help="Number of initial moves to select uniformly at random (default: 0). "
             "Increases opening diversity for training by randomizing the first N moves.",
    )

    # Difficulty mixing for training diversity
    parser.add_argument(
        "--temperature-mix",
        type=str,
        default=None,
        choices=["uniform", "weighted", "random"],
        help="Mix different temperature levels per game for training diversity. "
             "'uniform' = equal split across temps, 'weighted' = bias toward optimal, "
             "'random' = random temperature per batch. Temps: [0.5, 1.0, 2.0, 4.0]",
    )

    # Add ramdrive storage options
    add_ramdrive_args(parser)

    args = parser.parse_args()

    if args.benchmark_only:
        logger.info("Running GPU benchmark...")
        device = get_device()
        board_size = {"square8": 8, "square19": 19, "hex": 25, "hexagonal": 25, "hex8": 9}.get(args.board.lower(), 8)
        results = benchmark_parallel_games(
            batch_sizes=[32, 64, 128, 256, 512, 1024],
            board_size=board_size,
            max_moves=100,
            device=device,
        )
        logger.info("Benchmark results:")
        for i, bs in enumerate(results["batch_size"]):
            logger.info(
                f"  Batch {bs}: {results['games_per_second'][i]:.1f} games/sec, "
                f"{results['moves_per_second'][i]:.1f} moves/sec"
            )
        return

    # Load weights if specified
    weights = None
    if args.weights_file and args.profile:
        weights = load_weights_from_profile(args.weights_file, args.profile)
        logger.info(f"Loaded weights from {args.weights_file}:{args.profile}")

    # Resource guard: Check disk/memory/GPU before starting (80% limits)
    # Also import graceful degradation functions for dynamic resource management
    try:
        from app.utils.resource_guard import (
            check_disk_space, check_memory, check_gpu_memory,
            get_resource_status, LIMITS,
            should_proceed_with_priority, OperationPriority, get_degradation_level,
        )
        # Estimate output size: ~1KB per game for JSONL + ~100KB per 100 games for NPZ
        estimated_output_mb = (args.num_games * 0.001) + (args.num_games / 100 * 0.1) + 50
        if not check_disk_space(required_gb=max(2.0, estimated_output_mb / 1024)):
            logger.error(f"Insufficient disk space (limit: {LIMITS.DISK_MAX_PERCENT}%)")
            sys.exit(1)
        if not check_memory(required_gb=2.0):
            logger.error(f"Insufficient memory (limit: {LIMITS.MEMORY_MAX_PERCENT}%)")
            sys.exit(1)
        if not check_gpu_memory(required_gb=1.0):
            logger.warning("GPU memory constrained, may affect performance")
        logger.info(f"Resource check passed: {get_resource_status()['can_proceed']}")

        # Graceful degradation: GPU selfplay is NORMAL priority
        # Under heavy resource pressure, reduce workload or skip
        degradation = get_degradation_level()
        if degradation >= 4:  # CRITICAL - resources at/above limits
            logger.error("Resources at critical levels, aborting selfplay")
            sys.exit(1)
        elif degradation >= 3:  # HEAVY - only critical ops proceed
            if not should_proceed_with_priority(OperationPriority.NORMAL):
                logger.warning("Heavy resource pressure, reducing num_games by 75%")
                args.num_games = max(10, args.num_games // 4)
        elif degradation >= 2:  # MODERATE - reduce workload
            if not should_proceed_with_priority(OperationPriority.NORMAL):
                logger.warning("Moderate resource pressure, reducing num_games by 50%")
                args.num_games = max(10, args.num_games // 2)
        elif degradation >= 1:  # LIGHT - slight reduction
            logger.info(f"Light resource pressure (degradation level {degradation})")
    except ImportError:
        logger.debug("Resource guard not available, skipping checks")
        should_proceed_with_priority = None  # Mark as unavailable

    # Check coordination before spawning (using safe helpers)
    task_id = None
    start_time = time.time()
    if HAS_COORDINATION and TaskType is not None:
        import socket
        node_id = socket.gethostname()
        allowed, reason = can_spawn_safe(TaskType.GPU_SELFPLAY, node_id)
        if not allowed:
            logger.warning(f"Coordination denied spawn: {reason}")
            logger.info("Proceeding anyway (coordination is advisory)")

        # Register task for tracking (safe version handles errors internally)
        task_id = f"gpu_selfplay_{args.board}_{args.num_players}p_{os.getpid()}"
        if register_running_task_safe(task_id, "gpu_selfplay", 60.0):
            logger.info(f"Registered task {task_id} with coordinator")

    # Determine output directory: --output-dir > --ram-storage > default
    output_dir = args.output_dir
    syncer = None
    if getattr(args, 'ram_storage', False) and args.output_dir == "data/selfplay/gpu":
        # Only use ramdrive if output_dir wasn't explicitly set
        ramdrive_config = get_config_from_args(args)
        ramdrive_config.subdirectory = f"selfplay/gpu_{args.board}_{args.num_players}p"
        output_dir = str(get_games_directory(prefer_ramdrive=True, config=ramdrive_config))
        logger.info(f"Using ramdrive storage: {output_dir}")

        # Set up periodic sync if requested
        sync_interval = getattr(args, 'sync_interval', 0)
        sync_target = getattr(args, 'sync_target', '')
        if sync_interval > 0 and sync_target:
            syncer = RamdriveSyncer(
                source_dir=Path(output_dir),
                target_dir=Path(sync_target),
                interval=sync_interval,
                patterns=["*.db", "*.jsonl", "*.json", "*.npz"],
            )
            syncer.start()
            logger.info(f"Started ramdrive sync: {output_dir} -> {sync_target} every {sync_interval}s")

    try:
        run_gpu_selfplay(
            board_type=args.board,
            num_players=args.num_players,
            num_games=args.num_games,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_moves=args.max_moves,
            weights=weights,
            engine_mode=args.engine_mode,
            seed=args.seed,
            shadow_validation=args.shadow_validation,
            shadow_sample_rate=args.shadow_sample_rate,
            shadow_threshold=args.shadow_threshold,
            lps_victory_rounds=args.lps_victory_rounds,
            rings_per_player=args.rings_per_player,
            output_db=args.output_db,
            use_heuristic_selection=args.use_heuristic,
            weight_noise=args.weight_noise,
            use_policy=args.use_policy,
            policy_model_path=args.policy_model,
            temperature=args.temperature,
            noise_scale=args.noise_scale,
            min_game_length=args.min_game_length,
            random_opening_moves=args.random_opening_moves,
            temperature_mix=args.temperature_mix,
        )
    finally:
        # Stop ramdrive syncer and perform final sync
        if syncer:
            logger.info("Stopping ramdrive syncer and performing final sync...")
            syncer.stop(final_sync=True)
            logger.info(f"Ramdrive sync stats: {syncer.stats}")

        # Record task completion for duration learning (safe version handles errors)
        if HAS_COORDINATION and task_id:
            config = f"{args.board}_{args.num_players}p"
            actual_duration = time.time() - start_time
            if record_task_completion_safe(task_id, "gpu_selfplay", actual_duration):
                logger.info(f"Recorded task completion for duration learning")


if __name__ == "__main__":
    main()
