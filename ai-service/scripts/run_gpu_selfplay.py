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

import fcntl
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add app/ to path (must be early for app.* imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ramdrive utilities for high-speed I/O
import torch

from app.ai.gpu_batch import get_device
from app.ai.gpu_canonical_export import (
    _canvas_to_cube_coords,
    convert_gpu_move_to_canonical,
    gpu_move_type_to_canonical,
    gpu_phase_to_canonical,
    validate_canonical_move_sequence,
)

# Track canonical validation stats
_canonical_validation_stats = {
    "games_validated": 0,
    "games_valid": 0,
    "games_invalid": 0,
    "total_errors": 0,
}
from app.ai.gpu_parallel_games import (
    ParallelGameRunner,
    benchmark_parallel_games,
)
from app.ai.nnue import BatchNNUEEvaluator

# Import coordination helpers for task limits and duration tracking
from app.coordination.helpers import (
    can_spawn_safe,
    get_task_types,
    has_coordination,
    record_task_completion_safe,
    register_running_task_safe,
)
from app.game_engine import GameEngine
from app.models import GameState, Move, MoveType, Position
from app.models.core import BoardType
from app.training.initial_state import create_initial_state
from app.training.selfplay_config import SelfplayConfig, create_argument_parser
from app.training.temperature_scheduling import (
    TemperatureSchedule,
    LinearDecaySchedule,
    create_scheduler,
)
from app.utils.ramdrive import RamdriveSyncer, get_config_from_args, get_games_directory

# Curriculum feedback for adaptive training weights
try:
    from app.training.curriculum_feedback import get_curriculum_feedback, record_selfplay_game
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

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_gpu_selfplay")


def _emit_gpu_selfplay_complete(
    board_type: str,
    num_players: int,
    games_completed: int,
    total_samples: int,
    output_dir: str,
    db_path: str | None,
    duration_seconds: float,
) -> None:
    """Emit SELFPLAY_COMPLETE event for pipeline automation.

    Phase 3A.2 (Dec 2025): GPU selfplay now emits events to trigger
    downstream pipeline stages (sync, export, training).
    """
    import socket

    try:
        from app.coordination.event_router import publish_sync, StageEvent

        config_key = f"{board_type}_{num_players}p"
        publish_sync(
            event_type=StageEvent.GPU_SELFPLAY_COMPLETE,
            payload={
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "games_completed": games_completed,
                "games_count": games_completed,
                "total_samples": total_samples,
                "output_dir": output_dir,
                "db_path": str(db_path) if db_path else None,
                "duration_seconds": duration_seconds,
                "node_id": socket.gethostname(),
            },
            source="gpu_selfplay",
        )
        logger.info(f"[Pipeline] Emitted GPU_SELFPLAY_COMPLETE for {config_key}")
    except ImportError:
        logger.debug("[Pipeline] Event router not available")
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to emit GPU_SELFPLAY_COMPLETE: {e}")


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

# Persona Weights (full 45-weight profiles from heuristic_weights.py)
# =============================================================================
try:
    from app.ai.heuristic_weights import (
        HEURISTIC_V1_BALANCED,
        HEURISTIC_V1_AGGRESSIVE,
        HEURISTIC_V1_TERRITORIAL,
        HEURISTIC_V1_DEFENSIVE,
    )
    PERSONA_WEIGHTS = {
        "balanced": dict(HEURISTIC_V1_BALANCED),
        "aggressive": dict(HEURISTIC_V1_AGGRESSIVE),
        "territorial": dict(HEURISTIC_V1_TERRITORIAL),
        "defensive": dict(HEURISTIC_V1_DEFENSIVE),
    }
    HAS_PERSONAS = True
except ImportError:
    logger.warning("Could not import persona weights from heuristic_weights.py, using defaults")
    PERSONA_WEIGHTS = {
        "balanced": DEFAULT_WEIGHTS.copy(),
        "aggressive": DEFAULT_WEIGHTS.copy(),
        "territorial": DEFAULT_WEIGHTS.copy(),
        "defensive": DEFAULT_WEIGHTS.copy(),
    }
    HAS_PERSONAS = False

ALL_PERSONAS = list(PERSONA_WEIGHTS.keys())


def _parse_move(move_dict: dict[str, Any], move_number: int, timestamp: str) -> Move:
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
        # Line moves (RR-CANON: prefer choose_line_option)
        "choose_line_option": MoveType.CHOOSE_LINE_OPTION,
        "choose_line_reward": MoveType.CHOOSE_LINE_OPTION,  # Legacy -> canonical
        "process_line": MoveType.PROCESS_LINE,  # GPU internal format
        "no_line_action": MoveType.NO_LINE_ACTION,
        # Territory moves (RR-CANON: prefer choose_territory_option)
        "choose_territory_option": MoveType.CHOOSE_TERRITORY_OPTION,
        "process_territory_region": MoveType.CHOOSE_TERRITORY_OPTION,  # Legacy -> canonical
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

    # For all capture types, compute capture_target if not provided
    # The capture target is the midpoint between from and to (the stack being jumped over)
    capture_move_types = (
        MoveType.OVERTAKING_CAPTURE,
        MoveType.CONTINUE_CAPTURE_SEGMENT,
        MoveType.CHAIN_CAPTURE,
    )
    if move_type in capture_move_types and capture_target is None and from_pos and to_pos:
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
    moves: list[Move],
    initial_state: GameState,
) -> tuple[list[Move], GameState]:
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
                            disconnected_regions=terr_move.disconnected_regions,  # Dec 2025: Include geometry for replay
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

            elif phase == GamePhase.FORCED_ELIMINATION:
                # FORCED_ELIMINATION phase - player had no actions this turn but controls stacks
                # Must generate explicit forced_elimination move (RR-CANON-R070/R072)
                # Get available forced elimination moves
                fe_moves = GameEngine._get_forced_elimination_moves(current_state, player)

                if gpu_move.type != MoveType.FORCED_ELIMINATION and fe_moves:
                    # Apply the forced elimination move (typically only one valid target)
                    fe_move = fe_moves[0]
                    move_num += 1
                    phase_move = Move(
                        id=f"move-{move_num}",
                        type=MoveType.FORCED_ELIMINATION,
                        player=player,
                        to=fe_move.to,
                        timestamp=gpu_move.timestamp,
                        think_time=0,
                        move_number=move_num,
                    )
                    expanded_moves.append(phase_move)
                    current_state = GameEngine.apply_move(current_state, phase_move)
                    continue
                elif gpu_move.type != MoveType.FORCED_ELIMINATION:
                    # No forced elimination possible (player lost all stacks) - skip
                    logger.warning(f"FORCED_ELIMINATION phase but no FE moves for player {player}")
                    break

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
) -> dict[str, float]:
    """Load heuristic weights from a profile file."""
    if not os.path.exists(weights_file):
        logger.warning(f"Weights file not found: {weights_file}, using defaults")
        return DEFAULT_WEIGHTS.copy()

    with open(weights_file) as f:
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
        device: torch.device | None = None,
        weights: dict[str, float] | None = None,
        engine_mode: str = "heuristic-only",
        shadow_validation: bool = False,
        shadow_sample_rate: float = 0.05,
        shadow_threshold: float = 0.001,
        lps_victory_rounds: int = 3,
        rings_per_player: int | None = None,
        board_type: str | None = None,
        use_heuristic_selection: bool = False,
        weight_noise: float = 0.0,
        use_policy: bool = False,
        policy_model_path: str | None = None,
        temperature: float = 1.0,
        noise_scale: float = 0.1,
        min_game_length: int = 0,
        random_opening_moves: int = 0,
        temperature_mix: str | None = None,
        canonical_export: bool = False,
        snapshot_interval: int = 0,
        record_policy: bool = False,
        persona_pool: list[str] | None = None,
        per_player_personas: list[str] | None = None,
        temperature_schedule: TemperatureSchedule | None = None,
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.persona_pool = persona_pool
        self.per_player_personas = per_player_personas
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
        self.canonical_export = canonical_export
        self.snapshot_interval = snapshot_interval
        self.record_policy = record_policy
        # Store pending snapshots during batch generation: {game_idx: [(move_num, state_json), ...]}
        self._pending_snapshots: dict[int, list[tuple[int, str]]] = {}
        self._current_batch_persona: str | None = None  # Track current persona for metadata
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
                9: BoardType.HEX8,
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
            record_policy=record_policy,
            per_player_personas=per_player_personas,
        )

        # January 27, 2026 (Phase 2.2): Temperature scheduling integration
        # Store temperature schedule for per-move temperature updates during games
        self.temperature_schedule = temperature_schedule
        if temperature_schedule is not None:
            logger.info(f"Temperature scheduling ENABLED: {type(temperature_schedule).__name__}")
            # Create callback for run_games() to update temperature per step
            self._temperature_callback = lambda move_num: temperature_schedule.get_temperature(move_num)
        else:
            self._temperature_callback = None

        # Store persona pool for per-batch rotation (handled in generate_batch)
        if persona_pool:
            logger.info(f"Persona pool ENABLED: {persona_pool}")
            logger.info("  Weights will rotate through personas per batch")

        # Per-player personas: different weights per player position
        if per_player_personas:
            logger.info(f"Per-player personas ENABLED: {per_player_personas}")
            logger.info("  Each player uses different heuristic weights")

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
        self.wins_by_player = dict.fromkeys(range(1, num_players + 1), 0)
        self.draws = 0

        # Pre-compute initial state for training data compatibility
        # All GPU games start from the same initial state with custom rules applied
        # Hex boards use 2*radius+1 for size: HEX8 radius=4 -> size=9, HEXAGONAL radius=12 -> size=25
        board_type_map = {
            8: BoardType.SQUARE8,
            9: BoardType.HEX8,
            19: BoardType.SQUARE19,
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
        seed: int | None = None,
    ) -> dict[str, Any]:
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
        # For persona pool, rotate through personas per batch for training diversity
        if self.persona_pool:
            # Rotate through personas using batch index
            batch_idx = self.total_games // self.batch_size
            persona_idx = batch_idx % len(self.persona_pool)
            persona_name = self.persona_pool[persona_idx]
            batch_weights = PERSONA_WEIGHTS.get(persona_name, PERSONA_WEIGHTS["balanced"]).copy()
            weights_list = [batch_weights] * self.batch_size
            self._current_batch_persona = persona_name  # Track for record metadata
        elif self.weights is None:
            self._current_batch_persona = None
            weights_list = None  # Uniform random
        else:
            self._current_batch_persona = None
            weights_list = [self.weights] * self.batch_size

        # Clear pending snapshots and create callback if snapshot_interval enabled
        self._pending_snapshots.clear()
        snapshot_callback = None
        if self.snapshot_interval > 0:
            def snapshot_callback(game_idx: int, move_num: int, game_state) -> None:
                """Capture and serialize game state snapshot."""
                try:
                    state_json = game_state.model_dump_json()
                    if game_idx not in self._pending_snapshots:
                        self._pending_snapshots[game_idx] = []
                    self._pending_snapshots[game_idx].append((move_num, state_json))
                except Exception as e:
                    logger.debug(f"Snapshot serialization failed for game {game_idx}: {e}")

        results = self.runner.run_games(
            weights_list=weights_list,
            max_moves=self.max_moves,
            snapshot_interval=self.snapshot_interval,
            snapshot_callback=snapshot_callback,
            temperature_callback=self._temperature_callback,
        )
        elapsed = time.time() - start

        # Log snapshot count if enabled
        if self.snapshot_interval > 0:
            total_snapshots = sum(len(snaps) for snaps in self._pending_snapshots.values())
            logger.debug(f"Captured {total_snapshots} snapshots from {len(self._pending_snapshots)} games")

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
        output_file: str | None = None,
        output_db: str | None = None,
        progress_interval: int = 10,
    ) -> list[dict[str, Any]]:
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
            logger.info("  Note: DB output not supported for GPU games (use import_gpu_selfplay_to_db.py)")

        # Buffered write for better I/O performance (flush every N records)
        WRITE_BUFFER_SIZE = 100  # Records before flush - balances throughput vs data safety
        write_buffer: list[str] = []

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
                batch_records_start = len(all_records)
                batch_filtered_start = self.filtered_short_games

                # Generate batch
                try:
                    results = self.generate_batch(seed=batch_idx * 1000)
                except Exception as exc:
                    logger.exception(
                        "GPU selfplay batch failed",
                        extra={
                            "batch_idx": batch_idx,
                            "num_batches": num_batches,
                            "board_type": self.board_type,
                            "num_players": self.num_players,
                            "batch_size": self.batch_size,
                        },
                    )
                    raise exc

                required_keys = (
                    "move_counts",
                    "move_histories",
                    "winners",
                    "victory_types",
                    "stalemate_tiebreakers",
                )
                for key in required_keys:
                    values = results.get(key)
                    if values is None or not hasattr(values, "__len__"):
                        raise RuntimeError(f"GPU selfplay batch missing '{key}' results")
                    if len(values) < actual_batch:
                        raise RuntimeError(
                            f"GPU selfplay batch '{key}' length {len(values)} < expected {actual_batch}"
                        )

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

                    # Get moves - convert to canonical format if enabled
                    raw_moves = results["move_histories"][i]
                    if self.canonical_export:
                        # Convert each move to canonical format
                        # GPU move history uses: move_type (str), from_pos (tuple), to_pos (tuple), phase (str)
                        canonical_moves = []
                        # Map GPU move type strings to canonical strings
                        gpu_type_map = {
                            "PLACEMENT": "place_ring",
                            "MOVEMENT": "move_stack",
                            "CAPTURE": "overtaking_capture",
                            "LINE_FORMATION": "process_line",
                            "TERRITORY_CLAIM": "process_territory_region",
                            "SKIP": "skip_placement",
                            "NO_ACTION": "no_territory_action",
                            "RECOVERY_SLIDE": "recovery_slide",
                            "NO_PLACEMENT_ACTION": "no_placement_action",
                            "NO_MOVEMENT_ACTION": "no_movement_action",
                            "NO_LINE_ACTION": "no_line_action",
                            "NO_TERRITORY_ACTION": "no_territory_action",
                            "OVERTAKING_CAPTURE": "overtaking_capture",
                            "CONTINUE_CAPTURE_SEGMENT": "continue_capture_segment",
                            "SKIP_CAPTURE": "skip_capture",
                            "SKIP_RECOVERY": "skip_recovery",
                            "FORCED_ELIMINATION": "forced_elimination",
                            "CHOOSE_LINE_OPTION": "choose_line_option",
                            "CHOOSE_TERRITORY_OPTION": "choose_territory_option",
                            "SKIP_PLACEMENT": "skip_placement",
                        }
                        gpu_phase_map = {
                            "RING_PLACEMENT": "ring_placement",
                            "MOVEMENT": "movement",
                            "LINE_PROCESSING": "line_processing",
                            "TERRITORY_PROCESSING": "territory_processing",
                            "END_TURN": "movement",
                            "CAPTURE": "capture",
                            "CHAIN_CAPTURE": "chain_capture",
                            # Recovery moves are recorded in the movement phase.
                            "RECOVERY": "movement",
                            "FORCED_ELIMINATION": "forced_elimination",
                            "GAME_OVER": "game_over",
                        }
                        for move_idx, raw_move in enumerate(raw_moves):
                            move_type_str = raw_move.get("move_type") or "PLACEMENT"
                            phase_str = raw_move.get("phase") or "RING_PLACEMENT"
                            canonical_type = gpu_type_map.get(move_type_str, move_type_str.lower() if move_type_str else "unknown")
                            canonical_move = {
                                "type": canonical_type,
                                "player": raw_move.get("player", 1),
                                "phase": gpu_phase_map.get(phase_str, phase_str.lower() if phase_str else "ring_placement"),
                                "moveNumber": move_idx + 1,  # 1-indexed for NNUE training compatibility
                            }
                            # Add position fields if present (convert tuples to dicts)
                            # For hex boards, convert canvas coords to cube coords
                            from_pos = raw_move.get("from_pos")
                            to_pos = raw_move.get("to_pos")
                            if from_pos and isinstance(from_pos, tuple):
                                canonical_move["from"] = _canvas_to_cube_coords(from_pos[0], from_pos[1], board_type_str)
                            if to_pos and isinstance(to_pos, tuple):
                                canonical_move["to"] = _canvas_to_cube_coords(to_pos[0], to_pos[1], board_type_str)
                                # Add captureTarget for capture moves
                                # December 2025 FIX: Use actual capture_target from move history,
                                # not computed position. Multi-step captures have targets that are
                                # NOT simply "one step back from landing".
                                if canonical_type in ("overtaking_capture", "continue_capture_segment") and from_pos:
                                    # Use capture_target from raw_move if available (correct)
                                    capture_target = raw_move.get("capture_target")
                                    if capture_target and isinstance(capture_target, tuple):
                                        target_y, target_x = capture_target
                                    else:
                                        # Fallback: compute from direction (may be wrong for multi-step)
                                        from_y, from_x = from_pos
                                        to_y, to_x = to_pos
                                        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
                                        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
                                        target_y = to_y - dy
                                        target_x = to_x - dx
                                    canonical_move["captureTarget"] = _canvas_to_cube_coords(target_y, target_x, board_type_str)
                            canonical_moves.append(canonical_move)
                        moves_for_record = canonical_moves

                        # Validate canonical move sequence (sample 5% for performance)
                        if game_idx % 20 == 0:  # Sample every 20th game
                            is_valid, errors = validate_canonical_move_sequence(
                                canonical_moves, self.num_players
                            )
                            _canonical_validation_stats["games_validated"] += 1
                            if is_valid:
                                _canonical_validation_stats["games_valid"] += 1
                            else:
                                _canonical_validation_stats["games_invalid"] += 1
                                _canonical_validation_stats["total_errors"] += len(errors)
                    else:
                        # Add moveNumber to raw moves for training compatibility
                        moves_for_record = []
                        for move_idx, raw_move in enumerate(raw_moves):
                            move_with_number = dict(raw_move)
                            move_with_number["moveNumber"] = move_idx + 1
                            moves_for_record.append(move_with_number)

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
                        "engine_mode": self.engine_mode,
                        "opponent_type": "selfplay",
                        "player_types": ["gpu_batch"] * self.num_players,
                        "batch_id": batch_idx,
                        # === Training data (required for NPZ export) ===
                        "moves": moves_for_record,
                        "initial_state": self._initial_state_json,
                        # === Timing metadata ===
                        "timestamp": datetime.now().isoformat(),
                        "created_at": datetime.now().isoformat(),
                        # === Source tracking ===
                        "source": "run_gpu_selfplay.py",
                        "device": str(self.device),
                        # === Canonical export flag ===
                        "canonical_format": self.canonical_export,
                        # === Training metadata ===
                        "metadata": {
                            "persona": self._current_batch_persona,
                            "persona_pool": list(self.persona_pool) if self.persona_pool else None,
                        },
                    }

                    # Add trajectory snapshots if captured
                    if i in self._pending_snapshots:
                        snapshots = self._pending_snapshots[i]
                        record["trajectory_snapshots"] = [
                            {"move_number": move_num, "state": state_json}
                            for move_num, state_json in snapshots
                        ]

                    # Add policy data if captured
                    if self.record_policy:
                        policy_data = self.runner.pop_policy_data(i)
                        if policy_data:
                            record["move_policies"] = [
                                {"move_number": move_num, "policy": policy_dict}
                                for move_num, policy_dict in policy_data
                            ]

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

                batch_written = len(all_records) - batch_records_start
                batch_filtered = self.filtered_short_games - batch_filtered_start
                if batch_written == 0:
                    logger.warning(
                        "GPU selfplay batch produced zero records",
                        extra={
                            "batch_idx": batch_idx + 1,
                            "num_batches": num_batches,
                            "board_type": board_type_str,
                            "num_players": self.num_players,
                            "filtered_short_games": batch_filtered,
                        },
                    )
                elif batch_filtered > 0:
                    logger.info(
                        f"  Batch {batch_idx + 1}/{num_batches}: "
                        f"wrote {batch_written}/{actual_batch} games "
                        f"(filtered {batch_filtered})"
                    )

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

        if output_file and os.path.exists(output_file):
            if os.path.getsize(output_file) == 0:
                raise RuntimeError(
                    f"GPU selfplay output is empty: {output_file}. "
                    "Check for batch failures or min_game_length filtering."
                )

        if len(all_records) == 0:
            raise RuntimeError(
                "GPU selfplay produced zero records. "
                "Check for batch failures, min_game_length filtering, or upstream errors."
            )

        return all_records

    def get_statistics(self) -> dict[str, Any]:
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
            "canonical_export": self.canonical_export,
        }

        # Add canonical validation stats if enabled
        if self.canonical_export and _canonical_validation_stats["games_validated"] > 0:
            valid_rate = _canonical_validation_stats["games_valid"] / _canonical_validation_stats["games_validated"]
            stats["canonical_validation"] = {
                "games_validated": _canonical_validation_stats["games_validated"],
                "games_valid": _canonical_validation_stats["games_valid"],
                "games_invalid": _canonical_validation_stats["games_invalid"],
                "valid_rate": valid_rate,
                "total_errors": _canonical_validation_stats["total_errors"],
            }
            if valid_rate < 1.0:
                logger.warning(f"Canonical validation: {valid_rate:.1%} valid ({_canonical_validation_stats['games_invalid']} invalid)")
            else:
                logger.info(f"Canonical validation: 100% valid ({_canonical_validation_stats['games_validated']} sampled)")

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
# Quality Tier Selfplay (Hybrid / MaxN)
# =============================================================================


def run_quality_tier_selfplay(
    board_type: str,
    num_players: int,
    num_games: int,
    output_dir: str,
    quality_tier: str,
    policy_model_path: str | None = None,
    hybrid_depth: int = 2,
    hybrid_top_k: int = 8,
    max_moves: int = 500,
    seed: int = 42,
    output_db: str | None = None,
    canonical_export: bool = False,
) -> dict[str, Any]:
    """Run quality-tier selfplay using Hybrid or MaxN AI.

    This runs games sequentially (not parallel) to achieve higher quality
    move selection through tree search.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players
        num_games: Number of games to generate
        output_dir: Output directory
        quality_tier: "hybrid" or "maxn"
        policy_model_path: Path to policy model for hybrid tier
        hybrid_depth: Search depth for hybrid tier
        hybrid_top_k: Number of top moves to search in hybrid tier
        max_moves: Maximum moves per game
        seed: Random seed
        output_db: Optional SQLite database for game storage
        canonical_export: Export in canonical format

    Returns:
        Statistics dict
    """
    import random
    from datetime import datetime

    from app.models import AIConfig, BoardType, GameStatus, GamePhase
    from app.training.initial_state import create_initial_state
    from app.game_engine import GameEngine

    def is_game_over(state) -> bool:
        """Check if game is over."""
        return (
            state.game_status == GameStatus.COMPLETED or
            state.winner is not None or
            state.current_phase == GamePhase.GAME_OVER
        )

    def is_no_action_move(move) -> bool:
        """Check if a move is a no-action/skip move."""
        if move is None:
            return False
        move_type = move.type.value if hasattr(move.type, 'value') else str(move.type)
        return move_type in (
            "no_placement_action", "no_movement_action", "no_line_action",
            "no_territory_action", "skip_territory_processing", "skip_placement",
            "skip_capture", "NO_PLACEMENT_ACTION", "NO_MOVEMENT_ACTION",
            "NO_LINE_ACTION", "NO_TERRITORY_ACTION"
        )

    def determine_winner_by_scoring(state) -> int | None:
        """Determine winner by scoring when stalemate detected.

        Uses the canonical tiebreaker ladder:
        1. Most territory (collapsed spaces)
        2. Most eliminated rings
        3. Most markers
        4. Last player to make a real move
        """
        scores = {}
        for player in state.players:
            pid = player.player_number

            # Count territory (collapsed spaces)
            territory = sum(1 for owner in state.board.collapsed_spaces.values()
                          if owner == pid)

            # Eliminated rings
            eliminated = getattr(player, 'eliminated_rings', 0)

            # Markers
            markers = sum(1 for m in state.board.markers.values()
                         if getattr(m, 'player', None) == pid)

            scores[pid] = (territory, eliminated, markers)

        # Sort by score (higher is better)
        sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_players:
            return sorted_players[0][0]
        return None

    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Create AI instances based on quality tier
    ais = []
    for player_num in range(1, num_players + 1):
        config = AIConfig(difficulty=7)  # High difficulty for quality play

        if quality_tier == "hybrid":
            from app.ai.hybrid_tree_policy_ai import HybridTreePolicyAI
            ai = HybridTreePolicyAI(
                player_number=player_num,
                config=config,
                top_k=hybrid_top_k,
                search_depth=hybrid_depth,
            )
            if policy_model_path:
                ai.load_policy_model(policy_model_path)
            else:
                # Try to load default model
                default_path = os.path.join(
                    os.path.dirname(__file__), "..",
                    "models", "nnue", f"nnue_policy_{board_type}_{num_players}p_v3.pt"
                )
                if os.path.exists(default_path):
                    ai.load_policy_model(default_path)
            ais.append(ai)

        elif quality_tier == "maxn":
            from app.ai.maxn_ai import MaxNAI
            ai = MaxNAI(player_number=player_num, config=config)
            ais.append(ai)

        else:
            raise ValueError(f"Unknown quality tier: {quality_tier}")

    logger.info(f"Quality tier selfplay: {quality_tier}")
    logger.info(f"  Board: {board_type}, Players: {num_players}")
    logger.info(f"  Games: {num_games}, Max moves: {max_moves}")
    if quality_tier == "hybrid":
        logger.info(f"  Hybrid config: depth={hybrid_depth}, top_k={hybrid_top_k}")

    # Game engine for rules
    engine = GameEngine()

    # Output files
    games_file = os.path.join(output_dir, "games.jsonl")
    stats = {
        "total_games": 0,
        "completed_games": 0,
        "wins_by_player": {p: 0 for p in range(1, num_players + 1)},
        "draws": 0,
        "total_moves": 0,
        "avg_game_length": 0,
        "quality_tier": quality_tier,
    }

    start_time = time.time()
    game_records = []

    # Stalemate safety net: detect games stuck in no-action loops.
    # With the fix to phase_machine.py (SKIP_* moves now don't prevent forced
    # elimination), this should rarely trigger. If it does, it indicates a bug.
    # Threshold: 16 consecutive no-action moves across all players (2 full turns)
    STALEMATE_THRESHOLD = 16 * num_players

    for game_idx in range(num_games):
        # Create initial state
        state = create_initial_state(board_type, num_players)
        moves_made = []
        move_count = 0
        consecutive_no_action = 0  # Track consecutive no-action moves

        game_start = time.time()
        stalemate_detected = False

        while not is_game_over(state) and move_count < max_moves:
            current_player = state.current_player
            ai = ais[current_player - 1]

            # Get AI move
            move = ai.select_move(state)
            if move is None:
                break

            # Track no-action moves for stalemate detection
            if is_no_action_move(move):
                consecutive_no_action += 1
                if consecutive_no_action >= STALEMATE_THRESHOLD:
                    stalemate_detected = True
                    break
            else:
                consecutive_no_action = 0

            # Apply move
            state = engine.apply_move(state, move)
            moves_made.append(move)
            move_count += 1

        game_elapsed = time.time() - game_start
        stats["total_games"] += 1
        stats["total_moves"] += move_count

        # Determine winner
        winner = None
        game_over = is_game_over(state) or stalemate_detected

        if stalemate_detected:
            # Stalemate safety net triggered - this indicates a potential bug
            # since the phase_machine fix should prevent infinite no-action loops
            logger.warning(
                f"Stalemate safety net triggered in game {game_idx}: "
                f"{consecutive_no_action} consecutive no-action moves at move {move_count}. "
                f"This may indicate a bug in forced elimination logic."
            )
            winner = determine_winner_by_scoring(state)
            stats["completed_games"] += 1
            stats["stalemates"] = stats.get("stalemates", 0) + 1
        elif game_over:
            stats["completed_games"] += 1
            # Find winner (player with most points or last standing)
            if state.winner:
                winner = state.winner
            else:
                # Check eliminated players
                active_players = [p for p in state.players if not p.eliminated]
                if len(active_players) == 1:
                    winner = active_players[0].player_number
                elif len(active_players) > 1:
                    # Multiple players still active at game over - use scoring
                    winner = determine_winner_by_scoring(state)

        if winner:
            stats["wins_by_player"][winner] = stats["wins_by_player"].get(winner, 0) + 1
        else:
            stats["draws"] += 1

        # Create game record
        record = {
            "game_id": f"quality_{quality_tier}_{game_idx}",
            "board_type": board_type,
            "num_players": num_players,
            "quality_tier": quality_tier,
            "winner": winner,
            "move_count": move_count,
            "game_over": game_over,
            "elapsed_seconds": game_elapsed,
            "timestamp": datetime.now().isoformat(),
            "moves": [
                {
                    "type": m.type.value if hasattr(m.type, 'value') else str(m.type),
                    "player": m.player,
                    "to": {"x": m.to.x, "y": m.to.y} if m.to else None,
                    "from": {"x": m.from_pos.x, "y": m.from_pos.y} if m.from_pos else None,
                }
                for m in moves_made
            ],
        }
        game_records.append(record)

        # Log progress every 10 games
        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed
            logger.info(
                f"Game {game_idx + 1}/{num_games}: {move_count} moves, "
                f"winner=P{winner}, {games_per_sec:.2f} games/sec"
            )

    # Write games to JSONL
    with open(games_file, "w") as f:
        for record in game_records:
            f.write(json.dumps(record) + "\n")

    # Calculate final stats
    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["games_per_second"] = num_games / elapsed if elapsed > 0 else 0
    stats["avg_game_length"] = stats["total_moves"] / num_games if num_games > 0 else 0

    # Write stats
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("")
    logger.info(f"Quality tier selfplay complete: {num_games} games in {elapsed:.1f}s")
    logger.info(f"  Speed: {stats['games_per_second']:.3f} games/sec")
    logger.info(f"  Avg length: {stats['avg_game_length']:.1f} moves")
    logger.info(f"  Output: {games_file}")

    # Emit SELFPLAY_COMPLETE for pipeline automation (Phase 3A.2: Dec 2025)
    _emit_gpu_selfplay_complete(
        board_type=board_type,
        num_players=num_players,
        games_completed=num_games,
        total_samples=stats["total_moves"],
        output_dir=output_dir,
        db_path=None,  # Quality tier doesn't use DB
        duration_seconds=elapsed,
    )

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
    weights: dict[str, float] | None = None,
    engine_mode: str = "heuristic-only",
    seed: int = 42,
    shadow_validation: bool = False,
    shadow_sample_rate: float = 0.05,
    shadow_threshold: float = 0.001,
    lps_victory_rounds: int = 3,
    rings_per_player: int | None = None,
    output_db: str | None = None,
    use_heuristic_selection: bool = False,
    weight_noise: float = 0.0,
    use_policy: bool = False,
    policy_model_path: str | None = None,
    temperature: float = 1.0,
    noise_scale: float = 0.1,
    min_game_length: int = 0,
    random_opening_moves: int = 0,
    temperature_mix: str | None = None,
    canonical_export: bool = False,
    snapshot_interval: int = 0,
    record_policy: bool = False,
    persona_pool: list[str] | None = None,
    per_player_personas: list[str] | None = None,
    temperature_schedule: TemperatureSchedule | None = None,
) -> dict[str, Any]:
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
        canonical_export: Output moves in canonical format (with phase/type strings)
        snapshot_interval: Capture GameState snapshots every N moves (0 = disabled)
        record_policy: Record move policy distributions for training (requires heuristic mode)
        per_player_personas: List of persona names per player position (e.g., ['aggressive', 'defensive'])

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
            # S-invariant guarantees termination within ~164 turns for square8 2p
            # Each turn has multiple phase actions, so allow ~6x headroom
            ("square8", 2): 1000,
            ("square8", 3): 1200,
            ("square8", 4): 1600,
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
        logger.info("Move selection: policy-guided" + (f" ({policy_model_path})" if policy_model_path else ""))
    else:
        logger.info(f"Move selection: {'heuristic-based' if use_heuristic_selection else 'center-bias random'}")
    logger.info(f"Weight noise: {weight_noise:.1%}" if weight_noise > 0 else "Weight noise: disabled")
    if min_game_length > 0:
        logger.info(f"Min game length filter: {min_game_length} moves")
    if random_opening_moves > 0:
        logger.info(f"Random opening moves: {random_opening_moves}")
    if temperature_mix:
        logger.info(f"Temperature mixing: {temperature_mix} (temps: [0.5, 1.0, 2.0, 4.0])")
    if canonical_export:
        logger.info("Canonical export: ENABLED (phase/type strings)")
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
        canonical_export=canonical_export,
        snapshot_interval=snapshot_interval,
        record_policy=record_policy,
        persona_pool=persona_pool,
        per_player_personas=per_player_personas,
        temperature_schedule=temperature_schedule,
    )

    if temperature_schedule is not None:
        logger.info(f"Temperature schedule: {type(temperature_schedule).__name__} (per-move decay)")

    if persona_pool:
        logger.info(f"Persona pool: {persona_pool} (full 45-weight profiles)")

    if per_player_personas:
        logger.info(f"Per-player personas: {per_player_personas} (full 45-weight profiles)")

    if record_policy:
        if not use_heuristic_selection:
            logger.warning("Policy recording requires heuristic mode; enabling use_heuristic_selection")
        logger.info("Policy recording: ENABLED (move scores/probabilities will be recorded)")

    if snapshot_interval > 0:
        logger.info(f"Snapshot interval: {snapshot_interval} moves (trajectory capture enabled)")

    # Generate games - use unique filename per config to avoid lock contention
    games_file = os.path.join(output_dir, f"games_{board_type}_{num_players}p_{os.getpid()}.jsonl")
    records = generator.generate_games(
        num_games=num_games,
        output_file=games_file,
        progress_interval=10,
    )

    # Get and save statistics
    stats = generator.get_statistics()
    stats["recorded_games"] = len(records)
    stats["timestamp"] = datetime.now().isoformat()
    stats["seed"] = seed

    if stats["recorded_games"] < num_games:
        logger.warning(
            f"Recorded {stats['recorded_games']}/{num_games} games. "
            f"Filtered short games: {stats['filtered_short_games']}."
        )

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

    # Emit SELFPLAY_COMPLETE for pipeline automation (Phase 3A.2: Dec 2025)
    _emit_gpu_selfplay_complete(
        board_type=board_type,
        num_players=num_players,
        games_completed=stats["total_games"],
        total_samples=stats["total_moves"],
        output_dir=output_dir,
        db_path=output_db,
        duration_seconds=stats["total_time_seconds"],
    )

    return stats


def main():
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="GPU-accelerated self-play data generation",
        include_gpu=True,
        include_ramdrive=True,
    )

    # Add GPU-specific arguments
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default=None,
        help="Output SQLite DB path for canonical game storage (default: output-dir/games.db)",
    )
    parser.add_argument(
        "--profile",
        dest="weights_profile",
        type=str,
        help="Profile name in weights file (alias for --weights-profile)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run GPU benchmark",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.001,
        help="Max divergence rate before error (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--rings-per-player",
        type=int,
        default=None,
        help="Starting rings per player (default: board default - 18/72/96)",
    )
    parser.add_argument(
        "--use-heuristic",
        action="store_true",
        help="Use heuristic-based move selection instead of center-bias random",
    )
    parser.add_argument(
        "--weight-noise",
        type=float,
        default=0.0,
        help="Multiplicative noise (0.0-1.0) for heuristic weights diversity",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=["balanced", "aggressive", "territorial", "defensive", "mixed"],
        help="Use persona-based weights (full 45-weight profiles). 'mixed' rotates through all personas.",
    )
    parser.add_argument(
        "--per-player-personas",
        action="store_true",
        help="Assign different personas to different players (enables varied strategy matchups)",
    )
    parser.add_argument(
        "--matchup",
        type=str,
        default=None,
        help="Use a predefined matchup configuration (e.g., 'aggressive_vs_defensive', 'balanced_mirror'). "
             "See ParallelGameRunner.TRAINING_MATCHUPS for available matchups.",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        help="Comma-separated list of personas to use (e.g., 'balanced,aggressive,defensive'). "
             "Each game randomly samples a persona from this pool. Uses full 45-weight profiles.",
    )
    parser.add_argument(
        "--use-policy",
        action="store_true",
        help="Use policy network for move selection",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default=None,
        help="Path to custom policy model (default: auto-detect based on board type)",
    )
    parser.add_argument(
        "--quality-tier",
        type=str,
        default=None,
        choices=["bulk", "hybrid", "maxn"],
        help="Quality tier for selfplay: bulk (fast GPU parallel), hybrid (policy+tree search), "
             "maxn (full MaxN tree search). Overrides --use-policy when set.",
    )
    parser.add_argument(
        "--hybrid-depth",
        type=int,
        default=2,
        help="Search depth for hybrid quality tier (default: 2)",
    )
    parser.add_argument(
        "--hybrid-top-k",
        type=int,
        default=8,
        help="Number of top policy moves to search in hybrid tier (default: 8)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Scale of noise added to move scores (default: 0.1)",
    )
    parser.add_argument(
        "--temperature-mix",
        type=str,
        default=None,
        choices=["uniform", "weighted", "random"],
        help="Mix different temperature levels per game for training diversity",
    )
    # January 27, 2026 (Phase 2.2): Per-move temperature scheduling
    parser.add_argument(
        "--temperature-schedule",
        type=str,
        default=None,
        choices=["linear", "adaptive", "elo_adaptive", "cosine", "curriculum"],
        help="Per-move temperature schedule (decays temperature during game). "
             "linear: 1.0->0.3 over moves 15-60. "
             "adaptive: Based on position complexity. "
             "elo_adaptive: Based on model Elo rating. "
             "cosine: Cosine annealing. "
             "curriculum: Based on training progress.",
    )
    # Additional ramdrive args beyond base parser
    parser.add_argument("--ram-storage", action="store_true", help="Use ramdrive storage")
    parser.add_argument("--sync-target", type=str, help="Target directory for ramdrive sync")
    parser.add_argument("--skip-resource-check", action="store_true",
                       help="Skip resource limit checks (use when resources are known to be available)")
    # Canonical export is now enabled by default (Dec 2025)
    # Use --no-canonical-export to disable if needed for debugging
    parser.set_defaults(canonical_export=True)
    parser.add_argument(
        "--no-canonical-export",
        action="store_false",
        dest="canonical_export",
        help="Disable canonical export (use raw GPU coordinates - not recommended)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="Capture GameState snapshots every N moves per game for NNUE training "
             "trajectories. 0 = disabled (default). Recommended: 15-20 for good coverage.",
    )
    parser.add_argument(
        "--snapshot-db",
        type=str,
        default=None,
        help="SQLite database path to store trajectory snapshots. If not specified, "
             "snapshots are stored in-memory with game records (increases memory usage).",
    )
    parser.add_argument(
        "--record-policy",
        action="store_true",
        help="Record move policy distributions (candidate moves + scores/probabilities) "
             "for each move. Requires --engine heuristic-only. Enables policy training.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously in an infinite loop, restarting after each batch of games. "
             "Eliminates GPU idle cycles between runs.",
    )

    parsed = parser.parse_args()
    engine_mode = parsed.engine_mode

    # Jan 2026: GPU-accelerated modes that use neural network policy guidance
    # These modes use heuristic-only as the base and add --use-policy for NN inference
    GPU_ACCELERATED_MODES = {
        "gumbel-mcts", "gumbel", "gpu-gumbel", "mcts", "policy-only", "nn-descent",
        "nn-minimax", "gnn", "hybrid", "gmo", "ebmo", "ig-gmo", "cage",
    }

    if engine_mode == "random":
        engine_mode = "random-only"
    elif engine_mode in ("policy_only", "policy-only"):
        engine_mode = "policy_only"
    elif engine_mode in GPU_ACCELERATED_MODES:
        # GPU-accelerated modes use heuristic base + policy network
        logger.info(f"GPU-accelerated mode '{engine_mode}' - using heuristic base with policy network")
        if not parsed.use_policy:
            parsed.use_policy = True
            logger.info("Automatically enabling --use-policy for GPU-accelerated mode")
        engine_mode = "heuristic-only"
    elif engine_mode not in ("random-only", "heuristic-only", "nnue-guided"):
        logger.warning(
            "Unsupported engine_mode '%s' for GPU selfplay; falling back to heuristic-only",
            engine_mode,
        )
        engine_mode = "heuristic-only"

    # Create SelfplayConfig from parsed args
    selfplay_config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        batch_size=parsed.batch_size,
        output_dir=parsed.output_dir or "data/selfplay/gpu",
        seed=parsed.seed or 42,
        temperature=parsed.temperature,
        weights_file=parsed.weights_file,
        weights_profile=parsed.weights_profile,
        shadow_validation=parsed.shadow_validation,
        shadow_sample_rate=parsed.shadow_sample_rate,
        lps_victory_rounds=parsed.lps_victory_rounds,
        min_game_length=parsed.min_game_length,
        random_opening_moves=parsed.random_opening_moves,
        use_ramdrive=parsed.use_ramdrive,
        ramdrive_path=parsed.ramdrive_path,
        sync_interval=parsed.sync_interval,
        source="run_gpu_selfplay.py",
        extra_options={
            "max_moves": parsed.max_moves,
            "output_db": parsed.output_db,
            "benchmark_only": parsed.benchmark_only,
            "shadow_threshold": parsed.shadow_threshold,
            "rings_per_player": parsed.rings_per_player,
            "use_heuristic": parsed.use_heuristic,
            "weight_noise": parsed.weight_noise,
            "use_policy": parsed.use_policy,
            "policy_model": parsed.policy_model,
            "noise_scale": parsed.noise_scale,
            "temperature_mix": parsed.temperature_mix,
            "engine_mode": engine_mode,
            "ram_storage": getattr(parsed, "ram_storage", False),
            "sync_target": getattr(parsed, "sync_target", None),
            "skip_resource_check": getattr(parsed, "skip_resource_check", False),
            "canonical_export": getattr(parsed, "canonical_export", False),
            "snapshot_interval": getattr(parsed, "snapshot_interval", 0),
            "snapshot_db": getattr(parsed, "snapshot_db", None),
            "record_policy": getattr(parsed, "record_policy", False),
            "persona": getattr(parsed, "persona", None),
            "personas": getattr(parsed, "personas", None),
            "per_player_personas": getattr(parsed, "per_player_personas", False),
            "matchup": getattr(parsed, "matchup", None),
            "continuous": getattr(parsed, "continuous", False),
            "quality_tier": getattr(parsed, "quality_tier", None),
            "hybrid_depth": getattr(parsed, "hybrid_depth", 2),
            "hybrid_top_k": getattr(parsed, "hybrid_top_k", 8),
        },
    )

    # Create backward-compatible args object
    args = type("Args", (), {
        "board": selfplay_config.board_type,
        "num_players": selfplay_config.num_players,
        "num_games": selfplay_config.num_games,
        "batch_size": selfplay_config.batch_size,
        "output_dir": selfplay_config.output_dir,
        "seed": selfplay_config.seed,
        "temperature": selfplay_config.temperature,
        "weights_file": selfplay_config.weights_file,
        "weights_profile": selfplay_config.weights_profile,
        "shadow_validation": selfplay_config.shadow_validation,
        "shadow_sample_rate": selfplay_config.shadow_sample_rate,
        "lps_victory_rounds": selfplay_config.lps_victory_rounds,
        "min_game_length": selfplay_config.min_game_length,
        "random_opening_moves": selfplay_config.random_opening_moves,
        "max_moves": selfplay_config.extra_options["max_moves"],
        "output_db": selfplay_config.extra_options["output_db"],
        "benchmark_only": selfplay_config.extra_options["benchmark_only"],
        "shadow_threshold": selfplay_config.extra_options["shadow_threshold"],
        "rings_per_player": selfplay_config.extra_options["rings_per_player"],
        "use_heuristic": selfplay_config.extra_options["use_heuristic"],
        "weight_noise": selfplay_config.extra_options["weight_noise"],
        "use_policy": selfplay_config.extra_options["use_policy"],
        "policy_model": selfplay_config.extra_options["policy_model"],
        "noise_scale": selfplay_config.extra_options["noise_scale"],
        "temperature_mix": selfplay_config.extra_options["temperature_mix"],
        "engine_mode": selfplay_config.extra_options["engine_mode"],
        "ram_storage": selfplay_config.extra_options["ram_storage"],
        "sync_target": selfplay_config.extra_options["sync_target"],
        "sync_interval": selfplay_config.sync_interval,
        "skip_resource_check": selfplay_config.extra_options["skip_resource_check"],
        "canonical_export": selfplay_config.extra_options["canonical_export"],
        "snapshot_interval": selfplay_config.extra_options["snapshot_interval"],
        "snapshot_db": selfplay_config.extra_options["snapshot_db"],
        "record_policy": selfplay_config.extra_options["record_policy"],
        "persona": selfplay_config.extra_options["persona"],
        "personas": selfplay_config.extra_options["personas"],
        "per_player_personas": selfplay_config.extra_options["per_player_personas"],
        "matchup": selfplay_config.extra_options["matchup"],
        "continuous": selfplay_config.extra_options["continuous"],
        "quality_tier": selfplay_config.extra_options["quality_tier"],
        "hybrid_depth": selfplay_config.extra_options["hybrid_depth"],
        "hybrid_top_k": selfplay_config.extra_options["hybrid_top_k"],
    })()

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

    # Load weights - matchup/persona takes priority over file
    # Priority: --matchup > --personas > --persona mixed > --persona <single> > --weights-file
    weights = None
    persona_pool = None
    per_player_personas_list = None  # List of persona names per player position
    persona = getattr(args, "persona", None)
    personas_arg = getattr(args, "personas", None)
    per_player_personas = getattr(args, "per_player_personas", False)
    matchup = getattr(args, "matchup", None)

    if matchup:
        # --matchup takes highest priority: use predefined matchup configuration
        try:
            per_player_personas_list = ParallelGameRunner.get_training_matchup(matchup)
            logger.info(f"Using matchup '{matchup}': {per_player_personas_list}")
            logger.info("  Per-player personas ENABLED (different weights per player position)")
            if not args.use_heuristic:
                logger.info("Matchup mode enables heuristic-based move selection")
                args.use_heuristic = True
        except KeyError:
            available = list(ParallelGameRunner.get_all_training_matchups())
            logger.error(f"Unknown matchup '{matchup}'. Available: {available}")
            sys.exit(1)
    elif personas_arg:
        # --personas takes highest priority: comma-separated list of personas
        persona_pool = [p.strip() for p in personas_arg.split(",") if p.strip()]
        # Validate all personas exist
        valid_personas = set(ALL_PERSONAS)
        invalid = [p for p in persona_pool if p not in valid_personas]
        if invalid:
            logger.warning(f"Unknown personas ignored: {invalid}. Valid: {ALL_PERSONAS}")
            persona_pool = [p for p in persona_pool if p in valid_personas]
        if persona_pool:
            weights = None  # Let ParallelGameRunner handle per-game weights from pool
            logger.info(f"Using custom persona pool: {persona_pool}")
            logger.info("  Each game samples a random persona (full 45-weight profiles)")
            if not args.use_heuristic:
                logger.info("Persona mode enables heuristic-based move selection")
                args.use_heuristic = True
    elif persona:
        if persona == "mixed":
            # For mixed mode, use persona_pool for per-game variety (full 45-weight profiles)
            persona_pool = ALL_PERSONAS.copy()
            weights = None  # Let ParallelGameRunner handle per-game weights from pool
            logger.info(f"Using MIXED persona mode with persona_pool: {persona_pool}")
            logger.info("  Each game samples a random persona (full 45-weight profiles)")
        else:
            # Single persona - use its full weights
            weights = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["balanced"]).copy()
            logger.info(f"Using {persona} persona weights ({len(weights)} heuristic weights)")
        # Persona implies heuristic selection
        if not args.use_heuristic:
            logger.info("Persona mode enables heuristic-based move selection")
            args.use_heuristic = True
    elif args.weights_file and args.weights_profile:
        weights = load_weights_from_profile(args.weights_file, args.weights_profile)
        logger.info(f"Loaded weights from {args.weights_file}:{args.weights_profile}")

    if per_player_personas and persona:
        logger.info("Per-player personas: ENABLED (each player uses different weights)")

    # Resource guard: Check disk/memory/GPU before starting (80% limits)
    # Also import graceful degradation functions for dynamic resource management
    try:
        from app.utils.resource_guard import (
            LIMITS,
            OperationPriority,
            check_disk_space,
            check_gpu_memory,
            check_memory,
            get_degradation_level,
            get_resource_status,
            should_proceed_with_priority,
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
        if degradation >= 4 and not args.skip_resource_check:  # CRITICAL - resources at/above limits
            logger.error("Resources at critical levels, aborting selfplay")
            sys.exit(1)
        elif degradation >= 4:
            logger.warning("Resources at critical levels but --skip-resource-check set, proceeding anyway")
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
        iteration = 0
        continuous = getattr(args, "continuous", False)
        if continuous:
            logger.info("CONTINUOUS MODE: Will restart automatically after each batch")

        # Check for quality tier mode
        quality_tier = args.quality_tier

        # January 27, 2026 (Phase 2.2): Create temperature scheduler from CLI arg
        temperature_schedule = None
        if hasattr(args, 'temperature_schedule') and args.temperature_schedule:
            schedule_type = args.temperature_schedule
            if schedule_type == "linear":
                # Default linear decay: 1.0 -> 0.3 over moves 15-60
                temperature_schedule = LinearDecaySchedule(
                    initial_temp=1.0,
                    final_temp=0.3,
                    decay_start=15,
                    decay_end=60,
                )
            else:
                # Use create_scheduler for other schedule types
                temperature_schedule = create_scheduler(schedule_type)
            logger.info(f"Using temperature schedule: {schedule_type}")

        while True:
            iteration += 1
            current_seed = args.seed + (iteration - 1) * 10000 if args.seed else None

            if continuous and iteration > 1:
                logger.info("")
                logger.info(f"========== CONTINUOUS ITERATION {iteration} ==========")
                logger.info("")

            # Use quality tier selfplay if specified (hybrid or maxn)
            if quality_tier and quality_tier in ("hybrid", "maxn"):
                run_quality_tier_selfplay(
                    board_type=args.board,
                    num_players=args.num_players,
                    num_games=args.num_games,
                    output_dir=output_dir,
                    quality_tier=quality_tier,
                    policy_model_path=args.policy_model,
                    hybrid_depth=args.hybrid_depth,
                    hybrid_top_k=args.hybrid_top_k,
                    max_moves=args.max_moves,
                    seed=current_seed or 42,
                    output_db=args.output_db,
                    canonical_export=args.canonical_export,
                )
            else:
                # Standard GPU parallel selfplay (bulk tier or default)
                run_gpu_selfplay(
                    board_type=args.board,
                    num_players=args.num_players,
                    num_games=args.num_games,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    max_moves=args.max_moves,
                    weights=weights,
                    engine_mode=args.engine_mode,
                    seed=current_seed,
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
                    canonical_export=args.canonical_export,
                    snapshot_interval=args.snapshot_interval,
                    record_policy=args.record_policy,
                    persona_pool=persona_pool,
                    per_player_personas=per_player_personas_list,
                    temperature_schedule=temperature_schedule,
                )

            if not continuous:
                break

            # Brief pause before next iteration to allow resource checks
            time.sleep(2)
    finally:
        # Stop ramdrive syncer and perform final sync
        if syncer:
            logger.info("Stopping ramdrive syncer and performing final sync...")
            syncer.stop(final_sync=True)
            logger.info(f"Ramdrive sync stats: {syncer.stats}")

        # Record task completion for duration learning (safe version handles errors)
        if HAS_COORDINATION and task_id:
            actual_duration = time.time() - start_time
            if record_task_completion_safe(task_id, "gpu_selfplay", actual_duration):
                logger.info("Recorded task completion for duration learning")


if __name__ == "__main__":
    main()
