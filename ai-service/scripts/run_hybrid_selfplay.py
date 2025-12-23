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

import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add app/ to path (must be early for app.* imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ramdrive utilities for high-speed I/O
from app.utils.ramdrive import RamdriveSyncer, get_config_from_args, get_games_directory

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        LIMITS as RESOURCE_LIMITS,
        get_disk_usage as unified_get_disk_usage,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_disk_usage = None
    RESOURCE_LIMITS = None

# Disk monitoring thresholds - raised to 85% as of 2025-12-17 (238GB free is plenty)
DISK_WARNING_THRESHOLD = 75  # Pause selfplay
DISK_CRITICAL_THRESHOLD = 85  # Abort selfplay

# Timeout handling constants
DEFAULT_GAME_TIMEOUT = 300  # 5 minutes per game
DEFAULT_SESSION_TIMEOUT = 14400  # 4 hours for entire session
HEARTBEAT_INTERVAL = 60  # Log progress every 60 seconds

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
) -> dict[str, float]:
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

    with open(weights_file) as f:
        data = json.load(f)

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        logging.getLogger(__name__).warning(
            f"Profile '{profile_name}' not found in {weights_file}, using defaults"
        )
        return DEFAULT_WEIGHTS.copy()

    return profiles[profile_name].get("weights", DEFAULT_WEIGHTS.copy())


def get_disk_usage_percent(path: str = "/") -> int:
    """Get disk usage percentage for the filesystem containing path.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.
    """
    # Use unified utilities when available
    if HAS_RESOURCE_GUARD and unified_get_disk_usage is not None:
        try:
            percent, _, _ = unified_get_disk_usage(path)
            return int(percent)
        except Exception:
            pass  # Fall through to original implementation

    # Fallback to original implementation
    try:
        total, used, _free = shutil.disk_usage(path)
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


from app.db import (
    ParityValidationError,
    get_or_create_db,
    record_completed_game_with_parity_check,
)
from app.training.selfplay_config import SelfplayConfig, create_argument_parser

# Import shared victory type module
from app.utils.victory_type import derive_victory_type

# Import coordination for task limits and duration tracking
try:
    from app.coordination import (
        TaskCoordinator,
        TaskType,
        record_task_completion,
        register_running_task,
    )
    from app.coordination.helpers import can_spawn_safe as can_spawn
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    TaskCoordinator = None
    TaskType = None
    can_spawn = None

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_hybrid_selfplay")


def run_hybrid_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 100,
    output_dir: str = "data/selfplay/hybrid",
    max_moves: int | None = None,  # Auto-calculated based on board type
    seed: int = 42,
    use_numba: bool = True,
    engine_mode: str = "heuristic-only",
    p2_engine_mode: str | None = None,  # Different engine for player 2 (asymmetric matches)
    p3_engine_mode: str | None = None,  # Different engine for player 3 (3-4 player games)
    p4_engine_mode: str | None = None,  # Different engine for player 4 (4 player games)
    weights: dict[str, float] | None = None,
    mix_ratio: float = 0.8,
    record_db: str | None = None,
    lean_db: bool = False,
    enforce_canonical_history: bool = True,
    parity_mode: str | None = None,
    mcts_sims: int = 100,
    nnue_blend: float = 0.5,
    nn_model_id: str | None = None,
    game_timeout: int = DEFAULT_GAME_TIMEOUT,
    session_timeout: int = DEFAULT_SESSION_TIMEOUT,
) -> dict[str, Any]:
    """Run hybrid GPU-accelerated self-play.

    Args:
        board_type: Board type (square8, square19, hex)
        num_players: Number of players (2-4)
        num_games: Number of games to generate
        output_dir: Output directory
        max_moves: Maximum moves per game
        seed: Random seed
        use_numba: Use Numba JIT-compiled rules
        engine_mode: Engine mode for player 1 (random-only, heuristic-only, mixed, nnue-guided, or mcts)
        p2_engine_mode: Engine mode for player 2 (if different from player 1)
        p3_engine_mode: Engine mode for player 3 (if different, for 3-4 player games)
        p4_engine_mode: Engine mode for player 4 (if different, for 4 player games)
        weights: Heuristic weights dict (from CMA-ES profile or defaults)
        mix_ratio: For mixed mode: probability of heuristic (0.0-1.0). Default 0.8
        mcts_sims: Number of MCTS simulations per move (for mcts mode). Default 100
        nnue_blend: For nnue-guided mode: blend ratio of NNUE vs heuristic. Default 0.5

    Returns:
        Statistics dictionary
    """
    from app.ai.gpu_batch import get_device
    from app.ai.hybrid_gpu import (
        create_hybrid_evaluator,
    )
    from app.game_engine import GameEngine
    from app.models import BoardType
    from app.training.initial_state import create_initial_state

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    board_type_key = board_type.lower()
    board_size = {"square8": 8, "square19": 19, "hex8": 9, "hex": 25, "hexagonal": 25}.get(board_type_key, 8)

    # Auto-calculate max_moves based on board type and player count if not specified
    # Larger boards and more players need more moves to reach natural game end
    if max_moves is None:
        max_moves_table = {
            # (board_type, num_players): max_moves
            ("square8", 2): 500,
            ("square8", 3): 800,
            ("square8", 4): 1200,
            ("square19", 2): 1200,
            ("square19", 3): 1600,
            ("square19", 4): 2000,
            ("hex8", 2): 500,
            ("hex8", 3): 800,
            ("hex8", 4): 1200,
            ("hex", 2): 1200,
            ("hex", 3): 1600,
            ("hex", 4): 2000,
            ("hexagonal", 2): 1200,
            ("hexagonal", 3): 1600,
            ("hexagonal", 4): 2000,
        }
        key = (board_type.lower(), num_players)
        max_moves = max_moves_table.get(key, 2000)
        logger.info(f"Auto-adjusted max_moves to {max_moves} for {board_type} {num_players}p")
    board_type_enum_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex8": BoardType.HEX8,
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
    mcts_state_adapter_class = None  # Will hold the adapter class for MCTS
    if engine_mode == "mcts":
        try:
            import hashlib

            from app.mcts.improved_mcts import (
                GameState as MCTSGameState,
                ImprovedMCTS,
                MCTSConfig,
                NeuralNetworkInterface,
            )

            class GameStateAdapter(MCTSGameState):
                """Adapts app.models.GameState to MCTS GameState interface.

                MCTS expects integer move indices, but our game uses Move objects.
                This adapter maintains a bidirectional mapping between them.
                """

                def __init__(self, real_state, engine, move_list=None):
                    """
                    Args:
                        real_state: The actual GameState from app.models
                        engine: The rules engine for move generation/application
                        move_list: Optional pre-computed list of legal moves
                    """
                    self.real_state = real_state
                    self.engine = engine
                    self._current_player = real_state.current_player or 1
                    # Cache legal moves for consistent indexing
                    if move_list is not None:
                        self._legal_moves = move_list
                    else:
                        self._legal_moves = engine.get_valid_moves(real_state, self._current_player)

                def get_legal_moves(self) -> list[int]:
                    """Return indices [0, 1, 2, ...] for legal moves."""
                    return list(range(len(self._legal_moves)))

                def apply_move(self, move_idx: int) -> GameStateAdapter:
                    """Apply move by index and return new adapted state."""
                    if move_idx < 0 or move_idx >= len(self._legal_moves):
                        raise ValueError(f"Invalid move index {move_idx}, have {len(self._legal_moves)} moves")
                    move = self._legal_moves[move_idx]
                    new_real_state = self.engine.apply_move(
                        self.real_state.model_copy(deep=True), move
                    )
                    return GameStateAdapter(new_real_state, self.engine)

                def is_terminal(self) -> bool:
                    """Check if game is over."""
                    return self.real_state.game_status != "active"

                def get_outcome(self, player: int) -> float:
                    """Get outcome for specified player."""
                    if self.real_state.winner == player:
                        return 1.0
                    elif self.real_state.winner is not None:
                        return -1.0
                    return 0.0  # Draw or ongoing

                def current_player(self) -> int:
                    """Get current player (0-indexed for MCTS)."""
                    # MCTS typically uses 0/1, our game uses 1/2/3/4
                    return (self._current_player - 1) % 2

                def hash(self) -> str:
                    """Generate unique hash for transposition table."""
                    # Use board state and current player for hash
                    board_str = str(self.real_state.board)
                    player_str = str(self._current_player)
                    return hashlib.md5(f"{board_str}:{player_str}".encode()).hexdigest()

                def get_move_by_index(self, idx: int):
                    """Get the actual Move object for an index."""
                    if 0 <= idx < len(self._legal_moves):
                        return self._legal_moves[idx]
                    return None

                def get_move_index(self, move) -> int:
                    """Get index for a Move object."""
                    try:
                        return self._legal_moves.index(move)
                    except ValueError:
                        return -1

            # Store adapter class for use in game loop
            mcts_state_adapter_class = GameStateAdapter

            # Create a heuristic-based network wrapper for pure MCTS
            class HeuristicNetworkWrapper(NeuralNetworkInterface):
                """Wraps heuristic evaluator as a neural network interface for MCTS."""

                def __init__(self, evaluator, engine, board_size: int):
                    self.evaluator = evaluator
                    self.engine = engine
                    self.board_size = board_size

                def evaluate(self, state: GameStateAdapter) -> tuple[list[float], float]:
                    """Return uniform policy over legal moves and heuristic value.

                    Policy indices match the legal move indices from GameStateAdapter.
                    """
                    # Get the real state and legal moves from adapter
                    real_state = state.real_state
                    current_player = state._current_player
                    legal_moves = state._legal_moves
                    num_legal = len(legal_moves)

                    # Create policy - indexed by move position in legal_moves
                    # MCTS uses indices 0..num_legal-1, so policy should have
                    # non-zero values at these indices
                    max_policy_size = max(num_legal, self.board_size * self.board_size * 4)
                    policy = [0.0] * max_policy_size

                    if num_legal > 0:
                        prob = 1.0 / num_legal
                        for i in range(num_legal):
                            policy[i] = prob

                    # Use heuristic evaluation for value
                    try:
                        move_scores = self.evaluator.evaluate_moves(
                            real_state, legal_moves, current_player, self.engine
                        )
                        if move_scores:
                            # Normalize scores to [-1, 1] range
                            scores = [s for _, s in move_scores]
                            value = sum(scores) / len(scores) / 1000.0  # Normalize
                            value = max(-1.0, min(1.0, value))
                        else:
                            value = 0.0
                    except Exception:
                        value = 0.0

                    return policy, value

            mcts_config = MCTSConfig(
                num_simulations=mcts_sims,
                cpuct=1.414,
                root_dirichlet_alpha=0.3,
                root_noise_weight=0.25,
            )

            # Create heuristic network wrapper
            heuristic_network = HeuristicNetworkWrapper(evaluator, GameEngine, board_size)
            mcts_ai = ImprovedMCTS(network=heuristic_network, config=mcts_config)
            logger.info(f"MCTS initialized with {mcts_sims} simulations per move (heuristic network)")
        except ImportError as e:
            logger.warning(f"MCTS not available: {e}. Falling back to heuristic.")
            mcts_ai = None
        except Exception as e:
            logger.warning(f"MCTS initialization failed: {e}. Falling back to heuristic.")
            mcts_ai = None

    # Initialize Minimax AI for nn-minimax mode
    minimax_ai = None
    all_engine_modes = {engine_mode, p2_engine_mode, p3_engine_mode, p4_engine_mode}
    if "nn-minimax" in all_engine_modes:
        try:
            from app.ai.minimax_ai import MinimaxAI
            from app.models.core import AIConfig
            minimax_config = AIConfig(
                difficulty=6,
                think_time=2000,  # 2 second think time
                use_neural_net=True,
            )
            minimax_ai = MinimaxAI(config=minimax_config, player_number=1)
            logger.info("Minimax AI initialized with neural net evaluation")
        except ImportError as e:
            logger.warning(f"Minimax AI not available: {e}. Falling back to heuristic.")
            minimax_ai = None

    # Initialize Descent AI for nn-descent mode
    descent_ai = None
    if "nn-descent" in all_engine_modes:
        try:
            from app.ai.descent_ai import DescentAI
            from app.models.core import AIConfig
            descent_config = AIConfig(
                difficulty=6,
                think_time=2000,  # 2 second think time
                use_neural_net=True,
                nn_model_id=nn_model_id,  # Use specified model
            )
            descent_ai = DescentAI(config=descent_config, player_number=1)
            logger.info(f"Descent AI initialized with neural net evaluation (model={nn_model_id or 'default'})")
        except ImportError as e:
            logger.warning(f"Descent AI not available: {e}. Falling back to heuristic.")
            descent_ai = None

    # Initialize Policy-Only AI for fast NN-based selfplay (no search, ~100x faster)
    policy_only_ai = None
    if "policy-only" in all_engine_modes:
        try:
            from app.ai.policy_only_ai import PolicyOnlyAI
            from app.models.core import AIConfig, AIType
            policy_config = AIConfig(
                ai_type=AIType.POLICY_ONLY,
                difficulty=4,
                policy_temperature=1.0,  # Exploratory temperature for diversity
                use_neural_net=True,
                nn_model_id=nn_model_id,  # Use specified model
            )
            policy_only_ai = PolicyOnlyAI(player_number=1, config=policy_config, board_type=board_type)
            logger.info(f"Policy-Only AI initialized for fast selfplay (model={nn_model_id or 'default'})")
        except ImportError as e:
            logger.warning(f"Policy-Only AI not available: {e}. Falling back to heuristic.")
            policy_only_ai = None

    # Initialize Gumbel MCTS AI for efficient search-based selfplay
    gumbel_mcts_ai = None
    if "gumbel-mcts" in all_engine_modes:
        try:
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI
            from app.models.core import AIConfig, AIType
            gumbel_config = AIConfig(
                ai_type=AIType.GUMBEL_MCTS,
                difficulty=7,
                gumbel_num_sampled_actions=16,
                gumbel_simulation_budget=100,
                use_neural_net=True,
                nn_model_id=nn_model_id,
            )
            gumbel_mcts_ai = GumbelMCTSAI(player_number=1, config=gumbel_config, board_type=board_type)
            logger.info(f"Gumbel MCTS AI initialized (m=16, budget=100, model={nn_model_id or 'default'})")
        except ImportError as e:
            logger.warning(f"Gumbel MCTS AI not available: {e}. Falling back to heuristic.")
            gumbel_mcts_ai = None

    # Initialize MaxN AI for multi-player search
    maxn_ai = None
    if "maxn" in all_engine_modes:
        try:
            from app.ai.maxn_ai import MaxNAI
            from app.models.core import AIConfig, AIType
            maxn_config = AIConfig(
                ai_type=AIType.MAXN,
                difficulty=7,
                think_time=2000,
            )
            maxn_ai = MaxNAI(player_number=1, config=maxn_config)
            logger.info("MaxN AI initialized (multi-player minimax)")
        except ImportError as e:
            logger.warning(f"MaxN AI not available: {e}. Falling back to heuristic.")
            maxn_ai = None

    # Initialize BRS AI for best reply search
    brs_ai = None
    if "brs" in all_engine_modes:
        try:
            from app.ai.maxn_ai import BRSAI
            from app.models.core import AIConfig, AIType
            brs_config = AIConfig(
                ai_type=AIType.BRS,
                difficulty=7,
                think_time=2000,
            )
            brs_ai = BRSAI(player_number=1, config=brs_config)
            logger.info("BRS AI initialized (best reply search)")
        except ImportError as e:
            logger.warning(f"BRS AI not available: {e}. Falling back to heuristic.")
            brs_ai = None

    # Build per-player engine mode mapping for asymmetric matches
    player_engine_modes = {
        1: engine_mode,
        2: p2_engine_mode or engine_mode,
        3: p3_engine_mode or engine_mode,
        4: p4_engine_mode or engine_mode,
    }
    is_asymmetric = any(
        player_engine_modes[p] != engine_mode for p in range(2, num_players + 1)
    )
    if is_asymmetric:
        logger.info("ASYMMETRIC MATCH CONFIGURATION:")
        for p in range(1, num_players + 1):
            logger.info(f"  Player {p}: {player_engine_modes[p]}")

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
    wins_by_player = dict.fromkeys(range(1, num_players + 1), 0)
    draws = 0
    victory_type_counts: dict[str, int] = {}  # Track victory type distribution
    stalemate_by_tiebreaker: dict[str, int] = {}  # Track which tiebreaker resolved stalemates
    game_lengths: list[int] = []  # Track individual game lengths for detailed stats
    game_records = []

    games_file = os.path.join(output_dir, "games.jsonl")

    logger.info(f"Starting {num_games} games...")
    start_time = time.time()
    session_deadline = start_time + session_timeout

    # Heartbeat tracking for stuck detection
    games_completed = 0
    games_timed_out = 0
    current_game_start = time.time()
    heartbeat_stop = threading.Event()

    def heartbeat_thread():
        """Log progress periodically to detect stuck sessions."""
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(HEARTBEAT_INTERVAL)
            if heartbeat_stop.is_set():
                break
            elapsed = time.time() - start_time
            game_elapsed = time.time() - current_game_start
            remaining_session = session_deadline - time.time()
            logger.info(
                f"[Heartbeat] Session running for {elapsed:.0f}s, "
                f"{games_completed}/{num_games} games done, "
                f"current game: {game_elapsed:.0f}s, "
                f"session timeout in {remaining_session:.0f}s"
            )
            if game_elapsed > game_timeout:
                logger.warning(f"Current game taking {game_elapsed:.0f}s (timeout: {game_timeout}s)")

    heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
    heartbeat.start()

    try:
        with open(games_file, "w") as f:
            # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.error(f"Cannot acquire lock on {games_file} - another process is writing")
                sys.exit(1)
            for game_idx in range(num_games):
                # Check session timeout
                if time.time() > session_deadline:
                    logger.error(f"Session timeout ({session_timeout}s) exceeded at game {game_idx}")
                    break

                # Check disk space every 10 games
                if game_idx % 10 == 0:
                    disk_status = check_disk_space(logger, output_dir)
                    if disk_status == "critical":
                        logger.error(f"Aborting selfplay at game {game_idx} due to critical disk usage")
                        break
                    elif disk_status == "warning":
                        logger.warning(f"Disk space low, continuing cautiously at game {game_idx}")

                game_start = time.time()
                current_game_start = game_start  # Update for heartbeat tracking

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
                game_timed_out = False

                while game_state.game_status == "active" and move_count < max_moves:
                    # Check game timeout
                    if time.time() - game_start > game_timeout:
                        logger.warning(f"Game {game_idx} timed out after {game_timeout}s, aborting")
                        game_timed_out = True
                        games_timed_out += 1
                        break

                    current_player = game_state.current_player
                    mcts_policy_dist = None  # Reset for each move; populated if MCTS mode is used

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
                            # Apply the bookkeeping move and continue the game loop
                            move_timestamp = datetime.now(timezone.utc)
                            stamped_move = best_move.model_copy(
                                update={
                                    "id": f"move-{move_count + 1}",
                                    "timestamp": move_timestamp,
                                    "think_time": 0,
                                    "move_number": move_count + 1,
                                }
                            )
                            game_state = GameEngine.apply_move(game_state, stamped_move)
                            moves_for_db.append(stamped_move)
                            moves_played.append({
                                "type": stamped_move.type.value if hasattr(stamped_move.type, 'value') else str(stamped_move.type),
                                "player": stamped_move.player,
                            })
                            move_count += 1
                            continue  # Continue to next iteration of the game loop
                        else:
                            # No valid moves and no phase requirement - trigger victory check
                            GameEngine._check_victory(game_state)
                            break
                    else:
                        # Select move based on engine mode (per-player for asymmetric matches)
                        current_engine = player_engine_modes[current_player]

                        if current_engine == "random-only" or current_engine == "random":
                            # Uniform random move selection (no evaluation)
                            best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "mixed":
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
                        elif current_engine == "mcts" and mcts_ai is not None and mcts_state_adapter_class is not None:
                            # MCTS mode: Use Monte Carlo Tree Search for move selection
                            mcts_policy_dist = None  # Will be populated if MCTS succeeds
                            try:
                                # Wrap game state in adapter for MCTS (maps Move objects to int indices)
                                adapted_state = mcts_state_adapter_class(game_state, GameEngine, valid_moves)

                                # MCTS returns an integer move index
                                move_idx = mcts_ai.search(adapted_state)

                                if move_idx is None or move_idx < 0 or move_idx >= len(valid_moves):
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
                                else:
                                    # Convert index back to Move object
                                    best_move = adapted_state.get_move_by_index(move_idx)

                                    # Capture MCTS visit distribution for KL training
                                    try:
                                        policy_list = mcts_ai.get_policy(temperature=1.0)
                                        # Convert to sparse dict (only non-zero probs)
                                        # Indices are positions in valid_moves list
                                        mcts_policy_dist = {
                                            idx: prob for idx, prob in enumerate(policy_list)
                                            if prob > 1e-6
                                        }
                                    except Exception as policy_err:
                                        logger.debug(f"Failed to get MCTS policy: {policy_err}")
                            except Exception as e:
                                logger.debug(f"MCTS error: {e}, falling back to heuristic")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "nnue-guided" and nnue_evaluator is not None:
                            # NNUE-guided mode: Blend NNUE and heuristic scores
                            try:
                                # Get heuristic scores
                                heuristic_scores = evaluator.evaluate_moves(
                                    game_state, valid_moves, current_player, GameEngine
                                )
                                heuristic_dict = dict(heuristic_scores) if heuristic_scores else {}

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
                        elif current_engine == "nn-minimax" and minimax_ai is not None:
                            # NN-Minimax mode: Use minimax search with neural net evaluation
                            try:
                                # Update AI's player number for current player
                                minimax_ai.player_number = current_player
                                best_move = minimax_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to heuristic if minimax fails
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
                                logger.debug(f"Minimax error: {e}, falling back to heuristic")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "nn-descent" and descent_ai is not None:
                            # NN-Descent mode: Use descent search with neural net evaluation
                            try:
                                # Update AI's player number for current player
                                descent_ai.player_number = current_player
                                best_move = descent_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to heuristic if descent fails
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
                                logger.debug(f"Descent error: {e}, falling back to heuristic")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "policy-only" and policy_only_ai is not None:
                            # Policy-Only mode: Fast NN policy without search (~100x faster)
                            try:
                                # Update AI's player number for current player
                                policy_only_ai.player_number = current_player
                                best_move = policy_only_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to random if policy-only fails
                                    best_move = valid_moves[np.random.randint(len(valid_moves))]
                            except Exception as e:
                                logger.debug(f"Policy-Only error: {e}, falling back to random")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "gumbel-mcts" and gumbel_mcts_ai is not None:
                            # Gumbel MCTS mode: Efficient search with Sequential Halving
                            try:
                                # Update AI's player number for current player
                                gumbel_mcts_ai.player_number = current_player
                                best_move = gumbel_mcts_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to random if gumbel fails
                                    best_move = valid_moves[np.random.randint(len(valid_moves))]
                                else:
                                    # Capture visit distribution for soft policy targets
                                    dist_moves, dist_probs = gumbel_mcts_ai.get_visit_distribution()
                                    if dist_moves and dist_probs:
                                        # Map moves to indices in valid_moves
                                        move_to_idx = {m: i for i, m in enumerate(valid_moves)}
                                        mcts_policy_dist = {}
                                        for m, p in zip(dist_moves, dist_probs, strict=False):
                                            if m in move_to_idx and p > 1e-6:
                                                mcts_policy_dist[move_to_idx[m]] = p
                            except Exception as e:
                                logger.debug(f"Gumbel MCTS error: {e}, falling back to random")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "maxn" and maxn_ai is not None:
                            # MaxN mode: Multi-player minimax search
                            try:
                                # Update AI's player number for current player
                                maxn_ai.player_number = current_player
                                best_move = maxn_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to heuristic if maxn fails
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
                                logger.debug(f"MaxN error: {e}, falling back to heuristic")
                                best_move = valid_moves[np.random.randint(len(valid_moves))]
                        elif current_engine == "brs" and brs_ai is not None:
                            # BRS mode: Best Reply Search (faster alternative to MaxN)
                            try:
                                # Update AI's player number for current player
                                brs_ai.player_number = current_player
                                best_move = brs_ai.select_move(game_state)
                                if best_move is None or best_move not in valid_moves:
                                    # Fallback to heuristic if BRS fails
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
                                logger.debug(f"BRS error: {e}, falling back to heuristic")
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

                        # Add MCTS policy distribution for KL-divergence training
                        if mcts_policy_dist is not None:
                            move_record["mcts_policy"] = mcts_policy_dist

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

                # Derive effective game status for training data validity
                # Games that exit with status "active" need a distinct status based on why they ended
                effective_status = game_state.game_status
                if effective_status == "active":
                    if game_timed_out:
                        # Wall-clock timeout - game was forcibly stopped
                        effective_status = "timeout"
                    elif victory_type == "timeout":
                        # Hit max_moves limit without natural conclusion
                        effective_status = "max_moves"
                    else:
                        # Other non-natural endings (stalemate resolved by tiebreaker, etc.)
                        effective_status = "completed"

                record = {
                    # === Core game identifiers ===
                    "game_id": f"hybrid_{board_type}_{num_players}p_{game_idx}_{int(datetime.now().timestamp())}",
                    "board_type": board_type,  # square8, square19, hexagonal
                    "num_players": num_players,
                    # === Game outcome ===
                    "winner": winner,
                    "move_count": move_count,
                    "status": effective_status,  # completed, abandoned, etc.
                    "game_status": effective_status,  # Alias for compatibility
                    "victory_type": victory_type,  # territory, elimination, lps, stalemate, timeout
                    "stalemate_tiebreaker": stalemate_tiebreaker,  # territory, ring_elim, or None
                    "termination_reason": f"status:{effective_status}:{victory_type}",
                    # === Engine/opponent metadata ===
                    "engine_mode": engine_mode,  # Default engine mode (P1)
                    "player_engine_modes": {str(p): player_engine_modes[p] for p in range(1, num_players + 1)},  # Per-player engine modes
                    "is_asymmetric": is_asymmetric,  # Whether different players used different engines
                    "opponent_type": "selfplay" if not is_asymmetric else "ai_vs_ai",  # ai_vs_ai for asymmetric
                    "player_types": [player_engine_modes[p] for p in range(1, num_players + 1)],  # Engine type of each player
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

                # Clear move cache between games to prevent stale cache entries
                # causing infinite loops (esp. for hex boards with larger state spaces)
                GameEngine.clear_cache()

                # Progress logging
                if (game_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    games_per_sec = (game_idx + 1) / elapsed
                    eta = (num_games - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0

                    logger.info(
                        f"  Game {game_idx + 1}/{num_games}: "
                        f"{games_per_sec:.2f} g/s, ETA: {eta:.0f}s"
                    )

                    # Update games completed for heartbeat
                    games_completed = game_idx + 1

    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)

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
    from app.ai.gpu_batch import get_device
    from app.ai.hybrid_gpu import (
        benchmark_hybrid_evaluation,
        create_hybrid_evaluator,
    )
    from app.ai.numba_rules import (
        NUMBA_AVAILABLE,
        benchmark_numba_functions,
    )
    from app.game_engine import GameEngine
    from app.models import BoardType
    from app.training.initial_state import create_initial_state

    logger.info("=" * 60)
    logger.info("BENCHMARK: CPU vs Hybrid GPU Evaluation")
    logger.info("=" * 60)

    board_size = {"square8": 8, "square19": 19, "hex8": 9, "hex": 25, "hexagonal": 25}.get(board_type.lower(), 8)
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
            GameEngine.apply_move(game_state, move)
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
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Hybrid GPU-accelerated self-play with full rule fidelity",
        include_gpu=True,
        include_ramdrive=True,
    )

    # Add hybrid-specific arguments
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum moves per game (default: auto-calculated from board type)",
    )
    parser.add_argument(
        "--no-record-db",
        action="store_true",
        help="Disable DB recording even if --record-db is set.",
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
        "--benchmark",
        action="store_true",
        help="Run benchmark only",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT compilation",
    )
    # Per-player engine modes for asymmetric matches
    parser.add_argument(
        "--p2-engine-mode",
        type=str,
        default=None,
        help="Engine mode for Player 2 (if different from P1 for asymmetric matches)",
    )
    parser.add_argument(
        "--p3-engine-mode",
        type=str,
        default=None,
        help="Engine mode for Player 3 (for 3-4 player games)",
    )
    parser.add_argument(
        "--p4-engine-mode",
        type=str,
        default=None,
        help="Engine mode for Player 4 (for 4 player games)",
    )
    parser.add_argument(
        "--nn-model-id",
        type=str,
        default=None,
        help="Neural network model ID for gumbel-mcts/policy-only/mcts modes",
    )
    parser.add_argument(
        "--nnue-blend",
        type=float,
        default=0.5,
        help="For nnue-guided mode: blend ratio of NNUE vs heuristic (default: 0.5)",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.8,
        help="For mixed mode: probability of using heuristic (default: 0.8)",
    )
    parser.add_argument(
        "--profile",
        dest="weights_profile",
        type=str,
        help="Profile name in weights file (alias for --weights-profile)",
    )
    # Additional ramdrive args
    parser.add_argument("--ram-storage", action="store_true", help="Use ramdrive storage")
    parser.add_argument("--sync-target", type=str, help="Target directory for ramdrive sync")

    # Timeout handling
    parser.add_argument(
        "--game-timeout",
        type=int,
        default=DEFAULT_GAME_TIMEOUT,
        help=f"Timeout per game in seconds (default: {DEFAULT_GAME_TIMEOUT})",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=DEFAULT_SESSION_TIMEOUT,
        help=f"Timeout for entire session in seconds (default: {DEFAULT_SESSION_TIMEOUT})",
    )

    parsed = parser.parse_args()

    # Create SelfplayConfig from parsed args
    selfplay_config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        engine_mode=parsed.engine_mode,
        mcts_simulations=parsed.mcts_simulations,
        output_dir=parsed.output_dir,
        record_db=parsed.record_db,
        lean_db=parsed.lean_db,
        seed=parsed.seed or 42,
        weights_file=parsed.weights_file,
        weights_profile=parsed.weights_profile,
        use_ramdrive=parsed.use_ramdrive,
        ramdrive_path=parsed.ramdrive_path,
        sync_interval=parsed.sync_interval,
        source="run_hybrid_selfplay.py",
        extra_options={
            "max_moves": parsed.max_moves,
            "no_record_db": parsed.no_record_db,
            "no_enforce_canonical_history": parsed.no_enforce_canonical_history,
            "parity_mode": parsed.parity_mode,
            "benchmark": parsed.benchmark,
            "no_numba": parsed.no_numba,
            "p2_engine_mode": parsed.p2_engine_mode,
            "p3_engine_mode": parsed.p3_engine_mode,
            "p4_engine_mode": parsed.p4_engine_mode,
            "nn_model_id": parsed.nn_model_id,
            "nnue_blend": parsed.nnue_blend,
            "mix_ratio": parsed.mix_ratio,
            "ram_storage": getattr(parsed, "ram_storage", False),
            "sync_target": getattr(parsed, "sync_target", None),
        },
    )

    # Create backward-compatible args object
    args = type("Args", (), {
        "board_type": selfplay_config.board_type,
        "num_players": selfplay_config.num_players,
        "num_games": selfplay_config.num_games,
        "engine_mode": selfplay_config.engine_mode,
        "output_dir": selfplay_config.output_dir,
        "record_db": selfplay_config.record_db,
        "lean_db": selfplay_config.lean_db,
        "seed": selfplay_config.seed,
        "weights_file": selfplay_config.weights_file,
        "weights_profile": selfplay_config.weights_profile,
        "mcts_sims": selfplay_config.mcts_simulations,
        "max_moves": selfplay_config.extra_options["max_moves"],
        "no_record_db": selfplay_config.extra_options["no_record_db"],
        "no_enforce_canonical_history": selfplay_config.extra_options["no_enforce_canonical_history"],
        "parity_mode": selfplay_config.extra_options["parity_mode"],
        "benchmark": selfplay_config.extra_options["benchmark"],
        "no_numba": selfplay_config.extra_options["no_numba"],
        "p2_engine_mode": selfplay_config.extra_options["p2_engine_mode"],
        "p3_engine_mode": selfplay_config.extra_options["p3_engine_mode"],
        "p4_engine_mode": selfplay_config.extra_options["p4_engine_mode"],
        "nn_model_id": selfplay_config.extra_options["nn_model_id"],
        "nnue_blend": selfplay_config.extra_options["nnue_blend"],
        "mix_ratio": selfplay_config.extra_options["mix_ratio"],
        "ram_storage": selfplay_config.extra_options["ram_storage"],
        "sync_target": selfplay_config.extra_options["sync_target"],
        "sync_interval": selfplay_config.sync_interval,
        "game_timeout": parsed.game_timeout,
        "session_timeout": parsed.session_timeout,
    })()

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
        if args.weights_file and args.weights_profile:
            weights = load_weights_from_profile(args.weights_file, args.weights_profile)
            logger.info(f"Loaded weights from {args.weights_file}:{args.weights_profile}")
        elif args.weights_file or args.weights_profile:
            # Only one was specified - warn user
            logger.warning("Both --weights-file and --profile are required to load custom weights")

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
                should_proceed_with_priority,
            )
            # Estimate output size: ~2KB per game for JSONL/DB
            estimated_output_mb = (args.num_games * 0.002) + 50
            if not check_disk_space(required_gb=max(2.0, estimated_output_mb / 1024)):
                logger.error(f"Insufficient disk space (limit: {LIMITS.DISK_MAX_PERCENT}%)")
                sys.exit(1)
            if not check_memory(required_gb=2.0):
                logger.error(f"Insufficient memory (limit: {LIMITS.MEMORY_MAX_PERCENT}%)")
                sys.exit(1)
            if not check_gpu_memory(required_gb=1.0):
                logger.warning("GPU memory constrained, may affect performance")
            logger.info("Resource check passed: disk/memory/GPU within 80% limits")

            # Graceful degradation: Hybrid selfplay is NORMAL priority
            degradation = get_degradation_level()
            if degradation >= 4:  # CRITICAL - resources at/above limits
                logger.error("Resources at critical levels, aborting hybrid selfplay")
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
        # Priority: --output-dir > --ram-storage > default
        output_dir = args.output_dir
        syncer = None
        if output_dir is None:
            if getattr(args, 'ram_storage', False):
                # Use ramdrive for high-speed I/O
                ramdrive_config = get_config_from_args(args)
                ramdrive_config.subdirectory = f"selfplay/hybrid_{args.board_type}_{args.num_players}p"
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
                        patterns=["*.db", "*.jsonl", "*.json"],
                    )
                    syncer.start()
                    logger.info(f"Started ramdrive sync: {output_dir} -> {sync_target} every {sync_interval}s")
            else:
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
                p2_engine_mode=args.p2_engine_mode,
                p3_engine_mode=args.p3_engine_mode,
                p4_engine_mode=args.p4_engine_mode,
                weights=weights,
                mix_ratio=args.mix_ratio,
                record_db=None if args.no_record_db else (args.record_db or None),
                lean_db=bool(args.lean_db),
                enforce_canonical_history=not bool(args.no_enforce_canonical_history),
                parity_mode=args.parity_mode,
                mcts_sims=args.mcts_sims,
                nnue_blend=args.nnue_blend,
                nn_model_id=args.nn_model_id,
                game_timeout=args.game_timeout,
                session_timeout=args.session_timeout,
            )
        finally:
            # Stop ramdrive syncer and perform final sync
            if syncer:
                logger.info("Stopping ramdrive syncer and performing final sync...")
                syncer.stop(final_sync=True)
                logger.info(f"Ramdrive sync stats: {syncer.stats}")

            # Record task completion for duration learning
            if HAS_COORDINATION and task_id:
                try:
                    import socket
                    node_id = socket.gethostname()
                    config = f"{args.board_type}_{args.num_players}p"
                    # Args: task_type, host, started_at, completed_at, success, config
                    record_task_completion("hybrid_selfplay", node_id, start_time, time.time(), True, config)
                    logger.info("Recorded task completion for duration learning")
                except Exception as e:
                    logger.warning(f"Failed to record task completion: {e}")


if __name__ == "__main__":
    main()
