"""GPU-accelerated parallel game simulation for CMA-ES and selfplay.

This module enables running multiple RingRift games in parallel using GPU
acceleration for move evaluation. Key use cases:

1. CMA-ES fitness evaluation: Run 10-100+ games per candidate in parallel
2. Selfplay data generation: Generate training data 10x faster
3. Tournament evaluation: Run many tournament games concurrently

Architecture:
- Maintains batch of game states on GPU
- Vectorized move generation and application
- Batch neural network / heuristic evaluation
- Efficient memory management with game recycling

Performance targets:
- 10-100 games/sec on RTX 3090 (vs 1 game/sec CPU)
- 50-500 games/sec on A100 / RTX 5090

MPS (Apple Silicon) Performance Note:
-------------------------------------
MPS is currently ~100x SLOWER than CPU due to excessive CPU-GPU synchronization
from .item() calls in the game loop. For Apple Silicon, use device="cpu" for
game simulation. The CPU implementation uses vectorized numpy operations that
are more efficient than MPS kernel launches for the small tensor operations
in game state management.

CUDA provides the expected 6-10x speedups. After extensive optimization (Dec 2025),
only 1 .item() call remains:
1. Statistics tracking (once per batch at end of run_games, minimal impact)
All move selection (policy, heuristic, center-bias) is now fully vectorized with
segment-wise softmax sampling - no per-game loops or .item() calls in the hot path.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch


# =============================================================================
# Configuration Dataclass (Dec 2025 - fix 25-param constructor)
# =============================================================================


@dataclass
class GameRunnerConfig:
    """Configuration for ParallelGameRunner.

    Groups the 25+ parameters into logical categories for cleaner initialization.
    All parameters have sensible defaults matching the original constructor.

    Categories:
        - Board: batch_size, board_size, num_players, board_type, device
        - Validation: shadow_*, state_* parameters for GPU/CPU parity checking
        - Rules: swap_enabled, lps_victory_rounds, rings_per_player
        - Selection: use_heuristic_selection, weight_noise, temperature, noise_scale
        - Training: random_opening_moves, record_policy
        - Personas: persona_pool, per_player_personas

    Example:
        >>> config = GameRunnerConfig(
        ...     batch_size=128,
        ...     board_type="hex8",
        ...     num_players=2,
        ...     record_policy=True,
        ... )
        >>> runner = ParallelGameRunner(config=config)
    """

    # Board configuration
    batch_size: int = 64
    board_size: int = 8
    num_players: int = 2
    board_type: str | None = None
    device: torch.device | None = None

    # Shadow validation (move generation parity)
    shadow_validation: bool = False
    shadow_sample_rate: float = 0.05
    shadow_threshold: float = 0.001
    async_shadow_validation: bool = True

    # State validation (CPU oracle mode)
    state_validation: bool = False
    state_sample_rate: float = 0.01
    state_threshold: float = 0.001

    # Game rules
    swap_enabled: bool = False
    lps_victory_rounds: int | None = None
    rings_per_player: int | None = None

    # AI/Selection parameters
    use_heuristic_selection: bool = False
    weight_noise: float = 0.0
    temperature: float = 1.0
    noise_scale: float = 0.1

    # Training options
    random_opening_moves: int = 0
    record_policy: bool = False

    # Persona configuration
    persona_pool: list[str] | None = None
    per_player_personas: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.per_player_personas is not None:
            if len(self.per_player_personas) != self.num_players:
                raise ValueError(
                    f"per_player_personas length ({len(self.per_player_personas)}) "
                    f"must match num_players ({self.num_players})"
                )

    @classmethod
    def for_selfplay(
        cls,
        board_type: str,
        num_players: int,
        batch_size: int = 64,
        record_policy: bool = True,
        **kwargs: Any,
    ) -> "GameRunnerConfig":
        """Create config optimized for selfplay data generation.

        Args:
            board_type: Board type ("hex8", "square8", etc.)
            num_players: Number of players
            batch_size: Games to run in parallel
            record_policy: Whether to record policy distributions
            **kwargs: Additional config overrides

        Returns:
            Config with selfplay-appropriate defaults
        """
        return cls(
            board_type=board_type,
            num_players=num_players,
            batch_size=batch_size,
            record_policy=record_policy,
            use_heuristic_selection=True,
            weight_noise=0.1,  # Some variety
            **kwargs,
        )

    @classmethod
    def for_evaluation(
        cls,
        board_type: str,
        num_players: int,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> "GameRunnerConfig":
        """Create config optimized for model evaluation/gauntlet.

        Args:
            board_type: Board type
            num_players: Number of players
            batch_size: Games to run in parallel
            **kwargs: Additional config overrides

        Returns:
            Config with evaluation-appropriate defaults (no noise, deterministic)
        """
        return cls(
            board_type=board_type,
            num_players=num_players,
            batch_size=batch_size,
            record_policy=False,
            weight_noise=0.0,
            temperature=0.1,  # Near-deterministic
            **kwargs,
        )

    @classmethod
    def for_cmaes(
        cls,
        board_type: str,
        num_players: int,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> "GameRunnerConfig":
        """Create config optimized for CMA-ES fitness evaluation.

        Args:
            board_type: Board type
            num_players: Number of players
            batch_size: Games per fitness evaluation
            **kwargs: Additional config overrides

        Returns:
            Config with CMA-ES-appropriate defaults
        """
        return cls(
            board_type=board_type,
            num_players=num_players,
            batch_size=batch_size,
            use_heuristic_selection=True,
            record_policy=False,
            **kwargs,
        )

# Resource checking before GPU operations (December 2025)
from app.utils.resource_guard import check_gpu_memory, check_memory, clear_gpu_memory

from .gpu_batch import get_device
from .gpu_batch_state import BatchGameState
from .gpu_game_types import GamePhase, GameStatus, MAX_STACK_HEIGHT
from .gpu_line_detection import (
    apply_line_elimination_batch,
    detect_lines_vectorized,
    process_lines_batch,
)
from .gpu_move_application import (
    apply_capture_moves_batch,
    apply_movement_moves_batch,
    apply_no_action_moves_batch,
    apply_placement_moves_batch,
    apply_recovery_moves_vectorized,
    # Canonical phase tracking (December 2025)
    check_and_apply_forced_elimination_batch,
    mark_real_action_batch,
    reset_capture_chain_batch,
    reset_turn_tracking_batch,
)
from .gpu_move_generation import (
    BatchMoves,
    apply_single_chain_capture,
    apply_single_initial_capture,
    generate_capture_moves_batch,
    generate_chain_capture_moves_from_position,
    generate_movement_moves_batch,
    generate_placement_moves_batch,
    generate_recovery_moves_batch,
)
from .gpu_selection import (
    PolicyData,
    select_moves_heuristic,
    select_moves_heuristic_with_policy,
    select_moves_vectorized,
)
from .gpu_territory import compute_territory_batch
from .shadow_validation import (
    AsyncShadowValidator,
    ShadowValidator,
    StateValidator,
    create_async_shadow_validator,
    create_shadow_validator,
    create_state_validator,
)

if TYPE_CHECKING:
    from app.ai.nnue_policy import RingRiftNNUEWithPolicy
    from app.models import BoardType

logger = logging.getLogger(__name__)


# =============================================================================
# Batch Move Application
# =============================================================================


# =============================================================================
# Dynamic Batch Sizing (Dec 2025)
# =============================================================================


# Board cell counts for batch size calculation
BOARD_CELLS: dict[str, int] = {
    "hex8": 61,
    "square8": 64,
    "square19": 361,
    "hexagonal": 469,
}


def get_optimal_batch_size(
    board_type: str | None = None,
    num_players: int = 2,
    gpu_memory_gb: float | None = None,
    board_size: int | None = None,
) -> int:
    """Calculate optimal batch size based on board complexity and GPU memory.

    Uses heuristic formula that balances memory usage against parallelism.
    Smaller boards allow larger batches, larger boards require smaller batches
    to avoid OOM errors.

    Dec 2025: Added to dynamically tune batch sizes instead of hardcoded 64.

    Args:
        board_type: Board type string ("hex8", "square8", "square19", "hexagonal").
                   Takes precedence over board_size if provided.
        num_players: Number of players per game (2, 3, or 4)
        gpu_memory_gb: Available GPU memory in GB. Auto-detected if None.
        board_size: Board dimension (fallback if board_type not provided).

    Returns:
        Optimal batch size, clamped to [32, 512] range.

    Examples:
        >>> get_optimal_batch_size("hex8", 2)  # Small board
        256
        >>> get_optimal_batch_size("hexagonal", 4)  # Large board + 4p
        64
        >>> get_optimal_batch_size("square19", 2)  # Large board
        128
    """
    # Determine cell count from board type or size
    if board_type:
        cells = BOARD_CELLS.get(board_type, 64)
    elif board_size:
        cells = board_size * board_size
    else:
        cells = 64  # Default to square8

    # Auto-detect GPU memory if not provided
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            try:
                # Get available memory in GB
                device = torch.cuda.current_device()
                total_mem = torch.cuda.get_device_properties(device).total_memory
                gpu_memory_gb = total_mem / (1024 ** 3)
            except RuntimeError:
                # CUDA initialization errors (driver issues, OOM)
                gpu_memory_gb = 8.0  # Safe default
        else:
            gpu_memory_gb = 8.0  # CPU fallback

    # Base formula: memory_gb * scale_factor / (cells * player_overhead)
    # Higher cell count = more memory per game = smaller batch
    # More players = more state to track = smaller batch
    player_factor = 1.0 + (num_players - 2) * 0.3  # 3p=1.3, 4p=1.6

    # Scale factor tuned empirically:
    # - 8GB GPU with hex8 (61 cells, 2p) -> batch ~300
    # - 8GB GPU with hexagonal (469 cells, 4p) -> batch ~64
    scale_factor = 15000

    base_batch = int(gpu_memory_gb * scale_factor / (cells * player_factor))

    # Clamp to reasonable range
    # Min 32: Below this overhead dominates
    # Max 512: Above this diminishing returns
    result = max(32, min(512, base_batch))

    logger.debug(
        f"[BatchSizing] board={board_type or board_size}, cells={cells}, "
        f"players={num_players}, gpu_mem={gpu_memory_gb:.1f}GB -> batch={result}"
    )

    return result


# =============================================================================
# Parallel Game Runner
# =============================================================================


class ParallelGameRunner:
    """GPU-accelerated parallel game simulation.

    Runs multiple games simultaneously using batch operations on GPU.
    Supports different AI configurations per game for CMA-ES evaluation.

    Example (new style with config):
        config = GameRunnerConfig(
            batch_size=128,
            board_type="hex8",
            num_players=2,
            record_policy=True,
        )
        runner = ParallelGameRunner(config=config)

    Example (legacy style with individual params - still supported):
        runner = ParallelGameRunner(batch_size=64, device="cuda")

        # Run games with specific heuristic weights
        results = runner.run_games(
            weights_per_game=[weights1, weights2, ...],  # length 64
            max_moves=10000,
        )

        # Results contain win/loss/draw for each game

    Example (factory methods):
        config = GameRunnerConfig.for_selfplay("hex8", 2, batch_size=128)
        runner = ParallelGameRunner(config=config)
    """

    def __init__(
        self,
        # New config-based initialization (preferred)
        config: GameRunnerConfig | None = None,
        # Legacy individual parameters (for backward compatibility)
        batch_size: int = 64,
        board_size: int = 8,
        num_players: int = 2,
        device: torch.device | None = None,
        shadow_validation: bool = False,
        shadow_sample_rate: float = 0.05,
        shadow_threshold: float = 0.001,
        async_shadow_validation: bool = True,
        state_validation: bool = False,
        state_sample_rate: float = 0.01,
        state_threshold: float = 0.001,
        swap_enabled: bool = False,
        lps_victory_rounds: int | None = None,
        rings_per_player: int | None = None,
        board_type: str | None = None,
        use_heuristic_selection: bool = False,
        weight_noise: float = 0.0,
        temperature: float = 1.0,
        noise_scale: float = 0.1,
        random_opening_moves: int = 0,
        record_policy: bool = False,
        persona_pool: list[str] | None = None,
        per_player_personas: list[str] | None = None,
    ):
        """Initialize parallel game runner.

        Accepts either a GameRunnerConfig object (preferred) or individual parameters
        (legacy, for backward compatibility).

        Args:
            config: Configuration object (preferred). If provided, individual params
                   are ignored except for device which can override config.device.

            # Legacy parameters (used if config is None):
            batch_size: Number of games to run in parallel
            board_size: Board dimension
            num_players: Number of players per game
            device: GPU device (auto-detected if None)
            shadow_validation: Enable shadow validation against CPU rules (move generation)
            shadow_sample_rate: Fraction of moves to validate (0.0-1.0)
            shadow_threshold: Maximum divergence rate before halt
            async_shadow_validation: Use background thread for shadow validation (default True)
                                    Prevents GPU blocking during CPU validation overhead.
            state_validation: Enable CPU oracle mode (state validation)
            state_sample_rate: Fraction of states to validate (0.0-1.0)
            state_threshold: Maximum state divergence rate before halt
            swap_enabled: Enable pie rule (swap_sides) for 2-player games (RR-CANON R180-R184)
            lps_victory_rounds: Number of consecutive rounds one player must have exclusive
                               real actions to win via LPS (None = board default, respects env vars)
            rings_per_player: Starting rings per player (None = board default, respects env vars)
            board_type: Board type string ("square8", "square19", "hexagonal") for proper
                       initialization. If "hexagonal", marks out-of-bounds cells as collapsed.
            use_heuristic_selection: When True, use heuristic-based move selection instead of
                       center-bias. Provides better move quality but slightly slower.
            weight_noise: Multiplicative noise factor (0.0-1.0) for heuristic weights.
                       Each weight is multiplied by a random factor in [1-noise, 1+noise].
                       This increases training diversity by making each game use slightly
                       different evaluation. Default 0.0 (no noise).
            temperature: Softmax temperature for move sampling (higher = more random).
                       Used for curriculum learning. Default 1.0.
            noise_scale: Scale of noise added to move scores for exploration.
                       Used for curriculum learning diversity. Default 0.1.
            random_opening_moves: Number of initial moves to select uniformly at random
                       instead of using heuristic/policy. Increases opening diversity
                       for training. Default 0 (no random opening).
            record_policy: When True, record policy distributions (move scores and
                       probabilities) for each move. This enables capturing policy
                       targets during GPU selfplay for training. Default False.
            persona_pool: List of heuristic persona IDs to use for training variety.
                       Each game randomly samples a persona from this pool.
                       Valid IDs: "balanced", "aggressive", "territorial", "defensive".
                       If None, uses board-specific default weights.
                       Example: ["balanced", "aggressive", "defensive"] creates games
                       with 3 different play styles for more diverse training data.
            per_player_personas: List of persona IDs, one per player position.
                       Enables different play styles for different players in the same game.
                       Length must match num_players. Takes precedence over persona_pool.
                       Example for 2-player: ["aggressive", "defensive"] makes P1 aggressive
                       and P2 defensive in all games. For variety, combine with weight_noise.
                       Example for 4-player: ["aggressive", "balanced", "territorial", "defensive"]
        """
        # Handle config vs individual params
        if config is not None:
            # Use config object (preferred)
            self.config = config
            batch_size = config.batch_size
            board_size = config.board_size
            num_players = config.num_players
            # Allow device override, but prefer config if device param not explicitly set
            if device is None:
                device = config.device
            shadow_validation = config.shadow_validation
            shadow_sample_rate = config.shadow_sample_rate
            shadow_threshold = config.shadow_threshold
            async_shadow_validation = config.async_shadow_validation
            state_validation = config.state_validation
            state_sample_rate = config.state_sample_rate
            state_threshold = config.state_threshold
            swap_enabled = config.swap_enabled
            lps_victory_rounds = config.lps_victory_rounds
            rings_per_player = config.rings_per_player
            board_type = config.board_type
            use_heuristic_selection = config.use_heuristic_selection
            weight_noise = config.weight_noise
            temperature = config.temperature
            noise_scale = config.noise_scale
            random_opening_moves = config.random_opening_moves
            record_policy = config.record_policy
            persona_pool = config.persona_pool
            per_player_personas = config.per_player_personas
        else:
            # Create config from individual params (legacy mode)
            self.config = GameRunnerConfig(
                batch_size=batch_size,
                board_size=board_size,
                num_players=num_players,
                device=device,
                shadow_validation=shadow_validation,
                shadow_sample_rate=shadow_sample_rate,
                shadow_threshold=shadow_threshold,
                async_shadow_validation=async_shadow_validation,
                state_validation=state_validation,
                state_sample_rate=state_sample_rate,
                state_threshold=state_threshold,
                swap_enabled=swap_enabled,
                lps_victory_rounds=lps_victory_rounds,
                rings_per_player=rings_per_player,
                board_type=board_type,
                use_heuristic_selection=use_heuristic_selection,
                weight_noise=weight_noise,
                temperature=temperature,
                noise_scale=noise_scale,
                random_opening_moves=random_opening_moves,
                record_policy=record_policy,
                persona_pool=persona_pool,
                per_player_personas=per_player_personas,
            )

        # Store commonly accessed attributes directly for performance
        self.batch_size = batch_size
        # Handle board_size if passed as tuple (e.g., (8, 8) instead of 8)
        self.board_size = board_size[0] if isinstance(board_size, tuple) else board_size
        self.num_players = num_players
        self.swap_enabled = swap_enabled
        self.board_type = board_type
        self.use_heuristic_selection = use_heuristic_selection
        self.weight_noise = weight_noise
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.random_opening_moves = random_opening_moves
        self.record_policy = record_policy
        self.persona_pool = persona_pool
        self.per_player_personas = per_player_personas
        # Validation already done in GameRunnerConfig.__post_init__
        self.use_policy_selection = False
        self.policy_model: RingRiftNNUEWithPolicy | None = None
        # Policy recording buffer: {game_idx: [(move_num, policy_dict), ...]}
        self._pending_policy: dict[int, list[tuple[int, dict]]] = {}
        # Default LPS victory rounds to 3 if not specified
        self.lps_victory_rounds = lps_victory_rounds if lps_victory_rounds is not None else 3
        self.rings_per_player = rings_per_player

        if device is None:
            # NOTE: The GPU parallel game simulation is optimized for CUDA.
            # On Apple Silicon, torch MPS is typically *much* slower than CPU for
            # this workload (many small tensor ops + unavoidable sync points).
            # Default to CPU on MPS to keep self-play and tests performant.
            detected = get_device()
            self.device = torch.device("cpu") if detected.type == "mps" else detected
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Resource guard: check GPU/memory before large allocation
        if self.device.type == "cuda":
            if not check_gpu_memory():
                logger.warning("[ParallelGameRunner] GPU memory pressure, clearing cache")
                clear_gpu_memory()
            if not check_memory():
                logger.warning("[ParallelGameRunner] System memory pressure detected")

        # Pre-allocate state buffer
        self.state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=self.board_size,  # Use validated board_size (int, not tuple)
            num_players=num_players,
            device=self.device,
            rings_per_player=rings_per_player,
            board_type=board_type,
            lps_rounds_required=self.lps_victory_rounds,
        )

        # Shadow validation for GPU/CPU parity checking (Phase 2 - move generation)
        # Use async validator by default to prevent GPU blocking during validation
        self.shadow_validator: ShadowValidator | AsyncShadowValidator | None = None
        self._async_shadow_validation = async_shadow_validation
        if shadow_validation:
            if async_shadow_validation:
                self.shadow_validator = create_async_shadow_validator(
                    sample_rate=shadow_sample_rate,
                    threshold=shadow_threshold,
                    enabled=True,
                    max_queue_size=1000,
                )
                logger.info(
                    f"Async shadow validation enabled: sample_rate={shadow_sample_rate}, "
                    f"threshold={shadow_threshold} (non-blocking)"
                )
            else:
                self.shadow_validator = create_shadow_validator(
                    sample_rate=shadow_sample_rate,
                    threshold=shadow_threshold,
                    enabled=True,
                )
                logger.info(
                    f"Shadow validation enabled: sample_rate={shadow_sample_rate}, "
                    f"threshold={shadow_threshold}"
                )

        # State validation for CPU oracle mode (A1 - state parity)
        self.state_validator: StateValidator | None = None
        if state_validation:
            self.state_validator = create_state_validator(
                sample_rate=state_sample_rate,
                threshold=state_threshold,
                enabled=True,
            )
            logger.info(
                f"State validation (CPU oracle) enabled: sample_rate={state_sample_rate}, "
                f"threshold={state_threshold}"
            )

        # Statistics
        self._games_completed = 0
        self._total_moves = 0
        self._total_time = 0.0

        logger.info(
            f"ParallelGameRunner initialized: {batch_size} games, "
            f"{board_size}x{board_size} board, {num_players} players, "
            f"device={self.device}"
        )

    def reset_games(self) -> None:
        """Reset all games to initial state."""
        self.state = BatchGameState.create_batch(
            batch_size=self.batch_size,
            board_size=self.board_size,
            num_players=self.num_players,
            device=self.device,
            rings_per_player=self.rings_per_player,
            board_type=self.board_type,
        )
        # Clear policy recording buffer
        self._pending_policy.clear()

    def get_policy_data(self, game_idx: int) -> list[tuple[int, dict]]:
        """Get recorded policy data for a game.

        Args:
            game_idx: Index of the game

        Returns:
            List of (move_number, policy_dict) tuples where policy_dict contains:
            - candidates: list of {move_type, from_y, from_x, to_y, to_x, score, probability}
            - selected_idx: index of selected move in candidates list
        """
        return self._pending_policy.get(game_idx, [])

    def pop_policy_data(self, game_idx: int) -> list[tuple[int, dict]]:
        """Get and remove recorded policy data for a game.

        Args:
            game_idx: Index of the game

        Returns:
            List of (move_number, policy_dict) tuples
        """
        return self._pending_policy.pop(game_idx, [])

    def clear_policy_data(self, game_idx: int | None = None) -> None:
        """Clear recorded policy data.

        Args:
            game_idx: If provided, clear only this game's data. If None, clear all.
        """
        if game_idx is not None:
            self._pending_policy.pop(game_idx, None)
        else:
            self._pending_policy.clear()

    def set_temperature(self, temperature: float) -> None:
        """Dynamically set the temperature for move selection.

        Higher temperature = more random move selection.
        Useful for difficulty mixing during training data generation.

        Args:
            temperature: New temperature value (e.g., 0.5 for optimal, 4.0 for random)
        """
        self.temperature = temperature

    def load_policy_model(self, model_path: str | None = None) -> bool:
        """Load policy model for policy-based move selection.

        Args:
            model_path: Path to policy model checkpoint. If None, uses default.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            from ..models import BoardType
            from .nnue_policy import RingRiftNNUEWithPolicy

            if model_path is None:
                board_type_str = self.board_type or "square8"
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..",
                    "models", "nnue", f"nnue_policy_{board_type_str}_{self.num_players}p.pt"
                )
                model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                logger.debug(f"Policy model not found at {model_path}")
                return False

            # Use safe loading utility - tries weights_only=True first
            from app.utils.torch_utils import safe_load_checkpoint
            checkpoint = safe_load_checkpoint(model_path, map_location="cpu", warn_on_unsafe=False)
            hidden_dim = checkpoint.get("hidden_dim", 256)
            num_hidden_layers = checkpoint.get("num_hidden_layers", 2)

            # Determine board type
            board_type_str = self.board_type or "square8"
            board_type = BoardType(board_type_str)

            self.policy_model = RingRiftNNUEWithPolicy(
                board_type=board_type,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            self.policy_model.load_state_dict(checkpoint["model_state_dict"])
            self.policy_model.to(self.device)
            self.policy_model.eval()

            # Enable FP16 inference for faster GPU execution if CUDA available
            # Jan 2026: Added proper FP32 fallback for models with extreme weights (e.g., V4)
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    self.policy_model = self.policy_model.half()
                    logger.info("ParallelGameRunner: Enabled FP16 inference for policy model")
                except RuntimeError as e:
                    # FP16 overflow - model weights exceed Half precision range
                    if "Half" in str(e) or "overflow" in str(e):
                        logger.warning(f"ParallelGameRunner: FP16 conversion failed ({e}), using FP32")
                    else:
                        raise
                except Exception as e:
                    logger.debug(f"FP16 inference not available: {e}")

            self.use_policy_selection = True

            logger.info(f"ParallelGameRunner: Loaded policy model from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load policy model: {e}")
            return False

    def reload_policy_model(self, model_path: str | None = None) -> bool:
        """Hot-reload policy model without restarting the runner.

        This enables model hot-swap during selfplay for faster iteration.
        New models can be deployed immediately without restarting games.

        Args:
            model_path: Path to new model checkpoint. If None, uses default path.

        Returns:
            True if model reloaded successfully, False otherwise.
        """
        if self.policy_model is None:
            # No model loaded yet, use full load
            return self.load_policy_model(model_path)

        try:
            from app.utils.torch_utils import safe_load_checkpoint

            if model_path is None:
                board_type_str = self.board_type or "square8"
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..",
                    "models", "nnue", f"nnue_policy_{board_type_str}_{self.num_players}p.pt"
                )
                model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                logger.debug(f"Policy model not found at {model_path}")
                return False

            # Load new state dict
            checkpoint = safe_load_checkpoint(model_path, map_location=self.device, warn_on_unsafe=False)
            new_state_dict = checkpoint["model_state_dict"]

            # Hot-swap weights (in-place update)
            self.policy_model.load_state_dict(new_state_dict)
            self.policy_model.eval()

            # Re-enable FP16 if on CUDA (with FP32 fallback for models with extreme weights)
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    self.policy_model = self.policy_model.half()
                except RuntimeError as e:
                    # FP16 overflow - keep model in FP32
                    if "Half" in str(e) or "overflow" in str(e):
                        logger.debug(f"Hot-reload: FP16 failed ({e}), using FP32")
                    # Other errors silently ignored (fallback to FP32)

            logger.info(f"ParallelGameRunner: Hot-reloaded policy model from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to hot-reload policy model: {e}")
            return False

    def _select_moves(
        self,
        moves: BatchMoves,
        active_mask: torch.Tensor,
        weights_list: list[dict[str, float]] | None = None,
    ) -> torch.Tensor:
        """Select moves using configured selection strategy.

        Selection priority:
        1. Policy-based selection (if policy model loaded)
        2. Heuristic-based selection (if use_heuristic_selection=True)
        3. Fast center-bias selection (default)

        Uses self.temperature for softmax temperature (curriculum learning).
        During opening phase (move_count < random_opening_moves), uses very high
        temperature to make selections nearly uniform random.

        When self.record_policy is True and heuristic selection is used,
        policy data is captured and stored in self._pending_policy.

        Args:
            moves: BatchMoves containing flattened moves for all games
            active_mask: (batch_size,) bool tensor of games to process
            weights_list: Optional per-game weights for heuristic selection
                         (already resolved to current player's weights)
        """
        # Check if any games are in opening phase (need random selection)
        if self.random_opening_moves > 0:
            in_opening = self.state.move_count < self.random_opening_moves
            all_in_opening = (in_opening & active_mask).sum() == active_mask.sum()
            any_in_opening = in_opening.any()

            # If all active games are in opening, use uniform random (high temp)
            if all_in_opening:
                return select_moves_vectorized(
                    moves, active_mask, self.board_size, temperature=100.0
                )

            # If some are in opening but not all, we need to handle them separately
            # For simplicity, use standard selection with slightly elevated temperature
            if any_in_opening:
                # Use elevated temperature to increase randomness during mixed phase
                elevated_temp = max(self.temperature, 3.0)
                if self.use_policy_selection and self.policy_model is not None:
                    return self._select_moves_policy(moves, active_mask, temperature=elevated_temp)
                elif self.use_heuristic_selection:
                    if self.record_policy:
                        return self._select_moves_with_policy_recording(
                            moves, active_mask, temperature=elevated_temp, weights_list=weights_list
                        )
                    return select_moves_heuristic(
                        moves, self.state, active_mask, weights_list=weights_list, temperature=elevated_temp
                    )
                else:
                    return select_moves_vectorized(
                        moves, active_mask, self.board_size, temperature=elevated_temp
                    )

        # Normal selection with configured strategy
        if self.use_policy_selection and self.policy_model is not None:
            return self._select_moves_policy(moves, active_mask, temperature=self.temperature)
        elif self.use_heuristic_selection:
            if self.record_policy:
                return self._select_moves_with_policy_recording(
                    moves, active_mask, temperature=self.temperature, weights_list=weights_list
                )
            return select_moves_heuristic(
                moves, self.state, active_mask, weights_list=weights_list, temperature=self.temperature
            )
        else:
            return select_moves_vectorized(
                moves, active_mask, self.board_size, temperature=self.temperature
            )

    def _select_moves_with_policy_recording(
        self,
        moves: BatchMoves,
        active_mask: torch.Tensor,
        temperature: float = 1.0,
        weights_list: list[dict[str, float]] | None = None,
    ) -> torch.Tensor:
        """Select moves with heuristic scoring and record policy data.

        Uses select_moves_heuristic_with_policy to capture the full probability
        distribution over candidate moves, then stores policy data for each
        active game in self._pending_policy.

        Args:
            moves: BatchMoves containing flattened moves
            active_mask: (batch_size,) bool tensor of active games
            temperature: Softmax temperature
            weights_list: Optional per-game weights for heuristic selection

        Returns:
            (batch_size,) tensor of selected local move indices
        """
        selected, policy_data = select_moves_heuristic_with_policy(
            moves, self.state, active_mask, weights_list=weights_list, temperature=temperature
        )

        # Store policy data for each active game
        # Optimized 2025-12-24: Pre-extract move_count to avoid per-iteration .item() calls
        move_counts_np = self.state.move_count.cpu().numpy()
        moves_per_game_np = moves.moves_per_game.cpu().numpy()
        active_mask_np = active_mask.cpu().numpy()
        for g in range(self.batch_size):
            if active_mask_np[g] and moves_per_game_np[g] > 0:
                move_num = int(move_counts_np[g])
                policy_dict = policy_data.extract_for_game(g)
                if g not in self._pending_policy:
                    self._pending_policy[g] = []
                self._pending_policy[g].append((move_num, policy_dict))

        return selected

    def _select_moves_policy(
        self,
        moves: BatchMoves,
        active_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Select moves using policy network scores (batched).

        Uses vectorized operations for efficient batch inference:
        1. Extract features for all active games in parallel
        2. Run single batch forward pass through policy model
        3. Score and sample all moves using vectorized operations

        Falls back to center-bias if policy evaluation fails.
        """
        device = moves.device
        batch_size = active_mask.shape[0]

        # Initialize output: -1 for games with no moves
        selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

        if moves.total_moves == 0 or self.policy_model is None:
            return selected

        try:
            from ..models import BoardType
            from .nnue import get_feature_dim

            board_type_str = self.board_type or "square8"
            board_type = BoardType(board_type_str)

            # Get active game indices that have moves
            active_indices = torch.where(active_mask)[0]
            moves_per_game = moves.moves_per_game[active_indices]
            games_with_moves = active_indices[moves_per_game > 0]

            if len(games_with_moves) == 0:
                return selected

            # === Batched Feature Extraction ===
            # Extract features for all active games in one pass using vectorized ops
            feature_dim = get_feature_dim(board_type)
            features_batch = self._extract_features_batched(
                games_with_moves, board_type, feature_dim, device
            )

            if features_batch is None:
                raise RuntimeError("Batched feature extraction failed")

            # === Batched Policy Inference ===
            # Convert to half precision if model uses FP16
            model_dtype = next(self.policy_model.parameters()).dtype
            if model_dtype == torch.float16:
                features_batch = features_batch.half()

            with torch.no_grad():
                _, from_logits_batch, to_logits_batch = self.policy_model(
                    features_batch, return_policy=True
                )
                # Convert back to float32 for stable sampling
                from_logits_batch = from_logits_batch.float()
                to_logits_batch = to_logits_batch.float()
                # from_logits_batch: (num_active, H*W)
                # to_logits_batch: (num_active, H*W)

            # === Fully Vectorized Move Scoring (Optimized 2025-12-25) ===
            # Score ALL moves across ALL games in parallel with no per-game loops
            center = self.board_size // 2
            center_idx = center * self.board_size + center
            num_positions = self.board_size * self.board_size

            # Build mapping from global game index to local logits index
            # games_with_moves contains the actual game indices that have moves
            # We need to map each move's game_idx to the corresponding logits index
            game_to_local = torch.full((batch_size,), -1, dtype=torch.long, device=device)
            game_to_local[games_with_moves] = torch.arange(len(games_with_moves), device=device)

            # Get local logits index for each move
            move_game_idx = moves.game_idx.long()
            local_logits_idx = game_to_local[move_game_idx]

            # Mask for moves belonging to games with policy evaluation
            valid_moves_mask = local_logits_idx >= 0

            # Compute flat indices for ALL moves (use center for negative coords)
            from_idx = torch.where(
                moves.from_y >= 0,
                moves.from_y * self.board_size + moves.from_x,
                torch.full_like(moves.from_y, center_idx)
            ).long().clamp(0, num_positions - 1)

            to_idx = torch.where(
                moves.to_y >= 0,
                moves.to_y * self.board_size + moves.to_x,
                torch.full_like(moves.to_y, center_idx)
            ).long().clamp(0, num_positions - 1)

            # Initialize scores with large negative for invalid moves
            scores = torch.full((moves.total_moves,), float('-inf'), device=device)

            # Score valid moves using advanced indexing
            # from_logits_batch[local_logits_idx, from_idx] gives the from_logit for each move
            valid_local_idx = local_logits_idx[valid_moves_mask]
            valid_from_idx = from_idx[valid_moves_mask]
            valid_to_idx = to_idx[valid_moves_mask]

            scores[valid_moves_mask] = (
                from_logits_batch[valid_local_idx, valid_from_idx] +
                to_logits_batch[valid_local_idx, valid_to_idx]
            )

            # === Segment-wise Softmax Sampling (no .item() calls) ===
            # Apply temperature
            scores = scores / temperature

            # Segment-wise softmax: compute max per game for numerical stability
            neg_inf = torch.full((batch_size,), float('-inf'), device=device)
            max_per_game = neg_inf.scatter_reduce(0, move_game_idx, scores, reduce='amax')
            scores_stable = scores - max_per_game[move_game_idx]

            # Compute exp(scores) - invalid moves stay at 0 due to -inf
            exp_scores = torch.exp(scores_stable)

            # Sum exp per game
            sum_per_game = torch.zeros(batch_size, device=device)
            sum_per_game.scatter_add_(0, move_game_idx, exp_scores)

            # Compute probabilities
            probs = exp_scores / (sum_per_game[move_game_idx] + 1e-10)

            # Segment-wise multinomial sampling using cumsum trick
            game_idx = move_game_idx
            is_sorted = (game_idx[1:] >= game_idx[:-1]).all() if moves.total_moves > 1 else True

            if is_sorted:
                sorted_indices = torch.arange(moves.total_moves, device=device)
                sorted_game_idx = game_idx
                sorted_probs = probs
            else:
                sorted_indices = torch.argsort(game_idx)
                sorted_game_idx = game_idx[sorted_indices]
                sorted_probs = probs[sorted_indices]

            cumsum_probs = torch.cumsum(sorted_probs, dim=0)

            game_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
            game_starts[1:] = torch.searchsorted(sorted_game_idx, torch.arange(1, batch_size, device=device))

            cumsum_at_start = torch.zeros(batch_size, device=device)
            cumsum_at_start[1:] = cumsum_probs[game_starts[1:] - 1]

            per_game_cumsum = cumsum_probs - cumsum_at_start[sorted_game_idx]

            # Generate one random value per game
            rand_vals = torch.rand(batch_size, device=device)
            exceeds_rand = per_game_cumsum > rand_vals[sorted_game_idx]

            # Find first exceeding index per game
            large_val = moves.total_moves + 1
            indices_or_large = torch.where(
                exceeds_rand,
                torch.arange(moves.total_moves, device=device, dtype=torch.float32),
                torch.full((moves.total_moves,), float(large_val), device=device)
            )

            first_exceed_f = torch.full((batch_size,), float(large_val), dtype=torch.float32, device=device)
            first_exceed_f.scatter_reduce_(0, sorted_game_idx, indices_or_large, reduce='amin')
            first_exceed = first_exceed_f.long()

            # Map back to local indices
            has_moves = moves.moves_per_game > 0
            valid_selection = has_moves & active_mask & (first_exceed < large_val)

            original_indices = sorted_indices[first_exceed.clamp(0, moves.total_moves - 1)]
            local_idx = original_indices - moves.move_offsets

            selected[valid_selection] = local_idx[valid_selection]

        except Exception as e:
            logger.debug(f"Policy selection failed, falling back to center-bias: {e}")
            return select_moves_vectorized(
                moves, active_mask, self.board_size, temperature=temperature
            )

        return selected

    def _extract_features_batched(
        self,
        game_indices: torch.Tensor,
        board_type: BoardType,
        feature_dim: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Extract NNUE features for multiple games using vectorized operations.

        This is much more efficient than the per-game extraction as it uses
        batch tensor operations instead of Python loops.

        Args:
            game_indices: Tensor of game indices to extract features for
            board_type: Board type enum
            feature_dim: Expected feature dimension
            device: Target device

        Returns:
            Tensor of shape (num_games, feature_dim) or None on failure
        """
        try:
            num_games = len(game_indices)
            board_size = self.board_size
            num_positions = board_size * board_size

            # Initialize features tensor
            features = torch.zeros(
                (num_games, feature_dim), dtype=torch.float32, device=device
            )

            # Get state tensors for selected games
            current_player = self.state.current_player[game_indices]  # (N,)
            current_player = torch.where(
                current_player < 1,
                torch.ones_like(current_player),
                current_player
            )

            # Extract game state slices: (N, H, W)
            stack_owner = self.state.stack_owner[game_indices]
            stack_height = self.state.stack_height[game_indices]
            territory_owner = self.state.territory_owner[game_indices]

            # Create position indices (flat)
            y_coords = torch.arange(board_size, device=device).view(-1, 1).expand(
                board_size, board_size
            )
            x_coords = torch.arange(board_size, device=device).view(1, -1).expand(
                board_size, board_size
            )
            pos_indices = (y_coords * board_size + x_coords).flatten()  # (H*W,)

            # Process each player's perspective
            # For each game g and position (y,x), we compute:
            #   plane_offset = (owner - current_player[g]) % 4
            # Then set appropriate feature planes
            # Optimized 2025-12-14: Pre-extract current_player to avoid .item() in loop
            current_player_np = current_player.cpu().numpy()

            for g_local in range(num_games):
                cp = int(current_player_np[g_local])

                # Ring and stack features
                owner_slice = stack_owner[g_local].flatten()  # (H*W,)
                height_slice = stack_height[g_local].flatten()  # (H*W,)

                # Find occupied positions
                occupied = (owner_slice > 0) & (height_slice > 0)
                occupied_idx = torch.where(occupied)[0]

                if len(occupied_idx) > 0:
                    owners = owner_slice[occupied_idx]
                    heights = height_slice[occupied_idx]
                    positions = pos_indices[occupied_idx]

                    # Compute plane offsets (rotate perspective)
                    plane_offsets = ((owners - cp) % self.num_players).long()

                    # Set ring features (planes 0-3)
                    ring_indices = plane_offsets * num_positions + positions
                    valid_ring = ring_indices < feature_dim
                    features[g_local].scatter_(
                        0, ring_indices[valid_ring], torch.ones_like(ring_indices[valid_ring], dtype=torch.float32)
                    )

                    # Set stack height features (planes 4-7)
                    stack_indices = (4 + plane_offsets) * num_positions + positions
                    valid_stack = stack_indices < feature_dim
                    heights_scaled = torch.clamp(heights.float() / 5.0, 0.0, 1.0)
                    features[g_local].scatter_(
                        0, stack_indices[valid_stack], heights_scaled[valid_stack]
                    )

                # Territory features (planes 8-11)
                territory_slice = territory_owner[g_local].flatten()
                territory_occupied = territory_slice > 0
                territory_idx = torch.where(territory_occupied)[0]

                if len(territory_idx) > 0:
                    territory_owners = territory_slice[territory_idx]
                    territory_positions = pos_indices[territory_idx]

                    territory_offsets = ((territory_owners - cp) % self.num_players).long()
                    territory_plane_indices = (8 + territory_offsets) * num_positions + territory_positions
                    valid_territory = territory_plane_indices < feature_dim
                    features[g_local].scatter_(
                        0,
                        territory_plane_indices[valid_territory],
                        torch.ones_like(
                            territory_plane_indices[valid_territory],
                            dtype=torch.float32,
                        ),
                    )

            return features

        except Exception as e:
            logger.debug(f"Batched feature extraction failed: {e}")
            return None

    def _extract_features_for_game(
        self,
        game_idx: int,
        board_type: BoardType,
    ) -> np.ndarray | None:
        """Extract NNUE features from batch state for a single game.

        This is a simplified implementation that extracts features game-by-game.
        A more efficient implementation would batch this extraction.
        """
        try:
            import numpy as np

            from .nnue import get_feature_dim

            feature_dim = get_feature_dim(board_type)
            features = np.zeros(feature_dim, dtype=np.float32)
            board_size = self.board_size
            num_positions = board_size * board_size

            # Pre-extract to numpy to avoid per-cell .item() syncs (Dec 2025 optimization)
            current_player = int(self.state.current_player[game_idx].cpu().numpy())
            if current_player < 1:
                current_player = 1
            stack_owner_np = self.state.stack_owner[game_idx].cpu().numpy()
            stack_height_np = self.state.stack_height[game_idx].cpu().numpy()
            territory_owner_np = self.state.territory_owner[game_idx].cpu().numpy()

            # Extract stack ownership for each player (simplified)
            # Planes 0-3: Ring presence, 4-7: Stack presence, 8-11: Territory
            for y in range(board_size):
                for x in range(board_size):
                    pos_idx = y * board_size + x
                    owner = int(stack_owner_np[y, x])
                    height = int(stack_height_np[y, x])

                    if owner > 0 and height > 0:
                        # Rotate perspective so current player is always plane 0
                        plane_offset = ((owner - current_player) % self.num_players)

                        # Set ring and stack features
                        ring_plane = plane_offset * num_positions + pos_idx
                        stack_plane = (4 + plane_offset) * num_positions + pos_idx

                        if ring_plane < feature_dim:
                            features[ring_plane] = 1.0
                        if stack_plane < feature_dim:
                            features[stack_plane] = min(float(height) / 5.0, 1.0)

                    # Territory
                    territory_owner = int(territory_owner_np[y, x])
                    if territory_owner > 0:
                        plane_offset = ((territory_owner - current_player) % self.num_players)
                        territory_plane = (8 + plane_offset) * num_positions + pos_idx
                        if territory_plane < feature_dim:
                            features[territory_plane] = 1.0

            return features

        except Exception as e:
            logger.debug(f"Feature extraction failed for game {game_idx}: {e}")
            return None

    def _emit_gpu_selfplay_complete(
        self,
        games_count: int,
        elapsed_seconds: float,
        success: bool = True,
        task_id: str | None = None,
        iteration: int = 0,
        error: str = "",
    ) -> None:
        """Emit GPU_SELFPLAY_COMPLETE event (December 2025).

        This notifies downstream pipeline stages that GPU selfplay has completed,
        enabling coordination with training, evaluation, and data aggregation.

        Args:
            games_count: Number of games completed
            elapsed_seconds: Duration of selfplay in seconds
            success: Whether selfplay completed without error
            task_id: Optional task identifier
            iteration: Pipeline iteration number
            error: Error message if failed
        """
        try:
            import asyncio
            import socket

            from app.coordination.selfplay_orchestrator import emit_selfplay_completion

            node_id = socket.gethostname()
            if task_id is None:
                task_id = f"gpu_selfplay_{self.batch_size}_{int(time.time())}"

            board_type = self.board_type or f"square{self.board_size}"

            async def emit():
                return await emit_selfplay_completion(
                    task_id=task_id,
                    board_type=board_type,
                    num_players=self.num_players,
                    games_generated=games_count,
                    success=success,
                    node_id=node_id,
                    selfplay_type="gpu_selfplay",
                    iteration=iteration,
                    error=error,
                )

            try:
                loop = asyncio.get_running_loop()
                # Fire and forget in existing loop
                loop.create_task(emit())
            except RuntimeError:
                # No event loop running, run synchronously
                asyncio.run(emit())

            logger.debug(
                f"Emitted GPU_SELFPLAY_COMPLETE: {games_count} games, "
                f"{elapsed_seconds:.2f}s, task_id={task_id}"
            )
        except ImportError:
            pass  # SelfplayOrchestrator not available
        except Exception as e:
            logger.debug(f"Failed to emit GPU_SELFPLAY_COMPLETE: {e}")

    @torch.no_grad()
    def run_games(
        self,
        weights_list: list[list[dict[str, float]]] | None = None,
        max_moves: int = 10000,
        callback: Callable[[int, BatchGameState], None] | None = None,
        snapshot_interval: int = 0,
        snapshot_callback: Callable[[int, int, "GameState"], None] | None = None,
        emit_events: bool = True,
        task_id: str | None = None,
        iteration: int = 0,
        temperature_callback: Callable[[int], float] | None = None,
    ) -> dict[str, Any]:
        """Run all games to completion.

        Args:
            weights_list: List of weight dicts (one per game) or None for default
            max_moves: Maximum moves before declaring draw
            callback: Optional callback(move_num, state) after each batch move
            snapshot_interval: If > 0, capture GameState snapshots every N moves per game.
                             Useful for NNUE training with full game trajectories.
            snapshot_callback: Called with (game_idx, move_number, GameState) when a
                             snapshot is captured. Use to save to database.
            emit_events: If True, emit GPU_SELFPLAY_COMPLETE event on completion
                       (December 2025). Default True.
            task_id: Optional task identifier for event emission. Auto-generated if None.
            iteration: Pipeline iteration number for event metadata.
            temperature_callback: Optional callback for per-move temperature scheduling.
                       Called with mean move count across active games, returns new temperature.
                       Use with temperature_scheduling.py schedules:
                           scheduler = LinearDecaySchedule(initial_temp=1.0, final_temp=0.3)
                           callback = lambda move: scheduler.get_temperature(move)
                       January 27, 2026: Phase 2.2 temperature scheduling integration.

        Returns:
            Dictionary with:
                - winners: List of winner player numbers (0 for draw)
                - move_counts: List of move counts per game
                - game_lengths: List of game durations
                - snapshots_captured: Number of snapshots captured (if snapshot_interval > 0)
        """
        self.reset_games()
        start_time = time.perf_counter()

        # Track last snapshot move count per game for interval-based capture
        last_snapshot_move = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        snapshots_captured = 0

        # Use default weights if not provided (with optional noise for diversity)
        if weights_list is None:
            weights_list = self._generate_weights_list()
        elif weights_list and isinstance(weights_list[0], dict):
            # Normalize 1D weights list (one dict per game) to 2D format (per game, per player)
            # This maintains backward compatibility with callers that pass [weights] * batch_size
            weights_list = [[w] * self.num_players for w in weights_list]

        # In this runner, a single call to ``_step_games`` advances one phase
        # (ring_placement, movement, line_processing, territory_processing, end_turn).
        # ``max_moves`` is interpreted as a per-game *move record* limit (i.e.
        # ``BatchGameState.move_count``), not a phase-step limit. To prevent
        # pathological stalls where ``move_count`` does not advance, we also
        # apply a conservative safety cap on phase steps.
        phase_steps = 0
        max_phase_steps = max_moves * 20

        while self.state.count_active() > 0 and phase_steps < max_phase_steps:
            active_mask = self.state.get_active_mask()

            # Enforce per-game move limit based on the recorded move_count.
            reached_limit = active_mask & (self.state.move_count >= max_moves)
            if reached_limit.any():
                self.state.game_status[reached_limit] = GameStatus.MAX_MOVES

            if self.state.count_active() == 0:
                break

            # Generate and apply moves for all remaining active games.
            self._step_games(weights_list)

            phase_steps += 1
            if callback:
                callback(phase_steps, self.state)

            # Update temperature based on mean move count (Jan 27, 2026 - Phase 2.2)
            if temperature_callback is not None:
                active_moves = self.state.move_count[active_mask]
                if active_moves.numel() > 0:
                    mean_move = int(active_moves.float().mean().item())
                    new_temp = temperature_callback(mean_move)
                    if new_temp != self.temperature:
                        self.set_temperature(new_temp)

            # Capture snapshots for games that have crossed the interval threshold
            if snapshot_interval > 0 and snapshot_callback:
                current_moves = self.state.move_count
                # Find games where move_count has advanced past last_snapshot + interval
                need_snapshot = (current_moves >= last_snapshot_move + snapshot_interval) & active_mask
                if need_snapshot.any():
                    # Pre-extract to numpy to avoid per-game .item() syncs (Dec 2025 optimization)
                    current_moves_np = current_moves.cpu().numpy()
                    for g in need_snapshot.nonzero(as_tuple=True)[0].tolist():
                        try:
                            move_num = int(current_moves_np[g])
                            game_state = self.state.to_game_state(g)
                            snapshot_callback(g, move_num, game_state)
                            last_snapshot_move[g] = move_num
                            snapshots_captured += 1
                        except Exception as e:
                            logger.debug(f"Snapshot capture failed for game {g}: {e}")

            # Check for game completion.
            self._check_victory_conditions()

        # Mark any remaining active games as max-moves timeouts.
        active_mask = self.state.get_active_mask()
        self.state.game_status[active_mask] = GameStatus.MAX_MOVES

        elapsed = time.perf_counter() - start_time
        self._games_completed += self.batch_size
        self._total_moves += self.state.move_count.sum().item()
        self._total_time += elapsed

        # Extract move histories using optimized batch extraction (10x faster)
        move_histories = self.state.extract_move_history_batch_dicts()

        # Victory types using optimized batch extraction (5x faster)
        victory_types, stalemate_tiebreakers = self.state.derive_victory_types_batch(max_moves)

        # Build results
        results = {
            "winners": self.state.winner.cpu().tolist(),
            "move_counts": self.state.move_count.cpu().tolist(),
            "status": self.state.game_status.cpu().tolist(),
            "move_histories": move_histories,
            "victory_types": victory_types,
            "stalemate_tiebreakers": stalemate_tiebreakers,
            "elapsed_seconds": elapsed,
            "games_per_second": self.batch_size / elapsed,
            "snapshots_captured": snapshots_captured,
        }

        # Add validation reports if enabled
        if self.shadow_validator:
            results["shadow_validation"] = self.shadow_validator.get_report()
        if self.state_validator:
            results["state_validation"] = self.state_validator.get_report()

        # Emit GPU_SELFPLAY_COMPLETE event for pipeline coordination (December 2025)
        if emit_events:
            self._emit_gpu_selfplay_complete(
                games_count=self.batch_size,
                elapsed_seconds=elapsed,
                success=True,
                task_id=task_id,
                iteration=iteration,
            )

        return results

    def get_validation_reports(self) -> dict[str, Any]:
        """Get validation reports from both shadow and state validators.

        Returns:
            Dictionary with validation reports and combined status.
        """
        reports = {}

        if self.shadow_validator:
            reports["shadow_validation"] = self.shadow_validator.get_report()

        if self.state_validator:
            reports["state_validation"] = self.state_validator.get_report()

        # Compute combined status
        all_passed = True
        if self.shadow_validator and self.shadow_validator.stats.divergence_rate > self.shadow_validator.threshold:
            all_passed = False
        if self.state_validator and self.state_validator.stats.divergence_rate > self.state_validator.threshold:
            all_passed = False

        reports["combined_status"] = "PASS" if all_passed else "FAIL"
        return reports

    def reset_validation_stats(self) -> None:
        """Reset all validation statistics."""
        if self.shadow_validator:
            self.shadow_validator.reset_stats()
        if self.state_validator:
            self.state_validator.reset_stats()

    def _step_games(
        self, weights_list: list[list[dict[str, float]]] | list[dict[str, float]] | None = None
    ) -> None:
        """Execute one phase step for all active games using full rules FSM.

        Phase flow per turn (per RR-CANON):
        RING_PLACEMENT -> MOVEMENT -> LINE_PROCESSING -> TERRITORY_PROCESSING -> END_TURN

        Each call to _step_games processes ONE phase for all active games,
        then advances to the next phase.
        """
        active_mask = self.state.get_active_mask()

        if not active_mask.any():
            return

        # Snapshot phases at the start of the step so a single call processes
        # at most one phase per game. (If we re-read current_phase after each
        # handler, games that advance phases could be processed multiple times
        # per call, which breaks RR-CANON-R172 round-based LPS timing and makes
        # tests non-deterministic.)
        phase_snapshot = self.state.current_phase.clone()

        # Process each phase type separately based on current phase
        # Games may be in different phases, so we handle each group

        # PHASE: RING_PLACEMENT (0)
        placement_mask = active_mask & (phase_snapshot == GamePhase.RING_PLACEMENT)
        if placement_mask.any():
            self._step_placement_phase(placement_mask, weights_list)

        # PHASE: MOVEMENT (1)
        movement_mask = active_mask & (phase_snapshot == GamePhase.MOVEMENT)
        if movement_mask.any():
            self._step_movement_phase(movement_mask, weights_list)

        # PHASE: LINE_PROCESSING (2)
        line_mask = active_mask & (phase_snapshot == GamePhase.LINE_PROCESSING)
        if line_mask.any():
            self._step_line_phase(line_mask)

        # PHASE: TERRITORY_PROCESSING (3)
        territory_mask = active_mask & (phase_snapshot == GamePhase.TERRITORY_PROCESSING)
        if territory_mask.any():
            self._step_territory_phase(territory_mask)

        # PHASE: FORCED_ELIMINATION (8) - December 2025
        # Games that triggered forced elimination proceed directly to END_TURN
        forced_elim_mask = active_mask & (phase_snapshot == GamePhase.FORCED_ELIMINATION)
        if forced_elim_mask.any():
            # Forced elimination move was already recorded, just advance to END_TURN
            self.state.current_phase[forced_elim_mask] = GamePhase.END_TURN

        # PHASE: END_TURN (4)
        end_turn_mask = active_mask & (phase_snapshot == GamePhase.END_TURN)
        if end_turn_mask.any():
            self._step_end_turn_phase(end_turn_mask)

    def _validate_placement_moves_sample(
        self,
        moves: BatchMoves,
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of placement moves against CPU rules.

        Called when shadow_validator is enabled. Samples games probabilistically
        and validates GPU-generated moves against canonical CPU implementation.
        """
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()

        # Optimized 2025-12-24: Pre-extract move metadata to avoid per-iteration .item() calls
        move_offsets_np = moves.move_offsets.cpu().numpy()
        moves_per_game_np = moves.moves_per_game.cpu().numpy()
        from_y_np = moves.from_y.cpu().numpy()
        from_x_np = moves.from_x.cpu().numpy()
        current_player_np = self.state.current_player.cpu().numpy()

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Extract GPU moves for this game
            move_start = int(move_offsets_np[g])
            move_count = int(moves_per_game_np[g])

            if move_count == 0:
                continue

            gpu_positions = []
            for i in range(move_count):
                idx = move_start + i
                # Placement moves store position in from_y, from_x (target position)
                row = int(from_y_np[idx])
                col = int(from_x_np[idx])

                # Convert GPU grid coords to CPU format
                # For hex boards (25x25 embedding): convert to axial coords
                # CPU axial (x, y) = GPU grid (col - center, row - center)
                if self.board_type and self.board_type.lower() in ("hexagonal", "hex"):
                    center = self.board_size // 2  # 12 for 25x25
                    x = col - center
                    y = row - center
                else:
                    # Square boards: grid coords match directly
                    x = col
                    y = row

                gpu_positions.append((x, y))

            # Convert to CPU state and validate
            cpu_state = self.state.to_game_state(g)
            player = int(current_player_np[g])

            self.shadow_validator.validate_placement_moves(
                gpu_positions, cpu_state, player
            )

    def _validate_movement_moves_sample(
        self,
        movement_moves: BatchMoves,
        capture_moves: BatchMoves,
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of movement/capture moves against CPU rules.

        Called when shadow_validator is enabled. Validates both movement and
        capture move generation against canonical CPU implementation.
        """
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()

        # Optimized 2025-12-24: Pre-extract move metadata to avoid per-iteration .item() calls
        # Movement moves
        mv_offsets_np = movement_moves.move_offsets.cpu().numpy()
        mv_counts_np = movement_moves.moves_per_game.cpu().numpy()
        mv_from_y_np = movement_moves.from_y.cpu().numpy()
        mv_from_x_np = movement_moves.from_x.cpu().numpy()
        mv_to_y_np = movement_moves.to_y.cpu().numpy()
        mv_to_x_np = movement_moves.to_x.cpu().numpy()
        # Capture moves
        cap_offsets_np = capture_moves.move_offsets.cpu().numpy()
        cap_counts_np = capture_moves.moves_per_game.cpu().numpy()
        cap_from_y_np = capture_moves.from_y.cpu().numpy()
        cap_from_x_np = capture_moves.from_x.cpu().numpy()
        cap_to_y_np = capture_moves.to_y.cpu().numpy()
        cap_to_x_np = capture_moves.to_x.cpu().numpy()
        # Current player
        current_player_np = self.state.current_player.cpu().numpy()

        # Hex coordinate conversion helper (defined once outside loop)
        is_hex = self.board_type and self.board_type.lower() in ("hexagonal", "hex", "hex8")
        hex_center = self.board_size // 2 if is_hex else 0

        def to_cpu_coords(row: int, col: int):
            """Convert GPU grid to CPU coords."""
            if is_hex:
                return col - hex_center, row - hex_center
            return col, row

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Extract GPU movement moves
            move_start = int(mv_offsets_np[g])
            move_count = int(mv_counts_np[g])

            gpu_movement = []
            for i in range(move_count):
                idx = move_start + i
                from_row = int(mv_from_y_np[idx])
                from_col = int(mv_from_x_np[idx])
                to_row = int(mv_to_y_np[idx])
                to_col = int(mv_to_x_np[idx])
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_movement.append(((from_x, from_y), (to_x, to_y)))

            # Extract GPU capture moves
            cap_start = int(cap_offsets_np[g])
            cap_count = int(cap_counts_np[g])

            gpu_captures = []
            for i in range(cap_count):
                idx = cap_start + i
                from_row = int(cap_from_y_np[idx])
                from_col = int(cap_from_x_np[idx])
                to_row = int(cap_to_y_np[idx])
                to_col = int(cap_to_x_np[idx])
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_captures.append(((from_x, from_y), (to_x, to_y)))

            # Convert to CPU state and validate
            cpu_state = self.state.to_game_state(g)
            player = int(current_player_np[g])

            if gpu_movement:
                self.shadow_validator.validate_movement_moves(
                    gpu_movement, cpu_state, player
                )

            if gpu_captures:
                self.shadow_validator.validate_capture_moves(
                    gpu_captures, cpu_state, player
                )

    def _player_has_real_action_gpu(self, g: int, player: int) -> bool:
        """Return True if player has any RR-CANON-R172 real action in game g.

        Real actions are:
        - any legal placement (approximated here as ``rings_in_hand > 0``), OR
        - any legal non-capture movement or overtaking capture.

        Recovery and forced elimination do NOT count as real actions.

        Note: For better performance, use _check_real_actions_batch() which
        batches move generation across multiple games.

        Optimized Dec 2025: Pre-extract to numpy to minimize .item() calls.
        """
        # Pre-extract to numpy (Dec 2025 optimization)
        rings_in_hand_np = self.state.rings_in_hand[g].cpu().numpy()
        stack_owner_np = self.state.stack_owner[g].cpu().numpy()
        current_player_np = int(self.state.current_player[g].cpu().numpy())

        # Placement: treat any remaining rings in hand as a real action.
        if rings_in_hand_np[player] > 0:
            return True

        # Without controlled stacks, there is no movement/capture.
        if not (stack_owner_np == player).any():
            return False

        prev_player = current_player_np
        self.state.current_player[g] = player
        try:
            single_mask = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            single_mask[g] = True
            movement_moves = generate_movement_moves_batch(self.state, single_mask)
            capture_moves = generate_capture_moves_batch(self.state, single_mask)
            # Pre-extract move counts (Dec 2025 optimization)
            movement_count = int(movement_moves.moves_per_game[g].cpu().numpy())
            capture_count = int(capture_moves.moves_per_game[g].cpu().numpy())
            return movement_count > 0 or capture_count > 0
        finally:
            self.state.current_player[g] = prev_player

    def _check_real_actions_batch(
        self,
        game_mask: torch.Tensor,
        player: int,
    ) -> torch.Tensor:
        """Check if player has real actions in all games specified by mask.

        Batched version of _player_has_real_action_gpu for better GPU performance.
        Runs move generation once for all games instead of per-game.

        Args:
            game_mask: Boolean mask of games to check
            player: Player number to check for real actions

        Returns:
            Boolean tensor (batch_size,) indicating which games have real actions
        """
        device = self.device
        batch_size = self.batch_size

        # Initialize result - False for all games
        has_action = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if not game_mask.any():
            return has_action

        # Check 1: Rings in hand counts as real action (vectorized)
        rings_check = self.state.rings_in_hand[:, player] > 0
        has_action = has_action | (game_mask & rings_check)

        # Games that still need move generation check
        needs_move_check = game_mask & ~has_action

        if not needs_move_check.any():
            return has_action

        # Check 2: Must control at least one stack (vectorized)
        controls_stack = (self.state.stack_owner == player).flatten(1).any(dim=1)
        needs_move_check = needs_move_check & controls_stack

        if not needs_move_check.any():
            return has_action

        # Check 3: Generate moves for remaining games (batched)
        # Save and set current_player for all games being checked
        prev_players = self.state.current_player.clone()
        self.state.current_player[needs_move_check] = player

        try:
            # Generate moves for all games at once
            movement_moves = generate_movement_moves_batch(self.state, needs_move_check)
            capture_moves = generate_capture_moves_batch(self.state, needs_move_check)

            # Check which games have moves (vectorized)
            has_movement = movement_moves.moves_per_game > 0
            has_capture = capture_moves.moves_per_game > 0
            has_action = has_action | (needs_move_check & (has_movement | has_capture))
        finally:
            # Restore original players
            self.state.current_player = prev_players

        return has_action

    def _maybe_apply_lps_victory_at_turn_start(
        self,
        mask: torch.Tensor,
        player_has_rings: torch.Tensor | None = None,
    ) -> None:
        """Apply RR-CANON-R172 LPS victory at the start of a player's turn.

        This is called after updating round tracking in ``ring_placement``.
        We only run the expensive real-action check when a candidate has
        already achieved the required consecutive exclusive rounds.

        Optimized for GPU/MPS: Uses batched move generation per-player instead
        of per-game to minimize CPU-GPU synchronization.
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        lps_required = self.lps_victory_rounds
        candidate_mask = (
            active_mask
            & (self.state.lps_consecutive_exclusive_rounds >= lps_required)
            & (self.state.lps_consecutive_exclusive_player > 0)
            & (
                self.state.current_player
                == self.state.lps_consecutive_exclusive_player
            )
        )
        if not candidate_mask.any():
            return

        if player_has_rings is None:
            player_has_rings = self._compute_player_ring_status_batch()

        # === BATCHED APPROACH ===
        # Instead of checking per-game, we batch by player:
        # 1. For each player, check which candidate games need that player checked
        # 2. Run batched move generation once per player
        # 3. Aggregate results

        # Pre-compute real actions for each player across all candidate games
        # Shape: (num_players + 1, batch_size) - player 0 unused
        real_actions_by_player = {}
        for pid in range(1, self.num_players + 1):
            # Only check games where this player has rings (optimization)
            games_to_check = candidate_mask & (player_has_rings[:, pid] > 0)
            if games_to_check.any():
                real_actions_by_player[pid] = self._check_real_actions_batch(
                    games_to_check, pid
                )
            else:
                real_actions_by_player[pid] = torch.zeros(
                    self.batch_size, dtype=torch.bool, device=self.device
                )

        # Now determine winners using the pre-computed results (fully vectorized)
        candidates = self.state.lps_consecutive_exclusive_player

        # For each candidate game, check if candidate has action and others don't
        for pid in range(1, self.num_players + 1):
            # Games where this player is the candidate
            is_candidate = candidate_mask & (candidates == pid)
            if not is_candidate.any():
                continue

            # Candidate must have real action
            candidate_has_action = real_actions_by_player[pid]
            valid_candidates = is_candidate & candidate_has_action

            if not valid_candidates.any():
                continue

            # Check if any other player has real action
            others_have_action = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            for other_pid in range(1, self.num_players + 1):
                if other_pid == pid:
                    continue
                # Only consider players with rings
                other_has_rings = player_has_rings[:, other_pid] > 0
                others_have_action = others_have_action | (
                    other_has_rings & real_actions_by_player[other_pid]
                )

            # Winner: candidate has action AND no others have action
            winners = valid_candidates & ~others_have_action

            if winners.any():
                self.state.winner[winners] = pid
                self.state.game_status[winners] = GameStatus.COMPLETED

    def _update_lps_round_tracking_for_current_player(
        self,
        mask: torch.Tensor,
    ) -> None:
        """Update LPS round tracking (RR-CANON-R172) for games in mask.

        Mirrors the CPU/TS approach:
        - Track the first player of the current round.
        - Mark each non-permanently-eliminated player as "seen" once per round.
        - Record whether that player had any real action available at the start
          of their turn (placements count; recovery does not).
        - When cycling back to the first player, finalize the previous round and
          update consecutive-exclusive round counters.
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        player_has_rings = self._compute_player_ring_status_batch()
        current = self.state.current_player
        first = self.state.lps_current_round_first_player

        first_has_rings = torch.gather(
            player_has_rings,
            dim=1,
            index=first.unsqueeze(1).long(),
        ).squeeze(1)

        starting_new_cycle = active_mask & ((first == 0) | (~first_has_rings))

        round_has_entries = self.state.lps_current_round_seen_mask.any(dim=1)
        finalize_round = (
            active_mask
            & (~starting_new_cycle)
            & (current == first)
            & round_has_entries
        )

        if starting_new_cycle.any():
            idx = starting_new_cycle
            self.state.lps_round_index[idx] += 1
            self.state.lps_current_round_first_player[idx] = current[idx]
            self.state.lps_current_round_seen_mask[idx] = False
            self.state.lps_current_round_real_action_mask[idx] = False
            self.state.lps_exclusive_player_for_completed_round[idx] = 0

            # Reset consecutive tracking only if the prior exclusive player
            # also dropped out (per TS semantics).
            excl = self.state.lps_consecutive_exclusive_player
            excl_has_rings = torch.gather(
                player_has_rings,
                dim=1,
                index=excl.unsqueeze(1).long(),
            ).squeeze(1)
            reset_consecutive = idx & (~excl_has_rings)
            if reset_consecutive.any():
                self.state.lps_consecutive_exclusive_rounds[reset_consecutive] = 0
                self.state.lps_consecutive_exclusive_player[reset_consecutive] = 0

        if finalize_round.any():
            idx = finalize_round
            eligible = player_has_rings & self.state.lps_current_round_seen_mask
            real_action_players = eligible & self.state.lps_current_round_real_action_mask

            true_counts = real_action_players.sum(dim=1).to(torch.int16)

            # FIX 2025-12-26: Use masked multiplication to find exclusive player ID.
            # argmax on boolean tensor is semantically incorrect - it returns the first
            # True index, not necessarily the exclusive player. Use masked multiplication
            # with player indices instead to correctly identify the single True player.
            device = real_action_players.device
            player_indices = torch.arange(
                real_action_players.shape[1], device=device, dtype=torch.int8
            ).unsqueeze(0)  # Shape: (1, num_players+1)
            # Multiply boolean mask by indices - only the True player's index survives
            masked_indices = real_action_players.to(torch.int8) * player_indices
            # Take max to get the exclusive player ID (works because only one is non-zero)
            exclusive_pid_candidates = masked_indices.max(dim=1).values.to(torch.int8)

            exclusive_pid = torch.where(
                true_counts == 1,
                exclusive_pid_candidates,
                torch.zeros_like(exclusive_pid_candidates),
            )

            self.state.lps_exclusive_player_for_completed_round[idx] = exclusive_pid[idx]

            has_exclusive = idx & (exclusive_pid != 0)
            same_exclusive = has_exclusive & (
                exclusive_pid == self.state.lps_consecutive_exclusive_player
            )
            if same_exclusive.any():
                self.state.lps_consecutive_exclusive_rounds[same_exclusive] += 1

            diff_exclusive = has_exclusive & (~same_exclusive)
            if diff_exclusive.any():
                self.state.lps_consecutive_exclusive_player[diff_exclusive] = exclusive_pid[diff_exclusive]
                self.state.lps_consecutive_exclusive_rounds[diff_exclusive] = 1

            no_exclusive = idx & (exclusive_pid == 0)
            if no_exclusive.any():
                self.state.lps_consecutive_exclusive_rounds[no_exclusive] = 0
                self.state.lps_consecutive_exclusive_player[no_exclusive] = 0

            # Start a new round from the current (first) player.
            self.state.lps_round_index[idx] += 1
            self.state.lps_current_round_first_player[idx] = current[idx]
            self.state.lps_current_round_seen_mask[idx] = False
            self.state.lps_current_round_real_action_mask[idx] = False

        # Record that the current player started their turn in this round.
        game_indices = torch.where(active_mask)[0]
        player_indices = current[game_indices].long()
        self.state.lps_current_round_seen_mask[game_indices, player_indices] = True

        # Record initial real-action availability from placements only.
        # Movement/capture availability is filled in during MOVEMENT phase.
        rings_for_current = torch.gather(
            self.state.rings_in_hand,
            dim=1,
            index=current.unsqueeze(1).long(),
        ).squeeze(1)
        has_placement = rings_for_current > 0
        self.state.lps_current_round_real_action_mask[
            game_indices, player_indices
        ] |= has_placement[game_indices]

        # Apply LPS victory at the start of the candidate's next turn.
        self._maybe_apply_lps_victory_at_turn_start(active_mask, player_has_rings)

    def _step_placement_phase(
        self,
        mask: torch.Tensor,
        weights_list: list[list[dict[str, float]]] | list[dict[str, float]] | None = None,
    ) -> None:
        """Handle RING_PLACEMENT phase for games in mask.

        Per RR-CANON-R073, every turn begins in ``ring_placement``. If the current
        player has a legal placement they may place (we generate a single-ring
        placement for simplicity). Otherwise they proceed to ``movement``.

        Note: Placement is part of the player's turn; player rotation happens in
        ``END_TURN``.
        """
        # RR-CANON-R172: update round tracking at the start of the player's
        # turn (in ring_placement). LPS victory is applied here when eligible.
        self._update_lps_round_tracking_for_current_player(mask)

        # Some games may have ended by LPS; drop them from this phase step.
        mask = mask & self.state.get_active_mask()
        if not mask.any():
            return

        # Check which games have rings to place (vectorized).
        current_players = self.state.current_player  # (batch_size,)
        rings_for_current_player = torch.gather(
            self.state.rings_in_hand,
            dim=1,
            index=current_players.unsqueeze(1).long()
        ).squeeze(1)

        # If the player is recovery-eligible, allow movement so they can take
        # a recovery action (RR-CANON-R110): skipping placement is permitted
        # even when rings remain in hand.
        player_expanded = current_players.view(self.batch_size, 1, 1).expand_as(self.state.stack_owner)
        controls_stack = (self.state.stack_owner == player_expanded).flatten(1).any(dim=1)
        has_marker = (self.state.marker_owner == player_expanded).flatten(1).any(dim=1)
        buried_for_current = torch.gather(
            self.state.buried_rings,
            dim=1,
            index=current_players.unsqueeze(1).long(),
        ).squeeze(1)
        recovery_eligible = mask & (~controls_stack) & has_marker & (buried_for_current > 0)

        # December 2025: BUG FIX - Per RR-CANON-R086, recovery-eligible players with rings
        # MAY choose to place or skip. Previously, we forced skip_placement for all recovery-
        # eligible players, which violated canonical rules and caused GPU-CPU parity failures.
        # Now recovery-eligible players are included in games_with_rings and will place normally.
        # Skip_placement for recovery is handled when no valid placements exist.
        games_with_rings = mask & (rings_for_current_player > 0)
        # Games with no rings: no_placement_action
        games_no_placement = mask & (rings_for_current_player == 0)

        # Games WITH rings: generate and apply placement moves
        if games_with_rings.any():
            moves = generate_placement_moves_batch(self.state, games_with_rings)

            # Shadow validation: validate move generation against CPU
            self._validate_placement_moves_sample(moves, games_with_rings)

            if moves.total_moves > 0:
                # Use configured move selection strategy with per-game weights
                per_game_weights = self.get_weights_for_current_players(weights_list)
                selected = self._select_moves(moves, games_with_rings, per_game_weights)
                apply_placement_moves_batch(self.state, selected, moves)

                # December 2025: Track real action for forced elimination detection
                # Placement is a real action (not bookkeeping)
                games_placed = games_with_rings & (moves.moves_per_game > 0)
                mark_real_action_batch(self.state, games_placed)

        # Games with no rings: no_placement_action
        if games_no_placement.any():
            apply_no_action_moves_batch(self.state, games_no_placement)

        # Advance to MOVEMENT for this player's turn regardless of whether a
        # placement occurred (no legal placements, no rings in hand, or a
        # strategic skip to enable recovery).
        self.state.current_phase[mask] = GamePhase.MOVEMENT

    def _record_skip_placement_moves(self, mask: torch.Tensor) -> None:
        """Record skip_placement moves for canonical compliance.

        Per canonical contract, when a player skips placement (e.g., to enable
        recovery), an explicit skip_placement move must be recorded.

        Args:
            mask: Games where skip_placement should be recorded

        December 2025: Vectorized to avoid per-game .item() calls.
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        move_counts = self.state.move_count[game_indices]
        players = self.state.current_player[game_indices]

        # Filter games where move_count < max_history_moves (vectorized)
        valid_mask = move_counts < self.state.max_history_moves
        if not valid_mask.any():
            return

        valid_games = game_indices[valid_mask]
        valid_move_counts = move_counts[valid_mask].long()
        valid_players = players[valid_mask]

        # Vectorized move history update using advanced indexing
        # move_history shape: (num_games, max_history_moves, 7)
        self.state.move_history[valid_games, valid_move_counts, 0] = MoveType.SKIP_PLACEMENT
        self.state.move_history[valid_games, valid_move_counts, 1] = valid_players
        self.state.move_history[valid_games, valid_move_counts, 2] = -1  # No position for skip
        self.state.move_history[valid_games, valid_move_counts, 3] = -1
        self.state.move_history[valid_games, valid_move_counts, 4] = -1
        self.state.move_history[valid_games, valid_move_counts, 5] = -1
        self.state.move_history[valid_games, valid_move_counts, 6] = GamePhase.RING_PLACEMENT

        self.state.move_count[valid_games] += 1

    def _record_skip_capture_moves(self, mask: torch.Tensor) -> None:
        """Record skip_capture moves for canonical compliance.

        Per canonical contract, when a player finishes capturing (no more chain
        captures available), an explicit skip_capture move must be recorded before
        transitioning to line_processing.

        December 2025: Added for GPU/CPU phase machine parity.
        December 2025: Vectorized to avoid per-game .item() calls.

        Args:
            mask: Games where skip_capture should be recorded
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        move_counts = self.state.move_count[game_indices]
        players = self.state.current_player[game_indices]

        # Filter games where move_count < max_history_moves (vectorized)
        valid_mask = move_counts < self.state.max_history_moves
        if not valid_mask.any():
            return

        valid_games = game_indices[valid_mask]
        valid_move_counts = move_counts[valid_mask].long()
        valid_players = players[valid_mask]

        # Vectorized move history update using advanced indexing
        # move_history shape: (num_games, max_history_moves, 7)
        self.state.move_history[valid_games, valid_move_counts, 0] = MoveType.SKIP_CAPTURE
        self.state.move_history[valid_games, valid_move_counts, 1] = valid_players
        self.state.move_history[valid_games, valid_move_counts, 2] = -1  # No position for skip
        self.state.move_history[valid_games, valid_move_counts, 3] = -1
        self.state.move_history[valid_games, valid_move_counts, 4] = -1
        self.state.move_history[valid_games, valid_move_counts, 5] = -1
        # Use CAPTURE phase for skip_capture (chain_capture would also be valid)
        self.state.move_history[valid_games, valid_move_counts, 6] = GamePhase.CAPTURE

        self.state.move_count[valid_games] += 1

    def _step_movement_phase(
        self,
        mask: torch.Tensor,
        weights_list: list[list[dict[str, float]]] | list[dict[str, float]] | None = None,
    ) -> None:
        """Handle MOVEMENT phase for games in mask.

        Generate both non-capture movement and capture moves,
        select the best one, and apply it.

        Refactored 2025-12-11 to use vectorized selection and application:
        - select_moves_vectorized() for parallel move selection
        - apply_*_moves_vectorized() for batched move application
        - Reduces per-game Python loops and .item() calls
        - See GPU_PIPELINE_ROADMAP.md Section 2.1 (False Parallelism) for context
        """
        # Check which games have stacks to move (vectorized)
        # Create mask for each player slot and check ownership
        current_players = self.state.current_player  # (batch_size,)

        # Expand current_player to match stack_owner shape for comparison
        # stack_owner shape: (batch_size, board_size, board_size)
        player_expanded = current_players.view(self.batch_size, 1, 1).expand_as(self.state.stack_owner)
        has_any_stack = (self.state.stack_owner == player_expanded).flatten(1).any(dim=1)
        has_stacks = mask & has_any_stack

        games_with_stacks = has_stacks
        games_without_stacks = mask & ~has_stacks

        # Games WITHOUT stacks: check for recovery moves
        if games_without_stacks.any():
            recovery_moves = generate_recovery_moves_batch(self.state, games_without_stacks)

            # Identify games with recovery moves (vectorized)
            games_with_recovery = games_without_stacks & (recovery_moves.moves_per_game > 0)
            games_no_recovery = games_without_stacks & (recovery_moves.moves_per_game == 0)

            # Apply recovery moves using configured selection strategy
            if games_with_recovery.any():
                per_game_weights = self.get_weights_for_current_players(weights_list)
                selected_recovery = self._select_moves(recovery_moves, games_with_recovery, per_game_weights)
                apply_recovery_moves_vectorized(
                    self.state, selected_recovery, recovery_moves, games_with_recovery
                )
                # December 2025 BUG FIX: Players who did recovery were NOT blocked -
                # they successfully took an action. Mark as "real action" to prevent
                # incorrect forced_elimination after stack_strike gains stacks.
                # Per RR-CANON-R112, recovery is a valid action for recovery-eligible
                # players, not a blocked state.
                mark_real_action_batch(self.state, games_with_recovery)

            # No movement/capture/recovery action is available for the current player.
            # Do not treat this as an immediate stalemate (RR-CANON-R173 is global and
            # evaluated only for bare-board terminality); instead, record an explicit
            # NO_ACTION move and allow turn/round machinery (including LPS) to proceed.
            if games_no_recovery.any():
                apply_no_action_moves_batch(self.state, games_no_recovery)

        # Games WITH stacks: generate movement and capture moves
        if games_with_stacks.any():
            # Generate non-capture movement moves
            movement_moves = generate_movement_moves_batch(self.state, games_with_stacks)

            # Generate capture moves
            capture_moves = generate_capture_moves_batch(self.state, games_with_stacks)

            # RR-CANON-R172: movement/capture availability counts as a "real action"
            # for LPS purposes; recovery does not. Record availability at the
            # start of MOVEMENT (before applying any move).
            has_real_action = (movement_moves.moves_per_game > 0) | (capture_moves.moves_per_game > 0)
            lps_game_indices = torch.where(games_with_stacks)[0]
            if len(lps_game_indices) > 0:
                lps_players = current_players[lps_game_indices].long()
                self.state.lps_current_round_real_action_mask[
                    lps_game_indices, lps_players
                ] |= has_real_action[lps_game_indices]

            # Shadow validation: validate move generation against CPU
            self._validate_movement_moves_sample(movement_moves, capture_moves, games_with_stacks)

            # Identify which games have captures (prefer captures per RR-CANON)
            # Per RR-CANON-R103: After executing a capture, if additional legal captures
            # exist from the new landing position, the chain MUST continue.
            #
            # BUG FIX December 2025: If must_move_from is set (after a placement), the player
            # can ONLY move/capture FROM that specific position. Captures from other positions
            # are blocked, but captures FROM must_move_from are allowed.
            has_must_move_constraint = self.state.must_move_from_y >= 0  # (batch_size,)

            # For games with must_move_from constraint, check if there are captures FROM that position
            # Optimized 2025-12-24: Pre-extract to numpy to avoid per-iteration .item() calls
            games_with_constrained_captures = torch.zeros_like(games_with_stacks)
            if has_must_move_constraint.any():
                cap_moves_per_game_np = capture_moves.moves_per_game.cpu().numpy()
                cap_offsets_np = capture_moves.move_offsets.cpu().numpy()
                cap_from_y_np = capture_moves.from_y.cpu().numpy()
                cap_from_x_np = capture_moves.from_x.cpu().numpy()
                must_y_np = self.state.must_move_from_y.cpu().numpy()
                must_x_np = self.state.must_move_from_x.cpu().numpy()

                for g in torch.where(has_must_move_constraint & games_with_stacks)[0].tolist():
                    if cap_moves_per_game_np[g] == 0:
                        continue
                    # Check if any capture originates from must_move_from position
                    start = int(cap_offsets_np[g])
                    count = int(cap_moves_per_game_np[g])
                    must_y = int(must_y_np[g])
                    must_x = int(must_x_np[g])
                    for idx in range(start, start + count):
                        if cap_from_y_np[idx] == must_y and cap_from_x_np[idx] == must_x:
                            games_with_constrained_captures[g] = True
                            break

            # Games with captures: either unconstrained, or constrained but from must_move_from
            games_with_captures = (
                games_with_stacks &
                (capture_moves.moves_per_game > 0) &
                (~has_must_move_constraint | games_with_constrained_captures)
            )
            games_movement_only = games_with_stacks & ~games_with_captures & (movement_moves.moves_per_game > 0)
            games_no_action = games_with_stacks & (capture_moves.moves_per_game == 0) & (movement_moves.moves_per_game == 0)

            # Apply capture moves with chain capture support (RR-CANON-R103)
            # Split handling: unconstrained games use batch logic, constrained games use per-game logic
            games_unconstrained_captures = games_with_captures & ~has_must_move_constraint
            games_constrained_captures = games_with_captures & has_must_move_constraint

            if games_unconstrained_captures.any():
                per_game_weights = self.get_weights_for_current_players(weights_list)
                selected_captures = self._select_moves(capture_moves, games_unconstrained_captures, per_game_weights)
                apply_capture_moves_batch(
                    self.state, selected_captures, capture_moves
                )

                # Track landing positions for chain capture continuation
                # Pre-extract data in batch to minimize .item() calls
                game_indices = torch.where(games_unconstrained_captures)[0]

                if game_indices.numel() > 0:
                    # Batch extract local indices and compute global indices
                    local_indices = selected_captures[game_indices]
                    valid_local = local_indices >= 0

                    # Compute global indices for valid selections
                    offsets = capture_moves.move_offsets[game_indices]
                    global_indices = offsets + local_indices
                    valid_global = valid_local & (global_indices < capture_moves.total_moves)

                    # Get landing positions for all valid captures
                    clamped_global = global_indices.clamp(0, max(0, capture_moves.total_moves - 1))
                    landing_y_batch = capture_moves.to_y[clamped_global]
                    landing_x_batch = capture_moves.to_x[clamped_global]

                    # Convert to numpy for efficient iteration
                    game_indices_np = game_indices.cpu().numpy()
                    valid_global_np = valid_global.cpu().numpy()
                    landing_y_np = landing_y_batch.cpu().numpy()
                    landing_x_np = landing_x_batch.cpu().numpy()

                    # Process chain captures for each game
                    for idx, g in enumerate(game_indices_np):
                        if not valid_global_np[idx]:
                            continue

                        landing_y = int(landing_y_np[idx])
                        landing_x = int(landing_x_np[idx])

                        # Chain capture loop: continue capturing from landing position
                        # per RR-CANON-R103 (mandatory chain captures)
                        max_chain_depth = 10  # Safety limit to prevent infinite loops
                        chain_depth = 0

                        while chain_depth < max_chain_depth:
                            chain_depth += 1

                            # Generate captures from current landing position only
                            chain_captures = generate_chain_capture_moves_from_position(
                                self.state, int(g), landing_y, landing_x
                            )

                            if not chain_captures:
                                # No more captures available from this position
                                break

                            # Select a chain capture (use first available for simplicity)
                            # In training, randomizing might be better but for correctness
                            # any valid chain is acceptable
                            to_y, to_x = chain_captures[0]

                            # Apply the chain capture
                            new_y, new_x = apply_single_chain_capture(
                                self.state, int(g), landing_y, landing_x, to_y, to_x
                            )

                            # Update landing position for next iteration
                            landing_y, landing_x = new_y, new_x

            # Handle constrained captures (from must_move_from position)
            # Optimized 2025-12-24: Pre-extract to avoid per-iteration .item() calls
            if games_constrained_captures.any():
                must_from_y_np = self.state.must_move_from_y.cpu().numpy()
                must_from_x_np = self.state.must_move_from_x.cpu().numpy()
                for g in torch.where(games_constrained_captures)[0].tolist():
                    # Get captures from must_move_from position
                    from_y = int(must_from_y_np[g])
                    from_x = int(must_from_x_np[g])

                    captures = generate_chain_capture_moves_from_position(
                        self.state, g, from_y, from_x
                    )
                    if not captures:
                        continue

                    # Apply the initial capture
                    to_y, to_x = captures[0]
                    landing_y, landing_x = apply_single_initial_capture(
                        self.state, g, from_y, from_x, to_y, to_x
                    )

                    # Chain capture loop
                    max_chain_depth = 10
                    chain_depth = 0
                    while chain_depth < max_chain_depth:
                        chain_depth += 1
                        chain_captures = generate_chain_capture_moves_from_position(
                            self.state, g, landing_y, landing_x
                        )
                        if not chain_captures:
                            break
                        to_y, to_x = chain_captures[0]
                        new_y, new_x = apply_single_chain_capture(
                            self.state, g, landing_y, landing_x, to_y, to_x
                        )
                        landing_y, landing_x = new_y, new_x

                # Track constrained captures as real actions
                mark_real_action_batch(self.state, games_constrained_captures)

            # Apply movement moves for games without captures
            if games_movement_only.any():
                per_game_weights = self.get_weights_for_current_players(weights_list)
                selected_movements = self._select_moves(movement_moves, games_movement_only, per_game_weights)

                # Extract landing positions BEFORE applying (needed for post-movement capture check)
                movement_game_indices = torch.where(games_movement_only)[0]
                movement_local_indices = selected_movements[movement_game_indices]
                movement_offsets = movement_moves.move_offsets[movement_game_indices]
                movement_global_indices = movement_offsets + movement_local_indices
                movement_valid = (movement_local_indices >= 0) & (movement_global_indices < movement_moves.total_moves)
                movement_global_clamped = movement_global_indices.clamp(0, max(0, movement_moves.total_moves - 1))
                movement_landing_y = movement_moves.to_y[movement_global_clamped]
                movement_landing_x = movement_moves.to_x[movement_global_clamped]

                apply_movement_moves_batch(
                    self.state, selected_movements, movement_moves
                )
                # December 2025: Track real action for forced elimination detection
                mark_real_action_batch(self.state, games_movement_only)

                # December 2025: CPU phase machine parity - check for captures AFTER movement
                # Per RR-CANON fsm.py:560-566: After MOVE_STACK, if captures are available
                # FROM THE MOVED STACK, CPU enters CAPTURE phase. GPU must match this by
                # generating captures ONLY from the new landing position.
                #
                # CRITICAL: We must check captures ONLY from the stack that just moved,
                # not from all player stacks. Using generate_chain_capture_moves_from_position
                # ensures we only look at captures from the specific landing position.
                games_with_post_captures = torch.zeros_like(games_movement_only)
                post_capture_landings = {}  # game_idx -> (landing_y, landing_x)

                # Optimized 2025-12-24: Pre-extract landing coords to avoid per-iteration .item() calls
                landing_y_np = movement_landing_y.cpu().numpy()
                landing_x_np = movement_landing_x.cpu().numpy()
                movement_valid_np = movement_valid.cpu().numpy()

                for idx, g in enumerate(movement_game_indices.tolist()):
                    if not movement_valid_np[idx]:
                        continue
                    landing_y = int(landing_y_np[idx])
                    landing_x = int(landing_x_np[idx])

                    # Check captures from the moved stack's new position only
                    captures_from_landing = generate_chain_capture_moves_from_position(
                        self.state, g, landing_y, landing_x
                    )
                    if captures_from_landing:
                        games_with_post_captures[g] = True
                        post_capture_landings[g] = (landing_y, landing_x)


                if games_with_post_captures.any():
                    # Apply post-movement captures from the new position (mandatory per RR-CANON)
                    # We process each game individually since we need to capture from
                    # specific landing positions, not all player stacks.
                    for g, (from_y, from_x) in post_capture_landings.items():
                        # Get first available capture from the landing position
                        captures = generate_chain_capture_moves_from_position(
                            self.state, g, from_y, from_x
                        )
                        if not captures:
                            continue

                        # Apply the initial capture (uses OVERTAKING_CAPTURE + CAPTURE phase)
                        to_y, to_x = captures[0]
                        landing_y, landing_x = apply_single_initial_capture(
                            self.state, g, from_y, from_x, to_y, to_x
                        )

                        # Chain capture loop
                        max_chain_depth = 10
                        chain_depth = 0
                        while chain_depth < max_chain_depth:
                            chain_depth += 1
                            chain_captures = generate_chain_capture_moves_from_position(
                                self.state, g, landing_y, landing_x
                            )
                            if not chain_captures:
                                break
                            to_y, to_x = chain_captures[0]
                            new_y, new_x = apply_single_chain_capture(
                                self.state, g, landing_y, landing_x, to_y, to_x
                            )
                            landing_y, landing_x = new_y, new_x

                    # Track post-movement captures as real actions
                    mark_real_action_batch(self.state, games_with_post_captures)

            if games_no_action.any():
                apply_no_action_moves_batch(self.state, games_no_action)

            # December 2025: Track real action for captures
            # Captures are real actions (applied above with chain capture support)
            if games_with_captures.any():
                mark_real_action_batch(self.state, games_with_captures)

        # December 2025: SKIP_CAPTURE is NOT needed after completing captures
        # Per CPU phase machine (lines 298-313 in phase_machine.py):
        # After ANY capture (OVERTAKING_CAPTURE, CONTINUE_CAPTURE_SEGMENT, etc.),
        # if no more captures exist, CPU auto-advances to LINE_PROCESSING.
        # SKIP_CAPTURE is only for explicitly DECLINING available captures (when
        # player does MOVE_STACK but chooses not to take available captures).
        # Since GPU always takes available captures, SKIP_CAPTURE should never be emitted.

        # December 2025: Reset capture chain tracking before advancing phase
        reset_capture_chain_batch(self.state, mask)

        # After movement, advance to LINE_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.LINE_PROCESSING

    def _step_line_phase(self, mask: torch.Tensor) -> None:
        """Handle LINE_PROCESSING phase for games in mask.

        Detect lines and convert them to territory markers.

        Per RR-CANON-R121-R122: Process all eligible lines for the current player.
        After line processing, check for new lines formed by territory collapse
        (cascade processing per RR-CANON-R144).

        December 2025: Records canonical line processing moves to move_history.
        """
        # Check which games have lines BEFORE processing (for move recording)
        games_with_lines = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        for p in range(1, self.num_players + 1):
            _, line_counts = detect_lines_vectorized(self.state, p, mask)
            games_with_lines = games_with_lines | (line_counts > 0)

        # Process the lines (collapses markers to territory, sets pending elimination flag)
        # Returns first line position per game for move recording
        # Use option2_probability=0.0 for CPU parity (always use Option 1 with elimination)
        line_positions = process_lines_batch(self.state, mask, option2_probability=0.0)
        # RR-CANON-R123: Auto-apply line elimination for self-play (no interactive choice)
        # Capture elimination positions for move recording
        elimination_positions = apply_line_elimination_batch(self.state, mask)

        # Record canonical moves to move_history
        # Games WITH lines: record CHOOSE_LINE_OPTION (player "chose" to process lines)
        # Games WITHOUT lines: record NO_LINE_ACTION
        self._record_line_phase_moves(mask, games_with_lines, elimination_positions, line_positions)

        # After line processing, advance to TERRITORY_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.TERRITORY_PROCESSING

    def _record_line_phase_moves(
        self,
        mask: torch.Tensor,
        games_with_lines: torch.Tensor,
        elimination_positions: dict[int, tuple[int, int]],
        line_positions: dict[int, tuple[int, int]],
    ) -> None:
        """Record canonical line processing moves to move_history.

        Per canonical contract (RR-CANON-R123), LINE_PROCESSING phase must emit:
        - Games WITH lines: CHOOSE_LINE_OPTION + ELIMINATE_RINGS_FROM_STACK
        - Games WITHOUT lines: NO_LINE_ACTION

        The elimination move is separate from the line choice per RR-CANON-R123
        to support interactive stack selection in human games.

        Args:
            mask: Games being processed in this phase
            games_with_lines: Which games had lines to process
            elimination_positions: Dict mapping game_idx to (y, x) of eliminated stack
            line_positions: Dict mapping game_idx to (y, x) of first position of processed line
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        players = self.state.current_player[game_indices]

        # Determine move type for each game
        had_lines = games_with_lines[game_indices]

        # Optimized 2025-12-24: Pre-extract to avoid per-iteration .item() calls
        move_counts_np = self.state.move_count.cpu().numpy()
        players_np = players.cpu().numpy()
        had_lines_np = had_lines.cpu().numpy()

        # Record moves
        for i, g in enumerate(game_indices.tolist()):
            move_count = int(move_counts_np[g])
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players_np[i])

            if had_lines_np[i]:
                # RR-CANON-R123: Record CHOOSE_LINE_OPTION + ELIMINATE_RINGS_FROM_STACK
                # First: record the line choice move
                # Get first position of processed line for this game (for CPU parity)
                line_y, line_x = line_positions.get(g, (-1, -1))
                self.state.move_history[g, move_count, 0] = MoveType.CHOOSE_LINE_OPTION
                self.state.move_history[g, move_count, 1] = player
                self.state.move_history[g, move_count, 2] = -1  # from_y (not used)
                self.state.move_history[g, move_count, 3] = -1  # from_x (not used)
                self.state.move_history[g, move_count, 4] = line_y  # to_y: first pos of line
                self.state.move_history[g, move_count, 5] = line_x  # to_x: first pos of line
                self.state.move_history[g, move_count, 6] = GamePhase.LINE_PROCESSING
                self.state.move_count[g] += 1
                move_count += 1

                # Second: record the elimination move ONLY if elimination actually occurred
                # (player might have no controlled stacks, in which case no elimination is needed)
                if g in elimination_positions and move_count < self.state.max_history_moves:
                    elim_y, elim_x = elimination_positions[g]
                    self.state.move_history[g, move_count, 0] = MoveType.ELIMINATE_RINGS_FROM_STACK
                    self.state.move_history[g, move_count, 1] = player
                    self.state.move_history[g, move_count, 2] = -1  # from_y (not used)
                    self.state.move_history[g, move_count, 3] = -1  # from_x (not used)
                    self.state.move_history[g, move_count, 4] = elim_y  # to_y: row of eliminated stack
                    self.state.move_history[g, move_count, 5] = elim_x  # to_x: col of eliminated stack
                    self.state.move_history[g, move_count, 6] = GamePhase.LINE_PROCESSING
                    self.state.move_count[g] += 1
            else:
                # No lines: record NO_LINE_ACTION
                self.state.move_history[g, move_count, 0] = MoveType.NO_LINE_ACTION
                self.state.move_history[g, move_count, 1] = player
                self.state.move_history[g, move_count, 2] = -1
                self.state.move_history[g, move_count, 3] = -1
                self.state.move_history[g, move_count, 4] = -1
                self.state.move_history[g, move_count, 5] = -1
                self.state.move_history[g, move_count, 6] = GamePhase.LINE_PROCESSING
                self.state.move_count[g] += 1

    def _step_territory_phase(self, mask: torch.Tensor) -> None:
        """Handle TERRITORY_PROCESSING phase for games in mask.

        Calculate enclosed territory using flood-fill.

        Per RR-CANON-R144-R145: Territory processing may create conditions for
        new lines (e.g., markers that were separated now form a line due to
        territory collapse removing blocking pieces). In this case, we need to
        return to LINE_PROCESSING phase for cascade handling.

        SIMPLIFICATION (GPU training): We implement a limited cascade check.
        Full cascade would iteratively process line->territory->line until stable.
        For training efficiency, we do one round of cascade check.

        December 2025: Records canonical territory processing moves to move_history.
        """
        # Capture territory counts BEFORE processing (to detect if territory was claimed)
        territory_before = self.state.territory_count.clone()

        # Process territory claims and capture territory moves (region + elimination pairs)
        # Use current_player_only=True to match CPU semantics (process only current player's regions)
        territory_moves, territory_region_positions = compute_territory_batch(
            self.state, mask, current_player_only=True
        )

        # Check which games claimed territory
        territory_after = self.state.territory_count
        # Sum across players to get total territory change per game
        territory_changed = (territory_after.sum(dim=1) != territory_before.sum(dim=1))
        games_with_territory = mask & territory_changed

        # Check for territory victory AFTER region collapse but BEFORE recording elimination
        # Per RR-CANON-R171/R062: territory victory if player has >= territoryVictoryMinimum AND > all opponents
        # If territory victory achieved, skip self-elimination recording - game ends immediately
        territory_victory = self._check_territory_victory(mask & games_with_territory)

        # Record canonical territory moves (CHOOSE + ELIM pairs for each territory)
        # Skip ELIMINATE for games where territory victory was achieved (game ends after CHOOSE)
        self._record_territory_phase_moves(
            mask, games_with_territory, territory_moves, territory_region_positions,
            skip_elimination=territory_victory,
        )

        # Set game status for territory victories
        if territory_victory.any():
            self.state.game_status[territory_victory] = GameStatus.COMPLETED
            # Winner is the current player who achieved territory victory
            self.state.winner[territory_victory] = self.state.current_player[territory_victory]

        # Check if games that processed territory have MORE regions to process
        # Per CPU semantics: player can process multiple regions per turn (one at a time)
        # If more regions exist, stay in TERRITORY_PROCESSING for next step
        games_with_more_territory = self._check_for_more_territory(games_with_territory)

        # Games with more territory: stay in TERRITORY_PROCESSING (don't advance yet)
        # Games without more territory: proceed to cascade check and phase transition

        # Cascade check: Did territory processing create new marker lines?
        # This can happen if territory collapse removes stacks that were blocking
        # marker alignment, or if markers from captured stacks now form lines.
        games_to_check = mask & ~games_with_more_territory  # Only check games ready to leave territory phase
        cascade_games = self._check_for_new_lines(games_to_check)

        if cascade_games.any():
            # Games with new lines go back to LINE_PROCESSING
            self.state.current_phase[cascade_games] = GamePhase.LINE_PROCESSING
            # Games without new lines: check for forced elimination before END_TURN
            no_cascade = games_to_check & ~cascade_games
            # December 2025: Forced elimination check (RR-CANON-R160)
            # If player had no real action this turn but has stacks, record forced_elimination
            check_and_apply_forced_elimination_batch(self.state, no_cascade)
            # Games still in TERRITORY_PROCESSING advance to END_TURN
            # (forced elimination sets phase to FORCED_ELIMINATION, others stay)
            still_territory = no_cascade & (self.state.current_phase == GamePhase.TERRITORY_PROCESSING)
            self.state.current_phase[still_territory] = GamePhase.END_TURN
        else:
            # No cascade needed, check for forced elimination before END_TURN
            # December 2025: Forced elimination check (RR-CANON-R160)
            check_and_apply_forced_elimination_batch(self.state, games_to_check)
            # Games still in TERRITORY_PROCESSING advance to END_TURN
            still_territory = games_to_check & (self.state.current_phase == GamePhase.TERRITORY_PROCESSING)
            self.state.current_phase[still_territory] = GamePhase.END_TURN

        # December 2025: Check for player elimination after territory processing
        # If any player has been eliminated (no rings anywhere), end the game
        self._check_player_elimination(mask)

    def _check_opponent_eliminated(
        self, mask: torch.Tensor, games_with_territory: torch.Tensor
    ) -> torch.Tensor:
        """Check if opponent was eliminated by territory processing.

        Returns a boolean mask of games where opponent has no rings left.
        Used to skip self-elimination recording when game ends during territory.

        Optimized 2025-12-24: Fully vectorized for 2-player games.
        """
        result = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        active_mask = mask & games_with_territory & self.state.get_active_mask()
        if not active_mask.any():
            return result

        # Compute player ring status
        player_has_rings = self._compute_player_ring_status_batch()

        # In 2-player games, check if the opponent has no rings (vectorized)
        if self.num_players == 2:
            # Compute opponent player index: if current==1 then opponent==2, else opponent==1
            current_players = self.state.current_player  # (batch_size,)
            opponent_players = torch.where(
                current_players == 1,
                torch.tensor(2, device=self.device),
                torch.tensor(1, device=self.device)
            )
            # Gather opponent's ring status
            opponent_indices = opponent_players.unsqueeze(1).long()  # (batch_size, 1) - int64 for gather
            opponent_has_rings = player_has_rings.gather(1, opponent_indices).squeeze(1)  # (batch_size,)
            # Result: opponent eliminated (no rings) in active games
            result = active_mask & ~opponent_has_rings

        return result

    def _check_player_elimination(self, mask: torch.Tensor) -> None:
        """Check if any player has been eliminated (no rings anywhere).

        A player is eliminated when they have:
        - No controlled stacks
        - No rings in hand
        - No buried rings (cannot recover)

        When a player is eliminated, the remaining player wins.
        This matches CPU's game-over detection after territory processing.
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        # Compute player ring status for all players
        player_has_rings = self._compute_player_ring_status_batch()

        # For each player, check if they have no rings at all
        for p in range(1, self.num_players + 1):
            # Games where player p has no rings
            player_eliminated = active_mask & ~player_has_rings[:, p]

            if player_eliminated.any():
                # Player p is eliminated, determine winner
                # In 2-player game, the other player wins
                if self.num_players == 2:
                    winner = 2 if p == 1 else 1
                    self.state.winner[player_eliminated] = winner
                    self.state.game_status[player_eliminated] = GameStatus.COMPLETED

    def _check_territory_victory(self, mask: torch.Tensor) -> torch.Tensor:
        """Check for territory victory per RR-CANON-R171/R062.

        A player wins by territory when BOTH:
        1. territorySpaces[P] >= territoryVictoryMinimum (floor(totalSpaces / numPlayers) + 1)
        2. territorySpaces[P] > sum(territorySpaces[Q]) for all opponents Q

        Args:
            mask: Games to check

        Returns:
            Boolean mask of games where current player achieved territory victory.

        Optimized 2025-12-24: Fully vectorized to avoid per-game loops and .item() calls.
        """
        result = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        if not mask.any():
            return result

        # Territory victory minimum: floor(totalSpaces / numPlayers) + 1
        total_spaces = self.board_size * self.board_size
        min_threshold = total_spaces // self.num_players + 1

        # Vectorized: Get current player's territory for all games
        # territory_count is (batch_size, num_players + 1) where index 0 is unused
        current_players = self.state.current_player  # (batch_size,)

        # Use gather to get each game's current player's territory count
        # Expand current_players to match territory_count shape for gather
        # Note: gather() requires int64 indices
        player_indices = current_players.unsqueeze(1).to(torch.int64)  # (batch_size, 1)
        player_territory = self.state.territory_count.gather(1, player_indices).squeeze(1)  # (batch_size,)

        # Condition 1: player has minimum threshold (vectorized)
        has_minimum = player_territory >= min_threshold

        # Condition 2: player dominates opponents (vectorized)
        # Sum all territory, then subtract player's own to get opponent total
        total_territory = self.state.territory_count.sum(dim=1)  # (batch_size,)
        opponent_total = total_territory - player_territory

        # Player dominates if their territory > opponent total
        dominates = player_territory > opponent_total

        # Victory requires both conditions AND game is in mask
        result = mask & has_minimum & dominates

        return result

    def _check_for_more_territory(self, mask: torch.Tensor) -> torch.Tensor:
        """Check if games have more territory regions to process.

        Per CPU semantics, a player can process multiple regions per turn,
        choosing one at a time. After processing one region, check if more
        eligible regions exist for the current player.

        Args:
            mask: Games to check (typically games that just processed territory)

        Returns:
            Boolean mask of games that have more territory regions available.
        """
        from .gpu_territory import (
            _find_eligible_territory_cap,
            _find_regions_with_border_color,
            _is_color_disconnected,
        )

        result = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        if not mask.any():
            return result

        import numpy as np

        # Optimized 2025-12-24: Pre-extract to avoid per-iteration .item() calls
        game_status_np = self.state.game_status.cpu().numpy()
        current_player_np = self.state.current_player.cpu().numpy()

        for g in torch.where(mask)[0].tolist():
            # Check if game is still active
            if game_status_np[g] != GameStatus.ACTIVE:
                continue

            player = int(current_player_np[g])

            # Get board state as numpy for efficiency
            stack_owner_np = self.state.stack_owner[g].cpu().numpy()
            stack_height_np = self.state.stack_height[g].cpu().numpy()
            marker_owner_np = self.state.marker_owner[g].cpu().numpy()

            # RR-CANON-R142 fix: Don't early-exit based on controlling player count
            # ActiveColors includes ALL rings (buried too), not just controlling players.
            # Empty regions can be color-disconnected even with 1 controlling player
            # if another player has buried rings. Let _is_color_disconnected handle filtering.
            # Only skip if no stacks exist at all.
            has_any_stacks = (stack_height_np > 0).any()
            if not has_any_stacks:
                continue  # No stacks on board, no territory processing possible

            marker_colors = {int(c) for c in np.unique(marker_owner_np) if c > 0}

            if not marker_colors:
                continue

            # Find all candidate regions with marker barriers
            candidate_regions = []
            for border_color in marker_colors:
                regions = _find_regions_with_border_color(self.state, g, border_color)
                candidate_regions.extend(regions)

            # Check for color-disconnected regions the player can claim
            for region, border_player, _start_pos in candidate_regions:
                if not _is_color_disconnected(self.state, g, region):
                    continue

                # Check if player has eligible cap outside region
                cap = _find_eligible_territory_cap(
                    self.state, g, player, excluded_positions=region
                )
                if cap is not None:
                    result[g] = True
                    break  # Found one eligible region, that's enough

        return result

    def _record_territory_phase_moves(
        self,
        mask: torch.Tensor,
        games_with_territory: torch.Tensor,
        territory_moves: dict[int, list[tuple[int, int, int, int, int]]],
        region_positions: dict[int, tuple[int, int]],
        skip_elimination: torch.Tensor | None = None,
    ) -> None:
        """Record canonical territory processing moves to move_history.

        Per canonical contract (RR-CANON-R145), TERRITORY_PROCESSING phase must emit:
        - Games WITH territory: CHOOSE_TERRITORY_OPTION + ELIMINATE_RINGS_FROM_STACK pairs
        - Games WITHOUT territory: NO_TERRITORY_ACTION

        Per RR-CANON-R145, self-elimination is MANDATORY when processing territory.
        However, if territory victory is achieved during region collapse, the game
        ends immediately and self-elimination is skipped (per CPU behavior matching
        RR-CANON-R171).

        CPU parity: Each territory region requires a CHOOSE + ELIM pair. When multiple
        territories are processed (due to chain reactions), we emit multiple pairs.

        Args:
            mask: Games being processed in this phase
            games_with_territory: Which games had territory to claim
            territory_moves: Dict mapping game_idx to list of (player, region_y, region_x, elim_y, elim_x)
            region_positions: Dict mapping game_idx to (y, x) first region position (backwards compat)
            skip_elimination: Boolean mask - skip elimination recording for games with territory victory
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        players = self.state.current_player[game_indices]

        # Determine move type for each game
        had_territory = games_with_territory[game_indices]

        # Optimized 2025-12-24: Pre-extract to avoid per-iteration .item() calls
        move_counts_np = self.state.move_count.cpu().numpy()
        players_np = players.cpu().numpy()
        had_territory_np = had_territory.cpu().numpy()

        # Record moves
        for i, g in enumerate(game_indices.tolist()):
            move_count = int(move_counts_np[g])
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players_np[i])

            if had_territory_np[i]:
                # RR-CANON-R145: Record CHOOSE_TERRITORY_OPTION + ELIMINATE_RINGS_FROM_STACK pairs
                # CPU parity: Each territory needs its own CHOOSE + ELIM pair
                # Skip all recording if territory victory was achieved - game ends immediately
                if skip_elimination is not None and skip_elimination[g]:
                    # Record just the first territory choice for history, then skip
                    region_y, region_x = region_positions.get(g, (-1, -1))
                    self.state.move_history[g, move_count, 0] = MoveType.CHOOSE_TERRITORY_OPTION
                    self.state.move_history[g, move_count, 1] = player
                    self.state.move_history[g, move_count, 2] = -1
                    self.state.move_history[g, move_count, 3] = -1
                    self.state.move_history[g, move_count, 4] = region_y
                    self.state.move_history[g, move_count, 5] = region_x
                    self.state.move_history[g, move_count, 6] = GamePhase.TERRITORY_PROCESSING
                    self.state.move_count[g] += 1
                    continue  # Territory victory - game ends, no elimination moves

                # Record CHOOSE + ELIM pair for each territory processed
                moves = territory_moves.get(g, [])
                for elim_player, region_y, region_x, elim_y, elim_x in moves:
                    if elim_player != player:
                        continue  # Only record moves for current player
                    if move_count >= self.state.max_history_moves - 1:
                        break  # Need space for both CHOOSE and ELIM

                    # Record CHOOSE_TERRITORY_OPTION for this region
                    self.state.move_history[g, move_count, 0] = MoveType.CHOOSE_TERRITORY_OPTION
                    self.state.move_history[g, move_count, 1] = player
                    self.state.move_history[g, move_count, 2] = -1  # from_y: not applicable
                    self.state.move_history[g, move_count, 3] = -1  # from_x: not applicable
                    self.state.move_history[g, move_count, 4] = region_y  # to_y: region representative
                    self.state.move_history[g, move_count, 5] = region_x  # to_x: region representative
                    self.state.move_history[g, move_count, 6] = GamePhase.TERRITORY_PROCESSING
                    self.state.move_count[g] += 1
                    move_count += 1

                    # Record ELIMINATE_RINGS_FROM_STACK for this region
                    self.state.move_history[g, move_count, 0] = MoveType.ELIMINATE_RINGS_FROM_STACK
                    self.state.move_history[g, move_count, 1] = elim_player
                    self.state.move_history[g, move_count, 2] = -1  # from_y (not used)
                    self.state.move_history[g, move_count, 3] = -1  # from_x (not used)
                    self.state.move_history[g, move_count, 4] = elim_y  # to_y: row of eliminated stack
                    self.state.move_history[g, move_count, 5] = elim_x  # to_x: col of eliminated stack
                    self.state.move_history[g, move_count, 6] = GamePhase.TERRITORY_PROCESSING
                    self.state.move_count[g] += 1
                    move_count += 1
            else:
                # No territory: record NO_TERRITORY_ACTION
                self.state.move_history[g, move_count, 0] = MoveType.NO_TERRITORY_ACTION
                self.state.move_history[g, move_count, 1] = player
                self.state.move_history[g, move_count, 2] = -1
                self.state.move_history[g, move_count, 3] = -1
                self.state.move_history[g, move_count, 4] = -1
                self.state.move_history[g, move_count, 5] = -1
                self.state.move_history[g, move_count, 6] = GamePhase.TERRITORY_PROCESSING
                self.state.move_count[g] += 1

    def _check_for_new_lines(self, mask: torch.Tensor) -> torch.Tensor:
        """Check which games have new marker lines after territory processing.

        Used for cascade detection per RR-CANON-R144.

        Refactored 2025-12-13 for vectorized efficiency:
        - Use detect_lines_vectorized which returns line counts
        - O(1) check via count > 0

        Args:
            mask: Games to check

        Returns:
            Boolean tensor indicating which games have new lines
        """
        has_new_lines = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Check each player's lines across all masked games at once
        for p in range(1, self.num_players + 1):
            # detect_lines_vectorized returns (in_line_mask, line_counts)
            _, line_counts = detect_lines_vectorized(self.state, p, mask)
            # Games with line_count > 0 have lines
            has_new_lines = has_new_lines | (line_counts > 0)

        return has_new_lines

    def _step_end_turn_phase(self, mask: torch.Tensor) -> None:
        """Handle END_TURN phase for games in mask.

        Rotate to next player and reset phase to RING_PLACEMENT.

        Per updated rules: Players are only permanently eliminated if they have
        NO rings anywhere (no controlled stacks, no buried rings, no rings in hand).
        Players with only buried rings still get turns and can use recovery moves.

        Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
        NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action
        and proceed to movement, but they MUST enter ring_placement first.

        Refactored 2025-12-11 for vectorized player rotation:
        - Precompute player elimination status for all players in all games
        - Use vectorized rotation with fallback for eliminated player skipping
        """
        # NOTE: move_count is incremented in the vectorized move application functions
        # (apply_capture_moves_vectorized, apply_movement_moves_vectorized, etc.)
        # when recording moves to history, so we don't increment here.

        # Precompute player elimination status for all games and players
        # Shape: (batch_size, num_players+1) - player_has_rings[g, p] = True if player p has rings in game g
        player_has_rings = self._compute_player_ring_status_batch()

        # Vectorized player rotation with eliminated player skipping
        # For most games (2-player with no eliminations), this is a simple increment
        current_players = self.state.current_player.clone()  # (batch_size,)

        # Start with simple rotation: (current % num_players) + 1
        next_players = (current_players % self.num_players) + 1

        # For games where the next player is eliminated, find the next non-eliminated player
        # This handles the uncommon case where we need to skip eliminated players
        for _skip_round in range(self.num_players):
            # Check which games have an eliminated next player
            # Use gather to check player_has_rings[g, next_players[g]]
            next_player_has_rings = torch.gather(
                player_has_rings,
                dim=1,
                index=next_players.unsqueeze(1).long()
            ).squeeze(1)

            # Games where next player is eliminated AND we're in the mask
            needs_skip = mask & ~next_player_has_rings

            if not needs_skip.any():
                break

            # Rotate eliminated players to next candidate
            next_players[needs_skip] = (next_players[needs_skip] % self.num_players) + 1

        # Apply the computed next players
        self.state.current_player[mask] = next_players[mask]

        # Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
        # NO PHASE SKIPPING - this is a core invariant for parity with TS/Python engines.
        self.state.current_phase[mask] = GamePhase.RING_PLACEMENT
        self.state.must_move_from_y[mask] = -1
        self.state.must_move_from_x[mask] = -1

        # December 2025: Reset per-turn tracking for next player's turn
        # This resets turn_had_real_action, capture chain tracking
        reset_turn_tracking_batch(self.state, mask)

        # Swap sides (pie rule) check for 2-player games (RR-CANON R180-R184)
        # Offered to P2 immediately after P1's first complete turn
        if self.num_players == 2 and self.swap_enabled:
            self._check_and_apply_swap_sides(mask)

    def _check_and_apply_swap_sides(self, mask: torch.Tensor) -> None:
        """Check for swap_sides eligibility and mark it as offered.

        Per RR-CANON R180-R184: The pie rule allows P2 to swap colours/seats
        with P1 immediately after P1's first complete turn.

        Conditions for swap eligibility:
        1. 2-player game (already checked by caller)
        2. Current player is now P2
        3. Swap not already offered in this game
        4. P1 has completed at least one full turn (has moves in history)

        IMPORTANT (GPU parity):
        The canonical engines implement swap_sides as an *identity swap* that
        does not change board ownership or per-seat counters (ringsInHand,
        eliminatedRings, territorySpaces). GPU self-play does not currently
        record explicit swap_sides moves in its coarse move_history format, so
        applying a semantic swap here would create non-replayable traces.

        Until GPU move history is upgraded to represent swap_sides explicitly,
        we treat the pie rule as "offered but always declined" and only set
        swap_offered for observability/debugging.
        """
        # Identify games where swap should be offered:
        # - In mask (just completed END_TURN)
        # - Current player is now P2
        # - Swap not already offered
        is_p2_turn = self.state.current_player == 2
        not_yet_offered = ~self.state.swap_offered
        swap_eligible = mask & is_p2_turn & not_yet_offered

        if not swap_eligible.any():
            return

        # Mark swap as offered for these games (regardless of acceptance)
        self.state.swap_offered[swap_eligible] = True
        return

    def _compute_player_ring_status_batch(self) -> torch.Tensor:
        """Compute which players have any rings in each game (vectorized).

        Returns:
            Boolean tensor of shape (batch_size, num_players+1) where
            result[g, p] = True if player p has any rings in game g.
            Index 0 is unused (players are 1-indexed).

        A player has rings if ANY of:
        - rings_in_hand[g, p] > 0
        - Any cell where stack_owner[g, y, x] == p (controlled stacks)
        - buried_rings[g, p] > 0
        """
        device = self.device
        batch_size = self.batch_size
        num_players = self.num_players

        # Initialize result tensor
        has_rings = torch.zeros(batch_size, num_players + 1, dtype=torch.bool, device=device)

        for p in range(1, num_players + 1):
            # Check rings in hand
            has_in_hand = self.state.rings_in_hand[:, p] > 0

            # Check controlled stacks (any cell where stack_owner == p)
            has_controlled = (self.state.stack_owner == p).flatten(1).any(dim=1)

            # Check buried rings
            has_buried = self.state.buried_rings[:, p] > 0

            # Player has rings if any of the above is true
            has_rings[:, p] = has_in_hand | has_controlled | has_buried

        return has_rings

    def _player_has_any_rings_gpu(self, g: int, player: int) -> bool:
        """Check if a player has any rings anywhere (controlled, buried, or in hand).

        A player with no rings anywhere is permanently eliminated.
        A player who has rings (even if only buried) is NOT permanently eliminated
        and should still get turns (they may have recovery moves available).

        Args:
            g: Game index in batch
            player: Player number (1-indexed)

        Returns:
            True if player has any rings anywhere, False if permanently eliminated

        Optimized Dec 2025: Pre-extract to numpy to minimize .item() calls.
        """
        # Pre-extract to numpy (Dec 2025 optimization)
        rings_in_hand_np = self.state.rings_in_hand[g].cpu().numpy()
        stack_owner_np = self.state.stack_owner[g].cpu().numpy()
        buried_rings_np = self.state.buried_rings[g].cpu().numpy()

        # Check rings in hand
        if rings_in_hand_np[player] > 0:
            return True

        # Check controlled stacks (rings in stacks we control)
        if (stack_owner_np == player).any():
            return True

        # Check buried rings (rings in opponent-controlled stacks)
        # buried_rings uses 1-indexed players (shape is batch_size x num_players+1)
        return buried_rings_np[player] > 0

    def _apply_single_capture(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single capture move for game g at global index move_idx.

        Per RR-CANON-R100-R103:
        - Attacker moves onto defender stack
        - Defender's top ring is eliminated
        - Stacks merge (attacker on top)
        - Control transfers to attacker

        Optimized 2025-12-24: Pre-extract to numpy to minimize .item() calls.
        """
        state = self.state

        # Pre-extract all needed values to numpy/Python in one batch
        # This minimizes GPU↔CPU sync points
        from_y = int(moves.from_y[move_idx].cpu().numpy())
        from_x = int(moves.from_x[move_idx].cpu().numpy())
        to_y = int(moves.to_y[move_idx].cpu().numpy())
        to_x = int(moves.to_x[move_idx].cpu().numpy())
        player = int(state.current_player[g].cpu().numpy())
        move_type = int(moves.move_type[move_idx].cpu().numpy())
        move_count = int(state.move_count[g].cpu().numpy())
        is_chain = bool(state.in_capture_chain[g].cpu().numpy())
        current_phase = int(state.current_phase[g].cpu().numpy())
        attacker_height = int(state.stack_height[g, from_y, from_x].cpu().numpy())
        defender_height = int(state.stack_height[g, to_y, to_x].cpu().numpy())
        defender_owner = int(state.stack_owner[g, to_y, to_x].cpu().numpy())

        # Pre-extract marker_owner slice for the game
        marker_owner_g = state.marker_owner[g].cpu().numpy()

        # Record move in history
        # 9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x
            # Record phase based on capture chain state and current phase:
            # - Direct capture during MOVEMENT phase → MOVEMENT phase
            # - First capture after MOVE_STACK (CAPTURE phase) → CAPTURE phase
            # - Chain captures → CHAIN_CAPTURE phase
            if is_chain:
                record_phase = GamePhase.CHAIN_CAPTURE
            elif current_phase == GamePhase.MOVEMENT:
                record_phase = GamePhase.MOVEMENT  # Direct capture during MOVEMENT
            else:
                record_phase = GamePhase.CAPTURE  # First capture after MOVE_STACK
            state.move_history[g, move_count, 6] = record_phase
            # December 2025: Record capture target for canonical export
            # Note: This function uses to_y/to_x as target directly (not landing)
            state.move_history[g, move_count, 7] = to_y
            state.move_history[g, move_count, 8] = to_x

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_val = marker_owner_g[check_y, check_x]
            if marker_val != 0 and marker_val != player:
                # Flip opponent marker to our color
                state.marker_owner[g, check_y, check_x] = player

        # Eliminate defender's top ring
        defender_eliminated = 1
        defender_new_height = max(0, defender_height - defender_eliminated)

        # Track elimination
        if defender_owner > 0:
            # Defender LOSES the ring
            state.eliminated_rings[g, defender_owner] += defender_eliminated
            # Attacker (player) CAUSED the elimination (for victory check)
            state.rings_caused_eliminated[g, player] += defender_eliminated

        # Clear attacker origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        # Place merged stack at destination (attacker on top)
        new_height = attacker_height + defender_new_height
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(MAX_STACK_HEIGHT, new_height)  # Cap at MAX_STACK_HEIGHT

    def _apply_single_movement(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single movement move for game g at global index move_idx.

        Per RR-CANON-R090-R092:
        - Stack moves from origin to destination
        - Origin becomes empty
        - Destination gets merged stack (if own stack) or new stack
        - Markers along path: flip on pass, collapse cost on landing

        Optimized 2025-12-24: Pre-extract to numpy to minimize .item() calls.
        """
        state = self.state

        # Pre-extract all needed values to numpy/Python in one batch
        from_y = int(moves.from_y[move_idx].cpu().numpy())
        from_x = int(moves.from_x[move_idx].cpu().numpy())
        to_y = int(moves.to_y[move_idx].cpu().numpy())
        to_x = int(moves.to_x[move_idx].cpu().numpy())
        player = int(state.current_player[g].cpu().numpy())
        move_type = int(moves.move_type[move_idx].cpu().numpy())
        move_count = int(state.move_count[g].cpu().numpy())
        moving_height = int(state.stack_height[g, from_y, from_x].cpu().numpy())

        # Pre-extract marker_owner and stack info for the game
        marker_owner_g = state.marker_owner[g].cpu().numpy()
        dest_marker = int(marker_owner_g[to_y, to_x])
        dest_owner = int(state.stack_owner[g, to_y, to_x].cpu().numpy())
        dest_height = int(state.stack_height[g, to_y, to_x].cpu().numpy())

        # Record move in history
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x
            state.move_history[g, move_count, 6] = GamePhase.MOVEMENT

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_val = marker_owner_g[check_y, check_x]
            if marker_val != 0 and marker_val != player:
                # Flip opponent marker to our color
                state.marker_owner[g, check_y, check_x] = player

        # Handle landing on ANY marker (own or opponent) - per RR-CANON-R092:
        # The marker is removed and the top ring of the landing stack is eliminated.
        # Note: Landing does NOT collapse the position (only path markers collapse).
        landing_ring_cost = 0
        if dest_marker != 0:
            # Landing on any marker costs 1 ring (cap elimination)
            landing_ring_cost = 1
            state.marker_owner[g, to_y, to_x] = 0  # Marker removed

        # Clear origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        if dest_owner == 0:
            # Landing on empty
            new_height = moving_height - landing_ring_cost
            state.stack_owner[g, to_y, to_x] = player
            state.stack_height[g, to_y, to_x] = max(1, new_height)
        elif dest_owner == player:
            # Merging with own stack
            new_height = dest_height + moving_height - landing_ring_cost
            state.stack_height[g, to_y, to_x] = min(MAX_STACK_HEIGHT, new_height)  # Cap at MAX_STACK_HEIGHT

    def _apply_single_recovery(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single recovery slide move for game g at global index move_idx.

        Per RR-CANON-R110-R115:
        - Recovery slide: marker moves to adjacent empty cell
        - Costs 1 buried ring (deducted from buried_rings)
        - Origin marker is cleared, destination gets marker
        - Player gains recovery attempt toward un-burying rings

        Per RR-CANON-R114 (Recovery Cascade):
        - After recovery move completes, check for line formation
        - If line formed, process it (collapse markers, eliminate ring)
        - After line processing, check for territory claims
        - Cascade continues until no new lines are formed

        Optimized 2025-12-24: Pre-extract to numpy to minimize .item() calls.
        """
        state = self.state

        # Pre-extract all needed values to numpy/Python in one batch
        from_y = int(moves.from_y[move_idx].cpu().numpy())
        from_x = int(moves.from_x[move_idx].cpu().numpy())
        to_y = int(moves.to_y[move_idx].cpu().numpy())
        to_x = int(moves.to_x[move_idx].cpu().numpy())
        player = int(state.current_player[g].cpu().numpy())
        move_type = int(moves.move_type[move_idx].cpu().numpy())
        move_count = int(state.move_count[g].cpu().numpy())
        current_buried = int(state.buried_rings[g, player].cpu().numpy())

        # Record move in history
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x
            # Recovery is a movement-phase action in canonical rules.
            state.move_history[g, move_count, 6] = GamePhase.MOVEMENT

        # Move marker from origin to destination
        state.marker_owner[g, from_y, from_x] = 0  # Clear origin
        state.marker_owner[g, to_y, to_x] = player  # Place at destination

        # Deduct recovery cost: 1 buried ring
        # In the canonical rules, recovery costs rings from the buried pool
        # NOTE: buried_rings is 1-indexed (shape: batch_size, num_players + 1)
        if current_buried > 0:
            state.buried_rings[g, player] = current_buried - 1

        # Recovery Cascade per RR-CANON-R114:
        # After recovery move, the marker slide could form a line, which triggers
        # line processing, which could trigger territory claims, and so on.
        self._process_recovery_cascade(g, player)

    def _process_recovery_cascade(self, g: int, player: int, max_iterations: int = 5) -> None:
        """Process line formation and territory claims after a recovery move.

        Per RR-CANON-R114: After a recovery move, check if a line was formed.
        If so, process the line (collapse markers, eliminate ring from any stack).
        Then check for territory claims. This cascade continues until stable.

        Args:
            g: Game index
            player: Player who made the recovery move
            max_iterations: Safety limit to prevent infinite loops (default 5)
        """
        # Create a single-game mask for this game
        single_game_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        single_game_mask[g] = True

        for _iteration in range(max_iterations):
            # Check for lines for the current player using vectorized detection
            _, line_counts = detect_lines_vectorized(self.state, player, single_game_mask)

            # Pre-extract to numpy to avoid .item() sync (Dec 2025 optimization)
            if line_counts.cpu().numpy()[g] == 0:
                # No lines formed, we're done with cascade
                break

            # Process lines (collapse markers, set pending elimination flag)
            # Use option2_probability=0.0 for CPU parity (always use Option 1 with elimination)
            process_lines_batch(self.state, single_game_mask, option2_probability=0.0)
            # RR-CANON-R123: Auto-apply line elimination for cascade processing
            apply_line_elimination_batch(self.state, single_game_mask)

            # After line processing, check for territory claims (current player only)
            # Return values not needed here - cascade processing doesn't record moves
            compute_territory_batch(self.state, single_game_mask, current_player_only=True)

            # Continue loop to check if territory processing created new lines
            # (e.g., by removing markers that were blocking a line)

    def _select_best_moves(
        self,
        moves: BatchMoves,
        weights_list: list[list[dict[str, float]]],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select the best move for each game using GPU-accelerated heuristic evaluation.

        Uses fully vectorized heuristic scoring from gpu_selection.py:
        - Center distance scoring (closer = better)
        - Capture value scoring (target stack height)
        - Adjacency to own stacks
        - Line potential detection

        The heuristic weights from weights_list are resolved to the current player
        for each game and passed to the vectorized selection function.

        Optimized 2025-12-25: Uses select_moves_heuristic for proper GPU-accelerated
        heuristic evaluation instead of simple center-bias.
        """
        # Resolve per-game weights to current player's weights
        # weights_list is [game][player] -> weights dict
        # We need [game] -> current player's weights dict
        current_players = self.state.current_player.cpu().numpy()
        resolved_weights: list[dict[str, float]] = []
        for g in range(self.batch_size):
            player_idx = int(current_players[g]) - 1  # Convert 1-indexed to 0-indexed
            if weights_list and len(weights_list) > g and len(weights_list[g]) > player_idx:
                resolved_weights.append(weights_list[g][player_idx])
            else:
                # Default weights if not provided
                resolved_weights.append({})

        # Use fully vectorized heuristic selection (no .item() calls, no per-game loops)
        return select_moves_heuristic(
            moves, self.state, active_mask,
            weights_list=resolved_weights,
            temperature=1.0
        )

    def _check_victory_conditions(self) -> None:
        """Check and update victory conditions for all games.

        Implements canonical rules:
        - RR-CANON-R170: Ring-elimination victory (eliminatedRingsTotal >= victoryThreshold)
        - RR-CANON-R171: Territory-control victory (dual condition: threshold AND dominance)
        - RR-CANON-R172: Last-player-standing (round-based exclusive real actions)

        Victory thresholds per RR-CANON-R061/R062-v2:
        - victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
        - territoryVictoryMinimum = floor(totalSpaces / numPlayers) + 1 [plus dominance check]
        """
        active_mask = self.state.get_active_mask()

        # Canonical thresholds depend on board type and player count (RR-CANON-R061/R062).
        from app.models import BoardType
        from app.rules.core import get_territory_victory_minimum, get_victory_threshold

        board_type_map = {
            8: BoardType.SQUARE8,
            9: BoardType.HEX8,
            19: BoardType.SQUARE19,
            25: BoardType.HEXAGONAL,  # Fix: size=25 (bounding box), not 13
        }
        board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)
        ring_elimination_threshold = get_victory_threshold(board_type, self.num_players)
        # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
        territory_victory_minimum = get_territory_victory_minimum(board_type, self.num_players)

        for p in range(1, self.num_players + 1):
            # Check ring-elimination victory (RR-CANON-R170)
            # Per RR-CANON-R060/R170: Player wins when they have CAUSED >= victoryThreshold
            # rings to be eliminated (includes self-elimination via lines, territory, etc.)
            # rings_caused_eliminated[:, p] tracks rings that player p CAUSED to be eliminated
            ring_elimination_victory = self.state.rings_caused_eliminated[:, p] >= ring_elimination_threshold

            # Check territory victory per RR-CANON-R062-v2 (dual condition)
            # Condition 1: Territory >= floor(totalSpaces / numPlayers) + 1
            player_territory = self.state.territory_count[:, p]
            meets_threshold = player_territory >= territory_victory_minimum

            # Condition 2: Territory > sum of all opponents' territory
            total_territory = self.state.territory_count[:, 1:self.num_players + 1].sum(dim=1)
            opponents_territory = total_territory - player_territory
            dominates_opponents = player_territory > opponents_territory

            # Victory requires BOTH conditions
            territory_victory = meets_threshold & dominates_opponents

            # RR-CANON-R172 (LPS) is applied at turn start via the LPS round
            # tracker (see _update_lps_round_tracking_for_current_player).
            victory_mask = active_mask & (ring_elimination_victory | territory_victory)
            self.state.winner[victory_mask] = p
            self.state.game_status[victory_mask] = GameStatus.COMPLETED

    def _check_stalemate(self, mask: torch.Tensor) -> None:
        """Check for stalemate condition (no valid moves for current player).

        Per RR-CANON-R175 (implied): If the current player has no valid moves
        and cannot make any progress (no placement, no movement, no recovery),
        then the game ends in a stalemate (draw) or tiebreaker by stack count.

        This is called during MOVEMENT phase when a player has neither:
        - Controlled stacks to move
        - Rings in hand to place
        - Recovery moves available

        Stalemate resolution per canonical rules:
        - Winner determined by highest total stack height
        - Ties result in draw
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        state = self.state
        device = self.device
        batch_size = self.batch_size

        # === BATCHED STALEMATE CHECK ===
        # Get current players for all games (batch_size,)
        current_players = state.current_player

        # Check if current player has any stacks (vectorized)
        # Create player mask for each game: (batch_size, H, W)
        player_mask = (
            state.stack_owner
            == current_players.unsqueeze(-1).unsqueeze(-1)
        )
        has_stacks = player_mask.flatten(1).any(dim=1)

        # Check if player has rings in hand (vectorized using gather)
        player_indices = current_players.unsqueeze(-1).long()  # (batch_size, 1) - int64 for gather
        has_rings_in_hand = (
            torch.gather(state.rings_in_hand, 1, player_indices).squeeze(-1) > 0
        )

        # Check markers (vectorized)
        marker_mask = (
            state.marker_owner
            == current_players.unsqueeze(-1).unsqueeze(-1)
        )
        has_markers = marker_mask.flatten(1).any(dim=1)

        # Check buried rings (vectorized)
        has_buried = (
            torch.gather(state.buried_rings, 1, player_indices).squeeze(-1) > 0
        )

        # Can recover if has both markers and buried rings
        can_recover = has_markers & has_buried

        # Stalemate: no stacks, no rings in hand, can't recover
        is_stalemate = active_mask & ~has_stacks & ~has_rings_in_hand & ~can_recover

        if not is_stalemate.any():
            return

        # === TIEBREAKER RESOLUTION ===
        # For stalemate games, determine winner by stack height
        # Compute total height per player across all games at once

        # Stack heights per player: (batch_size, num_players)
        player_heights = torch.zeros(
            batch_size, self.num_players + 1, dtype=torch.float32, device=device
        )

        for p in range(1, self.num_players + 1):
            p_mask = state.stack_owner == p
            player_heights[:, p] = (
                state.stack_height.float() * p_mask.float()
            ).sum(dim=(1, 2))

        # Find max height per game (excluding player 0)
        heights_to_compare = player_heights[:, 1 : self.num_players + 1]
        max_heights, best_players = heights_to_compare.max(dim=1)
        best_players = best_players + 1  # Adjust for 1-indexed players

        # Check for ties (more than one player at max height)
        is_at_max = heights_to_compare == max_heights.unsqueeze(-1)
        num_at_max = is_at_max.sum(dim=1)
        is_tie = num_at_max > 1

        # Set winners
        winners = torch.where(
            is_tie | (max_heights == 0),
            torch.zeros_like(best_players),  # Draw
            best_players,
        )

        # Apply only to stalemate games
        state.winner[is_stalemate] = winners[is_stalemate]
        state.game_status[is_stalemate] = GameStatus.COMPLETED

    def _default_weights(self) -> dict[str, float]:
        """Load best heuristic weights for this board configuration.

        Priority order:
        1. Registry weights (PRODUCTION stage) - from CMA-ES optimization
        2. Registry weights (STAGING stage) - from recent CMA-ES runs
        3. Board-specific profile from heuristic_weights.py
        4. Fallback minimal weights

        This enables the positive feedback training loop:
        CMA-ES → Registry → Selfplay → Better training data → NN improves
        """
        # Determine board type for registry lookup
        board_type = self.board_type or "square8"
        if self.board_size == 19:
            board_type = "square19"
        elif self.board_size == 8 and board_type not in ("square8", "hex8"):
            board_type = "square8"

        # Try to load from model registry (CMA-ES optimized weights)
        try:
            from app.training.cmaes_registry_integration import (
                load_heuristic_weights_from_registry,
            )

            # Try production first, then staging
            for stage in ("production", "staging"):
                weights = load_heuristic_weights_from_registry(
                    board_type=board_type,
                    num_players=self.num_players,
                    stage=stage,
                )
                if weights:
                    logger.info(
                        f"Loaded {stage} weights from registry for "
                        f"{board_type}_{self.num_players}p"
                    )
                    return weights
        except Exception as e:
            logger.debug(f"Registry weight loading failed: {e}")

        # Fall back to board-specific profile from heuristic_weights.py
        try:
            from .heuristic_weights import get_weights_for_board

            weights = get_weights_for_board(board_type, self.num_players)
            if weights:
                logger.debug(
                    f"Using profile weights for {board_type}_{self.num_players}p"
                )
                return weights
        except Exception as e:
            logger.debug(f"Profile weight loading failed: {e}")

        # Final fallback - minimal weights (evaluate_positions_batch fills defaults)
        return {
            "stack_count": 1.0,
            "territory_count": 2.0,
            "rings_penalty": 0.1,
            "center_control": 0.3,
        }

    def _apply_weight_noise(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply multiplicative noise to weights for training diversity.

        Each weight is multiplied by a random factor in [1-noise, 1+noise].

        Args:
            weights: Base weights dictionary

        Returns:
            New weights dictionary with noise applied
        """
        if self.weight_noise <= 0:
            return weights.copy()

        import random
        noisy_weights = {}
        for key, value in weights.items():
            # Multiplicative noise: value * uniform(1-noise, 1+noise)
            noise_factor = 1.0 + random.uniform(-self.weight_noise, self.weight_noise)
            noisy_weights[key] = value * noise_factor
        return noisy_weights

    def _resolve_persona_weights(self, persona_id: str) -> dict[str, float]:
        """Resolve a persona ID to its weight dictionary.

        Args:
            persona_id: Short name ("aggressive") or full ID ("heuristic_v1_aggressive")

        Returns:
            Weight dictionary for the persona, or default weights if not found.
        """
        from .heuristic_weights import get_weights

        # Support both short names and full IDs
        full_id = persona_id
        if not persona_id.startswith("heuristic_"):
            full_id = f"heuristic_v1_{persona_id}"

        persona_weights = get_weights(full_id)
        if not persona_weights:
            logger.warning(f"Persona '{persona_id}' not found, using default")
            return self._default_weights()
        return persona_weights

    def _generate_weights_list(self) -> list[list[dict[str, float]]]:
        """Generate per-game, per-player weights with optional noise and persona variety.

        Weight generation priority:
        1. per_player_personas: Fixed persona per player position (e.g., P1=aggressive, P2=defensive)
        2. persona_pool: Random persona sampling per game (same for all players in that game)
        3. Default: Board-specific optimized weights

        Returns:
            2D list of weight dictionaries: [game_idx][player_idx] -> weights
            Shape: (batch_size, num_players)
        """
        import random

        weights_list: list[list[dict[str, float]]] = []

        # Priority 1: Per-player personas (fixed persona per player position)
        if self.per_player_personas:
            # Resolve each player's persona to weights
            player_weights = [
                self._resolve_persona_weights(pid) for pid in self.per_player_personas
            ]

            for _ in range(self.batch_size):
                game_weights = []
                for player_idx in range(self.num_players):
                    pw = player_weights[player_idx].copy()
                    # Apply noise if configured
                    if self.weight_noise > 0:
                        pw = self._apply_weight_noise(pw)
                    game_weights.append(pw)
                weights_list.append(game_weights)

            return weights_list

        # Priority 2: Persona pool (random per game, same for all players)
        if self.persona_pool:
            for _ in range(self.batch_size):
                # Sample a persona for this game
                persona_id = random.choice(self.persona_pool)
                persona_weights = self._resolve_persona_weights(persona_id)

                # Apply noise if configured
                if self.weight_noise > 0:
                    persona_weights = self._apply_weight_noise(persona_weights)

                # All players in this game use the same weights
                weights_list.append([persona_weights] * self.num_players)

            return weights_list

        # Priority 3: Default weights
        base_weights = self._default_weights()

        if self.weight_noise <= 0:
            # No noise - all games and players use same weights
            player_weights = [base_weights] * self.num_players
            return [player_weights] * self.batch_size
        else:
            # Each game gets unique noisy weights (same for all players in game)
            for _ in range(self.batch_size):
                noisy = self._apply_weight_noise(base_weights)
                weights_list.append([noisy] * self.num_players)
            return weights_list

    def get_weights_for_current_players(
        self, weights_list: list[list[dict[str, float]]] | list[dict[str, float]] | None
    ) -> list[dict[str, float]] | None:
        """Get weights for the current player of each game.

        Supports multiple input formats for backward compatibility:
        - 2D format: list[list[dict]] - [game][player] -> dict
        - 1D format: list[dict] - [game] -> dict (same for all players)
        - Empty/None: Returns None (use default weights)

        Args:
            weights_list: Weights in any supported format

        Returns:
            List of weight dicts (one per game) for current player, or None for defaults.
        """
        # Handle None or empty
        if not weights_list:
            return None

        # Detect format by checking first element
        first_elem = weights_list[0]

        # 2D format: first element is a list of dicts
        if isinstance(first_elem, list):
            current_players = self.state.current_player.cpu().numpy()
            return [
                weights_list[game_idx][int(current_players[game_idx]) - 1]  # Players are 1-indexed
                for game_idx in range(self.batch_size)
            ]

        # 1D format: first element is a dict
        if isinstance(first_elem, dict):
            # If it's empty dict or single-element list, return None for defaults
            if not first_elem or len(weights_list) == 1:
                return None
            # Otherwise return as-is (one dict per game)
            return weights_list

        return None

    @staticmethod
    def get_available_personas() -> list[str]:
        """Get list of available persona IDs for training variety.

        These personas represent different play styles:
        - balanced: Well-rounded, optimized via CMA-ES
        - aggressive: Favors captures and elimination
        - territorial: Emphasizes space control and markers
        - defensive: Prioritizes safety and flexibility

        Returns:
            List of persona ID strings that can be used in persona_pool.
        """
        return ["balanced", "aggressive", "territorial", "defensive"]

    @staticmethod
    def get_all_persona_profiles() -> dict[str, dict[str, float]]:
        """Get all persona profiles with their full weight dictionaries.

        Useful for understanding what each persona emphasizes.

        Returns:
            Dict mapping persona ID to weight dictionary.
        """
        from .heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        personas = {}
        for short_name in ParallelGameRunner.get_available_personas():
            full_id = f"heuristic_v1_{short_name}"
            if full_id in HEURISTIC_WEIGHT_PROFILES:
                personas[short_name] = HEURISTIC_WEIGHT_PROFILES[full_id]

    # ==========================================================================
    # Matchup Configurations for Training
    # ==========================================================================

    # Predefined matchup configurations for systematic training
    TRAINING_MATCHUPS: dict[str, list[str]] = {
        # ===== 2-PLAYER MATCHUPS =====
        "aggressive_vs_defensive": ["aggressive", "defensive"],
        "territorial_vs_aggressive": ["territorial", "aggressive"],
        "balanced_vs_aggressive": ["balanced", "aggressive"],
        "balanced_vs_defensive": ["balanced", "defensive"],
        "balanced_vs_territorial": ["balanced", "territorial"],
        "defensive_vs_territorial": ["defensive", "territorial"],
        # Mirror matchups (same style vs same style)
        "aggressive_mirror": ["aggressive", "aggressive"],
        "defensive_mirror": ["defensive", "defensive"],
        "balanced_mirror": ["balanced", "balanced"],
        "territorial_mirror": ["territorial", "territorial"],
        # ===== 3-PLAYER MATCHUPS =====
        "3p_balanced": ["balanced", "balanced", "balanced"],
        "3p_mixed": ["aggressive", "defensive", "territorial"],
        "3p_aggressive": ["aggressive", "aggressive", "aggressive"],
        "3p_defensive": ["defensive", "defensive", "defensive"],
        "3p_territorial": ["territorial", "territorial", "territorial"],
        "3p_agg_def_bal": ["aggressive", "defensive", "balanced"],
        "3p_ter_agg_bal": ["territorial", "aggressive", "balanced"],
        # ===== 4-PLAYER MATCHUPS =====
        "4p_balanced": ["balanced", "balanced", "balanced", "balanced"],
        "4p_mixed": ["aggressive", "defensive", "territorial", "balanced"],
        "4p_aggressive": ["aggressive", "aggressive", "aggressive", "aggressive"],
        "4p_defensive": ["defensive", "defensive", "defensive", "defensive"],
        "4p_territorial": ["territorial", "territorial", "territorial", "territorial"],
        "4p_agg_vs_def": ["aggressive", "aggressive", "defensive", "defensive"],
        "4p_ter_vs_bal": ["territorial", "territorial", "balanced", "balanced"],
    }

    @classmethod
    def get_training_matchup(cls, matchup_name: str) -> list[str]:
        """Get a predefined training matchup configuration.

        Args:
            matchup_name: Name of the matchup (e.g., "aggressive_vs_defensive")

        Returns:
            List of persona IDs for each player.

        Raises:
            ValueError: If matchup_name is not recognized.
        """
        if matchup_name not in cls.TRAINING_MATCHUPS:
            available = list(cls.TRAINING_MATCHUPS.keys())
            raise ValueError(f"Unknown matchup '{matchup_name}'. Available: {available}")
        return cls.TRAINING_MATCHUPS[matchup_name]

    @classmethod
    def get_all_training_matchups(cls) -> list[str]:
        """Get all available training matchup names.

        Returns:
            List of matchup configuration names.
        """
        return list(cls.TRAINING_MATCHUPS.keys())

    @classmethod
    def create_with_matchup(
        cls,
        matchup_name: str,
        batch_size: int = 64,
        **kwargs,
    ) -> "ParallelGameRunner":
        """Factory method to create a runner with a predefined matchup.

        Args:
            matchup_name: Name of the matchup configuration
            batch_size: Number of games to run in parallel
            **kwargs: Additional arguments for ParallelGameRunner

        Returns:
            ParallelGameRunner configured with the specified matchup.

        Example:
            >>> runner = ParallelGameRunner.create_with_matchup(
            ...     "aggressive_vs_defensive",
            ...     batch_size=1000,
            ...     use_heuristic_selection=True,
            ... )
            >>> results = runner.run_games()
        """
        matchup = cls.get_training_matchup(matchup_name)
        return cls(
            batch_size=batch_size,
            num_players=len(matchup),
            per_player_personas=matchup,
            use_heuristic_selection=True,  # Enable heuristic selection by default
            **kwargs,
        )

    @classmethod
    def run_matchup_tournament(
        cls,
        matchups: list[str] | None = None,
        games_per_matchup: int = 100,
        max_moves: int = 200,
        **runner_kwargs,
    ) -> dict[str, dict[str, Any]]:
        """Run a tournament across multiple matchup configurations.

        This is useful for systematic training data generation with variety.

        Args:
            matchups: List of matchup names to run, or None for all matchups
            games_per_matchup: Number of games per matchup
            max_moves: Maximum moves per game
            **runner_kwargs: Additional arguments for ParallelGameRunner

        Returns:
            Dictionary mapping matchup name to results:
            {
                "aggressive_vs_defensive": {
                    "p1_wins": 45,
                    "p2_wins": 38,
                    "draws": 17,
                    "total_games": 100,
                    "avg_game_length": 87.5,
                },
                ...
            }

        Example:
            >>> results = ParallelGameRunner.run_matchup_tournament(
            ...     matchups=["aggressive_vs_defensive", "balanced_mirror"],
            ...     games_per_matchup=500,
            ... )
            >>> print(results["aggressive_vs_defensive"]["p1_wins"])
        """
        if matchups is None:
            matchups = cls.get_all_training_matchups()

        tournament_results = {}

        for matchup_name in matchups:
            runner = cls.create_with_matchup(
                matchup_name,
                batch_size=games_per_matchup,
                **runner_kwargs,
            )
            game_results = runner.run_games(max_moves=max_moves)

            # Aggregate results
            winners = game_results.get("winners", [])
            move_counts = game_results.get("move_counts", [])

            p1_wins = sum(1 for w in winners if w == 1)
            p2_wins = sum(1 for w in winners if w == 2)
            draws = sum(1 for w in winners if w == 0 or w is None)

            tournament_results[matchup_name] = {
                "p1_wins": p1_wins,
                "p2_wins": p2_wins,
                "draws": draws,
                "total_games": len(winners),
                "avg_game_length": sum(move_counts) / len(move_counts) if move_counts else 0,
                "personas": cls.get_training_matchup(matchup_name),
            }

        return tournament_results

    def get_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        return {
            "games_completed": self._games_completed,
            "total_moves": self._total_moves,
            "total_time_seconds": self._total_time,
            "games_per_second": (
                self._games_completed / self._total_time
                if self._total_time > 0 else 0
            ),
            "moves_per_second": (
                self._total_moves / self._total_time
                if self._total_time > 0 else 0
            ),
        }

    def get_shadow_validation_report(self) -> dict[str, Any] | None:
        """Get shadow validation statistics if enabled.

        Returns:
            Validation report dict if shadow validation enabled, None otherwise.
            Report includes:
                - total_validations: Number of moves validated
                - total_divergences: Number of divergences detected
                - divergence_rate: Divergence rate (0.0-1.0)
                - threshold: Configured threshold
                - status: "PASS" or "FAIL"
                - by_move_type: Breakdown by move type
        """
        if self.shadow_validator is None:
            return None
        return self.shadow_validator.get_report()


# =============================================================================
# CMA-ES Integration
# =============================================================================


def evaluate_candidate_fitness_gpu(
    candidate_weights: dict[str, float],
    opponent_weights: dict[str, float],
    num_games: int = 10,
    board_size: int = 8,
    num_players: int = 2,
    max_moves: int = 10000,
    device: torch.device | None = None,
) -> float:
    """Evaluate CMA-ES candidate fitness using GPU parallel games.

    Runs multiple games between candidate and opponent, returns win rate.

    Args:
        candidate_weights: Heuristic weights for candidate
        opponent_weights: Heuristic weights for opponent
        num_games: Number of games to play
        board_size: Board dimension
        num_players: Number of players
        max_moves: Max moves per game
        device: GPU device

    Returns:
        Win rate (0.0 to 1.0) for candidate
    """
    runner = ParallelGameRunner(
        batch_size=num_games,
        board_size=board_size,
        num_players=num_players,
        device=device,
    )

    # Alternate who plays first
    weights_list = []
    for i in range(num_games):
        if i % 2 == 0:
            weights_list.append(candidate_weights)  # Candidate is P1
        else:
            weights_list.append(opponent_weights)   # Opponent is P1

    results = runner.run_games(weights_list=weights_list, max_moves=max_moves)

    # Count wins for candidate
    wins = 0
    for i, winner in enumerate(results["winners"]):
        if i % 2 == 0:  # Candidate was P1
            if winner == 1:
                wins += 1
        else:  # Candidate was P2
            if winner == 2:
                wins += 1

    win_rate = wins / num_games

    logger.debug(
        f"GPU fitness evaluation: {wins}/{num_games} wins ({win_rate:.1%}), "
        f"{results['games_per_second']:.1f} games/sec"
    )

    return win_rate


def benchmark_parallel_games(
    batch_sizes: list[int] | None = None,
    board_size: int = 8,
    max_moves: int = 100,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Benchmark parallel game simulation performance.

    Args:
        batch_sizes: List of batch sizes to test
        board_size: Board dimension
        max_moves: Max moves per game
        device: GPU device

    Returns:
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]
    results = {
        "batch_size": [],
        "games_per_second": [],
        "moves_per_second": [],
        "elapsed_seconds": [],
    }

    for batch_size in batch_sizes:
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=board_size,
            device=device,
        )

        # Warmup
        runner.run_games(max_moves=10)

        # Benchmark
        game_results = runner.run_games(max_moves=max_moves)

        results["batch_size"].append(batch_size)
        results["games_per_second"].append(game_results["games_per_second"])
        results["moves_per_second"].append(
            sum(game_results["move_counts"]) / game_results["elapsed_seconds"]
        )
        results["elapsed_seconds"].append(game_results["elapsed_seconds"])

        logger.info(
            f"Batch {batch_size}: {game_results['games_per_second']:.1f} games/sec"
        )

    return results
