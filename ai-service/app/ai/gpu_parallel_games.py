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

CUDA provides the expected speedups. MPS optimization would require eliminating
~80 .item() calls and fully vectorizing all conditional logic.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

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
from .gpu_selection import select_moves_heuristic, select_moves_vectorized
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
# Parallel Game Runner
# =============================================================================


class ParallelGameRunner:
    """GPU-accelerated parallel game simulation.

    Runs multiple games simultaneously using batch operations on GPU.
    Supports different AI configurations per game for CMA-ES evaluation.

    Example:
        runner = ParallelGameRunner(batch_size=64, device="cuda")

        # Run games with specific heuristic weights
        results = runner.run_games(
            weights_per_game=[weights1, weights2, ...],  # length 64
            max_moves=10000,
        )

        # Results contain win/loss/draw for each game
    """

    def __init__(
        self,
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
    ):
        """Initialize parallel game runner.

        Args:
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
        """
        self.batch_size = batch_size
        self.board_size = board_size
        self.num_players = num_players
        self.swap_enabled = swap_enabled
        self.board_type = board_type
        self.use_heuristic_selection = use_heuristic_selection
        self.weight_noise = weight_noise
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.random_opening_moves = random_opening_moves
        self.use_policy_selection = False
        self.policy_model: RingRiftNNUEWithPolicy | None = None
        # Default LPS victory rounds to 3 if not specified
        self.lps_victory_rounds = lps_victory_rounds if lps_victory_rounds is not None else 3
        self.rings_per_player = rings_per_player

        if device is None:
            self.device = get_device()
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
            board_size=board_size,
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
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    self.policy_model = self.policy_model.half()
                    logger.info("ParallelGameRunner: Enabled FP16 inference for policy model")
                except Exception as e:
                    logger.debug(f"FP16 inference not available: {e}")

            self.use_policy_selection = True

            logger.info(f"ParallelGameRunner: Loaded policy model from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load policy model: {e}")
            return False

    def _select_moves(
        self,
        moves: BatchMoves,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select moves using configured selection strategy.

        Selection priority:
        1. Policy-based selection (if policy model loaded)
        2. Heuristic-based selection (if use_heuristic_selection=True)
        3. Fast center-bias selection (default)

        Uses self.temperature for softmax temperature (curriculum learning).
        During opening phase (move_count < random_opening_moves), uses very high
        temperature to make selections nearly uniform random.
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
                    return select_moves_heuristic(
                        moves, self.state, active_mask, temperature=elevated_temp
                    )
                else:
                    return select_moves_vectorized(
                        moves, active_mask, self.board_size, temperature=elevated_temp
                    )

        # Normal selection with configured strategy
        if self.use_policy_selection and self.policy_model is not None:
            return self._select_moves_policy(moves, active_mask, temperature=self.temperature)
        elif self.use_heuristic_selection:
            return select_moves_heuristic(
                moves, self.state, active_mask, temperature=self.temperature
            )
        else:
            return select_moves_vectorized(
                moves, active_mask, self.board_size, temperature=self.temperature
            )

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

            # === Vectorized Move Scoring ===
            # Score all moves across all games in parallel
            # Optimized 2025-12-14: Pre-extract numpy arrays to avoid .item() calls in loop
            center = self.board_size // 2
            center_idx = center * self.board_size + center
            num_positions = self.board_size * self.board_size

            # Pre-extract game indices and move metadata to avoid per-iteration .item() calls
            game_indices_np = games_with_moves.cpu().numpy()
            move_offsets_np = moves.move_offsets[games_with_moves].cpu().numpy()
            moves_per_game_np = moves.moves_per_game[games_with_moves].cpu().numpy()

            for local_idx in range(len(game_indices_np)):
                g = int(game_indices_np[local_idx])
                move_start = int(move_offsets_np[local_idx])
                move_count = int(moves_per_game_np[local_idx])

                if move_count == 0:
                    continue

                # Get from/to positions for all moves of this game
                from_y = moves.from_y[move_start:move_start + move_count]
                from_x = moves.from_x[move_start:move_start + move_count]
                to_y = moves.to_y[move_start:move_start + move_count]
                to_x = moves.to_x[move_start:move_start + move_count]

                # Compute flat indices vectorized (use center for negative coords)
                from_idx = torch.where(
                    from_y >= 0,
                    from_y * self.board_size + from_x,
                    torch.full_like(from_y, center_idx)
                ).long()
                to_idx = torch.where(
                    to_y >= 0,
                    to_y * self.board_size + to_x,
                    torch.full_like(to_y, center_idx)
                ).long()

                # Clamp indices to valid range
                from_idx = from_idx.clamp(0, num_positions - 1)
                to_idx = to_idx.clamp(0, num_positions - 1)

                # Get logits for this game
                from_logits = from_logits_batch[local_idx]
                to_logits = to_logits_batch[local_idx]

                # Compute move scores vectorized
                move_scores = from_logits[from_idx] + to_logits[to_idx]

                # Sample move with temperature (single .item() per game for result)
                probs = torch.softmax(move_scores / temperature, dim=0)
                selected_local = int(torch.multinomial(probs, 1).item())
                selected[g] = selected_local

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

            current_player = self.state.current_player[game_idx].item()
            if current_player < 1:
                current_player = 1

            # Extract stack ownership for each player (simplified)
            # Planes 0-3: Ring presence, 4-7: Stack presence, 8-11: Territory
            for y in range(board_size):
                for x in range(board_size):
                    pos_idx = y * board_size + x
                    owner = self.state.stack_owner[game_idx, y, x].item()
                    height = self.state.stack_height[game_idx, y, x].item()

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
                    territory_owner = self.state.territory_owner[game_idx, y, x].item()
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
        weights_list: list[dict[str, float]] | None = None,
        max_moves: int = 10000,
        callback: Callable[[int, BatchGameState], None] | None = None,
        snapshot_interval: int = 0,
        snapshot_callback: Callable[[int, int, "GameState"], None] | None = None,
        emit_events: bool = True,
        task_id: str | None = None,
        iteration: int = 0,
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

            # Capture snapshots for games that have crossed the interval threshold
            if snapshot_interval > 0 and snapshot_callback:
                current_moves = self.state.move_count
                # Find games where move_count has advanced past last_snapshot + interval
                need_snapshot = (current_moves >= last_snapshot_move + snapshot_interval) & active_mask
                if need_snapshot.any():
                    for g in need_snapshot.nonzero(as_tuple=True)[0].tolist():
                        try:
                            move_num = current_moves[g].item()
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

    def _step_games(self, weights_list: list[dict[str, float]]) -> None:
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

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Extract GPU moves for this game
            move_start = moves.move_offsets[g].item()
            move_count = moves.moves_per_game[g].item()

            if move_count == 0:
                continue

            gpu_positions = []
            for i in range(move_count):
                idx = move_start + i
                # Placement moves store position in from_y, from_x (target position)
                row = moves.from_y[idx].item()
                col = moves.from_x[idx].item()

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
            player = self.state.current_player[g].item()

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

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Hex coordinate conversion helper
            is_hex = self.board_type and self.board_type.lower() in ("hexagonal", "hex", "hex8")
            hex_center = self.board_size // 2 if is_hex else 0

            def to_cpu_coords(row: int, col: int, *, _is_hex: bool = is_hex, _hex_center: int = hex_center):
                """Convert GPU grid to CPU coords."""
                if _is_hex:
                    return col - _hex_center, row - _hex_center
                return col, row

            # Extract GPU movement moves
            move_start = movement_moves.move_offsets[g].item()
            move_count = movement_moves.moves_per_game[g].item()

            gpu_movement = []
            for i in range(move_count):
                idx = move_start + i
                from_row = movement_moves.from_y[idx].item()
                from_col = movement_moves.from_x[idx].item()
                to_row = movement_moves.to_y[idx].item()
                to_col = movement_moves.to_x[idx].item()
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_movement.append(((from_x, from_y), (to_x, to_y)))

            # Extract GPU capture moves
            cap_start = capture_moves.move_offsets[g].item()
            cap_count = capture_moves.moves_per_game[g].item()

            gpu_captures = []
            for i in range(cap_count):
                idx = cap_start + i
                from_row = capture_moves.from_y[idx].item()
                from_col = capture_moves.from_x[idx].item()
                to_row = capture_moves.to_y[idx].item()
                to_col = capture_moves.to_x[idx].item()
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_captures.append(((from_x, from_y), (to_x, to_y)))

            # Convert to CPU state and validate
            cpu_state = self.state.to_game_state(g)
            player = self.state.current_player[g].item()

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
        """
        # Placement: treat any remaining rings in hand as a real action.
        if self.state.rings_in_hand[g, player].item() > 0:
            return True

        # Without controlled stacks, there is no movement/capture.
        if not (self.state.stack_owner[g] == player).any().item():
            return False

        prev_player = int(self.state.current_player[g].item())
        self.state.current_player[g] = player
        try:
            single_mask = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            single_mask[g] = True
            movement_moves = generate_movement_moves_batch(self.state, single_mask)
            capture_moves = generate_capture_moves_batch(self.state, single_mask)
            return bool(
                (movement_moves.moves_per_game[g].item() > 0)
                or (capture_moves.moves_per_game[g].item() > 0)
            )
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
        controls_stack = (self.state.stack_owner == player).any(dim=(1, 2))
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
            exclusive_pid = torch.argmax(real_action_players.to(torch.int8), dim=1).to(torch.int8)
            exclusive_pid = torch.where(
                true_counts == 1,
                exclusive_pid,
                torch.zeros_like(exclusive_pid),
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
        weights_list: list[dict[str, float]],
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
        controls_stack = (self.state.stack_owner == player_expanded).any(dim=(1, 2))
        has_marker = (self.state.marker_owner == player_expanded).any(dim=(1, 2))
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
                # Use configured move selection strategy
                selected = self._select_moves(moves, games_with_rings)
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
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        players = self.state.current_player[game_indices]

        for i, g in enumerate(game_indices.tolist()):
            move_count = int(self.state.move_count[g].item())
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players[i].item())

            # Record skip_placement (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
            self.state.move_history[g, move_count, 0] = MoveType.SKIP_PLACEMENT
            self.state.move_history[g, move_count, 1] = player
            self.state.move_history[g, move_count, 2] = -1  # No position for skip
            self.state.move_history[g, move_count, 3] = -1
            self.state.move_history[g, move_count, 4] = -1
            self.state.move_history[g, move_count, 5] = -1
            self.state.move_history[g, move_count, 6] = GamePhase.RING_PLACEMENT

            self.state.move_count[g] += 1

    def _record_skip_capture_moves(self, mask: torch.Tensor) -> None:
        """Record skip_capture moves for canonical compliance.

        Per canonical contract, when a player finishes capturing (no more chain
        captures available), an explicit skip_capture move must be recorded before
        transitioning to line_processing.

        December 2025: Added for GPU/CPU phase machine parity.

        Args:
            mask: Games where skip_capture should be recorded
        """
        from .gpu_game_types import MoveType

        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        game_indices = torch.where(active_mask)[0]
        players = self.state.current_player[game_indices]

        for i, g in enumerate(game_indices.tolist()):
            move_count = int(self.state.move_count[g].item())
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players[i].item())

            # Record skip_capture (7 columns: move_type, player, from_y, from_x, to_y, to_x, phase)
            self.state.move_history[g, move_count, 0] = MoveType.SKIP_CAPTURE
            self.state.move_history[g, move_count, 1] = player
            self.state.move_history[g, move_count, 2] = -1  # No position for skip
            self.state.move_history[g, move_count, 3] = -1
            self.state.move_history[g, move_count, 4] = -1
            self.state.move_history[g, move_count, 5] = -1
            # Use CAPTURE phase for skip_capture (chain_capture would also be valid)
            self.state.move_history[g, move_count, 6] = GamePhase.CAPTURE

            self.state.move_count[g] += 1

    def _step_movement_phase(
        self,
        mask: torch.Tensor,
        weights_list: list[dict[str, float]],
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
        has_any_stack = (self.state.stack_owner == player_expanded).any(dim=(1, 2))
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
                selected_recovery = self._select_moves(recovery_moves, games_with_recovery)
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
            games_with_constrained_captures = torch.zeros_like(games_with_stacks)
            if has_must_move_constraint.any():
                for g in torch.where(has_must_move_constraint & games_with_stacks)[0].tolist():
                    if capture_moves.moves_per_game[g].item() == 0:
                        continue
                    # Check if any capture originates from must_move_from position
                    start = capture_moves.move_offsets[g].item()
                    count = capture_moves.moves_per_game[g].item()
                    must_y = self.state.must_move_from_y[g].item()
                    must_x = self.state.must_move_from_x[g].item()
                    for idx in range(start, start + count):
                        if capture_moves.from_y[idx].item() == must_y and capture_moves.from_x[idx].item() == must_x:
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
                selected_captures = self._select_moves(capture_moves, games_unconstrained_captures)
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
            if games_constrained_captures.any():
                for g in torch.where(games_constrained_captures)[0].tolist():
                    # Get captures from must_move_from position
                    from_y = int(self.state.must_move_from_y[g].item())
                    from_x = int(self.state.must_move_from_x[g].item())

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
                selected_movements = self._select_moves(movement_moves, games_movement_only)

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

                for idx, g in enumerate(movement_game_indices.tolist()):
                    if not movement_valid[idx]:
                        continue
                    landing_y = int(movement_landing_y[idx].item())
                    landing_x = int(movement_landing_x[idx].item())

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

        # Record moves
        for i, g in enumerate(game_indices.tolist()):
            move_count = int(self.state.move_count[g].item())
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players[i].item())

            if had_lines[i]:
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
        """
        result = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        active_mask = mask & games_with_territory & self.state.get_active_mask()
        if not active_mask.any():
            return result

        # Get current player for each game
        current_players = self.state.current_player

        # Compute player ring status
        player_has_rings = self._compute_player_ring_status_batch()

        # In 2-player games, check if the opponent has no rings
        if self.num_players == 2:
            for g in torch.where(active_mask)[0].tolist():
                current_p = int(current_players[g].item())
                opponent_p = 2 if current_p == 1 else 1
                if not player_has_rings[g, opponent_p]:
                    result[g] = True

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
        """
        result = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        if not mask.any():
            return result

        # Territory victory minimum: floor(totalSpaces / numPlayers) + 1
        total_spaces = self.board_size * self.board_size
        min_threshold = total_spaces // self.num_players + 1

        for g in torch.where(mask)[0].tolist():
            current_p = int(self.state.current_player[g].item())
            player_territory = int(self.state.territory_count[g, current_p].item())

            # Check condition 1: player has minimum
            if player_territory < min_threshold:
                continue

            # Check condition 2: player dominates all opponents combined
            opponent_total = 0
            for opp in range(1, self.num_players + 1):
                if opp != current_p:
                    opponent_total += int(self.state.territory_count[g, opp].item())

            if player_territory > opponent_total:
                result[g] = True

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

        for g in torch.where(mask)[0].tolist():
            # Check if game is still active
            if self.state.game_status[g].item() != GameStatus.ACTIVE:
                continue

            player = int(self.state.current_player[g].item())

            # Get board state as numpy for efficiency
            marker_owner_np = self.state.marker_owner[g].cpu().numpy()
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

        # Record moves
        for i, g in enumerate(game_indices.tolist()):
            move_count = int(self.state.move_count[g].item())
            if move_count >= self.state.max_history_moves:
                continue

            player = int(players[i].item())

            if had_territory[i]:
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
            has_controlled = (self.state.stack_owner == p).any(dim=(1, 2))

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
        """
        # Check rings in hand
        if self.state.rings_in_hand[g, player].item() > 0:
            return True

        # Check controlled stacks (rings in stacks we control)
        has_controlled_stacks = (self.state.stack_owner[g] == player).any().item()
        if has_controlled_stacks:
            return True

        # Check buried rings (rings in opponent-controlled stacks)
        # buried_rings uses 1-indexed players (shape is batch_size x num_players+1)
        return self.state.buried_rings[g, player].item() > 0

    def _apply_single_capture(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single capture move for game g at global index move_idx.

        Per RR-CANON-R100-R103:
        - Attacker moves onto defender stack
        - Defender's top ring is eliminated
        - Stacks merge (attacker on top)
        - Control transfers to attacker
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        # 9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x
        move_count = state.move_count[g].item()
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
            is_chain = state.in_capture_chain[g].item()
            current_phase = state.current_phase[g].item()
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

        # Get moving stack info
        attacker_height = state.stack_height[g, from_y, from_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()
        defender_owner = state.stack_owner[g, to_y, to_x].item()

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
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
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        move_count = state.move_count[g].item()
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x
            state.move_history[g, move_count, 6] = GamePhase.MOVEMENT

        # Get moving stack info
        moving_height = state.stack_height[g, from_y, from_x].item()

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, int(dist)):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                # Flip opponent marker to our color
                state.marker_owner[g, check_y, check_x] = player

        # Handle landing on ANY marker (own or opponent) - per RR-CANON-R092:
        # The marker is removed and the top ring of the landing stack is eliminated.
        # Note: Landing does NOT collapse the position (only path markers collapse).
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 0
        if dest_marker != 0:
            # Landing on any marker costs 1 ring (cap elimination)
            landing_ring_cost = 1
            state.marker_owner[g, to_y, to_x] = 0  # Marker removed

        # Clear origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        # Handle destination
        dest_owner = state.stack_owner[g, to_y, to_x].item()
        dest_height = state.stack_height[g, to_y, to_x].item()

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
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        move_count = state.move_count[g].item()
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
        current_buried = state.buried_rings[g, player].item()
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

            if line_counts[g].item() == 0:
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
        weights_list: list[dict[str, float]],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select the best move for each game using heuristic evaluation.

        This is a simplified version - a full implementation would:
        1. Apply each candidate move to a temporary state
        2. Evaluate the resulting position
        3. Select the move with best score

        For now, we select randomly with bias toward center positions.
        """
        batch_size = self.batch_size
        device = self.device

        # Simple selection: prefer center positions
        selected = torch.zeros(batch_size, dtype=torch.int64, device=device)

        center = self.board_size // 2

        for g in range(batch_size):
            if not active_mask[g] or moves.moves_per_game[g] == 0:
                continue

            # Get moves for this game
            start_idx = moves.move_offsets[g]
            end_idx = start_idx + moves.moves_per_game[g]

            game_moves_y = moves.from_y[start_idx:end_idx]
            game_moves_x = moves.from_x[start_idx:end_idx]

            # Score by distance to center (lower is better -> invert for softmax)
            dist_to_center = (
                (game_moves_y.float() - center).abs() +
                (game_moves_x.float() - center).abs()
            )

            # Convert to scores: higher score for closer to center
            max_dist = center * 2  # Maximum possible Manhattan distance
            scores = (max_dist - dist_to_center) + torch.rand_like(dist_to_center) * 2.0

            # Softmax selection with temperature=1.0 for stochasticity
            probs = torch.softmax(scores, dim=0)
            best_local_idx = torch.multinomial(probs, 1).item()
            selected[g] = best_local_idx

        return selected

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
        has_stacks = player_mask.any(dim=(1, 2))

        # Check if player has rings in hand (vectorized using gather)
        player_indices = current_players.unsqueeze(-1)  # (batch_size, 1)
        has_rings_in_hand = (
            torch.gather(state.rings_in_hand, 1, player_indices).squeeze(-1) > 0
        )

        # Check markers (vectorized)
        marker_mask = (
            state.marker_owner
            == current_players.unsqueeze(-1).unsqueeze(-1)
        )
        has_markers = marker_mask.any(dim=(1, 2))

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

    def _generate_weights_list(self) -> list[dict[str, float]]:
        """Generate per-game weights with optional noise.

        Returns:
            List of weight dictionaries, one per game in batch.
        """
        base_weights = self._default_weights()
        if self.weight_noise <= 0:
            # No noise - all games use same weights
            return [base_weights] * self.batch_size
        else:
            # Each game gets unique noisy weights
            return [self._apply_weight_noise(base_weights) for _ in range(self.batch_size)]

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
