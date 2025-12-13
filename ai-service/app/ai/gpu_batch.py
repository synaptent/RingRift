"""GPU-accelerated batch inference and evaluation for RingRift AI.

This module provides GPU-accelerated batch processing for:
1. Neural network inference (batch evaluation of positions)
2. Feature extraction (board state to tensor conversion)
3. Heuristic batch evaluation (vectorized move scoring on GPU)
4. Async batch queue management for MCTS integration

The module auto-detects available hardware and falls back gracefully:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

Usage:
    from app.ai.gpu_batch import GPUBatchEvaluator, get_device

    # Auto-detect best device
    device = get_device()

    # Create evaluator with optional model
    evaluator = GPUBatchEvaluator(device=device, model=my_nn_model)

    # Batch evaluate positions
    values, policies = evaluator.evaluate_batch(states)

    # Async queue for MCTS
    async_evaluator = AsyncGPUEvaluator(evaluator, batch_size=64)
    async_evaluator.queue_position(state, callback)
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# Device Detection and Management
# =============================================================================


def get_device(prefer_gpu: bool = True, device_id: int = 0) -> torch.device:
    """Auto-detect the best available compute device.

    Priority order:
    1. CUDA (if available and prefer_gpu=True)
    2. MPS (Apple Silicon, if available and prefer_gpu=True)
    3. CPU (fallback)

    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        device_id: CUDA device ID to use (ignored for MPS/CPU)

    Returns:
        torch.device for the selected compute device
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            props = torch.cuda.get_device_properties(device_id)
            logger.info(
                f"Using CUDA device {device_id}: {props.name} "
                f"({props.total_memory / 1024**3:.1f}GB)"
            )
            return device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon)")
            return torch.device("mps")

    logger.info("Using CPU")
    return torch.device("cpu")


def get_all_cuda_devices() -> List[torch.device]:
    """Get list of all available CUDA devices."""
    if not torch.cuda.is_available():
        return []
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def clear_gpu_memory(device: Optional[torch.device] = None) -> None:
    """Clear GPU memory caches.

    Args:
        device: Specific device to clear, or None for all devices
    """
    gc.collect()

    if torch.cuda.is_available():
        if device is None or device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        if device is None or device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


# =============================================================================
# Model Compilation (torch.compile optimization)
# =============================================================================


def compile_model(
    model: nn.Module,
    device: Optional[torch.device] = None,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> nn.Module:
    """Apply torch.compile() optimization to a model for faster inference.

    torch.compile() provides 2-3x speedup for inference by:
    - Fusing operations into optimized kernels
    - Reducing Python overhead
    - Enabling GPU-specific optimizations (TensorRT-like)

    Args:
        model: PyTorch model to compile
        device: Target device (used to check compatibility)
        mode: Compilation mode:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Minimize Python overhead (best for inference)
            - "max-autotune": Maximum optimization (longer compile time)
        fullgraph: If True, require entire model in single graph (stricter)

    Returns:
        Compiled model (or original if compilation fails/unsupported)

    Note:
        - Compilation happens on first forward pass (lazy)
        - MPS has limited torch.compile() support (will skip gracefully)
        - Requires PyTorch 2.0+
    """
    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 0):
        logger.debug(f"torch.compile() requires PyTorch 2.0+, got {torch.__version__}")
        return model

    # MPS has limited compile support - skip for now
    if device is not None and device.type == "mps":
        logger.debug("Skipping torch.compile() on MPS (limited support)")
        return model

    # Check if already compiled
    if hasattr(model, "_compiled") and model._compiled:
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=True,  # Support dynamic batch sizes
        )
        # Mark as compiled to avoid re-compilation
        compiled._compiled = True
        logger.info(f"Applied torch.compile(mode={mode}) to model")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile() failed, using uncompiled model: {e}")
        return model


def warmup_compiled_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_warmup: int = 3,
) -> None:
    """Warmup a compiled model to trigger JIT compilation.

    The first few forward passes of a compiled model are slow due to
    JIT compilation. This function runs warmup passes to "pay" this
    cost upfront.

    Args:
        model: Compiled model to warmup
        sample_input: Representative input tensor
        num_warmup: Number of warmup passes
    """
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_input)

    # Sync GPU
    if sample_input.device.type == "cuda":
        torch.cuda.synchronize()

    logger.debug(f"Completed {num_warmup} warmup passes for compiled model")


# =============================================================================
# GPU Tensor Utilities
# =============================================================================


@dataclass
class GPUBoardState:
    """GPU-resident board state representation for batch processing.

    All tensors are kept on GPU to minimize data transfer overhead.
    """

    # Board state tensors (batch_size, board_size, board_size) or (batch_size, num_positions)
    stack_owner: torch.Tensor
    stack_height: torch.Tensor
    marker_owner: torch.Tensor
    territory_owner: torch.Tensor
    is_collapsed: torch.Tensor

    # Player state tensors (batch_size, num_players)
    rings_in_hand: torch.Tensor
    player_eliminated: torch.Tensor
    territory_count: torch.Tensor

    # Metadata
    device: torch.device
    batch_size: int
    board_size: int

    @classmethod
    def from_numpy_batch(
        cls,
        board_states: List[Dict[str, np.ndarray]],
        device: torch.device,
        board_size: int = 8,
    ) -> "GPUBoardState":
        """Create GPU board state from list of numpy state dicts.

        Args:
            board_states: List of state dictionaries with numpy arrays
            device: Target GPU device
            board_size: Board dimension

        Returns:
            GPUBoardState with all tensors on GPU
        """
        batch_size = len(board_states)

        # Stack numpy arrays and convert to GPU tensors
        stack_owner = torch.from_numpy(
            np.stack([s.get("stack_owner", np.zeros(board_size * board_size)) for s in board_states])
        ).to(device, dtype=torch.int8)

        stack_height = torch.from_numpy(
            np.stack([s.get("stack_height", np.zeros(board_size * board_size)) for s in board_states])
        ).to(device, dtype=torch.int8)

        marker_owner = torch.from_numpy(
            np.stack([s.get("marker_owner", np.zeros(board_size * board_size)) for s in board_states])
        ).to(device, dtype=torch.int8)

        territory_owner = torch.from_numpy(
            np.stack([s.get("territory_owner", np.zeros(board_size * board_size)) for s in board_states])
        ).to(device, dtype=torch.int8)

        is_collapsed = torch.from_numpy(
            np.stack([s.get("is_collapsed", np.zeros(board_size * board_size, dtype=bool)) for s in board_states])
        ).to(device, dtype=torch.bool)

        # Player state (assume max 4 players)
        rings_in_hand = torch.from_numpy(
            np.stack([s.get("rings_in_hand", np.zeros(5)) for s in board_states])
        ).to(device, dtype=torch.int16)

        player_eliminated = torch.from_numpy(
            np.stack([s.get("player_eliminated", np.zeros(5)) for s in board_states])
        ).to(device, dtype=torch.int16)

        territory_count = torch.from_numpy(
            np.stack([s.get("territory_count", np.zeros(5)) for s in board_states])
        ).to(device, dtype=torch.int16)

        return cls(
            stack_owner=stack_owner,
            stack_height=stack_height,
            marker_owner=marker_owner,
            territory_owner=territory_owner,
            is_collapsed=is_collapsed,
            rings_in_hand=rings_in_hand,
            player_eliminated=player_eliminated,
            territory_count=territory_count,
            device=device,
            batch_size=batch_size,
            board_size=board_size,
        )

    @classmethod
    def from_game_states(
        cls,
        game_states: List[Any],  # List[GameState] - use Any to avoid circular import
        device: "torch.device",
    ) -> "GPUBoardState":
        """Create GPU board state from list of GameState objects.

        Converts the high-level GameState representation to GPU tensors.

        Args:
            game_states: List of GameState objects
            device: Target GPU device

        Returns:
            GPUBoardState with all tensors on GPU
        """
        if not game_states:
            raise ValueError("game_states cannot be empty")

        # Infer board size from first state
        first_state = game_states[0]
        board_size = first_state.board.size

        batch_size = len(game_states)
        num_positions = board_size * board_size
        num_players = 5  # Max players + 1 for 0-indexing

        # Pre-allocate numpy arrays
        stack_owner = np.zeros((batch_size, num_positions), dtype=np.int8)
        stack_height = np.zeros((batch_size, num_positions), dtype=np.int8)
        marker_owner = np.zeros((batch_size, num_positions), dtype=np.int8)
        territory_owner = np.zeros((batch_size, num_positions), dtype=np.int8)
        is_collapsed = np.zeros((batch_size, num_positions), dtype=bool)
        rings_in_hand = np.zeros((batch_size, num_players), dtype=np.int16)
        player_eliminated = np.zeros((batch_size, num_players), dtype=np.int16)
        territory_count = np.zeros((batch_size, num_players), dtype=np.int16)

        for i, state in enumerate(game_states):
            board = state.board

            # Convert stacks
            for key, stack in board.stacks.items():
                try:
                    # Parse key "x,y" format
                    parts = key.split(",")
                    x, y = int(parts[0]), int(parts[1])
                    idx = y * board_size + x
                    if 0 <= idx < num_positions:
                        stack_owner[i, idx] = stack.controlling_player or 0
                        stack_height[i, idx] = stack.stack_height or 0
                except (ValueError, IndexError):
                    pass

            # Convert markers
            for key, marker in board.markers.items():
                try:
                    parts = key.split(",")
                    x, y = int(parts[0]), int(parts[1])
                    idx = y * board_size + x
                    if 0 <= idx < num_positions:
                        marker_owner[i, idx] = marker.player or 0
                except (ValueError, IndexError):
                    pass

            # Convert collapsed spaces
            for key, owner in board.collapsed_spaces.items():
                try:
                    parts = key.split(",")
                    x, y = int(parts[0]), int(parts[1])
                    idx = y * board_size + x
                    if 0 <= idx < num_positions:
                        is_collapsed[i, idx] = True
                        territory_owner[i, idx] = owner or 0
                except (ValueError, IndexError):
                    pass

            # Convert player state
            for player in state.players:
                pn = player.player_number
                if 0 <= pn < num_players:
                    rings_in_hand[i, pn] = player.rings_in_hand
                    player_eliminated[i, pn] = player.eliminated_rings
                    territory_count[i, pn] = player.territory_spaces

        # Convert to GPU tensors
        return cls(
            stack_owner=torch.from_numpy(stack_owner).to(device),
            stack_height=torch.from_numpy(stack_height).to(device),
            marker_owner=torch.from_numpy(marker_owner).to(device),
            territory_owner=torch.from_numpy(territory_owner).to(device),
            is_collapsed=torch.from_numpy(is_collapsed).to(device),
            rings_in_hand=torch.from_numpy(rings_in_hand).to(device),
            player_eliminated=torch.from_numpy(player_eliminated).to(device),
            territory_count=torch.from_numpy(territory_count).to(device),
            device=device,
            batch_size=batch_size,
            board_size=board_size,
        )


# =============================================================================
# GPU Batch Evaluator
# =============================================================================


class GPUBatchEvaluator:
    """GPU-accelerated batch evaluation for neural network inference.

    This class provides efficient batch processing of game positions using
    GPU acceleration. It handles:
    - Automatic batching of positions
    - Mixed precision inference (fp16) for faster throughput
    - Memory-efficient tensor management
    - Fallback to CPU when GPU is unavailable

    Example:
        evaluator = GPUBatchEvaluator(device="cuda", model=my_model)
        values, policies = evaluator.evaluate_batch(states, valid_moves_batch)
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[nn.Module] = None,
        use_mixed_precision: bool = True,
        max_batch_size: int = 256,
    ):
        """Initialize GPU batch evaluator.

        Args:
            device: Compute device (auto-detected if None)
            model: Neural network model for evaluation
            use_mixed_precision: Use fp16 for faster inference on supported GPUs
            max_batch_size: Maximum batch size for memory management
        """
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()

        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.max_batch_size = max_batch_size

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0

        logger.info(
            f"GPUBatchEvaluator initialized on {self.device} "
            f"(mixed_precision={self.use_mixed_precision})"
        )

    def set_model(self, model: nn.Module) -> None:
        """Set or update the neural network model.

        Args:
            model: Neural network model to use for evaluation
        """
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_batch(
        self,
        feature_batch: np.ndarray,
        global_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions using the neural network.

        Args:
            feature_batch: Batch of feature tensors (batch_size, channels, height, width)
            global_features: Optional global feature vectors (batch_size, num_features)

        Returns:
            Tuple of (values, policies) as numpy arrays
            - values: Shape (batch_size,) or (batch_size, num_players)
            - policies: Shape (batch_size, policy_size)
        """
        if self.model is None:
            raise ValueError("No model set for evaluation")

        batch_size = len(feature_batch)
        start_time = time.perf_counter()

        # Convert to GPU tensors
        features = torch.from_numpy(feature_batch).float().to(self.device)
        if global_features is not None:
            globals_t = torch.from_numpy(global_features).float().to(self.device)
        else:
            globals_t = None

        # Process in chunks if batch is too large
        if batch_size > self.max_batch_size:
            return self._evaluate_chunked(features, globals_t)

        # Run inference with optional mixed precision
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                values, policies = self._forward(features, globals_t)
        else:
            values, policies = self._forward(features, globals_t)

        # Convert back to numpy
        values_np = values.cpu().numpy()
        policies_np = policies.cpu().numpy()

        # Track performance
        elapsed = time.perf_counter() - start_time
        self._inference_count += batch_size
        self._total_inference_time += elapsed

        return values_np, policies_np

    def _forward(
        self,
        features: torch.Tensor,
        globals_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            features: Feature tensor on GPU
            globals_t: Global features tensor on GPU (optional)

        Returns:
            Tuple of (values, policies) tensors
        """
        if globals_t is not None:
            output = self.model(features, globals_t)
        else:
            output = self.model(features)

        # Handle different output formats
        if isinstance(output, tuple):
            values, policies = output[0], output[1]
        else:
            values = output
            policies = torch.zeros(features.shape[0], 1, device=self.device)

        return values, policies

    def _evaluate_chunked(
        self,
        features: torch.Tensor,
        globals_t: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate large batch in chunks to manage memory.

        Args:
            features: Full feature tensor
            globals_t: Full global features tensor (optional)

        Returns:
            Tuple of (values, policies) as numpy arrays
        """
        batch_size = features.shape[0]
        all_values = []
        all_policies = []

        for i in range(0, batch_size, self.max_batch_size):
            end_idx = min(i + self.max_batch_size, batch_size)
            chunk_features = features[i:end_idx]
            chunk_globals = globals_t[i:end_idx] if globals_t is not None else None

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    values, policies = self._forward(chunk_features, chunk_globals)
            else:
                values, policies = self._forward(chunk_features, chunk_globals)

            all_values.append(values.cpu().numpy())
            all_policies.append(policies.cpu().numpy())

        return np.concatenate(all_values), np.concatenate(all_policies)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics.

        Returns:
            Dictionary with inference count, total time, and average throughput
        """
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0
        )
        throughput = (
            self._inference_count / self._total_inference_time
            if self._total_inference_time > 0
            else 0
        )

        return {
            "inference_count": self._inference_count,
            "total_time_seconds": self._total_inference_time,
            "avg_time_per_sample": avg_time,
            "throughput_samples_per_sec": throughput,
        }


# =============================================================================
# GPU Heuristic Evaluator
# =============================================================================


class GPUHeuristicEvaluator:
    """GPU-accelerated heuristic position evaluation.

    Vectorized computation of heuristic features on GPU for batch move scoring.
    Significantly faster than CPU for large batches (>32 positions).
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        board_size: int = 8,
        num_players: int = 2,
    ):
        """Initialize GPU heuristic evaluator.

        Args:
            device: Compute device (auto-detected if None)
            board_size: Board dimension (8 or 19)
            num_players: Number of players (2-4)
        """
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.board_size = board_size
        self.num_players = num_players
        self.num_positions = board_size * board_size

        # Pre-compute center mask on GPU
        self._build_center_mask()

        # Default heuristic weights (can be overridden)
        self.weights = self._default_weights()

    def _build_center_mask(self) -> None:
        """Build center position mask on GPU."""
        center = self.board_size // 2
        mask = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        # Define center region (inner 40% of board)
        inner_radius = self.board_size * 0.2
        for y in range(self.board_size):
            for x in range(self.board_size):
                dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                if dist <= inner_radius:
                    mask[y, x] = 1.0
                elif dist <= inner_radius * 1.5:
                    mask[y, x] = 0.5

        self.center_mask = torch.from_numpy(mask).to(self.device)

    def _default_weights(self) -> Dict[str, float]:
        """Default heuristic weights."""
        return {
            "stack_count": 1.0,
            "ring_count": 0.5,
            "territory_count": 2.0,
            "center_control": 0.3,
            "mobility": 0.1,
            "no_stacks_penalty": -100.0,
        }

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set heuristic weights.

        Args:
            weights: Dictionary mapping feature names to weights
        """
        self.weights.update(weights)

    @torch.no_grad()
    def evaluate_batch(
        self,
        board_states: GPUBoardState,
        player_number: int,
    ) -> torch.Tensor:
        """Evaluate batch of positions for a player using heuristics.

        Args:
            board_states: GPUBoardState with batch of positions
            player_number: Player to evaluate for (1-4)

        Returns:
            Tensor of scores (batch_size,)
        """
        batch_size = board_states.batch_size

        # Compute features on GPU
        my_stacks = (board_states.stack_owner == player_number).sum(dim=-1).float()
        opp_stacks = (
            (board_states.stack_owner > 0) & (board_states.stack_owner != player_number)
        ).sum(dim=-1).float()

        my_territory = board_states.territory_count[:, player_number].float()
        my_rings = board_states.rings_in_hand[:, player_number].float()

        # Center control (requires reshape for 2D mask)
        stack_2d = board_states.stack_owner.view(batch_size, self.board_size, self.board_size)
        center_control = ((stack_2d == player_number).float() * self.center_mask).sum(dim=(1, 2))

        # Compute final score
        w = self.weights
        scores = (
            (my_stacks - opp_stacks) * w["stack_count"]
            + my_rings * w["ring_count"]
            + my_territory * w["territory_count"]
            + center_control * w["center_control"]
        )

        # No stacks penalty
        scores = torch.where(
            my_stacks == 0,
            scores + w["no_stacks_penalty"],
            scores,
        )

        return scores

    def evaluate_moves_batch(
        self,
        base_state: GPUBoardState,
        move_results: List[GPUBoardState],
        player_number: int,
    ) -> np.ndarray:
        """Evaluate multiple candidate moves from same base state.

        Args:
            base_state: Starting position
            move_results: List of states after each candidate move
            player_number: Player to evaluate for

        Returns:
            Array of scores for each move
        """
        # Combine all result states into single batch
        combined = self._combine_states(move_results)

        # Evaluate all at once
        scores = self.evaluate_batch(combined, player_number)

        return scores.cpu().numpy()

    def _combine_states(self, states: List[GPUBoardState]) -> GPUBoardState:
        """Combine multiple GPUBoardState objects into a single batch."""
        return GPUBoardState(
            stack_owner=torch.cat([s.stack_owner for s in states]),
            stack_height=torch.cat([s.stack_height for s in states]),
            marker_owner=torch.cat([s.marker_owner for s in states]),
            territory_owner=torch.cat([s.territory_owner for s in states]),
            is_collapsed=torch.cat([s.is_collapsed for s in states]),
            rings_in_hand=torch.cat([s.rings_in_hand for s in states]),
            player_eliminated=torch.cat([s.player_eliminated for s in states]),
            territory_count=torch.cat([s.territory_count for s in states]),
            device=self.device,
            batch_size=sum(s.batch_size for s in states),
            board_size=states[0].board_size,
        )


# =============================================================================
# Async Batch Queue for MCTS
# =============================================================================


@dataclass
class EvalRequest:
    """Request for async batch evaluation."""

    features: np.ndarray
    global_features: Optional[np.ndarray]
    callback: Callable[[np.ndarray, np.ndarray], None]
    timestamp: float = field(default_factory=time.perf_counter)


class AsyncGPUEvaluator:
    """Asynchronous GPU evaluator with request batching for MCTS.

    Queues evaluation requests and processes them in batches to maximize
    GPU utilization. Useful for MCTS where leaf evaluations arrive
    incrementally during tree search.

    Example:
        async_eval = AsyncGPUEvaluator(evaluator, batch_size=64, timeout_ms=5)

        # Queue positions for evaluation
        async_eval.queue_position(features, globals, callback)

        # Process pending batches
        async_eval.flush()
    """

    def __init__(
        self,
        evaluator: GPUBatchEvaluator,
        batch_size: int = 64,
        timeout_ms: float = 10.0,
        max_queue_size: int = 1024,
    ):
        """Initialize async GPU evaluator.

        Args:
            evaluator: Underlying GPUBatchEvaluator
            batch_size: Target batch size before processing
            timeout_ms: Max time to wait before processing incomplete batch
            max_queue_size: Maximum pending requests before blocking
        """
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.timeout_sec = timeout_ms / 1000.0
        self.max_queue_size = max_queue_size

        self._queue: queue.Queue[EvalRequest] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._last_batch_time = time.perf_counter()

        # Stats
        self._batches_processed = 0
        self._requests_processed = 0

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("AsyncGPUEvaluator worker started")

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        logger.info("AsyncGPUEvaluator worker stopped")

    def queue_position(
        self,
        features: np.ndarray,
        global_features: Optional[np.ndarray],
        callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Queue a position for async evaluation.

        Args:
            features: Feature tensor for this position
            global_features: Global features (optional)
            callback: Function to call with (value, policy) results
        """
        request = EvalRequest(
            features=features,
            global_features=global_features,
            callback=callback,
        )
        self._queue.put(request)

    def flush(self) -> None:
        """Force processing of all pending requests."""
        self._process_batch(force=True)

    def _worker_loop(self) -> None:
        """Background worker loop that processes batches."""
        while self._running:
            self._process_batch(force=False)
            time.sleep(0.001)  # Small sleep to prevent busy-waiting

    def _process_batch(self, force: bool = False) -> None:
        """Process pending requests as a batch.

        Args:
            force: Process even if batch is incomplete
        """
        requests: List[EvalRequest] = []

        # Collect pending requests
        while len(requests) < self.batch_size:
            try:
                request = self._queue.get_nowait()
                requests.append(request)
            except queue.Empty:
                break

        if not requests:
            return

        # Check if we should process (batch full, timeout, or forced)
        time_since_last = time.perf_counter() - self._last_batch_time
        should_process = (
            force
            or len(requests) >= self.batch_size
            or time_since_last >= self.timeout_sec
        )

        if not should_process:
            # Put requests back in queue
            for req in requests:
                self._queue.put(req)
            return

        # Process batch
        try:
            # Stack features
            feature_batch = np.stack([r.features for r in requests])
            global_batch = None
            if requests[0].global_features is not None:
                global_batch = np.stack([r.global_features for r in requests])

            # Evaluate
            values, policies = self.evaluator.evaluate_batch(feature_batch, global_batch)

            # Call callbacks
            for i, request in enumerate(requests):
                try:
                    request.callback(values[i : i + 1], policies[i : i + 1])
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            self._batches_processed += 1
            self._requests_processed += len(requests)
            self._last_batch_time = time.perf_counter()

        except Exception as e:
            logger.error(f"Batch evaluation error: {e}")
            # Return error to callbacks
            for request in requests:
                try:
                    request.callback(np.array([0.0]), np.zeros(1))
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get async evaluator statistics."""
        return {
            "batches_processed": self._batches_processed,
            "requests_processed": self._requests_processed,
            "queue_size": self._queue.qsize(),
            "avg_batch_size": (
                self._requests_processed / self._batches_processed
                if self._batches_processed > 0
                else 0
            ),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def benchmark_gpu_batch(
    evaluator: GPUBatchEvaluator,
    batch_sizes: List[int] = [1, 8, 32, 64, 128, 256],
    feature_shape: Tuple[int, ...] = (16, 8, 8),
    num_iterations: int = 100,
) -> Dict[str, List[float]]:
    """Benchmark GPU batch evaluation at different batch sizes.

    Args:
        evaluator: GPUBatchEvaluator to benchmark
        batch_sizes: List of batch sizes to test
        feature_shape: Shape of feature tensor (channels, height, width)
        num_iterations: Number of iterations per batch size

    Returns:
        Dictionary with throughput results for each batch size
    """
    results = {"batch_size": [], "throughput": [], "latency_ms": []}

    for batch_size in batch_sizes:
        # Generate random features
        features = np.random.randn(batch_size, *feature_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            evaluator.evaluate_batch(features)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            evaluator.evaluate_batch(features)
        elapsed = time.perf_counter() - start

        total_samples = batch_size * num_iterations
        throughput = total_samples / elapsed
        latency_ms = (elapsed / num_iterations) * 1000

        results["batch_size"].append(batch_size)
        results["throughput"].append(throughput)
        results["latency_ms"].append(latency_ms)

        logger.info(
            f"Batch size {batch_size}: {throughput:.0f} samples/sec, "
            f"{latency_ms:.2f} ms/batch"
        )

    return results
