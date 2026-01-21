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

import contextlib
import gc
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Resource checking for GPU operations (December 2025)
from app.utils.resource_guard import check_gpu_memory, check_memory

logger = logging.getLogger(__name__)

# =============================================================================
# Device Detection and Management
# =============================================================================


def get_device(prefer_gpu: bool = True, device_id: int = 0) -> torch.device:
    """Auto-detect the best available compute device.

    .. deprecated:: 2025-12
        Use ``app.utils.torch_utils.get_device`` instead for the canonical
        implementation with full distributed training support.

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
    # Delegate to canonical implementation (no deprecation warning here since
    # this is a high-frequency call path - just document deprecation)
    from app.utils.torch_utils import get_device as _canonical_get_device
    return _canonical_get_device(prefer_gpu=prefer_gpu, device_id=device_id)


def get_all_cuda_devices() -> list[torch.device]:
    """Get list of all available CUDA devices."""
    if not torch.cuda.is_available():
        return []
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def clear_gpu_memory(device: torch.device | None = None) -> None:
    """Clear GPU memory caches.

    Args:
        device: Specific device to clear, or None for all devices
    """
    gc.collect()

    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and (device is None or device.type == "mps"):
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


# =============================================================================
# CUDA Health Monitoring and Recovery (December 2025)
# =============================================================================


@dataclass
class CudaHealthStatus:
    """CUDA device health status."""
    available: bool
    device_id: int = 0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_total_mb: float = 0.0
    last_check_time: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ""


# Global health status per device
_cuda_health: dict[int, CudaHealthStatus] = {}
_cuda_health_lock = threading.Lock()


def check_cuda_health(device_id: int = 0, timeout_seconds: float = 5.0) -> CudaHealthStatus:
    """Check CUDA device health with a small test operation.

    Performs a quick tensor operation to verify device is responsive.
    Tracks consecutive failures to detect persistent issues.

    Args:
        device_id: CUDA device ID to check
        timeout_seconds: Max time for health check operation

    Returns:
        CudaHealthStatus with current device state
    """
    with _cuda_health_lock:
        if device_id not in _cuda_health:
            _cuda_health[device_id] = CudaHealthStatus(available=False, device_id=device_id)
        status = _cuda_health[device_id]

    status.last_check_time = time.time()

    if not torch.cuda.is_available():
        status.available = False
        status.last_error = "CUDA not available"
        return status

    try:
        device = torch.device(f"cuda:{device_id}")

        # Quick test operation (create small tensor, do computation)
        test_tensor = torch.zeros(10, 10, device=device)
        _ = test_tensor.sum()
        torch.cuda.synchronize(device)

        # Get memory stats
        status.memory_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        status.memory_reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        props = torch.cuda.get_device_properties(device)
        status.memory_total_mb = props.total_memory / (1024 * 1024)

        status.available = True
        status.consecutive_failures = 0
        status.last_error = ""

    except RuntimeError as e:
        status.available = False
        status.consecutive_failures += 1
        status.last_error = str(e)
        logger.warning(f"CUDA health check failed for device {device_id}: {e}")

    except Exception as e:
        status.available = False
        status.consecutive_failures += 1
        status.last_error = str(e)
        logger.warning(f"CUDA health check error for device {device_id}: {e}")

    with _cuda_health_lock:
        _cuda_health[device_id] = status

    return status


def recover_cuda_device(device_id: int = 0, max_attempts: int = 3) -> bool:
    """Attempt to recover a CUDA device after errors.

    Recovery steps:
    1. Clear CUDA memory cache
    2. Synchronize device
    3. Reset peak memory stats
    4. Retry health check

    Args:
        device_id: CUDA device ID to recover
        max_attempts: Maximum recovery attempts

    Returns:
        True if device recovered successfully
    """
    if not torch.cuda.is_available():
        return False

    device = torch.device(f"cuda:{device_id}")

    for attempt in range(max_attempts):
        logger.info(f"CUDA recovery attempt {attempt + 1}/{max_attempts} for device {device_id}")

        try:
            # Step 1: Garbage collect Python objects
            gc.collect()

            # Step 2: Clear CUDA cache
            torch.cuda.empty_cache()

            # Step 3: Synchronize to clear pending operations
            try:
                torch.cuda.synchronize(device)
            except RuntimeError:
                # Device may be in bad state, continue with recovery
                pass

            # Step 4: Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device)

            # Step 5: Short delay for device to stabilize
            time.sleep(0.5 * (attempt + 1))

            # Step 6: Verify recovery with health check
            status = check_cuda_health(device_id)
            if status.available:
                logger.info(f"CUDA device {device_id} recovered successfully")
                return True

        except Exception as e:
            logger.warning(f"CUDA recovery attempt {attempt + 1} failed: {e}")

    logger.error(f"CUDA device {device_id} recovery failed after {max_attempts} attempts")
    return False


def get_device_with_recovery(
    prefer_gpu: bool = True,
    device_id: int = 0,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """Get device with automatic recovery on CUDA failure.

    If CUDA device is unhealthy, attempts recovery before falling back to CPU.

    Args:
        prefer_gpu: Whether to prefer GPU
        device_id: CUDA device ID to use
        fallback_to_cpu: Whether to fallback to CPU on CUDA failure

    Returns:
        torch.device (CUDA, MPS, or CPU)
    """
    if not prefer_gpu:
        return torch.device("cpu")

    # Check CUDA first
    if torch.cuda.is_available():
        status = check_cuda_health(device_id)

        if status.available:
            return torch.device(f"cuda:{device_id}")

        # Device unhealthy - attempt recovery
        if status.consecutive_failures < 5:  # Don't keep trying if persistently failing
            if recover_cuda_device(device_id):
                return torch.device(f"cuda:{device_id}")

        logger.warning(
            f"CUDA device {device_id} unavailable after recovery "
            f"(failures={status.consecutive_failures}). Falling back to CPU."
        )

    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if fallback_to_cpu:
        return torch.device("cpu")

    raise RuntimeError(f"No available compute device (CUDA device {device_id} unhealthy)")


def log_cuda_memory_state(device_id: int = 0, prefix: str = "") -> None:
    """Log current CUDA memory state for debugging.

    Useful for diagnosing memory leaks and fragmentation.

    Args:
        device_id: CUDA device ID
        prefix: Optional prefix for log message
    """
    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(f"cuda:{device_id}")
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        logger.info(
            f"{prefix}CUDA memory (device {device_id}): "
            f"allocated={allocated:.1f}MB, reserved={reserved:.1f}MB, "
            f"peak={max_allocated:.1f}MB, fragmentation={reserved - allocated:.1f}MB"
        )
    except Exception as e:
        logger.debug(f"Failed to log CUDA memory state: {e}")


# =============================================================================
# Model Compilation (torch.compile optimization)
# =============================================================================


def _configure_dynamo_for_dynamic_shapes() -> None:
    """Configure torch._dynamo for better handling of dynamic batch sizes.

    This prevents excessive recompilation warnings when batch sizes vary
    during inference (e.g., single state eval vs batched MCTS evaluation).
    """
    try:
        import torch._dynamo as dynamo

        # Increase cache size limit to handle more shape variations
        # Default is 8, which is too low for MCTS with varying batch sizes
        dynamo.config.cache_size_limit = 64

        # Enable automatic dynamic shapes detection
        # This helps dynamo understand which dimensions are truly dynamic
        dynamo.config.automatic_dynamic_shapes = True

        # Suppress recompilation warnings after limit is hit
        # The model still works (falls back to eager mode), just with a warning
        dynamo.config.suppress_errors = True

    except (ImportError, AttributeError):
        # torch._dynamo not available or config attributes don't exist
        pass


# Configure dynamo once at module load
_configure_dynamo_for_dynamic_shapes()


def compile_model(
    model: nn.Module,
    device: torch.device | None = None,
    mode: str = "default",
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
            - "default": Good balance of compile time and speedup (recommended)
            - "reduce-overhead": Uses CUDA graphs (can fail with dynamic shapes)
            - "max-autotune": Maximum optimization (longer compile time)
        fullgraph: If True, require entire model in single graph (stricter)

    Returns:
        Compiled model (or original if compilation fails/unsupported)

    Note:
        - Compilation happens on first forward pass (lazy)
        - MPS has limited torch.compile() support (will skip gracefully)
        - Requires PyTorch 2.0+
        - Set RINGRIFT_DISABLE_TORCH_COMPILE=1 to skip compilation
        - dynamo cache_size_limit increased to 64 to handle varying batch sizes
    """
    # Check for env var to disable compilation
    import os
    if os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes"):
        logger.debug("Skipping torch.compile() (RINGRIFT_DISABLE_TORCH_COMPILE set)")
        return model

    # Check PyTorch version
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 0):
        logger.debug(f"torch.compile() requires PyTorch 2.0+, got {torch.__version__}")
        return model

    # torch.compile is most valuable on CUDA inference workloads. On CPU it can
    # be brittle (especially on macOS toolchains) and the speedups are often
    # marginal for our batch sizes, so we skip compilation unless the target is
    # a CUDA device.
    if device is None:
        return model

    if device.type == "mps":
        logger.debug("Skipping torch.compile() on MPS (limited support)")
        return model

    if device.type == "cpu":
        logger.debug("Skipping torch.compile() on CPU")
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
        board_states: list[dict[str, np.ndarray]],
        device: torch.device,
        board_size: int = 8,
    ) -> GPUBoardState:
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
        game_states: list[Any],  # List[GameState] - use Any to avoid circular import
        device: torch.device,
    ) -> GPUBoardState:
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

    # Default timeout threshold for warning (10 seconds per batch is very slow)
    DEFAULT_WARN_TIMEOUT_SECONDS = float(os.environ.get("RINGRIFT_NN_WARN_TIMEOUT", "10.0"))

    def __init__(
        self,
        device: str | torch.device | None = None,
        model: nn.Module | None = None,
        use_mixed_precision: bool = True,
        max_batch_size: int = 256,
        warn_timeout_seconds: float | None = None,
    ):
        """Initialize GPU batch evaluator.

        Args:
            device: Compute device (auto-detected if None)
            model: Neural network model for evaluation
            use_mixed_precision: Use fp16 for faster inference on supported GPUs
            max_batch_size: Maximum batch size for memory management
            warn_timeout_seconds: Log warning if batch takes longer than this (default: 10s)
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
        self.warn_timeout_seconds = warn_timeout_seconds or self.DEFAULT_WARN_TIMEOUT_SECONDS

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._slow_eval_count = 0  # Count of evaluations exceeding timeout

        logger.info(
            f"GPUBatchEvaluator initialized on {self.device} "
            f"(mixed_precision={self.use_mixed_precision}, warn_timeout={self.warn_timeout_seconds}s)"
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
        global_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        # Resource guard: check GPU memory before large tensor allocation
        if self.device.type == "cuda" and batch_size > 32:
            if not check_gpu_memory():
                logger.warning("[GPUBatchEvaluator] GPU memory pressure, clearing cache")
                clear_gpu_memory(self.device)

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
            with torch.amp.autocast('cuda'):
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

        # Timeout warning for slow batches (Dec 2025 monitoring)
        if elapsed > self.warn_timeout_seconds:
            self._slow_eval_count += 1
            logger.warning(
                f"[GPUBatchEvaluator] Slow NN evaluation: {elapsed:.2f}s for batch_size={batch_size} "
                f"(threshold={self.warn_timeout_seconds}s, slow_count={self._slow_eval_count}). "
                f"Consider reducing batch size or checking GPU health."
            )

        return values_np, policies_np

    def _forward(
        self,
        features: torch.Tensor,
        globals_t: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        globals_t: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
                with torch.amp.autocast('cuda'):
                    values, policies = self._forward(chunk_features, chunk_globals)
            else:
                values, policies = self._forward(chunk_features, chunk_globals)

            all_values.append(values.cpu().numpy())
            all_policies.append(policies.cpu().numpy())

        return np.concatenate(all_values), np.concatenate(all_policies)

    def get_performance_stats(self) -> dict[str, float]:
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
        device: str | torch.device | None = None,
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

    def _default_weights(self) -> dict[str, float]:
        """Default heuristic weights."""
        return {
            "stack_count": 1.0,
            "ring_count": 0.5,
            "territory_count": 2.0,
            "center_control": 0.3,
            "mobility": 0.1,
            "no_stacks_penalty": -100.0,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
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

        Uses symmetric (zero-sum) evaluation: each feature is computed as
        (my_value - max_opponent_value) so that the sum across all players
        is approximately zero. This prevents P1/P2 bias in self-play.

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

        # Compute max opponent values for symmetric evaluation
        max_opp_territory = torch.zeros(batch_size, device=board_states.stack_owner.device)
        max_opp_rings = torch.zeros(batch_size, device=board_states.stack_owner.device)
        max_opp_center = torch.zeros(batch_size, device=board_states.stack_owner.device)

        for opp in range(1, self.num_players + 1):
            if opp == player_number:
                continue
            opp_territory = board_states.territory_count[:, opp].float()
            opp_rings = board_states.rings_in_hand[:, opp].float()
            opp_center = ((stack_2d == opp).float() * self.center_mask).sum(dim=(1, 2))

            max_opp_territory = torch.maximum(max_opp_territory, opp_territory)
            max_opp_rings = torch.maximum(max_opp_rings, opp_rings)
            max_opp_center = torch.maximum(max_opp_center, opp_center)

        # Compute final score using symmetric (relative) features
        w = self.weights
        scores = (
            (my_stacks - opp_stacks) * w["stack_count"]
            + (my_rings - max_opp_rings) * w["ring_count"]
            + (my_territory - max_opp_territory) * w["territory_count"]
            + (center_control - max_opp_center) * w["center_control"]
        )

        # No stacks penalty (apply to both player and check opponent situation)
        my_no_stacks = my_stacks == 0
        opp_no_stacks = opp_stacks == 0
        scores = torch.where(
            my_no_stacks & ~opp_no_stacks,  # Only penalize if I have none but opponent does
            scores + w["no_stacks_penalty"],
            scores,
        )
        # Bonus if opponent has no stacks but I do
        scores = torch.where(
            ~my_no_stacks & opp_no_stacks,
            scores - w["no_stacks_penalty"],  # Subtract penalty = add bonus
            scores,
        )

        return scores

    def evaluate_moves_batch(
        self,
        base_state: GPUBoardState,
        move_results: list[GPUBoardState],
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

    def _combine_states(self, states: list[GPUBoardState]) -> GPUBoardState:
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
    global_features: np.ndarray | None
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
        self._worker_thread: threading.Thread | None = None
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
        global_features: np.ndarray | None,
        callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Queue a position for async evaluation.

        Args:
            features: Feature tensor for this position
            global_features: Global features (optional)
            callback: Function to call with (value, policy) results

        Note:
            Uses non-blocking put to avoid deadlock if worker thread isn't running.
            If queue is full and worker isn't running, the request is processed
            synchronously as a fallback.
        """
        if not self._running:
            logger.warning("AsyncGPUEvaluator.queue_position called without start() - auto-starting worker")
            self.start()

        request = EvalRequest(
            features=features,
            global_features=global_features,
            callback=callback,
        )
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            # Queue is full - process synchronously as fallback to avoid deadlock
            logger.warning("AsyncGPUEvaluator queue full, processing request synchronously")
            try:
                values, policies = self.evaluator.evaluate_batch(
                    features[np.newaxis],
                    global_features[np.newaxis] if global_features is not None else None
                )
                callback(values[0], policies[0])
            except Exception as e:
                logger.error(f"Synchronous fallback evaluation failed: {e}")
                # Call callback with neutral values as last resort
                callback(np.array([0.0]), np.zeros(1))

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
        requests: list[EvalRequest] = []

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
            # Put requests back in queue using non-blocking put to avoid deadlock
            for req in requests:
                try:
                    self._queue.put_nowait(req)
                except queue.Full:
                    # Queue is full - process this request synchronously
                    logger.warning("Queue full during re-insert, processing synchronously")
                    try:
                        values, policies = self.evaluator.evaluate_batch(
                            req.features[np.newaxis],
                            req.global_features[np.newaxis] if req.global_features is not None else None
                        )
                        req.callback(values[0], policies[0])
                    except Exception as e:
                        logger.error(f"Synchronous fallback failed: {e}")
                        req.callback(np.array([0.0]), np.zeros(1))
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
                with contextlib.suppress(Exception):
                    request.callback(np.array([0.0]), np.zeros(1))

    def get_stats(self) -> dict[str, Any]:
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
    batch_sizes: list[int] | None = None,
    feature_shape: tuple[int, ...] = (16, 8, 8),
    num_iterations: int = 100,
) -> dict[str, list[float]]:
    """Benchmark GPU batch evaluation at different batch sizes.

    Args:
        evaluator: GPUBatchEvaluator to benchmark
        batch_sizes: List of batch sizes to test
        feature_shape: Shape of feature tensor (channels, height, width)
        num_iterations: Number of iterations per batch size

    Returns:
        Dictionary with throughput results for each batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]
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
