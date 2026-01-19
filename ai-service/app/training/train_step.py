"""Core training step for RingRift Neural Network AI.

December 2025: Extracted from train.py to improve modularity.

This module provides the core training step logic for a single batch,
including forward pass, loss computation, backward pass, and optimizer step.

Usage:
    from app.training.train_step import TrainStepContext, run_training_step

    context = TrainStepContext(
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
    )
    result = run_training_step(context, batch_data, batch_idx=0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy imports for optional dependencies
# =============================================================================


def _try_import_masked_policy():
    """Lazy import masked policy functions."""
    try:
        from app.ai.neural_losses import (
            detect_masked_policy_output,
            masked_log_softmax,
            masked_policy_kl,
        )
        return detect_masked_policy_output, masked_log_softmax, masked_policy_kl
    except ImportError:
        return None, None, None


def _try_import_multi_player_loss():
    """Lazy import multi-player value loss."""
    try:
        from app.ai.neural_net import multi_player_value_loss
        return multi_player_value_loss
    except ImportError:
        return None


def _try_import_rank_targets():
    """Lazy import rank target builder."""
    try:
        from app.ai.neural_losses import build_rank_targets
        return build_rank_targets
    except ImportError:
        return None


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrainStepConfig:
    """Configuration for a training step.

    Groups related parameters to reduce function signature complexity.
    """

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0  # Jan 2026: Balance value vs policy learning
    rank_dist_weight: float = 0.1
    entropy_weight: float = 0.0

    # Policy processing
    policy_label_smoothing: float = 0.0

    # Multi-player training
    use_multi_player_loss: bool = False
    num_players: int = 2

    # Outcome-weighted policy
    enable_outcome_weighted_policy: bool = False
    outcome_weight_scale: float = 0.5

    # Gradient management
    gradient_accumulation_steps: int = 1
    gradient_clip_max_norm: float = 1.0
    use_adaptive_clipping: bool = False

    # Mixed precision
    use_mixed_precision: bool = False
    amp_dtype: torch.dtype = torch.float16

    # Model version (for heuristics support)
    model_version: str = "v2"

    # Debugging
    validate_policy_targets: bool = True
    log_interval: int = 500


@dataclass
class TrainStepContext:
    """Context for a training step.

    Holds all the state needed to execute training steps.
    """

    model: nn.Module
    optimizer: "Optimizer"
    device: torch.device
    config: TrainStepConfig

    # Optional components
    value_criterion: nn.Module | None = None
    grad_scaler: "GradScaler | None" = None
    adaptive_clipper: Any | None = None
    gradient_surgeon: Any | None = None
    training_breaker: Any | None = None
    enhancements_manager: Any | None = None
    training_facade: Any | None = None
    hard_example_miner: Any | None = None
    quality_trainer: Any | None = None
    hot_buffer: Any | None = None

    # State tracking
    accumulation_counter: int = 0

    def __post_init__(self):
        """Set up default value criterion if not provided."""
        if self.value_criterion is None:
            self.value_criterion = nn.MSELoss()


@dataclass
class TrainStepResult:
    """Result of a training step."""

    loss: float
    value_loss: float
    policy_loss: float
    rank_loss: float | None = None
    aux_loss: float | None = None
    grad_norm: float | None = None
    skipped: bool = False
    error: str | None = None


@dataclass
class BatchData:
    """Parsed batch data for training.

    Provides a structured way to access batch components regardless of
    the raw batch format (streaming, 4-tuple, 5-tuple, 6-tuple).
    """

    features: torch.Tensor
    globals_vec: torch.Tensor
    value_targets: torch.Tensor
    policy_targets: torch.Tensor
    num_players: torch.Tensor | None = None
    heuristics: torch.Tensor | None = None


# =============================================================================
# Batch Parsing
# =============================================================================


def parse_batch(
    raw_batch: tuple,
    is_streaming: bool = False,
    has_mp_values: bool = False,
) -> BatchData:
    """Parse raw batch data into structured BatchData.

    Handles different batch formats:
    - Streaming: ((features, globals), (value, policy)) or with mp values
    - 4-tuple: (features, globals, value, policy)
    - 5-tuple: (features, globals, value, policy, num_players) or heuristics
    - 6-tuple: (features, globals, value, policy, num_players, heuristics)

    Args:
        raw_batch: Raw batch from dataloader
        is_streaming: Whether batch is from streaming loader
        has_mp_values: Whether streaming loader has multi-player values

    Returns:
        Structured BatchData object
    """
    if is_streaming:
        if has_mp_values:
            # Streaming with multi-player values
            ((features, globals_vec), (value_targets, policy_targets),
             values_mp, num_players) = raw_batch
            if values_mp is not None:
                value_targets = values_mp
            return BatchData(
                features=features,
                globals_vec=globals_vec,
                value_targets=value_targets,
                policy_targets=policy_targets,
                num_players=num_players,
            )
        else:
            # Standard streaming
            ((features, globals_vec), (value_targets, policy_targets)) = raw_batch
            return BatchData(
                features=features,
                globals_vec=globals_vec,
                value_targets=value_targets,
                policy_targets=policy_targets,
            )

    # Non-streaming: 4, 5, or 6 tuple
    batch_len = len(raw_batch) if isinstance(raw_batch, (list, tuple)) else 0

    if batch_len == 6:
        # Full: with num_players and heuristics
        (features, globals_vec, value_targets, policy_targets,
         num_players, heuristics) = raw_batch
        return BatchData(
            features=features,
            globals_vec=globals_vec,
            value_targets=value_targets,
            policy_targets=policy_targets,
            num_players=num_players,
            heuristics=heuristics,
        )

    elif batch_len == 5:
        # Check if 5th element is num_players (int tensor) or heuristics (float)
        fifth_elem = raw_batch[4]
        if fifth_elem.dtype in (torch.int64, torch.int32, torch.long):
            return BatchData(
                features=raw_batch[0],
                globals_vec=raw_batch[1],
                value_targets=raw_batch[2],
                policy_targets=raw_batch[3],
                num_players=fifth_elem,
            )
        else:
            return BatchData(
                features=raw_batch[0],
                globals_vec=raw_batch[1],
                value_targets=raw_batch[2],
                policy_targets=raw_batch[3],
                heuristics=fifth_elem,
            )

    else:
        # 4-tuple
        (features, globals_vec, value_targets, policy_targets) = raw_batch[:4]
        return BatchData(
            features=features,
            globals_vec=globals_vec,
            value_targets=value_targets,
            policy_targets=policy_targets,
        )


def transfer_batch_to_device(batch: BatchData, device: torch.device) -> BatchData:
    """Transfer batch tensors to target device.

    Args:
        batch: Batch data to transfer
        device: Target device

    Returns:
        BatchData with tensors on target device
    """
    def maybe_transfer(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        if t.device != device:
            return t.to(device, non_blocking=True)
        return t

    return BatchData(
        features=maybe_transfer(batch.features),
        globals_vec=maybe_transfer(batch.globals_vec),
        value_targets=maybe_transfer(batch.value_targets),
        policy_targets=maybe_transfer(batch.policy_targets),
        num_players=maybe_transfer(batch.num_players),
        heuristics=maybe_transfer(batch.heuristics),
    )


# =============================================================================
# Policy Processing
# =============================================================================


def pad_policy_targets(
    policy_targets: torch.Tensor,
    model_policy_size: int,
) -> torch.Tensor:
    """Pad policy targets to match model policy size.

    Args:
        policy_targets: Policy targets tensor (batch, policy_size)
        model_policy_size: Target policy size from model

    Returns:
        Padded policy targets
    """
    if policy_targets.size(1) < model_policy_size:
        pad_size = model_policy_size - policy_targets.size(1)
        return torch.nn.functional.pad(policy_targets, (0, pad_size), value=0.0)
    return policy_targets


def apply_label_smoothing(
    policy_targets: torch.Tensor,
    epsilon: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply label smoothing to policy targets.

    Args:
        policy_targets: Policy targets tensor
        epsilon: Smoothing factor (0-1)
        valid_mask: Mask for valid (non-zero) targets

    Returns:
        Smoothed policy targets
    """
    if epsilon <= 0 or not torch.any(valid_mask):
        return policy_targets

    policy_size = policy_targets.size(1)
    uniform = 1.0 / policy_size

    smoothed = policy_targets.clone()
    smoothed[valid_mask] = (
        (1 - epsilon) * policy_targets[valid_mask] + epsilon * uniform
    )
    return smoothed


def validate_policy_targets(
    policy_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_idx: int,
) -> None:
    """Validate policy target normalization.

    Args:
        policy_targets: Policy targets tensor
        valid_mask: Mask for valid targets
        batch_idx: Batch index for error messages

    Raises:
        ValueError: If targets are severely denormalized
    """
    if not torch.any(valid_mask):
        return

    target_sums = policy_targets[valid_mask].sum(dim=1)
    ones = torch.ones_like(target_sums)

    if not torch.allclose(target_sums, ones, atol=1e-4):
        bad_sums = target_sums[~torch.isclose(target_sums, ones, atol=1e-4)]
        logger.error(
            f"Policy targets not normalized at batch {batch_idx}! "
            f"Expected sum=1.0, got: min={target_sums.min():.6f}, "
            f"max={target_sums.max():.6f}, num_bad={len(bad_sums)}/{len(target_sums)}"
        )
        if target_sums.min() < 0.5 or target_sums.max() > 1.5:
            raise ValueError(
                f"Policy targets severely denormalized at batch {batch_idx}. "
                f"Check data export pipeline."
            )


# =============================================================================
# Loss Computation
# =============================================================================


@dataclass
class LossComponents:
    """Individual loss components for training."""

    value_loss: torch.Tensor
    policy_loss: torch.Tensor
    rank_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    entropy_bonus: torch.Tensor | None = None
    quality_ranking_loss: torch.Tensor | None = None

    @property
    def total(self) -> torch.Tensor:
        """Compute total loss from components."""
        total = self.value_loss + self.policy_loss
        if self.rank_loss is not None:
            total = total + self.rank_loss
        if self.aux_loss is not None:
            total = total + self.aux_loss
        if self.entropy_bonus is not None:
            total = total + self.entropy_bonus
        if self.quality_ranking_loss is not None:
            total = total + self.quality_ranking_loss
        return total

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return losses as dictionary for gradient surgery."""
        result = {
            "value": self.value_loss,
            "policy": self.policy_loss,
        }
        if self.rank_loss is not None:
            result["rank"] = self.rank_loss
        if self.aux_loss is not None:
            result["aux"] = self.aux_loss
        return result


def compute_value_loss(
    value_pred: torch.Tensor,
    value_targets: torch.Tensor,
    num_players: torch.Tensor | int | None,
    use_multi_player: bool,
    value_criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute value prediction loss.

    Args:
        value_pred: Model value predictions
        value_targets: Target values
        num_players: Number of players (per-sample tensor or scalar)
        use_multi_player: Whether to use multi-player loss
        value_criterion: Loss criterion for scalar mode
        device: Target device

    Returns:
        Value loss tensor
    """
    if use_multi_player:
        multi_player_value_loss = _try_import_multi_player_loss()
        if multi_player_value_loss is None:
            logger.warning("Multi-player value loss not available, using MSE")
            use_multi_player = False

    if use_multi_player:
        effective_num_players = num_players if num_players is not None else 2
        return multi_player_value_loss(value_pred, value_targets, effective_num_players)
    else:
        # Scalar training uses only the first value head
        if value_pred.ndim == 2:
            value_pred_scalar = value_pred[:, 0]
        else:
            value_pred_scalar = value_pred
        return value_criterion(
            value_pred_scalar.reshape(-1),
            value_targets.reshape(-1),
        )


def compute_policy_loss(
    policy_pred: torch.Tensor,
    policy_targets: torch.Tensor,
    config: TrainStepConfig,
    value_targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute policy prediction loss.

    Args:
        policy_pred: Model policy logits
        policy_targets: Target policy distribution
        config: Training step configuration
        value_targets: Value targets (for outcome weighting)
        device: Target device

    Returns:
        Tuple of (policy_loss, policy_log_probs)
    """
    detect_masked, masked_log_softmax, masked_policy_kl = _try_import_masked_policy()

    # Apply log_softmax to policy prediction for KLDivLoss
    if detect_masked is not None and detect_masked(policy_pred):
        # Valid positions are either: (1) target > 0, or (2) logits > -1e3
        valid_mask = (policy_targets > 0) | (policy_pred > -1e3)
        policy_log_probs = masked_log_softmax(policy_pred, valid_mask)
    else:
        policy_log_probs = torch.log_softmax(policy_pred, dim=1)

    # Compute base policy loss
    if masked_policy_kl is not None:
        policy_loss = masked_policy_kl(policy_log_probs, policy_targets)
    else:
        # Fallback: manual KL computation
        valid_mask = policy_targets.sum(dim=1) > 0
        if valid_mask.any():
            policy_loss = -(policy_targets[valid_mask] * policy_log_probs[valid_mask]).sum(dim=1).mean()
        else:
            policy_loss = torch.tensor(0.0, device=device)

    # Apply outcome weighting if enabled
    if config.enable_outcome_weighted_policy and config.outcome_weight_scale > 0:
        with torch.no_grad():
            if value_targets.ndim == 2:
                outcome_signal = value_targets.mean(dim=1)
            else:
                outcome_signal = value_targets.reshape(-1)

            outcome_weights = 1.0 + config.outcome_weight_scale * outcome_signal.sign()
            outcome_weights = outcome_weights.clamp(min=0.1)

        per_sample_policy = -(policy_targets * policy_log_probs).sum(dim=1)
        valid_mask = policy_targets.sum(dim=1) > 0
        if valid_mask.any():
            policy_loss = (per_sample_policy[valid_mask] * outcome_weights[valid_mask]).mean()

    return config.policy_weight * policy_loss, policy_log_probs


def compute_entropy_bonus(
    policy_log_probs: torch.Tensor,
    entropy_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute entropy regularization bonus.

    Args:
        policy_log_probs: Log probabilities from policy head
        entropy_weight: Weight for entropy bonus
        device: Target device

    Returns:
        Entropy bonus (negative, to maximize entropy)
    """
    if entropy_weight <= 0:
        return torch.tensor(0.0, device=device)

    policy_probs = policy_log_probs.exp()
    policy_entropy = -(policy_probs * policy_log_probs.clamp(min=-20)).sum(dim=1).mean()
    return -entropy_weight * policy_entropy


# =============================================================================
# Backward and Optimization
# =============================================================================


def run_backward(
    loss: torch.Tensor,
    task_losses: dict[str, torch.Tensor] | None,
    context: TrainStepContext,
    batch_idx: int,
) -> bool:
    """Run backward pass with error handling.

    Args:
        loss: Total loss tensor
        task_losses: Individual task losses for gradient surgery
        context: Training step context
        batch_idx: Batch index for logging

    Returns:
        True if backward succeeded, False if skipped due to error
    """
    try:
        if context.gradient_surgeon is not None and task_losses is not None:
            context.gradient_surgeon.apply_surgery(context.model, task_losses)
        elif context.grad_scaler is not None:
            context.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        return True

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_msg = str(e).lower()
        is_cuda_error = any(x in error_msg for x in
                          ['cuda', 'out of memory', 'cublas', 'cudnn'])

        if is_cuda_error:
            logger.warning(f"CUDA error in batch {batch_idx}: {e}")
            if context.training_breaker:
                context.training_breaker.record_failure("batch_processing", e)

            # Clear gradients and memory
            context.optimizer.zero_grad(set_to_none=True)
            if context.device.type == 'cuda':
                torch.cuda.empty_cache()
            return False
        else:
            raise


def step_optimizer(
    context: TrainStepContext,
    batch_idx: int,
    is_last_batch: bool,
) -> float | None:
    """Step optimizer with gradient clipping.

    Args:
        context: Training step context
        batch_idx: Current batch index
        is_last_batch: Whether this is the last batch

    Returns:
        Gradient norm if stepping occurred, None otherwise
    """
    config = context.config
    accum_steps = config.gradient_accumulation_steps

    # Only step after accumulating gradients
    context.accumulation_counter += 1
    if context.accumulation_counter < accum_steps and not is_last_batch:
        return None

    context.accumulation_counter = 0

    # Unscale gradients if using AMP
    if context.grad_scaler is not None:
        context.grad_scaler.unscale_(context.optimizer)

    # Gradient clipping
    if context.adaptive_clipper is not None:
        grad_norm = context.adaptive_clipper.update_and_clip(context.model.parameters())
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            context.model.parameters(),
            max_norm=config.gradient_clip_max_norm,
        )

    # Step optimizer
    try:
        if context.grad_scaler is not None:
            context.grad_scaler.step(context.optimizer)
            context.grad_scaler.update()
        else:
            context.optimizer.step()

        # Record success for circuit breaker
        if context.training_breaker:
            context.training_breaker.record_success("batch_processing")

        return float(grad_norm) if grad_norm is not None else None

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_msg = str(e).lower()
        is_cuda_error = any(x in error_msg for x in
                          ['cuda', 'out of memory', 'cublas', 'cudnn'])

        if is_cuda_error:
            logger.warning(f"CUDA error in optimizer step at batch {batch_idx}: {e}")
            if context.training_breaker:
                context.training_breaker.record_failure("batch_processing", e)
            return None
        else:
            raise


# =============================================================================
# Main Training Step
# =============================================================================


def run_training_step(
    context: TrainStepContext,
    raw_batch: tuple,
    batch_idx: int,
    is_streaming: bool = False,
    has_mp_values: bool = False,
    is_last_batch: bool = False,
) -> TrainStepResult:
    """Execute a single training step on a batch.

    This is the main entry point for batch-level training logic.

    Args:
        context: Training step context with model, optimizer, etc.
        raw_batch: Raw batch data from dataloader
        batch_idx: Index of current batch
        is_streaming: Whether batch is from streaming loader
        has_mp_values: Whether streaming loader has multi-player values
        is_last_batch: Whether this is the last batch in the epoch

    Returns:
        TrainStepResult with loss values and status
    """
    config = context.config
    device = context.device
    model = context.model

    # Check circuit breaker
    if context.training_breaker and not context.training_breaker.can_execute("batch_processing"):
        if batch_idx % 100 == 0:
            logger.debug(f"Batch {batch_idx} skipped: circuit breaker open")
        return TrainStepResult(
            loss=0.0, value_loss=0.0, policy_loss=0.0, skipped=True
        )

    # Parse and transfer batch
    batch = parse_batch(raw_batch, is_streaming, has_mp_values)
    batch = transfer_batch_to_device(batch, device)

    # Pad policy targets if needed
    model_policy_size = getattr(model, 'policy_size', None)
    if model_policy_size and batch.policy_targets.size(1) < model_policy_size:
        batch = BatchData(
            features=batch.features,
            globals_vec=batch.globals_vec,
            value_targets=batch.value_targets,
            policy_targets=pad_policy_targets(batch.policy_targets, model_policy_size),
            num_players=batch.num_players,
            heuristics=batch.heuristics,
        )

    # Compute valid mask and validate
    policy_valid_mask = batch.policy_targets.sum(dim=1) > 0
    if config.validate_policy_targets:
        validate_policy_targets(batch.policy_targets, policy_valid_mask, batch_idx)

    # Apply label smoothing
    policy_targets = apply_label_smoothing(
        batch.policy_targets, config.policy_label_smoothing, policy_valid_mask
    )

    # Zero gradients at start of accumulation window
    if batch_idx % config.gradient_accumulation_steps == 0:
        context.optimizer.zero_grad()

    # Forward pass with optional mixed precision
    amp_enabled = config.use_mixed_precision and device.type == 'cuda'
    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=config.amp_dtype):
        # Check if model accepts heuristics
        model_accepts_heuristics = config.model_version in ('v5', 'v5-gnn', 'v5-heavy')

        # Forward pass
        if model_accepts_heuristics and batch.heuristics is not None:
            out = model(batch.features, batch.globals_vec, heuristics=batch.heuristics)
        else:
            out = model(batch.features, batch.globals_vec)

        # Parse model output
        if isinstance(out, tuple) and len(out) == 3:
            value_pred, policy_pred, rank_dist_pred = out
        else:
            value_pred, policy_pred = out[:2]
            rank_dist_pred = None

        # Check for NaN/Inf
        if torch.any(torch.isnan(policy_pred)) or torch.any(torch.isinf(policy_pred)):
            return TrainStepResult(
                loss=0.0, value_loss=0.0, policy_loss=0.0,
                skipped=True, error="NaN/Inf in policy predictions"
            )

        # Compute losses
        value_loss = compute_value_loss(
            value_pred, batch.value_targets, batch.num_players,
            config.use_multi_player_loss, context.value_criterion, device
        )

        policy_loss, policy_log_probs = compute_policy_loss(
            policy_pred, policy_targets, config, batch.value_targets, device
        )

        entropy_bonus = compute_entropy_bonus(
            policy_log_probs, config.entropy_weight, device
        )

        # Build loss components
        # Jan 2026: Apply value_weight to balance value vs policy learning
        losses = LossComponents(
            value_loss=config.value_weight * value_loss,
            policy_loss=policy_loss + entropy_bonus,
        )

        # Rank distribution loss (V3+ multi-player)
        if rank_dist_pred is not None and config.use_multi_player_loss:
            build_rank_targets = _try_import_rank_targets()
            if build_rank_targets and batch.value_targets.ndim == 2:
                effective_np = batch.num_players if batch.num_players is not None else config.num_players
                rank_targets, rank_mask = build_rank_targets(batch.value_targets, effective_np)
                rank_log_probs = torch.log(rank_dist_pred.clamp_min(1e-8))
                per_player_loss = -(rank_targets * rank_log_probs).sum(dim=-1)
                if torch.any(rank_mask):
                    losses.rank_loss = config.rank_dist_weight * per_player_loss[rank_mask].mean()

        # Total loss
        loss = losses.total

        # Scale for gradient accumulation
        accum_steps = config.gradient_accumulation_steps
        if accum_steps > 1:
            loss = loss / accum_steps

    # Backward pass
    task_losses = losses.as_dict() if context.gradient_surgeon else None
    if not run_backward(loss, task_losses, context, batch_idx):
        return TrainStepResult(
            loss=0.0, value_loss=0.0, policy_loss=0.0,
            skipped=True, error="Backward pass failed"
        )

    # Step optimizer
    grad_norm = step_optimizer(context, batch_idx, is_last_batch)

    return TrainStepResult(
        loss=float(loss.item()) * accum_steps,  # Un-scale for reporting
        value_loss=float(value_loss.item()),
        policy_loss=float(policy_loss.item()),
        rank_loss=float(losses.rank_loss.item()) if losses.rank_loss else None,
        grad_norm=grad_norm,
        skipped=False,
    )
