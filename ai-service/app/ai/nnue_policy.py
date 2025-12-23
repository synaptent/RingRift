"""NNUE with Policy Head for move prediction.

This module extends the base NNUE architecture with policy outputs
for move prediction. The policy head outputs "from" and "to" heatmaps
over board positions, enabling efficient move scoring.

For a given move, the policy score is computed as:
    score(move) = from_logits[from_pos] + to_logits[to_pos]

This design keeps the network small while providing useful move guidance.

Training:
- Value loss: MSE on game outcome
- Policy loss: Cross-entropy on actual move played
- Combined: L = value_weight * L_value + policy_weight * L_policy
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel

from ..models import BoardType, Position
from ..rules.legacy.move_type_aliases import convert_legacy_move_type

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..models import GameState, Move
from .nnue import (
    ClippedReLU,
    RingRiftNNUE,
    get_board_size,
    get_feature_dim,
)

try:  # pragma: no cover
    from ..training.selfplay_config import normalize_engine_mode
except Exception:  # pragma: no cover
    def normalize_engine_mode(raw_mode: str) -> str:
        return str(raw_mode).strip().lower()


def get_hidden_dim_for_board(board_type: BoardType, board_size: int = 0) -> int:
    """Auto-select hidden dimension based on board type and size.

    Args:
        board_type: The type of board
        board_size: For hexagonal boards, used to distinguish hex8 from full hex

    Returns:
        Recommended hidden dimension for the model

    Sizes:
        - Square8 (64 cells): 128 hidden
        - Hex8 (size <= 8): 128 hidden
        - Full hexagonal (size > 8): 1024 hidden
        - Square19 (361 cells): 512 hidden
    """
    if board_type == BoardType.SQUARE8:
        return 128  # 64 cells - smaller model
    elif board_type == BoardType.SQUARE19:
        return 512  # 361 cells - larger model
    elif board_type == BoardType.HEX8:
        return 128  # hex8 (61 cells) - smaller model matching square8
    elif board_type == BoardType.HEXAGONAL:
        # Distinguish hex8 vs full hex by board_size
        if board_size <= 8:
            return 128  # hex8 - smaller model matching square8
        else:
            return 1024  # full hexagonal (size 19+) - large model
    return 256  # default fallback


class RingRiftNNUEWithPolicy(nn.Module):
    """NNUE network with both value and policy heads.

    Extends RingRiftNNUE with policy outputs for move prediction.
    The policy head outputs "from" and "to" heatmaps over board positions.

    Architecture:
    - Same accumulator and hidden layers as RingRiftNNUE
    - Value head: Linear(32, 1) + tanh -> scalar in [-1, 1]
    - From head: Linear(32, H*W) -> from position logits
    - To head: Linear(32, H*W) -> to position logits
    """

    ARCHITECTURE_VERSION = "v1.1.0"

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 2,
        policy_dropout: float = 0.1,
    ):
        super().__init__()
        self.board_type = board_type
        self.board_size = get_board_size(board_type)
        self.policy_dropout_rate = policy_dropout

        # Auto-select hidden dimension if not specified
        if hidden_dim is None:
            hidden_dim = get_hidden_dim_for_board(board_type, self.board_size)

        input_dim = get_feature_dim(board_type)
        num_positions = self.board_size * self.board_size

        # Accumulator layer (same as RingRiftNNUE)
        self.accumulator = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden layers with ClippedReLU (same as RingRiftNNUE)
        layers: list[nn.Module] = []
        current_dim = hidden_dim * 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, 32))
            layers.append(ClippedReLU())
            current_dim = 32

        self.hidden = nn.Sequential(*layers)

        # Value head: single scalar output
        self.value_head = nn.Linear(32, 1)

        # Policy heads: from/to position logits with dropout for regularization
        self.policy_hidden = nn.Sequential(
            nn.Linear(32, 64),
            ClippedReLU(),
            nn.Dropout(policy_dropout),
        )
        self.from_head = nn.Linear(64, num_positions)
        self.to_head = nn.Linear(64, num_positions)

    def forward(
        self,
        features: torch.Tensor,
        return_policy: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Shape (batch, input_dim) sparse/dense input features
            return_policy: If True, return (value, from_logits, to_logits)

        Returns:
            If return_policy=False: Shape (batch, 1) values in [-1, 1]
            If return_policy=True: Tuple of (value, from_logits, to_logits)
        """
        acc = torch.clamp(self.accumulator(features), 0.0, 1.0)
        x = torch.cat([acc, acc], dim=-1)
        x = self.hidden(x)
        value = torch.tanh(self.value_head(x))

        if not return_policy:
            return value

        policy_features = self.policy_hidden(x)
        from_logits = self.from_head(policy_features)
        to_logits = self.to_head(policy_features)

        return value, from_logits, to_logits

    def forward_single(self, features: np.ndarray) -> float:
        """Convenience method for single-sample value inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features[None, ...]).float()
            device = next(self.parameters()).device
            x = x.to(device)
            value = self.forward(x, return_policy=False)
        return float(value.cpu().item())

    def score_moves(
        self,
        features: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of moves given game state features.

        Args:
            features: Shape (batch, input_dim) state features
            from_indices: Shape (batch, max_moves) flattened from position indices
            to_indices: Shape (batch, max_moves) flattened to position indices
            move_mask: Shape (batch, max_moves) boolean mask for valid moves

        Returns:
            Shape (batch, max_moves) move scores
        """
        _, from_logits, to_logits = self.forward(features, return_policy=True)
        from_scores = torch.gather(from_logits, 1, from_indices)
        to_scores = torch.gather(to_logits, 1, to_indices)
        move_scores = from_scores + to_scores
        if move_mask is not None:
            move_scores = move_scores.masked_fill(~move_mask, float('-inf'))
        return move_scores

    def get_move_probabilities(
        self,
        features: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
        move_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get move probabilities via softmax over legal moves.

        Args:
            features: Shape (batch, input_dim) state features
            from_indices: Shape (batch, max_moves) flattened from position indices
            to_indices: Shape (batch, max_moves) flattened to position indices
            move_mask: Shape (batch, max_moves) boolean mask for valid moves
            temperature: Softmax temperature (higher = more exploration)

        Returns:
            Shape (batch, max_moves) move probabilities
        """
        move_scores = self.score_moves(features, from_indices, to_indices, move_mask)
        return torch.softmax(move_scores / temperature, dim=-1)

    @classmethod
    def from_value_only(
        cls,
        value_model: RingRiftNNUE,
        freeze_value_weights: bool = False,
    ) -> "RingRiftNNUEWithPolicy":
        """Create a policy model from an existing value-only model.

        Loads the value model weights and initializes fresh policy heads.
        Useful for fine-tuning an existing NNUE model with policy learning.

        Args:
            value_model: Trained RingRiftNNUE model
            freeze_value_weights: If True, freeze value weights during training

        Returns:
            RingRiftNNUEWithPolicy with copied value weights
        """
        inferred_layers = sum(
            1 for module in value_model.hidden if isinstance(module, nn.Linear)
        )
        policy_model = cls(
            board_type=value_model.board_type,
            hidden_dim=value_model.accumulator.out_features,
            num_hidden_layers=inferred_layers or 2,
        )
        policy_model.accumulator.load_state_dict(value_model.accumulator.state_dict())
        policy_model.hidden.load_state_dict(value_model.hidden.state_dict())
        policy_model.value_head.load_state_dict(value_model.output.state_dict())

        if freeze_value_weights:
            for param in policy_model.accumulator.parameters():
                param.requires_grad = False
            for param in policy_model.hidden.parameters():
                param.requires_grad = False
            for param in policy_model.value_head.parameters():
                param.requires_grad = False

        return policy_model


def pos_to_flat_index(pos: Position, board_size: int, board_type: BoardType) -> int:
    """Convert a Position to a flattened board index for policy heads.

    Args:
        pos: Position with x, y coordinates
        board_size: Size of the board (e.g., 8 for square8)
        board_type: Board type for coordinate handling

    Returns:
        Flattened index in range [0, board_size * board_size)
    """
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        radius = (board_size - 1) // 2
        cx = pos.x + radius
        cy = pos.y + radius
        return cy * board_size + cx
    else:
        return pos.y * board_size + pos.x


def encode_moves_for_policy(
    moves: list,
    board_size: int,
    board_type: BoardType,
    max_moves: int = 256,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a list of moves for policy head input.

    Args:
        moves: List of Move objects
        board_size: Size of the board
        board_type: Board type
        max_moves: Maximum number of moves to encode
        device: Target device for tensors

    Returns:
        Tuple of (from_indices, to_indices, move_mask):
        - from_indices: Shape (max_moves,) flattened from position indices
        - to_indices: Shape (max_moves,) flattened to position indices
        - move_mask: Shape (max_moves,) boolean mask for valid moves
    """
    center = board_size // 2
    from_indices = torch.zeros(max_moves, dtype=torch.long, device=device)
    to_indices = torch.zeros(max_moves, dtype=torch.long, device=device)
    move_mask = torch.zeros(max_moves, dtype=torch.bool, device=device)

    for i, move in enumerate(moves[:max_moves]):
        from_pos = getattr(move, 'from_pos', None)
        if from_pos is None:
            from_idx = center * board_size + center
        else:
            from_idx = pos_to_flat_index(from_pos, board_size, board_type)

        to_pos = getattr(move, 'to', None)
        if to_pos is None:
            to_pos = from_pos
        if to_pos is None:
            to_idx = center * board_size + center
        else:
            to_idx = pos_to_flat_index(to_pos, board_size, board_type)

        from_indices[i] = from_idx
        to_indices[i] = to_idx
        move_mask[i] = True

    return from_indices, to_indices, move_mask


class NNUEPolicyTrainer:
    """Trainer for NNUE with policy head.

    Combines value loss (MSE) and policy loss (cross-entropy) for
    joint training of value and policy networks.

    Supports:
    - Temperature scaling for sharper/softer policy predictions
    - Label smoothing for regularization
    - Temperature annealing during training
    """

    def __init__(
        self,
        model: RingRiftNNUEWithPolicy,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        use_kl_loss: bool = False,
        kl_loss_weight: float = 1.0,  # Mix ratio: 1.0 = pure KL, 0.5 = 50% KL + 50% CE
        grad_clip: float = 1.0,
        lr_scheduler: str = "plateau",
        total_epochs: int = 100,
        use_amp: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        focal_gamma: float = 0.0,
        label_smoothing_warmup: int = 0,
        use_ddp: bool = False,
        ddp_rank: int = 0,
        use_swa: bool = False,
        swa_start_epoch: int = 0,
        swa_lr: float | None = None,
        progressive_batch_callback: Callable[[int, int], int] | None = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._accumulation_step = 0  # Track current step for gradient accumulation
        self.device = device
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.temperature = temperature
        self.initial_temperature = temperature
        self.label_smoothing = label_smoothing
        self.target_label_smoothing = label_smoothing
        self.label_smoothing_warmup = label_smoothing_warmup
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight  # Mix ratio for KL vs cross-entropy
        self.grad_clip = grad_clip
        self.lr_scheduler_type = lr_scheduler
        self.focal_gamma = focal_gamma
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate

        # Mixed precision training (AMP)
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Distributed Data Parallel (DDP)
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        if use_ddp:
            self.model = DDP(self.model, device_ids=[device.index] if device.type == "cuda" else None)

        # Exponential Moving Average
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_model = None
        if use_ema:
            self._init_ema()

        # Stochastic Weight Averaging (SWA)
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch if swa_start_epoch > 0 else int(total_epochs * 0.75)
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_lr = swa_lr if swa_lr is not None else learning_rate * 0.1

        # Progressive batch sizing callback: (epoch, total_epochs) -> batch_size
        self.progressive_batch_callback = progressive_batch_callback

        # Learning history for plotting
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_value_loss": [], "val_value_loss": [],
            "train_policy_loss": [], "val_policy_loss": [],
            "val_accuracy": [], "learning_rate": [],
        }

        self.optimizer = torch.optim.AdamW(
            self._get_model_params(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        if lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=learning_rate * 0.01
            )
        elif lr_scheduler == "cosine_warmup":
            # Cosine with linear warmup (5% of epochs)
            warmup_epochs = max(1, total_epochs // 20)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs),
                    torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=learning_rate * 0.01),
                ],
                milestones=[warmup_epochs],
            )
        else:  # plateau (default)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5,
            )

        self.value_criterion = nn.MSELoss()
        self.policy_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _get_model_params(self):
        """Get model parameters, handling DDP wrapper."""
        if self.use_ddp and hasattr(self.model, 'module'):
            return self.model.module.parameters()
        return self.model.parameters()

    def _get_base_model(self) -> RingRiftNNUEWithPolicy:
        """Get the base model, unwrapping DDP if needed."""
        if self.use_ddp and hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def _init_swa(self) -> None:
        """Initialize SWA model and scheduler."""
        base_model = self._get_base_model()
        self.swa_model = AveragedModel(base_model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)

    def update_swa(self, epoch: int) -> None:
        """Update SWA model if past start epoch."""
        if not self.use_swa:
            return
        if epoch >= self.swa_start_epoch:
            if self.swa_model is None:
                self._init_swa()
            self.swa_model.update_parameters(self._get_base_model())
            self.swa_scheduler.step()

    def finalize_swa(self, train_loader) -> None:
        """Finalize SWA by updating batch normalization statistics."""
        if not self.use_swa or self.swa_model is None:
            return
        # Update BN statistics for SWA model
        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)

    def get_swa_model(self) -> AveragedModel | None:
        """Get the SWA averaged model for inference."""
        return self.swa_model

    def get_batch_size(self, epoch: int, base_batch_size: int) -> int:
        """Get batch size for current epoch using progressive callback."""
        if self.progressive_batch_callback is not None:
            return self.progressive_batch_callback(epoch, self.total_epochs)
        return base_batch_size

    def _init_ema(self) -> None:
        """Initialize EMA model as a copy of the main model."""
        import copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters(), strict=False):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)

    def get_ema_model(self) -> RingRiftNNUEWithPolicy | None:
        """Get the EMA model for inference."""
        return self.ema_model

    def update_label_smoothing(self, epoch: int) -> None:
        """Warm up label smoothing over epochs."""
        if self.label_smoothing_warmup > 0 and epoch < self.label_smoothing_warmup:
            progress = epoch / self.label_smoothing_warmup
            self.label_smoothing = self.target_label_smoothing * progress
            self.policy_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute focal loss for hard sample mining.

        Args:
            logits: (batch, num_classes) model outputs
            targets: (batch,) target class indices
            reduction: 'mean', 'sum', or 'none' for per-sample loss
        """
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature for policy softmax."""
        self.temperature = max(0.01, temperature)  # Prevent division by zero

    def anneal_temperature(
        self,
        epoch: int,
        total_epochs: int,
        start_temp: float = 2.0,
        end_temp: float = 0.5,
        schedule: str = "linear",
    ) -> float:
        """Anneal temperature during training.

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            start_temp: Starting temperature (higher = softer)
            end_temp: Ending temperature (lower = sharper)
            schedule: "linear", "cosine", or "exponential"

        Returns:
            The new temperature value
        """
        progress = min(1.0, epoch / max(1, total_epochs - 1))

        if schedule == "cosine":
            # Cosine annealing: slower at start and end
            temp = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(math.pi * progress))
        elif schedule == "exponential":
            # Exponential decay
            temp = start_temp * ((end_temp / start_temp) ** progress)
        else:  # linear
            temp = start_temp + (end_temp - start_temp) * progress

        self.set_temperature(temp)
        return temp

    def train_step(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
        move_mask: torch.Tensor,
        target_move_idx: torch.Tensor,
        mcts_probs: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[float, float, float]:
        """Single training step with both value and policy loss.

        Args:
            features: (batch, input_dim) state features
            values: (batch, 1) target values
            from_indices: (batch, max_moves) from position indices
            to_indices: (batch, max_moves) to position indices
            move_mask: (batch, max_moves) valid move mask
            target_move_idx: (batch,) index of the move that was played
            mcts_probs: (batch, max_moves) optional MCTS visit distribution for KL loss
            sample_weights: (batch,) optional per-sample weights for loss

        Returns:
            Tuple of (total_loss, value_loss, policy_loss)
        """
        self.model.train()

        # Only zero gradients at start of accumulation cycle
        if self._accumulation_step == 0:
            self.optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=self.use_amp):
            # Forward pass with policy
            pred_value, from_logits, to_logits = self.model(features, return_policy=True)

            # Value loss (per-sample for weighting)
            value_loss_unreduced = torch.nn.functional.mse_loss(
                pred_value, values, reduction='none'
            ).squeeze(-1)  # (batch,)

            # Policy loss: compute move scores and apply cross-entropy or KL divergence
            # Apply temperature scaling (higher temp = softer targets, lower = sharper)
            from_scores = torch.gather(from_logits, 1, from_indices)
            to_scores = torch.gather(to_logits, 1, to_indices)
            move_scores = (from_scores + to_scores) / self.temperature
            # Use large negative instead of -inf to avoid numerical issues with label smoothing
            move_scores = move_scores.masked_fill(~move_mask, -1e4)  # Use -1e4 for AMP compatibility

            # Use KL divergence if enabled and MCTS probs available
            if self.use_kl_loss and mcts_probs is not None:
                # KL divergence: KL(mcts_probs || softmax(move_scores))
                log_policy = torch.log_softmax(move_scores, dim=-1)
                # Add small epsilon to avoid log(0)
                mcts_probs_safe = mcts_probs.clamp(min=1e-8)
                # Per-sample KL divergence
                kl_loss_unreduced = torch.sum(
                    mcts_probs_safe * (torch.log(mcts_probs_safe) - log_policy), dim=-1
                )  # (batch,)

                # Mix KL and cross-entropy based on kl_loss_weight
                if self.kl_loss_weight < 1.0:
                    ce_loss_unreduced = torch.nn.functional.cross_entropy(
                        move_scores, target_move_idx, reduction='none',
                        label_smoothing=self.policy_criterion.label_smoothing
                    )
                    policy_loss_unreduced = (
                        self.kl_loss_weight * kl_loss_unreduced +
                        (1.0 - self.kl_loss_weight) * ce_loss_unreduced
                    )
                else:
                    policy_loss_unreduced = kl_loss_unreduced
            elif self.focal_gamma > 0:
                # Focal loss for hard sample mining (per-sample)
                policy_loss_unreduced = self._focal_loss(move_scores, target_move_idx, reduction='none')
            else:
                # Cross-entropy per sample
                policy_loss_unreduced = torch.nn.functional.cross_entropy(
                    move_scores, target_move_idx, reduction='none',
                    label_smoothing=self.policy_criterion.label_smoothing
                )  # (batch,)

            # Apply sample weights if provided
            if sample_weights is not None:
                # Normalize weights to sum to batch size for consistent magnitude
                weights = sample_weights / (sample_weights.mean() + 1e-8)
                value_loss = (value_loss_unreduced * weights).mean()
                policy_loss = (policy_loss_unreduced * weights).mean()
            else:
                value_loss = value_loss_unreduced.mean()
                policy_loss = policy_loss_unreduced.mean()

            # Combined loss
            total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.gradient_accumulation_steps

        # Backward pass with gradient scaling for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Increment accumulation step
        self._accumulation_step += 1

        # Only step optimizer every gradient_accumulation_steps
        if self._accumulation_step >= self.gradient_accumulation_steps:
            if self.use_amp and self.scaler is not None:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self._accumulation_step = 0

            # Update EMA model only after optimizer step
            self._update_ema()

        # Return unscaled loss for logging
        unscaled_total = total_loss.item() * self.gradient_accumulation_steps
        return unscaled_total, value_loss.item(), policy_loss.item()

    def validate(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
        move_mask: torch.Tensor,
        target_move_idx: torch.Tensor,
        mcts_probs: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float]:
        """Validate on held-out data.

        Args:
            features: (batch, input_dim) state features
            values: (batch, 1) target values
            from_indices: (batch, max_moves) from position indices
            to_indices: (batch, max_moves) to position indices
            move_mask: (batch, max_moves) valid move mask
            target_move_idx: (batch,) index of the move that was played
            mcts_probs: (batch, max_moves) optional MCTS visit distribution for KL loss

        Returns:
            Tuple of (total_loss, value_loss, policy_loss, policy_accuracy)
        """
        self.model.eval()
        with torch.no_grad():
            pred_value, from_logits, to_logits = self.model(features, return_policy=True)

            # Value loss
            value_loss = self.value_criterion(pred_value, values)

            # Policy loss - apply temperature scaling consistent with training
            from_scores = torch.gather(from_logits, 1, from_indices)
            to_scores = torch.gather(to_logits, 1, to_indices)
            move_scores = (from_scores + to_scores) / self.temperature
            # Use large negative instead of -inf to avoid numerical issues
            move_scores = move_scores.masked_fill(~move_mask, -1e4)  # Use -1e4 for AMP compatibility

            # Use same loss function as training for comparable metrics
            if self.use_kl_loss and mcts_probs is not None:
                log_policy = torch.log_softmax(move_scores, dim=-1)
                mcts_probs_safe = mcts_probs.clamp(min=1e-8)
                policy_loss = torch.nn.functional.kl_div(
                    log_policy, mcts_probs_safe, reduction='batchmean'
                )
            elif self.focal_gamma > 0:
                # Use focal loss for consistency with training
                policy_loss = self._focal_loss(move_scores, target_move_idx, reduction='mean')
            else:
                policy_loss = self.policy_criterion(move_scores, target_move_idx)

            # Policy accuracy (always use argmax for accuracy metric)
            pred_move_idx = move_scores.argmax(dim=-1)
            policy_accuracy = (pred_move_idx == target_move_idx).float().mean()

            total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss

        return total_loss.item(), value_loss.item(), policy_loss.item(), policy_accuracy.item()

    def update_scheduler(self, val_loss: float, epoch: int | None = None) -> None:
        """Update learning rate scheduler.

        Args:
            val_loss: Validation loss (used by plateau scheduler)
            epoch: Current epoch (used by cosine schedulers)
        """
        if self.lr_scheduler_type == "plateau":
            self.scheduler.step(val_loss)
        else:
            # Cosine and cosine_warmup step by epoch
            self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def record_epoch(
        self,
        train_loss: float,
        train_value_loss: float,
        train_policy_loss: float,
        val_loss: float,
        val_value_loss: float,
        val_policy_loss: float,
        val_accuracy: float,
    ) -> None:
        """Record metrics for one epoch."""
        self.history["train_loss"].append(train_loss)
        self.history["train_value_loss"].append(train_value_loss)
        self.history["train_policy_loss"].append(train_policy_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_value_loss"].append(val_value_loss)
        self.history["val_policy_loss"].append(val_policy_loss)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["learning_rate"].append(self.get_lr())

    def save_learning_curves(self, path: str) -> None:
        """Save learning curves to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_learning_curves(self, path: str) -> None:
        """Plot and save learning curves as PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Loss curves
            ax = axes[0, 0]
            ax.plot(self.history["train_loss"], label="Train")
            ax.plot(self.history["val_loss"], label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total Loss")
            ax.set_title("Total Loss")
            ax.legend()
            ax.grid(True)

            # Policy loss
            ax = axes[0, 1]
            ax.plot(self.history["train_policy_loss"], label="Train")
            ax.plot(self.history["val_policy_loss"], label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Policy Loss")
            ax.set_title("Policy Loss")
            ax.legend()
            ax.grid(True)

            # Accuracy
            ax = axes[1, 0]
            ax.plot(self.history["val_accuracy"], label="Val Accuracy", color="green")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Policy Accuracy")
            ax.legend()
            ax.grid(True)

            # Learning rate
            ax = axes[1, 1]
            ax.plot(self.history["learning_rate"], label="LR", color="orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
        except ImportError:
            pass  # matplotlib not available


# =============================================================================
# Data Augmentation for Hexagonal Boards
# =============================================================================


def progressive_batch_schedule(
    epoch: int,
    total_epochs: int,
    min_batch: int = 64,
    max_batch: int = 512,
    warmup_fraction: float = 0.2,
) -> int:
    """Default progressive batch sizing schedule.

    Starts with small batches for better gradient signal, grows to larger
    batches for faster convergence in later epochs.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        min_batch: Starting batch size
        max_batch: Maximum batch size
        warmup_fraction: Fraction of epochs for warmup (linear growth)

    Returns:
        Batch size for this epoch
    """
    warmup_epochs = int(total_epochs * warmup_fraction)
    if epoch < warmup_epochs:
        # Linear growth during warmup
        progress = epoch / warmup_epochs
        batch = int(min_batch + (max_batch - min_batch) * progress)
    else:
        batch = max_batch
    # Round to nearest power of 2 for efficiency
    return min(max_batch, max(min_batch, 2 ** int(np.log2(batch))))


class LearningRateFinder:
    """Find optimal learning rate using the learning rate range test.

    Runs a short training sweep with exponentially increasing learning rate,
    recording loss at each step. The optimal LR is where loss decreases fastest
    (steepest negative gradient) before diverging.

    Usage:
        finder = LearningRateFinder(model, optimizer, criterion)
        optimal_lr = finder.find(train_loader, device, start_lr=1e-7, end_lr=10)
        finder.plot("lr_finder.png")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        trainer: "NNUEPolicyTrainer",
    ):
        """Initialize LR finder.

        Args:
            model: The model to train
            optimizer: Optimizer (will have LR modified during sweep)
            trainer: NNUEPolicyTrainer for train_step method
        """
        self.model = model
        self.optimizer = optimizer
        self.trainer = trainer
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self._initial_state: dict | None = None

    def _save_state(self) -> None:
        """Save model and optimizer state."""
        self._initial_state = {
            "model": {k: v.clone() for k, v in self.model.state_dict().items()},
            "optimizer": {k: v if not isinstance(v, torch.Tensor) else v.clone()
                         for k, v in self.optimizer.state_dict().items()},
        }

    def _restore_state(self) -> None:
        """Restore model and optimizer state."""
        if self._initial_state is not None:
            self.model.load_state_dict(self._initial_state["model"])
            self.optimizer.load_state_dict(self._initial_state["optimizer"])

    def find(
        self,
        train_loader: "DataLoader",
        device: torch.device,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> float:
        """Run learning rate range test.

        Args:
            train_loader: Training data loader
            device: Device to train on
            start_lr: Starting learning rate (very small)
            end_lr: Ending learning rate (where we expect divergence)
            num_iter: Number of iterations for sweep
            smooth_factor: Smoothing factor for loss (0-1)
            diverge_threshold: Stop when loss exceeds this multiple of min loss

        Returns:
            Suggested optimal learning rate
        """
        self._save_state()
        self.lrs = []
        self.losses = []

        # Compute LR multiplier per step for exponential growth
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        self.model.train()
        best_loss = float('inf')
        smoothed_loss = 0.0
        data_iter = iter(train_loader)

        for iteration in range(num_iter):
            # Get next batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Unpack batch
            features, values, from_idx, to_idx, mask, target, sample_weights, _mcts_probs = batch
            features = features.to(device)
            values = values.to(device)
            from_idx = from_idx.to(device)
            to_idx = to_idx.to(device)
            mask = mask.to(device)
            target = target.to(device)
            sample_weights = sample_weights.to(device)

            # Train step
            total_loss, _, _ = self.trainer.train_step(
                features, values, from_idx, to_idx, mask, target, None, sample_weights
            )

            # Smooth the loss
            if iteration == 0:
                smoothed_loss = total_loss
            else:
                smoothed_loss = smooth_factor * total_loss + (1 - smooth_factor) * smoothed_loss

            # Record LR and loss
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            self.losses.append(smoothed_loss)

            # Check for divergence
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            if smoothed_loss > best_loss * diverge_threshold:
                break

            # Increase learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult

        # Restore original state
        self._restore_state()

        # Find optimal LR (steepest descent point)
        return self._find_optimal_lr()

    def _find_optimal_lr(self) -> float:
        """Find optimal LR from recorded data.

        Uses the point of steepest descent (most negative gradient)
        as the suggested learning rate.
        """
        if len(self.lrs) < 10:
            return self.lrs[len(self.lrs) // 2] if self.lrs else 1e-3

        # Compute gradients in log space
        log_lrs = np.log10(self.lrs)
        losses = np.array(self.losses)

        # Smooth gradients
        gradients = np.gradient(losses, log_lrs)

        # Find point of steepest descent (most negative gradient)
        # Avoid very early points (noisy) and very late (diverging)
        start_idx = len(gradients) // 10
        end_idx = len(gradients) * 8 // 10
        search_range = gradients[start_idx:end_idx]

        if len(search_range) == 0:
            return self.lrs[len(self.lrs) // 2]

        min_grad_idx = start_idx + np.argmin(search_range)

        # Return LR slightly before the minimum gradient point
        suggested_idx = max(0, min_grad_idx - 2)
        return self.lrs[suggested_idx]  # type: ignore[call-overload]

    def plot(self, path: str) -> None:
        """Plot LR vs Loss curve and save to file."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            _fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.lrs, self.losses)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')
            ax.grid(True)

            # Mark suggested LR
            optimal_lr = self._find_optimal_lr()
            ax.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Suggested LR: {optimal_lr:.2e}')
            ax.legend()

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
        except ImportError:
            pass  # matplotlib not available


class HexBoardAugmenter:
    """Data augmentation for hexagonal boards using D6 symmetry transformations.

    Hexagonal boards have D6 symmetry (dihedral group of order 12):
    - 6 rotational symmetries (0°, 60°, 120°, 180°, 240°, 300°)
    - Each can be combined with reflection = 12 total transformations

    Uses axial coordinates (q, r) for hex grid transformations:
    - Rotation by 60°: (q, r) -> (-r, q + r)
    - Reflection: (q, r) -> (q, -r - q) or similar

    This class augments training samples by applying these transformations
    to the feature arrays and adjusting move indices accordingly.
    """

    def __init__(self, board_size: int, num_augmentations: int = 6):
        """Initialize augmenter.

        Args:
            board_size: Size of the hex board (e.g., 8 for hex8, 19 for full)
            num_augmentations: Number of augmentations per sample (1-12)
        """
        self.board_size = board_size
        self.num_augmentations = min(num_augmentations, 12)
        self._build_transformation_tables()

    def _axial_to_flat(self, q: int, r: int) -> int:
        """Convert axial coordinates to flat index."""
        # Offset coordinates: row = r, col = q + r // 2
        row = r + self.board_size // 2
        col = q + self.board_size // 2 + r // 2
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row * self.board_size + col
        return -1  # Out of bounds

    def _flat_to_axial(self, idx: int) -> tuple[int, int]:
        """Convert flat index to axial coordinates."""
        row = idx // self.board_size
        col = idx % self.board_size
        r = row - self.board_size // 2
        q = col - self.board_size // 2 - r // 2
        return q, r

    def _rotate_axial_60(self, q: int, r: int) -> tuple[int, int]:
        """Rotate axial coordinates by 60° clockwise."""
        return -r, q + r

    def _reflect_axial(self, q: int, r: int) -> tuple[int, int]:
        """Reflect axial coordinates across q-axis."""
        return q, -r - q

    def _build_transformation_tables(self) -> None:
        """Build lookup tables for all 12 D6 transformations."""
        num_cells = self.board_size * self.board_size
        self.transformation_maps = []

        for t in range(12):
            rotation = t % 6  # 0-5 rotations
            reflect = t >= 6   # Apply reflection for t >= 6

            mapping = np.full(num_cells, -1, dtype=np.int32)

            for idx in range(num_cells):
                q, r = self._flat_to_axial(idx)

                # Apply rotation (multiple 60° rotations)
                for _ in range(rotation):
                    q, r = self._rotate_axial_60(q, r)

                # Apply reflection if needed
                if reflect:
                    q, r = self._reflect_axial(q, r)

                new_idx = self._axial_to_flat(q, r)
                if new_idx >= 0:
                    mapping[idx] = new_idx

            self.transformation_maps.append(mapping)

    def _transform_indices(self, indices: "np.ndarray", transform_idx: int) -> "np.ndarray":
        """Apply transformation to position indices."""
        mapping = self.transformation_maps[transform_idx]
        result = indices.copy()
        for i, idx in enumerate(indices):
            if idx >= 0 and idx < len(mapping):
                new_idx = mapping[idx]
                result[i] = new_idx if new_idx >= 0 else idx
        return result

    def _transform_features(self, features: "np.ndarray", transform_idx: int) -> "np.ndarray":
        """Apply transformation to feature array.

        Features are organized by position, so we need to permute
        the position-based feature blocks according to the transformation.
        """
        mapping = self.transformation_maps[transform_idx]
        num_cells = self.board_size * self.board_size

        # If features are flat (per-position features concatenated)
        if len(features.shape) == 1:
            features_per_cell = len(features) // num_cells
            if features_per_cell * num_cells != len(features):
                # Features don't align with cells, return unchanged
                return features.copy()

            # Reshape, permute, flatten
            reshaped = features.reshape(num_cells, features_per_cell)
            permuted = np.zeros_like(reshaped)
            for old_idx in range(num_cells):
                new_idx = mapping[old_idx]
                if new_idx >= 0:
                    permuted[new_idx] = reshaped[old_idx]
            return permuted.flatten()

        return features.copy()

    def augment_sample(
        self,
        features: "np.ndarray",
        from_indices: "np.ndarray",
        to_indices: "np.ndarray",
        target_move_idx: int,
        include_original: bool = True,
    ) -> list[tuple["np.ndarray", "np.ndarray", "np.ndarray", int]]:
        """Augment a single sample with D6 symmetry transformations.

        Args:
            features: Feature array for the board state
            from_indices: Array of from position indices
            to_indices: Array of to position indices
            target_move_idx: Index of the target move
            include_original: Whether to include the original sample

        Returns:
            List of (features, from_indices, to_indices, target_idx) tuples
        """
        import random
        results = []

        if include_original:
            results.append((features, from_indices, to_indices, target_move_idx))

        # Apply random subset of transformations (skip identity t=0)
        num_to_select = min(self.num_augmentations - (1 if include_original else 0), 11)
        if num_to_select > 0:
            transforms = random.sample(range(1, 12), num_to_select)

            for t in transforms:
                # Transform features
                aug_features = self._transform_features(features, t)

                # Transform move indices
                aug_from = self._transform_indices(from_indices, t)
                aug_to = self._transform_indices(to_indices, t)

                # Target move index stays the same (it's an index into the move list)
                aug_target = target_move_idx

                results.append((aug_features, aug_from, aug_to, aug_target))

        return results


# =============================================================================
# Policy Training Dataset
# =============================================================================

import gzip
import json
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any

from torch.utils.data import Dataset

from .nnue import extract_features_from_gamestate

logger = logging.getLogger(__name__)


def _process_game_batch(
    db_path: str,
    game_ids: list[str],
    config_dict: dict[str, Any],
    board_size: int,
) -> list[dict[str, Any]]:
    """Worker function to process a batch of games in a separate process.

    Returns serialized sample dicts (not NNUEPolicySample objects) to avoid
    pickling issues with numpy arrays.
    """
    from ..models import BoardType, GameState, Move
    from ..rules.default_engine import DefaultRulesEngine

    # Reconstruct config from dict
    board_type = BoardType(config_dict['board_type'])
    sample_every_n = config_dict['sample_every_n_moves']
    max_moves = config_dict['max_moves_per_position']
    include_draws = config_dict['include_draws']
    distill_from_winners = config_dict['distill_from_winners']
    winner_weight_boost = config_dict['winner_weight_boost']
    weight_by_game_length = config_dict.get('weight_by_game_length', True)
    game_length_weight_cap = config_dict.get('game_length_weight_cap', 50)
    min_move_number = config_dict.get('min_move_number', 0)
    max_move_number = config_dict.get('max_move_number', 999999)

    samples = []
    engine = DefaultRulesEngine()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Detect schema for move count column
    cursor.execute("PRAGMA table_info(games)")
    columns = {row['name'] for row in cursor.fetchall()}
    moves_col = 'total_moves' if 'total_moves' in columns else 'move_count'

    for game_id in game_ids:
        try:
            # Get game info including move count for length weighting
            cursor.execute(
                f"SELECT winner, {moves_col} as game_length FROM games WHERE game_id = ?",
                (game_id,)
            )
            game_row = cursor.fetchone()
            if not game_row:
                continue
            winner = game_row['winner']
            game_length = game_row['game_length'] or 0

            # Get initial state - try database first, fall back to creating from board_type
            cursor.execute(
                "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
                (game_id,)
            )
            initial_row = cursor.fetchone()

            state = None
            if initial_row:
                initial_json = initial_row['initial_state_json']
                if initial_row['compressed']:
                    try:
                        if isinstance(initial_json, (bytes, bytearray)):
                            initial_json = gzip.decompress(bytes(initial_json)).decode("utf-8")
                        else:
                            initial_json = gzip.decompress(str(initial_json).encode("utf-8")).decode("utf-8")
                    except Exception:
                        initial_json = None

                if initial_json:
                    try:
                        state_dict = json.loads(initial_json)
                        state = GameState(**state_dict)
                    except Exception:
                        state = None

            # Fall back to creating initial state from board_type
            if state is None:
                try:
                    from ..training.initial_state import create_initial_state
                    state = create_initial_state(board_type, config_dict.get('num_players', 2))
                except Exception:
                    continue

            # Get all moves
            cursor.execute(
                "SELECT move_number, move_json FROM game_moves WHERE game_id = ? ORDER BY move_number",
                (game_id,)
            )
            moves = cursor.fetchall()

            if not moves:
                continue

            # Replay and sample
            for move_row in moves:
                move_number = move_row['move_number']
                move_json_str = move_row['move_json']

                if move_number % sample_every_n == 0:
                    current_player = state.current_player or 1
                    is_winner = winner == current_player

                    if is_winner:
                        value = 1.0
                    elif winner is None or winner == 0:
                        value = 0.0
                    else:
                        value = -1.0

                    # Filtering
                    should_include = True
                    if (distill_from_winners and not is_winner) or (value == 0.0 and not include_draws):
                        should_include = False
                    elif move_number < min_move_number or move_number > max_move_number:
                        should_include = False  # Curriculum: filter by move range

                    if should_include:
                        try:
                            legal_moves = engine.get_valid_moves(state, current_player)
                            if legal_moves:
                                move_dict = json.loads(move_json_str)
                                played_move = Move(**move_dict)

                                # Find move index
                                target_idx = _find_move_index_static(played_move, legal_moves)

                                # Skip if target is beyond max_moves encoding limit
                                if target_idx >= 0 and target_idx < max_moves:
                                    features = extract_features_from_gamestate(state, current_player)
                                    from_indices, to_indices, move_mask = _encode_legal_moves_static(
                                        legal_moves, board_size, max_moves, board_type
                                    )

                                    # Calculate sample weight
                                    sample_weight = 1.0

                                    # Apply game length weight (longer games = more signal)
                                    if weight_by_game_length and game_length_weight_cap > 0:
                                        sample_weight *= min(1.0, game_length / game_length_weight_cap)

                                    # Apply winner weight boost
                                    if is_winner and winner_weight_boost > 1.0:
                                        sample_weight *= winner_weight_boost

                                    # Store as dict for pickling
                                    samples.append({
                                        'features': features.tolist(),
                                        'value': value,
                                        'from_indices': from_indices.tolist(),
                                        'to_indices': to_indices.tolist(),
                                        'move_mask': move_mask.tolist(),
                                        'target_move_idx': target_idx,
                                        'player_number': current_player,
                                        'game_id': game_id,
                                        'move_number': move_number,
                                        'sample_weight': sample_weight,
                                    })
                        except Exception:
                            pass

                # Advance state
                try:
                    move_dict = json.loads(move_json_str)
                    move = Move(**move_dict)
                    state = engine.apply_move(state, move)
                except Exception:
                    break

        except Exception:
            continue

    conn.close()
    return samples


def _find_move_index_static(played_move, legal_moves: list) -> int:
    """Static version of move index finding for multiprocessing."""
    played_type = getattr(played_move, 'type', None)
    played_from = getattr(played_move, 'from_pos', None)
    played_to = getattr(played_move, 'to', None)

    for i, legal in enumerate(legal_moves):
        legal_type = getattr(legal, 'type', None)
        legal_from = getattr(legal, 'from_pos', None)
        legal_to = getattr(legal, 'to', None)

        if played_type != legal_type:
            continue
        if played_from is not None and legal_from is not None:
            if played_from.x != legal_from.x or played_from.y != legal_from.y:
                continue
        elif played_from is not None or legal_from is not None:
            continue
        if played_to is not None and legal_to is not None:
            if played_to.x != legal_to.x or played_to.y != legal_to.y:
                continue
        elif played_to is not None or legal_to is not None:
            continue
        return i
    return -1


def _encode_legal_moves_static(
    legal_moves: list,
    board_size: int,
    max_moves: int,
    board_type,
) -> tuple:
    """Static version of move encoding for multiprocessing."""
    import numpy as np

    center = board_size // 2
    center_idx = center * board_size + center

    from_indices = np.zeros(max_moves, dtype=np.int64)
    to_indices = np.zeros(max_moves, dtype=np.int64)
    move_mask = np.zeros(max_moves, dtype=bool)

    for i, move in enumerate(legal_moves[:max_moves]):
        from_pos = getattr(move, 'from_pos', None)
        if from_pos is None:
            from_idx = center_idx
        else:
            from_idx = pos_to_flat_index(from_pos, board_size, board_type)

        to_pos = getattr(move, 'to', None)
        if to_pos is None:
            to_pos = from_pos
        if to_pos is None:
            to_idx = center_idx
        else:
            to_idx = pos_to_flat_index(to_pos, board_size, board_type)

        from_indices[i] = from_idx
        to_indices[i] = to_idx
        move_mask[i] = True

    return from_indices, to_indices, move_mask


@dataclass
class NNUEPolicySample:
    """Training sample for NNUE with policy head."""
    features: "np.ndarray"  # Shape: (feature_dim,)
    value: float  # Game outcome: +1 win, -1 loss, 0 draw
    from_indices: "np.ndarray"  # Shape: (max_moves,) flattened from positions
    to_indices: "np.ndarray"  # Shape: (max_moves,) flattened to positions
    move_mask: "np.ndarray"  # Shape: (max_moves,) boolean valid moves
    target_move_idx: int  # Index of the move that was played
    player_number: int
    game_id: str
    move_number: int
    sample_weight: float = 1.0  # Weight for weighted loss (for distillation)
    # Optional MCTS visit distribution for KL divergence training
    # Shape: (max_moves,) normalized visit counts, None if not available
    mcts_visit_distribution: Optional["np.ndarray"] = None


@dataclass
class NNUEPolicyDatasetConfig:
    """Configuration for policy dataset generation."""
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    sample_every_n_moves: int = 1
    min_game_length: int = 10
    max_moves_per_position: int = 128  # Max legal moves to encode
    include_draws: bool = True

    # Policy distillation options (for training on strong games)
    distill_from_winners: bool = False  # Only include positions from winning players
    winner_weight_boost: float = 1.0    # Sample weight multiplier for winners (for weighted loss)

    # Sample weighting options
    weight_by_game_length: bool = True  # Weight samples by game length (longer = more signal)
    game_length_weight_cap: int = 50    # Games with >= this many moves get weight 1.0

    # Curriculum learning move range filter
    min_move_number: int = 0       # Only include positions with move_number >= this
    max_move_number: int = 999999  # Only include positions with move_number <= this


class NNUEPolicyDataset(Dataset):
    """PyTorch Dataset for NNUE policy training.

    Loads games from SQLite and/or JSONL files, replays them to get legal moves
    at each position, and encodes the move that was played for policy supervision.

    Supports parallel extraction via num_workers parameter for faster loading.
    JSONL files are particularly useful for MCTS selfplay data with embedded policy distributions.
    """

    def __init__(
        self,
        db_paths: list[str],
        config: NNUEPolicyDatasetConfig | None = None,
        max_samples: int | None = None,
        num_workers: int = 0,
        jsonl_paths: list[str] | None = None,
    ):
        self.db_paths = db_paths
        self.jsonl_paths = jsonl_paths or []
        self.config = config or NNUEPolicyDatasetConfig()
        self.max_samples = max_samples
        self.num_workers = num_workers if num_workers > 0 else max(1, cpu_count() - 2)
        self.board_size = get_board_size(self.config.board_type)
        self.feature_dim = get_feature_dim(self.config.board_type)
        self.samples: list[NNUEPolicySample] = []

        self._extract_samples()

    def _extract_samples(self) -> None:
        """Extract training samples from SQLite databases and JSONL files.

        Uses parallel processing when num_workers > 1 for significantly faster
        extraction from large databases.
        """
        len(self.db_paths) + len(self.jsonl_paths)
        logger.info(f"Extracting policy samples from {len(self.db_paths)} databases + {len(self.jsonl_paths)} JSONL files (workers={self.num_workers})")

        # Process JSONL files first (they often have MCTS policy data)
        for jsonl_path in self.jsonl_paths:
            if not os.path.exists(jsonl_path):
                logger.warning(f"JSONL file not found: {jsonl_path}")
                continue

            try:
                samples = self._extract_from_jsonl(jsonl_path)
                self.samples.extend(samples)
                logger.info(f"Extracted {len(samples)} policy samples from JSONL {jsonl_path}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to extract from JSONL {jsonl_path}: {e}\n{traceback.format_exc()}")

            if self.max_samples and len(self.samples) >= self.max_samples:
                self.samples = self.samples[:self.max_samples]
                logger.info(f"Total policy samples: {len(self.samples)}")
                return

        for db_path in self.db_paths:
            if not os.path.exists(db_path):
                logger.warning(f"Database not found: {db_path}")
                continue

            try:
                if self.num_workers > 1:
                    samples = self._extract_from_db_parallel(db_path)
                else:
                    samples = self._extract_from_db(db_path)
                self.samples.extend(samples)
                logger.info(f"Extracted {len(samples)} policy samples from {db_path}")
            except Exception as e:
                logger.error(f"Failed to extract from {db_path}: {e}")

            if self.max_samples and len(self.samples) >= self.max_samples:
                self.samples = self.samples[:self.max_samples]
                break

        logger.info(f"Total policy samples: {len(self.samples)}")

    def _extract_from_jsonl(self, jsonl_path: str) -> list[NNUEPolicySample]:
        """Extract samples from a JSONL file containing game records with moves.

        JSONL files should contain one JSON game object per line with:
        - board_type: string (e.g., "square8")
        - num_players: int
        - winner: int (player number who won)
        - moves: list of move dicts, each potentially containing mcts_policy
        - initial_state: optional initial GameState dict

        This is particularly useful for MCTS selfplay data that includes
        mcts_policy distributions in the move records.

        IMPORTANT: GPU selfplay JSONL has known incompatibilities with CPU engine:
        - Hex boards: GPU uses 8-dir grid movement, CPU uses 6-dir hex movement
        - Square boards: GPU records sub-actions (place/move/capture) as separate
          "moves" within a single turn, breaking CPU replay which expects
          one action per turn step

        For reliable training data, use:
        - DB data from CPU-based selfplay or MCTS
        - MCTS JSONL with mcts_policy distributions

        For GPU selfplay data, this method attempts to expand moves to canonical
        format by inserting required phase-handling moves, but results may be
        incomplete due to the fundamental format differences.
        """
        import numpy as np

        from ..game_engine import GameEngine
        from ..models import GameState, Move, MoveType
        from ..models.core import GamePhase
        from ..rules.default_engine import DefaultRulesEngine
        from ..rules.legacy import auto_advance_phase
        from ..training.initial_state import create_initial_state

        samples: list[NNUEPolicySample] = []
        DefaultRulesEngine()
        board_type_str = self.config.board_type.value.lower() if hasattr(self.config.board_type, 'value') else str(self.config.board_type).lower()

        import time as _time
        extraction_start = _time.time()
        games_processed = 0
        games_skipped = 0

        # Track GPU selfplay warnings (shown once per extraction)
        # GPU selfplay has known incompatibilities:
        # - Hex boards: 8-dir grid movement vs 6-dir hex movement
        # - Square boards: sub-action recording breaks CPU replay
        is_hex_board = self.config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        gpu_warned = False

        with open(jsonl_path) as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by board type
                game_board = game.get("board_type", "").lower()
                if board_type_str not in game_board:
                    games_skipped += 1
                    continue

                # Filter by num_players
                if game.get("num_players") != self.config.num_players:
                    games_skipped += 1
                    continue

                # Skip incomplete games
                if game.get("game_status") != "completed" or game.get("winner") is None:
                    games_skipped += 1
                    continue

                # Detect selfplay data source and coordinate format
                source = game.get("source", "")
                raw_engine_mode = game.get("engine_mode", "")
                engine_mode = normalize_engine_mode(raw_engine_mode) if raw_engine_mode else ""

                # GPU heuristic uses offset coords for hex boards
                is_gpu_heuristic = (
                    source.startswith("run_gpu") or
                    engine_mode == "gpu_heuristic"
                )
                # Gumbel MCTS and other CPU modes use cube coords (z omitted, derived from x+y)
                is_gumbel_mcts = (
                    "gumbel" in source.lower() or
                    engine_mode == "gumbel-mcts"
                )
                # For phase handling: both GPU and Gumbel need special treatment
                is_gpu_selfplay = is_gpu_heuristic or is_gumbel_mcts

                # Track coordinate format for position parsing
                uses_offset_coords = is_gpu_heuristic and not is_gumbel_mcts

                # Warn about phase-skipping selfplay incompatibility
                # Hex boards: GPU mode is completely incompatible (skip)
                # Square boards: Auto-advance through phases will be attempted
                if is_gpu_selfplay and not gpu_warned:
                    if is_hex_board and engine_mode == "gpu_heuristic":
                        logger.warning(
                            f"Skipping GPU selfplay JSONL for hex board ({self.config.board_type.value}). "
                            "GPU uses 8-dir grid movement, hex requires 6-dir hex movement. "
                            "Use DB data from CPU/MCTS selfplay instead."
                        )
                    else:
                        logger.info(
                            f"Phase-skipping selfplay detected (source={source[:30]}, engine={engine_mode}). "
                            "Auto-advancing through line/territory processing phases."
                        )
                    gpu_warned = True

                # Skip hex GPU heuristic selfplay entirely (incompatible movement rules)
                # Note: Gumbel MCTS on hex boards is fine, only GPU heuristic is broken
                if is_hex_board and engine_mode == "gpu_heuristic":
                    games_skipped += 1
                    continue

                winner = game.get("winner")
                moves = game.get("moves", [])
                game_id = game.get("game_id", f"jsonl_{line_num}")
                game_length = len(moves)

                if game_length < self.config.min_game_length:
                    games_skipped += 1
                    continue

                # Get initial state
                initial_state_dict = game.get("initial_state")
                if initial_state_dict:
                    try:
                        state = GameState(**initial_state_dict)
                    except Exception:
                        state = create_initial_state(self.config.board_type, self.config.num_players)
                else:
                    state = create_initial_state(self.config.board_type, self.config.num_players)

                # Check if this game already encodes explicit phase transitions.
                # If so, we skip auto-advance and apply moves directly.
                explicit_phase_markers = {
                    "process_line",
                    "choose_line_option",
                    "choose_territory_option",
                    "eliminate_rings_from_stack",
                    "forced_elimination",
                }
                has_explicit_bookkeeping = any(
                    (raw_type := str(m.get("type", "") or "").lower())
                    and (
                        (move_type := convert_legacy_move_type(raw_type, warn=False)).startswith("no_")
                        or move_type.startswith("skip_")
                        or move_type in explicit_phase_markers
                    )
                    for m in moves
                )

                # Replay game and extract samples
                for move_idx, move_dict in enumerate(moves):
                    move_number = move_idx + 1

                    # Check if this is a bookkeeping move (no_line_action, no_territory_action, etc.)
                    move_type_str = move_dict.get('type', '')
                    is_bookkeeping_move = move_type_str.startswith('no_')

                    # LEGACY PATH (RR-CANON-R075 VIOLATION - deprecated, removal Q2 2026):
                    # Auto-advance only for GPU selfplay data WITHOUT explicit bookkeeping moves.
                    # This violates canonical spec: new selfplay must emit explicit bookkeeping moves.
                    # Once all legacy GPU data is regenerated with Gumbel MCTS, remove this block.
                    game_status_str = state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status)
                    if is_gpu_selfplay and not has_explicit_bookkeeping and game_status_str == "active":
                        try:
                            state = auto_advance_phase(state)  # emits DeprecationWarning
                        except Exception:
                            break  # Can't continue if phase advance fails

                    # Skip sampling from bookkeeping moves (not decision points)
                    if is_bookkeeping_move:
                        # Still need to apply the move to advance state
                        try:
                            move = self._parse_jsonl_move(move_dict, move_idx, uses_offset_coords)
                            if move is not None:
                                state = GameEngine.apply_move(state, move)
                        except Exception:
                            break
                        continue

                    # Sample every Nth position
                    game_status_str = state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status)
                    if move_number % self.config.sample_every_n_moves == 0 and game_status_str == "active":
                        current_player = state.current_player or 1

                        # Calculate value from perspective of current player
                        is_winner = winner == current_player
                        if is_winner:
                            value = 1.0
                        elif winner is None or winner == 0:
                            value = 0.0
                        else:
                            value = -1.0

                        # Apply filters
                        skip_sample = False
                        if (self.config.distill_from_winners and not is_winner) or (value == 0.0 and not self.config.include_draws) or move_number < self.config.min_move_number or move_number > self.config.max_move_number:
                            skip_sample = True

                        if not skip_sample:
                            try:
                                # Get legal moves at this position
                                legal_moves = GameEngine.get_valid_moves(state, current_player)
                                if legal_moves:
                                    # Parse the move that was played
                                    played_move = self._parse_jsonl_move(move_dict, move_number, uses_offset_coords)
                                    if played_move is not None:
                                        # Find which legal move matches the played move
                                        target_idx = self._find_move_index(played_move, legal_moves)

                                        if target_idx >= 0 and target_idx < self.config.max_moves_per_position:
                                            # Extract features
                                            features = extract_features_from_gamestate(state, current_player)

                                            # Encode legal moves
                                            from_indices, to_indices, move_mask = self._encode_legal_moves(legal_moves)

                                            # Calculate sample weight
                                            sample_weight = 1.0
                                            if self.config.weight_by_game_length and self.config.game_length_weight_cap > 0:
                                                sample_weight *= min(1.0, game_length / self.config.game_length_weight_cap)
                                            if is_winner and self.config.winner_weight_boost > 1.0:
                                                sample_weight *= self.config.winner_weight_boost

                                            # Extract MCTS policy distribution if available
                                            mcts_visit_dist = None
                                            mcts_policy_dict = move_dict.get('mcts_policy') or move_dict.get('mctsPolicy')
                                            if mcts_policy_dict and isinstance(mcts_policy_dict, dict):
                                                mcts_visit_dist = np.zeros(self.config.max_moves_per_position, dtype=np.float32)
                                                for idx_str, prob in mcts_policy_dict.items():
                                                    idx = int(idx_str)
                                                    if 0 <= idx < self.config.max_moves_per_position:
                                                        mcts_visit_dist[idx] = float(prob)
                                                total = mcts_visit_dist.sum()
                                                if total > 0:
                                                    mcts_visit_dist /= total

                                            sample = NNUEPolicySample(
                                                features=features,
                                                value=value,
                                                from_indices=from_indices,
                                                to_indices=to_indices,
                                                move_mask=move_mask,
                                                target_move_idx=target_idx,
                                                player_number=current_player,
                                                game_id=game_id,
                                                move_number=move_number,
                                                sample_weight=sample_weight,
                                                mcts_visit_distribution=mcts_visit_dist,
                                            )
                                            samples.append(sample)
                            except Exception as e:
                                logger.debug(f"Failed to process {game_id}:{move_number}: {e}")

                    # Apply move to advance state
                    try:
                        move = self._parse_jsonl_move(move_dict, move_idx, uses_offset_coords)
                        if move is not None:
                            # For GPU selfplay captures, compute capture_target from current state
                            # GPU records 'to' as landing position, but CPU needs actual target
                            if (is_gpu_selfplay and move.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CHAIN_CAPTURE)
                                    and move.capture_target is None and move.from_pos and move.to):
                                target = self._compute_capture_target(state, move.from_pos, move.to)
                                if target:
                                    move = Move(
                                        id=move.id,
                                        type=move.type,
                                        player=move.player,
                                        from_pos=move.from_pos,
                                        to=move.to,
                                        capture_target=target,
                                        timestamp=move.timestamp,
                                        think_time=move.think_time,
                                        move_number=move.move_number,
                                    )
                            # Use GameEngine for consistent phase handling
                            state = GameEngine.apply_move(state, move)
                            # Auto-advance only for GPU selfplay data WITHOUT explicit bookkeeping moves
                            game_status_str = state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status)
                            if is_gpu_selfplay and not has_explicit_bookkeeping and game_status_str == "active":
                                state = auto_advance_phase(state)
                    except Exception:
                        break  # Can't continue if move application fails

                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples

                games_processed += 1
                if games_processed % 100 == 0:
                    elapsed = _time.time() - extraction_start
                    rate = games_processed / elapsed if elapsed > 0 else 0
                    logger.info(f"  JSONL progress: {games_processed} games ({rate:.1f}/s), {len(samples)} samples")

        total_time = _time.time() - extraction_start
        logger.info(f"  JSONL extraction: {games_processed} games, {games_skipped} skipped in {total_time:.1f}s, {len(samples)} samples")
        return samples

    def _parse_jsonl_move(
        self, move_dict: dict, move_number: int, uses_offset_coords: bool = False
    ) -> Optional["Move"]:
        """Parse a move dict from JSONL into a Move object.

        For hexagonal boards with GPU selfplay (uses_offset_coords=True), converts
        offset coordinates to cube coordinates. For Gumbel MCTS and other CPU modes,
        coordinates are already cube (just missing z, which is derived).
        """
        from datetime import datetime

        from ..models import Move, MoveType, Position

        move_type_str = str(move_dict.get("type") or "").strip()
        if not move_type_str or move_type_str.startswith("unknown_"):
            return None

        # Map GPU selfplay move type names to CPU MoveType values
        # GPU uses: place, move_stack, capture, no_action
        # CPU uses: place_ring, move_stack, overtaking_capture, etc.
        gpu_move_type_map = {
            "place": "place_ring",
            "capture": "overtaking_capture",
            "no_action": "no_movement_action",
            # move_stack matches directly
        }
        mapped_type = gpu_move_type_map.get(move_type_str.lower(), move_type_str)

        try:
            move_type = MoveType(mapped_type)
        except ValueError:
            return None

        # Check if we need coordinate conversion for hex boards
        is_hex_board = self.config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        center_offset = self.board_size // 2 if is_hex_board else 0

        def parse_pos(pos_data):
            if not pos_data:
                return None

            # Handle string format "y,x" from GPU selfplay
            if isinstance(pos_data, str):
                if "," in pos_data:
                    parts = pos_data.split(",")
                    if len(parts) >= 2:
                        y, x = int(parts[0]), int(parts[1])
                        if is_hex_board:
                            cube_x = x - center_offset
                            cube_y = y - center_offset
                            cube_z = -cube_x - cube_y
                            return Position(x=cube_x, y=cube_y, z=cube_z)
                        return Position(x=x, y=y)
                return None

            # Handle dict format {"x": ..., "y": ..., "z": ...}
            if not isinstance(pos_data, dict):
                return None
            x = pos_data.get("x", 0)
            y = pos_data.get("y", 0)
            z = pos_data.get("z")

            # Handle missing z for hex boards
            if is_hex_board and z is None:
                if uses_offset_coords:
                    # GPU selfplay uses offset coordinates: convert to cube
                    cube_x = x - center_offset
                    cube_y = y - center_offset
                    cube_z = -cube_x - cube_y  # Cube constraint: x+y+z=0
                    return Position(x=cube_x, y=cube_y, z=cube_z)
                else:
                    # Gumbel/canonical selfplay uses cube coords but omits z
                    # Just derive z from cube constraint: x + y + z = 0
                    return Position(x=x, y=y, z=-x - y)

            return Position(x=x, y=y, z=z)

        from_pos = parse_pos(move_dict.get("from") or move_dict.get("from_pos"))
        to_pos = parse_pos(move_dict.get("to"))
        capture_target = parse_pos(move_dict.get("capture_target") or move_dict.get("captureTarget"))

        # For capture moves, capture_target must be provided or computed.
        # GPU selfplay records 'to' as the LANDING position (where attacker ends up),
        # NOT the capture target. The actual target is between from and to.
        # We can't compute this here without game state, so we leave capture_target as None
        # and let the caller (extraction code) compute it from the current state if needed.
        # NOTE: Do NOT default capture_target to to_pos for GPU moves!
        # That's incorrect - to_pos is landing, not target.
        # The extraction code must compute target from game state.

        # For PLACE_RING from GPU selfplay, the position is stored in 'from' not 'to'
        # because there's no "from" position for placement. Swap them.
        if move_type == MoveType.PLACE_RING and from_pos is not None and to_pos is None:
            to_pos = from_pos
            from_pos = None

        return Move(
            id=move_dict.get("id", f"jsonl-{move_number}"),
            type=move_type,
            player=move_dict.get("player", 1),
            from_pos=from_pos,
            to=to_pos,
            capture_target=capture_target,
            timestamp=move_dict.get("timestamp", datetime.now()),
            think_time=move_dict.get("think_time", move_dict.get("thinkTime", 0)),
            move_number=move_dict.get("move_number", move_dict.get("moveNumber", move_number + 1)),
        )

    def _compute_capture_target(
        self, state: "GameState", from_pos: "Position", to_pos: "Position"
    ) -> Optional["Position"]:
        """Compute capture target by scanning from from_pos towards to_pos.

        For GPU selfplay captures, 'to' is the landing position, not the target.
        The actual target is the first occupied cell along the ray from 'from' to 'to'.

        Args:
            state: Current game state (before applying the capture)
            from_pos: Starting position of attacker
            to_pos: Landing position (where attacker ends up)

        Returns:
            Position of the capture target, or None if not found
        """
        from ..board_manager import BoardManager
        from ..models import Position

        board = state.board

        # Calculate direction from from_pos to to_pos
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y

        # Normalize to unit direction
        # For square boards, one of dx/dy is 0 or |dx| == |dy| (diagonal)
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)

        # Scan along ray from from_pos towards to_pos
        step = 1
        while True:
            check_x = from_pos.x + dx * step
            check_y = from_pos.y + dy * step
            check_z = None
            if hasattr(from_pos, 'z') and from_pos.z is not None:
                dz = to_pos.z - from_pos.z if hasattr(to_pos, 'z') and to_pos.z is not None else 0
                if dz != 0:
                    dz = dz // abs(dz)
                check_z = from_pos.z + dz * step

            check_pos = Position(x=check_x, y=check_y, z=check_z)

            # Stop if we've reached or passed landing
            if check_x == to_pos.x and check_y == to_pos.y:
                break  # Landed without finding target (shouldn't happen for valid capture)

            # Check if there's a stack at this position
            stack = BoardManager.get_stack(check_pos, board)
            if stack and stack.stack_height > 0:
                return check_pos  # Found the target

            step += 1
            # Safety limit
            if step > 20:
                break

        return None

    def _extract_from_db_parallel(self, db_path: str) -> list[NNUEPolicySample]:
        """Extract samples using parallel processing across multiple workers."""
        import numpy as np

        # Get list of game IDs to process
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Detect schema
        cursor.execute("PRAGMA table_info(games)")
        columns = {row['name'] for row in cursor.fetchall()}
        moves_col = 'total_moves' if 'total_moves' in columns else 'move_count'

        board_type_str = self.config.board_type.value.lower()
        query = f"""
            SELECT game_id
            FROM games
            WHERE game_status = 'completed'
              AND winner IS NOT NULL
              AND board_type = ?
              AND num_players = ?
              AND {moves_col} >= ?
        """
        cursor.execute(query, (
            board_type_str,
            self.config.num_players,
            self.config.min_game_length,
        ))
        game_ids = [row['game_id'] for row in cursor.fetchall()]
        conn.close()

        if not game_ids:
            return []

        # Estimate games needed based on max_samples
        if self.max_samples:
            # Rough estimate: ~10 samples per game on average
            games_needed = min(len(game_ids), self.max_samples // 5 + 100)
            game_ids = game_ids[:games_needed]

        logger.info(f"Processing {len(game_ids)} games with {self.num_workers} workers")

        # Split games into batches for each worker
        batch_size = max(1, len(game_ids) // self.num_workers)
        batches = []
        for i in range(0, len(game_ids), batch_size):
            batches.append(game_ids[i:i + batch_size])

        # Prepare config dict for pickling
        config_dict = {
            'board_type': self.config.board_type.value,
            'num_players': self.config.num_players,
            'sample_every_n_moves': self.config.sample_every_n_moves,
            'min_game_length': self.config.min_game_length,
            'max_moves_per_position': self.config.max_moves_per_position,
            'include_draws': self.config.include_draws,
            'distill_from_winners': self.config.distill_from_winners,
            'winner_weight_boost': self.config.winner_weight_boost,
            'weight_by_game_length': self.config.weight_by_game_length,
            'game_length_weight_cap': self.config.game_length_weight_cap,
            'min_move_number': self.config.min_move_number,
            'max_move_number': self.config.max_move_number,
        }

        # Process batches in parallel
        import time as _time
        parallel_start = _time.time()
        all_sample_dicts = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    _process_game_batch,
                    db_path,
                    batch,
                    config_dict,
                    self.board_size,
                ): i for i, batch in enumerate(batches)
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    batch_samples = future.result()
                    all_sample_dicts.extend(batch_samples)
                    completed += 1
                    if completed % 5 == 0 or completed == len(batches):
                        elapsed = _time.time() - parallel_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(batches) - completed) / rate if rate > 0 else 0
                        logger.info(f"  Completed {completed}/{len(batches)} batches ({rate:.2f} batches/s, ETA: {eta:.0f}s), {len(all_sample_dicts)} samples")
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")

                # Early exit if we have enough samples
                if self.max_samples and len(all_sample_dicts) >= self.max_samples:
                    break

        # Convert dicts back to NNUEPolicySample objects
        samples = []
        for d in all_sample_dicts[:self.max_samples] if self.max_samples else all_sample_dicts:
            samples.append(NNUEPolicySample(
                features=np.array(d['features'], dtype=np.float32),
                value=d['value'],
                from_indices=np.array(d['from_indices'], dtype=np.int64),
                to_indices=np.array(d['to_indices'], dtype=np.int64),
                move_mask=np.array(d['move_mask'], dtype=bool),
                target_move_idx=d['target_move_idx'],
                player_number=d['player_number'],
                game_id=d['game_id'],
                move_number=d['move_number'],
                sample_weight=d['sample_weight'],
            ))

        return samples

    def _extract_from_db(self, db_path: str) -> list[NNUEPolicySample]:
        """Extract samples from a single database via game replay.

        Handles GPU selfplay data by synthesizing bookkeeping moves on-the-fly.
        GPU selfplay records simplified moves (place_ring, move_stack, capture)
        without intermediate phase-handling moves. This method uses the same
        phase synthesis logic as parity testing to bridge the gap.
        """
        from ..game_engine import GameEngine
        from ..models import GameState, Move
        from ..rules.default_engine import DefaultRulesEngine

        def _advance_to_phase(state: GameState, target_phase_str: str, max_steps: int = 50) -> GameState:
            """Advance state through bookkeeping phases until target phase is reached.

            GPU selfplay records phases in a different order than CPU engine transitions.
            This function uses the same approach as import_gpu_selfplay_to_db.py to
            bridge the gap: synthesize bookkeeping moves until the CPU state matches
            the recorded GPU phase.
            """
            from ..models.core import GamePhase

            try:
                target = GamePhase(target_phase_str)
            except ValueError:
                return state  # Unknown phase, return as-is

            for _ in range(max_steps):
                if state.current_phase == target:
                    return state
                if state.game_status.value != "active":
                    return state

                req = GameEngine.get_phase_requirement(state, state.current_player)
                if req is not None:
                    synth = GameEngine.synthesize_bookkeeping_move(req, state)
                    if synth is not None:
                        state = GameEngine.apply_move(state, synth)
                        continue
                # No bookkeeping available, can't advance further
                break
            return state

        samples: list[NNUEPolicySample] = []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Detect schema - different databases use different column names
        cursor.execute("PRAGMA table_info(games)")
        columns = {row['name'] for row in cursor.fetchall()}
        moves_col = 'total_moves' if 'total_moves' in columns else 'move_count'

        # Get completed games
        board_type_str = self.config.board_type.value.lower()
        query = f"""
            SELECT game_id, winner, {moves_col} as moves
            FROM games
            WHERE game_status = 'completed'
              AND winner IS NOT NULL
              AND board_type = ?
              AND num_players = ?
              AND {moves_col} >= ?
        """
        cursor.execute(query, (
            board_type_str,
            self.config.num_players,
            self.config.min_game_length,
        ))
        games = cursor.fetchall()
        total_games = len(games)

        engine = DefaultRulesEngine()

        # Progress tracking
        import time as _time
        extraction_start = _time.time()

        for game_idx, game_row in enumerate(games):
            # Progress logging every 100 games
            if game_idx > 0 and game_idx % 100 == 0:
                elapsed = _time.time() - extraction_start
                rate = game_idx / elapsed if elapsed > 0 else 0
                eta = (total_games - game_idx) / rate if rate > 0 else 0
                logger.info(f"  Extraction progress: {game_idx}/{total_games} games ({rate:.1f}/s, ETA: {eta:.0f}s, {len(samples)} samples)")
            game_id = game_row['game_id']
            winner = game_row['winner']
            game_length = game_row['moves'] or 0

            # Get initial state
            cursor.execute(
                "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
                (game_id,)
            )
            initial_row = cursor.fetchone()
            if not initial_row:
                continue

            initial_json = initial_row['initial_state_json']
            if initial_row['compressed']:
                try:
                    if isinstance(initial_json, (bytes, bytearray)):
                        initial_json = gzip.decompress(bytes(initial_json)).decode("utf-8")
                    else:
                        initial_json = gzip.decompress(str(initial_json).encode("utf-8")).decode("utf-8")
                except Exception:
                    continue

            try:
                state_dict = json.loads(initial_json)
                state = GameState(**state_dict)
            except Exception:
                continue

            # Get all moves with phase information
            cursor.execute(
                "SELECT move_number, phase, move_type, move_json FROM game_moves WHERE game_id = ? ORDER BY move_number",
                (game_id,)
            )
            moves = cursor.fetchall()

            if not moves:
                continue

            # Replay and sample
            for move_row in moves:
                move_number = move_row['move_number']
                recorded_phase = move_row['phase']
                move_type = move_row['move_type']
                move_json_str = move_row['move_json']

                # Advance state to match the recorded phase
                # GPU selfplay records phases in a different order than CPU expects
                try:
                    state = _advance_to_phase(state, recorded_phase)
                except Exception:
                    break  # Can't continue if phase advancement fails

                # Skip bookkeeping moves for sampling (not decision points)
                is_bookkeeping = move_type and move_type.startswith('no_')

                # Sample every Nth position, but only for substantive moves
                if move_number % self.config.sample_every_n_moves == 0 and not is_bookkeeping:
                    current_player = state.current_player or 1

                    # Calculate value
                    is_winner = winner == current_player
                    if is_winner:
                        value = 1.0
                    elif winner is None or winner == 0:
                        value = 0.0
                    else:
                        value = -1.0

                    # Distillation filtering: only include winning players' positions
                    if self.config.distill_from_winners and not is_winner:
                        pass  # Skip non-winner positions but continue replay
                    elif value == 0.0 and not self.config.include_draws:
                        pass  # Skip draws but continue replay
                    elif move_number < self.config.min_move_number or move_number > self.config.max_move_number:
                        pass  # Curriculum: skip moves outside range
                    else:
                        # Get legal moves at this position
                        try:
                            legal_moves = engine.get_valid_moves(state, current_player)
                            if legal_moves:
                                # Parse the move that was played
                                move_dict = json.loads(move_json_str)
                                played_move = Move(**move_dict)

                                # Find which legal move matches the played move
                                target_idx = self._find_move_index(
                                    played_move, legal_moves
                                )

                                # Skip if target is beyond max_moves encoding limit
                                if target_idx >= 0 and target_idx < self.config.max_moves_per_position:
                                    # Extract features
                                    features = extract_features_from_gamestate(
                                        state, current_player
                                    )

                                    # Encode legal moves
                                    from_indices, to_indices, move_mask = self._encode_legal_moves(
                                        legal_moves
                                    )

                                    # Calculate sample weight
                                    sample_weight = 1.0

                                    # Apply game length weight (longer games = more signal)
                                    if self.config.weight_by_game_length and self.config.game_length_weight_cap > 0:
                                        sample_weight *= min(1.0, game_length / self.config.game_length_weight_cap)

                                    # Apply winner weight boost
                                    if is_winner and self.config.winner_weight_boost > 1.0:
                                        sample_weight *= self.config.winner_weight_boost

                                    # Extract MCTS policy distribution if available
                                    mcts_visit_dist = None
                                    mcts_policy_dict = move_dict.get('mcts_policy') or move_dict.get('mctsPolicy')
                                    if mcts_policy_dict and isinstance(mcts_policy_dict, dict):
                                        # Convert sparse dict to dense array
                                        mcts_visit_dist = np.zeros(self.config.max_moves_per_position, dtype=np.float32)
                                        for idx_str, prob in mcts_policy_dict.items():
                                            idx = int(idx_str)
                                            if 0 <= idx < self.config.max_moves_per_position:
                                                mcts_visit_dist[idx] = float(prob)
                                        # Renormalize to sum to 1 (in case of truncation)
                                        total = mcts_visit_dist.sum()
                                        if total > 0:
                                            mcts_visit_dist /= total

                                    sample = NNUEPolicySample(
                                        features=features,
                                        value=value,
                                        from_indices=from_indices,
                                        to_indices=to_indices,
                                        move_mask=move_mask,
                                        target_move_idx=target_idx,
                                        player_number=current_player,
                                        game_id=game_id,
                                        move_number=move_number,
                                        sample_weight=sample_weight,
                                        mcts_visit_distribution=mcts_visit_dist,
                                    )
                                    samples.append(sample)
                        except Exception as e:
                            logger.debug(f"Failed to process {game_id}:{move_number}: {e}")

                # Apply move to advance state
                try:
                    move_dict = json.loads(move_json_str)
                    move = Move(**move_dict)
                    state = engine.apply_move(state, move)
                except Exception:
                    break

                if self.max_samples and len(samples) >= self.max_samples:
                    conn.close()
                    return samples

        conn.close()
        # Final extraction summary
        total_time = _time.time() - extraction_start
        logger.info(f"  Extraction complete: {total_games} games in {total_time:.1f}s ({total_games/total_time:.1f}/s), {len(samples)} samples")
        return samples

    def _find_move_index(self, played_move, legal_moves: list) -> int:
        """Find the index of played_move in legal_moves.

        Matches by type, from_pos, and to positions.
        """
        played_type = getattr(played_move, 'type', None)
        played_from = getattr(played_move, 'from_pos', None)
        played_to = getattr(played_move, 'to', None)

        for i, legal in enumerate(legal_moves):
            legal_type = getattr(legal, 'type', None)
            legal_from = getattr(legal, 'from_pos', None)
            legal_to = getattr(legal, 'to', None)

            # Match by type
            if played_type != legal_type:
                continue

            # Match by from position (if both have one)
            if played_from is not None and legal_from is not None:
                if played_from.x != legal_from.x or played_from.y != legal_from.y:
                    continue
            elif played_from is not None or legal_from is not None:
                # One has from_pos and other doesn't
                continue

            # Match by to position (if both have one)
            if played_to is not None and legal_to is not None:
                if played_to.x != legal_to.x or played_to.y != legal_to.y:
                    continue
            elif played_to is not None or legal_to is not None:
                continue

            return i

        return -1  # Not found

    def _encode_legal_moves(
        self,
        legal_moves: list,
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Encode legal moves as numpy arrays."""
        import numpy as np

        max_moves = self.config.max_moves_per_position
        center = self.board_size // 2
        center_idx = center * self.board_size + center

        from_indices = np.zeros(max_moves, dtype=np.int64)
        to_indices = np.zeros(max_moves, dtype=np.int64)
        move_mask = np.zeros(max_moves, dtype=bool)

        for i, move in enumerate(legal_moves[:max_moves]):
            from_pos = getattr(move, 'from_pos', None)
            if from_pos is None:
                from_idx = center_idx
            else:
                from_idx = pos_to_flat_index(from_pos, self.board_size, self.config.board_type)

            to_pos = getattr(move, 'to', None)
            if to_pos is None:
                to_pos = from_pos
            if to_pos is None:
                to_idx = center_idx
            else:
                to_idx = pos_to_flat_index(to_pos, self.board_size, self.config.board_type)

            from_indices[i] = from_idx
            to_indices[i] = to_idx
            move_mask[i] = True

        return from_indices, to_indices, move_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a single sample as tensors.

        Returns:
            Tuple of (features, value, from_indices, to_indices, move_mask, target_idx, sample_weight, mcts_probs)
            Note: mcts_probs is zeros if not available for this sample
        """
        sample = self.samples[idx]
        # Return MCTS probs if available, otherwise zeros
        if sample.mcts_visit_distribution is not None:
            mcts_probs = torch.from_numpy(sample.mcts_visit_distribution).float()
        else:
            mcts_probs = torch.zeros(len(sample.move_mask), dtype=torch.float32)
        return (
            torch.from_numpy(sample.features).float(),
            torch.tensor([sample.value], dtype=torch.float32),
            torch.from_numpy(sample.from_indices).long(),
            torch.from_numpy(sample.to_indices).long(),
            torch.from_numpy(sample.move_mask).bool(),
            torch.tensor(sample.target_move_idx, dtype=torch.long),
            torch.tensor(sample.sample_weight, dtype=torch.float32),
            mcts_probs,
        )

    def get_mcts_policy_stats(self) -> dict[str, Any]:
        """Get statistics about MCTS policy availability in the dataset.

        Returns:
            Dict with keys:
                - total_samples: Total number of samples
                - samples_with_mcts: Number of samples with MCTS policy
                - mcts_coverage: Fraction of samples with MCTS policy (0.0-1.0)
                - recommend_kl_loss: Whether KL loss is recommended (coverage >= 0.5)
        """
        total = len(self.samples)
        with_mcts = sum(1 for s in self.samples if s.mcts_visit_distribution is not None)
        coverage = with_mcts / total if total > 0 else 0.0

        return {
            "total_samples": total,
            "samples_with_mcts": with_mcts,
            "mcts_coverage": coverage,
            "recommend_kl_loss": coverage >= 0.5,  # Recommend KL if >= 50% coverage
        }

    def should_use_kl_loss(self, min_coverage: float = 0.5, min_samples: int = 100) -> bool:
        """Determine if KL loss should be used based on MCTS policy availability.

        Args:
            min_coverage: Minimum fraction of samples with MCTS policy (default: 0.5)
            min_samples: Minimum number of samples with MCTS policy (default: 100)

        Returns:
            True if KL loss should be used, False otherwise
        """
        stats = self.get_mcts_policy_stats()
        return (
            stats["mcts_coverage"] >= min_coverage and
            stats["samples_with_mcts"] >= min_samples
        )


class NNUEPolicyAgent:
    """Agent that uses a trained NNUE policy network for move selection.

    This agent loads a trained RingRiftNNUEWithPolicy model and uses
    the policy head to score legal moves, returning the highest-scoring move.

    Usage:
        agent = NNUEPolicyAgent(
            model_path="models/ringrift_hex8_2p_v8.pth",
            board_type="hex8",
            num_players=2,
        )
        best_move = agent.get_best_move(state, legal_moves)
    """

    def __init__(
        self,
        model_path: str,
        board_type: str = "hex8",
        num_players: int = 2,
        device: str | None = None,
        temperature: float = 0.0,
    ):
        """Initialize the NNUE policy agent.

        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            board_type: Board type string (e.g., "hex8", "square8")
            num_players: Number of players in the game
            device: Device to run inference on (auto-detected if None)
            temperature: Softmax temperature for move selection (0.0 = greedy)
        """
        self.model_path = model_path
        self.board_type_str = board_type
        self.num_players = num_players
        self.temperature = temperature

        # Parse board type
        board_type_map = {
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
        }
        self.board_type = board_type_map.get(board_type.lower(), BoardType.HEX8)
        self.board_size = get_board_size(self.board_type)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> RingRiftNNUEWithPolicy:
        """Load the model from checkpoint."""
        from app.utils.torch_utils import safe_load_checkpoint
        logger = logging.getLogger(__name__)

        checkpoint = safe_load_checkpoint(self.model_path, map_location=str(self.device), warn_on_unsafe=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            # Assume it's a state dict directly
            state_dict = checkpoint

        # Infer model architecture from state_dict
        hidden_dim = None
        if "accumulator.weight" in state_dict:
            hidden_dim = state_dict["accumulator.weight"].shape[0]

        # Create model with inferred or default parameters
        model = RingRiftNNUEWithPolicy(
            board_type=self.board_type,
            hidden_dim=hidden_dim,
        )

        # Load weights (handle DataParallel prefix if present)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        logger.info(f"Loaded NNUE policy model from {self.model_path} (device={self.device})")
        return model

    def get_best_move(self, state, legal_moves):
        """Get the best move according to the policy network.

        Args:
            state: GameState or MutableGameState
            legal_moves: List of legal moves

        Returns:
            The highest-scoring legal move
        """
        if not legal_moves:
            return None

        # Extract features
        from .nnue import extract_features_from_gamestate, extract_features_from_mutable

        # Handle both GameState and MutableGameState
        if hasattr(state, "current_player"):
            player = state.current_player
        else:
            player = 0

        try:
            # Try MutableGameState first (faster)
            features = extract_features_from_mutable(state, player)
        except (AttributeError, TypeError):
            # Fall back to GameState
            features = extract_features_from_gamestate(state, player)

        # Encode legal moves
        max_moves = len(legal_moves)
        from_indices, to_indices, move_mask = _encode_legal_moves_static(
            legal_moves, self.board_size, max_moves, self.board_type
        )

        # Convert to tensors
        features_t = torch.from_numpy(features[None, ...]).float().to(self.device)
        from_t = torch.from_numpy(from_indices[None, ...]).long().to(self.device)
        to_t = torch.from_numpy(to_indices[None, ...]).long().to(self.device)
        mask_t = torch.from_numpy(move_mask[None, ...]).bool().to(self.device)

        # Score moves
        with torch.no_grad():
            if self.temperature > 0:
                probs = self.model.get_move_probabilities(
                    features_t, from_t, to_t, mask_t, temperature=self.temperature
                )
                # Sample from distribution
                move_idx = torch.multinomial(probs[0], 1).item()
            else:
                # Greedy selection
                scores = self.model.score_moves(features_t, from_t, to_t, mask_t)
                move_idx = torch.argmax(scores[0]).item()

        return legal_moves[move_idx]

    def get_move_scores(self, state, legal_moves) -> list[tuple]:
        """Get all moves with their policy scores.

        Args:
            state: GameState or MutableGameState
            legal_moves: List of legal moves

        Returns:
            List of (move, score) tuples sorted by score descending
        """
        if not legal_moves:
            return []

        from .nnue import extract_features_from_gamestate, extract_features_from_mutable

        if hasattr(state, "current_player"):
            player = state.current_player
        else:
            player = 0

        try:
            features = extract_features_from_mutable(state, player)
        except (AttributeError, TypeError):
            features = extract_features_from_gamestate(state, player)

        max_moves = len(legal_moves)
        from_indices, to_indices, move_mask = _encode_legal_moves_static(
            legal_moves, self.board_size, max_moves, self.board_type
        )

        features_t = torch.from_numpy(features[None, ...]).float().to(self.device)
        from_t = torch.from_numpy(from_indices[None, ...]).long().to(self.device)
        to_t = torch.from_numpy(to_indices[None, ...]).long().to(self.device)
        mask_t = torch.from_numpy(move_mask[None, ...]).bool().to(self.device)

        with torch.no_grad():
            scores = self.model.score_moves(features_t, from_t, to_t, mask_t)
            scores_np = scores[0].cpu().numpy()

        move_scores = [
            (legal_moves[i], float(scores_np[i]))
            for i in range(len(legal_moves))
        ]
        return sorted(move_scores, key=lambda x: x[1], reverse=True)
