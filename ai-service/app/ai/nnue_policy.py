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
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR

from ..models import BoardType, Position
from .nnue import (
    ClippedReLU,
    RingRiftNNUE,
    get_board_size,
    get_feature_dim,
)


def get_hidden_dim_for_board(board_type: BoardType, board_size: int = 0) -> int:
    """Auto-select hidden dimension based on board type and size.

    Args:
        board_type: The type of board
        board_size: For hexagonal boards, used to distinguish hex8 from full hex

    Returns:
        Recommended hidden dimension for the model

    Sizes:
        - Square8 (64 cells): 128 hidden
        - Hex8 (size <= 8): 256 hidden
        - Full hexagonal (size > 8): 1024 hidden
        - Square19 (361 cells): 512 hidden
    """
    if board_type == BoardType.SQUARE8:
        return 128  # 64 cells - smaller model
    elif board_type == BoardType.SQUARE19:
        return 512  # 361 cells - larger model
    elif board_type == BoardType.HEX8:
        return 256  # hex8 (61 cells) - medium model
    elif board_type == BoardType.HEXAGONAL:
        # Distinguish hex8 vs full hex by board_size
        if board_size <= 8:
            return 256  # hex8 - medium model
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
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.board_type = board_type
        self.board_size = get_board_size(board_type)

        # Auto-select hidden dimension if not specified
        if hidden_dim is None:
            hidden_dim = get_hidden_dim_for_board(board_type, self.board_size)

        input_dim = get_feature_dim(board_type)
        num_positions = self.board_size * self.board_size

        # Accumulator layer (same as RingRiftNNUE)
        self.accumulator = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden layers with ClippedReLU (same as RingRiftNNUE)
        layers: List[nn.Module] = []
        current_dim = hidden_dim * 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, 32))
            layers.append(ClippedReLU())
            current_dim = 32

        self.hidden = nn.Sequential(*layers)

        # Value head: single scalar output
        self.value_head = nn.Linear(32, 1)

        # Policy heads: from/to position logits
        self.policy_hidden = nn.Sequential(
            nn.Linear(32, 64),
            ClippedReLU(),
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
        move_mask: Optional[torch.Tensor] = None,
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
        move_mask: Optional[torch.Tensor] = None,
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
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        swa_lr: Optional[float] = None,
        progressive_batch_callback: Optional[Callable[[int, int], int]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.temperature = temperature
        self.initial_temperature = temperature
        self.label_smoothing = label_smoothing
        self.target_label_smoothing = label_smoothing
        self.label_smoothing_warmup = label_smoothing_warmup
        self.use_kl_loss = use_kl_loss
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
        self.history: Dict[str, List[float]] = {
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

    def get_swa_model(self) -> Optional[AveragedModel]:
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
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)

    def get_ema_model(self) -> Optional[RingRiftNNUEWithPolicy]:
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
        mcts_probs: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
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
                policy_loss_unreduced = torch.sum(
                    mcts_probs_safe * (torch.log(mcts_probs_safe) - log_policy), dim=-1
                )  # (batch,)
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

        # Backward pass with gradient scaling for AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        # Update EMA model
        self._update_ema()

        return total_loss.item(), value_loss.item(), policy_loss.item()

    def validate(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
        move_mask: torch.Tensor,
        target_move_idx: torch.Tensor,
        mcts_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float, float]:
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

            # Use KL divergence if enabled and MCTS probs available
            if self.use_kl_loss and mcts_probs is not None:
                log_policy = torch.log_softmax(move_scores, dim=-1)
                mcts_probs_safe = mcts_probs.clamp(min=1e-8)
                policy_loss = torch.nn.functional.kl_div(
                    log_policy, mcts_probs_safe, reduction='batchmean'
                )
            else:
                policy_loss = self.policy_criterion(move_scores, target_move_idx)

            # Policy accuracy (always use argmax for accuracy metric)
            pred_move_idx = move_scores.argmax(dim=-1)
            policy_accuracy = (pred_move_idx == target_move_idx).float().mean()

            total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss

        return total_loss.item(), value_loss.item(), policy_loss.item(), policy_accuracy.item()

    def update_scheduler(self, val_loss: float, epoch: Optional[int] = None) -> None:
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

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

    def _flat_to_axial(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to axial coordinates."""
        row = idx // self.board_size
        col = idx % self.board_size
        r = row - self.board_size // 2
        q = col - self.board_size // 2 - r // 2
        return q, r

    def _rotate_axial_60(self, q: int, r: int) -> Tuple[int, int]:
        """Rotate axial coordinates by 60° clockwise."""
        return -r, q + r

    def _reflect_axial(self, q: int, r: int) -> Tuple[int, int]:
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
    ) -> List[Tuple["np.ndarray", "np.ndarray", "np.ndarray", int]]:
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
from torch.utils.data import Dataset
from typing import Dict, Any

from .nnue import extract_features_from_gamestate

logger = logging.getLogger(__name__)


def _process_game_batch(
    db_path: str,
    game_ids: List[str],
    config_dict: Dict[str, Any],
    board_size: int,
) -> List[Dict[str, Any]]:
    """Worker function to process a batch of games in a separate process.

    Returns serialized sample dicts (not NNUEPolicySample objects) to avoid
    pickling issues with numpy arrays.
    """
    import numpy as np
    from ..models import GameState, Move, BoardType
    from ..rules.default_engine import DefaultRulesEngine

    # Reconstruct config from dict
    board_type = BoardType(config_dict['board_type'])
    sample_every_n = config_dict['sample_every_n_moves']
    min_game_length = config_dict['min_game_length']
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

            state_dict = json.loads(initial_json)
            state = GameState(**state_dict)

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
                    if distill_from_winners and not is_winner:
                        should_include = False
                    elif value == 0.0 and not include_draws:
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
        db_paths: List[str],
        config: Optional[NNUEPolicyDatasetConfig] = None,
        max_samples: Optional[int] = None,
        num_workers: int = 0,
        jsonl_paths: Optional[List[str]] = None,
    ):
        self.db_paths = db_paths
        self.jsonl_paths = jsonl_paths or []
        self.config = config or NNUEPolicyDatasetConfig()
        self.max_samples = max_samples
        self.num_workers = num_workers if num_workers > 0 else max(1, cpu_count() - 2)
        self.board_size = get_board_size(self.config.board_type)
        self.feature_dim = get_feature_dim(self.config.board_type)
        self.samples: List[NNUEPolicySample] = []

        self._extract_samples()

    def _extract_samples(self) -> None:
        """Extract training samples from SQLite databases and JSONL files.

        Uses parallel processing when num_workers > 1 for significantly faster
        extraction from large databases.
        """
        total_sources = len(self.db_paths) + len(self.jsonl_paths)
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
                logger.error(f"Failed to extract from JSONL {jsonl_path}: {e}")

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

    def _extract_from_jsonl(self, jsonl_path: str) -> List[NNUEPolicySample]:
        """Extract samples from a JSONL file containing game records with moves.

        JSONL files should contain one JSON game object per line with:
        - board_type: string (e.g., "square8")
        - num_players: int
        - winner: int (player number who won)
        - moves: list of move dicts, each potentially containing mcts_policy
        - initial_state: optional initial GameState dict

        This is particularly useful for MCTS selfplay data that includes
        mcts_policy distributions in the move records.
        """
        import numpy as np
        from ..models import GameState, Move, MoveType, Position
        from ..rules.default_engine import DefaultRulesEngine
        from ..training.generate_data import create_initial_state

        samples: List[NNUEPolicySample] = []
        engine = DefaultRulesEngine()
        board_type_str = self.config.board_type.value.lower()

        import time as _time
        extraction_start = _time.time()
        games_processed = 0
        games_skipped = 0

        with open(jsonl_path, 'r') as f:
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

                # Replay game and extract samples
                for move_idx, move_dict in enumerate(moves):
                    move_number = move_idx + 1

                    # Sample every Nth position
                    if move_number % self.config.sample_every_n_moves == 0 and state.game_status == "active":
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
                        if self.config.distill_from_winners and not is_winner:
                            skip_sample = True
                        elif value == 0.0 and not self.config.include_draws:
                            skip_sample = True
                        elif move_number < self.config.min_move_number or move_number > self.config.max_move_number:
                            skip_sample = True

                        if not skip_sample:
                            try:
                                # Get legal moves at this position
                                legal_moves = engine.get_valid_moves(state, current_player)
                                if legal_moves:
                                    # Parse the move that was played
                                    played_move = self._parse_jsonl_move(move_dict, move_number)
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
                        move = self._parse_jsonl_move(move_dict, move_idx)
                        if move is not None:
                            state = engine.apply_move(state, move)
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

    def _parse_jsonl_move(self, move_dict: dict, move_number: int) -> Optional["Move"]:
        """Parse a move dict from JSONL into a Move object."""
        from ..models import Move, MoveType, Position
        from datetime import datetime

        move_type_str = str(move_dict.get("type") or "").strip()
        if not move_type_str or move_type_str.startswith("unknown_"):
            return None

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            return None

        def parse_pos(pos_dict):
            if not pos_dict or not isinstance(pos_dict, dict):
                return None
            return Position(
                x=pos_dict.get("x", 0),
                y=pos_dict.get("y", 0),
                z=pos_dict.get("z"),
            )

        from_pos = parse_pos(move_dict.get("from") or move_dict.get("from_pos"))
        to_pos = parse_pos(move_dict.get("to"))
        capture_target = parse_pos(move_dict.get("capture_target") or move_dict.get("captureTarget"))

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

    def _extract_from_db_parallel(self, db_path: str) -> List[NNUEPolicySample]:
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

    def _extract_from_db(self, db_path: str) -> List[NNUEPolicySample]:
        """Extract samples from a single database via game replay."""
        from ..models import GameState, Move
        from ..rules.default_engine import DefaultRulesEngine

        samples: List[NNUEPolicySample] = []
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

                # Sample every Nth position
                if move_number % self.config.sample_every_n_moves == 0:
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
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
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

    def get_mcts_policy_stats(self) -> Dict[str, Any]:
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
