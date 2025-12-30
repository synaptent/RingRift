"""
Multi-Task Learning for RingRift AI.

Trains the neural network on multiple auxiliary tasks alongside
the main policy and value predictions:
- Game outcome prediction
- Move legality prediction
- Board state reconstruction

This improves representation learning and can lead to better
generalization.

Usage:
    from app.training.multi_task_learning import (
        MultiTaskHead,
        MultiTaskLoss,
        create_multi_task_model,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for an auxiliary task."""
    name: str
    weight: float = 1.0
    enabled: bool = True
    hidden_dim: int = 256
    output_dim: int | None = None


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""
    policy_weight: float = 1.0
    value_weight: float = 1.0

    # Auxiliary tasks
    outcome_prediction: TaskConfig = field(
        default_factory=lambda: TaskConfig(
            name="outcome",
            weight=0.1,
            output_dim=3,  # win/draw/loss
        )
    )
    legality_prediction: TaskConfig = field(
        default_factory=lambda: TaskConfig(
            name="legality",
            weight=0.05,
            output_dim=None,  # Same as policy
        )
    )
    state_reconstruction: TaskConfig = field(
        default_factory=lambda: TaskConfig(
            name="reconstruction",
            weight=0.05,
            enabled=False,  # Disabled by default (expensive)
        )
    )

    # Learning settings
    task_weighting: str = "fixed"  # "fixed", "uncertainty", "gradnorm"
    uncertainty_temp: float = 1.0


class OutcomePredictionHead(nn.Module):
    """
    Auxiliary task: Predict final game outcome from current position.

    Predicts win/draw/loss probability for the current player.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Shared representation from backbone

        Returns:
            Logits for win/draw/loss
        """
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)


class LegalityPredictionHead(nn.Module):
    """
    Auxiliary task: Predict which moves are legal.

    Binary classification for each possible move.
    """

    def __init__(
        self,
        input_dim: int,
        policy_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, policy_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Shared representation from backbone

        Returns:
            Logits for each move being legal
        """
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)


class StateReconstructionHead(nn.Module):
    """
    Auxiliary task: Reconstruct input state from hidden representation.

    Auto-encoder style task that encourages learning useful features.
    """

    def __init__(
        self,
        input_dim: int,
        state_shape: tuple[int, ...],
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.state_shape = state_shape
        output_size = 1
        for dim in state_shape:
            output_size *= dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Shared representation from backbone

        Returns:
            Reconstructed state tensor
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, *self.state_shape)


class MultiTaskHead(nn.Module):
    """
    Multi-task head that wraps auxiliary task predictions.

    Attaches to a backbone model and adds auxiliary task outputs.
    """

    def __init__(
        self,
        backbone_output_dim: int,
        policy_size: int,
        state_shape: tuple[int, ...] | None = None,
        config: MultiTaskConfig | None = None,
    ):
        super().__init__()
        self.config = config or MultiTaskConfig()

        self.auxiliary_heads = nn.ModuleDict()

        # Outcome prediction
        if self.config.outcome_prediction.enabled:
            self.auxiliary_heads['outcome'] = OutcomePredictionHead(
                input_dim=backbone_output_dim,
                hidden_dim=self.config.outcome_prediction.hidden_dim,
                num_classes=self.config.outcome_prediction.output_dim or 3,
            )

        # Legality prediction
        if self.config.legality_prediction.enabled:
            self.auxiliary_heads['legality'] = LegalityPredictionHead(
                input_dim=backbone_output_dim,
                policy_size=policy_size,
                hidden_dim=self.config.legality_prediction.hidden_dim,
            )

        # State reconstruction
        if self.config.state_reconstruction.enabled and state_shape is not None:
            self.auxiliary_heads['reconstruction'] = StateReconstructionHead(
                input_dim=backbone_output_dim,
                state_shape=state_shape,
                hidden_dim=self.config.state_reconstruction.hidden_dim,
            )

        # Learnable task weights for uncertainty weighting
        if self.config.task_weighting == "uncertainty":
            num_tasks = len(self.auxiliary_heads) + 2  # +2 for policy and value
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute auxiliary task outputs.

        Args:
            features: Shared representation from backbone

        Returns:
            Dictionary of task name -> output tensor
        """
        outputs = {}

        for name, head in self.auxiliary_heads.items():
            outputs[name] = head(features)

        return outputs


class GradNormWeighter(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Implements the GradNorm algorithm from "GradNorm: Gradient Normalization
    for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., 2018).

    The algorithm dynamically adjusts task weights to balance gradient magnitudes,
    ensuring no single task dominates training while prioritizing tasks that are
    learning slower relative to others.
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        initial_weights: list[float] | None = None,
    ):
        """
        Args:
            num_tasks: Number of tasks to balance
            alpha: Asymmetry parameter (higher = more aggressive rebalancing)
                   - alpha=0: Equal gradient norms (ignores learning rates)
                   - alpha=1.5: Standard GradNorm (recommended)
                   - alpha>2: Very aggressive rebalancing
            initial_weights: Initial task weights (default: equal weights)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Learnable task weights (stored as log for positivity)
        if initial_weights is not None:
            init_log_weights = torch.log(torch.tensor(initial_weights, dtype=torch.float32))
        else:
            init_log_weights = torch.zeros(num_tasks)
        self.log_weights = nn.Parameter(init_log_weights)

        # Track initial losses for relative training rate computation
        self.register_buffer('initial_losses', torch.ones(num_tasks))
        self.register_buffer('loss_ratios', torch.ones(num_tasks))
        self.initialized = False

    @property
    def weights(self) -> torch.Tensor:
        """Get current task weights (normalized to sum to num_tasks)."""
        raw_weights = torch.exp(self.log_weights)
        return raw_weights / raw_weights.sum() * self.num_tasks

    def initialize_losses(self, losses: torch.Tensor):
        """Record initial losses for training rate computation."""
        if not self.initialized:
            self.initial_losses.copy_(losses.detach())
            self.initialized = True

    def compute_loss_ratios(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute relative inverse training rate for each task."""
        # L_i(t) / L_i(0) - how much each task has improved
        ratios = losses.detach() / (self.initial_losses + 1e-8)
        # Normalize by mean ratio
        mean_ratio = ratios.mean()
        # Inverse training rate: tasks with higher ratio are learning slower
        inverse_train_rate = ratios / (mean_ratio + 1e-8)
        self.loss_ratios.copy_(inverse_train_rate)
        return inverse_train_rate

    def compute_grad_norm_loss(
        self,
        task_losses: list[torch.Tensor],
        shared_params: nn.Parameter,
    ) -> torch.Tensor:
        """
        Compute GradNorm loss for weight updates.

        Args:
            task_losses: List of individual task losses
            shared_params: Shared network parameters (last shared layer)

        Returns:
            GradNorm loss to minimize for weight optimization
        """
        losses = torch.stack(task_losses)

        # Initialize on first call
        self.initialize_losses(losses)

        # Compute weighted losses
        weights = self.weights
        weighted_losses = weights * losses

        # Compute gradient norms for each weighted task loss
        grad_norms = []
        for _i, wl in enumerate(weighted_losses):
            if shared_params.grad is not None:
                shared_params.grad.zero_()

            # Compute gradient of weighted loss w.r.t. shared params
            grad = torch.autograd.grad(
                wl, shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]

            if grad is not None:
                grad_norms.append(grad.norm())
            else:
                grad_norms.append(torch.tensor(0.0, device=losses.device))

        grad_norms = torch.stack(grad_norms)

        # Target gradient norm based on relative training rates
        mean_grad_norm = grad_norms.mean().detach()
        inverse_train_rate = self.compute_loss_ratios(losses)
        target_grad_norms = mean_grad_norm * (inverse_train_rate ** self.alpha)

        # GradNorm loss: minimize difference between actual and target grad norms
        gradnorm_loss = (grad_norms - target_grad_norms).abs().sum()

        return gradnorm_loss

    def get_stats(self) -> dict[str, float]:
        """Get current weight statistics for logging."""
        weights = self.weights.detach().cpu().numpy()
        ratios = self.loss_ratios.detach().cpu().numpy()
        return {
            f'task_{i}_weight': float(weights[i])
            for i in range(self.num_tasks)
        } | {
            f'task_{i}_train_rate': float(1.0 / (ratios[i] + 1e-8))
            for i in range(self.num_tasks)
        }


class MultiTaskLoss(nn.Module):
    """
    Computes combined loss for multi-task learning.

    Supports different task weighting strategies:
    - fixed: Use config weights directly
    - uncertainty: Learn task weights based on homoscedastic uncertainty
    - gradnorm: Gradient normalization for adaptive loss balancing
    """

    def __init__(
        self,
        config: MultiTaskConfig | None = None,
        shared_layer: nn.Parameter | None = None,
    ):
        super().__init__()
        self.config = config or MultiTaskConfig()
        self.shared_layer = shared_layer

        # Loss functions
        self.outcome_loss = nn.CrossEntropyLoss()
        self.legality_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()

        # GradNorm weighter (initialized lazily based on actual tasks)
        self._gradnorm_weighter: GradNormWeighter | None = None
        self._task_names: list[str] = []

    def _ensure_gradnorm_initialized(self, task_names: list[str], device: torch.device):
        """Initialize GradNorm weighter if needed."""
        if self._gradnorm_weighter is None or self._task_names != task_names:
            self._task_names = task_names
            initial_weights = []
            for name in task_names:
                if name == 'outcome':
                    initial_weights.append(self.config.outcome_prediction.weight)
                elif name == 'legality':
                    initial_weights.append(self.config.legality_prediction.weight)
                elif name == 'reconstruction':
                    initial_weights.append(self.config.state_reconstruction.weight)
                else:
                    initial_weights.append(1.0)

            self._gradnorm_weighter = GradNormWeighter(
                num_tasks=len(task_names),
                alpha=1.5,  # Standard GradNorm alpha
                initial_weights=initial_weights,
            ).to(device)

    def get_gradnorm_weighter(self) -> GradNormWeighter | None:
        """Get the GradNorm weighter for optimizer access."""
        return self._gradnorm_weighter

    def forward(
        self,
        auxiliary_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        log_vars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined auxiliary task loss.

        Args:
            auxiliary_outputs: Dict of task outputs
            targets: Dict of task targets
            log_vars: Optional learned log-variance for uncertainty weighting

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Handle empty outputs early to avoid StopIteration
        if not auxiliary_outputs:
            return torch.tensor(0.0), {'total_auxiliary_loss': 0.0}

        device = next(iter(auxiliary_outputs.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        task_losses = []
        task_names = []

        # Compute individual task losses
        if 'outcome' in auxiliary_outputs and 'outcome' in targets:
            outcome_loss = self.outcome_loss(
                auxiliary_outputs['outcome'],
                targets['outcome'],
            )
            task_losses.append(outcome_loss)
            task_names.append('outcome')
            loss_dict['outcome_loss'] = outcome_loss.item()

        if 'legality' in auxiliary_outputs and 'legality' in targets:
            legality_loss = self.legality_loss(
                auxiliary_outputs['legality'],
                targets['legality'].float(),
            )
            task_losses.append(legality_loss)
            task_names.append('legality')
            loss_dict['legality_loss'] = legality_loss.item()

        if 'reconstruction' in auxiliary_outputs and 'reconstruction' in targets:
            recon_loss = self.reconstruction_loss(
                auxiliary_outputs['reconstruction'],
                targets['reconstruction'],
            )
            task_losses.append(recon_loss)
            task_names.append('reconstruction')
            loss_dict['reconstruction_loss'] = recon_loss.item()

        if not task_losses:
            loss_dict['total_auxiliary_loss'] = 0.0
            return total_loss, loss_dict

        # Apply weighting strategy
        if self.config.task_weighting == "gradnorm":
            # GradNorm: use learned adaptive weights
            self._ensure_gradnorm_initialized(task_names, device)
            weights = self._gradnorm_weighter.weights

            for i, (loss, name) in enumerate(zip(task_losses, task_names, strict=False)):
                total_loss = total_loss + weights[i] * loss
                loss_dict[f'{name}_weight'] = weights[i].item()

            # Add GradNorm stats to loss dict
            if self._gradnorm_weighter is not None:
                loss_dict.update(self._gradnorm_weighter.get_stats())

        elif self.config.task_weighting == "uncertainty" and log_vars is not None:
            # Uncertainty weighting: learn precision for each task
            for i, (loss, _name) in enumerate(zip(task_losses, task_names, strict=False)):
                precision = torch.exp(-log_vars[i])
                weighted_loss = precision * loss + log_vars[i]
                total_loss = total_loss + weighted_loss

        else:
            # Fixed weighting: use config weights
            for loss, name in zip(task_losses, task_names, strict=False):
                if name == 'outcome':
                    weight = self.config.outcome_prediction.weight
                elif name == 'legality':
                    weight = self.config.legality_prediction.weight
                elif name == 'reconstruction':
                    weight = self.config.state_reconstruction.weight
                else:
                    weight = 1.0
                total_loss = total_loss + weight * loss

        loss_dict['total_auxiliary_loss'] = total_loss.item()
        return total_loss, loss_dict

    def compute_gradnorm_loss(
        self,
        task_losses: list[torch.Tensor],
        shared_params: nn.Parameter,
    ) -> torch.Tensor | None:
        """
        Compute GradNorm loss for weight optimization.

        Call this separately and add to weight optimizer's loss.

        Args:
            task_losses: List of unweighted task losses
            shared_params: Last shared layer parameters

        Returns:
            GradNorm loss if using gradnorm weighting, else None
        """
        if self.config.task_weighting != "gradnorm" or self._gradnorm_weighter is None:
            return None

        return self._gradnorm_weighter.compute_grad_norm_loss(task_losses, shared_params)


def create_auxiliary_targets(
    batch: dict[str, torch.Tensor],
    num_classes: int = 3,
) -> dict[str, torch.Tensor]:
    """
    Create auxiliary task targets from a training batch.

    Args:
        batch: Training batch with features, policy, value
        num_classes: Number of outcome classes

    Returns:
        Dictionary of auxiliary task targets
    """
    targets = {}

    # Outcome target from value (convert to class index)
    if 'values' in batch:
        values = batch['values']
        # Map [-1, 1] to [0, 1, 2] (loss, draw, win)
        outcome_classes = torch.clamp(
            (values + 1).long(),
            min=0,
            max=num_classes - 1,
        )
        targets['outcome'] = outcome_classes

    # Legality target from policy (non-zero entries are legal)
    if 'policy' in batch:
        policy = batch['policy']
        # Any non-zero probability indicates a legal move
        targets['legality'] = (policy > 0).float()

    # Reconstruction target is the input features
    if 'features' in batch:
        targets['reconstruction'] = batch['features']

    return targets


class MultiTaskModelWrapper(nn.Module):
    """
    Wraps a backbone model with multi-task heads.

    Usage:
        backbone = RingRiftCNN_v2(...)
        multi_task_model = MultiTaskModelWrapper(
            backbone,
            backbone_output_dim=256,
            policy_size=policy_size,
        )

        policy, value, auxiliary = multi_task_model(features)
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_output_dim: int,
        policy_size: int,
        state_shape: tuple[int, ...] | None = None,
        config: MultiTaskConfig | None = None,
        extract_features_fn: str | None = None,
    ):
        """
        Args:
            backbone: Base model with policy and value heads
            backbone_output_dim: Output dimension of backbone features
            policy_size: Size of policy output
            state_shape: Shape of state for reconstruction task
            config: Multi-task configuration
            extract_features_fn: Name of method to extract features from backbone
        """
        super().__init__()
        self.backbone = backbone
        self.extract_features_fn = extract_features_fn

        self.multi_task_head = MultiTaskHead(
            backbone_output_dim=backbone_output_dim,
            policy_size=policy_size,
            state_shape=state_shape,
            config=config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass with multi-task outputs.

        Args:
            x: Input tensor

        Returns:
            Tuple of (policy, value, auxiliary_outputs)
        """
        # Get backbone outputs
        backbone_output = self.backbone(x)

        if isinstance(backbone_output, tuple):
            policy, value = backbone_output[:2]
            # Try to get features if returned
            if len(backbone_output) > 2:
                features = backbone_output[2]
            else:
                features = None
        else:
            policy = backbone_output
            value = None
            features = None

        # Extract features if not returned and method specified
        if features is None and self.extract_features_fn:
            features = getattr(self.backbone, self.extract_features_fn)(x)

        # If still no features, use global average pooled conv output
        if features is None:
            # Fallback: run through backbone and global pool
            # This is a simplified version - actual implementation depends on backbone
            features = policy.view(policy.size(0), -1)

        # Get auxiliary outputs
        auxiliary = self.multi_task_head(features)

        return policy, value, auxiliary


def integrate_multi_task_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    config: MultiTaskConfig | None = None,
    log_vars: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combine primary and auxiliary losses.

    Args:
        policy_loss: Policy head loss
        value_loss: Value head loss
        auxiliary_loss: Combined auxiliary task loss
        config: Multi-task configuration
        log_vars: Learned log-variance for uncertainty weighting

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    config = config or MultiTaskConfig()
    loss_dict = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'auxiliary_loss': auxiliary_loss.item(),
    }

    if config.task_weighting == "uncertainty" and log_vars is not None:
        # Uncertainty weighting for main tasks
        policy_precision = torch.exp(-log_vars[-2])
        value_precision = torch.exp(-log_vars[-1])

        weighted_policy = policy_precision * policy_loss + log_vars[-2]
        weighted_value = value_precision * value_loss + log_vars[-1]

        total_loss = weighted_policy + weighted_value + auxiliary_loss
    else:
        # Fixed weighting
        total_loss = (
            config.policy_weight * policy_loss +
            config.value_weight * value_loss +
            auxiliary_loss
        )

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict
