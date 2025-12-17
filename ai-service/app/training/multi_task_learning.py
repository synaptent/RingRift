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
from typing import Any, Dict, List, Optional, Tuple

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
    output_dim: Optional[int] = None


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
        state_shape: Tuple[int, ...],
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
        state_shape: Optional[Tuple[int, ...]] = None,
        config: Optional[MultiTaskConfig] = None,
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
    ) -> Dict[str, torch.Tensor]:
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


class MultiTaskLoss(nn.Module):
    """
    Computes combined loss for multi-task learning.

    Supports different task weighting strategies:
    - fixed: Use config weights directly
    - uncertainty: Learn task weights based on homoscedastic uncertainty
    - gradnorm: Gradient normalization (not implemented yet)
    """

    def __init__(
        self,
        config: Optional[MultiTaskConfig] = None,
    ):
        super().__init__()
        self.config = config or MultiTaskConfig()

        # Loss functions
        self.outcome_loss = nn.CrossEntropyLoss()
        self.legality_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()

    def forward(
        self,
        auxiliary_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        log_vars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined auxiliary task loss.

        Args:
            auxiliary_outputs: Dict of task outputs
            targets: Dict of task targets
            log_vars: Optional learned log-variance for uncertainty weighting

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = torch.tensor(0.0, device=next(iter(auxiliary_outputs.values())).device)
        loss_dict = {}
        task_idx = 0

        # Outcome prediction loss
        if 'outcome' in auxiliary_outputs and 'outcome' in targets:
            outcome_loss = self.outcome_loss(
                auxiliary_outputs['outcome'],
                targets['outcome'],
            )

            if self.config.task_weighting == "uncertainty" and log_vars is not None:
                precision = torch.exp(-log_vars[task_idx])
                outcome_loss = precision * outcome_loss + log_vars[task_idx]

            weight = self.config.outcome_prediction.weight
            total_loss = total_loss + weight * outcome_loss
            loss_dict['outcome_loss'] = outcome_loss.item()
            task_idx += 1

        # Legality prediction loss
        if 'legality' in auxiliary_outputs and 'legality' in targets:
            legality_loss = self.legality_loss(
                auxiliary_outputs['legality'],
                targets['legality'].float(),
            )

            if self.config.task_weighting == "uncertainty" and log_vars is not None:
                precision = torch.exp(-log_vars[task_idx])
                legality_loss = precision * legality_loss + log_vars[task_idx]

            weight = self.config.legality_prediction.weight
            total_loss = total_loss + weight * legality_loss
            loss_dict['legality_loss'] = legality_loss.item()
            task_idx += 1

        # State reconstruction loss
        if 'reconstruction' in auxiliary_outputs and 'reconstruction' in targets:
            recon_loss = self.reconstruction_loss(
                auxiliary_outputs['reconstruction'],
                targets['reconstruction'],
            )

            if self.config.task_weighting == "uncertainty" and log_vars is not None:
                precision = torch.exp(-log_vars[task_idx])
                recon_loss = precision * recon_loss + log_vars[task_idx]

            weight = self.config.state_reconstruction.weight
            total_loss = total_loss + weight * recon_loss
            loss_dict['reconstruction_loss'] = recon_loss.item()

        loss_dict['total_auxiliary_loss'] = total_loss.item()
        return total_loss, loss_dict


def create_auxiliary_targets(
    batch: Dict[str, torch.Tensor],
    num_classes: int = 3,
) -> Dict[str, torch.Tensor]:
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
        state_shape: Optional[Tuple[int, ...]] = None,
        config: Optional[MultiTaskConfig] = None,
        extract_features_fn: Optional[str] = None,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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
    config: Optional[MultiTaskConfig] = None,
    log_vars: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
