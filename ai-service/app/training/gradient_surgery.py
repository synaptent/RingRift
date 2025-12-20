"""Gradient Surgery for Multi-Task Learning.

Handles conflicting gradients between value and policy heads by projecting
conflicting gradients to a neutral direction, preventing oscillation.

Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class GradientSurgeryConfig:
    """Configuration for gradient surgery."""
    enabled: bool = True
    method: str = "pcgrad"  # "pcgrad" or "cagrad"
    conflict_threshold: float = 0.0  # Angle threshold for conflict detection
    alpha: float = 0.5  # Blending factor for CAGrad


class GradientSurgeon:
    """Applies gradient surgery to handle conflicting task gradients."""

    def __init__(self, config: Optional[GradientSurgeryConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        self.config = config or GradientSurgeryConfig()
        self._grad_cache: Dict[str, torch.Tensor] = {}

    def compute_task_gradients(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for each task separately.

        Args:
            model: The model
            losses: Dictionary of task name -> loss tensor

        Returns:
            Dictionary of task name -> flattened gradient vector
        """
        task_grads = {}

        for task_name, loss in losses.items():
            model.zero_grad()
            loss.backward(retain_graph=True)

            # Flatten all gradients into single vector
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1).clone())

            if grads:
                task_grads[task_name] = torch.cat(grads)

        return task_grads

    def project_conflicting_gradients(
        self,
        grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply PCGrad: project conflicting gradients.

        Args:
            grads: Task gradients

        Returns:
            Projected gradients
        """
        task_names = list(grads.keys())
        if len(task_names) < 2:
            return grads

        projected = {name: grad.clone() for name, grad in grads.items()}

        for i, name_i in enumerate(task_names):
            for j, name_j in enumerate(task_names):
                if i == j:
                    continue

                grad_i = projected[name_i]
                grad_j = grads[name_j]

                # Check for conflict (negative inner product)
                dot_product = torch.dot(grad_i, grad_j)

                if dot_product < self.config.conflict_threshold:
                    # Project grad_i onto normal of grad_j
                    grad_j_norm_sq = torch.dot(grad_j, grad_j)
                    if grad_j_norm_sq > 1e-8:
                        projection = (dot_product / grad_j_norm_sq) * grad_j
                        projected[name_i] = grad_i - projection

        return projected

    def apply_surgery(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply gradient surgery and update model gradients.

        Args:
            model: The model
            losses: Dictionary of task losses

        Returns:
            Combined loss value
        """
        if not self.config.enabled:
            # Simple sum without surgery (losses must be non-empty)
            total_loss: torch.Tensor = sum(losses.values())  # type: ignore[assignment]
            model.zero_grad()
            total_loss.backward()
            return total_loss

        # Compute task gradients
        task_grads = self.compute_task_gradients(model, losses)

        # Apply gradient surgery
        if self.config.method == "pcgrad":
            projected_grads = self.project_conflicting_gradients(task_grads)
        else:
            projected_grads = task_grads

        # Average projected gradients
        combined_grad = sum(projected_grads.values()) / len(projected_grads)

        # Apply combined gradient to model
        model.zero_grad()
        idx = 0
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            numel = param.numel()
            param.grad.copy_(combined_grad[idx:idx + numel].view_as(param))
            idx += numel

        combined_loss: torch.Tensor = sum(losses.values())  # type: ignore[assignment]
        return combined_loss

    def check_gradient_conflict(
        self,
        grad1: torch.Tensor,
        grad2: torch.Tensor,
    ) -> Tuple[bool, float]:
        """Check if two gradients are in conflict.

        Args:
            grad1: First gradient vector
            grad2: Second gradient vector

        Returns:
            (is_conflict, cosine_similarity)
        """
        dot = torch.dot(grad1, grad2)
        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return False, 0.0

        cosine = dot / (norm1 * norm2)
        is_conflict = cosine < self.config.conflict_threshold

        return is_conflict, cosine.item()


class MultiTaskLossBalancer:
    """Balances losses across tasks dynamically."""

    def __init__(
        self,
        task_names: List[str],
        method: str = "uncertainty",
    ):
        """Initialize loss balancer.

        Args:
            task_names: Names of tasks
            method: Balancing method ("uncertainty", "gradnorm", "fixed")
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")

        self.task_names = task_names
        self.method = method

        # Learnable log-variance for uncertainty weighting
        self.log_vars = {name: torch.nn.Parameter(torch.zeros(1)) for name in task_names}

    def balance_losses(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute balanced total loss.

        Args:
            losses: Dictionary of task losses

        Returns:
            Balanced total loss
        """
        if self.method == "fixed":
            return sum(losses.values())

        elif self.method == "uncertainty":
            total = torch.tensor(0.0, device=next(iter(losses.values())).device)
            for name, loss in losses.items():
                precision = torch.exp(-self.log_vars[name])
                total = total + precision * loss + self.log_vars[name]
            return total

        else:
            return sum(losses.values())


def create_gradient_surgeon(method: str = "pcgrad") -> GradientSurgeon:
    """Create a gradient surgeon."""
    config = GradientSurgeryConfig(method=method)
    return GradientSurgeon(config)
