"""Model Distillation Pipeline for RingRift AI.

Knowledge distillation compresses a large teacher model (or ensemble)
into a smaller, faster student model while preserving performance.

Use cases:
1. Compress ensemble into single model for faster inference
2. Create lightweight models for edge deployment
3. Transfer knowledge from multiple training runs

Usage:
    from app.training.distillation import DistillationTrainer, DistillationConfig

    config = DistillationConfig(
        temperature=3.0,
        alpha=0.7,  # Weight of soft targets vs hard targets
    )

    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        config=config,
    )

    for batch in dataloader:
        loss = trainer.train_step(batch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 3.0  # Softmax temperature for soft targets
    alpha: float = 0.7  # Weight of soft targets (1-alpha = hard targets)
    value_temperature: float = 1.0  # Temperature for value head
    policy_temperature: float = 3.0  # Temperature for policy head
    use_attention_transfer: bool = False  # Transfer attention maps
    attention_beta: float = 0.1  # Weight of attention loss
    use_hint_loss: bool = False  # Use intermediate layer hints
    hint_layers: list[str] = None  # Layer names for hint loss
    hint_beta: float = 0.5  # Weight of hint loss


class SoftTargetLoss(nn.Module):
    """Soft target cross-entropy loss for distillation."""

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft target loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits

        Returns:
            Soft target KL divergence loss
        """
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
        return loss


class DistillationTrainer:
    """Trains a student model to mimic a teacher model.

    Combines soft targets from teacher with hard ground truth labels.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | None = None,
    ):
        """Initialize distillation trainer.

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            config: Distillation configuration
            optimizer: Optimizer for student model
            device: Training device
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")

        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        # Teacher in eval mode (no gradients)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Loss functions
        self.soft_loss = SoftTargetLoss(config.temperature)
        self.hard_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

        # Statistics
        self.step_count = 0
        self.total_loss = 0.0
        self.soft_loss_sum = 0.0
        self.hard_loss_sum = 0.0

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        value_targets: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform a single distillation training step.

        Args:
            inputs: Input batch
            targets: Hard targets (optional)
            value_targets: Value targets (optional)

        Returns:
            Dictionary of loss values
        """
        self.student.train()
        inputs = inputs.to(self.device)

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(inputs)
            if isinstance(teacher_output, tuple):
                teacher_value, teacher_policy = teacher_output[:2]
            else:
                teacher_value = teacher_output
                teacher_policy = None

        # Get student outputs
        student_output = self.student(inputs)
        if isinstance(student_output, tuple):
            student_value, student_policy = student_output[:2]
        else:
            student_value = student_output
            student_policy = None

        # Compute losses
        losses = {}

        # Value distillation loss
        value_distill_loss = F.mse_loss(
            student_value / self.config.value_temperature,
            teacher_value / self.config.value_temperature,
        ) * (self.config.value_temperature ** 2)
        losses["value_distill"] = value_distill_loss.item()

        total_loss = value_distill_loss

        # Policy distillation loss
        if student_policy is not None and teacher_policy is not None:
            policy_distill_loss = self.soft_loss(student_policy, teacher_policy)
            losses["policy_distill"] = policy_distill_loss.item()
            total_loss = total_loss + policy_distill_loss

        # Hard target loss (if targets provided)
        if targets is not None:
            targets = targets.to(self.device)
            if student_policy is not None:
                hard_loss = self.hard_loss(student_policy, targets)
            else:
                hard_loss = F.mse_loss(student_value.squeeze(), targets.float())
            losses["hard"] = hard_loss.item()
            total_loss = self.config.alpha * total_loss + (1 - self.config.alpha) * hard_loss

        # Value hard target loss
        if value_targets is not None:
            value_targets = value_targets.to(self.device)
            value_hard_loss = F.mse_loss(student_value.squeeze(), value_targets)
            losses["value_hard"] = value_hard_loss.item()
            total_loss = total_loss + (1 - self.config.alpha) * value_hard_loss

        losses["total"] = total_loss.item()

        # Backward pass
        if self.optimizer:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.step_count += 1
        self.total_loss += total_loss.item()

        return losses

    def get_stats(self) -> dict[str, float]:
        """Get training statistics."""
        return {
            "steps": self.step_count,
            "avg_loss": self.total_loss / max(1, self.step_count),
        }


class EnsembleTeacher:
    """Combines multiple models into a single teacher for distillation."""

    def __init__(
        self,
        models: list[nn.Module],
        weights: list[float] | None = None,
        device: torch.device | None = None,
    ):
        """Initialize ensemble teacher.

        Args:
            models: List of teacher models
            weights: Optional weights for each model (default: uniform)
            device: Training device
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")

        self.models = models
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # Move models to device and set eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get weighted ensemble predictions."""
        inputs = inputs.to(self.device)

        values = []
        policies = []

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights, strict=False):
                output = model(inputs)
                if isinstance(output, tuple):
                    v, p = output[:2]
                    values.append(v * weight)
                    policies.append(p * weight)
                else:
                    values.append(output * weight)

        # Weighted average
        ensemble_value = torch.stack(values, dim=0).sum(dim=0)

        if policies:
            ensemble_policy = torch.stack(policies, dim=0).sum(dim=0)
        else:
            ensemble_policy = None

        return ensemble_value, ensemble_policy

    def __call__(self, inputs: torch.Tensor):
        return self.forward(inputs)


def create_distillation_trainer(
    teacher: Union[nn.Module, list[nn.Module]],
    student: nn.Module,
    temperature: float = 3.0,
    alpha: float = 0.7,
    learning_rate: float = 1e-4,
) -> DistillationTrainer:
    """Factory function to create a distillation trainer.

    Args:
        teacher: Teacher model or list of models for ensemble
        student: Student model to train
        temperature: Softmax temperature
        alpha: Weight of soft targets
        learning_rate: Learning rate for student

    Returns:
        Configured DistillationTrainer
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    # Create ensemble if multiple teachers
    if isinstance(teacher, list):
        teacher = EnsembleTeacher(teacher)

    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    return DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        config=config,
        optimizer=optimizer,
    )


def distill_checkpoint_ensemble(
    checkpoint_paths: list[Path],
    student_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 10,
    temperature: float = 3.0,
    learning_rate: float = 1e-4,
) -> nn.Module:
    """Distill knowledge from multiple checkpoints into a single model.

    Args:
        checkpoint_paths: Paths to teacher model checkpoints
        student_model: Student model architecture
        dataloader: Training data
        epochs: Number of training epochs
        temperature: Distillation temperature
        learning_rate: Learning rate

    Returns:
        Trained student model
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    # Load teacher models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = []

    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device)
        # Assumes checkpoint has model_state_dict
        teacher = type(student_model)()  # Create same architecture
        teacher.load_state_dict(checkpoint["model_state_dict"])
        teachers.append(teacher)

    logger.info(f"[Distillation] Loaded {len(teachers)} teacher models")

    # Create ensemble teacher
    ensemble = EnsembleTeacher(teachers, device=device)

    # Create trainer
    config = DistillationConfig(temperature=temperature)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

    trainer = DistillationTrainer(
        teacher_model=ensemble,
        student_model=student_model,
        config=config,
        optimizer=optimizer,
        device=device,
    )

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
                value_targets = batch[2] if len(batch) > 2 else None
            else:
                inputs = batch
                targets = None
                value_targets = None

            losses = trainer.train_step(inputs, targets, value_targets)
            epoch_loss += losses["total"]
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        logger.info(f"[Distillation] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return student_model
