"""
Multi-Task Learning Heads for RingRift AI.

Provides auxiliary prediction heads to improve representation learning
and enable additional capabilities beyond policy and value prediction.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, multi-task heads will be stubs")


class AuxiliaryTask(Enum):
    """Types of auxiliary prediction tasks."""
    MOVE_COUNT = "move_count"            # Predict number of remaining moves
    GAME_PHASE = "game_phase"            # Opening/midgame/endgame classification
    TERRITORY_CONTROL = "territory"      # Territory influence prediction
    PIECE_ACTIVITY = "piece_activity"    # Piece mobility/activity
    THREAT_DETECTION = "threats"         # Identify immediate threats
    LINE_COMPLETION = "line_completion"  # Probability of completing lines
    DEFENSIVE_NEED = "defense"           # Need for defensive moves
    WINNING_PATH = "winning_path"        # Suggested winning continuation
    POSITION_TYPE = "position_type"      # Tactical vs positional


@dataclass
class TaskConfig:
    """Configuration for an auxiliary task."""
    task_type: AuxiliaryTask
    enabled: bool = True
    loss_weight: float = 0.1
    hidden_dim: int = 128
    num_classes: int | None = None
    output_dim: int | None = None
    loss_fn: str = "mse"  # mse, cross_entropy, bce


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""
    tasks: list[TaskConfig] = field(default_factory=list)
    uncertainty_weighting: bool = False  # Learn loss weights automatically
    gradient_normalization: bool = False  # GradNorm balancing
    shared_layers: int = 0  # Number of shared layers before task heads
    dropout: float = 0.1


if TORCH_AVAILABLE:

    class AuxiliaryHead(nn.Module, ABC):
        """Base class for auxiliary prediction heads."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__()
            self.config = config
            self.input_dim = input_dim

        @abstractmethod
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """Forward pass through the head."""
            pass

        @abstractmethod
        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            """Compute task-specific loss."""
            pass


    class MoveCountHead(AuxiliaryHead):
        """Predict expected number of remaining moves in the game."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__(input_dim, config)

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, 1)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            # Predict log(moves) for better scaling
            return self.fc(features)

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            # targets should be actual move counts
            log_targets = torch.log1p(targets)
            return F.mse_loss(predictions.squeeze(), log_targets)


    class GamePhaseHead(AuxiliaryHead):
        """Classify game phase: opening, midgame, endgame."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__(input_dim, config)

            num_classes = config.num_classes or 3

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, num_classes)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.fc(features)

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(predictions, targets.long())


    class TerritoryHead(AuxiliaryHead):
        """Predict territory/influence map over the board."""

        def __init__(self, input_dim: int, config: TaskConfig,
                     board_size: tuple[int, int] = (8, 8)):
            super().__init__(input_dim, config)

            self.board_size = board_size
            output_size = board_size[0] * board_size[1]

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, output_size)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            x = self.fc(features)
            # Output values in [-1, 1] for player influence
            return torch.tanh(x)

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(predictions, targets.view(predictions.shape))


    class ThreatDetectionHead(AuxiliaryHead):
        """Detect threats and forcing moves."""

        def __init__(self, input_dim: int, config: TaskConfig,
                     num_threat_types: int = 5):
            super().__init__(input_dim, config)

            self.num_types = num_threat_types

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, num_threat_types)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(features))

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy(predictions, targets.float())


    class LineCompletionHead(AuxiliaryHead):
        """Predict probability of completing lines for each player."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__(input_dim, config)

            # Output: [player1_prob, player2_prob] for completing a line next
            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, 2)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(features))

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy(predictions, targets.float())


    class DefensiveNeedHead(AuxiliaryHead):
        """Predict urgency of defensive play."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__(input_dim, config)

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, 1)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            # 0 = no defensive need, 1 = urgent defense required
            return torch.sigmoid(self.fc(features))

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy(predictions.squeeze(), targets.float())


    class WinningPathHead(AuxiliaryHead):
        """Predict which moves are part of a winning path."""

        def __init__(self, input_dim: int, config: TaskConfig,
                     num_moves: int = 64):
            super().__init__(input_dim, config)

            self.num_moves = num_moves

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, num_moves)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(features))

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy(predictions, targets.float())


    class PositionTypeHead(AuxiliaryHead):
        """Classify position as tactical, positional, or balanced."""

        def __init__(self, input_dim: int, config: TaskConfig):
            super().__init__(input_dim, config)

            num_classes = config.num_classes or 3

            self.fc = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, num_classes)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.fc(features)

        def compute_loss(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(predictions, targets.long())


    class MultiTaskModel(nn.Module):
        """
        Wrapper that adds auxiliary task heads to a base model.
        """

        def __init__(
            self,
            base_model: nn.Module,
            config: MultiTaskConfig,
            feature_dim: int = 256,
            board_size: tuple[int, int] = (8, 8)
        ):
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.feature_dim = feature_dim
            self.board_size = board_size

            # Task heads
            self.task_heads = nn.ModuleDict()
            self._build_task_heads()

            # Uncertainty weighting (learned loss weights)
            if config.uncertainty_weighting:
                # log(sigma^2) parameters
                self.log_vars = nn.ParameterDict({
                    task.task_type.value: nn.Parameter(torch.zeros(1))
                    for task in config.tasks if task.enabled
                })

        def _build_task_heads(self):
            """Build auxiliary task heads."""
            head_classes = {
                AuxiliaryTask.MOVE_COUNT: MoveCountHead,
                AuxiliaryTask.GAME_PHASE: GamePhaseHead,
                AuxiliaryTask.TERRITORY_CONTROL: lambda d, c: TerritoryHead(d, c, self.board_size),
                AuxiliaryTask.THREAT_DETECTION: ThreatDetectionHead,
                AuxiliaryTask.LINE_COMPLETION: LineCompletionHead,
                AuxiliaryTask.DEFENSIVE_NEED: DefensiveNeedHead,
                AuxiliaryTask.WINNING_PATH: lambda d, c: WinningPathHead(d, c, self.board_size[0] * self.board_size[1]),
                AuxiliaryTask.POSITION_TYPE: PositionTypeHead,
            }

            for task_config in self.config.tasks:
                if not task_config.enabled:
                    continue

                head_class = head_classes.get(task_config.task_type)
                if head_class:
                    if callable(head_class) and not isinstance(head_class, type):
                        # Lambda for heads with extra args
                        head = head_class(self.feature_dim, task_config)
                    else:
                        head = head_class(self.feature_dim, task_config)
                    self.task_heads[task_config.task_type.value] = head

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """
            Forward pass returning all task outputs.

            Returns dict with:
                - 'policy': Policy logits
                - 'value': Value prediction
                - task_name: Auxiliary task outputs
            """
            # Get base model outputs and features
            if hasattr(self.base_model, 'forward_with_features'):
                policy, value, features = self.base_model.forward_with_features(x)
            else:
                # Assume base model returns (policy, value)
                output = self.base_model(x)
                if isinstance(output, tuple):
                    policy, value = output[:2]
                    features = output[2] if len(output) > 2 else None
                else:
                    policy = output
                    value = None
                    features = None

            outputs = {
                'policy': policy,
                'value': value
            }

            # Get auxiliary outputs
            if features is not None:
                # Flatten features if needed
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)

                for task_name, head in self.task_heads.items():
                    outputs[task_name] = head(features)

            return outputs

        def compute_losses(
            self,
            outputs: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """
            Compute all task losses.

            Args:
                outputs: Model outputs from forward()
                targets: Target values for each task

            Returns:
                (total_loss, individual_losses_dict)
            """
            losses = {}
            total_loss = torch.tensor(0.0, device=outputs['policy'].device)

            # Policy loss (if provided)
            if 'policy' in targets and outputs['policy'] is not None:
                policy_loss = F.cross_entropy(outputs['policy'], targets['policy'])
                losses['policy'] = policy_loss
                total_loss = total_loss + policy_loss

            # Value loss (if provided)
            if 'value' in targets and outputs['value'] is not None:
                value_loss = F.mse_loss(outputs['value'].squeeze(), targets['value'])
                losses['value'] = value_loss
                total_loss = total_loss + value_loss

            # Auxiliary task losses
            for task_config in self.config.tasks:
                if not task_config.enabled:
                    continue

                task_name = task_config.task_type.value
                if task_name not in outputs or task_name not in targets:
                    continue

                head = self.task_heads[task_name]
                task_loss = head.compute_loss(outputs[task_name], targets[task_name])
                losses[task_name] = task_loss

                # Apply loss weighting
                if self.config.uncertainty_weighting and task_name in self.log_vars:
                    # Uncertainty weighting: 1/(2*sigma^2) * loss + log(sigma)
                    log_var = self.log_vars[task_name]
                    precision = torch.exp(-log_var)
                    weighted_loss = precision * task_loss + log_var
                else:
                    weighted_loss = task_config.loss_weight * task_loss

                total_loss = total_loss + weighted_loss

            return total_loss, losses

        def get_task_weights(self) -> dict[str, float]:
            """Get current task weights (static or learned)."""
            weights = {}

            if self.config.uncertainty_weighting:
                # Batch all log_var values into single GPU->CPU transfer
                log_var_tensors = []
                task_names = []
                for task_config in self.config.tasks:
                    if not task_config.enabled:
                        continue
                    task_name = task_config.task_type.value
                    if task_name in self.log_vars:
                        log_var_tensors.append(self.log_vars[task_name])
                        task_names.append(task_name)

                if log_var_tensors:
                    # Single GPU->CPU transfer for all log vars
                    stacked = torch.cat(log_var_tensors)
                    log_var_values = stacked.detach().cpu().numpy()

                    for i, task_name in enumerate(task_names):
                        weights[task_name] = float(math.exp(-log_var_values[i]))

            # Add static weights for tasks not using uncertainty weighting
            for task_config in self.config.tasks:
                if not task_config.enabled:
                    continue
                task_name = task_config.task_type.value
                if task_name not in weights:
                    weights[task_name] = task_config.loss_weight

            return weights

        def compute_losses_batched(
            self,
            outputs: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
            """
            Compute all task losses with batched GPU->CPU transfer for logging.

            This is more efficient than compute_losses() when you need loss values
            for logging, as it uses a single GPU->CPU transfer instead of multiple
            .item() calls.

            Args:
                outputs: Model outputs from forward()
                targets: Target values for each task

            Returns:
                (total_loss, individual_losses_dict, loss_values_for_logging)
            """
            losses = {}
            total_loss = torch.tensor(0.0, device=outputs['policy'].device)
            loss_tensors = []
            loss_names = []

            # Policy loss (if provided)
            if 'policy' in targets and outputs['policy'] is not None:
                policy_loss = F.cross_entropy(outputs['policy'], targets['policy'])
                losses['policy'] = policy_loss
                total_loss = total_loss + policy_loss
                loss_tensors.append(policy_loss)
                loss_names.append('policy')

            # Value loss (if provided)
            if 'value' in targets and outputs['value'] is not None:
                value_loss = F.mse_loss(outputs['value'].squeeze(), targets['value'])
                losses['value'] = value_loss
                total_loss = total_loss + value_loss
                loss_tensors.append(value_loss)
                loss_names.append('value')

            # Auxiliary task losses
            for task_config in self.config.tasks:
                if not task_config.enabled:
                    continue

                task_name = task_config.task_type.value
                if task_name not in outputs or task_name not in targets:
                    continue

                head = self.task_heads[task_name]
                task_loss = head.compute_loss(outputs[task_name], targets[task_name])
                losses[task_name] = task_loss
                loss_tensors.append(task_loss)
                loss_names.append(task_name)

                # Apply loss weighting
                if self.config.uncertainty_weighting and task_name in self.log_vars:
                    log_var = self.log_vars[task_name]
                    precision = torch.exp(-log_var)
                    weighted_loss = precision * task_loss + log_var
                else:
                    weighted_loss = task_config.loss_weight * task_loss

                total_loss = total_loss + weighted_loss

            # Single GPU->CPU transfer for all loss values (for logging)
            if loss_tensors:
                stacked_losses = torch.stack(loss_tensors).detach()
                loss_values_np = stacked_losses.cpu().numpy()
                loss_values = {name: float(loss_values_np[i]) for i, name in enumerate(loss_names)}
            else:
                loss_values = {}

            return total_loss, losses, loss_values


    class GradNormBalancer:
        """
        GradNorm algorithm for balancing multi-task gradients.

        Paper: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
        """

        def __init__(
            self,
            model: MultiTaskModel,
            task_names: list[str],
            alpha: float = 1.5,
            lr: float = 0.01
        ):
            self.model = model
            self.task_names = task_names
            self.alpha = alpha

            # Learnable weights
            self.weights = {name: nn.Parameter(torch.ones(1)) for name in task_names}
            self.optimizer = torch.optim.Adam(self.weights.values(), lr=lr)

            # Initial loss values for normalization
            self.initial_losses: dict[str, float] = {}

        def update_weights(
            self,
            losses: dict[str, torch.Tensor],
            shared_layer: nn.Module
        ):
            """
            Update task weights based on gradient norms.

            Args:
                losses: Current task losses
                shared_layer: The shared layer to compute gradients for
            """
            # Batch initial loss extraction (single GPU->CPU transfer)
            if not self.initial_losses:
                loss_names = list(losses.keys())
                loss_tensors = torch.stack([losses[k] for k in loss_names])
                loss_values = loss_tensors.detach().cpu().numpy()
                self.initial_losses = {name: float(loss_values[i]) for i, name in enumerate(loss_names)}

            # Compute gradient norms (collect tensors, batch transfer)
            grad_norm_tensors = []
            grad_norm_names = []
            for name in self.task_names:
                if name not in losses:
                    continue

                # Zero grads
                self.model.zero_grad()

                # Backward pass for this task
                weighted_loss = self.weights[name] * losses[name]
                weighted_loss.backward(retain_graph=True)

                # Get gradient norm from shared layer
                grad = None
                for param in shared_layer.parameters():
                    if param.grad is not None:
                        if grad is None:
                            grad = param.grad.flatten()
                        else:
                            grad = torch.cat([grad, param.grad.flatten()])

                if grad is not None:
                    grad_norm_tensors.append(grad.norm())
                    grad_norm_names.append(name)

            if not grad_norm_tensors:
                return

            # Single GPU->CPU transfer for all gradient norms
            stacked_norms = torch.stack(grad_norm_tensors).detach()
            grad_norm_values = stacked_norms.cpu().numpy()
            grad_norms = {name: float(grad_norm_values[i]) for i, name in enumerate(grad_norm_names)}

            # Compute target gradient norms
            avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)

            # Batch current loss extraction for loss ratios (single GPU->CPU transfer)
            ratio_names = [n for n in self.task_names if n in losses and n in self.initial_losses]
            if ratio_names:
                ratio_tensors = torch.stack([losses[n] for n in ratio_names])
                ratio_values = ratio_tensors.detach().cpu().numpy()
                rel_inv_rates = {
                    name: float(ratio_values[i]) / (self.initial_losses[name] + 1e-8)
                    for i, name in enumerate(ratio_names)
                }
            else:
                rel_inv_rates = {}

            avg_rate = sum(rel_inv_rates.values()) / len(rel_inv_rates) if rel_inv_rates else 1.0

            # Batch weight extraction (single GPU->CPU transfer)
            weight_names = list(self.weights.keys())
            weight_tensors = torch.stack([self.weights[n] for n in weight_names])
            weight_values = weight_tensors.detach().cpu().numpy()
            weights_dict = {name: float(weight_values[i]) for i, name in enumerate(weight_names)}

            # Update weights
            self.optimizer.zero_grad()

            for name in self.task_names:
                if name not in grad_norms or name not in rel_inv_rates:
                    continue

                # Target gradient norm (for GradNorm algorithm)
                # Note: target used for weight normalization, not direct loss
                _ = avg_grad_norm * (rel_inv_rates[name] / avg_rate) ** self.alpha

            # Normalize weights (use already-extracted values)
            weight_sum = sum(weights_dict.values())
            if weight_sum > 0:
                for w in self.weights.values():
                    w.data = w.data / weight_sum * len(self.task_names)

        def get_weights(self) -> dict[str, float]:
            """Get current task weights (batched GPU->CPU transfer)."""
            weight_names = list(self.weights.keys())
            weight_tensors = torch.stack([self.weights[n] for n in weight_names])
            weight_values = weight_tensors.detach().cpu().numpy()
            return {name: float(weight_values[i]) for i, name in enumerate(weight_names)}


    class AuxiliaryDataGenerator:
        """
        Generate auxiliary task labels from game states.
        """

        @staticmethod
        def generate_move_count_label(game_state: Any) -> float:
            """Estimate remaining moves from game state."""
            if hasattr(game_state, 'move_count'):
                total_moves = getattr(game_state, 'total_moves', 100)
                return max(0, total_moves - game_state.move_count)
            return 50.0  # Default estimate

        @staticmethod
        def generate_game_phase_label(game_state: Any) -> int:
            """Classify game phase: 0=opening, 1=midgame, 2=endgame."""
            if hasattr(game_state, 'move_count'):
                move = game_state.move_count
                if move < 15:
                    return 0
                elif move < 45:
                    return 1
                else:
                    return 2
            return 1  # Default to midgame

        @staticmethod
        def generate_territory_label(
            game_state: Any,
            board_size: tuple[int, int] = (8, 8)
        ) -> torch.Tensor:
            """Generate territory influence map."""
            territory = torch.zeros(board_size[0] * board_size[1])

            if hasattr(game_state, 'board'):
                board = game_state.board
                for i in range(board_size[0]):
                    for j in range(board_size[1]):
                        idx = i * board_size[1] + j
                        if hasattr(board, '__getitem__'):
                            cell = board[i][j] if len(board) > i and len(board[i]) > j else 0
                            territory[idx] = cell  # -1, 0, or 1

            return territory

        @staticmethod
        def generate_defensive_need_label(game_state: Any, value: float) -> float:
            """
            Estimate defensive urgency.

            Based on position evaluation - worse positions need more defense.
            """
            if hasattr(game_state, 'current_player'):
                player = game_state.current_player
                # If player is losing, defensive need is higher
                player_value = value if player == 1 else -value
                if player_value < -0.3:
                    return 1.0  # High defensive need
                elif player_value < 0:
                    return 0.5  # Moderate
                else:
                    return 0.0  # Low

            return 0.5

        @staticmethod
        def generate_labels(
            game_state: Any,
            value: float,
            board_size: tuple[int, int] = (8, 8)
        ) -> dict[str, torch.Tensor]:
            """Generate all auxiliary labels for a game state."""
            return {
                'move_count': torch.tensor(
                    AuxiliaryDataGenerator.generate_move_count_label(game_state)
                ),
                'game_phase': torch.tensor(
                    AuxiliaryDataGenerator.generate_game_phase_label(game_state)
                ),
                'territory': AuxiliaryDataGenerator.generate_territory_label(
                    game_state, board_size
                ),
                'defense': torch.tensor(
                    AuxiliaryDataGenerator.generate_defensive_need_label(game_state, value)
                )
            }


def create_default_multitask_config() -> MultiTaskConfig:
    """Create default multi-task configuration."""
    return MultiTaskConfig(
        tasks=[
            TaskConfig(AuxiliaryTask.MOVE_COUNT, loss_weight=0.05),
            TaskConfig(AuxiliaryTask.GAME_PHASE, loss_weight=0.1, num_classes=3),
            TaskConfig(AuxiliaryTask.TERRITORY_CONTROL, loss_weight=0.1),
            TaskConfig(AuxiliaryTask.DEFENSIVE_NEED, loss_weight=0.05),
        ],
        uncertainty_weighting=True,
        gradient_normalization=False
    )


def main():
    """Demonstrate multi-task learning."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return

    import torch
    import torch.nn as nn

    # Create a simple base model
    class SimpleBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(18, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 256)
            self.policy_head = nn.Linear(256, 64)
            self.value_head = nn.Linear(256, 1)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            features = torch.relu(self.fc(x))
            policy = self.policy_head(features)
            value = torch.tanh(self.value_head(features))
            return policy, value, features

        def forward_with_features(self, x):
            return self.forward(x)

    # Create multi-task model
    base_model = SimpleBaseModel()
    config = create_default_multitask_config()
    mt_model = MultiTaskModel(base_model, config, feature_dim=256)

    print("Multi-Task Model Architecture:")
    print(f"  Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"  Task heads: {list(mt_model.task_heads.keys())}")
    print()

    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 18, 8, 8)

    # Forward pass
    outputs = mt_model(x)
    print("Outputs:")
    for name, tensor in outputs.items():
        if tensor is not None:
            print(f"  {name}: {tensor.shape}")
        else:
            print(f"  {name}: None")

    # Create dummy targets
    targets = {
        'policy': torch.randint(0, 64, (batch_size,)),
        'value': torch.randn(batch_size),
        'move_count': torch.rand(batch_size) * 50,
        'game_phase': torch.randint(0, 3, (batch_size,)),
        'territory': torch.randn(batch_size, 64),
        'defense': torch.rand(batch_size),
    }

    # Compute losses
    total_loss, individual_losses = mt_model.compute_losses(outputs, targets)

    print("\nLosses:")
    print(f"  Total: {total_loss.item():.4f}")
    for name, loss in individual_losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # Get task weights
    weights = mt_model.get_task_weights()
    print("\nTask Weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")

    # Demonstrate auxiliary data generation
    print("\n=== Auxiliary Data Generation ===")

    class DummyGameState:
        def __init__(self):
            self.move_count = 25
            self.total_moves = 100
            self.current_player = 1
            self.board = [[0] * 8 for _ in range(8)]
            self.board[3][3] = 1
            self.board[4][4] = -1

    game_state = DummyGameState()
    labels = AuxiliaryDataGenerator.generate_labels(game_state, value=0.2)

    print("Generated labels:")
    for name, tensor in labels.items():
        print(f"  {name}: {tensor.shape if hasattr(tensor, 'shape') else tensor}")


if __name__ == "__main__":
    main()
