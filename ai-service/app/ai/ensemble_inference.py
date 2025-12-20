"""
Ensemble Inference for RingRift AI.

Combines predictions from multiple models for stronger play and
uncertainty estimation. Supports various ensemble strategies.

Features:
- Average ensemble (simple averaging)
- Weighted ensemble (weight by model strength/Elo)
- Voting ensemble (majority vote on best move)
- Mixture of experts (learned weighting)
- Uncertainty estimation via prediction disagreement

Usage:
    from app.ai.ensemble_inference import EnsemblePredictor

    # Create ensemble from multiple models
    ensemble = EnsemblePredictor(
        model_paths=["model1.pt", "model2.pt", "model3.pt"],
        strategy="weighted"
    )

    # Get predictions with uncertainty
    policy, value, uncertainty = ensemble.predict(state)

    # Get best move with confidence
    move, confidence = ensemble.get_best_move(state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, ensemble inference will be limited")


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    AVERAGE = "average"         # Simple average of predictions
    WEIGHTED = "weighted"       # Weight by model strength
    VOTING = "voting"           # Majority vote on best move
    MAX = "max"                 # Take maximum confidence
    BAYESIAN = "bayesian"       # Bayesian model averaging


@dataclass
class ModelConfig:
    """Configuration for a model in the ensemble."""
    path: Path
    weight: float = 1.0
    elo: float = 1500.0
    name: str = ""


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction."""
    policy: np.ndarray          # Combined policy distribution
    value: float                # Combined value estimate
    uncertainty: float          # Prediction uncertainty
    individual_values: list[float]  # Per-model values
    individual_policies: list[np.ndarray]  # Per-model policies
    agreement: float            # Agreement between models (0-1)


class EnsemblePredictor:
    """Combines multiple models for ensemble prediction."""

    def __init__(
        self,
        model_paths: list[Union[str, Path]] | None = None,
        model_configs: list[ModelConfig] | None = None,
        strategy: Union[str, EnsembleStrategy] = EnsembleStrategy.WEIGHTED,
        temperature: float = 1.0,
        min_agreement_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize ensemble predictor.

        Args:
            model_paths: List of paths to model files
            model_configs: Detailed model configurations (overrides model_paths)
            strategy: Ensemble combination strategy
            temperature: Temperature for softmax (higher = more exploration)
            min_agreement_threshold: Minimum agreement for confident predictions
            device: Device to run inference on
        """
        if isinstance(strategy, str):
            strategy = EnsembleStrategy(strategy)

        self.strategy = strategy
        self.temperature = temperature
        self.min_agreement_threshold = min_agreement_threshold
        self.device = device

        # Initialize model configs
        if model_configs:
            self.configs = model_configs
        elif model_paths:
            self.configs = [
                ModelConfig(path=Path(p), name=Path(p).stem)
                for p in model_paths
            ]
        else:
            self.configs = []

        # Load models
        self.models = []
        self._load_models()

    def _load_models(self):
        """Load all models in the ensemble."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available, cannot load models")
            return

        for config in self.configs:
            try:
                if config.path.exists():
                    model = torch.load(config.path, map_location=self.device)
                    if hasattr(model, "eval"):
                        model.eval()
                    self.models.append(model)
                    logger.info(f"Loaded model: {config.name}")
                else:
                    logger.warning(f"Model not found: {config.path}")
            except Exception as e:
                logger.error(f"Error loading model {config.path}: {e}")

    def add_model(self, path: Path, weight: float = 1.0, elo: float = 1500.0):
        """Add a model to the ensemble.

        Args:
            path: Path to model file
            weight: Model weight for weighted averaging
            elo: Model Elo for Elo-based weighting
        """
        config = ModelConfig(path=path, weight=weight, elo=elo, name=path.stem)
        self.configs.append(config)

        if HAS_TORCH and path.exists():
            try:
                model = torch.load(path, map_location=self.device)
                if hasattr(model, "eval"):
                    model.eval()
                self.models.append(model)
            except Exception as e:
                logger.error(f"Error loading model {path}: {e}")

    def _get_model_weights(self) -> np.ndarray:
        """Compute weights for each model."""
        if not self.configs:
            return np.array([])

        if self.strategy == EnsembleStrategy.AVERAGE:
            return np.ones(len(self.configs)) / len(self.configs)

        elif self.strategy == EnsembleStrategy.WEIGHTED:
            weights = np.array([c.weight for c in self.configs])
            return weights / weights.sum()

        elif self.strategy == EnsembleStrategy.BAYESIAN:
            # Weight by Elo (higher Elo = more weight)
            elos = np.array([c.elo for c in self.configs])
            # Convert Elo difference to weight using logistic function
            mean_elo = np.mean(elos)
            weights = 1.0 / (1.0 + 10 ** ((mean_elo - elos) / 400))
            return weights / weights.sum()

        else:
            return np.ones(len(self.configs)) / len(self.configs)

    def _forward_models(
        self,
        state: Any,
        features: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run forward pass on all models.

        Args:
            state: Game state (for models that take state directly)
            features: Pre-computed features tensor

        Returns:
            Tuple of (policies, values) from each model
        """
        policies = []
        values = []

        if not self.models:
            # Return uniform policy and neutral value if no models
            return [np.ones(100) / 100], [0.5]

        for model in self.models:
            try:
                if HAS_TORCH:
                    with torch.no_grad():
                        if features is not None:
                            input_tensor = torch.tensor(features, dtype=torch.float32)
                            if input_tensor.dim() == 3:
                                input_tensor = input_tensor.unsqueeze(0)
                            input_tensor = input_tensor.to(self.device)

                            output = model(input_tensor)

                            if isinstance(output, tuple):
                                policy_logits, value = output
                            else:
                                policy_logits = output
                                value = torch.tensor([0.5])

                            policy = F.softmax(policy_logits / self.temperature, dim=-1)
                            policies.append(policy.cpu().numpy().flatten())
                            values.append(float(value.cpu().numpy().flatten()[0]))
                        else:
                            # Model takes state directly
                            policy, value = model.predict(state)
                            policies.append(policy)
                            values.append(value)
                else:
                    # Fallback for non-PyTorch
                    policies.append(np.ones(100) / 100)
                    values.append(0.5)

            except Exception as e:
                logger.error(f"Error in model forward: {e}")
                policies.append(np.ones(100) / 100)
                values.append(0.5)

        return policies, values

    def predict(
        self,
        state: Any = None,
        features: np.ndarray | None = None,
    ) -> EnsemblePrediction:
        """Get ensemble prediction.

        Args:
            state: Game state
            features: Pre-computed features

        Returns:
            EnsemblePrediction with combined policy, value, and uncertainty
        """
        policies, values = self._forward_models(state, features)
        weights = self._get_model_weights()

        if not policies:
            return EnsemblePrediction(
                policy=np.ones(100) / 100,
                value=0.5,
                uncertainty=1.0,
                individual_values=[],
                individual_policies=[],
                agreement=0.0,
            )

        # Combine policies based on strategy
        if self.strategy == EnsembleStrategy.VOTING:
            # Each model votes for its top move
            combined_policy = self._voting_combine(policies)
        elif self.strategy == EnsembleStrategy.MAX:
            # Take maximum confidence for each move
            combined_policy = self._max_combine(policies)
        else:
            # Weighted average
            combined_policy = self._weighted_combine(policies, weights)

        # Combine values (always weighted average)
        combined_value = np.average(values, weights=weights[:len(values)])

        # Compute uncertainty and agreement
        uncertainty = self._compute_uncertainty(policies, values)
        agreement = self._compute_agreement(policies)

        return EnsemblePrediction(
            policy=combined_policy,
            value=float(combined_value),
            uncertainty=uncertainty,
            individual_values=values,
            individual_policies=policies,
            agreement=agreement,
        )

    def _weighted_combine(
        self,
        policies: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """Combine policies using weighted average."""
        if not policies:
            return np.ones(100) / 100

        # Ensure all policies have same size
        max_size = max(len(p) for p in policies)
        padded = []
        for p in policies:
            if len(p) < max_size:
                padded.append(np.pad(p, (0, max_size - len(p))))
            else:
                padded.append(p)

        stacked = np.stack(padded)
        combined = np.average(stacked, axis=0, weights=weights[:len(policies)])

        # Renormalize
        combined = combined / (combined.sum() + 1e-8)

        return combined

    def _voting_combine(self, policies: list[np.ndarray]) -> np.ndarray:
        """Combine policies using voting."""
        if not policies:
            return np.ones(100) / 100

        max_size = max(len(p) for p in policies)
        votes = np.zeros(max_size)

        for policy in policies:
            best_move = np.argmax(policy)
            if best_move < max_size:
                votes[best_move] += 1

        # Convert votes to probability
        combined = votes / (votes.sum() + 1e-8)

        return combined

    def _max_combine(self, policies: list[np.ndarray]) -> np.ndarray:
        """Combine policies using max confidence."""
        if not policies:
            return np.ones(100) / 100

        max_size = max(len(p) for p in policies)
        padded = []
        for p in policies:
            if len(p) < max_size:
                padded.append(np.pad(p, (0, max_size - len(p))))
            else:
                padded.append(p)

        stacked = np.stack(padded)
        combined = np.max(stacked, axis=0)

        # Renormalize
        combined = combined / (combined.sum() + 1e-8)

        return combined

    def _compute_uncertainty(
        self,
        policies: list[np.ndarray],
        values: list[float],
    ) -> float:
        """Compute prediction uncertainty based on model disagreement."""
        if len(policies) < 2:
            return 0.0

        # Policy uncertainty: average KL divergence from mean
        mean_policy = np.mean(list(policies), axis=0)
        policy_kl_divs = []
        for policy in policies:
            # KL(P || Q) = sum(P * log(P/Q))
            # Clip to avoid log(0)
            p = np.clip(policy, 1e-10, 1.0)
            q = np.clip(mean_policy[:len(p)], 1e-10, 1.0)
            kl = np.sum(p * np.log(p / q))
            policy_kl_divs.append(kl)

        policy_uncertainty = np.mean(policy_kl_divs)

        # Value uncertainty: standard deviation
        value_std = np.std(values) if len(values) > 1 else 0.0

        # Combined uncertainty (normalized to roughly 0-1)
        uncertainty = 0.5 * np.tanh(policy_uncertainty) + 0.5 * min(value_std * 2, 1.0)

        return float(uncertainty)

    def _compute_agreement(self, policies: list[np.ndarray]) -> float:
        """Compute agreement between models on best move."""
        if len(policies) < 2:
            return 1.0

        best_moves = [np.argmax(p) for p in policies]

        # Count most common best move
        from collections import Counter
        move_counts = Counter(best_moves)
        most_common_count = move_counts.most_common(1)[0][1]

        agreement = most_common_count / len(policies)

        return float(agreement)

    def get_best_move(
        self,
        state: Any = None,
        features: np.ndarray | None = None,
    ) -> tuple[int, float]:
        """Get best move with confidence score.

        Args:
            state: Game state
            features: Pre-computed features

        Returns:
            Tuple of (best_move_index, confidence)
        """
        prediction = self.predict(state, features)

        best_move = int(np.argmax(prediction.policy))
        confidence = float(prediction.policy[best_move] * prediction.agreement)

        return best_move, confidence

    def get_move_with_uncertainty(
        self,
        state: Any = None,
        features: np.ndarray | None = None,
        exploration_bonus: float = 0.1,
    ) -> tuple[int, float, float]:
        """Get move considering uncertainty for exploration.

        Uses Upper Confidence Bound (UCB) style exploration:
        score = policy + exploration_bonus * uncertainty

        Args:
            state: Game state
            features: Pre-computed features
            exploration_bonus: Weight for uncertainty exploration

        Returns:
            Tuple of (move_index, confidence, uncertainty)
        """
        prediction = self.predict(state, features)

        # UCB-style scoring
        ucb_scores = prediction.policy + exploration_bonus * prediction.uncertainty

        best_move = int(np.argmax(ucb_scores))
        confidence = float(prediction.policy[best_move])

        return best_move, confidence, prediction.uncertainty


class DynamicEnsemble(EnsemblePredictor):
    """Ensemble that dynamically adjusts weights based on performance."""

    def __init__(
        self,
        model_paths: list[Union[str, Path]] | None = None,
        model_configs: list[ModelConfig] | None = None,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        """
        Initialize dynamic ensemble.

        Args:
            model_paths: List of model paths
            model_configs: Model configurations
            learning_rate: Learning rate for weight updates
        """
        super().__init__(model_paths, model_configs, **kwargs)
        self.learning_rate = learning_rate
        self.performance_history: dict[int, list[float]] = {
            i: [] for i in range(len(self.configs))
        }

    def update_weights(self, model_idx: int, reward: float):
        """Update model weight based on reward.

        Args:
            model_idx: Index of model to update
            reward: Reward signal (1 for correct, 0 for incorrect)
        """
        if model_idx >= len(self.configs):
            return

        self.performance_history[model_idx].append(reward)

        # Compute running average
        history = self.performance_history[model_idx][-100:]  # Last 100 samples
        avg_performance = np.mean(history)

        # Update weight
        old_weight = self.configs[model_idx].weight
        new_weight = old_weight + self.learning_rate * (avg_performance - 0.5)
        new_weight = max(0.1, min(10.0, new_weight))  # Clip weights

        self.configs[model_idx].weight = new_weight

    def record_prediction_outcome(
        self,
        individual_predictions: list[int],
        correct_move: int,
    ):
        """Record outcome of predictions for weight learning.

        Args:
            individual_predictions: Each model's predicted move
            correct_move: The correct/winning move
        """
        for i, pred in enumerate(individual_predictions):
            reward = 1.0 if pred == correct_move else 0.0
            self.update_weights(i, reward)


def create_ensemble_from_directory(
    model_dir: Path,
    pattern: str = "*.pt",
    strategy: str = "weighted",
    max_models: int = 5,
) -> EnsemblePredictor:
    """Create ensemble from all models in a directory.

    Args:
        model_dir: Directory containing models
        pattern: Glob pattern for model files
        strategy: Ensemble strategy
        max_models: Maximum number of models to include

    Returns:
        EnsemblePredictor instance
    """
    model_paths = sorted(model_dir.glob(pattern))[-max_models:]

    if not model_paths:
        logger.warning(f"No models found in {model_dir} matching {pattern}")

    return EnsemblePredictor(
        model_paths=model_paths,
        strategy=strategy,
    )
