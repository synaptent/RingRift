"""
Temperature Scheduling for Self-Play in RingRift AI.

Controls exploration vs exploitation during self-play game generation
through sophisticated temperature schedules.
"""

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of temperature schedules."""
    CONSTANT = "constant"              # Fixed temperature
    LINEAR_DECAY = "linear_decay"      # Linear decrease over moves
    EXPONENTIAL_DECAY = "exp_decay"    # Exponential decrease
    STEP = "step"                      # Step function changes
    COSINE = "cosine"                  # Cosine annealing
    ADAPTIVE = "adaptive"              # Based on game state
    CURRICULUM = "curriculum"          # Based on training progress
    MIXED = "mixed"                    # Combination of schedules
    ELO_ADAPTIVE = "elo_adaptive"      # Based on model Elo rating (Dec 2025)


@dataclass
class TemperatureConfig:
    """Configuration for temperature scheduling."""
    schedule_type: ScheduleType = ScheduleType.LINEAR_DECAY
    initial_temp: float = 1.0
    final_temp: float = 0.1
    exploration_moves: int = 30        # Moves with high exploration
    decay_start_move: int = 10         # When to start decaying
    decay_end_move: int = 60           # When to reach final temp
    min_temp: float = 0.01             # Absolute minimum
    max_temp: float = 2.0              # Absolute maximum
    step_schedule: list[tuple[int, float]] = field(default_factory=list)
    adaptive_config: dict[str, Any] = field(default_factory=dict)
    add_noise: bool = False
    noise_scale: float = 0.1


class TemperatureSchedule(ABC):
    """Abstract base class for temperature schedules."""

    @abstractmethod
    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        """Get temperature for a given move."""

    def clip_temperature(self, temp: float, min_temp: float = 0.01,
                         max_temp: float = 2.0) -> float:
        """Clip temperature to valid range."""
        return max(min_temp, min(max_temp, temp))


class ConstantSchedule(TemperatureSchedule):
    """Constant temperature throughout the game."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        return self.temperature


class LinearDecaySchedule(TemperatureSchedule):
    """Linear decay from initial to final temperature."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        decay_start: int = 10,
        decay_end: int = 60
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_start = decay_start
        self.decay_end = decay_end

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        if move_number <= self.decay_start:
            return self.initial_temp

        if move_number >= self.decay_end:
            return self.final_temp

        # Linear interpolation
        progress = (move_number - self.decay_start) / (self.decay_end - self.decay_start)
        return self.initial_temp + progress * (self.final_temp - self.initial_temp)


class ExponentialDecaySchedule(TemperatureSchedule):
    """Exponential decay schedule."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        decay_rate: float = 0.05,
        decay_start: int = 10
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_rate = decay_rate
        self.decay_start = decay_start

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        if move_number <= self.decay_start:
            return self.initial_temp

        effective_move = move_number - self.decay_start
        temp = self.initial_temp * math.exp(-self.decay_rate * effective_move)
        return max(temp, self.final_temp)


class StepSchedule(TemperatureSchedule):
    """Step function temperature changes at specified moves."""

    def __init__(self, steps: list[tuple[int, float]], default_temp: float = 1.0):
        """
        Args:
            steps: List of (move_number, temperature) tuples, sorted by move
            default_temp: Temperature before first step
        """
        self.steps = sorted(steps, key=lambda x: x[0])
        self.default_temp = default_temp

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        temp = self.default_temp
        for step_move, step_temp in self.steps:
            if move_number >= step_move:
                temp = step_temp
            else:
                break
        return temp


class CosineAnnealingSchedule(TemperatureSchedule):
    """Cosine annealing temperature schedule."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        period_moves: int = 60,
        num_cycles: int = 1
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.period_moves = period_moves
        self.num_cycles = num_cycles

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        # Cosine annealing with warm restarts
        cycle_length = self.period_moves // self.num_cycles
        move_in_cycle = move_number % cycle_length
        progress = move_in_cycle / cycle_length

        # Cosine curve from initial to final
        cos_value = (1 + math.cos(math.pi * progress)) / 2
        return self.final_temp + cos_value * (self.initial_temp - self.final_temp)


class AdaptiveSchedule(TemperatureSchedule):
    """
    Adaptive temperature based on game state analysis.

    Adjusts temperature based on:
    - Position complexity
    - Number of available moves
    - Game phase
    - Model uncertainty
    """

    def __init__(
        self,
        base_temp: float = 1.0,
        complexity_weight: float = 0.3,
        uncertainty_weight: float = 0.3,
        phase_weight: float = 0.4,
        min_temp: float = 0.1,
        max_temp: float = 2.0
    ):
        self.base_temp = base_temp
        self.complexity_weight = complexity_weight
        self.uncertainty_weight = uncertainty_weight
        self.phase_weight = phase_weight
        self.min_temp = min_temp
        self.max_temp = max_temp

    def _estimate_complexity(self, game_state: Any) -> float:
        """Estimate position complexity (0-1)."""
        if game_state is None:
            return 0.5

        # Count available moves as complexity proxy
        if hasattr(game_state, 'get_legal_moves'):
            num_moves = len(game_state.get_legal_moves())
            # Normalize: more moves = higher complexity
            complexity = min(1.0, num_moves / 50)
        elif hasattr(game_state, 'legal_moves'):
            num_moves = len(game_state.legal_moves)
            complexity = min(1.0, num_moves / 50)
        else:
            complexity = 0.5

        return complexity

    def _estimate_phase(self, move_number: int, game_state: Any) -> float:
        """Estimate game phase (0=opening, 0.5=midgame, 1=endgame)."""
        # Simple heuristic based on move number
        if move_number < 20:
            return 0.0 + move_number / 40  # 0-0.5
        elif move_number < 60:
            return 0.5  # Midgame
        else:
            return min(1.0, 0.5 + (move_number - 60) / 40)

    def _get_uncertainty(self, game_state: Any) -> float:
        """Get model uncertainty from game state if available."""
        if game_state is not None and hasattr(game_state, 'last_policy_entropy'):
            # High entropy = high uncertainty
            entropy = game_state.last_policy_entropy
            return min(1.0, entropy / 2.0)  # Normalize
        return 0.5

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        complexity = self._estimate_complexity(game_state)
        phase = self._estimate_phase(move_number, game_state)
        uncertainty = self._get_uncertainty(game_state)

        # Higher complexity -> higher temperature (explore more)
        complexity_factor = 1.0 + self.complexity_weight * (complexity - 0.5)

        # Higher uncertainty -> higher temperature
        uncertainty_factor = 1.0 + self.uncertainty_weight * (uncertainty - 0.5)

        # Opening -> high temp, Endgame -> low temp
        phase_factor = 1.0 - self.phase_weight * phase

        temp = self.base_temp * complexity_factor * uncertainty_factor * phase_factor
        return self.clip_temperature(temp, self.min_temp, self.max_temp)


class CurriculumSchedule(TemperatureSchedule):
    """
    Temperature schedule that changes based on training progress.

    Early training: High temperature for diverse data
    Late training: Low temperature for quality data
    """

    def __init__(
        self,
        early_temp: float = 1.5,
        late_temp: float = 0.5,
        early_exploration_moves: int = 40,
        late_exploration_moves: int = 15,
        transition_start: float = 0.3,   # Training progress
        transition_end: float = 0.7
    ):
        self.early_temp = early_temp
        self.late_temp = late_temp
        self.early_exploration_moves = early_exploration_moves
        self.late_exploration_moves = late_exploration_moves
        self.transition_start = transition_start
        self.transition_end = transition_end

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        if training_progress is None:
            training_progress = 0.0

        # Interpolate between early and late configs
        if training_progress <= self.transition_start:
            progress = 0.0
        elif training_progress >= self.transition_end:
            progress = 1.0
        else:
            progress = (training_progress - self.transition_start) / \
                       (self.transition_end - self.transition_start)

        # Interpolate parameters
        base_temp = self.early_temp + progress * (self.late_temp - self.early_temp)
        exploration_moves = int(self.early_exploration_moves +
                                progress * (self.late_exploration_moves - self.early_exploration_moves))

        # Apply move-based decay
        if move_number <= exploration_moves:
            return base_temp
        else:
            # Decay after exploration phase
            decay = math.exp(-0.05 * (move_number - exploration_moves))
            return max(0.1, base_temp * decay)


class EloAdaptiveSchedule(TemperatureSchedule):
    """
    Temperature schedule that adapts based on model Elo rating.

    This supports the strength-driven training philosophy (December 2025):
    - Weak models (< 1300 Elo): High exploration to find strong moves
    - Medium models (1300-1500): Balanced exploration/exploitation
    - Strong models (1500-1700): More exploitation, less noise
    - Very strong (> 1700): Confident, low-temperature play

    Usage:
        schedule = EloAdaptiveSchedule(model_elo=1450)
        temp = schedule.get_temperature(move_number=15)
    """

    def __init__(
        self,
        model_elo: float = 1500.0,
        weak_temp: float = 1.5,
        medium_temp: float = 1.0,
        strong_temp: float = 0.7,
        very_strong_temp: float = 0.5,
        exploration_moves: int = 30,
        elo_weak_threshold: float = 1300.0,
        elo_medium_threshold: float = 1500.0,
        elo_strong_threshold: float = 1700.0,
    ):
        """Initialize Elo-adaptive temperature schedule.

        Args:
            model_elo: Current model Elo rating
            weak_temp: Base temperature for weak models (< 1300 Elo)
            medium_temp: Base temperature for medium models (1300-1500 Elo)
            strong_temp: Base temperature for strong models (1500-1700 Elo)
            very_strong_temp: Base temperature for very strong models (> 1700 Elo)
            exploration_moves: Number of moves before temperature decay
            elo_weak_threshold: Elo below which model is considered weak
            elo_medium_threshold: Elo below which model is considered medium
            elo_strong_threshold: Elo above which model is considered very strong
        """
        self.model_elo = model_elo
        self.weak_temp = weak_temp
        self.medium_temp = medium_temp
        self.strong_temp = strong_temp
        self.very_strong_temp = very_strong_temp
        self.exploration_moves = exploration_moves
        self.elo_weak_threshold = elo_weak_threshold
        self.elo_medium_threshold = elo_medium_threshold
        self.elo_strong_threshold = elo_strong_threshold

        # Determine base temperature from Elo
        self._base_temp = self._compute_base_temp()

    def _compute_base_temp(self) -> float:
        """Compute base temperature from model Elo."""
        if self.model_elo < self.elo_weak_threshold:
            return self.weak_temp
        elif self.model_elo < self.elo_medium_threshold:
            # Interpolate between weak and medium
            progress = (self.model_elo - self.elo_weak_threshold) / (
                self.elo_medium_threshold - self.elo_weak_threshold
            )
            return self.weak_temp + progress * (self.medium_temp - self.weak_temp)
        elif self.model_elo < self.elo_strong_threshold:
            # Interpolate between medium and strong
            progress = (self.model_elo - self.elo_medium_threshold) / (
                self.elo_strong_threshold - self.elo_medium_threshold
            )
            return self.medium_temp + progress * (self.strong_temp - self.medium_temp)
        else:
            # Very strong - use lowest temperature
            return self.very_strong_temp

    def update_elo(self, new_elo: float) -> None:
        """Update model Elo and recalculate base temperature.

        Call this when model Elo changes (e.g., after evaluation).
        """
        self.model_elo = new_elo
        self._base_temp = self._compute_base_temp()
        logger.debug(
            f"[EloAdaptiveSchedule] Updated Elo to {new_elo:.0f}, "
            f"base_temp now {self._base_temp:.2f}"
        )

    def get_temperature(
        self,
        move_number: int,
        game_state: Any | None = None,
        training_progress: float | None = None,
    ) -> float:
        """Get temperature for a move based on Elo and move number.

        Temperature is high in early moves (exploration phase) and
        decays exponentially afterward.
        """
        if move_number <= self.exploration_moves:
            return self._base_temp
        else:
            # Exponential decay after exploration phase
            effective_move = move_number - self.exploration_moves
            decay_rate = 0.03 if self.model_elo < self.elo_medium_threshold else 0.05
            decay = math.exp(-decay_rate * effective_move)
            final_temp = 0.1 if self.model_elo >= self.elo_strong_threshold else 0.2
            return max(final_temp, self._base_temp * decay)


class MixedSchedule(TemperatureSchedule):
    """
    Combines multiple schedules with different weights.
    """

    def __init__(self, schedules: list[tuple[TemperatureSchedule, float]]):
        """
        Args:
            schedules: List of (schedule, weight) tuples
        """
        self.schedules = schedules
        total_weight = sum(w for _, w in schedules)
        self.schedules = [(s, w / total_weight) for s, w in schedules]

    def get_temperature(self, move_number: int, game_state: Any | None = None,
                        training_progress: float | None = None) -> float:
        temp = 0.0
        for schedule, weight in self.schedules:
            temp += weight * schedule.get_temperature(move_number, game_state, training_progress)
        return temp


class TemperatureScheduler:
    """
    Main interface for temperature scheduling in self-play.
    """

    def __init__(self, config: TemperatureConfig | None = None):
        self.config = config or TemperatureConfig()
        self.schedule = self._create_schedule()
        self._training_progress = 0.0
        self._game_count = 0
        self._move_temperatures: list[float] = []
        # December 2025: Exploration boost from FeedbackLoopController
        # Values > 1.0 increase exploration (e.g., after failed promotion)
        # Values < 1.0 decrease exploration (e.g., after successful promotion)
        self._exploration_boost: float = 1.0

    def _create_schedule(self) -> TemperatureSchedule:
        """Create schedule based on configuration."""
        cfg = self.config

        if cfg.schedule_type == ScheduleType.CONSTANT:
            return ConstantSchedule(cfg.initial_temp)

        elif cfg.schedule_type == ScheduleType.LINEAR_DECAY:
            return LinearDecaySchedule(
                initial_temp=cfg.initial_temp,
                final_temp=cfg.final_temp,
                decay_start=cfg.decay_start_move,
                decay_end=cfg.decay_end_move
            )

        elif cfg.schedule_type == ScheduleType.EXPONENTIAL_DECAY:
            return ExponentialDecaySchedule(
                initial_temp=cfg.initial_temp,
                final_temp=cfg.final_temp,
                decay_rate=0.05,
                decay_start=cfg.decay_start_move
            )

        elif cfg.schedule_type == ScheduleType.STEP:
            return StepSchedule(cfg.step_schedule, cfg.initial_temp)

        elif cfg.schedule_type == ScheduleType.COSINE:
            return CosineAnnealingSchedule(
                initial_temp=cfg.initial_temp,
                final_temp=cfg.final_temp,
                period_moves=cfg.decay_end_move
            )

        elif cfg.schedule_type == ScheduleType.ADAPTIVE:
            return AdaptiveSchedule(
                base_temp=cfg.initial_temp,
                **cfg.adaptive_config
            )

        elif cfg.schedule_type == ScheduleType.CURRICULUM:
            return CurriculumSchedule(
                early_temp=cfg.initial_temp,
                late_temp=cfg.final_temp,
                early_exploration_moves=cfg.exploration_moves,
                late_exploration_moves=cfg.exploration_moves // 2
            )

        elif cfg.schedule_type == ScheduleType.MIXED:
            # Create a mixed schedule with linear and adaptive
            return MixedSchedule([
                (LinearDecaySchedule(cfg.initial_temp, cfg.final_temp,
                                     cfg.decay_start_move, cfg.decay_end_move), 0.6),
                (AdaptiveSchedule(cfg.initial_temp), 0.4)
            ])

        elif cfg.schedule_type == ScheduleType.ELO_ADAPTIVE:
            # Elo-adaptive schedule (December 2025)
            return EloAdaptiveSchedule(
                model_elo=cfg.adaptive_config.get('model_elo', 1500.0),
                weak_temp=cfg.adaptive_config.get('weak_temp', 1.5),
                medium_temp=cfg.adaptive_config.get('medium_temp', 1.0),
                strong_temp=cfg.adaptive_config.get('strong_temp', 0.7),
                very_strong_temp=cfg.adaptive_config.get('very_strong_temp', 0.5),
                exploration_moves=cfg.exploration_moves,
                elo_weak_threshold=cfg.adaptive_config.get('elo_weak_threshold', 1300.0),
                elo_medium_threshold=cfg.adaptive_config.get('elo_medium_threshold', 1500.0),
                elo_strong_threshold=cfg.adaptive_config.get('elo_strong_threshold', 1700.0),
            )

        else:
            return LinearDecaySchedule()

    def get_temperature(self, move_number: int, game_state: Any | None = None) -> float:
        """Get temperature for a move.

        Temperature is modified by:
        1. Base schedule (linear, exponential, adaptive, etc.)
        2. Exploration boost from FeedbackLoopController (1.0-2.0)
        3. Optional noise
        """
        temp = self.schedule.get_temperature(
            move_number, game_state, self._training_progress
        )

        # December 2025: Apply exploration boost from feedback loop
        # Boost > 1.0 increases temperature (more exploration after failed promotions)
        # Boost < 1.0 decreases temperature (less exploration after success)
        if self._exploration_boost != 1.0:
            temp = temp * self._exploration_boost

        # Add noise if configured
        if self.config.add_noise:
            noise = random.gauss(0, self.config.noise_scale)
            temp = temp * (1 + noise)

        # Clip to valid range
        temp = self.schedule.clip_temperature(
            temp, self.config.min_temp, self.config.max_temp
        )

        self._move_temperatures.append(temp)
        return temp

    def set_training_progress(self, progress: float):
        """Set current training progress (0-1)."""
        self._training_progress = max(0.0, min(1.0, progress))

    def set_exploration_boost(self, boost: float) -> None:
        """Set exploration boost from FeedbackLoopController.

        December 2025: Connects temperature scheduling to the feedback loop.
        When promotion fails, boost is increased (more exploration).
        When promotion succeeds, boost is decreased (more exploitation).

        Args:
            boost: Exploration multiplier (1.0 = normal, 1.3 = 30% more exploration)
                   Valid range: 0.5 to 2.0

        Example:
            scheduler.set_exploration_boost(1.3)  # After failed promotion
            scheduler.set_exploration_boost(1.0)  # After successful promotion
        """
        self._exploration_boost = max(0.5, min(2.0, boost))
        if boost != 1.0:
            logger.info(
                f"[TemperatureScheduler] Exploration boost set to {self._exploration_boost:.2f}"
            )

    def get_exploration_boost(self) -> float:
        """Get current exploration boost value."""
        return self._exploration_boost

    def new_game(self):
        """Signal start of a new game."""
        self._game_count += 1
        self._move_temperatures = []

    def get_game_stats(self) -> dict[str, float]:
        """Get temperature statistics for the current game."""
        if not self._move_temperatures:
            return {}

        return {
            'mean_temp': sum(self._move_temperatures) / len(self._move_temperatures),
            'max_temp': max(self._move_temperatures),
            'min_temp': min(self._move_temperatures),
            'final_temp': self._move_temperatures[-1] if self._move_temperatures else 0,
            'num_moves': len(self._move_temperatures)
        }


class AlphaZeroTemperature:
    """
    AlphaZero-style temperature with τ=1 for first N moves, then τ→0.
    """

    def __init__(
        self,
        exploration_moves: int = 30,
        exploration_temp: float = 1.0,
        exploitation_temp: float = 0.0
    ):
        self.exploration_moves = exploration_moves
        self.exploration_temp = exploration_temp
        self.exploitation_temp = exploitation_temp

    def get_temperature(self, move_number: int) -> float:
        """Get temperature (AlphaZero style)."""
        if move_number < self.exploration_moves:
            return self.exploration_temp
        return self.exploitation_temp

    def sample_move(self, policy: list[float], move_number: int) -> int:
        """Sample a move from policy with temperature."""
        temp = self.get_temperature(move_number)

        if temp == 0:
            # Deterministic selection (argmax)
            return max(range(len(policy)), key=lambda i: policy[i])

        # Apply temperature
        policy_array = [p ** (1.0 / temp) for p in policy]
        total = sum(policy_array)
        if total == 0:
            return random.randint(0, len(policy) - 1)

        policy_array = [p / total for p in policy_array]

        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(policy_array):
            cumsum += p
            if r <= cumsum:
                return i
        return len(policy) - 1


class DirichletNoiseTemperature:
    """
    Combines temperature with Dirichlet noise for root exploration.
    """

    def __init__(
        self,
        base_temp: float = 1.0,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
        exploration_moves: int = 30
    ):
        self.base_temp = base_temp
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.exploration_moves = exploration_moves

    def apply_noise(self, policy: list[float], move_number: int) -> list[float]:
        """Apply Dirichlet noise to policy for root exploration."""
        if move_number >= self.exploration_moves:
            return policy

        # Generate Dirichlet noise
        noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in policy]
        noise_sum = sum(noise)
        if noise_sum > 0:
            noise = [n / noise_sum for n in noise]

        # Mix policy with noise
        mixed = []
        for p, n in zip(policy, noise, strict=False):
            mixed_p = (1 - self.noise_fraction) * p + self.noise_fraction * n
            mixed.append(mixed_p)

        # Renormalize
        total = sum(mixed)
        if total > 0:
            mixed = [p / total for p in mixed]

        return mixed

    def apply_temperature(self, policy: list[float], move_number: int) -> list[float]:
        """Apply temperature to policy."""
        if move_number >= self.exploration_moves:
            # Argmax (greedy)
            max_idx = max(range(len(policy)), key=lambda i: policy[i])
            return [1.0 if i == max_idx else 0.0 for i in range(len(policy))]

        # Apply temperature
        temp_policy = [p ** (1.0 / self.base_temp) for p in policy]
        total = sum(temp_policy)
        if total > 0:
            temp_policy = [p / total for p in temp_policy]

        return temp_policy

    def process_policy(self, policy: list[float], move_number: int,
                       add_noise: bool = True) -> list[float]:
        """Process policy with noise and temperature."""
        if add_noise:
            policy = self.apply_noise(policy, move_number)
        return self.apply_temperature(policy, move_number)


def create_scheduler(
    preset: str = "default",
    config_key: str | None = None,
    auto_wire: bool = True,
    **kwargs,
) -> TemperatureScheduler:
    """Create a temperature scheduler from a preset.

    Args:
        preset: Preset name (default, alphazero, adaptive, curriculum, etc.)
        config_key: Optional config key for auto-registration and wiring.
            If provided, the scheduler will be registered and wired to
            receive exploration boost signals from FeedbackLoopController.
        auto_wire: If True and config_key is provided, automatically wire
            exploration boost subscription. Default True.
        **kwargs: Override any config attributes

    Returns:
        Configured TemperatureScheduler instance

    Example:
        >>> # Without auto-registration (legacy behavior)
        >>> scheduler = create_scheduler("adaptive")

        >>> # With auto-registration (recommended for selfplay)
        >>> scheduler = create_scheduler("adaptive", config_key="hex8_2p")
        >>> # Now automatically receives exploration boost from feedback loop

    December 2025: Added auto-registration to close the feedback loop gap.
    Previously, selfplay had to manually call register_active_scheduler()
    and wire_exploration_boost() which was often forgotten.
    """
    presets = {
        "default": TemperatureConfig(
            schedule_type=ScheduleType.LINEAR_DECAY,
            initial_temp=1.0,
            final_temp=0.1,
            decay_start_move=15,
            decay_end_move=60
        ),
        "alphazero": TemperatureConfig(
            schedule_type=ScheduleType.STEP,
            initial_temp=1.0,
            final_temp=0.0,
            step_schedule=[(0, 1.0), (30, 0.0)]
        ),
        "aggressive_exploration": TemperatureConfig(
            schedule_type=ScheduleType.LINEAR_DECAY,
            initial_temp=1.5,
            final_temp=0.2,
            decay_start_move=20,
            decay_end_move=80,
            add_noise=True,
            noise_scale=0.15
        ),
        "conservative": TemperatureConfig(
            schedule_type=ScheduleType.EXPONENTIAL_DECAY,
            initial_temp=0.8,
            final_temp=0.1,
            decay_start_move=5,
            decay_end_move=40
        ),
        "adaptive": TemperatureConfig(
            schedule_type=ScheduleType.ADAPTIVE,
            initial_temp=1.0,
            adaptive_config={
                'complexity_weight': 0.3,
                'uncertainty_weight': 0.3,
                'phase_weight': 0.4
            }
        ),
        "curriculum": TemperatureConfig(
            schedule_type=ScheduleType.CURRICULUM,
            initial_temp=1.5,
            final_temp=0.5,
            exploration_moves=40
        ),
        "cosine": TemperatureConfig(
            schedule_type=ScheduleType.COSINE,
            initial_temp=1.2,
            final_temp=0.1,
            decay_end_move=60
        ),
        "elo_adaptive": TemperatureConfig(
            schedule_type=ScheduleType.ELO_ADAPTIVE,
            initial_temp=1.0,  # Not used directly, but for consistency
            exploration_moves=30,
            adaptive_config={
                'model_elo': 1500.0,
                'weak_temp': 1.5,
                'medium_temp': 1.0,
                'strong_temp': 0.7,
                'very_strong_temp': 0.5,
                'elo_weak_threshold': 1300.0,
                'elo_medium_threshold': 1500.0,
                'elo_strong_threshold': 1700.0,
            }
        )
    }

    config = presets.get(preset, presets["default"])

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    scheduler = TemperatureScheduler(config)

    # December 2025: Auto-register and wire if config_key provided
    if config_key:
        register_active_scheduler(config_key, scheduler)
        if auto_wire:
            wired = wire_exploration_boost(scheduler, config_key)
            if wired:
                logger.debug(
                    f"[create_scheduler] Auto-wired {preset} scheduler for {config_key}"
                )

    return scheduler


def create_elo_adaptive_scheduler(
    model_elo: float,
    exploration_moves: int = 30,
    config_key: str | None = None,
    auto_wire: bool = True,
) -> TemperatureScheduler:
    """Create an Elo-adaptive temperature scheduler for a specific model Elo.

    This is a convenience function for the strength-driven training pipeline
    (December 2025). The scheduler automatically adjusts temperature based on
    model strength:
    - Weak models (< 1300 Elo): High exploration (temp ~1.5)
    - Medium models (1300-1500 Elo): Balanced (temp ~1.0)
    - Strong models (1500-1700 Elo): More exploitation (temp ~0.7)
    - Very strong (> 1700 Elo): Confident play (temp ~0.5)

    Args:
        model_elo: Current model Elo rating
        exploration_moves: Number of moves before temperature decay
        config_key: Optional config key for auto-registration and wiring.
            If provided, the scheduler will be registered and wired to
            receive exploration boost signals from FeedbackLoopController.
        auto_wire: If True and config_key is provided, automatically wire
            exploration boost subscription. Default True.

    Returns:
        TemperatureScheduler configured for Elo-adaptive temperature

    Example:
        >>> # Without auto-registration (legacy behavior)
        >>> scheduler = create_elo_adaptive_scheduler(model_elo=1450)
        >>> temp = scheduler.get_temperature(move_number=15)
        >>> print(f"Temperature: {temp:.2f}")
        Temperature: 1.25

        >>> # With auto-registration (recommended)
        >>> scheduler = create_elo_adaptive_scheduler(
        ...     model_elo=1450, config_key="hex8_2p"
        ... )
        >>> # Now automatically receives exploration boost from feedback loop
    """
    config = TemperatureConfig(
        schedule_type=ScheduleType.ELO_ADAPTIVE,
        exploration_moves=exploration_moves,
        adaptive_config={'model_elo': model_elo},
    )
    scheduler = TemperatureScheduler(config)

    # December 2025: Auto-register and wire if config_key provided
    if config_key:
        register_active_scheduler(config_key, scheduler)
        if auto_wire:
            wired = wire_exploration_boost(scheduler, config_key)
            if wired:
                logger.debug(
                    f"[create_elo_adaptive_scheduler] Auto-wired scheduler for "
                    f"{config_key} (Elo={model_elo:.0f})"
                )

    return scheduler


# =============================================================================
# Scheduler Registry (December 2025)
# Enables FeedbackLoopController to wire exploration boost to active schedulers
# =============================================================================

_active_schedulers: dict[str, "TemperatureScheduler"] = {}


def get_active_schedulers() -> dict[str, "TemperatureScheduler"]:
    """Get all active temperature schedulers by config_key.

    Returns:
        Dictionary mapping config_key (e.g., "hex8_2p") to TemperatureScheduler

    Example:
        >>> schedulers = get_active_schedulers()
        >>> for key, sched in schedulers.items():
        ...     print(f"{key}: boost={sched.get_exploration_boost():.2f}")
    """
    return dict(_active_schedulers)


def register_active_scheduler(config_key: str, scheduler: "TemperatureScheduler") -> None:
    """Register a temperature scheduler as active for a config.

    Called during selfplay initialization to make the scheduler available
    for exploration boost wiring from FeedbackLoopController.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        scheduler: TemperatureScheduler instance to register

    Example:
        >>> scheduler = create_scheduler("adaptive")
        >>> register_active_scheduler("hex8_2p", scheduler)
        >>> # Now FeedbackLoopController can wire exploration boost
    """
    _active_schedulers[config_key] = scheduler
    logger.debug(f"[TemperatureScheduler] Registered active scheduler for {config_key}")

    # December 2025: Emit event for lazy exploration boost wiring
    # This allows FeedbackLoopController to wire schedulers that register after startup
    _emit_scheduler_registered(config_key)


def _emit_scheduler_registered(config_key: str) -> None:
    """Emit SCHEDULER_REGISTERED event for lazy exploration boost wiring.

    This enables the feedback loop to wire exploration boost to schedulers
    that are created after FeedbackLoopController initializes.
    """
    try:
        from app.coordination.event_router import get_event_bus, RouterEvent, EventSource
        from app.distributed.data_events import DataEventType

        bus = get_event_bus()
        if bus is None:
            return

        # P0.6 Dec 2025: Use DataEventType enum for type-safe event emission
        event = RouterEvent(
            event_type=DataEventType.SCHEDULER_REGISTERED,
            payload={
                "config_key": config_key,
                "timestamp": time.time(),
            },
            source="temperature_scheduling",
            origin=EventSource.ROUTER,
        )

        # Fire-and-forget - use sync publish if available
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.publish(event))
        except RuntimeError:
            # No event loop running - try sync publish or skip
            if hasattr(bus, "publish_sync"):
                bus.publish_sync(event)
            else:
                logger.debug(
                    f"[TemperatureScheduler] Could not emit SCHEDULER_REGISTERED "
                    f"for {config_key} (no event loop)"
                )

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[TemperatureScheduler] Failed to emit SCHEDULER_REGISTERED: {e}")


def unregister_active_scheduler(config_key: str) -> None:
    """Unregister a temperature scheduler when selfplay completes.

    Args:
        config_key: Configuration key to unregister
    """
    if config_key in _active_schedulers:
        del _active_schedulers[config_key]
        logger.debug(f"[TemperatureScheduler] Unregistered scheduler for {config_key}")


def wire_exploration_boost(
    scheduler: TemperatureScheduler,
    config_key: str,
) -> bool:
    """Wire a temperature scheduler to receive exploration boost from FeedbackLoopController.

    December 2025: This connects the feedback loop to temperature scheduling,
    enabling automatic exploration adjustment based on promotion success/failure.

    Args:
        scheduler: TemperatureScheduler instance to receive boost updates
        config_key: Configuration key (e.g., "hex8_2p") to subscribe to

    Returns:
        True if successfully wired, False if event system not available

    Example:
        >>> scheduler = create_scheduler("adaptive")
        >>> wire_exploration_boost(scheduler, "hex8_2p")
        True
        >>> # Now scheduler will automatically adjust exploration based on feedback
    """
    try:
        from app.coordination.event_router import get_event_bus
        from app.distributed.data_events import DataEventType

        bus = get_event_bus()
        if bus is None:
            logger.debug("[TemperatureScheduler] Event bus not available for boost wiring")
            return False

        def on_exploration_boost_updated(event):
            """Handle EXPLORATION_BOOST_UPDATED from FeedbackLoopController."""
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config", "")

            if event_config != config_key:
                return

            boost = payload.get("exploration_boost", 1.0)
            scheduler.set_exploration_boost(boost)

        def on_promotion_failed(event):
            """Handle PROMOTION_FAILED - increase exploration."""
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config_key", "")

            if event_config != config_key:
                return

            # Increase exploration on failure (capped at 2.0)
            current_boost = scheduler.get_exploration_boost()
            new_boost = min(2.0, current_boost * 1.2)
            scheduler.set_exploration_boost(new_boost)
            logger.info(
                f"[TemperatureScheduler] Promotion failed for {config_key}, "
                f"boost increased to {new_boost:.2f}"
            )

        def on_model_promoted(event):
            """Handle MODEL_PROMOTED - decrease exploration."""
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config_key", "")

            if event_config != config_key:
                return

            # Reset exploration on success
            scheduler.set_exploration_boost(1.0)
            logger.info(
                f"[TemperatureScheduler] Promotion succeeded for {config_key}, "
                f"boost reset to 1.0"
            )

        def on_training_loss_trend(event):
            """Handle TRAINING_LOSS_TREND - adjust exploration based on training progress.

            Phase 21.2 (Dec 2025): This completes the TRAINING_LOSS → DATA_COLLECTION
            feedback loop. When training is stalled or degrading, we increase exploration
            to generate more diverse training games.

            Trend effects:
            - "improving": Keep current boost (training is working)
            - "stalled": Increase exploration by 10% (need more diverse data)
            - "degrading": Increase exploration by 20% (need significantly different data)
            """
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config_key", "")

            if event_config != config_key:
                return

            trend = payload.get("trend", "unknown")
            current_boost = scheduler.get_exploration_boost()

            if trend == "stalled":
                # Training is stalled - increase exploration to generate diverse games
                new_boost = min(2.0, current_boost * 1.1)
                scheduler.set_exploration_boost(new_boost)
                logger.info(
                    f"[TemperatureScheduler] Training stalled for {config_key}, "
                    f"boost increased to {new_boost:.2f}"
                )

            elif trend == "degrading":
                # Training is getting worse - significantly increase exploration
                new_boost = min(2.0, current_boost * 1.2)
                scheduler.set_exploration_boost(new_boost)
                logger.warning(
                    f"[TemperatureScheduler] Training degrading for {config_key}, "
                    f"boost increased to {new_boost:.2f}"
                )

            elif trend == "improving":
                # Training is improving - gradually reduce exploration boost
                if current_boost > 1.0:
                    new_boost = max(1.0, current_boost * 0.95)
                    scheduler.set_exploration_boost(new_boost)
                    logger.debug(
                        f"[TemperatureScheduler] Training improving for {config_key}, "
                        f"boost reduced to {new_boost:.2f}"
                    )

        def on_elo_significant_change(event):
            """Handle ELO_SIGNIFICANT_CHANGE - update Elo-adaptive temperature.

            December 2025: This completes the Elo → Temperature feedback loop.
            When model Elo changes significantly, we update the temperature
            scheduler to reflect the new model strength.

            - Elo increase: Reduce temperature (more confident play)
            - Elo decrease: Increase temperature (more exploration)
            """
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config", payload.get("config_key", ""))

            if event_config != config_key:
                return

            new_elo = payload.get("current_elo", payload.get("new_elo"))
            elo_delta = payload.get("elo_delta", payload.get("delta", 0))
            elo_velocity = payload.get("velocity", 0)

            if new_elo is None:
                return

            # Update EloAdaptiveSchedule if that's what we're using
            if isinstance(scheduler.schedule, EloAdaptiveSchedule):
                old_elo = scheduler.schedule.model_elo
                scheduler.schedule.update_elo(new_elo)
                logger.info(
                    f"[TemperatureScheduler] Elo updated for {config_key}: "
                    f"{old_elo:.0f} → {new_elo:.0f} (delta={elo_delta:+.0f})"
                )

                # Adjust exploration boost based on Elo velocity
                # High velocity (rapid improvement) = reduce exploration
                # Low velocity (stagnation) = increase exploration
                if elo_velocity is not None:
                    current_boost = scheduler.get_exploration_boost()
                    if elo_velocity > 2.0:
                        # Rapid improvement - reduce exploration
                        new_boost = max(0.7, current_boost * 0.9)
                        scheduler.set_exploration_boost(new_boost)
                        logger.info(
                            f"[TemperatureScheduler] High Elo velocity ({elo_velocity:.1f}/game) "
                            f"for {config_key}, boost reduced to {new_boost:.2f}"
                        )
                    elif elo_velocity < 0.5 and elo_velocity >= 0:
                        # Stagnation - increase exploration
                        new_boost = min(1.5, current_boost * 1.1)
                        scheduler.set_exploration_boost(new_boost)
                        logger.info(
                            f"[TemperatureScheduler] Low Elo velocity ({elo_velocity:.1f}/game) "
                            f"for {config_key}, boost increased to {new_boost:.2f}"
                        )
            else:
                # For non-Elo-adaptive schedules, just adjust exploration boost
                # based on Elo velocity
                if elo_velocity is not None and elo_velocity > 2.0:
                    new_boost = max(0.7, scheduler.get_exploration_boost() * 0.9)
                    scheduler.set_exploration_boost(new_boost)
                elif elo_velocity is not None and 0 <= elo_velocity < 0.5:
                    new_boost = min(1.5, scheduler.get_exploration_boost() * 1.1)
                    scheduler.set_exploration_boost(new_boost)

        def on_regression_detected(event):
            """Handle REGRESSION_DETECTED - significantly increase exploration.

            December 2025: This closes the REGRESSION_DETECTED → EXPLORATION_BOOST
            feedback loop. When model regression is detected, we increase exploration
            to generate more diverse training data to help the model recover.

            Severity effects:
            - MINOR: Increase exploration by 15%
            - MODERATE: Increase exploration by 25%
            - SEVERE/CRITICAL: Increase exploration by 40%
            """
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config_key", payload.get("config", ""))

            if event_config != config_key:
                return

            severity = payload.get("severity", "MODERATE")
            current_boost = scheduler.get_exploration_boost()

            # Determine boost multiplier based on severity
            severity_multipliers = {
                "MINOR": 1.15,
                "MODERATE": 1.25,
                "SEVERE": 1.40,
                "CRITICAL": 1.40,
            }
            multiplier = severity_multipliers.get(severity.upper(), 1.25)

            new_boost = min(2.0, current_boost * multiplier)
            scheduler.set_exploration_boost(new_boost)
            logger.warning(
                f"[TemperatureScheduler] Regression detected for {config_key} "
                f"(severity={severity}), boost increased {current_boost:.2f} → {new_boost:.2f}"
            )

        def on_exploration_boost(event):
            """Handle EXPLORATION_BOOST - directly apply exploration boost.

            P1.4 Dec 2025: This closes the EXPLORATION_BOOST → temperature adjustment
            feedback loop. When curriculum or quality systems request exploration boost,
            we apply it directly to the temperature scheduler.
            """
            payload = event.payload if hasattr(event, "payload") else event
            event_config = payload.get("config_key", payload.get("config", ""))

            if event_config != config_key:
                return

            boost_factor = payload.get("boost_factor", 1.3)
            current_boost = scheduler.get_exploration_boost()
            new_boost = min(2.0, current_boost * boost_factor)
            scheduler.set_exploration_boost(new_boost)
            logger.info(
                f"[TemperatureScheduler] Exploration boost applied for {config_key}: "
                f"{current_boost:.2f} → {new_boost:.2f} (factor={boost_factor})"
            )

        # Subscribe to feedback events
        bus.subscribe(DataEventType.PROMOTION_FAILED, on_promotion_failed)
        bus.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)
        bus.subscribe(DataEventType.TRAINING_LOSS_TREND, on_training_loss_trend)
        bus.subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, on_elo_significant_change)
        bus.subscribe(DataEventType.REGRESSION_DETECTED, on_regression_detected)
        # P1.4 Dec 2025: Subscribe to EXPLORATION_BOOST for direct temperature adjustment
        if hasattr(DataEventType, 'EXPLORATION_BOOST'):
            bus.subscribe(DataEventType.EXPLORATION_BOOST, on_exploration_boost)

        logger.debug(
            f"[TemperatureScheduler] Wired exploration boost, Elo updates, regression "
            f"detection, and EXPLORATION_BOOST for {config_key}"
        )
        return True

    except ImportError as e:
        logger.debug(f"[TemperatureScheduler] Event system not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"[TemperatureScheduler] Failed to wire exploration boost: {e}")
        return False


def main() -> None:
    """Demonstrate temperature scheduling."""
    import matplotlib.pyplot as plt

    # Create different schedulers
    schedulers = {
        'Linear Decay': create_scheduler('default'),
        'AlphaZero': create_scheduler('alphazero'),
        'Aggressive': create_scheduler('aggressive_exploration'),
        'Adaptive': create_scheduler('adaptive'),
        'Cosine': create_scheduler('cosine'),
        'Curriculum (early)': create_scheduler('curriculum'),
        'Curriculum (late)': create_scheduler('curriculum'),
    }

    # Set training progress for curriculum schedulers
    schedulers['Curriculum (early)'].set_training_progress(0.1)
    schedulers['Curriculum (late)'].set_training_progress(0.9)

    # Generate temperature curves
    moves = list(range(100))
    plt.figure(figsize=(12, 6))

    for name, scheduler in schedulers.items():
        temps = [scheduler.get_temperature(m) for m in moves]
        plt.plot(moves, temps, label=name, linewidth=2)

    plt.xlabel('Move Number')
    plt.ylabel('Temperature')
    plt.title('Temperature Schedules Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('temperature_schedules.png', dpi=150)
    print("Saved temperature_schedules.png")

    # Demonstrate AlphaZero-style temperature
    print("\nAlphaZero-style Temperature Demo:")
    az_temp = AlphaZeroTemperature(exploration_moves=30)

    # Sample policy
    policy = [0.1, 0.3, 0.2, 0.15, 0.25]
    print(f"Original policy: {policy}")

    for move in [0, 15, 29, 30, 50]:
        temp = az_temp.get_temperature(move)
        selected = az_temp.sample_move(policy, move)
        print(f"Move {move}: temp={temp:.2f}, selected={selected}")

    # Demonstrate Dirichlet noise
    print("\nDirichlet Noise Demo:")
    dirichlet = DirichletNoiseTemperature()

    policy = [0.1, 0.5, 0.2, 0.15, 0.05]
    print(f"Original policy: {[f'{p:.3f}' for p in policy]}")

    noisy = dirichlet.process_policy(policy, move_number=10, add_noise=True)
    print(f"With noise (move 10): {[f'{p:.3f}' for p in noisy]}")

    noisy = dirichlet.process_policy(policy, move_number=50, add_noise=True)
    print(f"Late game (move 50): {[f'{p:.3f}' for p in noisy]}")


if __name__ == "__main__":
    main()
