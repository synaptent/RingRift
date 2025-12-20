"""
Feedback Loops for the Pipeline Orchestrator.

This module provides closed-loop feedback mechanisms between pipeline stages
to enable intelligent adaptation of the training process.

Feedback Loops Implemented:
1. Evaluation → Curriculum: Adjust training focus based on weaknesses
2. Parity Failures → Selfplay: Adjust parameters on validation failures
3. Training Loss → Data Collection: Increase data when training struggles
4. Elo Trend → Optimization: Trigger CMA-ES/NAS on plateaus
5. Data Quality → Quarantine: Isolate problematic games

Integration:
    from app.integration.pipeline_feedback import PipelineFeedbackController

    # In PipelineOrchestrator.__init__:
    self.feedback = PipelineFeedbackController(self)

    # After each stage:
    await self.feedback.on_stage_complete(stage_name, result)
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackAction(Enum):
    """Actions that feedback can trigger."""
    INCREASE_DATA_COLLECTION = "increase_data"
    DECREASE_DATA_COLLECTION = "decrease_data"
    INCREASE_CURRICULUM_WEIGHT = "increase_weight"
    DECREASE_CURRICULUM_WEIGHT = "decrease_weight"
    TRIGGER_CMAES = "trigger_cmaes"
    TRIGGER_NAS = "trigger_nas"
    QUARANTINE_DATA = "quarantine_data"
    ADJUST_TEMPERATURE = "adjust_temperature"
    EXTEND_TRAINING = "extend_training"
    REDUCE_TRAINING = "reduce_training"
    NO_ACTION = "no_action"
    # Promotion-related actions
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_SUCCEEDED = "promotion_succeeded"
    URGENT_RETRAINING = "urgent_retraining"
    # Utilization-related actions (target 60-80% CPU/GPU)
    SCALE_UP_SELFPLAY = "scale_up_selfplay"
    SCALE_DOWN_SELFPLAY = "scale_down_selfplay"


@dataclass
class FeedbackSignal:
    """A feedback signal from one pipeline stage to another."""
    source_stage: str
    target_stage: str
    action: FeedbackAction
    magnitude: float  # 0.0 to 1.0
    reason: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackState:
    """Persistent state for feedback system."""
    # Curriculum weights by config (config_key -> weight multiplier)
    curriculum_weights: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))

    # Temperature adjustments
    temperature_multiplier: float = 1.0

    # Training parameters
    epochs_multiplier: float = 1.0
    batch_size_multiplier: float = 1.0

    # Data collection parameters
    games_per_worker_multiplier: float = 1.0

    # Elo tracking for plateau detection
    elo_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, elo)
    plateau_count: int = 0

    # Parity tracking
    parity_failure_rate: float = 0.0
    consecutive_parity_failures: int = 0

    # Quality tracking
    quarantined_games: int = 0
    data_quality_score: float = 1.0

    # CMA-ES/NAS tracking
    last_cmaes_trigger: float = 0.0
    last_nas_trigger: float = 0.0
    cmaes_cooldown_hours: float = 6.0
    nas_cooldown_hours: float = 24.0

    # Promotion tracking
    consecutive_promotion_failures: int = 0
    last_promotion_success: float = 0.0
    promotion_failure_configs: dict[str, int] = field(default_factory=dict)  # config -> failure count


class EvaluationAnalyzer:
    """Analyzes evaluation results to identify weaknesses."""

    def __init__(self):
        self.win_rates: dict[str, list[float]] = defaultdict(list)
        self.elo_trends: dict[str, list[float]] = defaultdict(list)

    def add_result(self, config_key: str, win_rate: float, elo: float):
        """Add an evaluation result."""
        self.win_rates[config_key].append(win_rate)
        self.elo_trends[config_key].append(elo)

        # Keep only last 20 results
        if len(self.win_rates[config_key]) > 20:
            self.win_rates[config_key] = self.win_rates[config_key][-20:]
        if len(self.elo_trends[config_key]) > 20:
            self.elo_trends[config_key] = self.elo_trends[config_key][-20:]

    def get_weak_configs(self, threshold: float = 0.45) -> list[str]:
        """Get configurations with below-threshold win rate."""
        weak = []
        for config_key, rates in self.win_rates.items():
            if len(rates) >= 3:  # Need at least 3 samples
                avg_rate = sum(rates[-5:]) / len(rates[-5:])
                if avg_rate < threshold:
                    weak.append(config_key)
        return weak

    def get_elo_trend(self, config_key: str, lookback: int = 5) -> float:
        """Get Elo trend (positive = improving)."""
        elos = self.elo_trends.get(config_key, [])
        if len(elos) < lookback:
            return 0.0

        recent = elos[-lookback:]
        return recent[-1] - recent[0]

    def is_plateau(self, config_key: str, min_improvement: float = 15.0, lookback: int = 5) -> bool:
        """Check if config is in a plateau."""
        trend = self.get_elo_trend(config_key, lookback)
        return trend < min_improvement


class DataQualityMonitor:
    """Monitors data quality and identifies problematic games."""

    def __init__(self):
        self.parity_results: list[bool] = []  # True = passed, False = failed
        self.game_lengths: list[int] = []
        self.outlier_games: list[str] = []

    def add_parity_result(self, passed: bool, game_id: str | None = None):
        """Record a parity validation result."""
        self.parity_results.append(passed)

        if not passed and game_id:
            self.outlier_games.append(game_id)

        # Keep bounded
        if len(self.parity_results) > 1000:
            self.parity_results = self.parity_results[-1000:]

    def get_parity_failure_rate(self) -> float:
        """Get recent parity failure rate."""
        if not self.parity_results:
            return 0.0

        recent = self.parity_results[-100:]
        return 1.0 - (sum(recent) / len(recent))

    def should_quarantine(self, failure_threshold: float = 0.1) -> bool:
        """Check if data collection should be paused due to quality issues."""
        return self.get_parity_failure_rate() > failure_threshold


class TrainingMonitor:
    """Monitors training progress and identifies issues."""

    def __init__(self):
        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.learning_rate_history: list[float] = []

    def add_training_metrics(self, loss: float, val_loss: float | None = None,
                             learning_rate: float | None = None):
        """Record training metrics."""
        self.loss_history.append(loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        if learning_rate is not None:
            self.learning_rate_history.append(learning_rate)

    def is_loss_plateau(self, lookback: int = 10, threshold: float = 0.01) -> bool:
        """Check if loss has plateaued."""
        if len(self.loss_history) < lookback:
            return False

        recent = self.loss_history[-lookback:]
        improvement = recent[0] - recent[-1]
        return improvement < threshold

    def is_overfitting(self) -> bool:
        """Check if model is overfitting (val_loss increasing while train_loss decreasing)."""
        if len(self.loss_history) < 5 or len(self.val_loss_history) < 5:
            return False

        train_trend = self.loss_history[-1] - self.loss_history[-5]
        val_trend = self.val_loss_history[-1] - self.val_loss_history[-5]

        return train_trend < 0 and val_trend > 0


class PipelineFeedbackController:
    """
    Main controller for pipeline feedback loops.

    Monitors pipeline stages and generates feedback signals to adjust
    parameters for subsequent stages.
    """

    def __init__(
        self,
        state_path: Path | None = None,
        config: dict[str, Any] | None = None
    ):
        self.config = config or {}

        # State persistence
        self.state_path = state_path
        self.state = FeedbackState()
        self._load_state()

        # Analyzers
        self.eval_analyzer = EvaluationAnalyzer()
        self.data_monitor = DataQualityMonitor()
        self.training_monitor = TrainingMonitor()

        # Signal history
        self.signals: list[FeedbackSignal] = []
        self._max_signals = 100

    def _load_state(self):
        """Load state from disk."""
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                # Restore state fields
                for key, value in data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
            except Exception as e:
                logger.warning(f"Failed to load feedback state: {e}")

    def _save_state(self):
        """Save state to disk."""
        if self.state_path:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_path, 'w') as f:
                    json.dump({
                        'curriculum_weights': dict(self.state.curriculum_weights),
                        'temperature_multiplier': self.state.temperature_multiplier,
                        'epochs_multiplier': self.state.epochs_multiplier,
                        'games_per_worker_multiplier': self.state.games_per_worker_multiplier,
                        'elo_history': self.state.elo_history[-50:],
                        'plateau_count': self.state.plateau_count,
                        'parity_failure_rate': self.state.parity_failure_rate,
                        'last_cmaes_trigger': self.state.last_cmaes_trigger,
                        'last_nas_trigger': self.state.last_nas_trigger,
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save feedback state: {e}")

    def _emit_signal(self, signal: FeedbackSignal):
        """Record and potentially act on a feedback signal."""
        self.signals.append(signal)
        if len(self.signals) > self._max_signals:
            self.signals = self.signals[-self._max_signals:]

        logger.info(f"Feedback: {signal.source_stage} → {signal.target_stage}: "
                    f"{signal.action.value} (magnitude={signal.magnitude:.2f}, reason={signal.reason})")

    # =========================================================================
    # Stage Completion Handlers
    # =========================================================================

    async def on_stage_complete(self, stage: str, result: dict[str, Any]):
        """Handle completion of a pipeline stage."""
        handlers = {
            'evaluation': self._on_evaluation_complete,
            'training': self._on_training_complete,
            'parity-validation': self._on_parity_validation_complete,
            'selfplay': self._on_selfplay_complete,
            'canonical-selfplay': self._on_selfplay_complete,
            'cmaes': self._on_cmaes_complete,
            'promotion': self._on_promotion_complete,
            'utilization': self._on_utilization_complete,
        }

        handler = handlers.get(stage)
        if handler:
            await handler(result)

        self._save_state()

    async def on_stage_failed(self, stage: str, result: dict[str, Any]):
        """Handle failure of a pipeline stage.

        This method is called when a stage fails (error, timeout, etc.) to:
        1. Track failure patterns
        2. Adjust parameters to prevent repeated failures
        3. Emit signals for recovery actions

        Args:
            stage: Name of the failed stage (training, evaluation, etc.)
            result: Dictionary with failure details (config_key, error, duration, etc.)
        """
        config_key = result.get('config_key', result.get('config', 'unknown'))
        error = result.get('error', 'unknown error')
        duration = result.get('duration', 0)

        logger.warning(f"Stage '{stage}' failed for {config_key}: {error}")

        # Track failures in state
        if not hasattr(self.state, 'failure_counts'):
            self.state.failure_counts = defaultdict(int)
        if not hasattr(self.state, 'consecutive_failures'):
            self.state.consecutive_failures = defaultdict(int)

        failure_key = f"{stage}:{config_key}"
        self.state.failure_counts[failure_key] += 1
        self.state.consecutive_failures[stage] += 1

        # Stage-specific failure handling
        if stage == 'training':
            await self._on_training_failed(config_key, error, duration)
        elif stage == 'evaluation':
            await self._on_evaluation_failed(config_key, error)
        elif stage == 'promotion':
            await self._on_promotion_failed(config_key, error)
        elif stage in ('selfplay', 'canonical-selfplay'):
            await self._on_selfplay_failed(config_key, error)
        elif stage == 'parity-validation':
            await self._on_parity_failed(config_key, error)

        self._save_state()

    async def _on_training_failed(self, config_key: str, error: str, duration: float):
        """Handle training failure - potentially adjust data or retry."""
        # If training fails repeatedly, might need more data diversity
        failure_key = f"training:{config_key}"
        if self.state.failure_counts.get(failure_key, 0) >= 2:
            self._emit_signal(FeedbackSignal(
                source_stage='training',
                target_stage='data_collection',
                action=FeedbackAction.INCREASE_DATA_COLLECTION,
                magnitude=0.3,
                reason=f"Repeated training failures for {config_key} - need more diverse data",
                metadata={'config': config_key, 'error': str(error)[:200]}
            ))

        # If OOM or similar, reduce batch size
        if 'memory' in error.lower() or 'oom' in error.lower() or 'cuda' in error.lower():
            self.state.batch_size_multiplier = max(0.5, self.state.batch_size_multiplier * 0.8)
            logger.info(f"Reduced batch size multiplier to {self.state.batch_size_multiplier} due to memory error")

    async def _on_evaluation_failed(self, config_key: str, error: str):
        """Handle evaluation failure - skip model, potentially adjust eval parameters."""
        # Track model as problematic
        config_key.split('/')[-1] if '/' in config_key else config_key

        # If evaluation times out repeatedly, reduce eval games
        if 'timeout' in error.lower():
            self._emit_signal(FeedbackSignal(
                source_stage='evaluation',
                target_stage='evaluation',
                action=FeedbackAction.REDUCE_TRAINING,  # Using as proxy for reduce_eval
                magnitude=0.2,
                reason=f"Evaluation timeout for {config_key}",
                metadata={'config': config_key}
            ))

    async def _on_promotion_failed(self, config_key: str, error: str):
        """Handle promotion failure - emit signal for retry or investigation."""
        self._emit_signal(FeedbackSignal(
            source_stage='promotion',
            target_stage='training',
            action=FeedbackAction.PROMOTION_FAILED,
            magnitude=0.5,
            reason=f"Promotion failed for {config_key}: {error[:100]}",
            metadata={'config': config_key, 'error': str(error)[:200]}
        ))

    async def _on_selfplay_failed(self, config_key: str, error: str):
        """Handle selfplay failure - adjust worker parameters."""
        # If workers are failing, might need to reduce parallelism
        if self.state.consecutive_failures.get('selfplay', 0) >= 3:
            self.state.games_per_worker_multiplier = max(0.5, self.state.games_per_worker_multiplier * 0.8)
            logger.info(f"Reduced games_per_worker to {self.state.games_per_worker_multiplier} due to repeated selfplay failures")

    async def _on_parity_failed(self, config_key: str, error: str):
        """Handle parity validation failure - quarantine data source."""
        self.state.consecutive_parity_failures += 1

        if self.state.consecutive_parity_failures >= 3:
            self._emit_signal(FeedbackSignal(
                source_stage='parity-validation',
                target_stage='data_collection',
                action=FeedbackAction.QUARANTINE_DATA,
                magnitude=0.8,
                reason=f"Repeated parity failures - quarantine data from {config_key}",
                metadata={'config': config_key}
            ))

    def reset_consecutive_failures(self, stage: str):
        """Reset consecutive failure count for a stage after success."""
        if hasattr(self.state, 'consecutive_failures'):
            self.state.consecutive_failures[stage] = 0

    async def _on_evaluation_complete(self, result: dict[str, Any]):
        """Handle evaluation completion - adjust curriculum weights."""
        config_key = result.get('config_key', 'default')
        win_rate = result.get('win_rate')
        elo = result.get('elo')

        if win_rate is not None:
            self.eval_analyzer.add_result(config_key, win_rate, elo or 1500)

        if elo is not None:
            self.state.elo_history.append((time.time(), elo))
            if len(self.state.elo_history) > 100:
                self.state.elo_history = self.state.elo_history[-100:]

        # Check for weak configurations
        weak_configs = self.eval_analyzer.get_weak_configs()
        for weak_config in weak_configs:
            # Increase curriculum weight for weak configs
            current_weight = self.state.curriculum_weights.get(weak_config, 1.0)
            new_weight = min(2.0, current_weight * 1.2)  # Cap at 2x
            self.state.curriculum_weights[weak_config] = new_weight

            self._emit_signal(FeedbackSignal(
                source_stage='evaluation',
                target_stage='training',
                action=FeedbackAction.INCREASE_CURRICULUM_WEIGHT,
                magnitude=0.2,
                reason=f"Low win rate for {weak_config}",
                metadata={'config': weak_config, 'new_weight': new_weight}
            ))

        # Check for plateau
        if self.eval_analyzer.is_plateau(config_key):
            self.state.plateau_count += 1

            if self._should_trigger_cmaes():
                self._emit_signal(FeedbackSignal(
                    source_stage='evaluation',
                    target_stage='cmaes',
                    action=FeedbackAction.TRIGGER_CMAES,
                    magnitude=1.0,
                    reason=f"Plateau detected (count={self.state.plateau_count})"
                ))
                self.state.last_cmaes_trigger = time.time()

            if self.state.plateau_count >= 3 and self._should_trigger_nas():
                self._emit_signal(FeedbackSignal(
                    source_stage='evaluation',
                    target_stage='nas',
                    action=FeedbackAction.TRIGGER_NAS,
                    magnitude=1.0,
                    reason="Severe plateau detected"
                ))
                self.state.last_nas_trigger = time.time()
        else:
            self.state.plateau_count = max(0, self.state.plateau_count - 1)

    async def _on_training_complete(self, result: dict[str, Any]):
        """Handle training completion - analyze loss patterns."""
        final_loss = result.get('final_loss')
        val_loss = result.get('val_loss')
        result.get('epochs', 0)

        if final_loss is not None:
            self.training_monitor.add_training_metrics(final_loss, val_loss)

        # Check for overfitting
        if self.training_monitor.is_overfitting():
            self._emit_signal(FeedbackSignal(
                source_stage='training',
                target_stage='data_collection',
                action=FeedbackAction.INCREASE_DATA_COLLECTION,
                magnitude=0.3,
                reason="Overfitting detected - need more diverse data"
            ))
            self.state.games_per_worker_multiplier = min(2.0, self.state.games_per_worker_multiplier * 1.2)

        # Check for loss plateau
        if self.training_monitor.is_loss_plateau():
            self._emit_signal(FeedbackSignal(
                source_stage='training',
                target_stage='training',
                action=FeedbackAction.ADJUST_TEMPERATURE,
                magnitude=0.2,
                reason="Loss plateau - increasing exploration"
            ))
            self.state.temperature_multiplier = min(1.5, self.state.temperature_multiplier * 1.1)

    async def _on_parity_validation_complete(self, result: dict[str, Any]):
        """Handle parity validation - adjust selfplay parameters."""
        passed = result.get('passed', 0)
        failed = result.get('failed', 0)
        total = passed + failed

        if total > 0:
            for _ in range(passed):
                self.data_monitor.add_parity_result(True)
            for _ in range(failed):
                self.data_monitor.add_parity_result(False)

        self.state.parity_failure_rate = self.data_monitor.get_parity_failure_rate()

        if self.state.parity_failure_rate > 0.1:  # More than 10% failures
            self.state.consecutive_parity_failures += 1

            self._emit_signal(FeedbackSignal(
                source_stage='parity-validation',
                target_stage='selfplay',
                action=FeedbackAction.QUARANTINE_DATA,
                magnitude=self.state.parity_failure_rate,
                reason=f"High parity failure rate: {self.state.parity_failure_rate:.1%}"
            ))

            # Reduce data collection rate if persistent issues
            if self.state.consecutive_parity_failures >= 3:
                self.state.games_per_worker_multiplier = max(0.5, self.state.games_per_worker_multiplier * 0.8)
                self._emit_signal(FeedbackSignal(
                    source_stage='parity-validation',
                    target_stage='selfplay',
                    action=FeedbackAction.DECREASE_DATA_COLLECTION,
                    magnitude=0.2,
                    reason="Persistent parity issues - reducing data rate"
                ))
        else:
            self.state.consecutive_parity_failures = 0

    async def _on_selfplay_complete(self, result: dict[str, Any]):
        """Handle selfplay completion - track data generation."""
        games_generated = result.get('games', 0)
        config_key = result.get('config_key')

        if config_key and games_generated > 0:
            # Could track per-config generation rates here
            pass

    async def _on_cmaes_complete(self, result: dict[str, Any]):
        """Handle CMA-ES completion - track optimization results."""
        improved = result.get('improved', False)

        if improved:
            # Reset plateau count on successful optimization
            self.state.plateau_count = 0

    async def _on_promotion_complete(self, result: dict[str, Any]):
        """Handle model promotion completion - track success/failure and adjust training.

        If promotion fails repeatedly for a config, increase its curriculum weight
        and potentially trigger urgent retraining or hyperparameter optimization.
        """
        config_key = result.get('config_key', 'default')
        success = result.get('success', False)
        elo_gain = result.get('elo_gain', 0)
        model_id = result.get('model_id', 'unknown')
        reason = result.get('reason', '')

        if success:
            # Promotion succeeded
            self.state.consecutive_promotion_failures = 0
            self.state.last_promotion_success = time.time()

            # Clear failure count for this config
            if config_key in self.state.promotion_failure_configs:
                del self.state.promotion_failure_configs[config_key]

            # Decrease plateau count on successful promotion
            self.state.plateau_count = max(0, self.state.plateau_count - 1)

            self._emit_signal(FeedbackSignal(
                source_stage='promotion',
                target_stage='training',
                action=FeedbackAction.PROMOTION_SUCCEEDED,
                magnitude=min(1.0, elo_gain / 50.0),  # Normalize by expected gain
                reason=f"Model {model_id} promoted (+{elo_gain} Elo)",
                metadata={'config': config_key, 'elo_gain': elo_gain, 'model_id': model_id}
            ))
        else:
            # Promotion failed
            self.state.consecutive_promotion_failures += 1

            # Track per-config failure count
            self.state.promotion_failure_configs[config_key] = (
                self.state.promotion_failure_configs.get(config_key, 0) + 1
            )
            config_failures = self.state.promotion_failure_configs[config_key]

            self._emit_signal(FeedbackSignal(
                source_stage='promotion',
                target_stage='training',
                action=FeedbackAction.PROMOTION_FAILED,
                magnitude=0.3 + (0.1 * min(config_failures, 5)),  # Increase urgency with failures
                reason=f"Promotion failed for {config_key}: {reason}",
                metadata={
                    'config': config_key,
                    'model_id': model_id,
                    'consecutive_failures': self.state.consecutive_promotion_failures,
                    'config_failures': config_failures,
                }
            ))

            # Increase curriculum weight for repeatedly failing configs
            if config_failures >= 2:
                current_weight = self.state.curriculum_weights.get(config_key, 1.0)
                new_weight = min(2.5, current_weight * 1.3)  # More aggressive increase
                self.state.curriculum_weights[config_key] = new_weight

                self._emit_signal(FeedbackSignal(
                    source_stage='promotion',
                    target_stage='training',
                    action=FeedbackAction.INCREASE_CURRICULUM_WEIGHT,
                    magnitude=0.3,
                    reason=f"Promotion failed {config_failures}x for {config_key}",
                    metadata={'config': config_key, 'new_weight': new_weight}
                ))

            # Trigger urgent retraining after 3 consecutive failures
            if self.state.consecutive_promotion_failures >= 3:
                self._emit_signal(FeedbackSignal(
                    source_stage='promotion',
                    target_stage='training',
                    action=FeedbackAction.URGENT_RETRAINING,
                    magnitude=1.0,
                    reason=f"{self.state.consecutive_promotion_failures} consecutive promotion failures",
                    metadata={'configs_affected': list(self.state.promotion_failure_configs.keys())}
                ))

                # Increase epochs for more thorough training
                self.state.epochs_multiplier = min(1.5, self.state.epochs_multiplier * 1.2)

            # Trigger CMA-ES after 5 consecutive failures
            if self.state.consecutive_promotion_failures >= 5 and self._should_trigger_cmaes():
                self._emit_signal(FeedbackSignal(
                    source_stage='promotion',
                    target_stage='cmaes',
                    action=FeedbackAction.TRIGGER_CMAES,
                    magnitude=1.0,
                    reason="5+ consecutive promotion failures, trying hyperparameter optimization"
                ))
                self.state.last_cmaes_trigger = time.time()

    async def _on_utilization_complete(self, result: dict[str, Any]):
        """Handle utilization report - adjust selfplay rate for optimal throughput.

        This feedback helps maintain 60-80% CPU/GPU utilization for maximum
        AI training throughput. Underutilization wastes capacity; overutilization
        causes throttling and OOM errors.
        """
        cpu_util = result.get('cpu_util', 0.0)
        gpu_util = result.get('gpu_util', 0.0)
        in_target_range = result.get('in_target_range', True)
        total_jobs = result.get('total_jobs', 0)

        # Track utilization history
        if not hasattr(self.state, 'utilization_history'):
            self.state.utilization_history = []

        self.state.utilization_history.append({
            'timestamp': time.time(),
            'cpu_util': cpu_util,
            'gpu_util': gpu_util,
        })
        # Keep last 100 samples
        if len(self.state.utilization_history) > 100:
            self.state.utilization_history = self.state.utilization_history[-100:]

        if not in_target_range:
            # Determine which direction is needed
            if cpu_util < 60 or gpu_util < 60:
                # Underutilized - increase selfplay to generate more data
                self._emit_signal(FeedbackSignal(
                    source_stage='utilization',
                    target_stage='selfplay',
                    action=FeedbackAction.SCALE_UP_SELFPLAY,
                    magnitude=min(1.0, (60 - min(cpu_util, gpu_util)) / 30),  # 0-1 based on gap
                    reason=f"Underutilized: CPU={cpu_util:.1f}%, GPU={gpu_util:.1f}% (target 60-80%)",
                    metadata={
                        'cpu_util': cpu_util,
                        'gpu_util': gpu_util,
                        'total_jobs': total_jobs,
                    }
                ))
            elif cpu_util > 80 or gpu_util > 80:
                # Overutilized - reduce selfplay to prevent throttling
                self._emit_signal(FeedbackSignal(
                    source_stage='utilization',
                    target_stage='selfplay',
                    action=FeedbackAction.SCALE_DOWN_SELFPLAY,
                    magnitude=min(1.0, (max(cpu_util, gpu_util) - 80) / 20),
                    reason=f"Overutilized: CPU={cpu_util:.1f}%, GPU={gpu_util:.1f}% (target 60-80%)",
                    metadata={
                        'cpu_util': cpu_util,
                        'gpu_util': gpu_util,
                        'total_jobs': total_jobs,
                    }
                ))

    # =========================================================================
    # Trigger Checks
    # =========================================================================

    def _should_trigger_cmaes(self) -> bool:
        """Check if CMA-ES should be triggered."""
        now = time.time()
        cooldown = self.state.cmaes_cooldown_hours * 3600
        return now - self.state.last_cmaes_trigger > cooldown

    def _should_trigger_nas(self) -> bool:
        """Check if NAS should be triggered."""
        now = time.time()
        cooldown = self.state.nas_cooldown_hours * 3600
        return now - self.state.last_nas_trigger > cooldown

    # =========================================================================
    # Parameter Getters (for pipeline stages to query)
    # =========================================================================

    def get_curriculum_weight(self, config_key: str) -> float:
        """Get curriculum weight for a configuration."""
        return self.state.curriculum_weights.get(config_key, 1.0)

    def get_temperature_multiplier(self) -> float:
        """Get temperature multiplier for selfplay."""
        return self.state.temperature_multiplier

    def get_epochs_multiplier(self) -> float:
        """Get epochs multiplier for training."""
        return self.state.epochs_multiplier

    def get_games_per_worker_multiplier(self) -> float:
        """Get games-per-worker multiplier."""
        return self.state.games_per_worker_multiplier

    def should_quarantine_data(self) -> bool:
        """Check if data should be quarantined due to quality issues."""
        return self.data_monitor.should_quarantine()

    def update_data_quality(
        self,
        parity_results: list[bool] | None = None,
        game_lengths: list[int] | None = None,
        draw_rate: float | None = None,
        timeout_rate: float | None = None,
    ) -> float:
        """Update data quality metrics and compute overall quality score.

        Args:
            parity_results: Recent parity validation results (True=passed)
            game_lengths: Recent game lengths for outlier detection
            draw_rate: Current draw rate (0-1)
            timeout_rate: Rate of games hitting move limits (0-1)

        Returns:
            Updated data_quality_score (0-1)
        """
        score = 1.0

        # Factor 1: Parity failure rate (most important)
        if parity_results:
            for result in parity_results:
                self.data_monitor.add_parity_result(result)

        parity_failure_rate = self.data_monitor.get_parity_failure_rate()
        self.state.parity_failure_rate = parity_failure_rate
        # Deduct based on parity failures (max 40% deduction)
        parity_penalty = min(0.4, parity_failure_rate * 4)
        score -= parity_penalty

        # Factor 2: Draw rate (target < 20%)
        if draw_rate is not None:
            draw_threshold = 0.20
            if draw_rate > draw_threshold:
                draw_penalty = min(0.2, (draw_rate - draw_threshold) * 2)
                score -= draw_penalty

        # Factor 3: Timeout/move limit rate (should be < 5%)
        if timeout_rate is not None:
            timeout_threshold = 0.05
            if timeout_rate > timeout_threshold:
                timeout_penalty = min(0.2, (timeout_rate - timeout_threshold) * 4)
                score -= timeout_penalty

        # Factor 4: Game length outliers (games with >1000 moves are suspicious)
        if game_lengths:
            self.data_monitor.game_lengths.extend(game_lengths)
            # Keep bounded
            if len(self.data_monitor.game_lengths) > 1000:
                self.data_monitor.game_lengths = self.data_monitor.game_lengths[-1000:]

            outlier_count = sum(1 for l in game_lengths if l > 1000)
            outlier_rate = outlier_count / len(game_lengths) if game_lengths else 0
            if outlier_rate > 0.01:  # More than 1% outliers
                outlier_penalty = min(0.2, outlier_rate * 10)
                score -= outlier_penalty

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        self.state.data_quality_score = score

        logger.info(f"Data quality updated: score={score:.2f}, "
                    f"parity_failure_rate={parity_failure_rate:.1%}")

        return score

    def get_pending_actions(self) -> list[FeedbackSignal]:
        """Get pending feedback actions that require external handling."""
        pending = []
        actionable_types = (
            FeedbackAction.TRIGGER_CMAES,
            FeedbackAction.TRIGGER_NAS,
            FeedbackAction.URGENT_RETRAINING,
            FeedbackAction.PROMOTION_FAILED,
        )
        for signal in self.signals[-10:]:  # Last 10 signals
            if signal.action in actionable_types:
                pending.append(signal)
        return pending

    def get_state_summary(self) -> dict[str, Any]:
        """Get summary of current feedback state."""
        return {
            'curriculum_weights': dict(self.state.curriculum_weights),
            'temperature_multiplier': self.state.temperature_multiplier,
            'epochs_multiplier': self.state.epochs_multiplier,
            'games_multiplier': self.state.games_per_worker_multiplier,
            'plateau_count': self.state.plateau_count,
            'parity_failure_rate': f"{self.state.parity_failure_rate:.1%}",
            'weak_configs': self.eval_analyzer.get_weak_configs(),
            'recent_signals': len(self.signals),
            # Promotion tracking
            'consecutive_promotion_failures': self.state.consecutive_promotion_failures,
            'promotion_failure_configs': dict(self.state.promotion_failure_configs),
            'last_promotion_success': self.state.last_promotion_success,
        }


# =============================================================================
# Feedback Signal Router
# =============================================================================

class FeedbackSignalRouter:
    """
    Routes feedback signals to registered handlers.

    This allows different components to subscribe to specific feedback actions
    and be notified when those actions are triggered.

    Usage:
        router = FeedbackSignalRouter()

        # Register handlers
        router.register_handler(FeedbackAction.TRIGGER_CMAES, cmaes_handler)
        router.register_handler(FeedbackAction.INCREASE_CURRICULUM_WEIGHT, curriculum_handler)

        # Route signals
        await router.route(signal)
    """

    def __init__(self):
        self._handlers: dict[FeedbackAction, list[callable]] = defaultdict(list)
        self._signal_history: list[tuple[float, FeedbackSignal, str]] = []  # (timestamp, signal, handler_result)
        self._max_history = 500

    def register_handler(
        self,
        action: FeedbackAction,
        handler: callable,
        name: str | None = None
    ):
        """Register a handler for a specific feedback action.

        Args:
            action: The FeedbackAction to handle
            handler: Async or sync callable that takes (signal: FeedbackSignal) -> bool
            name: Optional handler name for logging
        """
        handler_info = {
            'handler': handler,
            'name': name or handler.__name__ if hasattr(handler, '__name__') else 'anonymous',
        }
        self._handlers[action].append(handler_info)
        logger.info(f"Registered handler '{handler_info['name']}' for {action.value}")

    def unregister_handler(self, action: FeedbackAction, handler: callable):
        """Unregister a handler."""
        self._handlers[action] = [
            h for h in self._handlers[action]
            if h['handler'] != handler
        ]

    async def route(self, signal: FeedbackSignal) -> list[tuple[str, bool]]:
        """Route a signal to all registered handlers for its action.

        Returns:
            List of (handler_name, success) tuples
        """
        results = []
        handlers = self._handlers.get(signal.action, [])

        if not handlers:
            logger.debug(f"No handlers registered for {signal.action.value}")
            self._record_history(signal, "no_handlers")
            return results

        for handler_info in handlers:
            handler = handler_info['handler']
            name = handler_info['name']

            try:
                # Support both async and sync handlers
                if asyncio.iscoroutinefunction(handler):
                    success = await handler(signal)
                else:
                    success = handler(signal)

                results.append((name, success))
                self._record_history(signal, f"{name}:{'ok' if success else 'fail'}")
                logger.debug(f"Handler '{name}' processed {signal.action.value}: {success}")

            except Exception as e:
                logger.error(f"Handler '{name}' failed for {signal.action.value}: {e}")
                results.append((name, False))
                self._record_history(signal, f"{name}:error:{str(e)[:50]}")

        return results

    def _record_history(self, signal: FeedbackSignal, result: str):
        """Record signal routing to history."""
        self._signal_history.append((time.time(), signal, result))
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]

    def get_history(
        self,
        action: FeedbackAction | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get signal routing history.

        Args:
            action: Filter by action type
            limit: Maximum entries to return
        """
        history = self._signal_history
        if action:
            history = [(t, s, r) for t, s, r in history if s.action == action]

        return [
            {
                'timestamp': t,
                'action': s.action.value,
                'source': s.source_stage,
                'target': s.target_stage,
                'magnitude': s.magnitude,
                'reason': s.reason,
                'result': r,
            }
            for t, s, r in history[-limit:]
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        stats = {
            'total_signals_routed': len(self._signal_history),
            'handlers_registered': {
                action.value: len(handlers)
                for action, handlers in self._handlers.items()
            },
            'signals_by_action': {},
        }

        for _, signal, _ in self._signal_history:
            action = signal.action.value
            stats['signals_by_action'][action] = stats['signals_by_action'].get(action, 0) + 1

        return stats


class OpponentWinRateTracker:
    """
    Tracks win rates against specific opponent types.

    This enables opponent-specific curriculum adjustments - if the model
    struggles against a specific opponent (e.g., MCTS-1000), we can
    increase training weight for games against that opponent.

    Usage:
        tracker = OpponentWinRateTracker()
        tracker.record_game("ringrift_v3", "mcts_100", won=True)
        tracker.record_game("ringrift_v3", "mcts_1000", won=False)

        weak_opponents = tracker.get_weak_opponents("ringrift_v3")
    """

    def __init__(self, min_games: int = 10, weak_threshold: float = 0.45):
        self.min_games = min_games
        self.weak_threshold = weak_threshold

        # model_id -> opponent_id -> {wins, losses}
        self._records: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        )
        self._history: list[tuple[float, str, str, str]] = []  # (timestamp, model, opponent, result)

    def record_game(
        self,
        model_id: str,
        opponent_id: str,
        won: bool | None = None,
        draw: bool = False
    ):
        """Record a game result against an opponent."""
        if draw:
            self._records[model_id][opponent_id]['draws'] += 1
            result = 'draw'
        elif won:
            self._records[model_id][opponent_id]['wins'] += 1
            result = 'win'
        else:
            self._records[model_id][opponent_id]['losses'] += 1
            result = 'loss'

        self._history.append((time.time(), model_id, opponent_id, result))

        # Keep bounded
        if len(self._history) > 10000:
            self._history = self._history[-10000:]

    def get_win_rate(self, model_id: str, opponent_id: str) -> float | None:
        """Get win rate against a specific opponent. Returns None if insufficient games."""
        record = self._records.get(model_id, {}).get(opponent_id)
        if not record:
            return None

        total = record['wins'] + record['losses'] + record['draws']
        if total < self.min_games:
            return None

        # Count draws as half wins
        return (record['wins'] + 0.5 * record['draws']) / total

    def get_weak_opponents(self, model_id: str) -> list[tuple[str, float]]:
        """Get opponents where model has below-threshold win rate.

        Returns:
            List of (opponent_id, win_rate) tuples, sorted by win rate ascending
        """
        weak = []
        for opponent_id, _record in self._records.get(model_id, {}).items():
            win_rate = self.get_win_rate(model_id, opponent_id)
            if win_rate is not None and win_rate < self.weak_threshold:
                weak.append((opponent_id, win_rate))

        return sorted(weak, key=lambda x: x[1])

    def get_strong_opponents(self, model_id: str, threshold: float = 0.60) -> list[tuple[str, float]]:
        """Get opponents where model has above-threshold win rate."""
        strong = []
        for opponent_id, _record in self._records.get(model_id, {}).items():
            win_rate = self.get_win_rate(model_id, opponent_id)
            if win_rate is not None and win_rate >= threshold:
                strong.append((opponent_id, win_rate))

        return sorted(strong, key=lambda x: x[1], reverse=True)

    def get_summary(self, model_id: str) -> dict[str, Any]:
        """Get summary of opponent performance for a model."""
        opponents = self._records.get(model_id, {})
        summary = {
            'total_opponents': len(opponents),
            'opponents': {},
        }

        for opponent_id, record in opponents.items():
            win_rate = self.get_win_rate(model_id, opponent_id)
            total = record['wins'] + record['losses'] + record['draws']
            summary['opponents'][opponent_id] = {
                'wins': record['wins'],
                'losses': record['losses'],
                'draws': record['draws'],
                'total': total,
                'win_rate': win_rate,
            }

        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

def create_feedback_controller(
    ai_service_dir: Path,
    config: dict[str, Any] | None = None
) -> PipelineFeedbackController:
    """Create a feedback controller with standard paths."""
    # Ensure ai_service_dir is a Path object
    if isinstance(ai_service_dir, str):
        ai_service_dir = Path(ai_service_dir)
    state_path = ai_service_dir / "logs" / "feedback" / "feedback_state.json"
    return PipelineFeedbackController(state_path=state_path, config=config)


def create_feedback_router() -> FeedbackSignalRouter:
    """Create a feedback signal router."""
    return FeedbackSignalRouter()


def create_opponent_tracker(min_games: int = 10, weak_threshold: float = 0.45) -> OpponentWinRateTracker:
    """Create an opponent win rate tracker."""
    return OpponentWinRateTracker(min_games=min_games, weak_threshold=weak_threshold)
