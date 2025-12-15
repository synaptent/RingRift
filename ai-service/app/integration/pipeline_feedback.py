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

import json
import logging
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum


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


@dataclass
class FeedbackSignal:
    """A feedback signal from one pipeline stage to another."""
    source_stage: str
    target_stage: str
    action: FeedbackAction
    magnitude: float  # 0.0 to 1.0
    reason: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackState:
    """Persistent state for feedback system."""
    # Curriculum weights by config (config_key -> weight multiplier)
    curriculum_weights: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))

    # Temperature adjustments
    temperature_multiplier: float = 1.0

    # Training parameters
    epochs_multiplier: float = 1.0
    batch_size_multiplier: float = 1.0

    # Data collection parameters
    games_per_worker_multiplier: float = 1.0

    # Elo tracking for plateau detection
    elo_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, elo)
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


class EvaluationAnalyzer:
    """Analyzes evaluation results to identify weaknesses."""

    def __init__(self):
        self.win_rates: Dict[str, List[float]] = defaultdict(list)
        self.elo_trends: Dict[str, List[float]] = defaultdict(list)

    def add_result(self, config_key: str, win_rate: float, elo: float):
        """Add an evaluation result."""
        self.win_rates[config_key].append(win_rate)
        self.elo_trends[config_key].append(elo)

        # Keep only last 20 results
        if len(self.win_rates[config_key]) > 20:
            self.win_rates[config_key] = self.win_rates[config_key][-20:]
        if len(self.elo_trends[config_key]) > 20:
            self.elo_trends[config_key] = self.elo_trends[config_key][-20:]

    def get_weak_configs(self, threshold: float = 0.45) -> List[str]:
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
        self.parity_results: List[bool] = []  # True = passed, False = failed
        self.game_lengths: List[int] = []
        self.outlier_games: List[str] = []

    def add_parity_result(self, passed: bool, game_id: Optional[str] = None):
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
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.learning_rate_history: List[float] = []

    def add_training_metrics(self, loss: float, val_loss: Optional[float] = None,
                             learning_rate: Optional[float] = None):
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
        state_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
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
        self.signals: List[FeedbackSignal] = []
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

    async def on_stage_complete(self, stage: str, result: Dict[str, Any]):
        """Handle completion of a pipeline stage."""
        handlers = {
            'evaluation': self._on_evaluation_complete,
            'training': self._on_training_complete,
            'parity-validation': self._on_parity_validation_complete,
            'selfplay': self._on_selfplay_complete,
            'canonical-selfplay': self._on_selfplay_complete,
            'cmaes': self._on_cmaes_complete,
        }

        handler = handlers.get(stage)
        if handler:
            await handler(result)

        self._save_state()

    async def _on_evaluation_complete(self, result: Dict[str, Any]):
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

    async def _on_training_complete(self, result: Dict[str, Any]):
        """Handle training completion - analyze loss patterns."""
        final_loss = result.get('final_loss')
        val_loss = result.get('val_loss')
        epochs_completed = result.get('epochs', 0)

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

    async def _on_parity_validation_complete(self, result: Dict[str, Any]):
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

    async def _on_selfplay_complete(self, result: Dict[str, Any]):
        """Handle selfplay completion - track data generation."""
        games_generated = result.get('games', 0)
        config_key = result.get('config_key')

        if config_key and games_generated > 0:
            # Could track per-config generation rates here
            pass

    async def _on_cmaes_complete(self, result: Dict[str, Any]):
        """Handle CMA-ES completion - track optimization results."""
        improved = result.get('improved', False)

        if improved:
            # Reset plateau count on successful optimization
            self.state.plateau_count = 0

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

    def get_pending_actions(self) -> List[FeedbackSignal]:
        """Get pending feedback actions that require external handling."""
        pending = []
        for signal in self.signals[-10:]:  # Last 10 signals
            if signal.action in (FeedbackAction.TRIGGER_CMAES, FeedbackAction.TRIGGER_NAS):
                pending.append(signal)
        return pending

    def get_state_summary(self) -> Dict[str, Any]:
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
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_feedback_controller(
    ai_service_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> PipelineFeedbackController:
    """Create a feedback controller with standard paths."""
    state_path = ai_service_dir / "logs" / "feedback" / "feedback_state.json"
    return PipelineFeedbackController(state_path=state_path, config=config)
