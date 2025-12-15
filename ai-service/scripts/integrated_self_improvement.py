#!/usr/bin/env python3
"""
Integrated Self-Improvement Controller for RingRift AI.

This module unifies all AI training components into a cohesive self-improvement
loop with closed-loop feedback, automatic optimization, and robust fault tolerance.

Integrates:
- Training Dashboard & Monitoring
- Automated Model Registry
- Training Fault Tolerance
- Self-Play Temperature Scheduling
- Value Head Calibration
- Comprehensive Benchmark Suite
- Multi-Task Learning
- Human Evaluation Interface
- Transformer Integration
- MCTS Improvements

With existing infrastructure:
- P2P Orchestrator
- Pipeline Orchestrator
- Unified AI Loop
- Continuous Improvement Daemon
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import sqlite3
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import subprocess
import signal


# Add paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_DIR = SCRIPT_DIR.parent
RINGRIFT_DIR = AI_SERVICE_DIR.parent
sys.path.insert(0, str(AI_SERVICE_DIR))
sys.path.insert(0, str(RINGRIFT_DIR))


logger = logging.getLogger(__name__)


# =============================================================================
# Event System
# =============================================================================

class EventType(Enum):
    """Events in the self-improvement loop."""
    # Data events
    GAMES_COLLECTED = "games_collected"
    DATA_QUALITY_WARNING = "data_quality_warning"
    TRAINING_DATA_READY = "training_data_ready"

    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    CHECKPOINT_SAVED = "checkpoint_saved"

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    BENCHMARK_COMPLETED = "benchmark_completed"

    # Model lifecycle events
    MODEL_REGISTERED = "model_registered"
    MODEL_PROMOTED = "model_promoted"
    MODEL_REJECTED = "model_rejected"
    MODEL_ROLLBACK = "model_rollback"

    # Optimization events
    PLATEAU_DETECTED = "plateau_detected"
    CMAES_TRIGGERED = "cmaes_triggered"
    NAS_TRIGGERED = "nas_triggered"
    TEMPERATURE_ADJUSTED = "temperature_adjusted"

    # System events
    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"
    FAULT_DETECTED = "fault_detected"
    RECOVERY_COMPLETED = "recovery_completed"


@dataclass
class Event:
    """An event in the system."""
    event_type: EventType
    timestamp: datetime
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'payload': self.payload
        }


class EventBus:
    """Central event bus for component communication."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        for callback in self._subscribers[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")

    def get_recent_events(self, event_type: Optional[EventType] = None,
                          limit: int = 100) -> List[Event]:
        """Get recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


# =============================================================================
# State Management
# =============================================================================

@dataclass
class ComponentState:
    """State of a component."""
    name: str
    running: bool = False
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedState:
    """Complete state of the integrated system."""
    version: str = "1.0.0"
    started_at: Optional[datetime] = None

    # Component states
    components: Dict[str, ComponentState] = field(default_factory=dict)

    # Training state
    current_model_id: Optional[str] = None
    current_model_version: int = 0
    training_iteration: int = 0
    games_collected: int = 0
    games_since_training: int = 0

    # Evaluation state
    current_elo: float = 1500.0
    elo_history: List[Tuple[datetime, float]] = field(default_factory=list)
    plateau_count: int = 0

    # Optimization state
    temperature_schedule: str = "default"
    cmaes_last_run: Optional[datetime] = None
    nas_last_run: Optional[datetime] = None

    # Calibration state
    value_calibration_temp: float = 1.0
    calibration_ece: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat() if self.started_at else None
        d['elo_history'] = [(t.isoformat(), e) for t, e in self.elo_history]
        d['cmaes_last_run'] = self.cmaes_last_run.isoformat() if self.cmaes_last_run else None
        d['nas_last_run'] = self.nas_last_run.isoformat() if self.nas_last_run else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IntegratedState':
        d = d.copy()
        if d.get('started_at'):
            d['started_at'] = datetime.fromisoformat(d['started_at'])
        if d.get('elo_history'):
            d['elo_history'] = [(datetime.fromisoformat(t), e) for t, e in d['elo_history']]
        if d.get('cmaes_last_run'):
            d['cmaes_last_run'] = datetime.fromisoformat(d['cmaes_last_run'])
        if d.get('nas_last_run'):
            d['nas_last_run'] = datetime.fromisoformat(d['nas_last_run'])
        if d.get('components'):
            d['components'] = {
                k: ComponentState(**v) if isinstance(v, dict) else v
                for k, v in d['components'].items()
            }
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StateManager:
    """Manages persistent state."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "integrated_state.json"
        self.state = IntegratedState()
        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.state = IntegratedState.from_dict(data)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Save state to disk."""
        with self._lock:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state.to_dict(), f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def update_component(self, name: str, **kwargs):
        """Update component state."""
        with self._lock:
            if name not in self.state.components:
                self.state.components[name] = ComponentState(name=name)
            for k, v in kwargs.items():
                if hasattr(self.state.components[name], k):
                    setattr(self.state.components[name], k, v)
        self.save_state()


# =============================================================================
# Component Integrations
# =============================================================================

class ModelRegistryIntegration:
    """Integration with the Model Registry."""

    def __init__(self, registry_dir: Path, event_bus: EventBus):
        self.registry_dir = registry_dir
        self.event_bus = event_bus
        self._registry = None
        self._auto_promoter = None

    def _ensure_registry(self):
        """Lazy initialization of registry."""
        if self._registry is None:
            try:
                from app.training.model_registry import ModelRegistry, AutoPromoter
                self._registry = ModelRegistry(self.registry_dir)
                self._auto_promoter = AutoPromoter(
                    self._registry,
                    min_elo_improvement=20.0,
                    min_games=100,
                    min_win_rate_vs_current=0.52
                )
            except ImportError:
                logger.warning("Model registry not available")

    async def register_model(
        self,
        name: str,
        model_path: Path,
        metrics: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Optional[Tuple[str, int]]:
        """Register a new model version."""
        self._ensure_registry()
        if self._registry is None:
            return None

        try:
            from app.training.model_registry import ModelMetrics, TrainingConfig, ModelType

            model_metrics = ModelMetrics(
                elo=metrics.get('elo'),
                elo_uncertainty=metrics.get('elo_uncertainty'),
                win_rate=metrics.get('win_rate'),
                games_played=metrics.get('games_played', 0)
            )

            train_cfg = TrainingConfig(
                learning_rate=training_config.get('learning_rate', 0.001),
                batch_size=training_config.get('batch_size', 256),
                epochs=training_config.get('epochs', 100)
            )

            model_id, version = self._registry.register_model(
                name=name,
                model_path=model_path,
                model_type=ModelType.POLICY_VALUE,
                metrics=model_metrics,
                training_config=train_cfg
            )

            await self.event_bus.publish(Event(
                event_type=EventType.MODEL_REGISTERED,
                timestamp=datetime.now(),
                source="model_registry",
                payload={'model_id': model_id, 'version': version}
            ))

            return model_id, version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    async def try_promote(self, model_id: str, version: int) -> bool:
        """Try to automatically promote a model."""
        self._ensure_registry()
        if self._auto_promoter is None:
            return False

        try:
            new_stage = self._auto_promoter.auto_promote(model_id, version)
            if new_stage:
                await self.event_bus.publish(Event(
                    event_type=EventType.MODEL_PROMOTED,
                    timestamp=datetime.now(),
                    source="model_registry",
                    payload={
                        'model_id': model_id,
                        'version': version,
                        'new_stage': new_stage.value
                    }
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")

        return False

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Get the current production model."""
        self._ensure_registry()
        if self._registry is None:
            return None

        try:
            model = self._registry.get_production_model()
            if model:
                return {
                    'model_id': model.model_id,
                    'version': model.version,
                    'file_path': model.file_path,
                    'elo': model.metrics.elo
                }
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")

        return None


class DashboardIntegration:
    """Integration with the Training Dashboard."""

    def __init__(self, dashboard_dir: Path, event_bus: EventBus):
        self.dashboard_dir = dashboard_dir
        self.event_bus = event_bus
        self._collector = None

    def _ensure_collector(self):
        """Lazy initialization of metrics collector."""
        if self._collector is None:
            try:
                from app.monitoring.training_dashboard import MetricsCollector
                self._collector = MetricsCollector(self.dashboard_dir / "metrics.db")
            except ImportError:
                logger.warning("Dashboard not available")

    def record_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float
    ):
        """Record a training step."""
        self._ensure_collector()
        if self._collector:
            self._collector.record_training_step(
                epoch, step, loss, policy_loss, value_loss, learning_rate
            )

    def record_elo(self, model_id: str, elo: float, games_played: int):
        """Record Elo rating."""
        self._ensure_collector()
        if self._collector:
            self._collector.record_elo(model_id, elo, games_played)

    def record_cluster_status(
        self,
        host_name: str,
        cpu_percent: float,
        memory_percent: float,
        gpu_util: Optional[float],
        jobs_running: int
    ):
        """Record cluster node status."""
        self._ensure_collector()
        if self._collector:
            self._collector.record_cluster_status(
                host_name, cpu_percent, memory_percent, gpu_util, jobs_running
            )


class CalibrationIntegration:
    """Integration with Value Head Calibration."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._calibrator = None
        self._current_temperature = 1.0

    def _ensure_calibrator(self):
        """Lazy initialization."""
        if self._calibrator is None:
            try:
                from app.training.value_calibration import ValueCalibrator
                self._calibrator = ValueCalibrator(num_bins=10)
            except ImportError:
                logger.warning("Calibration not available")

    def add_samples(self, predictions: List[float], outcomes: List[float]):
        """Add prediction-outcome pairs for calibration."""
        self._ensure_calibrator()
        if self._calibrator:
            self._calibrator.add_batch(predictions, outcomes)

    async def compute_calibration(self) -> Optional[Dict[str, float]]:
        """Compute calibration and find optimal temperature."""
        self._ensure_calibrator()
        if self._calibrator is None:
            return None

        try:
            report = self._calibrator.compute_calibration()
            optimal_temp = self._calibrator.find_optimal_temperature()
            self._current_temperature = optimal_temp

            result = {
                'ece': report.ece,
                'mce': report.mce,
                'overconfidence': report.overconfidence,
                'optimal_temperature': optimal_temp,
                'brier_score': report.brier_score
            }

            logger.info(f"Calibration: ECE={report.ece:.4f}, temp={optimal_temp:.3f}")
            return result

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return None

    def calibrate_value(self, value: float) -> float:
        """Apply temperature scaling to a value prediction."""
        self._ensure_calibrator()
        if self._calibrator:
            return self._calibrator.calibrate_prediction(value)
        return value


class TemperatureScheduleIntegration:
    """Integration with Self-Play Temperature Scheduling."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._scheduler = None
        self._current_preset = "default"
        self._training_progress = 0.0

    def _ensure_scheduler(self):
        """Lazy initialization."""
        if self._scheduler is None:
            try:
                from app.training.temperature_scheduling import create_scheduler
                self._scheduler = create_scheduler(self._current_preset)
            except ImportError:
                logger.warning("Temperature scheduling not available")

    def set_preset(self, preset: str):
        """Set temperature schedule preset."""
        self._current_preset = preset
        self._scheduler = None  # Reset to recreate with new preset
        self._ensure_scheduler()

    def set_training_progress(self, progress: float):
        """Set current training progress (0-1)."""
        self._training_progress = max(0.0, min(1.0, progress))
        self._ensure_scheduler()
        if self._scheduler:
            self._scheduler.set_training_progress(self._training_progress)

    def get_temperature(self, move_number: int, game_state: Any = None) -> float:
        """Get temperature for a move."""
        self._ensure_scheduler()
        if self._scheduler:
            return self._scheduler.get_temperature(move_number, game_state)
        return 1.0  # Default


class BenchmarkIntegration:
    """Integration with Benchmark Suite."""

    def __init__(self, results_dir: Path, event_bus: EventBus):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus
        self._suite = None

    def _ensure_suite(self):
        """Lazy initialization."""
        if self._suite is None:
            try:
                from app.evaluation.benchmark_suite import create_default_suite
                self._suite = create_default_suite()
            except ImportError:
                logger.warning("Benchmark suite not available")

    async def run_benchmarks(self, model: Any, model_id: str) -> Optional[Dict[str, Any]]:
        """Run benchmark suite on a model."""
        self._ensure_suite()
        if self._suite is None:
            return None

        try:
            result = self._suite.run(model, model_id)

            # Save results
            result_path = self.results_dir / f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._suite.save_results(result, result_path)

            await self.event_bus.publish(Event(
                event_type=EventType.BENCHMARK_COMPLETED,
                timestamp=datetime.now(),
                source="benchmark",
                payload={
                    'model_id': model_id,
                    'aggregate_score': result.compute_aggregate_score(),
                    'result_path': str(result_path)
                }
            ))

            return result.to_dict()

        except Exception as e:
            logger.error(f"Benchmarks failed: {e}")
            return None


class FaultToleranceIntegration:
    """Integration with Training Fault Tolerance."""

    def __init__(self, checkpoint_dir: Path, event_bus: EventBus):
        self.checkpoint_dir = checkpoint_dir
        self.event_bus = event_bus
        self._trainer = None

    def _ensure_trainer(self):
        """Lazy initialization."""
        if self._trainer is None:
            try:
                from app.training.fault_tolerance import FaultTolerantTrainer
                self._trainer = FaultTolerantTrainer(
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_interval_steps=1000,
                    checkpoint_interval_epochs=1,
                    max_retries=3
                )
            except ImportError:
                logger.warning("Fault tolerance not available")

    def initialize_training(
        self,
        model_state: Dict[str, Any],
        training_config: Dict[str, Any],
        total_epochs: int,
        resume: bool = True
    ) -> Dict[str, Any]:
        """Initialize fault-tolerant training."""
        self._ensure_trainer()
        if self._trainer is None:
            return {'epoch': 0, 'global_step': 0}

        progress = self._trainer.initialize(
            model_state, training_config, total_epochs, resume
        )
        return progress.to_dict()

    def update_progress(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        metrics: Dict[str, float],
        model_state: Optional[Dict[str, Any]] = None
    ):
        """Update training progress."""
        if self._trainer:
            self._trainer.update_progress(
                epoch, batch_idx, global_step, metrics, model_state
            )

    async def checkpoint_if_needed(
        self,
        model_state: Dict[str, Any],
        force: bool = False
    ) -> bool:
        """Save checkpoint if needed."""
        if self._trainer is None:
            return False

        metadata = self._trainer.checkpoint_if_needed(model_state, force)
        if metadata:
            await self.event_bus.publish(Event(
                event_type=EventType.CHECKPOINT_SAVED,
                timestamp=datetime.now(),
                source="fault_tolerance",
                payload={'checkpoint_id': metadata.checkpoint_id}
            ))
            return True
        return False

    @property
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._trainer.should_stop if self._trainer else False


# =============================================================================
# Plateau Detection and Optimization Triggering
# =============================================================================

class PlateauDetector:
    """Detects training plateaus and triggers optimization."""

    def __init__(
        self,
        event_bus: EventBus,
        state_manager: StateManager,
        min_elo_improvement: float = 15.0,
        lookback_iterations: int = 5,
        cmaes_cooldown_hours: float = 6.0,
        nas_cooldown_hours: float = 24.0
    ):
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.min_elo_improvement = min_elo_improvement
        self.lookback_iterations = lookback_iterations
        self.cmaes_cooldown_hours = cmaes_cooldown_hours
        self.nas_cooldown_hours = nas_cooldown_hours

    async def check_plateau(self) -> bool:
        """Check if we're in a plateau and trigger optimization."""
        state = self.state_manager.state

        if len(state.elo_history) < self.lookback_iterations:
            return False

        # Get recent Elo values
        recent = state.elo_history[-self.lookback_iterations:]
        oldest_elo = recent[0][1]
        newest_elo = recent[-1][1]
        improvement = newest_elo - oldest_elo

        if improvement < self.min_elo_improvement:
            # Plateau detected
            state.plateau_count += 1

            await self.event_bus.publish(Event(
                event_type=EventType.PLATEAU_DETECTED,
                timestamp=datetime.now(),
                source="plateau_detector",
                payload={
                    'improvement': improvement,
                    'lookback': self.lookback_iterations,
                    'plateau_count': state.plateau_count
                }
            ))

            # Decide which optimization to trigger
            now = datetime.now()

            # Check CMA-ES cooldown
            if state.cmaes_last_run is None or \
               (now - state.cmaes_last_run).total_seconds() > self.cmaes_cooldown_hours * 3600:
                await self._trigger_cmaes()
                return True

            # Check NAS cooldown (less frequent)
            if state.nas_last_run is None or \
               (now - state.nas_last_run).total_seconds() > self.nas_cooldown_hours * 3600:
                if state.plateau_count >= 3:  # More severe plateau
                    await self._trigger_nas()
                    return True

            logger.info(f"Plateau detected but optimizations on cooldown")
            return True

        # Not in plateau
        if state.plateau_count > 0:
            state.plateau_count = max(0, state.plateau_count - 1)

        return False

    async def _trigger_cmaes(self):
        """Trigger CMA-ES optimization."""
        self.state_manager.state.cmaes_last_run = datetime.now()
        self.state_manager.save_state()

        await self.event_bus.publish(Event(
            event_type=EventType.CMAES_TRIGGERED,
            timestamp=datetime.now(),
            source="plateau_detector",
            payload={'reason': 'plateau_detected'}
        ))

    async def _trigger_nas(self):
        """Trigger Neural Architecture Search."""
        self.state_manager.state.nas_last_run = datetime.now()
        self.state_manager.save_state()

        await self.event_bus.publish(Event(
            event_type=EventType.NAS_TRIGGERED,
            timestamp=datetime.now(),
            source="plateau_detector",
            payload={'reason': 'severe_plateau'}
        ))


# =============================================================================
# Feedback Loops
# =============================================================================

class FeedbackController:
    """
    Manages feedback loops between components.

    Feedback loops:
    1. Evaluation → Curriculum: Adjust training focus based on weaknesses
    2. Calibration → Inference: Apply temperature scaling to predictions
    3. Benchmarks → Training: Identify areas needing improvement
    4. Plateau → Optimization: Trigger CMA-ES/NAS when stuck
    """

    def __init__(
        self,
        event_bus: EventBus,
        state_manager: StateManager,
        calibration: CalibrationIntegration,
        temperature: TemperatureScheduleIntegration
    ):
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.calibration = calibration
        self.temperature = temperature

        # Subscribe to events
        event_bus.subscribe(EventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
        event_bus.subscribe(EventType.BENCHMARK_COMPLETED, self._on_benchmark_completed)
        event_bus.subscribe(EventType.TRAINING_COMPLETED, self._on_training_completed)
        event_bus.subscribe(EventType.PLATEAU_DETECTED, self._on_plateau_detected)

    async def _on_evaluation_completed(self, event: Event):
        """Handle evaluation results - adjust curriculum weights."""
        payload = event.payload
        elo = payload.get('elo')
        win_rate = payload.get('win_rate')
        config_key = payload.get('config_key', 'default')

        if elo is not None:
            # Update Elo history
            state = self.state_manager.state
            state.current_elo = elo
            state.elo_history.append((datetime.now(), elo))

            # Keep only last 100 entries
            if len(state.elo_history) > 100:
                state.elo_history = state.elo_history[-100:]

            self.state_manager.save_state()

        # Adjust curriculum based on win rate
        if win_rate is not None and win_rate < 0.45:
            logger.info(f"Low win rate ({win_rate:.2f}) for {config_key}, increasing focus")
            # This would integrate with curriculum weights in the unified loop

    async def _on_benchmark_completed(self, event: Event):
        """Handle benchmark results - identify weak areas."""
        payload = event.payload
        results = payload.get('results', {})

        # Look for weak benchmarks
        for benchmark_name, score in results.items():
            if isinstance(score, (int, float)) and score < 0.5:
                logger.info(f"Weak benchmark: {benchmark_name} = {score:.3f}")

    async def _on_training_completed(self, event: Event):
        """Handle training completion - update calibration."""
        payload = event.payload

        # Update training progress for temperature scheduling
        iteration = payload.get('iteration', 0)
        total_iterations = payload.get('total_iterations', 100)
        progress = iteration / total_iterations if total_iterations > 0 else 0

        self.temperature.set_training_progress(progress)

    async def _on_plateau_detected(self, event: Event):
        """Handle plateau detection - adjust temperature."""
        payload = event.payload
        plateau_count = payload.get('plateau_count', 0)

        # Increase exploration when stuck
        if plateau_count >= 2:
            logger.info("Switching to aggressive exploration temperature schedule")
            self.temperature.set_preset("aggressive_exploration")

            await self.event_bus.publish(Event(
                event_type=EventType.TEMPERATURE_ADJUSTED,
                timestamp=datetime.now(),
                source="feedback_controller",
                payload={'new_preset': 'aggressive_exploration', 'reason': 'plateau'}
            ))


# =============================================================================
# P2P Orchestrator Integration
# =============================================================================

class P2PIntegration:
    """Integration with P2P Orchestrator for distributed execution."""

    def __init__(
        self,
        p2p_url: Optional[str],
        event_bus: EventBus
    ):
        self.p2p_url = p2p_url
        self.event_bus = event_bus
        self._session = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                logger.warning("aiohttp not available for P2P integration")

    async def submit_job(
        self,
        job_type: str,
        config: Dict[str, Any],
        priority: int = 0
    ) -> Optional[str]:
        """Submit a job to the P2P orchestrator."""
        if not self.p2p_url:
            return None

        await self._ensure_session()
        if self._session is None:
            return None

        try:
            async with self._session.post(
                f"{self.p2p_url}/api/jobs",
                json={
                    'job_type': job_type,
                    'config': config,
                    'priority': priority
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('job_id')
        except Exception as e:
            logger.error(f"Failed to submit P2P job: {e}")

        return None

    async def get_cluster_status(self) -> Optional[Dict[str, Any]]:
        """Get cluster status from P2P orchestrator."""
        if not self.p2p_url:
            return None

        await self._ensure_session()
        if self._session is None:
            return None

        try:
            async with self._session.get(f"{self.p2p_url}/api/status") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")

        return None

    async def trigger_selfplay(
        self,
        num_games: int,
        board_type: str = "square8",
        num_players: int = 2,
        temperature_schedule: str = "default"
    ) -> Optional[str]:
        """Trigger self-play game generation."""
        return await self.submit_job(
            job_type="SELFPLAY",
            config={
                'num_games': num_games,
                'board_type': board_type,
                'num_players': num_players,
                'temperature_schedule': temperature_schedule
            },
            priority=1
        )

    async def trigger_training(
        self,
        data_path: str,
        epochs: int = 10,
        batch_size: int = 256
    ) -> Optional[str]:
        """Trigger model training."""
        return await self.submit_job(
            job_type="TRAINING",
            config={
                'data_path': data_path,
                'epochs': epochs,
                'batch_size': batch_size
            },
            priority=2
        )

    async def trigger_cmaes(self, board_type: str = "square8") -> Optional[str]:
        """Trigger CMA-ES optimization."""
        return await self.submit_job(
            job_type="CMAES",
            config={'board_type': board_type},
            priority=3
        )

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()


# =============================================================================
# Main Integrated Controller
# =============================================================================

class IntegratedSelfImprovementController:
    """
    Main controller that orchestrates all components for self-improvement.
    """

    def __init__(
        self,
        base_dir: Path,
        p2p_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.base_dir = Path(base_dir)
        self.config = config or {}

        # Create directories
        self.state_dir = self.base_dir / "logs" / "integrated"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.registry_dir = self.base_dir / "model_registry"
        self.dashboard_dir = self.base_dir / "dashboard"
        self.benchmark_dir = self.base_dir / "benchmarks"

        for d in [self.state_dir, self.checkpoint_dir, self.registry_dir,
                  self.dashboard_dir, self.benchmark_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Core components
        self.event_bus = EventBus()
        self.state_manager = StateManager(self.state_dir)

        # Integrations
        self.model_registry = ModelRegistryIntegration(self.registry_dir, self.event_bus)
        self.dashboard = DashboardIntegration(self.dashboard_dir, self.event_bus)
        self.calibration = CalibrationIntegration(self.event_bus)
        self.temperature = TemperatureScheduleIntegration(self.event_bus)
        self.benchmarks = BenchmarkIntegration(self.benchmark_dir, self.event_bus)
        self.fault_tolerance = FaultToleranceIntegration(self.checkpoint_dir, self.event_bus)
        self.p2p = P2PIntegration(p2p_url, self.event_bus)

        # Controllers
        self.plateau_detector = PlateauDetector(
            self.event_bus,
            self.state_manager,
            min_elo_improvement=self.config.get('min_elo_improvement', 15.0),
            lookback_iterations=self.config.get('plateau_lookback', 5)
        )

        self.feedback = FeedbackController(
            self.event_bus,
            self.state_manager,
            self.calibration,
            self.temperature
        )

        # Runtime state
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the integrated controller."""
        logger.info("Starting Integrated Self-Improvement Controller")

        self._running = True
        self.state_manager.state.started_at = datetime.now()
        self.state_manager.save_state()

        await self.event_bus.publish(Event(
            event_type=EventType.COMPONENT_STARTED,
            timestamp=datetime.now(),
            source="integrated_controller",
            payload={'version': self.state_manager.state.version}
        ))

        # Start background loops
        self._tasks = [
            asyncio.create_task(self._data_collection_loop()),
            asyncio.create_task(self._training_loop()),
            asyncio.create_task(self._evaluation_loop()),
            asyncio.create_task(self._calibration_loop()),
            asyncio.create_task(self._plateau_detection_loop()),
            asyncio.create_task(self._cluster_monitoring_loop()),
            asyncio.create_task(self._state_persistence_loop()),
        ]

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Controller tasks cancelled")

    async def stop(self):
        """Stop the integrated controller."""
        logger.info("Stopping Integrated Self-Improvement Controller")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Cleanup
        await self.p2p.close()
        self.state_manager.save_state()

        await self.event_bus.publish(Event(
            event_type=EventType.COMPONENT_STOPPED,
            timestamp=datetime.now(),
            source="integrated_controller",
            payload={}
        ))

    async def _data_collection_loop(self):
        """Periodically sync game data."""
        interval = self.config.get('data_collection_interval', 60)

        while self._running:
            try:
                # Get cluster status
                status = await self.p2p.get_cluster_status()
                if status:
                    total_games = status.get('total_games', 0)
                    new_games = total_games - self.state_manager.state.games_collected

                    if new_games > 0:
                        self.state_manager.state.games_collected = total_games
                        self.state_manager.state.games_since_training += new_games

                        await self.event_bus.publish(Event(
                            event_type=EventType.GAMES_COLLECTED,
                            timestamp=datetime.now(),
                            source="data_collection",
                            payload={'new_games': new_games, 'total': total_games}
                        ))

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(interval)

    async def _training_loop(self):
        """Check training triggers and manage training."""
        interval = self.config.get('training_check_interval', 300)
        threshold = self.config.get('training_threshold', 500)

        while self._running:
            try:
                state = self.state_manager.state

                # Check if training should be triggered
                if state.games_since_training >= threshold:
                    logger.info(f"Training threshold reached: {state.games_since_training} games")

                    await self.event_bus.publish(Event(
                        event_type=EventType.TRAINING_DATA_READY,
                        timestamp=datetime.now(),
                        source="training_loop",
                        payload={'games': state.games_since_training}
                    ))

                    # Trigger training via P2P
                    job_id = await self.p2p.trigger_training(
                        data_path=str(self.base_dir / "data" / "training"),
                        epochs=self.config.get('training_epochs', 10),
                        batch_size=self.config.get('batch_size', 256)
                    )

                    if job_id:
                        state.games_since_training = 0
                        state.training_iteration += 1
                        self.state_manager.save_state()

                        await self.event_bus.publish(Event(
                            event_type=EventType.TRAINING_STARTED,
                            timestamp=datetime.now(),
                            source="training_loop",
                            payload={'job_id': job_id, 'iteration': state.training_iteration}
                        ))

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(interval)

    async def _evaluation_loop(self):
        """Periodically evaluate models."""
        interval = self.config.get('evaluation_interval', 3600)

        while self._running:
            try:
                # Get production model
                prod_model = self.model_registry.get_production_model()
                if prod_model:
                    # Run benchmarks (would need actual model loading)
                    logger.info(f"Running evaluation for {prod_model['model_id']}")

                    # In a real implementation, load model and run benchmarks
                    # result = await self.benchmarks.run_benchmarks(model, model_id)

                    await self.event_bus.publish(Event(
                        event_type=EventType.EVALUATION_COMPLETED,
                        timestamp=datetime.now(),
                        source="evaluation_loop",
                        payload={
                            'model_id': prod_model['model_id'],
                            'elo': prod_model.get('elo', 1500)
                        }
                    ))

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
                await asyncio.sleep(interval)

    async def _calibration_loop(self):
        """Periodically calibrate value predictions."""
        interval = self.config.get('calibration_interval', 7200)  # 2 hours

        while self._running:
            try:
                # Compute calibration
                result = await self.calibration.compute_calibration()
                if result:
                    self.state_manager.state.value_calibration_temp = result['optimal_temperature']
                    self.state_manager.state.calibration_ece = result['ece']
                    self.state_manager.save_state()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Calibration loop error: {e}")
                await asyncio.sleep(interval)

    async def _plateau_detection_loop(self):
        """Check for training plateaus."""
        interval = self.config.get('plateau_check_interval', 1800)  # 30 min

        while self._running:
            try:
                # Check for plateau
                is_plateau = await self.plateau_detector.check_plateau()

                if is_plateau:
                    logger.info("Plateau detected, optimization triggered")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Plateau detection error: {e}")
                await asyncio.sleep(interval)

    async def _cluster_monitoring_loop(self):
        """Monitor cluster health."""
        interval = self.config.get('cluster_monitor_interval', 60)

        while self._running:
            try:
                status = await self.p2p.get_cluster_status()
                if status:
                    nodes = status.get('nodes', {})
                    for node_name, node_info in nodes.items():
                        self.dashboard.record_cluster_status(
                            host_name=node_name,
                            cpu_percent=node_info.get('cpu', 0),
                            memory_percent=node_info.get('memory', 0),
                            gpu_util=node_info.get('gpu_util'),
                            jobs_running=node_info.get('jobs', 0)
                        )

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cluster monitoring error: {e}")
                await asyncio.sleep(interval)

    async def _state_persistence_loop(self):
        """Periodically save state."""
        interval = self.config.get('state_save_interval', 60)

        while self._running:
            try:
                self.state_manager.save_state()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State persistence error: {e}")
                await asyncio.sleep(interval)


# =============================================================================
# CLI Entry Point
# =============================================================================

def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    config = {
        'data_collection_interval': 60,
        'training_check_interval': 300,
        'training_threshold': 500,
        'training_epochs': 10,
        'batch_size': 256,
        'evaluation_interval': 3600,
        'calibration_interval': 7200,
        'plateau_check_interval': 1800,
        'plateau_lookback': 5,
        'min_elo_improvement': 15.0,
        'cluster_monitor_interval': 60,
        'state_save_interval': 60
    }

    if config_path and config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                loaded = yaml.safe_load(f)

            # Merge with defaults
            if loaded:
                for section in ['data_ingestion', 'training', 'evaluation', 'adaptive_control']:
                    if section in loaded:
                        for k, v in loaded[section].items():
                            config_key = f"{section}_{k}" if section != 'training' else k
                            config[config_key] = v
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    return config


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Integrated Self-Improvement Controller')
    parser.add_argument('--base-dir', type=Path, default=AI_SERVICE_DIR,
                        help='Base directory for AI service')
    parser.add_argument('--config', type=Path, default=None,
                        help='Configuration file (YAML)')
    parser.add_argument('--p2p-url', type=str, default=None,
                        help='P2P orchestrator URL (e.g., http://localhost:8765)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = load_config(args.config or AI_SERVICE_DIR / "config" / "unified_loop.yaml")

    # Create controller
    controller = IntegratedSelfImprovementController(
        base_dir=args.base_dir,
        p2p_url=args.p2p_url,
        config=config
    )

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(controller.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run
    try:
        await controller.start()
    except KeyboardInterrupt:
        pass
    finally:
        await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())
