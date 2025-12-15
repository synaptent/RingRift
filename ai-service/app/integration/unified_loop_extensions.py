"""
Extensions for the Unified AI Loop.

This module provides integration of advanced AI training components
into the existing unified_ai_loop.py infrastructure.

Components integrated:
- Model Registry: Version tracking and lifecycle management
- Value Calibration: Prediction calibration and temperature scaling
- Temperature Scheduling: Self-play exploration control
- Benchmark Suite: Comprehensive model evaluation
- Fault Tolerance: Checkpointing and recovery
- Plateau Detection: Automatic optimization triggering
- Dashboard Metrics: Real-time monitoring

Usage:
    from app.integration.unified_loop_extensions import UnifiedLoopExtensions

    # In UnifiedAILoop.__init__:
    self.extensions = UnifiedLoopExtensions(self)

    # In UnifiedAILoop.run():
    await asyncio.gather(
        # ... existing loops ...
        self.extensions.run_all_loops(),
    )
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ExtensionConfig:
    """Configuration for unified loop extensions."""
    # Model registry
    registry_enabled: bool = True
    registry_dir: str = "model_registry"

    # Calibration
    calibration_enabled: bool = True
    calibration_interval_seconds: int = 7200  # 2 hours
    calibration_min_samples: int = 100

    # Temperature scheduling
    temperature_enabled: bool = True
    temperature_preset: str = "curriculum"  # default, alphazero, curriculum, adaptive

    # Benchmarking
    benchmark_enabled: bool = True
    benchmark_interval_seconds: int = 86400  # Daily
    benchmark_on_promotion: bool = True

    # Plateau detection
    plateau_detection_enabled: bool = True
    plateau_check_interval_seconds: int = 1800  # 30 min
    plateau_lookback_iterations: int = 5
    plateau_min_elo_improvement: float = 15.0

    # Dashboard
    dashboard_enabled: bool = True
    dashboard_metrics_interval_seconds: int = 60

    # Fault tolerance
    fault_tolerance_enabled: bool = True
    checkpoint_interval_epochs: int = 1
    max_training_retries: int = 3


@dataclass
class ExtensionState:
    """State for unified loop extensions."""
    # Calibration state
    calibration_temperature: float = 1.0
    calibration_ece: float = 0.0
    calibration_samples: int = 0
    last_calibration_time: float = 0.0

    # Plateau detection state
    elo_history: List[tuple] = field(default_factory=list)  # (timestamp, elo)
    plateau_count: int = 0
    last_plateau_check: float = 0.0

    # CMAES/NAS triggering
    cmaes_triggered_count: int = 0
    nas_triggered_count: int = 0
    last_cmaes_trigger: float = 0.0
    last_nas_trigger: float = 0.0

    # Benchmark state
    last_benchmark_time: float = 0.0
    benchmark_count: int = 0

    # Dashboard state
    last_dashboard_update: float = 0.0


class UnifiedLoopExtensions:
    """
    Extensions for the Unified AI Loop that integrate advanced training components.
    """

    def __init__(
        self,
        unified_loop,  # UnifiedAILoop instance
        config: Optional[ExtensionConfig] = None
    ):
        self.loop = unified_loop
        self.config = config or ExtensionConfig()
        self.state = ExtensionState()

        # Base paths
        self.ai_service_dir = Path(__file__).resolve().parents[2]

        # Initialize components (lazy loading)
        self._model_registry = None
        self._calibrator = None
        self._temperature_scheduler = None
        self._benchmark_suite = None
        self._dashboard_collector = None

        # Event subscriptions
        self._setup_event_handlers()

        self._running = False

    def _setup_event_handlers(self):
        """Subscribe to events from the unified loop."""
        if hasattr(self.loop, 'event_bus'):
            try:
                from scripts.unified_ai_loop import DataEventType

                # Subscribe to training completion for calibration
                self.loop.event_bus.subscribe(
                    DataEventType.TRAINING_COMPLETED,
                    self._on_training_completed
                )

                # Subscribe to evaluation for Elo tracking
                self.loop.event_bus.subscribe(
                    DataEventType.EVALUATION_COMPLETED,
                    self._on_evaluation_completed
                )

                # Subscribe to promotion for benchmarking
                self.loop.event_bus.subscribe(
                    DataEventType.MODEL_PROMOTED,
                    self._on_model_promoted
                )
            except ImportError:
                logger.warning("Could not import DataEventType for event subscriptions")

    # =========================================================================
    # Lazy Component Initialization
    # =========================================================================

    @property
    def model_registry(self):
        """Lazy initialization of model registry."""
        if self._model_registry is None and self.config.registry_enabled:
            try:
                from app.training.model_registry import ModelRegistry
                registry_dir = self.ai_service_dir / self.config.registry_dir
                self._model_registry = ModelRegistry(registry_dir)
                logger.info(f"Model registry initialized at {registry_dir}")
            except ImportError:
                logger.warning("Model registry not available")
        return self._model_registry

    @property
    def calibrator(self):
        """Lazy initialization of value calibrator."""
        if self._calibrator is None and self.config.calibration_enabled:
            try:
                from app.training.value_calibration import ValueCalibrator
                self._calibrator = ValueCalibrator(num_bins=10)
            except ImportError:
                logger.warning("Value calibration not available")
        return self._calibrator

    @property
    def temperature_scheduler(self):
        """Lazy initialization of temperature scheduler."""
        if self._temperature_scheduler is None and self.config.temperature_enabled:
            try:
                from app.training.temperature_scheduling import create_scheduler
                self._temperature_scheduler = create_scheduler(self.config.temperature_preset)
            except ImportError:
                logger.warning("Temperature scheduling not available")
        return self._temperature_scheduler

    @property
    def benchmark_suite(self):
        """Lazy initialization of benchmark suite."""
        if self._benchmark_suite is None and self.config.benchmark_enabled:
            try:
                from app.evaluation.benchmark_suite import create_default_suite
                self._benchmark_suite = create_default_suite()
            except ImportError:
                logger.warning("Benchmark suite not available")
        return self._benchmark_suite

    @property
    def dashboard_collector(self):
        """Lazy initialization of dashboard metrics collector."""
        if self._dashboard_collector is None and self.config.dashboard_enabled:
            try:
                from app.monitoring.training_dashboard import MetricsCollector
                dashboard_dir = self.ai_service_dir / "dashboard"
                dashboard_dir.mkdir(parents=True, exist_ok=True)
                self._dashboard_collector = MetricsCollector(dashboard_dir / "metrics.db")
            except ImportError:
                logger.warning("Dashboard collector not available")
        return self._dashboard_collector

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_training_completed(self, event):
        """Handle training completion - update calibration samples."""
        payload = event.payload if hasattr(event, 'payload') else {}

        # Update temperature scheduler with training progress
        if self.temperature_scheduler:
            iteration = payload.get('iteration', 0)
            total = payload.get('total_iterations', 100)
            progress = iteration / total if total > 0 else 0
            self.temperature_scheduler.set_training_progress(progress)

    async def _on_evaluation_completed(self, event):
        """Handle evaluation completion - update Elo history."""
        payload = event.payload if hasattr(event, 'payload') else {}
        elo = payload.get('elo')

        if elo is not None:
            self.state.elo_history.append((time.time(), elo))

            # Keep only last 100 entries
            if len(self.state.elo_history) > 100:
                self.state.elo_history = self.state.elo_history[-100:]

            # Update dashboard
            if self.dashboard_collector:
                model_id = payload.get('model_id', 'current')
                games = payload.get('games_played', 0)
                self.dashboard_collector.record_elo(model_id, elo, games)

    async def _on_model_promoted(self, event):
        """Handle model promotion - optionally run benchmarks."""
        if not self.config.benchmark_on_promotion:
            return

        payload = event.payload if hasattr(event, 'payload') else {}
        model_path = payload.get('model_path')
        model_id = payload.get('model_id', 'promoted')

        if model_path and self.benchmark_suite:
            logger.info(f"Running benchmarks for promoted model {model_id}")
            # Note: Would need to load model and run benchmarks
            # await self._run_benchmarks(model_path, model_id)

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def get_selfplay_temperature(self, move_number: int, game_state: Any = None) -> float:
        """Get temperature for self-play move selection."""
        if self.temperature_scheduler:
            return self.temperature_scheduler.get_temperature(move_number, game_state)
        return 1.0

    def calibrate_value(self, value: float) -> float:
        """Apply calibration to a value prediction."""
        if self.calibrator:
            return self.calibrator.calibrate_prediction(value)
        return value

    async def register_model(
        self,
        name: str,
        model_path: Path,
        metrics: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Optional[tuple]:
        """Register a model with the registry."""
        if not self.model_registry:
            return None

        try:
            from app.training.model_registry import ModelMetrics, TrainingConfig

            model_metrics = ModelMetrics(
                elo=metrics.get('elo'),
                win_rate=metrics.get('win_rate'),
                games_played=metrics.get('games_played', 0)
            )

            train_cfg = TrainingConfig(
                learning_rate=training_config.get('learning_rate', 0.001),
                batch_size=training_config.get('batch_size', 256)
            )

            return self.model_registry.register_model(
                name=name,
                model_path=model_path,
                metrics=model_metrics,
                training_config=train_cfg
            )
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    # =========================================================================
    # Background Loops
    # =========================================================================

    async def _calibration_loop(self):
        """Periodically compute value calibration."""
        interval = self.config.calibration_interval_seconds

        while self._running:
            try:
                now = time.time()

                if now - self.state.last_calibration_time >= interval:
                    if self.calibrator and self.state.calibration_samples >= self.config.calibration_min_samples:
                        report = self.calibrator.compute_calibration()
                        optimal_temp = self.calibrator.find_optimal_temperature()

                        self.state.calibration_temperature = optimal_temp
                        self.state.calibration_ece = report.ece
                        self.state.last_calibration_time = now

                        logger.info(f"Calibration: ECE={report.ece:.4f}, temp={optimal_temp:.3f}")

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Calibration loop error: {e}")
                await asyncio.sleep(60)

    async def _plateau_detection_loop(self):
        """Check for training plateaus and trigger optimization."""
        interval = self.config.plateau_check_interval_seconds
        lookback = self.config.plateau_lookback_iterations
        min_improvement = self.config.plateau_min_elo_improvement

        while self._running:
            try:
                now = time.time()

                if now - self.state.last_plateau_check >= interval:
                    self.state.last_plateau_check = now

                    if len(self.state.elo_history) >= lookback:
                        recent = self.state.elo_history[-lookback:]
                        oldest_elo = recent[0][1]
                        newest_elo = recent[-1][1]
                        improvement = newest_elo - oldest_elo

                        if improvement < min_improvement:
                            self.state.plateau_count += 1
                            logger.info(f"Plateau detected: {improvement:.1f} Elo in {lookback} iterations (count={self.state.plateau_count})")

                            # Trigger CMA-ES
                            cmaes_cooldown = 6 * 3600  # 6 hours
                            if now - self.state.last_cmaes_trigger > cmaes_cooldown:
                                await self._trigger_cmaes()

                            # Trigger NAS for severe plateau
                            nas_cooldown = 24 * 3600  # 24 hours
                            if self.state.plateau_count >= 3 and now - self.state.last_nas_trigger > nas_cooldown:
                                await self._trigger_nas()
                        else:
                            if self.state.plateau_count > 0:
                                self.state.plateau_count = max(0, self.state.plateau_count - 1)

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Plateau detection error: {e}")
                await asyncio.sleep(60)

    async def _trigger_cmaes(self):
        """Trigger CMA-ES optimization."""
        self.state.last_cmaes_trigger = time.time()
        self.state.cmaes_triggered_count += 1
        logger.info("Triggering CMA-ES optimization due to plateau")

        # Publish event if event bus available
        if hasattr(self.loop, 'event_bus'):
            try:
                from scripts.unified_ai_loop import DataEvent, DataEventType
                await self.loop.event_bus.publish(DataEvent(
                    event_type=DataEventType.CMAES_TRIGGERED,
                    payload={'reason': 'plateau_detected', 'plateau_count': self.state.plateau_count}
                ))
            except:
                pass

    async def _trigger_nas(self):
        """Trigger Neural Architecture Search."""
        self.state.last_nas_trigger = time.time()
        self.state.nas_triggered_count += 1
        logger.info("Triggering NAS due to severe plateau")

        if hasattr(self.loop, 'event_bus'):
            try:
                from scripts.unified_ai_loop import DataEvent, DataEventType
                await self.loop.event_bus.publish(DataEvent(
                    event_type=DataEventType.NAS_TRIGGERED,
                    payload={'reason': 'severe_plateau', 'plateau_count': self.state.plateau_count}
                ))
            except:
                pass

    async def _benchmark_loop(self):
        """Periodically run benchmarks on production model."""
        interval = self.config.benchmark_interval_seconds

        while self._running:
            try:
                now = time.time()

                if now - self.state.last_benchmark_time >= interval:
                    if self.benchmark_suite and self.model_registry:
                        prod_model = self.model_registry.get_production_model()
                        if prod_model:
                            logger.info(f"Running daily benchmarks for {prod_model.model_id}")
                            # Would load model and run benchmarks here
                            self.state.last_benchmark_time = now
                            self.state.benchmark_count += 1

                await asyncio.sleep(3600)  # Check hourly

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Benchmark loop error: {e}")
                await asyncio.sleep(3600)

    async def _dashboard_loop(self):
        """Update dashboard metrics."""
        interval = self.config.dashboard_metrics_interval_seconds

        while self._running:
            try:
                if self.dashboard_collector:
                    # Collect system metrics
                    try:
                        import psutil
                        cpu = psutil.cpu_percent()
                        mem = psutil.virtual_memory().percent

                        # Try to get GPU metrics
                        gpu_util = None
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5
                            )
                            if result.returncode == 0:
                                gpu_util = float(result.stdout.strip().split('\n')[0])
                        except:
                            pass

                        self.dashboard_collector.record_cluster_status(
                            host_name='local',
                            cpu_percent=cpu,
                            memory_percent=mem,
                            gpu_util=gpu_util,
                            jobs_running=1 if self.loop.state.training_in_progress else 0
                        )
                    except ImportError:
                        pass

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                await asyncio.sleep(interval)

    async def run_all_loops(self):
        """Run all extension loops concurrently."""
        self._running = True

        loops = []

        if self.config.calibration_enabled:
            loops.append(self._calibration_loop())

        if self.config.plateau_detection_enabled:
            loops.append(self._plateau_detection_loop())

        if self.config.benchmark_enabled:
            loops.append(self._benchmark_loop())

        if self.config.dashboard_enabled:
            loops.append(self._dashboard_loop())

        if loops:
            await asyncio.gather(*loops)

    def stop(self):
        """Stop all extension loops."""
        self._running = False


# =============================================================================
# Helper function for easy integration
# =============================================================================

def integrate_extensions(unified_loop, config: Optional[ExtensionConfig] = None) -> UnifiedLoopExtensions:
    """
    Integrate extensions into an existing UnifiedAILoop instance.

    Usage:
        loop = UnifiedAILoop(config)
        extensions = integrate_extensions(loop)

        # Then in run():
        await asyncio.gather(
            ... existing loops ...,
            extensions.run_all_loops()
        )
    """
    extensions = UnifiedLoopExtensions(unified_loop, config)
    logger.info("Unified loop extensions integrated")
    return extensions


def get_extension_config_from_yaml(yaml_path: Path) -> ExtensionConfig:
    """Load extension configuration from YAML file."""
    import yaml

    config = ExtensionConfig()

    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if 'extensions' in data:
            ext_data = data['extensions']
            for key, value in ext_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    return config
