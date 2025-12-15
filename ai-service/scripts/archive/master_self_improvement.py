#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated. Use scripts/unified_ai_loop.py instead.

The unified_ai_loop.py provides all functionality with proper cluster coordination,
event-driven data flow, and feedback loop integration.

To migrate:
    python scripts/unified_ai_loop.py --foreground --verbose

---
Master Self-Improvement Loop for RingRift AI.

This script ties together all the integration components into a unified
self-improvement system that maximizes AI training effectiveness.

Components Integrated:
1. Model Registry - Version tracking and lifecycle management
2. Model Lifecycle Manager - Promotion decisions and rollback
3. Pipeline Feedback Controller - Adaptive training based on metrics
4. P2P Integration Manager - Distributed cluster coordination
5. Training Dashboard - Real-time monitoring
6. Value Calibration - Calibrated predictions
7. Temperature Scheduling - Exploration/exploitation balance
8. Benchmark Suite - Comprehensive evaluation
9. Fault Tolerance - Crash recovery
10. Unified Loop Extensions - Enhancement of base loop

Architecture:
                    ┌─────────────────────────────────────┐
                    │     Master Self-Improvement Loop     │
                    └─────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
    ┌───────────┐            ┌───────────────┐          ┌───────────────┐
    │   P2P     │◄──────────►│   Lifecycle   │◄────────►│   Pipeline    │
    │Integration│            │   Manager     │          │   Feedback    │
    └───────────┘            └───────────────┘          └───────────────┘
          │                          │                          │
          │                          │                          │
          ▼                          ▼                          ▼
    ┌───────────┐            ┌───────────────┐          ┌───────────────┐
    │  Cluster  │            │    Model      │          │   Training    │
    │   Nodes   │            │   Registry    │          │   Dashboard   │
    └───────────┘            └───────────────┘          └───────────────┘

Usage:
    # Start the master loop
    python scripts/master_self_improvement.py

    # With custom P2P endpoint
    python scripts/master_self_improvement.py --p2p-url http://192.168.1.100:8770

    # With specific mode
    python scripts/master_self_improvement.py --mode continuous
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

class ImprovementMode(Enum):
    """Modes of operation for the self-improvement loop."""
    CONTINUOUS = "continuous"    # Run indefinitely
    SINGLE_CYCLE = "single"      # Run one improvement cycle
    TRAINING_ONLY = "training"   # Training without selfplay
    EVALUATION_ONLY = "eval"     # Evaluation only


@dataclass
class MasterConfig:
    """Configuration for the master self-improvement loop."""
    # Operation mode
    mode: ImprovementMode = ImprovementMode.CONTINUOUS

    # P2P settings
    p2p_base_url: str = "http://localhost:8770"
    p2p_auth_token: Optional[str] = None

    # Registry settings
    registry_dir: str = "data/model_registry"
    model_storage_dir: str = "data/models"

    # Training settings - match app/config/unified_config.py
    min_games_for_training: int = 500  # Canonical: 500
    training_interval_hours: float = 4.0
    auto_train: bool = True

    # Evaluation settings - match app/config/unified_config.py
    min_games_for_promotion: int = 200
    elo_improvement_threshold: float = 25.0  # Canonical: 25 (was 20)
    tournament_games_per_pair: int = 50

    # Selfplay settings
    target_selfplay_rate: int = 1000  # games per hour
    auto_scale_selfplay: bool = True

    # Calibration settings
    calibrate_on_promotion: bool = True
    calibration_interval_hours: float = 12.0

    # Feedback settings
    enable_curriculum_feedback: bool = True
    enable_cmaes_trigger: bool = True
    plateau_detection_window: int = 5

    # Health settings
    health_check_interval: float = 30.0
    max_consecutive_failures: int = 3


# ============================================
# Component Wrappers
# ============================================

class ComponentManager:
    """Manages lifecycle of all integration components."""

    def __init__(self, config: MasterConfig):
        self.config = config
        self._components: Dict[str, Any] = {}
        self._running = False

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing components...")

        # Initialize P2P integration
        try:
            from app.integration.p2p_integration import (
                P2PIntegrationManager, P2PIntegrationConfig
            )
            p2p_config = P2PIntegrationConfig(
                p2p_base_url=self.config.p2p_base_url,
                auth_token=self.config.p2p_auth_token,
                auto_scale_selfplay=self.config.auto_scale_selfplay,
                target_selfplay_games_per_hour=self.config.target_selfplay_rate
            )
            self._components['p2p'] = P2PIntegrationManager(p2p_config)
            logger.info("P2P integration initialized")
        except ImportError as e:
            logger.warning(f"P2P integration not available: {e}")

        # Initialize lifecycle manager
        try:
            from app.integration.model_lifecycle import (
                ModelLifecycleManager, LifecycleConfig
            )
            lifecycle_config = LifecycleConfig(
                registry_dir=self.config.registry_dir,
                model_storage_dir=self.config.model_storage_dir,
                min_elo_improvement=self.config.elo_improvement_threshold,
                min_games_for_production=self.config.min_games_for_promotion,
                p2p_api_base=self.config.p2p_base_url
            )
            self._components['lifecycle'] = ModelLifecycleManager(lifecycle_config)
            logger.info("Lifecycle manager initialized")
        except ImportError as e:
            logger.warning(f"Lifecycle manager not available: {e}")

        # Initialize pipeline feedback
        try:
            from app.integration.pipeline_feedback import PipelineFeedbackController
            self._components['feedback'] = PipelineFeedbackController()
            logger.info("Pipeline feedback initialized")
        except ImportError as e:
            logger.warning(f"Pipeline feedback not available: {e}")

        # Initialize dashboard (optional)
        try:
            from app.monitoring.training_dashboard import DashboardServer
            self._components['dashboard'] = DashboardServer()
            logger.info("Dashboard initialized")
        except ImportError as e:
            logger.debug(f"Dashboard not available: {e}")

        # Initialize value calibrator (optional)
        try:
            from app.training.value_calibration import ValueCalibrator
            self._components['calibrator'] = ValueCalibrator()
            logger.info("Value calibrator initialized")
        except ImportError as e:
            logger.debug(f"Value calibrator not available: {e}")

        # Initialize benchmark suite (optional)
        try:
            from app.evaluation.benchmark_suite import BenchmarkSuite
            self._components['benchmark'] = BenchmarkSuite()
            logger.info("Benchmark suite initialized")
        except ImportError as e:
            logger.debug(f"Benchmark suite not available: {e}")

    async def start(self) -> None:
        """Start all components."""
        self._running = True

        # Start P2P integration
        if 'p2p' in self._components:
            await self._components['p2p'].start()

        # Start lifecycle manager
        if 'lifecycle' in self._components:
            await self._components['lifecycle'].start()

        # Start dashboard
        if 'dashboard' in self._components:
            try:
                await self._components['dashboard'].start()
            except Exception as e:
                logger.warning(f"Failed to start dashboard: {e}")

        logger.info("All components started")

    async def stop(self) -> None:
        """Stop all components."""
        self._running = False

        # Stop in reverse order
        if 'dashboard' in self._components:
            try:
                await self._components['dashboard'].stop()
            except Exception:
                pass

        if 'lifecycle' in self._components:
            await self._components['lifecycle'].stop()

        if 'p2p' in self._components:
            await self._components['p2p'].stop()

        logger.info("All components stopped")

    def get(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        return self._components.get(name)


# ============================================
# Event Bus
# ============================================

class EventBus:
    """Simple event bus for component communication."""

    def __init__(self):
        self._handlers: Dict[str, List] = {}

    def subscribe(self, event: str, handler) -> None:
        """Subscribe to an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    async def publish(self, event: str, **kwargs) -> None:
        """Publish an event."""
        for handler in self._handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(**kwargs)
                else:
                    handler(**kwargs)
            except Exception as e:
                logger.error(f"Event handler error for {event}: {e}")


# ============================================
# Master Loop
# ============================================

class MasterSelfImprovementLoop:
    """
    Master orchestrator for the AI self-improvement loop.

    Coordinates all components to maximize training effectiveness.
    """

    def __init__(self, config: Optional[MasterConfig] = None):
        self.config = config or MasterConfig()
        self.components = ComponentManager(self.config)
        self.event_bus = EventBus()

        # State
        self._running = False
        self._cycle_count = 0
        self._last_training: Optional[datetime] = None
        self._last_calibration: Optional[datetime] = None
        self._current_phase = "idle"

        # Tasks
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize the master loop."""
        logger.info("Initializing master self-improvement loop...")

        # Initialize components
        await self.components.initialize()

        # Set up event handlers
        self._setup_event_handlers()

        logger.info("Master loop initialized")

    def _setup_event_handlers(self) -> None:
        """Set up inter-component event handlers."""

        # Lifecycle -> P2P: Sync promoted models
        lifecycle = self.components.get('lifecycle')
        p2p = self.components.get('p2p')

        if lifecycle and p2p:
            async def on_model_promoted(**kwargs):
                if kwargs.get('stage') == 'production':
                    logger.info(f"Syncing promoted model to cluster")
                    # Trigger cluster sync

            lifecycle.register_callback('model_promoted', on_model_promoted)

        # Feedback -> Training: Trigger training on plateau
        feedback = self.components.get('feedback')
        if feedback and lifecycle:
            # Set up feedback to trigger training/optimization
            pass

    async def start(self) -> None:
        """Start the master loop."""
        self._running = True

        # Start components
        await self.components.start()

        # Start main loop based on mode
        if self.config.mode == ImprovementMode.CONTINUOUS:
            self._tasks.append(asyncio.create_task(self._continuous_loop()))
        elif self.config.mode == ImprovementMode.SINGLE_CYCLE:
            self._tasks.append(asyncio.create_task(self._single_cycle()))
        elif self.config.mode == ImprovementMode.TRAINING_ONLY:
            self._tasks.append(asyncio.create_task(self._training_loop()))
        elif self.config.mode == ImprovementMode.EVALUATION_ONLY:
            self._tasks.append(asyncio.create_task(self._evaluation_loop()))

        # Start monitoring loops
        self._tasks.append(asyncio.create_task(self._monitoring_loop()))
        self._tasks.append(asyncio.create_task(self._calibration_loop()))

        logger.info(f"Master loop started in {self.config.mode.value} mode")

    async def stop(self) -> None:
        """Stop the master loop."""
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop components
        await self.components.stop()

        logger.info("Master loop stopped")

    # ==========================================
    # Main Loops
    # ==========================================

    async def _continuous_loop(self) -> None:
        """Continuous improvement loop."""
        while self._running:
            try:
                # Run improvement cycle
                await self._run_cycle()

                self._cycle_count += 1
                logger.info(f"Completed improvement cycle {self._cycle_count}")

                # Wait before next cycle
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous loop error: {e}")
                await asyncio.sleep(300)

    async def _single_cycle(self) -> None:
        """Run a single improvement cycle."""
        try:
            await self._run_cycle()
            logger.info("Single cycle completed")
        except Exception as e:
            logger.error(f"Single cycle error: {e}")
        finally:
            self._running = False

    async def _training_loop(self) -> None:
        """Training-only loop."""
        while self._running:
            try:
                await self._run_training_phase()
                await asyncio.sleep(self.config.training_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(300)

    async def _evaluation_loop(self) -> None:
        """Evaluation-only loop."""
        while self._running:
            try:
                await self._run_evaluation_phase()
                await asyncio.sleep(3600)  # Evaluate hourly
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
                await asyncio.sleep(300)

    # ==========================================
    # Cycle Phases
    # ==========================================

    async def _run_cycle(self) -> None:
        """Run a complete improvement cycle."""
        logger.info("Starting improvement cycle...")

        # Phase 1: Check selfplay status and auto-scale
        self._current_phase = "selfplay_check"
        await self._check_selfplay()

        # Phase 2: Check if training is needed
        self._current_phase = "training_check"
        should_train, reason = await self._check_training_trigger()

        if should_train:
            self._current_phase = "training"
            await self._run_training_phase()

        # Phase 3: Evaluate new models
        self._current_phase = "evaluation"
        await self._run_evaluation_phase()

        # Phase 4: Check for optimization triggers
        self._current_phase = "optimization_check"
        await self._check_optimization_triggers()

        self._current_phase = "idle"

    async def _check_selfplay(self) -> None:
        """Check selfplay status and auto-scale if needed."""
        p2p = self.components.get('p2p')
        if not p2p:
            return

        try:
            # Get cluster status
            summary = await p2p.get_cluster_summary()

            if 'error' in summary:
                logger.warning(f"Failed to get cluster status: {summary['error']}")
                return

            # Log current state
            logger.info(
                f"Cluster: {summary.get('alive_nodes', 0)}/{summary.get('total_nodes', 0)} nodes, "
                f"{summary.get('total_selfplay_jobs', 0)} selfplay jobs"
            )

            # Auto-scale selfplay
            if self.config.auto_scale_selfplay:
                result = await p2p.selfplay.auto_scale()
                if result.get('actions'):
                    logger.info(f"Auto-scaled selfplay: {len(result['actions'])} actions")

        except Exception as e:
            logger.error(f"Selfplay check error: {e}")

    async def _check_training_trigger(self) -> tuple:
        """Check if training should be triggered."""
        lifecycle = self.components.get('lifecycle')

        if not lifecycle:
            return (False, "Lifecycle manager not available")

        if not self.config.auto_train:
            return (False, "Auto-training disabled")

        # Check time since last training
        if self._last_training:
            hours_since = (datetime.now() - self._last_training).total_seconds() / 3600
            if hours_since < self.config.training_interval_hours:
                return (False, f"Too soon since last training ({hours_since:.1f}h)")

        # Get game count from P2P
        p2p = self.components.get('p2p')
        if p2p:
            try:
                status = await p2p.client.get_cluster_status()
                games = status.get('total_games', 0)

                # Simple trigger based on game count
                if games >= self.config.min_games_for_training:
                    return (True, f"Sufficient games: {games}")
            except Exception as e:
                logger.error(f"Failed to check game count: {e}")

        return (False, "Conditions not met")

    async def _run_training_phase(self) -> None:
        """Run the training phase."""
        logger.info("Starting training phase...")

        p2p = self.components.get('p2p')
        lifecycle = self.components.get('lifecycle')

        if not p2p:
            logger.warning("P2P not available for training")
            return

        try:
            # Start training on cluster
            result = await p2p.trigger_training(wait_for_completion=False)

            if 'error' in result:
                logger.error(f"Training failed to start: {result['error']}")
                return

            logger.info(f"Training started: {result}")

            # Poll for completion
            while self._running:
                status = await p2p.training.get_status()
                train_status = status.get('status', 'unknown')

                if train_status == 'completed':
                    logger.info("Training completed successfully")
                    self._last_training = datetime.now()

                    # Register new model with lifecycle
                    if lifecycle:
                        model_path = status.get('model_path')
                        if model_path:
                            await lifecycle.register_model(
                                name="RingRift AI",
                                model_path=Path(model_path),
                                training_config=status.get('config', {})
                            )

                    break

                elif train_status == 'failed':
                    logger.error(f"Training failed: {status.get('error')}")
                    break

                await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Training phase error: {e}")

    async def _run_evaluation_phase(self) -> None:
        """Run the evaluation phase."""
        logger.info("Starting evaluation phase...")

        p2p = self.components.get('p2p')
        lifecycle = self.components.get('lifecycle')

        if not p2p:
            return

        try:
            # Get models to evaluate
            leaderboard = await p2p.evaluation.get_leaderboard()

            if not leaderboard:
                logger.info("No models to evaluate")
                return

            # Run tournament with recent models
            recent_models = [m['model_id'] for m in leaderboard[:5]]

            result = await p2p.evaluation.run_tournament(
                recent_models,
                self.config.tournament_games_per_pair
            )

            logger.info(f"Tournament started: {result}")

            # Update lifecycle with results
            if lifecycle and 'results' in result:
                for model_result in result['results']:
                    from app.integration.model_lifecycle import EvaluationResult

                    eval_result = EvaluationResult(
                        model_id=model_result.get('model_id', ''),
                        version=model_result.get('version', 1),
                        elo=model_result.get('elo'),
                        games_played=model_result.get('games', 0),
                        win_rate=model_result.get('win_rate')
                    )

                    await lifecycle.submit_evaluation(
                        eval_result.model_id,
                        eval_result.version,
                        eval_result
                    )

        except Exception as e:
            logger.error(f"Evaluation phase error: {e}")

    async def _check_optimization_triggers(self) -> None:
        """Check if optimization (CMA-ES, NAS) should be triggered."""
        feedback = self.components.get('feedback')

        if not feedback or not self.config.enable_cmaes_trigger:
            return

        try:
            # Check for plateau
            signals = feedback.get_pending_signals()

            for signal in signals:
                if signal.action.name == 'TRIGGER_CMAES':
                    logger.info(f"CMA-ES triggered: {signal.reason}")

                    p2p = self.components.get('p2p')
                    if p2p:
                        await p2p.client.start_cmaes()

        except Exception as e:
            logger.error(f"Optimization check error: {e}")

    # ==========================================
    # Supporting Loops
    # ==========================================

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect and log metrics
                await self._collect_metrics()
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _calibration_loop(self) -> None:
        """Background calibration loop."""
        while self._running:
            try:
                # Check if calibration is needed
                if self._should_calibrate():
                    await self._run_calibration()

                await asyncio.sleep(3600)  # Check hourly

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Calibration loop error: {e}")
                await asyncio.sleep(3600)

    def _should_calibrate(self) -> bool:
        """Check if calibration is needed."""
        if not self.config.calibrate_on_promotion:
            return False

        if self._last_calibration is None:
            return True

        hours_since = (datetime.now() - self._last_calibration).total_seconds() / 3600
        return hours_since >= self.config.calibration_interval_hours

    async def _run_calibration(self) -> None:
        """Run value head calibration."""
        calibrator = self.components.get('calibrator')
        if not calibrator:
            return

        logger.info("Running value calibration...")
        self._last_calibration = datetime.now()

    async def _collect_metrics(self) -> None:
        """Collect and publish metrics."""
        dashboard = self.components.get('dashboard')
        p2p = self.components.get('p2p')

        if dashboard and p2p:
            try:
                summary = await p2p.get_cluster_summary()
                # Dashboard would collect these metrics
            except Exception:
                pass

    # ==========================================
    # Status
    # ==========================================

    def get_status(self) -> Dict[str, Any]:
        """Get master loop status."""
        return {
            'running': self._running,
            'mode': self.config.mode.value,
            'current_phase': self._current_phase,
            'cycle_count': self._cycle_count,
            'last_training': self._last_training.isoformat() if self._last_training else None,
            'last_calibration': self._last_calibration.isoformat() if self._last_calibration else None,
            'config': asdict(self.config)
        }


# ============================================
# Main Entry Point
# ============================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master Self-Improvement Loop for RingRift AI"
    )
    parser.add_argument(
        '--p2p-url',
        default=os.environ.get('RINGRIFT_P2P_URL', 'http://localhost:8770'),
        help='P2P orchestrator URL'
    )
    parser.add_argument(
        '--mode',
        choices=['continuous', 'single', 'training', 'eval'],
        default='continuous',
        help='Operation mode'
    )
    parser.add_argument(
        '--no-auto-scale',
        action='store_true',
        help='Disable auto-scaling selfplay'
    )
    parser.add_argument(
        '--training-interval',
        type=float,
        default=4.0,
        help='Hours between training runs'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=500,
        help='Minimum games for training'
    )

    args = parser.parse_args()

    # Create config
    mode_map = {
        'continuous': ImprovementMode.CONTINUOUS,
        'single': ImprovementMode.SINGLE_CYCLE,
        'training': ImprovementMode.TRAINING_ONLY,
        'eval': ImprovementMode.EVALUATION_ONLY
    }

    config = MasterConfig(
        mode=mode_map[args.mode],
        p2p_base_url=args.p2p_url,
        auto_scale_selfplay=not args.no_auto_scale,
        training_interval_hours=args.training_interval,
        min_games_for_training=args.min_games
    )

    # Create master loop
    loop = MasterSelfImprovementLoop(config)

    # Handle shutdown
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

    try:
        # Initialize and start
        await loop.initialize()
        await loop.start()

        # Print status
        status = loop.get_status()
        logger.info(f"Master loop running: {json.dumps(status, indent=2, default=str)}")

        # Wait for shutdown
        await shutdown_event.wait()

    finally:
        await loop.stop()


if __name__ == "__main__":
    asyncio.run(main())
