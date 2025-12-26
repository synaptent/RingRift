#!/usr/bin/env python3
"""Master Loop Controller - Unified automation entry point for RingRift AI training.

This is the single daemon that orchestrates all automation:
- Selfplay allocation across cluster
- Training triggering based on data freshness
- Evaluation and promotion
- Model distribution
- Cluster health monitoring
- Feedback loop integration

Unlike the separate daemon scripts, this provides a unified control plane
that makes high-level decisions about resource allocation and priorities.

Architecture:
    MasterLoopController
    ├── DaemonManager: Lifecycle for all background daemons
    ├── ClusterMonitor: Real-time cluster health
    ├── AdaptiveResourceManager: Resource tracking
    ├── FeedbackLoopController: Training feedback signals
    ├── DataPipelineOrchestrator: Pipeline stage tracking
    └── SelfplayScheduler: Priority-based selfplay allocation

Usage:
    # Full automation mode
    python scripts/master_loop.py

    # Watch mode (show status, don't run loop)
    python scripts/master_loop.py --watch

    # Specific configs only
    python scripts/master_loop.py --configs hex8_2p,square8_2p

    # Dry run (preview actions without executing)
    python scripts/master_loop.py --dry-run

    # Skip daemons (for testing)
    python scripts/master_loop.py --skip-daemons

December 2025: Created as part of strategic integration plan.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.thresholds import (
    PROMOTION_THRESHOLDS_BY_CONFIG,
    GPU_MEMORY_WEIGHTS,
    SELFPLAY_GAMES_PER_GPU_TYPE,
    TRAINING_BATCH_SIZE_BY_GPU,
    MAX_CONCURRENT_GAUNTLETS_BY_GPU,
    MIN_SAMPLES_FOR_TRAINING,
    MIN_AVG_GAME_LENGTH,
    MAX_DRAW_RATE_FOR_TRAINING,
    MIN_SELFPLAY_WIN_RATE_VS_HEURISTIC,
    is_ephemeral_node,
    check_training_data_quality,
    get_promotion_thresholds,
    get_gpu_weight,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# All supported board type / player count configurations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Loop timing
LOOP_INTERVAL_SECONDS = 60  # Check every minute
TRAINING_CHECK_INTERVAL = 300  # Check training readiness every 5 minutes
ALLOCATION_CHECK_INTERVAL = 600  # Rebalance allocations every 10 minutes

# Thresholds
MIN_GAMES_FOR_EXPORT = 1000  # Minimum new games before triggering export
MAX_DATA_STALENESS_HOURS = 4.0  # Trigger sync if data older than this


@dataclass
class ConfigState:
    """Tracking state for a single board/player configuration."""
    config_key: str

    # Data freshness
    last_export_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    games_since_last_export: int = 0

    # Quality metrics
    last_quality_score: float = 0.0
    last_policy_accuracy: float = 0.0
    last_evaluation_win_rate: float = 0.0

    # Current allocations
    selfplay_nodes_allocated: list[str] = field(default_factory=list)
    training_node: str | None = None

    # Promotion status
    pending_evaluation: bool = False
    last_promotion_success: bool | None = None

    # Feedback signals
    training_intensity: str = "normal"  # normal, accelerated, hot_path
    exploration_boost: float = 1.0

    @property
    def data_staleness_hours(self) -> float:
        """Hours since last export."""
        if self.last_export_time == 0:
            return float("inf")
        return (time.time() - self.last_export_time) / 3600

    @property
    def needs_training(self) -> bool:
        """Check if config needs training."""
        # Has enough games and data is fresh enough
        return (
            self.games_since_last_export >= MIN_GAMES_FOR_EXPORT
            and self.last_quality_score >= 0.5
        )


@dataclass
class ClusterHealth:
    """Aggregated cluster health status."""
    total_nodes: int = 0
    healthy_nodes: int = 0
    training_nodes: int = 0
    selfplay_nodes: int = 0
    avg_gpu_utilization: float = 0.0
    avg_disk_usage: float = 0.0
    load_critical: bool = False
    errors: list[str] = field(default_factory=list)


class MasterLoopController:
    """Single daemon managing all automation.

    Coordinates:
    - Selfplay allocation across cluster
    - Training triggering based on data freshness
    - Evaluation and promotion
    - Model distribution
    - Cluster health monitoring
    """

    def __init__(
        self,
        configs: list[str] | None = None,
        dry_run: bool = False,
        skip_daemons: bool = False,
    ):
        self.active_configs = configs or ALL_CONFIGS
        self.dry_run = dry_run
        self.skip_daemons = skip_daemons

        # State tracking
        self._states: dict[str, ConfigState] = {
            cfg: ConfigState(config_key=cfg) for cfg in self.active_configs
        }

        # Timing
        self._last_training_check = 0.0
        self._last_allocation_check = 0.0

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Lazy-loaded managers
        self._daemon_manager = None
        self._cluster_monitor = None
        self._resource_manager = None
        self._feedback_controller = None
        self._pipeline_orchestrator = None

    # =========================================================================
    # Lazy-loaded dependencies
    # =========================================================================

    @property
    def daemon_manager(self):
        """Get DaemonManager (lazy load)."""
        if self._daemon_manager is None:
            from app.coordination.daemon_manager import get_daemon_manager
            self._daemon_manager = get_daemon_manager()
        return self._daemon_manager

    @property
    def cluster_monitor(self):
        """Get ClusterMonitor (lazy load)."""
        if self._cluster_monitor is None:
            from app.distributed.cluster_monitor import ClusterMonitor
            self._cluster_monitor = ClusterMonitor()
        return self._cluster_monitor

    @property
    def resource_manager(self):
        """Get AdaptiveResourceManager (lazy load)."""
        if self._resource_manager is None:
            from app.coordination.adaptive_resource_manager import get_resource_manager
            self._resource_manager = get_resource_manager()
        return self._resource_manager

    @property
    def feedback_controller(self):
        """Get FeedbackLoopController (lazy load)."""
        if self._feedback_controller is None:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller
            self._feedback_controller = get_feedback_loop_controller()
        return self._feedback_controller

    @property
    def pipeline_orchestrator(self):
        """Get DataPipelineOrchestrator (lazy load)."""
        if self._pipeline_orchestrator is None:
            from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
            self._pipeline_orchestrator = get_pipeline_orchestrator()
        return self._pipeline_orchestrator

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the master loop."""
        if self._running:
            logger.warning("MasterLoopController already running")
            return

        self._running = True
        logger.info(f"[MasterLoop] Starting with {len(self.active_configs)} configs")
        logger.info(f"[MasterLoop] Dry run: {self.dry_run}, Skip daemons: {self.skip_daemons}")

        # Start daemons
        if not self.skip_daemons:
            await self._start_daemons()

        # Subscribe to events
        self._subscribe_to_events()

        # Initialize state from current data
        await self._initialize_state()

        logger.info("[MasterLoop] Started successfully")

    async def stop(self) -> None:
        """Stop the master loop gracefully."""
        logger.info("[MasterLoop] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Stop daemons
        if not self.skip_daemons and self._daemon_manager is not None:
            await self.daemon_manager.shutdown()

        logger.info("[MasterLoop] Stopped")

    async def run(self) -> None:
        """Main automation loop."""
        await self.start()

        try:
            while self._running and not self._shutdown_event.is_set():
                loop_start = time.time()

                try:
                    # 1. Check cluster health, throttle if needed
                    health = await self._get_cluster_health()
                    if health.load_critical:
                        logger.warning("[MasterLoop] Cluster load critical, throttling")
                        await self._throttle_selfplay()

                    # 2. Check for training opportunities
                    now = time.time()
                    if now - self._last_training_check >= TRAINING_CHECK_INTERVAL:
                        await self._check_training_opportunities()
                        self._last_training_check = now

                    # 3. Check for allocation rebalancing
                    if now - self._last_allocation_check >= ALLOCATION_CHECK_INTERVAL:
                        await self._rebalance_allocations()
                        self._last_allocation_check = now

                    # 4. Check for pending evaluations
                    await self._check_pending_evaluations()

                    # 5. Log status
                    self._log_status(health)

                except Exception as e:
                    logger.error(f"[MasterLoop] Error in loop iteration: {e}")

                # Wait for next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed)

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

        finally:
            await self.stop()

    # =========================================================================
    # Daemon management
    # =========================================================================

    async def _start_daemons(self) -> None:
        """Start essential daemons."""
        from app.coordination.daemon_manager import DaemonType

        essential_daemons = [
            DaemonType.EVENT_ROUTER,
            DaemonType.FEEDBACK_LOOP,
            DaemonType.DATA_PIPELINE,
            DaemonType.AUTO_SYNC,
            DaemonType.MODEL_DISTRIBUTION,
        ]

        logger.info(f"[MasterLoop] Starting {len(essential_daemons)} essential daemons")

        for daemon_type in essential_daemons:
            try:
                if self.dry_run:
                    logger.info(f"[MasterLoop] [DRY RUN] Would start {daemon_type.value}")
                else:
                    await self.daemon_manager.start(daemon_type)
                    logger.debug(f"[MasterLoop] Started {daemon_type.value}")
            except Exception as e:
                logger.warning(f"[MasterLoop] Failed to start {daemon_type.value}: {e}")

        # Start FeedbackLoopController explicitly (December 2025 - Phase 2A.1)
        # The daemon manager starts FEEDBACK_LOOP daemon, but we also need to
        # ensure the controller itself is started with proper event subscriptions
        try:
            if not self.dry_run:
                await self.feedback_controller.start()
                logger.info("[MasterLoop] Started FeedbackLoopController")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to start FeedbackLoopController: {e}")

    # =========================================================================
    # Event handling
    # =========================================================================

    def _subscribe_to_events(self) -> None:
        """Subscribe to pipeline events."""
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.subscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)

            logger.info("[MasterLoop] Subscribed to pipeline events")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to subscribe to events: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion."""
        try:
            config_key = event.payload.get("config_key", "")
            games_added = event.payload.get("games_added", 0)

            if config_key in self._states:
                state = self._states[config_key]
                state.games_since_last_export += games_added
                logger.debug(f"[MasterLoop] {config_key}: +{games_added} games, total pending: {state.games_since_last_export}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling selfplay event: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion."""
        try:
            config_key = event.payload.get("config_key", "")
            policy_accuracy = event.payload.get("policy_accuracy", 0.0)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_training_time = time.time()
                state.last_policy_accuracy = policy_accuracy
                state.pending_evaluation = True  # Queue for evaluation
                state.training_node = None  # Release training node

                logger.info(f"[MasterLoop] {config_key}: Training complete, policy accuracy: {policy_accuracy:.2%}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling training event: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation completion."""
        try:
            config_key = event.payload.get("config_key", "")
            win_rate = event.payload.get("win_rate", 0.0)
            passed = event.payload.get("passed", False)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_evaluation_time = time.time()
                state.last_evaluation_win_rate = win_rate
                state.pending_evaluation = False

                logger.info(f"[MasterLoop] {config_key}: Evaluation complete, win rate: {win_rate:.2%}, passed: {passed}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling evaluation event: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion."""
        try:
            config_key = event.payload.get("config_key", "")
            success = event.payload.get("success", False)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_promotion_success = success

                # Apply feedback based on outcome
                if success:
                    # Accelerate training for this config
                    state.training_intensity = "accelerated"
                    state.exploration_boost = 1.0  # Reset exploration
                    logger.info(f"[MasterLoop] {config_key}: Promotion succeeded, accelerating training")
                else:
                    # Boost exploration on failure
                    state.exploration_boost = min(2.0, state.exploration_boost * 1.3)
                    logger.info(f"[MasterLoop] {config_key}: Promotion failed, exploration boost: {state.exploration_boost:.2f}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling promotion event: {e}")

    # =========================================================================
    # State initialization
    # =========================================================================

    async def _initialize_state(self) -> None:
        """Initialize state from current data."""
        logger.info("[MasterLoop] Initializing state from current data...")

        for config_key in self.active_configs:
            try:
                # Parse config key
                parts = config_key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))

                # Check for existing NPZ files
                npz_path = Path(f"data/training/{board_type}_{num_players}p.npz")
                if npz_path.exists():
                    mtime = npz_path.stat().st_mtime
                    self._states[config_key].last_export_time = mtime
                    logger.debug(f"[MasterLoop] {config_key}: Found NPZ from {datetime.fromtimestamp(mtime)}")

            except Exception as e:
                logger.debug(f"[MasterLoop] Error initializing {config_key}: {e}")

    # =========================================================================
    # Health monitoring
    # =========================================================================

    async def _get_cluster_health(self) -> ClusterHealth:
        """Get aggregated cluster health."""
        health = ClusterHealth()

        try:
            # Query cluster status
            status = self.cluster_monitor.get_cluster_status()

            health.total_nodes = status.total_nodes
            health.healthy_nodes = status.active_nodes
            health.avg_disk_usage = status.avg_disk_usage

            # Count training and selfplay nodes
            for node_id, node_status in status.nodes.items():
                if node_status.training_active:
                    health.training_nodes += 1
                # Assume non-training GPU nodes are doing selfplay
                elif node_status.gpu_utilization_percent > 10:
                    health.selfplay_nodes += 1

                health.avg_gpu_utilization += node_status.gpu_utilization_percent

            if health.total_nodes > 0:
                health.avg_gpu_utilization /= health.total_nodes

            # Check for critical load
            health.load_critical = (
                health.avg_disk_usage > 90
                or health.healthy_nodes < health.total_nodes * 0.5
            )

        except Exception as e:
            health.errors.append(str(e))
            logger.debug(f"[MasterLoop] Error getting cluster health: {e}")

        return health

    async def _throttle_selfplay(self) -> None:
        """Throttle selfplay when cluster is overloaded."""
        logger.info("[MasterLoop] Throttling selfplay due to cluster load")

        if self.dry_run:
            logger.info("[MasterLoop] [DRY RUN] Would pause selfplay jobs")
            return

        # Emit throttle signal via event bus
        try:
            from app.coordination.event_router import emit_event, DataEventType

            await emit_event(
                DataEventType.CLUSTER_HEALTH,
                {"action": "throttle_selfplay", "reason": "load_critical"}
            )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error emitting throttle event: {e}")

    # =========================================================================
    # Training coordination
    # =========================================================================

    async def _check_training_opportunities(self) -> None:
        """Check if any configs are ready for training."""
        logger.debug("[MasterLoop] Checking training opportunities...")

        for config_key, state in self._states.items():
            if state.training_node is not None:
                # Already training
                continue

            # Check readiness
            ready, reason = await self._check_training_readiness(config_key)

            if ready:
                logger.info(f"[MasterLoop] {config_key}: Ready for training")
                await self._trigger_training(config_key)
            elif state.games_since_last_export > 0:
                logger.debug(f"[MasterLoop] {config_key}: Not ready - {reason}")

    async def _check_training_readiness(self, config_key: str) -> tuple[bool, str]:
        """Check if a config is ready for training."""
        state = self._states[config_key]

        # Check minimum games
        if state.games_since_last_export < MIN_GAMES_FOR_EXPORT:
            return False, f"Insufficient games: {state.games_since_last_export} < {MIN_GAMES_FOR_EXPORT}"

        # Check quality score
        if state.last_quality_score < 0.5:
            return False, f"Low quality: {state.last_quality_score:.2f}"

        # Check if circuit breaker is tripped
        try:
            if self._pipeline_orchestrator is not None:
                breaker = self.pipeline_orchestrator._circuit_breaker
                if breaker and not breaker.can_execute():
                    return False, "Circuit breaker open"
        except Exception:
            pass

        return True, "Ready"

    async def _trigger_training(self, config_key: str) -> None:
        """Trigger training for a config."""
        if self.dry_run:
            logger.info(f"[MasterLoop] [DRY RUN] Would trigger training for {config_key}")
            return

        try:
            # Parse config
            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Emit training trigger event
            from app.coordination.event_router import emit_event, DataEventType

            await emit_event(
                DataEventType.TRIGGER_TRAINING,
                {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "intensity": self._states[config_key].training_intensity,
                }
            )

            logger.info(f"[MasterLoop] Triggered training for {config_key}")

            # Reset games counter
            self._states[config_key].games_since_last_export = 0
            self._states[config_key].last_export_time = time.time()

        except Exception as e:
            logger.error(f"[MasterLoop] Failed to trigger training for {config_key}: {e}")

    # =========================================================================
    # Allocation rebalancing
    # =========================================================================

    async def _rebalance_allocations(self) -> None:
        """Rebalance selfplay allocations based on priorities.

        December 2025 - Phase 2A.2: Now uses SelfplayScheduler for priority-based allocation.
        """
        logger.debug("[MasterLoop] Rebalancing allocations...")

        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()

            if self.dry_run:
                # Just show priorities in dry run mode
                priorities = await scheduler.get_priority_configs(top_n=6)
                for config_key, priority in priorities:
                    logger.info(f"[MasterLoop] [DRY RUN] Priority: {config_key} = {priority:.2f}")
                return

            # Get allocations from scheduler
            allocation = await scheduler.allocate_selfplay_batch(
                games_per_config=500,
                max_configs=6,
            )

            if allocation:
                # Emit job allocation events
                for config_key, nodes in allocation.items():
                    total_games = sum(nodes.values())
                    logger.info(
                        f"[MasterLoop] Allocated {config_key}: {total_games} games "
                        f"across {len(nodes)} nodes"
                    )

                    # Emit event for each node allocation
                    for node_id, num_games in nodes.items():
                        await self._emit_selfplay_job(node_id, config_key, num_games)

                logger.info(f"[MasterLoop] Rebalanced {len(allocation)} configs")
            else:
                logger.debug("[MasterLoop] No allocations needed")

        except ImportError:
            # Fallback to simple priority logging if scheduler not available
            priorities = self._get_priority_configs()
            for config_key, priority in priorities[:3]:
                logger.debug(f"[MasterLoop] Priority: {config_key} = {priority:.2f}")
        except Exception as e:
            logger.warning(f"[MasterLoop] Error in rebalancing: {e}")

    def _get_priority_configs(self) -> list[tuple[str, float]]:
        """Rank configs by priority for selfplay allocation."""
        priorities = []

        for config_key, state in self._states.items():
            # Priority factors:
            # - Data staleness (higher staleness = higher priority)
            # - Improvement potential (configs that are improving get more resources)
            # - Exploration boost (failing configs get boosted exploration)

            staleness_factor = min(state.data_staleness_hours / MAX_DATA_STALENESS_HOURS, 2.0)
            exploration_factor = state.exploration_boost

            # Bonus for accelerated training
            intensity_factor = 1.5 if state.training_intensity == "accelerated" else 1.0

            priority = staleness_factor * exploration_factor * intensity_factor
            priorities.append((config_key, priority))

        return sorted(priorities, key=lambda x: -x[1])

    async def _emit_selfplay_job(
        self,
        node_id: str,
        config_key: str,
        num_games: int,
    ) -> None:
        """Emit a selfplay job allocation event.

        December 2025 - Phase 2A.2: Emits events for work queue integration.
        """
        try:
            from app.coordination.event_router import emit_event, DataEventType

            # Parse config key
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                return

            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            await emit_event(
                DataEventType.REQUEST_SELFPLAY_PRIORITY,
                {
                    "node_id": node_id,
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": num_games,
                    "source": "master_loop",
                }
            )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error emitting selfplay job: {e}")

    # =========================================================================
    # Evaluation handling
    # =========================================================================

    async def _check_pending_evaluations(self) -> None:
        """Check for configs pending evaluation."""
        for config_key, state in self._states.items():
            if state.pending_evaluation:
                # Check if evaluation is already running
                # (would need integration with gauntlet runner)
                logger.debug(f"[MasterLoop] {config_key}: Pending evaluation")

    # =========================================================================
    # Status reporting
    # =========================================================================

    def _log_status(self, health: ClusterHealth) -> None:
        """Log current status."""
        # Only log periodically
        if int(time.time()) % 300 != 0:  # Every 5 minutes
            return

        logger.info(
            f"[MasterLoop] Status: "
            f"nodes={health.healthy_nodes}/{health.total_nodes}, "
            f"training={health.training_nodes}, "
            f"selfplay={health.selfplay_nodes}, "
            f"gpu_util={health.avg_gpu_utilization:.0f}%"
        )

        # Log config states
        for config_key, state in self._states.items():
            if state.games_since_last_export > 0 or state.pending_evaluation:
                logger.info(
                    f"[MasterLoop]   {config_key}: "
                    f"pending_games={state.games_since_last_export}, "
                    f"intensity={state.training_intensity}"
                )

    def get_status(self) -> dict[str, Any]:
        """Get current status as dict."""
        return {
            "running": self._running,
            "active_configs": self.active_configs,
            "dry_run": self.dry_run,
            "config_states": {
                cfg: {
                    "games_pending": state.games_since_last_export,
                    "data_staleness_hours": state.data_staleness_hours,
                    "training_intensity": state.training_intensity,
                    "exploration_boost": state.exploration_boost,
                    "pending_evaluation": state.pending_evaluation,
                }
                for cfg, state in self._states.items()
            },
        }


# =============================================================================
# Watch mode
# =============================================================================

async def watch_mode(controller: MasterLoopController, interval: int = 10) -> None:
    """Display live status updates."""
    import shutil

    term_width = shutil.get_terminal_size().columns

    while True:
        # Clear screen
        print("\033[2J\033[H", end="")

        # Header
        print("=" * term_width)
        print(f"RingRift Master Loop - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * term_width)

        status = controller.get_status()

        print(f"\nRunning: {status['running']}")
        print(f"Dry Run: {status['dry_run']}")
        print(f"Active Configs: {len(status['active_configs'])}")

        print("\nConfig States:")
        print("-" * term_width)
        print(f"{'Config':<15} {'Games':<8} {'Staleness':<12} {'Intensity':<12} {'Eval':<6}")
        print("-" * term_width)

        for cfg, state in status['config_states'].items():
            staleness = f"{state['data_staleness_hours']:.1f}h" if state['data_staleness_hours'] < float('inf') else "N/A"
            eval_status = "Y" if state['pending_evaluation'] else "-"
            print(
                f"{cfg:<15} "
                f"{state['games_pending']:<8} "
                f"{staleness:<12} "
                f"{state['training_intensity']:<12} "
                f"{eval_status:<6}"
            )

        print("\n(Press Ctrl+C to exit)")

        await asyncio.sleep(interval)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master Loop Controller - Unified RingRift AI automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of configs to manage (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--skip-daemons",
        action="store_true",
        help="Don't start/stop daemons (for testing)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - display live status",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Watch mode update interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Parse configs
    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",")]

    # Create controller
    controller = MasterLoopController(
        configs=configs,
        dry_run=args.dry_run,
        skip_daemons=args.skip_daemons,
    )

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(controller.stop()))

    if args.status:
        # Just show status
        await controller._initialize_state()
        status = controller.get_status()
        import json
        print(json.dumps(status, indent=2, default=str))
        return

    if args.watch:
        # Watch mode
        await controller.start()
        await watch_mode(controller, interval=args.interval)
    else:
        # Run main loop
        await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
