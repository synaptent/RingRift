"""Continuous Training Loop Daemon.

Automates the full training cycle: selfplay -> sync -> export -> train -> evaluate -> promote.
Runs as a daemon under DaemonManager for lifecycle management.

IMPORTANT: This is a LIGHTWEIGHT alternative to unified_ai_loop.py. If the full
unified_ai_loop is running, this daemon will automatically defer to it.

Loop Ecosystem:
- unified_ai_loop.py (8451 lines): Full system with streaming data, shadow tournaments,
  adaptive curriculum, PBT, NAS. Use for production deployments.
- continuous_loop.py (this file): Lightweight daemon that just orchestrates
  selfplay -> pipeline auto-trigger. Use for simpler deployments.

The two loops use the same coordination infrastructure (event_router, pipeline_orchestrator)
but unified_ai_loop has many more features. When both are started, this loop defers.

Usage:
    # As a daemon (via launch_daemons.py)
    python scripts/launch_daemons.py --continuous

    # Direct execution
    python -m app.coordination.continuous_loop --config hex8:2 --config square8:4

    # Single iteration for testing
    python -m app.coordination.continuous_loop --config hex8:2 --games 10 --max-iterations 1

    # Force run even if unified_ai_loop is active (not recommended)
    python -m app.coordination.continuous_loop --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """State of the continuous training loop."""
    IDLE = "idle"
    RUNNING_SELFPLAY = "running_selfplay"
    WAITING_PIPELINE = "waiting_pipeline"
    COOLING_DOWN = "cooling_down"
    DEFERRED = "deferred"  # Deferring to unified_ai_loop
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LoopConfig:
    """Configuration for continuous training loop."""

    # Board configurations to train: [(board_type, num_players), ...]
    configs: list[tuple[str, int]] = field(default_factory=lambda: [
        ("hex8", 2),
        ("square8", 2),
    ])

    # Selfplay settings
    selfplay_games_per_iteration: int = 1000
    selfplay_engine: str = "gumbel-mcts"

    # Pipeline settings
    parallel_configs: bool = False  # Run configs in parallel (requires more resources)
    pipeline_timeout_seconds: int = 7200  # 2 hours

    # Loop control
    max_iterations: int = 0  # 0 = infinite
    iteration_cooldown_seconds: float = 60.0

    # Failure handling
    max_consecutive_failures: int = 3
    failure_backoff_seconds: float = 300.0  # 5 minutes

    # Deferral settings
    force: bool = False  # Force run even if unified_ai_loop is active
    defer_check_interval_seconds: float = 60.0  # How often to check if we can take over


@dataclass
class LoopStats:
    """Statistics for the continuous training loop."""
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    consecutive_failures: int = 0
    last_iteration_time: float = 0.0
    last_config_trained: str = ""
    current_state: LoopState = LoopState.IDLE
    current_config: str = ""
    start_time: float = field(default_factory=time.time)
    last_error: str | None = None  # December 2025: Added for health_check() reporting


class ContinuousTrainingLoop:
    """Daemon that runs continuous training iterations.

    Each iteration:
    1. Runs selfplay to generate training data
    2. Waits for pipeline auto-triggers: sync -> export -> train -> evaluate -> promote
    3. Cools down before next iteration

    Handles failures with exponential backoff and circuit breaker pattern.

    NOTE: If unified_ai_loop is running, this loop will defer to it and wait
    in DEFERRED state until unified_ai_loop stops.
    """

    def __init__(self, config: LoopConfig | None = None):
        self.config = config or LoopConfig()
        self.stats = LoopStats()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._orchestrator: DataPipelineOrchestrator | None = None

    def _is_unified_loop_running(self) -> bool:
        """Check if the unified_ai_loop is running.

        Returns True if we should defer to unified_ai_loop.
        """
        if self.config.force:
            return False  # Force mode - don't defer

        try:
            from app.coordination.helpers import is_unified_loop_running
            return is_unified_loop_running()
        except ImportError:
            return False

    async def start(self) -> None:
        """Start the continuous training loop."""
        # December 2025: Coordinator-only mode check
        # This loop runs CPU/GPU intensive tasks - should NEVER run on coordinator nodes
        from app.config.env import env
        if env.is_coordinator:
            logger.info(
                f"[ContinuousLoop] Skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator})"
            )
            return

        if self._running:
            logger.warning("Continuous loop already running")
            return

        # December 2025 fix: Capture deferral state BEFORE creating new stats
        # (previously DEFERRED was set then immediately overwritten by LoopStats())
        should_defer = self._is_unified_loop_running()
        if should_defer:
            logger.info("unified_ai_loop is running - deferring to it")
            logger.info("Use --force to override (not recommended)")

        logger.info("Starting continuous training loop")
        logger.info(f"  Configs: {self.config.configs}")
        logger.info(f"  Games per iteration: {self.config.selfplay_games_per_iteration}")
        logger.info(f"  Engine: {self.config.selfplay_engine}")
        logger.info(f"  Max iterations: {self.config.max_iterations or 'infinite'}")
        if self.config.force:
            logger.warning("  Force mode: ignoring unified_ai_loop")

        self._running = True
        self._shutdown_event.clear()
        self.stats = LoopStats()

        # Restore deferral state after stats creation
        if should_defer:
            self.stats.current_state = LoopState.DEFERRED

        # Wire up pipeline events with auto-trigger
        self._setup_pipeline()

        # Start the main loop
        self._task = safe_create_task(
            self._run_loop(),
            name="continuous_training_loop",
        )

    async def stop(self) -> None:
        """Stop the continuous training loop gracefully."""
        if not self._running:
            return

        logger.info("Stopping continuous training loop...")
        self._running = False
        self._shutdown_event.set()

        if self._task:
            try:
                # Give the loop time to finish current operation
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Loop did not stop gracefully, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self.stats.current_state = LoopState.STOPPED
        logger.info("Continuous training loop stopped")

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ContinuousTrainingLoop not running",
            )

        # Check for error state
        if self.stats.current_state == LoopState.ERROR:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"ContinuousTrainingLoop in ERROR state: {self.stats.last_error or 'unknown'}",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"ContinuousTrainingLoop: {self.stats.current_state.value}, iterations={self.stats.total_iterations}",
            details=self.get_status(),
        )

    def _setup_pipeline(self) -> None:
        """Set up pipeline orchestrator with auto-trigger."""
        try:
            from app.coordination.data_pipeline_orchestrator import (
                get_pipeline_orchestrator,
                wire_pipeline_events,
            )

            # Wire up events with auto-trigger enabled
            wire_pipeline_events(auto_trigger=True)
            self._orchestrator = get_pipeline_orchestrator()

            logger.info("Pipeline events wired with auto-trigger enabled")

        except ImportError as e:
            logger.warning(f"Could not import pipeline orchestrator: {e}")
            logger.warning("Pipeline auto-trigger will not be available")

    async def _run_loop(self) -> None:
        """Main loop that runs training iterations."""
        iteration = 0
        config_index = 0

        while self._running:
            try:
                # Check if unified_ai_loop is running - defer if so
                if self._is_unified_loop_running():
                    if self.stats.current_state != LoopState.DEFERRED:
                        logger.info("unified_ai_loop detected - entering DEFERRED state")
                        self.stats.current_state = LoopState.DEFERRED
                    await self._wait_or_shutdown(self.config.defer_check_interval_seconds)
                    continue

                # Exit deferred state if we were waiting
                if self.stats.current_state == LoopState.DEFERRED:
                    logger.info("unified_ai_loop stopped - resuming continuous loop")
                    self.stats.current_state = LoopState.IDLE

                # Check iteration limit
                if self.config.max_iterations > 0 and iteration >= self.config.max_iterations:
                    logger.info(f"Reached max iterations ({self.config.max_iterations})")
                    break

                # Get next config to train
                config = self.config.configs[config_index]
                board_type, num_players = config
                config_key = f"{board_type}_{num_players}p"

                iteration += 1
                self.stats.total_iterations = iteration
                self.stats.current_config = config_key

                logger.info(f"=== Iteration {iteration}: {config_key} ===")

                # Run single iteration
                success = await self._run_single_config(board_type, num_players, iteration)

                if success:
                    self.stats.successful_iterations += 1
                    self.stats.consecutive_failures = 0
                    self.stats.last_config_trained = config_key
                else:
                    self.stats.failed_iterations += 1
                    self.stats.consecutive_failures += 1

                self.stats.last_iteration_time = time.time()

                # Check for too many consecutive failures
                if self.stats.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.error(
                        f"Circuit breaker: {self.stats.consecutive_failures} consecutive failures, "
                        f"backing off for {self.config.failure_backoff_seconds}s"
                    )
                    self.stats.current_state = LoopState.ERROR
                    await self._wait_or_shutdown(self.config.failure_backoff_seconds)
                    self.stats.consecutive_failures = 0  # Reset after backoff

                # Move to next config
                config_index = (config_index + 1) % len(self.config.configs)

                # Cooldown between iterations
                if self._running and self.config.iteration_cooldown_seconds > 0:
                    self.stats.current_state = LoopState.COOLING_DOWN
                    logger.info(f"Cooling down for {self.config.iteration_cooldown_seconds}s...")
                    await self._wait_or_shutdown(self.config.iteration_cooldown_seconds)

            except asyncio.CancelledError:
                logger.info("Loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Unexpected error in training loop: {e}")
                self.stats.last_error = str(e)  # December 2025: Record for health_check()
                self.stats.failed_iterations += 1
                self.stats.consecutive_failures += 1
                await self._wait_or_shutdown(60.0)  # Brief wait before retry

        self.stats.current_state = LoopState.STOPPED
        logger.info(f"Training loop completed: {self.stats.successful_iterations}/{self.stats.total_iterations} successful")

    async def _run_single_config(
        self,
        board_type: str,
        num_players: int,
        iteration: int,
    ) -> bool:
        """Run a single training iteration for one config.

        Returns True if iteration completed successfully.
        """
        config_key = f"{board_type}_{num_players}p"

        try:
            # Step 1: Run selfplay
            self.stats.current_state = LoopState.RUNNING_SELFPLAY
            logger.info(f"[{config_key}] Starting selfplay: {self.config.selfplay_games_per_iteration} games")

            selfplay_success = await self._run_selfplay(
                board_type=board_type,
                num_players=num_players,
                num_games=self.config.selfplay_games_per_iteration,
                engine=self.config.selfplay_engine,
            )

            if not selfplay_success:
                logger.error(f"[{config_key}] Selfplay failed")
                return False

            logger.info(f"[{config_key}] Selfplay complete, waiting for pipeline...")

            # Step 2: Wait for pipeline to complete
            self.stats.current_state = LoopState.WAITING_PIPELINE
            pipeline_success = await self._wait_for_pipeline(
                config_key=config_key,
                timeout_seconds=self.config.pipeline_timeout_seconds,
            )

            if not pipeline_success:
                logger.warning(f"[{config_key}] Pipeline did not complete successfully")
                # This is a soft failure - selfplay data is still valuable
                return False

            logger.info(f"[{config_key}] Iteration {iteration} complete!")
            return True

        except Exception as e:
            logger.exception(f"[{config_key}] Error in iteration: {e}")
            return False

    async def _run_selfplay(
        self,
        board_type: str,
        num_players: int,
        num_games: int,
        engine: str,
    ) -> bool:
        """Run selfplay and emit completion event.

        Returns True if selfplay completed successfully.
        """
        try:
            from app.training.selfplay_runner import run_selfplay

            # Run selfplay in executor to avoid blocking event loop
            # Dec 2025: Use get_running_loop() for async context
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: run_selfplay(
                    board_type=board_type,
                    num_players=num_players,
                    num_games=num_games,
                    engine=engine,
                ),
            )

            if stats and stats.games_completed > 0:
                logger.info(f"Selfplay completed: {stats.games_completed} games, {stats.total_samples} samples")
                return True
            else:
                logger.warning("Selfplay produced no games")
                return False

        except ImportError as e:
            logger.error(f"Could not import selfplay_runner: {e}")
            return False
        except Exception as e:
            logger.exception(f"Selfplay error: {e}")
            return False

    async def _wait_for_pipeline(
        self,
        config_key: str,
        timeout_seconds: int,
    ) -> bool:
        """Wait for the pipeline to complete all stages.

        Returns True if pipeline completed successfully.
        """
        if not self._orchestrator:
            logger.warning("No orchestrator available, skipping pipeline wait")
            return True  # Consider it success if no orchestrator

        try:
            from app.coordination.data_pipeline_orchestrator import PipelineStage
        except ImportError:
            logger.warning("Could not import PipelineStage")
            return True

        start_time = time.time()
        last_stage = None
        check_interval = 10.0  # Check every 10 seconds

        while time.time() - start_time < timeout_seconds:
            if not self._running:
                logger.info("Loop stopped, aborting pipeline wait")
                return False

            try:
                state = self._orchestrator.get_config_state(config_key)

                if state is None:
                    # No state yet, pipeline might not have started
                    await asyncio.sleep(check_interval)
                    continue

                current_stage = state.current_stage

                # Log stage transitions
                if current_stage != last_stage:
                    stage_name = current_stage.value if current_stage else "None"
                    logger.info(f"[{config_key}] Pipeline stage: {stage_name}")
                    last_stage = current_stage

                # Check for completion
                if current_stage == PipelineStage.IDLE:
                    if state.last_training_completed:
                        logger.info(f"[{config_key}] Pipeline completed successfully")
                        return True

                # Check for circuit breaker
                if self._orchestrator.circuit_breaker:
                    if not self._orchestrator.circuit_breaker.can_execute():
                        logger.error(f"[{config_key}] Pipeline circuit breaker is OPEN")
                        return False

            except Exception as e:
                logger.warning(f"Error checking pipeline state: {e}")

            await self._wait_or_shutdown(check_interval)

        logger.warning(f"[{config_key}] Pipeline timed out after {timeout_seconds}s")
        return False

    async def _wait_or_shutdown(self, seconds: float) -> bool:
        """Wait for specified seconds or until shutdown is requested.

        Returns True if waited full duration, False if shutdown requested.
        """
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=seconds,
            )
            # Shutdown was requested
            return False
        except asyncio.TimeoutError:
            # Normal timeout - waited full duration
            return True

    def get_status(self) -> dict:
        """Get current status of the training loop."""
        uptime = time.time() - self.stats.start_time

        # Check if we're deferring
        unified_loop_active = self._is_unified_loop_running()

        return {
            "running": self._running,
            "state": self.stats.current_state.value,
            "deferred_to_unified_loop": unified_loop_active and not self.config.force,
            "force_mode": self.config.force,
            "current_config": self.stats.current_config,
            "total_iterations": self.stats.total_iterations,
            "successful_iterations": self.stats.successful_iterations,
            "failed_iterations": self.stats.failed_iterations,
            "consecutive_failures": self.stats.consecutive_failures,
            "last_config_trained": self.stats.last_config_trained,
            "uptime_seconds": uptime,
            "configs": [f"{bt}_{np}p" for bt, np in self.config.configs],
            "games_per_iteration": self.config.selfplay_games_per_iteration,
            "engine": self.config.selfplay_engine,
        }


# Global instance for daemon manager
_loop_instance: ContinuousTrainingLoop | None = None


def get_continuous_loop() -> ContinuousTrainingLoop:
    """Get or create the global continuous training loop instance."""
    global _loop_instance
    if _loop_instance is None:
        _loop_instance = ContinuousTrainingLoop()
    return _loop_instance


async def start_continuous_loop(config: LoopConfig | None = None) -> ContinuousTrainingLoop:
    """Start the continuous training loop with given config."""
    global _loop_instance
    _loop_instance = ContinuousTrainingLoop(config)
    await _loop_instance.start()
    return _loop_instance


async def stop_continuous_loop() -> None:
    """Stop the global continuous training loop."""
    global _loop_instance
    if _loop_instance:
        await _loop_instance.stop()
        _loop_instance = None


def parse_config_arg(config_str: str) -> tuple[str, int]:
    """Parse a config argument like 'hex8:2' or 'square8_4p'."""
    # Handle 'board:players' format
    if ":" in config_str:
        parts = config_str.split(":")
        return (parts[0], int(parts[1]))

    # Handle 'board_Xp' format
    if "_" in config_str and config_str.endswith("p"):
        parts = config_str.rsplit("_", 1)
        return (parts[0], int(parts[1][:-1]))

    raise ValueError(f"Invalid config format: {config_str}. Use 'board:players' or 'board_Xp'")


def main():
    """CLI entry point for continuous training loop."""
    parser = argparse.ArgumentParser(
        description="Continuous Training Loop - automates selfplay -> train -> evaluate -> promote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        action="append",
        default=[],
        help="Board config to train (e.g., 'hex8:2', 'square8_4p'). Can be repeated.",
    )
    parser.add_argument(
        "--games", "-n",
        type=int,
        default=1000,
        help="Number of selfplay games per iteration (default: 1000)",
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="gumbel-mcts",
        choices=["heuristic", "gumbel-mcts", "mcts", "nnue-guided"],
        help="Selfplay engine (default: gumbel-mcts)",
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=0,
        help="Max iterations (0 = infinite, default: 0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=60.0,
        help="Seconds between iterations (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Pipeline timeout in seconds (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force run even if unified_ai_loop is active (not recommended)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse configs
    configs = []
    if args.config:
        for config_str in args.config:
            try:
                configs.append(parse_config_arg(config_str))
            except ValueError as e:
                parser.error(str(e))
    else:
        # Default configs
        configs = [("hex8", 2), ("square8", 2)]

    # Create config
    loop_config = LoopConfig(
        configs=configs,
        selfplay_games_per_iteration=args.games,
        selfplay_engine=args.engine,
        max_iterations=args.max_iterations,
        iteration_cooldown_seconds=args.cooldown,
        pipeline_timeout_seconds=args.timeout,
        force=args.force,
    )

    # Handle signals with proper async shutdown coordination
    loop = ContinuousTrainingLoop(loop_config)
    shutdown_requested = False
    shutdown_complete = asyncio.Event()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second signal - force exit, but still try to log
            logger.warning("Force exit requested (second signal)")
            # Don't use sys.exit(1) - let the event loop handle cleanup
            # Instead, set the event and let the main coroutine exit
            try:
                running_loop = asyncio.get_running_loop()
                running_loop.call_soon_threadsafe(shutdown_complete.set)
            except RuntimeError:
                # No running loop - exit directly as last resort
                import os
                os._exit(1)
            return
        shutdown_requested = True
        logger.info("Shutdown requested...")
        # Dec 2025: Signal handlers need special handling for async
        # Use call_soon_threadsafe for thread-safe scheduling
        try:
            running_loop = asyncio.get_running_loop()
            # Schedule graceful stop - this will set _running = False
            # and trigger the shutdown event
            running_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_graceful_shutdown())
            )
        except RuntimeError:
            # No running loop in signal handler - the main loop isn't running yet
            pass

    async def _graceful_shutdown():
        """Async helper for graceful shutdown coordination."""
        try:
            await loop.stop()
        finally:
            shutdown_complete.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the loop
    async def run():
        await loop.start()
        # Wait for loop to complete or shutdown signal
        try:
            while loop._running and not shutdown_complete.is_set():
                # Use wait_for to allow shutdown_complete to interrupt
                try:
                    await asyncio.wait_for(
                        shutdown_complete.wait(),
                        timeout=1.0,
                    )
                    break  # shutdown_complete was set
                except asyncio.TimeoutError:
                    continue  # Still running, check again
        except asyncio.CancelledError:
            # Ensure cleanup on cancellation
            if loop._running:
                await loop.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        # asyncio.run handles cleanup, but ensure our loop is stopped
        pass

    # Print final stats
    status = loop.get_status()
    print("\n" + "=" * 60)
    print("Continuous Training Loop - Final Status")
    print("=" * 60)
    print(f"Total iterations: {status['total_iterations']}")
    print(f"Successful: {status['successful_iterations']}")
    print(f"Failed: {status['failed_iterations']}")
    print(f"Uptime: {status['uptime_seconds']:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
