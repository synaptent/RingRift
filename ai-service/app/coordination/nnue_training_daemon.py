"""NNUE Automatic Training Daemon for RingRift.

Automatically triggers NNUE training when sufficient new games have been
accumulated for a board configuration. This enables continuous NNUE model
improvement without manual intervention.

Key Features:
    - Monitors game counts per config
    - Triggers training when threshold reached (default: 10K games)
    - Limits concurrent trainings to prevent resource exhaustion
    - Emits events for training lifecycle tracking
    - Integrates with EloService for post-training evaluation

Usage:
    from app.coordination.nnue_training_daemon import (
        get_nnue_training_daemon,
        NNUETrainingConfig,
    )

    # Get singleton daemon
    daemon = get_nnue_training_daemon()
    await daemon.start()

    # Or with custom config
    config = NNUETrainingConfig(
        default_game_threshold=5000,
        check_interval_seconds=1800,
    )
    daemon = NNUETrainingDaemon(config=config)
    await daemon.start()

December 2025 - Phase 7 of unified NN/NNUE multi-harness plan.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_handler_utils import extract_config_key, extract_model_path
from app.coordination.event_utils import parse_config_key
from pathlib import Path
from typing import Any, Callable, ClassVar

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class NNUETrainingConfig:
    """Configuration for NNUE training daemon.

    Attributes:
        game_thresholds: Per-config game count thresholds for triggering training
        default_game_threshold: Default threshold if config not in thresholds
        check_interval_seconds: How often to check game counts
        max_concurrent_trainings: Maximum number of concurrent trainings
        min_time_between_trainings: Minimum seconds between trainings for same config
        training_timeout_seconds: Timeout for training subprocess
    """

    # Per-config thresholds (larger boards need fewer games due to longer games)
    game_thresholds: dict[str, int] = field(default_factory=lambda: {
        "hex8_2p": 5000,
        "hex8_3p": 5000,
        "hex8_4p": 10000,
        "square8_2p": 5000,
        "square8_3p": 5000,
        "square8_4p": 10000,
        "square19_2p": 2000,
        "square19_3p": 3000,
        "square19_4p": 5000,
        "hexagonal_2p": 2000,
        "hexagonal_3p": 3000,
        "hexagonal_4p": 5000,
    })

    default_game_threshold: int = 5000
    check_interval_seconds: float = 3600.0  # Check every hour
    max_concurrent_trainings: int = 2
    min_time_between_trainings: float = 3600.0  # 1 hour minimum gap
    training_timeout_seconds: float = 7200.0  # 2 hour timeout

    def get_threshold(self, config_key: str) -> int:
        """Get game threshold for a config key."""
        return self.game_thresholds.get(config_key, self.default_game_threshold)


@dataclass
class NNUETrainingState:
    """Persistent state for NNUE training tracking.

    Tracks when each config was last trained and game counts at that time.
    """

    # Last training time per config
    last_training_time: dict[str, float] = field(default_factory=dict)

    # Game count at last training per config
    last_training_game_count: dict[str, int] = field(default_factory=dict)

    # Currently active trainings (config -> start_time)
    active_trainings: dict[str, float] = field(default_factory=dict)

    # Training history (for analytics)
    training_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "last_training_time": self.last_training_time,
            "last_training_game_count": self.last_training_game_count,
            "active_trainings": self.active_trainings,
            "training_history": self.training_history[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NNUETrainingState:
        """Deserialize from dictionary."""
        return cls(
            last_training_time=data.get("last_training_time", {}),
            last_training_game_count=data.get("last_training_game_count", {}),
            active_trainings=data.get("active_trainings", {}),
            training_history=data.get("training_history", []),
        )


class NNUETrainingDaemon(HandlerBase):
    """Daemon that automatically triggers NNUE training.

    Monitors game counts for each board configuration and triggers NNUE
    training when the game count since last training exceeds the threshold.

    This enables continuous NNUE model improvement without manual intervention.
    """

    _instance: ClassVar[NNUETrainingDaemon | None] = None

    def __init__(self, config: NNUETrainingConfig | None = None):
        """Initialize NNUE training daemon.

        Args:
            config: Training configuration (uses defaults if None)
        """
        # Store in _nnue_config to avoid HandlerBase._config overwrite
        self._nnue_config = config or NNUETrainingConfig()
        super().__init__(
            name="nnue_training",
            cycle_interval=self._nnue_config.check_interval_seconds,
        )

        self._state = NNUETrainingState()
        self._state_path = Path("data/nnue_training_state.json")
        self._load_state()

        # Current game counts (refreshed each cycle)
        self._current_game_counts: dict[str, int] = {}

        logger.info(
            f"NNUETrainingDaemon initialized (check_interval={self._nnue_config.check_interval_seconds}s, "
            f"max_concurrent={self._nnue_config.max_concurrent_trainings})"
        )

    @classmethod
    def get_instance(cls, config: NNUETrainingConfig | None = None) -> NNUETrainingDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            # Use provided config or create default
            effective_config = config or NNUETrainingConfig()
            cls._instance = cls(config=effective_config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if not self._state_path.exists():
            return

        try:
            import json
            with open(self._state_path) as f:
                data = json.load(f)
            self._state = NNUETrainingState.from_dict(data)
            logger.debug(f"NNUETrainingDaemon: Loaded state with {len(self._state.last_training_time)} configs")
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            logger.warning(f"NNUETrainingDaemon: Could not load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            import json
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except OSError as e:
            logger.error(f"NNUETrainingDaemon: Could not save state: {e}")

    def _get_event_subscriptions(self) -> dict[str, Callable[..., Any]]:
        """Define event subscriptions for this daemon."""
        return {
            "NEW_GAMES_AVAILABLE": self._on_new_games,
            "CONSOLIDATION_COMPLETE": self._on_consolidation_complete,
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            # Dec 2025: Phase 4 - Trigger NNUE training after NN training completes
            "TRAINING_COMPLETED": self._on_nn_training_completed,
        }

    async def _on_new_games(self, event: dict[str, Any]) -> None:
        """Handle new games available event."""
        config_key = extract_config_key(event)
        game_count = event.get("game_count", 0)

        if config_key and game_count:
            self._current_game_counts[config_key] = game_count
            logger.debug(f"NNUETrainingDaemon: Updated {config_key} count to {game_count}")

    async def _on_consolidation_complete(self, event: dict[str, Any]) -> None:
        """Handle consolidation complete event."""
        config_key = extract_config_key(event)
        if config_key:
            # Refresh game counts after consolidation
            await self._refresh_game_counts()

    async def _on_data_sync_completed(self, event: dict[str, Any]) -> None:
        """Handle data sync completed event."""
        # Data may have changed, refresh counts
        await self._refresh_game_counts()

    async def _on_nn_training_completed(self, event: dict[str, Any]) -> None:
        """Handle NN training completed event - trigger NNUE training if appropriate.

        This is the key integration point for Phase 4: when NN training completes,
        we check if NNUE training should be triggered for that config.

        Decision factors:
        1. Do we have enough games since last NNUE training?
        2. Did NN training succeed?
        3. Are we under the concurrent training limit?
        4. Has enough time passed since last NNUE training?

        Args:
            event: Training completed event with config_key, success, model_path
        """
        config_key = extract_config_key(event)
        success = event.get("success", False)
        model_path = extract_model_path(event)

        if not config_key:
            return

        if not success:
            logger.debug(
                f"NNUETrainingDaemon: NN training failed for {config_key}, skipping NNUE trigger"
            )
            return

        logger.info(
            f"NNUETrainingDaemon: NN training completed for {config_key}, "
            f"checking if NNUE training needed"
        )

        # Refresh game counts to get current state
        await self._refresh_game_counts()

        # Get current game count for this config
        current_count = self._current_game_counts.get(config_key, 0)

        # Check if NNUE training should be triggered
        if self._should_train(config_key, current_count):
            # Check concurrent limit
            available_slots = (
                self._nnue_config.max_concurrent_trainings
                - len(self._state.active_trainings)
            )
            if available_slots > 0:
                logger.info(
                    f"NNUETrainingDaemon: Triggering NNUE training for {config_key} "
                    f"after NN training (model: {model_path})"
                )
                await self._trigger_training(config_key, current_count)
            else:
                logger.info(
                    f"NNUETrainingDaemon: NNUE training needed for {config_key} "
                    f"but no slots available ({len(self._state.active_trainings)} active)"
                )
        else:
            logger.debug(
                f"NNUETrainingDaemon: NNUE training not needed yet for {config_key} "
                f"(games: {current_count})"
            )

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check for training opportunities."""
        # Clean up completed/timed-out trainings
        self._cleanup_active_trainings()

        # Refresh game counts
        await self._refresh_game_counts()

        # Check each config for training needs
        configs_to_train = []
        for config_key, current_count in self._current_game_counts.items():
            if self._should_train(config_key, current_count):
                configs_to_train.append((config_key, current_count))

        # Trigger trainings (respecting concurrency limit)
        available_slots = (
            self._nnue_config.max_concurrent_trainings
            - len(self._state.active_trainings)
        )

        for config_key, game_count in configs_to_train[:available_slots]:
            await self._trigger_training(config_key, game_count)

        # Save state
        self._save_state()

    async def _refresh_game_counts(self) -> None:
        """Refresh current game counts from GameDiscovery."""
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            databases = discovery.find_all_databases()

            # Aggregate counts per config
            counts: dict[str, int] = {}
            for db_info in databases:
                config_key = f"{db_info.board_type}_{db_info.num_players}p"
                counts[config_key] = counts.get(config_key, 0) + db_info.game_count

            self._current_game_counts = counts
            logger.debug(f"NNUETrainingDaemon: Refreshed counts for {len(counts)} configs")

        except (ImportError, OSError) as e:
            logger.warning(f"NNUETrainingDaemon: Could not refresh game counts: {e}")

    def _should_train(self, config_key: str, current_count: int) -> bool:
        """Check if NNUE training should be triggered for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            current_count: Current game count for this config

        Returns:
            True if training should be triggered
        """
        # Check if already training
        if config_key in self._state.active_trainings:
            return False

        # Check minimum time between trainings
        last_time = self._state.last_training_time.get(config_key, 0.0)
        if time.time() - last_time < self._nnue_config.min_time_between_trainings:
            return False

        # Check game count threshold
        last_count = self._state.last_training_game_count.get(config_key, 0)
        threshold = self._nnue_config.get_threshold(config_key)
        games_since_last = current_count - last_count

        if games_since_last >= threshold:
            logger.info(
                f"NNUETrainingDaemon: {config_key} has {games_since_last} new games "
                f"(threshold: {threshold}), training needed"
            )
            return True

        return False

    async def _trigger_training(self, config_key: str, game_count: int) -> None:
        """Trigger NNUE training for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            game_count: Current game count
        """
        logger.info(f"NNUETrainingDaemon: Triggering NNUE training for {config_key}")

        # Mark as active
        self._state.active_trainings[config_key] = time.time()

        # Emit training started event
        safe_emit_event(
            "NNUE_TRAINING_STARTED",
            {"config_key": config_key, "game_count": game_count, "timestamp": time.time()},
            context="nnue_training",
        )

        # Run training in background task
        asyncio.create_task(self._run_training(config_key, game_count))

    async def _run_training(self, config_key: str, game_count: int) -> None:
        """Dispatch NNUE training to cluster via P2P.

        December 29, 2025: Changed from local subprocess to cluster dispatch.
        Training runs on GPU nodes via P2P orchestrator, not locally.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            game_count: Current game count
        """
        import aiohttp

        start_time = time.time()
        success = False
        error_msg = ""
        job_id = ""

        try:
            # Parse config key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                raise ValueError(f"Invalid config_key format: {config_key}")

            board_type = parsed.board_type
            num_players = parsed.num_players

            # Dispatch to P2P leader via HTTP
            # The P2P orchestrator will find a suitable GPU node and start training
            p2p_port = int(Path.home().joinpath(".ringrift_p2p_port").read_text().strip()) \
                if Path.home().joinpath(".ringrift_p2p_port").exists() else 8770

            dispatch_url = f"http://localhost:{p2p_port}/training/nnue/start"
            payload = {
                "board_type": board_type,
                "num_players": num_players,
                "epochs": 100,  # Default epochs
                "batch_size": 4096,  # Default batch size
            }

            logger.info(f"NNUETrainingDaemon: Dispatching training for {config_key} to cluster")

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(dispatch_url, json=payload) as resp:
                    if resp.status != 200:
                        error_msg = f"Dispatch failed with status {resp.status}"
                        logger.error(f"NNUETrainingDaemon: {error_msg}")
                    else:
                        result = await resp.json()
                        if result.get("success"):
                            job_id = result.get("job_id", "unknown")
                            logger.info(
                                f"NNUETrainingDaemon: Training dispatched for {config_key}, "
                                f"job_id={job_id}"
                            )
                            # Wait for training to complete by polling status
                            success = await self._wait_for_training_completion(
                                session, p2p_port, config_key, job_id
                            )
                            if not success:
                                error_msg = "Training job failed or timed out"
                        else:
                            error_msg = result.get("error", "Unknown dispatch error")
                            logger.error(f"NNUETrainingDaemon: {error_msg}")

        except aiohttp.ClientError as e:
            error_msg = f"HTTP error: {e}"
            logger.error(f"NNUETrainingDaemon: Network error for {config_key}: {e}")
        except asyncio.TimeoutError:
            error_msg = "Dispatch request timed out"
            logger.error(f"NNUETrainingDaemon: Dispatch timed out for {config_key}")
        except (OSError, ValueError) as e:
            error_msg = str(e)
            logger.error(f"NNUETrainingDaemon: Training error for {config_key}: {e}")

        # Update state
        elapsed = time.time() - start_time
        self._state.active_trainings.pop(config_key, None)

        if success:
            self._state.last_training_time[config_key] = time.time()
            self._state.last_training_game_count[config_key] = game_count

        # Record in history
        self._state.training_history.append({
            "config_key": config_key,
            "timestamp": time.time(),
            "success": success,
            "duration_seconds": elapsed,
            "game_count": game_count,
            "error": error_msg if not success else None,
        })

        # Emit completion event
        safe_emit_event(
            "NNUE_TRAINING_COMPLETED",
            {
                "config_key": config_key,
                "success": success,
                "duration_seconds": elapsed,
                "game_count": game_count,
                "error": error_msg if not success else None,
                "timestamp": time.time(),
            },
            context="nnue_training",
        )

        self._save_state()

    async def _wait_for_training_completion(
        self,
        session: Any,  # aiohttp.ClientSession
        p2p_port: int,
        config_key: str,
        job_id: str,
    ) -> bool:
        """Poll P2P orchestrator for training job completion.

        Args:
            session: aiohttp client session
            p2p_port: P2P orchestrator port
            config_key: Configuration key being trained
            job_id: Job ID to monitor

        Returns:
            True if training completed successfully, False otherwise
        """
        import aiohttp

        status_url = f"http://localhost:{p2p_port}/training/status"
        poll_interval = 60.0  # Check every minute
        timeout = self._nnue_config.training_timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        jobs = result.get("jobs", {})

                        # December 31, 2025: Handle both dict and list formats
                        # P2P may return jobs as a list of job dicts or a dict keyed by job_id
                        job_status = None
                        if isinstance(jobs, dict):
                            job_status = jobs.get(job_id, {}).get("status")
                        elif isinstance(jobs, list):
                            # Search list for matching job_id
                            for job in jobs:
                                if isinstance(job, dict) and job.get("job_id") == job_id:
                                    job_status = job.get("status")
                                    break

                        if job_status == "completed":
                            logger.info(
                                f"NNUETrainingDaemon: Training completed for {config_key}"
                            )
                            return True
                        elif job_status in ("failed", "cancelled"):
                            logger.error(
                                f"NNUETrainingDaemon: Training {job_status} for {config_key}"
                            )
                            return False
                        # Still running, continue polling

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(f"NNUETrainingDaemon: Status poll error: {e}")

            await asyncio.sleep(poll_interval)

        logger.warning(f"NNUETrainingDaemon: Timed out waiting for {config_key} training")
        return False

    def _cleanup_active_trainings(self) -> None:
        """Clean up timed-out active trainings."""
        now = time.time()
        timeout = self._nnue_config.training_timeout_seconds

        timed_out = [
            config_key
            for config_key, start_time in self._state.active_trainings.items()
            if now - start_time > timeout
        ]

        for config_key in timed_out:
            logger.warning(f"NNUETrainingDaemon: Training for {config_key} timed out, cleaning up")
            self._state.active_trainings.pop(config_key, None)

    def health_check(self) -> HealthCheckResult:
        """Return health check result for this daemon."""
        from app.coordination.contracts import CoordinatorStatus

        is_healthy = self.is_running

        details = {
            "configs_tracked": len(self._current_game_counts),
            "active_trainings": len(self._state.active_trainings),
            "total_trainings": len(self._state.training_history),
            "last_check": self._stats.last_activity,
        }

        # Add recent training info
        if self._state.training_history:
            recent = self._state.training_history[-1]
            details["last_training"] = {
                "config": recent.get("config_key"),
                "success": recent.get("success"),
                "time": recent.get("timestamp"),
            }

        return HealthCheckResult(
            healthy=is_healthy,
            status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.STOPPED,
            details=details,
        )

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics for monitoring."""
        successful = sum(1 for h in self._state.training_history if h.get("success"))
        failed = len(self._state.training_history) - successful

        return {
            "total_trainings": len(self._state.training_history),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / max(1, len(self._state.training_history)),
            "active_trainings": list(self._state.active_trainings.keys()),
            "configs_tracked": list(self._current_game_counts.keys()),
        }


# ============================================
# Convenience Functions
# ============================================


def get_nnue_training_daemon(
    config: NNUETrainingConfig | None = None,
) -> NNUETrainingDaemon:
    """Get the singleton NNUE training daemon."""
    return NNUETrainingDaemon.get_instance(config=config)


async def trigger_nnue_training_if_needed(
    config_key: str,
    game_count: int,
) -> bool:
    """Manually check if NNUE training is needed for a config.

    Returns True if training was triggered.
    """
    daemon = get_nnue_training_daemon()
    daemon._current_game_counts[config_key] = game_count

    if daemon._should_train(config_key, game_count):
        await daemon._trigger_training(config_key, game_count)
        return True
    return False
