"""Reanalysis Daemon - Orchestrates reanalysis of historical games.

Re-evaluates historical games with improved models to get better training targets.
Triggers when a model's Elo improves significantly, leveraging the existing
ReanalysisEngine from app/training/reanalysis.py.

January 27, 2026 - Phase 2.1: Reanalysis Pipeline Integration
Expected Impact: +25-50 Elo from improved training targets

Usage:
    from app.coordination.reanalysis_daemon import ReanalysisDaemon

    daemon = ReanalysisDaemon()
    await daemon.start()

Events Subscribed:
    - MODEL_PROMOTED: Triggers reanalysis check when model is promoted
    - EVALUATION_COMPLETED: Updates Elo tracking for reanalysis decisions

Events Emitted:
    - REANALYSIS_STARTED: Reanalysis job started for a config
    - REANALYSIS_COMPLETED: Reanalysis job finished successfully
    - REANALYSIS_FAILED: Reanalysis job failed
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.coordination.event_router import get_event_payload
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

# Default path for state persistence
DEFAULT_STATE_PATH = Path("data/coordination/reanalysis_state.json")

# Default paths for training data
DEFAULT_TRAINING_DIR = Path("data/training")
DEFAULT_REANALYSIS_OUTPUT_DIR = Path("data/reanalysis")

__all__ = [
    "ReanalysisConfig",
    "ReanalysisDaemon",
    "create_reanalysis_daemon",
    "get_reanalysis_daemon",
    "reset_reanalysis_daemon",
]


@dataclass
class ReanalysisConfig:
    """Configuration for the reanalysis daemon.

    Attributes:
        min_elo_delta: Minimum Elo improvement to trigger reanalysis (default: 50)
        min_interval_hours: Minimum hours between reanalysis runs per config (default: 6)
        max_games_per_run: Maximum games to reanalyze per run (default: 1000)
        value_blend_ratio: Weight for new model's value predictions (default: 0.7)
        policy_blend_ratio: Weight for new model's policy predictions (default: 0.8)
        enabled: Whether reanalysis is enabled (default: True)
        dry_run: If True, log actions without executing (default: False)
        use_mcts: Use GPU MCTS for higher quality reanalysis (default: False)
        mcts_simulations: MCTS simulations per position if use_mcts (default: 100)
        state_path: Path for persisting state (default: data/coordination/reanalysis_state.json)
        training_dir: Directory containing training NPZ files (default: data/training)
        output_dir: Directory for reanalyzed NPZ files (default: data/reanalysis)
    """

    min_elo_delta: int = 50
    min_interval_hours: float = 6.0
    max_games_per_run: int = 1000
    value_blend_ratio: float = 0.7
    policy_blend_ratio: float = 0.8
    enabled: bool = True
    dry_run: bool = False
    use_mcts: bool = False
    mcts_simulations: int = 100
    state_path: Path = DEFAULT_STATE_PATH
    training_dir: Path = DEFAULT_TRAINING_DIR
    output_dir: Path = DEFAULT_REANALYSIS_OUTPUT_DIR

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)
        if isinstance(self.training_dir, str):
            self.training_dir = Path(self.training_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ConfigReanalysisState:
    """State for a single config's reanalysis history."""

    config_key: str
    last_reanalysis_elo: float = 0.0
    last_reanalysis_time: float = 0.0
    current_elo: float = 0.0
    reanalysis_count: int = 0
    last_npz_path: str = ""
    last_output_path: str = ""


class ReanalysisDaemon(HandlerBase):
    """Daemon that orchestrates reanalysis of historical games.

    When a model is promoted with significant Elo improvement, this daemon
    triggers reanalysis of historical games using the new model to generate
    improved training targets.

    This leverages the ReanalysisEngine from app/training/reanalysis.py.
    """

    _event_source = "ReanalysisDaemon"
    _instance: ReanalysisDaemon | None = None
    _lock = asyncio.Lock()

    def __init__(self, config: ReanalysisConfig | None = None) -> None:
        """Initialize the reanalysis daemon.

        Args:
            config: Daemon configuration. Uses defaults if not provided.
        """
        super().__init__(name="reanalysis_daemon", cycle_interval=300.0)  # 5 min cycle

        self.config = config or ReanalysisConfig()
        self._config_states: dict[str, ConfigReanalysisState] = {}
        self._reanalysis_in_progress: set[str] = set()
        self._total_reanalyses: int = 0
        self._successful_reanalyses: int = 0
        self._failed_reanalyses: int = 0
        self._last_check_time: float = 0.0

        # Load persisted state
        self._load_state()

    @classmethod
    async def get_instance(cls, config: ReanalysisConfig | None = None) -> ReanalysisDaemon:
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for this daemon."""
        return {
            "model_promoted": self._on_model_promoted,
            "evaluation_completed": self._on_evaluation_completed,
        }

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check for pending reanalysis work."""
        if not self.config.enabled:
            return

        self._last_check_time = time.time()

        # Check for any configs that need reanalysis based on Elo delta
        for config_key, state in self._config_states.items():
            if config_key in self._reanalysis_in_progress:
                continue

            if self._should_reanalyze(state):
                await self._trigger_reanalysis(config_key)

    def _should_reanalyze(self, state: ConfigReanalysisState) -> bool:
        """Check if a config should be reanalyzed.

        Args:
            state: Config's reanalysis state

        Returns:
            True if reanalysis should be triggered
        """
        # Check Elo delta
        elo_delta = state.current_elo - state.last_reanalysis_elo
        if elo_delta < self.config.min_elo_delta:
            return False

        # Check time since last reanalysis
        hours_since = (time.time() - state.last_reanalysis_time) / 3600
        if hours_since < self.config.min_interval_hours:
            return False

        return True

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle MODEL_PROMOTED event.

        When a model is promoted, check if reanalysis should be triggered
        based on the Elo improvement.
        """
        if not self.config.enabled:
            return

        # Deduplicate event
        if self._is_duplicate_event(event):
            return

        # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
        payload = get_event_payload(event)
        config_key = payload.get("config_key", "")
        new_elo = payload.get("elo", payload.get("new_elo", 0))
        model_path = payload.get("model_path", "")

        if not config_key:
            logger.warning("[ReanalysisDaemon] MODEL_PROMOTED missing config_key")
            return

        logger.info(
            f"[ReanalysisDaemon] Model promoted for {config_key}: "
            f"Elo={new_elo}, path={model_path}"
        )

        # Update config state
        state = self._get_or_create_state(config_key)
        state.current_elo = new_elo

        # Check if reanalysis should be triggered
        if self._should_reanalyze(state):
            await self._trigger_reanalysis(config_key, model_path=model_path)
        else:
            elo_delta = new_elo - state.last_reanalysis_elo
            logger.debug(
                f"[ReanalysisDaemon] Skipping reanalysis for {config_key}: "
                f"Elo delta {elo_delta:.1f} < threshold {self.config.min_elo_delta}"
            )

        # Save state
        self._save_state()

    async def _on_evaluation_completed(self, event: dict[str, Any]) -> None:
        """Handle EVALUATION_COMPLETED event.

        Updates Elo tracking for reanalysis decisions.
        """
        if not self.config.enabled:
            return

        # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
        payload = get_event_payload(event)
        config_key = payload.get("config_key", "")
        elo = payload.get("elo", 0)

        if not config_key:
            return

        # Update config state
        state = self._get_or_create_state(config_key)
        if elo > state.current_elo:
            state.current_elo = elo

    async def _trigger_reanalysis(
        self,
        config_key: str,
        model_path: str | None = None,
    ) -> None:
        """Trigger reanalysis for a config.

        Args:
            config_key: The config to reanalyze (e.g., "hex8_2p")
            model_path: Path to the model to use (optional, will find canonical)
        """
        if config_key in self._reanalysis_in_progress:
            logger.warning(f"[ReanalysisDaemon] Reanalysis already in progress for {config_key}")
            return

        state = self._get_or_create_state(config_key)

        logger.info(
            f"[ReanalysisDaemon] Triggering reanalysis for {config_key}: "
            f"Elo delta={state.current_elo - state.last_reanalysis_elo:.1f}"
        )

        if self.config.dry_run:
            logger.info(f"[ReanalysisDaemon] DRY RUN: Would reanalyze {config_key}")
            return

        self._reanalysis_in_progress.add(config_key)
        self._total_reanalyses += 1

        # Emit start event
        await self._safe_emit_event_async(
            "reanalysis_started",
            {
                "config_key": config_key,
                "current_elo": state.current_elo,
                "last_elo": state.last_reanalysis_elo,
                "elo_delta": state.current_elo - state.last_reanalysis_elo,
                "model_path": model_path or "",
            },
        )

        try:
            success = await self._run_reanalysis(config_key, model_path)

            if success:
                self._successful_reanalyses += 1
                state.last_reanalysis_elo = state.current_elo
                state.last_reanalysis_time = time.time()
                state.reanalysis_count += 1

                # Emit completion event
                await self._safe_emit_event_async(
                    "reanalysis_completed",
                    {
                        "config_key": config_key,
                        "elo": state.current_elo,
                        "output_path": state.last_output_path,
                        "reanalysis_count": state.reanalysis_count,
                    },
                )

                logger.info(f"[ReanalysisDaemon] Reanalysis completed for {config_key}")
            else:
                self._failed_reanalyses += 1
                await self._safe_emit_event_async(
                    "reanalysis_failed",
                    {
                        "config_key": config_key,
                        "reason": "reanalysis_engine_failed",
                    },
                )

        except Exception as e:
            self._failed_reanalyses += 1
            logger.error(f"[ReanalysisDaemon] Reanalysis failed for {config_key}: {e}")
            await self._safe_emit_event_async(
                "reanalysis_failed",
                {
                    "config_key": config_key,
                    "reason": str(e),
                },
            )
        finally:
            self._reanalysis_in_progress.discard(config_key)
            self._save_state()

    async def _run_reanalysis(
        self,
        config_key: str,
        model_path: str | None = None,
    ) -> bool:
        """Run the actual reanalysis using ReanalysisEngine.

        Args:
            config_key: Config to reanalyze
            model_path: Model path to use

        Returns:
            True if reanalysis succeeded
        """
        try:
            # Import here to avoid circular imports and optional dependencies
            from app.training.reanalysis import ReanalysisConfig as EngineConfig
            from app.training.reanalysis import ReanalysisEngine
        except ImportError as e:
            logger.error(f"[ReanalysisDaemon] Cannot import reanalysis engine: {e}")
            return False

        # Find NPZ file for this config
        npz_path = self._find_npz_for_config(config_key)
        if not npz_path:
            logger.warning(f"[ReanalysisDaemon] No NPZ file found for {config_key}")
            return False

        # Find or load model
        model = await self._load_model(config_key, model_path)
        if model is None:
            logger.warning(f"[ReanalysisDaemon] Cannot load model for {config_key}")
            return False

        # Configure reanalysis engine
        engine_config = EngineConfig(
            batch_size=64,
            max_games_per_run=self.config.max_games_per_run,
            value_blend_ratio=self.config.value_blend_ratio,
            policy_blend_ratio=self.config.policy_blend_ratio,
            min_model_elo_delta=self.config.min_elo_delta,
            reanalysis_interval_hours=self.config.min_interval_hours,
            use_mcts=self.config.use_mcts,
            mcts_simulations=self.config.mcts_simulations,
        )

        # Create output path
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_path = self.config.output_dir / f"{config_key}_reanalyzed_{timestamp}.npz"

        # Run reanalysis in thread pool to avoid blocking
        try:
            engine = ReanalysisEngine(model, engine_config)

            result_path = await asyncio.to_thread(
                engine.reanalyze_npz,
                npz_path,
                output_path,
            )

            if result_path and result_path.exists():
                state = self._get_or_create_state(config_key)
                state.last_npz_path = str(npz_path)
                state.last_output_path = str(result_path)
                return True
            return False

        except Exception as e:
            logger.error(f"[ReanalysisDaemon] ReanalysisEngine failed for {config_key}: {e}")
            return False

    def _find_npz_for_config(self, config_key: str) -> Path | None:
        """Find the NPZ training file for a config.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Path to NPZ file if found
        """
        # Try common naming patterns
        patterns = [
            f"{config_key}.npz",
            f"canonical_{config_key}.npz",
            f"{config_key}_combined.npz",
            f"{config_key}_training.npz",
        ]

        for pattern in patterns:
            path = self.config.training_dir / pattern
            if path.exists():
                return path

        # Try glob pattern
        matches = list(self.config.training_dir.glob(f"*{config_key}*.npz"))
        if matches:
            # Return most recent
            return max(matches, key=lambda p: p.stat().st_mtime)

        return None

    async def _load_model(
        self,
        config_key: str,
        model_path: str | None = None,
    ):
        """Load the model for reanalysis.

        Args:
            config_key: Config key for model lookup
            model_path: Explicit model path (optional)

        Returns:
            Loaded model or None
        """
        try:
            import torch
            from app.utils.torch_utils import safe_load_checkpoint
        except ImportError:
            logger.error("[ReanalysisDaemon] PyTorch not available")
            return None

        # Find model path
        if not model_path:
            model_dir = Path("models")
            candidates = [
                model_dir / f"canonical_{config_key}.pth",
                model_dir / f"ringrift_best_{config_key}.pth",
                model_dir / f"{config_key}.pth",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    break

        if not model_path or not Path(model_path).exists():
            logger.warning(f"[ReanalysisDaemon] Model not found for {config_key}")
            return None

        try:
            # Load model architecture
            from app.ai.neural_net.architecture_registry import get_model_for_checkpoint

            checkpoint = await asyncio.to_thread(safe_load_checkpoint, model_path)
            model = get_model_for_checkpoint(checkpoint)
            return model

        except Exception as e:
            logger.error(f"[ReanalysisDaemon] Failed to load model from {model_path}: {e}")
            return None

    def _get_or_create_state(self, config_key: str) -> ConfigReanalysisState:
        """Get or create state for a config."""
        if config_key not in self._config_states:
            self._config_states[config_key] = ConfigReanalysisState(config_key=config_key)
        return self._config_states[config_key]

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if not self.config.state_path.exists():
            return

        try:
            with open(self.config.state_path) as f:
                data = json.load(f)

            for config_key, state_data in data.get("config_states", {}).items():
                self._config_states[config_key] = ConfigReanalysisState(
                    config_key=config_key,
                    last_reanalysis_elo=state_data.get("last_reanalysis_elo", 0),
                    last_reanalysis_time=state_data.get("last_reanalysis_time", 0),
                    current_elo=state_data.get("current_elo", 0),
                    reanalysis_count=state_data.get("reanalysis_count", 0),
                    last_npz_path=state_data.get("last_npz_path", ""),
                    last_output_path=state_data.get("last_output_path", ""),
                )

            self._total_reanalyses = data.get("total_reanalyses", 0)
            self._successful_reanalyses = data.get("successful_reanalyses", 0)
            self._failed_reanalyses = data.get("failed_reanalyses", 0)

            logger.info(
                f"[ReanalysisDaemon] Loaded state: {len(self._config_states)} configs, "
                f"{self._total_reanalyses} total reanalyses"
            )

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ReanalysisDaemon] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.config.state_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "config_states": {
                    key: {
                        "last_reanalysis_elo": state.last_reanalysis_elo,
                        "last_reanalysis_time": state.last_reanalysis_time,
                        "current_elo": state.current_elo,
                        "reanalysis_count": state.reanalysis_count,
                        "last_npz_path": state.last_npz_path,
                        "last_output_path": state.last_output_path,
                    }
                    for key, state in self._config_states.items()
                },
                "total_reanalyses": self._total_reanalyses,
                "successful_reanalyses": self._successful_reanalyses,
                "failed_reanalyses": self._failed_reanalyses,
                "saved_at": time.time(),
            }

            with open(self.config.state_path, "w") as f:
                json.dump(data, f, indent=2)

        except OSError as e:
            logger.error(f"[ReanalysisDaemon] Failed to save state: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        # Calculate health based on success rate
        total = self._successful_reanalyses + self._failed_reanalyses
        success_rate = self._successful_reanalyses / total if total > 0 else 1.0

        is_healthy = (
            self.config.enabled
            and len(self._reanalysis_in_progress) < 5  # Not overloaded
            and success_rate >= 0.5  # At least 50% success rate
        )

        status = "healthy" if is_healthy else "degraded"
        if not self.config.enabled:
            status = "disabled"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            details={
                "enabled": self.config.enabled,
                "configs_tracked": len(self._config_states),
                "in_progress": len(self._reanalysis_in_progress),
                "total_reanalyses": self._total_reanalyses,
                "successful": self._successful_reanalyses,
                "failed": self._failed_reanalyses,
                "success_rate": round(success_rate, 3),
                "last_check_time": self._last_check_time,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            "enabled": self.config.enabled,
            "configs_tracked": len(self._config_states),
            "in_progress": list(self._reanalysis_in_progress),
            "total_reanalyses": self._total_reanalyses,
            "successful_reanalyses": self._successful_reanalyses,
            "failed_reanalyses": self._failed_reanalyses,
            "config_states": {
                key: {
                    "current_elo": state.current_elo,
                    "last_reanalysis_elo": state.last_reanalysis_elo,
                    "elo_delta": state.current_elo - state.last_reanalysis_elo,
                    "reanalysis_count": state.reanalysis_count,
                }
                for key, state in self._config_states.items()
            },
        }


# Factory functions
def create_reanalysis_daemon(config: ReanalysisConfig | None = None) -> ReanalysisDaemon:
    """Create a reanalysis daemon instance.

    Args:
        config: Optional configuration

    Returns:
        New ReanalysisDaemon instance
    """
    return ReanalysisDaemon(config)


async def get_reanalysis_daemon() -> ReanalysisDaemon:
    """Get the singleton reanalysis daemon instance."""
    return await ReanalysisDaemon.get_instance()


def reset_reanalysis_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    ReanalysisDaemon.reset_instance()
