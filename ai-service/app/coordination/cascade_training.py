"""
Cascade Training Orchestrator for multiplayer bootstrapping.

December 29, 2025

This module implements cascade training for multiplayer models:
- Train 2-player models first (simplest, most data available)
- Transfer 2p weights to initialize 3-player models
- Transfer 3p weights to initialize 4-player models

This "bootstrapping" approach helps multiplayer models converge faster
by starting from learned features rather than random initialization.

Usage:
    orchestrator = CascadeTrainingOrchestrator()
    await orchestrator.start()

    # Or get cascade status
    status = orchestrator.get_cascade_status("hex8")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from app.coordination.handler_base import HandlerBase, HealthCheckResult

if TYPE_CHECKING:
    from app.coordination.event_router import UnifiedEventRouter

logger = logging.getLogger(__name__)


class CascadeStage(Enum):
    """Stage in the cascade training pipeline."""

    NOT_STARTED = "not_started"
    TRAINING_2P = "training_2p"
    READY_2P = "ready_2p"
    TRAINING_3P = "training_3p"
    READY_3P = "ready_3p"
    TRAINING_4P = "training_4p"
    COMPLETE = "complete"


@dataclass
class CascadeState:
    """State of cascade training for a board type."""

    board_type: str
    stage: CascadeStage = CascadeStage.NOT_STARTED

    # Model paths (None if not yet trained)
    model_2p: str | None = None
    model_3p: str | None = None
    model_4p: str | None = None

    # Elo ratings for gating decisions
    elo_2p: float = 0.0
    elo_3p: float = 0.0
    elo_4p: float = 0.0

    # Training state
    training_started: datetime | None = None
    last_updated: datetime = field(default_factory=datetime.now)

    # Quality gates
    min_elo_for_transfer: float = 1200.0  # Minimum Elo before transfer
    min_games_for_transfer: int = 1000    # Minimum games in training data

    def can_transfer_to_3p(self) -> bool:
        """Check if 2p model is ready to transfer to 3p."""
        return (
            self.model_2p is not None
            and self.elo_2p >= self.min_elo_for_transfer
            and self.stage in (CascadeStage.READY_2P, CascadeStage.TRAINING_3P)
        )

    def can_transfer_to_4p(self) -> bool:
        """Check if 3p model is ready to transfer to 4p."""
        return (
            self.model_3p is not None
            and self.elo_3p >= self.min_elo_for_transfer
            and self.stage in (CascadeStage.READY_3P, CascadeStage.TRAINING_4P)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "board_type": self.board_type,
            "stage": self.stage.value,
            "model_2p": self.model_2p,
            "model_3p": self.model_3p,
            "model_4p": self.model_4p,
            "elo_2p": self.elo_2p,
            "elo_3p": self.elo_3p,
            "elo_4p": self.elo_4p,
            "min_elo_for_transfer": self.min_elo_for_transfer,
            "min_games_for_transfer": self.min_games_for_transfer,
            "training_started": self.training_started.isoformat() if self.training_started else None,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class CascadeConfig:
    """Configuration for cascade training."""

    # Minimum Elo threshold before transferring weights
    min_elo_for_transfer: float = 1200.0

    # Minimum games in training data before attempting training
    min_games_for_training: int = 500

    # How much to boost selfplay for bootstrapping configs
    bootstrap_selfplay_multiplier: float = 1.5

    # Board types to manage (default: all canonical)
    board_types: list[str] = field(
        default_factory=lambda: ["hex8", "square8", "square19", "hexagonal"]
    )

    # Directories
    models_dir: str = "models"
    data_dir: str = "data/games"

    # Check interval (seconds)
    check_interval: float = 300.0  # 5 minutes


class CascadeTrainingOrchestrator(HandlerBase):
    """Orchestrates cascade training for multiplayer bootstrapping.

    The cascade training approach:
    1. Prioritize 2-player training (most efficient, stable)
    2. Once 2p reaches quality threshold, transfer to 3p
    3. Once 3p reaches threshold, transfer to 4p

    This accelerates multiplayer training by starting from
    learned features rather than random initialization.
    """

    _instance: CascadeTrainingOrchestrator | None = None

    def __init__(self, config: CascadeConfig | None = None):
        super().__init__(
            name="CascadeTrainingOrchestrator",
            cycle_interval=config.check_interval if config else 300.0,
        )
        self.config = config or CascadeConfig()
        self._states: dict[str, CascadeState] = {}
        self._event_router: UnifiedEventRouter | None = None

        # Initialize states for all board types
        for board_type in self.config.board_types:
            self._states[board_type] = CascadeState(
                board_type=board_type,
                min_elo_for_transfer=self.config.min_elo_for_transfer,
                min_games_for_transfer=self.config.min_games_for_training,
            )

    @classmethod
    def get_instance(cls, config: CascadeConfig | None = None) -> CascadeTrainingOrchestrator:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for this handler."""
        return {
            "TRAINING_COMPLETED": self._on_training_completed,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "MODEL_PROMOTED": self._on_model_promoted,
            "ELO_UPDATED": self._on_elo_updated,
            "CASCADE_TRANSFER_TRIGGERED": self._on_cascade_transfer_triggered,
        }

    async def _on_cascade_transfer_triggered(self, event: dict) -> None:
        """Handle cascade transfer trigger - actually run the transfer.

        December 29, 2025: Added to execute the transfer_2p_to_np script
        when cascade advancement is triggered.
        """
        board_type = event.get("board_type")
        source_players = event.get("source_players")
        target_players = event.get("target_players")
        source_model = event.get("source_model")

        if not all([board_type, source_players, target_players, source_model]):
            logger.warning(f"[CascadeTraining] Invalid transfer event: {event}")
            return

        await self._execute_transfer(
            board_type=board_type,
            source_model=source_model,
            source_players=source_players,
            target_players=target_players,
        )

    async def _execute_transfer(
        self,
        board_type: str,
        source_model: str,
        source_players: int,
        target_players: int,
    ) -> bool:
        """Execute the weight transfer from source to target player count.

        December 29, 2025: Implements the actual transfer by calling
        scripts/transfer_2p_to_4p.py or directly using the transfer function.

        Args:
            board_type: Board type (hex8, square8, etc.)
            source_model: Path to source model
            source_players: Source player count (2 or 3)
            target_players: Target player count (3 or 4)

        Returns:
            True if transfer succeeded
        """
        config_key = f"{board_type}_{target_players}p"
        output_path = Path(self.config.models_dir) / f"transfer_{board_type}_{target_players}p_init.pth"

        logger.info(
            f"[CascadeTraining] Executing transfer: {source_model} → {output_path} "
            f"({source_players}p → {target_players}p)"
        )

        try:
            # Run transfer in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._run_transfer_sync,
                source_model,
                str(output_path),
                board_type,
                target_players,
            )

            # Verify output exists
            if not output_path.exists():
                logger.error(f"[CascadeTraining] Transfer failed: output not found at {output_path}")
                return False

            logger.info(f"[CascadeTraining] Transfer complete: {output_path}")

            # Emit event to trigger training with transferred weights
            await self._emit_event(
                "TRAINING_REQUESTED",
                {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": target_players,
                    "init_weights": str(output_path),
                    "cascade_stage": True,
                    "transfer_from": source_model,
                },
            )

            return True

        except (OSError, RuntimeError, ValueError, TypeError, asyncio.CancelledError) as e:
            logger.error(f"[CascadeTraining] Transfer failed: {e}")
            await self._emit_event(
                "CASCADE_TRANSFER_FAILED",
                {
                    "config_key": config_key,
                    "board_type": board_type,
                    "source_players": source_players,
                    "target_players": target_players,
                    "error": str(e),
                },
            )
            return False

    def _run_transfer_sync(
        self,
        source_path: str,
        output_path: str,
        board_type: str,
        target_players: int,
    ) -> None:
        """Run the transfer synchronously (for executor)."""
        from scripts.transfer_2p_to_4p import transfer_2p_to_np

        transfer_2p_to_np(
            source_path=source_path,
            output_path=output_path,
            board_type=board_type,
            target_players=target_players,
        )

    async def _on_training_completed(self, event: dict) -> None:
        """Handle training completion event."""
        config_key = event.get("config_key", "")
        if not config_key:
            return

        # Parse config_key: "hex8_2p" -> board_type="hex8", num_players=2
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return

        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return

        if board_type not in self._states:
            return

        state = self._states[board_type]
        model_path = event.get("model_path")

        if num_players == 2:
            state.model_2p = model_path
            if state.stage == CascadeStage.TRAINING_2P:
                state.stage = CascadeStage.READY_2P
                logger.info(f"[CascadeTraining] {board_type} 2p training complete")
        elif num_players == 3:
            state.model_3p = model_path
            if state.stage == CascadeStage.TRAINING_3P:
                state.stage = CascadeStage.READY_3P
                logger.info(f"[CascadeTraining] {board_type} 3p training complete")
        elif num_players == 4:
            state.model_4p = model_path
            if state.stage == CascadeStage.TRAINING_4P:
                state.stage = CascadeStage.COMPLETE
                logger.info(f"[CascadeTraining] {board_type} cascade COMPLETE!")

        state.last_updated = datetime.now()

    async def _on_evaluation_completed(self, event: dict) -> None:
        """Handle evaluation completion - may trigger cascade advancement."""
        config_key = event.get("config_key", "")
        elo = event.get("elo", 0.0)

        if not config_key or not elo:
            return

        await self._on_elo_updated({"config_key": config_key, "elo": elo})

    async def _on_elo_updated(self, event: dict) -> None:
        """Handle Elo update - check if we can advance cascade."""
        config_key = event.get("config_key", "")
        elo = event.get("elo", 0.0)

        if not config_key:
            return

        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return

        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return

        if board_type not in self._states:
            return

        state = self._states[board_type]

        if num_players == 2:
            state.elo_2p = elo
            if state.can_transfer_to_3p() and state.stage == CascadeStage.READY_2P:
                await self._trigger_transfer(board_type, 2, 3)
        elif num_players == 3:
            state.elo_3p = elo
            if state.can_transfer_to_4p() and state.stage == CascadeStage.READY_3P:
                await self._trigger_transfer(board_type, 3, 4)
        elif num_players == 4:
            state.elo_4p = elo

        state.last_updated = datetime.now()

    async def _on_model_promoted(self, event: dict) -> None:
        """Handle model promotion - update cascade state."""
        config_key = event.get("config_key", "")
        model_path = event.get("model_path")

        if not config_key or not model_path:
            return

        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return

        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return

        if board_type not in self._states:
            return

        state = self._states[board_type]

        if num_players == 2:
            state.model_2p = model_path
        elif num_players == 3:
            state.model_3p = model_path
        elif num_players == 4:
            state.model_4p = model_path

        state.last_updated = datetime.now()

    async def _trigger_transfer(
        self, board_type: str, source_players: int, target_players: int
    ) -> None:
        """Trigger weight transfer and training for next player count."""
        state = self._states[board_type]
        source_model = state.model_2p if source_players == 2 else state.model_3p

        if not source_model:
            logger.warning(
                f"[CascadeTraining] Cannot transfer {board_type} {source_players}p→{target_players}p: "
                f"no source model"
            )
            return

        logger.info(
            f"[CascadeTraining] Triggering transfer: {board_type} {source_players}p → {target_players}p"
        )

        # Update stage
        if target_players == 3:
            state.stage = CascadeStage.TRAINING_3P
        elif target_players == 4:
            state.stage = CascadeStage.TRAINING_4P

        state.training_started = datetime.now()

        # Emit event to trigger transfer
        await self._emit_event(
            "CASCADE_TRANSFER_TRIGGERED",
            {
                "board_type": board_type,
                "source_players": source_players,
                "target_players": target_players,
                "source_model": source_model,
                "config_key": f"{board_type}_{target_players}p",
            },
        )

    async def _emit_event(self, event_type: str, payload: dict) -> None:
        """Emit event via router if available."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            await router.publish(event_type, payload)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[CascadeTraining] Event emission failed: {e}")

    async def _run_cycle(self) -> None:
        """Run periodic cascade state check."""
        for board_type, state in self._states.items():
            await self._check_cascade_state(board_type, state)

    async def _check_cascade_state(self, board_type: str, state: CascadeState) -> None:
        """Check and potentially advance cascade state for a board type."""
        # Discover existing models
        await self._discover_models(board_type, state)

        # Check for automatic stage transitions
        if state.stage == CascadeStage.NOT_STARTED:
            if state.model_2p:
                state.stage = CascadeStage.READY_2P
            else:
                # Check if we have enough data to train 2p
                game_count = await self._get_game_count(board_type, 2)
                if game_count >= self.config.min_games_for_training:
                    state.stage = CascadeStage.TRAINING_2P
                    await self._request_training(board_type, 2)

        elif state.stage == CascadeStage.READY_2P:
            if state.can_transfer_to_3p():
                await self._trigger_transfer(board_type, 2, 3)

        elif state.stage == CascadeStage.READY_3P:
            if state.can_transfer_to_4p():
                await self._trigger_transfer(board_type, 3, 4)

    async def _discover_models(self, board_type: str, state: CascadeState) -> None:
        """Discover existing models for a board type."""
        models_dir = Path(self.config.models_dir)

        for num_players in [2, 3, 4]:
            # Check multiple naming conventions
            patterns = [
                f"canonical_{board_type}_{num_players}p.pth",
                f"ringrift_best_{board_type}_{num_players}p.pth",
                f"{board_type}_{num_players}p*.pth",
            ]

            for pattern in patterns:
                matches = list(models_dir.glob(pattern))
                if matches:
                    model_path = str(matches[0])
                    if num_players == 2 and not state.model_2p:
                        state.model_2p = model_path
                    elif num_players == 3 and not state.model_3p:
                        state.model_3p = model_path
                    elif num_players == 4 and not state.model_4p:
                        state.model_4p = model_path
                    break

    async def _get_game_count(self, board_type: str, num_players: int) -> int:
        """Get count of games available for training."""
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery(self.config.data_dir)
            databases = discovery.find_databases_for_config(board_type, num_players)
            return sum(db.game_count for db in databases)
        except (ImportError, FileNotFoundError, OSError) as e:
            logger.debug(f"[CascadeTraining] Game count lookup failed: {e}")
            return 0

    async def _request_training(self, board_type: str, num_players: int) -> None:
        """Request training for a configuration."""
        config_key = f"{board_type}_{num_players}p"
        logger.info(f"[CascadeTraining] Requesting training for {config_key}")

        await self._emit_event(
            "TRAINING_REQUESTED",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "cascade_stage": True,
            },
        )

    def get_cascade_status(self, board_type: str) -> CascadeState | None:
        """Get cascade status for a board type."""
        return self._states.get(board_type)

    def get_all_cascade_status(self) -> dict[str, dict]:
        """Get cascade status for all board types."""
        return {bt: state.to_dict() for bt, state in self._states.items()}

    def get_bootstrap_priority(self, config_key: str) -> float:
        """Get priority boost for bootstrapping configs.

        Returns a multiplier (1.0 = normal, >1.0 = boosted).
        Configs that are blocking cascade advancement get boosted.
        """
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return 1.0

        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return 1.0

        if board_type not in self._states:
            return 1.0

        state = self._states[board_type]

        # Boost 2p if it's blocking 3p/4p cascade
        if num_players == 2 and state.stage in (
            CascadeStage.NOT_STARTED,
            CascadeStage.TRAINING_2P,
        ):
            return self.config.bootstrap_selfplay_multiplier

        # Boost 3p if it's the next step in cascade
        if num_players == 3 and state.stage in (
            CascadeStage.READY_2P,
            CascadeStage.TRAINING_3P,
        ):
            return self.config.bootstrap_selfplay_multiplier

        # Boost 4p if it's the final step
        if num_players == 4 and state.stage in (
            CascadeStage.READY_3P,
            CascadeStage.TRAINING_4P,
        ):
            return self.config.bootstrap_selfplay_multiplier

        return 1.0

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        # Count stages
        complete = sum(1 for s in self._states.values() if s.stage == CascadeStage.COMPLETE)
        in_progress = sum(
            1 for s in self._states.values()
            if s.stage not in (CascadeStage.NOT_STARTED, CascadeStage.COMPLETE)
        )

        return HealthCheckResult(
            healthy=True,
            message=f"Cascade: {complete}/{len(self._states)} complete, {in_progress} in progress",
            details={
                "board_types": len(self._states),
                "complete": complete,
                "in_progress": in_progress,
                "states": {bt: s.stage.value for bt, s in self._states.items()},
            },
        )


# Singleton accessor
def get_cascade_orchestrator(
    config: CascadeConfig | None = None,
) -> CascadeTrainingOrchestrator:
    """Get the cascade training orchestrator singleton."""
    return CascadeTrainingOrchestrator.get_instance(config)
