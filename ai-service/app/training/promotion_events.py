"""Event definitions and emitters for automated model promotion.

This module provides event types and helper functions for emitting
promotion-related events that integrate with the feedback loop system.

Events emitted:
- MODEL_AUTO_PROMOTED: Model passed evaluation and was promoted
- MODEL_PROMOTION_REJECTED: Model failed evaluation criteria
- MODEL_PROMOTION_DEFERRED: Evaluation deferred (e.g., insufficient data)

Usage:
    from app.training.promotion_events import (
        PromotionEventType,
        emit_promotion_event,
    )

    emit_promotion_event(
        event_type=PromotionEventType.MODEL_AUTO_PROMOTED,
        model_path="models/hex8_2p_epoch50.pth",
        board_type="hex8",
        num_players=2,
        reason="Elo parity achieved: 1450 >= 1400",
    )

January 2026: Created for automated promotion feedback integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PromotionEventType(str, Enum):
    """Event types for model promotion decisions.

    These events integrate with the feedback loop system to:
    - Update curriculum weights after promotions
    - Trigger model distribution to cluster
    - Log promotion history for analysis
    """
    MODEL_AUTO_PROMOTED = "model_auto_promoted"
    MODEL_PROMOTION_REJECTED = "model_promotion_rejected"
    MODEL_PROMOTION_DEFERRED = "model_promotion_deferred"
    # Batch promotion events
    BATCH_PROMOTION_STARTED = "batch_promotion_started"
    BATCH_PROMOTION_COMPLETED = "batch_promotion_completed"


@dataclass
class PromotionEventData:
    """Structured data for promotion events.

    Attributes:
        model_path: Path to the model being evaluated
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        approved: Whether the model was promoted
        reason: Human-readable explanation
        criterion_met: Which criterion triggered approval (if any)
        estimated_elo: Model's estimated Elo rating
        win_rate_vs_random: Win rate against random baseline
        win_rate_vs_heuristic: Win rate against heuristic baseline
        promoted_path: Path where model was promoted (if approved)
        timestamp: When the event occurred
    """
    model_path: str
    board_type: str
    num_players: int
    approved: bool
    reason: str
    criterion_met: str | None = None
    estimated_elo: float | None = None
    win_rate_vs_random: float | None = None
    win_rate_vs_heuristic: float | None = None
    promoted_path: str | None = None
    timestamp: datetime | None = None

    @property
    def config_key(self) -> str:
        """Get the configuration key."""
        return f"{self.board_type}_{self.num_players}p"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for event emission."""
        data = {
            "model_path": self.model_path,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
            "approved": self.approved,
            "reason": self.reason,
            "timestamp": (self.timestamp or datetime.now(timezone.utc)).isoformat(),
        }

        if self.criterion_met:
            data["criterion_met"] = self.criterion_met
        if self.estimated_elo is not None:
            data["estimated_elo"] = self.estimated_elo
        if self.win_rate_vs_random is not None:
            data["win_rate_vs_random"] = self.win_rate_vs_random
        if self.win_rate_vs_heuristic is not None:
            data["win_rate_vs_heuristic"] = self.win_rate_vs_heuristic
        if self.promoted_path:
            data["promoted_path"] = self.promoted_path

        return data


def _get_event_router() -> Any | None:
    """Get the event router instance."""
    try:
        from app.coordination.event_router import get_router
        return get_router()
    except ImportError:
        return None


def _get_safe_emit() -> Any | None:
    """Get the safe_emit_event function."""
    try:
        from app.coordination.event_router import safe_emit_event
        return safe_emit_event
    except ImportError:
        return None


def emit_promotion_event(
    event_type: PromotionEventType,
    model_path: str,
    board_type: str,
    num_players: int,
    reason: str,
    approved: bool = False,
    criterion_met: str | None = None,
    estimated_elo: float | None = None,
    win_rate_vs_random: float | None = None,
    win_rate_vs_heuristic: float | None = None,
    promoted_path: str | None = None,
) -> bool:
    """Emit a promotion event to the event router.

    This function is safe to call even if the event router is not available.
    It will log a debug message if emission fails.

    Args:
        event_type: Type of promotion event
        model_path: Path to the model being evaluated
        board_type: Board type
        num_players: Number of players
        reason: Human-readable explanation
        approved: Whether the model was promoted
        criterion_met: Which criterion triggered approval
        estimated_elo: Model's estimated Elo
        win_rate_vs_random: Win rate against random
        win_rate_vs_heuristic: Win rate against heuristic
        promoted_path: Path where model was promoted

    Returns:
        True if event was emitted successfully, False otherwise
    """
    event_data = PromotionEventData(
        model_path=model_path,
        board_type=board_type,
        num_players=num_players,
        approved=approved,
        reason=reason,
        criterion_met=criterion_met,
        estimated_elo=estimated_elo,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_heuristic=win_rate_vs_heuristic,
        promoted_path=promoted_path,
        timestamp=datetime.now(timezone.utc),
    )

    # Try safe_emit_event first (preferred)
    safe_emit = _get_safe_emit()
    if safe_emit is not None:
        try:
            safe_emit(event_type.value, event_data.to_dict())
            logger.debug(f"[PromotionEvents] Emitted {event_type.value} for {event_data.config_key}")
            return True
        except Exception as e:
            logger.debug(f"[PromotionEvents] safe_emit_event failed: {e}")

    # Fall back to router.emit()
    router = _get_event_router()
    if router is not None:
        try:
            router.emit(event_type.value, event_data.to_dict())
            logger.debug(f"[PromotionEvents] Emitted {event_type.value} for {event_data.config_key}")
            return True
        except Exception as e:
            logger.debug(f"[PromotionEvents] router.emit failed: {e}")

    logger.debug(f"[PromotionEvents] Event router not available, {event_type.value} not emitted")
    return False


def emit_auto_promoted(
    model_path: str,
    board_type: str,
    num_players: int,
    reason: str,
    criterion_met: str,
    estimated_elo: float,
    promoted_path: str,
    **kwargs: Any,
) -> bool:
    """Convenience function to emit MODEL_AUTO_PROMOTED event.

    Args:
        model_path: Path to the original model
        board_type: Board type
        num_players: Number of players
        reason: Explanation of why model was promoted
        criterion_met: Which criterion triggered promotion
        estimated_elo: Model's Elo rating
        promoted_path: Path to the promoted canonical model
        **kwargs: Additional fields to include

    Returns:
        True if event was emitted successfully
    """
    return emit_promotion_event(
        event_type=PromotionEventType.MODEL_AUTO_PROMOTED,
        model_path=model_path,
        board_type=board_type,
        num_players=num_players,
        reason=reason,
        approved=True,
        criterion_met=criterion_met,
        estimated_elo=estimated_elo,
        promoted_path=promoted_path,
        **kwargs,
    )


def emit_promotion_rejected(
    model_path: str,
    board_type: str,
    num_players: int,
    reason: str,
    estimated_elo: float | None = None,
    win_rate_vs_random: float | None = None,
    win_rate_vs_heuristic: float | None = None,
    **kwargs: Any,
) -> bool:
    """Convenience function to emit MODEL_PROMOTION_REJECTED event.

    Args:
        model_path: Path to the model
        board_type: Board type
        num_players: Number of players
        reason: Explanation of why model was rejected
        estimated_elo: Model's Elo rating
        win_rate_vs_random: Win rate against random
        win_rate_vs_heuristic: Win rate against heuristic
        **kwargs: Additional fields to include

    Returns:
        True if event was emitted successfully
    """
    return emit_promotion_event(
        event_type=PromotionEventType.MODEL_PROMOTION_REJECTED,
        model_path=model_path,
        board_type=board_type,
        num_players=num_players,
        reason=reason,
        approved=False,
        estimated_elo=estimated_elo,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_heuristic=win_rate_vs_heuristic,
        **kwargs,
    )
