"""Training Status Routes for RingRift AI Service.

Provides endpoints for training loop monitoring and Elo velocity tracking:
- GET /training/{config_key}/velocity - Elo velocity and trend
- GET /training/{config_key}/momentum - Full momentum status
- GET /training/status - Aggregate training status
- GET /training/feedback - FeedbackAccelerator status

Usage:
    from app.routes import training_router

    app.include_router(training_router, prefix="/api", tags=["training"])

December 2025: Added for Phase 15 integration - exposing Elo velocity to operators.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

__all__ = ["router"]


# =============================================================================
# Response Models
# =============================================================================


class EloVelocityResponse(BaseModel):
    """Elo velocity response."""
    config_key: str = Field(..., description="Configuration key (e.g., 'hex8_2p')")
    elo_per_hour: float = Field(..., description="Elo points gained per hour")
    elo_per_game: float = Field(0.0, description="Elo points gained per game")
    trend: str = Field(..., description="Trend: 'improving', 'stable', or 'declining'")
    lookback_hours: float = Field(24.0, description="Lookback period for calculation")
    games_in_period: int = Field(0, description="Games played in lookback period")
    current_elo: float = Field(1500.0, description="Current Elo rating")


class MomentumResponse(BaseModel):
    """Full momentum status response."""
    config_key: str = Field(..., description="Configuration key")
    current_elo: float = Field(1500.0, description="Current Elo rating")
    momentum_state: str = Field(..., description="Momentum state")
    intensity: str = Field(..., description="Training intensity")
    games_since_training: int = Field(0, description="Games since last training")
    consecutive_improvements: int = Field(0, description="Consecutive improvements")
    consecutive_plateaus: int = Field(0, description="Consecutive plateaus")
    elo_trend: float = Field(0.0, description="Recent Elo trend")
    improvement_rate_per_hour: float = Field(0.0, description="Improvement rate")


class TrainingStatusResponse(BaseModel):
    """Aggregate training status response."""
    total_configs: int = Field(0, description="Total tracked configurations")
    improving_configs: int = Field(0, description="Configs currently improving")
    plateau_configs: int = Field(0, description="Configs in plateau")
    total_improvement_rate_per_hour: float = Field(0.0, description="Total improvement rate")
    configs: dict[str, Any] = Field(default_factory=dict, description="Per-config status")


class FeedbackStatusResponse(BaseModel):
    """FeedbackAccelerator status response."""
    improving: list[str] = Field(default_factory=list, description="Improving config keys")
    plateau: list[str] = Field(default_factory=list, description="Plateau config keys")
    aggregate_momentum: str = Field("unknown", description="Aggregate momentum state")
    recommended_multiplier: float = Field(1.0, description="Recommended selfplay multiplier")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/{config_key}/velocity", response_model=EloVelocityResponse)
async def get_elo_velocity(config_key: str, lookback_hours: float = 24.0) -> EloVelocityResponse:
    """Get Elo velocity (improvement rate) for a configuration.

    This endpoint exposes the Elo velocity tracking added in December 2025.
    Use it to monitor training progress and detect stalls.

    Args:
        config_key: Configuration key (e.g., "hex8_2p", "square8_4p")
        lookback_hours: Hours to look back for trend calculation (default: 24)

    Returns:
        Elo velocity metrics including trend and current Elo
    """
    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()
        momentum = accelerator.get_config_momentum(config_key)

        if momentum is None:
            return EloVelocityResponse(
                config_key=config_key,
                elo_per_hour=0.0,
                elo_per_game=0.0,
                trend="unknown",
                lookback_hours=lookback_hours,
                games_in_period=0,
                current_elo=1500.0,
            )

        # Compute velocity
        improvement_rate = momentum.get_improvement_rate()
        elo_trend = momentum.get_elo_trend()

        # Determine trend
        if improvement_rate > 10:
            trend = "improving"
        elif improvement_rate < -5:
            trend = "declining"
        else:
            trend = "stable"

        # Compute games in period
        games_in_period = 0
        if len(momentum.elo_history) >= 2:
            games_in_period = (
                momentum.elo_history[-1].games_played - momentum.elo_history[0].games_played
            )

        return EloVelocityResponse(
            config_key=config_key,
            elo_per_hour=improvement_rate,
            elo_per_game=elo_trend,
            trend=trend,
            lookback_hours=lookback_hours,
            games_in_period=games_in_period,
            current_elo=momentum.current_elo,
        )

    except ImportError as e:
        logger.warning(f"FeedbackAccelerator not available: {e}")
        raise HTTPException(status_code=503, detail="FeedbackAccelerator not available")
    except Exception as e:
        logger.error(f"Error getting Elo velocity for {config_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{config_key}/momentum", response_model=MomentumResponse)
async def get_momentum(config_key: str) -> MomentumResponse:
    """Get full momentum status for a configuration.

    Provides detailed momentum tracking including intensity, consecutive
    improvements/plateaus, and improvement rate.

    Args:
        config_key: Configuration key (e.g., "hex8_2p", "square8_4p")

    Returns:
        Full momentum status
    """
    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()
        momentum = accelerator.get_config_momentum(config_key)

        if momentum is None:
            raise HTTPException(
                status_code=404,
                detail=f"No momentum data for config: {config_key}"
            )

        return MomentumResponse(
            config_key=config_key,
            current_elo=momentum.current_elo,
            momentum_state=momentum.momentum_state.value,
            intensity=momentum.intensity.value,
            games_since_training=momentum.games_since_training,
            consecutive_improvements=momentum.consecutive_improvements,
            consecutive_plateaus=momentum.consecutive_plateaus,
            elo_trend=momentum.get_elo_trend(),
            improvement_rate_per_hour=momentum.get_improvement_rate(),
        )

    except HTTPException:
        raise
    except ImportError as e:
        logger.warning(f"FeedbackAccelerator not available: {e}")
        raise HTTPException(status_code=503, detail="FeedbackAccelerator not available")
    except Exception as e:
        logger.error(f"Error getting momentum for {config_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status() -> TrainingStatusResponse:
    """Get aggregate training status across all configurations.

    Provides overview of training progress across all tracked configs.

    Returns:
        Aggregate training status
    """
    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()
        status = accelerator.get_status_summary()

        return TrainingStatusResponse(
            total_configs=status.get("total_configs", 0),
            improving_configs=status.get("improving_configs", 0),
            plateau_configs=status.get("plateau_configs", 0),
            total_improvement_rate_per_hour=status.get("total_improvement_rate_per_hour", 0.0),
            configs=status.get("configs", {}),
        )

    except ImportError as e:
        logger.warning(f"FeedbackAccelerator not available: {e}")
        raise HTTPException(status_code=503, detail="FeedbackAccelerator not available")
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback", response_model=FeedbackStatusResponse)
async def get_feedback_status() -> FeedbackStatusResponse:
    """Get FeedbackAccelerator status with recommendations.

    Provides improving/plateau configs and aggregate selfplay recommendations.

    Returns:
        Feedback loop status with recommendations
    """
    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()

        improving = accelerator.get_improving_configs()
        plateau = accelerator.get_plateau_configs()
        aggregate = accelerator.get_aggregate_selfplay_recommendation()

        return FeedbackStatusResponse(
            improving=improving,
            plateau=plateau,
            aggregate_momentum=aggregate.get("aggregate_momentum", "unknown"),
            recommended_multiplier=aggregate.get("recommended_multiplier", 1.0),
        )

    except ImportError as e:
        logger.warning(f"FeedbackAccelerator not available: {e}")
        raise HTTPException(status_code=503, detail="FeedbackAccelerator not available")
    except Exception as e:
        logger.error(f"Error getting feedback status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
