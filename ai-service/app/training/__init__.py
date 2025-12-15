"""Training module for RingRift AI.

This package provides training infrastructure including:
- Promotion controller for model promotion decisions
- Elo service for rating management
- Model registry for lifecycle tracking
- Tier promotion for difficulty ladder

Usage:
    from app.training import (
        PromotionController,
        PromotionType,
        PromotionCriteria,
        get_promotion_controller,
    )

    controller = get_promotion_controller()
    decision = controller.evaluate_promotion(
        model_id="model_v42",
        board_type="square8",
        num_players=2,
    )
    if decision.should_promote:
        controller.execute_promotion(decision)
"""

# Import promotion controller if available
try:
    from app.training.promotion_controller import (
        PromotionController,
        PromotionType,
        PromotionCriteria,
        PromotionDecision,
        get_promotion_controller,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False

__all__ = []

if HAS_PROMOTION_CONTROLLER:
    __all__.extend([
        "PromotionController",
        "PromotionType",
        "PromotionCriteria",
        "PromotionDecision",
        "get_promotion_controller",
        "HAS_PROMOTION_CONTROLLER",
    ])
