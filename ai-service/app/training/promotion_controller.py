"""Unified Promotion Controller for RingRift AI.

Provides a single entry point for all model promotion decisions, consolidating
logic from multiple modules:
- app/training/model_registry.py - Model lifecycle tracking
- app/training/tier_promotion_registry.py - Tier-based promotion for difficulty ladder
- app/integration/model_lifecycle.py - Full lifecycle management with evaluation

This controller does NOT replace those modules - it delegates to them based on
the promotion type. Use this controller for NEW code to ensure consistent
promotion criteria across the system.

Usage:
    from app.training.promotion_controller import PromotionController, PromotionType

    controller = PromotionController()

    # Check if a model should be promoted
    decision = controller.evaluate_promotion(
        model_id="model_v42",
        board_type="square8",
        num_players=2,
        promotion_type=PromotionType.PRODUCTION,
    )

    if decision.should_promote:
        controller.execute_promotion(decision)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromotionType(Enum):
    """Types of promotion decisions."""
    STAGING = "staging"          # Development → Staging
    PRODUCTION = "production"    # Staging → Production
    TIER = "tier"                # Tier ladder promotion (D1→D2, etc.)
    CHAMPION = "champion"        # Tournament champion promotion
    ROLLBACK = "rollback"        # Rollback to previous version


@dataclass
class PromotionCriteria:
    """Criteria for promotion evaluation.

    These are the canonical thresholds - sourced from unified_config.py.
    """
    min_elo_improvement: float = 25.0
    min_games_played: int = 50
    min_win_rate: float = 0.52
    max_value_mse_degradation: float = 0.05
    confidence_threshold: float = 0.95

    # Tier-specific
    tier_elo_threshold: Optional[float] = None
    tier_games_required: int = 100


@dataclass
class PromotionDecision:
    """Result of a promotion evaluation."""
    model_id: str
    promotion_type: PromotionType
    should_promote: bool
    reason: str

    # Evaluation metrics
    current_elo: Optional[float] = None
    elo_improvement: Optional[float] = None
    games_played: int = 0
    win_rate: Optional[float] = None
    confidence: Optional[float] = None

    # For tier promotions
    current_tier: Optional[str] = None
    target_tier: Optional[str] = None

    # Metadata
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    criteria_used: Optional[PromotionCriteria] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "promotion_type": self.promotion_type.value,
            "should_promote": self.should_promote,
            "reason": self.reason,
            "current_elo": self.current_elo,
            "elo_improvement": self.elo_improvement,
            "games_played": self.games_played,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "current_tier": self.current_tier,
            "target_tier": self.target_tier,
            "evaluated_at": self.evaluated_at,
        }


class PromotionController:
    """Unified controller for all model promotion decisions.

    Consolidates promotion logic from multiple systems while preserving
    backward compatibility with existing code.
    """

    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
        elo_service: Optional[Any] = None,
        model_registry: Optional[Any] = None,
    ):
        """Initialize the promotion controller.

        Args:
            criteria: Custom promotion criteria (uses defaults if None)
            elo_service: EloService instance (lazy-loaded if None)
            model_registry: ModelRegistry instance (lazy-loaded if None)
        """
        self.criteria = criteria or PromotionCriteria()
        self._elo_service = elo_service
        self._model_registry = model_registry
        self._lifecycle_manager = None
        self._tier_registry = None

    @property
    def elo_service(self):
        """Lazy-load EloService."""
        if self._elo_service is None:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except ImportError:
                logger.warning("EloService not available")
        return self._elo_service

    @property
    def model_registry(self):
        """Lazy-load ModelRegistry."""
        if self._model_registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._model_registry = ModelRegistry()
            except ImportError:
                logger.warning("ModelRegistry not available")
        return self._model_registry

    @property
    def lifecycle_manager(self):
        """Lazy-load ModelLifecycleManager."""
        if self._lifecycle_manager is None:
            try:
                from app.integration.model_lifecycle import (
                    ModelLifecycleManager,
                    LifecycleConfig,
                )
                self._lifecycle_manager = ModelLifecycleManager(LifecycleConfig())
            except ImportError:
                logger.warning("ModelLifecycleManager not available")
        return self._lifecycle_manager

    def evaluate_promotion(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        promotion_type: PromotionType = PromotionType.PRODUCTION,
        baseline_model_id: Optional[str] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether a model should be promoted.

        Args:
            model_id: ID of the model to evaluate
            board_type: Board type for evaluation
            num_players: Number of players
            promotion_type: Type of promotion to evaluate
            baseline_model_id: Model to compare against (for PRODUCTION/CHAMPION)
            **kwargs: Additional arguments for specific promotion types

        Returns:
            PromotionDecision with evaluation results
        """
        if promotion_type == PromotionType.TIER:
            return self._evaluate_tier_promotion(
                model_id, board_type, num_players, **kwargs
            )
        elif promotion_type == PromotionType.CHAMPION:
            return self._evaluate_champion_promotion(
                model_id, board_type, num_players, baseline_model_id, **kwargs
            )
        elif promotion_type == PromotionType.ROLLBACK:
            return self._evaluate_rollback(
                model_id, board_type, num_players, **kwargs
            )
        else:
            return self._evaluate_standard_promotion(
                model_id, board_type, num_players, promotion_type, baseline_model_id
            )

    def _evaluate_standard_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        promotion_type: PromotionType,
        baseline_model_id: Optional[str],
    ) -> PromotionDecision:
        """Evaluate standard staging/production promotion."""
        # Get current Elo and stats
        current_elo = None
        games_played = 0
        win_rate = None
        elo_improvement = None

        if self.elo_service:
            try:
                rating = self.elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played
                    win_rate = rating.win_rate
            except Exception as e:
                logger.warning(f"Failed to get Elo for {model_id}: {e}")

        # Get baseline Elo for comparison
        baseline_elo = None
        if baseline_model_id and self.elo_service:
            try:
                baseline = self.elo_service.get_rating(
                    baseline_model_id, board_type, num_players
                )
                if baseline:
                    baseline_elo = baseline.rating
                    if current_elo is not None:
                        elo_improvement = current_elo - baseline_elo
            except Exception as e:
                logger.warning(f"Failed to get baseline Elo: {e}")

        # Check promotion criteria
        should_promote = False
        reason = ""

        if games_played < self.criteria.min_games_played:
            reason = f"Insufficient games ({games_played} < {self.criteria.min_games_played})"
        elif elo_improvement is not None and elo_improvement < self.criteria.min_elo_improvement:
            reason = f"Insufficient Elo improvement ({elo_improvement:.1f} < {self.criteria.min_elo_improvement})"
        elif win_rate is not None and win_rate < self.criteria.min_win_rate:
            reason = f"Win rate too low ({win_rate:.2%} < {self.criteria.min_win_rate:.2%})"
        else:
            should_promote = True
            reason = f"Meets all criteria: Elo +{elo_improvement or 0:.1f}, {games_played} games"

        decision = PromotionDecision(
            model_id=model_id,
            promotion_type=promotion_type,
            should_promote=should_promote,
            reason=reason,
            current_elo=current_elo,
            elo_improvement=elo_improvement,
            games_played=games_played,
            win_rate=win_rate,
            criteria_used=self.criteria,
        )

        # Emit metrics
        self._emit_decision_metrics(decision)

        return decision

    def _emit_decision_metrics(self, decision: PromotionDecision) -> None:
        """Emit Prometheus metrics for a promotion decision."""
        try:
            from app.metrics import record_promotion_decision
            record_promotion_decision(
                promotion_type=decision.promotion_type.value,
                approved=decision.should_promote,
                elo_improvement=decision.elo_improvement,
            )
        except ImportError:
            pass

    def _evaluate_tier_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        current_tier: Optional[str] = None,
        target_tier: Optional[str] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate tier-based promotion for difficulty ladder."""
        # Load tier registry
        try:
            from app.training.tier_promotion_registry import (
                load_square8_two_player_registry,
                get_current_ladder_model_for_tier,
            )
            from app.config.ladder_config import get_tier_threshold

            # Get current model stats
            current_elo = None
            games_played = 0

            if self.elo_service:
                rating = self.elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played

            # Check against tier threshold
            if target_tier:
                threshold = get_tier_threshold(target_tier)
                if threshold and current_elo is not None:
                    if current_elo >= threshold and games_played >= self.criteria.tier_games_required:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.TIER,
                            should_promote=True,
                            reason=f"Elo {current_elo:.0f} >= tier {target_tier} threshold {threshold}",
                            current_elo=current_elo,
                            games_played=games_played,
                            current_tier=current_tier,
                            target_tier=target_tier,
                        )
                    else:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.TIER,
                            should_promote=False,
                            reason=f"Elo {current_elo or 0:.0f} < tier {target_tier} threshold {threshold}",
                            current_elo=current_elo,
                            games_played=games_played,
                            current_tier=current_tier,
                            target_tier=target_tier,
                        )
        except ImportError as e:
            logger.warning(f"Tier promotion modules not available: {e}")

        return PromotionDecision(
            model_id=model_id,
            promotion_type=PromotionType.TIER,
            should_promote=False,
            reason="Tier promotion evaluation failed - modules not available",
        )

    def _evaluate_champion_promotion(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        baseline_model_id: Optional[str],
        tournament_results: Optional[Dict] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate champion promotion after tournament win."""
        # Use lifecycle manager if available
        if self.lifecycle_manager:
            try:
                from app.integration.model_lifecycle import EvaluationResult

                result = EvaluationResult(
                    model_id=model_id,
                    version=0,
                    games_played=tournament_results.get("games_played", 0) if tournament_results else 0,
                    win_rate=tournament_results.get("win_rate") if tournament_results else None,
                )

                # Use the lifecycle manager's promotion gate
                decision, reason = self.lifecycle_manager.promotion_gate.evaluate_for_staging(result)

                return PromotionDecision(
                    model_id=model_id,
                    promotion_type=PromotionType.CHAMPION,
                    should_promote=decision.value == "promote",
                    reason=reason,
                    games_played=result.games_played,
                    win_rate=result.win_rate,
                )
            except Exception as e:
                logger.warning(f"Lifecycle manager evaluation failed: {e}")

        # Fallback to standard evaluation
        return self._evaluate_standard_promotion(
            model_id, board_type, num_players, PromotionType.CHAMPION, baseline_model_id
        )

    def _evaluate_rollback(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        regression_threshold: Optional[float] = None,
        **kwargs,
    ) -> PromotionDecision:
        """Evaluate whether to rollback to a previous model."""
        threshold = regression_threshold or -30.0

        # Get recent Elo trend
        if self.elo_service:
            try:
                history = self.elo_service.get_rating_history(
                    model_id, board_type, num_players, limit=10
                )
                if len(history) >= 2:
                    recent_elo = history[0].get("rating", 0)
                    older_elo = history[-1].get("rating", 0)
                    elo_change = recent_elo - older_elo

                    if elo_change < threshold:
                        return PromotionDecision(
                            model_id=model_id,
                            promotion_type=PromotionType.ROLLBACK,
                            should_promote=True,
                            reason=f"Elo regression {elo_change:.1f} < threshold {threshold}",
                            current_elo=recent_elo,
                            elo_improvement=elo_change,
                        )
            except Exception as e:
                logger.warning(f"Failed to evaluate rollback: {e}")

        return PromotionDecision(
            model_id=model_id,
            promotion_type=PromotionType.ROLLBACK,
            should_promote=False,
            reason="No significant regression detected",
        )

    def execute_promotion(
        self,
        decision: PromotionDecision,
        dry_run: bool = False,
    ) -> bool:
        """Execute a promotion decision.

        Args:
            decision: The promotion decision to execute
            dry_run: If True, only log what would happen

        Returns:
            True if promotion was successful
        """
        if not decision.should_promote:
            logger.info(f"Skipping promotion for {decision.model_id}: {decision.reason}")
            return False

        if dry_run:
            logger.info(f"[DRY RUN] Would promote {decision.model_id}: {decision.reason}")
            self._emit_execution_metrics(decision, success=True, dry_run=True)
            return True

        success = False
        try:
            if decision.promotion_type == PromotionType.TIER:
                success = self._execute_tier_promotion(decision)
            elif decision.promotion_type in (PromotionType.STAGING, PromotionType.PRODUCTION):
                success = self._execute_stage_promotion(decision)
            elif decision.promotion_type == PromotionType.CHAMPION:
                success = self._execute_champion_promotion(decision)
            elif decision.promotion_type == PromotionType.ROLLBACK:
                success = self._execute_rollback(decision)
            else:
                logger.error(f"Unknown promotion type: {decision.promotion_type}")
                success = False
        except Exception as e:
            logger.error(f"Promotion failed for {decision.model_id}: {e}")
            success = False

        self._emit_execution_metrics(decision, success=success, dry_run=False)
        return success

    def _emit_execution_metrics(
        self,
        decision: PromotionDecision,
        success: bool,
        dry_run: bool,
    ) -> None:
        """Emit Prometheus metrics for a promotion execution."""
        try:
            from app.metrics import record_promotion_execution
            record_promotion_execution(
                promotion_type=decision.promotion_type.value,
                success=success,
                dry_run=dry_run,
            )
        except ImportError:
            pass

    def _execute_stage_promotion(self, decision: PromotionDecision) -> bool:
        """Execute staging or production promotion."""
        if self.model_registry:
            try:
                from app.training.model_registry import ModelStage

                target_stage = (
                    ModelStage.STAGING if decision.promotion_type == PromotionType.STAGING
                    else ModelStage.PRODUCTION
                )

                self.model_registry.promote_model(decision.model_id, target_stage)
                logger.info(f"Promoted {decision.model_id} to {target_stage.value}")
                return True
            except Exception as e:
                logger.error(f"Model registry promotion failed: {e}")
        return False

    def _execute_tier_promotion(self, decision: PromotionDecision) -> bool:
        """Execute tier promotion."""
        try:
            from app.training.tier_promotion_registry import (
                load_square8_two_player_registry,
                save_square8_two_player_registry,
            )

            registry = load_square8_two_player_registry()
            tier = decision.target_tier

            if tier and tier not in registry.get("tiers", {}):
                registry.setdefault("tiers", {})[tier] = {}

            if tier:
                registry["tiers"][tier]["promoted_model"] = decision.model_id
                registry["tiers"][tier]["promoted_at"] = decision.evaluated_at
                registry["tiers"][tier]["elo"] = decision.current_elo

                save_square8_two_player_registry(registry)
                logger.info(f"Promoted {decision.model_id} to tier {tier}")
                return True
        except Exception as e:
            logger.error(f"Tier promotion failed: {e}")
        return False

    def _execute_champion_promotion(self, decision: PromotionDecision) -> bool:
        """Execute champion promotion."""
        # Champion promotion typically goes to production
        return self._execute_stage_promotion(
            PromotionDecision(
                model_id=decision.model_id,
                promotion_type=PromotionType.PRODUCTION,
                should_promote=True,
                reason=f"Champion promotion: {decision.reason}",
                current_elo=decision.current_elo,
                games_played=decision.games_played,
                win_rate=decision.win_rate,
            )
        )

    def _execute_rollback(self, decision: PromotionDecision) -> bool:
        """Execute rollback to previous model."""
        if self.model_registry:
            try:
                # Archive current production model
                from app.training.model_registry import ModelStage

                current_production = self.model_registry.get_production_model()
                if current_production:
                    self.model_registry.promote_model(
                        current_production, ModelStage.ARCHIVED
                    )

                # Restore previous model to production
                self.model_registry.promote_model(
                    decision.model_id, ModelStage.PRODUCTION
                )
                logger.info(f"Rolled back to {decision.model_id}")
                return True
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
        return False


@dataclass
class RollbackCriteria:
    """Criteria for automatic rollback decisions."""
    # Elo regression threshold (negative = regression)
    elo_regression_threshold: float = -30.0
    # Minimum games required before considering rollback
    min_games_for_regression: int = 20
    # Number of consecutive regression checks before triggering rollback
    consecutive_checks_required: int = 3
    # Win rate threshold below which rollback may be considered
    min_win_rate: float = 0.40
    # Time window in seconds for regression detection
    time_window_seconds: int = 3600  # 1 hour


class NotificationHook:
    """Base class for rollback notification hooks.

    Extend this class to implement custom notification handlers.

    Example:
        class SlackHook(NotificationHook):
            def __init__(self, webhook_url: str):
                self.webhook_url = webhook_url

            def on_rollback_triggered(self, event):
                import requests
                requests.post(self.webhook_url, json={
                    "text": f"Rollback triggered: {event.reason}"
                })
    """

    def on_regression_detected(self, model_id: str, status: Dict[str, Any]) -> None:
        """Called when regression is detected but rollback not yet triggered."""
        pass

    def on_at_risk(self, model_id: str, status: Dict[str, Any]) -> None:
        """Called when model enters at-risk state."""
        pass

    def on_rollback_triggered(self, event: "RollbackEvent") -> None:
        """Called when rollback is triggered (before execution)."""
        pass

    def on_rollback_completed(self, event: "RollbackEvent", success: bool) -> None:
        """Called after rollback execution."""
        pass


class LoggingNotificationHook(NotificationHook):
    """Default hook that logs notifications."""

    def __init__(self, logger_name: str = "ringrift.rollback"):
        import logging
        self.logger = logging.getLogger(logger_name)

    def on_regression_detected(self, model_id: str, status: Dict[str, Any]) -> None:
        self.logger.warning(
            f"Regression detected for {model_id}: "
            f"consecutive={status.get('consecutive_regressions', 0)}, "
            f"avg={status.get('avg_regression', 0):.1f}"
        )

    def on_at_risk(self, model_id: str, status: Dict[str, Any]) -> None:
        self.logger.warning(
            f"MODEL AT RISK: {model_id} - "
            f"consecutive regressions: {status.get('consecutive_regressions', 0)}"
        )

    def on_rollback_triggered(self, event: "RollbackEvent") -> None:
        self.logger.critical(
            f"ROLLBACK TRIGGERED: {event.current_model_id} -> {event.rollback_model_id} "
            f"(reason: {event.reason})"
        )

    def on_rollback_completed(self, event: "RollbackEvent", success: bool) -> None:
        if success:
            self.logger.info(
                f"Rollback completed: {event.current_model_id} -> {event.rollback_model_id}"
            )
        else:
            self.logger.error(
                f"Rollback FAILED: {event.current_model_id} -> {event.rollback_model_id}"
            )


class WebhookNotificationHook(NotificationHook):
    """Notification hook that sends webhooks to external services.

    Supports Slack, Discord, and generic webhook endpoints.
    """

    def __init__(
        self,
        webhook_url: str,
        webhook_type: str = "generic",  # "slack", "discord", "generic"
        timeout: int = 10,
    ):
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type
        self.timeout = timeout

    def _send_webhook(self, message: str, level: str = "info") -> bool:
        """Send a webhook notification."""
        try:
            import json
            import urllib.request
            import urllib.error

            if self.webhook_type == "slack":
                # Slack format
                color = {"info": "good", "warning": "warning", "critical": "danger"}.get(level, "good")
                payload = {
                    "attachments": [{
                        "color": color,
                        "text": message,
                        "footer": "RingRift AI Rollback Monitor",
                    }]
                }
            elif self.webhook_type == "discord":
                # Discord format
                color = {"info": 0x00FF00, "warning": 0xFFFF00, "critical": 0xFF0000}.get(level, 0x00FF00)
                payload = {
                    "embeds": [{
                        "description": message,
                        "color": color,
                        "footer": {"text": "RingRift AI Rollback Monitor"},
                    }]
                }
            else:
                # Generic format
                payload = {
                    "message": message,
                    "level": level,
                    "source": "ringrift_rollback_monitor",
                }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            return True

        except Exception as e:
            print(f"[WebhookHook] Failed to send webhook: {e}")
            return False

    def on_at_risk(self, model_id: str, status: Dict[str, Any]) -> None:
        self._send_webhook(
            f"⚠️ Model at risk: `{model_id}` - "
            f"{status.get('consecutive_regressions', 0)} consecutive regressions",
            level="warning",
        )

    def on_rollback_triggered(self, event: "RollbackEvent") -> None:
        self._send_webhook(
            f"🔄 Rollback triggered: `{event.current_model_id}` → `{event.rollback_model_id}`\n"
            f"Reason: {event.reason}",
            level="critical",
        )

    def on_rollback_completed(self, event: "RollbackEvent", success: bool) -> None:
        if success:
            self._send_webhook(
                f"✅ Rollback completed: `{event.current_model_id}` → `{event.rollback_model_id}`",
                level="info",
            )
        else:
            self._send_webhook(
                f"❌ Rollback FAILED: `{event.current_model_id}` → `{event.rollback_model_id}`",
                level="critical",
            )


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    triggered_at: str
    current_model_id: str
    rollback_model_id: str
    reason: str
    elo_regression: Optional[float] = None
    games_played: int = 0
    win_rate: Optional[float] = None
    auto_triggered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered_at": self.triggered_at,
            "current_model_id": self.current_model_id,
            "rollback_model_id": self.rollback_model_id,
            "reason": self.reason,
            "elo_regression": self.elo_regression,
            "games_played": self.games_played,
            "win_rate": self.win_rate,
            "auto_triggered": self.auto_triggered,
        }


class RollbackMonitor:
    """Automated rollback monitoring for promoted models.

    Monitors model performance after promotion and triggers automatic
    rollback if significant regression is detected.

    Usage:
        from app.training.promotion_controller import RollbackMonitor, RollbackCriteria

        monitor = RollbackMonitor(
            criteria=RollbackCriteria(elo_regression_threshold=-25.0)
        )

        # Check if rollback is needed
        should_rollback, event = monitor.check_for_regression(
            model_id="model_v42",
            board_type="square8",
            num_players=2,
            previous_model_id="model_v41",
        )

        if should_rollback:
            success = monitor.execute_rollback(event)
    """

    def __init__(
        self,
        criteria: Optional[RollbackCriteria] = None,
        promotion_controller: Optional[PromotionController] = None,
        notification_hooks: Optional[List[NotificationHook]] = None,
    ):
        self.criteria = criteria or RollbackCriteria()
        self._controller = promotion_controller
        # Track regression history per model: model_id -> list of (timestamp, elo_diff)
        self._regression_history: Dict[str, List[Tuple[str, float]]] = {}
        # Track rollback events
        self._rollback_events: List[RollbackEvent] = []
        # Notification hooks
        self._hooks: List[NotificationHook] = notification_hooks or []
        # Track which models we've already notified about being at-risk (avoid spam)
        self._at_risk_notified: set = set()

    def add_notification_hook(self, hook: NotificationHook) -> None:
        """Add a notification hook."""
        self._hooks.append(hook)

    def _notify_regression_detected(self, model_id: str, status: Dict[str, Any]) -> None:
        """Notify hooks about detected regression."""
        for hook in self._hooks:
            try:
                hook.on_regression_detected(model_id, status)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_at_risk(self, model_id: str, status: Dict[str, Any]) -> None:
        """Notify hooks about model entering at-risk state."""
        if model_id in self._at_risk_notified:
            return  # Already notified
        self._at_risk_notified.add(model_id)
        for hook in self._hooks:
            try:
                hook.on_at_risk(model_id, status)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_rollback_triggered(self, event: RollbackEvent) -> None:
        """Notify hooks about rollback being triggered."""
        for hook in self._hooks:
            try:
                hook.on_rollback_triggered(event)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    def _notify_rollback_completed(self, event: RollbackEvent, success: bool) -> None:
        """Notify hooks about rollback completion."""
        # Clear at-risk notification state for this model
        if success and event.current_model_id in self._at_risk_notified:
            self._at_risk_notified.discard(event.current_model_id)
        for hook in self._hooks:
            try:
                hook.on_rollback_completed(event, success)
            except Exception as e:
                logger.warning(f"Notification hook error: {e}")

    @property
    def controller(self) -> PromotionController:
        """Lazy-load PromotionController."""
        if self._controller is None:
            self._controller = PromotionController()
        return self._controller

    def check_for_regression(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        previous_model_id: Optional[str] = None,
        baseline_elo: Optional[float] = None,
    ) -> Tuple[bool, Optional[RollbackEvent]]:
        """Check if a model has regressed and should be rolled back.

        Args:
            model_id: Current model to check
            board_type: Board type
            num_players: Number of players
            previous_model_id: Model to rollback to if needed
            baseline_elo: Expected Elo at promotion time (for comparison)

        Returns:
            Tuple of (should_rollback, RollbackEvent or None)
        """
        now = datetime.now().isoformat()

        # Get current model stats
        current_elo = None
        games_played = 0
        win_rate = None

        elo_service = self.controller.elo_service
        if elo_service:
            try:
                rating = elo_service.get_rating(model_id, board_type, num_players)
                if rating:
                    current_elo = rating.rating
                    games_played = rating.games_played
                    win_rate = rating.win_rate
            except Exception as e:
                logger.warning(f"Failed to get Elo for {model_id}: {e}")
                return False, None

        # Not enough games to make a decision
        if games_played < self.criteria.min_games_for_regression:
            return False, None

        # Calculate regression from baseline
        elo_regression = None
        if baseline_elo is not None and current_elo is not None:
            elo_regression = current_elo - baseline_elo
        elif previous_model_id and elo_service:
            try:
                prev_rating = elo_service.get_rating(previous_model_id, board_type, num_players)
                if prev_rating and current_elo is not None:
                    elo_regression = current_elo - prev_rating.rating
            except Exception:
                pass

        # Record this check in history
        if model_id not in self._regression_history:
            self._regression_history[model_id] = []

        if elo_regression is not None:
            self._regression_history[model_id].append((now, elo_regression))
            # Keep only recent history
            self._prune_history(model_id)

        # Check regression criteria
        should_rollback = False
        reason = ""

        # Check for severe win rate drop
        if win_rate is not None and win_rate < self.criteria.min_win_rate:
            should_rollback = True
            reason = f"Win rate {win_rate:.2%} below threshold {self.criteria.min_win_rate:.2%}"

        # Check for consecutive regression
        elif self._has_consecutive_regression(model_id):
            should_rollback = True
            avg_regression = self._get_average_regression(model_id)
            reason = f"Consecutive Elo regression detected: avg {avg_regression:.1f}"

        # Check for immediate severe regression
        elif elo_regression is not None and elo_regression < self.criteria.elo_regression_threshold * 2:
            # Severe regression triggers immediate rollback
            should_rollback = True
            reason = f"Severe Elo regression: {elo_regression:.1f}"

        # Get regression status for notifications
        status = self.get_regression_status(model_id)

        # Notify if regression detected but not triggering rollback
        if elo_regression is not None and elo_regression < self.criteria.elo_regression_threshold:
            self._notify_regression_detected(model_id, status)

        # Notify if model is at risk
        if status.get("at_risk"):
            self._notify_at_risk(model_id, status)

        if not should_rollback:
            self._emit_check_metrics(model_id, False, elo_regression)
            return False, None

        # Create rollback event
        rollback_model = previous_model_id or self._get_previous_production_model(model_id)
        if not rollback_model:
            logger.warning(f"No rollback target found for {model_id}")
            return False, None

        event = RollbackEvent(
            triggered_at=now,
            current_model_id=model_id,
            rollback_model_id=rollback_model,
            reason=reason,
            elo_regression=elo_regression,
            games_played=games_played,
            win_rate=win_rate,
            auto_triggered=True,
        )

        # Notify hooks about rollback being triggered
        self._notify_rollback_triggered(event)

        self._emit_check_metrics(model_id, True, elo_regression)
        return True, event

    def check_against_baselines(
        self,
        model_id: str,
        board_type: str = "square8",
        num_players: int = 2,
        baseline_model_ids: Optional[List[str]] = None,
        num_baselines: int = 3,
    ) -> Dict[str, Any]:
        """Compare model against multiple baseline models.

        Args:
            model_id: Current model to check
            board_type: Board type
            num_players: Number of players
            baseline_model_ids: Specific models to compare against (if None, use recent production)
            num_baselines: Number of recent models to compare against if baseline_model_ids not provided

        Returns:
            Dict with comparison results for each baseline
        """
        elo_service = self.controller.elo_service
        if not elo_service:
            return {"error": "No Elo service available"}

        # Get current model's rating
        try:
            current_rating = elo_service.get_rating(model_id, board_type, num_players)
            if not current_rating:
                return {"error": f"No rating found for {model_id}"}
        except Exception as e:
            return {"error": f"Failed to get rating: {e}"}

        current_elo = current_rating.rating

        # Get baseline models
        if baseline_model_ids is None:
            baseline_model_ids = self._get_recent_production_models(model_id, num_baselines)

        if not baseline_model_ids:
            return {
                "model_id": model_id,
                "current_elo": current_elo,
                "baselines": [],
                "summary": "No baseline models found",
            }

        # Compare against each baseline
        comparisons = []
        for baseline_id in baseline_model_ids:
            try:
                baseline_rating = elo_service.get_rating(baseline_id, board_type, num_players)
                if baseline_rating:
                    diff = current_elo - baseline_rating.rating
                    comparisons.append({
                        "baseline_id": baseline_id,
                        "baseline_elo": baseline_rating.rating,
                        "elo_diff": diff,
                        "is_regression": diff < self.criteria.elo_regression_threshold,
                    })
            except Exception:
                comparisons.append({
                    "baseline_id": baseline_id,
                    "error": "Failed to get rating",
                })

        # Calculate summary stats
        valid_diffs = [c["elo_diff"] for c in comparisons if "elo_diff" in c]
        if valid_diffs:
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            min_diff = min(valid_diffs)
            max_diff = max(valid_diffs)
            regressions = sum(1 for d in valid_diffs if d < self.criteria.elo_regression_threshold)
        else:
            avg_diff = min_diff = max_diff = 0.0
            regressions = 0

        summary = "healthy"
        if regressions == len(valid_diffs) and len(valid_diffs) > 0:
            summary = "regression_against_all"
        elif regressions > len(valid_diffs) / 2:
            summary = "regression_against_majority"
        elif regressions > 0:
            summary = "regression_against_some"

        return {
            "model_id": model_id,
            "current_elo": current_elo,
            "games_played": current_rating.games_played,
            "baselines": comparisons,
            "avg_diff": avg_diff,
            "min_diff": min_diff,
            "max_diff": max_diff,
            "regressions": regressions,
            "total_baselines": len(valid_diffs),
            "summary": summary,
        }

    def _get_recent_production_models(
        self,
        exclude_model_id: str,
        count: int = 3,
    ) -> List[str]:
        """Get list of recent production models to use as baselines."""
        registry = self.controller.model_registry
        if not registry:
            return []

        try:
            history = registry.get_model_history(limit=count + 5)
            models = []
            for entry in history:
                model_id = entry.get("model_id")
                if model_id and model_id != exclude_model_id:
                    if entry.get("stage") in ("production", "staging"):
                        models.append(model_id)
                        if len(models) >= count:
                            break
            return models
        except Exception:
            return []

    def _prune_history(self, model_id: str) -> None:
        """Prune old entries from regression history."""
        if model_id not in self._regression_history:
            return

        cutoff_time = datetime.now().timestamp() - self.criteria.time_window_seconds
        history = self._regression_history[model_id]

        # Keep only entries within time window (compare ISO strings)
        cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()
        self._regression_history[model_id] = [
            (ts, val) for ts, val in history if ts >= cutoff_iso
        ]

    def _has_consecutive_regression(self, model_id: str) -> bool:
        """Check if model has consecutive regression checks exceeding threshold."""
        history = self._regression_history.get(model_id, [])
        if len(history) < self.criteria.consecutive_checks_required:
            return False

        # Check last N entries
        recent = history[-self.criteria.consecutive_checks_required:]
        return all(
            val < self.criteria.elo_regression_threshold
            for _, val in recent
        )

    def _get_average_regression(self, model_id: str) -> float:
        """Get average regression from recent history."""
        history = self._regression_history.get(model_id, [])
        if not history:
            return 0.0
        return sum(val for _, val in history) / len(history)

    def _get_previous_production_model(self, current_model_id: str) -> Optional[str]:
        """Get the previous production model to rollback to."""
        registry = self.controller.model_registry
        if not registry:
            return None

        try:
            # Get model history
            history = registry.get_model_history(limit=10)
            found_current = False
            for entry in history:
                if entry.get("model_id") == current_model_id:
                    found_current = True
                    continue
                if found_current and entry.get("stage") in ("production", "staging"):
                    return entry.get("model_id")
        except Exception:
            pass

        return None

    def _emit_check_metrics(
        self,
        model_id: str,
        triggered: bool,
        elo_regression: Optional[float],
    ) -> None:
        """Emit metrics for regression check."""
        try:
            from app.metrics import record_rollback_check
            status = self.get_regression_status(model_id)
            record_rollback_check(
                model_id=model_id,
                triggered=triggered,
                elo_regression=elo_regression or 0.0,
                at_risk=status.get("at_risk", False),
            )
        except ImportError:
            pass

    def execute_rollback(
        self,
        event: RollbackEvent,
        dry_run: bool = False,
    ) -> bool:
        """Execute an automatic rollback.

        Args:
            event: The rollback event to execute
            dry_run: If True, only log what would happen

        Returns:
            True if rollback was successful
        """
        logger.warning(
            f"{'[DRY RUN] ' if dry_run else ''}Executing automatic rollback: "
            f"{event.current_model_id} -> {event.rollback_model_id} "
            f"(reason: {event.reason})"
        )

        if dry_run:
            self._emit_rollback_metrics(event, success=True, dry_run=True)
            return True

        # Create rollback decision
        decision = PromotionDecision(
            model_id=event.rollback_model_id,
            promotion_type=PromotionType.ROLLBACK,
            should_promote=True,
            reason=f"Auto-rollback from {event.current_model_id}: {event.reason}",
            elo_improvement=event.elo_regression,
            games_played=event.games_played,
            win_rate=event.win_rate,
        )

        # Execute via controller
        success = self.controller.execute_promotion(decision)

        if success:
            self._rollback_events.append(event)
            # Clear regression history for the model we rolled back from
            if event.current_model_id in self._regression_history:
                del self._regression_history[event.current_model_id]

        # Notify hooks about rollback completion
        self._notify_rollback_completed(event, success)

        self._emit_rollback_metrics(event, success=success, dry_run=False)
        return success

    def _emit_rollback_metrics(
        self,
        event: RollbackEvent,
        success: bool,
        dry_run: bool,
    ) -> None:
        """Emit metrics for rollback execution."""
        try:
            from app.metrics import record_auto_rollback
            record_auto_rollback(
                from_model=event.current_model_id,
                to_model=event.rollback_model_id,
                success=success,
                dry_run=dry_run,
                reason=event.reason,
            )
        except ImportError:
            pass

    def get_rollback_history(self) -> List[RollbackEvent]:
        """Get history of rollback events."""
        return self._rollback_events.copy()

    def get_regression_status(self, model_id: str) -> Dict[str, Any]:
        """Get current regression status for a model."""
        history = self._regression_history.get(model_id, [])
        if not history:
            return {
                "model_id": model_id,
                "checks": 0,
                "avg_regression": 0.0,
                "consecutive_regressions": 0,
                "at_risk": False,
            }

        # Count consecutive regressions
        consecutive = 0
        for _, val in reversed(history):
            if val < self.criteria.elo_regression_threshold:
                consecutive += 1
            else:
                break

        return {
            "model_id": model_id,
            "checks": len(history),
            "avg_regression": self._get_average_regression(model_id),
            "consecutive_regressions": consecutive,
            "at_risk": consecutive >= self.criteria.consecutive_checks_required - 1,
        }


# Convenience function
def get_promotion_controller(
    criteria: Optional[PromotionCriteria] = None,
) -> PromotionController:
    """Get a configured promotion controller instance."""
    return PromotionController(criteria=criteria)


def get_rollback_monitor(
    criteria: Optional[RollbackCriteria] = None,
) -> RollbackMonitor:
    """Get a configured rollback monitor instance."""
    return RollbackMonitor(criteria=criteria)
