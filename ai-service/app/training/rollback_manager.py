"""Model Rollback Manager.

Provides automated and manual rollback capabilities for production models.
Monitors model performance and can trigger automatic rollbacks when
performance degrades below thresholds.

Features:
- Manual rollback to any previous version
- Automatic rollback on performance degradation
- Rollback history tracking
- Integration with model registry
- Prometheus metrics for rollback events

Usage:
    from app.training.rollback_manager import RollbackManager

    manager = RollbackManager(registry)

    # Manual rollback
    result = manager.rollback_model("square8_2p", reason="Performance issues")

    # Check if rollback is needed
    if manager.should_rollback("square8_2p"):
        manager.rollback_model("square8_2p", reason="Auto-rollback: Elo degraded")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.paths import DATA_DIR

if TYPE_CHECKING:
    from app.training.regression_detector import RegressionEvent

logger = logging.getLogger(__name__)

ROLLBACK_HISTORY_PATH = DATA_DIR / "rollback_history.json"


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    model_id: str
    from_version: int
    to_version: int
    reason: str
    triggered_by: str  # "manual", "auto_elo", "auto_error", etc.
    timestamp: str
    from_metrics: dict[str, Any] = field(default_factory=dict)
    to_metrics: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RollbackEvent:
        return cls(**d)


@dataclass
class RollbackThresholds:
    """Thresholds for automatic rollback triggers."""
    elo_drop_threshold: float = 50.0  # Elo points drop to trigger rollback
    elo_drop_window_hours: float = 24.0  # Time window to measure drop
    win_rate_drop_threshold: float = 0.10  # 10% win rate drop
    error_rate_threshold: float = 0.05  # 5% error rate
    min_games_for_evaluation: int = 50  # Minimum games before evaluating


class RollbackManager:
    """Manages model rollbacks with automatic detection and history tracking."""

    def __init__(
        self,
        registry,  # ModelRegistry instance
        thresholds: RollbackThresholds | None = None,
        history_path: Path | None = None,
    ):
        self.registry = registry
        self.thresholds = thresholds or RollbackThresholds()
        self.history_path = history_path or ROLLBACK_HISTORY_PATH
        self._history: list[RollbackEvent] = []
        self._load_history()

        # Performance baseline cache (model_id -> metrics snapshot)
        self._baselines: dict[str, dict[str, Any]] = {}

    def _load_history(self):
        """Load rollback history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    data = json.load(f)
                    self._history = [RollbackEvent.from_dict(e) for e in data]
            except Exception as e:
                logger.warning(f"Failed to load rollback history: {e}")

    def _save_history(self):
        """Save rollback history to disk."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump([e.to_dict() for e in self._history], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save rollback history: {e}")

    def set_baseline(self, model_id: str, metrics: dict[str, Any]):
        """Set performance baseline for a model.

        Call this after promoting a model to production to establish
        the expected performance level.
        """
        self._baselines[model_id] = {
            "timestamp": datetime.now().isoformat(),
            "elo": metrics.get("elo"),
            "win_rate": metrics.get("win_rate"),
            "games_played": metrics.get("games_played", 0),
        }
        logger.info(f"Set baseline for {model_id}: Elo={metrics.get('elo')}")

    def check_performance(
        self,
        model_id: str,
        current_metrics: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if current performance is acceptable.

        Returns:
            Tuple of (is_degraded, reason)
        """
        baseline = self._baselines.get(model_id)
        if not baseline:
            return False, "No baseline set"

        # Need minimum games to evaluate
        current_games = current_metrics.get("games_played", 0)
        baseline_games = baseline.get("games_played", 0)
        games_since_baseline = current_games - baseline_games

        if games_since_baseline < self.thresholds.min_games_for_evaluation:
            return False, f"Insufficient games ({games_since_baseline})"

        # Check Elo drop
        baseline_elo = baseline.get("elo")
        current_elo = current_metrics.get("elo")

        if baseline_elo and current_elo:
            elo_drop = baseline_elo - current_elo
            if elo_drop >= self.thresholds.elo_drop_threshold:
                return True, f"Elo dropped by {elo_drop:.0f} (threshold: {self.thresholds.elo_drop_threshold})"

        # Check win rate drop
        baseline_wr = baseline.get("win_rate")
        current_wr = current_metrics.get("win_rate")

        if baseline_wr and current_wr:
            wr_drop = baseline_wr - current_wr
            if wr_drop >= self.thresholds.win_rate_drop_threshold:
                return True, f"Win rate dropped by {wr_drop:.1%} (threshold: {self.thresholds.win_rate_drop_threshold:.1%})"

        return False, "Performance within acceptable range"

    def should_rollback(
        self,
        model_id: str,
        current_metrics: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if a rollback should be triggered for the model.

        Args:
            model_id: The model to check
            current_metrics: Current performance metrics (if not provided,
                           will be fetched from registry)

        Returns:
            Tuple of (should_rollback, reason)
        """
        # Get current production model
        prod_model = self.registry.get_production_model()
        if not prod_model or prod_model.model_id != model_id:
            return False, "Model not in production"

        # Get current metrics
        if current_metrics is None:
            current_metrics = prod_model.metrics.to_dict() if prod_model.metrics else {}

        # Check for rollback candidates (archived versions)
        from app.training.model_registry import ModelStage
        archived = self.registry.list_models(stage=ModelStage.ARCHIVED)
        candidates = [m for m in archived if m["model_id"] == model_id]

        if not candidates:
            return False, "No rollback candidates available"

        # Check performance degradation
        is_degraded, reason = self.check_performance(model_id, current_metrics)
        if is_degraded:
            return True, reason

        return False, "No rollback needed"

    def get_rollback_candidate(self, model_id: str) -> dict[str, Any] | None:
        """Get the best candidate for rolling back to.

        Returns the most recent archived version with good metrics.
        """
        from app.training.model_registry import ModelStage

        archived = self.registry.list_models(stage=ModelStage.ARCHIVED)
        candidates = [m for m in archived if m["model_id"] == model_id]

        if not candidates:
            return None

        # Sort by version descending (most recent first)
        candidates.sort(key=lambda x: x["version"], reverse=True)

        # Prefer candidates with good metrics
        for candidate in candidates:
            metrics = candidate.get("metrics", {})
            elo = metrics.get("elo")
            if elo and elo > 0:
                return candidate

        # Fall back to most recent
        return candidates[0] if candidates else None

    def rollback_model(
        self,
        model_id: str,
        to_version: int | None = None,
        reason: str = "Manual rollback",
        triggered_by: str = "manual",
    ) -> dict[str, Any]:
        """Execute a rollback to a previous version.

        Args:
            model_id: Model to rollback
            to_version: Specific version to rollback to (None = auto-select)
            reason: Reason for rollback
            triggered_by: What triggered the rollback

        Returns:
            Result dict with success status and details
        """
        from app.training.model_registry import ModelStage

        result = {
            "success": False,
            "model_id": model_id,
            "reason": reason,
        }

        # Get current production model
        prod_model = self.registry.get_production_model()
        if not prod_model:
            result["error"] = "No production model found"
            return result

        if prod_model.model_id != model_id:
            # Check if any production model matches
            all_prod = self.registry.list_models(stage=ModelStage.PRODUCTION)
            prod_match = next((m for m in all_prod if m["model_id"] == model_id), None)
            if not prod_match:
                result["error"] = f"Model {model_id} is not in production"
                return result
            from_version = prod_match["version"]
            from_metrics = prod_match.get("metrics", {})
        else:
            from_version = prod_model.version
            from_metrics = prod_model.metrics.to_dict() if prod_model.metrics else {}

        # Find rollback target
        if to_version:
            target = self.registry.get_model(model_id, to_version)
            if not target:
                result["error"] = f"Version {to_version} not found"
                return result
            target_dict = {
                "model_id": target.model_id,
                "version": target.version,
                "metrics": target.metrics.to_dict() if target.metrics else {},
            }
        else:
            target_dict = self.get_rollback_candidate(model_id)
            if not target_dict:
                result["error"] = "No rollback candidate found"
                return result

        target_version = target_dict["version"]
        target_metrics = target_dict.get("metrics", {})

        logger.info(
            f"Rolling back {model_id} from v{from_version} to v{target_version}: {reason}"
        )

        try:
            # Execute rollback through registry promotions
            # First: restore archived model to development
            target_model = self.registry.get_model(model_id, target_version)
            if target_model.stage == ModelStage.ARCHIVED:
                self.registry.promote(
                    model_id,
                    target_version,
                    ModelStage.DEVELOPMENT,
                    reason="Rollback: restoring from archive",
                    promoted_by="rollback_manager",
                )

            # Then: move through staging
            self.registry.promote(
                model_id,
                target_version,
                ModelStage.STAGING,
                reason="Rollback: staging for production",
                promoted_by="rollback_manager",
            )

            # Finally: promote to production (archives current prod)
            self.registry.promote(
                model_id,
                target_version,
                ModelStage.PRODUCTION,
                reason=f"Rollback: {reason}",
                promoted_by="rollback_manager",
            )

            # Record rollback event
            event = RollbackEvent(
                model_id=model_id,
                from_version=from_version,
                to_version=target_version,
                reason=reason,
                triggered_by=triggered_by,
                timestamp=datetime.now().isoformat(),
                from_metrics=from_metrics,
                to_metrics=target_metrics,
                success=True,
            )
            self._history.append(event)
            self._save_history()

            # Clear baseline after rollback
            if model_id in self._baselines:
                del self._baselines[model_id]

            # Emit Prometheus metric if available
            self._emit_rollback_metric(model_id, triggered_by)

            result["success"] = True
            result["from_version"] = from_version
            result["to_version"] = target_version
            result["from_metrics"] = from_metrics
            result["to_metrics"] = target_metrics

            logger.info(f"Rollback successful: {model_id} v{from_version} -> v{target_version}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Rollback failed: {error_msg}")

            # Record failed rollback
            event = RollbackEvent(
                model_id=model_id,
                from_version=from_version,
                to_version=target_version,
                reason=reason,
                triggered_by=triggered_by,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=error_msg,
            )
            self._history.append(event)
            self._save_history()

            result["error"] = error_msg

        return result

    def _emit_rollback_metric(self, model_id: str, triggered_by: str):
        """Emit Prometheus metric for rollback event."""
        try:
            from prometheus_client import REGISTRY

            # Try to get existing metric or create new
            metric_name = "ringrift_model_rollbacks_total"
            try:
                metric = REGISTRY._names_to_collectors.get(metric_name)
                if metric:
                    metric.labels(model_id=model_id, trigger=triggered_by).inc()
            except Exception:
                pass
        except ImportError:
            pass

    def get_rollback_history(
        self,
        model_id: str | None = None,
        limit: int = 20,
    ) -> list[RollbackEvent]:
        """Get rollback history, optionally filtered by model."""
        history = self._history
        if model_id:
            history = [e for e in history if e.model_id == model_id]

        # Return most recent first
        return sorted(history, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_rollback_stats(self) -> dict[str, Any]:
        """Get statistics about rollbacks."""
        total = len(self._history)
        successful = sum(1 for e in self._history if e.success)
        failed = total - successful

        by_trigger = {}
        for event in self._history:
            trigger = event.triggered_by
            by_trigger[trigger] = by_trigger.get(trigger, 0) + 1

        recent_24h = sum(
            1 for e in self._history
            if (datetime.now() - datetime.fromisoformat(e.timestamp)).total_seconds() < 86400
        )

        return {
            "total_rollbacks": total,
            "successful": successful,
            "failed": failed,
            "by_trigger": by_trigger,
            "recent_24h": recent_24h,
        }


def create_rollback_alert_rules() -> str:
    """Generate Prometheus alerting rules for rollback monitoring.

    Returns YAML content for alert rules.
    """
    return """
groups:
  - name: model_rollback_alerts
    rules:
      - alert: ModelRollbackTriggered
        expr: increase(ringrift_model_rollbacks_total[1h]) > 0
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "Model rollback triggered"
          description: "A model rollback was triggered for {{ $labels.model_id }} (trigger: {{ $labels.trigger }})"

      - alert: MultipleRollbacksDetected
        expr: increase(ringrift_model_rollbacks_total[24h]) > 2
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Multiple rollbacks in 24 hours"
          description: "More than 2 rollbacks have occurred in the last 24 hours, indicating potential model stability issues"

      - alert: EloDegradation
        expr: (ringrift_production_model_elo - ringrift_production_model_baseline_elo) < -50
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Production model Elo degradation"
          description: "Production model {{ $labels.model_id }} Elo has dropped by more than 50 points from baseline"
"""


# =============================================================================
# Auto-Rollback Handler (December 2025)
# Wires RegressionDetector events to RollbackManager actions
# =============================================================================

class AutoRollbackHandler:
    """Automatic rollback handler that responds to regression events.

    Implements the RegressionListener protocol from regression_detector.py.
    Automatically triggers rollbacks for SEVERE and CRITICAL regressions.

    Usage:
        from app.training.rollback_manager import AutoRollbackHandler, RollbackManager
        from app.training.regression_detector import get_regression_detector

        rollback_mgr = RollbackManager(registry)
        auto_handler = AutoRollbackHandler(rollback_mgr)

        detector = get_regression_detector()
        detector.add_listener(auto_handler)

        # Or use the convenience function:
        from app.training.rollback_manager import wire_regression_to_rollback
        wire_regression_to_rollback(registry)
    """

    def __init__(
        self,
        rollback_manager: RollbackManager,
        auto_rollback_enabled: bool = True,
        require_approval_for_severe: bool = True,
    ):
        """Initialize the auto-rollback handler.

        Args:
            rollback_manager: RollbackManager instance to use for rollbacks
            auto_rollback_enabled: If True, automatically trigger rollbacks
            require_approval_for_severe: If True, only auto-rollback CRITICAL,
                log warning for SEVERE (default True for safety)
        """
        self.rollback_manager = rollback_manager
        self.auto_rollback_enabled = auto_rollback_enabled
        self.require_approval_for_severe = require_approval_for_severe
        self._pending_rollbacks: dict[str, dict[str, Any]] = {}

    def on_regression(self, event: RegressionEvent) -> None:
        """Handle regression event from RegressionDetector.

        Implements the RegressionListener protocol.
        """
        from app.training.regression_detector import RegressionSeverity

        model_id = event.model_id
        severity = event.severity

        logger.warning(
            f"[AutoRollbackHandler] Received {severity.name} regression for {model_id}: "
            f"{event.reason}"
        )

        if not self.auto_rollback_enabled:
            logger.info("[AutoRollbackHandler] Auto-rollback disabled, logging only")
            return

        if severity == RegressionSeverity.CRITICAL:
            # CRITICAL: Immediate auto-rollback
            logger.warning(
                f"[AutoRollbackHandler] CRITICAL regression - triggering auto-rollback for {model_id}"
            )
            self._execute_rollback(
                model_id,
                reason=f"Auto-rollback: CRITICAL regression - {event.reason}",
                triggered_by="auto_regression_critical",
            )

        elif severity == RegressionSeverity.SEVERE:
            if self.require_approval_for_severe:
                # Log pending rollback for manual approval
                logger.warning(
                    f"[AutoRollbackHandler] SEVERE regression detected for {model_id}. "
                    f"Manual approval required. Use pending_rollback({model_id!r}) to execute."
                )
                self._pending_rollbacks[model_id] = {
                    "event": event.to_dict(),
                    "timestamp": event.timestamp,
                    "reason": event.reason,
                }
            else:
                # Auto-rollback SEVERE if approval not required
                logger.warning(
                    f"[AutoRollbackHandler] SEVERE regression - triggering auto-rollback for {model_id}"
                )
                self._execute_rollback(
                    model_id,
                    reason=f"Auto-rollback: SEVERE regression - {event.reason}",
                    triggered_by="auto_regression_severe",
                )

        elif severity == RegressionSeverity.MODERATE:
            logger.info(
                f"[AutoRollbackHandler] MODERATE regression for {model_id} - monitoring"
            )
            # Track but don't act

        # MINOR regressions are logged by the detector but not acted upon here

    # =========================================================================
    # Event Bus Subscription (December 2025)
    # =========================================================================

    def subscribe_to_regression_events(self) -> bool:
        """Subscribe to REGRESSION_DETECTED events from the event bus.

        This allows the handler to receive regression events directly from
        the event bus in addition to the listener pattern.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_event)
            bus.subscribe(DataEventType.REGRESSION_SEVERE, self._on_regression_event)
            bus.subscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_event)
            # Also subscribe to ELO_SIGNIFICANT_CHANGE for early detection (December 2025)
            bus.subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)
            self._event_subscribed = True
            logger.info("[AutoRollbackHandler] Subscribed to regression and Elo events")
            return True
        except Exception as e:
            logger.warning(f"[AutoRollbackHandler] Failed to subscribe to events: {e}")
            return False

    def unsubscribe_from_regression_events(self) -> None:
        """Unsubscribe from regression events."""
        if not getattr(self, '_event_subscribed', False):
            return

        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_event)
            bus.unsubscribe(DataEventType.REGRESSION_SEVERE, self._on_regression_event)
            bus.unsubscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_event)
            self._event_subscribed = False
        except Exception:
            pass

    def _on_regression_event(self, event) -> None:
        """Handle REGRESSION_DETECTED event from event bus.

        Converts the event bus event to the RegressionEvent format and
        delegates to on_regression().
        """
        from app.training.regression_detector import RegressionSeverity

        payload = event.payload if hasattr(event, 'payload') else {}

        model_id = payload.get("model_id", "")
        if not model_id:
            return

        # Determine severity from event type or payload
        event_type = event.event_type.value if hasattr(event, 'event_type') else ""
        severity_str = payload.get("severity", "moderate")

        if "critical" in event_type.lower() or severity_str.lower() == "critical":
            severity = RegressionSeverity.CRITICAL
        elif "severe" in event_type.lower() or severity_str.lower() == "severe":
            severity = RegressionSeverity.SEVERE
        elif severity_str.lower() == "moderate":
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR

        # Build a pseudo RegressionEvent for on_regression
        class EventBusRegressionEvent:
            def __init__(self, model_id, severity, reason, timestamp, payload):
                self.model_id = model_id
                self.severity = severity
                self.reason = reason
                self.timestamp = timestamp
                self._payload = payload

            def to_dict(self):
                return self._payload

        reason = payload.get("reason", payload.get("message", "Regression detected via event bus"))
        timestamp = payload.get("timestamp", event.timestamp if hasattr(event, 'timestamp') else 0)

        pseudo_event = EventBusRegressionEvent(
            model_id=model_id,
            severity=severity,
            reason=reason,
            timestamp=timestamp,
            payload=payload,
        )

        self.on_regression(pseudo_event)

        # Phase 21.2 (December 2025): Emit SELFPLAY_TARGET_UPDATED with exploration boost
        # After rollback, diversify selfplay to recover from regression
        self._emit_regression_recovery_selfplay(model_id, severity, payload)

    def _emit_regression_recovery_selfplay(
        self, model_id: str, severity: str, payload: dict[str, Any]
    ) -> None:
        """Emit SELFPLAY_TARGET_UPDATED with exploration boost after regression.

        Phase 21.2: When a regression is detected and handled, we need to:
        1. Increase selfplay target to generate more diverse training data
        2. Apply exploration boost to temperature scheduling
        3. Signal the feedback loop to prioritize this config

        This closes the loop: REGRESSION_DETECTED → exploration boost → better data → recovery
        """
        try:
            from app.distributed.data_events import emit_selfplay_target_updated

            config_key = payload.get("config_key", "")
            if not config_key:
                # Try to infer from model_id (e.g., "hex8_2p_v15" -> "hex8_2p")
                parts = model_id.rsplit("_v", 1)
                config_key = parts[0] if parts else model_id

            # Calculate boost based on severity
            boost_factor = self._get_exploration_boost_factor(severity)
            base_target = payload.get("current_target", 1000)
            new_target = int(base_target * boost_factor)

            logger.info(
                f"[AutoRollbackHandler] Emitting regression recovery selfplay: "
                f"{config_key} target={new_target} (boost={boost_factor}x), "
                f"exploration_boost={boost_factor}"
            )

            # Emit with exploration boost metadata
            import asyncio
            asyncio.create_task(
                emit_selfplay_target_updated(
                    config_key=config_key,
                    target_games=new_target,
                    reason=f"regression_recovery_{severity}",
                    priority=1,  # High priority for regression recovery
                    source="rollback_manager.py",
                    exploration_boost=boost_factor,
                    recovery_mode=True,
                )
            )

        except ImportError:
            logger.debug("[AutoRollbackHandler] data_events not available, skipping selfplay emission")
        except Exception as e:
            logger.warning(f"[AutoRollbackHandler] Failed to emit regression recovery selfplay: {e}")

    def _get_exploration_boost_factor(self, severity: str) -> float:
        """Get exploration boost factor based on regression severity.

        Args:
            severity: One of 'critical', 'major', 'moderate', 'minor'

        Returns:
            Boost factor for selfplay target and exploration temperature
        """
        severity_boosts = {
            "critical": 2.0,  # Double selfplay, max exploration
            "major": 1.75,
            "moderate": 1.5,
            "minor": 1.25,
        }
        return severity_boosts.get(severity, 1.5)

    def _on_elo_significant_change(self, event) -> None:
        """Handle ELO_SIGNIFICANT_CHANGE event for early regression detection (December 2025).

        Large negative Elo changes may indicate a problem before formal regression
        detection triggers. This allows earlier warnings.
        """
        payload = event.payload if hasattr(event, 'payload') else {}
        model_id = payload.get("model_id", "")
        elo_delta = payload.get("elo_delta", 0.0)

        if not model_id:
            return

        # Only react to significant drops (>50 Elo loss)
        if elo_delta < -50:
            logger.warning(
                f"[AutoRollbackHandler] Significant Elo drop detected: "
                f"{model_id} {elo_delta:.1f} Elo, monitoring for regression"
            )
            # Track for potential rollback consideration
            if model_id not in self._pending_rollbacks:
                self._pending_rollbacks[model_id] = {
                    "severity": "elo_drop",
                    "reason": f"Significant Elo drop: {elo_delta:.1f}",
                    "elo_delta": elo_delta,
                    "detected_at": payload.get("timestamp", ""),
                    "auto_approved": False,  # Requires confirmation for Elo-based rollback
                }

    def _execute_rollback(self, model_id: str, reason: str, triggered_by: str) -> bool:
        """Execute a rollback via the RollbackManager."""
        try:
            result = self.rollback_manager.rollback_model(
                model_id=model_id,
                reason=reason,
                triggered_by=triggered_by,
            )

            if result.get("success"):
                logger.info(
                    f"[AutoRollbackHandler] Rollback successful for {model_id}: "
                    f"v{result.get('from_version')} -> v{result.get('to_version')}"
                )
                # Clear any pending rollback for this model
                self._pending_rollbacks.pop(model_id, None)

                # Emit PROMOTION_ROLLED_BACK event to pause training (December 2025)
                self._emit_rollback_completed_event(model_id, result, reason, triggered_by)

                return True
            else:
                logger.error(
                    f"[AutoRollbackHandler] Rollback failed for {model_id}: {result.get('error')}"
                )
                return False

        except Exception as e:
            logger.error(f"[AutoRollbackHandler] Rollback exception for {model_id}: {e}")
            return False

    def _emit_rollback_completed_event(
        self, model_id: str, result: dict[str, Any], reason: str, triggered_by: str
    ) -> None:
        """Emit PROMOTION_ROLLED_BACK event after successful rollback (December 2025).

        This event triggers:
        1. TrainingCoordinator to pause training for this config
        2. SelfplayOrchestrator to pause selfplay for this config
        3. Metrics/alerting systems to notify of rollback

        Args:
            model_id: Model that was rolled back (e.g., "hex8_2p")
            result: Rollback result dict with from_version, to_version, metrics
            reason: Reason for rollback
            triggered_by: What triggered the rollback (auto/manual)
        """
        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            if not bus:
                logger.debug("[AutoRollbackHandler] No event bus available for rollback event")
                return

            # Extract config_key from model_id (e.g., "hex8_2p_v15" -> "hex8_2p")
            config_key = model_id
            if "_v" in model_id:
                config_key = model_id.rsplit("_v", 1)[0]

            payload = {
                "model_id": model_id,
                "config_key": config_key,
                "from_version": result.get("from_version"),
                "to_version": result.get("to_version"),
                "from_metrics": result.get("from_metrics", {}),
                "to_metrics": result.get("to_metrics", {}),
                "reason": reason,
                "triggered_by": triggered_by,
                "timestamp": datetime.now().isoformat(),
                "action_required": "pause_training",  # Signal to pause training
            }

            # Use async publish if event loop is running
            try:
                import asyncio
                from app.utils.async_utils import fire_and_forget

                asyncio.get_running_loop()
                fire_and_forget(
                    bus.publish(
                        event_type=DataEventType.PROMOTION_ROLLED_BACK,
                        payload=payload,
                        source="auto_rollback_handler",
                    ),
                    name="rollback_completed",
                )
                logger.info(
                    f"[AutoRollbackHandler] Emitted PROMOTION_ROLLED_BACK for {config_key} "
                    f"(v{result.get('from_version')} → v{result.get('to_version')})"
                )
            except RuntimeError:
                # No running event loop - use sync if available
                if hasattr(bus, 'publish_sync'):
                    from app.distributed.data_events import DataEvent

                    sync_event = DataEvent(
                        event_type=DataEventType.PROMOTION_ROLLED_BACK,
                        payload=payload,
                        source="auto_rollback_handler",
                    )
                    bus.publish_sync(sync_event)
                    logger.info(
                        f"[AutoRollbackHandler] Emitted PROMOTION_ROLLED_BACK (sync) for {config_key}"
                    )

        except ImportError as e:
            logger.debug(f"[AutoRollbackHandler] Event bus not available: {e}")
        except Exception as e:
            logger.warning(f"[AutoRollbackHandler] Failed to emit rollback event: {e}")

    def approve_pending_rollback(self, model_id: str) -> dict[str, Any]:
        """Approve and execute a pending rollback.

        Args:
            model_id: Model ID to rollback

        Returns:
            Rollback result dict
        """
        if model_id not in self._pending_rollbacks:
            return {"success": False, "error": f"No pending rollback for {model_id}"}

        pending = self._pending_rollbacks[model_id]
        reason = pending.get("reason", "SEVERE regression - manual approval")

        result = self.rollback_manager.rollback_model(
            model_id=model_id,
            reason=f"Approved rollback: {reason}",
            triggered_by="auto_regression_approved",
        )

        if result.get("success"):
            self._pending_rollbacks.pop(model_id, None)

        return result

    def get_pending_rollbacks(self) -> dict[str, dict[str, Any]]:
        """Get all pending rollbacks awaiting approval."""
        return self._pending_rollbacks.copy()

    def clear_pending_rollback(self, model_id: str) -> bool:
        """Clear a pending rollback without executing it."""
        if model_id in self._pending_rollbacks:
            del self._pending_rollbacks[model_id]
            logger.info(f"[AutoRollbackHandler] Cleared pending rollback for {model_id}")
            return True
        return False


# Singleton and wiring functions
_auto_handler: AutoRollbackHandler | None = None


def wire_regression_to_rollback(
    registry,
    auto_rollback_enabled: bool = True,
    require_approval_for_severe: bool = True,
    subscribe_to_events: bool = True,
) -> AutoRollbackHandler:
    """Wire RegressionDetector to RollbackManager for automatic rollbacks.

    This is the main entry point for connecting regression detection to
    automatic rollback execution. Supports both the listener pattern (direct
    callback) and event bus subscription.

    Args:
        registry: ModelRegistry instance
        auto_rollback_enabled: Enable automatic rollbacks
        require_approval_for_severe: Require manual approval for SEVERE regressions
        subscribe_to_events: Also subscribe to REGRESSION_DETECTED events (default True)

    Returns:
        AutoRollbackHandler instance
    """
    global _auto_handler

    try:
        from app.training.regression_detector import get_regression_detector

        # Create managers
        rollback_mgr = RollbackManager(registry)
        _auto_handler = AutoRollbackHandler(
            rollback_mgr,
            auto_rollback_enabled=auto_rollback_enabled,
            require_approval_for_severe=require_approval_for_severe,
        )

        # Wire to regression detector (listener pattern)
        detector = get_regression_detector()
        detector.add_listener(_auto_handler)

        # Also subscribe to event bus events (December 2025)
        if subscribe_to_events:
            _auto_handler.subscribe_to_regression_events()

        logger.info(
            f"[wire_regression_to_rollback] Regression detector wired to rollback manager "
            f"(auto_enabled={auto_rollback_enabled}, require_approval_severe={require_approval_for_severe}, "
            f"event_bus={subscribe_to_events})"
        )

        return _auto_handler

    except Exception as e:
        logger.error(f"[wire_regression_to_rollback] Failed to wire: {e}")
        raise


def get_auto_rollback_handler() -> AutoRollbackHandler | None:
    """Get the global auto-rollback handler if configured."""
    return _auto_handler


# =============================================================================
# Quality-to-Rollback Wiring (December 2025)
# =============================================================================

class QualityRollbackWatcher:
    """Watches for low quality events and triggers model rollback.

    Subscribes to LOW_QUALITY_DATA_WARNING events and triggers automatic
    rollback when data quality drops below critical threshold.

    This prevents training on poisoned data from degraded selfplay.
    """

    # Quality thresholds
    CRITICAL_QUALITY_THRESHOLD = 0.3  # Below this triggers rollback
    WARNING_QUALITY_THRESHOLD = 0.5  # Below this triggers warning
    SUSTAINED_LOW_QUALITY_MINUTES = 30  # Must be low for this long

    def __init__(
        self,
        rollback_manager: RollbackManager,
        critical_threshold: float = CRITICAL_QUALITY_THRESHOLD,
        sustained_minutes: float = SUSTAINED_LOW_QUALITY_MINUTES,
    ):
        """Initialize the quality rollback watcher.

        Args:
            rollback_manager: RollbackManager instance for rollback execution
            critical_threshold: Quality score below which triggers rollback
            sustained_minutes: Minutes quality must be low before rollback
        """
        self.rollback_manager = rollback_manager
        self.critical_threshold = critical_threshold
        self.sustained_minutes = sustained_minutes

        # Track low quality duration per config
        self._low_quality_start: dict[str, float] = {}
        self._subscribed = False
        self._rollbacks_triggered = 0

    def subscribe_to_quality_events(self) -> bool:
        """Subscribe to LOW_QUALITY_DATA_WARNING events.

        Returns:
            True if subscription succeeded
        """
        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            if bus is None:
                logger.debug("[QualityRollbackWatcher] Event bus not available")
                return False

            bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality)
            bus.subscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_quality_recovered)
            self._subscribed = True
            logger.info("[QualityRollbackWatcher] Subscribed to quality events")
            return True

        except ImportError as e:
            logger.debug(f"[QualityRollbackWatcher] Import failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"[QualityRollbackWatcher] Failed to subscribe: {e}")
            return False

    def _on_low_quality(self, event) -> None:
        """Handle LOW_QUALITY_DATA_WARNING event."""
        try:
            # Extract event data
            if hasattr(event, 'payload'):
                data = event.payload
            elif isinstance(event, dict):
                data = event
            else:
                data = {}

            quality_score = data.get("quality_score", 1.0)
            config_key = data.get("config_key", "all")  # May be per-config or global
            timestamp = data.get("timestamp", time.time())

            logger.warning(
                f"[QualityRollbackWatcher] Low quality detected: {quality_score:.3f} for {config_key}"
            )

            # Check if critically low
            if quality_score < self.critical_threshold:
                # Start or update tracking
                if config_key not in self._low_quality_start:
                    self._low_quality_start[config_key] = timestamp
                    logger.warning(
                        f"[QualityRollbackWatcher] Critical quality {quality_score:.3f} - "
                        f"monitoring for {self.sustained_minutes} minutes before rollback"
                    )
                else:
                    # Check if sustained long enough
                    start_time = self._low_quality_start[config_key]
                    duration_minutes = (timestamp - start_time) / 60.0

                    if duration_minutes >= self.sustained_minutes:
                        # Trigger rollback
                        self._trigger_quality_rollback(config_key, quality_score, duration_minutes)

        except Exception as e:
            logger.error(f"[QualityRollbackWatcher] Error handling low quality event: {e}")

    def _on_quality_recovered(self, event) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE event."""
        try:
            if hasattr(event, 'payload'):
                data = event.payload
            elif isinstance(event, dict):
                data = event
            else:
                data = {}

            config_key = data.get("config_key", "all")

            # Clear low quality tracking
            if config_key in self._low_quality_start:
                del self._low_quality_start[config_key]
                logger.info(f"[QualityRollbackWatcher] Quality recovered for {config_key}")

        except Exception as e:
            logger.debug(f"[QualityRollbackWatcher] Error handling quality recovery: {e}")

    def _trigger_quality_rollback(
        self,
        config_key: str,
        quality_score: float,
        duration_minutes: float,
    ) -> None:
        """Trigger model rollback due to sustained low quality.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            quality_score: Current quality score
            duration_minutes: How long quality has been low
        """
        logger.warning(
            f"[QualityRollbackWatcher] Triggering rollback for {config_key}: "
            f"quality={quality_score:.3f} sustained for {duration_minutes:.1f} minutes"
        )

        try:
            # Trigger rollback via RollbackManager
            result = self.rollback_manager.rollback_model(
                model_id=config_key,
                reason=f"Sustained low quality ({quality_score:.3f}) for {duration_minutes:.1f} minutes",
            )

            if result.success:
                logger.info(
                    f"[QualityRollbackWatcher] Rollback succeeded: {config_key} "
                    f"v{result.from_version} -> v{result.to_version}"
                )
                self._rollbacks_triggered += 1
            else:
                logger.error(
                    f"[QualityRollbackWatcher] Rollback failed: {result.error_message}"
                )

            # Clear tracking
            if config_key in self._low_quality_start:
                del self._low_quality_start[config_key]

            # Emit rollback event
            self._emit_rollback_event(config_key, quality_score, result)

        except Exception as e:
            logger.error(f"[QualityRollbackWatcher] Rollback execution failed: {e}")

    def _emit_rollback_event(self, config_key: str, quality_score: float, result) -> None:
        """Emit a quality-triggered rollback event."""
        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType, DataEvent

            bus = get_event_bus()
            if bus:
                event = DataEvent(
                    event_type=DataEventType.PROMOTION_ROLLED_BACK,
                    payload={
                        "model_id": config_key,
                        "reason": "quality_degradation",
                        "quality_score": quality_score,
                        "from_version": getattr(result, "from_version", None),
                        "to_version": getattr(result, "to_version", None),
                        "success": getattr(result, "success", False),
                    },
                    source="quality_rollback_watcher",
                )
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(bus.publish(event))
                except RuntimeError:
                    pass
        except Exception as e:
            logger.debug(f"Failed to emit rollback event: {e}")

    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return {
            "subscribed": self._subscribed,
            "rollbacks_triggered": self._rollbacks_triggered,
            "configs_being_monitored": list(self._low_quality_start.keys()),
            "critical_threshold": self.critical_threshold,
            "sustained_minutes": self.sustained_minutes,
        }


# Singleton for quality watcher
_quality_rollback_watcher: QualityRollbackWatcher | None = None


def wire_quality_to_rollback(
    registry,
    critical_threshold: float = 0.3,
    sustained_minutes: float = 30.0,
) -> QualityRollbackWatcher:
    """Wire quality events to automatic rollback.

    Subscribes to LOW_QUALITY_DATA_WARNING events and triggers model
    rollback when data quality is critically low for sustained period.

    Args:
        registry: ModelRegistry instance
        critical_threshold: Quality score below which triggers rollback
        sustained_minutes: Minutes quality must be low before rollback

    Returns:
        QualityRollbackWatcher instance

    Usage:
        from app.training.rollback_manager import wire_quality_to_rollback

        watcher = wire_quality_to_rollback(registry)
    """
    global _quality_rollback_watcher

    rollback_mgr = RollbackManager(registry)
    _quality_rollback_watcher = QualityRollbackWatcher(
        rollback_manager=rollback_mgr,
        critical_threshold=critical_threshold,
        sustained_minutes=sustained_minutes,
    )
    _quality_rollback_watcher.subscribe_to_quality_events()

    logger.info(
        f"[wire_quality_to_rollback] Quality events wired to rollback "
        f"(threshold={critical_threshold}, sustained={sustained_minutes}min)"
    )

    return _quality_rollback_watcher


def get_quality_rollback_watcher() -> QualityRollbackWatcher | None:
    """Get the global quality rollback watcher if configured."""
    return _quality_rollback_watcher
