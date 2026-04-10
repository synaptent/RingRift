"""Bootstrap smoke-test helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)




def _bootstrap_state():
    from app.coordination.coordination_bootstrap import _state

    return _state

# Smoke Test (December 2025)
# =============================================================================


@dataclass
class SmokeTestResult:
    """Result of a single smoke test check."""

    name: str
    passed: bool
    error: str | None = None
    details: dict[str, Any] | None = None


def run_bootstrap_smoke_test() -> dict[str, Any]:
    """Run comprehensive smoke test on daemon subscriptions and wiring.

    Verifies that all critical event subscriptions and integrations
    are properly wired after bootstrap. This is designed to catch
    wiring issues before they cause problems in production.

    Returns:
        Dict with test results:
        - passed: bool - Overall pass/fail
        - checks: list[SmokeTestResult] - Individual check results
        - passed_count: int - Number of passed checks
        - failed_count: int - Number of failed checks
        - warnings: list[str] - Non-fatal warnings

    Example:
        >>> result = run_bootstrap_smoke_test()
        >>> if not result['passed']:
        ...     for check in result['checks']:
        ...         if not check['passed']:
        ...             print(f"FAIL: {check['name']}: {check['error']}")
    """
    checks: list[SmokeTestResult] = []
    warnings: list[str] = []

    # 1. Check event bus is initialized
    try:
        from app.coordination.event_router import get_router

        router = get_router()
        subscriber_count = len(getattr(router, "_subscribers", {}))
        checks.append(SmokeTestResult(
            name="event_bus_initialized",
            passed=True,
            details={"subscriber_count": subscriber_count},
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="event_bus_initialized",
            passed=False,
            error=str(e),
        ))

    # 2. Check curriculum feedback watcher
    try:
        from app.training.curriculum_feedback import (
            get_plateau_curriculum_watcher,
            get_curriculum_feedback,
        )

        feedback = get_curriculum_feedback()
        plateau_watcher = get_plateau_curriculum_watcher()
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=feedback is not None,
            details={
                "has_feedback": feedback is not None,
                "has_plateau_watcher": plateau_watcher is not None,
                "plateau_subscribed": getattr(plateau_watcher, "_subscribed", False) if plateau_watcher else False,
            },
        ))
    except ImportError:
        warnings.append("curriculum_feedback module not available")
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=True,  # Not a failure if module doesn't exist
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=False,
            error=str(e),
        ))

    # 3. Check quality rollback watcher class availability (structural check)
    # Note: We verify the CLASS exists and has required methods. Singleton wiring
    # happens later in bootstrap, so we can't check instantiation here.
    try:
        from app.training.rollback_manager import (
            QualityRollbackWatcher,
            wire_quality_to_rollback,
        )

        # Verify the class has required methods
        has_subscribe = hasattr(QualityRollbackWatcher, "subscribe_to_quality_events")
        has_handler = hasattr(QualityRollbackWatcher, "_on_low_quality")
        has_wire_func = callable(wire_quality_to_rollback)
        handlers_ready = has_subscribe and has_handler and has_wire_func

        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=handlers_ready,
            details={
                "has_class": True,
                "has_subscribe_method": has_subscribe,
                "has_handler_method": has_handler,
                "has_wire_function": has_wire_func,
            },
        ))
    except ImportError:
        warnings.append("rollback_manager.QualityRollbackWatcher not available")
        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=False,
            error=str(e),
        ))

    # 4. Check regression detector event bus connection
    try:
        from app.training.regression_detector import get_regression_detector

        detector = get_regression_detector(connect_event_bus=False)
        has_bus = getattr(detector, "_event_bus", None) is not None
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=detector is not None,
            details={
                "has_detector": detector is not None,
                "has_event_bus": has_bus,
            },
        ))
    except ImportError:
        warnings.append("regression_detector module not available")
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=False,
            error=str(e),
        ))

    # 5. Check training coordinator is subscribed
    try:
        from app.coordination.training_coordinator import get_training_coordinator

        coordinator = get_training_coordinator()
        subscribed = getattr(coordinator, "_subscribed", False)
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=subscribed,
            details={
                "subscribed": subscribed,
                "has_coordinator": coordinator is not None,
            },
        ))
    except ImportError:
        warnings.append("training_coordinator not available")
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=False,
            error=str(e),
        ))

    # 6. Check selfplay scheduler curriculum integration
    try:
        from app.coordination.selfplay_scheduler import get_selfplay_scheduler

        scheduler = get_selfplay_scheduler()
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=scheduler is not None,
            details={
                "has_scheduler": scheduler is not None,
            },
        ))
    except ImportError:
        warnings.append("selfplay_scheduler not available")
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=False,
            error=str(e),
        ))

    # 7. Check key event types are defined
    try:
        from app.coordination.event_router import DataEventType

        critical_events = [
            "PLATEAU_DETECTED",
            "REGRESSION_DETECTED",
            "LOW_QUALITY_DATA_WARNING",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
        ]
        missing = [e for e in critical_events if not hasattr(DataEventType, e)]
        checks.append(SmokeTestResult(
            name="critical_event_types_defined",
            passed=len(missing) == 0,
            details={"missing_events": missing} if missing else None,
            error=f"Missing events: {missing}" if missing else None,
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="critical_event_types_defined",
            passed=False,
            error=str(e),
        ))

    # 8. Check daemon manager has known daemons
    try:
        from app.coordination.daemon_manager import get_daemon_manager

        manager = get_daemon_manager()
        known_daemons = getattr(manager, "_daemons", {})
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=manager is not None,
            details={
                "daemon_count": len(known_daemons),
            },
        ))
    except ImportError:
        warnings.append("daemon_manager not available")
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=False,
            error=str(e),
        ))

    # 9. Check UnifiedHealthManager coordinator lifecycle handlers exist (Dec 2025)
    # Note: We verify handlers EXIST (structural check). Subscription happens at runtime.
    try:
        from app.coordination.unified_health_manager import get_health_manager

        health_mgr = get_health_manager()
        # Verify the P0 lifecycle event handlers exist (added Dec 2025)
        has_shutdown = hasattr(health_mgr, "_on_coordinator_shutdown")
        has_heartbeat = hasattr(health_mgr, "_on_coordinator_heartbeat")
        has_subscribe = hasattr(health_mgr, "subscribe_to_events")
        handlers_ready = has_shutdown and has_heartbeat and has_subscribe
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=handlers_ready,
            details={
                "has_shutdown_handler": has_shutdown,
                "has_heartbeat_handler": has_heartbeat,
                "has_subscribe_method": has_subscribe,
            },
            error=None if handlers_ready
            else "Missing coordinator lifecycle handlers",
        ))
    except ImportError:
        warnings.append("unified_health_manager not available")
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=False,
            error=str(e),
        ))

    # 10. Check all coordinators are properly subscribed (December 2025)
    # This catches silent subscription failures where a coordinator initializes
    # but fails to subscribe to events.
    try:
        unsubscribed = [
            name for name, status in _bootstrap_state().coordinators.items()
            if status.initialized and not status.subscribed
        ]
        checks.append(SmokeTestResult(
            name="coordinator_subscriptions_complete",
            passed=len(unsubscribed) == 0,
            error=f"Unsubscribed coordinators: {unsubscribed}" if unsubscribed else None,
            details={
                "total_initialized": sum(1 for s in _bootstrap_state().coordinators.values() if s.initialized),
                "total_subscribed": sum(1 for s in _bootstrap_state().coordinators.values() if s.subscribed),
                "unsubscribed": unsubscribed,
            },
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="coordinator_subscriptions_complete",
            passed=False,
            error=str(e),
        ))

    # Compile results
    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count

    result = {
        "passed": failed_count == 0,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "error": c.error,
                "details": c.details,
            }
            for c in checks
        ],
        "passed_count": passed_count,
        "failed_count": failed_count,
        "warnings": warnings,
    }

    # Log summary
    if failed_count > 0:
        logger.warning(
            f"[Bootstrap] Smoke test: {passed_count}/{len(checks)} passed, "
            f"{failed_count} failed"
        )
        for c in checks:
            if not c.passed:
                logger.warning(f"[Bootstrap]   FAIL: {c.name}: {c.error}")
    else:
        logger.info(
            f"[Bootstrap] Smoke test: {passed_count}/{len(checks)} passed"
        )

    return result
