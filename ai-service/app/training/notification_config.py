"""Notification hooks configuration loader.

Loads notification hook configurations from YAML and creates the appropriate
hook instances for the RollbackMonitor.

Usage:
    from app.training.notification_config import load_notification_hooks, load_rollback_config

    # Load hooks from default config
    hooks = load_notification_hooks()
    monitor = RollbackMonitor(notification_hooks=hooks)

    # Load full configuration including criteria overrides
    config = load_rollback_config()
    monitor = RollbackMonitor(
        criteria=config.criteria,
        notification_hooks=config.hooks,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.training.promotion_controller import (
    LoggingNotificationHook,
    NotificationHook,
    RollbackCriteria,
    WebhookNotificationHook,
)
from app.utils.yaml_utils import load_config_yaml as _load_yaml_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "notification_hooks.yaml"


@dataclass
class RollbackConfig:
    """Complete rollback configuration from YAML."""
    hooks: list[NotificationHook]
    criteria: RollbackCriteria
    enabled: bool = True


class FilteredWebhookHook(WebhookNotificationHook):
    """Webhook hook that filters events based on configuration."""

    def __init__(
        self,
        webhook_url: str,
        webhook_type: str = "generic",
        timeout: int = 10,
        events: dict[str, bool] | None = None,
    ):
        super().__init__(webhook_url, webhook_type, timeout)
        self.events = events or {
            "at_risk": True,
            "rollback_triggered": True,
            "rollback_completed": True,
        }

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        if self.events.get("at_risk", True):
            super().on_at_risk(model_id, status)

    def on_rollback_triggered(self, event) -> None:
        if self.events.get("rollback_triggered", True):
            super().on_rollback_triggered(event)

    def on_rollback_completed(self, event, success: bool) -> None:
        if self.events.get("rollback_completed", True):
            super().on_rollback_completed(event, success)


class PagerDutyNotificationHook(NotificationHook):
    """Notification hook for PagerDuty Events API v2."""

    PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        severity_mapping: dict[str, str] | None = None,
        timeout: int = 10,
    ):
        self.routing_key = routing_key
        self.timeout = timeout
        self.severity_mapping = severity_mapping or {
            "at_risk": "warning",
            "rollback_triggered": "critical",
            "rollback_completed": "info",
        }

    def _send_event(
        self,
        summary: str,
        severity: str,
        dedup_key: str,
        action: str = "trigger",
        custom_details: dict | None = None,
    ) -> bool:
        """Send an event to PagerDuty."""
        try:
            import json
            import urllib.error
            import urllib.request

            payload = {
                "routing_key": self.routing_key,
                "event_action": action,
                "dedup_key": dedup_key,
                "payload": {
                    "summary": summary,
                    "severity": severity,
                    "source": "ringrift-rollback-monitor",
                    "custom_details": custom_details or {},
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.PAGERDUTY_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            return True

        except Exception as e:
            logger.warning(f"[PagerDuty] Failed to send event: {e}")
            return False

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        self._send_event(
            summary=f"Model {model_id} at risk of rollback - {status.get('consecutive_regressions', 0)} consecutive regressions",
            severity=self.severity_mapping.get("at_risk", "warning"),
            dedup_key=f"ringrift-at-risk-{model_id}",
            custom_details=status,
        )

    def on_rollback_triggered(self, event) -> None:
        self._send_event(
            summary=f"Rollback triggered: {event.current_model_id} -> {event.rollback_model_id}",
            severity=self.severity_mapping.get("rollback_triggered", "critical"),
            dedup_key=f"ringrift-rollback-{event.current_model_id}",
            custom_details=event.to_dict(),
        )

    def on_rollback_completed(self, event, success: bool) -> None:
        action = "resolve" if success else "trigger"
        severity = "info" if success else "critical"
        self._send_event(
            summary=f"Rollback {'completed' if success else 'FAILED'}: {event.current_model_id} -> {event.rollback_model_id}",
            severity=self.severity_mapping.get("rollback_completed", severity),
            dedup_key=f"ringrift-rollback-{event.current_model_id}",
            action=action,
            custom_details={"success": success, **event.to_dict()},
        )


class OpsGenieNotificationHook(NotificationHook):
    """Notification hook for OpsGenie Alerts API."""

    def __init__(
        self,
        api_key: str,
        region: str = "us",
        priority_mapping: dict[str, str] | None = None,
        timeout: int = 10,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.priority_mapping = priority_mapping or {
            "at_risk": "P3",
            "rollback_triggered": "P1",
            "rollback_completed": "P5",
        }
        # OpsGenie API endpoint varies by region
        if region == "eu":
            self.base_url = "https://api.eu.opsgenie.com/v2/alerts"
        else:
            self.base_url = "https://api.opsgenie.com/v2/alerts"

    def _send_alert(
        self,
        message: str,
        priority: str,
        alias: str,
        description: str | None = None,
        details: dict | None = None,
    ) -> bool:
        """Send an alert to OpsGenie."""
        try:
            import json
            import urllib.error
            import urllib.request

            payload = {
                "message": message,
                "alias": alias,
                "priority": priority,
                "source": "ringrift-rollback-monitor",
            }
            if description:
                payload["description"] = description
            if details:
                payload["details"] = details

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.base_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"GenieKey {self.api_key}",
                },
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            return True

        except Exception as e:
            logger.warning(f"[OpsGenie] Failed to send alert: {e}")
            return False

    def _close_alert(self, alias: str) -> bool:
        """Close an alert in OpsGenie."""
        try:
            import json
            import urllib.error
            import urllib.request

            url = f"{self.base_url}/{alias}/close?identifierType=alias"
            data = json.dumps({"source": "ringrift-rollback-monitor"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"GenieKey {self.api_key}",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            return True

        except Exception as e:
            logger.warning(f"[OpsGenie] Failed to close alert: {e}")
            return False

    def on_at_risk(self, model_id: str, status: dict[str, Any]) -> None:
        self._send_alert(
            message=f"Model {model_id} at risk of rollback",
            priority=self.priority_mapping.get("at_risk", "P3"),
            alias=f"ringrift-at-risk-{model_id}",
            description=f"{status.get('consecutive_regressions', 0)} consecutive regressions detected",
            details=status,
        )

    def on_rollback_triggered(self, event) -> None:
        self._send_alert(
            message=f"Rollback triggered: {event.current_model_id} -> {event.rollback_model_id}",
            priority=self.priority_mapping.get("rollback_triggered", "P1"),
            alias=f"ringrift-rollback-{event.current_model_id}",
            description=event.reason,
            details=event.to_dict(),
        )

    def on_rollback_completed(self, event, success: bool) -> None:
        if success:
            # Close the alert on successful rollback
            self._close_alert(f"ringrift-rollback-{event.current_model_id}")
            self._close_alert(f"ringrift-at-risk-{event.current_model_id}")
        else:
            # Escalate on failure
            self._send_alert(
                message=f"Rollback FAILED: {event.current_model_id} -> {event.rollback_model_id}",
                priority="P1",  # Always P1 for failures
                alias=f"ringrift-rollback-failed-{event.current_model_id}",
                description="Manual intervention required",
                details={"success": False, **event.to_dict()},
            )


def load_config_yaml(config_path: Path | None = None) -> dict[str, Any]:
    """Load the notification hooks configuration YAML.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        Dict containing the parsed YAML configuration
    """
    path = config_path or DEFAULT_CONFIG_PATH
    return _load_yaml_config(
        default_path=path,
        env_var="RINGRIFT_NOTIFICATION_CONFIG",
        defaults={"enabled": True, "logging": {"enabled": True}},
    )


def load_notification_hooks(config_path: Path | None = None) -> list[NotificationHook]:
    """Load notification hooks from configuration.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        List of configured NotificationHook instances
    """
    config = load_config_yaml(config_path)
    hooks: list[NotificationHook] = []

    if not config.get("enabled", True):
        logger.info("Notification hooks disabled in config")
        return hooks

    # Logging hook
    logging_config = config.get("logging", {})
    if logging_config.get("enabled", True):
        logger_name = logging_config.get("logger_name", "ringrift.rollback")
        hooks.append(LoggingNotificationHook(logger_name=logger_name))

    # Slack webhook
    slack_config = config.get("slack", {})
    if slack_config.get("enabled") and slack_config.get("webhook_url"):
        hooks.append(FilteredWebhookHook(
            webhook_url=slack_config["webhook_url"],
            webhook_type="slack",
            timeout=slack_config.get("timeout_seconds", 10),
            events=slack_config.get("events"),
        ))

    # Discord webhook
    discord_config = config.get("discord", {})
    if discord_config.get("enabled") and discord_config.get("webhook_url"):
        hooks.append(FilteredWebhookHook(
            webhook_url=discord_config["webhook_url"],
            webhook_type="discord",
            timeout=discord_config.get("timeout_seconds", 10),
            events=discord_config.get("events"),
        ))

    # PagerDuty
    pd_config = config.get("pagerduty", {})
    if pd_config.get("enabled") and pd_config.get("routing_key"):
        hooks.append(PagerDutyNotificationHook(
            routing_key=pd_config["routing_key"],
            severity_mapping=pd_config.get("severity_mapping"),
            timeout=pd_config.get("timeout_seconds", 10),
        ))

    # OpsGenie
    og_config = config.get("opsgenie", {})
    if og_config.get("enabled") and og_config.get("api_key"):
        hooks.append(OpsGenieNotificationHook(
            api_key=og_config["api_key"],
            region=og_config.get("region", "us"),
            priority_mapping=og_config.get("priority_mapping"),
            timeout=og_config.get("timeout_seconds", 10),
        ))

    # Generic webhook
    generic_config = config.get("generic", {})
    if generic_config.get("enabled") and generic_config.get("webhook_url"):
        hooks.append(FilteredWebhookHook(
            webhook_url=generic_config["webhook_url"],
            webhook_type="generic",
            timeout=generic_config.get("timeout_seconds", 10),
            events=generic_config.get("events"),
        ))

    logger.info(f"Loaded {len(hooks)} notification hooks from config")
    return hooks


def load_rollback_criteria(config_path: Path | None = None) -> RollbackCriteria:
    """Load rollback criteria from configuration.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        RollbackCriteria with any overrides from config
    """
    config = load_config_yaml(config_path)
    overrides = config.get("criteria_overrides", {})

    if not overrides:
        return RollbackCriteria()

    return RollbackCriteria(
        elo_regression_threshold=overrides.get("elo_regression_threshold", -30.0),
        min_games_for_regression=overrides.get("min_games_for_regression", 20),
        consecutive_checks_required=overrides.get("consecutive_checks_required", 3),
        min_win_rate=overrides.get("min_win_rate", 0.40),
        time_window_seconds=overrides.get("time_window_seconds", 3600),
        cooldown_seconds=overrides.get("cooldown_seconds", 3600),
        max_rollbacks_per_day=overrides.get("max_rollbacks_per_day", 3),
    )


def load_rollback_config(config_path: Path | None = None) -> RollbackConfig:
    """Load complete rollback configuration including hooks and criteria.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        RollbackConfig with hooks and criteria
    """
    config = load_config_yaml(config_path)
    return RollbackConfig(
        hooks=load_notification_hooks(config_path),
        criteria=load_rollback_criteria(config_path),
        enabled=config.get("enabled", True),
    )
