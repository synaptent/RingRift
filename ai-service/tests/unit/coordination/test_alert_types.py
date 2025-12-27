#!/usr/bin/env python3
"""Tests for app/coordination/alert_types.py.

December 27, 2025: Initial test coverage for unified alert types.
"""

import time
import unittest
from unittest.mock import patch

from app.coordination.alert_types import (
    # Core types
    AlertSeverity,
    AlertCategory,
    AlertState,
    Alert,
    # Factory
    create_alert,
    # Backward-compatible aliases
    AlertLevel,
    ErrorSeverity,
    StallSeverity,
    ReplicationAlertLevel,
    RegressionSeverity,
    ValidationSeverity,
    # Helpers
    severity_to_log_level,
    severity_to_color,
    severity_to_emoji,
)


class TestAlertSeverity(unittest.TestCase):
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test that severity enum has expected values."""
        self.assertEqual(AlertSeverity.DEBUG.value, 0)
        self.assertEqual(AlertSeverity.INFO.value, 1)
        self.assertEqual(AlertSeverity.WARNING.value, 2)
        self.assertEqual(AlertSeverity.ERROR.value, 3)
        self.assertEqual(AlertSeverity.CRITICAL.value, 4)

    def test_severity_ordering(self):
        """Test that severities can be compared."""
        self.assertLess(AlertSeverity.DEBUG, AlertSeverity.INFO)
        self.assertLess(AlertSeverity.INFO, AlertSeverity.WARNING)
        self.assertLess(AlertSeverity.WARNING, AlertSeverity.ERROR)
        self.assertLess(AlertSeverity.ERROR, AlertSeverity.CRITICAL)

    def test_severity_gte_comparison(self):
        """Test >= comparison for threshold checks."""
        self.assertTrue(AlertSeverity.WARNING >= AlertSeverity.INFO)
        self.assertTrue(AlertSeverity.ERROR >= AlertSeverity.WARNING)
        self.assertTrue(AlertSeverity.CRITICAL >= AlertSeverity.ERROR)
        self.assertFalse(AlertSeverity.INFO >= AlertSeverity.WARNING)


class TestAlertCategory(unittest.TestCase):
    """Tests for AlertCategory enum."""

    def test_category_values(self):
        """Test that category enum has expected values."""
        self.assertEqual(AlertCategory.TRAINING.value, "training")
        self.assertEqual(AlertCategory.EVALUATION.value, "evaluation")
        self.assertEqual(AlertCategory.RESOURCE.value, "resource")
        self.assertEqual(AlertCategory.CLUSTER.value, "cluster")
        self.assertEqual(AlertCategory.SYNC.value, "sync")
        self.assertEqual(AlertCategory.QUALITY.value, "quality")
        self.assertEqual(AlertCategory.SYSTEM.value, "system")

    def test_category_is_string(self):
        """Test that category values are strings."""
        for category in AlertCategory:
            self.assertIsInstance(category.value, str)


class TestAlertState(unittest.TestCase):
    """Tests for AlertState enum."""

    def test_state_values(self):
        """Test that state enum has expected values."""
        self.assertEqual(AlertState.ACTIVE.value, "active")
        self.assertEqual(AlertState.ACKNOWLEDGED.value, "acknowledged")
        self.assertEqual(AlertState.RESOLVED.value, "resolved")
        self.assertEqual(AlertState.SUPPRESSED.value, "suppressed")


class TestAlert(unittest.TestCase):
    """Tests for Alert dataclass."""

    def test_alert_creation_minimal(self):
        """Test creating an alert with minimal args."""
        alert = Alert(title="Test", message="Test message")
        self.assertEqual(alert.title, "Test")
        self.assertEqual(alert.message, "Test message")
        self.assertEqual(alert.severity, AlertSeverity.WARNING)  # Default
        self.assertEqual(alert.category, AlertCategory.SYSTEM)  # Default
        self.assertEqual(alert.state, AlertState.ACTIVE)  # Default

    def test_alert_creation_full(self):
        """Test creating an alert with all args."""
        alert = Alert(
            title="High GPU",
            message="GPU at 95%",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.RESOURCE,
            state=AlertState.ACKNOWLEDGED,
            source="gpu_monitor",
            config_key="hex8_2p",
            node_id="worker-1",
            metadata={"gpu_id": 0, "utilization": 95},
        )
        self.assertEqual(alert.title, "High GPU")
        self.assertEqual(alert.severity, AlertSeverity.ERROR)
        self.assertEqual(alert.category, AlertCategory.RESOURCE)
        self.assertEqual(alert.state, AlertState.ACKNOWLEDGED)
        self.assertEqual(alert.source, "gpu_monitor")
        self.assertEqual(alert.config_key, "hex8_2p")
        self.assertEqual(alert.node_id, "worker-1")
        self.assertEqual(alert.metadata["gpu_id"], 0)

    def test_alert_id_generated(self):
        """Test that alert_id is auto-generated."""
        alert1 = Alert(title="Test", message="Test")
        alert2 = Alert(title="Test", message="Test")
        self.assertIsNotNone(alert1.alert_id)
        self.assertIsNotNone(alert2.alert_id)
        self.assertNotEqual(alert1.alert_id, alert2.alert_id)

    def test_alert_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        alert = Alert(title="Test", message="Test")
        after = time.time()
        self.assertGreaterEqual(alert.timestamp, before)
        self.assertLessEqual(alert.timestamp, after)

    def test_is_critical_property(self):
        """Test is_critical property."""
        critical = Alert(title="T", message="M", severity=AlertSeverity.CRITICAL)
        error = Alert(title="T", message="M", severity=AlertSeverity.ERROR)
        warning = Alert(title="T", message="M", severity=AlertSeverity.WARNING)

        self.assertTrue(critical.is_critical)
        self.assertFalse(error.is_critical)
        self.assertFalse(warning.is_critical)

    def test_is_error_or_above_property(self):
        """Test is_error_or_above property."""
        critical = Alert(title="T", message="M", severity=AlertSeverity.CRITICAL)
        error = Alert(title="T", message="M", severity=AlertSeverity.ERROR)
        warning = Alert(title="T", message="M", severity=AlertSeverity.WARNING)
        info = Alert(title="T", message="M", severity=AlertSeverity.INFO)

        self.assertTrue(critical.is_error_or_above)
        self.assertTrue(error.is_error_or_above)
        self.assertFalse(warning.is_error_or_above)
        self.assertFalse(info.is_error_or_above)

    def test_age_seconds_property(self):
        """Test age_seconds property."""
        # Create alert with timestamp 5 seconds ago
        old_time = time.time() - 5.0
        alert = Alert(title="T", message="M", timestamp=old_time)
        age = alert.age_seconds
        self.assertGreaterEqual(age, 4.9)
        self.assertLessEqual(age, 6.0)

    def test_to_dict(self):
        """Test to_dict serialization."""
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRAINING,
            source="test_source",
            metadata={"key": "value"},
        )
        d = alert.to_dict()

        self.assertEqual(d["title"], "Test")
        self.assertEqual(d["message"], "Test message")
        self.assertEqual(d["severity"], "ERROR")
        self.assertEqual(d["severity_value"], 3)
        self.assertEqual(d["category"], "training")
        self.assertEqual(d["state"], "active")
        self.assertEqual(d["source"], "test_source")
        self.assertEqual(d["metadata"], {"key": "value"})
        self.assertIn("alert_id", d)
        self.assertIn("timestamp", d)


class TestCreateAlert(unittest.TestCase):
    """Tests for create_alert factory function."""

    def test_create_alert_basic(self):
        """Test basic alert creation."""
        alert = create_alert(title="Test", message="Test message")
        self.assertEqual(alert.title, "Test")
        self.assertEqual(alert.message, "Test message")
        self.assertEqual(alert.severity, AlertSeverity.WARNING)

    def test_create_alert_with_options(self):
        """Test alert creation with all options."""
        alert = create_alert(
            title="Critical Error",
            message="System failure",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            source="daemon_manager",
            config_key="square8_4p",
            node_id="coordinator",
            extra_field="extra_value",
        )
        self.assertEqual(alert.title, "Critical Error")
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertEqual(alert.category, AlertCategory.SYSTEM)
        self.assertEqual(alert.source, "daemon_manager")
        self.assertEqual(alert.config_key, "square8_4p")
        self.assertEqual(alert.node_id, "coordinator")
        self.assertEqual(alert.metadata["extra_field"], "extra_value")

    def test_create_alert_metadata_kwargs(self):
        """Test that **kwargs become metadata."""
        alert = create_alert(
            title="T",
            message="M",
            job_id="123",
            error_code=500,
            retry_count=3,
        )
        self.assertEqual(alert.metadata["job_id"], "123")
        self.assertEqual(alert.metadata["error_code"], 500)
        self.assertEqual(alert.metadata["retry_count"], 3)


class TestBackwardCompatibleAliases(unittest.TestCase):
    """Tests for backward-compatible severity aliases."""

    def test_alert_level_alias(self):
        """Test that AlertLevel is alias for AlertSeverity."""
        self.assertIs(AlertLevel, AlertSeverity)
        self.assertEqual(AlertLevel.CRITICAL, AlertSeverity.CRITICAL)

    def test_error_severity_conversion(self):
        """Test ErrorSeverity to AlertSeverity conversion."""
        self.assertEqual(
            ErrorSeverity.LOW.to_alert_severity(),
            AlertSeverity.INFO,
        )
        self.assertEqual(
            ErrorSeverity.MEDIUM.to_alert_severity(),
            AlertSeverity.WARNING,
        )
        self.assertEqual(
            ErrorSeverity.HIGH.to_alert_severity(),
            AlertSeverity.ERROR,
        )
        self.assertEqual(
            ErrorSeverity.CRITICAL.to_alert_severity(),
            AlertSeverity.CRITICAL,
        )

    def test_stall_severity_conversion(self):
        """Test StallSeverity to AlertSeverity conversion."""
        self.assertEqual(
            StallSeverity.LOW.to_alert_severity(),
            AlertSeverity.INFO,
        )
        self.assertEqual(
            StallSeverity.MEDIUM.to_alert_severity(),
            AlertSeverity.WARNING,
        )
        self.assertEqual(
            StallSeverity.HIGH.to_alert_severity(),
            AlertSeverity.ERROR,
        )
        self.assertEqual(
            StallSeverity.CRITICAL.to_alert_severity(),
            AlertSeverity.CRITICAL,
        )

    def test_replication_alert_level_conversion(self):
        """Test ReplicationAlertLevel to AlertSeverity conversion."""
        self.assertEqual(
            ReplicationAlertLevel.INFO.to_alert_severity(),
            AlertSeverity.INFO,
        )
        self.assertEqual(
            ReplicationAlertLevel.WARNING.to_alert_severity(),
            AlertSeverity.WARNING,
        )
        self.assertEqual(
            ReplicationAlertLevel.ERROR.to_alert_severity(),
            AlertSeverity.ERROR,
        )
        self.assertEqual(
            ReplicationAlertLevel.CRITICAL.to_alert_severity(),
            AlertSeverity.CRITICAL,
        )

    def test_regression_severity_conversion(self):
        """Test RegressionSeverity to AlertSeverity conversion."""
        self.assertEqual(
            RegressionSeverity.LOW.to_alert_severity(),
            AlertSeverity.INFO,
        )
        self.assertEqual(
            RegressionSeverity.MEDIUM.to_alert_severity(),
            AlertSeverity.WARNING,
        )
        self.assertEqual(
            RegressionSeverity.HIGH.to_alert_severity(),
            AlertSeverity.ERROR,
        )
        self.assertEqual(
            RegressionSeverity.CRITICAL.to_alert_severity(),
            AlertSeverity.CRITICAL,
        )

    def test_validation_severity_conversion(self):
        """Test ValidationSeverity to AlertSeverity conversion."""
        self.assertEqual(
            ValidationSeverity.INFO.to_alert_severity(),
            AlertSeverity.INFO,
        )
        self.assertEqual(
            ValidationSeverity.WARNING.to_alert_severity(),
            AlertSeverity.WARNING,
        )
        self.assertEqual(
            ValidationSeverity.ERROR.to_alert_severity(),
            AlertSeverity.ERROR,
        )
        self.assertEqual(
            ValidationSeverity.CRITICAL.to_alert_severity(),
            AlertSeverity.CRITICAL,
        )


class TestHelperFunctions(unittest.TestCase):
    """Tests for alert helper functions."""

    def test_severity_to_log_level(self):
        """Test severity_to_log_level conversion."""
        self.assertEqual(severity_to_log_level(AlertSeverity.DEBUG), "DEBUG")
        self.assertEqual(severity_to_log_level(AlertSeverity.INFO), "INFO")
        self.assertEqual(severity_to_log_level(AlertSeverity.WARNING), "WARNING")
        self.assertEqual(severity_to_log_level(AlertSeverity.ERROR), "ERROR")
        self.assertEqual(severity_to_log_level(AlertSeverity.CRITICAL), "CRITICAL")

    def test_severity_to_color(self):
        """Test severity_to_color returns ANSI codes."""
        # Just check they're non-empty strings starting with escape
        for severity in AlertSeverity:
            color = severity_to_color(severity)
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith("\033["))

    def test_severity_to_color_distinct(self):
        """Test that different severities have different colors."""
        colors = {severity_to_color(s) for s in AlertSeverity}
        self.assertEqual(len(colors), 5)  # 5 distinct colors

    def test_severity_to_emoji(self):
        """Test severity_to_emoji returns strings."""
        for severity in AlertSeverity:
            emoji = severity_to_emoji(severity)
            self.assertIsInstance(emoji, str)


class TestAlertWorkflows(unittest.TestCase):
    """Integration tests for common alert workflows."""

    def test_create_and_serialize_alert(self):
        """Test creating an alert and serializing it."""
        alert = create_alert(
            title="Training stalled",
            message="No progress for 30 minutes",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRAINING,
            source="training_monitor",
            config_key="hex8_2p",
            duration_minutes=30,
        )

        # Serialize
        d = alert.to_dict()

        # Verify key fields
        self.assertEqual(d["title"], "Training stalled")
        self.assertEqual(d["severity"], "ERROR")
        self.assertEqual(d["category"], "training")
        self.assertEqual(d["metadata"]["duration_minutes"], 30)

    def test_alert_severity_filtering(self):
        """Test filtering alerts by severity."""
        alerts = [
            create_alert("Debug", "D", severity=AlertSeverity.DEBUG),
            create_alert("Info", "I", severity=AlertSeverity.INFO),
            create_alert("Warning", "W", severity=AlertSeverity.WARNING),
            create_alert("Error", "E", severity=AlertSeverity.ERROR),
            create_alert("Critical", "C", severity=AlertSeverity.CRITICAL),
        ]

        # Filter error and above
        errors = [a for a in alerts if a.is_error_or_above]
        self.assertEqual(len(errors), 2)
        self.assertEqual({a.title for a in errors}, {"Error", "Critical"})

        # Filter critical only
        critical = [a for a in alerts if a.is_critical]
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].title, "Critical")

    def test_convert_legacy_severity_to_alert(self):
        """Test converting legacy severity to unified alert."""
        # Simulate legacy code path
        legacy_severity = ErrorSeverity.HIGH

        # Create alert with converted severity
        alert = create_alert(
            title="Legacy alert",
            message="From old system",
            severity=legacy_severity.to_alert_severity(),
        )

        self.assertEqual(alert.severity, AlertSeverity.ERROR)


if __name__ == "__main__":
    unittest.main()
