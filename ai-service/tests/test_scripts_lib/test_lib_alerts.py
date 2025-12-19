"""Tests for scripts.lib.alerts module."""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from scripts.lib.alerts import (
    AlertSeverity,
    AlertType,
    Alert,
    AlertThresholds,
    AlertManager,
    create_alert,
    check_disk_alert,
    check_memory_alert,
    check_cpu_alert,
    log_handler,
    console_handler,
    file_handler,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_levels(self):
        assert AlertSeverity.DEBUG.level == 0
        assert AlertSeverity.INFO.level == 1
        assert AlertSeverity.WARNING.level == 2
        assert AlertSeverity.ERROR.level == 3
        assert AlertSeverity.CRITICAL.level == 4

    def test_severity_comparison(self):
        assert AlertSeverity.DEBUG < AlertSeverity.INFO
        assert AlertSeverity.WARNING <= AlertSeverity.WARNING
        assert AlertSeverity.CRITICAL > AlertSeverity.ERROR
        assert AlertSeverity.INFO >= AlertSeverity.DEBUG

    def test_severity_values(self):
        assert AlertSeverity.DEBUG.value == "debug"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertType:
    """Tests for AlertType enum."""

    def test_resource_alerts(self):
        assert AlertType.HIGH_DISK_USAGE.value == "high_disk_usage"
        assert AlertType.HIGH_MEMORY_USAGE.value == "high_memory_usage"
        assert AlertType.GPU_IDLE.value == "gpu_idle"

    def test_cluster_alerts(self):
        assert AlertType.LEADER_UNREACHABLE.value == "leader_unreachable"
        assert AlertType.NODE_UNREACHABLE.value == "node_unreachable"

    def test_training_alerts(self):
        assert AlertType.TRAINING_FAILED.value == "training_failed"
        assert AlertType.LOSS_SPIKE.value == "loss_spike"

    def test_data_quality_alerts(self):
        assert AlertType.HIGH_DRAW_RATE.value == "high_draw_rate"
        assert AlertType.GAMES_AT_MOVE_LIMIT.value == "games_at_move_limit"
        assert AlertType.NO_GAMES.value == "no_games"
        assert AlertType.DATABASE_ERROR.value == "database_error"

    def test_elo_alerts(self):
        assert AlertType.ELO_REGRESSION.value == "elo_regression"
        assert AlertType.ELO_STAGNATION.value == "elo_stagnation"
        assert AlertType.MODEL_DEGRADATION.value == "model_degradation"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_create_alert(self):
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.HIGH_DISK_USAGE,
            message="Disk usage high",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == AlertType.HIGH_DISK_USAGE
        assert alert.message == "Disk usage high"
        assert alert.acknowledged is False

    def test_alert_str(self):
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.TRAINING_FAILED,
            message="Training crashed",
        )
        assert str(alert) == "[CRITICAL] Training crashed"

    def test_alert_repr(self):
        alert = Alert(
            severity=AlertSeverity.INFO,
            alert_type=AlertType.THRESHOLD_EXCEEDED,
            message="Test",
        )
        assert "Alert(" in repr(alert)

    def test_alert_to_dict(self):
        alert = Alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.HIGH_MEMORY_USAGE,
            message="Memory high",
            details={"percent": 85.0},
            source="test",
        )
        data = alert.to_dict()

        assert data["severity"] == "warning"
        assert data["alert_type"] == "high_memory_usage"
        assert data["message"] == "Memory high"
        assert data["details"]["percent"] == 85.0
        assert data["source"] == "test"
        assert "timestamp" in data

    def test_alert_from_dict(self):
        data = {
            "severity": "error",
            "alert_type": "training_failed",
            "message": "Training error",
            "timestamp": "2025-12-19T10:00:00",
            "details": {"error": "OOM"},
            "source": "trainer",
        }
        alert = Alert.from_dict(data)

        assert alert.severity == AlertSeverity.ERROR
        assert alert.alert_type == AlertType.TRAINING_FAILED
        assert alert.message == "Training error"
        assert alert.details["error"] == "OOM"

    def test_alert_json_roundtrip(self):
        original = Alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.DATA_CORRUPTION,
            message="Bad data",
            details={"file": "test.npz"},
        )

        json_str = original.to_json()
        restored = Alert.from_json(json_str)

        assert restored.severity == original.severity
        assert restored.alert_type == original.alert_type
        assert restored.message == original.message
        assert restored.details == original.details


class TestCreateAlert:
    """Tests for create_alert convenience function."""

    def test_create_alert_basic(self):
        alert = create_alert(
            AlertSeverity.INFO,
            AlertType.HEALTH_CHECK_FAILED,
            "Health check failed",
        )
        assert alert.severity == AlertSeverity.INFO
        assert alert.message == "Health check failed"

    def test_create_alert_with_details(self):
        alert = create_alert(
            AlertSeverity.WARNING,
            AlertType.GPU_HOT,
            "GPU temperature high",
            details={"temp": 85, "gpu_id": 0},
            source="gpu_monitor",
        )
        assert alert.details["temp"] == 85
        assert alert.source == "gpu_monitor"


class TestAlertThresholds:
    """Tests for AlertThresholds dataclass."""

    def test_default_thresholds(self):
        thresholds = AlertThresholds()
        assert thresholds.disk_warning_percent == 65.0
        assert thresholds.disk_critical_percent == 70.0
        assert thresholds.memory_warning_percent == 80.0
        assert thresholds.memory_critical_percent == 95.0

    def test_custom_thresholds(self):
        thresholds = AlertThresholds(
            disk_warning_percent=50.0,
            memory_critical_percent=90.0,
        )
        assert thresholds.disk_warning_percent == 50.0
        assert thresholds.memory_critical_percent == 90.0


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_add_alert(self):
        manager = AlertManager("test")
        alert = create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",
        )

        added = manager.add_alert(alert)
        assert added is True
        assert len(manager.alerts) == 1

    def test_min_severity_filter(self):
        manager = AlertManager("test", min_severity=AlertSeverity.WARNING)

        # Info should be filtered
        info_alert = create_alert(
            AlertSeverity.INFO,
            AlertType.THRESHOLD_EXCEEDED,
            "Info message",
        )
        assert manager.add_alert(info_alert) is False
        assert len(manager.alerts) == 0

        # Warning should be added
        warn_alert = create_alert(
            AlertSeverity.WARNING,
            AlertType.THRESHOLD_EXCEEDED,
            "Warning message",
        )
        assert manager.add_alert(warn_alert) is True
        assert len(manager.alerts) == 1

    def test_deduplication(self):
        manager = AlertManager("test", dedup_window_seconds=300)

        alert1 = create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",
        )
        alert2 = create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",  # Same message
        )

        assert manager.add_alert(alert1) is True
        assert manager.add_alert(alert2) is False  # Deduplicated
        assert len(manager.alerts) == 1

    def test_flush_clears_pending(self):
        manager = AlertManager("test")
        manager.add_alert(create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",
        ))

        pending = manager.flush()
        assert len(pending) == 1
        assert len(manager.alerts) == 0
        assert len(manager.alert_history) == 1

    def test_add_handler(self):
        manager = AlertManager("test")
        handled_alerts = []

        def test_handler(alert):
            handled_alerts.append(alert)

        manager.add_handler(test_handler)
        manager.add_alert(create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",
        ))
        manager.flush()

        assert len(handled_alerts) == 1

    def test_get_alerts_by_severity(self):
        manager = AlertManager("test", min_severity=AlertSeverity.DEBUG)
        manager.add_alert(create_alert(AlertSeverity.INFO, AlertType.UNKNOWN, "Info"))
        manager.add_alert(create_alert(AlertSeverity.WARNING, AlertType.UNKNOWN, "Warning"))
        manager.add_alert(create_alert(AlertSeverity.CRITICAL, AlertType.UNKNOWN, "Critical"))

        warnings_plus = manager.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warnings_plus) == 2

    def test_get_summary(self):
        manager = AlertManager("test", min_severity=AlertSeverity.DEBUG)
        manager.add_alert(create_alert(AlertSeverity.WARNING, AlertType.HIGH_DISK_USAGE, "Disk 1"))
        manager.add_alert(create_alert(AlertSeverity.WARNING, AlertType.HIGH_MEMORY_USAGE, "Mem 1"))
        manager.add_alert(create_alert(AlertSeverity.CRITICAL, AlertType.TRAINING_FAILED, "Train 1"))

        summary = manager.get_summary()
        assert summary["total_alerts"] == 3
        assert summary["by_severity"]["warning"] == 2
        assert summary["by_severity"]["critical"] == 1


class TestAlertCheckers:
    """Tests for alert checker functions."""

    def test_check_disk_alert_ok(self):
        thresholds = AlertThresholds()
        alert = check_disk_alert(50.0, thresholds)
        assert alert is None

    def test_check_disk_alert_warning(self):
        thresholds = AlertThresholds()
        alert = check_disk_alert(68.0, thresholds)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.alert_type == AlertType.HIGH_DISK_USAGE

    def test_check_disk_alert_critical(self):
        thresholds = AlertThresholds()
        alert = check_disk_alert(75.0, thresholds)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_check_memory_alert_ok(self):
        thresholds = AlertThresholds()
        alert = check_memory_alert(60.0, thresholds)
        assert alert is None

    def test_check_memory_alert_warning(self):
        thresholds = AlertThresholds()
        alert = check_memory_alert(85.0, thresholds)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_check_memory_alert_critical(self):
        thresholds = AlertThresholds()
        alert = check_memory_alert(97.0, thresholds)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_check_cpu_alert_ok(self):
        thresholds = AlertThresholds()
        alert = check_cpu_alert(50.0, thresholds)
        assert alert is None

    def test_check_cpu_alert_warning(self):
        thresholds = AlertThresholds()
        alert = check_cpu_alert(85.0, thresholds)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING


class TestAlertHandlers:
    """Tests for alert handlers."""

    def test_file_handler(self, tmp_path):
        filepath = tmp_path / "alerts.jsonl"
        handler = file_handler(filepath)

        alert = create_alert(
            AlertSeverity.WARNING,
            AlertType.HIGH_DISK_USAGE,
            "Disk high",
        )
        handler(alert)

        assert filepath.exists()
        with open(filepath) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["message"] == "Disk high"

    def test_console_handler(self, capsys):
        alert = create_alert(
            AlertSeverity.CRITICAL,
            AlertType.TRAINING_FAILED,
            "Training failed",
        )
        console_handler(alert)

        captured = capsys.readouterr()
        assert "[CRITICAL]" in captured.out
        assert "Training failed" in captured.out
