"""Tests for DaemonHealthAnalyzer.

Sprint 17.9+ (Jan 5, 2026): Initial test suite.
"""

import pytest

from app.coordination.daemon_health_types import (
    AnalyzerConfig,
    DaemonFailurePattern,
    FailureCategory,
)
from app.coordination.daemon_health_analyzer import (
    DaemonHealthAnalyzer,
    get_daemon_health_analyzer,
    reset_daemon_health_analyzer,
)


@pytest.fixture(autouse=True)
def reset_analyzer():
    """Reset the singleton analyzer before each test."""
    reset_daemon_health_analyzer()
    yield
    reset_daemon_health_analyzer()


@pytest.fixture
def analyzer():
    """Create a fresh analyzer for each test."""
    return DaemonHealthAnalyzer(AnalyzerConfig())


@pytest.fixture
def healthy_result():
    """Create a healthy HealthCheckResult as dict."""
    return {"healthy": True, "status": "running", "message": "OK"}


@pytest.fixture
def unhealthy_result():
    """Create an unhealthy HealthCheckResult as dict."""
    return {"healthy": False, "status": "error", "message": "Connection timeout"}


class TestFailureCategory:
    """Test FailureCategory enum."""

    def test_category_values(self):
        """Categories should have correct string values."""
        assert FailureCategory.TRANSIENT.value == "transient"
        assert FailureCategory.DEGRADED.value == "degraded"
        assert FailureCategory.PERSISTENT.value == "persistent"
        assert FailureCategory.CRITICAL.value == "critical"

    def test_category_ordering(self):
        """Categories should be ordered by severity."""
        categories = list(FailureCategory)
        assert categories.index(FailureCategory.TRANSIENT) < categories.index(
            FailureCategory.DEGRADED
        )
        assert categories.index(FailureCategory.DEGRADED) < categories.index(
            FailureCategory.PERSISTENT
        )
        assert categories.index(FailureCategory.PERSISTENT) < categories.index(
            FailureCategory.CRITICAL
        )


class TestDaemonFailurePattern:
    """Test DaemonFailurePattern dataclass."""

    def test_initial_state(self):
        """New pattern should have zero failures."""
        pattern = DaemonFailurePattern(daemon_name="test")
        assert pattern.daemon_name == "test"
        assert pattern.consecutive_failures == 0
        assert pattern.total_failures == 0
        assert pattern.total_checks == 0
        assert pattern.failure_rate == 0.0

    def test_record_healthy_check(self):
        """Recording healthy check should reset consecutive failures."""
        pattern = DaemonFailurePattern(daemon_name="test")
        pattern.consecutive_failures = 3
        pattern.record_check(healthy=True)

        assert pattern.consecutive_failures == 0
        assert pattern.total_checks == 1
        assert pattern.last_success_time > 0

    def test_record_unhealthy_check(self):
        """Recording unhealthy check should increment failures."""
        pattern = DaemonFailurePattern(daemon_name="test")
        pattern.record_check(healthy=False, message="error")

        assert pattern.consecutive_failures == 1
        assert pattern.total_failures == 1
        assert pattern.total_checks == 1
        assert pattern.last_failure_time > 0
        assert "error" in pattern.failure_messages

    def test_failure_rate_calculation(self):
        """Failure rate should be calculated correctly."""
        pattern = DaemonFailurePattern(daemon_name="test")
        pattern.record_check(healthy=True)
        pattern.record_check(healthy=False, message="fail")

        assert pattern.failure_rate == 0.5

    def test_recent_failure_count(self):
        """Recent failures should be counted within window."""
        import time

        pattern = DaemonFailurePattern(daemon_name="test")
        # Add 5 failures
        for _ in range(5):
            pattern.record_check(healthy=False)

        # All 5 should be within default 5-minute window
        assert pattern.recent_failure_count(300) == 5

        # With very short window, still 5 (just added)
        assert pattern.recent_failure_count(1) == 5

    def test_message_limit(self):
        """Should keep only last 10 failure messages."""
        pattern = DaemonFailurePattern(daemon_name="test")
        for i in range(15):
            pattern.record_check(healthy=False, message=f"error_{i}")

        assert len(pattern.failure_messages) == 10
        assert pattern.failure_messages[0] == "error_5"
        assert pattern.failure_messages[-1] == "error_14"

    def test_reset(self):
        """Reset should clear all pattern state."""
        pattern = DaemonFailurePattern(daemon_name="test")
        pattern.record_check(healthy=False, message="error")
        pattern.record_check(healthy=True)

        pattern.reset()

        assert pattern.consecutive_failures == 0
        assert pattern.total_failures == 0
        assert pattern.total_checks == 0
        assert len(pattern.failure_messages) == 0


class TestDaemonHealthAnalyzer:
    """Test DaemonHealthAnalyzer class."""

    def test_single_failure_is_transient(self, analyzer, unhealthy_result):
        """Single failure should be classified as transient."""
        unhealthy_result["status"] = "running"  # Not error/stopped
        result = analyzer.analyze("test_daemon", unhealthy_result)

        assert result.category == FailureCategory.TRANSIENT
        assert result.recommended_action == "monitor"

    def test_consecutive_failures_escalate_to_persistent(self, analyzer):
        """3 consecutive failures should escalate to persistent."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}

        for i in range(3):
            result = analyzer.analyze("test_daemon", unhealthy)

        assert result.category == FailureCategory.PERSISTENT
        assert result.recommended_action == "restart_daemon"

    def test_consecutive_failures_escalate_to_critical(self, analyzer):
        """5 consecutive failures should escalate to critical."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}

        for i in range(5):
            result = analyzer.analyze("test_daemon", unhealthy)

        assert result.category == FailureCategory.CRITICAL
        assert result.recommended_action == "escalate_to_recovery"

    def test_success_resets_consecutive_failures(self, analyzer):
        """Success should reset consecutive failure count."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}
        healthy = {"healthy": True, "status": "running", "message": "OK"}

        # 2 failures
        analyzer.analyze("test_daemon", unhealthy)
        analyzer.analyze("test_daemon", unhealthy)

        # 1 success resets
        analyzer.analyze("test_daemon", healthy)

        # Next failure is transient again
        result = analyzer.analyze("test_daemon", unhealthy)
        assert result.category == FailureCategory.TRANSIENT

    def test_error_status_is_critical(self, analyzer):
        """ERROR status should be classified as CRITICAL regardless of count."""
        error_result = {"healthy": False, "status": "error", "message": "Fatal"}

        result = analyzer.analyze("test_daemon", error_result)

        assert result.category == FailureCategory.CRITICAL

    def test_stopped_status_is_critical(self, analyzer):
        """STOPPED status should be classified as CRITICAL."""
        stopped_result = {"healthy": False, "status": "stopped", "message": "Shutdown"}

        result = analyzer.analyze("test_daemon", stopped_result)

        assert result.category == FailureCategory.CRITICAL

    def test_degraded_status_classification(self, analyzer):
        """DEGRADED status should be classified as DEGRADED."""
        degraded_result = {"healthy": False, "status": "degraded", "message": "Partial"}

        result = analyzer.analyze("test_daemon", degraded_result)

        assert result.category == FailureCategory.DEGRADED
        assert result.recommended_action == "log_warning"

    def test_paused_status_is_degraded(self, analyzer):
        """PAUSED status should be classified as DEGRADED."""
        paused_result = {"healthy": False, "status": "paused", "message": "Maintenance"}

        result = analyzer.analyze("test_daemon", paused_result)

        assert result.category == FailureCategory.DEGRADED

    def test_different_daemons_tracked_separately(self, analyzer):
        """Different daemons should have separate failure patterns."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}

        # 3 failures for daemon1 = PERSISTENT
        for _ in range(3):
            analyzer.analyze("daemon1", unhealthy)

        # 1 failure for daemon2 = TRANSIENT
        result = analyzer.analyze("daemon2", unhealthy)

        assert result.category == FailureCategory.TRANSIENT
        assert analyzer.get_pattern("daemon1").consecutive_failures == 3
        assert analyzer.get_pattern("daemon2").consecutive_failures == 1


class TestEventEmission:
    """Test event emission logic."""

    def test_critical_always_emits(self, analyzer):
        """CRITICAL failures should always emit events."""
        error_result = {"healthy": False, "status": "error", "message": "Fatal"}

        result = analyzer.analyze("test", error_result)

        assert result.should_emit_event is True

    def test_transient_no_emit(self, analyzer):
        """Single transient failure should not emit."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}

        result = analyzer.analyze("test", unhealthy)

        assert result.category == FailureCategory.TRANSIENT
        assert result.should_emit_event is False

    def test_escalation_to_persistent_emits(self, analyzer):
        """Escalation to PERSISTENT should emit event."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}

        # First 2 failures don't emit (transient)
        for _ in range(2):
            result = analyzer.analyze("test", unhealthy)
            assert result.should_emit_event is False

        # 3rd failure escalates to PERSISTENT and emits
        result = analyzer.analyze("test", unhealthy)
        assert result.category == FailureCategory.PERSISTENT
        assert result.should_emit_event is True

    def test_recovery_emits_after_persistent(self, analyzer):
        """Recovery from PERSISTENT should emit event."""
        unhealthy = {"healthy": False, "status": "running", "message": "timeout"}
        healthy = {"healthy": True, "status": "running", "message": "OK"}

        # Get to PERSISTENT
        for _ in range(3):
            analyzer.analyze("test", unhealthy)

        # Recovery emits
        result = analyzer.analyze("test", healthy)
        assert result.should_emit_event is True


class TestSingletonPattern:
    """Test singleton accessor functions."""

    def test_get_returns_singleton(self):
        """get_daemon_health_analyzer should return same instance."""
        a1 = get_daemon_health_analyzer()
        a2 = get_daemon_health_analyzer()

        assert a1 is a2

    def test_reset_clears_instance(self):
        """reset_daemon_health_analyzer should clear the singleton."""
        a1 = get_daemon_health_analyzer()
        reset_daemon_health_analyzer()
        a2 = get_daemon_health_analyzer()

        assert a1 is not a2


class TestAnalyzerQueries:
    """Test analyzer query methods."""

    def test_get_pattern_returns_none_for_unknown(self, analyzer):
        """get_pattern should return None for unknown daemon."""
        pattern = analyzer.get_pattern("unknown_daemon")
        assert pattern is None

    def test_get_pattern_returns_existing(self, analyzer):
        """get_pattern should return pattern after analyze."""
        analyzer.analyze("test", {"healthy": False, "message": "error"})

        pattern = analyzer.get_pattern("test")
        assert pattern is not None
        assert pattern.daemon_name == "test"

    def test_get_all_patterns(self, analyzer):
        """get_all_patterns should return all tracked daemons."""
        analyzer.analyze("daemon1", {"healthy": False})
        analyzer.analyze("daemon2", {"healthy": False})

        patterns = analyzer.get_all_patterns()
        assert len(patterns) == 2
        assert "daemon1" in patterns
        assert "daemon2" in patterns

    def test_get_failing_daemons(self, analyzer):
        """get_failing_daemons should filter by category."""
        unhealthy = {"healthy": False, "status": "running"}

        # 1 failure = TRANSIENT
        analyzer.analyze("daemon_transient", unhealthy)

        # 3 failures = PERSISTENT
        for _ in range(3):
            analyzer.analyze("daemon_persistent", unhealthy)

        # Get PERSISTENT+ only
        failing = analyzer.get_failing_daemons(FailureCategory.PERSISTENT)
        assert len(failing) == 1
        assert failing[0][0] == "daemon_persistent"

    def test_get_critical_daemons(self, analyzer):
        """get_critical_daemons should return only CRITICAL daemons."""
        analyzer.analyze("daemon1", {"healthy": False, "status": "error"})
        analyzer.analyze("daemon2", {"healthy": False, "status": "running"})

        critical = analyzer.get_critical_daemons()
        assert critical == ["daemon1"]

    def test_reset_pattern_clears_specific(self, analyzer):
        """reset_pattern should clear only specified daemon."""
        analyzer.analyze("daemon1", {"healthy": False})
        analyzer.analyze("daemon2", {"healthy": False})

        analyzer.reset_pattern("daemon1")

        assert analyzer.get_pattern("daemon1") is None
        assert analyzer.get_pattern("daemon2") is not None

    def test_clear_all_patterns(self, analyzer):
        """clear_all_patterns should remove all patterns."""
        analyzer.analyze("daemon1", {"healthy": False})
        analyzer.analyze("daemon2", {"healthy": False})

        analyzer.clear_all_patterns()

        assert len(analyzer.get_all_patterns()) == 0


class TestAnalyzerConfig:
    """Test AnalyzerConfig configuration."""

    def test_default_thresholds(self):
        """Default config should have expected thresholds."""
        config = AnalyzerConfig()

        assert config.transient_threshold == 1
        assert config.persistent_threshold == 3
        assert config.critical_threshold == 5

    def test_custom_thresholds(self):
        """Custom thresholds should be used."""
        config = AnalyzerConfig(
            persistent_threshold=2,
            critical_threshold=4,
        )
        analyzer = DaemonHealthAnalyzer(config)
        unhealthy = {"healthy": False, "status": "running"}

        # 2 failures should now be PERSISTENT
        analyzer.analyze("test", unhealthy)
        result = analyzer.analyze("test", unhealthy)

        assert result.category == FailureCategory.PERSISTENT

    def test_from_env_defaults(self):
        """from_env should work with no env vars set."""
        config = AnalyzerConfig.from_env()

        assert config.transient_threshold == 1
        assert config.persistent_threshold == 3
        assert config.critical_threshold == 5


class TestAnalysisResultProperties:
    """Test AnalysisResult properties."""

    def test_is_healthy_false_on_failure(self, analyzer):
        """is_healthy should be False when failing."""
        result = analyzer.analyze("test", {"healthy": False})
        assert result.is_healthy is False

    def test_is_healthy_true_on_success(self, analyzer):
        """is_healthy should be True after success."""
        result = analyzer.analyze("test", {"healthy": True})
        assert result.is_healthy is True

    def test_needs_intervention_for_persistent(self, analyzer):
        """needs_intervention should be True for PERSISTENT+."""
        unhealthy = {"healthy": False, "status": "running"}

        for _ in range(3):
            result = analyzer.analyze("test", unhealthy)

        assert result.needs_intervention is True

    def test_needs_intervention_false_for_transient(self, analyzer):
        """needs_intervention should be False for TRANSIENT."""
        result = analyzer.analyze("test", {"healthy": False, "status": "running"})
        assert result.needs_intervention is False

    def test_details_contains_metrics(self, analyzer):
        """Analysis result details should contain useful metrics."""
        result = analyzer.analyze("test", {"healthy": False, "message": "timeout"})

        assert "consecutive_failures" in result.details
        assert "failure_rate" in result.details
        assert "recent_failures" in result.details
        assert result.details["consecutive_failures"] == 1
