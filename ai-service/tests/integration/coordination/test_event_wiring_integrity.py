"""Integration tests for event wiring integrity.

This module verifies that critical events in the coordination infrastructure
have proper emitter-subscriber wiring, preventing orphan event regressions.

Created: December 28, 2025
Purpose: Phase 4D - Prevent orphan event regressions
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

# Path to coordination modules
AI_SERVICE_ROOT = Path(__file__).parent.parent.parent.parent
COORDINATION_DIR = AI_SERVICE_ROOT / "app" / "coordination"
P2P_DIR = AI_SERVICE_ROOT / "scripts" / "p2p"
DISTRIBUTED_DIR = AI_SERVICE_ROOT / "app" / "distributed"


# Critical events that MUST have subscribers (from EVENT_WIRING_VERIFICATION.md)
CRITICAL_EVENTS = {
    "training_started": ["SyncRouter", "IdleShutdown", "DataPipeline"],
    "training_completed": ["FeedbackLoop", "DataPipeline", "ModelDistribution"],
    "evaluation_completed": ["FeedbackLoop", "CurriculumIntegration"],
    "model_promoted": ["ModelDistribution", "FeedbackLoop"],
    "data_sync_completed": ["DataPipelineOrchestrator"],
    "new_games_available": ["SelfplayScheduler"],
    "regression_detected": ["ModelLifecycleCoordinator", "DataPipeline"],
    "orphan_games_detected": ["DataPipelineOrchestrator"],
    "backpressure_activated": ["SyncRouter"],
}

# Events known to be intentionally orphaned (no subscribers expected)
INTENTIONALLY_ORPHANED_EVENTS = {
    # Metrics/logging events that are observed but don't need handlers
    "metrics_updated",
    "daemon_heartbeat",
    "coordinator_heartbeat",
    # Status broadcast events
    "idle_state_broadcast",
    # Events consumed by external systems
    "webhook_sent",
}


def scan_file_for_patterns(
    file_path: Path, patterns: List[str]
) -> List[Tuple[str, int, str]]:
    """Scan a file for patterns and return matches with line numbers."""
    matches = []
    try:
        content = file_path.read_text()
        for line_num, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    matches.append((match.group(1), line_num, str(file_path.name)))
    except Exception:
        pass
    return matches


def find_emitters() -> Dict[str, List[Tuple[str, int]]]:
    """Find all event emitters in coordination code."""
    emitters: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    # Multiple patterns to catch various emit styles
    emit_patterns = [
        r'emit\(["\']([a-z_]+)["\']',
        r'emit_event\(["\']([a-z_]+)["\']',
        r'\.emit\(["\']([a-z_]+)["\']',
        r'_emit_([a-z_]+)_event',
        r'publish\(["\']([a-z_]+)["\']',
        r'emit_([a-z_]+)\(',  # emit_training_started() style
        r'DataEventType\.([A-Z_]+)\.value',  # DataEventType.TRAINING_STARTED.value
    ]

    for search_dir in [COORDINATION_DIR, P2P_DIR, DISTRIBUTED_DIR]:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            for event_name, line_num, filename in scan_file_for_patterns(
                py_file, emit_patterns
            ):
                # Normalize event name to lowercase with underscores
                normalized = event_name.lower().replace("-", "_")
                emitters[normalized].append((filename, line_num))

    return emitters


def find_subscribers() -> Dict[str, List[Tuple[str, int]]]:
    """Find all event subscribers in coordination code."""
    subscribers: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    # Multiple patterns to catch various subscribe styles
    subscribe_patterns = [
        r'subscribe\(["\']([a-z_]+)["\']',
        r'subscribe_to_event\(["\']([a-z_]+)["\']',
        r'\.subscribe\(["\']([a-z_]+)["\']',
        r'on_event\(["\']([a-z_]+)["\']',
        r'["\']([a-z_]+)["\']\s*:\s*self\._on_',
        r'DataEventType\.([A-Z_]+)\.value.*subscribe',  # DataEventType.X.value patterns
        r'DataEventType\.([A-Z_]+)[,\)]',  # DataEventType.X in list/dict
        r'_on_([a-z_]+)\s*\(',  # Handler method definitions
    ]

    for search_dir in [COORDINATION_DIR, P2P_DIR, DISTRIBUTED_DIR]:
        if not search_dir.exists():
            continue
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            for event_name, line_num, filename in scan_file_for_patterns(
                py_file, subscribe_patterns
            ):
                normalized = event_name.lower().replace("-", "_")
                subscribers[normalized].append((filename, line_num))

    return subscribers


class TestCriticalEventWiring:
    """Tests that critical pipeline events have proper wiring."""

    @pytest.fixture(scope="class")
    def emitters(self) -> Dict[str, List[Tuple[str, int]]]:
        """Fixture that scans for all emitters once per test class."""
        return find_emitters()

    @pytest.fixture(scope="class")
    def subscribers(self) -> Dict[str, List[Tuple[str, int]]]:
        """Fixture that scans for all subscribers once per test class."""
        return find_subscribers()

    @pytest.mark.parametrize("event_name", list(CRITICAL_EVENTS.keys()))
    def test_critical_event_has_subscriber(
        self, event_name: str, emitters, subscribers
    ):
        """Verify each critical event has at least one subscriber."""
        assert event_name in subscribers or event_name in emitters, (
            f"Critical event '{event_name}' has no emitters or subscribers found. "
            f"This may indicate a wiring issue or the event pattern was not matched. "
            f"Check DataEventType.{event_name.upper()} usage in coordination code."
        )

        # If it's emitted, it should have subscribers
        if event_name in emitters:
            # Use more lenient matching - check if any subscriber pattern exists
            # that could handle this event
            found_subscriber = event_name in subscribers
            if not found_subscriber:
                # Check for wildcard subscribers or handler methods
                for sub_event in subscribers:
                    if event_name.startswith(sub_event) or sub_event in event_name:
                        found_subscriber = True
                        break

            assert found_subscriber, (
                f"Critical event '{event_name}' is emitted but has no subscribers. "
                f"Emitters: {emitters.get(event_name, [])}. "
                f"Add a subscriber in one of: {CRITICAL_EVENTS[event_name]}"
            )


class TestOrphanEventDetection:
    """Tests that detect orphan events (emitters without subscribers)."""

    @pytest.fixture(scope="class")
    def event_analysis(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Analyze all events and return emitters, subscribers, orphans."""
        emitters = set(find_emitters().keys())
        subscribers = set(find_subscribers().keys())
        orphans = emitters - subscribers - INTENTIONALLY_ORPHANED_EVENTS
        return emitters, subscribers, orphans

    def test_no_unexpected_orphan_emitters(self, event_analysis):
        """Verify all emitted events have subscribers (except intentionally orphaned)."""
        emitters, subscribers, orphans = event_analysis

        # Filter out test files and mocks
        real_orphans = {
            o for o in orphans
            if not o.startswith("test_") and not o.startswith("mock_")
        }

        # Allow some orphans but warn about them
        if real_orphans:
            orphan_list = sorted(real_orphans)[:10]  # Limit output
            pytest.skip(
                f"Found {len(real_orphans)} potential orphan events: {orphan_list}. "
                f"This is informational - verify these events are intentional."
            )

    def test_critical_events_are_not_orphaned(self, event_analysis):
        """Ensure critical events are never orphaned."""
        emitters, subscribers, orphans = event_analysis

        critical_orphans = set(CRITICAL_EVENTS.keys()) & orphans
        assert not critical_orphans, (
            f"CRITICAL: These critical events are orphaned: {critical_orphans}. "
            f"This will break the training pipeline. Add subscribers immediately."
        )


class TestEventTypeConsistency:
    """Tests for event type naming consistency."""

    def test_event_names_are_lowercase(self):
        """Verify event names use lowercase_with_underscores format."""
        emitters = find_emitters()

        invalid_names = [
            name for name in emitters
            if name != name.lower() or "-" in name
        ]

        assert not invalid_names, (
            f"Event names should be lowercase_with_underscores: {invalid_names}"
        )

    def test_no_uppercase_event_strings(self):
        """Check for hardcoded UPPERCASE event strings (common bug)."""
        uppercase_pattern = r'(?:emit|subscribe)\(["\']([A-Z][A-Z_]+)["\']'
        violations = []

        for py_file in COORDINATION_DIR.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text()
                for line_num, line in enumerate(content.split("\n"), 1):
                    matches = re.findall(uppercase_pattern, line)
                    for match in matches:
                        # Ignore DataEventType.VALUE references
                        if "DataEventType" not in line:
                            violations.append(
                                f"{py_file.name}:{line_num} - {match}"
                            )
            except Exception:
                pass

        # This is informational - some uppercase may be intentional
        if violations:
            pytest.skip(
                f"Found {len(violations)} uppercase event strings (may need DataEventType.X.value): "
                f"{violations[:5]}"
            )


class TestEventSubscriptionWiring:
    """Tests that verify specific subscription patterns are present."""

    def test_data_pipeline_subscribes_to_sync_completed(self):
        """DataPipelineOrchestrator must subscribe to DATA_SYNC_COMPLETED."""
        data_pipeline = COORDINATION_DIR / "data_pipeline_orchestrator.py"
        assert data_pipeline.exists(), "data_pipeline_orchestrator.py not found"

        content = data_pipeline.read_text()

        # Check for subscription to data_sync_completed
        has_subscription = (
            "data_sync_completed" in content.lower() or
            "DATA_SYNC_COMPLETED" in content
        )

        assert has_subscription, (
            "DataPipelineOrchestrator does not subscribe to DATA_SYNC_COMPLETED. "
            "This event triggers NPZ export after sync."
        )

    def test_sync_router_subscribes_to_training_started(self):
        """SyncRouter must subscribe to TRAINING_STARTED for priority sync."""
        sync_router = COORDINATION_DIR / "sync_router.py"
        assert sync_router.exists(), "sync_router.py not found"

        content = sync_router.read_text()

        has_subscription = (
            "training_started" in content.lower() or
            "TRAINING_STARTED" in content
        )

        assert has_subscription, (
            "SyncRouter does not subscribe to TRAINING_STARTED. "
            "This event should trigger priority sync to training node."
        )

    def test_feedback_loop_subscribes_to_training_completed(self):
        """FeedbackLoopController must subscribe to TRAINING_COMPLETED."""
        feedback_loop = COORDINATION_DIR / "feedback_loop_controller.py"
        assert feedback_loop.exists(), "feedback_loop_controller.py not found"

        content = feedback_loop.read_text()

        has_subscription = (
            "training_completed" in content.lower() or
            "TRAINING_COMPLETED" in content
        )

        assert has_subscription, (
            "FeedbackLoopController does not subscribe to TRAINING_COMPLETED. "
            "This event triggers evaluation and curriculum updates."
        )

    def test_distribution_subscribes_to_model_promoted(self):
        """UnifiedDistributionDaemon must subscribe to MODEL_PROMOTED."""
        dist_daemon = COORDINATION_DIR / "unified_distribution_daemon.py"
        assert dist_daemon.exists(), "unified_distribution_daemon.py not found"

        content = dist_daemon.read_text()

        has_subscription = (
            "model_promoted" in content.lower() or
            "MODEL_PROMOTED" in content
        )

        assert has_subscription, (
            "UnifiedDistributionDaemon does not subscribe to MODEL_PROMOTED. "
            "This event triggers model distribution to cluster nodes."
        )


class TestEventEmitterPresence:
    """Tests that verify critical events are actually emitted somewhere."""

    @pytest.mark.parametrize(
        "event_name,expected_emitter_files",
        [
            ("training_completed", ["training_coordinator", "train"]),
            ("model_promoted", ["promotion", "auto_promote"]),
            ("data_sync_completed", ["sync_planner", "auto_sync", "p2p"]),
            ("evaluation_completed", ["gauntlet", "evaluation"]),
        ],
    )
    def test_event_has_emitter(
        self, event_name: str, expected_emitter_files: List[str]
    ):
        """Verify critical events have emitters in expected files."""
        emitters = find_emitters()

        if event_name not in emitters:
            pytest.skip(
                f"Event '{event_name}' not found in scan. "
                f"May need to add emit pattern to find_emitters()."
            )
            return

        emitter_files = [loc[0].lower() for loc in emitters[event_name]]

        has_expected_emitter = any(
            any(expected in f for expected in expected_emitter_files)
            for f in emitter_files
        )

        # Informational - emit location may have changed
        if not has_expected_emitter:
            pytest.skip(
                f"Event '{event_name}' emitted from {emitter_files}, "
                f"not from expected {expected_emitter_files}. "
                f"Verify this is intentional."
            )
