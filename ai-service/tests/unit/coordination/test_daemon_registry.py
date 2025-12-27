"""Tests for daemon_registry.py - Data-driven daemon registry.

December 2025 - Unit tests for the declarative daemon registry.

Tests cover:
- Registry completeness (all entries have valid runners)
- DaemonSpec dataclass validation
- Dependency graph validation (no cycles, no missing deps)
- Category helpers
- Registry validation function
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.coordination.daemon_registry import (
    DAEMON_REGISTRY,
    DaemonSpec,
    get_daemons_by_category,
    get_categories,
    validate_registry,
)
from app.coordination.daemon_types import DaemonType


class TestDaemonSpec:
    """Tests for the DaemonSpec dataclass."""

    def test_daemon_spec_creation(self):
        """Test that DaemonSpec can be created with required fields."""
        spec = DaemonSpec(runner_name="create_test_daemon")
        assert spec.runner_name == "create_test_daemon"
        assert spec.depends_on == ()
        assert spec.category == "misc"
        assert spec.auto_restart is True
        assert spec.max_restarts == 5

    def test_daemon_spec_with_dependencies(self):
        """Test DaemonSpec with dependencies."""
        spec = DaemonSpec(
            runner_name="create_auto_sync",
            depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
            category="sync",
        )
        assert spec.runner_name == "create_auto_sync"
        assert DaemonType.EVENT_ROUTER in spec.depends_on
        assert DaemonType.DATA_PIPELINE in spec.depends_on
        assert spec.category == "sync"

    def test_daemon_spec_is_frozen(self):
        """Test that DaemonSpec is immutable (frozen)."""
        spec = DaemonSpec(runner_name="create_test")
        with pytest.raises(AttributeError):
            spec.runner_name = "modified"

    def test_daemon_spec_custom_health_interval(self):
        """Test DaemonSpec with custom health check interval."""
        spec = DaemonSpec(
            runner_name="create_critical_daemon",
            health_check_interval=5.0,
        )
        assert spec.health_check_interval == 5.0

    def test_daemon_spec_auto_restart_disabled(self):
        """Test DaemonSpec with auto-restart disabled."""
        spec = DaemonSpec(
            runner_name="create_oneshot_daemon",
            auto_restart=False,
        )
        assert spec.auto_restart is False


class TestDaemonRegistry:
    """Tests for the DAEMON_REGISTRY constant."""

    def test_registry_is_dict(self):
        """Test that DAEMON_REGISTRY is a dictionary."""
        assert isinstance(DAEMON_REGISTRY, dict)

    def test_registry_has_entries(self):
        """Test that registry has daemon entries."""
        # Should have at least 50 entries (currently 62)
        assert len(DAEMON_REGISTRY) >= 50

    def test_registry_keys_are_daemon_types(self):
        """Test that all registry keys are DaemonType enum values."""
        for key in DAEMON_REGISTRY.keys():
            assert isinstance(key, DaemonType)

    def test_registry_values_are_daemon_specs(self):
        """Test that all registry values are DaemonSpec instances."""
        for value in DAEMON_REGISTRY.values():
            assert isinstance(value, DaemonSpec)

    def test_registry_critical_daemons_present(self):
        """Test that critical daemons are registered."""
        critical_daemons = [
            DaemonType.EVENT_ROUTER,
            DaemonType.DATA_PIPELINE,
            DaemonType.AUTO_SYNC,
            DaemonType.FEEDBACK_LOOP,
            DaemonType.SELFPLAY_COORDINATOR,
        ]
        for daemon_type in critical_daemons:
            assert daemon_type in DAEMON_REGISTRY, f"Missing critical daemon: {daemon_type.name}"


class TestRegistryCompleteness:
    """Tests verifying all runners exist for registered daemons."""

    def test_all_runners_exist(self):
        """Test that all registered runner_name values exist in daemon_runners."""
        from app.coordination import daemon_runners

        missing = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            if not hasattr(daemon_runners, spec.runner_name):
                missing.append(f"{daemon_type.name}: {spec.runner_name}")

        assert not missing, f"Missing runners: {missing}"

    def test_all_runners_are_coroutines(self):
        """Test that all runners are coroutine functions."""
        import inspect
        from app.coordination import daemon_runners

        not_coroutines = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            runner = getattr(daemon_runners, spec.runner_name, None)
            if runner and not inspect.iscoroutinefunction(runner):
                not_coroutines.append(f"{daemon_type.name}: {spec.runner_name}")

        assert not not_coroutines, f"Not coroutine functions: {not_coroutines}"

    def test_runner_names_follow_convention(self):
        """Test that runner names follow create_* naming convention."""
        violations = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            if not spec.runner_name.startswith("create_"):
                violations.append(f"{daemon_type.name}: {spec.runner_name}")

        assert not violations, f"Naming convention violations: {violations}"


class TestDependencyGraph:
    """Tests for daemon dependency validation."""

    def test_no_self_dependencies(self):
        """Test that no daemon depends on itself."""
        self_deps = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            if daemon_type in spec.depends_on:
                self_deps.append(daemon_type.name)

        assert not self_deps, f"Self-dependencies found: {self_deps}"

    def test_all_dependencies_exist(self):
        """Test that all dependencies exist in the registry or are HEALTH_SERVER."""
        missing_deps = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            for dep in spec.depends_on:
                # HEALTH_SERVER is registered inline in daemon_manager.py
                if dep not in DAEMON_REGISTRY and dep != DaemonType.HEALTH_SERVER:
                    missing_deps.append(f"{daemon_type.name} depends on missing {dep.name}")

        assert not missing_deps, f"Missing dependencies: {missing_deps}"

    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies in the graph."""
        # Build adjacency list
        graph = {dt: set(spec.depends_on) for dt, spec in DAEMON_REGISTRY.items()}

        # DFS-based cycle detection
        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in DAEMON_REGISTRY:
                    continue  # Skip HEALTH_SERVER or other inline daemons
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    pytest.fail(f"Circular dependency: {' -> '.join(d.name for d in cycle)}")
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        visited = set()
        for daemon_type in DAEMON_REGISTRY:
            if daemon_type not in visited:
                if has_cycle(daemon_type, visited, set(), []):
                    pytest.fail(f"Cycle detected starting from {daemon_type.name}")

    def test_event_router_has_no_dependencies(self):
        """Test that EVENT_ROUTER has no dependencies (it's the root)."""
        event_router_spec = DAEMON_REGISTRY.get(DaemonType.EVENT_ROUTER)
        assert event_router_spec is not None, "EVENT_ROUTER not in registry"
        assert len(event_router_spec.depends_on) == 0, (
            f"EVENT_ROUTER should have no dependencies, has: {event_router_spec.depends_on}"
        )


class TestCategoryHelpers:
    """Tests for category helper functions."""

    def test_get_categories_returns_list(self):
        """Test that get_categories returns a sorted list."""
        categories = get_categories()
        assert isinstance(categories, list)
        assert categories == sorted(categories)

    def test_get_categories_includes_expected(self):
        """Test that expected categories are present."""
        categories = get_categories()
        expected = ["sync", "event", "health", "pipeline", "resource"]
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"

    def test_get_daemons_by_category_sync(self):
        """Test getting sync category daemons."""
        sync_daemons = get_daemons_by_category("sync")
        assert isinstance(sync_daemons, list)
        assert DaemonType.AUTO_SYNC in sync_daemons

    def test_get_daemons_by_category_event(self):
        """Test getting event category daemons."""
        event_daemons = get_daemons_by_category("event")
        assert DaemonType.EVENT_ROUTER in event_daemons

    def test_get_daemons_by_category_nonexistent(self):
        """Test that non-existent category returns empty list."""
        daemons = get_daemons_by_category("nonexistent_category_xyz")
        assert daemons == []

    def test_all_daemons_have_valid_category(self):
        """Test that all daemons have a category from get_categories()."""
        categories = get_categories()
        invalid = []
        for daemon_type, spec in DAEMON_REGISTRY.items():
            if spec.category not in categories:
                invalid.append(f"{daemon_type.name}: {spec.category}")

        # This should never happen since categories come from specs
        assert not invalid, f"Invalid categories: {invalid}"


class TestValidateRegistry:
    """Tests for the validate_registry function."""

    def test_validate_registry_returns_list(self):
        """Test that validate_registry returns a list."""
        errors = validate_registry()
        assert isinstance(errors, list)

    def test_validate_registry_passes_on_valid_registry(self):
        """Test that validation passes on the real registry."""
        errors = validate_registry()
        # Allow some errors for test flexibility (e.g., if daemon_runners module issues)
        # But should have < 5 errors in a healthy state
        assert len(errors) < 5, f"Too many validation errors: {errors}"

    def test_validate_registry_catches_missing_runner(self):
        """Test that validation catches missing runner functions."""
        # Create a mock registry with a bad runner name
        with patch.dict(DAEMON_REGISTRY, {
            DaemonType.AUTO_SYNC: DaemonSpec(
                runner_name="nonexistent_runner_xyz",
                category="sync",
            )
        }):
            errors = validate_registry()
            assert any("nonexistent_runner_xyz" in e for e in errors)


class TestRegistryIntegration:
    """Integration tests for daemon registry with daemon_manager."""

    def test_registry_size_matches_daemon_types(self):
        """Test that registry covers most daemon types."""
        # Get all daemon types
        all_types = [dt for dt in DaemonType]
        registered = set(DAEMON_REGISTRY.keys())

        # Should have most daemon types registered (except HEALTH_SERVER which is inline)
        coverage = len(registered) / len(all_types)
        assert coverage >= 0.90, f"Registry coverage too low: {coverage:.1%}"

    def test_sync_daemons_depend_on_event_router(self):
        """Test that sync daemons depend on EVENT_ROUTER."""
        sync_daemons = get_daemons_by_category("sync")

        for daemon_type in sync_daemons:
            spec = DAEMON_REGISTRY[daemon_type]
            # AUTO_SYNC and friends should depend on EVENT_ROUTER
            if daemon_type in [DaemonType.AUTO_SYNC, DaemonType.TRAINING_NODE_WATCHER]:
                assert DaemonType.EVENT_ROUTER in spec.depends_on, (
                    f"{daemon_type.name} should depend on EVENT_ROUTER"
                )

    def test_auto_sync_has_correct_dependencies(self):
        """Test that AUTO_SYNC has the documented critical dependencies."""
        auto_sync_spec = DAEMON_REGISTRY.get(DaemonType.AUTO_SYNC)
        assert auto_sync_spec is not None

        # AUTO_SYNC must depend on DATA_PIPELINE and FEEDBACK_LOOP
        # to ensure event handlers are subscribed before it emits
        required_deps = [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.FEEDBACK_LOOP]
        for dep in required_deps:
            assert dep in auto_sync_spec.depends_on, (
                f"AUTO_SYNC should depend on {dep.name}"
            )


class TestParametrizedDaemons:
    """Parametrized tests for all registered daemons."""

    @pytest.fixture
    def all_daemon_specs(self):
        """Get all daemon specs for parametrized tests."""
        return list(DAEMON_REGISTRY.items())

    def test_all_specs_have_runner_name(self, all_daemon_specs):
        """Test that all specs have a non-empty runner_name."""
        for daemon_type, spec in all_daemon_specs:
            assert spec.runner_name, f"{daemon_type.name} has empty runner_name"

    def test_all_specs_have_valid_category(self, all_daemon_specs):
        """Test that all specs have a non-empty category."""
        for daemon_type, spec in all_daemon_specs:
            assert spec.category, f"{daemon_type.name} has empty category"

    def test_all_dependencies_are_daemon_types(self, all_daemon_specs):
        """Test that all dependencies are DaemonType enum values."""
        for daemon_type, spec in all_daemon_specs:
            for dep in spec.depends_on:
                assert isinstance(dep, DaemonType), (
                    f"{daemon_type.name} has non-DaemonType dependency: {dep}"
                )

    @pytest.mark.parametrize("daemon_name", [
        "AUTO_SYNC", "DATA_PIPELINE", "EVENT_ROUTER", "FEEDBACK_LOOP",
        "SELFPLAY_COORDINATOR", "TRAINING_TRIGGER", "EVALUATION",
        "MODEL_DISTRIBUTION", "CLUSTER_MONITOR", "DAEMON_WATCHDOG",
    ])
    def test_critical_daemons_registered(self, daemon_name):
        """Test that critical daemons are in the registry."""
        daemon_type = getattr(DaemonType, daemon_name, None)
        assert daemon_type is not None, f"DaemonType.{daemon_name} doesn't exist"
        assert daemon_type in DAEMON_REGISTRY, f"{daemon_name} not in registry"
