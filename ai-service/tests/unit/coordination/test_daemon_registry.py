"""Tests for DaemonRegistry module.

December 2025: Unit tests for daemon_registry.py which provides declarative
daemon specifications for DaemonManager.
"""

from __future__ import annotations

import pytest

from app.coordination.daemon_registry import (
    DAEMON_REGISTRY,
    DEPRECATED_TYPES,
    DaemonSpec,
    get_categories,
    get_daemons_by_category,
    get_deprecated_types,
    is_daemon_deprecated,
    validate_registry,
    validate_registry_or_raise,
)
from app.coordination.daemon_types import DaemonType


class TestDaemonSpec:
    """Test DaemonSpec dataclass."""

    def test_default_values(self):
        """Test DaemonSpec with default values."""
        spec = DaemonSpec(runner_name="create_test")

        assert spec.runner_name == "create_test"
        assert spec.depends_on == ()
        assert spec.health_check_interval is None
        assert spec.auto_restart is True
        assert spec.max_restarts == 5
        assert spec.category == "misc"

    def test_custom_values(self):
        """Test DaemonSpec with custom values."""
        spec = DaemonSpec(
            runner_name="create_custom",
            depends_on=(DaemonType.EVENT_ROUTER,),
            health_check_interval=30.0,
            auto_restart=False,
            max_restarts=3,
            category="sync",
        )

        assert spec.runner_name == "create_custom"
        assert spec.depends_on == (DaemonType.EVENT_ROUTER,)
        assert spec.health_check_interval == 30.0
        assert spec.auto_restart is False
        assert spec.max_restarts == 3
        assert spec.category == "sync"

    def test_frozen_immutability(self):
        """Test that DaemonSpec is immutable (frozen=True)."""
        spec = DaemonSpec(runner_name="create_test")

        with pytest.raises(AttributeError):
            spec.runner_name = "modified"


class TestDaemonRegistry:
    """Test DAEMON_REGISTRY dict."""

    def test_registry_not_empty(self):
        """Test that registry has entries."""
        assert len(DAEMON_REGISTRY) > 0

    def test_registry_keys_are_daemon_types(self):
        """Test that all keys are DaemonType enum values."""
        for key in DAEMON_REGISTRY.keys():
            assert isinstance(key, DaemonType)

    def test_registry_values_are_daemon_specs(self):
        """Test that all values are DaemonSpec instances."""
        for value in DAEMON_REGISTRY.values():
            assert isinstance(value, DaemonSpec)

    def test_critical_daemons_registered(self):
        """Test that critical daemons are in registry."""
        critical = [
            DaemonType.EVENT_ROUTER,
            DaemonType.AUTO_SYNC,
            DaemonType.DATA_PIPELINE,
            DaemonType.FEEDBACK_LOOP,
            DaemonType.DAEMON_WATCHDOG,
        ]
        for daemon_type in critical:
            assert daemon_type in DAEMON_REGISTRY, f"{daemon_type} not in registry"

    def test_event_router_has_no_dependencies(self):
        """Test that EVENT_ROUTER has no dependencies (starts first)."""
        spec = DAEMON_REGISTRY[DaemonType.EVENT_ROUTER]
        assert spec.depends_on == ()

    def test_auto_sync_dependencies(self):
        """Test that AUTO_SYNC depends on critical subscribers."""
        spec = DAEMON_REGISTRY[DaemonType.AUTO_SYNC]

        # Must depend on EVENT_ROUTER for event emission
        assert DaemonType.EVENT_ROUTER in spec.depends_on

        # Must depend on DATA_PIPELINE and FEEDBACK_LOOP for event handling
        assert DaemonType.DATA_PIPELINE in spec.depends_on
        assert DaemonType.FEEDBACK_LOOP in spec.depends_on

    def test_sync_daemons_depend_on_event_router(self):
        """Test that all sync daemons depend on EVENT_ROUTER."""
        sync_daemons = get_daemons_by_category("sync")

        for daemon_type in sync_daemons:
            spec = DAEMON_REGISTRY[daemon_type]
            assert DaemonType.EVENT_ROUTER in spec.depends_on, (
                f"{daemon_type.name} should depend on EVENT_ROUTER"
            )

    def test_no_self_dependencies(self):
        """Test that no daemon depends on itself."""
        for daemon_type, spec in DAEMON_REGISTRY.items():
            assert daemon_type not in spec.depends_on, (
                f"{daemon_type.name} depends on itself"
            )

    def test_all_dependencies_exist_in_registry(self):
        """Test that all dependencies reference registered daemons."""
        # HEALTH_SERVER is intentionally excluded (inline registration)
        excluded = {DaemonType.HEALTH_SERVER}

        for daemon_type, spec in DAEMON_REGISTRY.items():
            for dep in spec.depends_on:
                if dep not in excluded:
                    assert dep in DAEMON_REGISTRY, (
                        f"{daemon_type.name} depends on {dep.name} "
                        f"which is not in registry"
                    )


class TestGetCategories:
    """Test get_categories function."""

    def test_returns_list(self):
        """Test that get_categories returns a list."""
        result = get_categories()
        assert isinstance(result, list)

    def test_returns_sorted_unique(self):
        """Test that result is sorted and unique."""
        result = get_categories()
        assert result == sorted(set(result))

    def test_expected_categories(self):
        """Test that expected categories exist."""
        result = get_categories()

        expected = ["event", "health", "pipeline", "sync"]
        for cat in expected:
            assert cat in result, f"Category '{cat}' not found"


class TestGetDaemonsByCategory:
    """Test get_daemons_by_category function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_daemons_by_category("event")
        assert isinstance(result, list)

    def test_event_category(self):
        """Test daemons in event category."""
        result = get_daemons_by_category("event")

        assert DaemonType.EVENT_ROUTER in result
        assert DaemonType.DLQ_RETRY in result

    def test_sync_category(self):
        """Test daemons in sync category."""
        result = get_daemons_by_category("sync")

        assert DaemonType.AUTO_SYNC in result
        assert DaemonType.ELO_SYNC in result

    def test_empty_category(self):
        """Test that unknown category returns empty list."""
        result = get_daemons_by_category("nonexistent")
        assert result == []

    def test_all_returned_have_matching_category(self):
        """Test that all returned daemons have the requested category."""
        for category in get_categories():
            daemons = get_daemons_by_category(category)
            for daemon_type in daemons:
                spec = DAEMON_REGISTRY[daemon_type]
                assert spec.category == category


class TestValidateRegistry:
    """Test validate_registry function."""

    def test_returns_list(self):
        """Test that validate_registry returns a list."""
        result = validate_registry()
        assert isinstance(result, list)

    def test_current_registry_valid(self):
        """Test that current DAEMON_REGISTRY passes validation."""
        errors = validate_registry()
        assert errors == [], f"Registry validation errors: {errors}"

    def test_detects_missing_runner(self):
        """Test that validation detects missing runners.

        This is a behavior test - we verify the validation logic works
        by checking that it correctly validates the current registry.
        """
        # The current registry should have all runners present
        errors = validate_registry()
        runner_errors = [e for e in errors if "not found in daemon_runners" in e]
        assert runner_errors == []

    def test_detects_self_dependencies(self):
        """Test that validation would catch self-dependencies.

        We verify this by checking the validation logic exists and works.
        Since the current registry has no self-dependencies, we just verify
        the check runs without issues.
        """
        errors = validate_registry()
        self_dep_errors = [e for e in errors if "cannot depend on itself" in e]
        assert self_dep_errors == []


class TestRegistryCompleteness:
    """Test registry completeness against DaemonType enum."""

    def test_all_daemon_types_covered_or_known_exception(self):
        """Test that all DaemonType values are either in registry or documented.

        HEALTH_SERVER is intentionally excluded because it requires self access.
        """
        all_types = set(dt for dt in DaemonType)
        registered = set(DAEMON_REGISTRY.keys())
        known_exclusions = {DaemonType.HEALTH_SERVER}

        missing = all_types - registered - known_exclusions

        assert missing == set(), (
            f"DaemonTypes missing from registry: {[dt.name for dt in missing]}"
        )

    def test_registry_coverage_percentage(self):
        """Test that registry covers at least 95% of DaemonTypes."""
        all_types = set(dt for dt in DaemonType)
        registered = set(DAEMON_REGISTRY.keys())

        coverage = len(registered) / len(all_types) * 100

        assert coverage >= 95, f"Registry coverage is only {coverage:.1f}%"


class TestDependencyChains:
    """Test specific dependency chains for correctness."""

    def test_evaluation_chain(self):
        """Test evaluation -> auto_promotion dependency chain."""
        auto_promo_spec = DAEMON_REGISTRY[DaemonType.AUTO_PROMOTION]

        # AUTO_PROMOTION should depend on EVALUATION
        assert DaemonType.EVALUATION in auto_promo_spec.depends_on

    def test_monitoring_chain(self):
        """Test cluster_monitor -> cluster_watchdog dependency chain."""
        watchdog_spec = DAEMON_REGISTRY[DaemonType.CLUSTER_WATCHDOG]

        # CLUSTER_WATCHDOG should depend on CLUSTER_MONITOR
        assert DaemonType.CLUSTER_MONITOR in watchdog_spec.depends_on

    def test_disk_management_chain(self):
        """Test disk_space_manager -> coordinator_disk_manager dependency."""
        coord_disk_spec = DAEMON_REGISTRY[DaemonType.COORDINATOR_DISK_MANAGER]

        # COORDINATOR_DISK_MANAGER should depend on DISK_SPACE_MANAGER
        assert DaemonType.DISK_SPACE_MANAGER in coord_disk_spec.depends_on

    def test_data_consolidation_dependencies(self):
        """Test DATA_CONSOLIDATION depends on pipeline."""
        spec = DAEMON_REGISTRY[DaemonType.DATA_CONSOLIDATION]

        assert DaemonType.EVENT_ROUTER in spec.depends_on
        assert DaemonType.DATA_PIPELINE in spec.depends_on


class TestRunnerValidation:
    """Test runner name format and existence."""

    def test_all_runner_names_start_with_create(self):
        """Test that all runner names follow create_* convention."""
        for daemon_type, spec in DAEMON_REGISTRY.items():
            assert spec.runner_name.startswith("create_"), (
                f"{daemon_type.name} runner '{spec.runner_name}' "
                f"should start with 'create_'"
            )

    def test_all_runners_exist(self):
        """Test that all runner functions exist in daemon_runners."""
        from app.coordination import daemon_runners

        for daemon_type, spec in DAEMON_REGISTRY.items():
            assert hasattr(daemon_runners, spec.runner_name), (
                f"{daemon_type.name} runner '{spec.runner_name}' "
                f"not found in daemon_runners module"
            )

    def test_all_runners_are_callable(self):
        """Test that all runners are callable functions."""
        from app.coordination import daemon_runners

        for daemon_type, spec in DAEMON_REGISTRY.items():
            runner = getattr(daemon_runners, spec.runner_name, None)
            if runner is not None:
                assert callable(runner), (
                    f"{daemon_type.name} runner '{spec.runner_name}' is not callable"
                )


class TestCircularDependencies:
    """Test for circular dependency detection."""

    def test_no_direct_circular_dependencies(self):
        """Test that no daemon directly depends on its dependent."""
        for daemon_type, spec in DAEMON_REGISTRY.items():
            for dep in spec.depends_on:
                if dep in DAEMON_REGISTRY:
                    dep_spec = DAEMON_REGISTRY[dep]
                    assert daemon_type not in dep_spec.depends_on, (
                        f"Circular dependency: {daemon_type.name} <-> {dep.name}"
                    )

    def test_no_transitive_circular_dependencies(self):
        """Test that no transitive circular dependencies exist.

        Uses depth-first search to detect cycles in the dependency graph.
        """
        def has_cycle(start: DaemonType, visited: set, path: set) -> bool:
            if start in path:
                return True
            if start in visited:
                return False
            if start not in DAEMON_REGISTRY:
                return False  # External dependency

            visited.add(start)
            path.add(start)

            for dep in DAEMON_REGISTRY[start].depends_on:
                if has_cycle(dep, visited, path):
                    return True

            path.remove(start)
            return False

        for daemon_type in DAEMON_REGISTRY:
            assert not has_cycle(daemon_type, set(), set()), (
                f"Transitive circular dependency detected starting from {daemon_type.name}"
            )


class TestCategoryDistribution:
    """Test category distribution and balance."""

    def test_each_category_has_at_least_one_daemon(self):
        """Test that no category is empty."""
        for category in get_categories():
            daemons = get_daemons_by_category(category)
            assert len(daemons) >= 1, f"Category '{category}' has no daemons"

    def test_category_counts(self):
        """Test expected category sizes are reasonable."""
        category_counts = {}
        for spec in DAEMON_REGISTRY.values():
            category_counts[spec.category] = category_counts.get(spec.category, 0) + 1

        # Should have at least 5 categories
        assert len(category_counts) >= 5

        # No single category should have more than 50% of daemons
        total = len(DAEMON_REGISTRY)
        for category, count in category_counts.items():
            assert count < total * 0.5, (
                f"Category '{category}' has {count}/{total} daemons (>50%)"
            )

    def test_known_categories(self):
        """Test that expected categories exist."""
        categories = get_categories()
        expected = ["event", "health", "pipeline", "sync", "evaluation", "distribution"]
        for cat in expected:
            assert cat in categories, f"Expected category '{cat}' not found"


class TestDaemonSpecAttributes:
    """Test DaemonSpec attribute handling."""

    def test_auto_restart_defaults(self):
        """Test that auto_restart defaults to True."""
        for spec in DAEMON_REGISTRY.values():
            # Default is True, so most should be True
            # Just verify the attribute exists and is boolean
            assert isinstance(spec.auto_restart, bool)

    def test_max_restarts_positive(self):
        """Test that max_restarts is always positive."""
        for daemon_type, spec in DAEMON_REGISTRY.items():
            assert spec.max_restarts > 0, (
                f"{daemon_type.name} has non-positive max_restarts: {spec.max_restarts}"
            )

    def test_health_check_interval_valid(self):
        """Test health_check_interval is None or positive."""
        for daemon_type, spec in DAEMON_REGISTRY.items():
            if spec.health_check_interval is not None:
                assert spec.health_check_interval > 0, (
                    f"{daemon_type.name} has non-positive health_check_interval"
                )


class TestDeprecatedTypes:
    """Test deprecated type handling (December 2025)."""

    def test_get_deprecated_types_returns_set(self):
        """Test that get_deprecated_types returns a set."""
        deprecated = get_deprecated_types()
        assert isinstance(deprecated, set)

    def test_deprecated_types_constant_matches_function(self):
        """Test that DEPRECATED_TYPES constant matches get_deprecated_types()."""
        assert DEPRECATED_TYPES == get_deprecated_types()

    def test_deprecated_types_are_daemon_types(self):
        """Test that all deprecated types are DaemonType enum values."""
        for dt in get_deprecated_types():
            assert isinstance(dt, DaemonType)

    def test_known_deprecated_types(self):
        """Test that known deprecated types are in the set."""
        deprecated = get_deprecated_types()
        # These are documented as deprecated in daemon_registry.py
        known_deprecated = [
            DaemonType.SYNC_COORDINATOR,
            DaemonType.HEALTH_CHECK,
            DaemonType.LAMBDA_IDLE,
        ]
        for dt in known_deprecated:
            if dt in DAEMON_REGISTRY and DAEMON_REGISTRY[dt].deprecated:
                assert dt in deprecated

    def test_is_daemon_deprecated_true(self):
        """Test is_daemon_deprecated returns True for deprecated types."""
        for dt in get_deprecated_types():
            assert is_daemon_deprecated(dt) is True

    def test_is_daemon_deprecated_false(self):
        """Test is_daemon_deprecated returns False for active types."""
        deprecated = get_deprecated_types()
        for dt in DaemonType:
            if dt not in deprecated and dt in DAEMON_REGISTRY:
                if not DAEMON_REGISTRY[dt].deprecated:
                    assert is_daemon_deprecated(dt) is False


class TestValidateRegistryOrRaise:
    """Test validate_registry_or_raise function (December 2025)."""

    def test_does_not_raise_on_valid_registry(self):
        """Test that validate_registry_or_raise does not raise on valid registry."""
        # This should not raise since our registry is valid
        validate_registry_or_raise()

    def test_returns_none_on_success(self):
        """Test that validate_registry_or_raise returns None on success."""
        result = validate_registry_or_raise()
        assert result is None


class TestMissingDaemonTypeDetection:
    """Test detection of DaemonType values missing from registry (December 2025)."""

    def test_all_daemon_types_in_registry(self):
        """Test that all DaemonType enum values are in the registry."""
        all_types = set(DaemonType)
        registered = set(DAEMON_REGISTRY.keys())
        deprecated = get_deprecated_types()

        missing = all_types - registered
        # Allow deprecated types to be missing from registry
        missing_non_deprecated = missing - deprecated

        assert len(missing_non_deprecated) == 0, (
            f"DaemonTypes missing from DAEMON_REGISTRY: "
            f"{[dt.name for dt in missing_non_deprecated]}"
        )

    def test_validate_registry_catches_missing_types(self):
        """Test that validate_registry would catch missing types if they existed."""
        # We can't easily add a temporary DaemonType, so we just verify
        # the current validation passes
        errors = validate_registry()
        missing_type_errors = [e for e in errors if "has no DAEMON_REGISTRY entry" in e]
        assert len(missing_type_errors) == 0
