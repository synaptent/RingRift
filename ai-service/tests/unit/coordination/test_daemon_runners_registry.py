"""Unit tests for daemon_runners registry infrastructure.

December 30, 2025: Tests the new registry-driven factory pattern.

Tests coverage:
- InstantiationStyle enum
- WaitStyle enum
- StartMethod enum
- RunnerSpec dataclass
- RUNNER_SPECS registry completeness
- _create_runner_from_spec factory
- get_runner_spec helper
- Backward compatibility with legacy create_* functions
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from app.coordination.daemon_runners import (
    InstantiationStyle,
    WaitStyle,
    StartMethod,
    RunnerSpec,
    RUNNER_SPECS,
    get_runner_spec,
    _create_runner_from_spec,
    get_runner,
    get_all_runners,
)


class TestInstantiationStyle:
    """Test InstantiationStyle enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert InstantiationStyle.DIRECT.value == "direct"
        assert InstantiationStyle.SINGLETON.value == "singleton"
        assert InstantiationStyle.FACTORY.value == "factory"
        assert InstantiationStyle.WITH_CONFIG.value == "with_config"
        assert InstantiationStyle.ASYNC_FACTORY.value == "async_factory"

    def test_enum_count(self):
        """Test expected number of styles."""
        assert len(InstantiationStyle) == 5


class TestWaitStyle:
    """Test WaitStyle enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert WaitStyle.DAEMON.value == "daemon"
        assert WaitStyle.FOREVER_LOOP.value == "forever_loop"
        assert WaitStyle.RUN_FOREVER.value == "run_forever"
        assert WaitStyle.NONE.value == "none"
        assert WaitStyle.CUSTOM.value == "custom"

    def test_enum_count(self):
        """Test expected number of styles."""
        assert len(WaitStyle) == 5


class TestStartMethod:
    """Test StartMethod enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert StartMethod.ASYNC_START.value == "async_start"
        assert StartMethod.SYNC_START.value == "sync_start"
        assert StartMethod.INITIALIZE.value == "initialize"
        assert StartMethod.START_SERVER.value == "start_server"
        assert StartMethod.NONE.value == "none"

    def test_enum_count(self):
        """Test expected number of methods."""
        assert len(StartMethod) == 5


class TestRunnerSpec:
    """Test RunnerSpec dataclass."""

    def test_required_fields(self):
        """Test required fields are enforced."""
        spec = RunnerSpec(
            module="app.coordination.test_module",
            class_name="TestDaemon",
        )
        assert spec.module == "app.coordination.test_module"
        assert spec.class_name == "TestDaemon"

    def test_default_values(self):
        """Test default values are applied."""
        spec = RunnerSpec(
            module="app.coordination.test",
            class_name="TestDaemon",
        )
        assert spec.style == InstantiationStyle.DIRECT
        assert spec.config_class is None
        assert spec.factory_func is None
        assert spec.start_method == StartMethod.ASYNC_START
        assert spec.wait == WaitStyle.DAEMON
        assert spec.wait_interval == 60.0
        assert spec.deprecated is False
        assert spec.deprecation_message == ""
        assert spec.notes == ""
        assert spec.extra_imports == []

    def test_with_config_style(self):
        """Test WITH_CONFIG style spec."""
        spec = RunnerSpec(
            module="app.coordination.test",
            class_name="TestDaemon",
            style=InstantiationStyle.WITH_CONFIG,
            config_class="TestConfig",
        )
        assert spec.style == InstantiationStyle.WITH_CONFIG
        assert spec.config_class == "TestConfig"

    def test_factory_style(self):
        """Test FACTORY style spec."""
        spec = RunnerSpec(
            module="app.coordination.test",
            class_name="TestDaemon",
            style=InstantiationStyle.FACTORY,
            factory_func="get_test_daemon",
        )
        assert spec.style == InstantiationStyle.FACTORY
        assert spec.factory_func == "get_test_daemon"

    def test_deprecated_spec(self):
        """Test deprecated spec with message."""
        spec = RunnerSpec(
            module="app.coordination.test",
            class_name="TestDaemon",
            deprecated=True,
            deprecation_message="Use NewDaemon instead",
        )
        assert spec.deprecated is True
        assert "NewDaemon" in spec.deprecation_message


class TestRunnerSpecsRegistry:
    """Test RUNNER_SPECS registry."""

    def test_registry_not_empty(self):
        """Test registry has entries."""
        assert len(RUNNER_SPECS) > 0

    def test_registry_minimum_count(self):
        """Test registry has expected minimum entries."""
        # We expect at least 80 daemon types
        assert len(RUNNER_SPECS) >= 80

    def test_all_specs_have_module(self):
        """Test all specs have module field."""
        for name, spec in RUNNER_SPECS.items():
            assert spec.module, f"Spec {name} missing module"

    def test_all_specs_have_class_name(self):
        """Test all specs have class_name field."""
        for name, spec in RUNNER_SPECS.items():
            assert spec.class_name, f"Spec {name} missing class_name"

    def test_factory_specs_have_factory_func(self):
        """Test FACTORY style specs have factory_func."""
        for name, spec in RUNNER_SPECS.items():
            if spec.style in (InstantiationStyle.FACTORY, InstantiationStyle.ASYNC_FACTORY):
                assert spec.factory_func, f"Spec {name} uses FACTORY style but missing factory_func"

    def test_with_config_specs_have_config_class(self):
        """Test WITH_CONFIG style specs have config_class."""
        for name, spec in RUNNER_SPECS.items():
            if spec.style == InstantiationStyle.WITH_CONFIG:
                assert spec.config_class, f"Spec {name} uses WITH_CONFIG but missing config_class"

    def test_deprecated_specs_have_message(self):
        """Test deprecated specs have deprecation message."""
        for name, spec in RUNNER_SPECS.items():
            if spec.deprecated:
                assert spec.deprecation_message, f"Deprecated spec {name} missing deprecation_message"

    def test_key_specs_present(self):
        """Test essential daemon specs are present."""
        essential = [
            "auto_sync",
            "event_router",
            "data_pipeline",
            "feedback_loop",
            "evaluation",
            "quality_monitor",
        ]
        for name in essential:
            assert name in RUNNER_SPECS, f"Essential spec {name} not in registry"

    def test_spec_names_lowercase(self):
        """Test all spec names are lowercase with underscores."""
        for name in RUNNER_SPECS:
            assert name == name.lower(), f"Spec name {name} not lowercase"
            assert " " not in name, f"Spec name {name} contains spaces"


class TestGetRunnerSpec:
    """Test get_runner_spec helper."""

    def test_get_existing_spec(self):
        """Test getting an existing spec."""
        spec = get_runner_spec("auto_sync")
        assert spec is not None
        assert spec.class_name == "AutoSyncDaemon"

    def test_get_nonexistent_spec(self):
        """Test getting a nonexistent spec returns None."""
        spec = get_runner_spec("nonexistent_daemon")
        assert spec is None

    def test_get_deprecated_spec(self):
        """Test getting a deprecated spec still works."""
        spec = get_runner_spec("sync_coordinator")
        assert spec is not None
        assert spec.deprecated is True


class TestCreateRunnerFromSpec:
    """Test _create_runner_from_spec factory."""

    @pytest.mark.asyncio
    async def test_unknown_runner_raises(self):
        """Test unknown runner name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown runner"):
            await _create_runner_from_spec("nonexistent_daemon")

    @pytest.mark.asyncio
    async def test_import_error_propagates(self):
        """Test ImportError is propagated."""
        # Create a spec with invalid module
        with patch.dict(RUNNER_SPECS, {
            "test_invalid": RunnerSpec(
                module="app.coordination.does_not_exist",
                class_name="NonexistentDaemon",
            )
        }):
            with pytest.raises(ImportError):
                await _create_runner_from_spec("test_invalid")

    @pytest.mark.asyncio
    async def test_deprecated_runner_warns(self):
        """Test deprecated runner emits warning."""
        with patch.dict(RUNNER_SPECS, {
            "test_deprecated": RunnerSpec(
                module="app.coordination.auto_sync_daemon",  # Use real module
                class_name="AutoSyncDaemon",
                deprecated=True,
                deprecation_message="Test deprecation warning",
                wait=WaitStyle.NONE,  # Don't actually wait
            )
        }):
            with pytest.warns(DeprecationWarning, match="Test deprecation warning"):
                # Mock the daemon to avoid actual start
                with patch("app.coordination.auto_sync_daemon.AutoSyncDaemon") as mock:
                    mock.return_value.start = AsyncMock()
                    await _create_runner_from_spec("test_deprecated")


class TestBackwardCompatibility:
    """Test backward compatibility with legacy functions."""

    def test_get_runner_exists(self):
        """Test get_runner function exists."""
        from app.coordination.daemon_types import DaemonType
        runner = get_runner(DaemonType.AUTO_SYNC)
        assert runner is not None
        assert callable(runner)

    def test_get_all_runners_exists(self):
        """Test get_all_runners function exists."""
        runners = get_all_runners()
        assert isinstance(runners, dict)
        assert len(runners) > 0

    def test_legacy_create_functions_exist(self):
        """Test legacy create_* functions still exist."""
        from app.coordination.daemon_runners import (
            create_auto_sync,
            create_event_router,
            create_feedback_loop,
        )
        assert callable(create_auto_sync)
        assert callable(create_event_router)
        assert callable(create_feedback_loop)


class TestRegistryDaemonTypeCoverage:
    """Test registry covers all DaemonType values."""

    def test_registry_covers_daemon_types(self):
        """Test registry has entries for most DaemonType values."""
        from app.coordination.daemon_types import DaemonType

        # Get all daemon type names
        daemon_names = {dt.name.lower() for dt in DaemonType}

        # Get all registry names
        registry_names = set(RUNNER_SPECS.keys())

        # Check coverage
        missing = daemon_names - registry_names

        # Allow some missing (new types added but not yet in registry)
        # But there should be at most 10 missing
        assert len(missing) <= 10, f"Too many DaemonTypes missing from registry: {missing}"


class TestSpecCategorization:
    """Test specs are properly categorized by style."""

    def test_direct_style_count(self):
        """Test number of DIRECT style specs."""
        count = sum(1 for s in RUNNER_SPECS.values() if s.style == InstantiationStyle.DIRECT)
        # Most daemons use direct instantiation
        assert count >= 30

    def test_factory_style_count(self):
        """Test number of FACTORY style specs."""
        count = sum(1 for s in RUNNER_SPECS.values() if s.style == InstantiationStyle.FACTORY)
        assert count >= 15

    def test_singleton_style_count(self):
        """Test number of SINGLETON style specs."""
        count = sum(1 for s in RUNNER_SPECS.values() if s.style == InstantiationStyle.SINGLETON)
        assert count >= 3

    def test_deprecated_count(self):
        """Test number of deprecated specs."""
        count = sum(1 for s in RUNNER_SPECS.values() if s.deprecated)
        # Should have some deprecated specs
        assert count >= 3
        # But not too many
        assert count <= 15


class TestSpecModulePaths:
    """Test spec module paths are valid."""

    def test_module_paths_start_with_app(self):
        """Test most module paths start with app."""
        for name, spec in RUNNER_SPECS.items():
            assert spec.module.startswith("app.") or spec.module.startswith("scripts."), \
                f"Spec {name} has unexpected module path: {spec.module}"

    def test_coordination_module_paths(self):
        """Test coordination daemon modules are in app.coordination."""
        coordination_specs = [
            name for name, spec in RUNNER_SPECS.items()
            if "sync" not in name.lower() or name == "auto_sync"
        ]
        # Most coordination daemons should be in app.coordination
        coordination_module_count = sum(
            1 for name in coordination_specs
            if RUNNER_SPECS[name].module.startswith("app.coordination")
        )
        assert coordination_module_count >= 50
