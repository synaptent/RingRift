"""Tests for feature_registry.py - Optional dependency management.

Tests feature availability checking, export retrieval, and backward compatibility.
"""

from __future__ import annotations

from unittest import mock

import pytest

from app.training.feature_registry import (
    FEATURE_SPECS,
    FeatureRegistry,
    FeatureSpec,
    FeatureStatus,
    get_feature,
    get_feature_status,
    has_feature,
    require_feature,
    reset_feature_registry,
)


# =============================================================================
# FeatureSpec Tests
# =============================================================================


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_init_with_required_fields(self):
        """Test spec initialization with required fields."""
        spec = FeatureSpec(
            name="test_feature",
            module_path="app.test.module",
            exports=["TestClass", "test_func"],
        )

        assert spec.name == "test_feature"
        assert spec.module_path == "app.test.module"
        assert spec.exports == ["TestClass", "test_func"]
        assert spec.description == ""

    def test_init_with_description(self):
        """Test spec initialization with description."""
        spec = FeatureSpec(
            name="test_feature",
            module_path="app.test.module",
            exports=["TestClass"],
            description="A test feature for unit tests",
        )

        assert spec.description == "A test feature for unit tests"


class TestFeatureStatus:
    """Tests for FeatureStatus dataclass."""

    def test_init_available(self):
        """Test status initialization for available feature."""
        status = FeatureStatus(
            available=True,
            exports={"TestClass": object, "test_func": lambda: None},
        )

        assert status.available is True
        assert len(status.exports) == 2
        assert status.error == ""

    def test_init_unavailable(self):
        """Test status initialization for unavailable feature."""
        status = FeatureStatus(
            available=False,
            error="ModuleNotFoundError: No module named 'test'",
        )

        assert status.available is False
        assert status.exports == {}
        assert "ModuleNotFoundError" in status.error


# =============================================================================
# FeatureRegistry Tests
# =============================================================================


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before and after each test."""
        reset_feature_registry()
        yield
        reset_feature_registry()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        reg1 = FeatureRegistry()
        reg2 = FeatureRegistry()

        assert reg1 is reg2

    def test_has_feature_known_feature(self):
        """Test has_feature for a feature that likely exists."""
        registry = FeatureRegistry()

        # This module should exist in the codebase
        result = registry.has_feature("data_catalog")

        # Result depends on whether the module is installed
        assert isinstance(result, bool)

    def test_has_feature_unknown_feature(self):
        """Test has_feature for an unknown feature."""
        registry = FeatureRegistry()

        result = registry.has_feature("nonexistent_feature_xyz")

        assert result is False

    def test_get_feature_returns_none_when_unavailable(self):
        """Test get_feature returns None for unavailable features."""
        registry = FeatureRegistry()

        result = registry.get_feature("nonexistent_feature", "SomeClass")

        assert result is None

    def test_require_feature_raises_for_unavailable(self):
        """Test require_feature raises ImportError for unavailable features."""
        registry = FeatureRegistry()

        with pytest.raises(ImportError, match="not available"):
            registry.require_feature("nonexistent_feature", "SomeClass")

    def test_get_all_exports_returns_empty_for_unavailable(self):
        """Test get_all_exports returns empty dict for unavailable features."""
        registry = FeatureRegistry()

        result = registry.get_all_exports("nonexistent_feature")

        assert result == {}

    def test_get_status_report_includes_all_specs(self):
        """Test get_status_report includes all registered features."""
        registry = FeatureRegistry()

        report = registry.get_status_report()

        # Should include all features from FEATURE_SPECS
        assert len(report) == len(FEATURE_SPECS)
        for feature_name in FEATURE_SPECS:
            assert feature_name in report
            assert "available" in report[feature_name]
            assert "description" in report[feature_name]

    def test_clear_cache(self):
        """Test clear_cache resets status cache."""
        registry = FeatureRegistry()

        # Populate cache
        registry.has_feature("data_catalog")
        assert len(registry._status_cache) > 0

        # Clear cache
        registry.clear_cache()

        assert len(registry._status_cache) == 0

    def test_reset_instance(self):
        """Test reset_instance creates new singleton."""
        reg1 = FeatureRegistry()
        FeatureRegistry.reset_instance()
        reg2 = FeatureRegistry()

        assert reg1 is not reg2


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before and after each test."""
        reset_feature_registry()
        yield
        reset_feature_registry()

    def test_has_feature_function(self):
        """Test has_feature module function."""
        result = has_feature("data_catalog")

        assert isinstance(result, bool)

    def test_get_feature_function(self):
        """Test get_feature module function."""
        result = get_feature("nonexistent_feature", "SomeClass")

        assert result is None

    def test_require_feature_function_raises(self):
        """Test require_feature raises for unavailable features."""
        with pytest.raises(ImportError):
            require_feature("nonexistent_feature", "SomeClass")

    def test_get_feature_status_function(self):
        """Test get_feature_status module function."""
        status = get_feature_status()

        assert isinstance(status, dict)
        assert len(status) == len(FEATURE_SPECS)

    def test_reset_feature_registry_function(self):
        """Test reset_feature_registry module function."""
        # Use the registry
        has_feature("data_catalog")

        # Reset
        reset_feature_registry()

        # Should work after reset
        result = has_feature("data_catalog")
        assert isinstance(result, bool)


# =============================================================================
# Integration Tests (with real modules)
# =============================================================================


class TestRealFeatureIntegration:
    """Integration tests with actual modules."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before and after each test."""
        reset_feature_registry()
        yield
        reset_feature_registry()

    def test_data_catalog_feature_if_available(self):
        """Test data_catalog feature registration."""
        assert has_feature("data_catalog")

        DataCatalog = get_feature("data_catalog", "DataCatalog")
        get_data_catalog = get_feature("data_catalog", "get_data_catalog")

        assert DataCatalog is not None
        assert get_data_catalog is not None

    def test_training_facade_feature_if_available(self):
        """Test training_facade feature registration."""
        assert has_feature("training_facade")

        FacadeConfig = get_feature("training_facade", "FacadeConfig")
        TrainingEnhancementsFacade = get_feature(
            "training_facade", "TrainingEnhancementsFacade"
        )

        assert FacadeConfig is not None
        assert TrainingEnhancementsFacade is not None


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward-compatible HAS_* constants."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before and after each test."""
        reset_feature_registry()
        yield
        reset_feature_registry()

    def test_has_constant_via_getattr(self):
        """Test HAS_* constants work via __getattr__."""
        from app.training import feature_registry

        # Access HAS_DATA_CATALOG which maps to has_feature('data_catalog')
        result = feature_registry.HAS_DATA_CATALOG

        assert isinstance(result, bool)

    def test_has_constant_case_insensitive(self):
        """Test HAS_* constants convert to lowercase feature names."""
        from app.training import feature_registry

        # HAS_TRAINING_FACADE -> training_facade
        result = feature_registry.HAS_TRAINING_FACADE

        assert isinstance(result, bool)

    def test_unknown_attribute_raises(self):
        """Test non-HAS_* attributes raise AttributeError."""
        from app.training import feature_registry

        with pytest.raises(AttributeError):
            _ = feature_registry.NONEXISTENT_ATTRIBUTE


# =============================================================================
# Mock Tests (for testing import failure handling)
# =============================================================================


class TestImportFailureHandling:
    """Tests for handling import failures."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before and after each test."""
        reset_feature_registry()
        yield
        reset_feature_registry()

    def test_import_error_handling(self):
        """Test that ImportError is handled gracefully."""
        registry = FeatureRegistry()

        # Add a spec that will definitely fail
        registry._specs["definitely_not_a_module"] = FeatureSpec(
            name="definitely_not_a_module",
            module_path="this.module.does.not.exist.anywhere",
            exports=["NonexistentClass"],
            description="A module that does not exist",
        )

        result = registry.has_feature("definitely_not_a_module")

        assert result is False

    def test_status_captures_error_message(self):
        """Test that status captures error message on failure."""
        registry = FeatureRegistry()

        # Add a spec that will definitely fail
        registry._specs["bad_module"] = FeatureSpec(
            name="bad_module",
            module_path="definitely.not.a.real.module.path",
            exports=["SomeClass"],
            description="A module that does not exist",
        )

        status = registry._check_feature("bad_module")

        assert status.available is False
        assert len(status.error) > 0


# =============================================================================
# FEATURE_SPECS Validation Tests
# =============================================================================


class TestFeatureSpecsValidation:
    """Tests to validate FEATURE_SPECS configuration."""

    def test_all_specs_have_required_fields(self):
        """Test all specs have required fields populated."""
        for name, spec in FEATURE_SPECS.items():
            assert spec.name == name, f"{name}: name mismatch"
            assert spec.module_path, f"{name}: missing module_path"
            assert len(spec.exports) > 0, f"{name}: no exports defined"

    def test_no_duplicate_feature_names(self):
        """Test there are no duplicate feature names."""
        names = list(FEATURE_SPECS.keys())
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_exports_are_strings(self):
        """Test all exports are string names."""
        for name, spec in FEATURE_SPECS.items():
            for export in spec.exports:
                assert isinstance(export, str), (
                    f"{name}: export {export} is not a string"
                )
