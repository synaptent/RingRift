"""Tests for priority_boosts.py - Priority boost multiplier calculations.

January 2026: Tests for the extracted priority boost functions.
"""

import pytest


class TestGetCascadePriority:
    """Tests for get_cascade_priority function."""

    def test_returns_float(self):
        """Test always returns a float."""
        from app.coordination.selfplay.priority_boosts import get_cascade_priority

        result = get_cascade_priority("hex8_2p")
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_positive_value(self):
        """Test returns positive value for any config."""
        from app.coordination.selfplay.priority_boosts import get_cascade_priority

        for config in ["hex8_2p", "square8_4p", "hexagonal_3p"]:
            result = get_cascade_priority(config)
            assert result >= 0.0
            assert result <= 10.0  # Reasonable upper bound


class TestGetImprovementBoosts:
    """Tests for get_improvement_boosts function."""

    def test_returns_dict(self):
        """Test returns a dictionary."""
        from app.coordination.selfplay.priority_boosts import get_improvement_boosts

        result = get_improvement_boosts()
        assert isinstance(result, dict)

    def test_boost_values_are_floats(self):
        """Test all boost values are floats."""
        from app.coordination.selfplay.priority_boosts import get_improvement_boosts

        result = get_improvement_boosts()
        for config_key, boost in result.items():
            assert isinstance(config_key, str)
            assert isinstance(boost, (int, float))

    def test_boost_values_in_range(self):
        """Test boost values are in expected range."""
        from app.coordination.selfplay.priority_boosts import get_improvement_boosts

        result = get_improvement_boosts()
        for boost in result.values():
            assert -0.5 <= boost <= 0.5  # Reasonable range


class TestGetMomentumMultipliers:
    """Tests for get_momentum_multipliers function."""

    def test_returns_dict(self):
        """Test returns a dictionary."""
        from app.coordination.selfplay.priority_boosts import get_momentum_multipliers

        result = get_momentum_multipliers()
        assert isinstance(result, dict)

    def test_multiplier_values_are_positive(self):
        """Test all multiplier values are positive."""
        from app.coordination.selfplay.priority_boosts import get_momentum_multipliers

        result = get_momentum_multipliers()
        for config_key, multiplier in result.items():
            assert isinstance(config_key, str)
            assert isinstance(multiplier, (int, float))
            assert multiplier > 0

    def test_multiplier_values_in_range(self):
        """Test multiplier values are in expected range."""
        from app.coordination.selfplay.priority_boosts import get_momentum_multipliers

        result = get_momentum_multipliers()
        for multiplier in result.values():
            assert 0.5 <= multiplier <= 2.0  # Reasonable range


class TestGetArchitectureBoosts:
    """Tests for get_architecture_boosts function."""

    def test_returns_dict(self):
        """Test returns a dictionary."""
        from app.coordination.selfplay.priority_boosts import get_architecture_boosts

        result = get_architecture_boosts()
        assert isinstance(result, dict)

    def test_boost_values_in_range(self):
        """Test boost values are in expected range (0.0 to 0.30)."""
        from app.coordination.selfplay.priority_boosts import get_architecture_boosts

        result = get_architecture_boosts()
        for config_key, boost in result.items():
            assert isinstance(config_key, str)
            assert isinstance(boost, (int, float))
            assert 0.0 <= boost <= 0.35  # Allow small margin


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_exist(self):
        """Test that __all__ exports are valid."""
        from app.coordination.selfplay import priority_boosts

        for name in priority_boosts.__all__:
            assert hasattr(priority_boosts, name), f"{name} not in module"

    def test_imports_from_package(self):
        """Test functions can be imported from package."""
        from app.coordination.selfplay import (
            get_cascade_priority,
            get_improvement_boosts,
            get_momentum_multipliers,
            get_architecture_boosts,
        )

        # All imports should be callable
        assert callable(get_cascade_priority)
        assert callable(get_improvement_boosts)
        assert callable(get_momentum_multipliers)
        assert callable(get_architecture_boosts)

    def test_functions_have_docstrings(self):
        """Test all functions have docstrings."""
        from app.coordination.selfplay.priority_boosts import (
            get_cascade_priority,
            get_improvement_boosts,
            get_momentum_multipliers,
            get_architecture_boosts,
        )

        assert get_cascade_priority.__doc__ is not None
        assert get_improvement_boosts.__doc__ is not None
        assert get_momentum_multipliers.__doc__ is not None
        assert get_architecture_boosts.__doc__ is not None
