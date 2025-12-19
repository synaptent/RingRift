"""Tests for assertion utilities."""

import pytest

from app.utils.assertions import (
    AssertionError,
    PreconditionError,
    PostconditionError,
    InvariantError,
    require,
    ensure,
    check,
    require_type,
    require_not_none,
    require_in_range,
    require_non_empty,
    unreachable,
)


class TestRequire:
    """Tests for require() precondition."""

    def test_passes_when_true(self):
        require(True, "should pass")
        require(1 == 1, "math works")
        require([1, 2, 3], "list is truthy")

    def test_raises_when_false(self):
        with pytest.raises(PreconditionError) as exc_info:
            require(False, "expected to fail")
        assert "expected to fail" in str(exc_info.value)

    def test_includes_context(self):
        with pytest.raises(PreconditionError) as exc_info:
            require(False, "validation failed", value=42, name="test")
        error = exc_info.value
        assert error.context["value"] == 42
        assert error.context["name"] == "test"


class TestEnsure:
    """Tests for ensure() postcondition."""

    def test_passes_when_true(self):
        ensure(True, "should pass")

    def test_raises_when_false(self):
        with pytest.raises(PostconditionError) as exc_info:
            ensure(False, "postcondition failed")
        assert "postcondition failed" in str(exc_info.value)


class TestCheck:
    """Tests for check() invariant."""

    def test_passes_when_true(self):
        check(True, "invariant holds")

    def test_raises_when_false(self):
        with pytest.raises(InvariantError) as exc_info:
            check(False, "invariant violated")
        assert "invariant violated" in str(exc_info.value)


class TestRequireType:
    """Tests for require_type()."""

    def test_passes_for_correct_type(self):
        require_type("hello", str, "greeting")
        require_type(42, int, "number")
        require_type([1, 2], list, "items")

    def test_passes_for_tuple_of_types(self):
        require_type("hello", (str, int), "value")
        require_type(42, (str, int), "value")

    def test_raises_for_wrong_type(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_type("hello", int, "number")
        assert "number must be int" in str(exc_info.value)
        assert "got str" in str(exc_info.value)

    def test_error_shows_tuple_types(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_type([1, 2], (str, int), "value")
        assert "str or int" in str(exc_info.value)


class TestRequireNotNone:
    """Tests for require_not_none()."""

    def test_returns_value_when_not_none(self):
        result = require_not_none("hello", "greeting")
        assert result == "hello"

        result = require_not_none(0, "number")  # 0 is not None
        assert result == 0

    def test_raises_when_none(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_not_none(None, "user")
        assert "user must not be None" in str(exc_info.value)


class TestRequireInRange:
    """Tests for require_in_range()."""

    def test_passes_when_in_range_inclusive(self):
        require_in_range(0, 0, 100, "value")
        require_in_range(50, 0, 100, "value")
        require_in_range(100, 0, 100, "value")

    def test_passes_when_in_range_exclusive(self):
        require_in_range(1, 0, 100, "value", inclusive=False)
        require_in_range(50, 0, 100, "value", inclusive=False)
        require_in_range(99, 0, 100, "value", inclusive=False)

    def test_raises_when_below_range(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_in_range(-1, 0, 100, "volume")
        assert "volume must be in range" in str(exc_info.value)
        assert "-1" in str(exc_info.value)

    def test_raises_when_above_range(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_in_range(101, 0, 100, "volume")
        assert "volume must be in range" in str(exc_info.value)

    def test_raises_on_boundary_when_exclusive(self):
        with pytest.raises(PreconditionError):
            require_in_range(0, 0, 100, "value", inclusive=False)
        with pytest.raises(PreconditionError):
            require_in_range(100, 0, 100, "value", inclusive=False)

    def test_works_with_floats(self):
        require_in_range(0.5, 0.0, 1.0, "probability")


class TestRequireNonEmpty:
    """Tests for require_non_empty()."""

    def test_passes_for_non_empty_list(self):
        require_non_empty([1, 2, 3], "items")

    def test_passes_for_non_empty_string(self):
        require_non_empty("hello", "name")

    def test_passes_for_non_empty_dict(self):
        require_non_empty({"key": "value"}, "config")

    def test_raises_for_empty_list(self):
        with pytest.raises(PreconditionError) as exc_info:
            require_non_empty([], "items")
        assert "items must not be empty" in str(exc_info.value)

    def test_raises_for_empty_string(self):
        with pytest.raises(PreconditionError):
            require_non_empty("", "name")


class TestUnreachable:
    """Tests for unreachable()."""

    def test_always_raises(self):
        with pytest.raises(InvariantError) as exc_info:
            unreachable("should not get here")
        assert "Unreachable code reached" in str(exc_info.value)
        assert "should not get here" in str(exc_info.value)


class TestAssertionError:
    """Tests for custom AssertionError."""

    def test_str_without_context(self):
        error = AssertionError("something failed")
        assert str(error) == "something failed"

    def test_str_with_context(self):
        error = AssertionError("validation failed", {"value": 42, "limit": 100})
        error_str = str(error)
        assert "validation failed" in error_str
        assert "value=42" in error_str
        assert "limit=100" in error_str

    def test_is_exception(self):
        error = AssertionError("test")
        assert isinstance(error, Exception)


class TestPreconditionError:
    """Tests for PreconditionError."""

    def test_is_assertion_error(self):
        error = PreconditionError("test")
        assert isinstance(error, AssertionError)


class TestPostconditionError:
    """Tests for PostconditionError."""

    def test_is_assertion_error(self):
        error = PostconditionError("test")
        assert isinstance(error, AssertionError)


class TestInvariantError:
    """Tests for InvariantError."""

    def test_is_assertion_error(self):
        error = InvariantError("test")
        assert isinstance(error, AssertionError)
