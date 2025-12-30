"""Unit tests for ConfigHelper utility.

December 30, 2025: Created as part of Priority 4 architectural improvement.
Tests all ConfigHelper methods with comprehensive edge cases.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from app.config.config_helper import ConfigHelper, ConfigValidationError


class TestGetStr:
    """Tests for ConfigHelper.get_str()."""

    def test_returns_value_when_set(self):
        """Should return environment variable value when set."""
        with mock.patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert ConfigHelper.get_str("TEST_KEY") == "test_value"

    def test_returns_default_when_not_set(self):
        """Should return default when env var is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_str("NONEXISTENT", default="fallback") == "fallback"

    def test_returns_empty_string_as_default(self):
        """Should return empty string as default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_str("NONEXISTENT") == ""

    def test_raises_when_required_and_not_set(self):
        """Should raise ConfigValidationError when required and not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigHelper.get_str("REQUIRED_KEY", required=True)
            assert "REQUIRED_KEY" in str(exc_info.value)
            assert "Required value not set" in str(exc_info.value)

    def test_returns_value_when_required_and_set(self):
        """Should return value when required and set."""
        with mock.patch.dict(os.environ, {"REQUIRED": "value"}):
            assert ConfigHelper.get_str("REQUIRED", required=True) == "value"


class TestGetInt:
    """Tests for ConfigHelper.get_int()."""

    def test_parses_valid_integer(self):
        """Should parse valid integer string."""
        with mock.patch.dict(os.environ, {"TEST_INT": "42"}):
            assert ConfigHelper.get_int("TEST_INT") == 42

    def test_returns_default_when_not_set(self):
        """Should return default when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_int("NONEXISTENT", default=100) == 100

    def test_returns_default_for_invalid_value(self, caplog):
        """Should return default and log warning for invalid integer."""
        with mock.patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = ConfigHelper.get_int("TEST_INT", default=50)
            assert result == 50
            assert "not a valid integer" in caplog.text

    def test_clamps_to_min_value(self, caplog):
        """Should clamp value to min_val and log warning."""
        with mock.patch.dict(os.environ, {"TEST_INT": "5"}):
            result = ConfigHelper.get_int("TEST_INT", min_val=10)
            assert result == 10
            assert "below minimum" in caplog.text

    def test_clamps_to_max_value(self, caplog):
        """Should clamp value to max_val and log warning."""
        with mock.patch.dict(os.environ, {"TEST_INT": "100"}):
            result = ConfigHelper.get_int("TEST_INT", max_val=50)
            assert result == 50
            assert "above maximum" in caplog.text

    def test_value_within_range_not_clamped(self):
        """Should not clamp value within valid range."""
        with mock.patch.dict(os.environ, {"TEST_INT": "25"}):
            result = ConfigHelper.get_int("TEST_INT", min_val=10, max_val=50)
            assert result == 25

    def test_parses_negative_integer(self):
        """Should parse negative integers."""
        with mock.patch.dict(os.environ, {"TEST_INT": "-42"}):
            assert ConfigHelper.get_int("TEST_INT") == -42

    def test_raises_when_required_and_not_set(self):
        """Should raise ConfigValidationError when required and not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigValidationError):
                ConfigHelper.get_int("REQUIRED", required=True)


class TestGetFloat:
    """Tests for ConfigHelper.get_float()."""

    def test_parses_valid_float(self):
        """Should parse valid float string."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert ConfigHelper.get_float("TEST_FLOAT") == pytest.approx(3.14)

    def test_parses_integer_as_float(self):
        """Should parse integer string as float."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            assert ConfigHelper.get_float("TEST_FLOAT") == 42.0

    def test_returns_default_when_not_set(self):
        """Should return default when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_float("NONEXISTENT", default=1.5) == 1.5

    def test_returns_default_for_invalid_value(self, caplog):
        """Should return default and log warning for invalid float."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            result = ConfigHelper.get_float("TEST_FLOAT", default=0.5)
            assert result == 0.5
            assert "not a valid float" in caplog.text

    def test_clamps_to_min_value(self, caplog):
        """Should clamp value to min_val and log warning."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "0.1"}):
            result = ConfigHelper.get_float("TEST_FLOAT", min_val=0.5)
            assert result == 0.5
            assert "below minimum" in caplog.text

    def test_clamps_to_max_value(self, caplog):
        """Should clamp value to max_val and log warning."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "0.9"}):
            result = ConfigHelper.get_float("TEST_FLOAT", max_val=0.5)
            assert result == 0.5
            assert "above maximum" in caplog.text

    def test_parses_scientific_notation(self):
        """Should parse scientific notation."""
        with mock.patch.dict(os.environ, {"TEST_FLOAT": "1e-5"}):
            assert ConfigHelper.get_float("TEST_FLOAT") == pytest.approx(0.00001)


class TestGetBool:
    """Tests for ConfigHelper.get_bool()."""

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes", "on", "ON", "enabled"])
    def test_recognizes_true_values(self, value):
        """Should recognize various true value representations."""
        with mock.patch.dict(os.environ, {"TEST_BOOL": value}):
            assert ConfigHelper.get_bool("TEST_BOOL") is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "No", "off", "OFF", "disabled", ""])
    def test_recognizes_false_values(self, value):
        """Should recognize various false value representations."""
        with mock.patch.dict(os.environ, {"TEST_BOOL": value}):
            assert ConfigHelper.get_bool("TEST_BOOL") is False

    def test_returns_default_when_not_set(self):
        """Should return default when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_bool("NONEXISTENT", default=True) is True
            assert ConfigHelper.get_bool("NONEXISTENT", default=False) is False

    def test_returns_default_for_unrecognized_value(self, caplog):
        """Should return default and log warning for unrecognized value."""
        with mock.patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            result = ConfigHelper.get_bool("TEST_BOOL", default=True)
            assert result is True
            assert "not a recognized boolean value" in caplog.text

    def test_handles_whitespace(self):
        """Should handle values with leading/trailing whitespace."""
        with mock.patch.dict(os.environ, {"TEST_BOOL": "  true  "}):
            assert ConfigHelper.get_bool("TEST_BOOL") is True


class TestGetPath:
    """Tests for ConfigHelper.get_path()."""

    def test_returns_path_when_set(self):
        """Should return Path when environment variable is set."""
        with mock.patch.dict(os.environ, {"TEST_PATH": "/some/path"}):
            result = ConfigHelper.get_path("TEST_PATH")
            assert result == Path("/some/path")

    def test_returns_none_when_not_set_and_no_default(self):
        """Should return None when not set and no default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_path("NONEXISTENT") is None

    def test_returns_default_path_when_not_set(self):
        """Should return default Path when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = ConfigHelper.get_path("NONEXISTENT", default="/default/path")
            assert result == Path("/default/path")

    def test_accepts_path_as_default(self):
        """Should accept Path object as default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            default = Path("/default/path")
            result = ConfigHelper.get_path("NONEXISTENT", default=default)
            assert result == default

    def test_logs_warning_when_must_exist_and_missing(self, caplog, tmp_path):
        """Should log warning when must_exist=True and path doesn't exist."""
        with mock.patch.dict(os.environ, {"TEST_PATH": "/nonexistent/path/12345"}):
            result = ConfigHelper.get_path("TEST_PATH", must_exist=True)
            assert result == Path("/nonexistent/path/12345")
            assert "path does not exist" in caplog.text

    def test_no_warning_when_path_exists(self, tmp_path, caplog):
        """Should not log warning when path exists."""
        existing_file = tmp_path / "test_file"
        existing_file.touch()
        with mock.patch.dict(os.environ, {"TEST_PATH": str(existing_file)}):
            result = ConfigHelper.get_path("TEST_PATH", must_exist=True)
            assert result == existing_file
            assert "path does not exist" not in caplog.text

    def test_raises_when_required_and_not_set(self):
        """Should raise ConfigValidationError when required and not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigValidationError):
                ConfigHelper.get_path("REQUIRED", required=True)


class TestGetListStr:
    """Tests for ConfigHelper.get_list_str()."""

    def test_parses_comma_separated_values(self):
        """Should parse comma-separated string into list."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "a,b,c"}):
            result = ConfigHelper.get_list_str("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_strips_whitespace(self):
        """Should strip whitespace from values."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "  a , b , c  "}):
            result = ConfigHelper.get_list_str("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_filters_empty_values(self):
        """Should filter out empty values."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "a,,b,,,c"}):
            result = ConfigHelper.get_list_str("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_returns_default_when_not_set(self):
        """Should return default when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = ConfigHelper.get_list_str("NONEXISTENT", default=["x", "y"])
            assert result == ["x", "y"]

    def test_returns_empty_list_by_default(self):
        """Should return empty list when not set and no default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert ConfigHelper.get_list_str("NONEXISTENT") == []

    def test_custom_separator(self):
        """Should use custom separator."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "a;b;c"}):
            result = ConfigHelper.get_list_str("TEST_LIST", separator=";")
            assert result == ["a", "b", "c"]


class TestGetListInt:
    """Tests for ConfigHelper.get_list_int()."""

    def test_parses_comma_separated_integers(self):
        """Should parse comma-separated integers into list."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "1,2,3"}):
            result = ConfigHelper.get_list_int("TEST_LIST")
            assert result == [1, 2, 3]

    def test_strips_whitespace(self):
        """Should strip whitespace from values."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "  1 , 2 , 3  "}):
            result = ConfigHelper.get_list_int("TEST_LIST")
            assert result == [1, 2, 3]

    def test_skips_invalid_integers(self, caplog):
        """Should skip invalid integers with warning."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "1,bad,3"}):
            result = ConfigHelper.get_list_int("TEST_LIST")
            assert result == [1, 3]
            assert "skipping invalid integer" in caplog.text

    def test_returns_default_when_not_set(self):
        """Should return default when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = ConfigHelper.get_list_int("NONEXISTENT", default=[1, 2])
            assert result == [1, 2]

    def test_parses_negative_integers(self):
        """Should parse negative integers."""
        with mock.patch.dict(os.environ, {"TEST_LIST": "-1,0,1"}):
            result = ConfigHelper.get_list_int("TEST_LIST")
            assert result == [-1, 0, 1]


class TestInstanceMethods:
    """Tests for ConfigHelper instance methods with prefix."""

    def test_prefix_is_prepended(self):
        """Instance methods should prepend prefix to key."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_VALUE": "test"}):
            assert helper.str("VALUE") == "test"

    def test_int_with_prefix(self):
        """Should work with int method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_PORT": "8080"}):
            assert helper.int("PORT") == 8080

    def test_float_with_prefix(self):
        """Should work with float method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_RATE": "0.75"}):
            assert helper.float("RATE") == pytest.approx(0.75)

    def test_bool_with_prefix(self):
        """Should work with bool method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_ENABLED": "true"}):
            assert helper.bool("ENABLED") is True

    def test_path_with_prefix(self):
        """Should work with path method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_DIR": "/data"}):
            assert helper.path("DIR") == Path("/data")

    def test_list_str_with_prefix(self):
        """Should work with list_str method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_HOSTS": "a,b,c"}):
            assert helper.list_str("HOSTS") == ["a", "b", "c"]

    def test_list_int_with_prefix(self):
        """Should work with list_int method and prefix."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_PORTS": "80,443,8080"}):
            assert helper.list_int("PORTS") == [80, 443, 8080]

    def test_validation_works_with_instance(self):
        """Validation should work with instance methods."""
        helper = ConfigHelper(prefix="MY_PREFIX")
        with mock.patch.dict(os.environ, {"MY_PREFIX_PERCENT": "50"}):
            result = helper.int("PERCENT", min_val=0, max_val=100)
            assert result == 50

    def test_empty_prefix(self):
        """Empty prefix should use key as-is."""
        helper = ConfigHelper(prefix="")
        with mock.patch.dict(os.environ, {"DIRECT_KEY": "value"}):
            assert helper.str("DIRECT_KEY") == "value"


class TestLoadDict:
    """Tests for ConfigHelper.load_dict()."""

    def test_loads_multiple_values(self):
        """Should load multiple values at once."""
        with mock.patch.dict(os.environ, {
            "MY_ENABLED": "true",
            "MY_PORT": "8080",
            "MY_RATE": "0.5",
            "MY_NAME": "test",
        }):
            result = ConfigHelper.load_dict({
                "enabled": ("ENABLED", bool, False),
                "port": ("PORT", int, 80),
                "rate": ("RATE", float, 0.0),
                "name": ("NAME", str, "default"),
            }, prefix="MY")

            assert result["enabled"] is True
            assert result["port"] == 8080
            assert result["rate"] == pytest.approx(0.5)
            assert result["name"] == "test"

    def test_uses_defaults_when_not_set(self):
        """Should use defaults for missing values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = ConfigHelper.load_dict({
                "enabled": ("ENABLED", bool, True),
                "port": ("PORT", int, 8080),
            }, prefix="MY")

            assert result["enabled"] is True
            assert result["port"] == 8080

    def test_handles_path_type(self, tmp_path):
        """Should handle Path type."""
        with mock.patch.dict(os.environ, {"MY_DIR": str(tmp_path)}):
            result = ConfigHelper.load_dict({
                "directory": ("DIR", Path, None),
            }, prefix="MY")

            assert result["directory"] == tmp_path


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_exception_attributes(self):
        """Should have key, value, and reason attributes."""
        error = ConfigValidationError("MY_KEY", "bad_value", "Invalid format")
        assert error.key == "MY_KEY"
        assert error.value == "bad_value"
        assert error.reason == "Invalid format"

    def test_exception_message(self):
        """Should format message correctly."""
        error = ConfigValidationError("MY_KEY", "bad_value", "Invalid format")
        assert "MY_KEY" in str(error)
        assert "bad_value" in str(error)
        assert "Invalid format" in str(error)

    def test_inherits_from_value_error(self):
        """Should inherit from ValueError."""
        error = ConfigValidationError("KEY", "value", "reason")
        assert isinstance(error, ValueError)
