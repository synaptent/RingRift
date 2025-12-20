"""Tests for YAML utilities."""

import os
from pathlib import Path

import pytest

from app.utils.yaml_utils import (
    YAMLLoadError,
    dump_yaml,
    dumps_yaml,
    load_config_yaml,
    load_yaml,
    load_yaml_with_defaults,
    safe_load_yaml,
    validate_yaml_schema,
)


class TestLoadYaml:
    """Tests for load_yaml()."""

    def test_load_valid_yaml(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("key: value\nnum: 42")

        result = load_yaml(filepath)

        assert result == {"key": "value", "num": 42}

    def test_load_nested_yaml(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("""
parent:
  child: value
  list:
    - item1
    - item2
""")

        result = load_yaml(filepath)

        assert result["parent"]["child"] == "value"
        assert result["parent"]["list"] == ["item1", "item2"]

    def test_load_empty_file_returns_empty_dict(self, tmp_path):
        filepath = tmp_path / "empty.yaml"
        filepath.write_text("")

        result = load_yaml(filepath)

        assert result == {}

    def test_missing_file_raises_when_required(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_yaml(filepath, required=True)

    def test_missing_file_returns_none_when_not_required(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        result = load_yaml(filepath, required=False)

        assert result is None

    def test_invalid_yaml_raises(self, tmp_path):
        filepath = tmp_path / "invalid.yaml"
        filepath.write_text("invalid: yaml: syntax: {{{")

        with pytest.raises(YAMLLoadError):
            load_yaml(filepath)

    def test_accepts_string_path(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("key: value")

        result = load_yaml(str(filepath))

        assert result == {"key": "value"}


class TestLoadYamlWithDefaults:
    """Tests for load_yaml_with_defaults()."""

    def test_merges_with_defaults(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("override: true")

        defaults = {"default": "value", "override": False}
        result = load_yaml_with_defaults(filepath, defaults)

        assert result["default"] == "value"
        assert result["override"] is True

    def test_returns_defaults_for_missing_file(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        defaults = {"key": "value"}
        result = load_yaml_with_defaults(filepath, defaults)

        assert result == {"key": "value"}

    def test_deep_merge(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("""
parent:
  child2: override
""")

        defaults = {
            "parent": {
                "child1": "default1",
                "child2": "default2",
            }
        }
        result = load_yaml_with_defaults(filepath, defaults)

        assert result["parent"]["child1"] == "default1"
        assert result["parent"]["child2"] == "override"

    def test_no_merge_nested_when_disabled(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("""
parent:
  child2: override
""")

        defaults = {
            "parent": {
                "child1": "default1",
                "child2": "default2",
            }
        }
        result = load_yaml_with_defaults(filepath, defaults, merge_nested=False)

        # entire parent dict should be replaced
        assert "child1" not in result["parent"]
        assert result["parent"]["child2"] == "override"

    def test_does_not_mutate_defaults(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("new_key: new_value")

        defaults = {"default": "value"}
        original_defaults = defaults.copy()

        load_yaml_with_defaults(filepath, defaults)

        assert defaults == original_defaults


class TestSafeLoadYaml:
    """Tests for safe_load_yaml()."""

    def test_loads_valid_yaml(self, tmp_path):
        filepath = tmp_path / "test.yaml"
        filepath.write_text("key: value")

        result = safe_load_yaml(filepath)

        assert result == {"key": "value"}

    def test_returns_default_on_missing(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        result = safe_load_yaml(filepath, default={"fallback": True})

        assert result == {"fallback": True}

    def test_returns_default_on_invalid_yaml(self, tmp_path):
        filepath = tmp_path / "invalid.yaml"
        filepath.write_text("invalid: yaml: {{{")

        result = safe_load_yaml(filepath, default=[], log_errors=False)

        assert result == []

    def test_returns_none_by_default_on_error(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        result = safe_load_yaml(filepath)

        assert result is None


class TestDumpYaml:
    """Tests for dump_yaml()."""

    def test_writes_yaml_file(self, tmp_path):
        filepath = tmp_path / "output.yaml"
        data = {"key": "value", "num": 42}

        dump_yaml(data, filepath)

        assert filepath.exists()
        content = filepath.read_text()
        assert "key: value" in content

    def test_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "nested" / "dir" / "output.yaml"

        dump_yaml({"key": "value"}, filepath)

        assert filepath.exists()

    def test_sort_keys(self, tmp_path):
        filepath = tmp_path / "output.yaml"

        dump_yaml({"z": 1, "a": 2}, filepath, sort_keys=True)

        content = filepath.read_text()
        # 'a' should come before 'z'
        assert content.index("a:") < content.index("z:")

    def test_accepts_string_path(self, tmp_path):
        filepath = tmp_path / "output.yaml"

        dump_yaml({"key": "value"}, str(filepath))

        assert filepath.exists()


class TestDumpsYaml:
    """Tests for dumps_yaml()."""

    def test_returns_string(self):
        result = dumps_yaml({"key": "value"})

        assert isinstance(result, str)
        assert "key: value" in result

    def test_handles_nested_data(self):
        data = {
            "parent": {
                "child": "value",
                "list": [1, 2, 3],
            }
        }

        result = dumps_yaml(data)

        assert "parent:" in result
        assert "child: value" in result

    def test_sort_keys(self):
        result = dumps_yaml({"z": 1, "a": 2}, sort_keys=True)

        assert result.index("a:") < result.index("z:")


class TestValidateYamlSchema:
    """Tests for validate_yaml_schema()."""

    def test_valid_with_required_keys(self):
        data = {"host": "localhost", "port": 8080}

        is_valid, errors = validate_yaml_schema(data, required_keys=["host", "port"])

        assert is_valid is True
        assert errors == []

    def test_missing_required_key(self):
        data = {"host": "localhost"}

        is_valid, errors = validate_yaml_schema(data, required_keys=["host", "port"])

        assert is_valid is False
        assert len(errors) == 1
        assert "port" in errors[0]

    def test_strict_mode_rejects_unknown(self):
        data = {"host": "localhost", "unknown": "value"}

        is_valid, errors = validate_yaml_schema(
            data,
            required_keys=["host"],
            optional_keys=["port"],
            strict=True,
        )

        assert is_valid is False
        assert any("unknown" in e for e in errors)

    def test_strict_mode_allows_optional(self):
        data = {"host": "localhost", "port": 8080}

        is_valid, errors = validate_yaml_schema(
            data,
            required_keys=["host"],
            optional_keys=["port"],
            strict=True,
        )

        assert is_valid is True
        assert errors == []

    def test_non_strict_allows_unknown(self):
        data = {"host": "localhost", "unknown": "value"}

        is_valid, _errors = validate_yaml_schema(
            data,
            required_keys=["host"],
            strict=False,
        )

        assert is_valid is True


class TestLoadConfigYaml:
    """Tests for load_config_yaml()."""

    def test_loads_from_default_path(self, tmp_path):
        filepath = tmp_path / "config.yaml"
        filepath.write_text("key: value")

        result = load_config_yaml(filepath)

        assert result == {"key": "value"}

    def test_env_var_override(self, tmp_path, monkeypatch):
        default_path = tmp_path / "default.yaml"
        override_path = tmp_path / "override.yaml"

        default_path.write_text("source: default")
        override_path.write_text("source: override")

        monkeypatch.setenv("TEST_CONFIG_PATH", str(override_path))

        result = load_config_yaml(default_path, env_var="TEST_CONFIG_PATH")

        assert result == {"source": "override"}

    def test_with_defaults(self, tmp_path):
        filepath = tmp_path / "config.yaml"
        filepath.write_text("key: value")

        result = load_config_yaml(
            filepath,
            defaults={"key": "default", "other": "value"},
        )

        assert result["key"] == "value"
        assert result["other"] == "value"

    def test_missing_file_returns_empty(self, tmp_path):
        filepath = tmp_path / "nonexistent.yaml"

        result = load_config_yaml(filepath)

        assert result == {}


class TestYAMLLoadError:
    """Tests for YAMLLoadError exception."""

    def test_is_exception(self):
        error = YAMLLoadError("test error")

        assert isinstance(error, Exception)

    def test_message(self):
        error = YAMLLoadError("test message")

        assert str(error) == "test message"
