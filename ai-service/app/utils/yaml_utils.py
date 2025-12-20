"""YAML loading utilities with error handling and validation.

This module provides standardized YAML loading with:
- Graceful error handling
- Optional schema validation
- Fallback defaults
- Consistent patterns across the codebase

Usage:
    from app.utils.yaml_utils import load_yaml, load_yaml_with_defaults, safe_load_yaml

    # Basic loading
    config = load_yaml("config/settings.yaml")

    # With defaults for missing file
    config = load_yaml_with_defaults("config/settings.yaml", {"key": "default"})

    # Safe loading that returns None on error
    config = safe_load_yaml("config/settings.yaml")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TypeVar, Union

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigDict",
    "YAMLLoadError",
    "dump_yaml",
    "dumps_yaml",
    "load_config_yaml",
    "load_yaml",
    "load_yaml_with_defaults",
    "safe_load_yaml",
    "validate_yaml_schema",
]

# Type for configuration dictionaries
ConfigDict = dict[str, Any]
T = TypeVar("T")

# Check for yaml availability
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore


class YAMLLoadError(Exception):
    """Raised when YAML loading fails."""
    pass


def _ensure_yaml_available() -> None:
    """Ensure pyyaml is installed."""
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for YAML loading. "
            "Install it with: pip install pyyaml"
        )


def load_yaml(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    required: bool = True,
) -> ConfigDict | None:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file
        encoding: File encoding (default: utf-8)
        required: If True, raises FileNotFoundError if file doesn't exist.
                  If False, returns None for missing files.

    Returns:
        Parsed YAML content as a dictionary, or None if file doesn't exist
        and required=False

    Raises:
        FileNotFoundError: If file doesn't exist and required=True
        YAMLLoadError: If YAML parsing fails
        ImportError: If pyyaml is not installed

    Example:
        config = load_yaml("config/settings.yaml")
        if config:
            threshold = config.get("threshold", 100)
    """
    _ensure_yaml_available()

    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"YAML file not found: {path}")
        return None

    try:
        with open(path, encoding=encoding) as f:
            data = yaml.safe_load(f)
            # Handle empty files
            return data if data is not None else {}
    except yaml.YAMLError as e:
        raise YAMLLoadError(f"Failed to parse YAML file {path}: {e}") from e
    except Exception as e:
        raise YAMLLoadError(f"Failed to load YAML file {path}: {e}") from e


def load_yaml_with_defaults(
    path: Union[str, Path],
    defaults: ConfigDict,
    *,
    encoding: str = "utf-8",
    merge_nested: bool = True,
) -> ConfigDict:
    """Load a YAML file, falling back to defaults if file doesn't exist.

    This function merges the loaded YAML with the provided defaults,
    with YAML values taking precedence.

    Args:
        path: Path to the YAML file
        defaults: Default configuration to use if file missing or incomplete
        encoding: File encoding (default: utf-8)
        merge_nested: If True, recursively merge nested dicts. If False,
                      top-level YAML values completely override defaults.

    Returns:
        Merged configuration dictionary

    Example:
        defaults = {"threshold": 100, "enabled": True}
        config = load_yaml_with_defaults("config/settings.yaml", defaults)
    """
    loaded = load_yaml(path, encoding=encoding, required=False)

    if loaded is None:
        return defaults.copy()

    if merge_nested:
        return _deep_merge(defaults, loaded)
    else:
        result = defaults.copy()
        result.update(loaded)
        return result


def safe_load_yaml(
    path: Union[str, Path],
    *,
    default: ConfigDict | None = None,
    encoding: str = "utf-8",
    log_errors: bool = True,
) -> ConfigDict | None:
    """Safely load a YAML file, returning default on any error.

    This is the most forgiving YAML loader - it never raises exceptions.

    Args:
        path: Path to the YAML file
        default: Value to return on any error (default: None)
        encoding: File encoding (default: utf-8)
        log_errors: Whether to log errors (default: True)

    Returns:
        Parsed YAML content, or default on any error

    Example:
        config = safe_load_yaml("config/settings.yaml", default={})
    """
    try:
        result = load_yaml(path, encoding=encoding, required=False)
        # Return default if file doesn't exist (load_yaml returns None)
        return result if result is not None else default
    except Exception as e:
        if log_errors:
            logger.warning(f"Failed to load YAML from {path}: {e}")
        return default


def dump_yaml(
    data: Any,
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    default_flow_style: bool = False,
    sort_keys: bool = False,
    allow_unicode: bool = True,
) -> None:
    """Write data to a YAML file.

    Args:
        data: Data to serialize to YAML
        path: Output file path
        encoding: File encoding (default: utf-8)
        default_flow_style: Use flow style for all collections (default: False)
        sort_keys: Sort dictionary keys (default: False)
        allow_unicode: Allow unicode characters (default: True)

    Raises:
        ImportError: If pyyaml is not installed
        IOError: If file cannot be written

    Example:
        dump_yaml({"key": "value"}, "config/output.yaml")
    """
    _ensure_yaml_available()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding=encoding) as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=allow_unicode,
        )


def dumps_yaml(
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    allow_unicode: bool = True,
) -> str:
    """Serialize data to a YAML string.

    Args:
        data: Data to serialize
        default_flow_style: Use flow style for all collections (default: False)
        sort_keys: Sort dictionary keys (default: False)
        allow_unicode: Allow unicode characters (default: True)

    Returns:
        YAML string representation

    Example:
        yaml_str = dumps_yaml({"key": "value"})
    """
    _ensure_yaml_available()

    return yaml.safe_dump(
        data,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge two dictionaries.

    Args:
        base: Base dictionary (lower priority)
        override: Override dictionary (higher priority)

    Returns:
        Merged dictionary with override values taking precedence
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def validate_yaml_schema(
    data: ConfigDict,
    required_keys: list[str] | None = None,
    optional_keys: list[str] | None = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate YAML data against a simple schema.

    Args:
        data: Parsed YAML data to validate
        required_keys: List of keys that must be present
        optional_keys: List of keys that may be present (used with strict=True)
        strict: If True, reject unknown keys not in required_keys or optional_keys

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Example:
        is_valid, errors = validate_yaml_schema(
            config,
            required_keys=["host", "port"],
            optional_keys=["timeout"],
            strict=True
        )
        if not is_valid:
            for error in errors:
                print(f"Config error: {error}")
    """
    errors = []
    required_keys = required_keys or []
    optional_keys = optional_keys or []

    # Check required keys
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Check for unknown keys in strict mode
    if strict:
        allowed_keys = set(required_keys) | set(optional_keys)
        for key in data:
            if key not in allowed_keys:
                errors.append(f"Unknown key: {key}")

    return len(errors) == 0, errors


# Convenience function for loading config with environment variable override
def load_config_yaml(
    default_path: Union[str, Path],
    env_var: str | None = None,
    defaults: ConfigDict | None = None,
) -> ConfigDict:
    """Load configuration YAML with environment variable path override.

    This is a convenience function for the common pattern of loading
    config from a default path that can be overridden by an environment
    variable.

    Args:
        default_path: Default path to config file
        env_var: Environment variable that can override the path
        defaults: Default values to merge with loaded config

    Returns:
        Loaded and merged configuration

    Example:
        config = load_config_yaml(
            "config/settings.yaml",
            env_var="MY_APP_CONFIG",
            defaults={"debug": False}
        )
    """
    path = default_path
    if env_var:
        path = os.environ.get(env_var, str(default_path))

    if defaults:
        return load_yaml_with_defaults(path, defaults)
    else:
        return load_yaml(path, required=False) or {}
