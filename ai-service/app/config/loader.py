"""Unified Configuration Loader.

This module provides a single entry point for loading configuration files
with automatic format detection, environment variable overrides, validation,
and type conversion to dataclasses.

Usage:
    from app.config.loader import load_config, ConfigLoader

    # Auto-detect format from extension
    config = load_config("config/settings.yaml")
    config = load_config("config/settings.json")

    # Load into a typed dataclass
    from dataclasses import dataclass

    @dataclass
    class MyConfig:
        host: str = "localhost"
        port: int = 8080

    config = load_config("config/app.yaml", target=MyConfig)
    print(config.port)  # Type-safe access

    # With environment variable prefix
    config = load_config("config/app.yaml", env_prefix="MYAPP_")
    # MYAPP_HOST=remote.server will override config.host

Environment Variables (December 2025):
    - Config files can be overridden via environment variables
    - Use env_prefix to namespace your overrides
    - Environment values are type-coerced to match dataclass fields
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from app.utils.json_utils import load_json, save_json
from app.utils.yaml_utils import (
    ConfigDict,
    dump_yaml,
    load_yaml,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigLoadError",
    "ConfigLoader",
    "ConfigSource",
    "env_override",
    "load_config",
    "merge_configs",
    "save_config",
    "validate_config",
]

T = TypeVar("T")


class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""


class ConfigSource:
    """Configuration source with metadata."""

    def __init__(
        self,
        path: Union[str, Path],
        format: str | None = None,
        required: bool = True,
        env_var: str | None = None,
    ):
        """Initialize a config source.

        Args:
            path: Path to config file
            format: File format ("json", "yaml"). Auto-detected if None.
            required: Whether file must exist
            env_var: Environment variable that can override the path
        """
        self.path = Path(path)
        self.format = format or self._detect_format(self.path)
        self.required = required
        self.env_var = env_var

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Detect config format from file extension."""
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return "yaml"
        elif suffix == ".json":
            return "json"
        else:
            # Default to YAML for unknown extensions
            return "yaml"

    def get_path(self) -> Path:
        """Get the actual path, considering environment override."""
        if self.env_var:
            env_path = os.environ.get(self.env_var)
            if env_path:
                return Path(env_path)
        return self.path


def load_config(
    path: Union[str, Path],
    *,
    target: type[T] | None = None,
    defaults: dict[str, Any] | None = None,
    env_prefix: str | None = None,
    required: bool = True,
    validate: bool = True,
) -> Union[T, ConfigDict]:
    """Load configuration from a file with optional type conversion.

    This is the main entry point for loading configuration. It automatically
    detects the file format and can convert to typed dataclasses.

    Args:
        path: Path to the config file
        target: Optional dataclass type to convert to
        defaults: Default values to merge with loaded config
        env_prefix: Prefix for environment variable overrides
        required: Whether file must exist (False returns defaults or empty dict)
        validate: Whether to validate the config after loading

    Returns:
        Loaded configuration as dict or typed dataclass

    Examples:
        # Simple dictionary load
        config = load_config("config/app.yaml")

        # Load into dataclass
        @dataclass
        class AppConfig:
            debug: bool = False
            port: int = 8080

        config = load_config("config/app.yaml", target=AppConfig)
        print(config.debug)  # Type-safe
    """
    path = Path(path)
    source = ConfigSource(path, required=required)

    # Load raw config
    raw_config = _load_raw_config(source, defaults)

    if raw_config is None:
        if target:
            return target()  # Return default instance
        return {}

    # Apply environment overrides
    if env_prefix:
        raw_config = env_override(raw_config, env_prefix, target)

    # Convert to target type if specified
    if target:
        try:
            result = _dict_to_dataclass(raw_config, target)
            if validate:
                _validate_dataclass(result)
            return result
        except Exception as e:
            raise ConfigLoadError(f"Failed to convert config to {target.__name__}: {e}")

    return raw_config


def _load_raw_config(
    source: ConfigSource,
    defaults: dict[str, Any] | None = None,
) -> ConfigDict | None:
    """Load raw config from source."""
    actual_path = source.get_path()

    if not actual_path.exists():
        if source.required:
            raise ConfigLoadError(f"Config file not found: {actual_path}")
        return defaults.copy() if defaults else None

    try:
        if source.format == "json":
            data = load_json(actual_path)
        else:  # yaml
            data = load_yaml(actual_path, required=source.required)

        if data is None:
            return defaults.copy() if defaults else {}

        if defaults:
            return merge_configs(defaults, data)
        return data

    except Exception as e:
        raise ConfigLoadError(f"Failed to load config from {actual_path}: {e}")


def save_config(
    config: Union[dict[str, Any], Any],
    path: Union[str, Path],
    *,
    format: str | None = None,
) -> None:
    """Save configuration to a file.

    Args:
        config: Configuration dict or dataclass
        path: Output file path
        format: File format ("json", "yaml"). Auto-detected from path if None.
    """
    path = Path(path)
    format = format or ConfigSource._detect_format(path)

    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, type):
        from dataclasses import asdict
        config = asdict(config)

    if format == "json":
        save_json(path, config)
    else:
        dump_yaml(config, path)


def env_override(
    config: ConfigDict,
    prefix: str,
    target: type | None = None,
) -> ConfigDict:
    """Apply environment variable overrides to config.

    Environment variables are named as PREFIX_KEY_NAME (uppercase).
    Nested keys use underscores: PREFIX_NESTED_KEY becomes config["nested"]["key"]

    Args:
        config: Configuration dictionary to override
        prefix: Environment variable prefix (e.g., "MYAPP_")
        target: Optional dataclass type for type coercion

    Returns:
        Config with environment overrides applied
    """
    result = config.copy()

    # Get field types from target if available
    type_hints = {}
    if target and is_dataclass(target):
        for f in fields(target):
            type_hints[f.name] = f.type

    # Find matching environment variables
    prefix_upper = prefix.upper()
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix_upper):
            continue

        # Extract config key from env var name
        config_key = env_key[len(prefix_upper):].lower()

        # Handle nested keys (double underscore)
        if "__" in config_key:
            parts = config_key.split("__")
            _set_nested(result, parts, env_value, type_hints.get(parts[0]))
        else:
            # Coerce type if we know the target type
            result[config_key] = _coerce_type(
                env_value, type_hints.get(config_key)
            )

    return result


def _set_nested(
    config: dict[str, Any],
    keys: list[str],
    value: str,
    type_hint: type | None = None,
) -> None:
    """Set a nested config value."""
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = _coerce_type(value, type_hint)


def _coerce_type(value: str, type_hint: type | None) -> Any:
    """Coerce a string value to the target type."""
    if type_hint is None:
        # Try to infer type
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    # Handle Optional types
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Get first non-None type
        for arg in args:
            if arg is not type(None):
                type_hint = arg
                break

    # Coerce to target type
    if type_hint is bool:
        return value.lower() in ("true", "1", "yes")
    elif type_hint is int:
        return int(value)
    elif type_hint is float:
        return float(value)
    elif type_hint is str:
        return value
    elif type_hint is list or (origin and origin is list):
        # Comma-separated values
        return [v.strip() for v in value.split(",")]
    else:
        return value


def _dict_to_dataclass(data: dict[str, Any], target: type[T]) -> T:
    """Convert a dictionary to a dataclass instance.

    Handles nested dataclasses and provides defaults for missing fields.
    """
    if not is_dataclass(target):
        raise TypeError(f"{target} is not a dataclass")

    kwargs = {}
    for f in fields(target):
        if f.name not in data:
            # Use default if not in data
            continue

        value = data[f.name]

        # Handle nested dataclasses
        if is_dataclass(f.type) and isinstance(value, dict):
            value = _dict_to_dataclass(value, f.type)
        # Handle List of dataclasses
        elif get_origin(f.type) is list:
            args = get_args(f.type)
            if args and is_dataclass(args[0]) and isinstance(value, list):
                value = [_dict_to_dataclass(item, args[0]) for item in value]

        kwargs[f.name] = value

    return target(**kwargs)


def _validate_dataclass(obj: Any) -> None:
    """Basic validation for dataclass instances."""
    if not is_dataclass(obj):
        return

    for f in fields(obj):
        value = getattr(obj, f.name)
        # Check for None in non-Optional fields (simple check)
        if value is None and "Optional" not in str(f.type):
            logger.warning(f"Field {f.name} is None but may be required")


def merge_configs(
    base: ConfigDict,
    override: ConfigDict,
    *,
    deep: bool = True,
) -> ConfigDict:
    """Merge two configuration dictionaries.

    Args:
        base: Base configuration (lower priority)
        override: Override configuration (higher priority)
        deep: If True, recursively merge nested dicts

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def validate_config(
    config: Union[dict[str, Any], Any],
    required_keys: list[str] | None = None,
    validators: dict[str, Callable[[Any], bool]] | None = None,
) -> tuple[bool, list[str]]:
    """Validate configuration.

    Args:
        config: Configuration to validate
        required_keys: List of keys that must be present
        validators: Dict mapping keys to validation functions

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, type):
        from dataclasses import asdict
        config = asdict(config)

    # Check required keys
    required_keys = required_keys or []
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")
        elif config[key] is None:
            errors.append(f"Required key is None: {key}")

    # Run custom validators
    validators = validators or {}
    for key, validator in validators.items():
        if key in config:
            try:
                if not validator(config[key]):
                    errors.append(f"Validation failed for key: {key}")
            except Exception as e:
                errors.append(f"Validator error for {key}: {e}")

    return len(errors) == 0, errors


class ConfigLoader(Generic[T]):
    """Reusable config loader with caching.

    Usage:
        loader = ConfigLoader("config/app.yaml", target=AppConfig)
        config = loader.load()  # Cached after first load

        # Force reload
        config = loader.load(force_reload=True)

        # Access current config
        config = loader.current
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        target: type[T] | None = None,
        defaults: dict[str, Any] | None = None,
        env_prefix: str | None = None,
        auto_reload: bool = False,
    ):
        """Initialize the config loader.

        Args:
            path: Path to config file
            target: Optional dataclass type to convert to
            defaults: Default values
            env_prefix: Environment variable prefix for overrides
            auto_reload: If True, reload when file changes
        """
        self.path = Path(path)
        self.target = target
        self.defaults = defaults
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        self._cached: Union[T, ConfigDict] | None = None
        self._mtime: float | None = None

    def load(self, force_reload: bool = False) -> Union[T, ConfigDict]:
        """Load configuration, using cache if available.

        Args:
            force_reload: If True, bypass cache

        Returns:
            Loaded configuration
        """
        if self.auto_reload:
            current_mtime = self.path.stat().st_mtime if self.path.exists() else None
            if current_mtime != self._mtime:
                force_reload = True
                self._mtime = current_mtime

        if self._cached is not None and not force_reload:
            return self._cached

        self._cached = load_config(
            self.path,
            target=self.target,
            defaults=self.defaults,
            env_prefix=self.env_prefix,
        )
        return self._cached

    @property
    def current(self) -> Union[T, ConfigDict]:
        """Get current config, loading if needed."""
        if self._cached is None:
            self.load()
        return self._cached  # type: ignore

    def reload(self) -> Union[T, ConfigDict]:
        """Force reload configuration."""
        return self.load(force_reload=True)

    def invalidate(self) -> None:
        """Invalidate the cache."""
        self._cached = None
        self._mtime = None
