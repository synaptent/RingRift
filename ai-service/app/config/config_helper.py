"""Configuration Helper Utilities for standardized environment variable parsing.

December 30, 2025: Created for Priority 4 architectural improvement.
Provides consistent, validated, and well-logged environment variable parsing.

Problems solved:
- Inconsistent boolean parsing ("1", "true", "yes", "on" all now work consistently)
- Silent failures when parsing bad values (now logs warnings)
- No validation of numeric ranges (now supports min/max validation)
- Duplicated conversion code across 30+ Config classes

Usage:
    from app.config.config_helper import ConfigHelper

    # In a Config dataclass:
    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_MY") -> "MyConfig":
        return cls(
            enabled=ConfigHelper.get_bool(f"{prefix}_ENABLED", default=True),
            interval=ConfigHelper.get_int(f"{prefix}_INTERVAL", default=300, min_val=1),
            threshold=ConfigHelper.get_float(f"{prefix}_THRESHOLD", default=0.5, min_val=0.0, max_val=1.0),
            name=ConfigHelper.get_str(f"{prefix}_NAME", default="default"),
            ports=ConfigHelper.get_list_int(f"{prefix}_PORTS", default=[8770, 8780]),
        )

    # Or with prefix helper:
    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_MY") -> "MyConfig":
        helper = ConfigHelper(prefix)
        return cls(
            enabled=helper.bool("ENABLED", default=True),
            interval=helper.int("INTERVAL", default=300, min_val=1),
            threshold=helper.float("THRESHOLD", default=0.5, min_val=0.0, max_val=1.0),
        )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "ConfigHelper",
    "ConfigValidationError",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Values that are treated as True for boolean parsing
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
# Values that are treated as False for boolean parsing
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled", ""})


class ConfigValidationError(ValueError):
    """Raised when a configuration value fails validation."""

    def __init__(self, key: str, value: Any, reason: str):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(f"Config validation failed for {key}={value!r}: {reason}")


@dataclass(frozen=True)
class ConfigHelper:
    """Helper for parsing and validating environment variables.

    Can be used as static methods or instantiated with a prefix for convenience.

    Static usage:
        ConfigHelper.get_int("RINGRIFT_PORT", default=8770)

    Instance usage:
        helper = ConfigHelper("RINGRIFT_MY_DAEMON")
        helper.int("PORT", default=8770)  # Reads RINGRIFT_MY_DAEMON_PORT
    """

    prefix: str = ""

    # =========================================================================
    # Instance methods (use prefix)
    # =========================================================================

    def _key(self, suffix: str) -> str:
        """Build full key from prefix and suffix."""
        if not self.prefix:
            return suffix
        # Handle case where suffix already starts with underscore
        if suffix.startswith("_"):
            return f"{self.prefix}{suffix}"
        return f"{self.prefix}_{suffix}"

    def str(
        self,
        key: str,
        *,
        default: str = "",
        required: bool = False,
    ) -> str:
        """Get string value from environment."""
        return self.get_str(self._key(key), default=default, required=required)

    def int(
        self,
        key: str,
        *,
        default: int = 0,
        min_val: int | None = None,
        max_val: int | None = None,
        required: bool = False,
    ) -> int:
        """Get integer value from environment."""
        return self.get_int(
            self._key(key),
            default=default,
            min_val=min_val,
            max_val=max_val,
            required=required,
        )

    def float(
        self,
        key: str,
        *,
        default: float = 0.0,
        min_val: float | None = None,
        max_val: float | None = None,
        required: bool = False,
    ) -> float:
        """Get float value from environment."""
        return self.get_float(
            self._key(key),
            default=default,
            min_val=min_val,
            max_val=max_val,
            required=required,
        )

    def bool(
        self,
        key: str,
        *,
        default: bool = False,
    ) -> bool:
        """Get boolean value from environment."""
        return self.get_bool(self._key(key), default=default)

    def path(
        self,
        key: str,
        *,
        default: Path | str | None = None,
        must_exist: bool = False,
        required: bool = False,
    ) -> Path | None:
        """Get Path value from environment."""
        return self.get_path(
            self._key(key),
            default=default,
            must_exist=must_exist,
            required=required,
        )

    def list_str(
        self,
        key: str,
        *,
        default: list[str] | None = None,
        separator: str = ",",
    ) -> list[str]:
        """Get list of strings from environment (comma-separated by default)."""
        return self.get_list_str(self._key(key), default=default, separator=separator)

    def list_int(
        self,
        key: str,
        *,
        default: list[int] | None = None,
        separator: str = ",",
    ) -> list[int]:
        """Get list of integers from environment (comma-separated by default)."""
        return self.get_list_int(self._key(key), default=default, separator=separator)

    # =========================================================================
    # Static methods (use full key)
    # =========================================================================

    @staticmethod
    def get_str(
        key: str,
        *,
        default: str = "",
        required: bool = False,
    ) -> str:
        """Get string value from environment variable.

        Args:
            key: Environment variable name.
            default: Default value if not set.
            required: If True, raise ConfigValidationError if not set.

        Returns:
            The environment variable value or default.

        Raises:
            ConfigValidationError: If required and not set.
        """
        value = os.environ.get(key)

        if value is None:
            if required:
                raise ConfigValidationError(key, None, "Required value not set")
            return default

        return value

    @staticmethod
    def get_int(
        key: str,
        *,
        default: int = 0,
        min_val: int | None = None,
        max_val: int | None = None,
        required: bool = False,
    ) -> int:
        """Get integer value from environment variable with optional validation.

        Args:
            key: Environment variable name.
            default: Default value if not set or invalid.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
            required: If True, raise ConfigValidationError if not set.

        Returns:
            The parsed integer value, clamped to [min_val, max_val] if specified.

        Raises:
            ConfigValidationError: If required and not set.
        """
        raw = os.environ.get(key)

        if raw is None:
            if required:
                raise ConfigValidationError(key, None, "Required value not set")
            return default

        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                f"Config {key}={raw!r} is not a valid integer, using default={default}"
            )
            return default

        # Validation with clamping and warning
        if min_val is not None and value < min_val:
            logger.warning(
                f"Config {key}={value} is below minimum {min_val}, clamping to {min_val}"
            )
            value = min_val
        if max_val is not None and value > max_val:
            logger.warning(
                f"Config {key}={value} is above maximum {max_val}, clamping to {max_val}"
            )
            value = max_val

        return value

    @staticmethod
    def get_float(
        key: str,
        *,
        default: float = 0.0,
        min_val: float | None = None,
        max_val: float | None = None,
        required: bool = False,
    ) -> float:
        """Get float value from environment variable with optional validation.

        Args:
            key: Environment variable name.
            default: Default value if not set or invalid.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
            required: If True, raise ConfigValidationError if not set.

        Returns:
            The parsed float value, clamped to [min_val, max_val] if specified.

        Raises:
            ConfigValidationError: If required and not set.
        """
        raw = os.environ.get(key)

        if raw is None:
            if required:
                raise ConfigValidationError(key, None, "Required value not set")
            return default

        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                f"Config {key}={raw!r} is not a valid float, using default={default}"
            )
            return default

        # Validation with clamping and warning
        if min_val is not None and value < min_val:
            logger.warning(
                f"Config {key}={value} is below minimum {min_val}, clamping to {min_val}"
            )
            value = min_val
        if max_val is not None and value > max_val:
            logger.warning(
                f"Config {key}={value} is above maximum {max_val}, clamping to {max_val}"
            )
            value = max_val

        return value

    @staticmethod
    def get_bool(
        key: str,
        *,
        default: bool = False,
    ) -> bool:
        """Get boolean value from environment variable.

        Recognizes multiple true/false representations for convenience:
        - True: "1", "true", "yes", "on", "enabled" (case-insensitive)
        - False: "0", "false", "no", "off", "disabled", "" (case-insensitive)

        Unknown values are logged as warnings and default is returned.

        Args:
            key: Environment variable name.
            default: Default value if not set or unrecognized.

        Returns:
            The parsed boolean value.
        """
        raw = os.environ.get(key)

        if raw is None:
            return default

        normalized = raw.lower().strip()

        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

        logger.warning(
            f"Config {key}={raw!r} is not a recognized boolean value, using default={default}"
        )
        return default

    @staticmethod
    def get_path(
        key: str,
        *,
        default: Path | str | None = None,
        must_exist: bool = False,
        required: bool = False,
    ) -> Path | None:
        """Get Path value from environment variable.

        Args:
            key: Environment variable name.
            default: Default path if not set.
            must_exist: If True, log warning if path doesn't exist.
            required: If True, raise ConfigValidationError if not set.

        Returns:
            The Path object or None.

        Raises:
            ConfigValidationError: If required and not set.
        """
        raw = os.environ.get(key)

        if raw is None:
            if required:
                raise ConfigValidationError(key, None, "Required value not set")
            if default is None:
                return None
            return Path(default) if isinstance(default, str) else default

        path = Path(raw)

        if must_exist and not path.exists():
            logger.warning(f"Config {key}={raw!r} path does not exist")

        return path

    @staticmethod
    def get_list_str(
        key: str,
        *,
        default: list[str] | None = None,
        separator: str = ",",
    ) -> list[str]:
        """Get list of strings from environment variable.

        Args:
            key: Environment variable name.
            default: Default list if not set.
            separator: Separator between values (default: comma).

        Returns:
            List of string values, with whitespace stripped.
        """
        raw = os.environ.get(key)

        if raw is None:
            return default if default is not None else []

        # Split, strip whitespace, filter empty
        return [item.strip() for item in raw.split(separator) if item.strip()]

    @staticmethod
    def get_list_int(
        key: str,
        *,
        default: list[int] | None = None,
        separator: str = ",",
    ) -> list[int]:
        """Get list of integers from environment variable.

        Invalid integers are skipped with a warning.

        Args:
            key: Environment variable name.
            default: Default list if not set.
            separator: Separator between values (default: comma).

        Returns:
            List of integer values.
        """
        raw = os.environ.get(key)

        if raw is None:
            return default if default is not None else []

        result = []
        for item in raw.split(separator):
            item = item.strip()
            if not item:
                continue
            try:
                result.append(int(item))
            except ValueError:
                logger.warning(f"Config {key}: skipping invalid integer {item!r}")

        return result

    # =========================================================================
    # Convenience: Bulk loading
    # =========================================================================

    @staticmethod
    def load_dict(
        mapping: dict[str, tuple[str, type, Any]],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Load multiple environment variables at once.

        Args:
            mapping: Dict of {attr_name: (env_suffix, type, default)}
                     type can be: str, int, float, bool, Path
            prefix: Environment variable prefix.

        Returns:
            Dict of {attr_name: parsed_value}

        Example:
            values = ConfigHelper.load_dict({
                "enabled": ("ENABLED", bool, True),
                "interval": ("INTERVAL", int, 300),
                "threshold": ("THRESHOLD", float, 0.5),
            }, prefix="RINGRIFT_MY")
        """
        result = {}
        helper = ConfigHelper(prefix) if prefix else ConfigHelper()

        for attr_name, (suffix, value_type, default) in mapping.items():
            if value_type is str:
                result[attr_name] = helper.str(suffix, default=default)
            elif value_type is int:
                result[attr_name] = helper.int(suffix, default=default)
            elif value_type is float:
                result[attr_name] = helper.float(suffix, default=default)
            elif value_type is bool:
                result[attr_name] = helper.bool(suffix, default=default)
            elif value_type is Path:
                result[attr_name] = helper.path(suffix, default=default)
            else:
                # Fallback to string
                result[attr_name] = helper.str(suffix, default=str(default))

        return result
