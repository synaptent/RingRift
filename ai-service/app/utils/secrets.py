"""Utilities for handling sensitive data securely.

This module provides functions for:
1. Sanitizing sensitive values for logging (masking API keys, tokens, etc.)
2. Safe string representation of objects that may contain secrets
3. Environment variable loading with validation

Usage:
    from app.utils.secrets import sanitize_for_log, mask_secret, SecretString

    # Mask a secret value
    api_key = os.environ.get("VAST_API_KEY", "")
    logger.info(f"Using API key: {mask_secret(api_key)}")  # "vst_...abc"

    # Sanitize a dict that may contain secrets
    config = {"api_key": "secret123", "host": "example.com"}
    logger.debug(f"Config: {sanitize_for_log(config)}")

    # Use SecretString for automatic masking
    secret = SecretString(api_key)
    logger.info(f"Key: {secret}")  # Automatically masked
"""

from __future__ import annotations

import os
import re
from typing import Any, Union

__all__ = [
    "SENSITIVE_KEY_PATTERNS",
    "SecretString",
    "get_env_masked",
    "is_sensitive_key",
    "load_secret_from_env",
    "mask_secret",
    "mask_secret_prefix",
    "sanitize_for_log",
    "sanitize_value",
]


# Keys that should be masked when sanitizing dicts/objects
SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "password",
    "passwd",
    "token",
    "auth",
    "authorization",
    "credential",
    "private_key",
    "privatekey",
    "access_key",
    "accesskey",
    "secret_key",
    "secretkey",
}

# Regex patterns for detecting secrets in string values
SECRET_VALUE_PATTERNS = [
    # API keys with common prefixes
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),  # OpenAI-style
    re.compile(r"(vst_[a-zA-Z0-9]{20,})", re.IGNORECASE),  # Vast.ai-style
    re.compile(r"(ghp_[a-zA-Z0-9]{36,})", re.IGNORECASE),  # GitHub PAT
    re.compile(r"(gho_[a-zA-Z0-9]{36,})", re.IGNORECASE),  # GitHub OAuth
    re.compile(r"(xoxb-[a-zA-Z0-9-]+)", re.IGNORECASE),  # Slack bot token
    re.compile(r"(Bearer\s+[a-zA-Z0-9._-]{20,})", re.IGNORECASE),  # Bearer tokens
    # AWS-style keys
    re.compile(r"(AKIA[0-9A-Z]{16})", re.IGNORECASE),  # AWS access key
    # Generic long alphanumeric strings (potential secrets)
    re.compile(r"([a-zA-Z0-9]{40,})"),  # 40+ char alphanumeric
]


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only the last few characters.

    Args:
        value: The secret value to mask
        visible_chars: Number of characters to show at the end (default: 4)

    Returns:
        Masked string like "***abc" or "[empty]" if value is None/empty

    Examples:
        >>> mask_secret("my_secret_api_key_12345")
        '***2345'
        >>> mask_secret("short")
        '***ort'
        >>> mask_secret("")
        '[empty]'
    """
    if not value:
        return "[empty]"

    if len(value) <= visible_chars:
        return "*" * len(value)

    return "***" + value[-visible_chars:]


def mask_secret_prefix(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only the first few characters.

    Args:
        value: The secret value to mask
        visible_chars: Number of characters to show at the start (default: 4)

    Returns:
        Masked string like "my_s***" or "[empty]" if value is None/empty
    """
    if not value:
        return "[empty]"

    if len(value) <= visible_chars:
        return "*" * len(value)

    return value[:visible_chars] + "***"


def is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests it contains sensitive data.

    Args:
        key: The key name to check

    Returns:
        True if the key likely contains sensitive data
    """
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS)


def sanitize_value(value: Any, key: str | None = None) -> Any:
    """Sanitize a value for safe logging.

    Args:
        value: The value to sanitize
        key: Optional key name (used to detect sensitive fields)

    Returns:
        Sanitized value safe for logging
    """
    # If key suggests sensitivity, mask the value
    if key and is_sensitive_key(key):
        if isinstance(value, str):
            return mask_secret(value)
        return "[REDACTED]"

    # For strings, check for embedded secrets
    if isinstance(value, str):
        result = value
        for pattern in SECRET_VALUE_PATTERNS:
            result = pattern.sub(lambda m: mask_secret(m.group(1)), result)
        return result

    # For dicts, recursively sanitize
    if isinstance(value, dict):
        return sanitize_for_log(value)

    # For lists, sanitize each element
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]

    return value


def sanitize_for_log(
    data: Union[dict[str, Any], Any],
    additional_sensitive_keys: set[str] | None = None,
) -> Union[dict[str, Any], Any]:
    """Sanitize a dictionary for safe logging by masking sensitive values.

    Args:
        data: Dictionary or other value to sanitize
        additional_sensitive_keys: Extra key names to treat as sensitive

    Returns:
        Sanitized copy of the data safe for logging

    Examples:
        >>> sanitize_for_log({"api_key": "secret123", "host": "example.com"})
        {'api_key': '***123', 'host': 'example.com'}
    """
    if not isinstance(data, dict):
        return sanitize_value(data)

    sensitive_keys = SENSITIVE_KEY_PATTERNS.copy()
    if additional_sensitive_keys:
        sensitive_keys.update(k.lower() for k in additional_sensitive_keys)

    result = {}
    for key, value in data.items():
        sanitized_value = sanitize_value(value, key)
        result[key] = sanitized_value

    return result


class SecretString:
    """A string wrapper that automatically masks its value when converted to string.

    Use this for sensitive values that might accidentally be logged or printed.

    Examples:
        >>> secret = SecretString("my_api_key_12345")
        >>> str(secret)
        '***2345'
        >>> secret.get_value()
        'my_api_key_12345'
        >>> f"Using key: {secret}"
        'Using key: ***2345'
    """

    def __init__(self, value: str, visible_chars: int = 4):
        """Initialize a SecretString.

        Args:
            value: The secret value
            visible_chars: Number of characters to show when masked
        """
        self._value = value
        self._visible_chars = visible_chars

    def get_value(self) -> str:
        """Get the actual secret value (use carefully!)."""
        return self._value

    def __str__(self) -> str:
        """Return masked version of the secret."""
        return mask_secret(self._value, self._visible_chars)

    def __repr__(self) -> str:
        """Return masked version of the secret."""
        return f"SecretString({mask_secret(self._value, self._visible_chars)})"

    def __bool__(self) -> bool:
        """Return True if the secret has a value."""
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        """Compare with another SecretString or string."""
        if isinstance(other, SecretString):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return False

    def __hash__(self) -> int:
        """Hash based on the actual value."""
        return hash(self._value)


def load_secret_from_env(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> SecretString | None:
    """Load a secret from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raise ValueError if not set and no default

    Returns:
        SecretString wrapping the value, or None if not set and not required

    Raises:
        ValueError: If required=True and value is not set
    """
    value = os.environ.get(key, default)

    if value is None and required:
        raise ValueError(f"Required environment variable {key} is not set")

    if value is None:
        return None

    return SecretString(value)


def get_env_masked(key: str, default: str = "") -> str:
    """Get an environment variable value in masked form for logging.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Masked version of the value
    """
    value = os.environ.get(key, default)
    return mask_secret(value)
