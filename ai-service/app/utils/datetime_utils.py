"""Unified datetime utilities for the RingRift AI service.

This module provides consistent datetime handling across all scripts.
All timestamps should be in UTC for consistency across distributed nodes.

Usage:
    from app.utils.datetime_utils import (
        utc_now,
        utc_timestamp,
        iso_now,
        parse_iso,
        time_ago,
        format_duration,
    )

    # Get current UTC time
    now = utc_now()

    # Get ISO-formatted timestamp with Z suffix
    ts = iso_now()  # "2025-12-19T10:30:00Z"

    # Parse ISO timestamp
    dt = parse_iso("2025-12-19T10:30:00Z")

    # Get datetime from N hours ago
    yesterday = time_ago(hours=24)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Union

__all__ = [
    "date_str",
    "format_age",
    "format_duration",
    "iso_now",
    "iso_now_ms",
    "parse_iso",
    "time_ago",
    "timestamp_str",
    "to_iso",
    "utc_now",
    "utc_timestamp",
]


def utc_now() -> datetime:
    """Get the current UTC datetime (timezone-aware).

    Returns:
        Current datetime in UTC with timezone info.

    Example:
        >>> now = utc_now()
        >>> now.tzinfo
        datetime.timezone.utc
    """
    return datetime.now(timezone.utc)


def utc_timestamp() -> float:
    """Get the current Unix timestamp (seconds since epoch).

    Returns:
        Current Unix timestamp as float.

    Example:
        >>> ts = utc_timestamp()
        >>> ts > 1700000000
        True
    """
    return datetime.now(timezone.utc).timestamp()


def iso_now() -> str:
    """Get the current UTC time as an ISO 8601 string with Z suffix.

    Returns:
        ISO 8601 formatted string like "2025-12-19T10:30:00Z"

    Example:
        >>> ts = iso_now()
        >>> ts.endswith("Z")
        True
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_now_ms() -> str:
    """Get the current UTC time as an ISO 8601 string with milliseconds.

    Returns:
        ISO 8601 formatted string like "2025-12-19T10:30:00.123Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def to_iso(dt: datetime) -> str:
    """Convert a datetime to ISO 8601 string with Z suffix.

    If the datetime is naive (no timezone), assumes UTC.

    Args:
        dt: Datetime to convert.

    Returns:
        ISO 8601 formatted string.
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        # Convert to UTC
        dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(timestamp: str) -> datetime:
    """Parse an ISO 8601 timestamp string to datetime.

    Handles both Z suffix and +00:00 offset formats.

    Args:
        timestamp: ISO 8601 formatted string.

    Returns:
        Timezone-aware datetime in UTC.

    Example:
        >>> dt = parse_iso("2025-12-19T10:30:00Z")
        >>> dt.year
        2025
    """
    # Handle Z suffix
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"

    # Try parsing with microseconds
    try:
        if "." in timestamp:
            # Has microseconds
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = datetime.fromisoformat(timestamp)
    except ValueError:
        # Fallback for common formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(timestamp, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse timestamp: {timestamp}")

    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def time_ago(
    *,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    seconds: float = 0,
) -> datetime:
    """Get a datetime in the past relative to now.

    Args:
        days: Number of days ago.
        hours: Number of hours ago.
        minutes: Number of minutes ago.
        seconds: Number of seconds ago.

    Returns:
        Timezone-aware datetime in UTC.

    Example:
        >>> yesterday = time_ago(days=1)
        >>> yesterday < utc_now()
        True
    """
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return datetime.now(timezone.utc) - delta


def format_duration(seconds: Union[int, float]) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string.

    Example:
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(90)
        '1m 30s'
        >>> format_duration(45)
        '45s'
    """
    if seconds < 0:
        return "0s"

    seconds = int(seconds)

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_age(dt: datetime) -> str:
    """Format the age of a datetime as a human-readable string.

    Args:
        dt: Datetime to compare to now.

    Returns:
        Human-readable age string.

    Example:
        >>> from datetime import timedelta
        >>> old = utc_now() - timedelta(hours=2, minutes=30)
        >>> "2h" in format_age(old)
        True
    """
    now = datetime.now(timezone.utc)

    # Make naive datetime UTC-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = now - dt

    if delta.total_seconds() < 0:
        return "in the future"

    return format_duration(delta.total_seconds())


def date_str(dt: datetime | None = None, format: str = "%Y%m%d") -> str:
    """Get a date string suitable for filenames.

    Args:
        dt: Datetime to format. Defaults to now.
        format: strftime format string.

    Returns:
        Formatted date string.

    Example:
        >>> len(date_str()) == 8
        True
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime(format)


def timestamp_str(dt: datetime | None = None) -> str:
    """Get a timestamp string suitable for filenames.

    Args:
        dt: Datetime to format. Defaults to now.

    Returns:
        Formatted timestamp string like "20251219_103000"
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")
