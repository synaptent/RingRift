"""Datetime utilities for training scripts.

Provides utilities for common timestamp operations:
- File age calculations
- Elapsed time formatting
- Timestamp ID generation
- ISO timestamp parsing

Usage:
    from scripts.lib.datetime_utils import (
        get_file_age,
        find_files_older_than,
        format_elapsed_time,
        timestamp_id,
        parse_timestamp,
    )

    # Check file age
    age = get_file_age(path)
    if age.total_seconds() > 3600:  # older than 1 hour
        archive_file(path)

    # Find old files
    old_files = find_files_older_than(directory, hours=24)

    # Format elapsed time
    elapsed = format_elapsed_time(123.5)  # "2m 3s"

    # Generate unique timestamp ID
    job_id = f"job_{timestamp_id()}"  # "job_20241219_143052"
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Union

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# File Age Operations
# =============================================================================


def get_file_age(path: Union[str, Path]) -> timedelta:
    """Get the age of a file as a timedelta.

    Args:
        path: Path to the file

    Returns:
        timedelta representing the file's age (time since last modification)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    mtime = path.stat().st_mtime
    age_seconds = time.time() - mtime
    return timedelta(seconds=age_seconds)


def get_file_age_hours(path: Union[str, Path]) -> float:
    """Get the age of a file in hours.

    Args:
        path: Path to the file

    Returns:
        Age in hours as a float
    """
    return get_file_age(path).total_seconds() / 3600


def get_file_age_days(path: Union[str, Path]) -> float:
    """Get the age of a file in days.

    Args:
        path: Path to the file

    Returns:
        Age in days as a float
    """
    return get_file_age(path).total_seconds() / 86400


def is_file_older_than(
    path: Union[str, Path],
    hours: Optional[float] = None,
    days: Optional[float] = None,
    minutes: Optional[float] = None,
) -> bool:
    """Check if a file is older than the specified duration.

    Args:
        path: Path to the file
        hours: Age threshold in hours
        days: Age threshold in days
        minutes: Age threshold in minutes

    Returns:
        True if file is older than the specified duration

    Example:
        if is_file_older_than(path, hours=24):
            archive(path)
    """
    path = Path(path)
    if not path.exists():
        return False

    threshold_seconds = 0
    if hours:
        threshold_seconds += hours * 3600
    if days:
        threshold_seconds += days * 86400
    if minutes:
        threshold_seconds += minutes * 60

    if threshold_seconds == 0:
        raise ValueError("Must specify at least one of: hours, days, minutes")

    age_seconds = get_file_age(path).total_seconds()
    return age_seconds > threshold_seconds


def find_files_older_than(
    directory: Union[str, Path],
    hours: Optional[float] = None,
    days: Optional[float] = None,
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """Find files older than the specified duration.

    Args:
        directory: Directory to search
        hours: Age threshold in hours
        days: Age threshold in days
        pattern: Glob pattern for files (default: "*")
        recursive: If True, search recursively

    Returns:
        List of paths to files older than the threshold

    Example:
        old_logs = find_files_older_than("logs/", days=7, pattern="*.log")
    """
    directory = Path(directory)

    threshold_seconds = 0
    if hours:
        threshold_seconds += hours * 3600
    if days:
        threshold_seconds += days * 86400

    if threshold_seconds == 0:
        raise ValueError("Must specify at least one of: hours, days")

    cutoff_time = time.time() - threshold_seconds
    old_files = []

    glob_method = directory.rglob if recursive else directory.glob
    for path in glob_method(pattern):
        if path.is_file():
            try:
                if path.stat().st_mtime < cutoff_time:
                    old_files.append(path)
            except OSError:
                continue

    return sorted(old_files, key=lambda p: p.stat().st_mtime)


def iter_files_by_age(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
    newest_first: bool = False,
) -> Iterator[Path]:
    """Iterate over files sorted by modification time.

    Args:
        directory: Directory to search
        pattern: Glob pattern for files
        recursive: If True, search recursively
        newest_first: If True, return newest files first

    Yields:
        Paths sorted by modification time
    """
    directory = Path(directory)
    glob_method = directory.rglob if recursive else directory.glob

    files = []
    for path in glob_method(pattern):
        if path.is_file():
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:
                continue

    files.sort(reverse=newest_first)
    for _, path in files:
        yield path


# =============================================================================
# Elapsed Time Operations
# =============================================================================


def format_elapsed_time(seconds: float, precision: int = 0) -> str:
    """Format elapsed time as a human-readable string.

    Args:
        seconds: Elapsed time in seconds
        precision: Decimal places for seconds (default: 0)

    Returns:
        Formatted string like "2h 15m 30s" or "45.5s"

    Example:
        print(f"Completed in {format_elapsed_time(elapsed)}")
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")

    if precision > 0:
        parts.append(f"{secs:.{precision}f}s")
    else:
        parts.append(f"{int(secs)}s")

    return " ".join(parts)


def format_elapsed_time_short(seconds: float) -> str:
    """Format elapsed time in short notation (e.g., "1:23:45").

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted string like "1:23:45" or "5:30"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


class ElapsedTimer:
    """Context manager for timing operations.

    Example:
        with ElapsedTimer() as timer:
            do_work()
        print(f"Took {timer.elapsed:.2f}s")

        # Or with logging
        with ElapsedTimer("Processing data") as timer:
            process()
        # Logs: "Processing data completed in 1.5s"
    """

    def __init__(self, description: Optional[str] = None, log_on_exit: bool = True):
        """Initialize timer.

        Args:
            description: Optional description for logging
            log_on_exit: If True and description provided, log elapsed time on exit
        """
        self.description = description
        self.log_on_exit = log_on_exit
        self.start_time: float = 0
        self.end_time: float = 0

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Get formatted elapsed time string."""
        return format_elapsed_time(self.elapsed, precision=1)

    def __enter__(self) -> "ElapsedTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        if self.log_on_exit and self.description:
            logger.info(f"{self.description} completed in {self.elapsed_str}")


# =============================================================================
# Timestamp Generation
# =============================================================================


def timestamp_id() -> str:
    """Generate a timestamp-based ID in format YYYYMMDD_HHMMSS.

    Returns:
        Timestamp string suitable for filenames and IDs

    Example:
        job_id = f"training_{timestamp_id()}"  # "training_20241219_143052"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_id_ms() -> str:
    """Generate a timestamp-based ID with milliseconds.

    Returns:
        Timestamp string with milliseconds: YYYYMMDD_HHMMSS_fff

    Example:
        unique_id = timestamp_id_ms()  # "20241219_143052_123"
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"


def timestamp_for_log() -> str:
    """Generate a timestamp suitable for log messages.

    Returns:
        Timestamp string in format HH:MM:SS
    """
    return datetime.now().strftime("%H:%M:%S")


def timestamp_iso() -> str:
    """Generate an ISO 8601 timestamp.

    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def timestamp_iso_utc() -> str:
    """Generate an ISO 8601 timestamp in UTC.

    Returns:
        ISO format timestamp string with Z suffix
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# Timestamp Parsing
# =============================================================================


def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """Parse a timestamp from various formats.

    Handles:
    - ISO 8601 strings (with or without timezone)
    - Unix timestamps (int or float)
    - datetime objects (passed through)

    Args:
        value: Timestamp in various formats

    Returns:
        datetime object

    Raises:
        ValueError: If timestamp format is not recognized
    """
    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)

    if isinstance(value, str):
        # Try ISO format
        value = value.strip()

        # Handle Z suffix
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%Y%m%d_%H%M%S",
            "%Y-%m-%dT%H:%M:%S",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        raise ValueError(f"Cannot parse timestamp: {value}")

    raise ValueError(f"Unsupported timestamp type: {type(value)}")


def parse_timestamp_safe(
    value: Union[str, int, float, datetime, None],
    default: Optional[datetime] = None,
) -> Optional[datetime]:
    """Parse a timestamp, returning default on failure.

    Args:
        value: Timestamp in various formats
        default: Value to return on parse failure

    Returns:
        datetime object or default
    """
    if value is None:
        return default

    try:
        return parse_timestamp(value)
    except (ValueError, TypeError):
        return default


def timestamp_age(value: Union[str, int, float, datetime]) -> timedelta:
    """Calculate the age of a timestamp.

    Args:
        value: Timestamp in various formats

    Returns:
        timedelta representing the age
    """
    ts = parse_timestamp(value)

    # Make timezone-aware comparison
    now = datetime.now()
    if ts.tzinfo is not None:
        now = datetime.now(timezone.utc)
        if ts.tzinfo != timezone.utc:
            ts = ts.astimezone(timezone.utc)

    return now - ts


__all__ = [
    # File age operations
    "get_file_age",
    "get_file_age_hours",
    "get_file_age_days",
    "is_file_older_than",
    "find_files_older_than",
    "iter_files_by_age",
    # Elapsed time operations
    "format_elapsed_time",
    "format_elapsed_time_short",
    "ElapsedTimer",
    # Timestamp generation
    "timestamp_id",
    "timestamp_id_ms",
    "timestamp_for_log",
    "timestamp_iso",
    "timestamp_iso_utc",
    # Timestamp parsing
    "parse_timestamp",
    "parse_timestamp_safe",
    "timestamp_age",
]
