"""File format utilities for training scripts.

Provides utilities for working with common file formats:
- JSONL files with automatic gzip detection
- JSON files with error handling and atomic writes
- Gzip compression detection and handling

Usage:
    from scripts.lib.file_formats import (
        is_gzip_file,
        open_jsonl_file,
        read_jsonl_lines,
        load_json,
        save_json,
        load_json_if_exists,
    )

    # Automatic gzip detection
    if is_gzip_file(path):
        print("File is gzip-compressed")

    # Read JSONL with auto-detection
    with open_jsonl_file(path) as f:
        for line in f:
            data = json.loads(line)

    # JSON load/save with error handling
    config = load_json(path, default={})
    save_json(path, config, atomic=True)
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, Optional, TextIO, Union

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Gzip Detection
# =============================================================================


def is_gzip_file(filepath: Union[str, Path]) -> bool:
    """Check if a file is gzip-compressed by reading magic bytes.

    Args:
        filepath: Path to the file to check

    Returns:
        True if file is gzip-compressed, False otherwise
    """
    try:
        with open(filepath, "rb") as f:
            magic = f.read(2)
            return magic == b"\x1f\x8b"  # Gzip magic number
    except (IOError, OSError):
        return False


# =============================================================================
# JSONL File Handling
# =============================================================================


@contextmanager
def open_jsonl_file(filepath: Union[str, Path]) -> Generator[TextIO, None, None]:
    """Open a JSONL file, automatically detecting gzip compression.

    Args:
        filepath: Path to the JSONL file

    Yields:
        File-like object for reading lines

    Example:
        with open_jsonl_file("data.jsonl.gz") as f:
            for line in f:
                record = json.loads(line)
    """
    filepath = Path(filepath)

    if is_gzip_file(filepath):
        handle = gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    else:
        handle = open(filepath, "r", encoding="utf-8", errors="replace")

    try:
        yield handle
    finally:
        handle.close()


def read_jsonl_lines(
    filepath: Union[str, Path],
    limit: Optional[int] = None,
    skip_invalid: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Read and parse lines from a JSONL file.

    Args:
        filepath: Path to the JSONL file
        limit: Maximum number of records to return (None = no limit)
        skip_invalid: If True, skip invalid JSON lines; if False, raise

    Yields:
        Parsed JSON objects

    Example:
        for record in read_jsonl_lines("games.jsonl", limit=1000):
            process(record)
    """
    count = 0

    with open_jsonl_file(filepath) as f:
        for line_num, line in enumerate(f, start=1):
            if limit is not None and count >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                count += 1
                yield obj
            except json.JSONDecodeError as e:
                if skip_invalid:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                else:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e


def count_jsonl_lines(filepath: Union[str, Path]) -> int:
    """Count the number of non-empty lines in a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        Number of non-empty lines
    """
    count = 0
    with open_jsonl_file(filepath) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_jsonl_lines(
    filepath: Union[str, Path],
    records: Iterator[Dict[str, Any]],
    compress: bool = False,
    append: bool = False,
) -> int:
    """Write records to a JSONL file.

    Args:
        filepath: Path to the output file
        records: Iterator of dictionaries to write
        compress: If True, gzip compress the output
        append: If True, append to existing file; otherwise overwrite

    Returns:
        Number of records written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    mode = "at" if append else "wt"

    if compress or filepath.suffix == ".gz":
        handle = gzip.open(filepath, mode, encoding="utf-8")
    else:
        handle = open(filepath, mode, encoding="utf-8")

    count = 0
    try:
        for record in records:
            json.dump(record, handle, separators=(",", ":"))
            handle.write("\n")
            count += 1
    finally:
        handle.close()

    return count


# =============================================================================
# JSON File Handling
# =============================================================================


def load_json(
    filepath: Union[str, Path],
    default: Any = None,
    encoding: str = "utf-8",
) -> Any:
    """Load JSON from a file with error handling.

    Args:
        filepath: Path to the JSON file
        default: Value to return if file doesn't exist or is invalid
        encoding: File encoding

    Returns:
        Parsed JSON data, or default if file doesn't exist/is invalid

    Example:
        config = load_json("config.json", default={})
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return default

    try:
        with open(filepath, "r", encoding=encoding) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load JSON from {filepath}: {e}")
        return default


def load_json_strict(
    filepath: Union[str, Path],
    encoding: str = "utf-8",
) -> Any:
    """Load JSON from a file, raising on errors.

    Args:
        filepath: Path to the JSON file
        encoding: File encoding

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding=encoding) as f:
        return json.load(f)


def load_json_if_exists(
    filepath: Union[str, Path],
    default: Any = None,
) -> Any:
    """Load JSON only if file exists, otherwise return default.

    This is a convenience alias for load_json() that makes intent clearer.

    Args:
        filepath: Path to the JSON file
        default: Value to return if file doesn't exist

    Returns:
        Parsed JSON data, or default if file doesn't exist
    """
    return load_json(filepath, default=default)


def save_json(
    filepath: Union[str, Path],
    data: Any,
    indent: int = 2,
    atomic: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Save data to a JSON file.

    Args:
        filepath: Path to the output file
        data: Data to serialize to JSON
        indent: Indentation level (None for compact)
        atomic: If True, write to temp file first then rename (safer)
        encoding: File encoding

    Example:
        save_json("config.json", {"key": "value"}, atomic=True)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        # Write to temp file, then rename (atomic on POSIX)
        dir_path = filepath.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding=encoding) as f:
                json.dump(data, f, indent=indent)
                f.write("\n")  # Trailing newline
            shutil.move(tmp_path, filepath)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    else:
        with open(filepath, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent)
            f.write("\n")


def save_json_compact(
    filepath: Union[str, Path],
    data: Any,
    atomic: bool = True,
) -> None:
    """Save data to a JSON file in compact format (no indentation).

    Args:
        filepath: Path to the output file
        data: Data to serialize to JSON
        atomic: If True, write to temp file first then rename
    """
    save_json(filepath, data, indent=None, atomic=atomic)


def update_json(
    filepath: Union[str, Path],
    updates: Dict[str, Any],
    atomic: bool = True,
) -> Dict[str, Any]:
    """Load a JSON file, update it with new values, and save.

    Args:
        filepath: Path to the JSON file
        updates: Dictionary of updates to apply
        atomic: If True, use atomic write

    Returns:
        The updated data

    Example:
        update_json("config.json", {"version": 2})
    """
    data = load_json(filepath, default={})

    if isinstance(data, dict):
        data.update(updates)
    else:
        raise TypeError(f"Cannot update non-dict JSON: {type(data)}")

    save_json(filepath, data, atomic=atomic)
    return data


# =============================================================================
# File Information
# =============================================================================


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes.

    Args:
        filepath: Path to the file

    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0.0
    return filepath.stat().st_size / (1024 * 1024)


def get_uncompressed_size_estimate(filepath: Union[str, Path]) -> int:
    """Estimate uncompressed size of a gzip file.

    Note: This uses the gzip footer which stores the original size mod 2^32,
    so it's only accurate for files < 4GB.

    Args:
        filepath: Path to the gzip file

    Returns:
        Estimated uncompressed size in bytes, or file size if not gzip
    """
    filepath = Path(filepath)

    if not is_gzip_file(filepath):
        return filepath.stat().st_size

    try:
        with open(filepath, "rb") as f:
            f.seek(-4, 2)  # Seek to last 4 bytes
            size_bytes = f.read(4)
            return int.from_bytes(size_bytes, "little")
    except (IOError, OSError):
        return filepath.stat().st_size


__all__ = [
    # Gzip detection
    "is_gzip_file",
    # JSONL handling
    "open_jsonl_file",
    "read_jsonl_lines",
    "count_jsonl_lines",
    "write_jsonl_lines",
    # JSON handling
    "load_json",
    "load_json_strict",
    "load_json_if_exists",
    "save_json",
    "save_json_compact",
    "update_json",
    # File info
    "get_file_size_mb",
    "get_uncompressed_size_estimate",
]
