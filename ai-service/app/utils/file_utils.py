"""File operation utilities for safe, atomic file handling.

This module provides utilities for file operations that are safe against
crashes, interruptions, and concurrent access.

Usage:
    from app.utils.file_utils import atomic_write, read_safe, write_atomic

    # Atomic write with context manager
    with atomic_write("/path/to/file.txt") as f:
        f.write("content")

    # Safe read with fallback
    content = read_safe("/path/to/file.txt", default="")

    # Convenience functions
    write_atomic("/path/to/file.txt", "content")
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union


@contextmanager
def atomic_write(
    path: Union[str, Path],
    mode: str = "w",
    encoding: str = "utf-8",
    sync: bool = False,
) -> Generator:
    """Write to a file atomically using a temporary file and rename.

    This ensures that the file is either fully written or not modified at all,
    preventing partial writes on crash or interrupt.

    Args:
        path: Target file path
        mode: File mode ("w" for text, "wb" for binary)
        encoding: Text encoding (ignored for binary mode)
        sync: If True, fsync before rename (slower but safer)

    Yields:
        File object for writing

    Example:
        with atomic_write("config.json") as f:
            json.dump(data, f, indent=2)

        with atomic_write("data.bin", mode="wb") as f:
            f.write(binary_data)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory to ensure same filesystem
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        if "b" in mode:
            with os.fdopen(fd, mode) as f:
                yield f
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
        else:
            with os.fdopen(fd, mode, encoding=encoding) as f:
                yield f
                if sync:
                    f.flush()
                    os.fsync(f.fileno())

        # Atomic rename (same filesystem guaranteed)
        os.replace(tmp_path, path)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def write_atomic(
    path: Union[str, Path],
    content: Union[str, bytes],
    encoding: str = "utf-8",
    sync: bool = False,
) -> None:
    """Write content to a file atomically.

    Convenience function for atomic file writes without context manager.

    Args:
        path: Target file path
        content: Content to write (str or bytes)
        encoding: Text encoding (only used for str content)
        sync: If True, fsync before rename

    Example:
        write_atomic("config.txt", "key=value")
        write_atomic("data.bin", b"\\x00\\x01\\x02")
    """
    mode = "wb" if isinstance(content, bytes) else "w"
    with atomic_write(path, mode=mode, encoding=encoding, sync=sync) as f:
        f.write(content)


def read_safe(
    path: Union[str, Path],
    default: Optional[str] = None,
    encoding: str = "utf-8",
) -> Optional[str]:
    """Read a text file safely with fallback on error.

    Args:
        path: File to read
        default: Value to return if file doesn't exist or can't be read
        encoding: Text encoding

    Returns:
        File contents or default value

    Example:
        content = read_safe("config.txt", default="")
    """
    path = Path(path)
    if not path.exists():
        return default

    try:
        return path.read_text(encoding=encoding)
    except (IOError, OSError, UnicodeDecodeError):
        return default


def read_bytes_safe(
    path: Union[str, Path],
    default: Optional[bytes] = None,
) -> Optional[bytes]:
    """Read a binary file safely with fallback on error.

    Args:
        path: File to read
        default: Value to return if file doesn't exist or can't be read

    Returns:
        File contents or default value
    """
    path = Path(path)
    if not path.exists():
        return default

    try:
        return path.read_bytes()
    except (IOError, OSError):
        return default


def file_exists(path: Union[str, Path]) -> bool:
    """Check if a file exists (not a directory).

    Args:
        path: Path to check

    Returns:
        True if path exists and is a file
    """
    path = Path(path)
    return path.exists() and path.is_file()


def ensure_file_dir(path: Union[str, Path]) -> Path:
    """Ensure the parent directory of a file exists.

    Args:
        path: File path

    Returns:
        The path as a Path object
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backup_file(
    path: Union[str, Path],
    suffix: str = ".bak",
    overwrite: bool = True,
) -> Optional[Path]:
    """Create a backup copy of a file.

    Args:
        path: File to backup
        suffix: Suffix for backup file
        overwrite: Whether to overwrite existing backup

    Returns:
        Path to backup file, or None if source doesn't exist
    """
    import shutil

    path = Path(path)
    if not path.exists():
        return None

    backup_path = path.with_suffix(path.suffix + suffix)
    if backup_path.exists() and not overwrite:
        return backup_path

    shutil.copy2(path, backup_path)
    return backup_path


def remove_safe(path: Union[str, Path]) -> bool:
    """Remove a file, returning success status.

    Args:
        path: File to remove

    Returns:
        True if file was removed, False if it didn't exist
    """
    path = Path(path)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes, or 0 if file doesn't exist.

    Args:
        path: File path

    Returns:
        File size in bytes, or 0 if not found
    """
    path = Path(path)
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def get_file_mtime(path: Union[str, Path]) -> float:
    """Get file modification time as Unix timestamp.

    Args:
        path: File path

    Returns:
        Modification time as float, or 0.0 if not found
    """
    path = Path(path)
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


__all__ = [
    "atomic_write",
    "write_atomic",
    "read_safe",
    "read_bytes_safe",
    "file_exists",
    "ensure_file_dir",
    "backup_file",
    "remove_safe",
    "get_file_size",
    "get_file_mtime",
]
