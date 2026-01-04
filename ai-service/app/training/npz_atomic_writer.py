"""Atomic NPZ file writer utility.

January 4, 2026: Created to prevent NPZ corruption from partial writes.
Previously, 4 corrupted NPZ files caused training job failures with
"File is not a zip file" errors.

Problem: Direct np.savez() can leave corrupt files on:
- Process crash during write
- Power failure / system reboot
- Disk space exhaustion mid-write
- Network interruption during transfer

Solution: This module provides atomic NPZ writes via:
1. Write to temporary file (.npz.tmp)
2. Validate with numpy.load() to ensure integrity
3. Atomic rename to final destination
4. Include metadata for tracking (creation time, sample count)

Usage:
    from app.training.npz_atomic_writer import atomic_npz_write, AtomicNPZWriter

    # Simple usage
    atomic_npz_write(
        path="data/training/hex8_2p.npz",
        features=features_array,
        policy=policy_array,
        value=value_array,
    )

    # With context manager
    with AtomicNPZWriter("data/training/hex8_2p.npz") as writer:
        writer.add_array("features", features_array)
        writer.add_array("policy", policy_array)
        writer.add_array("value", value_array)
        writer.add_metadata(config_key="hex8_2p", source_db="canonical_hex8_2p.db")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AtomicWriteResult:
    """Result of an atomic NPZ write operation."""

    success: bool
    path: Path
    temp_path: Path | None = None
    sample_count: int = 0
    bytes_written: int = 0
    write_time_seconds: float = 0.0
    validation_passed: bool = False
    error: str | None = None


class AtomicNPZWriter:
    """Context manager for atomic NPZ file writing.

    Provides safe, atomic writes to NPZ files with:
    - Write to temp file first
    - Validation before rename
    - Automatic cleanup on failure
    - Metadata embedding

    Usage:
        with AtomicNPZWriter("output.npz") as writer:
            writer.add_array("features", features)
            writer.add_array("policy", policy)
            # If exception occurs, temp file is cleaned up
            # On success, atomic rename to final path
    """

    def __init__(self, path: str | Path, validate: bool = True):
        """Initialize atomic writer.

        Args:
            path: Final destination path for NPZ file
            validate: Whether to validate file after write (recommended)
        """
        self.path = Path(path)
        self.validate = validate
        self._arrays: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._temp_path: Path | None = None
        self._start_time: float | None = None

    def add_array(self, name: str, array: np.ndarray) -> None:
        """Add an array to be written.

        Args:
            name: Array name (key in NPZ file)
            array: NumPy array to write
        """
        self._arrays[name] = array

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to be embedded in the NPZ file.

        Metadata is stored as a JSON string in the '_metadata' array.

        Args:
            **kwargs: Arbitrary metadata key-value pairs
        """
        self._metadata.update(kwargs)

    def __enter__(self) -> "AtomicNPZWriter":
        """Enter context - prepare for writing."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - write file or clean up on error."""
        if exc_type is not None:
            # Exception occurred, clean up temp file if exists
            if self._temp_path and self._temp_path.exists():
                try:
                    self._temp_path.unlink()
                    logger.debug(f"Cleaned up temp file after error: {self._temp_path}")
                except OSError:
                    pass
            return

        # No exception - write the file
        try:
            result = self._write_atomic()
            if not result.success:
                raise IOError(f"Atomic write failed: {result.error}")
        except Exception:
            # Clean up temp file on write failure
            if self._temp_path and self._temp_path.exists():
                try:
                    self._temp_path.unlink()
                except OSError:
                    pass
            raise

    def _write_atomic(self) -> AtomicWriteResult:
        """Perform atomic write operation."""
        # Add metadata
        if self._metadata or self._arrays:
            meta = {
                "created_at": time.time(),
                "sample_count": self._get_sample_count(),
                **self._metadata,
            }
            self._arrays["_metadata"] = np.array([json.dumps(meta)])

        # Create parent directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory (for atomic rename)
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".npz.tmp",
            prefix=self.path.stem + "_",
            dir=self.path.parent,
        )
        os.close(fd)  # Close fd, we'll use np.savez
        self._temp_path = Path(temp_path_str)

        try:
            # Write to temp file
            np.savez_compressed(self._temp_path, **self._arrays)

            # Get file size
            bytes_written = self._temp_path.stat().st_size

            # Validate if requested
            validation_passed = True
            if self.validate:
                validation_passed = self._validate_npz(self._temp_path)
                if not validation_passed:
                    return AtomicWriteResult(
                        success=False,
                        path=self.path,
                        temp_path=self._temp_path,
                        error="Validation failed - file is corrupt",
                    )

            # Atomic rename
            self._temp_path.rename(self.path)

            write_time = time.time() - (self._start_time or time.time())
            logger.info(
                f"Atomic NPZ write: {self.path} "
                f"({bytes_written / 1024 / 1024:.1f} MB, "
                f"{self._get_sample_count()} samples, "
                f"{write_time:.2f}s)"
            )

            return AtomicWriteResult(
                success=True,
                path=self.path,
                sample_count=self._get_sample_count(),
                bytes_written=bytes_written,
                write_time_seconds=write_time,
                validation_passed=validation_passed,
            )

        except Exception as e:
            logger.error(f"Atomic write failed for {self.path}: {e}")
            # Clean up temp file
            if self._temp_path.exists():
                try:
                    self._temp_path.unlink()
                except OSError:
                    pass
            return AtomicWriteResult(
                success=False,
                path=self.path,
                temp_path=self._temp_path,
                error=str(e),
            )

    def _get_sample_count(self) -> int:
        """Get sample count from arrays."""
        for name in ["features", "policy", "value"]:
            if name in self._arrays:
                return len(self._arrays[name])
        if self._arrays:
            first_array = next(iter(self._arrays.values()))
            return len(first_array) if first_array.ndim > 0 else 0
        return 0

    def _validate_npz(self, path: Path) -> bool:
        """Validate NPZ file by attempting to load it.

        Args:
            path: Path to NPZ file

        Returns:
            True if file is valid, False otherwise
        """
        try:
            with np.load(path, allow_pickle=False) as data:
                # Access each array to verify integrity
                for key in data.files:
                    _ = data[key].shape
            return True
        except Exception as e:
            logger.warning(f"NPZ validation failed for {path}: {e}")
            return False


def atomic_npz_write(
    path: str | Path,
    validate: bool = True,
    **arrays: np.ndarray,
) -> AtomicWriteResult:
    """Write NPZ file atomically.

    Simple function interface for atomic NPZ writing.

    Args:
        path: Destination path for NPZ file
        validate: Whether to validate file after write
        **arrays: Named arrays to write (e.g., features=arr1, policy=arr2)

    Returns:
        AtomicWriteResult with success status and metadata

    Example:
        result = atomic_npz_write(
            "data/training/hex8_2p.npz",
            features=features,
            policy=policy,
            value=value,
        )
        if result.success:
            print(f"Wrote {result.sample_count} samples")
    """
    with AtomicNPZWriter(path, validate=validate) as writer:
        for name, array in arrays.items():
            writer.add_array(name, array)
    return AtomicWriteResult(
        success=True,
        path=Path(path),
        sample_count=len(next(iter(arrays.values()))) if arrays else 0,
    )


@contextmanager
def atomic_npz_context(
    path: str | Path,
    validate: bool = True,
):
    """Context manager for atomic NPZ writing.

    Provides access to the writer for adding arrays and metadata.

    Args:
        path: Destination path for NPZ file
        validate: Whether to validate file after write

    Yields:
        AtomicNPZWriter instance

    Example:
        with atomic_npz_context("output.npz") as writer:
            writer.add_array("features", features)
            writer.add_metadata(source="hex8_2p")
    """
    writer = AtomicNPZWriter(path, validate=validate)
    with writer:
        yield writer


def validate_npz_file(path: str | Path) -> tuple[bool, str | None]:
    """Validate an existing NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"

    try:
        with np.load(path, allow_pickle=False) as data:
            files = data.files
            total_size = 0
            for key in files:
                arr = data[key]
                total_size += arr.nbytes
        return True, None
    except Exception as e:
        return False, str(e)


def get_npz_metadata(path: str | Path) -> dict[str, Any] | None:
    """Extract metadata from an NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        Metadata dict if present, None otherwise
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            if "_metadata" in data.files:
                meta_json = str(data["_metadata"][0])
                return json.loads(meta_json)
    except Exception as e:
        logger.debug(f"Could not extract metadata from {path}: {e}")
    return None


__all__ = [
    "AtomicNPZWriter",
    "AtomicWriteResult",
    "atomic_npz_write",
    "atomic_npz_context",
    "validate_npz_file",
    "get_npz_metadata",
]
