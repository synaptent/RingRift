"""Checksum computation utilities.

This module provides standardized checksum/hash computation for:
- File integrity verification
- Data deduplication
- Transfer verification

Usage:
    from app.utils.checksum_utils import (
        compute_file_checksum,
        compute_bytes_checksum,
        compute_string_checksum,
    )

    # File checksum (full SHA256)
    checksum = compute_file_checksum(Path("model.pt"))

    # Truncated checksum for IDs
    short_hash = compute_file_checksum(Path("model.pt"), truncate=16)

    # Bytes checksum
    checksum = compute_bytes_checksum(data)

    # String checksum (for deduplication)
    hash_id = compute_string_checksum(content, truncate=32)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

# Default chunk size for file reading (8KB - good balance for most files)
DEFAULT_CHUNK_SIZE = 8192

# Large chunk size for big files (64KB - better for large files)
LARGE_CHUNK_SIZE = 65536


def compute_file_checksum(
    path: Union[str, Path],
    *,
    algorithm: str = "sha256",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    truncate: Optional[int] = None,
    return_empty_for_missing: bool = True,
) -> str:
    """Compute checksum of a file.

    Args:
        path: Path to the file
        algorithm: Hash algorithm (sha256, sha1, md5). Default: sha256
        chunk_size: Size of chunks to read. Default: 8192.
                   Use LARGE_CHUNK_SIZE (65536) for large files.
        truncate: If set, truncate the hex digest to this many characters.
                  Common values: 16 (short ID), 32 (medium), None (full 64)
        return_empty_for_missing: If True, return "" for missing files.
                                  If False, raise FileNotFoundError.

    Returns:
        Hex-encoded hash string, possibly truncated

    Raises:
        FileNotFoundError: If file doesn't exist and return_empty_for_missing=False
        ValueError: If algorithm is not supported

    Example:
        # Full SHA256 checksum
        checksum = compute_file_checksum(Path("data.db"))

        # Short hash for checkpoint IDs
        short_hash = compute_file_checksum(checkpoint_path, truncate=16)

        # Large file with bigger chunks
        checksum = compute_file_checksum(
            large_file,
            chunk_size=LARGE_CHUNK_SIZE
        )
    """
    path = Path(path)

    if not path.exists():
        if return_empty_for_missing:
            return ""
        raise FileNotFoundError(f"File not found: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    digest = hasher.hexdigest()
    return digest[:truncate] if truncate else digest


def compute_bytes_checksum(
    data: bytes,
    *,
    algorithm: str = "sha256",
    truncate: Optional[int] = None,
) -> str:
    """Compute checksum of bytes data.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm (sha256, sha1, md5). Default: sha256
        truncate: If set, truncate the hex digest to this many characters

    Returns:
        Hex-encoded hash string, possibly truncated

    Example:
        checksum = compute_bytes_checksum(file_content)
        short_id = compute_bytes_checksum(data, truncate=32)
    """
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(data)
    digest = hasher.hexdigest()
    return digest[:truncate] if truncate else digest


def compute_string_checksum(
    content: str,
    *,
    algorithm: str = "sha256",
    truncate: Optional[int] = None,
    encoding: str = "utf-8",
) -> str:
    """Compute checksum of a string.

    Args:
        content: String to hash
        algorithm: Hash algorithm (sha256, sha1, md5). Default: sha256
        truncate: If set, truncate the hex digest to this many characters
        encoding: String encoding. Default: utf-8

    Returns:
        Hex-encoded hash string, possibly truncated

    Example:
        # Deduplication hash
        dedup_id = compute_string_checksum(json_content, truncate=32)
    """
    return compute_bytes_checksum(
        content.encode(encoding),
        algorithm=algorithm,
        truncate=truncate,
    )


def verify_file_checksum(
    path: Union[str, Path],
    expected_checksum: str,
    *,
    algorithm: str = "sha256",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Verify a file matches an expected checksum.

    Handles truncated checksums by comparing only the prefix.

    Args:
        path: Path to the file
        expected_checksum: Expected hex-encoded checksum (may be truncated)
        algorithm: Hash algorithm used. Default: sha256
        chunk_size: Size of chunks to read

    Returns:
        True if checksum matches, False otherwise

    Example:
        if verify_file_checksum(downloaded_file, expected_hash):
            process_file(downloaded_file)
        else:
            raise IntegrityError("Checksum mismatch")
    """
    if not expected_checksum:
        return False

    actual = compute_file_checksum(
        path,
        algorithm=algorithm,
        chunk_size=chunk_size,
        truncate=len(expected_checksum),
        return_empty_for_missing=True,
    )

    return actual == expected_checksum


def compute_content_id(
    game_id: str,
    data: str,
    *,
    truncate: int = 32,
) -> str:
    """Compute a content ID for deduplication.

    Combines a game ID with JSON data to create a unique content identifier.

    Args:
        game_id: Unique game identifier
        data: Serialized data (typically JSON string)
        truncate: Truncate hash to this length. Default: 32

    Returns:
        Truncated hash suitable for deduplication

    Example:
        content_id = compute_content_id(game_id, json.dumps(data, sort_keys=True))
    """
    content = f"{game_id}:{data}"
    return compute_string_checksum(content, truncate=truncate)
