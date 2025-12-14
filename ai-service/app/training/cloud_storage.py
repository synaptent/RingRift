"""Cloud storage abstraction for distributed self-play data generation.

This module provides a unified interface for writing game data to local files
or cloud storage (S3, GCS) for large-scale neural network training data
collection.

Usage:
    # Local storage (default)
    storage = get_storage("file:///path/to/output")

    # AWS S3
    storage = get_storage("s3://bucket/prefix")

    # Google Cloud Storage
    storage = get_storage("gs://bucket/prefix")

    # Write training samples
    storage.write_training_sample(sample)
    storage.flush()
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample for neural network training.

    Contains a game state with outcome labels suitable for supervised
    or reinforcement learning.

    Attributes:
        state_json: Serialized GameState as JSON string
        outcome: Game outcome from perspective of current_player
                 1.0 = win, 0.0 = loss, 0.5 = draw/timeout
        move_json: Optional move that was played (for policy learning)
        move_probs: Optional dict mapping move_id -> probability (for MCTS data)
        board_type: Board type string for filtering
        game_id: Source game identifier for provenance
        move_number: Move number within the game
        ply_to_end: Number of moves until game end (for value discounting)
        metadata: Additional metadata (source, engine_mode, etc.)
    """
    state_json: str
    outcome: float
    board_type: str
    game_id: str
    move_number: int
    ply_to_end: int
    move_json: Optional[str] = None
    move_probs: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "state": json.loads(self.state_json),
            "outcome": self.outcome,
            "board_type": self.board_type,
            "game_id": self.game_id,
            "move_number": self.move_number,
            "ply_to_end": self.ply_to_end,
        }
        if self.move_json:
            d["move"] = json.loads(self.move_json)
        if self.move_probs:
            d["move_probs"] = self.move_probs
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write_line(self, data: str) -> None:
        """Write a single line of data (JSONL format)."""
        pass

    @abstractmethod
    def write_training_sample(self, sample: TrainingSample) -> None:
        """Write a training sample."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered data."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about written data."""
        pass


class LocalFileStorage(StorageBackend):
    """Local file storage backend.

    Writes JSONL data to a local file, optionally compressed with gzip.
    """

    def __init__(
        self,
        path: str,
        compress: bool = False,
        buffer_size: int = 1000,
    ):
        """Initialize local file storage.

        Args:
            path: Output file path
            compress: Whether to gzip compress the output
            buffer_size: Number of samples to buffer before flushing
        """
        self._path = Path(path)
        self._compress = compress
        self._buffer_size = buffer_size
        self._buffer: List[str] = []
        self._samples_written = 0
        self._bytes_written = 0

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Open file (gzip if requested)
        if compress:
            self._file = gzip.open(str(self._path), "at", encoding="utf-8")
        else:
            self._file = open(str(self._path), "a", encoding="utf-8")

    def write_line(self, data: str) -> None:
        """Write a single line."""
        self._buffer.append(data)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def write_training_sample(self, sample: TrainingSample) -> None:
        """Write a training sample."""
        self.write_line(sample.to_json())

    def flush(self) -> None:
        """Flush buffered data to file."""
        if not self._buffer:
            return

        for line in self._buffer:
            self._file.write(line)
            self._file.write("\n")
            self._bytes_written += len(line) + 1
            self._samples_written += 1

        self._buffer.clear()
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self.flush()
        self._file.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get write statistics."""
        return {
            "backend": "local",
            "path": str(self._path),
            "samples_written": self._samples_written,
            "bytes_written": self._bytes_written,
            "compressed": self._compress,
        }


class S3Storage(StorageBackend):
    """AWS S3 storage backend.

    Buffers samples locally and uploads in batches to S3.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        buffer_size: int = 10000,
        compress: bool = True,
        partition_size: int = 100000,
    ):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for uploaded files
            buffer_size: Samples to buffer before uploading
            compress: Whether to gzip compress uploads
            partition_size: Samples per partition file
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )

        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._buffer_size = buffer_size
        self._compress = compress
        self._partition_size = partition_size

        self._s3 = boto3.client("s3")
        self._buffer: List[str] = []
        self._current_partition = 0
        self._samples_in_partition = 0
        self._total_samples = 0
        self._total_bytes = 0
        self._uploaded_files: List[str] = []

        # Worker ID for uniqueness across distributed workers
        self._worker_id = os.environ.get("WORKER_ID", str(uuid.uuid4())[:8])

    def write_line(self, data: str) -> None:
        """Write a single line."""
        self._buffer.append(data)
        self._samples_in_partition += 1

        if len(self._buffer) >= self._buffer_size:
            self._upload_buffer()

        if self._samples_in_partition >= self._partition_size:
            self._finalize_partition()

    def write_training_sample(self, sample: TrainingSample) -> None:
        """Write a training sample."""
        self.write_line(sample.to_json())

    def _upload_buffer(self) -> None:
        """Upload buffered data to S3."""
        if not self._buffer:
            return

        # Generate partition key
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = ".jsonl.gz" if self._compress else ".jsonl"
        key = f"{self._prefix}/{self._worker_id}/part_{self._current_partition:06d}_{timestamp}{ext}"

        # Prepare data
        data = "\n".join(self._buffer) + "\n"
        data_bytes = data.encode("utf-8")

        if self._compress:
            data_bytes = gzip.compress(data_bytes)

        # Upload to S3
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data_bytes,
                ContentType="application/x-ndjson",
                ContentEncoding="gzip" if self._compress else "identity",
            )
            self._total_samples += len(self._buffer)
            self._total_bytes += len(data_bytes)
            self._uploaded_files.append(f"s3://{self._bucket}/{key}")
            logger.info(
                f"Uploaded {len(self._buffer)} samples to s3://{self._bucket}/{key}"
            )
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

        self._buffer.clear()

    def _finalize_partition(self) -> None:
        """Finalize current partition and start a new one."""
        self._upload_buffer()
        self._current_partition += 1
        self._samples_in_partition = 0

    def flush(self) -> None:
        """Flush any remaining buffered data."""
        self._upload_buffer()

    def close(self) -> None:
        """Close storage and upload any remaining data."""
        self.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        return {
            "backend": "s3",
            "bucket": self._bucket,
            "prefix": self._prefix,
            "worker_id": self._worker_id,
            "total_samples": self._total_samples,
            "total_bytes": self._total_bytes,
            "partitions": self._current_partition + 1,
            "uploaded_files": self._uploaded_files,
            "compressed": self._compress,
        }


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend.

    Buffers samples locally and uploads in batches to GCS.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        buffer_size: int = 10000,
        compress: bool = True,
        partition_size: int = 100000,
    ):
        """Initialize GCS storage.

        Args:
            bucket: GCS bucket name
            prefix: Object prefix for uploaded files
            buffer_size: Samples to buffer before uploading
            compress: Whether to gzip compress uploads
            partition_size: Samples per partition file
        """
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )

        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/")
        self._buffer_size = buffer_size
        self._compress = compress
        self._partition_size = partition_size

        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket)
        self._buffer: List[str] = []
        self._current_partition = 0
        self._samples_in_partition = 0
        self._total_samples = 0
        self._total_bytes = 0
        self._uploaded_files: List[str] = []

        self._worker_id = os.environ.get("WORKER_ID", str(uuid.uuid4())[:8])

    def write_line(self, data: str) -> None:
        """Write a single line."""
        self._buffer.append(data)
        self._samples_in_partition += 1

        if len(self._buffer) >= self._buffer_size:
            self._upload_buffer()

        if self._samples_in_partition >= self._partition_size:
            self._finalize_partition()

    def write_training_sample(self, sample: TrainingSample) -> None:
        """Write a training sample."""
        self.write_line(sample.to_json())

    def _upload_buffer(self) -> None:
        """Upload buffered data to GCS."""
        if not self._buffer:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = ".jsonl.gz" if self._compress else ".jsonl"
        blob_name = f"{self._prefix}/{self._worker_id}/part_{self._current_partition:06d}_{timestamp}{ext}"

        data = "\n".join(self._buffer) + "\n"
        data_bytes = data.encode("utf-8")

        if self._compress:
            data_bytes = gzip.compress(data_bytes)

        try:
            blob = self._bucket.blob(blob_name)
            blob.upload_from_string(
                data_bytes,
                content_type="application/x-ndjson",
            )
            self._total_samples += len(self._buffer)
            self._total_bytes += len(data_bytes)
            self._uploaded_files.append(f"gs://{self._bucket_name}/{blob_name}")
            logger.info(
                f"Uploaded {len(self._buffer)} samples to gs://{self._bucket_name}/{blob_name}"
            )
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise

        self._buffer.clear()

    def _finalize_partition(self) -> None:
        """Finalize current partition and start a new one."""
        self._upload_buffer()
        self._current_partition += 1
        self._samples_in_partition = 0

    def flush(self) -> None:
        """Flush any remaining buffered data."""
        self._upload_buffer()

    def close(self) -> None:
        """Close storage and upload any remaining data."""
        self.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        return {
            "backend": "gcs",
            "bucket": self._bucket_name,
            "prefix": self._prefix,
            "worker_id": self._worker_id,
            "total_samples": self._total_samples,
            "total_bytes": self._total_bytes,
            "partitions": self._current_partition + 1,
            "uploaded_files": self._uploaded_files,
            "compressed": self._compress,
        }


def get_storage(
    uri: str,
    buffer_size: int = 10000,
    compress: bool = True,
    partition_size: int = 100000,
) -> StorageBackend:
    """Get a storage backend for the given URI.

    Args:
        uri: Storage URI:
             - file:///path/to/file.jsonl or /path/to/file.jsonl (local)
             - s3://bucket/prefix (AWS S3)
             - gs://bucket/prefix (Google Cloud Storage)
        buffer_size: Number of samples to buffer before writing/uploading
        compress: Whether to compress output (gzip)
        partition_size: Samples per partition file (for cloud storage)

    Returns:
        StorageBackend instance for the specified URI
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "" or scheme == "file":
        # Local file storage
        # Combine netloc and path (for file://$HOME/path format)
        # then expand ~ and environment variables
        path = parsed.path or uri
        if parsed.netloc:
            # file://$HOME/path -> netloc='$HOME', path='/path'
            path = parsed.netloc + path
        path = os.path.expanduser(os.path.expandvars(path))
        return LocalFileStorage(
            path=path,
            compress=compress,
            buffer_size=buffer_size,
        )

    elif scheme == "s3":
        # AWS S3
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return S3Storage(
            bucket=bucket,
            prefix=prefix,
            buffer_size=buffer_size,
            compress=compress,
            partition_size=partition_size,
        )

    elif scheme == "gs":
        # Google Cloud Storage
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return GCSStorage(
            bucket=bucket,
            prefix=prefix,
            buffer_size=buffer_size,
            compress=compress,
            partition_size=partition_size,
        )

    else:
        raise ValueError(
            f"Unsupported storage scheme: {scheme}. "
            f"Use file://, s3://, or gs://"
        )
