"""Storage backend implementations for cloud and local file operations.

This module provides a unified interface for file operations across:
- Local filesystem
- AWS S3
- Google Cloud Storage

Usage:
    from app.storage import get_storage_backend

    # Local storage (default)
    storage = get_storage_backend("local", base_path="/data")

    # AWS S3
    storage = get_storage_backend("s3", bucket="my-bucket", prefix="ringrift-ai")

    # Google Cloud Storage
    storage = get_storage_backend("gcs", bucket="my-bucket", prefix="ringrift-ai")

    # Upload a file
    storage.upload("/local/model.pth", "models/v1/model.pth")

    # Download a file
    storage.download("models/v1/model.pth", "/local/model.pth")

    # List files
    files = storage.list("models/")

Environment variables:
    STORAGE_BACKEND: local, s3, gcs (default: local)
    STORAGE_BUCKET: S3/GCS bucket name
    STORAGE_PREFIX: Key prefix for cloud storage
    STORAGE_BASE_PATH: Base path for local storage
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Upload a local file to remote storage.

        Args:
            local_path: Path to local file
            remote_key: Remote key/path for the file
        """
        pass

    @abstractmethod
    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Download a file from remote storage to local path.

        Args:
            remote_key: Remote key/path of the file
            local_path: Local path to save the file
        """
        pass

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        """List files with the given prefix.

        Args:
            prefix: Prefix to filter files

        Returns:
            List of file keys matching the prefix
        """
        pass

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if a file exists.

        Args:
            remote_key: Remote key/path to check

        Returns:
            True if file exists
        """
        pass

    @abstractmethod
    def delete(self, remote_key: str) -> None:
        """Delete a file from storage.

        Args:
            remote_key: Remote key/path to delete
        """
        pass

    def download_if_newer(
        self, remote_key: str, local_path: str | Path
    ) -> bool:
        """Download file only if remote is newer than local.

        Args:
            remote_key: Remote key/path
            local_path: Local path to save

        Returns:
            True if file was downloaded, False if local is up-to-date
        """
        local_path = Path(local_path)
        if not local_path.exists():
            self.download(remote_key, local_path)
            return True
        # Default implementation: always download
        # Subclasses can optimize with metadata comparison
        self.download(remote_key, local_path)
        return True


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str | Path = "."):
        """Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self._base_path = Path(base_path).resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve a key to a full path."""
        return self._base_path / key

    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Copy a file to storage location."""
        local_path = Path(local_path)
        dest_path = self._resolve_path(remote_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.resolve() != dest_path.resolve():
            shutil.copy2(local_path, dest_path)
            logger.debug(f"Copied {local_path} to {dest_path}")

    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Copy a file from storage to local path."""
        source_path = self._resolve_path(remote_key)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        if source_path.resolve() != local_path.resolve():
            shutil.copy2(source_path, local_path)
            logger.debug(f"Copied {source_path} to {local_path}")

    def list(self, prefix: str = "") -> List[str]:
        """List files with prefix."""
        search_path = self._resolve_path(prefix)

        if search_path.is_file():
            return [prefix]

        if not search_path.exists():
            # Try as a prefix pattern
            parent = search_path.parent
            pattern = search_path.name + "*"
            if parent.exists():
                results = []
                for path in parent.rglob(pattern):
                    if path.is_file():
                        rel_path = path.relative_to(self._base_path)
                        results.append(str(rel_path))
                return sorted(results)
            return []

        results = []
        for path in search_path.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self._base_path)
                results.append(str(rel_path))
        return sorted(results)

    def exists(self, remote_key: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(remote_key).exists()

    def delete(self, remote_key: str) -> None:
        """Delete a file."""
        path = self._resolve_path(remote_key)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted {path}")


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
    ):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all operations
            region: AWS region (optional, uses default if not specified)
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )

        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/")

        if region:
            self._s3 = boto3.client("s3", region_name=region)
        else:
            self._s3 = boto3.client("s3")

        self._bucket = boto3.resource("s3").Bucket(bucket)

    def _full_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Upload file to S3."""
        local_path = Path(local_path)
        full_key = self._full_key(remote_key)

        self._s3.upload_file(str(local_path), self._bucket_name, full_key)
        logger.info(f"Uploaded {local_path} to s3://{self._bucket_name}/{full_key}")

    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Download file from S3."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        full_key = self._full_key(remote_key)

        self._s3.download_file(self._bucket_name, full_key, str(local_path))
        logger.info(f"Downloaded s3://{self._bucket_name}/{full_key} to {local_path}")

    def list(self, prefix: str = "") -> List[str]:
        """List files with prefix."""
        full_prefix = self._full_key(prefix)

        results = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Remove prefix to return relative keys
                if self._prefix and key.startswith(self._prefix + "/"):
                    key = key[len(self._prefix) + 1:]
                results.append(key)

        return sorted(results)

    def exists(self, remote_key: str) -> bool:
        """Check if file exists in S3."""
        full_key = self._full_key(remote_key)
        try:
            self._s3.head_object(Bucket=self._bucket_name, Key=full_key)
            return True
        except Exception:
            return False

    def delete(self, remote_key: str) -> None:
        """Delete file from S3."""
        full_key = self._full_key(remote_key)
        self._s3.delete_object(Bucket=self._bucket_name, Key=full_key)
        logger.info(f"Deleted s3://{self._bucket_name}/{full_key}")


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
    ):
        """Initialize GCS storage.

        Args:
            bucket: GCS bucket name
            prefix: Object prefix for all operations
        """
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS. "
                "Install with: pip install google-cloud-storage"
            )

        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/")

        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket)

    def _full_key(self, key: str) -> str:
        """Get full GCS object name with prefix."""
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    def upload(self, local_path: str | Path, remote_key: str) -> None:
        """Upload file to GCS."""
        local_path = Path(local_path)
        full_key = self._full_key(remote_key)

        blob = self._bucket.blob(full_key)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{self._bucket_name}/{full_key}")

    def download(self, remote_key: str, local_path: str | Path) -> None:
        """Download file from GCS."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        full_key = self._full_key(remote_key)

        blob = self._bucket.blob(full_key)
        blob.download_to_filename(str(local_path))
        logger.info(f"Downloaded gs://{self._bucket_name}/{full_key} to {local_path}")

    def list(self, prefix: str = "") -> List[str]:
        """List files with prefix."""
        full_prefix = self._full_key(prefix)

        results = []
        for blob in self._client.list_blobs(self._bucket_name, prefix=full_prefix):
            key = blob.name
            # Remove prefix to return relative keys
            if self._prefix and key.startswith(self._prefix + "/"):
                key = key[len(self._prefix) + 1:]
            results.append(key)

        return sorted(results)

    def exists(self, remote_key: str) -> bool:
        """Check if file exists in GCS."""
        full_key = self._full_key(remote_key)
        blob = self._bucket.blob(full_key)
        return blob.exists()

    def delete(self, remote_key: str) -> None:
        """Delete file from GCS."""
        full_key = self._full_key(remote_key)
        blob = self._bucket.blob(full_key)
        blob.delete()
        logger.info(f"Deleted gs://{self._bucket_name}/{full_key}")


def get_storage_backend(
    backend: Optional[str] = None,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    base_path: Optional[str] = None,
) -> StorageBackend:
    """Get a storage backend based on configuration.

    Args:
        backend: Storage type: local, s3, gcs (default from env or local)
        bucket: Bucket name for cloud storage (default from env)
        prefix: Key prefix for cloud storage (default from env)
        base_path: Base path for local storage (default from env or cwd)

    Returns:
        Configured StorageBackend instance

    Environment variables:
        STORAGE_BACKEND: local, s3, gcs
        STORAGE_BUCKET: S3/GCS bucket name
        STORAGE_PREFIX: Key prefix
        STORAGE_BASE_PATH: Local storage base path
    """
    backend = backend or os.getenv("STORAGE_BACKEND", "local")
    bucket = bucket or os.getenv("STORAGE_BUCKET", "")
    prefix = prefix or os.getenv("STORAGE_PREFIX", "ringrift-ai")
    base_path = base_path or os.getenv("STORAGE_BASE_PATH", ".")

    backend = backend.lower()

    if backend == "local":
        return LocalStorage(base_path=base_path)

    elif backend == "s3":
        if not bucket:
            raise ValueError("STORAGE_BUCKET is required for S3 backend")
        return S3Storage(bucket=bucket, prefix=prefix)

    elif backend == "gcs":
        if not bucket:
            raise ValueError("STORAGE_BUCKET is required for GCS backend")
        return GCSStorage(bucket=bucket, prefix=prefix)

    else:
        raise ValueError(
            f"Unknown storage backend: {backend}. "
            f"Supported: local, s3, gcs"
        )


def get_storage_from_uri(uri: str) -> StorageBackend:
    """Get storage backend from a URI.

    Args:
        uri: Storage URI:
            - /path/to/dir or file:///path (local)
            - s3://bucket/prefix (AWS S3)
            - gs://bucket/prefix (Google Cloud Storage)

    Returns:
        Configured StorageBackend instance
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "" or scheme == "file":
        path = parsed.path or uri
        return LocalStorage(base_path=path)

    elif scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return S3Storage(bucket=bucket, prefix=prefix)

    elif scheme == "gs":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return GCSStorage(bucket=bucket, prefix=prefix)

    else:
        raise ValueError(
            f"Unsupported storage URI scheme: {scheme}. "
            f"Use file://, s3://, or gs://"
        )
