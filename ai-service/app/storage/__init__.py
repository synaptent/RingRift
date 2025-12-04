"""Storage abstraction layer for cloud and local file operations.

Provides a unified interface for:
- Uploading artifacts (models, checkpoints, datasets)
- Downloading training data and state pools
- Listing available files

Supports local filesystem, S3, and GCS backends.
"""

from app.storage.backends import (
    StorageBackend,
    LocalStorage,
    S3Storage,
    GCSStorage,
    get_storage_backend,
)

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "get_storage_backend",
]
