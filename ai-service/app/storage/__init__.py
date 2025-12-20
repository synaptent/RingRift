"""Storage abstraction layer for cloud and local file operations.

Provides a unified interface for:
- Uploading artifacts (models, checkpoints, datasets)
- Downloading training data and state pools
- Listing available files

Supports local filesystem, S3, and GCS backends.
"""

from app.storage.backends import (
    GCSStorage,
    LocalStorage,
    S3Storage,
    StorageBackend,
    get_storage_backend,
)

__all__ = [
    "GCSStorage",
    "LocalStorage",
    "S3Storage",
    "StorageBackend",
    "get_storage_backend",
]
