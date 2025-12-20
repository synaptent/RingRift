"""Tests for the storage module.

Comprehensive tests for:
- StorageBackend abstract interface
- LocalStorage: upload, download, list, exists, delete
- get_storage_backend factory function
- Cloud backends (S3, GCS) mocked tests
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary local storage instance."""
        from app.storage import LocalStorage

        return LocalStorage(base_path=tmp_path)

    def test_local_storage_creation(self, tmp_path):
        """Test creating local storage."""
        from app.storage import LocalStorage

        storage = LocalStorage(base_path=tmp_path)
        assert storage is not None

    def test_local_storage_creates_base_path(self, tmp_path):
        """Test that local storage creates the base path if it doesn't exist."""
        from app.storage import LocalStorage

        new_path = tmp_path / "new_directory"
        LocalStorage(base_path=new_path)

        assert new_path.exists()

    def test_upload_and_download(self, temp_storage, tmp_path):
        """Test upload and download operations."""
        # Create a test file
        source_file = tmp_path / "source.txt"
        source_file.write_text("test content")

        # Upload
        temp_storage.upload(source_file, "uploaded/file.txt")

        # Download to new location
        dest_file = tmp_path / "dest.txt"
        temp_storage.download("uploaded/file.txt", dest_file)

        assert dest_file.read_text() == "test content"

    def test_upload_creates_directories(self, temp_storage, tmp_path):
        """Test that upload creates intermediate directories."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("test")

        temp_storage.upload(source_file, "deep/nested/path/file.txt")

        assert temp_storage.exists("deep/nested/path/file.txt")

    def test_exists(self, temp_storage, tmp_path):
        """Test exists method."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("test")

        temp_storage.upload(source_file, "file.txt")

        assert temp_storage.exists("file.txt")
        assert not temp_storage.exists("missing.txt")

    def test_delete(self, temp_storage, tmp_path):
        """Test delete method."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("test")

        temp_storage.upload(source_file, "file.txt")
        assert temp_storage.exists("file.txt")

        temp_storage.delete("file.txt")
        assert not temp_storage.exists("file.txt")

    def test_list_files(self, temp_storage, tmp_path):
        """Test listing files."""
        # Create multiple files
        for i in range(3):
            source = tmp_path / f"source_{i}.txt"
            source.write_text(f"content {i}")
            temp_storage.upload(source, f"files/file_{i}.txt")

        files = temp_storage.list("files/")

        assert len(files) == 3
        assert all("file_" in f for f in files)

    def test_list_empty_prefix(self, temp_storage, tmp_path):
        """Test listing with empty prefix."""
        source = tmp_path / "source.txt"
        source.write_text("test")
        temp_storage.upload(source, "file.txt")

        files = temp_storage.list("")

        assert len(files) >= 1

    def test_list_no_matches(self, temp_storage):
        """Test listing with no matching files."""
        files = temp_storage.list("nonexistent/")

        assert files == []

    def test_download_missing_file_raises(self, temp_storage, tmp_path):
        """Test that downloading missing file raises error."""
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError):
            temp_storage.download("missing.txt", dest)


class TestGetStorageBackend:
    """Tests for get_storage_backend factory function."""

    def test_get_local_backend_default(self):
        """Test getting local backend as default."""
        from app.storage import LocalStorage, get_storage_backend

        storage = get_storage_backend(backend="local")

        assert isinstance(storage, LocalStorage)

    def test_get_local_backend_with_base_path(self, tmp_path):
        """Test getting local backend with custom base path."""
        from app.storage import get_storage_backend

        storage = get_storage_backend(backend="local", base_path=str(tmp_path))

        assert storage is not None

    def test_get_backend_from_env(self, tmp_path, monkeypatch):
        """Test getting backend from environment variables."""
        from app.storage import LocalStorage, get_storage_backend

        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_BASE_PATH", str(tmp_path))

        storage = get_storage_backend()

        assert isinstance(storage, LocalStorage)


try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestS3Storage:
    """Tests for S3Storage backend (mocked)."""

    def test_s3_storage_creation(self):
        """Test creating S3 storage (mocked)."""
        with patch("boto3.client"):
            from app.storage import S3Storage

            storage = S3Storage(bucket="test-bucket", prefix="test-prefix")

            assert storage is not None

    def test_s3_upload_mocked(self):
        """Test S3 upload with mocked client."""
        with patch("boto3.client") as mock_client:
            from app.storage import S3Storage

            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            storage = S3Storage(bucket="test-bucket")

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"test content")
                f.flush()

                try:
                    storage.upload(f.name, "test-key")
                    mock_s3.upload_file.assert_called_once()
                finally:
                    os.unlink(f.name)

    def test_s3_download_mocked(self):
        """Test S3 download with mocked client."""
        with patch("boto3.client") as mock_client:
            from app.storage import S3Storage

            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            storage = S3Storage(bucket="test-bucket")

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = os.path.join(tmpdir, "dest.txt")
                storage.download("test-key", dest)

                mock_s3.download_file.assert_called_once()

    def test_s3_exists_mocked(self):
        """Test S3 exists with mocked client."""
        with patch("boto3.client") as mock_client:
            from app.storage import S3Storage

            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            mock_s3.head_object.return_value = {}

            storage = S3Storage(bucket="test-bucket")

            assert storage.exists("test-key") is True

    def test_s3_list_mocked(self):
        """Test S3 list with mocked client."""
        with patch("boto3.client") as mock_client:
            from app.storage import S3Storage

            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            mock_s3.list_objects_v2.return_value = {
                "Contents": [
                    {"Key": "prefix/file1.txt"},
                    {"Key": "prefix/file2.txt"},
                ]
            }

            storage = S3Storage(bucket="test-bucket")
            files = storage.list("prefix/")

            assert len(files) == 2


@pytest.mark.skipif(not HAS_GCS, reason="google-cloud-storage not installed")
class TestGCSStorage:
    """Tests for GCSStorage backend (mocked)."""

    def test_gcs_storage_creation(self):
        """Test creating GCS storage (mocked)."""
        with patch("google.cloud.storage.Client") as mock_client:
            from app.storage import GCSStorage

            mock_bucket = MagicMock()
            mock_client.return_value.bucket.return_value = mock_bucket

            storage = GCSStorage(bucket="test-bucket", prefix="test-prefix")

            assert storage is not None


class TestStorageBackendInterface:
    """Test that StorageBackend defines the expected interface."""

    def test_abstract_methods(self):
        """Test that StorageBackend has required abstract methods."""
        import inspect

        from app.storage import StorageBackend

        # Get abstract methods
        abstract_methods = set()
        for name, method in inspect.getmembers(StorageBackend):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        expected = {"upload", "download", "list", "exists", "delete"}
        assert expected.issubset(abstract_methods)


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_all_exports(self):
        """Test importing all exports from storage module."""
        from app.storage import (
            GCSStorage,
            LocalStorage,
            S3Storage,
            StorageBackend,
            get_storage_backend,
        )

        assert StorageBackend is not None
        assert LocalStorage is not None
        assert S3Storage is not None
        assert GCSStorage is not None
        assert callable(get_storage_backend)
