"""Tests for file operation utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from app.utils.file_utils import (
    atomic_write,
    write_atomic,
    read_safe,
    read_bytes_safe,
    file_exists,
    ensure_file_dir,
    backup_file,
    remove_safe,
    get_file_size,
    get_file_mtime,
)


class TestAtomicWrite:
    """Tests for atomic_write context manager."""

    def test_basic_write(self, tmp_path):
        file_path = tmp_path / "test.txt"
        with atomic_write(file_path) as f:
            f.write("hello world")

        assert file_path.exists()
        assert file_path.read_text() == "hello world"

    def test_binary_write(self, tmp_path):
        file_path = tmp_path / "test.bin"
        with atomic_write(file_path, mode="wb") as f:
            f.write(b"\x00\x01\x02")

        assert file_path.read_bytes() == b"\x00\x01\x02"

    def test_creates_parent_dirs(self, tmp_path):
        file_path = tmp_path / "nested" / "dir" / "test.txt"
        with atomic_write(file_path) as f:
            f.write("content")

        assert file_path.exists()
        assert file_path.read_text() == "content"

    def test_no_partial_write_on_error(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")

        with pytest.raises(ValueError):
            with atomic_write(file_path) as f:
                f.write("new content")
                raise ValueError("simulated error")

        # Original file should be unchanged
        assert file_path.read_text() == "original"

    def test_cleans_up_temp_file_on_error(self, tmp_path):
        file_path = tmp_path / "test.txt"

        with pytest.raises(ValueError):
            with atomic_write(file_path) as f:
                f.write("content")
                raise ValueError("simulated error")

        # No temp files should remain
        temp_files = list(tmp_path.glob(".*test.txt*.tmp"))
        assert temp_files == []


class TestWriteAtomic:
    """Tests for write_atomic convenience function."""

    def test_write_string(self, tmp_path):
        file_path = tmp_path / "test.txt"
        write_atomic(file_path, "hello world")
        assert file_path.read_text() == "hello world"

    def test_write_bytes(self, tmp_path):
        file_path = tmp_path / "test.bin"
        write_atomic(file_path, b"\x00\x01\x02")
        assert file_path.read_bytes() == b"\x00\x01\x02"


class TestReadSafe:
    """Tests for read_safe function."""

    def test_read_existing_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello")
        assert read_safe(file_path) == "hello"

    def test_nonexistent_file_returns_default(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert read_safe(file_path, default="default") == "default"

    def test_nonexistent_file_returns_none(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert read_safe(file_path) is None

    def test_encoding(self, tmp_path):
        file_path = tmp_path / "test.txt"
        expected = "h\u00e9llo"
        file_path.write_text(expected, encoding="utf-8")
        assert read_safe(file_path, encoding="utf-8") == expected


class TestReadBytesSafe:
    """Tests for read_bytes_safe function."""

    def test_read_existing_file(self, tmp_path):
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(b"\x00\x01\x02")
        assert read_bytes_safe(file_path) == b"\x00\x01\x02"

    def test_nonexistent_file_returns_default(self, tmp_path):
        file_path = tmp_path / "nonexistent.bin"
        assert read_bytes_safe(file_path, default=b"default") == b"default"


class TestFileExists:
    """Tests for file_exists function."""

    def test_existing_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.touch()
        assert file_exists(file_path) is True

    def test_nonexistent_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert file_exists(file_path) is False

    def test_directory_returns_false(self, tmp_path):
        # tmp_path is a directory, should return False
        assert file_exists(tmp_path) is False


class TestEnsureFileDir:
    """Tests for ensure_file_dir function."""

    def test_creates_parent_dirs(self, tmp_path):
        file_path = tmp_path / "a" / "b" / "c" / "file.txt"
        result = ensure_file_dir(file_path)

        assert result == file_path
        assert file_path.parent.exists()

    def test_existing_dir(self, tmp_path):
        file_path = tmp_path / "file.txt"
        result = ensure_file_dir(file_path)
        assert result == file_path


class TestBackupFile:
    """Tests for backup_file function."""

    def test_creates_backup(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")

        backup_path = backup_file(file_path)

        assert backup_path == file_path.with_suffix(".txt.bak")
        assert backup_path.exists()
        assert backup_path.read_text() == "original"

    def test_custom_suffix(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")

        backup_path = backup_file(file_path, suffix=".backup")

        assert backup_path.suffix == ".backup"
        assert backup_path.read_text() == "original"

    def test_nonexistent_file_returns_none(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert backup_file(file_path) is None

    def test_overwrite_existing_backup(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")
        backup_path = file_path.with_suffix(".txt.bak")
        backup_path.write_text("old backup")

        result = backup_file(file_path, overwrite=True)

        assert result.read_text() == "original"

    def test_no_overwrite_preserves_backup(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")
        backup_path = file_path.with_suffix(".txt.bak")
        backup_path.write_text("old backup")

        result = backup_file(file_path, overwrite=False)

        assert result.read_text() == "old backup"


class TestRemoveSafe:
    """Tests for remove_safe function."""

    def test_removes_existing_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.touch()

        result = remove_safe(file_path)

        assert result is True
        assert not file_path.exists()

    def test_nonexistent_file_returns_false(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        result = remove_safe(file_path)
        assert result is False


class TestGetFileSize:
    """Tests for get_file_size function."""

    def test_existing_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello")  # 5 bytes
        assert get_file_size(file_path) == 5

    def test_nonexistent_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert get_file_size(file_path) == 0

    def test_empty_file(self, tmp_path):
        file_path = tmp_path / "empty.txt"
        file_path.touch()
        assert get_file_size(file_path) == 0


class TestGetFileMtime:
    """Tests for get_file_mtime function."""

    def test_existing_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.touch()
        mtime = get_file_mtime(file_path)
        assert mtime > 0

    def test_nonexistent_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"
        assert get_file_mtime(file_path) == 0.0
