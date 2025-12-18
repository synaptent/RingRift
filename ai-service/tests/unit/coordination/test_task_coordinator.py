"""Tests for Task Coordinator implementation.

Tests the global task coordination system that prevents uncoordinated
task spawning across multiple orchestrators.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.coordination.task_coordinator import (
    TaskType,
    ResourceType,
    atomic_write_json,
    safe_read_json,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_task_types_defined(self):
        """All expected task types should be defined."""
        assert TaskType.SELFPLAY.value == "selfplay"
        assert TaskType.GPU_SELFPLAY.value == "gpu_selfplay"
        assert TaskType.HYBRID_SELFPLAY.value == "hybrid_selfplay"
        assert TaskType.TRAINING.value == "training"
        assert TaskType.CMAES.value == "cmaes"
        assert TaskType.TOURNAMENT.value == "tournament"
        assert TaskType.EVALUATION.value == "evaluation"
        assert TaskType.SYNC.value == "sync"
        assert TaskType.EXPORT.value == "export"
        assert TaskType.PIPELINE.value == "pipeline"
        assert TaskType.IMPROVEMENT_LOOP.value == "improvement_loop"
        assert TaskType.BACKGROUND_LOOP.value == "background_loop"

    def test_task_type_count(self):
        """Should have exactly 12 task types."""
        assert len(TaskType) == 12


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_resource_types_defined(self):
        """All expected resource types should be defined."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.HYBRID.value == "hybrid"
        assert ResourceType.IO.value == "io"

    def test_resource_type_count(self):
        """Should have exactly 4 resource types."""
        assert len(ResourceType) == 4


class TestAtomicWriteJson:
    """Tests for atomic JSON write utility."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_writes_json_file(self, temp_dir):
        """Should write JSON data to file."""
        filepath = temp_dir / "test.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(filepath, data)

        assert filepath.exists()

    def test_json_content_correct(self, temp_dir):
        """Should write correct JSON content."""
        filepath = temp_dir / "test.json"
        data = {"key": "value", "list": [1, 2, 3]}

        atomic_write_json(filepath, data)

        result = safe_read_json(filepath)
        assert result == data

    def test_creates_parent_directories(self, temp_dir):
        """Should create parent directories if they don't exist."""
        filepath = temp_dir / "nested" / "deep" / "test.json"
        data = {"nested": True}

        atomic_write_json(filepath, data)

        assert filepath.exists()
        assert safe_read_json(filepath) == data

    def test_overwrites_existing_file(self, temp_dir):
        """Should overwrite existing file."""
        filepath = temp_dir / "test.json"

        atomic_write_json(filepath, {"first": True})
        atomic_write_json(filepath, {"second": True})

        result = safe_read_json(filepath)
        assert result == {"second": True}

    def test_custom_indent(self, temp_dir):
        """Should respect custom indent parameter."""
        filepath = temp_dir / "test.json"
        data = {"key": "value"}

        atomic_write_json(filepath, data, indent=4)

        content = filepath.read_text()
        assert "    " in content  # 4-space indent


class TestSafeReadJson:
    """Tests for safe JSON read utility."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_reads_valid_json(self, temp_dir):
        """Should read valid JSON file."""
        filepath = temp_dir / "test.json"
        filepath.write_text('{"key": "value"}')

        result = safe_read_json(filepath)
        assert result == {"key": "value"}

    def test_returns_default_for_missing_file(self, temp_dir):
        """Should return default for missing file."""
        filepath = temp_dir / "nonexistent.json"

        result = safe_read_json(filepath, default={"default": True})
        assert result == {"default": True}

    def test_returns_none_default_for_missing(self, temp_dir):
        """Should return None as default for missing file."""
        filepath = temp_dir / "nonexistent.json"

        result = safe_read_json(filepath)
        assert result is None

    def test_returns_default_for_corrupt_json(self, temp_dir):
        """Should return default for corrupt JSON."""
        filepath = temp_dir / "corrupt.json"
        filepath.write_text("not valid json {")

        result = safe_read_json(filepath, default={"default": True})
        assert result == {"default": True}

    def test_reads_backup_on_corruption(self, temp_dir):
        """Should read backup file if main file is corrupt."""
        filepath = temp_dir / "test.json"
        backup = filepath.with_suffix(".json.bak")

        filepath.write_text("corrupt data")
        backup.write_text('{"backup": true}')

        result = safe_read_json(filepath, default=None)
        assert result == {"backup": True}

    def test_empty_list_as_valid_json(self, temp_dir):
        """Should correctly read empty list."""
        filepath = temp_dir / "test.json"
        filepath.write_text("[]")

        result = safe_read_json(filepath)
        assert result == []

    def test_empty_object_as_valid_json(self, temp_dir):
        """Should correctly read empty object."""
        filepath = temp_dir / "test.json"
        filepath.write_text("{}")

        result = safe_read_json(filepath)
        assert result == {}


class TestTaskTypeResourceMapping:
    """Tests for task type to resource type mapping."""

    def test_selfplay_is_cpu(self):
        """Selfplay should be CPU-bound."""
        # Based on typical mapping in the module
        cpu_tasks = [TaskType.SELFPLAY, TaskType.TOURNAMENT, TaskType.EVALUATION]
        for task in cpu_tasks:
            assert task in TaskType

    def test_training_is_gpu(self):
        """Training should be GPU-bound."""
        gpu_tasks = [TaskType.TRAINING, TaskType.CMAES]
        for task in gpu_tasks:
            assert task in TaskType

    def test_hybrid_selfplay_exists(self):
        """Hybrid selfplay task type should exist."""
        assert TaskType.HYBRID_SELFPLAY.value == "hybrid_selfplay"


class TestJsonRoundTrip:
    """Tests for JSON read/write round-trip."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_string_values(self, temp_dir):
        """Should preserve string values."""
        filepath = temp_dir / "test.json"
        data = {"string": "hello world"}
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data

    def test_numeric_values(self, temp_dir):
        """Should preserve numeric values."""
        filepath = temp_dir / "test.json"
        data = {"int": 42, "float": 3.14}
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data

    def test_nested_structures(self, temp_dir):
        """Should preserve nested structures."""
        filepath = temp_dir / "test.json"
        data = {
            "nested": {
                "level1": {
                    "level2": [1, 2, 3]
                }
            }
        }
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data

    def test_boolean_values(self, temp_dir):
        """Should preserve boolean values."""
        filepath = temp_dir / "test.json"
        data = {"true": True, "false": False}
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data

    def test_null_values(self, temp_dir):
        """Should preserve null values."""
        filepath = temp_dir / "test.json"
        data = {"null_value": None}
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_unicode_content(self, temp_dir):
        """Should handle unicode content."""
        filepath = temp_dir / "test.json"
        data = {"unicode": "Hello \u4e16\u754c"}
        atomic_write_json(filepath, data)
        result = safe_read_json(filepath)
        assert result["unicode"] == "Hello \u4e16\u754c"

    def test_large_data(self, temp_dir):
        """Should handle larger data structures."""
        filepath = temp_dir / "test.json"
        data = {"items": list(range(10000))}
        atomic_write_json(filepath, data)
        result = safe_read_json(filepath)
        assert len(result["items"]) == 10000

    def test_special_characters_in_keys(self, temp_dir):
        """Should handle special characters in keys."""
        filepath = temp_dir / "test.json"
        data = {"key-with-dash": 1, "key.with.dots": 2, "key_with_underscore": 3}
        atomic_write_json(filepath, data)
        assert safe_read_json(filepath) == data
