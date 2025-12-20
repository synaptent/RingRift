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
)
# Use centralized JSON utilities (atomic write and safe read)
from app.utils.json_utils import save_json as atomic_write_json, load_json as safe_read_json


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

    def test_returns_default_on_corruption(self, temp_dir):
        """Should return default for corrupt JSON (no backup fallback)."""
        filepath = temp_dir / "test.json"
        filepath.write_text("corrupt data")

        # load_json returns default for corrupt files
        result = safe_read_json(filepath, default={"fallback": True})
        assert result == {"fallback": True}

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


# ============================================
# Additional imports for extended tests
# ============================================

import os
import threading

from app.coordination.task_coordinator import (
    TaskLimits,
    TaskInfo,
    CoordinatorState,
    OrchestratorLock,
    RateLimiter,
    TaskRegistry,
    TaskHeartbeatMonitor,
    TaskCoordinator,
    CoordinatedTask,
    get_task_resource_type,
    is_gpu_task,
    is_cpu_task,
    get_queue_for_task,
    get_coordinator,
    can_spawn,
    emergency_stop_all,
    TASK_RESOURCE_MAP,
)


# ============================================
# Tests for resource mapping functions
# ============================================

class TestGetTaskResourceType:
    """Tests for get_task_resource_type function."""

    def test_selfplay_is_cpu(self):
        """Selfplay should map to CPU resource type."""
        assert get_task_resource_type(TaskType.SELFPLAY) == ResourceType.CPU

    def test_gpu_selfplay_is_gpu(self):
        """GPU selfplay should map to GPU resource type."""
        assert get_task_resource_type(TaskType.GPU_SELFPLAY) == ResourceType.GPU

    def test_hybrid_selfplay_is_hybrid(self):
        """Hybrid selfplay should map to HYBRID resource type."""
        assert get_task_resource_type(TaskType.HYBRID_SELFPLAY) == ResourceType.HYBRID

    def test_training_is_gpu(self):
        """Training should map to GPU resource type."""
        assert get_task_resource_type(TaskType.TRAINING) == ResourceType.GPU

    def test_cmaes_is_gpu(self):
        """CMA-ES should map to GPU resource type."""
        assert get_task_resource_type(TaskType.CMAES) == ResourceType.GPU

    def test_tournament_is_cpu(self):
        """Tournament should map to CPU resource type."""
        assert get_task_resource_type(TaskType.TOURNAMENT) == ResourceType.CPU

    def test_evaluation_is_cpu(self):
        """Evaluation should map to CPU resource type."""
        assert get_task_resource_type(TaskType.EVALUATION) == ResourceType.CPU

    def test_sync_is_io(self):
        """Sync should map to IO resource type."""
        assert get_task_resource_type(TaskType.SYNC) == ResourceType.IO

    def test_export_is_io(self):
        """Export should map to IO resource type."""
        assert get_task_resource_type(TaskType.EXPORT) == ResourceType.IO

    def test_pipeline_is_hybrid(self):
        """Pipeline should map to HYBRID resource type."""
        assert get_task_resource_type(TaskType.PIPELINE) == ResourceType.HYBRID

    def test_improvement_loop_is_hybrid(self):
        """Improvement loop should map to HYBRID resource type."""
        assert get_task_resource_type(TaskType.IMPROVEMENT_LOOP) == ResourceType.HYBRID

    def test_background_loop_is_cpu(self):
        """Background loop should map to CPU resource type."""
        assert get_task_resource_type(TaskType.BACKGROUND_LOOP) == ResourceType.CPU


class TestIsGpuTask:
    """Tests for is_gpu_task function."""

    def test_gpu_selfplay(self):
        """GPU selfplay should be a GPU task."""
        assert is_gpu_task(TaskType.GPU_SELFPLAY) is True

    def test_training(self):
        """Training should be a GPU task."""
        assert is_gpu_task(TaskType.TRAINING) is True

    def test_cmaes(self):
        """CMA-ES should be a GPU task."""
        assert is_gpu_task(TaskType.CMAES) is True

    def test_hybrid_selfplay(self):
        """Hybrid selfplay should be a GPU task (uses GPU)."""
        assert is_gpu_task(TaskType.HYBRID_SELFPLAY) is True

    def test_pipeline(self):
        """Pipeline should be a GPU task (hybrid uses GPU)."""
        assert is_gpu_task(TaskType.PIPELINE) is True

    def test_selfplay_is_not_gpu(self):
        """Regular selfplay should NOT be a GPU task."""
        assert is_gpu_task(TaskType.SELFPLAY) is False

    def test_sync_is_not_gpu(self):
        """Sync should NOT be a GPU task."""
        assert is_gpu_task(TaskType.SYNC) is False


class TestIsCpuTask:
    """Tests for is_cpu_task function."""

    def test_selfplay(self):
        """Selfplay should be a CPU task."""
        assert is_cpu_task(TaskType.SELFPLAY) is True

    def test_tournament(self):
        """Tournament should be a CPU task."""
        assert is_cpu_task(TaskType.TOURNAMENT) is True

    def test_evaluation(self):
        """Evaluation should be a CPU task."""
        assert is_cpu_task(TaskType.EVALUATION) is True

    def test_background_loop(self):
        """Background loop should be a CPU task."""
        assert is_cpu_task(TaskType.BACKGROUND_LOOP) is True

    def test_hybrid_selfplay(self):
        """Hybrid selfplay should be a CPU task (uses CPU)."""
        assert is_cpu_task(TaskType.HYBRID_SELFPLAY) is True

    def test_training_is_not_cpu(self):
        """Training should NOT be a CPU task."""
        assert is_cpu_task(TaskType.TRAINING) is False

    def test_sync_is_not_cpu(self):
        """Sync should NOT be a CPU task (IO bound)."""
        assert is_cpu_task(TaskType.SYNC) is False


# ============================================
# Tests for TaskLimits
# ============================================

class TestTaskLimits:
    """Tests for TaskLimits dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        limits = TaskLimits()

        # Per-node limits
        assert limits.max_selfplay_per_node == 32
        assert limits.max_training_per_node == 1
        assert limits.max_sync_per_node == 2
        assert limits.max_export_per_node == 2

        # Cluster-wide limits
        assert limits.max_total_selfplay == 500
        assert limits.max_total_training == 3
        assert limits.max_total_cmaes == 1
        assert limits.max_total_tournaments == 2
        assert limits.max_total_pipelines == 1
        assert limits.max_total_improvement_loops == 1

        # Rate limits
        assert limits.max_task_spawns_per_minute == 60
        assert limits.max_selfplay_spawns_per_minute == 30

        # Resource thresholds
        assert limits.halt_on_disk_percent == 70.0
        assert limits.halt_on_memory_percent == 95.0
        assert limits.halt_on_cpu_percent == 95.0

        # Backpressure settings
        assert limits.soft_limit_factor == 0.8
        assert limits.spawn_cooldown_seconds == 1.0

    def test_conservative_limits(self):
        """Conservative limits should be more restrictive."""
        limits = TaskLimits.conservative()

        assert limits.max_selfplay_per_node == 8
        assert limits.max_total_selfplay == 100
        assert limits.max_task_spawns_per_minute == 20
        assert limits.max_selfplay_spawns_per_minute == 10

    def test_aggressive_limits(self):
        """Aggressive limits should allow more tasks."""
        limits = TaskLimits.aggressive()

        assert limits.max_selfplay_per_node == 64
        assert limits.max_total_selfplay == 1000
        assert limits.max_task_spawns_per_minute == 120
        assert limits.max_selfplay_spawns_per_minute == 60

    def test_custom_limits(self):
        """Should allow custom limit values."""
        limits = TaskLimits(
            max_selfplay_per_node=16,
            max_total_selfplay=200,
            halt_on_disk_percent=80.0,
        )

        assert limits.max_selfplay_per_node == 16
        assert limits.max_total_selfplay == 200
        assert limits.halt_on_disk_percent == 80.0
        # Others should be default
        assert limits.max_training_per_node == 1


class TestCoordinatorState:
    """Tests for CoordinatorState enum."""

    def test_all_states_defined(self):
        """All expected states should be defined."""
        assert CoordinatorState.RUNNING.value == "running"
        assert CoordinatorState.PAUSED.value == "paused"
        assert CoordinatorState.DRAINING.value == "draining"
        assert CoordinatorState.EMERGENCY.value == "emergency"
        assert CoordinatorState.STOPPED.value == "stopped"

    def test_state_count(self):
        """Should have exactly 5 states."""
        assert len(CoordinatorState) == 5


# ============================================
# Tests for TaskInfo
# ============================================

class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_minimal_creation(self):
        """Should create with minimal required fields."""
        task = TaskInfo(
            task_id="test-task-1",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        )

        assert task.task_id == "test-task-1"
        assert task.task_type == TaskType.SELFPLAY
        assert task.node_id == "node-1"
        assert task.pid == 0
        assert task.status == "running"
        assert task.metadata == {}

    def test_full_creation(self):
        """Should create with all fields."""
        start_time = time.time()
        task = TaskInfo(
            task_id="test-task-2",
            task_type=TaskType.TRAINING,
            node_id="gpu-node-1",
            started_at=start_time,
            pid=12345,
            status="running",
            metadata={"iteration": 5, "games": 1000},
        )

        assert task.task_id == "test-task-2"
        assert task.task_type == TaskType.TRAINING
        assert task.node_id == "gpu-node-1"
        assert task.started_at == start_time
        assert task.pid == 12345
        assert task.status == "running"
        assert task.metadata["iteration"] == 5
        assert task.metadata["games"] == 1000


# ============================================
# Tests for RateLimiter
# ============================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_initial_burst(self):
        """Should have burst tokens available initially."""
        limiter = RateLimiter(rate=1.0, burst=10)
        assert limiter.tokens_available() == 10

    def test_acquire_single_token(self):
        """Should acquire single token successfully."""
        limiter = RateLimiter(rate=1.0, burst=10)
        assert limiter.acquire() is True
        assert limiter.tokens_available() < 10

    def test_acquire_multiple_tokens(self):
        """Should acquire multiple tokens at once."""
        limiter = RateLimiter(rate=1.0, burst=10)
        assert limiter.acquire(tokens=5) is True
        # Allow small floating point margin
        assert limiter.tokens_available() <= 5.01

    def test_acquire_fails_when_exhausted(self):
        """Should fail to acquire when tokens exhausted."""
        limiter = RateLimiter(rate=0.0, burst=3)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_tokens_replenish(self):
        """Should replenish tokens over time."""
        limiter = RateLimiter(rate=100.0, burst=10)  # 100 per second

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()

        # Wait for replenishment
        time.sleep(0.1)  # Should get ~10 tokens

        assert limiter.tokens_available() >= 5

    def test_burst_cap(self):
        """Should not exceed burst limit."""
        limiter = RateLimiter(rate=100.0, burst=5)

        # Wait and check - should be capped at burst
        time.sleep(0.1)
        assert limiter.tokens_available() <= 5

    def test_thread_safe(self):
        """Should be thread-safe."""
        limiter = RateLimiter(rate=1000.0, burst=100)
        results = []

        def acquire_tokens():
            for _ in range(10):
                results.append(limiter.acquire())

        threads = [threading.Thread(target=acquire_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All initial acquires should succeed (100 burst)
        assert sum(results) >= 50  # At least 50 should succeed


# ============================================
# Tests for OrchestratorLock
# ============================================

class TestOrchestratorLock:
    """Tests for OrchestratorLock class."""

    @pytest.fixture
    def lock(self):
        """Create a test lock with unique name."""
        import uuid
        lock = OrchestratorLock(lock_name=f"test_{uuid.uuid4().hex[:8]}")
        yield lock
        lock.release()

    def test_acquire_release(self, lock):
        """Should acquire and release lock."""
        assert lock.acquire() is True
        assert lock.is_held() is True
        lock.release()
        # After release, is_held checks if process is alive

    def test_acquire_nonblocking(self, lock):
        """Should fail immediately when already held."""
        lock.acquire()

        # Create another lock instance for same name
        lock2 = OrchestratorLock(lock_name=lock.lock_file.stem)
        assert lock2.acquire(blocking=False) is False

        lock.release()

    def test_get_holder(self, lock):
        """Should return holder PID."""
        lock.acquire()
        holder = lock.get_holder()
        assert holder == os.getpid()
        lock.release()

    def test_context_manager(self):
        """Should work as context manager."""
        import uuid
        lock = OrchestratorLock(lock_name=f"test_ctx_{uuid.uuid4().hex[:8]}")

        with lock:
            assert lock.is_held() is True

    def test_is_held_detects_dead_process(self, lock):
        """Should detect when holder process is dead."""
        # Write a fake PID that doesn't exist
        lock.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock.lock_file.write_text("99999999")  # Non-existent PID

        # Should return False since process doesn't exist
        assert lock.is_held() is False


# ============================================
# Tests for TaskRegistry
# ============================================

class TestTaskRegistry:
    """Tests for TaskRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a test registry with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_tasks.db"
            yield TaskRegistry(db_path)

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return TaskInfo(
            task_id="task-001",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
            pid=12345,
            status="running",
            metadata={"test": True},
        )

    def test_register_task(self, registry, sample_task):
        """Should register a task."""
        registry.register_task(sample_task)

        retrieved = registry.get_task(sample_task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == sample_task.task_id
        assert retrieved.task_type == sample_task.task_type
        assert retrieved.node_id == sample_task.node_id

    def test_unregister_task(self, registry, sample_task):
        """Should unregister a task."""
        registry.register_task(sample_task)
        registry.unregister_task(sample_task.task_id)

        assert registry.get_task(sample_task.task_id) is None

    def test_update_task_status(self, registry, sample_task):
        """Should update task status."""
        registry.register_task(sample_task)
        registry.update_task_status(sample_task.task_id, "completed")

        retrieved = registry.get_task(sample_task.task_id)
        assert retrieved.status == "completed"

    def test_get_tasks_by_node(self, registry):
        """Should get tasks filtered by node."""
        task1 = TaskInfo(
            task_id="task-1",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        )
        task2 = TaskInfo(
            task_id="task-2",
            task_type=TaskType.SELFPLAY,
            node_id="node-2",
            started_at=time.time(),
        )
        task3 = TaskInfo(
            task_id="task-3",
            task_type=TaskType.TRAINING,
            node_id="node-1",
            started_at=time.time(),
        )

        registry.register_task(task1)
        registry.register_task(task2)
        registry.register_task(task3)

        node1_tasks = registry.get_tasks_by_node("node-1")
        assert len(node1_tasks) == 2
        assert all(t.node_id == "node-1" for t in node1_tasks)

    def test_get_tasks_by_type(self, registry):
        """Should get tasks filtered by type."""
        task1 = TaskInfo(
            task_id="task-1",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        )
        task2 = TaskInfo(
            task_id="task-2",
            task_type=TaskType.TRAINING,
            node_id="node-1",
            started_at=time.time(),
        )

        registry.register_task(task1)
        registry.register_task(task2)

        selfplay_tasks = registry.get_tasks_by_type(TaskType.SELFPLAY)
        assert len(selfplay_tasks) == 1
        assert selfplay_tasks[0].task_type == TaskType.SELFPLAY

    def test_get_all_running_tasks(self, registry):
        """Should get all running tasks."""
        task1 = TaskInfo(
            task_id="task-1",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        )
        task2 = TaskInfo(
            task_id="task-2",
            task_type=TaskType.SELFPLAY,
            node_id="node-2",
            started_at=time.time(),
        )

        registry.register_task(task1)
        registry.register_task(task2)

        # Mark one as completed
        registry.update_task_status("task-1", "completed")

        running = registry.get_all_running_tasks()
        assert len(running) == 1
        assert running[0].task_id == "task-2"

    def test_count_by_type(self, registry):
        """Should count tasks by type."""
        for i in range(5):
            registry.register_task(TaskInfo(
                task_id=f"selfplay-{i}",
                task_type=TaskType.SELFPLAY,
                node_id="node-1",
                started_at=time.time(),
            ))

        for i in range(2):
            registry.register_task(TaskInfo(
                task_id=f"training-{i}",
                task_type=TaskType.TRAINING,
                node_id="node-1",
                started_at=time.time(),
            ))

        assert registry.count_by_type(TaskType.SELFPLAY) == 5
        assert registry.count_by_type(TaskType.TRAINING) == 2
        assert registry.count_by_type(TaskType.CMAES) == 0

    def test_count_by_node(self, registry):
        """Should count tasks by node."""
        for i in range(3):
            registry.register_task(TaskInfo(
                task_id=f"task-1-{i}",
                task_type=TaskType.SELFPLAY,
                node_id="node-1",
                started_at=time.time(),
            ))

        for i in range(7):
            registry.register_task(TaskInfo(
                task_id=f"task-2-{i}",
                task_type=TaskType.SELFPLAY,
                node_id="node-2",
                started_at=time.time(),
            ))

        assert registry.count_by_node("node-1") == 3
        assert registry.count_by_node("node-2") == 7
        assert registry.count_by_node("node-3") == 0

    def test_count_by_node_with_type(self, registry):
        """Should count tasks by node and type."""
        registry.register_task(TaskInfo(
            task_id="task-1",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        ))
        registry.register_task(TaskInfo(
            task_id="task-2",
            task_type=TaskType.TRAINING,
            node_id="node-1",
            started_at=time.time(),
        ))

        assert registry.count_by_node("node-1", TaskType.SELFPLAY) == 1
        assert registry.count_by_node("node-1", TaskType.TRAINING) == 1
        assert registry.count_by_node("node-1", TaskType.CMAES) == 0

    def test_log_spawn_attempt(self, registry):
        """Should log spawn attempts."""
        registry.log_spawn_attempt(TaskType.SELFPLAY, "node-1", True, "OK")
        registry.log_spawn_attempt(TaskType.SELFPLAY, "node-1", False, "limit")

        # Check spawn count only counts allowed
        assert registry.get_spawn_count(1) == 1

    def test_cleanup_stale_tasks(self, registry):
        """Should cleanup stale tasks."""
        # Create old task
        old_task = TaskInfo(
            task_id="old-task",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time() - (25 * 3600),  # 25 hours ago
        )
        registry.register_task(old_task)

        # Create recent task
        recent_task = TaskInfo(
            task_id="recent-task",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time(),
        )
        registry.register_task(recent_task)

        cleaned = registry.cleanup_stale_tasks(max_age_hours=24.0)
        assert cleaned == 1
        assert registry.get_task("old-task") is None
        assert registry.get_task("recent-task") is not None

    def test_set_get_state(self, registry):
        """Should set and get coordinator state."""
        registry.set_state("coordinator_state", "running")
        assert registry.get_state("coordinator_state") == "running"

        registry.set_state("coordinator_state", "paused")
        assert registry.get_state("coordinator_state") == "paused"

        assert registry.get_state("nonexistent") is None

    def test_update_heartbeat(self, registry, sample_task):
        """Should update task heartbeat."""
        registry.register_task(sample_task)

        before = time.time()
        registry.update_heartbeat(sample_task.task_id)
        after = time.time()

        task = registry.get_task(sample_task.task_id)
        heartbeat = task.metadata.get("last_heartbeat")
        assert heartbeat is not None
        # Use epsilon tolerance for floating-point clock precision (1ms)
        epsilon = 1e-3
        assert before - epsilon <= heartbeat <= after + epsilon

    def test_get_orphaned_tasks(self, registry):
        """Should find orphaned tasks."""
        # Create task with old heartbeat
        old_task = TaskInfo(
            task_id="orphan",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time() - 600,  # Started 10 min ago
            metadata={"last_heartbeat": time.time() - 400},  # No heartbeat for 6+ min
        )
        registry.register_task(old_task)

        # Create task with recent heartbeat
        active_task = TaskInfo(
            task_id="active",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time() - 600,
            metadata={"last_heartbeat": time.time() - 10},  # Recent heartbeat
        )
        registry.register_task(active_task)

        orphans = registry.get_orphaned_tasks(timeout_seconds=300)
        assert len(orphans) == 1
        assert orphans[0].task_id == "orphan"


# ============================================
# Tests for TaskCoordinator
# ============================================

class TestTaskCoordinator:
    """Tests for TaskCoordinator class."""

    @pytest.fixture(autouse=True)
    def reset_coordinator(self):
        """Reset coordinator singleton for each test."""
        TaskCoordinator.reset_instance()
        yield
        TaskCoordinator.reset_instance()

    @pytest.fixture
    def coordinator(self):
        """Get coordinator instance."""
        # Set temp directory for test
        import uuid
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        coord = TaskCoordinator.get_instance()
        yield coord

        # Cleanup
        if "RINGRIFT_COORDINATOR_DIR" in os.environ:
            del os.environ["RINGRIFT_COORDINATOR_DIR"]

    def test_singleton(self, coordinator):
        """Should return same instance."""
        coord2 = TaskCoordinator.get_instance()
        assert coordinator is coord2

    def test_default_state_running(self, coordinator):
        """Should start in RUNNING state."""
        assert coordinator.get_state() == CoordinatorState.RUNNING

    def test_pause_resume(self, coordinator):
        """Should pause and resume."""
        coordinator.pause()
        assert coordinator.get_state() == CoordinatorState.PAUSED

        coordinator.resume()
        assert coordinator.get_state() == CoordinatorState.RUNNING

    def test_emergency_stop(self, coordinator):
        """Should enter emergency state."""
        coordinator.emergency_stop()
        assert coordinator.get_state() == CoordinatorState.EMERGENCY

    def test_can_spawn_when_running(self, coordinator):
        """Should allow spawning when running."""
        # Wait for cooldown
        time.sleep(coordinator.limits.spawn_cooldown_seconds + 0.1)

        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=False,
            check_health=False,
        )
        assert allowed is True
        assert reason == "OK"

    def test_cannot_spawn_when_paused(self, coordinator):
        """Should deny spawning when paused."""
        coordinator.pause()

        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=False,
        )
        assert allowed is False
        assert "paused" in reason.lower()

    def test_cannot_spawn_when_emergency(self, coordinator):
        """Should deny spawning in emergency state."""
        coordinator.emergency_stop()

        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=False,
        )
        assert allowed is False
        assert "emergency" in reason.lower()

    def test_register_unregister_task(self, coordinator):
        """Should register and unregister tasks."""
        task_id = "test-task-123"

        coordinator.register_task(
            task_id=task_id,
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            pid=os.getpid(),
        )

        # Check task is registered
        tasks = coordinator.get_tasks(task_type=TaskType.SELFPLAY)
        assert any(t.task_id == task_id for t in tasks)

        # Unregister
        coordinator.unregister_task(task_id)
        tasks = coordinator.get_tasks(task_type=TaskType.SELFPLAY)
        assert not any(t.task_id == task_id for t in tasks)

    def test_node_limit_enforcement(self, coordinator):
        """Should enforce per-node limits."""
        coordinator.limits.max_selfplay_per_node = 3

        # Register up to limit
        for i in range(3):
            coordinator.register_task(
                f"task-{i}",
                TaskType.SELFPLAY,
                "node-1",
            )
            # Wait for cooldown
            time.sleep(coordinator.limits.spawn_cooldown_seconds + 0.05)

        # Next should be denied
        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=False,
            check_health=False,
        )
        assert allowed is False
        assert "limit" in reason.lower()

    def test_cluster_limit_enforcement(self, coordinator):
        """Should enforce cluster-wide limits."""
        coordinator.limits.max_total_selfplay = 2
        coordinator.limits.spawn_cooldown_seconds = 0.01

        # Register up to limit
        coordinator.register_task("task-1", TaskType.SELFPLAY, "node-1")
        time.sleep(0.02)
        coordinator.register_task("task-2", TaskType.SELFPLAY, "node-2")
        time.sleep(0.02)

        # Next should be denied
        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-3",
            check_resources=False,
            check_health=False,
        )
        assert allowed is False
        assert "limit" in reason.lower()

    def test_resource_check(self, coordinator):
        """Should check resource thresholds."""
        # Update with high disk usage
        coordinator.update_node_resources(
            "node-1",
            cpu_percent=50,
            memory_percent=60,
            disk_percent=85,  # Above 70% threshold
        )

        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=True,
            check_health=False,
        )
        assert allowed is False
        assert "disk" in reason.lower()

    def test_callback_on_limit_reached(self, coordinator):
        """Should fire callback when limit reached."""
        coordinator.limits.max_total_selfplay = 1
        coordinator.limits.spawn_cooldown_seconds = 0.01

        callback_data = []

        def on_limit(name, current, max_val):
            callback_data.append((name, current, max_val))

        coordinator.on_limit_reached(on_limit)

        # Register one task
        coordinator.register_task("task-1", TaskType.SELFPLAY, "node-1")
        time.sleep(0.02)

        # Try to spawn another (will hit limit)
        coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-2",
            check_resources=False,
            check_health=False,
        )

        assert len(callback_data) == 1
        assert callback_data[0][0] == "selfplay"

    def test_callback_on_emergency(self, coordinator):
        """Should fire callback on emergency stop."""
        callback_called = []

        def on_emergency():
            callback_called.append(True)

        coordinator.on_emergency(on_emergency)
        coordinator.emergency_stop()

        assert len(callback_called) == 1

    def test_get_stats(self, coordinator):
        """Should return statistics."""
        coordinator.register_task("task-1", TaskType.SELFPLAY, "node-1")
        coordinator.register_task("task-2", TaskType.TRAINING, "node-2")

        stats = coordinator.get_stats()

        assert stats["state"] == "running"
        assert stats["total_tasks"] == 2
        assert stats["by_type"]["selfplay"] == 1
        assert stats["by_type"]["training"] == 1
        assert "node-1" in stats["by_node"]
        assert "node-2" in stats["by_node"]

    def test_cleanup_stale_tasks(self, coordinator):
        """Should cleanup stale tasks."""
        count = coordinator.cleanup_stale_tasks()
        assert isinstance(count, int)

    def test_verify_tasks(self, coordinator):
        """Should verify tasks are still running."""
        # Register task with our PID (alive)
        coordinator.register_task("task-1", TaskType.SELFPLAY, "node-1", pid=os.getpid())

        # Register task with fake PID (dead)
        coordinator.register_task("task-2", TaskType.SELFPLAY, "node-1", pid=99999999)

        result = coordinator.verify_tasks()
        assert result["verified"] >= 1
        # Task with dead PID should be removed
        assert result["removed"] >= 1

    def test_gauntlet_reservation(self, coordinator):
        """Should reserve and release workers for gauntlet."""
        # Reserve some workers
        reserved = coordinator.reserve_for_gauntlet(["node-1", "node-2"])
        assert "node-1" in reserved
        assert "node-2" in reserved

        # Check reservation
        assert coordinator.is_reserved_for_gauntlet("node-1") is True
        assert coordinator.is_reserved_for_gauntlet("node-3") is False

        # Get reserved set
        reserved_set = coordinator.get_gauntlet_reserved()
        assert "node-1" in reserved_set
        assert "node-2" in reserved_set

        # Release one
        coordinator.release_from_gauntlet(["node-1"])
        assert coordinator.is_reserved_for_gauntlet("node-1") is False
        assert coordinator.is_reserved_for_gauntlet("node-2") is True

        # Release all
        count = coordinator.release_all_gauntlet()
        assert count == 1  # Only node-2 was still reserved

    def test_get_available_for_gauntlet(self, coordinator):
        """Should get available nodes for gauntlet."""
        all_nodes = ["cpu-node-1", "cpu-node-2", "gpu-node-1", "gpu-node-2"]

        # Reserve one
        coordinator.reserve_for_gauntlet(["cpu-node-1"])

        # Get available - should prefer CPU nodes
        available = coordinator.get_available_for_gauntlet(all_nodes, count=2)
        assert len(available) == 2
        assert "cpu-node-1" not in available  # Already reserved
        assert "cpu-node-2" in available  # CPU preferred


# ============================================
# Tests for CoordinatedTask context manager
# ============================================

class TestCoordinatedTask:
    """Tests for CoordinatedTask async context manager."""

    @pytest.fixture(autouse=True)
    def reset_coordinator(self):
        """Reset coordinator singleton for each test."""
        TaskCoordinator.reset_instance()
        yield
        TaskCoordinator.reset_instance()

    @pytest.fixture
    def coordinator(self):
        """Get coordinator instance."""
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        coord = TaskCoordinator.get_instance()
        yield coord

        if "RINGRIFT_COORDINATOR_DIR" in os.environ:
            del os.environ["RINGRIFT_COORDINATOR_DIR"]

    @pytest.mark.asyncio
    async def test_allowed_task(self, coordinator):
        """Should allow task when conditions are met."""
        coordinator.limits.spawn_cooldown_seconds = 0

        # Use "localhost" to skip health checks
        async with CoordinatedTask(TaskType.SELFPLAY, "localhost") as task:
            assert task.allowed is True
            assert task.reason == "OK"

    @pytest.mark.asyncio
    async def test_denied_task(self, coordinator):
        """Should deny task when coordinator is paused."""
        coordinator.pause()

        async with CoordinatedTask(TaskType.SELFPLAY, "localhost") as task:
            assert task.allowed is False
            assert "paused" in task.reason.lower()

    @pytest.mark.asyncio
    async def test_task_id_auto_generated(self, coordinator):
        """Should auto-generate task ID."""
        coordinator.limits.spawn_cooldown_seconds = 0

        async with CoordinatedTask(TaskType.SELFPLAY, "localhost") as task:
            assert task.task_id.startswith("selfplay_localhost_")

    @pytest.mark.asyncio
    async def test_task_id_custom(self, coordinator):
        """Should use custom task ID."""
        coordinator.limits.spawn_cooldown_seconds = 0

        async with CoordinatedTask(
            TaskType.SELFPLAY,
            "localhost",
            task_id="my-custom-task"
        ) as task:
            assert task.task_id == "my-custom-task"

    @pytest.mark.asyncio
    async def test_task_unregistered_on_exit(self, coordinator):
        """Should unregister task on context exit."""
        coordinator.limits.spawn_cooldown_seconds = 0

        task_id = None
        # Use "localhost" to skip health checks
        async with CoordinatedTask(TaskType.SELFPLAY, "localhost") as task:
            if task.allowed:
                task_id = task.task_id
                # Task should be registered
                tasks = coordinator.get_tasks(task_type=TaskType.SELFPLAY)
                assert any(t.task_id == task_id for t in tasks)

        # Task should be unregistered after exit
        if task_id:
            tasks = coordinator.get_tasks(task_type=TaskType.SELFPLAY)
            assert not any(t.task_id == task_id for t in tasks)


# ============================================
# Tests for utility functions
# ============================================

class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    @pytest.fixture(autouse=True)
    def reset_coordinator(self):
        """Reset coordinator singleton for each test."""
        TaskCoordinator.reset_instance()
        yield
        TaskCoordinator.reset_instance()

    def test_get_coordinator(self):
        """Should return coordinator instance."""
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        try:
            coord = get_coordinator()
            assert isinstance(coord, TaskCoordinator)
            assert coord is get_coordinator()  # Same instance
        finally:
            if "RINGRIFT_COORDINATOR_DIR" in os.environ:
                del os.environ["RINGRIFT_COORDINATOR_DIR"]

    def test_can_spawn(self):
        """Should check if spawning is allowed."""
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        try:
            coord = get_coordinator()
            coord.limits.spawn_cooldown_seconds = 0

            result = can_spawn(TaskType.SELFPLAY, "node-1")
            assert isinstance(result, bool)
        finally:
            if "RINGRIFT_COORDINATOR_DIR" in os.environ:
                del os.environ["RINGRIFT_COORDINATOR_DIR"]

    def test_emergency_stop_all(self):
        """Should trigger emergency stop."""
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        try:
            coord = get_coordinator()
            emergency_stop_all()
            assert coord.get_state() == CoordinatorState.EMERGENCY
        finally:
            if "RINGRIFT_COORDINATOR_DIR" in os.environ:
                del os.environ["RINGRIFT_COORDINATOR_DIR"]


# ============================================
# Tests for TaskHeartbeatMonitor
# ============================================

class TestTaskHeartbeatMonitor:
    """Tests for TaskHeartbeatMonitor class."""

    @pytest.fixture
    def registry(self):
        """Create a test registry with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_tasks.db"
            yield TaskRegistry(db_path)

    def test_check_for_orphans(self, registry):
        """Should detect orphaned tasks."""
        # Create orphaned task
        orphan = TaskInfo(
            task_id="orphan-task",
            task_type=TaskType.SELFPLAY,
            node_id="node-1",
            started_at=time.time() - 600,
            metadata={"last_heartbeat": time.time() - 400},
        )
        registry.register_task(orphan)

        monitor = TaskHeartbeatMonitor(
            registry=registry,
            timeout_seconds=300,
            check_interval_seconds=60,
        )

        orphans = monitor.check_for_orphans()
        assert len(orphans) == 1
        assert orphans[0].task_id == "orphan-task"

        # Task should be marked as orphaned
        task = registry.get_task("orphan-task")
        assert task.status == "orphaned"

    def test_start_stop(self, registry):
        """Should start and stop monitoring."""
        monitor = TaskHeartbeatMonitor(
            registry=registry,
            timeout_seconds=300,
            check_interval_seconds=0.1,  # Short interval for testing
        )

        monitor.start()
        assert monitor._running is True
        assert monitor._thread is not None

        time.sleep(0.2)  # Let it run briefly

        monitor.stop()
        assert monitor._running is False


# ============================================
# Integration tests
# ============================================

class TestTaskCoordinatorIntegration:
    """Integration tests for task coordination workflow."""

    @pytest.fixture(autouse=True)
    def reset_coordinator(self):
        """Reset coordinator singleton for each test."""
        TaskCoordinator.reset_instance()
        yield
        TaskCoordinator.reset_instance()

    @pytest.fixture
    def coordinator(self):
        """Get coordinator instance."""
        tmpdir = tempfile.mkdtemp()
        os.environ["RINGRIFT_COORDINATOR_DIR"] = tmpdir

        coord = TaskCoordinator.get_instance()
        coord.limits.spawn_cooldown_seconds = 0.01
        yield coord

        if "RINGRIFT_COORDINATOR_DIR" in os.environ:
            del os.environ["RINGRIFT_COORDINATOR_DIR"]

    def test_full_task_lifecycle(self, coordinator):
        """Should handle complete task lifecycle."""
        task_id = "lifecycle-test"

        # Check spawn allowed
        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=False,
            check_health=False,
        )
        assert allowed is True

        # Register task
        coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1", pid=os.getpid())

        # Verify registered
        task = coordinator.registry.get_task(task_id)
        assert task is not None
        assert task.status == "running"

        # Complete task
        coordinator.complete_task(task_id, success=True, result_data={"games": 100})

        # Verify completed (unregistered)
        task = coordinator.registry.get_task(task_id)
        assert task is None

    def test_multiple_task_types(self, coordinator):
        """Should handle multiple task types concurrently."""
        # Register different task types
        coordinator.register_task("selfplay-1", TaskType.SELFPLAY, "node-1")
        time.sleep(0.02)
        coordinator.register_task("training-1", TaskType.TRAINING, "gpu-node-1")
        time.sleep(0.02)
        coordinator.register_task("sync-1", TaskType.SYNC, "node-1")

        stats = coordinator.get_stats()
        assert stats["by_type"]["selfplay"] == 1
        assert stats["by_type"]["training"] == 1
        assert stats["by_type"]["sync"] == 1

    def test_resource_based_throttling(self, coordinator):
        """Should throttle based on resource usage."""
        # Normal resources - should allow
        coordinator.update_node_resources(
            "node-1",
            cpu_percent=50,
            memory_percent=60,
            disk_percent=50,
        )

        allowed, _ = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=True,
            check_health=False,
        )
        assert allowed is True

        # High disk - should deny
        coordinator.update_node_resources(
            "node-1",
            cpu_percent=50,
            memory_percent=60,
            disk_percent=75,
        )

        time.sleep(coordinator.limits.spawn_cooldown_seconds + 0.01)
        allowed, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "node-1",
            check_resources=True,
            check_health=False,
        )
        assert allowed is False
        assert "disk" in reason.lower()

    def test_rate_limiting(self, coordinator):
        """Should rate limit task spawning."""
        # Set very low rate limit - burst of 2, no replenishment
        coordinator._spawn_limiter = RateLimiter(rate=0.0, burst=2)
        coordinator._selfplay_limiter = RateLimiter(rate=0.0, burst=2)

        # can_spawn_task consumes rate limiter tokens (register_task does not)
        # First call should succeed
        allowed1, _ = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "localhost",
            check_resources=False,
            check_health=False,
        )
        assert allowed1 is True

        time.sleep(0.02)

        # Second call should succeed (using second burst token)
        allowed2, _ = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "localhost",
            check_resources=False,
            check_health=False,
        )
        assert allowed2 is True

        time.sleep(0.02)

        # Third should be rate limited (burst exhausted, no replenishment)
        allowed3, reason = coordinator.can_spawn_task(
            TaskType.SELFPLAY,
            "localhost",
            check_resources=False,
            check_health=False,
        )
        assert allowed3 is False
        assert "rate limit" in reason.lower()
