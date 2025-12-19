"""Tests for CoordinatorBase and related mixins."""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorProtocol,
    CoordinatorRegistry,
    CoordinatorStatus,
    CoordinatorStats,
    SQLitePersistenceMixin,
    SingletonMixin,
    CallbackMixin,
    is_coordinator,
    get_coordinator_registry,
    shutdown_all_coordinators,
)


class TestCoordinatorStatus:
    """Tests for CoordinatorStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert CoordinatorStatus.INITIALIZING.value == "initializing"
        assert CoordinatorStatus.READY.value == "ready"
        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.PAUSED.value == "paused"
        assert CoordinatorStatus.DRAINING.value == "draining"
        assert CoordinatorStatus.ERROR.value == "error"
        assert CoordinatorStatus.STOPPED.value == "stopped"


class TestCoordinatorStats:
    """Tests for CoordinatorStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = CoordinatorStats()
        assert stats.status == CoordinatorStatus.INITIALIZING
        assert stats.uptime_seconds == 0.0
        assert stats.operations_count == 0
        assert stats.errors_count == 0
        assert stats.last_error is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = CoordinatorStats(
            status=CoordinatorStatus.RUNNING,
            operations_count=10,
            errors_count=2,
            extra={"custom_key": "custom_value"},
        )
        data = stats.to_dict()
        assert data["status"] == "running"
        assert data["operations_count"] == 10
        assert data["custom_key"] == "custom_value"


class SimpleCoordinator(CoordinatorBase):
    """Simple coordinator for testing."""

    def __init__(self, name: str = "test"):
        super().__init__(name=name)
        self.started = False
        self.stopped = False

    async def _do_start(self) -> None:
        self.started = True

    async def _do_stop(self) -> None:
        self.stopped = True

    async def get_stats(self) -> Dict[str, Any]:
        base_stats = await super().get_stats()
        base_stats["custom_field"] = "test_value"
        return base_stats


class TestCoordinatorBase:
    """Tests for CoordinatorBase class."""

    def test_initialization(self):
        """Test coordinator initialization."""
        coord = SimpleCoordinator("my_coordinator")
        assert coord.name == "my_coordinator"
        assert coord.status == CoordinatorStatus.INITIALIZING
        assert not coord.is_running
        assert coord.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """Test coordinator lifecycle: initialize -> start -> stop."""
        coord = SimpleCoordinator()

        # Initialize
        await coord.initialize()
        assert coord.status == CoordinatorStatus.READY

        # Start
        await coord.start()
        assert coord.status == CoordinatorStatus.RUNNING
        assert coord.is_running
        assert coord.started
        assert coord.uptime_seconds >= 0

        # Stop
        await coord.stop()
        assert coord.status == CoordinatorStatus.STOPPED
        assert not coord.is_running
        assert coord.stopped

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        """Test pausing and resuming."""
        coord = SimpleCoordinator()
        await coord.start()

        await coord.pause()
        assert coord.status == CoordinatorStatus.PAUSED
        assert coord.is_running  # Still running, just paused

        await coord.resume()
        assert coord.status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dependency_injection(self):
        """Test dependency injection pattern."""
        coord = SimpleCoordinator()

        # Set dependencies
        coord.set_dependency("work_queue", MagicMock())
        coord.set_dependency("notifier", MagicMock())

        assert coord.has_dependency("work_queue")
        assert coord.has_dependency("notifier")
        assert not coord.has_dependency("unknown")

        work_queue = coord.get_dependency("work_queue")
        assert work_queue is not None

        unknown = coord.get_dependency("unknown", default="default_value")
        assert unknown == "default_value"

    def test_record_operations(self):
        """Test operation and error recording."""
        coord = SimpleCoordinator()

        coord.record_operation()
        coord.record_operation()
        assert coord._operations_count == 2

        coord.record_error(Exception("Test error"))
        assert coord._errors_count == 1
        assert coord._last_error == "Test error"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting stats."""
        coord = SimpleCoordinator("stats_test")
        await coord.start()

        stats = await coord.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["status"] == "running"
        assert stats["is_running"] is True
        assert stats["custom_field"] == "test_value"

    @pytest.mark.asyncio
    async def test_start_auto_initializes(self):
        """Test that start() auto-initializes if needed."""
        coord = SimpleCoordinator()
        assert coord.status == CoordinatorStatus.INITIALIZING

        await coord.start()
        assert coord.status == CoordinatorStatus.RUNNING
        assert coord.started


class SQLiteCoordinator(CoordinatorBase, SQLitePersistenceMixin):
    """Coordinator with SQLite persistence for testing."""

    def __init__(self, db_path: Path):
        super().__init__(name="sqlite_test")
        self.init_db(db_path)

    def _get_schema(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS test_data (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
        """

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestSQLitePersistenceMixin:
    """Tests for SQLitePersistenceMixin."""

    def test_db_initialization(self):
        """Test database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            assert db_path.exists()

            # Verify table was created
            conn = coord._get_connection()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_data'"
            )
            assert cursor.fetchone() is not None

    def test_thread_local_connection(self):
        """Test that connections are thread-local."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            conn1 = coord._get_connection()
            conn2 = coord._get_connection()
            assert conn1 is conn2  # Same thread, same connection

    def test_wal_mode(self):
        """Test that WAL mode is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            conn = coord._get_connection()
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"


class SingletonCoordinator(CoordinatorBase, SingletonMixin):
    """Singleton coordinator for testing."""

    def __init__(self, value: str = "default"):
        super().__init__(name="singleton_test")
        self.value = value

    @classmethod
    def get_instance(cls, value: str = "default") -> "SingletonCoordinator":
        return cls._get_or_create_instance(value)

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestSingletonMixin:
    """Tests for SingletonMixin."""

    def teardown_method(self):
        """Clean up singleton instances after each test."""
        SingletonCoordinator._clear_instance()

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        inst1 = SingletonCoordinator.get_instance("first")
        inst2 = SingletonCoordinator.get_instance("second")

        assert inst1 is inst2
        assert inst1.value == "first"  # First value is preserved

    def test_clear_instance(self):
        """Test clearing the singleton instance."""
        inst1 = SingletonCoordinator.get_instance("first")
        SingletonCoordinator._clear_instance()

        inst2 = SingletonCoordinator.get_instance("second")
        assert inst1 is not inst2
        assert inst2.value == "second"


class CallbackCoordinator(CoordinatorBase, CallbackMixin):
    """Coordinator with callbacks for testing."""

    def __init__(self):
        super().__init__(name="callback_test")
        self.__init_callbacks__()

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestCallbackMixin:
    """Tests for CallbackMixin."""

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        """Test synchronous callback invocation."""
        coord = CallbackCoordinator()
        results = []

        def on_event(value):
            results.append(value)
            return "sync_result"

        coord.register_callback("test_event", on_event)
        returned = await coord.invoke_callbacks("test_event", "test_value")

        assert results == ["test_value"]
        assert returned == ["sync_result"]

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test asynchronous callback invocation."""
        coord = CallbackCoordinator()
        results = []

        async def on_event(value):
            results.append(value)
            return "async_result"

        coord.register_callback("test_event", on_event)
        returned = await coord.invoke_callbacks("test_event", "test_value")

        assert results == ["test_value"]
        assert returned == ["async_result"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Test multiple callbacks for same event."""
        coord = CallbackCoordinator()
        results = []

        coord.register_callback("test_event", lambda x: results.append(f"1:{x}"))
        coord.register_callback("test_event", lambda x: results.append(f"2:{x}"))

        await coord.invoke_callbacks("test_event", "value")

        assert results == ["1:value", "2:value"]

    @pytest.mark.asyncio
    async def test_unregister_callback(self):
        """Test unregistering a callback."""
        coord = CallbackCoordinator()
        results = []

        def my_callback(x):
            results.append(x)

        coord.register_callback("test_event", my_callback)
        coord.unregister_callback("test_event", my_callback)

        await coord.invoke_callbacks("test_event", "value")

        assert results == []

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test that errors in callbacks don't break invocation."""
        coord = CallbackCoordinator()
        results = []

        def failing_callback(x):
            raise ValueError("Intentional error")

        def working_callback(x):
            results.append(x)
            return "worked"

        coord.register_callback("test_event", failing_callback)
        coord.register_callback("test_event", working_callback)

        returned = await coord.invoke_callbacks("test_event", "value")

        assert results == ["value"]
        assert returned == [None, "worked"]  # First is None due to error


class TestCoordinatorProtocol:
    """Tests for CoordinatorProtocol."""

    def test_protocol_check(self):
        """Test that protocol checking works."""
        coord = SimpleCoordinator()
        assert isinstance(coord, CoordinatorProtocol)

    def test_is_coordinator_function(self):
        """Test is_coordinator helper function."""
        coord = SimpleCoordinator()
        assert is_coordinator(coord)

        not_coord = {"status": "running"}
        assert not is_coordinator(not_coord)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class FailingCoordinator(CoordinatorBase):
    """Coordinator that fails during lifecycle methods."""

    def __init__(self, fail_on: str = "start"):
        super().__init__(name="failing_test")
        self.fail_on = fail_on

    async def _do_initialize(self) -> None:
        if self.fail_on == "initialize":
            raise ValueError("Intentional init failure")

    async def _do_start(self) -> None:
        if self.fail_on == "start":
            raise ValueError("Intentional start failure")

    async def _do_stop(self) -> None:
        if self.fail_on == "stop":
            raise ValueError("Intentional stop failure")

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestCoordinatorBaseEdgeCases:
    """Edge case tests for CoordinatorBase."""

    @pytest.mark.asyncio
    async def test_start_failure_sets_error_state(self):
        """Test that start failure sets error state."""
        coord = FailingCoordinator(fail_on="start")

        with pytest.raises(ValueError, match="Intentional start failure"):
            await coord.start()

        assert coord.status == CoordinatorStatus.ERROR
        assert not coord.is_running
        assert "Intentional start failure" in coord._last_error

    @pytest.mark.asyncio
    async def test_initialize_failure_sets_error_state(self):
        """Test that initialize failure sets error state."""
        coord = FailingCoordinator(fail_on="initialize")

        with pytest.raises(ValueError, match="Intentional init failure"):
            await coord.initialize()

        assert coord.status == CoordinatorStatus.ERROR

    @pytest.mark.asyncio
    async def test_stop_failure_sets_error_state(self):
        """Test that stop failure sets error state."""
        coord = FailingCoordinator(fail_on="stop")
        await coord.start()  # Start successfully

        with pytest.raises(ValueError, match="Intentional stop failure"):
            await coord.stop()

        assert coord.status == CoordinatorStatus.ERROR

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        """Test that starting an already running coordinator is a no-op."""
        coord = SimpleCoordinator()
        await coord.start()
        first_start_time = coord._start_time

        await asyncio.sleep(0.01)  # Small delay
        await coord.start()

        assert coord._start_time == first_start_time  # Unchanged

    @pytest.mark.asyncio
    async def test_double_stop_is_noop(self):
        """Test that stopping an already stopped coordinator is a no-op."""
        coord = SimpleCoordinator()
        await coord.start()
        await coord.stop()
        assert coord.status == CoordinatorStatus.STOPPED

        await coord.stop()  # Should not raise
        assert coord.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_calls_stop(self):
        """Test that shutdown calls stop first."""
        coord = SimpleCoordinator()
        await coord.start()

        await coord.shutdown()

        assert coord.status == CoordinatorStatus.STOPPED
        assert coord.stopped

    @pytest.mark.asyncio
    async def test_pause_when_not_running(self):
        """Test pause when not running is ignored."""
        coord = SimpleCoordinator()

        await coord.pause()
        assert coord.status == CoordinatorStatus.INITIALIZING  # Unchanged

    @pytest.mark.asyncio
    async def test_resume_when_not_paused(self):
        """Test resume when not paused is ignored."""
        coord = SimpleCoordinator()
        await coord.start()

        await coord.resume()
        assert coord.status == CoordinatorStatus.RUNNING  # Unchanged


class TestSQLitePersistenceMixinEdgeCases:
    """Edge case tests for SQLitePersistenceMixin."""

    def test_get_connection_without_init_raises(self):
        """Test that getting connection before init raises error."""
        class UninitializedCoord(CoordinatorBase, SQLitePersistenceMixin):
            def __init__(self):
                super().__init__(name="uninit_test")
                # Don't call init_db()

            def _get_schema(self) -> str:
                return ""

            async def get_stats(self) -> Dict[str, Any]:
                return {}

        coord = UninitializedCoord()
        with pytest.raises(RuntimeError, match="Database not initialized"):
            coord._get_connection()

    def test_close_connection(self):
        """Test that connection can be closed and reopened."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            conn1 = coord._get_connection()
            coord._close_connection()

            # Connection should be None after close
            assert not hasattr(coord._db_local, "conn") or coord._db_local.conn is None

            # New connection should work
            conn2 = coord._get_connection()
            assert conn2 is not None
            assert conn1 is not conn2

    def test_multiple_coordinators_same_db(self):
        """Test multiple coordinators can share the same database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "shared.db"

            coord1 = SQLiteCoordinator(db_path)
            coord2 = SQLiteCoordinator(db_path)

            # Both should have their own connections
            conn1 = coord1._get_connection()
            conn2 = coord2._get_connection()

            # Both should be able to access the table
            conn1.execute("INSERT INTO test_data (value) VALUES ('from_coord1')")
            conn1.commit()

            cursor = conn2.execute("SELECT value FROM test_data")
            assert cursor.fetchone()[0] == "from_coord1"

    def test_multithreaded_access(self):
        """Test thread-local connections work correctly."""
        import concurrent.futures
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "threaded.db"
            coord = SQLiteCoordinator(db_path)

            thread_ids = []
            conn_ids = []
            errors = []

            def get_conn():
                try:
                    thread_ids.append(threading.current_thread().ident)
                    conn = coord._get_connection()
                    conn_ids.append(id(conn))
                    # Do some work
                    conn.execute("INSERT INTO test_data (value) VALUES (?)", (f"thread_{id(conn)}",))
                    conn.commit()
                except Exception as e:
                    errors.append(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(get_conn) for _ in range(4)]
                concurrent.futures.wait(futures)

            assert not errors, f"Errors during multithreaded access: {errors}"
            # Each unique thread should get its own unique connection
            # Note: ThreadPoolExecutor may reuse threads, so we check that unique threads == unique connections
            assert len(set(thread_ids)) == len(set(conn_ids))


class TestSingletonMixinEdgeCases:
    """Edge case tests for SingletonMixin."""

    def teardown_method(self):
        """Clean up singleton instances after each test."""
        SingletonCoordinator._clear_instance()

    def test_singleton_thread_safety(self):
        """Test singleton pattern is thread-safe."""
        import concurrent.futures

        instances = []

        def get_inst():
            inst = SingletonCoordinator.get_instance("test")
            instances.append(id(inst))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_inst) for _ in range(10)]
            concurrent.futures.wait(futures)

        # All should be the same instance
        assert len(set(instances)) == 1


class TestCallbackMixinEdgeCases:
    """Edge case tests for CallbackMixin."""

    @pytest.mark.asyncio
    async def test_invoke_with_no_callbacks(self):
        """Test invoking event with no registered callbacks."""
        coord = CallbackCoordinator()

        results = await coord.invoke_callbacks("nonexistent_event", "value")
        assert results == []

    @pytest.mark.asyncio
    async def test_invoke_with_no_args(self):
        """Test invoking callback with no arguments."""
        coord = CallbackCoordinator()
        called = []

        coord.register_callback("no_args", lambda: called.append("called"))
        await coord.invoke_callbacks("no_args")

        assert called == ["called"]

    @pytest.mark.asyncio
    async def test_manual_clear_callbacks(self):
        """Test manually clearing callbacks for an event."""
        coord = CallbackCoordinator()
        results = []

        cb1 = lambda: results.append(1)
        cb2 = lambda: results.append(2)
        coord.register_callback("event", cb1)
        coord.register_callback("event", cb2)

        # Manually clear by unregistering both
        coord.unregister_callback("event", cb1)
        coord.unregister_callback("event", cb2)

        await coord.invoke_callbacks("event")
        assert results == []

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_callback(self):
        """Test unregistering a callback that doesn't exist."""
        coord = CallbackCoordinator()

        # Should not raise
        coord.unregister_callback("event", lambda: None)
        coord.unregister_callback("nonexistent", lambda: None)


# =============================================================================
# Coordinator Registry Tests
# =============================================================================

class TestCoordinatorRegistry:
    """Tests for CoordinatorRegistry."""

    def setup_method(self):
        """Reset singleton before each test."""
        CoordinatorRegistry.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        CoordinatorRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        reg1 = CoordinatorRegistry.get_instance()
        reg2 = CoordinatorRegistry.get_instance()
        assert reg1 is reg2

    def test_register_coordinator(self):
        """Test registering a coordinator."""
        registry = CoordinatorRegistry.get_instance()
        coord = SimpleCoordinator("test_coord")

        registry.register(coord)

        assert "test_coord" in registry.list_coordinators()
        assert registry.get("test_coord") is coord

    def test_unregister_coordinator(self):
        """Test unregistering a coordinator."""
        registry = CoordinatorRegistry.get_instance()
        coord = SimpleCoordinator("test_coord")
        registry.register(coord)

        removed = registry.unregister("test_coord")

        assert removed is coord
        assert "test_coord" not in registry.list_coordinators()

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent coordinator."""
        registry = CoordinatorRegistry.get_instance()
        removed = registry.unregister("nonexistent")
        assert removed is None

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test graceful shutdown of all coordinators."""
        registry = CoordinatorRegistry.get_instance()

        coord1 = SimpleCoordinator("coord1")
        coord2 = SimpleCoordinator("coord2")

        await coord1.start()
        await coord2.start()

        registry.register(coord1)
        registry.register(coord2)

        results = await registry.shutdown_all(timeout=5.0)

        assert results["coord1"] is True
        assert results["coord2"] is True
        assert coord1.status == CoordinatorStatus.STOPPED
        assert coord2.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_with_timeout(self):
        """Test shutdown timeout handling."""

        class SlowCoordinator(CoordinatorBase):
            async def _do_stop(self) -> None:
                await asyncio.sleep(10)  # Slow shutdown

            async def get_stats(self) -> Dict[str, Any]:
                return {}

        registry = CoordinatorRegistry.get_instance()
        slow = SlowCoordinator("slow")
        await slow.start()
        registry.register(slow)

        results = await registry.shutdown_all(timeout=0.1)

        assert results["slow"] is False  # Timed out

    def test_get_health_summary(self):
        """Test health summary generation."""
        registry = CoordinatorRegistry.get_instance()

        coord = SimpleCoordinator("health_test")
        registry.register(coord)

        summary = registry.get_health_summary()

        assert summary["coordinator_count"] == 1
        assert "health_test" in summary["coordinators"]
        assert summary["coordinators"]["health_test"]["status"] == "initializing"

    @pytest.mark.asyncio
    async def test_drain_all(self):
        """Test draining all coordinators."""
        registry = CoordinatorRegistry.get_instance()

        coord1 = SimpleCoordinator("drain1")
        coord2 = SimpleCoordinator("drain2")

        await coord1.start()
        await coord2.start()

        registry.register(coord1)
        registry.register(coord2)

        await registry.drain_all(timeout=5.0)

        assert coord1.status == CoordinatorStatus.STOPPED
        assert coord2.status == CoordinatorStatus.STOPPED

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        registry = get_coordinator_registry()
        assert isinstance(registry, CoordinatorRegistry)

    @pytest.mark.asyncio
    async def test_shutdown_all_convenience(self):
        """Test shutdown_all_coordinators convenience function."""
        # Should work even with no coordinators
        results = await shutdown_all_coordinators(timeout=1.0)
        assert results == {}
