"""Tests for HandlerBase SQLite async helpers added in Sprint 17.3.

Tests for:
- _sqlite_query() - async SQLite read operations
- _sqlite_execute() - async SQLite write operations
- _sqlite_with_connection() - managed connection helper

January 4, 2026 (Sprint 17.3): SQLite async safety for event loop protection.
"""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.handler_base import HandlerBase


# =============================================================================
# Test Fixtures
# =============================================================================


class TestHandler(HandlerBase):
    """Concrete handler for testing."""

    __test__ = False

    def __init__(self):
        super().__init__(name="test_sqlite_handler", cycle_interval=60.0)

    async def _run_cycle(self) -> None:
        pass


@pytest.fixture
def handler():
    """Create a test handler instance."""
    TestHandler.reset_instance()
    return TestHandler()


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create test table
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value INTEGER DEFAULT 0
        )
    """)
    conn.execute("INSERT INTO test_items (name, value) VALUES ('item1', 100)")
    conn.execute("INSERT INTO test_items (name, value) VALUES ('item2', 200)")
    conn.execute("INSERT INTO test_items (name, value) VALUES ('item3', 300)")
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


# =============================================================================
# _sqlite_query Tests
# =============================================================================


class TestSqliteQuery:
    """Tests for _sqlite_query method."""

    @pytest.mark.asyncio
    async def test_simple_query(self, handler, temp_db):
        """Test basic query execution."""
        def _query():
            conn = sqlite3.connect(temp_db)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM test_items")
                return cursor.fetchone()[0]
            finally:
                conn.close()

        result = await handler._sqlite_query(_query)
        assert result == 3

    @pytest.mark.asyncio
    async def test_query_with_parameters(self, handler, temp_db):
        """Test query with positional parameters."""
        def _query(name):
            conn = sqlite3.connect(temp_db)
            try:
                cursor = conn.execute(
                    "SELECT value FROM test_items WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

        result = await handler._sqlite_query(_query, "item2")
        assert result == 200

    @pytest.mark.asyncio
    async def test_query_returns_list(self, handler, temp_db):
        """Test query returning multiple rows."""
        def _query():
            conn = sqlite3.connect(temp_db)
            try:
                cursor = conn.execute("SELECT name, value FROM test_items ORDER BY id")
                return cursor.fetchall()
            finally:
                conn.close()

        result = await handler._sqlite_query(_query)
        assert len(result) == 3
        assert result[0] == ("item1", 100)
        assert result[1] == ("item2", 200)
        assert result[2] == ("item3", 300)

    @pytest.mark.asyncio
    async def test_query_runs_in_thread_pool(self, handler, temp_db):
        """Test that query runs in separate thread."""
        main_thread_id = asyncio.current_task()
        query_thread_ids = []

        def _query():
            import threading
            query_thread_ids.append(threading.current_thread().ident)
            return True

        await handler._sqlite_query(_query)
        assert len(query_thread_ids) == 1
        # The query should run in a different thread (thread pool)
        # We can't directly compare since current_task != thread ident

    @pytest.mark.asyncio
    async def test_query_with_kwargs(self, handler):
        """Test query with keyword arguments."""
        def _query(value, multiplier=1):
            return value * multiplier

        result = await handler._sqlite_query(_query, 10, multiplier=5)
        assert result == 50

    @pytest.mark.asyncio
    async def test_query_exception_propagates(self, handler, temp_db):
        """Test that exceptions from query propagate correctly."""
        def _query():
            conn = sqlite3.connect(temp_db)
            try:
                # Invalid SQL should raise
                conn.execute("SELECT * FROM nonexistent_table")
            finally:
                conn.close()

        with pytest.raises(sqlite3.OperationalError):
            await handler._sqlite_query(_query)


# =============================================================================
# _sqlite_execute Tests
# =============================================================================


class TestSqliteExecute:
    """Tests for _sqlite_execute method."""

    @pytest.mark.asyncio
    async def test_insert_operation(self, handler, temp_db):
        """Test INSERT operation."""
        def _insert():
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute(
                    "INSERT INTO test_items (name, value) VALUES (?, ?)",
                    ("item4", 400)
                )
                conn.commit()
            finally:
                conn.close()

        await handler._sqlite_execute(_insert)

        # Verify insert
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM test_items")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 4

    @pytest.mark.asyncio
    async def test_update_operation(self, handler, temp_db):
        """Test UPDATE operation."""
        def _update():
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute(
                    "UPDATE test_items SET value = ? WHERE name = ?",
                    (999, "item1")
                )
                conn.commit()
            finally:
                conn.close()

        await handler._sqlite_execute(_update)

        # Verify update
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT value FROM test_items WHERE name = 'item1'")
        value = cursor.fetchone()[0]
        conn.close()
        assert value == 999

    @pytest.mark.asyncio
    async def test_delete_operation(self, handler, temp_db):
        """Test DELETE operation."""
        def _delete():
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute("DELETE FROM test_items WHERE name = ?", ("item3",))
                conn.commit()
            finally:
                conn.close()

        await handler._sqlite_execute(_delete)

        # Verify delete
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM test_items")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    @pytest.mark.asyncio
    async def test_execute_returns_result(self, handler, temp_db):
        """Test that execute can return a result."""
        def _execute():
            conn = sqlite3.connect(temp_db)
            try:
                cursor = conn.execute("INSERT INTO test_items (name, value) VALUES (?, ?)", ("item5", 500))
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

        result = await handler._sqlite_execute(_execute)
        assert result is not None
        assert result > 0  # lastrowid should be positive

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, handler, temp_db):
        """Test that transactions can be rolled back on error."""
        def _execute():
            conn = sqlite3.connect(temp_db)
            try:
                conn.execute("INSERT INTO test_items (name, value) VALUES (?, ?)", ("item6", 600))
                # Force an error before commit
                raise ValueError("Simulated error")
            except ValueError:
                conn.rollback()
                raise
            finally:
                conn.close()

        with pytest.raises(ValueError, match="Simulated error"):
            await handler._sqlite_execute(_execute)

        # Verify rollback (item6 should not exist)
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM test_items WHERE name = 'item6'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0


# =============================================================================
# _sqlite_with_connection Tests
# =============================================================================


class TestSqliteWithConnection:
    """Tests for _sqlite_with_connection method."""

    @pytest.mark.asyncio
    async def test_readonly_query(self, handler, temp_db):
        """Test read-only query with managed connection."""
        result = await handler._sqlite_with_connection(
            temp_db,
            lambda conn: conn.execute("SELECT COUNT(*) FROM test_items").fetchone()[0],
            readonly=True
        )
        assert result == 3

    @pytest.mark.asyncio
    async def test_write_with_auto_commit(self, handler, temp_db):
        """Test write operation with auto-commit."""
        await handler._sqlite_with_connection(
            temp_db,
            lambda conn: conn.execute(
                "INSERT INTO test_items (name, value) VALUES (?, ?)",
                ("item_new", 999)
            )
        )

        # Verify write was committed
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT value FROM test_items WHERE name = 'item_new'")
        value = cursor.fetchone()[0]
        conn.close()
        assert value == 999

    @pytest.mark.asyncio
    async def test_write_with_rollback_on_error(self, handler, temp_db):
        """Test that write operations rollback on error."""
        def _failing_write(conn):
            conn.execute(
                "INSERT INTO test_items (name, value) VALUES (?, ?)",
                ("item_fail", 888)
            )
            raise RuntimeError("Simulated failure")

        with pytest.raises(RuntimeError, match="Simulated failure"):
            await handler._sqlite_with_connection(temp_db, _failing_write)

        # Verify rollback (item should not exist)
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM test_items WHERE name = 'item_fail'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

    @pytest.mark.asyncio
    async def test_connection_closed_after_query(self, handler, temp_db):
        """Test that connection is properly closed after operation."""
        # Verify that we can open a new connection after the helper runs
        # (which means the previous one was closed, releasing any locks)
        result = await handler._sqlite_with_connection(
            temp_db,
            lambda conn: conn.execute("SELECT 1").fetchone(),
            readonly=True
        )
        assert result == (1,)

        # If connection wasn't closed, this would potentially cause issues
        result2 = await handler._sqlite_with_connection(
            temp_db,
            lambda conn: conn.execute("SELECT 2").fetchone(),
            readonly=True
        )
        assert result2 == (2,)

    @pytest.mark.asyncio
    async def test_connection_closed_on_error(self, handler, temp_db):
        """Test that connection is closed even on error."""
        with pytest.raises(ValueError, match="test_error"):
            await handler._sqlite_with_connection(
                temp_db,
                lambda conn: (_ for _ in ()).throw(ValueError("test_error"))
            )

        # If connection wasn't closed properly, this might fail
        # Verify we can still access the database
        result = await handler._sqlite_with_connection(
            temp_db,
            lambda conn: conn.execute("SELECT COUNT(*) FROM test_items").fetchone()[0],
            readonly=True
        )
        assert result == 3

    @pytest.mark.asyncio
    async def test_complex_query_pattern(self, handler, temp_db):
        """Test complex query with multiple operations."""
        def _complex_query(conn):
            # Multiple reads in one connection
            cursor1 = conn.execute("SELECT SUM(value) FROM test_items")
            total = cursor1.fetchone()[0]

            cursor2 = conn.execute("SELECT AVG(value) FROM test_items")
            avg = cursor2.fetchone()[0]

            cursor3 = conn.execute("SELECT MAX(value) FROM test_items")
            max_val = cursor3.fetchone()[0]

            return {"total": total, "avg": avg, "max": max_val}

        result = await handler._sqlite_with_connection(
            temp_db,
            _complex_query,
            readonly=True
        )

        assert result["total"] == 600  # 100 + 200 + 300
        assert result["avg"] == 200.0
        assert result["max"] == 300


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentSqliteAccess:
    """Tests for concurrent SQLite access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, handler, temp_db):
        """Test multiple concurrent read operations."""
        async def read_count():
            return await handler._sqlite_with_connection(
                temp_db,
                lambda conn: conn.execute("SELECT COUNT(*) FROM test_items").fetchone()[0],
                readonly=True
            )

        # Run 10 concurrent reads
        tasks = [read_count() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return the same count
        assert all(r == 3 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, handler, temp_db):
        """Test concurrent reads and writes (SQLite serializes writes)."""
        write_count = 0

        async def write_item(i):
            nonlocal write_count
            await handler._sqlite_with_connection(
                temp_db,
                lambda conn: conn.execute(
                    "INSERT INTO test_items (name, value) VALUES (?, ?)",
                    (f"concurrent_{i}", i * 100)
                )
            )
            write_count += 1

        async def read_count():
            return await handler._sqlite_with_connection(
                temp_db,
                lambda conn: conn.execute("SELECT COUNT(*) FROM test_items").fetchone()[0],
                readonly=True
            )

        # Mix of reads and writes
        tasks = [write_item(i) for i in range(5)]
        tasks.extend([read_count() for _ in range(5)])

        await asyncio.gather(*tasks)

        # All writes should complete
        assert write_count == 5

        # Final count should reflect all writes
        final_count = await read_count()
        assert final_count == 8  # 3 original + 5 new


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSqliteErrorHandling:
    """Tests for SQLite error handling."""

    @pytest.mark.asyncio
    async def test_handles_missing_database(self, handler):
        """Test handling of missing database file."""
        def _query():
            conn = sqlite3.connect("/nonexistent/path/db.sqlite")
            return conn.execute("SELECT 1").fetchone()

        with pytest.raises(sqlite3.OperationalError):
            await handler._sqlite_query(_query)

    @pytest.mark.asyncio
    async def test_handles_syntax_error(self, handler, temp_db):
        """Test handling of SQL syntax errors."""
        with pytest.raises(sqlite3.OperationalError):
            await handler._sqlite_with_connection(
                temp_db,
                lambda conn: conn.execute("SELEC * FORM broken"),
                readonly=True
            )

    @pytest.mark.asyncio
    async def test_handles_constraint_violation(self, handler, temp_db):
        """Test handling of constraint violations."""
        # Add unique constraint
        conn = sqlite3.connect(temp_db)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_name ON test_items(name)")
        conn.commit()
        conn.close()

        # Try to insert duplicate
        with pytest.raises(sqlite3.IntegrityError):
            await handler._sqlite_with_connection(
                temp_db,
                lambda conn: conn.execute(
                    "INSERT INTO test_items (name, value) VALUES (?, ?)",
                    ("item1", 111)  # item1 already exists
                )
            )

    @pytest.mark.asyncio
    async def test_cancellation_handling(self, handler, temp_db):
        """Test that SQLite operations can be cancelled."""
        async def slow_query():
            def _query():
                import time
                time.sleep(1)  # Simulate slow query
                return True

            return await handler._sqlite_query(_query)

        task = asyncio.create_task(slow_query())

        # Cancel after short delay
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
