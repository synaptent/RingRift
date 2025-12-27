"""Tests for P2PMixinBase, EventSubscriptionMixin, and P2PManagerBase.

December 27, 2025: Comprehensive tests for the consolidated P2P mixin classes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch
from typing import Any

import pytest

from scripts.p2p.p2p_mixin_base import (
    P2PMixinBase,
    EventSubscriptionMixin,
    P2PManagerBase,
)


# =============================================================================
# P2PMixinBase Tests
# =============================================================================


class TestP2PMixinBaseDatabaseHelpers:
    """Test database helper methods."""

    def test_execute_db_query_fetch(self, tmp_path: Path) -> None:
        """Test _execute_db_query with fetch=True."""
        db_path = tmp_path / "test.db"

        # Create test database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'hello')")
        conn.execute("INSERT INTO test VALUES (2, 'world')")
        conn.commit()
        conn.close()

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.db_path = db_path

        result = mixin._execute_db_query(
            "SELECT * FROM test WHERE id = ?", (1,), fetch=True
        )
        assert result == [(1, "hello")]

    def test_execute_db_query_insert(self, tmp_path: Path) -> None:
        """Test _execute_db_query with fetch=False (insert)."""
        db_path = tmp_path / "test.db"

        # Create test database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        conn.commit()
        conn.close()

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.db_path = db_path

        result = mixin._execute_db_query(
            "INSERT INTO test VALUES (1, 'hello')", (), fetch=False
        )
        assert result == 1  # rowcount

    def test_execute_db_query_no_db_path(self) -> None:
        """Test _execute_db_query when db_path is None."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        result = mixin._execute_db_query("SELECT 1", fetch=True)
        assert result is None

    def test_db_connection_context_manager(self, tmp_path: Path) -> None:
        """Test _db_connection context manager."""
        db_path = tmp_path / "test.db"

        # Create test database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.db_path = db_path

        with mixin._db_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 0


class TestP2PMixinBaseStateHelpers:
    """Test state initialization helpers."""

    def test_ensure_state_attr_creates_dict(self) -> None:
        """Test _ensure_state_attr creates dict for cache-like names."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin._ensure_state_attr("_peer_cache")
        assert mixin._peer_cache == {}

    def test_ensure_state_attr_uses_default(self) -> None:
        """Test _ensure_state_attr uses provided default."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin._ensure_state_attr("_counter", 42)
        assert mixin._counter == 42

    def test_ensure_state_attr_idempotent(self) -> None:
        """Test _ensure_state_attr doesn't overwrite existing values."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin._counter = 100
        mixin._ensure_state_attr("_counter", 0)
        assert mixin._counter == 100

    def test_ensure_multiple_state_attrs(self) -> None:
        """Test _ensure_multiple_state_attrs initializes all attrs."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin._ensure_multiple_state_attrs(
            {"_cache": {}, "_count": 0, "_flag": False}
        )
        assert mixin._cache == {}
        assert mixin._count == 0
        assert mixin._flag is False


class TestP2PMixinBasePeerHelpers:
    """Test peer management helpers."""

    def test_count_alive_peers_empty(self) -> None:
        """Test _count_alive_peers with empty list."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.node_id = "self"

        assert mixin._count_alive_peers([]) == 0

    def test_count_alive_peers_includes_self(self) -> None:
        """Test _count_alive_peers counts self as alive."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.peers = {}
        mixin.peers_lock = threading.RLock()
        mixin.node_id = "self"

        assert mixin._count_alive_peers(["self"]) == 1

    def test_count_alive_peers_checks_is_alive(self) -> None:
        """Test _count_alive_peers calls is_alive() on peers."""

        class MockPeer:
            def __init__(self, alive: bool):
                self._alive = alive

            def is_alive(self) -> bool:
                return self._alive

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.peers = {
            "peer1": MockPeer(True),
            "peer2": MockPeer(False),
            "peer3": MockPeer(True),
        }
        mixin.peers_lock = threading.RLock()
        mixin.node_id = "self"

        assert mixin._count_alive_peers(["peer1", "peer2", "peer3"]) == 2

    def test_get_alive_peer_list(self) -> None:
        """Test _get_alive_peer_list returns correct IDs."""

        class MockPeer:
            def __init__(self, alive: bool):
                self._alive = alive

            def is_alive(self) -> bool:
                return self._alive

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        mixin.peers = {
            "peer1": MockPeer(True),
            "peer2": MockPeer(False),
        }
        mixin.peers_lock = threading.RLock()
        mixin.node_id = "self"

        result = mixin._get_alive_peer_list(["self", "peer1", "peer2"])
        assert set(result) == {"self", "peer1"}


class TestP2PMixinBaseEventHelpers:
    """Test event emission helpers."""

    def test_safe_emit_event_with_emit_event(self) -> None:
        """Test _safe_emit_event uses _emit_event if available."""
        emitted = []

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

            def _emit_event(self, event_type: str, payload: dict) -> None:
                emitted.append((event_type, payload))

        mixin = TestMixin()
        result = mixin._safe_emit_event("TEST_EVENT", {"key": "value"})

        assert result is True
        assert emitted == [("TEST_EVENT", {"key": "value"})]

    def test_safe_emit_event_no_handler(self) -> None:
        """Test _safe_emit_event returns False when no handler."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        result = mixin._safe_emit_event("TEST_EVENT", {})
        assert result is False

    def test_safe_emit_event_handles_exception(self) -> None:
        """Test _safe_emit_event catches exceptions."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

            def _emit_event(self, event_type: str, payload: dict) -> None:
                raise RuntimeError("Emit failed")

        mixin = TestMixin()
        mixin.verbose = True
        result = mixin._safe_emit_event("TEST_EVENT", {})
        assert result is False


class TestP2PMixinBaseLoggingHelpers:
    """Test logging helpers."""

    def test_logging_methods_use_prefix(self) -> None:
        """Test logging methods include MIXIN_TYPE prefix."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "my_component"

        mixin = TestMixin()

        with patch("scripts.p2p.p2p_mixin_base.logger") as mock_logger:
            mixin._log_debug("debug message")
            mock_logger.debug.assert_called_with("[my_component] debug message")

            mixin._log_info("info message")
            mock_logger.info.assert_called_with("[my_component] info message")

            mixin._log_warning("warning message")
            mock_logger.warning.assert_called_with("[my_component] warning message")

            mixin._log_error("error message")
            mock_logger.error.assert_called_with("[my_component] error message")


class TestP2PMixinBaseTimingHelpers:
    """Test timing helpers."""

    def test_get_timestamp_returns_time(self) -> None:
        """Test _get_timestamp returns current time."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        before = time.time()
        result = mixin._get_timestamp()
        after = time.time()

        assert before <= result <= after

    def test_is_expired_true(self) -> None:
        """Test _is_expired returns True for old timestamps."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        old_time = time.time() - 100
        assert mixin._is_expired(old_time, 50) is True

    def test_is_expired_false(self) -> None:
        """Test _is_expired returns False for recent timestamps."""

        class TestMixin(P2PMixinBase):
            MIXIN_TYPE = "test"

        mixin = TestMixin()
        recent_time = time.time() - 10
        assert mixin._is_expired(recent_time, 50) is False


# =============================================================================
# EventSubscriptionMixin Tests
# =============================================================================


class TestEventSubscriptionMixin:
    """Test EventSubscriptionMixin functionality."""

    def test_init_subscription_state(self) -> None:
        """Test _init_subscription_state initializes correctly."""

        class TestMixin(EventSubscriptionMixin):
            pass

        mixin = TestMixin()
        mixin._init_subscription_state()

        assert mixin._subscribed is False
        assert isinstance(mixin._subscription_lock, type(threading.RLock()))

    def test_init_subscription_state_idempotent(self) -> None:
        """Test _init_subscription_state doesn't overwrite existing values."""

        class TestMixin(EventSubscriptionMixin):
            pass

        mixin = TestMixin()
        mixin._subscribed = True
        mixin._subscription_lock = threading.RLock()

        mixin._init_subscription_state()
        assert mixin._subscribed is True  # Not reset

    def test_is_subscribed_default_false(self) -> None:
        """Test is_subscribed returns False when not subscribed."""

        class TestMixin(EventSubscriptionMixin):
            pass

        mixin = TestMixin()
        assert mixin.is_subscribed() is False

    def test_get_subscription_status(self) -> None:
        """Test get_subscription_status returns correct dict."""

        class TestMixin(EventSubscriptionMixin):
            def _get_event_subscriptions(self) -> dict:
                return {"EVENT_A": lambda e: None, "EVENT_B": lambda e: None}

        mixin = TestMixin()
        status = mixin.get_subscription_status()

        assert status["subscribed"] is False
        assert status["subscription_count"] == 2

    def test_subscribe_to_events_empty_subscriptions(self) -> None:
        """Test subscribe_to_events marks as subscribed with no events."""

        class TestMixin(EventSubscriptionMixin):
            _subscription_log_prefix = "TestMixin"

            def _get_event_subscriptions(self) -> dict:
                return {}

        mixin = TestMixin()
        mixin.subscribe_to_events()
        assert mixin.is_subscribed() is True

    def test_subscribe_to_events_thread_safe(self) -> None:
        """Test subscribe_to_events is thread-safe."""
        call_count = 0

        class TestMixin(EventSubscriptionMixin):
            _subscription_log_prefix = "TestMixin"

            def _get_event_subscriptions(self) -> dict:
                return {}

        mixin = TestMixin()

        def subscribe_thread():
            nonlocal call_count
            mixin.subscribe_to_events()
            call_count += 1

        # Spawn multiple threads trying to subscribe
        threads = [threading.Thread(target=subscribe_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads completed
        assert call_count == 10
        # Only one subscription actually happened
        assert mixin.is_subscribed() is True


# =============================================================================
# P2PManagerBase Tests
# =============================================================================


class TestP2PManagerBase:
    """Test P2PManagerBase combined functionality."""

    def test_inherits_from_both_mixins(self) -> None:
        """Test P2PManagerBase inherits from both parent classes."""
        assert issubclass(P2PManagerBase, P2PMixinBase)
        assert issubclass(P2PManagerBase, EventSubscriptionMixin)

    def test_has_all_methods(self) -> None:
        """Test P2PManagerBase has methods from both parents."""
        # From P2PMixinBase
        assert hasattr(P2PManagerBase, "_execute_db_query")
        assert hasattr(P2PManagerBase, "_ensure_state_attr")
        assert hasattr(P2PManagerBase, "_count_alive_peers")
        assert hasattr(P2PManagerBase, "_safe_emit_event")
        assert hasattr(P2PManagerBase, "_log_info")

        # From EventSubscriptionMixin
        assert hasattr(P2PManagerBase, "_init_subscription_state")
        assert hasattr(P2PManagerBase, "subscribe_to_events")
        assert hasattr(P2PManagerBase, "is_subscribed")
        assert hasattr(P2PManagerBase, "get_subscription_status")

    def test_combined_usage(self, tmp_path: Path) -> None:
        """Test using P2PManagerBase with both functionalities."""
        db_path = tmp_path / "test.db"

        # Create test database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE peers (node_id TEXT)")
        conn.commit()
        conn.close()

        class MyManager(P2PManagerBase):
            MIXIN_TYPE = "my_manager"
            _subscription_log_prefix = "MyManager"

            def __init__(self, db_path: Path):
                self.db_path = db_path
                self._init_subscription_state()

            def _get_event_subscriptions(self) -> dict:
                return {}

            def health_check(self) -> dict:
                sub_status = self.get_subscription_status()
                return {
                    "status": "healthy" if sub_status["subscribed"] else "degraded",
                    **sub_status,
                }

        manager = MyManager(db_path)
        manager.subscribe_to_events()

        # Test database helpers work
        result = manager._execute_db_query(
            "SELECT COUNT(*) FROM peers", fetch=True
        )
        assert result == [(0,)]

        # Test subscription works
        assert manager.is_subscribed() is True

        # Test health check
        health = manager.health_check()
        assert health["status"] == "healthy"
        assert health["subscribed"] is True


# =============================================================================
# Config Loading Tests
# =============================================================================


class TestP2PMixinBaseConfigHelpers:
    """Test configuration loading helpers."""

    def test_load_config_constant_success(self) -> None:
        """Test _load_config_constant loads existing constant."""
        # Use a known module and constant
        result = P2PMixinBase._load_config_constant(
            "Path", None, module_path="pathlib"
        )
        from pathlib import Path

        assert result is Path

    def test_load_config_constant_fallback(self) -> None:
        """Test _load_config_constant uses default on missing constant."""
        result = P2PMixinBase._load_config_constant(
            "NONEXISTENT_CONSTANT", 42, module_path="pathlib"
        )
        assert result == 42

    def test_load_config_constant_module_not_found(self) -> None:
        """Test _load_config_constant uses default on missing module."""
        result = P2PMixinBase._load_config_constant(
            "SOMETHING", "default", module_path="nonexistent.module.path"
        )
        assert result == "default"

    def test_load_config_constants_multiple(self) -> None:
        """Test _load_config_constants loads multiple at once."""
        result = P2PMixinBase._load_config_constants(
            {"Path": None, "MISSING": 123},
            module_path="pathlib",
        )
        from pathlib import Path

        assert result["Path"] is Path
        assert result["MISSING"] == 123
