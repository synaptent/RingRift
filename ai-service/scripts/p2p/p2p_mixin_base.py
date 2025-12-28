"""P2P Mixin Base Class - Shared Functionality for P2P Mixins.

This module provides a unified base class for all P2P orchestrator mixins,
eliminating duplicate code patterns across 6 mixin files.

Shared Functionality:
- Database connection helpers with automatic cleanup
- State initialization patterns
- Peer alive counting
- Event emission with error handling
- Configuration constant loading

Usage:
    class MyMixin(P2PMixinBase):
        MIXIN_TYPE = "my_mixin"

        def my_method(self):
            # Use shared DB helper
            result = self._execute_db_query(
                "SELECT * FROM peers WHERE active = ?",
                (True,),
                fetch=True,
            )

            # Use state initialization
            self._ensure_state_attr("_my_cache", {})

            # Use peer counting
            alive_count = self._count_alive_peers(self.voter_node_ids)

Created: December 27, 2025
Consolidates patterns from: peer_manager.py, membership_mixin.py, leader_election.py,
                           gossip_protocol.py, consensus_mixin.py, metrics_manager.py
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from threading import RLock

    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


class P2PMixinBase:
    """Unified base class for all P2P orchestrator mixins.

    Provides shared functionality for database operations, state management,
    peer tracking, and event emission. Subclasses should set MIXIN_TYPE
    for logging identification.

    Required parent class attributes (type hints for IDE support):
    - node_id: str - This node's identifier
    - peers: dict[str, NodeInfo] - Active peer connections
    - peers_lock: RLock - Thread lock for peers dict
    - db_path: Path - Path to SQLite database (optional, for DB methods)
    - verbose: bool - Verbose logging flag (optional)

    Subclasses should set:
    - MIXIN_TYPE: str - Identifier for logging (e.g., "peer_manager")
    """

    # Identifier for logging - subclasses should override
    MIXIN_TYPE: ClassVar[str] = "base"

    # Required attributes from parent class (P2POrchestrator)
    # These are type hints only - actual values come from parent
    node_id: str
    peers: dict[str, Any]  # dict[str, NodeInfo]
    peers_lock: "RLock"
    db_path: Path
    verbose: bool

    # =========================================================================
    # Database Helpers
    # =========================================================================

    def _execute_db_query(
        self,
        query: str,
        params: tuple = (),
        fetch: bool = False,
        commit: bool = True,
        timeout: float = 5.0,
    ) -> Any | None:
        """Execute a database query with automatic connection cleanup.

        Eliminates boilerplate try-finally-close pattern found across all mixins.

        Args:
            query: SQL query string
            params: Query parameters (default empty tuple)
            fetch: If True, return fetchall() results; otherwise return rowcount
            commit: If True, commit transaction after execution
            timeout: SQLite connection timeout in seconds

        Returns:
            Query results if fetch=True, affected rowcount if fetch=False,
            or None on error

        Example:
            # Fetch query
            rows = self._execute_db_query(
                "SELECT * FROM peers WHERE active = ?",
                (True,),
                fetch=True,
            )

            # Insert/update
            affected = self._execute_db_query(
                "UPDATE peers SET last_seen = ? WHERE node_id = ?",
                (time.time(), node_id),
                fetch=False,
            )
        """
        conn = None
        try:
            db_path = getattr(self, "db_path", None)
            if db_path is None:
                logger.debug(f"[{self.MIXIN_TYPE}] No db_path available")
                return None

            conn = sqlite3.connect(str(db_path), timeout=timeout)
            cursor = conn.cursor()
            cursor.execute(query, params)

            if fetch:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount

            if commit:
                conn.commit()

            return result

        except sqlite3.Error as e:
            verbose = getattr(self, "verbose", False)
            if verbose:
                logger.debug(f"[{self.MIXIN_TYPE}] DB query error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[{self.MIXIN_TYPE}] Unexpected DB error: {e}")
            return None
        finally:
            if conn:
                conn.close()

    @contextlib.contextmanager
    def _db_connection(
        self,
        timeout: float = 5.0,
    ) -> "Generator[sqlite3.Connection | None, None, None]":
        """Context manager for database connections.

        Provides automatic connection cleanup. Yields None if connection fails.

        Args:
            timeout: SQLite connection timeout in seconds

        Yields:
            sqlite3.Connection or None if connection failed

        Example:
            with self._db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM peers")
                    conn.commit()
        """
        conn = None
        try:
            db_path = getattr(self, "db_path", None)
            if db_path is None:
                yield None
                return

            conn = sqlite3.connect(str(db_path), timeout=timeout)
            yield conn

        except sqlite3.Error as e:
            verbose = getattr(self, "verbose", False)
            if verbose:
                logger.debug(f"[{self.MIXIN_TYPE}] Connection error: {e}")
            yield None
        except Exception as e:
            logger.warning(f"[{self.MIXIN_TYPE}] Unexpected connection error: {e}")
            yield None
        finally:
            if conn:
                conn.close()

    # =========================================================================
    # State Initialization Helpers
    # =========================================================================

    def _ensure_state_attr(
        self,
        attr_name: str,
        default_value: Any = None,
    ) -> None:
        """Ensure a state attribute exists, initialize if not present.

        Eliminates the `if not hasattr(self, x): self.x = {}` pattern
        found across all mixins.

        Args:
            attr_name: Name of the attribute to ensure exists
            default_value: Default value if attribute doesn't exist
                          (uses empty dict if None and attr name suggests dict)

        Example:
            self._ensure_state_attr("_peer_cache", {})
            self._ensure_state_attr("_error_count", 0)
            self._ensure_state_attr("_running", False)
        """
        if not hasattr(self, attr_name):
            # If no default provided and name suggests a collection, use dict
            if default_value is None and attr_name.startswith("_") and (
                "cache" in attr_name or
                "dict" in attr_name or
                "map" in attr_name or
                "states" in attr_name or
                "peers" in attr_name
            ):
                default_value = {}
            setattr(self, attr_name, default_value)

    def _ensure_multiple_state_attrs(
        self,
        attrs: dict[str, Any],
    ) -> None:
        """Ensure multiple state attributes exist.

        Args:
            attrs: Dictionary of {attr_name: default_value}

        Example:
            self._ensure_multiple_state_attrs({
                "_peer_cache": {},
                "_error_count": 0,
                "_running": False,
            })
        """
        for attr_name, default_value in attrs.items():
            self._ensure_state_attr(attr_name, default_value)

    # =========================================================================
    # Peer Management Helpers
    # =========================================================================

    def _count_alive_peers(self, node_ids: list[str]) -> int:
        """Count how many of the given node IDs are currently alive.

        Thread-safe peer counting using peers_lock.
        Counts self as alive if present in node_ids.

        Args:
            node_ids: List of node IDs to check

        Returns:
            Number of alive peers (including self if in list)

        Example:
            alive_voters = self._count_alive_peers(self.voter_node_ids)
            if alive_voters >= quorum:
                # Have quorum
        """
        if not node_ids:
            return 0

        alive = 0

        # Get snapshot of peers under lock
        peers_lock = getattr(self, "peers_lock", None)
        peers = getattr(self, "peers", {})
        node_id = getattr(self, "node_id", None)

        if peers_lock:
            with peers_lock:
                peers = dict(peers)
        else:
            peers = dict(peers)

        for nid in node_ids:
            # Count self as alive
            if nid == node_id:
                alive += 1
                continue

            # Check if peer is alive
            peer = peers.get(nid)
            if peer and hasattr(peer, "is_alive") and peer.is_alive():
                alive += 1

        return alive

    def _get_alive_peer_list(self, node_ids: list[str]) -> list[str]:
        """Get list of node IDs that are currently alive.

        Thread-safe version that returns the actual IDs, not just count.

        Args:
            node_ids: List of node IDs to check

        Returns:
            List of node IDs that are alive
        """
        if not node_ids:
            return []

        alive = []

        # Get snapshot of peers under lock
        peers_lock = getattr(self, "peers_lock", None)
        peers = getattr(self, "peers", {})
        node_id = getattr(self, "node_id", None)

        if peers_lock:
            with peers_lock:
                peers = dict(peers)
        else:
            peers = dict(peers)

        for nid in node_ids:
            # Self is alive
            if nid == node_id:
                alive.append(nid)
                continue

            # Check if peer is alive
            peer = peers.get(nid)
            if peer and hasattr(peer, "is_alive") and peer.is_alive():
                alive.append(nid)

        return alive

    # =========================================================================
    # Event Emission Helpers
    # =========================================================================

    def _safe_emit_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Safely emit an event if handler exists.

        Wraps event emission in try-catch to prevent event failures
        from crashing the caller.

        Args:
            event_type: Event type string to emit
            payload: Optional event payload dict

        Returns:
            True if event was emitted successfully, False otherwise

        Example:
            self._safe_emit_event(
                "HOST_OFFLINE",
                {"node_id": peer_id, "reason": "timeout"},
            )
        """
        try:
            emit_fn = getattr(self, "_emit_event", None)
            if callable(emit_fn):
                emit_fn(event_type, payload or {})
                return True

            # Try alternate emit method names
            for method_name in ("emit_event", "publish_event", "_publish"):
                alt_fn = getattr(self, method_name, None)
                if callable(alt_fn):
                    alt_fn(event_type, payload or {})
                    return True

            return False

        except Exception as e:
            verbose = getattr(self, "verbose", False)
            if verbose:
                logger.debug(f"[{self.MIXIN_TYPE}] Event emission error: {e}")
            return False

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    @staticmethod
    def _load_config_constant(
        constant_name: str,
        default_value: Any,
        module_path: str = "scripts.p2p.constants",
    ) -> Any:
        """Safely load a configuration constant with fallback.

        Eliminates the try-except import pattern found at the top of every mixin.

        Args:
            constant_name: Name of the constant to import
            default_value: Value to return if import fails
            module_path: Module path to import from

        Returns:
            The constant value, or default_value if import fails

        Example:
            PEER_CACHE_TTL = P2PMixinBase._load_config_constant(
                "PEER_CACHE_TTL_SECONDS",
                604800,  # 7 days default
            )
        """
        try:
            module = importlib.import_module(module_path)
            return getattr(module, constant_name, default_value)
        except ImportError:
            return default_value
        except AttributeError:
            # Module exists but constant not found - use default
            return default_value

    @classmethod
    def _load_config_constants(
        cls,
        constants: dict[str, Any],
        module_path: str = "scripts.p2p.constants",
    ) -> dict[str, Any]:
        """Load multiple configuration constants with fallbacks.

        Args:
            constants: Dict of {constant_name: default_value}
            module_path: Module path to import from

        Returns:
            Dict of {constant_name: loaded_value}

        Example:
            config = P2PMixinBase._load_config_constants({
                "PEER_CACHE_TTL_SECONDS": 604800,
                "PEER_REPUTATION_ALPHA": 0.2,
                "MAX_PEERS": 100,
            })
        """
        result = {}
        for name, default in constants.items():
            result[name] = cls._load_config_constant(name, default, module_path)
        return result

    # =========================================================================
    # Logging Helpers
    # =========================================================================

    def _log_debug(self, message: str) -> None:
        """Log debug message with mixin type prefix."""
        logger.debug(f"[{self.MIXIN_TYPE}] {message}")

    def _log_info(self, message: str) -> None:
        """Log info message with mixin type prefix."""
        logger.info(f"[{self.MIXIN_TYPE}] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message with mixin type prefix."""
        logger.warning(f"[{self.MIXIN_TYPE}] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message with mixin type prefix."""
        logger.error(f"[{self.MIXIN_TYPE}] {message}")

    # =========================================================================
    # Timing Helpers
    # =========================================================================

    def _get_timestamp(self) -> float:
        """Get current timestamp (for testing/mocking)."""
        return time.time()

    def _is_expired(
        self,
        timestamp: float,
        ttl_seconds: float,
    ) -> bool:
        """Check if a timestamp has expired given a TTL.

        Args:
            timestamp: The timestamp to check
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if timestamp is older than TTL
        """
        return (self._get_timestamp() - timestamp) > ttl_seconds


class EventSubscriptionMixin:
    """Mixin providing standardized event subscription for P2P managers.

    This mixin consolidates the duplicated event subscription pattern found
    across 6 P2P managers (~100 LOC duplicated). It provides:
    - Thread-safe double-checked locking for subscription
    - Safe event router import with graceful fallback
    - Declarative event subscription via _get_event_subscriptions()
    - Health check integration for subscription status

    Usage:
        class MyManager(EventSubscriptionMixin):
            def _get_event_subscriptions(self) -> dict:
                '''Return mapping of event types to handlers.'''
                return {
                    "HOST_OFFLINE": self._on_host_offline,
                    "NODE_RECOVERED": self._on_node_recovered,
                }

            async def _on_host_offline(self, event) -> None:
                '''Handle HOST_OFFLINE event.'''
                pass

        # Then call during initialization:
        manager.subscribe_to_events()

    Subclasses MUST implement:
        _get_event_subscriptions() -> dict[str, Callable]

    Created: December 27, 2025
    Consolidates patterns from: job_manager.py, training_coordinator.py,
                               selfplay_scheduler.py, node_selector.py,
                               state_manager.py, sync_planner.py
    """

    # Type hints for IDE support - actual values set by __init__ or subscribe_to_events
    _subscribed: bool
    _subscription_lock: "threading.RLock"

    # Override in subclass for logging prefix
    _subscription_log_prefix: str = "EventSubscriptionMixin"

    def _init_subscription_state(self) -> None:
        """Initialize subscription state. Call from __init__.

        Sets up the _subscribed flag and _subscription_lock if not already present.
        This method is idempotent - safe to call multiple times.
        """
        import threading

        if not hasattr(self, "_subscribed"):
            self._subscribed = False
        if not hasattr(self, "_subscription_lock"):
            self._subscription_lock = threading.RLock()

    def _get_event_subscriptions(self) -> dict:
        """Return mapping of event type names to handler methods.

        Subclasses MUST override this method to declare their subscriptions.

        Returns:
            Dict mapping event type names (strings matching DataEventType attrs)
            to async handler callables.

        Example:
            def _get_event_subscriptions(self) -> dict:
                return {
                    "HOST_OFFLINE": self._on_host_offline,
                    "NODE_RECOVERED": self._on_node_recovered,
                    "TRAINING_COMPLETED": self._on_training_completed,
                }
        """
        return {}

    def subscribe_to_events(self) -> None:
        """Subscribe to events declared in _get_event_subscriptions().

        Thread-safe with double-checked locking. Gracefully handles:
        - Event router not available (ImportError)
        - Missing event types (AttributeError)
        - Runtime errors during subscription

        On failure, _subscribed remains False and health_check will report degraded.
        """
        # Ensure state is initialized
        self._init_subscription_state()

        # Fast path without lock
        if self._subscribed:
            return

        # Slow path with lock to prevent race condition
        with self._subscription_lock:
            if self._subscribed:
                return

            prefix = getattr(self, "_subscription_log_prefix", "EventSubscriptionMixin")
            subscriptions = self._get_event_subscriptions()

            if not subscriptions:
                # No subscriptions declared, mark as subscribed (vacuously true)
                self._subscribed = True
                return

            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus is None:
                    logger.debug(f"[{prefix}] Event bus not available")
                    return

                subscribed_count = 0
                for event_name, handler in subscriptions.items():
                    if hasattr(DataEventType, event_name):
                        event_type = getattr(DataEventType, event_name)
                        bus.subscribe(event_type, handler)
                        subscribed_count += 1
                        logger.info(f"[{prefix}] Subscribed to {event_name}")
                    else:
                        logger.debug(f"[{prefix}] Event type {event_name} not found")

                self._subscribed = True
                if subscribed_count > 0:
                    logger.info(f"[{prefix}] Event subscriptions complete ({subscribed_count} events)")

            except ImportError:
                logger.debug(f"[{prefix}] Event router not available")
                self._subscribed = False
            except (RuntimeError, AttributeError) as e:
                logger.warning(f"[{prefix}] Failed to subscribe: {e}")
                self._subscribed = False

    def is_subscribed(self) -> bool:
        """Check if event subscriptions are active.

        Returns:
            True if subscribe_to_events() completed successfully.
        """
        return getattr(self, "_subscribed", False)

    def get_subscription_status(self) -> dict:
        """Get subscription status for health check inclusion.

        Returns:
            Dict with subscription state suitable for health check details.

        Example usage in health_check():
            def health_check(self) -> dict:
                status = "healthy"
                sub_status = self.get_subscription_status()
                if not sub_status["subscribed"]:
                    status = "degraded"
                return {"status": status, **sub_status}
        """
        self._init_subscription_state()
        return {
            "subscribed": self._subscribed,
            "subscription_count": len(self._get_event_subscriptions()),
        }

    # =========================================================================
    # Event Handler Helpers (December 28, 2025)
    # =========================================================================

    def _extract_event_payload(self, event: Any) -> dict[str, Any]:
        """Safely extract payload from an event object.

        Handles both DataEvent objects with `.payload` attribute and raw dicts.
        Centralizes the duplicate pattern found across all event handlers.

        Args:
            event: Event object or dict

        Returns:
            Event payload as a dict (empty dict if extraction fails)

        Example:
            async def _on_host_offline(self, event) -> None:
                payload = self._extract_event_payload(event)
                node_id = payload.get("node_id", "")
        """
        try:
            if hasattr(event, "payload"):
                return event.payload if isinstance(event.payload, dict) else {}
            if isinstance(event, dict):
                return event
            return {}
        except (AttributeError, TypeError):
            return {}

    def _log_info(self, message: str) -> None:
        """Log info message with mixin type prefix.

        Uses MIXIN_TYPE if available (from P2PMixinBase inheritance),
        otherwise uses _subscription_log_prefix.
        """
        prefix = getattr(self, "MIXIN_TYPE", None) or getattr(
            self, "_subscription_log_prefix", "EventSubscriptionMixin"
        )
        logger.info(f"[{prefix}] {message}")

    def _log_debug(self, message: str) -> None:
        """Log debug message with mixin type prefix."""
        prefix = getattr(self, "MIXIN_TYPE", None) or getattr(
            self, "_subscription_log_prefix", "EventSubscriptionMixin"
        )
        logger.debug(f"[{prefix}] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message with mixin type prefix."""
        prefix = getattr(self, "MIXIN_TYPE", None) or getattr(
            self, "_subscription_log_prefix", "EventSubscriptionMixin"
        )
        logger.warning(f"[{prefix}] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message with mixin type prefix."""
        prefix = getattr(self, "MIXIN_TYPE", None) or getattr(
            self, "_subscription_log_prefix", "EventSubscriptionMixin"
        )
        logger.error(f"[{prefix}] {message}")

    def _safe_emit_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Safely emit an event using the event router.

        Wraps event emission in try-catch to prevent event failures
        from crashing the caller. Tries multiple emission methods.

        Args:
            event_type: Event type string to emit
            payload: Optional event payload dict

        Returns:
            True if event was emitted successfully, False otherwise

        Example:
            self._safe_emit_event(
                "TASK_ABANDONED",
                {"job_id": job_id, "reason": "host_offline"},
            )
        """
        try:
            # Try using event router directly
            try:
                from app.coordination.event_router import emit_sync
                emit_sync(event_type, payload or {})
                return True
            except ImportError:
                pass

            # Fallback to instance emit methods
            emit_fn = getattr(self, "_emit_event", None)
            if callable(emit_fn):
                emit_fn(event_type, payload or {})
                return True

            for method_name in ("emit_event", "publish_event", "_publish"):
                alt_fn = getattr(self, method_name, None)
                if callable(alt_fn):
                    alt_fn(event_type, payload or {})
                    return True

            return False

        except (OSError, ConnectionError, RuntimeError, TypeError) as e:
            # Dec 28, 2025: Specific exception types for event emission errors
            self._log_debug(f"Event emission error: {e}")
            return False


class P2PManagerBase(P2PMixinBase, EventSubscriptionMixin):
    """Combined base class for P2P managers with event subscription support.

    This class combines:
    - P2PMixinBase: Database helpers, state initialization, peer management, logging
    - EventSubscriptionMixin: Thread-safe event subscription with health integration

    Usage:
        class MyManager(P2PManagerBase):
            MIXIN_TYPE = "my_manager"
            _subscription_log_prefix = "MyManager"

            def __init__(self, ...):
                # Manager-specific initialization
                self._init_subscription_state()

            def _get_event_subscriptions(self) -> dict:
                return {
                    "HOST_OFFLINE": self._on_host_offline,
                    "TRAINING_COMPLETED": self._on_training_completed,
                }

            async def _on_host_offline(self, event) -> None:
                self._log_info(f"Host went offline: {event}")

            def health_check(self) -> dict:
                status = "healthy"
                sub_status = self.get_subscription_status()
                if not sub_status["subscribed"]:
                    status = "degraded"
                return {
                    "status": status,
                    "manager": self.MIXIN_TYPE,
                    **sub_status,
                }

    Created: December 27, 2025
    """

    pass  # All functionality inherited from parent classes
