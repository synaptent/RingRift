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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from threading import RLock

    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Event Emission Circuit Breaker (Phase 3.3 - Dec 29, 2025)
# =============================================================================
#
# Circuit breaker for event emission to prevent cascading failures when the
# event router is temporarily unavailable. Opens after consecutive failures
# and resets after a timeout period.

_event_circuit_lock = threading.Lock()
_event_emission_failures = 0
_event_circuit_open = False
_event_circuit_opened_at: float = 0.0

# Circuit breaker thresholds - use centralized defaults when available
# Sprint 4 (Jan 2, 2026): Integrated with CircuitBreakerDefaults for consistency
try:
    from app.config.coordination_defaults import CircuitBreakerDefaults

    EVENT_CIRCUIT_FAILURE_THRESHOLD = CircuitBreakerDefaults.P2P_FAILURE_THRESHOLD
    EVENT_CIRCUIT_RESET_TIMEOUT = CircuitBreakerDefaults.P2P_RECOVERY_TIMEOUT
except ImportError:
    # Fallback for standalone usage
    EVENT_CIRCUIT_FAILURE_THRESHOLD = 3  # Reduced from 10 to match P2P defaults
    EVENT_CIRCUIT_RESET_TIMEOUT = 45.0  # Match P2P recovery timeout

# =============================================================================
# IP Mapping Cache (Sprint 3.5 - Jan 2, 2026)
# =============================================================================
#
# Cache for distributed_hosts.yaml IP mappings to avoid reading the file
# on every _count_alive_peers() call (which happens 100+ times per minute).
# TTL of 60 seconds ensures config changes are picked up within a minute.

_ip_mapping_cache_lock = threading.Lock()
_ip_mapping_cache: dict[str, set[str]] | None = None
_ip_mapping_cache_timestamp: float = 0.0
_ip_mapping_cache_config_path: str = ""
IP_MAPPING_CACHE_TTL = 60.0  # Refresh cache every 60 seconds


def _get_cached_ip_mapping(config_path: Path) -> dict[str, set[str]] | None:
    """Get cached IP mapping if still valid.

    Args:
        config_path: Path to distributed_hosts.yaml

    Returns:
        Cached mapping if valid, None if cache miss or stale
    """
    global _ip_mapping_cache, _ip_mapping_cache_timestamp, _ip_mapping_cache_config_path

    with _ip_mapping_cache_lock:
        config_path_str = str(config_path)
        now = time.time()

        # Check if cache is valid
        if (
            _ip_mapping_cache is not None
            and _ip_mapping_cache_config_path == config_path_str
            and (now - _ip_mapping_cache_timestamp) < IP_MAPPING_CACHE_TTL
        ):
            return _ip_mapping_cache

        return None


def _set_cached_ip_mapping(config_path: Path, mapping: dict[str, set[str]]) -> None:
    """Update the IP mapping cache.

    Args:
        config_path: Path to distributed_hosts.yaml
        mapping: The IP mapping to cache
    """
    global _ip_mapping_cache, _ip_mapping_cache_timestamp, _ip_mapping_cache_config_path

    with _ip_mapping_cache_lock:
        _ip_mapping_cache = mapping
        _ip_mapping_cache_timestamp = time.time()
        _ip_mapping_cache_config_path = str(config_path)


def _get_event_circuit_state() -> tuple[bool, int]:
    """Get current circuit breaker state.

    Returns:
        Tuple of (is_open, failure_count)
    """
    global _event_circuit_open, _event_circuit_opened_at
    with _event_circuit_lock:
        # Check if circuit should auto-reset (half-open state)
        if _event_circuit_open:
            if time.time() - _event_circuit_opened_at >= EVENT_CIRCUIT_RESET_TIMEOUT:
                # Move to half-open state - allow a test request
                _event_circuit_open = False
                logger.info("[EventCircuitBreaker] Circuit moving to half-open, allowing test request")
        return _event_circuit_open, _event_emission_failures


def _record_event_emission_success() -> None:
    """Record successful event emission, reset failure count."""
    global _event_emission_failures, _event_circuit_open
    with _event_circuit_lock:
        if _event_emission_failures > 0 or _event_circuit_open:
            logger.debug("[EventCircuitBreaker] Event emission succeeded, resetting circuit")
        _event_emission_failures = 0
        _event_circuit_open = False


def _record_event_emission_failure() -> None:
    """Record failed event emission, potentially open circuit."""
    global _event_emission_failures, _event_circuit_open, _event_circuit_opened_at
    with _event_circuit_lock:
        _event_emission_failures += 1
        if _event_emission_failures >= EVENT_CIRCUIT_FAILURE_THRESHOLD and not _event_circuit_open:
            _event_circuit_open = True
            _event_circuit_opened_at = time.time()
            logger.warning(
                f"[EventCircuitBreaker] Circuit OPENED after {_event_emission_failures} failures, "
                f"will retry in {EVENT_CIRCUIT_RESET_TIMEOUT}s"
            )


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

    def _ensure_table(
        self,
        table_name: str,
        schema_sql: str,
        index_sql: str | None = None,
        timeout: float = 5.0,
    ) -> bool:
        """Ensure a table exists, creating it if necessary.

        Consolidates the duplicate CREATE TABLE IF NOT EXISTS pattern found
        across metrics_manager.py, state_manager.py, and other modules.

        Args:
            table_name: Name of the table (for logging)
            schema_sql: Full CREATE TABLE IF NOT EXISTS statement
            index_sql: Optional CREATE INDEX IF NOT EXISTS statement
            timeout: SQLite connection timeout in seconds

        Returns:
            True if table was created or already exists, False on error

        Example:
            created = self._ensure_table(
                "metrics_history",
                '''
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL
                )
                ''',
                index_sql="CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics_history(timestamp DESC)",
            )
        """
        with self._db_connection(timeout=timeout) as conn:
            if conn is None:
                return False
            try:
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                if index_sql:
                    cursor.execute(index_sql)
                conn.commit()
                return True
            except sqlite3.OperationalError as e:
                # Table already exists or schema issue
                verbose = getattr(self, "verbose", False)
                if verbose:
                    logger.debug(f"[{self.MIXIN_TYPE}] Table {table_name} setup: {e}")
                return "already exists" in str(e).lower() or True
            except sqlite3.Error as e:
                logger.warning(f"[{self.MIXIN_TYPE}] Failed to create table {table_name}: {e}")
                return False

    def _table_exists(self, table_name: str, timeout: float = 5.0) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check
            timeout: SQLite connection timeout

        Returns:
            True if table exists, False otherwise
        """
        result = self._execute_db_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
            fetch=True,
            commit=False,
            timeout=timeout,
        )
        return bool(result)

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

        Uses SWIM-based failure detection when available (MembershipMixin),
        falling back to HTTP heartbeat-based checks otherwise.

        Jan 2, 2026: Added IP:port matching because SWIM discovers peers
        as IP:port format (e.g., "135.181.39.239:7947") but node_ids are
        proper names (e.g., "hetzner-cpu1"). This method now checks both.

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

        # Jan 2, 2026: Build IP mapping for node_ids from distributed_hosts.yaml
        # This allows matching peers discovered as IP:port against node_ids
        node_ip_map = self._build_node_ip_mapping(node_ids)

        # Check if SWIM-based hybrid check is available (from MembershipMixin)
        # SWIM provides 5s failure detection vs 60-90s for HTTP heartbeats
        hybrid_check = getattr(self, "is_peer_alive_hybrid", None)
        use_hybrid = callable(hybrid_check)

        counted_nodes: set[str] = set()

        for nid in node_ids:
            if nid in counted_nodes:
                continue

            # Count self as alive
            if nid == node_id:
                alive += 1
                counted_nodes.add(nid)
                continue

            # Use SWIM-based check if available
            # Jan 5, 2026: Fixed to fall through to HTTP check when SWIM returns False
            # Previously, SWIM returning False would `continue` without HTTP fallback,
            # causing voters to show as unreachable when port 7947 is blocked by firewall.
            if use_hybrid:
                try:
                    if hybrid_check(nid):
                        alive += 1
                        counted_nodes.add(nid)
                        continue  # Only continue if SWIM says peer is alive
                    # SWIM says peer is NOT alive - fall through to HTTP check
                    self._log_debug(f"SWIM reports {nid} not alive, checking via HTTP heartbeat")
                except Exception as e:
                    # Dec 30, 2025: Log SWIM check failures for observability
                    self._log_debug(f"SWIM check failed for {nid}, falling back to HTTP: {type(e).__name__}")

            # Check 1: Direct node_id match in peers
            peer = peers.get(nid)
            if peer and hasattr(peer, "is_alive") and peer.is_alive():
                alive += 1
                counted_nodes.add(nid)
                continue

            # Jan 2, 2026: Check 2: IP:port match - look for any peer whose IP matches
            node_ips = node_ip_map.get(nid, set())
            if node_ips:
                for peer_id, peer in peers.items():
                    if nid in counted_nodes:
                        break
                    # Extract IP from peer_id (format: "IP:port")
                    if ":" in peer_id:
                        peer_ip = peer_id.split(":")[0]
                        if peer_ip in node_ips and hasattr(peer, "is_alive") and peer.is_alive():
                            alive += 1
                            counted_nodes.add(nid)
                            break

        return alive

    def _build_node_ip_mapping(self, node_ids: list[str]) -> dict[str, set[str]]:
        """Build a mapping from node_ids to their known IPs.

        Jan 2, 2026: Added to support peer matching when peers are discovered
        via SWIM as IP:port format instead of proper node_ids.

        Sprint 3.5 (Jan 2, 2026): Now uses caching to avoid repeated YAML loads.
        The distributed_hosts.yaml was being loaded on every _count_alive_peers()
        call (100+ times per minute). Cache has 60s TTL for config freshness.

        Args:
            node_ids: List of node IDs to map

        Returns:
            Dict mapping node_id -> set of known IPs (tailscale_ip, ssh_host)
        """
        if not node_ids:
            return {}

        # Determine config path
        ringrift_path = getattr(self, "ringrift_path", None)
        if not ringrift_path:
            return {}

        rp = Path(ringrift_path)
        # Jan 2, 2026: Handle ringrift_path that already includes ai-service suffix
        # Config files are stored at .../ai-service/config/, so don't add ai-service again
        if rp.name == "ai-service":
            cfg_path = rp / "config" / "distributed_hosts.yaml"
        else:
            cfg_path = rp / "ai-service" / "config" / "distributed_hosts.yaml"
        if not cfg_path.exists():
            return {}

        # Sprint 3.5: Check cache first
        full_mapping = _get_cached_ip_mapping(cfg_path)

        if full_mapping is None:
            # Cache miss - load from YAML
            try:
                import yaml
                data = yaml.safe_load(cfg_path.read_text()) or {}
                hosts = data.get("hosts", {}) or {}
            except Exception:
                return {}

            # Build full mapping for all hosts in config
            full_mapping = {}
            for nid, host_cfg in hosts.items():
                ips: set[str] = set()

                # Collect all known IPs for this node
                if host_cfg.get("tailscale_ip"):
                    ips.add(host_cfg["tailscale_ip"])
                if host_cfg.get("ssh_host"):
                    ssh_host = host_cfg["ssh_host"]
                    # Only add if it's an IP (not a hostname like "ssh5.vast.ai")
                    if ssh_host and not any(c.isalpha() for c in ssh_host.replace(".", "")):
                        ips.add(ssh_host)

                if ips:
                    full_mapping[nid] = ips

            # Update cache
            _set_cached_ip_mapping(cfg_path, full_mapping)

        # Filter to requested node_ids
        return {nid: full_mapping[nid] for nid in node_ids if nid in full_mapping}

    def _get_alive_peer_list(self, node_ids: list[str]) -> list[str]:
        """Get list of node IDs that are currently alive.

        Thread-safe version that returns the actual IDs, not just count.

        Uses SWIM-based failure detection when available (MembershipMixin),
        falling back to HTTP heartbeat-based checks otherwise.

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

        # Check if SWIM-based hybrid check is available (from MembershipMixin)
        # SWIM provides 5s failure detection vs 60-90s for HTTP heartbeats
        hybrid_check = getattr(self, "is_peer_alive_hybrid", None)
        use_hybrid = callable(hybrid_check)

        for nid in node_ids:
            # Self is alive
            if nid == node_id:
                alive.append(nid)
                continue

            # Use SWIM-based check if available
            if use_hybrid:
                try:
                    if hybrid_check(nid):
                        alive.append(nid)
                    continue
                except Exception as e:
                    # Dec 30, 2025: Log SWIM check failures for observability
                    self._log_debug(f"SWIM check failed for {nid}, falling back to HTTP: {type(e).__name__}")
                    pass

            # HTTP heartbeat fallback
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

        except (AttributeError, RuntimeError, ImportError, TypeError) as e:
            # AttributeError - method doesn't exist, RuntimeError - no event loop
            # ImportError - event router unavailable, TypeError - wrong call signature
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

    # =========================================================================
    # Health Check Helpers (December 28, 2025)
    # =========================================================================

    def _build_health_response(
        self,
        is_healthy: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standardized health check response.

        Consolidates the duplicate health_check() pattern found across all mixins.
        Each mixin implements a specific check (e.g., election_health_check()),
        then wraps it with the standard format using this helper.

        Args:
            is_healthy: Whether the mixin/component is healthy
            message: Human-readable health message
            details: Additional details dict (e.g., from specific_health_check())

        Returns:
            Standardized health response dict with:
            - healthy: bool
            - message: str
            - details: dict (merged with mixin_type)

        Example:
            def health_check(self) -> dict[str, Any]:
                status = self.election_health_check()
                is_healthy = status.get("is_healthy", False)
                role = status.get("role", "unknown")
                return self._build_health_response(
                    is_healthy=is_healthy,
                    message=f"Election (role={role})" if is_healthy else "No quorum",
                    details=status,
                )
        """
        response = {
            "healthy": is_healthy,
            "message": message,
            "details": {
                "mixin_type": self.MIXIN_TYPE,
                **(details or {}),
            },
        }
        return response

    def _health_check_from_status(
        self,
        status: dict[str, Any],
        healthy_key: str = "is_healthy",
        message_template_healthy: str = "{mixin_type} healthy",
        message_template_unhealthy: str = "{mixin_type} unhealthy",
    ) -> dict[str, Any]:
        """Build health response from a status dict.

        Convenience wrapper for _build_health_response() when the specific
        health check returns a dict with an is_healthy key.

        Args:
            status: Status dict from specific health check (e.g., election_health_check())
            healthy_key: Key in status dict indicating health (default: "is_healthy")
            message_template_healthy: Format string for healthy message
            message_template_unhealthy: Format string for unhealthy message

        Returns:
            Standardized health response dict

        Example:
            def health_check(self) -> dict[str, Any]:
                status = self.peer_health_check()
                return self._health_check_from_status(
                    status,
                    message_template_healthy="Peer manager ({active_peers} active)",
                    message_template_unhealthy="No peers available",
                )
        """
        is_healthy = status.get(healthy_key, False)

        # Build message from template
        template = message_template_healthy if is_healthy else message_template_unhealthy
        try:
            # Try to format with status keys and mixin_type
            message = template.format(mixin_type=self.MIXIN_TYPE, **status)
        except (KeyError, ValueError):
            # Fallback to simple message
            message = template.format(mixin_type=self.MIXIN_TYPE) if "{mixin_type}" in template else template

        return self._build_health_response(
            is_healthy=is_healthy,
            message=message,
            details=status,
        )


# =========================================================================
# Event Subscription Retry Configuration (December 28, 2025)
# =========================================================================


@dataclass
class SubscriptionRetryConfig:
    """Configuration for event subscription retry behavior.

    Used by EventSubscriptionMixin.subscribe_to_events_with_retry() to control
    retry attempts when event subscriptions fail at startup.

    Attributes:
        max_attempts: Maximum number of subscription attempts (default: 3)
        initial_delay_seconds: Initial delay before first retry (default: 1.0)
        max_delay_seconds: Maximum delay cap (default: 8.0)
        backoff_multiplier: Delay multiplier per attempt (default: 2.0)

    Example delays with defaults:
        - Attempt 1: immediate
        - Attempt 2: 1.0s delay
        - Attempt 3: 2.0s delay
        Total worst-case startup delay: 3.0s

    Usage:
        config = SubscriptionRetryConfig(max_attempts=5, initial_delay_seconds=0.5)
        manager.subscribe_to_events_with_retry(config)
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    backoff_multiplier: float = 2.0


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

    def subscribe_to_events_with_retry(
        self,
        config: SubscriptionRetryConfig | None = None,
    ) -> bool:
        """Subscribe to events with exponential backoff retry.

        Wraps subscribe_to_events() with retry logic to handle transient failures
        during startup (e.g., event router not yet initialized, import race conditions).

        Args:
            config: Retry configuration. Uses SubscriptionRetryConfig defaults if None.

        Returns:
            True if subscriptions succeeded, False if all retries exhausted.

        Example:
            # Use default retry config (3 attempts, 1s/2s/4s delays)
            if not manager.subscribe_to_events_with_retry():
                logger.error("Failed to subscribe to events")

            # Custom config for critical managers
            config = SubscriptionRetryConfig(max_attempts=5, initial_delay_seconds=0.5)
            manager.subscribe_to_events_with_retry(config)

        Note:
            This method is synchronous and uses time.sleep() for delays.
            It's designed to be called during __init__ which is sync context.
        """
        config = config or SubscriptionRetryConfig()
        prefix = getattr(self, "_subscription_log_prefix", "EventSubscriptionMixin")

        for attempt in range(config.max_attempts):
            # Reset subscription state for retry
            self._init_subscription_state()
            self._subscribed = False

            # Attempt subscription
            self.subscribe_to_events()

            if self._subscribed:
                if attempt > 0:
                    logger.info(
                        f"[{prefix}] Event subscription succeeded on attempt {attempt + 1}"
                    )
                return True

            # Don't sleep after final failed attempt
            if attempt < config.max_attempts - 1:
                # Calculate delay with exponential backoff
                delay = min(
                    config.initial_delay_seconds * (config.backoff_multiplier ** attempt),
                    config.max_delay_seconds,
                )
                logger.warning(
                    f"[{prefix}] Event subscription failed, "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{config.max_attempts})"
                )
                time.sleep(delay)

        logger.error(
            f"[{prefix}] CRITICAL: Event subscription failed "
            f"after {config.max_attempts} attempts"
        )
        return False

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
        max_retries: int = 3,
    ) -> bool:
        """Safely emit an event using the event router with circuit breaker and retry.

        Phase 3.3 Dec 29, 2025: Added circuit breaker and retry with exponential backoff.
        Wraps event emission in try-catch to prevent event failures from crashing
        the caller. Tries multiple emission methods.

        Args:
            event_type: Event type string to emit
            payload: Optional event payload dict
            max_retries: Maximum retry attempts (default 3)

        Returns:
            True if event was emitted successfully, False otherwise

        Example:
            self._safe_emit_event(
                "TASK_ABANDONED",
                {"job_id": job_id, "reason": "host_offline"},
            )
        """
        # Phase 3.3 Dec 29, 2025: Check circuit breaker state
        is_open, failure_count = _get_event_circuit_state()
        if is_open:
            self._log_debug(f"Event emission circuit open, skipping {event_type}")
            return False

        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Try using event router directly
                try:
                    from app.coordination.event_router import emit_sync
                    emit_sync(event_type, payload or {})
                    _record_event_emission_success()
                    return True
                except ImportError:
                    pass

                # Fallback to instance emit methods
                emit_fn = getattr(self, "_emit_event", None)
                if callable(emit_fn):
                    emit_fn(event_type, payload or {})
                    _record_event_emission_success()
                    return True

                for method_name in ("emit_event", "publish_event", "_publish"):
                    alt_fn = getattr(self, method_name, None)
                    if callable(alt_fn):
                        alt_fn(event_type, payload or {})
                        _record_event_emission_success()
                        return True

                # No emission method available - not a retry-able error
                return False

            except (OSError, ConnectionError, RuntimeError, TypeError) as e:
                last_error = e
                _record_event_emission_failure()

                # Retry with exponential backoff
                if attempt < max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                    time.sleep(backoff)
                    self._log_debug(
                        f"Event emission retry {attempt + 1}/{max_retries} for {event_type} "
                        f"after {backoff}s backoff"
                    )

        # All retries exhausted
        if last_error:
            self._log_debug(f"Event emission failed after {max_retries} attempts: {last_error}")
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
