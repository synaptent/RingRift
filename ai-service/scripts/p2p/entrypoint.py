"""P2P Orchestrator entrypoint and startup utilities.

Extracted from scripts/p2p_orchestrator.py and scripts/p2p/startup_infrastructure.py
as Target 5 of the P2P decomposition plan.

Contains standalone startup functions that don't reference the P2POrchestrator class:
  - main()                              CLI entrypoint
  - should_master_loop_manage_p2p()     Used by master_loop.py
  - Singleton lock management           (_acquire/_release_singleton_lock)
  - Supervisor coordination             (_claim/_release_supervisor_role)
  - Zombie/port detection               (_check_and_kill_zombie_p2p, _check_port_available_and_responsive)
  - Tailscale IP detection              (_wait_for_tailscale_ip)
  - Node ID auto-detection              (_auto_detect_node_id)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.lib.logging_config import setup_script_logging
from scripts.lib.process import (
    SingletonLock,
    find_processes_by_pattern,
    kill_process,
)

logger = setup_script_logging("p2p_orchestrator")

# Singleton lock for duplicate process prevention (December 2025)
_P2P_LOCK: SingletonLock | None = None

# =============================================================================
# SUPERVISOR COORDINATION (January 21, 2026 - Phase 2)
# =============================================================================
# This section prevents conflicts between manual P2P starts and master_loop
# automated recovery by creating a coordination file that tracks which
# management path is in control.

SUPERVISOR_FILE_PATH = Path(__file__).parent.parent.parent / "data" / "coordination" / "p2p_supervisor.json"


def _wait_for_tailscale_ip(timeout_seconds: int = 90, interval_seconds: float = 1.0) -> str:
    """Wait for Tailscale IP to become available at startup.

    Jan 12, 2026: Increased timeout from 30s to 90s after observing mac-studio
    consistently advertising local IP (10.0.0.62) instead of Tailscale IP.

    Root cause: When P2P starts before Tailscale CLI is ready, _get_tailscale_ip()
    returns empty and the code falls back to local IP (e.g., 10.0.0.62). This
    persists even after Tailscale becomes available later, causing P2P connectivity
    issues since other nodes can't reach the local IP. On mac-studio specifically,
    Tailscale can take 45-60s to initialize after boot.

    Fix: Retry Tailscale IP detection with exponential backoff for up to 90 seconds
    at startup with faster initial polling (1s intervals). This gives Tailscale
    enough time to initialize even on slow boot scenarios.

    Args:
        timeout_seconds: Maximum time to wait for Tailscale (default 90s)
        interval_seconds: Initial retry interval (doubles with each retry, max 5s)

    Returns:
        Tailscale IP if available within timeout, else empty string
    """
    from scripts.p2p.resource_detector import ResourceDetector

    detector = ResourceDetector()
    start_time = time.time()
    attempt = 0
    current_interval = interval_seconds

    while (time.time() - start_time) < timeout_seconds:
        attempt += 1
        ts_ip = detector.get_tailscale_ip()
        if ts_ip:
            if attempt > 1:
                logger.info(f"[TAILSCALE] IP acquired after {attempt} attempts: {ts_ip}")
            return ts_ip

        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if elapsed >= 5 and attempt <= 3:
            logger.warning(f"[TAILSCALE] Still waiting for IP (attempt {attempt}, {elapsed:.1f}s elapsed)")

        if remaining <= 0:
            break

        # Sleep with exponential backoff (max 5s between retries)
        sleep_time = min(current_interval, remaining, 5.0)
        time.sleep(sleep_time)
        current_interval = min(current_interval * 1.5, 5.0)

    logger.warning(f"[TAILSCALE] Timed out waiting for IP after {timeout_seconds}s ({attempt} attempts)")
    return ""


def _auto_detect_node_id() -> str | None:
    """Auto-detect node ID using unified identity resolution.

    Jan 2, 2026: Added to prevent startup failures when --node-id is forgotten.
    Jan 12, 2026: Added /etc/ringrift/node-id file support and IP normalization.
    Jan 13, 2026: Delegated to app.config.node_identity module (P2P Cluster Stability Plan).

    Detection order (from node_identity module):
    0. /etc/ringrift/node-id file (canonical source, written by deployment)
    1. RINGRIFT_NODE_ID environment variable
    2. /etc/default/ringrift-p2p file (legacy compatibility)
    3. Hostname match against distributed_hosts.yaml
    4. Tailscale IP match against distributed_hosts.yaml
    5. Fall back to get_node_id_safe() which uses hostname

    Returns:
        Detected node_id string, or None if detection failed
    """
    try:
        from app.config.node_identity import (
            get_node_identity,
            get_node_id_safe,
            NodeIdentityError,
        )

        # Try strict resolution first
        try:
            identity = get_node_identity()
            logger.info(
                f"[NODE-ID] Resolved node ID via {identity.resolution_method}: "
                f"{identity.canonical_id}"
            )
            return identity.canonical_id
        except NodeIdentityError as e:
            # Strict resolution failed, use safe fallback
            logger.warning(f"[NODE-ID] Strict resolution failed: {e}")
            node_id = get_node_id_safe()
            logger.warning(
                f"[NODE-ID] Using fallback node ID: {node_id} - "
                f"Run 'python scripts/provision_node_id.py --auto-detect' to fix"
            )
            return node_id

    except ImportError as e:
        # Module not available (running standalone or tests)
        logger.debug(f"[NODE-ID] node_identity module not available: {e}")

        # Minimal fallback: check canonical file and env var
        try:
            with open("/etc/ringrift/node-id") as f:
                node_id = f.read().strip()
                if node_id:
                    logger.info(f"[NODE-ID] Using node-id from /etc/ringrift/node-id: {node_id}")
                    return node_id
        except (FileNotFoundError, PermissionError):
            pass

        node_id = os.environ.get("RINGRIFT_NODE_ID")
        if node_id:
            return node_id

        # Fall back to hostname
        import socket
        hostname = socket.gethostname()
        if "." in hostname:
            hostname = hostname.split(".")[0]
        logger.warning(
            f"[NODE-ID] Falling back to hostname '{hostname}' - "
            f"Set RINGRIFT_NODE_ID or run provision_node_id.py"
        )
        return hostname


def _acquire_singleton_lock(
    kill_duplicates: bool = False,
    force_takeover: bool = False,
) -> bool:
    """Acquire singleton lock to prevent duplicate P2P orchestrator instances.

    Uses atomic file locking (fcntl) which is more reliable than PID file checks.
    Automatically handles stale locks from crashed processes.

    Args:
        kill_duplicates: If True, kill any duplicate P2P processes before acquiring
        force_takeover: If True, force-kill any lock holder (even if not P2P).
                        Use when lock is held by a recycled PID.

    Returns:
        True if lock acquired successfully
    """
    global _P2P_LOCK

    lock_dir = Path(__file__).parent.parent.parent / "data" / "coordination"
    lock_dir.mkdir(parents=True, exist_ok=True)

    if kill_duplicates:
        # Find and kill any existing p2p_orchestrator processes
        pattern = r"p2p_orchestrator\.py"
        existing = find_processes_by_pattern(pattern, exclude_self=True)
        if existing:
            logger.info(f"[P2P] Found {len(existing)} duplicate processes, killing...")
            for proc in existing:
                logger.info(f"[P2P] Killing duplicate: PID {proc.pid}")
                if kill_process(proc.pid, wait=True, timeout=5.0):
                    logger.info(f"[P2P] Killed PID {proc.pid}")
                else:
                    logger.warning(f"[P2P] Failed to kill PID {proc.pid}")
            # Wait a moment for locks to release
            time.sleep(0.5)

    # Create lock with auto-cleanup of stale locks (from dead processes)
    _P2P_LOCK = SingletonLock(
        "p2p_orchestrator",
        lock_dir=lock_dir,
        auto_cleanup_stale=True,  # Automatically handle dead process locks
    )

    if not _P2P_LOCK.acquire():
        # Lock acquisition failed - provide detailed diagnostics
        status = _P2P_LOCK.get_lock_status()
        holder_pid = status.get("holder_pid")
        holder_alive = status.get("holder_alive", False)
        holder_command = status.get("holder_command", "")
        is_stale = status.get("is_stale", False)

        if is_stale:
            # This shouldn't happen with auto_cleanup_stale=True, but handle it
            logger.warning(
                f"[P2P] Stale lock detected (dead PID {holder_pid}). "
                f"Attempting force cleanup..."
            )
            if _P2P_LOCK.force_release():
                # Retry acquisition after cleanup
                if _P2P_LOCK.acquire():
                    logger.info(f"[P2P] Acquired lock after stale cleanup (PID {os.getpid()})")
                    return True
            logger.error("[P2P] Failed to clean up stale lock")
            return False

        if holder_pid and holder_alive:
            # Another live process is holding the lock
            is_p2p = _P2P_LOCK.is_holder_expected_process("p2p_orchestrator")
            if is_p2p:
                logger.error(
                    f"[P2P] Another P2P orchestrator is already running (PID {holder_pid}). "
                    f"Use --kill-duplicates to automatically terminate it."
                )
            else:
                # PID reuse - different process now holds the lock file
                # This happens when the old P2P crashed and the PID was reused
                if force_takeover:
                    logger.warning(
                        f"[P2P] Lock held by unexpected process (PID {holder_pid}: {holder_command[:80] if holder_command else 'unknown'}). "
                        f"Force takeover requested - killing holder."
                    )
                    if _P2P_LOCK.force_release(kill_holder=True):
                        if _P2P_LOCK.acquire():
                            logger.info(f"[P2P] Acquired lock after force takeover (PID {os.getpid()})")
                            return True
                    logger.error("[P2P] Force takeover failed")
                else:
                    logger.warning(
                        f"[P2P] Lock held by unexpected process (PID {holder_pid}: {holder_command[:80] if holder_command else 'unknown'}). "
                        f"This may indicate PID reuse after a crash. "
                        f"Use --force-takeover to automatically recover."
                    )
        else:
            logger.error(
                "[P2P] Failed to acquire lock (unknown reason). "
                f"Lock status: {status}"
            )
        return False

    logger.info(f"[P2P] Acquired singleton lock (PID {os.getpid()})")
    return True


def _release_singleton_lock() -> None:
    """Release the singleton lock on shutdown."""
    global _P2P_LOCK
    if _P2P_LOCK:
        _P2P_LOCK.release()
        logger.debug("[P2P] Released singleton lock")
        _P2P_LOCK = None


# =============================================================================
# PORT-FIRST CHECK (January 21, 2026 - Phase 1)
# =============================================================================
# This provides fast-fail duplicate detection BEFORE zombie detection or lock
# acquisition. If a healthy P2P is already running, exit immediately.

def _check_port_available_and_responsive(port: int = 8770, timeout: float = 3.0) -> tuple[bool, str]:
    """Check if port is available or if existing P2P is healthy.

    January 21, 2026: Added as Phase 1 of duplicate process prevention.
    This is the FIRST check at startup, before zombie detection or lock acquisition.
    Provides fast-fail when a healthy P2P is already running.

    Args:
        port: The P2P HTTP port to check (default 8770)
        timeout: HTTP health check timeout in seconds

    Returns:
        (should_continue, reason) tuple:
        - (True, "port_free") - Port is free, proceed with startup
        - (True, "port_check_failed") - Couldn't determine, proceed cautiously
        - (False, "healthy_p2p_running") - Another healthy P2P is running, exit
    """
    import socket
    import urllib.request
    import urllib.error

    # Step 1: Try to bind to port (instant availability check)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.close()
        return (True, "port_free")
    except OSError:
        pass  # Port in use, check if healthy

    # Step 2: Check if existing process on port is responsive
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            headers={"User-Agent": "p2p-startup-check"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return (False, "healthy_p2p_running")
    except urllib.error.URLError as e:
        # Connection refused means port not actually listening
        if "Connection refused" in str(e):
            return (True, "port_check_failed")
        # Timeout or other error - proceed cautiously
        return (True, "port_check_failed")
    except Exception:
        # Unexpected error - proceed cautiously
        return (True, "port_check_failed")

    # Should not reach here, but proceed if we do
    return (True, "port_check_failed")


def _read_supervisor_file() -> dict | None:
    """Read the supervisor coordination file."""
    try:
        if SUPERVISOR_FILE_PATH.exists():
            content = SUPERVISOR_FILE_PATH.read_text()
            return json.loads(content)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"[P2P] Could not read supervisor file: {e}")
    return None


def _write_supervisor_file(managed_by: str, pid: int, force: bool = False) -> bool:
    """Write the supervisor coordination file."""
    from datetime import datetime

    if not force:
        existing = _read_supervisor_file()
        if existing and existing.get("managed_by") not in ("none", None):
            existing_pid = existing.get("pid")
            if existing_pid and _is_process_running_check(existing_pid):
                return False

    try:
        SUPERVISOR_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "managed_by": managed_by,
            "pid": pid,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "last_health_check": datetime.utcnow().isoformat() + "Z",
        }
        SUPERVISOR_FILE_PATH.write_text(json.dumps(state, indent=2))
        logger.info(f"[P2P] Claimed supervisor role: {managed_by} (PID {pid})")
        return True
    except OSError as e:
        logger.warning(f"[P2P] Failed to write supervisor file: {e}")
        return False


def _is_process_running_check(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _claim_supervisor_role(mode: str = "manual", force: bool = False) -> tuple[bool, str]:
    """Claim P2P management role."""
    from datetime import datetime, timedelta

    existing = _read_supervisor_file()

    if existing is None or existing.get("managed_by") in ("none", None):
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    existing_manager = existing.get("managed_by")
    existing_pid = existing.get("pid")

    # Check if PID is dead
    if existing_pid and not _is_process_running_check(existing_pid):
        logger.info(f"[P2P] Previous manager (PID {existing_pid}) is dead, taking over")
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    # Jan 23, 2026: Check for stale claims based on timestamp
    # If last_health_check is older than 10 minutes, consider it stale
    last_health = existing.get("last_health_check") or existing.get("started_at")
    if last_health:
        try:
            last_health_dt = datetime.fromisoformat(last_health.replace("Z", "+00:00"))
            now = datetime.now(last_health_dt.tzinfo) if last_health_dt.tzinfo else datetime.utcnow()
            stale_threshold = timedelta(minutes=10)
            if now - last_health_dt > stale_threshold:
                logger.info(f"[P2P] Previous manager (PID {existing_pid}) has stale health check ({last_health}), taking over")
                if _write_supervisor_file(mode, os.getpid(), force=True):
                    return (True, "claimed")
                return (False, "write_failed")
        except (ValueError, TypeError) as e:
            logger.debug(f"[P2P] Could not parse health check timestamp: {e}")

    if existing_pid == os.getpid():
        return (True, "already_manager")

    if existing_manager == "master_loop" and mode == "manual":
        if not force:
            return (False, "master_loop_managing")
        logger.warning("[P2P] Forcing takeover from master_loop")

    if force:
        if _write_supervisor_file(mode, os.getpid(), force=True):
            return (True, "claimed")
        return (False, "write_failed")

    return (False, "other_manager")


def _release_supervisor_role() -> None:
    """Release P2P management role on shutdown."""
    try:
        existing = _read_supervisor_file()
        if existing and existing.get("pid") == os.getpid():
            _write_supervisor_file("none", 0, force=True)
            logger.info("[P2P] Released supervisor role")
    except Exception as e:
        logger.debug(f"[P2P] Could not release supervisor role: {e}")


def should_master_loop_manage_p2p() -> tuple[bool, str]:
    """Check if master_loop should manage P2P or defer to manual management."""
    from datetime import datetime, timedelta

    existing = _read_supervisor_file()

    if existing is None:
        return (True, "no_manager")

    managed_by = existing.get("managed_by")
    if managed_by in ("none", None):
        return (True, "no_manager")

    existing_pid = existing.get("pid")

    if existing_pid and not _is_process_running_check(existing_pid):
        return (True, "manager_dead")

    if managed_by == "manual":
        started_at_str = existing.get("started_at", "")
        try:
            started_at = datetime.fromisoformat(started_at_str.rstrip("Z"))
            age = datetime.utcnow() - started_at
            if age < timedelta(hours=1):
                return (False, "manual_manager")
            return (True, "manual_expired")
        except (ValueError, TypeError):
            return (False, "manual_manager")

    if managed_by == "master_loop":
        return (True, "master_loop_manager")

    return (False, "manager_healthy")


def _check_and_kill_zombie_p2p(port: int = 8770, timeout: float = 5.0) -> bool:
    """Check for zombie P2P process and kill it if found.

    A zombie P2P process is one that is bound to the port but not responding
    to HTTP requests. This can happen when the process is stuck in a bad state.

    Args:
        port: The P2P HTTP port to check (default 8770)
        timeout: HTTP request timeout in seconds

    Returns:
        True if a zombie was found and killed, False otherwise
    """
    import urllib.request
    import urllib.error

    # Step 1: Check if anything is listening on the port
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # Nothing listening on the port
            return False
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        if not pids:
            return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        # lsof failed or timed out, assume no zombie
        return False

    # Step 2: Try to hit the /status endpoint
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/status",
            headers={"User-Agent": "zombie-detector"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                # Process is responding, not a zombie
                return False
    except urllib.error.URLError as e:
        # Connection refused means nothing is really listening (lsof race)
        if "Connection refused" in str(e):
            return False
        # Other errors (timeout, etc.) mean zombie detected
        logger.warning(f"[P2P] Port {port} occupied but unresponsive: {e}")
    except Exception as e:
        # Timeout or other error - this is a zombie
        logger.warning(f"[P2P] Port {port} occupied but unresponsive: {e}")

    # Step 3: Kill the zombie process(es)
    logger.warning(f"[P2P] Detected zombie P2P process on port {port}, killing PIDs: {pids}")
    killed = False
    for pid in pids:
        # Skip ourselves
        if pid == os.getpid():
            continue
        try:
            if kill_process(pid, wait=True, timeout=5.0):
                logger.info(f"[P2P] Killed zombie process PID {pid}")
                killed = True
            else:
                logger.warning(f"[P2P] Failed to kill zombie PID {pid}")
        except Exception as e:
            logger.error(f"[P2P] Error killing zombie PID {pid}: {e}")

    if killed:
        # Give the port time to be released
        time.sleep(0.5)

    return killed


def main():
    import argparse
    import asyncio
    import sqlite3

    # Import lazily inside main() to avoid circular imports.
    # P2POrchestrator is defined in p2p_orchestrator.py which imports from us
    # via startup_infrastructure.py.
    from scripts.p2p.startup_infrastructure import (
        _validate_preflight_dependencies,
        _check_event_emitters,
        set_selfplay_disabled_override,
    )
    from scripts.p2p.constants import (
        ADVERTISE_HOST_ENV,
        ADVERTISE_PORT_ENV,
        AUTH_TOKEN_ENV,
        DEFAULT_PORT,
    )

    # ==========================================================================
    # PRE-FLIGHT VALIDATION (January 2026)
    # ==========================================================================
    # Validate critical dependencies before any complex initialization.
    # This prevents cryptic runtime errors from missing packages.
    deps_ok, dep_errors = _validate_preflight_dependencies()
    if not deps_ok:
        print("[P2P] FATAL: Missing critical dependencies", file=sys.stderr)
        for err in dep_errors:
            print(f"  {err}", file=sys.stderr)
        print("\n[P2P] Fix: pip install aiohttp psutil pyyaml", file=sys.stderr)
        sys.exit(1)

    # Parse lock-related args early (before full argparse)
    kill_duplicates = "--kill-duplicates" in sys.argv
    force_takeover = "--force-takeover" in sys.argv
    skip_zombie_check = "--no-zombie-check" in sys.argv
    skip_port_check = "--skip-port-check" in sys.argv
    ignore_supervisor = "--ignore-supervisor" in sys.argv
    force_supervisor = "--force-supervisor" in sys.argv
    is_master_loop = "--managed-by-master-loop" in sys.argv

    # ==========================================================================
    # PORT-FIRST CHECK (January 21, 2026)
    # ==========================================================================
    # Check if port is available or if a healthy P2P is already running.
    # This provides fast-fail before zombie detection or lock acquisition.
    if not skip_port_check:
        can_start, reason = _check_port_available_and_responsive(DEFAULT_PORT)
        if not can_start:
            print(f"[P2P] Exiting: {reason} - another healthy P2P is already running on port {DEFAULT_PORT}")
            sys.exit(0)
        elif reason == "port_free":
            print("[P2P] Port is free, proceeding with startup")

    # ==========================================================================
    # SUPERVISOR COORDINATION (January 21, 2026)
    # ==========================================================================
    # Check if another manager (master_loop or manual) is controlling P2P.
    if not ignore_supervisor:
        management_mode = "master_loop" if is_master_loop else "manual"
        claimed, claim_reason = _claim_supervisor_role(mode=management_mode, force=force_supervisor)
        if not claimed:
            if claim_reason == "master_loop_managing":
                print("[P2P] Exiting: master_loop.py is managing P2P. Use --force-supervisor to override.")
            else:
                print(f"[P2P] Exiting: Another manager is active ({claim_reason}). Use --force-supervisor to override.")
            sys.exit(0)
        print(f"[P2P] Claimed supervisor role: {management_mode}")

    # ==========================================================================
    # ZOMBIE DETECTION (January 2026)
    # ==========================================================================
    # Check for zombie P2P processes that are bound to the port but unresponsive.
    # This happens when the P2P process gets stuck in a bad state (e.g., 100% CPU).
    if not skip_zombie_check:
        if _check_and_kill_zombie_p2p():
            print("[P2P] Killed zombie P2P process, proceeding with startup")

    # Acquire singleton lock (December 2025: improved atomic locking with stale cleanup)
    if not _acquire_singleton_lock(
        kill_duplicates=kill_duplicates,
        force_takeover=force_takeover,
    ):
        sys.exit(1)

    parser = argparse.ArgumentParser(description="P2P Orchestrator for RingRift cluster")
    parser.add_argument("--node-id", required=False, help="Unique identifier for this node (auto-detects if not provided)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument(
        "--advertise-host",
        default=None,
        help=f"Host to advertise to peers (or set {ADVERTISE_HOST_ENV})",
    )
    parser.add_argument(
        "--advertise-port",
        type=int,
        default=None,
        help=f"Port to advertise to peers (or set {ADVERTISE_PORT_ENV})",
    )
    parser.add_argument("--peers", help="Comma-separated list of known peers (host[:port] or http(s)://host[:port])")
    parser.add_argument("--relay-peers", help="Comma-separated list of peers to use relay heartbeats with (for NAT-blocked nodes)")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")
    parser.add_argument("--auth-token", help=f"Shared auth token (or set {AUTH_TOKEN_ENV})")
    parser.add_argument("--require-auth", action="store_true", help="Require auth token to be set")
    parser.add_argument("--storage-type", choices=["disk", "ramdrive", "auto"], default="auto",
                        help="Storage type: 'disk', 'ramdrive' (/dev/shm), or 'auto' (detect based on RAM/disk)")
    parser.add_argument("--sync-to-disk-interval", type=int, default=300,
                        help="When using ramdrive, sync to disk every N seconds (0 = no sync, default: 300)")
    parser.add_argument("--supervised", action="store_true",
                        help="Running under cluster_supervisor.py - disable self-restart logic")
    parser.add_argument("--kill-duplicates", action="store_true",
                        help="Kill any existing P2P orchestrator processes before starting")
    parser.add_argument("--force-takeover", action="store_true",
                        help="Force acquire lock even if held by another process (use when PID was recycled after crash)")
    parser.add_argument("--no-zombie-check", action="store_true",
                        help="Skip automatic zombie P2P detection (zombies are processes bound to port but not responding)")
    parser.add_argument("--skip-port-check", action="store_true",
                        help="Skip the port availability check at startup (Jan 21, 2026)")
    parser.add_argument("--ignore-supervisor", action="store_true",
                        help="Skip supervisor coordination file check (Jan 21, 2026)")
    parser.add_argument("--force-supervisor", action="store_true",
                        help="Force takeover of supervisor role even if another manager is active")
    parser.add_argument("--managed-by-master-loop", action="store_true",
                        help="Internal flag: indicates P2P was started by master_loop.py")
    parser.add_argument("--training-only", action="store_true",
                        help="Run as training-only node (no selfplay dispatch). Prevents OOM from training + selfplay conflicts.")

    args = parser.parse_args()

    # Jan 2026: Set training-only mode if flag is set
    if args.training_only:
        set_selfplay_disabled_override(disabled=True)
        logger.info("[P2P] Running in training-only mode - selfplay disabled")

    # Jan 2, 2026: Auto-detect node_id if not provided
    if not args.node_id:
        args.node_id = _auto_detect_node_id()
        if not args.node_id:
            logger.error("Could not auto-detect node-id. Please provide --node-id explicitly.")
            sys.exit(1)
        logger.info(f"Auto-detected node-id: {args.node_id}")

    known_peers = []
    if args.peers:
        known_peers = [p.strip() for p in args.peers.split(',')]

    relay_peers = []
    if args.relay_peers:
        relay_peers = [p.strip() for p in args.relay_peers.split(',')]

    # Import P2POrchestrator here (not at module level) to avoid circular imports.
    # p2p_orchestrator.py -> startup_infrastructure.py -> (us), so importing
    # P2POrchestrator at module level would create a cycle.
    from scripts.p2p_orchestrator import P2POrchestrator

    # Wrap orchestrator creation and run in try/except to ensure crashes are logged
    orchestrator = None
    try:
        logger.info(f"Initializing P2P orchestrator: node_id={args.node_id}")
        orchestrator = P2POrchestrator(
            node_id=args.node_id,
            host=args.host,
            port=args.port,
            known_peers=known_peers,
            relay_peers=relay_peers,
            ringrift_path=args.ringrift_path,
            advertise_host=args.advertise_host,
            advertise_port=args.advertise_port,
            auth_token=args.auth_token,
            require_auth=args.require_auth,
            storage_type=args.storage_type,
            sync_to_disk_interval=args.sync_to_disk_interval,
        )
        logger.info(f"P2P orchestrator initialized successfully: {args.node_id}")

        # December 28, 2025: Validate event emitters at startup
        # This provides early warning if event system is not properly configured
        if _check_event_emitters():
            logger.info("[P2P] Event emitters available - P2P events will be published")
        else:
            logger.warning(
                "[P2P] Event emitters NOT available - P2P events will be silent. "
                "Ensure app.coordination.event_emitters is importable for full integration."
            )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to initialize P2P orchestrator: {e}")
        # January 2026: Release lock on initialization failure to prevent
        # stale locks that block future startups
        _release_singleton_lock()
        sys.exit(1)

    # Handle shutdown gracefully - avoid race conditions with async tasks
    # December 2025: Fixed signal handler race condition that caused threading exceptions
    _shutdown_requested = False
    _start_time = time.time()

    def signal_handler(sig, frame):
        nonlocal _shutdown_requested
        import traceback

        uptime = time.time() - _start_time
        sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"

        if _shutdown_requested:
            # Force exit on second signal
            logger.warning(f"Forced shutdown (second {sig_name}) after {uptime:.1f}s uptime")
            os._exit(1)
        _shutdown_requested = True

        # Enhanced logging to identify what's sending signals
        logger.warning(f"=== SIGNAL RECEIVED: {sig_name} ===")
        logger.warning(f"PID: {os.getpid()}, Uptime: {uptime:.1f}s, Node: {args.node_id}")
        logger.warning(f"Stack trace at signal:\n{''.join(traceback.format_stack(frame))}")
        logger.info("Shutdown requested, stopping gracefully...")
        if orchestrator:
            orchestrator.running = False
            # Cancel all background tasks for graceful shutdown (Dec 2025)
            if hasattr(orchestrator, '_background_tasks'):
                for task in orchestrator._background_tasks:
                    if not task.done():
                        task.cancel()
            # Schedule ramdrive sync in a thread to avoid blocking signal handler
            # Don't call sys.exit() - let asyncio loop exit cleanly

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Mar 2026: Event loop watchdog -- runs in a separate OS thread so a blocked
    # asyncio event loop doesn't prevent detection. Polls /health every 30s;
    # after 5 consecutive timeouts (~150s of unresponsiveness) calls os._exit(1)
    # to trigger systemd/supervisor restart. Startup grace period: 300s.
    def _event_loop_watchdog(port: int) -> None:
        import urllib.request
        import os as _os
        _STARTUP_GRACE = 300   # seconds before watchdog activates (was 180, increased to survive init)
        _CHECK_INTERVAL = 30   # seconds between health checks
        _TIMEOUT = 20          # seconds per HTTP check
        _MAX_FAILURES = 5      # consecutive failures before exit (was 3, increased for coordinator stability)
        import time as _time
        _time.sleep(_STARTUP_GRACE)
        consecutive_failures = 0
        while True:
            try:
                url = f"http://127.0.0.1:{port}/health"
                with urllib.request.urlopen(url, timeout=_TIMEOUT):
                    consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    f"[Watchdog] /health unresponsive ({consecutive_failures}/{_MAX_FAILURES})"
                )
                if consecutive_failures >= _MAX_FAILURES:
                    logger.error(
                        "[Watchdog] Event loop appears blocked — triggering restart via os._exit(1)"
                    )
                    _os._exit(1)
            _time.sleep(_CHECK_INTERVAL)

    import threading as _threading
    _watchdog_thread = _threading.Thread(
        target=_event_loop_watchdog,
        args=(args.port,),
        daemon=True,
        name="EventLoopWatchdog",
    )
    _watchdog_thread.start()
    logger.info(f"[Watchdog] Started event loop watchdog (grace={300}s, interval={30}s, max_failures={5})")

    # Mar 2026: Start ProcessWatchdog on coordinator to kill runaway subprocess trees.
    if os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes"):
        try:
            from app.utils.coordinator_governor import ProcessWatchdog
            _proc_watchdog = ProcessWatchdog()  # Uses _COORDINATOR_MAX_DESCENDANTS (120)
            _proc_watchdog.start()
        except Exception as _wd_err:
            logger.warning(f"[P2P] Failed to start ProcessWatchdog: {_wd_err}")

    # Run with exception logging
    try:
        logger.info(f"Starting P2P orchestrator main loop: {args.node_id}")
        asyncio.run(orchestrator.run())
    except Exception as e:  # noqa: BLE001
        logger.exception(f"P2P orchestrator crashed: {e}")
        sys.exit(1)
    finally:
        # Ensure ramdrive is synced on exit (moved from signal handler to avoid race)
        if orchestrator:
            try:
                orchestrator.stop_ramdrive_syncer(final_sync=True)
                logger.info("Ramdrive sync completed on shutdown")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Ramdrive sync on shutdown failed: {e}")
            # December 2025: Close webhook notifier to prevent memory leaks
            try:
                if hasattr(orchestrator, 'notifier') and orchestrator.notifier:
                    orchestrator.notifier.close_sync()
            except (RuntimeError, OSError, AttributeError) as e:
                # Dec 2025: Narrowed from bare Exception; best effort cleanup
                logger.debug(f"Notifier close failed (best effort): {e}")

            # Mar 2026: Kill child processes to prevent zombie accumulation.
            # Without this, every P2P restart leaves orphan selfplay/training/
            # gauntlet subprocesses that accumulate (148 zombies observed).
            try:
                if hasattr(orchestrator, 'job_manager') and orchestrator.job_manager:
                    import asyncio as _async_cleanup
                    try:
                        loop = _async_cleanup.get_event_loop()
                        if loop.is_running():
                            # Can't await in finally block, use sync kill
                            killed = orchestrator.job_manager._kill_all_processes_sync()
                        else:
                            killed = _async_cleanup.run(
                                orchestrator.job_manager.cleanup_active_processes()
                            )
                    except RuntimeError:
                        killed = orchestrator.job_manager._kill_all_processes_sync()
                    if killed:
                        logger.info(f"Killed {killed} child processes on shutdown")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Child process cleanup failed (best effort): {e}")

            # December 2025: Close work queue to persist final stats
            try:
                from app.coordination.work_queue import reset_work_queue
                reset_work_queue()
            except (ImportError, RuntimeError, sqlite3.Error) as e:
                logger.debug(f"Work queue cleanup failed (best effort): {e}")

            # December 2025: Release singleton lock on shutdown
            _release_singleton_lock()

            # January 2026: Release supervisor role on shutdown
            _release_supervisor_role()
