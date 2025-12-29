"""Process management utilities for scripts.

Provides common patterns for process management:
- Signal handling and graceful shutdown
- Singleton locks (prevent duplicate processes)
- PID file management
- Process discovery and control
- Subprocess execution helpers

Usage:
    from scripts.lib.process import (
        SingletonLock,
        SignalHandler,
        is_process_running,
        find_processes_by_pattern,
        kill_process,
        run_command,
    )

    # Ensure only one instance runs
    with SingletonLock("my-daemon") as lock:
        if not lock.acquired:
            print("Already running")
            sys.exit(0)
        run_daemon()

    # Graceful shutdown on signals
    handler = SignalHandler()
    while handler.running:
        do_work()
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# December 29, 2025: Configurable process grace period
# Allows longer cleanup time for P2P daemon and other processes
# Default raised from 5s to 30s to prevent premature SIGKILL
PROCESS_GRACE_PERIOD = float(os.environ.get("RINGRIFT_PROCESS_GRACE_PERIOD", "30.0"))
PROCESS_FORCE_KILL_DELAY = float(os.environ.get("RINGRIFT_PROCESS_FORCE_KILL_DELAY", "10.0"))


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    command: str = ""
    args: list[str] = field(default_factory=list)

    @property
    def full_command(self) -> str:
        """Get full command with arguments."""
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command


@dataclass
class CommandOutput:
    """Result of a local command execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: str

    def __bool__(self) -> bool:
        return self.success

    @property
    def output(self) -> str:
        """Return stdout, or stderr if stdout is empty."""
        return self.stdout.strip() or self.stderr.strip()


class SingletonLock:
    """File-based singleton lock to prevent duplicate processes.

    Uses fcntl for cross-platform file locking. The lock is automatically
    released when the process exits.

    Features:
    - Automatic stale lock detection and cleanup
    - Configurable retry with stale lock recovery
    - Process liveness verification before reporting conflicts

    Usage:
        lock = SingletonLock("my-daemon")
        if not lock.acquire():
            print("Already running")
            sys.exit(1)
        # ... run daemon ...
        lock.release()

        # Or as context manager:
        with SingletonLock("my-daemon") as lock:
            if not lock.acquired:
                sys.exit(1)
            run_daemon()

        # With automatic stale lock cleanup:
        lock = SingletonLock("my-daemon", auto_cleanup_stale=True)
        if not lock.acquire():
            # Only fails if another live process holds the lock
            sys.exit(1)
    """

    def __init__(
        self,
        name: str,
        lock_dir: Path | None = None,
        write_pid: bool = True,
        auto_cleanup_stale: bool = True,
        stale_timeout_seconds: float = 0,
    ):
        """Initialize singleton lock.

        Args:
            name: Unique name for this lock (used in filename)
            lock_dir: Directory for lock file (default: /tmp)
            write_pid: Write current PID to lock file
            auto_cleanup_stale: Automatically clean up locks from dead processes
            stale_timeout_seconds: Consider lock stale if holder dead for this long
                                   (0 = immediate cleanup when holder is dead)
        """
        self.name = name
        self.lock_dir = lock_dir or Path("/tmp")
        self.write_pid = write_pid
        self.auto_cleanup_stale = auto_cleanup_stale
        self.stale_timeout_seconds = stale_timeout_seconds
        self.lock_path = self.lock_dir / f"ringrift_{name}.lock"
        self._file_handle: Any | None = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        """Check if lock was successfully acquired."""
        return self._acquired

    def _is_holder_alive(self, pid: int) -> bool:
        """Check if the lock holder process is still running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, False if dead or doesn't exist
        """
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it - assume alive
            return True
        except (OSError, ValueError):
            return False

    def _cleanup_stale_lock(self) -> bool:
        """Clean up a stale lock file from a dead process.

        Returns:
            True if cleanup was performed, False otherwise
        """
        holder_pid = self.get_holder_pid()
        if holder_pid is None:
            # No PID in lock file, safe to proceed
            return True

        if self._is_holder_alive(holder_pid):
            # Holder is still alive, not stale
            return False

        # Holder is dead - log and attempt cleanup
        logger.info(
            f"[SingletonLock] Cleaning up stale lock for '{self.name}' "
            f"(dead PID {holder_pid})"
        )

        try:
            # Remove the stale lock file
            self.lock_path.unlink(missing_ok=True)
            return True
        except OSError as e:
            logger.warning(f"[SingletonLock] Failed to remove stale lock: {e}")
            return False

    def acquire(self, blocking: bool = False, max_retries: int = 2) -> bool:
        """Acquire the singleton lock.

        Args:
            blocking: If True, wait until lock is available
            max_retries: Number of retries after stale lock cleanup (default: 2)

        Returns:
            True if lock was acquired, False if already held by live process
        """
        retries = 0
        while retries <= max_retries:
            try:
                self.lock_dir.mkdir(parents=True, exist_ok=True)
                self._file_handle = open(self.lock_path, "a+")

                flags = fcntl.LOCK_EX
                if not blocking:
                    flags |= fcntl.LOCK_NB

                fcntl.flock(self._file_handle.fileno(), flags)

                if self.write_pid:
                    self._file_handle.seek(0)
                    self._file_handle.truncate()
                    self._file_handle.write(str(os.getpid()))
                    self._file_handle.flush()

                self._acquired = True
                return True

            except OSError:
                if self._file_handle:
                    self._file_handle.close()
                    self._file_handle = None

                # Check if we should attempt stale lock cleanup
                if self.auto_cleanup_stale and retries < max_retries:
                    holder_pid = self.get_holder_pid()
                    if holder_pid and not self._is_holder_alive(holder_pid):
                        # Holder is dead - this shouldn't normally happen with
                        # fcntl (kernel releases lock on process death), but can
                        # occur with NFS or other edge cases
                        if self._cleanup_stale_lock():
                            retries += 1
                            time.sleep(0.1)  # Brief pause before retry
                            continue

                return False

        return False

    def release(self) -> None:
        """Release the singleton lock."""
        if self._file_handle:
            try:
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
            except (OSError, ValueError):
                pass
            self._file_handle = None
        self._acquired = False

    def get_holder_pid(self) -> int | None:
        """Get PID of process holding the lock, if any."""
        try:
            if self.lock_path.exists():
                content = self.lock_path.read_text().strip()
                if content:
                    return int(content)
        except (OSError, ValueError):
            pass
        return None

    def is_holder_alive(self) -> bool:
        """Check if the current lock holder process is still running.

        Returns:
            True if holder exists and is running, False otherwise
        """
        holder_pid = self.get_holder_pid()
        if holder_pid is None:
            return False
        return self._is_holder_alive(holder_pid)

    def get_holder_command(self) -> str | None:
        """Get the command line of the lock holder process.

        Returns:
            Command line string, or None if not available
        """
        holder_pid = self.get_holder_pid()
        if holder_pid is None:
            return None

        try:
            # Try Linux /proc first
            cmdline_path = Path(f"/proc/{holder_pid}/cmdline")
            if cmdline_path.exists():
                cmdline = cmdline_path.read_bytes().decode("utf-8", errors="replace")
                return cmdline.replace("\x00", " ").strip()
        except (OSError, PermissionError):
            pass

        try:
            # Fall back to ps
            result = subprocess.run(
                ["ps", "-p", str(holder_pid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

        return None

    def is_holder_expected_process(self, expected_pattern: str) -> bool:
        """Check if the lock holder is running the expected command.

        Args:
            expected_pattern: Substring to match in holder's command line

        Returns:
            True if holder command contains expected_pattern
        """
        command = self.get_holder_command()
        if command is None:
            return False
        return expected_pattern in command

    def force_release(self, kill_holder: bool = False, signal_num: int = signal.SIGTERM) -> bool:
        """Force release a lock, optionally killing the holder.

        WARNING: Use with caution. This can cause issues if the holder is
        legitimately running.

        Args:
            kill_holder: If True, send signal to holder process
            signal_num: Signal to send if kill_holder is True

        Returns:
            True if lock was released/cleaned up
        """
        holder_pid = self.get_holder_pid()

        if kill_holder and holder_pid and self._is_holder_alive(holder_pid):
            logger.warning(f"[SingletonLock] Force-killing holder PID {holder_pid}")
            try:
                os.kill(holder_pid, signal_num)
                # Wait a bit for process to die
                for _ in range(30):  # 3 seconds max
                    time.sleep(0.1)
                    if not self._is_holder_alive(holder_pid):
                        break
                else:
                    # Try SIGKILL if still alive
                    try:
                        os.kill(holder_pid, signal.SIGKILL)
                        time.sleep(0.5)
                    except ProcessLookupError:
                        pass
            except (ProcessLookupError, PermissionError) as e:
                logger.warning(f"[SingletonLock] Could not kill holder: {e}")

        # Remove the lock file
        try:
            self.lock_path.unlink(missing_ok=True)
            logger.info(f"[SingletonLock] Force-released lock for '{self.name}'")
            return True
        except OSError as e:
            logger.error(f"[SingletonLock] Failed to force-release: {e}")
            return False

    def get_lock_status(self) -> dict:
        """Get detailed status of this lock.

        Returns:
            Dict with lock status information
        """
        holder_pid = self.get_holder_pid()
        holder_alive = self._is_holder_alive(holder_pid) if holder_pid else False
        holder_command = self.get_holder_command() if holder_pid and holder_alive else None

        return {
            "name": self.name,
            "lock_path": str(self.lock_path),
            "lock_exists": self.lock_path.exists(),
            "holder_pid": holder_pid,
            "holder_alive": holder_alive,
            "holder_command": holder_command,
            "is_stale": holder_pid is not None and not holder_alive,
            "acquired_by_us": self._acquired,
            "our_pid": os.getpid(),
        }

    def is_holder_stale(
        self,
        heartbeat_path: Path | None = None,
        heartbeat_table: str = "heartbeat",
        max_heartbeat_age: float = 90.0,
    ) -> tuple[bool, str]:
        """Check if the lock holder is stale based on heartbeat.

        A holder is considered stale if:
        1. The holder process is dead (PID doesn't exist)
        2. The heartbeat is older than max_heartbeat_age seconds

        This is useful for watchdog processes to detect hung daemons
        that are still "running" but not making progress.

        Args:
            heartbeat_path: Path to SQLite database with heartbeat table.
                           Default: data/coordination/master_loop_state.db
            heartbeat_table: Name of table containing heartbeat (default: "heartbeat")
            max_heartbeat_age: Maximum age of heartbeat in seconds before
                              considering holder stale (default: 90s)

        Returns:
            Tuple of (is_stale, reason_string)

        Example:
            lock = SingletonLock("master_loop")
            is_stale, reason = lock.is_holder_stale(
                heartbeat_path=Path("data/coordination/master_loop_state.db"),
                max_heartbeat_age=60.0,
            )
            if is_stale:
                logger.warning(f"Master loop is stale: {reason}")
                lock.force_release(kill_holder=True)

        December 2025: Added as part of 48-hour autonomous operation plan.
        """
        holder_pid = self.get_holder_pid()

        # No holder - not stale (nothing to check)
        if holder_pid is None:
            return False, "No holder"

        # Check if holder process is alive
        if not self._is_holder_alive(holder_pid):
            return True, f"Holder process {holder_pid} is dead"

        # Process is alive - check heartbeat if path provided
        if heartbeat_path is None:
            # Use default path relative to this file
            heartbeat_path = Path(__file__).parent.parent.parent / "data" / "coordination" / "master_loop_state.db"

        if not heartbeat_path.exists():
            # No heartbeat file - can't determine staleness from heartbeat
            return False, "No heartbeat database"

        try:
            import sqlite3
            conn = sqlite3.connect(heartbeat_path, timeout=5.0)
            row = conn.execute(
                f"SELECT last_beat FROM {heartbeat_table} WHERE id = 1"
            ).fetchone()
            conn.close()

            if row is None:
                return False, "No heartbeat record"

            last_beat = row[0]
            heartbeat_age = time.time() - last_beat

            if heartbeat_age > max_heartbeat_age:
                return True, f"Heartbeat stale ({heartbeat_age:.1f}s > {max_heartbeat_age}s threshold)"

            return False, f"Heartbeat fresh ({heartbeat_age:.1f}s old)"

        except Exception as e:
            logger.warning(f"[SingletonLock] Failed to check heartbeat: {e}")
            return False, f"Heartbeat check failed: {e}"

    def __enter__(self) -> SingletonLock:
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()


class SignalHandler:
    """Graceful signal handler for daemon processes.

    Catches SIGTERM and SIGINT, sets a flag for clean shutdown.

    Usage:
        handler = SignalHandler()
        while handler.running:
            do_work()

        # Or with callback:
        def on_shutdown():
            cleanup()

        handler = SignalHandler(on_shutdown=on_shutdown)
    """

    def __init__(
        self,
        on_shutdown: Callable[[], None] | None = None,
        signals: Sequence[int] | None = None,
    ):
        """Initialize signal handler.

        Args:
            on_shutdown: Callback to run when shutdown signal received
            signals: Signals to handle (default: SIGTERM, SIGINT)
        """
        self.running = True
        self.shutdown_requested = False
        self._on_shutdown = on_shutdown
        self._original_handlers: dict[int, Any] = {}

        signals = signals or [signal.SIGTERM, signal.SIGINT]
        for sig in signals:
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.shutdown_requested = True
        self.running = False

        if self._on_shutdown:
            try:
                self._on_shutdown()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

    def restore_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        True if process exists and is running
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def find_processes_by_pattern(
    pattern: str,
    exclude_self: bool = True,
) -> list[ProcessInfo]:
    """Find running processes matching a pattern.

    Uses pgrep to find processes. Pattern is matched against full command line.

    Args:
        pattern: Regex pattern to match against process command line
        exclude_self: Exclude current process from results

    Returns:
        List of ProcessInfo for matching processes
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "-a", pattern],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return []

        processes = []
        current_pid = os.getpid()

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) >= 1:
                try:
                    pid = int(parts[0])
                    if exclude_self and pid == current_pid:
                        continue
                    command = parts[1] if len(parts) > 1 else ""
                    processes.append(ProcessInfo(pid=pid, command=command))
                except ValueError:
                    continue

        return processes

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def count_processes_by_pattern(pattern: str, exclude_self: bool = True) -> int:
    """Count running processes matching a pattern.

    Args:
        pattern: Regex pattern to match against process command line
        exclude_self: Exclude current process from count

    Returns:
        Number of matching processes
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "-c", pattern],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return 0

        count = int(result.stdout.strip())
        if exclude_self:
            # Check if we match the pattern
            own_cmd = " ".join(sys.argv)
            if pattern in own_cmd:
                count = max(0, count - 1)
        return count

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return 0


def kill_process(
    pid: int,
    sig: int = signal.SIGTERM,
    wait: bool = False,
    timeout: float | None = None,
) -> bool:
    """Kill a process by PID.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default: SIGTERM)
        wait: Wait for process to terminate
        timeout: Timeout for waiting (if wait=True). Defaults to
            RINGRIFT_PROCESS_GRACE_PERIOD env var (30s).

    Returns:
        True if process was killed or already dead
    """
    if timeout is None:
        timeout = PROCESS_GRACE_PERIOD
    try:
        os.kill(pid, sig)

        if wait:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not is_process_running(pid):
                    return True
                time.sleep(0.1)

            # Process still running, try SIGKILL
            if is_process_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass

        return True

    except ProcessLookupError:
        return True  # Already dead
    except PermissionError:
        logger.warning(f"Permission denied killing PID {pid}")
        return False


def kill_processes_by_pattern(
    pattern: str,
    sig: int = signal.SIGTERM,
    wait: bool = True,
    timeout: float | None = None,
    force_after: float | None = None,
) -> int:
    """Kill all processes matching a pattern.

    Args:
        pattern: Regex pattern to match against process command line
        sig: Initial signal to send (default: SIGTERM)
        wait: Wait for processes to terminate
        timeout: Total timeout for waiting. Defaults to
            RINGRIFT_PROCESS_GRACE_PERIOD env var (30s).
        force_after: Send SIGKILL after this many seconds if still running.
            Defaults to RINGRIFT_PROCESS_FORCE_KILL_DELAY env var (10s).

    Returns:
        Number of processes killed
    """
    if timeout is None:
        timeout = PROCESS_GRACE_PERIOD
    if force_after is None:
        force_after = PROCESS_FORCE_KILL_DELAY
    processes = find_processes_by_pattern(pattern)
    if not processes:
        return 0

    # Send initial signal
    killed = 0
    for proc in processes:
        try:
            os.kill(proc.pid, sig)
            killed += 1
        except (ProcessLookupError, PermissionError):
            continue

    if not wait:
        return killed

    # Wait and escalate if needed
    deadline = time.time() + timeout
    force_time = time.time() + force_after

    while time.time() < deadline:
        still_running = [p for p in processes if is_process_running(p.pid)]
        if not still_running:
            break

        # Escalate to SIGKILL after force_after seconds
        if time.time() >= force_time:
            for proc in still_running:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    continue
            force_time = deadline + 1  # Only try once

        time.sleep(0.2)

    return killed


def run_command(
    command: str | list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
    shell: bool = False,
    capture_output: bool = True,
    check: bool = False,
) -> CommandOutput:
    """Run a local command and return structured output.

    Args:
        command: Command to run (string or list of args)
        cwd: Working directory
        env: Environment variables (merged with current)
        timeout: Command timeout in seconds
        shell: Run via shell
        capture_output: Capture stdout/stderr
        check: Raise exception on non-zero exit

    Returns:
        CommandOutput with execution details

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    start_time = time.time()

    # Prepare environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Prepare command
    if isinstance(command, str) and not shell:
        cmd_args = command.split()
        cmd_str = command
    elif isinstance(command, list):
        cmd_args = command
        cmd_str = " ".join(command)
    else:
        cmd_args = command
        cmd_str = command if isinstance(command, str) else " ".join(command)

    try:
        result = subprocess.run(
            cmd_args if not shell else command,
            cwd=str(cwd) if cwd else None,
            env=run_env,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            shell=shell,
        )

        duration = time.time() - start_time
        output = CommandOutput(
            success=result.returncode == 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=result.returncode,
            duration_seconds=duration,
            command=cmd_str,
        )

        if check and not output.success:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd_str,
                result.stdout,
                result.stderr,
            )

        return output

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        output = CommandOutput(
            success=False,
            stdout=e.stdout or "" if hasattr(e, "stdout") else "",
            stderr=e.stderr or "" if hasattr(e, "stderr") else "",
            exit_code=-1,
            duration_seconds=duration,
            command=cmd_str,
        )
        if check:
            raise
        return output


@contextmanager
def daemon_context(
    name: str,
    lock_dir: Path | None = None,
    exit_if_running: bool = True,
) -> Generator[SignalHandler]:
    """Context manager for daemon processes.

    Provides singleton locking and signal handling.

    Args:
        name: Daemon name for lock file
        lock_dir: Directory for lock file
        exit_if_running: Exit if another instance is running

    Yields:
        SignalHandler for checking shutdown status

    Usage:
        with daemon_context("my-daemon") as handler:
            while handler.running:
                do_work()
    """
    lock = SingletonLock(name, lock_dir)

    if not lock.acquire():
        if exit_if_running:
            logger.info(f"{name} is already running")
            sys.exit(0)
        else:
            yield SignalHandler()
            return

    handler = SignalHandler()
    try:
        yield handler
    finally:
        lock.release()


def wait_for_process_exit(
    pid: int,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for a process to exit.

    Args:
        pid: Process ID to wait for
        timeout: Maximum time to wait
        poll_interval: Time between checks

    Returns:
        True if process exited, False if timeout
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_process_running(pid):
            return True
        time.sleep(poll_interval)
    return False


def get_process_start_time(pid: int) -> float | None:
    """Get process start time as Unix timestamp.

    Args:
        pid: Process ID

    Returns:
        Start time as Unix timestamp, or None if not available
    """
    try:
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.exists():
            # Linux: parse /proc/pid/stat
            content = stat_path.read_text()
            # Field 22 is starttime in clock ticks since boot
            parts = content.split()
            if len(parts) > 21:
                starttime = int(parts[21])
                # Get system boot time and clock ticks per second
                uptime = float(Path("/proc/uptime").read_text().split()[0])
                boot_time = time.time() - uptime
                clock_ticks = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 2))
                return boot_time + (starttime / clock_ticks)
    except (OSError, FileNotFoundError, ValueError, IndexError, KeyError, AttributeError):
        pass

    # macOS: use ps
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse date format like "Mon Dec 16 10:30:00 2024"
            import datetime
            dt = datetime.datetime.strptime(
                result.stdout.strip(),
                "%a %b %d %H:%M:%S %Y"
            )
            return dt.timestamp()
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError):
        pass

    return None
