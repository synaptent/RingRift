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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections.abc import Callable, Generator, Sequence

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        name: str,
        lock_dir: Path | None = None,
        write_pid: bool = True,
    ):
        """Initialize singleton lock.

        Args:
            name: Unique name for this lock (used in filename)
            lock_dir: Directory for lock file (default: /tmp)
            write_pid: Write current PID to lock file
        """
        self.name = name
        self.lock_dir = lock_dir or Path("/tmp")
        self.write_pid = write_pid
        self.lock_path = self.lock_dir / f"ringrift_{name}.lock"
        self._file_handle: Any | None = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        """Check if lock was successfully acquired."""
        return self._acquired

    def acquire(self, blocking: bool = False) -> bool:
        """Acquire the singleton lock.

        Args:
            blocking: If True, wait until lock is available

        Returns:
            True if lock was acquired, False if already held
        """
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
            return False

    def release(self) -> None:
        """Release the singleton lock."""
        if self._file_handle:
            try:
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
            except Exception:
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

    def __enter__(self) -> "SingletonLock":
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
    timeout: float = 5.0,
) -> bool:
    """Kill a process by PID.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default: SIGTERM)
        wait: Wait for process to terminate
        timeout: Timeout for waiting (if wait=True)

    Returns:
        True if process was killed or already dead
    """
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
    timeout: float = 5.0,
    force_after: float = 3.0,
) -> int:
    """Kill all processes matching a pattern.

    Args:
        pattern: Regex pattern to match against process command line
        sig: Initial signal to send (default: SIGTERM)
        wait: Wait for processes to terminate
        timeout: Total timeout for waiting
        force_after: Send SIGKILL after this many seconds if still running

    Returns:
        Number of processes killed
    """
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
    command: Union[str, list[str]],
    cwd: Union[str, Path] | None = None,
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
) -> Generator[SignalHandler, None, None]:
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
    except Exception:
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
    except Exception:
        pass

    return None
