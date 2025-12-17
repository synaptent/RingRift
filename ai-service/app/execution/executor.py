"""Unified execution framework for local and remote command execution.

This module provides a consistent interface for running commands locally
or via SSH, with retry logic, timeout handling, and progress monitoring.

All orchestrators should use this instead of implementing their own
subprocess or SSH logic.

Usage:
    from app.execution import LocalExecutor, SSHExecutor, run_command

    # Simple local command
    result = await run_command("python train.py")

    # SSH command with retry
    executor = SSHExecutor(host="worker-1")
    result = await executor.run("python scripts/selfplay.py", timeout=3600)

    # Pool of executors for parallel work
    pool = ExecutorPool()
    pool.add_ssh("worker-1", host="192.168.1.10")
    pool.add_ssh("worker-2", host="192.168.1.11")
    results = await pool.run_all("python benchmark.py")
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from app.core.error_handler import retry_async, SSHError, RetryableError
from app.utils.resource_guard import (
    can_proceed,
    wait_for_resources,
    check_memory,
    check_cpu,
    get_resource_status,
    LIMITS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Resource-Aware Execution
# ============================================================================

async def check_resources_before_spawn(
    wait_if_unavailable: bool = True,
    wait_timeout: float = 300.0,
    required_mem_gb: float = 1.0,
) -> bool:
    """Check if resources are available before spawning a subprocess.

    Args:
        wait_if_unavailable: If True, wait for resources. If False, return immediately.
        wait_timeout: Max time to wait for resources (seconds).
        required_mem_gb: Minimum memory required in GB.

    Returns:
        True if resources are available, False otherwise.
    """
    if can_proceed(check_disk=False, mem_required_gb=required_mem_gb):
        return True

    if not wait_if_unavailable:
        status = get_resource_status()
        logger.warning(
            f"Resources unavailable for spawn: CPU={status['cpu']['used_percent']:.1f}%, "
            f"Memory={status['memory']['used_percent']:.1f}%"
        )
        return False

    logger.info(f"Waiting up to {wait_timeout}s for resources before spawn...")
    return wait_for_resources(timeout=wait_timeout, mem_required_gb=required_mem_gb)


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str
    host: str = "local"
    timed_out: bool = False

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        return f"{self.stdout}\n{self.stderr}".strip()

    def __bool__(self) -> bool:
        return self.success


@dataclass
class SSHConfig:
    """SSH connection configuration."""

    host: str
    user: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    connect_timeout: int = 10
    options: Dict[str, str] = field(default_factory=dict)

    @property
    def ssh_target(self) -> str:
        """Get SSH target string (user@host or just host)."""
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host

    def build_ssh_command(self) -> List[str]:
        """Build the base SSH command with options."""
        cmd = ["ssh"]

        # Standard options for non-interactive use
        cmd.extend(["-o", "BatchMode=yes"])
        cmd.extend(["-o", f"ConnectTimeout={self.connect_timeout}"])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])

        # Port
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])

        # Key file
        if self.key_path:
            cmd.extend(["-i", self.key_path])

        # Custom options
        for key, value in self.options.items():
            cmd.extend(["-o", f"{key}={value}"])

        cmd.append(self.ssh_target)
        return cmd


class BaseExecutor(ABC):
    """Base class for command executors."""

    @abstractmethod
    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
    ) -> ExecutionResult:
        """Execute a command.

        Args:
            command: Command to execute (string, will be parsed)
            timeout: Timeout in seconds (None for no timeout)
            cwd: Working directory
            env: Additional environment variables
            capture_output: Whether to capture stdout/stderr

        Returns:
            ExecutionResult with command output
        """
        pass

    @abstractmethod
    async def check_available(self) -> bool:
        """Check if this executor is available (e.g., SSH host reachable)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this executor."""
        pass


class LocalExecutor(BaseExecutor):
    """Execute commands on the local machine."""

    def __init__(
        self,
        working_dir: Optional[str] = None,
        check_resources: bool = False,
        required_mem_gb: float = 1.0,
    ):
        """Initialize local executor.

        Args:
            working_dir: Default working directory for commands.
            check_resources: If True, check resources before each spawn.
            required_mem_gb: Memory required for spawned processes.
        """
        self.working_dir = working_dir
        self.check_resources = check_resources
        self.required_mem_gb = required_mem_gb

    @property
    def name(self) -> str:
        return "local"

    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
    ) -> ExecutionResult:
        """Execute a local command."""
        effective_cwd = cwd or self.working_dir
        effective_env = {**os.environ, **(env or {})}

        start_time = time.time()
        timed_out = False

        # Check resources before spawning if enabled
        if self.check_resources:
            resources_ok = await check_resources_before_spawn(
                wait_if_unavailable=True,
                wait_timeout=60.0,
                required_mem_gb=self.required_mem_gb,
            )
            if not resources_ok:
                return ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"Resource limits exceeded (CPU>{LIMITS.CPU_MAX_PERCENT}% or Memory>{LIMITS.MEMORY_MAX_PERCENT}%)",
                    duration_seconds=time.time() - start_time,
                    command=command,
                    host="local",
                    timed_out=False,
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=effective_cwd,
                env=effective_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                returncode = proc.returncode or 0

            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                await proc.wait()
                stdout = ""
                stderr = f"Command timed out after {timeout}s"
                returncode = -1

        except Exception as e:
            logger.error(f"Local execution failed: {e}")
            stdout = ""
            stderr = str(e)
            returncode = -1

        duration = time.time() - start_time

        return ExecutionResult(
            success=returncode == 0 and not timed_out,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            command=command,
            host="local",
            timed_out=timed_out,
        )

    async def check_available(self) -> bool:
        """Local executor is always available."""
        return True


class SSHExecutor(BaseExecutor):
    """Execute commands on a remote host via SSH."""

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        port: int = 22,
        key_path: Optional[str] = None,
        connect_timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.config = SSHConfig(
            host=host,
            user=user,
            port=port,
            key_path=key_path,
            connect_timeout=connect_timeout,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return f"ssh:{self.config.ssh_target}"

    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
    ) -> ExecutionResult:
        """Execute a command via SSH."""
        # Build the remote command
        remote_cmd = command
        if cwd:
            remote_cmd = f"cd {shlex.quote(cwd)} && {command}"
        if env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            remote_cmd = f"{env_prefix} {remote_cmd}"

        # Build SSH command
        ssh_cmd = self.config.build_ssh_command()
        ssh_cmd.append(remote_cmd)

        start_time = time.time()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                returncode = proc.returncode or 0

            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                await proc.wait()
                stdout = ""
                stderr = f"SSH command timed out after {timeout}s"
                returncode = -1

        except Exception as e:
            logger.error(f"SSH execution to {self.config.host} failed: {e}")
            stdout = ""
            stderr = str(e)
            returncode = -1

        duration = time.time() - start_time

        return ExecutionResult(
            success=returncode == 0 and not timed_out,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            command=command,
            host=self.config.ssh_target,
            timed_out=timed_out,
        )

    async def run_with_retry(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute with automatic retry on transient failures."""
        last_result = None

        for attempt in range(1, self.max_retries + 1):
            result = await self.run(command, timeout, cwd, env)

            if result.success:
                return result

            last_result = result

            # Check for retryable errors
            is_retryable = (
                result.timed_out
                or "Connection refused" in result.stderr
                or "Connection timed out" in result.stderr
                or "No route to host" in result.stderr
            )

            if not is_retryable or attempt == self.max_retries:
                break

            logger.warning(
                f"SSH to {self.config.host} failed (attempt {attempt}/{self.max_retries}), "
                f"retrying in {self.retry_delay}s..."
            )
            await asyncio.sleep(self.retry_delay)

        return last_result or ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="All retry attempts failed",
            duration_seconds=0,
            command=command,
            host=self.config.ssh_target,
        )

    async def check_available(self) -> bool:
        """Check if SSH host is reachable."""
        result = await self.run("echo ping", timeout=10)
        self._available = result.success and "ping" in result.stdout
        return self._available


class ExecutorPool:
    """Pool of executors for parallel command execution."""

    def __init__(self):
        self.executors: Dict[str, BaseExecutor] = {}
        self._local = LocalExecutor()

    def add_local(self, name: str = "local", working_dir: Optional[str] = None) -> None:
        """Add local executor to the pool."""
        self.executors[name] = LocalExecutor(working_dir)

    def add_ssh(
        self,
        name: str,
        host: str,
        user: Optional[str] = None,
        port: int = 22,
        key_path: Optional[str] = None,
    ) -> None:
        """Add SSH executor to the pool."""
        self.executors[name] = SSHExecutor(host, user, port, key_path)

    def get(self, name: str) -> Optional[BaseExecutor]:
        """Get an executor by name."""
        return self.executors.get(name)

    async def check_all_available(self) -> Dict[str, bool]:
        """Check availability of all executors."""
        tasks = {
            name: executor.check_available()
            for name, executor in self.executors.items()
        }

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Availability check failed for {name}: {e}")
                results[name] = False

        return results

    async def run_on(
        self,
        executor_name: str,
        command: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Run command on a specific executor."""
        executor = self.executors.get(executor_name)
        if not executor:
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Unknown executor: {executor_name}",
                duration_seconds=0,
                command=command,
            )
        return await executor.run(command, timeout)

    async def run_all(
        self,
        command: str,
        timeout: Optional[int] = None,
        only_available: bool = True,
    ) -> Dict[str, ExecutionResult]:
        """Run command on all executors in parallel."""
        if only_available:
            available = await self.check_all_available()
            executors = {
                name: ex for name, ex in self.executors.items()
                if available.get(name, False)
            }
        else:
            executors = self.executors

        tasks = {
            name: executor.run(command, timeout)
            for name, executor in executors.items()
        }

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=0,
                    command=command,
                    host=name,
                )

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

async def run_command(
    command: str,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> ExecutionResult:
    """Run a local command (convenience function)."""
    executor = LocalExecutor()
    return await executor.run(command, timeout, cwd, env)


def run_command_sync(
    command: str,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
) -> ExecutionResult:
    """Synchronous version of run_command for non-async contexts."""
    effective_env = {**os.environ, **(env or {})}
    start_time = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=effective_env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode

    except subprocess.TimeoutExpired:
        timed_out = True
        stdout = ""
        stderr = f"Command timed out after {timeout}s"
        returncode = -1

    except Exception as e:
        stdout = ""
        stderr = str(e)
        returncode = -1

    duration = time.time() - start_time

    return ExecutionResult(
        success=returncode == 0 and not timed_out,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        command=command,
        host="local",
        timed_out=timed_out,
    )


async def run_ssh_command(
    host: str,
    command: str,
    user: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: int = 3,
) -> ExecutionResult:
    """Run a command via SSH (convenience function)."""
    executor = SSHExecutor(host, user, max_retries=max_retries)
    return await executor.run_with_retry(command, timeout)


async def run_ssh_command_async(
    host: str,
    command: str,
    user: Optional[str] = None,
    timeout: Optional[int] = None,
) -> ExecutionResult:
    """Alias for run_ssh_command for naming consistency."""
    return await run_ssh_command(host, command, user, timeout)


def run_ssh_command_sync(
    host: str,
    command: str,
    user: Optional[str] = None,
    port: int = 22,
    key_path: Optional[str] = None,
    timeout: int = 60,
) -> ExecutionResult:
    """Synchronous SSH command execution."""
    config = SSHConfig(host=host, user=user, port=port, key_path=key_path)
    ssh_cmd = config.build_ssh_command()
    ssh_cmd.append(command)

    start_time = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        returncode = result.returncode

    except subprocess.TimeoutExpired:
        timed_out = True
        stdout = ""
        stderr = f"SSH command timed out after {timeout}s"
        returncode = -1

    except Exception as e:
        stdout = ""
        stderr = str(e)
        returncode = -1

    duration = time.time() - start_time

    return ExecutionResult(
        success=returncode == 0 and not timed_out,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        command=command,
        host=config.ssh_target,
        timed_out=timed_out,
    )
