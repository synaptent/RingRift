"""Cluster host configuration, memory detection, and SSH utilities.

This module provides shared infrastructure for distributed operations:
- YAML-based host configuration loading
- Local and remote memory detection
- Host eligibility filtering by memory requirements
- SSH command execution utilities

Used by:
- run_distributed_selfplay_soak.py
- run_cmaes_optimization.py (distributed mode)
- Distributed NNUE training
- Memory profiling and benchmarks
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default paths relative to ai-service/
CONFIG_FILE_PATH = "config/distributed_hosts.yaml"
TEMPLATE_CONFIG_PATH = "config/distributed_hosts.template.yaml"

# Memory requirements per board type (in GB)
# Based on empirical testing:
# - 16GB machine: only 8x8 works reliably
# - 64GB machine: can run 19x19/hex with memory pressure
# - 96GB machine: runs everything comfortably
BOARD_MEMORY_REQUIREMENTS: Dict[str, int] = {
    "square8": 8,      # 8GB minimum for 8x8 games
    "square19": 48,    # 48GB minimum for 19x19 games
    "hexagonal": 48,   # 48GB minimum for hex games
}

# Default SSH key for cluster operations
DEFAULT_SSH_KEY = "~/.ssh/id_cluster"


@dataclass
class HostMemoryInfo:
    """Memory information for a host."""
    total_gb: int
    available_gb: int

    def __str__(self) -> str:
        return f"{self.total_gb}GB total, {self.available_gb}GB available"

    @property
    def is_high_memory(self) -> bool:
        """Check if host has high memory (48GB+) for heavy workloads."""
        return self.total_gb >= 48


@dataclass
class HostConfig:
    """Configuration for a remote host."""
    name: str
    ssh_host: str
    tailscale_ip: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    memory_gb: Optional[int] = None
    work_dir: Optional[str] = None
    venv_activate: Optional[str] = None
    python_path: Optional[str] = None
    max_parallel_jobs: int = 1
    worker_port: int = 8765  # Default HTTP worker port
    worker_url: Optional[str] = None  # Optional explicit worker URL
    properties: Dict = field(default_factory=dict)

    @property
    def ssh_target(self) -> str:
        """Return SSH target string (user@host when needed)."""
        if "@" in self.ssh_host:
            return self.ssh_host
        if self.ssh_user:
            return f"{self.ssh_user}@{self.ssh_host}"
        return self.ssh_host

    @property
    def ssh_targets(self) -> List[str]:
        """Return candidate SSH targets in priority order.

        When a ``tailscale_ip`` is configured, prefer it (mesh routing is often
        more reliable than public IPs / NATed providers). Fall back to the
        primary ``ssh_host``.
        """
        candidates: List[str] = []
        for raw in (self.tailscale_ip, self.ssh_host):
            if not raw:
                continue
            host = str(raw).strip()
            if not host:
                continue
            if "@" in host:
                target = host
            elif self.ssh_user:
                target = f"{self.ssh_user}@{host}"
            else:
                target = host
            if target not in candidates:
                candidates.append(target)
        return candidates or [self.ssh_target]

    @property
    def ssh_key_path(self) -> str:
        """Get the SSH key path, with default fallback."""
        return os.path.expanduser(self.ssh_key or DEFAULT_SSH_KEY)

    @property
    def work_directory(self) -> str:
        """Get the working directory for remote execution."""
        base = (self.work_dir or "~/Development/RingRift").rstrip("/")
        if base.endswith("ai-service"):
            return base
        return f"{base}/ai-service"

    @property
    def http_worker_url(self) -> str:
        """Get the HTTP worker URL for this host."""
        if self.worker_url:
            return self.worker_url
        host = (self.tailscale_ip or self.ssh_host) if self.ssh_host else (self.tailscale_ip or "")
        if "@" in host:
            host = host.split("@", 1)[1]
        return f"http://{host}:{self.worker_port}"


# Global host configuration cache
_HOST_CONFIG_CACHE: Dict[str, HostConfig] = {}
_HOST_MEMORY_CACHE: Dict[str, HostMemoryInfo] = {}


def get_ai_service_dir() -> Path:
    """Get the path to the ai-service directory."""
    # Navigate up from this file to ai-service/
    return Path(__file__).parent.parent.parent


def load_remote_hosts(config_path: Optional[str] = None) -> Dict[str, HostConfig]:
    """Load remote host configuration from YAML file.

    Args:
        config_path: Optional explicit path to config file.
                    If not provided, looks in config/distributed_hosts.yaml

    Returns:
        Dict mapping host names to HostConfig objects.

    Note:
        Copy config/distributed_hosts.template.yaml to config/distributed_hosts.yaml
        and fill in your actual host details.
    """
    global _HOST_CONFIG_CACHE

    # Return cached if already loaded
    if _HOST_CONFIG_CACHE and not config_path:
        return _HOST_CONFIG_CACHE

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Run 'pip install pyyaml' for host configuration.")
        return {}

    ai_service_dir = get_ai_service_dir()

    # Try explicit path first, then default
    if config_path:
        if not os.path.isabs(config_path):
            config_path = str(ai_service_dir / config_path)
        paths_to_try = [config_path]
    else:
        paths_to_try = [str(ai_service_dir / CONFIG_FILE_PATH)]

    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            hosts_dict = config.get("hosts", {})
            hosts = {}

            for name, host_data in hosts_dict.items():
                ssh_host_raw = host_data.get("ssh_host", name)
                tailscale_ip = host_data.get("tailscale_ip")
                ssh_user = host_data.get("ssh_user")
                ssh_host = ssh_host_raw

                # Support configs that embed the username in ssh_host (e.g. "user@1.2.3.4")
                # while keeping `ssh_host` clean for HTTP URLs.
                if "@" in ssh_host_raw:
                    user_part, host_part = ssh_host_raw.split("@", 1)
                    if not ssh_user:
                        ssh_user = user_part
                    ssh_host = host_part

                hosts[name] = HostConfig(
                    name=name,
                    ssh_host=ssh_host,
                    tailscale_ip=tailscale_ip,
                    ssh_user=ssh_user,
                    ssh_port=int(host_data.get("ssh_port", 22) or 22),
                    ssh_key=host_data.get("ssh_key"),
                    memory_gb=host_data.get("memory_gb"),
                    work_dir=host_data.get("ringrift_path") or host_data.get("work_dir"),
                    venv_activate=host_data.get("venv_activate"),
                    python_path=host_data.get("python_path"),
                    max_parallel_jobs=host_data.get("max_parallel_jobs", 1),
                    worker_port=host_data.get("worker_port", 8765),
                    worker_url=host_data.get("worker_url"),
                    properties=host_data,
                )

            if hosts:
                logger.info(f"Loaded {len(hosts)} remote host(s) from {path}")
                _HOST_CONFIG_CACHE.update(hosts)

            return hosts

    # No config found
    template_path = ai_service_dir / TEMPLATE_CONFIG_PATH
    if template_path.exists():
        logger.info(f"No host configuration found. Copy {template_path} to {ai_service_dir / CONFIG_FILE_PATH}")

    return {}


def get_local_memory_gb() -> Tuple[int, int]:
    """Get total and available physical memory on local machine in GB.

    Returns:
        Tuple of (total_gb, available_gb)
    """
    total_gb = 8
    available_gb = 8

    try:
        # macOS: use sysctl for total
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            bytes_total = int(result.stdout.strip())
            total_gb = bytes_total // (1024 ** 3)

        # macOS: use vm_stat for available (free + inactive pages)
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            page_size = 4096
            free_pages = 0
            inactive_pages = 0

            for line in result.stdout.split('\n'):
                if 'page size of' in line:
                    match = re.search(r'page size of (\d+)', line)
                    if match:
                        page_size = int(match.group(1))
                elif 'Pages free:' in line:
                    free_pages = int(line.split(':')[1].strip().rstrip('.'))
                elif 'Pages inactive:' in line:
                    inactive_pages = int(line.split(':')[1].strip().rstrip('.'))

            available_bytes = (free_pages + inactive_pages) * page_size
            available_gb = available_bytes // (1024 ** 3)

        return total_gb, available_gb

    except Exception:
        pass

    try:
        # Linux: read /proc/meminfo
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_gb = kb // (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    available_gb = kb // (1024 * 1024)
        return total_gb, available_gb
    except Exception:
        pass

    logger.warning("Could not detect local memory, assuming 8GB total, 4GB available")
    return 8, 4


def get_remote_memory_gb(host: HostConfig) -> Tuple[int, int]:
    """Get total and available physical memory on remote host in GB via SSH.

    Args:
        host: HostConfig for the remote host

    Returns:
        Tuple of (total_gb, available_gb)
    """
    # Use configured memory if available
    config_total = host.memory_gb

    ssh_cmd_base = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if host.ssh_port and int(host.ssh_port) != 22:
        ssh_cmd_base.extend(["-p", str(int(host.ssh_port))])
    if host.ssh_key:
        ssh_cmd_base.extend(["-i", host.ssh_key_path])

    total_gb = 8
    available_gb = 4

    try:
        # Get total memory
        if config_total:
            total_gb = config_total
        else:
            ssh_cmd = ssh_cmd_base + [
                host.ssh_target,
                "sysctl -n hw.memsize 2>/dev/null || grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2 * 1024}'",
            ]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                bytes_total = int(result.stdout.strip())
                total_gb = bytes_total // (1024 ** 3)

        # Get available memory via vm_stat on macOS
        vm_stat_script = '''
pagesize=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
free=$(vm_stat 2>/dev/null | grep "Pages free:" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}')
inactive=$(vm_stat 2>/dev/null | grep "Pages inactive:" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}')
if [ -n "$free" ] && [ -n "$inactive" ]; then
    echo $(( (free + inactive) * pagesize / 1073741824 ))
elif [ -f /proc/meminfo ]; then
    grep MemAvailable /proc/meminfo | awk '{print int($2/1048576)}'
else
    echo 4
fi
'''
        ssh_cmd = ssh_cmd_base + [host.ssh_target, vm_stat_script]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            available_gb = int(result.stdout.strip())

        return total_gb, available_gb

    except Exception as e:
        logger.warning(f"Could not detect memory for {host.name}: {e}")

    # Fallback
    if config_total:
        return config_total, config_total // 2
    return 8, 4


def detect_host_memory(host_name: str) -> HostMemoryInfo:
    """Detect memory for a single host (local or remote).

    Args:
        host_name: Either "local" or a configured remote host name

    Returns:
        HostMemoryInfo with total and available memory
    """
    global _HOST_MEMORY_CACHE

    if host_name in _HOST_MEMORY_CACHE:
        return _HOST_MEMORY_CACHE[host_name]

    if host_name == "local":
        total_gb, available_gb = get_local_memory_gb()
    else:
        hosts = load_remote_hosts()
        if host_name in hosts:
            total_gb, available_gb = get_remote_memory_gb(hosts[host_name])
        else:
            logger.warning(f"Unknown host: {host_name}, assuming 8GB")
            total_gb, available_gb = 8, 4

    info = HostMemoryInfo(total_gb=total_gb, available_gb=available_gb)
    _HOST_MEMORY_CACHE[host_name] = info
    return info


def detect_all_host_memory(host_names: List[str]) -> Dict[str, HostMemoryInfo]:
    """Detect memory for all specified hosts.

    Args:
        host_names: List of host names ("local" or configured remote hosts)

    Returns:
        Dict mapping host name to HostMemoryInfo
    """
    results = {}
    for name in host_names:
        results[name] = detect_host_memory(name)
    return results


def get_eligible_hosts_for_board(
    board_type: str,
    host_names: List[str],
) -> List[str]:
    """Get hosts with enough memory for a given board type.

    Args:
        board_type: Board type (square8, square19, hexagonal)
        host_names: List of host names to filter

    Returns:
        List of host names that have sufficient memory
    """
    required_memory = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
    eligible = []

    for name in host_names:
        info = detect_host_memory(name)
        if info.total_gb >= required_memory:
            eligible.append(name)

    return eligible


def get_high_memory_hosts(host_names: List[str]) -> List[str]:
    """Get hosts with high memory (48GB+) for heavy AI workloads.

    Args:
        host_names: List of host names to filter

    Returns:
        List of host names with 48GB+ memory
    """
    high_mem = []
    for name in host_names:
        info = detect_host_memory(name)
        if info.is_high_memory:
            high_mem.append(name)
    return high_mem


# Valid host statuses for job dispatch
READY_HOST_STATUSES = {"ready"}  # Only hosts with these statuses receive jobs


def get_host_status(host: HostConfig) -> str:
    """Get the status of a host from its properties.

    Args:
        host: HostConfig to check

    Returns:
        Status string (e.g., "ready", "disabled", "setup", "unstable", "stopped")
        Returns "ready" if no status is configured.
    """
    return str(host.properties.get("status", "ready")).lower().strip()


def is_host_ready(host: HostConfig) -> bool:
    """Check if a host is ready to receive jobs.

    Args:
        host: HostConfig to check

    Returns:
        True if host status is in READY_HOST_STATUSES
    """
    return get_host_status(host) in READY_HOST_STATUSES


def load_ready_hosts(config_path: Optional[str] = None) -> Dict[str, HostConfig]:
    """Load only hosts with status='ready' from YAML configuration.

    This is the preferred function for job dispatch - it filters out disabled,
    setup, unstable, and stopped hosts automatically.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Dict mapping host names to HostConfig objects for ready hosts only.
    """
    all_hosts = load_remote_hosts(config_path)
    return {name: host for name, host in all_hosts.items() if is_host_ready(host)}


def filter_ready_hosts(hosts: Dict[str, HostConfig]) -> Dict[str, HostConfig]:
    """Filter a hosts dict to only include ready hosts.

    Args:
        hosts: Dict of host name to HostConfig

    Returns:
        Filtered dict with only ready hosts
    """
    return {name: host for name, host in hosts.items() if is_host_ready(host)}


def clear_memory_cache() -> None:
    """Clear the memory detection cache to force re-detection."""
    global _HOST_MEMORY_CACHE
    _HOST_MEMORY_CACHE.clear()


class SSHExecutor:
    """Execute commands on remote hosts via SSH."""

    def __init__(self, host: HostConfig):
        self.host = host

    def _normalize_activate(self, command: str) -> str:
        """Normalize venv activation commands for POSIX shells."""
        cleaned = command.strip()
        if cleaned.startswith("source "):
            return ". " + cleaned[len("source ") :]
        return command

    def _build_ssh_cmd(self, ssh_target: str) -> List[str]:
        """Build the base SSH command."""
        cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
        ]
        if self.host.ssh_port and int(self.host.ssh_port) != 22:
            cmd.extend(["-p", str(int(self.host.ssh_port))])
        if self.host.ssh_key:
            cmd.extend(["-i", self.host.ssh_key_path])
        cmd.append(ssh_target)
        return cmd

    def _is_ssh_connection_failure(self, proc: subprocess.CompletedProcess) -> bool:
        """Heuristic: detect SSH-level failures so we can try fallback targets."""
        if proc.returncode == 255:
            return True
        stderr = (proc.stderr or "").lower()
        stdout = (proc.stdout or "").lower()
        combined = f"{stderr}\n{stdout}"
        return any(
            token in combined
            for token in (
                "could not resolve hostname",
                "connection timed out",
                "connection refused",
                "no route to host",
                "network is unreachable",
                "connection reset by peer",
                "broken pipe",
                "permission denied (publickey",
            )
        )

    def run(
        self,
        command: str,
        timeout: int = 60,
        capture_output: bool = True,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command on the remote host.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            cwd: Working directory (defaults to host's work_dir)

        Returns:
            CompletedProcess with stdout, stderr, and return code
        """
        work_dir = cwd or self.host.work_directory
        venv_activate = (
            self._normalize_activate(self.host.venv_activate)
            if self.host.venv_activate
            else None
        )
        prefix = f"{venv_activate} && " if venv_activate else ""
        full_cmd = f"cd {work_dir} && {prefix}{command}"

        last_result: Optional[subprocess.CompletedProcess] = None
        for target in self.host.ssh_targets:
            ssh_cmd = self._build_ssh_cmd(target) + [full_cmd]
            try:
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                result = subprocess.CompletedProcess(
                    ssh_cmd,
                    returncode=255,
                    stdout=getattr(exc, "stdout", "") or "",
                    stderr=getattr(exc, "stderr", "") or "SSH timeout",
                )
            last_result = result
            if result.returncode == 0:
                return result
            if not self._is_ssh_connection_failure(result):
                return result

        return last_result or subprocess.CompletedProcess([], returncode=255, stdout="", stderr="SSH failed")

    def run_async(
        self,
        command: str,
        log_file: str,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously on the remote host (nohup + background).

        Args:
            command: Command to execute
            log_file: Path to log file on remote host
            cwd: Working directory (defaults to host's work_dir)

        Returns:
            CompletedProcess for the SSH connection (command runs in background)
        """
        work_dir = cwd or self.host.work_directory
        venv_activate = (
            self._normalize_activate(self.host.venv_activate)
            if self.host.venv_activate
            else None
        )
        prefix = f"{venv_activate} && " if venv_activate else ""
        # Use nohup to detach from SSH session
        full_cmd = f"cd {work_dir} && {prefix}nohup {command} > {log_file} 2>&1 &"
        last_result: Optional[subprocess.CompletedProcess] = None
        for target in self.host.ssh_targets:
            ssh_cmd = self._build_ssh_cmd(target) + [full_cmd]
            try:
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                result = subprocess.CompletedProcess(
                    ssh_cmd,
                    returncode=255,
                    stdout=getattr(exc, "stdout", "") or "",
                    stderr=getattr(exc, "stderr", "") or "SSH timeout",
                )
            last_result = result
            if result.returncode == 0:
                return result
            if not self._is_ssh_connection_failure(result):
                return result

        return last_result or subprocess.CompletedProcess([], returncode=255, stdout="", stderr="SSH failed")

    def scp_from(
        self,
        remote_path: str,
        local_path: str,
        *,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess:
        """Copy a file from the remote host to local filesystem via scp."""
        last_result: Optional[subprocess.CompletedProcess] = None
        for target in self.host.ssh_targets:
            cmd = ["scp", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
            if self.host.ssh_key:
                cmd.extend(["-i", self.host.ssh_key_path])
            if self.host.ssh_port and int(self.host.ssh_port) != 22:
                cmd.extend(["-P", str(int(self.host.ssh_port))])
            cmd.extend([f"{target}:{remote_path}", local_path])
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                result = subprocess.CompletedProcess(
                    cmd,
                    returncode=255,
                    stdout=getattr(exc, "stdout", "") or "",
                    stderr=getattr(exc, "stderr", "") or "SCP timeout",
                )
            last_result = result
            if result.returncode == 0:
                return result
            if not self._is_ssh_connection_failure(result):
                return result

        return last_result or subprocess.CompletedProcess([], returncode=255, stdout="", stderr="SCP failed")

    def get_process_memory(self, pattern: str) -> Optional[int]:
        """Get RSS memory usage in MB for processes matching a pattern.

        Args:
            pattern: Pattern to grep for in ps output

        Returns:
            RSS memory in MB, or None if no matching process
        """
        cmd = f"ps aux | grep '{pattern}' | grep -v grep | awk '{{sum += $6}} END {{print int(sum/1024)}}'"
        result = self.run(cmd, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            try:
                return int(result.stdout.strip())
            except ValueError:
                pass
        return None

    def is_alive(self) -> bool:
        """Check if the remote host is reachable."""
        try:
            result = self.run("echo ok", timeout=10)
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False


def get_ssh_executor(host_name: str) -> Optional[SSHExecutor]:
    """Get an SSHExecutor for a configured host.

    Args:
        host_name: Name of a configured remote host

    Returns:
        SSHExecutor or None if host not found
    """
    hosts = load_remote_hosts()
    if host_name in hosts:
        return SSHExecutor(hosts[host_name])
    return None
