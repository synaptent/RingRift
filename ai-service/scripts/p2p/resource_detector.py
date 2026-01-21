"""Resource Detector for P2P Orchestrator.

Extracted from p2p_orchestrator.py on December 26, 2025.

This module provides:
- GPU detection (NVIDIA and Apple MPS)
- Memory detection (Linux and macOS)
- Resource usage monitoring (CPU, memory, disk, GPU)
- NFS accessibility checking
- External work detection (CMA-ES, gauntlet, tournaments)

Usage as standalone:
    detector = ResourceDetector(ringrift_path="/path/to/ringrift")
    has_gpu, gpu_name = detector.detect_gpu()
    memory_gb = detector.detect_memory()
    usage = detector.get_resource_usage()

Usage as mixin (in P2POrchestrator):
    class P2POrchestrator(ResourceDetectorMixin, ...):
        pass
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_valid_ip(ip_str: str, family: str = "ipv4") -> bool:
    """Validate that a string looks like an IP address.

    Jan 7, 2026: Prevents error messages from being stored as IP addresses.
    The Tailscale CLI can print errors to stdout with exit code 0, which
    previously caused error messages to be treated as IP addresses.

    Args:
        ip_str: String to validate
        family: "ipv4" for 100.x.x.x Tailscale IPs, "ipv6" for fd7a:* IPs

    Returns:
        True if string appears to be a valid IP address
    """
    if not ip_str or len(ip_str) > 45:  # Max IPv6 length is 45 chars
        return False

    if family == "ipv4":
        # Tailscale IPv4 always starts with 100.
        if not ip_str.startswith("100."):
            return False
        parts = ip_str.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False
    elif family == "ipv6":
        # Tailscale IPv6 starts with fd7a:
        if not ip_str.startswith("fd7a:"):
            return False
        # Basic validation: contains colons, no spaces/letters other than hex
        if " " in ip_str or any(c.isalpha() and c.lower() not in "abcdef" for c in ip_str):
            return False
        return ":" in ip_str
    return False


# January 2026: Use centralized subprocess timeouts from loop_constants
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    SUBPROCESS_QUICK_TIMEOUT = LoopTimeouts.SUBPROCESS_QUICK  # 5.0
    SUBPROCESS_LONG_TIMEOUT = LoopTimeouts.SUBPROCESS_LONG  # 30.0
except ImportError:
    SUBPROCESS_QUICK_TIMEOUT = 5.0
    SUBPROCESS_LONG_TIMEOUT = 30.0


class ResourceDetector:
    """Standalone resource detector for system resources and capabilities.

    Features:
    - GPU detection (NVIDIA via nvidia-smi, Apple MPS)
    - Memory detection (platform-aware)
    - Real-time resource usage monitoring
    - NFS mount accessibility checking
    - External work detection (non-P2P processes)
    """

    def __init__(
        self,
        ringrift_path: Path | str | None = None,
        start_time: float | None = None,
        startup_grace_period: float = 30.0,
    ):
        """Initialize resource detector.

        Args:
            ringrift_path: Path to RingRift installation (for disk checks)
            start_time: Timestamp when orchestrator started (for grace period)
            startup_grace_period: Seconds to skip heavy I/O after startup
        """
        self.ringrift_path = Path(ringrift_path) if ringrift_path else Path.cwd()
        self.start_time = start_time or time.time()
        self.startup_grace_period = startup_grace_period

        # Cache detected values
        self._cached_gpu: tuple[bool, str] | None = None
        self._cached_memory: int | None = None

        # Jan 2026: Track Tailscale CLI health for event emission
        self._tailscale_cli_healthy: bool | None = None  # None = unknown, True = healthy, False = error
        self._tailscale_cli_error_emitted: bool = False

    def detect_gpu(self) -> tuple[bool, str]:
        """Detect if GPU is available and its name.

        Returns:
            Tuple of (has_gpu, gpu_name)
        """
        if self._cached_gpu is not None:
            return self._cached_gpu

        # Try NVIDIA GPU via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=int(SUBPROCESS_QUICK_TIMEOUT),
            )
            if result.returncode == 0 and result.stdout.strip():
                self._cached_gpu = (True, result.stdout.strip().split("\n")[0])
                return self._cached_gpu
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            OSError,
            KeyError,
            IndexError,
            AttributeError,
        ):
            pass

        # Try Apple MPS (Apple Silicon)
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True,
                text=True,
                timeout=int(SUBPROCESS_QUICK_TIMEOUT * 2),  # 10s - Python import takes longer
            )
            if "True" in result.stdout:
                self._cached_gpu = (True, "Apple MPS")
                return self._cached_gpu
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            ImportError,
        ):
            pass

        self._cached_gpu = (False, "")
        return self._cached_gpu

    def validate_pytorch_cuda(self) -> dict[str, Any]:
        """Validate PyTorch has CUDA support when GPU is present.

        Jan 9, 2026: Added to detect CPU-only PyTorch installations on GPU nodes.
        This caused lambda-gh200-10 to run CPU selfplay with 0% GPU utilization.

        Returns:
            dict with pytorch_cuda_available, cuda_version, device_count, warning
        """
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
            device_count = torch.cuda.device_count() if cuda_available else 0

            # Check for mismatch: GPU detected but PyTorch has no CUDA
            gpu_detected, gpu_name = self.detect_gpu()
            warning = None
            if gpu_detected and not cuda_available:
                warning = (
                    f"GPU detected ({gpu_name}) but PyTorch has no CUDA support. "
                    "Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128"
                )
                logger.warning(f"[ResourceDetector] {warning}")

            return {
                "pytorch_cuda_available": cuda_available,
                "pytorch_cuda_version": cuda_version,
                "cuda_device_count": device_count,
                "warning": warning,
            }
        except ImportError:
            return {"pytorch_cuda_available": False, "error": "PyTorch not installed"}

    def detect_memory(self) -> int:
        """Detect total system memory in GB.

        Returns:
            Total memory in GB (defaults to 16 if detection fails)
        """
        if self._cached_memory is not None:
            return self._cached_memory

        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self._cached_memory = int(result.stdout.strip()) // (1024**3)
                return self._cached_memory
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            self._cached_memory = int(line.split()[1]) // (1024**2)
                            return self._cached_memory
        except (OSError, ValueError):
            pass

        self._cached_memory = 16  # Default assumption
        return self._cached_memory

    def get_local_ip(self) -> str:
        """Get local IP address.

        Returns:
            Local IP address or "127.0.0.1" if detection fails
        """
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"

    def get_tailscale_ip(self, prefer_ipv6: bool = False) -> str:
        """Return this node's Tailscale IP address.

        Jan 2026: Added IPv6 support. Tailscale IPv6 (fd7a:115c:a1e0::*) bypasses
        NAT traversal issues and provides more reliable connectivity.

        Jan 14, 2026: Changed default to prefer_ipv6=False. IPv4 (100.x.x.x) is
        more widely compatible because many systems have issues resolving Tailscale
        IPv6 addresses (fd7a:...) causing sync failures.

        Args:
            prefer_ipv6: If True, return IPv6 when available. Default False.

        Returns:
            Tailscale IP if available, else empty string
        """
        # Jan 2, 2026: Delegate to specific methods that use _find_tailscale_binary()
        ipv4 = self.get_tailscale_ipv4()
        ipv6 = self.get_tailscale_ipv6()

        # Return based on preference (Jan 14, 2026: default to IPv4 for compatibility)
        if prefer_ipv6 and ipv6:
            return ipv6
        return ipv4 or ipv6 or ""

    @staticmethod
    def _find_tailscale_binary() -> str | None:
        """Find the tailscale binary path.

        Jan 2, 2026: Checks multiple locations to handle macOS and Linux.

        Returns:
            Path to tailscale binary or None if not found
        """
        # Candidate paths in order of preference
        candidates = [
            "tailscale",  # In PATH (Linux, some macOS setups)
            "/usr/local/bin/tailscale",  # Common Linux/macOS location
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",  # macOS app
        ]

        import shutil
        for candidate in candidates:
            # Check if it's in PATH
            if "/" not in candidate:
                path = shutil.which(candidate)
                if path:
                    return path
            elif os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return None

    def get_tailscale_ipv6(self) -> str:
        """Return this node's Tailscale IPv6 (fd7a:...) when available.

        Returns:
            Tailscale IPv6 if available, else empty string
        """
        binary = self._find_tailscale_binary()
        if not binary:
            return ""

        try:
            result = subprocess.run(
                [binary, "ip", "-6"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ""
            ip = (result.stdout or "").strip().splitlines()[0].strip()
            # Jan 7, 2026: Validate before returning to prevent error messages as IPs
            return ip if _is_valid_ip(ip, "ipv6") else ""
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            return ""

    def get_tailscale_ipv4(self) -> str:
        """Return this node's Tailscale IPv4 (100.x) when available.

        Jan 2026: Enhanced with health event emission for monitoring.
        Emits TAILSCALE_CLI_ERROR on failure, TAILSCALE_CLI_RECOVERED on recovery.

        Returns:
            Tailscale IPv4 if available, else empty string
        """
        binary = self._find_tailscale_binary()
        if not binary:
            self._emit_tailscale_health_event(success=False, error="tailscale binary not found")
            return ""

        try:
            result = subprocess.run(
                [binary, "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                # Check stderr for common Tailscale errors
                stderr = (result.stderr or "").strip()
                self._emit_tailscale_health_event(success=False, error=stderr or f"exit code {result.returncode}")
                return ""
            ip = (result.stdout or "").strip().splitlines()[0].strip()
            # Jan 7, 2026: Validate IP before returning to prevent error messages as IPs
            # The Tailscale CLI can print errors to stdout with exit code 0
            if not _is_valid_ip(ip, "ipv4"):
                error_msg = f"invalid output from tailscale ip -4: {ip[:100]!r}"
                logger.warning(f"[ResourceDetector] {error_msg}")
                self._emit_tailscale_health_event(success=False, error=error_msg)
                return ""
            # Successful IP retrieval - mark as healthy
            self._emit_tailscale_health_event(success=True)
            return ip
        except subprocess.TimeoutExpired as e:
            self._emit_tailscale_health_event(success=False, error=f"timeout after {e.timeout}s")
            return ""
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
            self._emit_tailscale_health_event(success=False, error=str(e))
            return ""

    def _emit_tailscale_health_event(self, success: bool, error: str | None = None) -> None:
        """Emit Tailscale CLI health events on state changes.

        Jan 2026: Prevents event flooding by only emitting on state transitions.

        Args:
            success: True if Tailscale CLI worked, False if it failed
            error: Optional error message for logging
        """
        # Only emit on state change
        if self._tailscale_cli_healthy is None:
            # First call - just record state, don't emit
            self._tailscale_cli_healthy = success
            if not success:
                logger.warning(f"[ResourceDetector] Tailscale CLI initial check failed: {error}")
            return

        if success and not self._tailscale_cli_healthy:
            # Recovered from error
            self._tailscale_cli_healthy = True
            self._tailscale_cli_error_emitted = False
            logger.info("[ResourceDetector] Tailscale CLI recovered")
            self._try_emit_event("TAILSCALE_CLI_RECOVERED", {"node_id": self._get_node_id()})
        elif not success and self._tailscale_cli_healthy:
            # Transitioned to error
            self._tailscale_cli_healthy = False
            if not self._tailscale_cli_error_emitted:
                self._tailscale_cli_error_emitted = True
                logger.warning(f"[ResourceDetector] Tailscale CLI error: {error}")
                self._try_emit_event("TAILSCALE_CLI_ERROR", {
                    "node_id": self._get_node_id(),
                    "error": error,
                })

    def _try_emit_event(self, event_name: str, payload: dict) -> None:
        """Try to emit event, gracefully handling missing dependencies.

        Args:
            event_name: Name of event to emit
            payload: Event payload dict
        """
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType
            event_type = getattr(DataEventType, event_name, None)
            if event_type:
                emit_event(event_type, payload)
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"[ResourceDetector] Could not emit {event_name}: {e}")

    def _get_node_id(self) -> str:
        """Get the current node ID for event emission."""
        return os.environ.get("RINGRIFT_NODE_ID", os.environ.get("HOSTNAME", "unknown"))

    def get_all_ips(self) -> set[str]:
        """Get all discoverable IP addresses for this node.

        Jan 2, 2026: Added for multi-IP advertising. Returns all IPs that peers
        could potentially use to reach this node:
        - Tailscale IPv4 (100.x.x.x) - mesh network
        - Tailscale IPv6 (fd7a:...) - mesh network, NAT-friendly
        - Local IP (from socket) - direct network
        - All interface IPs - fallback

        Returns:
            Set of IP addresses (excludes localhost/loopback)
        """
        ips: set[str] = set()

        # Tailscale IPs (highest priority - mesh reachable)
        ts_ipv4 = self.get_tailscale_ipv4()
        if ts_ipv4:
            ips.add(ts_ipv4)

        ts_ipv6 = self.get_tailscale_ipv6()
        if ts_ipv6:
            ips.add(ts_ipv6)

        # Local IP from socket (what we'd use for outbound)
        local_ip = self.get_local_ip()
        if local_ip and local_ip != "127.0.0.1":
            ips.add(local_ip)

        # All network interface IPs
        try:
            import socket
            hostname = socket.gethostname()
            # Get all IPs for this hostname
            for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
                addr = info[4][0]
                # Skip localhost/loopback
                if addr and not addr.startswith("127.") and addr != "::1":
                    ips.add(addr)
        except (OSError, socket.gaierror):
            pass

        # Try netifaces for more complete interface enumeration
        try:
            import netifaces
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                # IPv4
                for addr_info in addrs.get(netifaces.AF_INET, []):
                    addr = addr_info.get("addr", "")
                    if addr and not addr.startswith("127."):
                        ips.add(addr)
                # IPv6
                for addr_info in addrs.get(netifaces.AF_INET6, []):
                    addr = addr_info.get("addr", "").split("%")[0]  # Remove scope
                    if addr and addr != "::1" and not addr.startswith("fe80"):  # Skip link-local
                        ips.add(addr)
        except ImportError:
            pass  # netifaces not available
        except Exception:
            pass

        return ips

    def is_in_startup_grace_period(self) -> bool:
        """Check if we're still in the startup grace period.

        During this period, skip heavy I/O operations like JSONL scanning
        to ensure HTTP server remains responsive.

        Returns:
            True if still in grace period
        """
        return (time.time() - self.start_time) < self.startup_grace_period

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage.

        Returns:
            Dict with cpu_percent, memory_percent, disk_percent,
            gpu_percent, gpu_memory_percent, disk_free_gb
        """
        result: dict[str, float] = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "gpu_percent": 0.0,
            "gpu_memory_percent": 0.0,
        }

        try:
            # CPU
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["ps", "-A", "-o", "%cpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cpus = [float(x) for x in out.stdout.strip().split("\n")[1:] if x.strip()]
                cpu_count = os.cpu_count() or 1
                result["cpu_percent"] = min(100.0, sum(cpus) / cpu_count)
            else:
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                    cpu_count = os.cpu_count() or 1
                    result["cpu_percent"] = min(100.0, load * 100 / cpu_count)

            # Memory - use psutil for accurate cross-platform measurement
            # Jan 21, 2026: psutil correctly handles macOS memory pressure calculation
            # by including inactive/purgeable/cached pages as "available"
            try:
                import psutil
                mem = psutil.virtual_memory()
                result["memory_percent"] = mem.percent
            except ImportError:
                # Fallback when psutil not available
                if sys.platform == "darwin":
                    out = subprocess.run(
                        ["vm_stat"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    lines = out.stdout.strip().split("\n")
                    stats: dict[str, int] = {}
                    for line in lines[1:]:
                        if ":" in line:
                            key, val = line.split(":")
                            stats[key.strip()] = int(val.strip().rstrip("."))
                    page_size = 16384
                    # Include free + purgeable + inactive as available
                    free_pages = stats.get("Pages free", 0)
                    purgeable_pages = stats.get("Pages purgeable", 0)
                    inactive_pages = stats.get("Pages inactive", 0)
                    available = (free_pages + purgeable_pages + inactive_pages) * page_size
                    total = self.detect_memory() * (1024**3)
                    result["memory_percent"] = 100.0 * (1 - available / total) if total > 0 else 0.0
                else:
                    # Linux fallback
                    with open("/proc/meminfo") as f:
                        meminfo: dict[str, int] = {}
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                meminfo[parts[0].rstrip(":")] = int(parts[1])
                        total = meminfo.get("MemTotal", 1)
                        avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
                        result["memory_percent"] = 100.0 * (1 - avail / total)

            # Disk
            usage = shutil.disk_usage(self.ringrift_path)
            result["disk_percent"] = 100.0 * usage.used / usage.total
            result["disk_free_gb"] = usage.free / (1024**3)

            # GPU (NVIDIA) - handle multi-GPU by using max
            # Jan 2026: Check cached GPU detection first to avoid spinning nvidia-smi on non-GPU nodes
            has_nvidia, gpu_name = self.detect_gpu()
            if has_nvidia and "MPS" not in gpu_name:  # Only run nvidia-smi for NVIDIA GPUs
                try:
                    out = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0:
                        lines = out.stdout.strip().split("\n")
                        gpu_utils: list[float] = []
                        mem_percents: list[float] = []
                        for line in lines:
                            parts = line.strip().split(",")
                            if len(parts) >= 3:
                                try:
                                    gpu_utils.append(float(parts[0].strip()))
                                    mem_used = float(parts[1].strip())
                                    mem_total = float(parts[2].strip())
                                    if mem_total > 0:
                                        mem_percents.append(100.0 * mem_used / mem_total)
                                except (ValueError, IndexError):
                                    continue
                        if gpu_utils:
                            # Use max utilization across GPUs (more representative)
                            result["gpu_percent"] = max(gpu_utils)
                        if mem_percents:
                            result["gpu_memory_percent"] = max(mem_percents)
                except (ValueError, KeyError, IndexError, AttributeError, subprocess.TimeoutExpired, OSError):
                    # Silently ignore nvidia-smi errors
                    pass

        except Exception as e:
            logger.info(f"Resource check error: {e}")

        return result

    def check_nfs_accessible(self) -> bool:
        """Check if NFS mount is accessible.

        Tests common NFS mount points for accessibility.

        Can be disabled via RINGRIFT_SKIP_NFS_CHECK=1 environment variable
        for selfplay-only nodes that don't need NFS access.

        Returns:
            True if NFS is accessible, False otherwise
        """
        # Allow disabling NFS check for selfplay-only nodes
        skip_nfs = os.environ.get("RINGRIFT_SKIP_NFS_CHECK", "").lower()
        if skip_nfs in ("1", "true", "yes"):
            return True

        nfs_paths = [
            Path("/mnt/nfs/ringrift"),
            Path("/home/shared/ringrift"),
            Path(os.environ.get("RINGRIFT_NFS_PATH", "/mnt/nfs/ringrift")),
        ]

        for nfs_path in nfs_paths:
            try:
                if nfs_path.exists() and nfs_path.is_dir():
                    # Try to list directory (actual access check)
                    list(nfs_path.iterdir())[:1]
                    return True
            except (OSError, KeyError, IndexError, AttributeError):
                continue

        # NFS not found or not accessible
        return False

    def validate_gpu_capability(self, configured_role: str | None = None) -> dict[str, Any]:
        """Validate that configured role matches actual GPU presence.

        Jan 8, 2026: Added for runtime GPU validation. Detects mismatches between
        configured role (from distributed_hosts.yaml) and actual GPU presence.

        This helps catch:
        - GPU nodes where GPU has failed/disconnected
        - Misconfigured nodes marked as GPU but without one
        - VMs where GPU passthrough failed

        Args:
            configured_role: The role from config (e.g., "gpu_selfplay", "gpu_training_primary").
                            If None, attempts to read from RINGRIFT_NODE_ROLE environment variable.

        Returns:
            Dict with:
                - valid: bool - True if config matches reality
                - gpu_detected: bool - Whether a GPU was found
                - gpu_name: str - Name of detected GPU (empty if none)
                - configured_role: str - The role that was checked
                - fallback: str | None - Suggested fallback role if invalid
                - message: str - Human-readable status message
        """
        # Get configured role from argument or environment
        if configured_role is None:
            configured_role = os.environ.get("RINGRIFT_NODE_ROLE", "")

        # Detect actual GPU
        gpu_detected, gpu_name = self.detect_gpu()

        # Check if role expects GPU
        role_requires_gpu = any(term in configured_role.lower() for term in ["gpu", "training_primary", "cuda"])

        result: dict[str, Any] = {
            "valid": True,
            "gpu_detected": gpu_detected,
            "gpu_name": gpu_name,
            "configured_role": configured_role,
            "fallback": None,
            "message": "",
        }

        if role_requires_gpu and not gpu_detected:
            # Mismatch: role expects GPU but none found
            result["valid"] = False
            result["fallback"] = "cpu_selfplay"
            result["message"] = f"Role '{configured_role}' requires GPU but none detected"
            logger.warning(f"[ResourceDetector] {result['message']}")

            # Emit event for monitoring
            self._try_emit_event("GPU_CAPABILITY_MISMATCH", {
                "node_id": self._get_node_id(),
                "configured_role": configured_role,
                "gpu_detected": False,
                "gpu_name": "",
                "fallback": "cpu_selfplay",
            })
        elif not role_requires_gpu and gpu_detected:
            # Info: GPU available but role doesn't use it (not an error)
            result["message"] = f"GPU '{gpu_name}' available but role '{configured_role}' doesn't require it"
            logger.info(f"[ResourceDetector] {result['message']}")
        elif gpu_detected:
            result["message"] = f"GPU '{gpu_name}' available and role '{configured_role}' uses it"
        else:
            result["message"] = f"No GPU detected, role '{configured_role}' doesn't require it"

        # Jan 9, 2026: Add PyTorch CUDA validation
        pytorch_status = self.validate_pytorch_cuda()
        result["pytorch"] = pytorch_status

        if pytorch_status.get("warning"):
            result["warnings"] = result.get("warnings", []) + [pytorch_status["warning"]]
            # Emit event for PyTorch CUDA mismatch
            self._try_emit_event("PYTORCH_CUDA_MISMATCH", {
                "node_id": self._get_node_id(),
                "warning": pytorch_status["warning"],
                "gpu_detected": gpu_detected,
                "pytorch_cuda_available": False,
            })

        return result

    def detect_local_external_work(self) -> dict[str, bool]:
        """Detect external work running on this node (not tracked by P2P orchestrator).

        This detects:
        - CMA-ES optimization (HeuristicAI weight tuning)
        - Gauntlet runs (baseline or two-stage)
        - ELO tournaments
        - Data merge/aggregation jobs

        Returns:
            Dict with boolean flags for each work type
        """
        result = {
            "cmaes_running": False,
            "gauntlet_running": False,
            "tournament_running": False,
            "data_merge_running": False,
        }

        try:
            # Use pgrep to check for running processes (efficient)
            checks = [
                ("cmaes_running", "HeuristicAI.*json|cmaes_distributed|run_cpu_cmaes"),
                ("gauntlet_running", "baseline_gauntlet|two_stage_gauntlet"),
                ("tournament_running", "run_model_elo_tournament"),
                ("data_merge_running", "merge_game_dbs|aggregate_jsonl|export_training"),
            ]

            for key, pattern in checks:
                try:
                    proc = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    # pgrep returns 0 if matches found
                    result[key] = proc.returncode == 0 and proc.stdout.strip() != ""
                except (
                    subprocess.SubprocessError,
                    subprocess.TimeoutExpired,
                    OSError,
                    KeyError,
                    IndexError,
                    AttributeError,
                ):
                    pass

        except Exception as e:
            logger.debug(f"External work detection error: {e}")

        return result


class ResourceDetectorMixin:
    """Mixin class for adding resource detection to P2POrchestrator.

    This mixin provides backward-compatible method names for P2POrchestrator.
    The orchestrator should initialize self._resource_detector in __init__.

    Example:
        class P2POrchestrator(ResourceDetectorMixin, ...):
            def __init__(self, ...):
                ...
                self._resource_detector = ResourceDetector(
                    ringrift_path=self.ringrift_path,
                    start_time=self.start_time,
                    startup_grace_period=STARTUP_JSONL_GRACE_PERIOD_SECONDS,
                )
    """

    _resource_detector: ResourceDetector

    def _detect_gpu(self) -> tuple[bool, str]:
        """Detect GPU (delegates to ResourceDetector)."""
        return self._resource_detector.detect_gpu()

    def _detect_memory(self) -> int:
        """Detect memory (delegates to ResourceDetector)."""
        return self._resource_detector.detect_memory()

    def _get_local_ip(self) -> str:
        """Get local IP (delegates to ResourceDetector)."""
        return self._resource_detector.get_local_ip()

    def _get_tailscale_ip(self) -> str:
        """Get Tailscale IP (delegates to ResourceDetector)."""
        return self._resource_detector.get_tailscale_ip()

    def _get_all_ips(self) -> set[str]:
        """Get all discoverable IPs (delegates to ResourceDetector)."""
        return self._resource_detector.get_all_ips()

    def _is_in_startup_grace_period(self) -> bool:
        """Check startup grace period (delegates to ResourceDetector)."""
        return self._resource_detector.is_in_startup_grace_period()

    def _get_resource_usage(self) -> dict[str, Any]:
        """Get resource usage (delegates to ResourceDetector)."""
        return self._resource_detector.get_resource_usage()

    def _check_nfs_accessible(self) -> bool:
        """Check NFS accessibility (delegates to ResourceDetector)."""
        return self._resource_detector.check_nfs_accessible()

    def _detect_local_external_work(self) -> dict[str, bool]:
        """Detect external work (delegates to ResourceDetector)."""
        return self._resource_detector.detect_local_external_work()

    # Dec 30, 2025: Async versions to avoid blocking event loop
    # These wrap blocking subprocess calls in asyncio.to_thread()

    async def _get_resource_usage_async(self) -> dict[str, Any]:
        """Get resource usage without blocking event loop.

        Wraps get_resource_usage() in asyncio.to_thread() to avoid
        blocking the event loop with subprocess calls.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.get_resource_usage)

    async def _detect_gpu_async(self) -> tuple[bool, str]:
        """Detect GPU without blocking event loop.

        Wraps detect_gpu() in asyncio.to_thread() to avoid blocking
        the event loop with subprocess calls.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.detect_gpu)

    async def _detect_memory_async(self) -> int:
        """Detect memory without blocking event loop.

        Wraps detect_memory() in asyncio.to_thread() to avoid blocking
        the event loop with subprocess calls (macOS sysctl).
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.detect_memory)

    async def _get_tailscale_ip_async(self, prefer_ipv6: bool = True) -> str:
        """Get Tailscale IP without blocking event loop.

        Jan 7, 2026: Added for async safety in P2P orchestrator.
        Wraps get_tailscale_ip() in asyncio.to_thread() to avoid blocking
        the event loop with subprocess calls to tailscale CLI.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.get_tailscale_ip, prefer_ipv6)

    async def _get_tailscale_ipv4_async(self) -> str:
        """Get Tailscale IPv4 without blocking event loop.

        Jan 7, 2026: Added for async safety in P2P orchestrator.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.get_tailscale_ipv4)

    async def get_tailscale_ipv4_with_retry(
        self, max_retries: int = 5, delay: float = 2.0
    ) -> str | None:
        """Get Tailscale IPv4 with retry logic for startup timing issues.

        Jan 12, 2026: Added to fix IP advertisement timing issue where
        Tailscale daemon is not ready at P2P orchestrator startup, causing
        fallback to local IP (10.0.0.x) instead of Tailscale IP (100.x.x.x).

        Args:
            max_retries: Maximum number of attempts. Default 5 (10s total).
            delay: Seconds between retries. Default 2.0.

        Returns:
            Tailscale IPv4 address or None if not available after retries.
        """
        import asyncio

        for attempt in range(max_retries):
            ip = await self._get_tailscale_ipv4_async()
            if ip and not ip.startswith("10."):  # Not local network
                return ip
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)

        return None

    async def _get_tailscale_ipv6_async(self) -> str:
        """Get Tailscale IPv6 without blocking event loop.

        Jan 7, 2026: Added for async safety in P2P orchestrator.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.get_tailscale_ipv6)

    async def _get_all_ips_async(self) -> set[str]:
        """Get all discoverable IPs without blocking event loop.

        Jan 7, 2026: Added for async safety in P2P orchestrator.
        Wraps get_all_ips() in asyncio.to_thread() to avoid blocking
        the event loop with subprocess calls to tailscale CLI.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.get_all_ips)

    async def _detect_local_external_work_async(self) -> dict[str, bool]:
        """Detect external work without blocking event loop.

        Jan 7, 2026: Added for async safety in P2P orchestrator.
        Wraps detect_local_external_work() in asyncio.to_thread() to avoid
        blocking the event loop with pgrep subprocess calls.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.detect_local_external_work)

    def _validate_gpu_capability(self, configured_role: str | None = None) -> dict[str, Any]:
        """Validate GPU capability matches configured role (delegates to ResourceDetector).

        Jan 8, 2026: Added for runtime GPU validation.

        Args:
            configured_role: The role from config. If None, reads from environment.

        Returns:
            Dict with validation result, GPU info, and fallback suggestion.
        """
        return self._resource_detector.validate_gpu_capability(configured_role)

    async def _validate_gpu_capability_async(self, configured_role: str | None = None) -> dict[str, Any]:
        """Validate GPU capability without blocking event loop.

        Jan 8, 2026: Added for async safety in P2P orchestrator.
        Wraps validate_gpu_capability() in asyncio.to_thread() to avoid
        blocking the event loop with nvidia-smi subprocess calls.

        Args:
            configured_role: The role from config. If None, reads from environment.

        Returns:
            Dict with validation result, GPU info, and fallback suggestion.
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.validate_gpu_capability, configured_role)

    def _validate_pytorch_cuda(self) -> dict[str, Any]:
        """Validate PyTorch has CUDA support when GPU is present.

        Jan 9, 2026: Added to detect CPU-only PyTorch installations on GPU nodes.

        Returns:
            Dict with pytorch_cuda_available, cuda_version, device_count, warning
        """
        return self._resource_detector.validate_pytorch_cuda()

    async def _validate_pytorch_cuda_async(self) -> dict[str, Any]:
        """Validate PyTorch CUDA support without blocking event loop.

        Jan 9, 2026: Added for async safety in P2P orchestrator.
        Wraps validate_pytorch_cuda() in asyncio.to_thread() to avoid
        blocking the event loop with torch import and CUDA checks.

        Returns:
            Dict with pytorch_cuda_available, cuda_version, device_count, warning
        """
        import asyncio
        return await asyncio.to_thread(self._resource_detector.validate_pytorch_cuda)
