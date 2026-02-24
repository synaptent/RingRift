#!/usr/bin/env python3
"""Universal Keepalive Daemon - Keeps cluster nodes online across all platforms.

This script provides unified keepalive functionality for all node types:
- Vast.ai instances: Prevents idle termination, monitors workers
- Mac nodes: Prevents sleep, monitors Tailscale
- Lambda/cloud: Monitors connectivity, restarts services
- Hetzner/bare metal: Service monitoring

Features:
- Platform detection and appropriate keepalive methods
- P2P orchestrator health monitoring with TIERED RECOVERY
- Tailscale connectivity validation
- Automatic service restart on failure
- Heartbeat pings to prevent network timeouts
- Predictive health checks (disk, memory, GPU)
- Coordinator registration for centralized monitoring

Recovery Tiers:
    Tier 1: Soft restart (restart P2P process) - 60s cooldown, 3 attempts
    Tier 2: Service restart (systemctl restart) - 5min cooldown, 2 attempts
    Tier 3: Tailscale reset (restart tailscaled) - 10min cooldown, 1 attempt
    Tier 4: Human escalation (Slack alert) - 1hr cooldown

Usage:
    python scripts/universal_keepalive.py --node-id my-node

    # As systemd service or launchd daemon:
    python scripts/universal_keepalive.py --node-id my-node --daemon
"""
from __future__ import annotations


import argparse
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CHECK_INTERVAL = 30  # seconds between health checks
P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))
# Load from environment or leave empty; configure via RINGRIFT_LAMBDA_IPS
LAMBDA_IPS = os.environ.get("RINGRIFT_LAMBDA_IPS", "").split(",") if os.environ.get("RINGRIFT_LAMBDA_IPS") else []

# Notification settings
SLACK_WEBHOOK_URL = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
NOTIFICATION_COOLDOWN = 300  # 5 minutes between repeated alerts
_last_notification_time: dict[str, float] = {}

# Predictive health thresholds - aligned with app.config.thresholds
try:
    from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
    DISK_WARNING_PERCENT = DISK_SYNC_TARGET_PERCENT - 5  # 65
    DISK_CRITICAL_PERCENT = DISK_SYNC_TARGET_PERCENT  # 70
except ImportError:
    DISK_WARNING_PERCENT = 65
    DISK_CRITICAL_PERCENT = 70
MEMORY_WARNING_PERCENT = 75
MEMORY_CRITICAL_PERCENT = 85

# Escalation state file
ESCALATION_STATE_FILE = Path("/tmp/ringrift_escalation_state.json")


# =============================================================================
# Tiered Escalation System
# =============================================================================

class EscalationTier(IntEnum):
    """Recovery tier levels."""
    NONE = 0
    SOFT_RESTART = 1      # Restart P2P process
    SERVICE_RESTART = 2   # Restart systemd service
    TAILSCALE_RESET = 3   # Restart tailscaled
    HUMAN_ESCALATION = 4  # Alert humans


@dataclass
class TierConfig:
    """Configuration for a recovery tier."""
    name: str
    cooldown_seconds: int
    max_attempts: int
    severity: str = "warning"


# Recovery tier configurations
TIER_CONFIGS: dict[EscalationTier, TierConfig] = {
    EscalationTier.SOFT_RESTART: TierConfig("soft_restart", 60, 3, "warning"),
    EscalationTier.SERVICE_RESTART: TierConfig("service_restart", 300, 2, "warning"),
    EscalationTier.TAILSCALE_RESET: TierConfig("tailscale_reset", 600, 1, "error"),
    EscalationTier.HUMAN_ESCALATION: TierConfig("human_escalation", 3600, 1, "error"),
}


@dataclass
class EscalationState:
    """Current escalation state for a node."""
    issue_type: str = ""
    current_tier: EscalationTier = EscalationTier.NONE
    tier_attempts: dict[int, int] = field(default_factory=dict)
    tier_last_attempt: dict[int, float] = field(default_factory=dict)
    first_failure_time: float = 0.0
    last_success_time: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "issue_type": self.issue_type,
            "current_tier": int(self.current_tier),
            "tier_attempts": {str(k): v for k, v in self.tier_attempts.items()},
            "tier_last_attempt": {str(k): v for k, v in self.tier_last_attempt.items()},
            "first_failure_time": self.first_failure_time,
            "last_success_time": self.last_success_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationState":
        """Deserialize from dict."""
        return cls(
            issue_type=data.get("issue_type", ""),
            current_tier=EscalationTier(data.get("current_tier", 0)),
            tier_attempts={int(k): v for k, v in data.get("tier_attempts", {}).items()},
            tier_last_attempt={int(k): v for k, v in data.get("tier_last_attempt", {}).items()},
            first_failure_time=data.get("first_failure_time", 0.0),
            last_success_time=data.get("last_success_time", 0.0),
        )


@dataclass
class PredictiveHealthIssue:
    """A predicted health issue."""
    issue_type: str  # disk_filling, memory_pressure, gpu_error
    severity: str    # warning, error, critical
    value: float     # Current value (e.g., 67% disk usage)
    threshold: float # Threshold that was exceeded
    message: str


class EscalationManager:
    """Manages tiered escalation with cooldowns and state persistence."""

    def __init__(self, node_id: str, state_file: Path = ESCALATION_STATE_FILE):
        self.node_id = node_id
        self.state_file = state_file
        self.states: dict[str, EscalationState] = {}
        self._load_state()

    def _load_state(self):
        """Load escalation state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    for issue_type, state_data in data.items():
                        self.states[issue_type] = EscalationState.from_dict(state_data)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load escalation state: {e}")

    def _save_state(self):
        """Save escalation state to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.states.items()}
            with open(self.state_file, "w") as f:
                json.dump(data, f)
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save escalation state: {e}")

    def get_state(self, issue_type: str) -> EscalationState:
        """Get or create escalation state for an issue."""
        if issue_type not in self.states:
            self.states[issue_type] = EscalationState(issue_type=issue_type)
        return self.states[issue_type]

    def should_escalate(self, issue_type: str) -> tuple[EscalationTier, bool]:
        """Check if we should escalate and to which tier.

        Returns:
            (tier, should_act): tier to escalate to, and whether to take action
        """
        state = self.get_state(issue_type)
        now = time.time()

        # Record first failure if not set
        if state.first_failure_time == 0:
            state.first_failure_time = now

        # Determine next tier
        next_tier = EscalationTier(min(int(state.current_tier) + 1, 4))

        # Check if we've exhausted current tier
        current_config = TIER_CONFIGS.get(state.current_tier)
        if current_config:
            attempts = state.tier_attempts.get(int(state.current_tier), 0)
            if attempts >= current_config.max_attempts:
                # Move to next tier
                state.current_tier = next_tier
                logger.info(f"Escalating {issue_type} to tier {next_tier.name}")

        # If at tier 0, start at tier 1
        if state.current_tier == EscalationTier.NONE:
            state.current_tier = EscalationTier.SOFT_RESTART

        # Check cooldown for current tier
        config = TIER_CONFIGS.get(state.current_tier)
        if not config:
            return state.current_tier, False

        last_attempt = state.tier_last_attempt.get(int(state.current_tier), 0)
        if now - last_attempt < config.cooldown_seconds:
            # Still in cooldown
            return state.current_tier, False

        # Can take action
        state.tier_attempts[int(state.current_tier)] = state.tier_attempts.get(int(state.current_tier), 0) + 1
        state.tier_last_attempt[int(state.current_tier)] = now
        self._save_state()

        return state.current_tier, True

    def mark_resolved(self, issue_type: str):
        """Mark an issue as resolved, resetting escalation state."""
        if issue_type in self.states:
            state = self.states[issue_type]
            state.last_success_time = time.time()
            # Reset for next failure
            state.current_tier = EscalationTier.NONE
            state.tier_attempts.clear()
            state.tier_last_attempt.clear()
            state.first_failure_time = 0.0
            self._save_state()
            logger.info(f"Issue {issue_type} resolved, resetting escalation state")

    def get_escalation_summary(self) -> dict[str, Any]:
        """Get summary of all escalation states."""
        return {
            issue_type: {
                "tier": state.current_tier.name,
                "attempts": state.tier_attempts,
                "duration_seconds": time.time() - state.first_failure_time if state.first_failure_time else 0,
            }
            for issue_type, state in self.states.items()
            if state.current_tier != EscalationTier.NONE
        }


def send_notification(event_type: str, message: str, node_id: str, severity: str = "warning") -> bool:
    """Send notification via Slack webhook.

    Args:
        event_type: Type of event (p2p_down, tailscale_down, node_offline, etc.)
        message: Human-readable message
        node_id: Affected node identifier
        severity: warning, error, or info

    Returns:
        True if notification sent successfully
    """
    global _last_notification_time

    # Check cooldown
    cache_key = f"{node_id}:{event_type}"
    now = time.time()
    if cache_key in _last_notification_time and now - _last_notification_time[cache_key] < NOTIFICATION_COOLDOWN:
        return False  # Skip due to cooldown

    if not SLACK_WEBHOOK_URL:
        logger.debug(f"No Slack webhook configured, skipping notification: {message}")
        return False

    # Color based on severity
    colors = {"error": "#ff0000", "warning": "#ffaa00", "info": "#00aa00"}
    color = colors.get(severity, "#808080")

    # Build Slack payload
    payload = {
        "attachments": [{
            "color": color,
            "title": f"RingRift Cluster Alert: {event_type}",
            "text": message,
            "fields": [
                {"title": "Node", "value": node_id, "short": True},
                {"title": "Severity", "value": severity.upper(), "short": True},
                {"title": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
            ],
            "footer": "RingRift Keepalive Daemon"
        }]
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                _last_notification_time[cache_key] = now
                logger.info(f"Sent notification: {event_type}")
                return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.error(f"Failed to send notification: {e}")

    return False


def send_recovery_notification(event_type: str, node_id: str) -> bool:
    """Send recovery notification when an issue is resolved."""
    message = f"Node {node_id} has recovered from {event_type}"
    return send_notification(f"{event_type}_recovered", message, node_id, severity="info")


class NodeType:
    """Detected node types."""
    MAC = "mac"
    VAST = "vast"
    LAMBDA = "lambda"
    HETZNER = "hetzner"
    LINUX = "linux"
    UNKNOWN = "unknown"


# Hetzner CPU node IP ranges (Tailscale)
HETZNER_IP_PREFIXES = ["100.94.174.", "100.67.131.", "100.126.21."]


def detect_node_type() -> str:
    """Detect what type of node we're running on."""
    system = platform.system().lower()

    if system == "darwin":
        return NodeType.MAC

    # Check for Vast.ai indicators
    if os.path.exists("/workspace") or "vast" in socket.gethostname().lower():
        return NodeType.VAST

    # Check for Lambda indicators
    hostname = socket.gethostname().lower()
    if "lambda" in hostname or "gh200" in hostname or "h100" in hostname or "a10" in hostname:
        return NodeType.LAMBDA

    # Check for Hetzner indicators
    if "hetzner" in hostname or "cpu" in hostname:
        return NodeType.HETZNER

    # Check Tailscale IP for Hetzner
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ts_ip = result.stdout.strip()
            for prefix in HETZNER_IP_PREFIXES:
                if ts_ip.startswith(prefix):
                    return NodeType.HETZNER
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if system == "linux":
        return NodeType.LINUX

    return NodeType.UNKNOWN


def get_ai_service_root() -> Path:
    """Get the AI service root directory."""
    # Try common locations
    candidates = [
        Path(__file__).parent.parent,  # Relative to script
        Path.home() / "ringrift" / "ai-service",
        Path.home() / "Development" / "RingRift" / "ai-service",
        Path("/workspace/ringrift/ai-service"),
    ]

    for path in candidates:
        if path.exists() and (path / "scripts").exists():
            return path

    return Path(__file__).parent.parent


class UniversalKeepalive:
    """Universal keepalive daemon for all node types."""

    def __init__(self, node_id: str, daemon_mode: bool = False):
        self.node_id = node_id
        self.daemon_mode = daemon_mode
        self.node_type = detect_node_type()
        self.ai_service_root = get_ai_service_root()
        self.running = True
        self.caffeinate_pid: int | None = None

        # Tiered escalation manager
        self.escalation_manager = EscalationManager(node_id)

        # Track previous health state for recovery detection
        self._prev_health: dict[str, bool] = {}

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(f"Initialized keepalive for {node_id} (type: {self.node_type})")
        logger.info(f"AI service root: {self.ai_service_root}")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def check_p2p_health(self) -> tuple[bool, dict | None]:
        """Check if P2P orchestrator is healthy."""
        try:
            url = f"http://localhost:{P2P_PORT}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return True, data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.debug(f"P2P health check failed: {e}")
            return False, None

    def is_p2p_running(self) -> bool:
        """Check if P2P orchestrator process is running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "p2p_orchestrator"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def start_p2p(self) -> bool:
        """Start the P2P orchestrator."""
        logger.info("Starting P2P orchestrator...")

        try:
            # Check for systemd service first
            if self.node_type in [NodeType.VAST, NodeType.LAMBDA, NodeType.LINUX]:
                result = subprocess.run(
                    ["systemctl", "is-enabled", "ringrift-p2p.service"],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["sudo", "systemctl", "restart", "ringrift-p2p.service"],
                        timeout=30
                    )
                    logger.info("Restarted P2P via systemctl")
                    return True

            # Fallback to direct start
            python_cmd = sys.executable
            p2p_script = self.ai_service_root / "scripts" / "p2p_orchestrator.py"

            if not p2p_script.exists():
                logger.error(f"P2P script not found: {p2p_script}")
                return False

            log_file = self.ai_service_root / "logs" / "p2p_orchestrator.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log:
                subprocess.Popen(
                    [python_cmd, str(p2p_script), "--node-id", self.node_id],
                    cwd=str(self.ai_service_root),
                    stdout=log,
                    stderr=log,
                    start_new_session=True
                )

            logger.info(f"Started P2P orchestrator with node-id: {self.node_id}")
            return True

        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to start P2P: {e}")
            return False

    def check_tailscale(self) -> bool:
        """Check Tailscale connectivity."""
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False

            data = json.loads(result.stdout)
            return data.get("BackendState") == "Running"
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError) as e:
            logger.debug(f"Tailscale check failed: {e}")
            return False

    def restart_tailscale(self) -> bool:
        """Attempt to restart Tailscale."""
        logger.warning("Attempting to restart Tailscale...")
        try:
            if self.node_type == NodeType.MAC:
                subprocess.run(
                    ["sudo", "launchctl", "kickstart", "-k", "system/com.tailscale.tailscaled"],
                    timeout=30
                )
            else:
                subprocess.run(["sudo", "systemctl", "restart", "tailscaled"], timeout=30)
            return True
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to restart Tailscale: {e}")
            return False

    def restart_p2p_service(self) -> bool:
        """Restart P2P via systemctl (Tier 2 recovery)."""
        logger.warning("Attempting systemctl restart of P2P service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "ringrift-p2p.service"],
                timeout=30
            )
            return True
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to restart P2P service: {e}")
            return False

    # =========================================================================
    # Predictive Health Checks
    # =========================================================================

    def get_disk_usage(self) -> float:
        """Get disk usage percentage for the data directory."""
        try:
            data_dir = self.ai_service_root / "data"
            if not data_dir.exists():
                data_dir = self.ai_service_root

            usage = shutil.disk_usage(data_dir)
            return (usage.used / usage.total) * 100
        except (OSError, FileNotFoundError) as e:
            logger.debug(f"Failed to get disk usage: {e}")
            return 0.0

    def get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            if self.node_type == NodeType.MAC:
                # macOS uses vm_stat
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse vm_stat output
                    lines = result.stdout.strip().split('\n')
                    stats = {}
                    for line in lines[1:]:
                        if ':' in line:
                            key, val = line.split(':')
                            val = val.strip().rstrip('.')
                            try:
                                stats[key.strip()] = int(val)
                            except ValueError:
                                pass

                    page_size = 16384  # Default for Apple Silicon
                    free_pages = stats.get('Pages free', 0)
                    active_pages = stats.get('Pages active', 0)
                    inactive_pages = stats.get('Pages inactive', 0)
                    wired_pages = stats.get('Pages wired down', 0)
                    total = free_pages + active_pages + inactive_pages + wired_pages
                    if total > 0:
                        used = active_pages + wired_pages
                        return (used / total) * 100
            else:
                # Linux uses /proc/meminfo
                with open('/proc/meminfo') as f:
                    mem_info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            mem_info[parts[0].rstrip(':')] = int(parts[1])

                total = mem_info.get('MemTotal', 1)
                available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
                used = total - available
                return (used / total) * 100
        except (OSError, FileNotFoundError, ValueError, subprocess.SubprocessError) as e:
            logger.debug(f"Failed to get memory usage: {e}")
        return 0.0

    def check_gpu_errors(self) -> list[str]:
        """Check for GPU errors (CUDA/MPS)."""
        errors = []
        try:
            if self.node_type == NodeType.MAC:
                # MPS doesn't have a standard error check
                pass
            else:
                # Check nvidia-smi for GPU errors
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=gpu_name,ecc.errors.corrected.volatile.total",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                ecc_errors = parts[1].strip()
                                if ecc_errors not in ['0', '[N/A]', '']:
                                    try:
                                        if int(ecc_errors) > 100:
                                            errors.append(f"High ECC errors: {ecc_errors}")
                                    except ValueError:
                                        pass
        except FileNotFoundError:
            pass  # nvidia-smi not available
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug(f"Failed to check GPU: {e}")
        return errors

    def check_predictive_health(self) -> list[PredictiveHealthIssue]:
        """Check for impending failures before they happen."""
        issues = []

        # Check disk
        disk_usage = self.get_disk_usage()
        if disk_usage >= DISK_CRITICAL_PERCENT:
            issues.append(PredictiveHealthIssue(
                issue_type="disk_critical",
                severity="error",
                value=disk_usage,
                threshold=DISK_CRITICAL_PERCENT,
                message=f"Disk usage critical: {disk_usage:.1f}% (threshold: {DISK_CRITICAL_PERCENT}%)"
            ))
        elif disk_usage >= DISK_WARNING_PERCENT:
            issues.append(PredictiveHealthIssue(
                issue_type="disk_warning",
                severity="warning",
                value=disk_usage,
                threshold=DISK_WARNING_PERCENT,
                message=f"Disk usage high: {disk_usage:.1f}% (threshold: {DISK_WARNING_PERCENT}%)"
            ))

        # Check memory
        memory_usage = self.get_memory_usage()
        if memory_usage >= MEMORY_CRITICAL_PERCENT:
            issues.append(PredictiveHealthIssue(
                issue_type="memory_critical",
                severity="error",
                value=memory_usage,
                threshold=MEMORY_CRITICAL_PERCENT,
                message=f"Memory usage critical: {memory_usage:.1f}% (threshold: {MEMORY_CRITICAL_PERCENT}%)"
            ))
        elif memory_usage >= MEMORY_WARNING_PERCENT:
            issues.append(PredictiveHealthIssue(
                issue_type="memory_warning",
                severity="warning",
                value=memory_usage,
                threshold=MEMORY_WARNING_PERCENT,
                message=f"Memory usage high: {memory_usage:.1f}% (threshold: {MEMORY_WARNING_PERCENT}%)"
            ))

        # Check GPU
        gpu_errors = self.check_gpu_errors()
        for error in gpu_errors:
            issues.append(PredictiveHealthIssue(
                issue_type="gpu_error",
                severity="warning",
                value=0,
                threshold=0,
                message=error
            ))

        return issues

    def cleanup_disk(self) -> bool:
        """Attempt to free disk space by cleaning up temporary files."""
        logger.info("Attempting disk cleanup...")
        freed_bytes = 0

        try:
            # Clean up old logs (>30 days)
            logs_dir = self.ai_service_root / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        if time.time() - log_file.stat().st_mtime > 30 * 86400:
                            size = log_file.stat().st_size
                            log_file.unlink()
                            freed_bytes += size
                    except (OSError, FileNotFoundError):
                        pass

            # Clean up /tmp keepalive files (>1 day)
            for f in Path("/tmp").glob("keepalive_*"):
                try:
                    if time.time() - f.stat().st_mtime > 86400:
                        size = f.stat().st_size
                        f.unlink()
                        freed_bytes += size
                except (OSError, FileNotFoundError):
                    pass

            # Clean up fallback selfplay data (always safe)
            fallback_dir = self.ai_service_root / "data" / "selfplay" / "fallback"
            if fallback_dir.exists():
                for db_file in fallback_dir.glob("*.db"):
                    try:
                        size = db_file.stat().st_size
                        db_file.unlink()
                        freed_bytes += size
                    except (OSError, FileNotFoundError):
                        pass

            logger.info(f"Disk cleanup freed {freed_bytes / (1024*1024):.1f} MB")
            return freed_bytes > 0

        except (OSError, FileNotFoundError) as e:
            logger.error(f"Disk cleanup failed: {e}")
            return False

    # =========================================================================
    # Tiered Recovery Actions
    # =========================================================================

    def execute_recovery(self, issue_type: str, tier: EscalationTier) -> bool:
        """Execute recovery action for the given tier.

        Returns:
            True if recovery action was taken (may or may not have succeeded)
        """
        config = TIER_CONFIGS.get(tier)
        if not config:
            return False

        logger.info(f"Executing {tier.name} recovery for {issue_type}")

        if tier == EscalationTier.SOFT_RESTART:
            # Kill and restart P2P process
            try:
                subprocess.run(["pkill", "-f", "p2p_orchestrator"], timeout=5)
                time.sleep(2)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
            return self.start_p2p()

        elif tier == EscalationTier.SERVICE_RESTART:
            # Restart via systemctl
            return self.restart_p2p_service()

        elif tier == EscalationTier.TAILSCALE_RESET:
            # Reset Tailscale
            return self.restart_tailscale()

        elif tier == EscalationTier.HUMAN_ESCALATION:
            # Send alert to humans
            state = self.escalation_manager.get_state(issue_type)
            duration = time.time() - state.first_failure_time
            send_notification(
                f"{issue_type}_escalation",
                f"ESCALATION: {issue_type} on {self.node_id} unresolved after {duration/60:.1f} minutes. "
                f"Automatic recovery failed. Manual intervention required.",
                self.node_id,
                severity="error"
            )
            return True

        return False

    def send_keepalive_ping(self) -> bool:
        """Send keepalive ping to maintain connectivity."""
        for ip in LAMBDA_IPS:
            try:
                # Quick ping to keep network alive
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", ip],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return False

    # Mac-specific methods
    def mac_prevent_sleep(self):
        """Prevent Mac from sleeping (Mac only)."""
        if self.node_type != NodeType.MAC:
            return

        if self.caffeinate_pid is not None:
            # Check if still running
            try:
                os.kill(self.caffeinate_pid, 0)
                return  # Still running
            except OSError:
                self.caffeinate_pid = None

        try:
            # Start caffeinate in background
            proc = subprocess.Popen(
                ["caffeinate", "-dis"],  # Prevent display, idle, and system sleep
                start_new_session=True
            )
            self.caffeinate_pid = proc.pid
            logger.info(f"Started caffeinate (PID: {self.caffeinate_pid})")
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to start caffeinate: {e}")

    def mac_cleanup(self):
        """Clean up Mac-specific resources."""
        if self.caffeinate_pid:
            try:
                os.kill(self.caffeinate_pid, signal.SIGTERM)
                logger.info("Stopped caffeinate")
            except OSError:
                pass

    # Vast-specific methods
    def vast_send_keepalive(self):
        """Send keepalive to prevent Vast.ai idle termination."""
        if self.node_type != NodeType.VAST:
            return

        try:
            # Write keepalive file (Vast monitors for activity)
            keepalive_file = Path("/tmp") / f"keepalive_{int(time.time())}"
            keepalive_file.write_text("alive")

            # Clean up old keepalive files
            for f in Path("/tmp").glob("keepalive_*"):
                try:
                    if f != keepalive_file and time.time() - f.stat().st_mtime > 300:
                        f.unlink()
                except (OSError, FileNotFoundError):
                    pass
        except (OSError, FileNotFoundError) as e:
            logger.debug(f"Vast keepalive write failed: {e}")

    def run_health_check(self) -> dict[str, bool]:
        """Run all health checks and return status."""
        status = {
            "p2p_running": False,
            "p2p_healthy": False,
            "tailscale_ok": False,
            "network_ok": False,
        }

        # Check P2P
        status["p2p_running"] = self.is_p2p_running()
        if status["p2p_running"]:
            healthy, _ = self.check_p2p_health()
            status["p2p_healthy"] = healthy

        # Check Tailscale
        status["tailscale_ok"] = self.check_tailscale()

        # Check network
        status["network_ok"] = self.send_keepalive_ping()

        return status

    def run(self):
        """Main keepalive loop with tiered escalation."""
        logger.info(f"Starting keepalive daemon (interval: {CHECK_INTERVAL}s)")
        logger.info(f"Node type: {self.node_type}, Escalation enabled")

        while self.running:
            try:
                # Platform-specific pre-checks
                if self.node_type == NodeType.MAC:
                    self.mac_prevent_sleep()
                elif self.node_type == NodeType.VAST:
                    self.vast_send_keepalive()

                # Run health checks
                status = self.run_health_check()

                # Log status periodically
                if not all(status.values()):
                    logger.warning(f"Health check: {status}")
                else:
                    logger.debug("Health check: all OK")

                # Check predictive health (disk, memory, GPU)
                predictive_issues = self.check_predictive_health()
                for issue in predictive_issues:
                    logger.warning(issue.message)
                    send_notification(
                        issue.issue_type,
                        issue.message,
                        self.node_id,
                        severity=issue.severity
                    )
                    # Auto-cleanup for disk issues
                    if issue.issue_type in ("disk_warning", "disk_critical"):
                        self.cleanup_disk()

                # Handle P2P issues with tiered escalation
                if not status["p2p_running"] or not status["p2p_healthy"]:
                    issue_type = "p2p_down" if not status["p2p_running"] else "p2p_unhealthy"
                    tier, should_act = self.escalation_manager.should_escalate(issue_type)

                    if should_act:
                        logger.info(f"Escalation tier {tier.name} for {issue_type}")
                        self.execute_recovery(issue_type, tier)

                    self._prev_health["p2p"] = False
                else:
                    # P2P is healthy - mark resolved if it was down
                    if self._prev_health.get("p2p") is False:
                        self.escalation_manager.mark_resolved("p2p_down")
                        self.escalation_manager.mark_resolved("p2p_unhealthy")
                        send_recovery_notification("p2p_down", self.node_id)
                    self._prev_health["p2p"] = True

                # Handle Tailscale issues with escalation
                if not status["tailscale_ok"]:
                    tier, should_act = self.escalation_manager.should_escalate("tailscale_down")
                    if should_act:
                        if tier == EscalationTier.TAILSCALE_RESET or tier.value >= 3:
                            self.restart_tailscale()
                        elif tier == EscalationTier.HUMAN_ESCALATION:
                            self.execute_recovery("tailscale_down", tier)

                    if self._prev_health.get("tailscale") is not False:
                        send_notification(
                            "tailscale_down",
                            f"Tailscale connectivity lost on {self.node_id}",
                            self.node_id,
                            severity="error"
                        )
                    self._prev_health["tailscale"] = False
                else:
                    if self._prev_health.get("tailscale") is False:
                        self.escalation_manager.mark_resolved("tailscale_down")
                        send_recovery_notification("tailscale_down", self.node_id)
                    self._prev_health["tailscale"] = True

                # Handle network issues (informational, no escalation)
                if not status["network_ok"]:
                    if self._prev_health.get("network") is not False:
                        send_notification(
                            "network_unreachable",
                            f"Network connectivity issues on {self.node_id}",
                            self.node_id,
                            severity="warning"
                        )
                    self._prev_health["network"] = False
                else:
                    if self._prev_health.get("network") is False:
                        send_recovery_notification("network_unreachable", self.node_id)
                    self._prev_health["network"] = True

                # Log escalation summary if any issues are being escalated
                summary = self.escalation_manager.get_escalation_summary()
                if summary:
                    logger.info(f"Active escalations: {summary}")

            except (OSError, FileNotFoundError, subprocess.SubprocessError, urllib.error.URLError) as e:
                logger.error(f"Health check error: {e}")
                send_notification(
                    "keepalive_error",
                    f"Keepalive daemon error on {self.node_id}: {e!s}",
                    self.node_id,
                    severity="error"
                )

            # Sleep until next check
            time.sleep(CHECK_INTERVAL)

        # Cleanup
        if self.node_type == NodeType.MAC:
            self.mac_cleanup()

        logger.info("Keepalive daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="Universal Keepalive Daemon")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args = parser.parse_args()

    global CHECK_INTERVAL
    CHECK_INTERVAL = args.interval

    keepalive = UniversalKeepalive(args.node_id, args.daemon)
    keepalive.run()


if __name__ == "__main__":
    main()
