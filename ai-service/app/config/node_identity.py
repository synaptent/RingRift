"""Unified Node Identity System.

Single source of truth for node identification across the cluster.
Resolves identity using multiple sources with clear priority order.

Usage:
    from app.config.node_identity import NodeIdentity, get_node_identity

    # Get canonical node identity
    identity = get_node_identity()
    print(f"This node is: {identity.canonical_id}")
    print(f"Tailscale IP: {identity.tailscale_ip}")

    # Validate a peer's identity claim
    from app.config.node_identity import validate_identity_claim
    valid, reason = validate_identity_claim("lambda-gh200-1", {"100.69.164.58"})

Priority Order (Phase 1 of P2P Cluster Stability Plan):
    0. /etc/ringrift/node-id file (canonical, written by deployment)
    1. RINGRIFT_NODE_ID environment variable (explicit override)
    2. /etc/default/ringrift-p2p file (legacy compatibility)
    3. Hostname match against distributed_hosts.yaml
    4. Tailscale IP match against distributed_hosts.yaml

This module solves the identity mismatch problem where:
- socket.gethostname() returns "ip-192-222-57-210" (Lambda cloud hostname)
- Config expects "lambda-gh200-1" (human-readable name)
- P2P voters use config names, causing quorum failures
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "NodeIdentity",
    "IdentityError",
    "NodeIdentityError",  # Alias for clarity
    "get_node_identity",
    "get_tailscale_ip",
    "resolve_node_id",
    "validate_identity_claim",
    "clear_identity_cache",
    "get_node_id_safe",
    "NODE_ID_FILE",
    "LEGACY_P2P_CONFIG",
]

# Canonical node identity file - written by deployment scripts
# This is the single source of truth for node identification
NODE_ID_FILE = Path("/etc/ringrift/node-id")

# Legacy P2P config file (for backward compatibility)
LEGACY_P2P_CONFIG = Path("/etc/default/ringrift-p2p")

logger = logging.getLogger(__name__)


class IdentityError(Exception):
    """Raised when node identity cannot be resolved.

    This is a fail-fast error that should stop the process rather than
    allow it to run with an unknown identity that could cause data
    corruption or cluster instability.
    """

    pass


# Alias for clearer naming in imports
NodeIdentityError = IdentityError


def _read_node_id_file(path: Path) -> str | None:
    """Read node ID from a file, returning None if not available.

    Args:
        path: Path to the node ID file

    Returns:
        Node ID string or None if file doesn't exist or is invalid
    """
    try:
        if not path.exists():
            return None
        content = path.read_text().strip()
        # Validate: must be non-empty and not contain newlines
        if content and "\n" not in content:
            return content
        logger.debug(f"Invalid node ID content in {path}")
        return None
    except (OSError, PermissionError) as e:
        logger.debug(f"Cannot read {path}: {e}")
        return None


def _read_legacy_p2p_config() -> str | None:
    """Read NODE_ID from /etc/default/ringrift-p2p (shell format).

    Returns:
        Node ID string or None if not available
    """
    try:
        if not LEGACY_P2P_CONFIG.exists():
            return None
        for line in LEGACY_P2P_CONFIG.read_text().splitlines():
            line = line.strip()
            if line.startswith("NODE_ID="):
                # Handle both NODE_ID=value and NODE_ID="value"
                value = line.split("=", 1)[1].strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value:
                    return value
        return None
    except (OSError, PermissionError) as e:
        logger.debug(f"Cannot read {LEGACY_P2P_CONFIG}: {e}")
        return None


@dataclass
class NodeIdentity:
    """Canonical node identity - single source of truth.

    Attributes:
        canonical_id: Config key from distributed_hosts.yaml (authoritative)
        tailscale_ip: Tailscale VPN IP if available
        ssh_host: SSH host (may be public IP or Tailscale IP)
        hostname: System hostname from socket.gethostname()
        role: Node role from config (coordinator, gpu_training, etc.)
        resolution_method: How the identity was resolved (env_var, hostname, tailscale)
    """

    canonical_id: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    hostname: str = field(default_factory=socket.gethostname)
    role: str | None = None
    resolution_method: str = "unknown"

    @classmethod
    def resolve(cls, config: dict[str, Any] | None = None) -> NodeIdentity:
        """Resolve identity with fail-fast validation.

        Args:
            config: Optional pre-loaded config dict. If None, loads from YAML.

        Returns:
            Resolved NodeIdentity

        Raises:
            IdentityError: If identity cannot be resolved
        """
        # Load config if not provided
        if config is None:
            config = _load_hosts_config()

        hosts = config.get("hosts", {})
        hostname = socket.gethostname()

        # Priority 0: Canonical file (written by deployment)
        # This is the single source of truth
        file_id = _read_node_id_file(NODE_ID_FILE)
        if file_id:
            if file_id in hosts:
                logger.debug(f"Node ID resolved from {NODE_ID_FILE}: {file_id}")
                return cls._from_config(file_id, hosts, "canonical_file")
            else:
                # File exists but ID not in config - configuration mismatch
                logger.warning(
                    f"Node ID from {NODE_ID_FILE} ({file_id}) not in config, "
                    "trying other resolution methods"
                )

        # Priority 1: Explicit env var
        # When explicitly set, always use it (user intent is clear)
        env_id = os.environ.get("RINGRIFT_NODE_ID")
        if env_id:
            if env_id in hosts:
                return cls._from_config(env_id, hosts, "env_var")

            # Env var set but not in config - use it anyway (user explicitly set it)
            # January 2026: Changed to respect explicit env var over hostname matching
            ts_ip = get_tailscale_ip()
            logger.warning(
                f"RINGRIFT_NODE_ID={env_id} not in config, using it anyway"
            )
            return cls(
                canonical_id=env_id,
                tailscale_ip=ts_ip,
                hostname=hostname,
                resolution_method="env_var_unverified",
            )

        # Priority 2: Legacy P2P config file
        legacy_id = _read_legacy_p2p_config()
        if legacy_id:
            if legacy_id in hosts:
                logger.debug(f"Node ID resolved from {LEGACY_P2P_CONFIG}: {legacy_id}")
                return cls._from_config(legacy_id, hosts, "legacy_config")

        # Priority 2: Hostname match (direct config key match)
        if hostname in hosts:
            return cls._from_config(hostname, hosts, "hostname")

        # Priority 3: Hostname pattern match (e.g., "ip-192-222-57-210" patterns)
        for node_id, node_cfg in hosts.items():
            if isinstance(node_cfg, dict):
                # Check ssh_host match
                if node_cfg.get("ssh_host") == hostname:
                    return cls._from_config(node_id, hosts, "ssh_host")

        # Priority 4: Tailscale IP match
        ts_ip = get_tailscale_ip()
        if ts_ip:
            for node_id, node_cfg in hosts.items():
                if isinstance(node_cfg, dict):
                    if node_cfg.get("tailscale_ip") == ts_ip:
                        return cls._from_config(node_id, hosts, "tailscale")

        # Cannot resolve - fail fast with helpful message
        # Note: If RINGRIFT_NODE_ID was set, we would have returned earlier at Priority 1
        node_ids = list(hosts.keys()) if hosts else []
        raise IdentityError(
            f"Cannot resolve node identity.\n"
            f"Hostname: {hostname}\n"
            f"Tailscale IP: {ts_ip}\n"
            f"Known nodes: {node_ids[:10]}{'...' if len(node_ids) > 10 else ''}\n\n"
            f"Fix: Set RINGRIFT_NODE_ID to one of the known node IDs,\n"
            f"or add this node to distributed_hosts.yaml"
        )

    @classmethod
    def _from_config(
        cls, node_id: str, hosts: dict[str, Any], method: str
    ) -> NodeIdentity:
        """Create NodeIdentity from config entry."""
        node_cfg = hosts.get(node_id, {})
        if not isinstance(node_cfg, dict):
            node_cfg = {}

        return cls(
            canonical_id=node_id,
            tailscale_ip=node_cfg.get("tailscale_ip"),
            ssh_host=node_cfg.get("ssh_host"),
            hostname=socket.gethostname(),
            role=node_cfg.get("role"),
            resolution_method=method,
        )

    @classmethod
    def resolve_safe(cls, config: dict[str, Any] | None = None) -> NodeIdentity | None:
        """Resolve identity without raising exceptions.

        Returns None if resolution fails, allowing callers to handle gracefully.
        """
        try:
            return cls.resolve(config)
        except IdentityError as e:
            logger.error(f"Node identity resolution failed: {e}")
            return None

    def matches_peer(self, peer_info: dict[str, Any]) -> bool:
        """Check if this identity matches a peer info dict.

        Useful for matching P2P peers to voter configuration.

        Args:
            peer_info: Dict with 'addresses', 'node_id', etc.

        Returns:
            True if any identifier matches
        """
        # Check node_id match
        if peer_info.get("node_id") == self.canonical_id:
            return True

        # Check address match
        peer_addresses = set(peer_info.get("addresses", []))
        my_addresses = {self.tailscale_ip, self.ssh_host} - {None}

        return bool(peer_addresses & my_addresses)


def get_tailscale_ip() -> str | None:
    """Get the local Tailscale IP address.

    Returns:
        Tailscale IP (100.x.x.x) or None if not available
    """
    # Try multiple paths - subprocess can't see shell aliases
    tailscale_paths = [
        "tailscale",  # If in PATH
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale",  # macOS app
        "/usr/bin/tailscale",  # Linux system
        "/usr/local/bin/tailscale",  # Linux local
    ]

    for tailscale_cmd in tailscale_paths:
        try:
            result = subprocess.run(
                [tailscale_cmd, "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                break
        except FileNotFoundError:
            continue
    else:
        logger.debug("tailscale command not found in any known path")
        return None

    try:
        status = json.loads(result.stdout)
        self_info = status.get("Self", {})
        ips = self_info.get("TailscaleIPs", [])

        # Return first IPv4 (100.x.x.x)
        for ip in ips:
            if ip.startswith("100."):
                return ip

        return ips[0] if ips else None

    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse tailscale status: {e}")
        return None


def _load_hosts_config() -> dict[str, Any]:
    """Load distributed_hosts.yaml configuration.

    Returns:
        Config dict with 'hosts' key
    """
    # Try multiple locations
    search_paths = [
        Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml",
        Path.cwd() / "config" / "distributed_hosts.yaml",
        Path.home() / "ringrift" / "ai-service" / "config" / "distributed_hosts.yaml",
    ]

    # Also check env var override
    env_config = os.environ.get("RINGRIFT_CONFIG_PATH")
    if env_config:
        search_paths.insert(0, Path(env_config))

    for path in search_paths:
        if path.exists():
            try:
                import yaml

                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
                continue

    logger.warning("No distributed_hosts.yaml found")
    return {}


# Cached singleton accessor
_identity_cache: NodeIdentity | None = None


def get_node_identity(force_refresh: bool = False) -> NodeIdentity:
    """Get the canonical node identity (cached).

    Args:
        force_refresh: If True, re-resolve identity

    Returns:
        NodeIdentity for this node

    Raises:
        IdentityError: If identity cannot be resolved
    """
    global _identity_cache

    if _identity_cache is None or force_refresh:
        _identity_cache = NodeIdentity.resolve()

    return _identity_cache


def resolve_node_id() -> str:
    """Convenience function to get just the canonical node ID.

    This is the primary function to use when you just need the node ID string.

    Returns:
        Canonical node ID string

    Raises:
        IdentityError: If identity cannot be resolved
    """
    return get_node_identity().canonical_id


@lru_cache(maxsize=1)
def get_node_id_safe() -> str:
    """Get node ID with fallback to hostname if resolution fails.

    Use this for non-critical code paths where a fallback is acceptable.
    For critical code paths, use resolve_node_id() which will fail fast.

    Returns:
        Canonical node ID or hostname as fallback
    """
    identity = NodeIdentity.resolve_safe()
    if identity:
        return identity.canonical_id

    # Fallback to old behavior
    return os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())


def validate_identity_claim(
    claimed_id: str,
    claimed_ips: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate a peer's identity claim against the cluster configuration.

    This function is used to verify that a peer claiming to be a particular
    node actually has credentials (IP addresses) that match the expected
    configuration. This prevents impersonation and helps detect
    misconfigured nodes.

    Args:
        claimed_id: The node ID the peer claims to be
        claimed_ips: Set of IP addresses the peer is reachable from
        config: Optional pre-loaded config dict. If None, loads from YAML.

    Returns:
        Tuple of (valid: bool, reason: str)
        - If valid is True, reason is "OK"
        - If valid is False, reason explains the validation failure

    Examples:
        >>> valid, reason = validate_identity_claim(
        ...     "lambda-gh200-1",
        ...     {"100.69.164.58", "192.168.1.100"}
        ... )
        >>> if not valid:
        ...     logger.warning(f"Identity claim rejected: {reason}")
    """
    # Load config if not provided
    if config is None:
        config = _load_hosts_config()

    hosts = config.get("hosts", {})

    # Check if the claimed ID exists in config
    if claimed_id not in hosts:
        return False, f"Unknown node ID: {claimed_id}"

    node_cfg = hosts.get(claimed_id, {})
    if not isinstance(node_cfg, dict):
        return False, f"Invalid config for node: {claimed_id}"

    # If no IPs provided, we can't validate further
    if not claimed_ips:
        return True, "OK (no IP validation)"

    # Build expected IP set from config
    expected_ips: set[str] = set()

    if node_cfg.get("tailscale_ip"):
        expected_ips.add(node_cfg["tailscale_ip"])

    if node_cfg.get("ssh_host"):
        # ssh_host might be hostname or IP
        ssh_host = node_cfg["ssh_host"]
        # Only add if it looks like an IP
        if ssh_host.replace(".", "").isdigit():
            expected_ips.add(ssh_host)

    if node_cfg.get("public_ip"):
        expected_ips.add(node_cfg["public_ip"])

    # Check for IP overlap
    if not expected_ips:
        # No IPs in config to validate against
        return True, "OK (no expected IPs in config)"

    if not (claimed_ips & expected_ips):
        return False, (
            f"IP mismatch for {claimed_id}: "
            f"claimed {claimed_ips}, expected {expected_ips}"
        )

    return True, "OK"


def clear_identity_cache() -> None:
    """Clear the cached node identity.

    Call this after updating the node ID file or config to force re-resolution.
    """
    global _identity_cache
    _identity_cache = None
    get_node_id_safe.cache_clear()
