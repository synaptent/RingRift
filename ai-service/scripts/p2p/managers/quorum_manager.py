"""Quorum Manager - Voter quorum management for P2P leader election.

Jan 2026: Extracted from p2p_orchestrator.py as part of Phase 2 modularization.
Handles voter configuration loading, IP-to-node mapping, and voter health tracking.

Key responsibilities:
- Load voter node IDs from config (env var, YAML)
- Build IP-to-node and voter-to-IP mappings
- Count alive voters for quorum decisions
- Check voter health and emit alerts
- Handle voter set adoption from gossip

Usage:
    from scripts.p2p.managers.quorum_manager import QuorumManager, QuorumConfig

    manager = QuorumManager(
        config=QuorumConfig(
            node_id="my-node",
            config_path=Path("config/distributed_hosts.yaml"),
        ),
        get_peers=lambda: peers_dict,
        get_peers_lock=lambda: peers_lock,
    )

    # Load voters at startup
    voters = manager.load_voter_node_ids()

    # Check voter health
    health = manager.check_voter_health(voter_quorum_size=4)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class QuorumConfig:
    """Configuration for QuorumManager."""

    node_id: str
    """This node's identifier."""

    config_path: Path | None = None
    """Path to distributed_hosts.yaml (optional, auto-detected if not provided)."""

    ringrift_path: Path | None = None
    """Path to RingRift root (for config path resolution)."""

    def __post_init__(self) -> None:
        if self.config_path is None and self.ringrift_path:
            # Auto-detect config path
            rp = Path(self.ringrift_path)
            if rp.name == "ai-service":
                self.config_path = rp / "config" / "distributed_hosts.yaml"
            else:
                self.config_path = rp / "ai-service" / "config" / "distributed_hosts.yaml"


@dataclass
class VoterHealthStatus:
    """Status of voter health check."""

    voters_total: int = 0
    voters_alive: int = 0
    voters_offline: list[str] = field(default_factory=list)
    quorum_size: int = 0
    quorum_ok: bool = True
    quorum_threatened: bool = False
    voter_status: dict[str, dict[str, Any]] = field(default_factory=dict)


# ============================================================================
# Protocol for peer info access
# ============================================================================


class PeerInfo(Protocol):
    """Protocol for peer information."""

    node_id: str
    host: str

    def is_alive(self) -> bool: ...


# ============================================================================
# QuorumManager
# ============================================================================


class QuorumManager:
    """Manager for voter quorum tracking and health monitoring.

    Handles:
    - Loading voter node IDs from config (env > YAML)
    - Building IP-to-node mappings for peer translation
    - Counting alive voters for quorum decisions
    - Checking voter health and detecting quorum issues
    - Adopting voter sets from gossip for cluster convergence
    """

    def __init__(
        self,
        config: QuorumConfig,
        get_peers: Callable[[], dict[str, Any]],
        get_peers_lock: Callable[[], threading.RLock] | None = None,
    ) -> None:
        """Initialize QuorumManager.

        Args:
            config: Quorum configuration.
            get_peers: Callback to get current peers dict.
            get_peers_lock: Callback to get peers lock (optional).
        """
        self._config = config
        self._get_peers = get_peers
        self._get_peers_lock = get_peers_lock

        # State
        self._voter_node_ids: list[str] = []
        self._voter_config_source: str = "none"
        self._ip_to_node_map: dict[str, str] = {}
        self._voter_ip_map: dict[str, set[str]] = {}
        self._last_offline_voters: list[str] = []

        # Thread safety for internal state
        self._state_lock = threading.RLock()

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def voter_node_ids(self) -> list[str]:
        """Get configured voter node IDs."""
        with self._state_lock:
            return list(self._voter_node_ids)

    @property
    def voter_config_source(self) -> str:
        """Get source of voter configuration ('env', 'config', 'none')."""
        return self._voter_config_source

    # ========================================================================
    # Public API
    # ========================================================================

    def load_voter_node_ids(self) -> list[str]:
        """Load the set of P2P voter node_ids (for quorum-based leadership).

        If no voters are configured, returns an empty list and quorum checks are
        disabled (backwards compatible).

        Priority order:
        1. Environment variable RINGRIFT_P2P_VOTERS
        2. cluster_config.get_p2p_voters()
        3. Direct YAML load from distributed_hosts.yaml

        Jan 15, 2026 (Phase 5 P2P Resilience): Voters with status: offline
        in the hosts config are filtered out with a warning.

        Returns:
            List of voter node IDs (sorted, deduplicated).
        """
        voters: list[str] = []

        # Priority 1: Environment variable override (highest priority)
        env = (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip()
        if env:
            self._voter_config_source = "env"
            voters = [t.strip() for t in env.split(",") if t.strip()]
        else:
            # Priority 2: Use cluster_config for YAML-based voter loading
            try:
                from app.config.cluster_config import get_p2p_voters

                config_voters = get_p2p_voters()
                if config_voters:
                    self._voter_config_source = "config"
                    voters = config_voters
            except ImportError:
                logger.debug("[QuorumManager] cluster_config not available, falling back to direct YAML load")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[QuorumManager] Failed to load voters via cluster_config: {e}")

            # Priority 3: Fallback - direct YAML load for legacy compatibility
            if not voters:
                cfg_path = self._config.config_path
                if cfg_path and cfg_path.exists():
                    try:
                        import yaml

                        data = yaml.safe_load(cfg_path.read_text()) or {}
                        p2p_voters_list = data.get("p2p_voters", []) or []
                        if p2p_voters_list and isinstance(p2p_voters_list, list):
                            voters = sorted({str(v).strip() for v in p2p_voters_list if str(v).strip()})
                            if voters:
                                self._voter_config_source = "config"
                    except (OSError, ValueError, KeyError) as e:
                        logger.debug(f"[QuorumManager] Failed to load voter config from file: {e}")
                    except Exception as e:  # noqa: BLE001
                        # YAML errors
                        logger.debug(f"[QuorumManager] Failed to parse voter config: {e}")

        if not voters:
            self._voter_config_source = "none"
            return []

        # Jan 15, 2026: Filter out offline voters (Phase 5 P2P Resilience)
        active_voters = self._filter_offline_voters(voters)

        with self._state_lock:
            self._voter_node_ids = sorted(set(active_voters))
        return self._voter_node_ids

    def _filter_offline_voters(self, voters: list[str]) -> list[str]:
        """Filter out voters with status: offline in hosts config.

        Jan 15, 2026: Phase 5 of P2P Resilience Plan.

        Args:
            voters: List of voter node IDs

        Returns:
            List of voter IDs that are not marked offline
        """
        cfg_path = self._config.config_path
        if not cfg_path or not cfg_path.exists():
            return voters  # Can't check, keep all

        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[QuorumManager] Could not load hosts for offline filter: {e}")
            return voters

        active_voters = []
        offline_voters = []

        for voter_id in voters:
            host_cfg = hosts.get(voter_id, {})
            if host_cfg.get("status") == "offline":
                offline_voters.append(voter_id)
            else:
                active_voters.append(voter_id)

        if offline_voters:
            logger.warning(
                f"[QuorumManager] Filtered {len(offline_voters)} offline voter(s): "
                f"{', '.join(offline_voters)}"
            )

        return active_voters

    def build_voter_ip_mapping(self) -> dict[str, set[str]]:
        """Build a mapping from voter node_ids to their known IPs.

        Returns:
            Dict mapping voter node_id -> set of known IPs (tailscale_ip, ssh_host).
        """
        with self._state_lock:
            voter_ids = list(self._voter_node_ids)
        if not voter_ids:
            return {}

        cfg_path = self._config.config_path
        if not cfg_path or not cfg_path.exists():
            return {}

        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
        except ImportError:
            # yaml module not available
            return {}
        except (OSError, IOError):
            # File read error
            return {}
        except yaml.YAMLError:
            # Malformed YAML
            return {}
        except (TypeError, KeyError, AttributeError):
            # Unexpected data structure
            return {}

        voter_ip_map: dict[str, set[str]] = {}
        for voter_id in voter_ids:
            host_cfg = hosts.get(voter_id, {})
            ips: set[str] = set()

            # Collect all known IPs for this voter
            if host_cfg.get("tailscale_ip"):
                ips.add(host_cfg["tailscale_ip"])
            if host_cfg.get("ssh_host"):
                ssh_host = host_cfg["ssh_host"]
                # Only add if it's an IP (not a hostname like "ssh5.vast.ai")
                if ssh_host and not any(c.isalpha() for c in ssh_host.replace(".", "")):
                    ips.add(ssh_host)

            if ips:
                voter_ip_map[voter_id] = ips

        with self._state_lock:
            self._voter_ip_map = voter_ip_map
        return voter_ip_map

    def build_ip_to_node_map(self) -> dict[str, str]:
        """Build a reverse mapping from IP addresses to node names.

        Returns:
            Dict mapping IP address -> node_id.
        """
        cfg_path = self._config.config_path
        if not cfg_path or not cfg_path.exists():
            return {}

        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
        except ImportError:
            # yaml module not available
            return {}
        except (OSError, IOError):
            # File read error
            return {}
        except yaml.YAMLError:
            # Malformed YAML
            return {}
        except (TypeError, KeyError, AttributeError):
            # Unexpected data structure
            return {}

        ip_to_node: dict[str, str] = {}
        for node_id, host_cfg in hosts.items():
            ts_ip = host_cfg.get("tailscale_ip")
            if ts_ip:
                ip_to_node[ts_ip] = node_id
            ssh_host = host_cfg.get("ssh_host")
            if ssh_host and not any(c.isalpha() for c in ssh_host.replace(".", "")):
                ip_to_node[ssh_host] = node_id

        with self._state_lock:
            self._ip_to_node_map = ip_to_node
        return ip_to_node

    def resolve_peer_id_to_node_name(self, peer_id: str) -> str:
        """Translate a SWIM peer ID (IP:port) to a node name if possible.

        Args:
            peer_id: Peer identifier (may be node_id or IP:port).

        Returns:
            Node name if translation found, otherwise original peer_id.
        """
        if ":" not in peer_id or not peer_id.split(":")[0].replace(".", "").isdigit():
            return peer_id
        ip = peer_id.split(":")[0]
        with self._state_lock:
            return self._ip_to_node_map.get(ip, peer_id)

    def find_voter_peer_by_ip(
        self, voter_id: str
    ) -> tuple[str | None, Any | None]:
        """Find a voter's peer entry by matching their known IPs against peers dict.

        Args:
            voter_id: The voter's friendly node_id (e.g., 'hetzner-cpu1').

        Returns:
            Tuple of (peer_key, peer_info) where peer_key is the key in
            peers, or (None, None) if not found.
        """
        peers = self._get_peers_safely()

        # Strategy 1: Direct node_id match
        if voter_id in peers:
            return voter_id, peers[voter_id]

        # Strategy 2: Get voter's known IPs from config
        voter_ip_map = self.build_voter_ip_mapping()
        voter_ips = voter_ip_map.get(voter_id, set())
        if not voter_ips:
            return None, None

        # Strategy 3: Check peer info 'host' field for IP match
        for peer_key, peer_info in peers.items():
            if isinstance(peer_info, dict):
                peer_host = peer_info.get("host", "")
                if peer_host in voter_ips:
                    return peer_key, peer_info
            elif hasattr(peer_info, "host") and peer_info.host in voter_ips:
                return peer_key, peer_info

        # Strategy 4: Extract IP from peer_key (IP:PORT format)
        for peer_key, peer_info in peers.items():
            if ":" in peer_key:
                peer_ip = peer_key.rsplit(":", 1)[0]
                if peer_ip in voter_ips:
                    return peer_key, peer_info

        return None, None

    def count_alive_voters(self) -> int:
        """Count alive voters by checking both node_id and IP:port matches.

        Returns:
            Number of alive voters (including self if we are a voter).
        """
        with self._state_lock:
            voter_ids = list(self._voter_node_ids)
        if not voter_ids:
            return 0

        peers = self._get_peers_safely()
        alive_count = 0

        # Build voter IP mapping
        voter_ip_map = self.build_voter_ip_mapping()

        # Track which voters we've counted
        counted_voters: set[str] = set()

        for voter_id in voter_ids:
            if voter_id in counted_voters:
                continue

            # Check 1: Is this voter us?
            if voter_id == self._config.node_id:
                alive_count += 1
                counted_voters.add(voter_id)
                continue

            # Check 2: Direct node_id match in peers
            peer = peers.get(voter_id)
            if peer and self._is_peer_alive_from_info(peer):
                alive_count += 1
                counted_voters.add(voter_id)
                continue

            # Check 3: Peer info 'host' field match
            voter_ips = voter_ip_map.get(voter_id, set())
            for peer_id, peer in peers.items():
                if voter_id in counted_voters:
                    break
                # Skip SWIM protocol entries (IP:7947)
                if self.is_swim_peer_id(peer_id):
                    continue
                # Check peer info 'host' field
                if isinstance(peer, dict):
                    peer_host = peer.get("host", "")
                    if peer_host in voter_ips and self._is_peer_alive_from_info(peer):
                        alive_count += 1
                        counted_voters.add(voter_id)
                        break
                elif hasattr(peer, "host") and peer.host in voter_ips:
                    if hasattr(peer, "is_alive") and peer.is_alive():
                        alive_count += 1
                        counted_voters.add(voter_id)
                        break

            if voter_id in counted_voters:
                continue

            # Check 4: IP:port extraction from peer_id
            for peer_id, peer in peers.items():
                if voter_id in counted_voters:
                    break
                # Skip SWIM protocol entries
                if self.is_swim_peer_id(peer_id):
                    continue
                if ":" in peer_id:
                    peer_ip = peer_id.rsplit(":", 1)[0]
                    if peer_ip in voter_ips:
                        if self._is_peer_alive_from_info(peer):
                            alive_count += 1
                            counted_voters.add(voter_id)
                            break

        return alive_count

    def check_voter_health(self, voter_quorum_size: int) -> VoterHealthStatus:
        """Check health status of all configured voters.

        Args:
            voter_quorum_size: Required quorum size for leader election.

        Returns:
            VoterHealthStatus with details about voter health.
        """
        with self._state_lock:
            voter_ids = list(self._voter_node_ids)

        if not voter_ids:
            return VoterHealthStatus(quorum_ok=True)

        voter_ip_map = self.build_voter_ip_mapping()
        peers = self._get_peers_safely()

        # Check each voter's status
        voter_status: dict[str, dict[str, Any]] = {}
        alive_voters: list[str] = []
        offline_voters: list[str] = []

        # Track last known status for change detection
        with self._state_lock:
            prev_offline = set(self._last_offline_voters)

        for voter_id in voter_ids:
            is_alive = False
            status_detail = "unknown"

            # Check 1: Is this voter us?
            if voter_id == self._config.node_id:
                is_alive = True
                status_detail = "self"
            else:
                # Check 2: Direct node_id match in peers
                peer = peers.get(voter_id)
                if peer and self._is_peer_alive_from_info(peer):
                    is_alive = True
                    status_detail = "direct"
                else:
                    # Check 3: IP:port match
                    voter_ips = voter_ip_map.get(voter_id, set())
                    for peer_id, p in peers.items():
                        # Skip SWIM protocol entries
                        if self.is_swim_peer_id(peer_id):
                            continue
                        if ":" in peer_id:
                            peer_ip = peer_id.split(":")[0]
                            if peer_ip in voter_ips and self._is_peer_alive_from_info(p):
                                is_alive = True
                                status_detail = f"ip_match:{peer_ip}"
                                break
                    if not is_alive:
                        status_detail = "unreachable"

            voter_status[voter_id] = {"alive": is_alive, "detail": status_detail}
            if is_alive:
                alive_voters.append(voter_id)
            else:
                offline_voters.append(voter_id)

        # Store for next comparison
        with self._state_lock:
            self._last_offline_voters = offline_voters

        voters_alive = len(alive_voters)
        voters_total = len(voter_ids)
        quorum_ok = voters_alive >= voter_quorum_size
        quorum_threatened = voters_alive <= voter_quorum_size

        # Log status changes
        newly_offline = set(offline_voters) - prev_offline
        newly_online = prev_offline - set(offline_voters)

        for voter_id in newly_offline:
            logger.warning(
                f"[QuorumManager] Voter {voter_id} went OFFLINE "
                f"({voters_alive}/{voters_total} alive, quorum={voter_quorum_size})"
            )

        for voter_id in newly_online:
            logger.info(
                f"[QuorumManager] Voter {voter_id} came ONLINE "
                f"({voters_alive}/{voters_total} alive, quorum={voter_quorum_size})"
            )

        # Periodic summary log
        if offline_voters:
            logger.warning(
                f"[QuorumManager] Status: {voters_alive}/{voters_total} voters alive, "
                f"quorum={'OK' if quorum_ok else 'LOST'}, offline: {offline_voters}"
            )

        return VoterHealthStatus(
            voters_total=voters_total,
            voters_alive=voters_alive,
            voters_offline=offline_voters,
            quorum_size=voter_quorum_size,
            quorum_ok=quorum_ok,
            quorum_threatened=quorum_threatened,
            voter_status=voter_status,
        )

    def maybe_adopt_voter_node_ids(
        self, voter_node_ids: list[str], *, source: str
    ) -> bool:
        """Adopt/override the voter set when it's not explicitly configured.

        This is a convergence mechanism: some nodes may boot without local
        config (or with stale config), which would disable quorum gating.
        Leaders propagate the stable voter set via gossip so the cluster converges.

        Args:
            voter_node_ids: List of voter node IDs to adopt.
            source: Source of the voter list (for logging).

        Returns:
            True if voter set was adopted, False otherwise.
        """
        # If explicitly configured via env var, never override
        if (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip():
            return False

        # If explicitly configured via YAML, never override from gossip
        if self._voter_config_source == "config":
            return False

        normalized = sorted({str(v).strip() for v in (voter_node_ids or []) if str(v).strip()})
        if not normalized:
            return False

        # Check if we're about to change voters
        with self._state_lock:
            current = self._voter_node_ids
            if current == normalized:
                return False

            # Adopt the voter set
            self._voter_node_ids = normalized
            self._voter_config_source = f"adopted:{source}"

        logger.info(
            f"[QuorumManager] Adopted voter set from {source}: {normalized}"
        )
        return True

    @staticmethod
    def is_swim_peer_id(peer_id: str) -> bool:
        """Check if peer_id is a SWIM protocol entry (IP:7947 format).

        SWIM entries use port 7947 and should not be in the HTTP peer list.
        These leak from the SWIM membership layer and cause peer pollution.

        Args:
            peer_id: Node identifier to check.

        Returns:
            True if this is a SWIM-format peer ID (should be skipped).
        """
        if not peer_id or ":" not in peer_id:
            return False
        parts = peer_id.rsplit(":", 1)
        return len(parts) == 2 and parts[1] == "7947"

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self, voter_quorum_size: int = 0) -> dict[str, Any]:
        """Get health status for daemon manager integration.

        Args:
            voter_quorum_size: Required quorum size (0 to skip health check).

        Returns:
            Dict with status, details, and metrics.
        """
        with self._state_lock:
            voter_count = len(self._voter_node_ids)

        if voter_count == 0:
            return {
                "status": "idle",
                "message": "No voters configured",
                "voter_count": 0,
                "config_source": self._voter_config_source,
            }

        if voter_quorum_size > 0:
            health = self.check_voter_health(voter_quorum_size)
            status = "healthy" if health.quorum_ok else "error"
            if health.quorum_threatened and health.quorum_ok:
                status = "degraded"

            return {
                "status": status,
                "message": f"{health.voters_alive}/{health.voters_total} voters alive",
                "voter_count": voter_count,
                "voters_alive": health.voters_alive,
                "quorum_ok": health.quorum_ok,
                "quorum_threatened": health.quorum_threatened,
                "config_source": self._voter_config_source,
                "offline_voters": health.voters_offline,
            }

        return {
            "status": "healthy",
            "message": f"{voter_count} voters configured",
            "voter_count": voter_count,
            "config_source": self._voter_config_source,
        }

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _get_peers_safely(self) -> dict[str, Any]:
        """Get peers dict with optional lock."""
        if self._get_peers_lock:
            with self._get_peers_lock():
                return dict(self._get_peers())
        return dict(self._get_peers())

    def _is_peer_alive_from_info(self, peer_info: Any) -> bool:
        """Check if a peer is alive based on its info."""
        if isinstance(peer_info, dict):
            status = peer_info.get("status", "unknown")
            return status in ("alive", "healthy", "connected")
        if hasattr(peer_info, "is_alive"):
            return peer_info.is_alive()
        return False


# ============================================================================
# Factory Functions
# ============================================================================


_quorum_manager: QuorumManager | None = None
_quorum_manager_lock = threading.Lock()


def get_quorum_manager() -> QuorumManager | None:
    """Get the singleton QuorumManager instance (if initialized)."""
    return _quorum_manager


def set_quorum_manager(manager: QuorumManager) -> None:
    """Set the singleton QuorumManager instance."""
    global _quorum_manager
    with _quorum_manager_lock:
        _quorum_manager = manager


def create_quorum_manager(
    node_id: str,
    get_peers: Callable[[], dict[str, Any]],
    get_peers_lock: Callable[[], threading.RLock] | None = None,
    config_path: Path | None = None,
    ringrift_path: Path | None = None,
) -> QuorumManager:
    """Create and register a QuorumManager instance.

    Args:
        node_id: This node's identifier.
        get_peers: Callback to get current peers dict.
        get_peers_lock: Callback to get peers lock (optional).
        config_path: Path to distributed_hosts.yaml (optional).
        ringrift_path: Path to RingRift root (optional).

    Returns:
        Configured QuorumManager instance.
    """
    config = QuorumConfig(
        node_id=node_id,
        config_path=config_path,
        ringrift_path=ringrift_path,
    )
    manager = QuorumManager(
        config=config,
        get_peers=get_peers,
        get_peers_lock=get_peers_lock,
    )
    set_quorum_manager(manager)
    return manager
