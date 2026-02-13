"""Consolidation Eligibility Manager.

Determines which nodes can run data consolidation based on configurable
criteria including disk space, CPU, memory, and role.

Dec 30, 2025: Phase 3.3 of Distributed Data Pipeline Architecture.

Design:
- Nodes are eligible if they meet disk, CPU, and role requirements
- Eligibility can be configured per-node in distributed_hosts.yaml
- Best consolidation node is selected based on data locality and capacity

Usage:
    from app.coordination.consolidation_eligibility import (
        get_eligibility_manager,
        ConsolidationEligibilityManager,
    )

    manager = get_eligibility_manager()

    # Check all eligible nodes
    eligible = manager.get_eligible_nodes()

    # Check specific node
    is_eligible, reason = manager.is_node_eligible("nebius-h100-1")

    # Get best node for a config
    best = manager.get_best_consolidation_node("hex8_2p")
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.sqlite_utils import connect_safe

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Default eligible roles for consolidation
DEFAULT_ELIGIBLE_ROLES = [
    "coordinator",
    "gpu_training",
    "gpu_training_primary",
    "backbone",
    "gpu_selfplay_primary",
]


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation eligibility.

    Attributes:
        enabled: Whether consolidation is enabled
        min_disk_free_gb: Minimum free disk space required (GB)
        max_cpu_percent: Maximum CPU utilization allowed
        min_games_to_consolidate: Minimum games before consolidation
        max_games_per_batch: Maximum games per consolidation batch
        eligible_roles: Node roles eligible for consolidation
        excluded_nodes: Explicitly excluded node IDs
        prefer_nodes_with_most_data: Prefer nodes with most data for config
        consolidation_interval_seconds: Interval between consolidation runs
    """

    enabled: bool = True
    min_disk_free_gb: float = 20.0
    max_cpu_percent: float = 80.0
    min_games_to_consolidate: int = 50
    max_games_per_batch: int = 5000
    eligible_roles: list[str] = field(default_factory=lambda: DEFAULT_ELIGIBLE_ROLES.copy())
    excluded_nodes: list[str] = field(default_factory=list)
    prefer_nodes_with_most_data: bool = True
    consolidation_interval_seconds: int = 300

    @classmethod
    def from_yaml_config(cls, config: dict[str, Any]) -> ConsolidationConfig:
        """Create config from YAML dictionary.

        Args:
            config: Dictionary from distributed_hosts.yaml consolidation section

        Returns:
            ConsolidationConfig instance
        """
        return cls(
            enabled=config.get("enabled", True),
            min_disk_free_gb=config.get("min_disk_free_gb", 20.0),
            max_cpu_percent=config.get("max_cpu_percent", 80.0),
            min_games_to_consolidate=config.get("min_games_to_consolidate", 50),
            max_games_per_batch=config.get("max_games_per_batch", 5000),
            eligible_roles=config.get("eligible_roles", DEFAULT_ELIGIBLE_ROLES.copy()),
            excluded_nodes=config.get("excluded_nodes", []),
            prefer_nodes_with_most_data=config.get("prefer_nodes_with_most_data", True),
            consolidation_interval_seconds=config.get("consolidation_interval_seconds", 300),
        )


@dataclass
class EligibilityResult:
    """Result of an eligibility check."""

    is_eligible: bool
    reason: str
    disk_free_gb: float | None = None
    cpu_percent: float | None = None
    role: str | None = None


@dataclass
class NodeConsolidationInfo:
    """Information about a node's consolidation capabilities."""

    node_id: str
    is_eligible: bool
    reason: str
    disk_free_gb: float
    game_counts: dict[str, int]  # config_key -> game count
    role: str | None = None


class ConsolidationEligibilityManager:
    """Determines which nodes can run consolidation.

    Uses a combination of static configuration and runtime metrics
    to determine eligibility. Can be integrated with P2P cluster
    to query remote node states.

    Thread-safe: Uses read-only config and queries remote state.
    """

    _instance: ConsolidationEligibilityManager | None = None

    def __init__(
        self,
        root_path: Path | str | None = None,
        config: ConsolidationConfig | None = None,
    ):
        """Initialize the eligibility manager.

        Args:
            root_path: Root path to ai-service directory
            config: Consolidation configuration (loads from YAML if None)
        """
        if root_path is None:
            root_path = Path(__file__).parent.parent.parent
        self.root_path = Path(root_path)

        # Load config from YAML if not provided
        if config is None:
            config = self._load_config_from_yaml()
        self.config = config

        # Cache for host configs
        self._host_configs: dict[str, dict] = {}
        self._load_host_configs()

    @classmethod
    def get_instance(cls) -> ConsolidationEligibilityManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_config_from_yaml(self) -> ConsolidationConfig:
        """Load configuration from distributed_hosts.yaml."""
        try:
            import yaml

            config_path = self.root_path / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    full_config = yaml.safe_load(f) or {}
                consolidation_config = full_config.get("consolidation", {})
                if consolidation_config.get("enabled", True):
                    return ConsolidationConfig.from_yaml_config(consolidation_config)
        except ImportError:
            # yaml module not available
            pass
        except (OSError, yaml.YAMLError) as e:
            # File I/O errors or malformed YAML
            logger.warning(f"[ConsolidationEligibility] Failed to load YAML config: {e}")

        return ConsolidationConfig()

    def _load_host_configs(self) -> None:
        """Load host configurations from distributed_hosts.yaml."""
        try:
            import yaml

            config_path = self.root_path / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    full_config = yaml.safe_load(f) or {}
                self._host_configs = full_config.get("hosts", {})
        except ImportError:
            # yaml module not available
            pass
        except (OSError, yaml.YAMLError) as e:
            # File I/O errors or malformed YAML
            logger.warning(f"[ConsolidationEligibility] Failed to load host configs: {e}")

    def get_eligible_nodes(self) -> list[str]:
        """Return list of nodes eligible for consolidation.

        Returns:
            List of node IDs that can run consolidation
        """
        eligible = []
        for node_id in self._host_configs:
            is_eligible, _ = self.is_node_eligible(node_id)
            if is_eligible:
                eligible.append(node_id)
        return eligible

    def is_node_eligible(self, node_id: str) -> tuple[bool, str]:
        """Check if specific node can consolidate.

        Args:
            node_id: Node identifier

        Returns:
            Tuple of (is_eligible, reason)
        """
        result = self._check_eligibility(node_id)
        return result.is_eligible, result.reason

    def _check_eligibility(self, node_id: str) -> EligibilityResult:
        """Perform detailed eligibility check.

        Args:
            node_id: Node identifier

        Returns:
            EligibilityResult with details
        """
        # Check if consolidation is enabled
        if not self.config.enabled:
            return EligibilityResult(
                is_eligible=False,
                reason="Consolidation disabled globally",
            )

        # Check if node is explicitly excluded
        if node_id in self.config.excluded_nodes:
            return EligibilityResult(
                is_eligible=False,
                reason="Node explicitly excluded",
            )

        # Get node config
        host_config = self._host_configs.get(node_id)
        if host_config is None:
            # January 2026: Allow consolidation for nodes not in config if they have local data.
            # This enables consolidation on local development machines and any node with
            # game data in data/games/ that needs to be merged into canonical DBs.
            # The node may have received data via OWC import, P2P sync, or local selfplay.
            if self._has_local_game_data():
                return EligibilityResult(
                    is_eligible=True,
                    reason="Node not in config but has local game data",
                    disk_free_gb=self._get_local_disk_free_gb_unsafe(),
                )
            return EligibilityResult(
                is_eligible=False,
                reason="Node not found in configuration and no local game data",
            )

        # Check node status
        status = host_config.get("status", "unknown")
        if status not in ("ready", "active"):
            return EligibilityResult(
                is_eligible=False,
                reason=f"Node status is {status}, not ready",
            )

        # Check per-node override
        consolidation_enabled = host_config.get("consolidation_enabled")
        if consolidation_enabled is False:
            return EligibilityResult(
                is_eligible=False,
                reason="Consolidation explicitly disabled for this node",
            )

        # Check role eligibility
        role = host_config.get("role", "unknown")
        if role not in self.config.eligible_roles:
            # Allow override via consolidation_enabled: true
            if consolidation_enabled is not True:
                return EligibilityResult(
                    is_eligible=False,
                    reason=f"Role {role} not in eligible roles",
                    role=role,
                )

        # Check disk space (for local node only)
        disk_free_gb = self._get_local_disk_free_gb(node_id)
        if disk_free_gb is not None:
            if disk_free_gb < self.config.min_disk_free_gb:
                return EligibilityResult(
                    is_eligible=False,
                    reason=f"Disk free {disk_free_gb:.1f}GB < {self.config.min_disk_free_gb}GB",
                    disk_free_gb=disk_free_gb,
                    role=role,
                )

        return EligibilityResult(
            is_eligible=True,
            reason="All eligibility criteria met",
            disk_free_gb=disk_free_gb,
            role=role,
        )

    def _get_local_disk_free_gb(self, node_id: str) -> float | None:
        """Get disk free space for local node only.

        Args:
            node_id: Node identifier

        Returns:
            Free disk space in GB, or None if not local
        """
        # Check if this is the local node
        try:
            from app.config.env import env

            local_node_id = env.node_id
            if node_id != local_node_id:
                return None  # Can't check remote node disk space here

            disk_usage = shutil.disk_usage(self.root_path)
            return disk_usage.free / (1024**3)
        except ImportError:
            # env module not available
            return None
        except OSError:
            # Disk access error (permission, path not found, etc.)
            return None

    def _get_local_disk_free_gb_unsafe(self) -> float | None:
        """Get disk free space for the current node (no node_id validation).

        Used when checking eligibility for nodes not in configuration.

        Returns:
            Free disk space in GB, or None on error
        """
        try:
            disk_usage = shutil.disk_usage(self.root_path)
            return disk_usage.free / (1024**3)
        except OSError:
            return None

    def _has_local_game_data(self) -> bool:
        """Check if this node has local game data that could be consolidated.

        Scans common source directories for non-canonical game databases.
        Returns True if any source databases exist with games.

        January 2026: Used to allow consolidation on nodes not in config.

        Returns:
            True if local game data exists
        """
        data_dir = self.root_path / "data" / "games"
        search_dirs = [
            data_dir,
            data_dir / "owc_imports",
            data_dir / "synced",
            data_dir / "selfplay",
            data_dir / "p2p_gpu",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Look for any .db files that aren't canonical databases
            for db_path in search_dir.glob("**/*.db"):
                if "canonical" in db_path.name.lower():
                    continue

                # Quick check: does it have any games?
                try:
                    conn = connect_safe(db_path, timeout=2.0, row_factory=None)
                    cursor = conn.execute("SELECT 1 FROM games LIMIT 1")
                    has_games = cursor.fetchone() is not None
                    conn.close()
                    if has_games:
                        logger.debug(
                            f"[ConsolidationEligibility] Found local game data in {db_path}"
                        )
                        return True
                except (sqlite3.Error, OSError):
                    continue

        return False

    def get_best_consolidation_node(
        self,
        config_key: str,
        exclude_nodes: list[str] | None = None,
    ) -> str | None:
        """Select best node for consolidating a specific config.

        Prefers:
        1. Nodes with most data for this config (if prefer_nodes_with_most_data)
        2. Nodes with most free disk space
        3. First eligible node alphabetically

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            exclude_nodes: Node IDs to exclude from selection

        Returns:
            Best node ID, or None if no eligible nodes
        """
        exclude_nodes = exclude_nodes or []
        candidates: list[NodeConsolidationInfo] = []

        for node_id in self._host_configs:
            if node_id in exclude_nodes:
                continue

            is_eligible, reason = self.is_node_eligible(node_id)
            if not is_eligible:
                continue

            # Get game counts for this node (from P2P manifest if available)
            game_counts = self._get_node_game_counts(node_id)
            disk_free = self._get_local_disk_free_gb(node_id) or 0.0

            candidates.append(
                NodeConsolidationInfo(
                    node_id=node_id,
                    is_eligible=True,
                    reason=reason,
                    disk_free_gb=disk_free,
                    game_counts=game_counts,
                    role=self._host_configs.get(node_id, {}).get("role"),
                )
            )

        if not candidates:
            return None

        # Sort by preference
        def sort_key(info: NodeConsolidationInfo) -> tuple:
            games_for_config = info.game_counts.get(config_key, 0)
            if self.config.prefer_nodes_with_most_data:
                # Higher games first, then higher disk space
                return (-games_for_config, -info.disk_free_gb, info.node_id)
            else:
                # Higher disk space first
                return (-info.disk_free_gb, info.node_id)

        candidates.sort(key=sort_key)
        return candidates[0].node_id

    def _get_node_game_counts(self, node_id: str) -> dict[str, int]:
        """Get game counts for a node from cluster manifest.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary of config_key -> game count
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            counts: dict[str, int] = {}

            with manifest._connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT board_type || '_' || num_players || 'p' as config,
                           COUNT(DISTINCT game_id)
                    FROM game_locations
                    WHERE node_id = ?
                    GROUP BY board_type, num_players
                    """,
                    (node_id,),
                )
                for row in cursor:
                    config_key = row[0]
                    if config_key and config_key != "None_Nonep":
                        counts[config_key] = row[1]

            return counts
        except ImportError:
            # cluster_manifest module not available
            return {}
        except sqlite3.Error as e:
            # Database query error
            logger.debug(f"[ConsolidationEligibility] Error getting game counts for {node_id}: {e}")
            return {}

    def get_all_node_info(self) -> list[NodeConsolidationInfo]:
        """Get consolidation info for all configured nodes.

        Returns:
            List of NodeConsolidationInfo for all nodes
        """
        result = []
        for node_id in self._host_configs:
            is_eligible, reason = self.is_node_eligible(node_id)
            game_counts = self._get_node_game_counts(node_id)
            disk_free = self._get_local_disk_free_gb(node_id) or 0.0

            result.append(
                NodeConsolidationInfo(
                    node_id=node_id,
                    is_eligible=is_eligible,
                    reason=reason,
                    disk_free_gb=disk_free,
                    game_counts=game_counts,
                    role=self._host_configs.get(node_id, {}).get("role"),
                )
            )
        return result


# ============================================================================
# Singleton accessor
# ============================================================================


def get_eligibility_manager() -> ConsolidationEligibilityManager:
    """Get the singleton ConsolidationEligibilityManager instance.

    Returns:
        The singleton instance
    """
    return ConsolidationEligibilityManager.get_instance()
